use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, UInt32Array, UInt64Array,
};
use arrow_ord::sort::{SortColumn, SortOptions, lexsort_to_indices};
use arrow_schema::SchemaRef;

use crate::error::HirnDbError;
use crate::store::{ExactMatchFilter, ScanOptions, ScanOrdering};

/// Apply scan options (filter, projection, limit, offset) to in-memory record batches.
///
/// This is used by `MemoryStore` for applying scan semantics to its in-memory data.
/// LancePhysicalStore pushes these down to the Lance Scanner instead.
pub fn apply_scan_options(
    batches: &[RecordBatch],
    opts: &ScanOptions,
) -> Result<Vec<RecordBatch>, HirnDbError> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    let mut result: Vec<RecordBatch> = batches.to_vec();

    // Apply filter first (before projection removes columns needed by the filter)
    if let Some(ref exact_filter) = opts.exact_filter {
        result = filter_batches_exact(exact_filter, &result)?;
    }

    if let Some(ref filter) = opts.filter {
        result = filter_batches(filter, &result)?;
    }

    // Apply ordering before projection so callers can sort by columns they do
    // not need in the final output.
    if let Some(ref ordering) = opts.order_by {
        result = apply_ordering(&result, ordering)?;
    }

    // Apply column projection
    if let Some(ref columns) = opts.columns {
        result = project_batches(&result, columns)?;
    }

    // Apply offset and limit
    result = apply_limit_offset(&result, opts.limit, opts.offset);

    Ok(result)
}

fn filter_batches_exact(
    filter: &ExactMatchFilter,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    match filter {
        ExactMatchFilter::Utf8In { column, values } => {
            filter_batches_utf8_in(column, values, batches)
        }
        ExactMatchFilter::Utf8MultiColumnOr { columns, value } => {
            filter_batches_utf8_multi_column_or(columns, value, batches)
        }
    }
}

fn filter_batches_utf8_in(
    column: &str,
    values: &[String],
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    if values.is_empty() {
        return Ok(Vec::new());
    }

    let value_set: std::collections::HashSet<&str> = values.iter().map(String::as_str).collect();
    let mut result = Vec::new();

    for batch in batches {
        let schema = batch.schema();
        let col_idx = schema.index_of(column).map_err(|_| {
            HirnDbError::InvalidArgument(format!("column `{column}` not found in schema"))
        })?;
        let col = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                HirnDbError::InvalidArgument(format!(
                    "exact UTF-8 filter requires Utf8 column `{column}`"
                ))
            })?;

        let mask = BooleanArray::from(
            (0..batch.num_rows())
                .map(|row| (!col.is_null(row)) && value_set.contains(col.value(row)))
                .collect::<Vec<_>>(),
        );
        let filtered = arrow_select::filter::filter_record_batch(batch, &mask)
            .map_err(HirnDbError::ArrowError)?;
        if filtered.num_rows() > 0 {
            result.push(filtered);
        }
    }

    Ok(result)
}

fn filter_batches_utf8_multi_column_or(
    columns: &[String],
    value: &str,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    let mut result = Vec::new();
    for batch in batches {
        let schema = batch.schema();
        let mut row_mask = vec![false; batch.num_rows()];

        for column in columns {
            let col_idx = match schema.index_of(column) {
                Ok(idx) => idx,
                Err(_) => continue, // column absent in this batch — skip
            };
            let col = batch
                .column(col_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    HirnDbError::InvalidArgument(format!(
                        "multi-column OR filter requires Utf8 column `{column}`"
                    ))
                })?;
            for (row, mask_slot) in row_mask.iter_mut().enumerate() {
                if (!col.is_null(row)) && col.value(row) == value {
                    *mask_slot = true;
                }
            }
        }

        let mask = BooleanArray::from(row_mask);
        let filtered = arrow_select::filter::filter_record_batch(batch, &mask)
            .map_err(HirnDbError::ArrowError)?;
        if filtered.num_rows() > 0 {
            result.push(filtered);
        }
    }
    Ok(result)
}

/// Project record batches to only include the specified columns.
pub fn project_batches(
    batches: &[RecordBatch],
    columns: &[String],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    let mut projected = Vec::with_capacity(batches.len());
    for batch in batches {
        let schema = batch.schema();
        let mut indices = Vec::with_capacity(columns.len());
        for col_name in columns {
            match schema.index_of(col_name) {
                Ok(idx) => indices.push(idx),
                Err(_) => {
                    return Err(HirnDbError::InvalidArgument(format!(
                        "column `{col_name}` not found in schema"
                    )));
                }
            }
        }
        let projected_batch = batch.project(&indices).map_err(HirnDbError::ArrowError)?;
        projected.push(projected_batch);
    }
    Ok(projected)
}

/// Order record batches lexicographically across one or more columns.
pub fn apply_ordering(
    batches: &[RecordBatch],
    ordering: &[ScanOrdering],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    if batches.is_empty() || ordering.is_empty() {
        return Ok(batches.to_vec());
    }

    let schema = batches[0].schema();
    let combined = arrow_select::concat::concat_batches(&schema, batches.iter())
        .map_err(HirnDbError::ArrowError)?;
    if combined.num_rows() == 0 {
        return Ok(Vec::new());
    }

    let sort_columns = ordering
        .iter()
        .map(|order| {
            let idx = schema.index_of(&order.column).map_err(|_| {
                HirnDbError::InvalidArgument(format!(
                    "column `{}` not found in schema",
                    order.column
                ))
            })?;
            Ok(SortColumn {
                values: combined.column(idx).clone(),
                options: Some(SortOptions {
                    descending: !order.ascending,
                    nulls_first: order.nulls_first,
                }),
            })
        })
        .collect::<Result<Vec<_>, HirnDbError>>()?;

    let indices = lexsort_to_indices(&sort_columns, None).map_err(HirnDbError::ArrowError)?;
    let sorted_columns = combined
        .columns()
        .iter()
        .map(|column| {
            arrow_select::take::take(column.as_ref(), &indices, None)
                .map_err(HirnDbError::ArrowError)
        })
        .collect::<Result<Vec<_>, HirnDbError>>()?;

    let sorted = RecordBatch::try_new(schema, sorted_columns).map_err(HirnDbError::ArrowError)?;
    Ok(vec![sorted])
}

/// Apply limit and offset to record batches. Returns a new vec of batches
/// containing at most `limit` total rows, starting from `offset`.
pub fn apply_limit_offset(
    batches: &[RecordBatch],
    limit: Option<usize>,
    offset: Option<usize>,
) -> Vec<RecordBatch> {
    let offset = offset.unwrap_or(0);
    let limit = limit.unwrap_or(usize::MAX);

    let mut result = Vec::new();
    let mut skipped = 0usize;
    let mut taken = 0usize;

    for batch in batches {
        if taken >= limit {
            break;
        }

        let batch_len = batch.num_rows();

        // Skip rows for offset
        if skipped + batch_len <= offset {
            skipped += batch_len;
            continue;
        }

        let start = offset.saturating_sub(skipped);
        skipped += start;

        let remaining = limit - taken;
        let end = start.saturating_add(remaining).min(batch_len);
        let slice_len = end - start;

        if slice_len > 0 {
            result.push(batch.slice(start, slice_len));
            taken += slice_len;
        }

        skipped = skipped.max(offset);
    }

    result
}

/// Compute the total row count across batches.
pub fn total_row_count(batches: &[RecordBatch]) -> u64 {
    batches.iter().map(|b| b.num_rows() as u64).sum()
}

/// Concatenate all batches into one (or return empty if none).
pub fn concat_batches(
    schema: &SchemaRef,
    batches: &[RecordBatch],
) -> Result<Option<RecordBatch>, HirnDbError> {
    if batches.is_empty() {
        return Ok(None);
    }
    let combined =
        arrow_select::concat::concat_batches(schema, batches).map_err(HirnDbError::ArrowError)?;
    Ok(Some(combined))
}

// ── SQL-like filter evaluation for MemoryStore ────────────────────────

/// Filter record batches using a SQL-like predicate expression.
///
/// Supports:
/// - `column = 'value'` / `column = value`
/// - `column != 'value'` / `column <> 'value'`
/// - `column > value`, `column < value`, `column >= value`, `column <= value`
/// - `column IN ('v1', 'v2', ...)`
/// - `column IS NULL` / `column IS NOT NULL`
/// - `NOT expr`
/// - `expr AND expr`
/// - `expr OR expr`
/// - Parenthesized grouping: `(expr)`
///
/// Comparisons follow SQL three-valued logic: a comparison against a NULL cell
/// evaluates to UNKNOWN, and only rows whose predicate evaluates to TRUE are
/// kept — matching what Lance's SQL filter pushdown does. `IS [NOT] NULL` is
/// the only way to match NULL cells.
pub fn filter_batches(
    predicate: &str,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    filter_batches_impl(predicate, batches, false)
}

/// Like [`filter_batches`] but keeps rows that **do not** match (inverted mask).
///
/// Follows SQL `DELETE ... WHERE` semantics: rows whose predicate evaluates to
/// FALSE *or* UNKNOWN do not match, so both are kept by the inverted filter.
pub fn filter_batches_inverted(
    predicate: &str,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    filter_batches_impl(predicate, batches, true)
}

/// Evaluate `predicate` against a single batch, returning one boolean per row.
///
/// A row is `true` only when the predicate is definitely TRUE (SQL three-valued
/// logic: FALSE and UNKNOWN both yield `false`), matching `WHERE`/`filter_batches`
/// semantics. Unlike [`filter_batches`], the row identity (index) is preserved,
/// which is what a conditional `merge_insert` needs to decide, per target row,
/// whether an update is permitted.
pub fn evaluate_predicate_mask(
    predicate: &str,
    batch: &RecordBatch,
) -> Result<Vec<bool>, HirnDbError> {
    let expr = parse_filter_expr(predicate)?;
    let tri_mask = eval_expr(&expr, batch)?;
    Ok(tri_mask.into_iter().map(|v| v == Some(true)).collect())
}

fn filter_batches_impl(
    predicate: &str,
    batches: &[RecordBatch],
    invert: bool,
) -> Result<Vec<RecordBatch>, HirnDbError> {
    let expr = parse_filter_expr(predicate)?;
    let mut result = Vec::new();
    for batch in batches {
        // Three-valued predicate result per row: Some(true) / Some(false) /
        // None (UNKNOWN). Only TRUE rows match; inversion keeps everything else.
        let tri_mask = eval_expr(&expr, batch)?;
        let final_mask = BooleanArray::from(
            tri_mask
                .iter()
                .map(|v| {
                    let matched = *v == Some(true);
                    Some(if invert { !matched } else { matched })
                })
                .collect::<Vec<_>>(),
        );
        let filtered = arrow_select::filter::filter_record_batch(batch, &final_mask)
            .map_err(HirnDbError::ArrowError)?;
        if filtered.num_rows() > 0 {
            result.push(filtered);
        }
    }
    Ok(result)
}

/// A parsed filter expression tree.
#[derive(Debug)]
enum FilterExpr {
    Comparison {
        column: String,
        op: CmpOp,
        value: String,
    },
    In {
        column: String,
        values: Vec<String>,
    },
    IsNull {
        column: String,
        negated: bool,
    },
    Not(Box<FilterExpr>),
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
}

#[derive(Debug, Clone, Copy)]
enum CmpOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Ge,
    Le,
}

fn column_for_expr<'a>(batch: &'a RecordBatch, column: &str) -> Result<&'a ArrayRef, HirnDbError> {
    let index = batch
        .schema()
        .index_of(column)
        .map_err(|_| HirnDbError::InvalidPredicate(format!("column `{column}` not in schema")))?;
    Ok(batch.column(index))
}

/// Evaluate a filter expression against a RecordBatch, producing one
/// three-valued result per row (`None` = SQL UNKNOWN).
fn eval_expr(expr: &FilterExpr, batch: &RecordBatch) -> Result<Vec<Option<bool>>, HirnDbError> {
    match expr {
        FilterExpr::Comparison { column, op, value } => {
            let col = column_for_expr(batch, column)?;
            let mut bits = Vec::with_capacity(batch.num_rows());
            for row in 0..batch.num_rows() {
                // NULL operand → UNKNOWN, never a match (SQL three-valued logic).
                let Some(cell) = array_value_to_string(col, row) else {
                    bits.push(None);
                    continue;
                };
                let ordering = cmp_values(&cell, value, col.data_type().is_numeric());
                let matched = match op {
                    CmpOp::Eq => ordering == std::cmp::Ordering::Equal,
                    CmpOp::Ne => ordering != std::cmp::Ordering::Equal,
                    CmpOp::Gt => ordering == std::cmp::Ordering::Greater,
                    CmpOp::Lt => ordering == std::cmp::Ordering::Less,
                    CmpOp::Ge => ordering != std::cmp::Ordering::Less,
                    CmpOp::Le => ordering != std::cmp::Ordering::Greater,
                };
                bits.push(Some(matched));
            }
            Ok(bits)
        }
        FilterExpr::In { column, values } => {
            let col = column_for_expr(batch, column)?;
            let value_set: std::collections::HashSet<&str> =
                values.iter().map(String::as_str).collect();
            let mut bits = Vec::with_capacity(batch.num_rows());
            for row in 0..batch.num_rows() {
                bits.push(
                    array_value_to_string(col, row).map(|cell| value_set.contains(cell.as_str())),
                );
            }
            Ok(bits)
        }
        FilterExpr::IsNull { column, negated } => {
            let col = column_for_expr(batch, column)?;
            Ok((0..batch.num_rows())
                .map(|row| Some(col.is_null(row) != *negated))
                .collect())
        }
        FilterExpr::Not(inner) => {
            // Kleene NOT: UNKNOWN stays UNKNOWN.
            let inner = eval_expr(inner, batch)?;
            Ok(inner.into_iter().map(|v| v.map(|b| !b)).collect())
        }
        FilterExpr::And(lhs, rhs) => {
            let l = eval_expr(lhs, batch)?;
            let r = eval_expr(rhs, batch)?;
            Ok(l.into_iter()
                .zip(r)
                .map(|(a, b)| match (a, b) {
                    // Kleene AND: FALSE dominates, then UNKNOWN.
                    (Some(false), _) | (_, Some(false)) => Some(false),
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                })
                .collect())
        }
        FilterExpr::Or(lhs, rhs) => {
            let l = eval_expr(lhs, batch)?;
            let r = eval_expr(rhs, batch)?;
            Ok(l.into_iter()
                .zip(r)
                .map(|(a, b)| match (a, b) {
                    // Kleene OR: TRUE dominates, then UNKNOWN.
                    (Some(true), _) | (_, Some(true)) => Some(true),
                    (Some(false), Some(false)) => Some(false),
                    _ => None,
                })
                .collect())
        }
    }
}

/// Compare a column cell value `a` to a filter literal `b`.
///
/// Numeric coercion is applied **only when the column is numeric** (R-43): a
/// numeric column compares by value so `col = 1.0` matches the stored integer
/// `1` and `col > 9` correctly orders `10` after `9`. A `Utf8` (string) column
/// is compared lexicographically and is never coerced to `f64` — otherwise a
/// text cell like `"1.0"`, `"1e3"`, or `"007"` would spuriously match a numeric
/// literal, and the Lance backend (which compares typed values) would disagree.
fn cmp_values(a: &str, b: &str, column_is_numeric: bool) -> std::cmp::Ordering {
    if column_is_numeric && let (Ok(x), Ok(y)) = (a.parse::<f64>(), b.parse::<f64>()) {
        return x.partial_cmp(&y).unwrap_or(std::cmp::Ordering::Equal);
    }
    a.cmp(b)
}

/// Stringify the cell at `row`, or `None` when the cell is NULL.
fn array_value_to_string(array: &ArrayRef, row: usize) -> Option<String> {
    if array.is_null(row) {
        return None;
    }
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        return Some(a.value(row).to_string());
    }
    if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
        return Some(a.value(row).to_string());
    }
    if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        return Some(a.value(row).to_string());
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(a.value(row).to_string());
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt32Array>() {
        return Some(a.value(row).to_string());
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        return Some(a.value(row).to_string());
    }
    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        return Some(a.value(row).to_string());
    }
    if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
        return Some(a.value(row).to_string());
    }
    Some(format!("{:?}", array.slice(row, 1)))
}

// ── Recursive-descent filter parser ───────────────────────────────────
//
// Grammar:
//   expr     → or_expr
//   or_expr  → and_expr ( "OR" and_expr )*
//   and_expr → atom ( "AND" atom )*
//   atom     → "(" expr ")" | "NOT" atom | comparison
//   comparison → IDENT OP VALUE
//              | IDENT "IN" "(" VALUE ("," VALUE)* ")"
//              | IDENT "IS" ["NOT"] "NULL"
//
// Tokens are whitespace-separated, except operators which may abut values.

fn parse_filter_expr(input: &str) -> Result<FilterExpr, HirnDbError> {
    let tokens = tokenize(input)?;
    let mut pos = 0;
    let expr = parse_or(&tokens, &mut pos)?;
    if pos < tokens.len() {
        return Err(HirnDbError::InvalidPredicate(format!(
            "unexpected token at position {pos}: {:?}",
            tokens[pos]
        )));
    }
    Ok(expr)
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Ident(String),
    StringLit(String),
    Op(String),
    LParen,
    RParen,
    Comma,
    And,
    Or,
    In,
    Is,
    Not,
    Null,
}

fn tokenize(input: &str) -> Result<Vec<Token>, HirnDbError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // Skip whitespace
        if chars[i].is_whitespace() {
            i += 1;
            continue;
        }
        // Parentheses
        if chars[i] == '(' {
            tokens.push(Token::LParen);
            i += 1;
            continue;
        }
        if chars[i] == ')' {
            tokens.push(Token::RParen);
            i += 1;
            continue;
        }
        // Comma (for IN lists)
        if chars[i] == ',' {
            tokens.push(Token::Comma);
            i += 1;
            continue;
        }
        // String literal — SQL standard: '' is an escaped single quote
        if chars[i] == '\'' {
            i += 1;
            let mut s = String::new();
            loop {
                if i >= chars.len() {
                    break;
                }
                if chars[i] == '\'' {
                    // Check for '' escape sequence
                    if i + 1 < chars.len() && chars[i + 1] == '\'' {
                        s.push('\'');
                        i += 2;
                    } else {
                        i += 1; // skip closing quote
                        break;
                    }
                } else {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            tokens.push(Token::StringLit(s));
            continue;
        }
        // Operators: !=, <>, >=, <=, =, >, <
        if chars[i] == '!' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push(Token::Op("!=".to_string()));
            i += 2;
            continue;
        }
        if chars[i] == '<' && i + 1 < chars.len() && chars[i + 1] == '>' {
            tokens.push(Token::Op("!=".to_string()));
            i += 2;
            continue;
        }
        if chars[i] == '>' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push(Token::Op(">=".to_string()));
            i += 2;
            continue;
        }
        if chars[i] == '<' && i + 1 < chars.len() && chars[i + 1] == '=' {
            tokens.push(Token::Op("<=".to_string()));
            i += 2;
            continue;
        }
        if chars[i] == '=' {
            tokens.push(Token::Op("=".to_string()));
            i += 1;
            continue;
        }
        if chars[i] == '>' {
            tokens.push(Token::Op(">".to_string()));
            i += 1;
            continue;
        }
        if chars[i] == '<' {
            tokens.push(Token::Op("<".to_string()));
            i += 1;
            continue;
        }
        // Identifier or keyword (AND/OR)
        if chars[i].is_alphanumeric() || chars[i] == '_' {
            let start = i;
            while i < chars.len()
                && (chars[i].is_alphanumeric() || chars[i] == '_' || chars[i] == '.')
            {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            match word.to_uppercase().as_str() {
                "AND" => tokens.push(Token::And),
                "OR" => tokens.push(Token::Or),
                "IN" => tokens.push(Token::In),
                "IS" => tokens.push(Token::Is),
                "NOT" => tokens.push(Token::Not),
                "NULL" => tokens.push(Token::Null),
                _ => tokens.push(Token::Ident(word)),
            }
            continue;
        }
        // Unrecognized
        return Err(HirnDbError::InvalidPredicate(format!(
            "unexpected character '{}' in filter",
            chars[i]
        )));
    }
    Ok(tokens)
}

fn parse_or(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, HirnDbError> {
    let mut left = parse_and(tokens, pos)?;
    while *pos < tokens.len() && tokens[*pos] == Token::Or {
        *pos += 1;
        let right = parse_and(tokens, pos)?;
        left = FilterExpr::Or(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_and(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, HirnDbError> {
    let mut left = parse_atom(tokens, pos)?;
    while *pos < tokens.len() && tokens[*pos] == Token::And {
        *pos += 1;
        let right = parse_atom(tokens, pos)?;
        left = FilterExpr::And(Box::new(left), Box::new(right));
    }
    Ok(left)
}

fn parse_atom(tokens: &[Token], pos: &mut usize) -> Result<FilterExpr, HirnDbError> {
    if *pos >= tokens.len() {
        return Err(HirnDbError::InvalidPredicate(
            "unexpected end of filter expression".to_string(),
        ));
    }
    // Parenthesized expression
    if tokens[*pos] == Token::LParen {
        *pos += 1;
        let expr = parse_or(tokens, pos)?;
        if *pos >= tokens.len() || tokens[*pos] != Token::RParen {
            return Err(HirnDbError::InvalidPredicate(
                "missing closing parenthesis".to_string(),
            ));
        }
        *pos += 1;
        return Ok(expr);
    }
    // Unary NOT (binds tighter than AND/OR, looser than comparisons)
    if tokens[*pos] == Token::Not {
        *pos += 1;
        let inner = parse_atom(tokens, pos)?;
        return Ok(FilterExpr::Not(Box::new(inner)));
    }
    // Comparison: IDENT OP VALUE, IDENT IN (value, ...), or IDENT IS [NOT] NULL
    let column = match &tokens[*pos] {
        Token::Ident(s) => s.clone(),
        other => {
            return Err(HirnDbError::InvalidPredicate(format!(
                "expected column name, got {other:?}"
            )));
        }
    };
    *pos += 1;
    if *pos >= tokens.len() {
        return Err(HirnDbError::InvalidPredicate(format!(
            "expected operator after `{column}`"
        )));
    }

    // Handle IS [NOT] NULL
    if tokens[*pos] == Token::Is {
        *pos += 1;
        let negated = if *pos < tokens.len() && tokens[*pos] == Token::Not {
            *pos += 1;
            true
        } else {
            false
        };
        if *pos >= tokens.len() || tokens[*pos] != Token::Null {
            return Err(HirnDbError::InvalidPredicate(format!(
                "expected NULL after IS for `{column}`"
            )));
        }
        *pos += 1;
        return Ok(FilterExpr::IsNull { column, negated });
    }

    // Handle IN (value, value, ...)
    if tokens[*pos] == Token::In {
        *pos += 1;
        if *pos >= tokens.len() || tokens[*pos] != Token::LParen {
            return Err(HirnDbError::InvalidPredicate(
                "expected '(' after IN".to_string(),
            ));
        }
        *pos += 1; // skip '('
        let mut values = Vec::new();
        loop {
            if *pos >= tokens.len() {
                return Err(HirnDbError::InvalidPredicate(
                    "unexpected end of IN list".to_string(),
                ));
            }
            if tokens[*pos] == Token::RParen {
                *pos += 1;
                break;
            }
            match &tokens[*pos] {
                Token::StringLit(s) => values.push(s.clone()),
                Token::Ident(s) => values.push(s.clone()),
                other => {
                    return Err(HirnDbError::InvalidPredicate(format!(
                        "expected value in IN list, got {other:?}"
                    )));
                }
            }
            *pos += 1;
            // Optional comma
            if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                *pos += 1;
            }
        }
        return Ok(FilterExpr::In { column, values });
    }

    let op = match &tokens[*pos] {
        Token::Op(s) => match s.as_str() {
            "=" => CmpOp::Eq,
            "!=" => CmpOp::Ne,
            ">" => CmpOp::Gt,
            "<" => CmpOp::Lt,
            ">=" => CmpOp::Ge,
            "<=" => CmpOp::Le,
            other => {
                return Err(HirnDbError::InvalidPredicate(format!(
                    "unsupported operator: {other}"
                )));
            }
        },
        other => {
            return Err(HirnDbError::InvalidPredicate(format!(
                "expected operator, got {other:?}"
            )));
        }
    };
    *pos += 1;
    if *pos >= tokens.len() {
        return Err(HirnDbError::InvalidPredicate(format!(
            "expected value after operator for `{column}`"
        )));
    }
    let value = match &tokens[*pos] {
        Token::StringLit(s) => s.clone(),
        Token::Ident(s) => s.clone(),
        other => {
            return Err(HirnDbError::InvalidPredicate(format!(
                "expected value, got {other:?}"
            )));
        }
    };
    *pos += 1;
    Ok(FilterExpr::Comparison { column, op, value })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray, UInt32Array};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn sample_batch(n: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let ids: Vec<i32> = (0..n as i32).collect();
        let names: Vec<String> = (0..n).map(|i| format!("item_{i}")).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(names)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_limit_offset() {
        let batch = sample_batch(10);
        let result = apply_limit_offset(&[batch], Some(3), Some(2));
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 3);
    }

    /// R-43: a `Utf8` column value must not be numerically coerced when compared
    /// to a numeric-looking literal. Text cells `"1.0"` / `"01"` must NOT match
    /// `code = '1'`; only the exact string `"1"` matches, matching Lance's typed
    /// comparison.
    #[test]
    fn utf8_column_is_not_coerced_to_number() {
        let schema = Arc::new(Schema::new(vec![Field::new("code", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(StringArray::from(vec!["1", "1.0", "01", "2"]))],
        )
        .unwrap();

        let matched = filter_batches("code = '1'", std::slice::from_ref(&batch)).unwrap();
        let total = total_row_count(&matched);
        assert_eq!(
            total, 1,
            "only the exact string '1' matches, not '1.0'/'01'"
        );
        let codes = matched[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(codes.value(0), "1");
    }

    /// R-43: numeric columns still compare by value (ordering, not lexicographic),
    /// so `n > 9` correctly orders `10` after `9`.
    #[test]
    fn numeric_column_compares_by_value() {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![2, 9, 10, 100]))],
        )
        .unwrap();

        let matched = filter_batches("n > 9", std::slice::from_ref(&batch)).unwrap();
        // Lexicographic would wrongly drop "10"/"100" (they sort before "9").
        assert_eq!(
            total_row_count(&matched),
            2,
            "10 and 100 are > 9 numerically"
        );
    }

    #[test]
    fn test_limit_only() {
        let batch = sample_batch(10);
        let result = apply_limit_offset(&[batch], Some(5), None);
        assert_eq!(result[0].num_rows(), 5);
    }

    #[test]
    fn test_offset_only() {
        let batch = sample_batch(10);
        let result = apply_limit_offset(&[batch], None, Some(7));
        assert_eq!(result[0].num_rows(), 3);
    }

    #[test]
    fn test_apply_ordering() {
        let batch = sample_batch(5);
        let ordered = apply_ordering(&[batch], &[ScanOrdering::desc("name")]).unwrap();
        let names = ordered[0]
            .column_by_name("name")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(names.value(0), "item_4");
        assert_eq!(names.value(1), "item_3");
        assert_eq!(names.value(4), "item_0");
    }

    #[test]
    fn test_project() {
        let batch = sample_batch(5);
        let projected = project_batches(&[batch], &["name".to_string()]).unwrap();
        assert_eq!(projected[0].num_columns(), 1);
        assert_eq!(projected[0].schema().field(0).name(), "name");
    }

    #[test]
    fn test_project_missing_column() {
        let batch = sample_batch(5);
        let err = project_batches(&[batch], &["missing".to_string()]).unwrap_err();
        assert!(matches!(err, HirnDbError::InvalidArgument(_)));
    }

    #[test]
    fn test_total_row_count() {
        let b1 = sample_batch(5);
        let b2 = sample_batch(3);
        assert_eq!(total_row_count(&[b1, b2]), 8);
    }

    #[test]
    fn test_concat_batches() {
        let b1 = sample_batch(3);
        let b2 = sample_batch(2);
        let schema = b1.schema();
        let combined = concat_batches(&schema, &[b1, b2]).unwrap().unwrap();
        assert_eq!(combined.num_rows(), 5);
    }

    #[test]
    fn test_concat_empty() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let combined = concat_batches(&schema, &[]).unwrap();
        assert!(combined.is_none());
    }

    fn edge_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("source", DataType::Utf8, false),
            Field::new("target", DataType::Utf8, false),
            Field::new("relation", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "a", "c"])),
                Arc::new(StringArray::from(vec!["b", "c", "c", "a"])),
                Arc::new(StringArray::from(vec![
                    "causes",
                    "causes",
                    "contradicts",
                    "contradicts",
                ])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_filter_simple_eq() {
        let batch = edge_batch();
        let result = filter_batches("relation = 'contradicts'", &[batch]).unwrap();
        assert_eq!(result[0].num_rows(), 2);
    }

    #[test]
    fn test_filter_and() {
        let batch = edge_batch();
        let result = filter_batches("source = 'a' AND relation = 'contradicts'", &[batch]).unwrap();
        assert_eq!(result[0].num_rows(), 1);
    }

    #[test]
    fn test_filter_or() {
        let batch = edge_batch();
        let result = filter_batches("source = 'a' OR target = 'a'", &[batch]).unwrap();
        assert_eq!(result[0].num_rows(), 3);
    }

    #[test]
    fn test_filter_and_or_grouped() {
        let batch = edge_batch();
        let result = filter_batches(
            "(source = 'a' OR target = 'a') AND relation = 'contradicts'",
            &[batch],
        )
        .unwrap();
        assert_eq!(result[0].num_rows(), 2);
    }

    #[test]
    fn test_filter_no_match() {
        let batch = edge_batch();
        let result = filter_batches("relation = 'derived_from'", &[batch]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_in() {
        let batch = edge_batch();
        let result = filter_batches("source IN ('a', 'c')", &[batch]).unwrap();
        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3); // rows with source = 'a' (2) + source = 'c' (1)
    }

    #[test]
    fn test_filter_in_single_value() {
        let batch = edge_batch();
        let result = filter_batches("source IN ('b')", &[batch]).unwrap();
        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn test_filter_in_combined_with_and() {
        let batch = edge_batch();
        let result =
            filter_batches("source IN ('a', 'b') AND relation = 'causes'", &[batch]).unwrap();
        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2); // a->b causes, b->c causes
    }

    #[test]
    fn test_filter_in_empty_result() {
        let batch = edge_batch();
        let result = filter_batches("source IN ('x', 'y', 'z')", &[batch]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_filter_uint32_equality() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("memory_id", DataType::Utf8, false),
            Field::new("blob_index", DataType::UInt32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["m1", "m1", "m2"])),
                Arc::new(UInt32Array::from(vec![0_u32, 1_u32, 0_u32])),
            ],
        )
        .unwrap();

        let result = filter_batches("memory_id = 'm1' AND blob_index = 1", &[batch]).unwrap();
        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 1);
    }

    fn nullable_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("note", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::Int64Array::from(vec![1_i64, 2, 3, 4])),
                Arc::new(StringArray::from(vec![
                    Some("alpha"),
                    None,
                    Some(""),
                    Some("beta"),
                ])),
            ],
        )
        .unwrap()
    }

    fn collect_ids(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|batch| {
                let ids = batch
                    .column_by_name("id")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<arrow_array::Int64Array>()
                    .unwrap();
                (0..batch.num_rows())
                    .map(|row| ids.value(row))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn test_filter_eq_empty_string_does_not_match_null() {
        let batch = nullable_batch();
        let result = filter_batches("note = ''", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![3]);
    }

    #[test]
    fn test_filter_ne_excludes_null() {
        let batch = nullable_batch();
        let result = filter_batches("note != 'alpha'", &[batch]).unwrap();
        // NULL comparison is UNKNOWN → excluded; only definite non-matches remain.
        assert_eq!(collect_ids(&result), vec![3, 4]);
    }

    #[test]
    fn test_filter_in_excludes_null() {
        let batch = nullable_batch();
        let result = filter_batches("note IN ('alpha', 'beta')", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![1, 4]);
    }

    #[test]
    fn test_filter_is_null() {
        let batch = nullable_batch();
        let result = filter_batches("note IS NULL", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![2]);
    }

    #[test]
    fn test_filter_is_not_null() {
        let batch = nullable_batch();
        let result = filter_batches("note IS NOT NULL", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![1, 3, 4]);
    }

    #[test]
    fn test_filter_not() {
        let batch = nullable_batch();
        // NOT over UNKNOWN stays UNKNOWN, so the NULL row remains excluded.
        let result = filter_batches("NOT note = 'alpha'", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![3, 4]);

        let batch = nullable_batch();
        let result = filter_batches("NOT (note IS NULL)", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![1, 3, 4]);
    }

    #[test]
    fn test_filter_inverted_keeps_null_rows() {
        let batch = nullable_batch();
        // DELETE-style inversion: rows where the predicate is FALSE or UNKNOWN
        // are kept, so the NULL row survives.
        let result = filter_batches_inverted("note = 'alpha'", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![2, 3, 4]);
    }

    #[test]
    fn test_filter_numeric_eq_across_representations() {
        let batch = nullable_batch();
        // Float literal matches integer cell numerically.
        let result = filter_batches("id = 1.0", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![1]);

        let batch = nullable_batch();
        let result = filter_batches("id != 1.0", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![2, 3, 4]);
    }

    #[test]
    fn test_filter_null_and_or_kleene() {
        let batch = nullable_batch();
        // UNKNOWN OR TRUE → TRUE: the NULL row matches through the id branch.
        let result = filter_batches("note = 'alpha' OR id = 2", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![1, 2]);

        let batch = nullable_batch();
        // UNKNOWN AND TRUE → UNKNOWN: the NULL row never matches.
        let result = filter_batches("note != 'x' AND id >= 1", &[batch]).unwrap();
        assert_eq!(collect_ids(&result), vec![1, 3, 4]);
    }

    #[test]
    fn test_apply_exact_utf8_filter() {
        let batch = edge_batch();
        let result = apply_scan_options(
            &[batch],
            &ScanOptions {
                exact_filter: Some(ExactMatchFilter::Utf8In {
                    column: "source".to_string(),
                    values: vec!["a".to_string(), "c".to_string()],
                }),
                ..Default::default()
            },
        )
        .unwrap();

        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 3);
    }

    #[test]
    fn test_apply_exact_utf8_filter_then_sql_filter() {
        let batch = edge_batch();
        let result = apply_scan_options(
            &[batch],
            &ScanOptions {
                exact_filter: Some(ExactMatchFilter::Utf8In {
                    column: "source".to_string(),
                    values: vec!["a".to_string(), "c".to_string()],
                }),
                filter: Some("relation = 'contradicts'".to_string()),
                ..Default::default()
            },
        )
        .unwrap();

        let total: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 2);
    }
}
