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
/// - `expr AND expr`
/// - `expr OR expr`
/// - Parenthesized grouping: `(expr)`
pub fn filter_batches(
    predicate: &str,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    filter_batches_impl(predicate, batches, false)
}

/// Like [`filter_batches`] but keeps rows that **do not** match (inverted mask).
pub fn filter_batches_inverted(
    predicate: &str,
    batches: &[RecordBatch],
) -> Result<Vec<RecordBatch>, HirnDbError> {
    filter_batches_impl(predicate, batches, true)
}

fn filter_batches_impl(
    predicate: &str,
    batches: &[RecordBatch],
    invert: bool,
) -> Result<Vec<RecordBatch>, HirnDbError> {
    let expr = parse_filter_expr(predicate)?;
    let mut result = Vec::new();
    for batch in batches {
        let mask = eval_expr(&expr, batch)?;
        let final_mask = if invert {
            BooleanArray::from(
                mask.iter()
                    .map(|v| Some(!v.unwrap_or(false)))
                    .collect::<Vec<_>>(),
            )
        } else {
            mask
        };
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

/// Evaluate a filter expression against a RecordBatch, producing a boolean mask.
fn eval_expr(expr: &FilterExpr, batch: &RecordBatch) -> Result<BooleanArray, HirnDbError> {
    match expr {
        FilterExpr::Comparison { column, op, value } => {
            let schema = batch.schema();
            let col_idx = schema.index_of(column).map_err(|_| {
                HirnDbError::InvalidPredicate(format!("column `{column}` not in schema"))
            })?;
            let col = batch.column(col_idx);
            let mut bits = Vec::with_capacity(batch.num_rows());
            for row in 0..batch.num_rows() {
                let cell = array_value_to_string(col, row);
                let matched = match op {
                    CmpOp::Eq => cell == *value,
                    CmpOp::Ne => cell != *value,
                    CmpOp::Gt => cmp_numeric(&cell, value, |a, b| a > b),
                    CmpOp::Lt => cmp_numeric(&cell, value, |a, b| a < b),
                    CmpOp::Ge => cmp_numeric(&cell, value, |a, b| a >= b),
                    CmpOp::Le => cmp_numeric(&cell, value, |a, b| a <= b),
                };
                bits.push(matched);
            }
            Ok(BooleanArray::from(bits))
        }
        FilterExpr::In { column, values } => {
            let schema = batch.schema();
            let col_idx = schema.index_of(column).map_err(|_| {
                HirnDbError::InvalidPredicate(format!("column `{column}` not in schema"))
            })?;
            let col = batch.column(col_idx);
            let value_set: std::collections::HashSet<&str> =
                values.iter().map(String::as_str).collect();
            let mut bits = Vec::with_capacity(batch.num_rows());
            for row in 0..batch.num_rows() {
                let cell = array_value_to_string(col, row);
                bits.push(value_set.contains(cell.as_str()));
            }
            Ok(BooleanArray::from(bits))
        }
        FilterExpr::And(lhs, rhs) => {
            let l = eval_expr(lhs, batch)?;
            let r = eval_expr(rhs, batch)?;
            let bits: Vec<bool> = (0..batch.num_rows())
                .map(|i| l.value(i) && r.value(i))
                .collect();
            Ok(BooleanArray::from(bits))
        }
        FilterExpr::Or(lhs, rhs) => {
            let l = eval_expr(lhs, batch)?;
            let r = eval_expr(rhs, batch)?;
            let bits: Vec<bool> = (0..batch.num_rows())
                .map(|i| l.value(i) || r.value(i))
                .collect();
            Ok(BooleanArray::from(bits))
        }
    }
}

fn cmp_numeric(a: &str, b: &str, f: fn(f64, f64) -> bool) -> bool {
    a.parse::<f64>()
        .ok()
        .zip(b.parse::<f64>().ok())
        .map(|(x, y)| f(x, y))
        .unwrap_or(false)
}

fn array_value_to_string(array: &ArrayRef, row: usize) -> String {
    if array.is_null(row) {
        return String::new();
    }
    if let Some(a) = array.as_any().downcast_ref::<StringArray>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<BooleanArray>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<Int32Array>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt32Array>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<UInt64Array>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<Float64Array>() {
        return a.value(row).to_string();
    }
    if let Some(a) = array.as_any().downcast_ref::<Float32Array>() {
        return a.value(row).to_string();
    }
    format!("{:?}", array.slice(row, 1))
}

// ── Recursive-descent filter parser ───────────────────────────────────
//
// Grammar:
//   expr     → or_expr
//   or_expr  → and_expr ( "OR" and_expr )*
//   and_expr → atom ( "AND" atom )*
//   atom     → "(" expr ")" | comparison
//   comparison → IDENT OP VALUE
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
    // Comparison: IDENT OP VALUE, or IDENT IN (value, ...)
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
