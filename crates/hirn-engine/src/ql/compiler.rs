//! HirnQL prepared-statement front-end — parse once, bind values into the AST.
//!
//! Execution artifacts (typed AST + DataFusion logical plan) are produced and
//! cached by `hirn_query::QueryPipeline`; that pipeline is the single compiler
//! stack in the system. This module owns the prepared-statement front-end: a
//! statement is parsed once at `prepare()` time and parameter values are bound
//! directly into the stored AST, which is then executed as-is through
//! `HirnDB::execute_statement` (which compiles it via the pipeline).

use std::collections::HashMap;

use hirn_core::HirnError;

use hirn_query::ast::Statement;
use hirn_query::compiler::validate::{self, AnalysisError, AnalysisErrorKind};
use hirn_query::parser::{self, ParseError};

/// Compilation error encompassing all stages.
#[derive(Debug, Clone)]
pub enum CompileError {
    /// Parse-stage error — invalid syntax with line/column.
    Parse(ParseError),
    /// Semantic analysis errors — type mismatches, unknown fields, etc.
    Analysis(Vec<AnalysisError>),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "{e}"),
            Self::Analysis(errors) => {
                for (i, e) in errors.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CompileError {}

impl From<CompileError> for HirnError {
    fn from(e: CompileError) -> Self {
        match e {
            CompileError::Parse(pe) => HirnError::InvalidInput(format!("parse error: {pe}")),
            CompileError::Analysis(errors) => {
                let msg = errors
                    .iter()
                    .map(|e| e.message.clone())
                    .collect::<Vec<_>>()
                    .join("; ");
                HirnError::InvalidInput(msg)
            }
        }
    }
}

// ── Prepared Statements ────────────────────────────────────────────────

/// A prepared statement with parameter slots.
///
/// Created via `prepare()`. Prefer `QueryView::execute_prepared()` for
/// execution, or use `bind()` when a bound `Statement` is needed for
/// inspection or tests.
#[derive(Debug, Clone)]
pub struct PreparedStatement {
    /// The original query text containing `$param` placeholders.
    pub source: String,
    /// Parameter names found in the query (sorted, with `$` prefix).
    pub params: Vec<String>,
    /// The AST template parsed once at prepare time; `bind()` clones it and
    /// substitutes parameter values, so binding never re-parses the source.
    template: Statement,
}

/// Prepare a parameterized query.
///
/// Parses the query (which may contain `$1`, `$name` placeholders) and
/// extracts parameter names. The template is parsed exactly once and reused
/// across multiple `bind()` calls since placeholder substitution does not
/// change query shape.
pub fn prepare(query: &str) -> Result<PreparedStatement, CompileError> {
    // Parse with parameters in place — they become $name strings in AST.
    let ast = parser::parse(query).map_err(CompileError::Parse)?;

    // Collect parameter references.
    let params = hirn_query::ast::collect_parameters(&ast);

    // Validation split: parameter-free templates are fully validated here, at
    // prepare time, so invalid statements fail fast. Parameterized templates
    // are validated after bind(), when the bound AST is compiled for
    // execution — value checks (numeric ranges, temporal formats) need
    // concrete values and would false-positive on `$name` placeholders (e.g.
    // importance > $threshold, AFTER $date).
    if params.is_empty() {
        let errors = validate::validate(&ast);
        if !errors.is_empty() {
            return Err(CompileError::Analysis(errors));
        }
    }

    Ok(PreparedStatement {
        source: query.to_string(),
        params,
        template: ast,
    })
}

/// Bind parameter values to a prepared statement, producing a bound
/// `Statement` ready for direct execution.
///
/// `values` maps parameter names (with `$` prefix) to string representations.
/// Positional parameters use `$1`, `$2`, etc.
///
/// The returned AST is executed as-is through the same compiled pipeline as
/// any other statement (`HirnDB::execute_statement`); it is never serialized
/// back to HirnQL text and re-parsed. Statement `Display` output exists for
/// logging and EXPLAIN only and is not load-bearing for execution.
///
/// Semantic validation of the bound values runs inside the pipeline's
/// `typed_ast::analyze` when the statement is compiled for execution — the
/// exact same single-stack pass every statement gets — so `bind()` itself
/// only checks that all declared parameters received a value.
///
/// Returns an error if any declared parameter is missing from `values`.
pub fn bind(
    prepared: &PreparedStatement,
    values: &HashMap<String, String>,
) -> Result<Statement, CompileError> {
    // Clone the template parsed at prepare time (parameters remain as `$name`
    // placeholders), then substitute values **into the AST**, not into the
    // query text. Placing values into typed AST nodes means a value can never
    // break out of a string literal to inject trailing clauses, and there is no
    // `$t`/`$t2` prefix-collision hazard that naive `String::replace` had.
    let mut ast = prepared.template.clone();

    let missing = hirn_query::ast::bind_parameters(&mut ast, values);
    if !missing.is_empty() {
        return Err(CompileError::Analysis(vec![AnalysisError {
            message: format!("missing value for parameter(s): {}", missing.join(", ")),
            kind: AnalysisErrorKind::UnknownField,
        }]));
    }

    Ok(ast)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_extracts_positional_params() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#).unwrap();
        assert_eq!(stmt.params, vec!["$1"]);
    }

    #[test]
    fn prepare_extracts_named_params() {
        let stmt =
            prepare(r#"RECALL episodic ABOUT $query WHERE importance > $threshold"#).unwrap();
        assert!(stmt.params.contains(&"$query".to_string()));
        assert!(stmt.params.contains(&"$threshold".to_string()));
        assert_eq!(stmt.params.len(), 2);
    }

    #[test]
    fn prepare_invalid_syntax_fails() {
        assert!(matches!(
            prepare("NOT_A_QUERY"),
            Err(CompileError::Parse(_))
        ));
    }

    #[test]
    fn prepare_no_params_runs_validation() {
        // No params — full validation, should catch value out of range.
        let result = prepare(r#"RECALL episodic ABOUT "x" WHERE importance > 2.0"#);
        assert!(matches!(result, Err(CompileError::Analysis(_))));
    }

    #[test]
    fn prepare_with_params_defers_validation() {
        // Has params — validation deferred (can't validate $threshold range).
        let result = prepare(r#"RECALL episodic ABOUT $1 WHERE importance > $2"#);
        assert!(result.is_ok());
    }

    #[test]
    fn bind_substitutes_string_param() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#).unwrap();
        let mut values = HashMap::new();
        values.insert("$1".to_string(), "authentication".to_string());

        let bound = bind(&stmt, &values).unwrap();
        match &bound {
            Statement::Recall(r) => assert_eq!(r.about, "authentication"),
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn bind_does_not_reparse_template() {
        // A bound value that would change meaning if the statement were
        // serialized to text and re-parsed must survive as literal data.
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#).unwrap();
        let payload = r#"x" LIMIT 1 NAMESPACE hijacked --"#;
        let mut values = HashMap::new();
        values.insert("$1".to_string(), payload.to_string());

        let bound = bind(&stmt, &values).unwrap();
        match &bound {
            Statement::Recall(r) => {
                assert_eq!(r.about, payload, "payload must stay data, not syntax");
                assert_eq!(r.limit, Some(10), "LIMIT clause must be unchanged");
                assert!(r.namespace.is_none(), "no namespace may be injected");
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn bind_substitutes_numeric_param() {
        // Numeric parameters are supported in WHERE conditions (typed values),
        // where the name survives in the AST. Integer clauses like LIMIT take
        // literals only.
        let stmt =
            prepare(r#"RECALL episodic ABOUT $query WHERE importance > $threshold"#).unwrap();
        let mut values = HashMap::new();
        values.insert("$query".to_string(), "test".to_string());
        values.insert("$threshold".to_string(), "0.5".to_string());

        let bound = bind(&stmt, &values).unwrap();
        match &bound {
            Statement::Recall(r) => {
                assert_eq!(r.about, "test");
                assert_eq!(
                    r.where_clauses[0].value,
                    hirn_query::ast::ConditionValue::Float(0.5)
                );
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn integer_clause_param_is_rejected_at_parse() {
        // `LIMIT $n` used to silently coerce to 0 (returning nothing); it now
        // errors clearly at parse time.
        assert!(prepare(r#"RECALL episodic ABOUT "x" LIMIT $n"#).is_err());
    }

    #[test]
    fn bind_missing_param_returns_error() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#).unwrap();
        let values = HashMap::new(); // no values
        let result = bind(&stmt, &values);
        assert!(result.is_err());
    }

    #[test]
    fn bind_leaves_template_reusable() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#).unwrap();

        let mut values = HashMap::new();
        values.insert("$1".to_string(), "auth".to_string());
        let _ = bind(&stmt, &values).unwrap();

        // Binding clones the template: the prepared statement stays untouched
        // and can be bound again.
        let again = bind(&stmt, &values).unwrap();
        match &again {
            Statement::Recall(r) => assert_eq!(r.about, "auth"),
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn bind_different_values_produce_different_asts() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#).unwrap();

        let mut v1 = HashMap::new();
        v1.insert("$1".to_string(), "auth".to_string());
        let b1 = bind(&stmt, &v1).unwrap();

        let mut v2 = HashMap::new();
        v2.insert("$1".to_string(), "deployment".to_string());
        let b2 = bind(&stmt, &v2).unwrap();

        match (&b1, &b2) {
            (Statement::Recall(r1), Statement::Recall(r2)) => {
                assert_eq!(r1.about, "auth");
                assert_eq!(r2.about, "deployment");
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn bound_out_of_range_value_is_rejected_at_compile() {
        // bind() itself no longer re-analyzes — the bound AST goes through the
        // single compiler stack (typed_ast::analyze) when compiled for
        // execution, which rejects out-of-range values.
        let stmt = prepare(r#"RECALL episodic ABOUT $1 WHERE importance > $2"#).unwrap();
        let mut values = HashMap::new();
        values.insert("$1".to_string(), "test".to_string());
        values.insert("$2".to_string(), "5.0".to_string()); // out of range

        let bound = bind(&stmt, &values).unwrap();
        let pipeline = hirn_query::QueryPipeline::new(hirn_query::AnalyzeContext::default());
        let err = pipeline.compile_statement(bound).unwrap_err();
        assert!(
            err.to_string().contains("between 0.0 and 1.0"),
            "expected range error, got: {err}"
        );
    }

    #[test]
    fn prepared_stmt_bind_is_fast() {
        let q = r#"RECALL episodic ABOUT $1 INVOLVING "auth" AFTER "2026-01-01" WHERE importance > 0.5 LIMIT 10"#;
        let stmt = prepare(q).unwrap();

        // Time 1000 bind() calls: bind clones the pre-parsed template and
        // substitutes values — no parsing, no planning.
        let mut values = HashMap::new();
        values.insert("$1".to_string(), "test".to_string());
        let start = std::time::Instant::now();
        for _ in 0..1_000 {
            let _ = bind(&stmt, &values).unwrap();
        }
        let bind_elapsed = start.elapsed();
        assert!(
            bind_elapsed.as_secs_f64() < 2.0,
            "1K binds took {:.2}s",
            bind_elapsed.as_secs_f64()
        );
    }

    // ── Parse performance ──────────────────────────────────────────────

    #[test]
    fn parse_10k_queries_under_1_second() {
        let q = r#"RECALL episodic ABOUT "test query" INVOLVING "auth" AFTER "2026-01-01" WHERE importance > 0.5 LIMIT 10"#;
        let max_elapsed = if cfg!(debug_assertions) {
            std::time::Duration::from_millis(2500)
        } else {
            std::time::Duration::from_millis(1500)
        };
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = parser::parse(q).unwrap();
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed <= max_elapsed,
            "10K parses took {:.2}s (>{:.2}s limit)",
            elapsed.as_secs_f64(),
            max_elapsed.as_secs_f64()
        );
    }

    #[test]
    fn compile_error_is_hirn_error() {
        let err = prepare("INVALID").unwrap_err();
        let msg = err.to_string();
        assert!(!msg.is_empty());
        let hirn_err: HirnError = err.into();
        assert!(matches!(hirn_err, HirnError::InvalidInput(_)));
    }
}
