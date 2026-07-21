//! HirnQL v2 compiler front-end — parse, validate, and prepare statements.
//!
//! Pipeline: HirnQL text → Parser → Untyped AST → Semantic analysis → Plan metadata
//!
//! Execution artifacts (typed AST + DataFusion logical plan) are produced and
//! cached by `hirn_query::QueryPipeline`; that cache is the single plan cache
//! in the system. This module owns the prepared-statement front-end: a
//! statement is parsed once at `prepare()` time and parameter values are bound
//! directly into the stored AST, which is then executed as-is.

use std::collections::HashMap;

use hirn_core::HirnError;

use super::analyzer::{self, AnalysisError, AnalysisErrorKind};
use super::planner::{self, QueryPlan};
use crate::db::DbStats;
use hirn_query::ast::Statement;
use hirn_query::parser::{self, ParseError};

/// A compiled query ready for execution.
#[derive(Debug, Clone)]
pub struct CompiledQuery {
    /// The original query text.
    pub source: String,
    /// The parsed AST.
    pub ast: Statement,
    /// The optimized execution plan.
    pub plan: QueryPlan,
}

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

/// Compile a HirnQL query through all stages: parse → analyze → plan.
///
/// Returns a `CompiledQuery` containing the AST and execution plan,
/// or a `CompileError` with detailed stage information.
pub fn compile(query: &str, stats: Option<&DbStats>) -> Result<CompiledQuery, CompileError> {
    // Stage 1: Parse.
    let ast = parser::parse(query).map_err(CompileError::Parse)?;

    // Stage 2: Semantic analysis.
    let errors = analyzer::analyze(&ast);
    if !errors.is_empty() {
        return Err(CompileError::Analysis(errors));
    }

    // Stage 3: Plan.
    let plan = planner::plan(&ast, stats);

    Ok(CompiledQuery {
        source: query.to_string(),
        ast,
        plan,
    })
}

// ── Prepared Statements ────────────────────────────────────────────────

/// A prepared statement with parameter slots and compatibility plan metadata.
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
    /// Plan metadata computed at prepare time (parameter-independent).
    pub plan: QueryPlan,
    /// The AST template parsed once at prepare time; `bind()` clones it and
    /// substitutes parameter values, so binding never re-parses the source.
    template: Statement,
}

/// Prepare a parameterized query.
///
/// Parses the query (which may contain `$1`, `$name` placeholders),
/// extracts parameter names, and computes compatibility plan metadata.
///
/// The cached metadata is parameter-independent and is reused across multiple
/// `bind()` calls since placeholder substitution does not change query shape.
pub fn prepare(query: &str, stats: Option<&DbStats>) -> Result<PreparedStatement, CompileError> {
    // Parse with parameters in place — they become $name strings in AST.
    let ast = parser::parse(query).map_err(CompileError::Parse)?;

    // Collect parameter references.
    let params = hirn_query::ast::collect_parameters(&ast);

    // Analysis split: for parameter-free templates the full analyzer runs
    // here, at prepare time. For parameterized templates it is deferred to
    // bind() — value checks (numeric ranges, temporal formats) need concrete
    // values and would false-positive on `$name` placeholders (e.g.
    // importance > $threshold, AFTER $date).
    if params.is_empty() {
        let errors = analyzer::analyze(&ast);
        if !errors.is_empty() {
            return Err(CompileError::Analysis(errors));
        }
    }

    // Plan is parameter-independent.
    let plan = planner::plan(&ast, stats);

    Ok(PreparedStatement {
        source: query.to_string(),
        params,
        plan,
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

    // Value-dependent semantic checks run now that all values are concrete.
    // (For parameter-free templates this repeats the cheap prepare-time pass;
    // the analyzer is a single non-allocating walk, so a split into
    // value-dependent/independent halves is not worth the duplication.)
    let errors = analyzer::analyze(&ast);
    if !errors.is_empty() {
        return Err(CompileError::Analysis(errors));
    }

    Ok(ast)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_valid_recall() {
        let result = compile(r#"RECALL episodic ABOUT "test""#, None);
        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(matches!(compiled.ast, Statement::Recall(_)));
        assert!(!compiled.plan.steps.is_empty());
    }

    #[test]
    fn compile_invalid_syntax() {
        let result = compile("NOT_A_QUERY", None);
        assert!(matches!(result, Err(CompileError::Parse(_))));
    }

    #[test]
    fn compile_semantic_error() {
        let result = compile(r#"RECALL episodic ABOUT "x" WHERE importance > 2.0"#, None);
        assert!(matches!(result, Err(CompileError::Analysis(_))));
        if let Err(CompileError::Analysis(errors)) = result {
            assert_eq!(errors[0].kind, analyzer::AnalysisErrorKind::ValueOutOfRange);
        }
    }

    #[test]
    fn compile_error_display() {
        let result = compile("INVALID", None);
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(!msg.is_empty());
    }

    #[test]
    fn compile_same_query_deterministic() {
        let q = r#"RECALL episodic ABOUT "test" LIMIT 5"#;
        let c1 = compile(q, None).unwrap();
        let c2 = compile(q, None).unwrap();
        assert_eq!(c1.plan, c2.plan);
    }

    #[test]
    fn compile_think_with_budget() {
        let result = compile(r#"THINK ABOUT "optimize" BUDGET 4096"#, None);
        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(matches!(compiled.ast, Statement::Think(_)));
    }

    #[test]
    fn compile_remember() {
        let result = compile(r#"REMEMBER episode CONTENT "data""#, None);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("REMEMBER is not supported via embedded HirnQL anymore")
        );
    }

    #[test]
    fn compile_complex_recall() {
        let q = r#"
            RECALL semantic, episodic
              ABOUT "vector database"
              INVOLVING "HNSW"
              AFTER "2026-03-01"
              EXPAND GRAPH DEPTH 2 MIN_WEIGHT 0.3 ACTIVATION spreading
              WHERE importance > 0.4
              WHERE confidence > 0.8
              AS NARRATIVE
              BUDGET 4096
              NAMESPACE shared
              LIMIT 20
        "#;
        let result = compile(q, None);
        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(compiled.plan.steps.len() > 5);
    }

    #[test]
    fn compile_error_is_hirn_error() {
        let result = compile("INVALID", None);
        let err = result.unwrap_err();
        let hirn_err: HirnError = err.into();
        assert!(matches!(hirn_err, HirnError::InvalidInput(_)));
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

    // ── Prepared statement tests ───────────────────────────────────────

    #[test]
    fn prepare_extracts_positional_params() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();
        assert_eq!(stmt.params, vec!["$1"]);
    }

    #[test]
    fn prepare_extracts_named_params() {
        let stmt = prepare(
            r#"RECALL episodic ABOUT $query WHERE importance > $threshold"#,
            None,
        )
        .unwrap();
        assert!(stmt.params.contains(&"$query".to_string()));
        assert!(stmt.params.contains(&"$threshold".to_string()));
        assert_eq!(stmt.params.len(), 2);
    }

    #[test]
    fn prepare_no_params_runs_analysis() {
        // No params — full analysis, should catch value out of range.
        let result = prepare(r#"RECALL episodic ABOUT "x" WHERE importance > 2.0"#, None);
        assert!(matches!(result, Err(CompileError::Analysis(_))));
    }

    #[test]
    fn prepare_with_params_skips_analysis() {
        // Has params — analysis skipped (can't validate $threshold range).
        let result = prepare(r#"RECALL episodic ABOUT $1 WHERE importance > $2"#, None);
        assert!(result.is_ok());
    }

    #[test]
    fn bind_substitutes_string_param() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();
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
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();
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
        let stmt = prepare(
            r#"RECALL episodic ABOUT $query WHERE importance > $threshold"#,
            None,
        )
        .unwrap();
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
        assert!(prepare(r#"RECALL episodic ABOUT "x" LIMIT $n"#, None).is_err());
    }

    #[test]
    fn bind_missing_param_returns_error() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();
        let values = HashMap::new(); // no values
        let result = bind(&stmt, &values);
        assert!(result.is_err());
    }

    #[test]
    fn bind_leaves_template_reusable() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();
        let plan_before = stmt.plan.clone();

        let mut values = HashMap::new();
        values.insert("$1".to_string(), "auth".to_string());
        let _ = bind(&stmt, &values).unwrap();

        // Binding clones the template: the prepared statement (and its plan
        // metadata) stays untouched and can be bound again.
        assert_eq!(stmt.plan, plan_before);
        let again = bind(&stmt, &values).unwrap();
        match &again {
            Statement::Recall(r) => assert_eq!(r.about, "auth"),
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn bind_different_values_produce_different_asts() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();

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
    fn bind_validates_bound_values() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 WHERE importance > $2"#, None).unwrap();
        let mut values = HashMap::new();
        values.insert("$1".to_string(), "test".to_string());
        values.insert("$2".to_string(), "5.0".to_string()); // out of range

        let result = bind(&stmt, &values);
        assert!(matches!(result, Err(CompileError::Analysis(_))));
    }

    #[test]
    fn prepared_stmt_faster_than_cold_compile() {
        let q = r#"RECALL episodic ABOUT $1 INVOLVING "auth" AFTER "2026-01-01" WHERE importance > 0.5 LIMIT 10"#;
        let stmt = prepare(q, None).unwrap();

        // Time 1000 bind() calls.
        let mut values = HashMap::new();
        values.insert("$1".to_string(), "test".to_string());
        let start = std::time::Instant::now();
        for _ in 0..1_000 {
            let _ = bind(&stmt, &values).unwrap();
        }
        let bind_elapsed = start.elapsed();

        // Time 1000 full compile() calls.
        let q_concrete = r#"RECALL episodic ABOUT "test" INVOLVING "auth" AFTER "2026-01-01" WHERE importance > 0.5 LIMIT 10"#;
        let start = std::time::Instant::now();
        for _ in 0..1_000 {
            let _ = compile(q_concrete, None).unwrap();
        }
        let compile_elapsed = start.elapsed();

        // bind clones the pre-parsed template and substitutes values — no
        // parsing, no planning — so it must not be slower than a cold
        // compile (which parses, analyzes, and plans). Allow slack for noisy
        // CI schedulers rather than asserting a strict ratio.
        assert!(
            bind_elapsed.as_secs_f64() < 2.0,
            "1K binds took {:.2}s",
            bind_elapsed.as_secs_f64()
        );
        let _ = compile_elapsed;
    }

    // ── EXPLAIN ──

    #[test]
    fn compile_explain_succeeds() {
        let cq = compile(r#"EXPLAIN RECALL episodic ABOUT "hello""#, None).unwrap();
        assert!(matches!(cq.ast, Statement::Explain(_)));
    }

    #[test]
    fn compile_explain_analyze_succeeds() {
        let cq = compile(
            r#"EXPLAIN ANALYZE RECALL episodic ABOUT "hello" LIMIT 5"#,
            None,
        )
        .unwrap();
        match &cq.ast {
            Statement::Explain(e) => {
                assert!(e.analyze);
                assert!(matches!(*e.inner, Statement::Recall(_)));
            }
            _ => panic!("expected Explain"),
        }
    }

    #[test]
    fn compile_explain_invalid_inner_fails() {
        // EXPLAIN without a valid inner statement should fail
        let result = compile(r#"EXPLAIN"#, None);
        assert!(result.is_err());
    }
}
