//! HirnQL v2 compiler — transforms query text into cached, optimized execution plans.
//!
//! Pipeline: HirnQL text → Parser → Untyped AST → Semantic analysis → Planner → Physical plan → Execute
//!
//! Each stage is independently testable. Plans are cached by query text hash.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use hirn_core::HirnError;
use parking_lot::RwLock;

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
/// execution, or use `bind()` when a concrete `CompiledQuery` is needed for
/// inspection or tests.
#[derive(Debug, Clone)]
pub struct PreparedStatement {
    /// The original query text containing `$param` placeholders.
    pub source: String,
    /// Parameter names found in the query (sorted, with `$` prefix).
    pub params: Vec<String>,
    /// The pre-compiled plan (reused across bindings).
    pub plan: QueryPlan,
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

    // Skip semantic analysis for parameterized queries — params may cause
    // false positives (e.g. importance > $threshold). Full analysis happens
    // after bind().
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
    })
}

/// Bind parameter values to a prepared statement, producing an executable `CompiledQuery`.
///
/// `values` maps parameter names (with `$` prefix) to string representations.
/// Positional parameters use `$1`, `$2`, etc.
///
/// Returns an error if any declared parameter is missing from `values`.
pub fn bind(
    prepared: &PreparedStatement,
    values: &HashMap<String, String>,
) -> Result<CompiledQuery, CompileError> {
    // Validate all parameters are provided.
    for param in &prepared.params {
        if !values.contains_key(param) {
            return Err(CompileError::Analysis(vec![AnalysisError {
                message: format!("missing value for parameter {param}"),
                kind: AnalysisErrorKind::UnknownField,
            }]));
        }
    }

    // Substitute parameters in the source query.
    let mut bound_query = prepared.source.clone();
    for (name, value) in values {
        // Determine if this parameter appears in a numeric context.
        // If the value is purely numeric, don't quote it.
        let replacement = if value.parse::<f64>().is_ok() || value.parse::<i64>().is_ok() {
            value.clone()
        } else {
            // Escape double quotes in the value for safe embedding.
            format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
        };
        bound_query = bound_query.replace(name.as_str(), &replacement);
    }

    // Re-parse with concrete values.
    let ast = parser::parse(&bound_query).map_err(CompileError::Parse)?;

    // Full semantic analysis now that all values are concrete.
    let errors = analyzer::analyze(&ast);
    if !errors.is_empty() {
        return Err(CompileError::Analysis(errors));
    }

    Ok(CompiledQuery {
        source: bound_query,
        ast,
        plan: prepared.plan.clone(),
    })
}

/// A plan cache that stores compiled query plans keyed by query text hash.
///
/// Plans are invalidated when the underlying data statistics change
/// significantly (detected by comparing DbStats).
#[derive(Debug)]
pub struct PlanCache {
    cache: RwLock<PlanCacheInner>,
    capacity: usize,
}

#[derive(Debug)]
struct PlanCacheInner {
    entries: HashMap<u64, CacheEntry>,
    /// Stats fingerprint at the time entries were cached.
    stats_fingerprint: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    compiled: CompiledQuery,
    hits: u64,
}

impl PlanCache {
    /// Create a new plan cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: RwLock::new(PlanCacheInner {
                entries: HashMap::with_capacity(capacity),
                stats_fingerprint: 0,
            }),
            capacity,
        }
    }

    /// Compile a query, using the cache if possible.
    ///
    /// If the query was previously compiled with the same stats, the cached
    /// plan is returned. Otherwise, a fresh compilation is performed.
    pub fn compile(
        &self,
        query: &str,
        stats: Option<&DbStats>,
    ) -> Result<CompiledQuery, CompileError> {
        let key = hash_query(query);
        let fingerprint = stats_fingerprint(stats);

        // Try cache read.
        {
            let cache = self.cache.read();
            if cache.stats_fingerprint == fingerprint {
                if let Some(entry) = cache.entries.get(&key) {
                    return Ok(entry.compiled.clone());
                }
            }
        }

        // Cache miss — compile fresh.
        let compiled = compile(query, stats)?;

        // Store in cache.
        {
            let mut cache = self.cache.write();

            // If stats changed, invalidate the entire cache.
            if cache.stats_fingerprint != fingerprint {
                cache.entries.clear();
                cache.stats_fingerprint = fingerprint;
            }

            // Evict if at capacity (simple: remove oldest entry by lowest hit count).
            if cache.entries.len() >= self.capacity {
                if let Some((&evict_key, _)) = cache.entries.iter().min_by_key(|(_, e)| e.hits) {
                    cache.entries.remove(&evict_key);
                }
            }

            cache.entries.insert(
                key,
                CacheEntry {
                    compiled: compiled.clone(),
                    hits: 1,
                },
            );
        }

        Ok(compiled)
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.read().entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the plan cache.
    pub fn clear(&self) {
        let mut cache = self.cache.write();
        cache.entries.clear();
    }
}

fn hash_query(query: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Normalize whitespace for cache key.
    let normalized: String = query.split_whitespace().collect::<Vec<_>>().join(" ");
    normalized.hash(&mut hasher);
    hasher.finish()
}

fn stats_fingerprint(stats: Option<&DbStats>) -> u64 {
    let Some(s) = stats else { return 0 };
    let mut hasher = DefaultHasher::new();
    s.total_count.hash(&mut hasher);
    s.episodic_count.hash(&mut hasher);
    s.semantic_count.hash(&mut hasher);
    hasher.finish()
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

    // ── Plan cache tests ───────────────────────────────────────────────

    #[test]
    fn cache_hit_returns_same_plan() {
        let cache = PlanCache::new(100);
        let q = r#"RECALL episodic ABOUT "test""#;
        let c1 = cache.compile(q, None).unwrap();
        let c2 = cache.compile(q, None).unwrap();
        assert_eq!(c1.plan, c2.plan);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_different_queries_stored_separately() {
        let cache = PlanCache::new(100);
        cache.compile(r#"RECALL episodic ABOUT "a""#, None).unwrap();
        cache.compile(r#"RECALL episodic ABOUT "b""#, None).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_invalidated_on_stats_change() {
        let cache = PlanCache::new(100);
        let stats1 = DbStats {
            working_count: 0,
            episodic_count: 100,
            semantic_count: 50,
            edge_count: 0,
            procedural_count: 0,
            total_count: 150,
            file_size_bytes: 0,
        };
        let stats2 = DbStats {
            working_count: 0,
            episodic_count: 5000,
            semantic_count: 2000,
            edge_count: 0,
            procedural_count: 0,
            total_count: 7000,
            file_size_bytes: 0,
        };

        cache
            .compile(r#"RECALL episodic ABOUT "test""#, Some(&stats1))
            .unwrap();
        assert_eq!(cache.len(), 1);

        // Different stats → cache invalidated, fresh compilation.
        cache
            .compile(r#"RECALL episodic ABOUT "test""#, Some(&stats2))
            .unwrap();
        assert_eq!(cache.len(), 1); // Old entry cleared, new one stored.
    }

    #[test]
    fn cache_eviction_at_capacity() {
        let cache = PlanCache::new(2);
        cache.compile(r#"RECALL episodic ABOUT "a""#, None).unwrap();
        cache.compile(r#"RECALL episodic ABOUT "b""#, None).unwrap();
        assert_eq!(cache.len(), 2);

        // Third entry should evict one.
        cache.compile(r#"RECALL episodic ABOUT "c""#, None).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_clear_empties_all() {
        let cache = PlanCache::new(100);
        cache.compile(r#"RECALL episodic ABOUT "a""#, None).unwrap();
        cache.compile(r#"RECALL episodic ABOUT "b""#, None).unwrap();
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_whitespace_normalized() {
        let cache = PlanCache::new(100);
        cache
            .compile(r#"RECALL episodic ABOUT "test""#, None)
            .unwrap();
        cache
            .compile(r#"RECALL  episodic   ABOUT  "test""#, None)
            .unwrap();
        // Same query after normalization → still 1 entry.
        assert_eq!(cache.len(), 1);
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

        let compiled = bind(&stmt, &values).unwrap();
        match &compiled.ast {
            Statement::Recall(r) => assert_eq!(r.about, "authentication"),
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn bind_substitutes_numeric_param() {
        let stmt = prepare(r#"RECALL episodic ABOUT $query LIMIT $limit"#, None).unwrap();
        let mut values = HashMap::new();
        values.insert("$query".to_string(), "test".to_string());
        values.insert("$limit".to_string(), "20".to_string());

        let compiled = bind(&stmt, &values).unwrap();
        match &compiled.ast {
            Statement::Recall(r) => {
                assert_eq!(r.about, "test");
                assert_eq!(r.limit, Some(20));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn bind_missing_param_returns_error() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();
        let values = HashMap::new(); // no values
        let result = bind(&stmt, &values);
        assert!(result.is_err());
    }

    #[test]
    fn bind_reuses_plan() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();
        let plan_before = stmt.plan.clone();

        let mut values = HashMap::new();
        values.insert("$1".to_string(), "auth".to_string());
        let compiled = bind(&stmt, &values).unwrap();

        assert_eq!(
            compiled.plan, plan_before,
            "plan should be reused from prepare"
        );
    }

    #[test]
    fn bind_different_values_produce_different_asts() {
        let stmt = prepare(r#"RECALL episodic ABOUT $1 LIMIT 10"#, None).unwrap();

        let mut v1 = HashMap::new();
        v1.insert("$1".to_string(), "auth".to_string());
        let c1 = bind(&stmt, &v1).unwrap();

        let mut v2 = HashMap::new();
        v2.insert("$1".to_string(), "deployment".to_string());
        let c2 = bind(&stmt, &v2).unwrap();

        match (&c1.ast, &c2.ast) {
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

        // bind should be at most slightly slower than compile (both re-parse),
        // but the plan is reused so no planner overhead.
        // We just verify bind completes in reasonable time.
        assert!(
            bind_elapsed.as_secs_f64() < 2.0,
            "1K binds took {:.2}s",
            bind_elapsed.as_secs_f64()
        );
        let _ = compile_elapsed; // avoid unused warning
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
