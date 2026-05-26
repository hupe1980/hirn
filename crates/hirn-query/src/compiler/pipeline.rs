//! `QueryPipeline` — 7-stage HirnQL compilation and execution pipeline.
//!
//! ```text
//! Stage 1: Parse     — HirnQL text → AST
//! Stage 2: Analyze   — AST → TypedStatement (namespace resolution, validation)
//! Stage 3: Rewrite   — logical plan rewrite pass (no-op; Cedar policy runs at
//!                       physical optimizer level via `PolicyPushdownRule` in hirn-exec)
//! Stage 4: Plan      — TypedStatement → DataFusion LogicalPlan
//! Stage 5: Optimize  — DataFusion optimizer + custom rules
//! Stage 6: Execute   — LogicalPlan → PhysicalPlan → RecordBatchStream
//! Stage 7: Collect   — RecordBatchStream → results
//! ```
//!
//! Stages 1–4 live here (pure transformations). Stages 5–7 require a
//! DataFusion `SessionContext` and live in `hirn-engine`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use dashmap::DashMap;
use datafusion_expr::LogicalPlan;
use datafusion_expr::logical_plan::Extension;
use parking_lot::Mutex;

use hirn_core::error::{HirnError, HirnResult};

use super::plan_compiler;
use super::typed_ast::{self, AnalyzeContext, TypedStatement};
use crate::parser;
use crate::parser::ast::Statement;

/// Compiled query ready for optimization + execution (stages 5–7).
#[derive(Debug, Clone)]
pub struct CompiledPlan {
    /// The original query text.
    pub source: String,
    /// The raw parsed AST.
    pub ast: Statement,
    /// The resolved typed AST.
    pub typed: TypedStatement,
    /// DataFusion logical plan.
    pub plan: LogicalPlan,
}

/// DashMap-backed plan cache with O(log N) LRU eviction via a min-heap.
///
/// Cache key is a hash of normalized query text. Cache entries store the
/// normalized source string so that 64-bit hash collisions are detected
/// and rejected rather than silently returning a wrong plan (N-M19).
///
/// # Eviction (N-H14)
///
/// A `BinaryHeap<(Reverse<u64>, u64)>` tracks `(Reverse<access_count>, key)`
/// pairs. Eviction pops from the min-heap using lazy deletion: entries whose
/// `access_count` has increased since they were pushed to the heap are skipped
/// and a fresh entry is pushed so the updated count is reflected.
pub struct PlanCache {
    entries: DashMap<u64, CacheEntry>,
    /// Min-heap for O(log N) eviction. Entries are `(Reverse<access_count>, key)`.
    /// Stale heap entries (count changed) are skipped lazily during eviction.
    eviction_heap: Mutex<BinaryHeap<(Reverse<u64>, u64)>>,
    max_entries: usize,
}

#[derive(Clone)]
struct CacheEntry {
    /// Normalized source query used to detect 64-bit hash collisions (N-M19).
    normalized_source: Arc<str>,
    plan: Arc<CompiledPlan>,
    access_count: u64,
}

impl PlanCache {
    /// Create a plan cache with the given maximum number of entries.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: DashMap::with_capacity(max_entries.min(256)),
            eviction_heap: Mutex::new(BinaryHeap::with_capacity(max_entries.min(256))),
            max_entries,
        }
    }

    /// Look up a cached plan by query hash.
    ///
    /// Returns `None` on a miss **or** on a hash collision (N-M19): the caller
    /// must pass the same normalized source string that was used to compute
    /// `key` so that the stored string is compared for equality before serving
    /// the cached plan.
    pub fn get(&self, key: u64, normalized_source: &str) -> Option<Arc<CompiledPlan>> {
        self.entries.get_mut(&key).and_then(|mut entry| {
            // Reject hash collisions: key matches but source differs.
            if entry.normalized_source.as_ref() != normalized_source {
                tracing::warn!(
                    key,
                    cached_source = %entry.normalized_source,
                    incoming_source = %normalized_source,
                    "plan cache: 64-bit hash collision — skipping cached plan"
                );
                return None;
            }
            entry.access_count += 1;
            // Push the updated count into the heap so the lazy-deletion
            // eviction always reflects the most recent access frequency.
            self.eviction_heap
                .lock()
                .push((Reverse(entry.access_count), key));
            Some(Arc::clone(&entry.plan))
        })
    }

    /// Insert a compiled plan. Evicts the least-recently-used entry when at
    /// capacity using O(log N) heap-pop with lazy deletion (N-H14).
    ///
    /// Concurrent `put` calls may temporarily exceed `max_entries` by the number
    /// of concurrent writers; the cap is enforced on a best-effort basis without
    /// a global write lock (N-M03).
    pub fn put(&self, key: u64, normalized_source: Arc<str>, plan: Arc<CompiledPlan>) {
        if self.entries.len() >= self.max_entries {
            let evicted = self.try_evict_one();
            // If the heap was exhausted (all entries freshly accessed), force-remove
            // an arbitrary entry so the cache stays bounded (N-M03).
            if !evicted && self.entries.len() >= self.max_entries {
                // Eagerly extract the key before the if-let so the DashMap iterator
                // guard is dropped before the subsequent remove (N-M03 / significant-drop).
                let arbitrary_key = self.entries.iter().next().map(|e| *e.key());
                if let Some(entry) = arbitrary_key {
                    self.entries.remove(&entry);
                }
            }
        }
        self.eviction_heap.lock().push((Reverse(1), key));
        self.entries.insert(
            key,
            CacheEntry {
                normalized_source,
                plan,
                access_count: 1,
            },
        );
    }

    /// Pop the min-heap until one live (non-stale) entry is evicted.
    /// Returns `true` if an entry was removed from `self.entries`.
    fn try_evict_one(&self) -> bool {
        let mut heap = self.eviction_heap.lock();
        loop {
            match heap.pop() {
                None => return false,
                Some((Reverse(snapshot_count), evict_key)) => {
                    // `remove_if` is atomic: evict only if count matches snapshot.
                    if self
                        .entries
                        .remove_if(&evict_key, |_, v| v.access_count == snapshot_count)
                        .is_some()
                    {
                        return true;
                    }
                    // Count changed — this heap entry is stale; try next.
                }
            }
        }
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all cached entries.
    pub fn clear(&self) {
        self.entries.clear();
    }
}

impl std::fmt::Debug for PlanCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanCache")
            .field("len", &self.entries.len())
            .field("max_entries", &self.max_entries)
            // eviction_heap omitted — locking during Debug is undesirable.
            .finish_non_exhaustive()
    }
}

/// The 7-stage query pipeline. Stages 1–4 are executed here; stages 5–7
/// are deferred to the engine which holds the `SessionContext`.
pub struct QueryPipeline {
    ctx: AnalyzeContext,
    cache: Option<Arc<PlanCache>>,
}

impl QueryPipeline {
    /// Create a new pipeline with the given context.
    pub fn new(ctx: AnalyzeContext) -> Self {
        Self { ctx, cache: None }
    }

    /// Attach a shared plan cache.
    pub fn with_cache(mut self, cache: Arc<PlanCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Run stages 1–4, returning a `CompiledPlan`.
    ///
    /// If a cache is attached and the query was previously compiled *with the
    /// same default namespace*, returns the cached plan directly.
    ///
    /// Use [`compile_with_ctx`] to override the default [`AnalyzeContext`] on a
    /// per-call basis (e.g. for requests from a specific agent whose default
    /// namespace differs from `self.ctx`).
    pub fn compile(&self, query: &str) -> HirnResult<Arc<CompiledPlan>> {
        self.compile_with_ctx(query, &self.ctx)
    }

    /// Run stages 1–4 with an explicit [`AnalyzeContext`], returning a `CompiledPlan`.
    ///
    /// The plan cache key is mixed with the caller-supplied default namespace so
    /// that two requests for the same query text but different default namespaces
    /// receive correctly resolved plans (not the same cached entry).
    ///
    /// If a cache is attached and there is a hit for `(query, ctx.default_namespace)`,
    /// returns the cached plan directly (skipping parse + analyze + plan).
    pub fn compile_with_ctx(&self, query: &str, ctx: &AnalyzeContext) -> HirnResult<Arc<CompiledPlan>> {
        let (normalized, base_key) = plan_compiler::query_normalize_and_hash(query);
        // Mix the default-namespace interned ID into the cache key so that the
        // same query text compiled under different default namespaces produces
        // independent cache entries.  Uses FNV-style mix to avoid trivial
        // cancellation.
        let ns_id = ctx.default_namespace.as_interned_id();
        let key = base_key
            .wrapping_mul(0x9e37_79b9_7f4a_7c15_u64)
            .wrapping_add(ns_id as u64);

        // Cache hit? Verified against stored normalized source to catch hash
        // collisions (N-M19).
        if let Some(ref cache) = self.cache {
            if let Some(plan) = cache.get(key, &normalized) {
                return Ok(plan);
            }
        }

        // Stage 1: Parse.
        let ast = parser::parse(query)
            .map_err(|e| HirnError::InvalidInput(format!("parse error: {e}")))?;

        // Stage 2: Analyze.
        let typed = typed_ast::analyze(&ast, ctx)?;

        // Stage 3: Rewrite — logical rewrite pass.  Policy enforcement is handled at
        // the *physical* optimizer level by `PolicyPushdownRule` in `hirn-exec`;
        // this stage is reserved for future logical-level rewrites (e.g. expansion
        // macros, cross-namespace normalization).
        let typed = self.rewrite(typed)?;

        // Stage 4: Plan.
        let plan = plan_compiler::compile(&typed)?;

        let compiled = Arc::new(CompiledPlan {
            source: query.to_string(),
            ast,
            typed,
            plan,
        });

        // Store in cache with normalized source for future collision checks (N-M19).
        if let Some(ref cache) = self.cache {
            cache.put(key, normalized.into(), Arc::clone(&compiled));
        }

        Ok(compiled)
    }

    /// Stage 3: Rewrite — logical rewrite pass.
    ///
    /// Currently a no-op; Cedar policy enforcement runs later as
    /// `PolicyPushdownRule` in the DataFusion physical optimizer (hirn-exec).
    /// Reserved for future logical-level rewrites.
    fn rewrite(&self, typed: TypedStatement) -> HirnResult<TypedStatement> {
        Ok(typed)
    }

    /// Format a query's logical plan as an indented text tree (like PostgreSQL EXPLAIN).
    ///
    /// Returns the plan tree as a `String`. The plan is compiled through stages 1–4
    /// and then formatted using DataFusion's `display_indent_schema()`.
    pub fn explain(&self, query: &str) -> HirnResult<String> {
        let compiled = self.compile(query)?;
        Ok(format_plan_tree(&compiled.plan))
    }

    /// Access the analyze context.
    pub fn context(&self) -> &AnalyzeContext {
        &self.ctx
    }
}

/// Format a DataFusion `LogicalPlan` as an indented plan tree.
///
/// Each operator is printed on its own line with 2-space indentation per depth level,
/// similar to PostgreSQL's `EXPLAIN` output.
pub fn format_plan_tree(plan: &LogicalPlan) -> String {
    let mut lines = Vec::new();
    format_plan_node(plan, 0, &mut lines);
    lines.join("\n")
}

fn format_plan_node(plan: &LogicalPlan, depth: usize, lines: &mut Vec<String>) {
    let indent = "  ".repeat(depth);
    lines.push(format!("{}{}", indent, plan_node_label(plan)));
    for child in plan.inputs() {
        format_plan_node(child, depth + 1, lines);
    }
}

fn plan_node_label(plan: &LogicalPlan) -> String {
    match plan {
        LogicalPlan::Extension(Extension { node }) => node.name().to_string(),
        _ => plan.display().to_string(),
    }
}

impl std::fmt::Debug for QueryPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryPipeline")
            .field("ctx", &self.ctx)
            .field("cache", &self.cache.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn pipeline() -> QueryPipeline {
        QueryPipeline::new(AnalyzeContext::default())
    }

    #[test]
    fn compile_recall_produces_plan() {
        let p = pipeline();
        let result = p.compile(r#"RECALL episodic ABOUT "test" LIMIT 5"#);
        assert!(result.is_ok());
        let compiled = result.unwrap();
        assert!(matches!(compiled.ast, Statement::Recall(_)));
        assert!(matches!(compiled.typed, TypedStatement::Recall(_)));
        let display = format!("{}", compiled.plan);
        assert!(display.contains("HybridSearch"), "plan: {display}");
    }

    #[test]
    fn compile_think_produces_plan() {
        let p = pipeline();
        let compiled = p.compile(r#"THINK ABOUT "test" BUDGET 4096"#).unwrap();
        assert!(matches!(compiled.typed, TypedStatement::Think(_)));
        let display = format!("{}", compiled.plan);
        assert!(display.contains("QualityGate"), "plan: {display}");
    }

    #[test]
    fn compile_rejects_removed_embedded_mutation_verbs() {
        let p = pipeline();
        for query in [
            r#"REMEMBER episode CONTENT "event happened""#,
            r#"FORGET "01J000000000000000000000""#,
            "WATCH ALL FORMAT json",
            "CONSOLIDATE WHERE episodic.access_count > 5",
        ] {
            let err = p.compile(query).unwrap_err();
            assert!(
                err.to_string().contains("not supported"),
                "unexpected error for `{query}`: {err}"
            );
        }
    }

    #[test]
    fn compile_parse_error() {
        let p = pipeline();
        let err = p.compile("NOT_A_QUERY").unwrap_err();
        assert!(matches!(err, HirnError::InvalidInput(_)));
    }

    #[test]
    fn cache_hit() {
        let cache = Arc::new(PlanCache::new(100));
        let p = pipeline().with_cache(cache.clone());
        let q = r#"RECALL episodic ABOUT "test" LIMIT 10"#;
        p.compile(q).unwrap();
        assert_eq!(cache.len(), 1);
        // Second compile should hit cache.
        p.compile(q).unwrap();
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn cache_different_queries() {
        let cache = Arc::new(PlanCache::new(100));
        let p = pipeline().with_cache(cache.clone());
        p.compile(r#"RECALL episodic ABOUT "a""#).unwrap();
        p.compile(r#"RECALL episodic ABOUT "b""#).unwrap();
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn cache_eviction() {
        let cache = Arc::new(PlanCache::new(2));
        let p = pipeline().with_cache(cache.clone());
        p.compile(r#"RECALL episodic ABOUT "a""#).unwrap();
        p.compile(r#"RECALL episodic ABOUT "b""#).unwrap();
        assert_eq!(cache.len(), 2);
        p.compile(r#"RECALL episodic ABOUT "c""#).unwrap();
        assert_eq!(cache.len(), 2); // One evicted.
    }

    #[test]
    fn cache_clear() {
        let cache = Arc::new(PlanCache::new(100));
        let p = pipeline().with_cache(cache.clone());
        p.compile(r#"RECALL episodic ABOUT "a""#).unwrap();
        p.compile(r#"RECALL episodic ABOUT "b""#).unwrap();
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn pipeline_without_cache() {
        let p = pipeline();
        let compiled = p.compile(r#"RECALL episodic ABOUT "test""#).unwrap();
        assert!(!compiled.source.is_empty());
    }

    #[test]
    fn stages_independently_callable() {
        // Stage 1: Parse.
        let ast = parser::parse(r#"RECALL episodic ABOUT "test" LIMIT 5"#).unwrap();
        // Stage 2: Analyze.
        let ctx = AnalyzeContext::default();
        let typed = typed_ast::analyze(&ast, &ctx).unwrap();
        // Stage 4: Plan.
        let plan = plan_compiler::compile(&typed).unwrap();
        let display = format!("{plan}");
        assert!(display.contains("HybridSearch"), "plan: {display}");
    }

    #[test]
    fn explain_returns_plan_tree() {
        let p = pipeline();
        let tree = p
            .explain(r#"RECALL episodic ABOUT "test" LIMIT 5"#)
            .unwrap();
        // Plan tree should contain operator names at different indentation levels.
        assert!(
            tree.contains("HybridSearch") || tree.contains("Limit"),
            "plan tree: {tree}"
        );
        // Should have multiple lines (indented children).
        assert!(tree.lines().count() > 1, "plan tree: {tree}");
    }

    #[test]
    fn explain_correct_shows_extension_name() {
        let p = pipeline();
        let tree = p
            .explain(r#"EXPLAIN CORRECT "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "updated""#)
            .unwrap();
        assert!(tree.contains("HirnDirectCorrect"), "plan tree: {tree}");
    }

    #[test]
    fn explain_supersede_shows_extension_name() {
        let p = pipeline();
        let tree = p
            .explain(
                r#"EXPLAIN SUPERSEDE "01ARZ3NDEKTSV4RRFFQ69G5FAV" SET description = "replacement""#,
            )
            .unwrap();
        assert!(tree.contains("HirnDirectSupersede"), "plan tree: {tree}");
    }

    #[test]
    fn explain_merge_memory_shows_extension_name() {
        let p = pipeline();
        let tree = p
            .explain(
                r#"EXPLAIN MERGE MEMORY "01ARZ3NDEKTSV4RRFFQ69G5FAA" INTO "01ARZ3NDEKTSV4RRFFQ69G5FAV""#,
            )
            .unwrap();
        assert!(tree.contains("HirnDirectMergeMemory"), "plan tree: {tree}");
    }

    #[test]
    fn explain_history_shows_extension_name() {
        let p = pipeline();
        let tree = p
            .explain(r#"EXPLAIN HISTORY "01ARZ3NDEKTSV4RRFFQ69G5FAV" NAMESPACE custom"#)
            .unwrap();
        assert!(
            tree.contains("HirnSemanticHistoryScan"),
            "plan tree: {tree}"
        );
    }

    #[test]
    fn explain_retract_shows_extension_name() {
        let p = pipeline();
        let tree = p
            .explain(r#"EXPLAIN RETRACT "01ARZ3NDEKTSV4RRFFQ69G5FAV" REASON "obsolete""#)
            .unwrap();
        assert!(tree.contains("HirnDirectRetract"), "plan tree: {tree}");
    }

    #[test]
    fn explain_of_cached_query_still_shows_plan() {
        let cache = Arc::new(PlanCache::new(10));
        let p = pipeline().with_cache(cache);
        // First call compiles and caches.
        let tree1 = p.explain(r#"RECALL episodic ABOUT "test""#).unwrap();
        // Second call hits cache but should still produce the same plan tree.
        let tree2 = p.explain(r#"RECALL episodic ABOUT "test""#).unwrap();
        assert_eq!(tree1, tree2);
        assert!(!tree1.is_empty());
    }

    #[test]
    fn format_plan_tree_indents_children() {
        let p = pipeline();
        let compiled = p
            .compile(r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 2 LIMIT 5"#)
            .unwrap();
        let tree = super::format_plan_tree(&compiled.plan);
        // Root should start at column 0, children should be indented.
        let lines: Vec<&str> = tree.lines().collect();
        assert!(!lines.is_empty());
        // First line has no indentation.
        assert!(!lines[0].starts_with(' '), "root: {}", lines[0]);
        // If there are children, they should be indented.
        if lines.len() > 1 {
            assert!(lines[1].starts_with("  "), "child: {}", lines[1]);
        }
    }

    #[test]
    fn cached_query_executes_under_5us() {
        let cache = Arc::new(PlanCache::new(100));
        let p = pipeline().with_cache(cache.clone());
        // Warm up: compile and cache
        p.compile(r#"RECALL episodic ABOUT "test""#).unwrap();
        // Measure cached hit
        let start = std::time::Instant::now();
        let iterations = 1000;
        for _ in 0..iterations {
            let _ = p.compile(r#"RECALL episodic ABOUT "test""#).unwrap();
        }
        let elapsed = start.elapsed();
        let per_op = elapsed / iterations;
        assert!(
            per_op.as_micros() < 5,
            "cached query took {per_op:?} per op, expected < 5µs"
        );
    }
}
