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

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

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

/// Bounded LRU cache for compiled plans.
///
/// A single mutex guards a `HashMap` of entries plus a `VecDeque` of
/// `(generation, key)` recency stamps. Every hit and insert assigns the entry
/// a fresh generation and appends one stamp, so both operations are O(1);
/// older stamps for the same entry become stale. Eviction pops stamps from
/// the front of the queue (oldest first) and skips stale ones, which yields
/// true least-recently-used order: the entry whose live stamp is oldest is
/// removed first.
///
/// The stamp queue is compacted (stale stamps dropped) whenever it grows past
/// twice the live-entry watermark, so its memory stays O(capacity) even under
/// hit-heavy workloads.
///
/// Cache keys are 64-bit hashes of normalized query text. Entries store the
/// normalized source string so a hash collision is detected and treated as a
/// miss instead of silently serving a wrong plan.
pub struct PlanCache {
    inner: Mutex<PlanCacheInner>,
    max_entries: usize,
}

struct PlanCacheInner {
    entries: HashMap<u64, CacheEntry>,
    /// Recency stamps `(generation, key)`, oldest at the front. An entry's
    /// only live stamp is the one matching its current `generation`.
    recency: VecDeque<(u64, u64)>,
    /// Monotonic counter; bumped on every hit and insert.
    generation: u64,
}

struct CacheEntry {
    /// Normalized source query used to detect 64-bit hash collisions.
    normalized_source: Arc<str>,
    plan: Arc<CompiledPlan>,
    /// Generation of this entry's live recency stamp.
    generation: u64,
}

impl PlanCacheInner {
    /// Remove the least-recently-used live entry. Skips stale stamps.
    fn evict_lru(&mut self) {
        while let Some((generation, key)) = self.recency.pop_front() {
            let live = self
                .entries
                .get(&key)
                .is_some_and(|e| e.generation == generation);
            if live {
                self.entries.remove(&key);
                return;
            }
        }
    }

    /// Drop stale stamps once the queue exceeds twice the live watermark.
    /// Amortized O(1) per operation; bounds the queue at O(capacity).
    fn compact_if_needed(&mut self, max_entries: usize) {
        if self.recency.len() <= self.entries.len().max(max_entries) * 2 {
            return;
        }
        let entries = &self.entries;
        self.recency.retain(|(generation, key)| {
            entries
                .get(key)
                .is_some_and(|e| e.generation == *generation)
        });
    }
}

impl PlanCache {
    /// Create a plan cache with the given maximum number of entries.
    pub fn new(max_entries: usize) -> Self {
        Self {
            inner: Mutex::new(PlanCacheInner {
                entries: HashMap::with_capacity(max_entries.min(256)),
                recency: VecDeque::with_capacity(max_entries.min(256)),
                generation: 0,
            }),
            max_entries,
        }
    }

    /// Look up a cached plan by query hash, marking it most-recently-used.
    ///
    /// Returns `None` on a miss **or** on a hash collision: the caller must
    /// pass the same normalized source string that was used to compute `key`
    /// so the stored string is compared for equality before serving the
    /// cached plan.
    pub fn get(&self, key: u64, normalized_source: &str) -> Option<Arc<CompiledPlan>> {
        let mut guard = self.inner.lock();
        let inner = &mut *guard;
        let entry = inner.entries.get_mut(&key)?;
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
        inner.generation += 1;
        entry.generation = inner.generation;
        let plan = Arc::clone(&entry.plan);
        inner.recency.push_back((inner.generation, key));
        inner.compact_if_needed(self.max_entries);
        Some(plan)
    }

    /// Insert a compiled plan as the most-recently-used entry, evicting the
    /// least-recently-used entry first when at capacity.
    ///
    /// The capacity bound is strict: eviction and insert happen under the
    /// same lock, so the cache never holds more than `max_entries` entries.
    pub fn put(&self, key: u64, normalized_source: Arc<str>, plan: Arc<CompiledPlan>) {
        if self.max_entries == 0 {
            return;
        }
        let mut guard = self.inner.lock();
        let inner = &mut *guard;
        if !inner.entries.contains_key(&key) && inner.entries.len() >= self.max_entries {
            inner.evict_lru();
        }
        inner.generation += 1;
        let generation = inner.generation;
        inner.entries.insert(
            key,
            CacheEntry {
                normalized_source,
                plan,
                generation,
            },
        );
        inner.recency.push_back((generation, key));
        inner.compact_if_needed(self.max_entries);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().entries.is_empty()
    }

    /// Remove all cached entries.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.recency.clear();
    }

    /// Current length of the recency stamp queue (test instrumentation for
    /// the boundedness guarantee).
    #[cfg(test)]
    fn recency_len(&self) -> usize {
        self.inner.lock().recency.len()
    }
}

impl std::fmt::Debug for PlanCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanCache")
            .field("len", &self.len())
            .field("max_entries", &self.max_entries)
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
    pub fn compile_with_ctx(
        &self,
        query: &str,
        ctx: &AnalyzeContext,
    ) -> HirnResult<Arc<CompiledPlan>> {
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

        let compiled = Arc::new(self.compile_parsed(ast, query.to_string(), ctx)?);

        // Store in cache with normalized source for future collision checks (N-M19).
        if let Some(ref cache) = self.cache {
            cache.put(key, normalized.into(), Arc::clone(&compiled));
        }

        Ok(compiled)
    }

    /// Compile an already-parsed statement through stages 2–4, using the
    /// pipeline's default [`AnalyzeContext`].
    ///
    /// This is the prepared-statement execution entry: after `bind()` has
    /// substituted parameter values into the template AST, the bound AST is
    /// compiled and executed directly. It is never serialized back to HirnQL
    /// text and re-parsed, so statement `Display` output is not load-bearing
    /// for execution (it remains in use for logging and EXPLAIN only).
    ///
    /// Bypasses the plan cache: the cache is keyed by query text, and bound
    /// statements embed per-execution parameter values.
    pub fn compile_statement(&self, ast: Statement) -> HirnResult<Arc<CompiledPlan>> {
        self.compile_statement_with_ctx(ast, &self.ctx)
    }

    /// Compile an already-parsed statement through stages 2–4 with an
    /// explicit [`AnalyzeContext`]. See [`Self::compile_statement`].
    pub fn compile_statement_with_ctx(
        &self,
        ast: Statement,
        ctx: &AnalyzeContext,
    ) -> HirnResult<Arc<CompiledPlan>> {
        // `source` is derived via Display for diagnostics/logging only;
        // execution consumes `typed`/`plan` and never re-parses it.
        let source = ast.to_string();
        Ok(Arc::new(self.compile_parsed(ast, source, ctx)?))
    }

    /// Stages 2–4 (analyze → rewrite → plan), shared by the text entry
    /// ([`Self::compile_with_ctx`]) and the AST entry
    /// ([`Self::compile_statement_with_ctx`]) so both paths run the exact
    /// same pipeline.
    fn compile_parsed(
        &self,
        ast: Statement,
        source: String,
        ctx: &AnalyzeContext,
    ) -> HirnResult<CompiledPlan> {
        // Stage 2: Analyze.
        let typed = typed_ast::analyze(&ast, ctx)?;

        // Stage 3: Rewrite — logical rewrite pass.  Policy enforcement is handled at
        // the *physical* optimizer level by `PolicyPushdownRule` in `hirn-exec`;
        // this stage is reserved for future logical-level rewrites (e.g. expansion
        // macros, cross-namespace normalization).
        let typed = self.rewrite(typed)?;

        // Stage 4: Plan.
        let plan = plan_compiler::compile(&typed)?;

        Ok(CompiledPlan {
            source,
            ast,
            typed,
            plan,
        })
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

    // ── Direct-AST compilation (prepared-statement path) ───────────────

    #[test]
    fn compile_statement_matches_text_compile() {
        let p = pipeline();
        let query = r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 2 MIN_WEIGHT 1.0 LIMIT 5"#;
        let from_text = p.compile(query).unwrap();
        let ast = parser::parse(query).unwrap();
        let from_ast = p.compile_statement(ast).unwrap();
        assert_eq!(
            super::format_plan_tree(&from_text.plan),
            super::format_plan_tree(&from_ast.plan),
            "text and AST entries must produce identical plans"
        );
    }

    #[test]
    fn compile_statement_does_not_reparse() {
        // Inject a payload into the AST that would change meaning if the
        // statement were serialized back to text and re-parsed. The direct
        // AST entry must keep it as opaque data.
        let p = pipeline();
        let mut ast = parser::parse(r#"RECALL episodic ABOUT "placeholder" LIMIT 5"#).unwrap();
        let payload = r#"x" LIMIT 1 NAMESPACE hijacked --"#;
        match &mut ast {
            Statement::Recall(r) => r.about = payload.to_string(),
            _ => unreachable!(),
        }
        let compiled = p.compile_statement(ast).unwrap();
        match &compiled.typed {
            TypedStatement::Recall(r) => {
                assert_eq!(r.query, payload, "payload must survive as literal data");
            }
            other => panic!("expected TypedStatement::Recall, got {other:?}"),
        }
    }

    // ── LRU cache behavior ─────────────────────────────────────────────

    fn dummy_plan() -> Arc<CompiledPlan> {
        pipeline()
            .compile(r#"RECALL episodic ABOUT "lru-fixture""#)
            .unwrap()
    }

    #[test]
    fn cache_lru_recently_used_survives_eviction() {
        let cache = PlanCache::new(2);
        let plan = dummy_plan();
        cache.put(1, "a".into(), Arc::clone(&plan));
        cache.put(2, "b".into(), Arc::clone(&plan));
        // Touch entry 1 so entry 2 becomes least-recently-used.
        assert!(cache.get(1, "a").is_some());
        cache.put(3, "c".into(), Arc::clone(&plan));
        assert_eq!(cache.len(), 2);
        assert!(cache.get(1, "a").is_some(), "recently used must survive");
        assert!(cache.get(2, "b").is_none(), "stale entry must be evicted");
        assert!(cache.get(3, "c").is_some());
    }

    #[test]
    fn cache_reinsert_refreshes_recency() {
        let cache = PlanCache::new(2);
        let plan = dummy_plan();
        cache.put(1, "a".into(), Arc::clone(&plan));
        cache.put(2, "b".into(), Arc::clone(&plan));
        // Re-inserting key 1 makes it most-recently-used.
        cache.put(1, "a".into(), Arc::clone(&plan));
        cache.put(3, "c".into(), Arc::clone(&plan));
        assert!(cache.get(1, "a").is_some());
        assert!(cache.get(2, "b").is_none());
    }

    #[test]
    fn cache_bounded_under_hit_heavy_workload() {
        let cache = PlanCache::new(4);
        let plan = dummy_plan();
        let sources = ["a", "b", "c", "d"];
        for (i, src) in sources.iter().enumerate() {
            cache.put(i as u64, (*src).into(), Arc::clone(&plan));
        }
        for round in 0..10_000u64 {
            let i = (round % 4) as usize;
            assert!(cache.get(i as u64, sources[i]).is_some());
        }
        assert_eq!(cache.len(), 4, "no growth after warmup");
        // The recency queue must stay O(capacity), not O(hits).
        assert!(
            cache.recency_len() <= 8,
            "recency queue grew unbounded: {}",
            cache.recency_len()
        );
    }

    #[test]
    fn cache_hash_collision_rejected() {
        let cache = PlanCache::new(4);
        let plan = dummy_plan();
        cache.put(1, "a".into(), Arc::clone(&plan));
        assert!(
            cache.get(1, "different source").is_none(),
            "same key with different source must be treated as a miss"
        );
    }

    #[test]
    fn cache_zero_capacity_stores_nothing() {
        let cache = PlanCache::new(0);
        let plan = dummy_plan();
        cache.put(1, "a".into(), plan);
        assert!(cache.is_empty());
        assert!(cache.get(1, "a").is_none());
    }

    #[test]
    fn cache_concurrent_gets_and_puts_stay_bounded() {
        let cache = Arc::new(PlanCache::new(8));
        let plan = dummy_plan();
        std::thread::scope(|scope| {
            for t in 0..8u64 {
                let cache = Arc::clone(&cache);
                let plan = Arc::clone(&plan);
                scope.spawn(move || {
                    for i in 0..500u64 {
                        let key = (t * 500 + i) % 32;
                        let source = format!("q{key}");
                        if cache.get(key, &source).is_none() {
                            cache.put(key, source.into(), Arc::clone(&plan));
                        }
                    }
                });
            }
        });
        assert!(
            cache.len() <= 8,
            "capacity bound violated: {} entries",
            cache.len()
        );
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
