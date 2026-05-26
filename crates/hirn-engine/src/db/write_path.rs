//! Write-path intelligence: RPE scoring, prospective indexing, and SVO extraction.
//!
//! These functions are called from `remember_inner()` to implement the
//! fast-path / slow-path branching described in BACKLOG5.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot;

use arrow_array::{Array, RecordBatch};
use hirn_core::MemoryId;
use hirn_core::embed::Embedder;
use hirn_core::prospective::ProspectiveImplication;
use hirn_core::svo_event::SvoEvent;
use hirn_core::types::Namespace;
use hirn_storage::PhysicalStore;
use hirn_storage::store::VectorSearchOptions;
use tracing;

/// Result of RPE (Reward Prediction Error) scoring for a single memory.
#[derive(Debug, Clone)]
pub struct RpeResult {
    /// RPE score in [0.0, 2.0]. Lower = more familiar, higher = more novel.
    pub score: f32,
    /// Max similarity found across existing memories.
    pub max_similarity: f32,
    /// Whether this should take the fast path (RPE < threshold).
    pub is_fast_path: bool,
}

/// Running population statistics for RPE distance values, enabling z-score
/// computation across writes (Welford's online algorithm).
///
/// Type alias for `hirn_core::WelfordStats`. See its documentation for the
/// jackknife-style z-score computation semantics.
pub type RunningRpeStats = hirn_core::WelfordStats;

/// Workload boundary for partitioned RPE baselines.
///
/// Separate z-score accumulators are maintained per (realm × namespace × model × layer)
/// to prevent cross-layer baseline contamination (e.g., procedural skills inflating
/// the episodic novelty baseline).
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct RpePartitionKey {
    realm: String,
    namespace: Namespace,
    model_id: String,
    /// Memory layer being written — isolates per-layer z-score baselines.
    layer: hirn_core::types::Layer,
}

impl RpePartitionKey {
    pub fn new(
        realm: impl Into<String>,
        namespace: Namespace,
        model_id: impl Into<String>,
        layer: hirn_core::types::Layer,
    ) -> Self {
        Self {
            realm: realm.into(),
            namespace,
            model_id: model_id.into(),
            layer,
        }
    }

    pub fn realm(&self) -> &str {
        &self.realm
    }

    pub fn namespace(&self) -> Namespace {
        self.namespace
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    #[cfg(test)]
    pub fn layer(&self) -> hirn_core::types::Layer {
        self.layer
    }
}

pub const RPE_THRESHOLD_NEAR_EPSILON: f32 = 0.05;

pub fn rpe_threshold_band(score: f32, threshold: f32) -> &'static str {
    if (score - threshold).abs() <= RPE_THRESHOLD_NEAR_EPSILON {
        "near"
    } else {
        "far"
    }
}

/// Number of consecutive RPE search failures before the circuit opens.
pub const RPE_CIRCUIT_FAILURE_THRESHOLD: u32 = 5;
/// Seconds the RPE circuit stays open before attempting a probe.
pub const RPE_CIRCUIT_OPEN_SECS: u64 = 30;

/// Half-open circuit breaker for `compute_rpe` vector-search calls.
///
/// When `RPE_CIRCUIT_FAILURE_THRESHOLD` consecutive searches fail
/// (e.g. storage unavailable) the circuit opens and `compute_rpe` returns
/// a default `RpeResult` immediately without hitting the store.
/// After `RPE_CIRCUIT_OPEN_SECS` the circuit moves to half-open and allows
/// a single probe. A successful probe closes the circuit; another failure
/// resets the open timer.
#[derive(Debug, Default)]
pub struct RpeCircuitBreaker {
    consecutive_failures: AtomicU32,
    open_until_unix_secs: AtomicU64,
}

impl RpeCircuitBreaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns `true` if the circuit is currently open (calls should be skipped).
    pub fn is_open(&self) -> bool {
        let until = self.open_until_unix_secs.load(Ordering::Relaxed);
        if until == 0 {
            return false;
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now < until
    }

    /// Record a successful RPE call; closes the circuit.
    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.open_until_unix_secs.store(0, Ordering::Relaxed);
    }

    /// Record a failed RPE call; opens the circuit after the threshold.
    pub fn record_failure(&self, open_secs: u64) {
        let failures = self
            .consecutive_failures
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        if failures >= RPE_CIRCUIT_FAILURE_THRESHOLD {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            // Only advance the timer (never retreat it on concurrent calls).
            let target = now.saturating_add(open_secs);
            let _ = self.open_until_unix_secs.fetch_update(
                Ordering::Relaxed,
                Ordering::Relaxed,
                |current| if target > current { Some(target) } else { None },
            );
        }
    }
}

/// Compute RPE score for an incoming memory based on embedding similarity.
///
/// RPE = (1 - max_similarity) × (1 + z_score), clamped to [0, 2].
///
/// The z-score is computed against a **running population** of historical
/// distance values (via `RunningRpeStats`), not against the neighbor
/// distances from this single search. This avoids the mathematical
/// impossibility of z-scoring the minimum of a set against itself
/// (which always yields ≤ 0).
///
/// When `stats.count < 2`, z_score = 0 and RPE = distance.
///
/// If `circuit_breaker` is open (too many consecutive storage failures) this
/// function returns a default `RpeResult` immediately without touching the
/// store, and emits a `tracing::warn` once per circuit-open event.
pub async fn compute_rpe(
    storage: &dyn PhysicalStore,
    embedding: &[f32],
    threshold: f32,
    search_limit: usize,
    stats: &mut RunningRpeStats,
    circuit_breaker: &RpeCircuitBreaker,
) -> RpeResult {
    // ── Circuit-breaker check ─────────────────────────────────────────────
    if circuit_breaker.is_open() {
        tracing::warn!("RPE circuit open — skipping vector search, using fast-path default");
        return RpeResult {
            score: 0.0,
            max_similarity: 1.0,
            is_fast_path: true,
        };
    }

    let datasets = ["episodic", "semantic", "procedural"];
    let mut max_sim: f32 = 0.0;
    let mut any_search_error = false;

    for dataset in &datasets {
        let exists = matches!(storage.exists(dataset).await, Ok(true));
        if !exists {
            continue;
        }

        let opts = VectorSearchOptions {
            query: embedding.to_vec(),
            column: "embedding".into(),
            limit: search_limit,
            ..Default::default()
        };

        match storage.vector_search(dataset, opts).await {
            Ok(batches) => {
                for batch in &batches {
                    if let Some(dist_col) = batch.column_by_name("_distance") {
                        if let Some(dists) = dist_col
                            .as_any()
                            .downcast_ref::<arrow_array::Float32Array>()
                        {
                            for j in 0..dists.len() {
                                if !dists.is_null(j) {
                                    let dist = dists.value(j);
                                    let sim = 1.0 / (1.0 + dist);
                                    max_sim = max_sim.max(sim);
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!(dataset, error = %e, "RPE vector search failed");
                any_search_error = true;
            }
        }
    }

    // Update circuit-breaker state based on whether any search error occurred.
    if any_search_error {
        circuit_breaker.record_failure(RPE_CIRCUIT_OPEN_SECS);
    } else {
        circuit_breaker.record_success();
    }

    let distance = 1.0 - max_sim;

    // Z-score: compare this memory's distance against the historical
    // population of distances from all prior writes. Positive z = more
    // novel than average (amplify), negative z = less novel (attenuate).
    let z_score = stats.z_score(distance as f64) as f32;

    // Feed this distance into the running population for future z-scores.
    stats.update(distance as f64);

    // RPE = distance × (1 + z_score), clamped to [0, 2].
    let rpe = (distance * (1.0 + z_score)).clamp(0.0, 2.0);

    RpeResult {
        score: rpe,
        max_similarity: max_sim,
        is_fast_path: rpe < threshold,
    }
}

/// Compute per-record max-similarity against episodic/semantic/procedural in batch.
///
/// Replaces the N×3 serial `vector_search` calls in `batch_remember` with 3 batched
/// `vector_search_many` calls (one per dataset). Returns one `f32` max_similarity
/// value per input embedding (index-aligned); 0.0 when no existing memories are found.
///
/// Circuit-breaker semantics mirror `compute_rpe`: open circuit → all results 1.0
/// (fast-path all records). Failure in any single dataset still updates the breaker.
/// Output of a batch vector-search operation.
pub struct BatchSearchResult {
    /// Per-embedding max-similarity values (same length as input).
    pub max_sims: Vec<f32>,
    /// `true` if at least one dataset search returned an error.
    pub had_storage_error: bool,
}

/// Batch RPE vector search across the 3 cognitive datasets.
///
/// Circuit-breaker logic is **not** applied here — callers are responsible
/// for pre-checking the circuit and recording success/failure on the
/// per-partition circuit breakers after this call returns.
///
/// Returns `None` if `embeddings` is empty.
pub async fn batch_vector_search_max_sim(
    storage: &dyn PhysicalStore,
    embeddings: &[Vec<f32>],
    search_limit: usize,
) -> Option<BatchSearchResult> {
    if embeddings.is_empty() {
        return None;
    }

    let n = embeddings.len();
    let mut max_sims = vec![0.0_f32; n];
    let mut had_storage_error = false;

    let datasets = ["episodic", "semantic", "procedural"];

    for dataset in &datasets {
        let Ok(true) = storage.exists(dataset).await else {
            continue;
        };

        let queries: Vec<VectorSearchOptions> = embeddings
            .iter()
            .map(|emb| VectorSearchOptions {
                query: emb.clone(),
                column: "embedding".into(),
                limit: search_limit,
                ..Default::default()
            })
            .collect();

        match storage.vector_search_many(dataset, queries).await {
            Ok(results) => {
                for (i, batches) in results.into_iter().enumerate() {
                    for batch in &batches {
                        if let Some(dist_col) = batch.column_by_name("_distance") {
                            if let Some(dists) =
                                dist_col.as_any().downcast_ref::<arrow_array::Float32Array>()
                            {
                                for j in 0..dists.len() {
                                    if !dists.is_null(j) {
                                        let sim = 1.0 / (1.0 + dists.value(j));
                                        max_sims[i] = max_sims[i].max(sim);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                tracing::debug!(dataset, error = %e, "RPE batch vector search failed");
                had_storage_error = true;
            }
        }
    }

    Some(BatchSearchResult {
        max_sims,
        had_storage_error,
    })
}

/// Generate heuristic questions and store as prospective implications.
///
/// Returns the number of implications stored, or 0 on failure.
pub async fn prepare_prospective_implications_batch(
    embedder: &dyn Embedder,
    source_id: MemoryId,
    content: &str,
    num_questions: usize,
    timeout_secs: u64,
    templates: &[String],
    namespace: &str,
) -> Option<RecordBatch> {
    let words: Vec<&str> = content.split_whitespace().collect();
    if words.len() < 3 {
        return None;
    }

    let truncated = hirn_core::text_util::truncate_at_word_boundary(content, 80);
    let questions: Vec<String> = templates
        .iter()
        .take(num_questions)
        .map(|t| t.replace("{content}", &truncated))
        .collect();

    if questions.is_empty() {
        return None;
    }

    let refs: Vec<&str> = questions.iter().map(|q| q.as_str()).collect();
    let embeddings = match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        embedder.embed(&refs),
    )
    .await
    {
        Ok(Ok(embs)) => embs,
        Ok(Err(e)) => {
            tracing::warn!(error = %e, "Prospective embedding failed");
            return None;
        }
        Err(_) => {
            tracing::warn!(timeout_secs, "Prospective embedding timed out");
            return None;
        }
    };

    if embeddings.is_empty() {
        return None;
    }

    if embeddings.len() != questions.len() {
        tracing::warn!(
            expected = questions.len(),
            actual = embeddings.len(),
            "Prospective embedding count mismatch"
        );
    }

    let count = questions.len().min(embeddings.len());
    if count == 0 {
        return None;
    }

    let records: Vec<ProspectiveImplication> = questions
        .into_iter()
        .take(count)
        .map(|question| ProspectiveImplication::new(source_id, question))
        .collect();
    let embedding_values: Vec<Option<Vec<f32>>> = embeddings
        .into_iter()
        .take(count)
        .map(|embedding| Some(embedding.vector))
        .collect();
    let embedding_dims = embedding_values
        .iter()
        .find_map(|embedding| embedding.as_ref().map(Vec::len))
        .unwrap_or(0);
    let namespaces: Vec<&str> = std::iter::repeat_n(namespace, count).collect();

    match hirn_storage::datasets::prospective_implications::to_batch_with_namespaces(
        &records,
        &embedding_values,
        &namespaces,
        embedding_dims,
    ) {
        Ok(batch) => Some(batch),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to build prospective batch");
            None
        }
    }
}

pub async fn store_prospective_implications(
    storage: &dyn PhysicalStore,
    embedder: &dyn Embedder,
    source_id: MemoryId,
    content: &str,
    num_questions: usize,
    timeout_secs: u64,
    templates: &[String],
    namespace: &str,
) -> usize {
    let Some(batch) = prepare_prospective_implications_batch(
        embedder,
        source_id,
        content,
        num_questions,
        timeout_secs,
        templates,
        namespace,
    )
    .await
    else {
        return 0;
    };

    let count = batch.num_rows();

    match storage
        .append(
            hirn_storage::datasets::prospective_implications::DATASET_NAME,
            batch,
        )
        .await
    {
        Ok(()) => count,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to write prospective implications");
            0
        }
    }
}

/// Extract SVO events from content and store them.
///
/// Returns the number of SVO events stored, or 0 on failure.
pub fn prepare_svo_events_batch(
    source_id: MemoryId,
    content: &str,
    confidence_threshold: f32,
    namespace: &str,
    embedding_dims: usize,
) -> Option<RecordBatch> {
    // Reuse the regex extraction from hirn-exec.
    let events = hirn_exec::operators::extract_svo_regex(content, confidence_threshold);

    if events.is_empty() {
        return None;
    }

    let count = events.len();
    let records: Vec<SvoEvent> = events
        .into_iter()
        .map(|event| {
            SvoEvent::from_extraction(
                event.subject,
                event.verb,
                event.object,
                event.time_start,
                event.time_end,
                event.confidence,
                vec![source_id],
            )
        })
        .collect();
    let embeddings = vec![None; count];
    let namespaces: Vec<&str> = std::iter::repeat_n(namespace, count).collect();

    match hirn_storage::datasets::svo_events::to_batch_with_namespaces(
        &records,
        &embeddings,
        &namespaces,
        embedding_dims,
    ) {
        Ok(batch) => Some(batch),
        Err(e) => {
            tracing::warn!(error = %e, "Failed to build SVO batch");
            None
        }
    }
}

pub async fn extract_and_store_svo_events(
    storage: &dyn PhysicalStore,
    source_id: MemoryId,
    content: &str,
    confidence_threshold: f32,
    namespace: &str,
    embedding_dims: usize,
) -> usize {
    let Some(batch) = prepare_svo_events_batch(
        source_id,
        content,
        confidence_threshold,
        namespace,
        embedding_dims,
    ) else {
        return 0;
    };

    let count = batch.num_rows();

    match storage
        .append(hirn_storage::datasets::svo_events::DATASET_NAME, batch)
        .await
    {
        Ok(()) => count,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to write SVO events");
            0
        }
    }
}

// ── Adaptive Consolidation ──────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PendingConsolidation {
    snapshot_scores: std::collections::HashMap<hirn_core::types::Namespace, f32>,
}

/// Root cause for an interference-driven consolidation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceTriggerCause {
    ThresholdExceeded,
}

impl InterferenceTriggerCause {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ThresholdExceeded => "threshold_exceeded",
        }
    }
}

/// Reason a consolidation request was suppressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterferenceSuppressionReason {
    CooldownActive,
    AwaitingFeedback,
}

impl InterferenceSuppressionReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CooldownActive => "cooldown_active",
            Self::AwaitingFeedback => "awaiting_feedback",
        }
    }
}

/// Result of interference accumulation check.
#[derive(Debug)]
pub enum InterferenceAction {
    /// Below threshold, no action needed.
    None,
    /// Threshold exceeded and cooldown elapsed — trigger consolidation.
    TriggerConsolidation {
        /// Namespaces that accumulated interference — scope consolidation to these.
        namespaces: Vec<hirn_core::types::Namespace>,
        /// Total unresolved backlog score when the request was emitted.
        backlog_score: f32,
        /// Why this request was emitted.
        cause: InterferenceTriggerCause,
    },
    /// Threshold exceeded but a new request would be redundant.
    Suppressed {
        reason: InterferenceSuppressionReason,
        backlog_score: f32,
    },
}

/// Feedback from an actual consolidation execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationFeedback {
    Succeeded { progress_made: bool },
    Failed,
}

/// Outcome after applying consolidation feedback to the backlog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationFeedbackOutcome {
    ProgressRecorded,
    NoProgress,
    Failed,
    NoPendingTrigger,
}

impl ConsolidationFeedbackOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProgressRecorded => "progress",
            Self::NoProgress => "no_progress",
            Self::Failed => "failed",
            Self::NoPendingTrigger => "no_pending_trigger",
        }
    }
}

/// Result of applying consolidation feedback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsolidationFeedbackResult {
    pub outcome: ConsolidationFeedbackOutcome,
    pub reduced_score: f32,
    pub remaining_score: f32,
}

/// Snapshot of the current interference backlog state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterferenceStateSnapshot {
    pub backlog_score: f32,
    pub namespace_count: usize,
    pub awaiting_feedback: bool,
}

fn sorted_namespaces(
    namespaces: impl Iterator<Item = hirn_core::types::Namespace>,
) -> Vec<hirn_core::types::Namespace> {
    let mut namespaces: Vec<_> = namespaces.collect();
    namespaces.sort_by(|left, right| left.as_str().cmp(right.as_str()));
    namespaces
}

/// Compute interference score for a write based on max similarity to existing memories.
/// High similarity = high interference (near-duplicate or supersession).
pub fn interference_score_from_similarity(max_similarity: f32) -> f32 {
    // Interference peaks when similarity is high (near-duplicate territory).
    // Score = max(0, similarity - 0.5) * 2, clamped to [0, 1].
    ((max_similarity - 0.5) * 2.0).clamp(0.0, 1.0)
}

// ── ShardedInterferenceTracker ──────────────────────────────────────────

/// Global trigger-control state — only locked when the total backlog exceeds
/// the configured threshold (rare path).
#[derive(Debug, Default)]
struct TriggerState {
    last_trigger: Option<std::time::Instant>,
    pending_consolidation: Option<PendingConsolidation>,
}

/// Lock-split interference tracker for the high-throughput write path.
///
/// **Hot path (score accumulation):** updates a single `DashMap` shard for the
/// affected namespace.  Under 30 K+ writes/sec the per-shard lock eliminates
/// the global-Mutex bottleneck of the single-lock `InterferenceTracker`.
///
/// **Control path (trigger/cooldown):** acquires a narrow
/// `Mutex<TriggerState>` only when the total backlog exceeds the configured
/// threshold — which is rare in well-operating systems.
///
/// # Semantics
///
/// Trigger/cooldown/feedback semantics are identical to `InterferenceTracker`;
/// this type is a concurrency-optimised drop-in replacement for use in
/// `WriteRuntime`.
pub struct ShardedInterferenceTracker {
    /// Per-namespace backlog scores.  `DashMap` provides independent shard
    /// locking so writes to different namespaces never contend.
    backlog: dashmap::DashMap<hirn_core::types::Namespace, f32>,
    /// Narrow global lock — only held during the threshold-exceeded code path.
    trigger: parking_lot::Mutex<TriggerState>,
}

impl Default for ShardedInterferenceTracker {
    fn default() -> Self {
        Self {
            backlog: dashmap::DashMap::new(),
            trigger: parking_lot::Mutex::new(TriggerState::default()),
        }
    }
}

impl std::fmt::Debug for ShardedInterferenceTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShardedInterferenceTracker")
            .field("namespace_count", &self.backlog.len())
            .field("total_backlog_score", &self.total_backlog_score())
            .finish_non_exhaustive()
    }
}

impl ShardedInterferenceTracker {
    /// Accumulate interference score from a write operation.
    ///
    /// **Fast path (below threshold):** updates only the per-namespace DashMap
    /// shard — no global lock acquired.
    ///
    /// **Slow path (threshold exceeded):** acquires `Mutex<TriggerState>` to
    /// check cooldown and update pending-consolidation state.
    pub fn accumulate(
        &self,
        score: f32,
        namespace: hirn_core::types::Namespace,
        threshold: f32,
        cooldown_secs: u64,
    ) -> InterferenceAction {
        // Hot path: update per-namespace score (DashMap shard lock only).
        *self.backlog.entry(namespace).or_default() += score.max(0.0);

        // Fast-path check: read total without the narrow trigger lock.
        let backlog_score = self.total_backlog_score();
        if backlog_score < threshold {
            return InterferenceAction::None;
        }

        // Slow path: threshold exceeded — acquire the narrow trigger lock.
        let mut trigger = self.trigger.lock();

        if trigger.pending_consolidation.is_some() {
            return InterferenceAction::Suppressed {
                reason: InterferenceSuppressionReason::AwaitingFeedback,
                backlog_score,
            };
        }

        if let Some(last) = trigger.last_trigger {
            if last.elapsed().as_secs() < cooldown_secs {
                return InterferenceAction::Suppressed {
                    reason: InterferenceSuppressionReason::CooldownActive,
                    backlog_score,
                };
            }
        }

        // Take a consistent snapshot of current scores under the trigger lock.
        let snapshot_scores: std::collections::HashMap<_, _> =
            self.backlog.iter().map(|e| (*e.key(), *e.value())).collect();
        let namespaces = sorted_namespaces(snapshot_scores.keys().copied());

        trigger.pending_consolidation = Some(PendingConsolidation { snapshot_scores });
        trigger.last_trigger = Some(std::time::Instant::now());

        tracing::info!(
            namespace_count = namespaces.len(),
            backlog_score,
            "Interference-driven consolidation triggered"
        );

        InterferenceAction::TriggerConsolidation {
            namespaces,
            backlog_score,
            cause: InterferenceTriggerCause::ThresholdExceeded,
        }
    }

    /// Apply feedback from a completed consolidation run.
    pub fn record_consolidation_feedback(
        &self,
        feedback: ConsolidationFeedback,
    ) -> ConsolidationFeedbackResult {
        let mut trigger = self.trigger.lock();
        let Some(pending) = trigger.pending_consolidation.take() else {
            return ConsolidationFeedbackResult {
                outcome: ConsolidationFeedbackOutcome::NoPendingTrigger,
                reduced_score: 0.0,
                remaining_score: self.total_backlog_score(),
            };
        };

        match feedback {
            ConsolidationFeedback::Succeeded { progress_made: true } => {
                let reduced_score = self.subtract_snapshot(&pending.snapshot_scores);
                ConsolidationFeedbackResult {
                    outcome: ConsolidationFeedbackOutcome::ProgressRecorded,
                    reduced_score,
                    remaining_score: self.total_backlog_score(),
                }
            }
            ConsolidationFeedback::Succeeded {
                progress_made: false,
            } => ConsolidationFeedbackResult {
                outcome: ConsolidationFeedbackOutcome::NoProgress,
                reduced_score: 0.0,
                remaining_score: self.total_backlog_score(),
            },
            ConsolidationFeedback::Failed => ConsolidationFeedbackResult {
                outcome: ConsolidationFeedbackOutcome::Failed,
                reduced_score: 0.0,
                remaining_score: self.total_backlog_score(),
            },
        }
    }

    /// Total interference backlog score summed across all namespaces.
    pub fn total_backlog_score(&self) -> f32 {
        self.backlog.iter().map(|e| *e.value()).sum()
    }

    /// Snapshot of the current interference state.
    pub fn snapshot(&self) -> InterferenceStateSnapshot {
        let trigger = self.trigger.lock();
        InterferenceStateSnapshot {
            backlog_score: self.total_backlog_score(),
            namespace_count: self.backlog.len(),
            awaiting_feedback: trigger.pending_consolidation.is_some(),
        }
    }

    fn subtract_snapshot(
        &self,
        snapshot_scores: &std::collections::HashMap<hirn_core::types::Namespace, f32>,
    ) -> f32 {
        let mut reduced_score = 0.0;
        for (&namespace, &snapshot_score) in snapshot_scores {
            if let Some(mut entry) = self.backlog.get_mut(&namespace) {
                let applied = (*entry).min(snapshot_score);
                *entry -= applied;
                reduced_score += applied;
            }
        }
        // Prune zero-score entries to keep the map compact.
        self.backlog.retain(|_, v| *v > 1e-6);
        reduced_score
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interference_score_low_similarity() {
        assert_eq!(interference_score_from_similarity(0.3), 0.0);
        assert_eq!(interference_score_from_similarity(0.5), 0.0);
    }

    #[test]
    fn interference_score_high_similarity() {
        let score = interference_score_from_similarity(0.8);
        assert!((score - 0.6).abs() < 1e-6);
        assert_eq!(interference_score_from_similarity(1.0), 1.0);
    }

    #[test]
    fn tracker_below_threshold_returns_none() {
        let ns = hirn_core::types::Namespace::default();
        let tracker = ShardedInterferenceTracker::default();
        let action = tracker.accumulate(0.1, ns, 0.3, 300);
        assert!(matches!(action, InterferenceAction::None));
    }

    #[test]
    fn tracker_triggers_above_threshold() {
        let ns = hirn_core::types::Namespace::default();
        let tracker = ShardedInterferenceTracker::default();
        // Accumulate past threshold.
        let _ = tracker.accumulate(0.2, ns, 0.3, 300);
        let action = tracker.accumulate(0.2, ns, 0.3, 300);
        assert!(matches!(
            action,
            InterferenceAction::TriggerConsolidation {
                backlog_score,
                cause: InterferenceTriggerCause::ThresholdExceeded,
                ..
            } if backlog_score >= 0.4
        ));
    }

    #[test]
    fn tracker_suppresses_while_feedback_is_pending() {
        let ns = hirn_core::types::Namespace::default();
        let tracker = ShardedInterferenceTracker::default();
        let _ = tracker.accumulate(0.4, ns, 0.3, 300);
        let action = tracker.accumulate(0.4, ns, 0.3, 300);
        assert!(matches!(
            action,
            InterferenceAction::Suppressed {
                reason: InterferenceSuppressionReason::AwaitingFeedback,
                backlog_score,
            } if backlog_score >= 0.8
        ));
    }

    #[test]
    fn tracker_success_feedback_preserves_post_trigger_backlog() {
        let ns_a = hirn_core::types::Namespace::default();
        let ns_b = hirn_core::types::Namespace::shared();
        let tracker = ShardedInterferenceTracker::default();
        let _ = tracker.accumulate(0.4, ns_a, 0.3, 300);
        let _ = tracker.accumulate(0.2, ns_a, 0.3, 300);
        let _ = tracker.accumulate(0.3, ns_b, 0.3, 300);

        let feedback = tracker.record_consolidation_feedback(ConsolidationFeedback::Succeeded {
            progress_made: true,
        });
        assert!(
            matches!(
                feedback,
                ConsolidationFeedbackResult {
                    outcome: ConsolidationFeedbackOutcome::ProgressRecorded,
                    ..
                }
            ),
            "Expected progress to be recorded, got {feedback:?}",
        );
        assert!((feedback.reduced_score - 0.4).abs() < 1e-6);
        assert!((feedback.remaining_score - 0.5).abs() < 1e-6);

        let snapshot = tracker.snapshot();
        assert!((snapshot.backlog_score - 0.5).abs() < 1e-6);
        assert_eq!(snapshot.namespace_count, 2);
        assert!(!snapshot.awaiting_feedback);
    }

    #[test]
    fn tracker_no_progress_feedback_keeps_backlog() {
        let ns = hirn_core::types::Namespace::default();
        let tracker = ShardedInterferenceTracker::default();
        let _ = tracker.accumulate(0.4, ns, 0.3, 300);

        let feedback = tracker.record_consolidation_feedback(ConsolidationFeedback::Succeeded {
            progress_made: false,
        });

        assert_eq!(feedback.outcome, ConsolidationFeedbackOutcome::NoProgress);
        assert_eq!(feedback.reduced_score, 0.0);
        assert!((feedback.remaining_score - 0.4).abs() < 1e-6);
    }

    #[test]
    fn tracker_failure_feedback_keeps_backlog() {
        let ns = hirn_core::types::Namespace::default();
        let tracker = ShardedInterferenceTracker::default();
        let _ = tracker.accumulate(0.4, ns, 0.3, 300);

        let feedback = tracker.record_consolidation_feedback(ConsolidationFeedback::Failed);

        assert_eq!(feedback.outcome, ConsolidationFeedbackOutcome::Failed);
        assert_eq!(feedback.reduced_score, 0.0);
        assert!((feedback.remaining_score - 0.4).abs() < 1e-6);
    }

    #[test]
    fn tracker_cooldown_is_checked_after_feedback() {
        let ns = hirn_core::types::Namespace::default();
        let tracker = ShardedInterferenceTracker::default();
        let _ = tracker.accumulate(0.4, ns, 0.3, 300);
        let _ = tracker.record_consolidation_feedback(ConsolidationFeedback::Succeeded {
            progress_made: true,
        });

        let action = tracker.accumulate(0.4, ns, 0.3, 300);
        assert!(matches!(
            action,
            InterferenceAction::Suppressed {
                reason: InterferenceSuppressionReason::CooldownActive,
                backlog_score,
            } if backlog_score >= 0.4
        ));
    }

    #[test]
    fn tracker_scopes_namespaces() {
        let ns_a = hirn_core::types::Namespace::default();
        let ns_b = hirn_core::types::Namespace::shared();
        let tracker = ShardedInterferenceTracker::default();
        let _ = tracker.accumulate(0.1, ns_a, 0.3, 300);
        let action = tracker.accumulate(0.3, ns_b, 0.3, 300);
        assert!(matches!(
            action,
            InterferenceAction::TriggerConsolidation { ref namespaces, .. }
                if namespaces.len() == 2
                    && namespaces.contains(&ns_a)
                    && namespaces.contains(&ns_b)
        ));
    }

    // ── RunningRpeStats tests ───────────────────────────────────────

    #[test]
    fn rpe_stats_empty_returns_zero_zscore() {
        let stats = RunningRpeStats::default();
        assert_eq!(stats.z_score(0.5), 0.0);
    }

    #[test]
    fn rpe_stats_single_sample_returns_zero_zscore() {
        let mut stats = RunningRpeStats::default();
        stats.update(0.5);
        assert_eq!(stats.z_score(0.5), 0.0);
    }

    #[test]
    fn rpe_stats_novel_content_gets_positive_zscore() {
        let mut stats = RunningRpeStats::default();
        // Simulate several writes with varying familiar distances.
        for d in &[0.1, 0.15, 0.12, 0.08, 0.11, 0.09, 0.14, 0.13, 0.10, 0.07] {
            stats.update(*d);
        }
        // Novel content: distance = 0.9, well above the mean of ~0.11.
        let z = stats.z_score(0.9);
        assert!(
            z > 0.0,
            "Novel content should get positive z-score, got {z}",
        );
    }

    #[test]
    fn rpe_stats_familiar_content_gets_negative_zscore() {
        let mut stats = RunningRpeStats::default();
        // Simulate several writes with varying novel distances.
        for d in &[0.7, 0.8, 0.75, 0.85, 0.9, 0.72, 0.88, 0.79, 0.82, 0.77] {
            stats.update(*d);
        }
        // Familiar content: distance = 0.05, well below the mean of ~0.8.
        let z = stats.z_score(0.05);
        assert!(
            z < 0.0,
            "Familiar content should get negative z-score, got {z}",
        );
    }

    #[test]
    fn rpe_stats_identical_samples_return_zero_zscore() {
        let mut stats = RunningRpeStats::default();
        for _ in 0..5 {
            stats.update(0.5);
        }
        // Zero stddev → z-score = 0.
        assert_eq!(stats.z_score(0.5), 0.0);
    }
}

// ── Background Embed Retry ──────────────────────────────────────────

/// Tracks memory IDs that were stored without embeddings due to provider
/// failure, enabling background retry when the provider recovers.
#[derive(Debug)]
pub struct PendingEmbedQueue {
    /// Memory IDs awaiting embedding, with retry metadata.
    pending: std::collections::VecDeque<PendingEmbed>,
    /// Maximum number of pending items to retain (prevents unbounded growth).
    max_capacity: usize,
}

#[derive(Debug, Clone)]
pub struct PendingEmbed {
    /// The memory that needs embedding.
    pub id: MemoryId,
    /// Number of retry attempts so far.
    pub attempts: u32,
    /// When the embed failure occurred (used for backoff scheduling).
    #[allow(dead_code)] // Public API — used by consumers for backoff decisions
    pub enqueued_at: std::time::Instant,
}

impl Default for PendingEmbedQueue {
    fn default() -> Self {
        Self {
            pending: std::collections::VecDeque::new(),
            max_capacity: 10_000,
        }
    }
}

impl PendingEmbedQueue {
    /// Enqueue a memory ID for background embed retry.
    pub fn enqueue(&mut self, id: MemoryId) {
        if self.pending.len() >= self.max_capacity {
            // Drop oldest entry to prevent unbounded growth.
            self.pending.pop_front();
            tracing::warn!(
                max_capacity = self.max_capacity,
                "PendingEmbedQueue capacity reached, dropping oldest entry"
            );
        }
        self.pending.push_back(PendingEmbed {
            id,
            attempts: 0,
            enqueued_at: std::time::Instant::now(),
        });
    }

    /// Drain all pending items for processing. Returns the items and
    /// clears the queue.
    pub fn drain_all(&mut self) -> Vec<PendingEmbed> {
        self.pending.drain(..).collect()
    }

    /// Return items that failed retry back to the queue with incremented
    /// attempt count. Items exceeding `max_attempts` are dropped.
    pub fn requeue_failed(&mut self, items: Vec<PendingEmbed>, max_attempts: u32) {
        for mut item in items {
            item.attempts += 1;
            if item.attempts < max_attempts {
                self.pending.push_back(item);
            } else {
                tracing::warn!(
                    id = %item.id,
                    attempts = item.attempts,
                    "Dropping pending embed after max retry attempts"
                );
            }
        }
    }

    /// Number of pending items.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether the queue is empty.
    #[allow(dead_code)] // Public API — checked by callers to decide whether to retry
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}
