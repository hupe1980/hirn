//! Token Budget Gate — enforces per-agent memory token budgets.
//!
//! Tracks the total token count for each agent and rejects new memories
//! that would push the agent over its budget.
//!
//! Accounting is reserve/commit: `evaluate` atomically reserves the
//! candidate's tokens against `persisted + reserved` (so two concurrent
//! evaluations can never jointly exceed the budget), and the reservation is
//! converted to persisted usage by `commit` or dropped by `release` — the
//! admission pipeline guarantees exactly one of the two happens. Without
//! this, a candidate rejected by a *later* controller (or a failed write)
//! left the speculative total inflated until the next invalidation.

use std::collections::HashMap;
use std::sync::Arc;

use futures::TryStreamExt;
use hirn_core::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::tokenizer::Tokenizer;
use hirn_core::types::AgentId;
use hirn_storage::PhysicalStore;
use hirn_storage::store::ScanOptions;
use tokio::sync::RwLock;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};

/// Per-agent usage: confirmed persisted tokens + in-flight reservations.
#[derive(Debug, Default, Clone, Copy)]
struct AgentTokens {
    persisted: usize,
    reserved: usize,
}

/// All mutable gate state behind one lock so read-check-reserve is atomic.
#[derive(Debug, Default)]
struct BudgetState {
    agents: HashMap<AgentId, AgentTokens>,
    /// Candidate id → (agent, reserved amount) for commit/release.
    reservations: HashMap<MemoryId, (AgentId, usize)>,
}

/// Cognitive datasets whose per-agent content counts against the budget.
///
/// R-57: the budget previously counted only `episodic`, so semantic and
/// procedural writes escaped it entirely. All three long-term cognitive tiers
/// are summed so an agent cannot bypass its budget by writing to another tier.
const COGNITIVE_DATASETS: [&str; 3] = [
    hirn_storage::datasets::episodic::DATASET_NAME,
    hirn_storage::datasets::semantic::DATASET_NAME,
    hirn_storage::datasets::procedural::DATASET_NAME,
];

/// Per-agent token budget enforcement.
pub struct TokenBudgetGate {
    storage: Arc<dyn PhysicalStore>,
    tokenizer: Arc<dyn Tokenizer>,
    /// Datasets summed for an agent's usage. Multiple entries are added
    /// together (R-57) so writes across cognitive tiers all count.
    datasets: Vec<String>,
    /// Maximum tokens per agent. Default: 500_000.
    max_tokens: usize,
    state: RwLock<BudgetState>,
}

impl TokenBudgetGate {
    pub fn new(
        storage: Arc<dyn PhysicalStore>,
        tokenizer: Arc<dyn Tokenizer>,
        dataset: impl Into<String>,
        max_tokens: usize,
    ) -> Self {
        Self {
            storage,
            tokenizer,
            datasets: vec![dataset.into()],
            max_tokens,
            state: RwLock::new(BudgetState::default()),
        }
    }

    /// Enforce the budget across ALL cognitive datasets (episodic + semantic +
    /// procedural). This is the constructor the default write pipeline uses so
    /// a per-agent budget cannot be bypassed by writing to a non-episodic tier
    /// (R-57).
    pub fn new_cognitive(
        storage: Arc<dyn PhysicalStore>,
        tokenizer: Arc<dyn Tokenizer>,
        max_tokens: usize,
    ) -> Self {
        Self {
            storage,
            tokenizer,
            datasets: COGNITIVE_DATASETS
                .iter()
                .map(|d| (*d).to_string())
                .collect(),
            max_tokens,
            state: RwLock::new(BudgetState::default()),
        }
    }

    /// Create with the default budget of 500,000 tokens per agent.
    pub fn with_defaults(
        storage: Arc<dyn PhysicalStore>,
        tokenizer: Arc<dyn Tokenizer>,
        dataset: impl Into<String>,
    ) -> Self {
        Self::new(storage, tokenizer, dataset, 500_000)
    }

    /// Invalidate the cached usage for an agent (e.g., after a forget
    /// operation). The agent's in-flight reservations are dropped too — a
    /// pending commit for one of them becomes a no-op, and the next
    /// evaluation re-scans storage, which by then reflects the write.
    pub async fn invalidate(&self, agent_id: &AgentId) {
        let mut state = self.state.write().await;
        state.agents.remove(agent_id);
        state.reservations.retain(|_, (agent, _)| agent != agent_id);
    }

    /// Invalidate all cached counts and reservations.
    pub async fn invalidate_all(&self) {
        let mut state = self.state.write().await;
        state.agents.clear();
        state.reservations.clear();
    }

    /// Compute the current token count for an agent by scanning storage,
    /// summed across every configured dataset (R-57).
    async fn compute_tokens(&self, agent_id: &AgentId) -> HirnResult<usize> {
        let mut total_tokens = 0usize;
        for dataset in &self.datasets {
            total_tokens += self.compute_tokens_for_dataset(dataset, agent_id).await?;
        }
        Ok(total_tokens)
    }

    /// Sum the tokens an agent's content occupies in a single dataset.
    ///
    /// Episodic exposes top-level `agent_id`/`content` columns, so its scan
    /// pushes the agent filter down to Lance and reads only `content`. Semantic
    /// and procedural keep the authoring agent inside `provenance_json` and have
    /// no single `content` column, so they are decoded to typed records and the
    /// authoring agent is matched in-process (R-57).
    async fn compute_tokens_for_dataset(
        &self,
        dataset: &str,
        agent_id: &AgentId,
    ) -> HirnResult<usize> {
        let exists = self
            .storage
            .exists(dataset)
            .await
            .map_err(hirn_core::HirnError::storage)?;
        if !exists {
            return Ok(0);
        }

        if dataset == hirn_storage::datasets::episodic::DATASET_NAME {
            self.compute_episodic_tokens(dataset, agent_id).await
        } else {
            self.compute_decoded_tokens(dataset, agent_id).await
        }
    }

    /// Episodic: push the agent filter down and read only the `content` column.
    async fn compute_episodic_tokens(
        &self,
        dataset: &str,
        agent_id: &AgentId,
    ) -> HirnResult<usize> {
        let agent_str = agent_id.as_str();
        let options = ScanOptions {
            columns: Some(vec!["content".into()]),
            filter: Some(format!("agent_id = '{}'", agent_str.replace('\'', "''"))),
            exact_filter: None,
            order_by: None,
            limit: None,
            offset: None,
        };

        let mut batches = self
            .storage
            .scan_stream(dataset, options)
            .await
            .map_err(hirn_core::HirnError::storage)?;

        let mut total_tokens = 0usize;
        while let Some(batch) = batches
            .try_next()
            .await
            .map_err(hirn_core::HirnError::storage)?
        {
            use arrow_array::Array;
            let content_col = batch.column_by_name("content");
            let content_arr = match content_col {
                Some(c) => c,
                None => continue,
            };

            if let Some(arr) = content_arr
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
            {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        total_tokens += self.tokenizer.count_tokens(arr.value(i));
                    }
                }
            }
        }

        Ok(total_tokens)
    }

    /// Semantic/procedural: the agent lives in `provenance_json` and there is no
    /// single text column, so decode typed records and match the author
    /// in-process, counting the tokens of each record's text fields.
    async fn compute_decoded_tokens(&self, dataset: &str, agent_id: &AgentId) -> HirnResult<usize> {
        let batches = self
            .storage
            .scan(dataset, ScanOptions::default())
            .await
            .map_err(hirn_core::HirnError::storage)?;

        let mut total_tokens = 0usize;
        for batch in &batches {
            if dataset == hirn_storage::datasets::semantic::DATASET_NAME {
                for record in hirn_storage::datasets::semantic::from_batch(batch)
                    .map_err(hirn_core::HirnError::storage)?
                {
                    if record.provenance.created_by == *agent_id {
                        total_tokens += self.tokenizer.count_tokens(&record.concept);
                        total_tokens += self.tokenizer.count_tokens(&record.description);
                    }
                }
            } else if dataset == hirn_storage::datasets::procedural::DATASET_NAME {
                for record in hirn_storage::datasets::procedural::from_batch(batch)
                    .map_err(hirn_core::HirnError::storage)?
                {
                    if record.provenance.created_by == *agent_id {
                        total_tokens += self.tokenizer.count_tokens(&record.name);
                        total_tokens += self.tokenizer.count_tokens(&record.description);
                    }
                }
            }
        }

        Ok(total_tokens)
    }

    /// Ensure the agent has a persisted-usage baseline in the state map,
    /// computing it from storage if absent. The storage scan runs OUTSIDE the
    /// state lock; insertion is `or_insert` so a concurrent computation
    /// cannot clobber reservations taken in the meantime.
    async fn ensure_baseline(&self, agent_id: &AgentId) -> HirnResult<()> {
        {
            let state = self.state.read().await;
            if state.agents.contains_key(agent_id) {
                return Ok(());
            }
        }
        let computed = self.compute_tokens(agent_id).await?;
        let mut state = self.state.write().await;
        state.agents.entry(agent_id.clone()).or_insert(AgentTokens {
            persisted: computed,
            reserved: 0,
        });
        Ok(())
    }
}

#[async_trait::async_trait]
impl AdmissionController for TokenBudgetGate {
    fn name(&self) -> &str {
        "token_budget_gate"
    }

    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
        let candidate_tokens = self.tokenizer.count_tokens(&candidate.content);
        self.ensure_baseline(&candidate.agent_id).await?;

        // Single critical section: read usage, decide, and reserve — so two
        // concurrent evaluations for the same agent serialize and cannot both
        // be admitted against the same headroom.
        let mut state = self.state.write().await;
        let usage = state
            .agents
            .get(&candidate.agent_id)
            .copied()
            .unwrap_or_default();
        let current = usage.persisted + usage.reserved;
        let projected = current + candidate_tokens;

        if projected > self.max_tokens {
            Ok(AdmissionDecision::Reject {
                reason: format!(
                    "token budget exceeded for agent '{}': {current} + {candidate_tokens} = \
                     {projected} > {max} max",
                    candidate.agent_id.as_str(),
                    max = self.max_tokens,
                ),
            })
        } else {
            let entry = state
                .agents
                .entry(candidate.agent_id.clone())
                .or_insert(usage);
            entry.reserved += candidate_tokens;
            state
                .reservations
                .insert(candidate.id, (candidate.agent_id.clone(), candidate_tokens));
            Ok(AdmissionDecision::Accept {
                importance_override: None,
                flags: Vec::new(),
            })
        }
    }

    async fn commit(&self, candidate: &MemoryCandidate) {
        let mut state = self.state.write().await;
        if let Some((agent, amount)) = state.reservations.remove(&candidate.id)
            && let Some(entry) = state.agents.get_mut(&agent)
        {
            entry.reserved = entry.reserved.saturating_sub(amount);
            entry.persisted += amount;
        }
    }

    async fn release(&self, candidate: &MemoryCandidate) {
        let mut state = self.state.write().await;
        if let Some((agent, amount)) = state.reservations.remove(&candidate.id)
            && let Some(entry) = state.agents.get_mut(&agent)
        {
            entry.reserved = entry.reserved.saturating_sub(amount);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::tokenizer::EstimatingTokenizer;
    use hirn_core::types::{AgentId, Namespace};
    use hirn_storage::{HirnDb, HirnDbConfig};

    fn candidate_with_agent(content: &str, agent: &str) -> MemoryCandidate {
        MemoryCandidate {
            id: MemoryId::new(),
            content: content.into(),
            entities: vec![],
            embedding: None,
            agent_id: AgentId::new(agent).unwrap(),
            provenance: hirn_core::provenance::Provenance::direct(AgentId::new(agent).unwrap()),
            namespace: Namespace::shared(),
            importance: 0.5,
            surprise: 0.5,
            metadata: Metadata::default(),
        }
    }

    async fn temp_storage() -> (Arc<dyn PhysicalStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance");
        let config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config.clone()).await.unwrap();
        (backend.store_arc(), dir)
    }

    async fn insert_content(storage: &Arc<dyn PhysicalStore>, content: &str, agent: &str) {
        let emb: Vec<f32> = vec![0.0; 32];
        let rec = hirn_core::episodic::EpisodicRecord::builder()
            .content(content)
            .embedding(emb)
            .agent_id(AgentId::new(agent).unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::episodic::to_batch(std::slice::from_ref(&rec), 32).unwrap();
        storage.append("episodic", batch).await.unwrap();
    }

    async fn insert_semantic(storage: &Arc<dyn PhysicalStore>, description: &str, agent: &str) {
        let emb: Vec<f32> = vec![0.0; 32];
        let rec = hirn_core::semantic::SemanticRecord::builder()
            .concept("concept")
            .description(description)
            .embedding(emb)
            .agent_id(AgentId::new(agent).unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::semantic::to_batch(std::slice::from_ref(&rec), 32).unwrap();
        storage
            .append(hirn_storage::datasets::semantic::DATASET_NAME, batch)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn cognitive_budget_counts_semantic_writes() {
        // R-57: semantic (and procedural) writes must count against the budget,
        // not only episodic. A single-dataset gate would miss the semantic
        // usage and admit; the cognitive gate rejects.
        let (storage, _dir) = temp_storage().await;

        // ~2500 tokens of semantic content for agent-a.
        let big_description = "a ".repeat(5000);
        insert_semantic(&storage, &big_description, "agent-a").await;

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);

        // Episodic-only gate ignores the semantic footprint → admits.
        let episodic_only =
            TokenBudgetGate::new(storage.clone(), tokenizer.clone(), "episodic", 3000);
        assert!(
            episodic_only
                .evaluate(&candidate_with_agent("more content", "agent-a"))
                .await
                .unwrap()
                .is_accept(),
            "episodic-only gate wrongly ignores semantic usage"
        );

        // Cognitive gate sums the semantic footprint → the same candidate now
        // pushes over the 3000-token budget.
        let cognitive = TokenBudgetGate::new_cognitive(storage, tokenizer, 3000);
        let more_content = "b ".repeat(1200); // ~600 tokens; 2500 + 600 > 3000
        assert!(
            cognitive
                .evaluate(&candidate_with_agent(&more_content, "agent-a"))
                .await
                .unwrap()
                .is_reject(),
            "cognitive gate must count semantic usage against the budget"
        );
    }

    #[tokio::test]
    async fn within_budget_accepted() {
        let (storage, _dir) = temp_storage().await;
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);
        let gate = TokenBudgetGate::new(storage, tokenizer, "episodic", 100_000);
        let result = gate
            .evaluate(&candidate_with_agent("hello world", "agent-a"))
            .await
            .unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn over_budget_rejected() {
        let (storage, _dir) = temp_storage().await;

        // Insert a large block of content for the agent.
        let big_content = "a ".repeat(5000); // ~2500 tokens via estimator
        insert_content(&storage, &big_content, "agent-a").await;

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);
        // Budget = 3000 tokens.
        let gate = TokenBudgetGate::new(storage, tokenizer, "episodic", 3000);

        // First candidate should push over budget.
        let more_content = "b ".repeat(1000); // ~500 tokens
        let result = gate
            .evaluate(&candidate_with_agent(&more_content, "agent-a"))
            .await
            .unwrap();
        // 2500 + 500 = 3000, which is not > 3000, so should accept.
        assert!(result.is_accept());

        // One more should push over.
        let result2 = gate
            .evaluate(&candidate_with_agent("enough already", "agent-a"))
            .await
            .unwrap();
        // Now the speculative cache has 3000 + a few more → rejected.
        assert!(result2.is_reject());
    }

    #[tokio::test]
    async fn invalidate_resets_cache() {
        let (storage, _dir) = temp_storage().await;
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);
        let gate = TokenBudgetGate::new(storage, tokenizer, "episodic", 100);

        let agent = AgentId::new("agent-a").unwrap();

        // Accept and cache.
        let result = gate
            .evaluate(&candidate_with_agent("hello", "agent-a"))
            .await
            .unwrap();
        assert!(result.is_accept());

        // Invalidate.
        gate.invalidate(&agent).await;

        // Next evaluate re-scans storage (which has 0 tokens since we didn't actually write).
        let result = gate
            .evaluate(&candidate_with_agent("hello", "agent-a"))
            .await
            .unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn downstream_reject_releases_reservation() {
        use crate::admission::AdmissionPipeline;

        /// Always-reject controller placed AFTER the gate.
        struct RejectAll;
        #[async_trait::async_trait]
        impl crate::admission::AdmissionController for RejectAll {
            fn name(&self) -> &str {
                "reject_all"
            }
            async fn evaluate(
                &self,
                _: &MemoryCandidate,
            ) -> HirnResult<crate::admission::AdmissionDecision> {
                Ok(crate::admission::AdmissionDecision::Reject {
                    reason: "downstream".into(),
                })
            }
        }

        let (storage, _dir) = temp_storage().await;
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);
        // Budget fits ~one candidate at a time.
        let gate = TokenBudgetGate::new(storage, tokenizer, "episodic", 60);
        let pipeline = AdmissionPipeline::new().with(gate).with(RejectAll);

        // Each attempt reserves ~50 tokens in the gate, then the downstream
        // controller rejects. Without release-on-short-circuit the second
        // attempt would be rejected by the gate on leaked reservations.
        for _ in 0..3 {
            let candidate = candidate_with_agent(&"t ".repeat(100), "agent-a");
            let result = pipeline.evaluate(&candidate).await.unwrap();
            assert!(result.decision.is_reject());
            assert_eq!(
                result.verdicts.last().unwrap().controller,
                "reject_all",
                "the gate must keep accepting — its reservation was released \
                 after each downstream reject"
            );
        }
    }

    #[tokio::test]
    async fn concurrent_evaluations_cannot_jointly_exceed_budget() {
        let (storage, _dir) = temp_storage().await;
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);
        // Each candidate is ~60% of the budget: admitting both would exceed it.
        let gate = Arc::new(TokenBudgetGate::new(storage, tokenizer, "episodic", 100));

        let content = "c ".repeat(120); // ~60 tokens via estimator
        let a = candidate_with_agent(&content, "agent-a");
        let b = candidate_with_agent(&content, "agent-a");

        let (ra, rb) = tokio::join!(gate.evaluate(&a), gate.evaluate(&b));
        let accepts = [ra.unwrap(), rb.unwrap()]
            .iter()
            .filter(|d| d.is_accept())
            .count();
        assert_eq!(
            accepts, 1,
            "read-check-reserve must be atomic: only one 60%-of-budget \
             candidate may be admitted"
        );
    }

    #[tokio::test]
    async fn commit_and_release_settle_reservations() {
        let (storage, _dir) = temp_storage().await;
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);
        let gate = TokenBudgetGate::new(storage, tokenizer, "episodic", 100);

        // Reserve ~60, release → headroom restored.
        let a = candidate_with_agent(&"a ".repeat(120), "agent-a");
        assert!(gate.evaluate(&a).await.unwrap().is_accept());
        gate.release(&a).await;
        let b = candidate_with_agent(&"b ".repeat(120), "agent-a");
        assert!(
            gate.evaluate(&b).await.unwrap().is_accept(),
            "released reservation must free headroom"
        );

        // Commit b → its usage is persistent; the next 60%-candidate rejects.
        gate.commit(&b).await;
        let c = candidate_with_agent(&"c ".repeat(120), "agent-a");
        assert!(
            gate.evaluate(&c).await.unwrap().is_reject(),
            "committed usage must keep counting against the budget"
        );
    }

    #[tokio::test]
    async fn two_agents_independent_budgets() {
        let (storage, _dir) = temp_storage().await;

        let big_content = "x ".repeat(4000); // ~2000 tokens
        insert_content(&storage, &big_content, "agent-a").await;

        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);
        let gate = TokenBudgetGate::new(storage, tokenizer, "episodic", 2500);

        // Agent A is near budget → adding 600 more tokens should reject.
        let result_a = gate
            .evaluate(&candidate_with_agent(&"y ".repeat(1200), "agent-a"))
            .await
            .unwrap();
        assert!(result_a.is_reject());

        // Agent B has zero usage → should accept.
        let result_b = gate
            .evaluate(&candidate_with_agent("small note", "agent-b"))
            .await
            .unwrap();
        assert!(result_b.is_accept());
    }
}
