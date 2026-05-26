//! Token Budget Gate — enforces per-agent memory token budgets.
//!
//! Tracks the total token count for each agent and rejects new memories
//! that would push the agent over its budget.

use std::collections::HashMap;
use std::sync::Arc;

use futures::TryStreamExt;
use hirn_core::HirnResult;
use hirn_core::tokenizer::Tokenizer;
use hirn_core::types::AgentId;
use hirn_storage::PhysicalStore;
use hirn_storage::store::ScanOptions;
use tokio::sync::RwLock;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};

/// Per-agent token budget enforcement.
pub struct TokenBudgetGate {
    storage: Arc<dyn PhysicalStore>,
    tokenizer: Arc<dyn Tokenizer>,
    dataset: String,
    /// Maximum tokens per agent. Default: 500_000.
    max_tokens: usize,
    /// Cached token counts per agent (invalidated on forget).
    cache: RwLock<HashMap<AgentId, usize>>,
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
            dataset: dataset.into(),
            max_tokens,
            cache: RwLock::new(HashMap::new()),
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

    /// Invalidate the cache for an agent (e.g., after a forget operation).
    pub async fn invalidate(&self, agent_id: &AgentId) {
        self.cache.write().await.remove(agent_id);
    }

    /// Invalidate all cached counts.
    pub async fn invalidate_all(&self) {
        self.cache.write().await.clear();
    }

    /// Compute the current token count for an agent by scanning storage.
    async fn compute_tokens(&self, agent_id: &AgentId) -> HirnResult<usize> {
        let exists = self
            .storage
            .exists(&self.dataset)
            .await
            .map_err(hirn_core::HirnError::storage)?;
        if !exists {
            return Ok(0);
        }

        // Use the top-level agent_id column to push the filter down to Lance,
        // reading only the content column for matching rows.
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
            .scan_stream(&self.dataset, options)
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

    /// Get or compute the current token count for an agent.
    async fn get_tokens(&self, agent_id: &AgentId) -> HirnResult<usize> {
        // Check cache first.
        {
            let cache = self.cache.read().await;
            if let Some(&count) = cache.get(agent_id) {
                return Ok(count);
            }
        }

        // Compute and cache.
        let count = self.compute_tokens(agent_id).await?;
        self.cache.write().await.insert(agent_id.clone(), count);
        Ok(count)
    }
}

#[async_trait::async_trait]
impl AdmissionController for TokenBudgetGate {
    fn name(&self) -> &str {
        "token_budget_gate"
    }

    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
        let current = self.get_tokens(&candidate.agent_id).await?;
        let candidate_tokens = self.tokenizer.count_tokens(&candidate.content);
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
            // Speculatively update the cache with the projected total.
            self.cache
                .write()
                .await
                .insert(candidate.agent_id.clone(), projected);
            Ok(AdmissionDecision::Accept {
                importance_override: None,
            })
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
