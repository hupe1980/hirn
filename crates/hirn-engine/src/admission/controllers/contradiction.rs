//! Contradiction Gate — flags memories that contradict existing high-confidence knowledge.
//!
//! Uses LLM to evaluate whether a candidate memory contradicts existing
//! semantic records. Contradicting candidates are deferred for review.

use std::sync::Arc;

use hirn_core::HirnResult;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider};
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::types::EdgeRelation;
use hirn_storage::PhysicalStore;
use hirn_storage::store::VectorSearchOptions;

use crate::admission::{AdmissionController, AdmissionDecision, MemoryCandidate};
use crate::persistent_graph::PersistentGraph;

/// Checks candidates against existing semantic records for contradictions.
pub struct ContradictionGate {
    storage: Arc<dyn PhysicalStore>,
    llm: Arc<dyn LlmProvider>,
    /// Dataset containing semantic records.
    dataset: String,
    /// Minimum confidence of existing records to compare against.
    confidence_threshold: f32,
    /// Number of existing records to compare against.
    top_k: usize,
    /// Optional graph for creating `contradicts` edges.
    graph: Option<PersistentGraph>,
}

impl ContradictionGate {
    pub fn new(
        storage: Arc<dyn PhysicalStore>,
        llm: Arc<dyn LlmProvider>,
        dataset: impl Into<String>,
        confidence_threshold: f32,
        top_k: usize,
    ) -> Self {
        Self {
            storage,
            llm,
            dataset: dataset.into(),
            confidence_threshold,
            top_k,
            graph: None,
        }
    }

    /// Create with defaults: confidence ≥ 0.7, top 5 records.
    pub fn with_defaults(
        storage: Arc<dyn PhysicalStore>,
        llm: Arc<dyn LlmProvider>,
        dataset: impl Into<String>,
    ) -> Self {
        Self::new(storage, llm, dataset, 0.7, 5)
    }

    /// Attach a persistent graph for creating `contradicts` edges.
    pub fn with_graph(mut self, graph: PersistentGraph) -> Self {
        self.graph = Some(graph);
        self
    }

    /// Build the prompt for contradiction detection.
    fn build_prompt(candidate_content: &str, existing_facts: &[String]) -> Vec<ChatMessage> {
        let facts_block = existing_facts
            .iter()
            .enumerate()
            .map(|(i, f)| format!("{}. {f}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");

        vec![
            ChatMessage {
                role: "system".into(),
                content: "You are a contradiction detector. Given a new statement and a list of \
                          existing facts, determine if the new statement contradicts any existing \
                          fact. Respond with ONLY 'CONTRADICTION: <fact_number>' if a \
                          contradiction exists, or 'NO_CONTRADICTION' if there is none. Be \
                          precise — a contradiction means the two statements cannot both be true."
                    .into(),
            },
            ChatMessage {
                role: "user".into(),
                content: format!(
                    "New statement: {candidate_content}\n\nExisting facts:\n{facts_block}"
                ),
            },
        ]
    }

    /// Parse the LLM response. Returns the 0-based index of the contradicting fact, if any.
    fn parse_response(response: &str) -> Option<usize> {
        let trimmed = response.trim().to_uppercase();
        if !trimmed.starts_with("CONTRADICTION") {
            return None;
        }
        // Try to extract the fact number after the colon (1-based → 0-based).
        trimmed
            .split(':')
            .nth(1)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|n| n.saturating_sub(1))
    }
}

#[async_trait::async_trait]
impl AdmissionController for ContradictionGate {
    fn name(&self) -> &str {
        "contradiction_gate"
    }

    async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
        let embedding = match &candidate.embedding {
            Some(emb) => emb,
            None => {
                return Ok(AdmissionDecision::Accept {
                    importance_override: None,
                });
            }
        };

        let exists = self
            .storage
            .exists(&self.dataset)
            .await
            .map_err(hirn_core::HirnError::storage)?;
        if !exists {
            return Ok(AdmissionDecision::Accept {
                importance_override: None,
            });
        }

        // Find the most similar high-confidence semantic records.
        let options = VectorSearchOptions {
            query: embedding.clone(),
            column: "embedding".into(),
            limit: self.top_k,
            filter: Some(format!(
                "confidence >= {} AND (archived IS NULL OR archived = false)",
                self.confidence_threshold
            )),
            ..Default::default()
        };

        let batches = self
            .storage
            .vector_search(&self.dataset, options)
            .await
            .map_err(hirn_core::HirnError::storage)?;

        let existing_facts = extract_descriptions(&batches);
        let existing_ids = extract_ids(&batches);

        if existing_facts.is_empty() {
            return Ok(AdmissionDecision::Accept {
                importance_override: None,
            });
        }

        // Ask LLM about contradictions.
        let messages = Self::build_prompt(&candidate.content, &existing_facts);
        let llm_options = LlmOptions {
            temperature: 0.0,
            max_tokens: 64,
            ..Default::default()
        };

        let response = self.llm.generate_text(&messages, &llm_options).await?;
        let contradiction_idx = Self::parse_response(&response);

        if contradiction_idx.is_some() {
            // Create `contradicts` edge if graph is available and we know the target.
            if let (Some(graph), Some(idx)) = (&self.graph, contradiction_idx) {
                if let Some(target_id) = existing_ids.get(idx) {
                    // Best-effort edge creation — don't fail admission on edge error.
                    let _ = graph
                        .add_edge(
                            candidate.id,
                            *target_id,
                            EdgeRelation::Contradicts,
                            1.0,
                            Metadata::default(),
                        )
                        .await;
                }
            }

            // Defer for manual review / consolidation.
            let now = hirn_core::timestamp::Timestamp::now();
            Ok(AdmissionDecision::Defer {
                until: now.timestamp_ms() + 3_600_000, // 1 hour in ms
            })
        } else {
            Ok(AdmissionDecision::Accept {
                importance_override: None,
            })
        }
    }
}

/// Extract description strings from result batches.
fn extract_descriptions(batches: &[arrow_array::RecordBatch]) -> Vec<String> {
    use arrow_array::Array;
    let mut out = Vec::new();
    for batch in batches {
        if let Some(col) = batch.column_by_name("description") {
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        out.push(arr.value(i).to_string());
                    }
                }
            }
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        out.push(arr.value(i).to_string());
                    }
                }
            }
        }
    }
    out
}

/// Extract memory IDs from result batches.
fn extract_ids(batches: &[arrow_array::RecordBatch]) -> Vec<MemoryId> {
    use arrow_array::Array;
    let mut out = Vec::new();
    for batch in batches {
        if let Some(col) = batch.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        if let Ok(id) = MemoryId::parse(arr.value(i)) {
                            out.push(id);
                        }
                    }
                }
            }
            if let Some(arr) = col.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        if let Ok(id) = MemoryId::parse(arr.value(i)) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::embed::{ChatMessage, LlmOptions};
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, Namespace};
    use hirn_storage::{HirnDb, HirnDbConfig};
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn candidate(content: &str, embedding: Vec<f32>) -> MemoryCandidate {
        MemoryCandidate {
            id: MemoryId::new(),
            content: content.into(),
            entities: vec![],
            embedding: Some(embedding),
            agent_id: AgentId::new("test").unwrap(),
            namespace: Namespace::shared(),
            importance: 0.5,
            surprise: 0.5,
            metadata: Metadata::default(),
        }
    }

    fn rand_vec(seed: u128) -> Vec<f32> {
        (0..32)
            .map(|i| (seed as f64 * 0.618_033 + i as f64 * 0.414_213).sin() as f32)
            .collect()
    }

    /// MockProvider that returns a configurable response.
    struct MockLlm {
        response: String,
        call_count: AtomicUsize,
    }

    impl MockLlm {
        fn new(response: &str) -> Self {
            Self {
                response: response.into(),
                call_count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl LlmProvider for MockLlm {
        async fn generate_text(
            &self,
            _messages: &[ChatMessage],
            _options: &LlmOptions,
        ) -> HirnResult<String> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(self.response.clone())
        }

        fn model_id(&self) -> &str {
            "mock-llm"
        }
    }

    async fn temp_storage() -> (Arc<dyn PhysicalStore>, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let lance_path = dir.path().join("lance");
        let config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend = HirnDb::open(config.clone()).await.unwrap();
        (backend.store_arc(), dir)
    }

    async fn insert_semantic(
        storage: &Arc<dyn PhysicalStore>,
        description: &str,
        emb: Vec<f32>,
        confidence: f32,
    ) {
        let rec = hirn_core::semantic::SemanticRecord::builder()
            .concept("test-concept")
            .description(description)
            .embedding(emb)
            .confidence(confidence)
            .agent_id(AgentId::new("test").unwrap())
            .build()
            .unwrap();
        let batch =
            hirn_storage::datasets::semantic::to_batch(std::slice::from_ref(&rec), 32).unwrap();
        storage.append("semantic", batch).await.unwrap();
    }

    #[tokio::test]
    async fn no_embedding_accepts() {
        let (storage, _dir) = temp_storage().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlm::new("NO_CONTRADICTION"));
        let gate = ContradictionGate::with_defaults(storage, llm, "semantic");
        let mut c = candidate("anything", rand_vec(1));
        c.embedding = None;
        let result = gate.evaluate(&c).await.unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn empty_database_accepts() {
        let (storage, _dir) = temp_storage().await;
        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlm::new("NO_CONTRADICTION"));
        let gate = ContradictionGate::with_defaults(storage, llm, "semantic");
        let result = gate
            .evaluate(&candidate("test", rand_vec(1)))
            .await
            .unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn contradiction_detected_defers() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(1);
        insert_semantic(&storage, "The sky is blue", emb.clone(), 0.9).await;

        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlm::new("CONTRADICTION: 1"));
        let gate = ContradictionGate::with_defaults(storage, llm, "semantic");
        let result = gate
            .evaluate(&candidate("The sky is green", emb))
            .await
            .unwrap();
        assert!(matches!(result, AdmissionDecision::Defer { .. }));
    }

    #[tokio::test]
    async fn no_contradiction_accepts() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(1);
        insert_semantic(&storage, "The sky is blue", emb.clone(), 0.9).await;

        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlm::new("NO_CONTRADICTION"));
        let gate = ContradictionGate::with_defaults(storage, llm, "semantic");
        let result = gate
            .evaluate(&candidate("Water is wet", emb))
            .await
            .unwrap();
        assert!(result.is_accept());
    }

    #[tokio::test]
    async fn low_confidence_records_skipped() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(1);
        // Insert with low confidence (below threshold).
        insert_semantic(&storage, "The sky is blue", emb.clone(), 0.3).await;

        let llm = Arc::new(MockLlm::new("CONTRADICTION: 1"));
        let gate = ContradictionGate::with_defaults(
            storage,
            llm.clone() as Arc<dyn LlmProvider>,
            "semantic",
        );
        let result = gate
            .evaluate(&candidate("The sky is green", emb))
            .await
            .unwrap();
        // LLM should not be called since no records meet the confidence threshold.
        assert!(result.is_accept());
        assert_eq!(llm.calls(), 0);
    }

    #[tokio::test]
    async fn causal_contradiction_flagged() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(1);
        insert_semantic(&storage, "X causes Y", emb.clone(), 0.9).await;

        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlm::new("CONTRADICTION: 1"));
        let gate = ContradictionGate::with_defaults(storage, llm, "semantic");
        let result = gate
            .evaluate(&candidate("X does not cause Y", emb))
            .await
            .unwrap();
        assert!(matches!(result, AdmissionDecision::Defer { .. }));
    }

    #[tokio::test]
    async fn contradiction_creates_edge_in_graph() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(1);
        insert_semantic(&storage, "The sky is blue", emb.clone(), 0.9).await;

        let graph = PersistentGraph::open(Arc::clone(&storage)).await.unwrap();
        let llm: Arc<dyn LlmProvider> = Arc::new(MockLlm::new("CONTRADICTION: 1"));
        let gate = ContradictionGate::with_defaults(Arc::clone(&storage), llm, "semantic")
            .with_graph(graph);

        let c = candidate("The sky is green", emb);
        let candidate_id = c.id;
        let result = gate.evaluate(&c).await.unwrap();
        assert!(matches!(result, AdmissionDecision::Defer { .. }));

        // Verify a `Contradicts` edge was created.
        let graph = PersistentGraph::open(storage).await.unwrap();
        let edges = graph.get_edges_from(candidate_id).await.unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation, EdgeRelation::Contradicts);
    }

    #[test]
    fn prompt_format() {
        let messages = ContradictionGate::build_prompt(
            "The sky is green",
            &["The sky is blue".into(), "Water is wet".into()],
        );
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert!(messages[1].content.contains("The sky is green"));
        assert!(messages[1].content.contains("1. The sky is blue"));
        assert!(messages[1].content.contains("2. Water is wet"));
    }

    #[test]
    fn parse_contradiction_response() {
        assert_eq!(
            ContradictionGate::parse_response("CONTRADICTION: 1"),
            Some(0)
        );
        assert_eq!(
            ContradictionGate::parse_response("  contradiction: 2  "),
            Some(1)
        );
        assert_eq!(ContradictionGate::parse_response("NO_CONTRADICTION"), None);
        assert_eq!(
            ContradictionGate::parse_response("no contradiction found"),
            None
        );
    }
}
