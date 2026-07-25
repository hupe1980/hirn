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
                    flags: Vec::new(),
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
                flags: Vec::new(),
            });
        }

        // Find the most similar high-confidence semantic records. The search is
        // scoped to the candidate's namespace so a contradiction (and any
        // `Contradicts` edge) can never be drawn against a foreign tenant's
        // record.
        let options = VectorSearchOptions {
            query: embedding.clone(),
            column: "embedding".into(),
            limit: self.top_k,
            filter: Some(format!(
                "confidence >= {} AND (archived IS NULL OR archived = false) AND {}",
                self.confidence_threshold,
                super::namespace_eq_filter(&candidate.namespace)
            )),
            ..Default::default()
        };

        let batches = self
            .storage
            .vector_search(&self.dataset, options)
            .await
            .map_err(hirn_core::HirnError::storage)?;

        // Extract `(id, description)` as aligned pairs in a single pass so the
        // LLM's 1-based fact index always maps back to the correct record. A row
        // missing *either* a parseable id or a description is skipped entirely —
        // independent null-skipping passes could otherwise misalign the index and
        // draw a `Contradicts` edge to the wrong record.
        let candidates = extract_id_description_pairs(&batches);

        if candidates.is_empty() {
            return Ok(AdmissionDecision::Accept {
                importance_override: None,
                flags: Vec::new(),
            });
        }

        let existing_ids: Vec<MemoryId> = candidates.iter().map(|(id, _)| *id).collect();
        // Sanitize candidate content and existing facts before embedding them in
        // the LLM prompt to neutralize prompt-injection payloads carried in
        // stored content.
        let existing_facts: Vec<String> = candidates
            .iter()
            .map(|(_, desc)| hirn_core::sanitize::sanitize_for_llm(desc))
            .collect();

        // Ask LLM about contradictions.
        let sanitized_content = hirn_core::sanitize::sanitize_for_llm(&candidate.content);
        let messages = Self::build_prompt(&sanitized_content, &existing_facts);
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
                flags: Vec::new(),
            })
        }
    }
}

/// Extract aligned `(id, description)` pairs from result batches.
///
/// Both columns are read in a single pass and a row is emitted only when it
/// carries *both* a parseable `id` and a non-null `description`. This keeps the
/// returned vector's index — which becomes the LLM's 1-based fact number —
/// pointing at exactly the record it describes. Independent null-skipping passes
/// over the two columns could drift out of alignment (e.g. a row with a null
/// description but a valid id), so they are deliberately fused here.
fn extract_id_description_pairs(batches: &[arrow_array::RecordBatch]) -> Vec<(MemoryId, String)> {
    use arrow_array::Array;

    fn string_at(col: &dyn Array, i: usize) -> Option<String> {
        if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
            return (!arr.is_null(i)).then(|| arr.value(i).to_string());
        }
        if let Some(arr) = col.as_any().downcast_ref::<arrow_array::LargeStringArray>() {
            return (!arr.is_null(i)).then(|| arr.value(i).to_string());
        }
        None
    }

    let mut out = Vec::new();
    for batch in batches {
        let (Some(id_col), Some(desc_col)) = (
            batch.column_by_name("id"),
            batch.column_by_name("description"),
        ) else {
            continue;
        };
        for i in 0..batch.num_rows() {
            let (Some(id_str), Some(description)) = (string_at(id_col, i), string_at(desc_col, i))
            else {
                continue;
            };
            let Ok(id) = MemoryId::parse(&id_str) else {
                continue;
            };
            out.push((id, description));
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
        candidate_in(content, embedding, Namespace::shared())
    }

    fn candidate_in(content: &str, embedding: Vec<f32>, namespace: Namespace) -> MemoryCandidate {
        MemoryCandidate {
            id: MemoryId::new(),
            content: content.into(),
            entities: vec![],
            embedding: Some(embedding),
            agent_id: AgentId::new("test").unwrap(),
            provenance: hirn_core::provenance::Provenance::direct(AgentId::new("test").unwrap()),
            namespace,
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
        insert_semantic_ns(storage, description, emb, confidence, Namespace::shared()).await;
    }

    async fn insert_semantic_ns(
        storage: &Arc<dyn PhysicalStore>,
        description: &str,
        emb: Vec<f32>,
        confidence: f32,
        namespace: Namespace,
    ) -> MemoryId {
        let rec = hirn_core::semantic::SemanticRecord::builder()
            .concept("test-concept")
            .description(description)
            .embedding(emb)
            .confidence(confidence)
            .agent_id(AgentId::new("test").unwrap())
            .namespace(namespace)
            .build()
            .unwrap();
        let id = rec.id;
        let batch =
            hirn_storage::datasets::semantic::to_batch(std::slice::from_ref(&rec), 32).unwrap();
        storage.append("semantic", batch).await.unwrap();
        id
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

    /// R-10(a) regression: a candidate in namespace A must not be compared
    /// against a high-confidence record in a foreign namespace B — the scoped
    /// search returns no facts, so the LLM is never consulted and no
    /// `Contradicts` edge can be drawn cross-tenant.
    #[tokio::test]
    async fn cross_namespace_contradiction_not_flagged() {
        let (storage, _dir) = temp_storage().await;
        let emb = rand_vec(1);
        let foreign_ns = Namespace::private_for(&AgentId::new("agent-b").unwrap());
        insert_semantic_ns(&storage, "The sky is blue", emb.clone(), 0.9, foreign_ns).await;

        let llm = Arc::new(MockLlm::new("CONTRADICTION: 1"));
        let gate = ContradictionGate::with_defaults(
            storage,
            llm.clone() as Arc<dyn LlmProvider>,
            "semantic",
        );
        let own_ns = Namespace::private_for(&AgentId::new("agent-a").unwrap());
        let result = gate
            .evaluate(&candidate_in("The sky is green", emb, own_ns))
            .await
            .unwrap();
        assert!(
            result.is_accept(),
            "cross-namespace record must not trigger a contradiction, got {result:?}"
        );
        assert_eq!(
            llm.calls(),
            0,
            "LLM must not be consulted when no in-namespace facts exist"
        );
    }

    /// R-10(c) regression: `(id, description)` pairs stay aligned when an
    /// intermediate row is missing a description. A separate id-only pass would
    /// have mapped fact #1 to the first id even though that row has no
    /// description — the fused pass drops the row entirely instead.
    #[test]
    fn id_description_pairs_stay_aligned_on_null_description() {
        use arrow_array::{RecordBatch, StringArray};
        use std::sync::Arc as StdArc;

        let id0 = MemoryId::new();
        let id1 = MemoryId::new();
        let ids = StringArray::from(vec![Some(id0.to_string()), Some(id1.to_string())]);
        // First row has NO description; second row does.
        let descriptions = StringArray::from(vec![None, Some("the sky is blue".to_string())]);

        let schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Utf8, true),
            arrow_schema::Field::new("description", arrow_schema::DataType::Utf8, true),
        ]);
        let batch = RecordBatch::try_new(
            StdArc::new(schema),
            vec![StdArc::new(ids), StdArc::new(descriptions)],
        )
        .unwrap();

        let pairs = extract_id_description_pairs(&[batch]);
        // Only the fully-populated row survives, and it maps to id1 — not id0.
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, id1);
        assert_eq!(pairs[0].1, "the sky is blue");
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
