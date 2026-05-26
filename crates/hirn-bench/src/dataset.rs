//! Synthetic dataset generator with deterministic embeddings and ground-truth relevance.

use hirn::MemoryId;
use hirn::agent::AgentId;
use hirn::episodic::{EpisodicRecord, EventType};
use hirn::semantic::{KnowledgeType, SemanticRecord};

use crate::metrics::BenchmarkConfig;

/// A synthetic dataset for benchmarking.
pub struct SyntheticDataset {
    pub episodic_records: Vec<EpisodicRecord>,
    pub semantic_records: Vec<SemanticRecord>,
    pub queries: Vec<Query>,
}

/// A benchmark query with ground-truth relevant IDs.
pub struct Query {
    /// Topic label retained for debugging and reporting.
    #[allow(dead_code)]
    pub label: String,
    pub embedding: Vec<f32>,
    pub relevant_ids: Vec<MemoryId>,
}

/// Deterministic pseudo-embedding from text (matches existing bench pattern).
pub fn pseudo_embedding(text: &str, dims: usize) -> Vec<f32> {
    let mut embedding = vec![0.0f32; dims];
    let bytes = text.as_bytes();
    for (i, window) in bytes.windows(3).enumerate() {
        let hash = (window[0] as u32)
            .wrapping_mul(31)
            .wrapping_add(window[1] as u32)
            .wrapping_mul(31)
            .wrapping_add(window[2] as u32);
        let idx = (hash as usize).wrapping_add(i) % dims;
        embedding[idx] += 1.0;
    }
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut embedding {
            *v /= norm;
        }
    } else {
        embedding[0] = 1.0;
    }
    embedding
}

/// Topic clusters for generating records with known relationships.
const TOPICS: &[&str] = &[
    "deployment strategies for microservices",
    "caching patterns in distributed systems",
    "machine learning model training pipelines",
    "database indexing and query optimization",
    "container orchestration with kubernetes",
    "event driven architecture patterns",
    "authentication and authorization protocols",
    "observability and distributed tracing",
    "CI/CD pipeline automation practices",
    "data serialization and schema evolution",
];

fn agent() -> AgentId {
    AgentId::well_known("bench")
}

/// Generate a deterministic synthetic dataset.
///
/// Records cluster around `TOPICS`. Each query targets one topic, and
/// ground-truth relevant IDs are the records generated from that topic.
pub fn generate(config: &BenchmarkConfig) -> SyntheticDataset {
    let dims = config.embedding_dims;
    let topic_count = TOPICS.len();
    let records_per_topic = config.num_records / topic_count;
    let sem_per_topic = records_per_topic / 5;

    let mut episodic_records = Vec::new();
    let mut semantic_records = Vec::new();
    let mut topic_ids: Vec<Vec<MemoryId>> = vec![Vec::new(); topic_count];

    // Generate episodic records clustered by topic.
    for (t_idx, topic) in TOPICS.iter().enumerate() {
        for r in 0..records_per_topic {
            let content = format!(
                "Episode {r} about {topic}: variant-{} detail-{}",
                r % 7,
                r % 13
            );
            let emb = pseudo_embedding(&content, dims);
            let rec = EpisodicRecord::builder()
                .event_type(EventType::Observation)
                .content(&content)
                .summary(format!("ep-{t_idx}-{r}"))
                .importance(0.3 + (r as f32 % 7.0) * 0.1)
                .agent_id(agent())
                .embedding(emb)
                .build()
                .unwrap();
            topic_ids[t_idx].push(rec.id);
            episodic_records.push(rec);
        }

        // Semantic records for this topic.
        for s in 0..sem_per_topic {
            let desc = format!("Semantic knowledge: {topic} aspect-{s}");
            let emb = pseudo_embedding(&desc, dims);
            let rec = SemanticRecord::builder()
                .concept(format!("topic_{t_idx}_concept_{s}"))
                .knowledge_type(KnowledgeType::Propositional)
                .description(&desc)
                .confidence(0.8)
                .embedding(emb)
                .agent_id(agent())
                .build()
                .unwrap();
            topic_ids[t_idx].push(rec.id);
            semantic_records.push(rec);
        }
    }

    // Generate queries — select a subset of topics.
    let query_count = config.num_queries.min(topic_count);
    let mut queries = Vec::with_capacity(query_count);
    for t_idx in 0..query_count {
        let topic = TOPICS[t_idx];
        let emb = pseudo_embedding(topic, dims);
        queries.push(Query {
            label: topic.to_string(),
            embedding: emb,
            relevant_ids: topic_ids[t_idx].clone(),
        });
    }

    SyntheticDataset {
        episodic_records,
        semantic_records,
        queries,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::BenchmarkConfig;

    #[test]
    fn pseudo_embedding_deterministic() {
        let a = pseudo_embedding("hello world", 64);
        let b = pseudo_embedding("hello world", 64);
        assert_eq!(a, b);
    }

    #[test]
    fn pseudo_embedding_normalized() {
        let emb = pseudo_embedding("test input", 128);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn pseudo_embedding_different_inputs_differ() {
        let a = pseudo_embedding("alpha", 64);
        let b = pseudo_embedding("beta", 64);
        assert_ne!(a, b);
    }

    #[test]
    fn generate_dataset() {
        let config = BenchmarkConfig {
            num_records: 100,
            embedding_dims: 64,
            num_queries: 5,
            k: 10,
            ..Default::default()
        };
        let ds = generate(&config);

        // 100 records / 10 topics = 10 per topic
        assert_eq!(ds.episodic_records.len(), 100);
        // 10/5 = 2 semantic per topic × 10 topics = 20
        assert_eq!(ds.semantic_records.len(), 20);
        assert_eq!(ds.queries.len(), 5);

        // Each query has relevant IDs: 10 episodic + 2 semantic = 12
        for q in &ds.queries {
            assert_eq!(q.relevant_ids.len(), 12);
        }
    }

    #[test]
    fn generate_dataset_deterministic_ids() {
        let config = BenchmarkConfig {
            num_records: 50,
            embedding_dims: 32,
            num_queries: 3,
            k: 5,
            ..Default::default()
        };
        let ds1 = generate(&config);
        let ds2 = generate(&config);

        // Embeddings should match (deterministic), but IDs will differ (MemoryId::new() uses ULID).
        // We verify embedding determinism instead.
        for (a, b) in ds1.episodic_records.iter().zip(ds2.episodic_records.iter()) {
            assert_eq!(a.embedding, b.embedding);
            assert_eq!(a.content, b.content);
        }
    }
}
