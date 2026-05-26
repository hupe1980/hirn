//! Causal reasoning, trust scoring, and contradiction detection.
//!
//! This module provides:
//! - Causal chain extraction via directed graph traversal
//! - Causal influence scoring (ε weight in composite formula)
//! - Counterfactual constraint detection
//! - Trustworthiness scoring engine
//! - Automatic contradiction detection on insertion

use std::collections::HashSet;

use hirn_core::id::MemoryId;
use hirn_core::provenance::Provenance;
use hirn_core::record::MemoryRecord;
use hirn_core::types::{EdgeRelation, Namespace, Origin};

use crate::graph_store::GraphStore;

use hirn_core::error::HirnResult;

// ── Causal Chain Types ─────────────────────────────────────────────────

/// A single link in a causal chain: source → (edge) → target.
#[derive(Debug, Clone)]
pub struct CausalLink {
    pub source: MemoryId,
    pub target: MemoryId,
    pub weight: f32,
    pub edge_id: MemoryId,
    /// Causal strength in `[0, 1]`.  `None` for legacy edges.
    pub strength: Option<f32>,
    /// Confidence in `[0, 1]`.  `None` for legacy edges.
    pub confidence: Option<f32>,
    /// Evidence count supporting this causal link.
    pub evidence_count: Option<u32>,
    /// Free-text provenance tag (e.g. "RCT", "observational").
    pub provenance: Option<String>,
    /// Mechanism description (e.g. "dopamine release").
    pub mechanism: Option<String>,
}

/// A complete causal chain: an ordered sequence of links forming a path.
#[derive(Debug, Clone)]
pub struct CausalChain {
    pub links: Vec<CausalLink>,
}

impl CausalChain {
    /// The starting node of this chain.
    pub fn start(&self) -> Option<MemoryId> {
        self.links.first().map(|l| l.source)
    }

    /// The ending node of this chain.
    pub fn end(&self) -> Option<MemoryId> {
        self.links.last().map(|l| l.target)
    }

    /// All node IDs in this chain (in order).
    pub fn node_ids(&self) -> Vec<MemoryId> {
        if self.links.is_empty() {
            return vec![];
        }
        let mut ids = vec![self.links[0].source];
        for link in &self.links {
            ids.push(link.target);
        }
        ids
    }

    /// Number of hops in this chain.
    pub fn depth(&self) -> usize {
        self.links.len()
    }

    /// Average edge weight across the chain.
    pub fn avg_weight(&self) -> f32 {
        if self.links.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.links.iter().map(|l| l.weight).sum();
        sum / self.links.len() as f32
    }
}

/// Result of causal chain extraction.
#[derive(Debug, Clone)]
pub struct CausalChainResult {
    /// All causal chains found from the starting node.
    pub chains: Vec<CausalChain>,
    /// Whether any cycles were detected (and broken).
    pub cycles_detected: bool,
}

// ── Causal Relevance Scoring ────────────────────────────────────────────

/// Compute the causal-relevance score ε for a memory that participates in one
/// or more causal chains.  Returns a value in `[0, 1]`.
///
/// The score is the **maximum** across all chains that touch the memory,
/// computed as:
///
///   `chain_score = avg(link_score)` over every link in the chain
///   `link_score  = strength * confidence` when both are present,
///                  or `weight` as a fallback for legacy edges.
///
/// Using the max (rather than mean) ensures that a single strong causal
/// connection is not diluted by unrelated weak chains.
pub fn causal_relevance(result: &CausalChainResult) -> f32 {
    if result.chains.is_empty() {
        return 0.0;
    }
    let mut max_score: f32 = 0.0;
    for chain in &result.chains {
        if chain.links.is_empty() {
            continue;
        }
        let sum: f32 = chain.links.iter().map(|l| link_score(l)).sum();
        let avg = sum / chain.links.len() as f32;
        max_score = max_score.max(avg);
    }
    max_score.clamp(0.0, 1.0)
}

/// Per-link score: `strength × confidence × ln(1 + evidence_count)` when
/// rich fields are available, falls back to `weight` for legacy edges.
fn link_score(link: &CausalLink) -> f32 {
    match (link.strength, link.confidence) {
        (Some(s), Some(c)) => {
            let ev = link.evidence_count.unwrap_or(1).max(1) as f32;
            s * c * (1.0 + ev).ln()
        }
        (Some(s), None) => s,
        (None, Some(c)) => c,
        (None, None) => link.weight,
    }
}

// ── Causal Chain Extraction ─────────────────────────────────────────────

/// Extract causal chains forward from a starting node via any [`GraphStore`].
pub async fn causal_chain_forward(
    store: &dyn GraphStore,
    start: MemoryId,
    max_depth: usize,
    confidence_threshold: f32,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<CausalChainResult> {
    extract_causal_chains(
        store,
        start,
        max_depth,
        EdgeRelation::Causes,
        confidence_threshold,
        allowed_namespaces,
    )
    .await
}

/// Extract causal chains backward from a starting node via any [`GraphStore`].
pub async fn causal_chain_backward(
    store: &dyn GraphStore,
    start: MemoryId,
    max_depth: usize,
    confidence_threshold: f32,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<CausalChainResult> {
    extract_causal_chains(
        store,
        start,
        max_depth,
        EdgeRelation::CausedBy,
        confidence_threshold,
        allowed_namespaces,
    )
    .await
}

/// Async core causal chain extraction: iterative DFS with cycle detection.
async fn extract_causal_chains(
    store: &dyn GraphStore,
    start: MemoryId,
    max_depth: usize,
    relation: EdgeRelation,
    confidence_threshold: f32,
    allowed_namespaces: Option<&[Namespace]>,
) -> HirnResult<CausalChainResult> {
    if max_depth == 0 || !store.has_node(start).await? {
        return Ok(CausalChainResult {
            chains: vec![],
            cycles_detected: false,
        });
    }

    if let Some(allowed) = allowed_namespaces {
        let Some(namespace) = store.node_namespace(start).await? else {
            return Ok(CausalChainResult {
                chains: vec![],
                cycles_detected: false,
            });
        };
        if !allowed.contains(&namespace) {
            return Ok(CausalChainResult {
                chains: vec![],
                cycles_detected: false,
            });
        }
    }

    let mut chains = Vec::new();
    let mut cycles_detected = false;

    let mut stack: Vec<(MemoryId, Vec<CausalLink>, HashSet<MemoryId>)> = Vec::new();
    let mut initial_visited = HashSet::new();
    initial_visited.insert(start);
    stack.push((start, Vec::new(), initial_visited));

    while let Some((current, path, visited)) = stack.pop() {
        if path.len() >= max_depth {
            if !path.is_empty() {
                chains.push(CausalChain { links: path });
            }
            continue;
        }

        let edges = store.get_edges_of_type(current, relation).await?;
        let mut outgoing = Vec::new();
        for edge in &edges {
            if edge.source != current {
                continue;
            }

            let confidence = edge.confidence().unwrap_or(0.5);
            if confidence < confidence_threshold {
                continue;
            }

            if let Some(allowed) = allowed_namespaces {
                let Some(namespace) = store.node_namespace(edge.target).await? else {
                    continue;
                };
                if !allowed.contains(&namespace) {
                    continue;
                }
            }

            outgoing.push(edge);
        }

        if outgoing.is_empty() {
            if !path.is_empty() {
                chains.push(CausalChain { links: path });
            }
            continue;
        }

        let mut any_extended = false;
        for edge in &outgoing {
            let target = edge.target;
            if visited.contains(&target) {
                cycles_detected = true;
                if !path.is_empty() {
                    chains.push(CausalChain {
                        links: path.clone(),
                    });
                }
                continue;
            }

            any_extended = true;
            let link = CausalLink {
                source: current,
                target,
                weight: edge.weight,
                edge_id: edge.id,
                strength: edge.strength(),
                confidence: edge.confidence(),
                evidence_count: edge.evidence_count(),
                provenance: edge.provenance().map(str::to_owned),
                mechanism: edge.mechanism().map(str::to_owned),
            };
            let mut new_path = path.clone();
            new_path.push(link);
            let mut new_visited = visited.clone();
            new_visited.insert(target);
            stack.push((target, new_path, new_visited));
        }

        if !any_extended && !path.is_empty() {
            // Already recorded in cycle detection above.
        }
    }

    chains.sort_by_key(|c| std::cmp::Reverse(c.depth()));
    chains.dedup_by(|a, b| a.node_ids() == b.node_ids());

    Ok(CausalChainResult {
        chains,
        cycles_detected,
    })
}

// ── Counterfactual Detection ───────────────────────────────────────────

/// A counterfactual constraint: if memory A is true, memory B is under tension.
#[derive(Debug, Clone)]
pub struct Counterfactual {
    pub memory_a: MemoryId,
    pub memory_b: MemoryId,
    pub constraint: CounterfactualConstraint,
    pub explanation: String,
}

/// The type of counterfactual constraint detected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterfactualConstraint {
    /// Direct contradiction via `contradicts` edge.
    DirectContradiction,
    /// Temporal impossibility: events are temporally inconsistent.
    TemporalImpossibility,
    /// F-44: Temporal supersession: a newer record updates/replaces an older one.
    /// The newer record (memory_a) supersedes the older (memory_b).
    TemporalSupersession,
}

// ── Trust Scoring Engine ───────────────────────────────────────────────

/// Compute the trust score for a memory record.
///
/// Trust factors:
/// 1. Origin type: direct observation = 1.0, user = 0.9, LLM = 0.7, etc.
/// 2. Evidence diversity: more diverse sources = higher trust
/// 3. Reconsolidation penalty: each mutation slightly reduces trust
///    (unless supported by new evidence)
///
/// Returns a score in [0.0, 1.0].
pub fn compute_trust_score(provenance: &Provenance, contradiction_count: usize) -> f32 {
    // Base trust from origin type.
    let origin_trust = match *provenance.origin() {
        Origin::DirectObservation => 1.0,
        Origin::UserProvided => 0.9,
        Origin::LlmExtraction => 0.7,
        Origin::Consolidation => {
            // Consolidation trust depends on evidence diversity.
            let evidence_count = provenance.confidence_basis.len();
            if evidence_count >= 5 {
                0.9
            } else if evidence_count >= 3 {
                0.8
            } else if evidence_count >= 1 {
                0.6
            } else {
                0.4
            }
        }
        Origin::CrossAgent => 0.6,
        Origin::DreamReplay => 0.3, // low trust until validated
    };

    // Evidence diversity bonus: unique source IDs.
    let unique_sources: HashSet<MemoryId> = provenance
        .confidence_basis
        .iter()
        .map(|e| e.source_id)
        .collect();
    let diversity_bonus = if unique_sources.len() >= 5 {
        0.1
    } else if unique_sources.len() >= 3 {
        0.05
    } else {
        0.0
    };

    // Mutation penalty: each mutation without new evidence slightly reduces trust.
    let mutations_without_evidence = count_mutations_without_evidence(provenance);
    let mutation_penalty = (mutations_without_evidence as f32 * 0.05).min(0.3);

    // Contradiction penalty.
    let contradiction_penalty = (contradiction_count as f32 * 0.1).min(0.3);

    let score = origin_trust + diversity_bonus - mutation_penalty - contradiction_penalty;
    score.clamp(0.0, 1.0)
}

/// Count mutations that are not supported by new evidence.
fn count_mutations_without_evidence(provenance: &Provenance) -> usize {
    // Each evidence ref added AFTER a mutation counts as "supported".
    // Simple heuristic: mutations beyond the evidence count are unsupported.
    let evidence_count = provenance.confidence_basis.len();
    let mutation_count = provenance.mutation_log.len();
    mutation_count.saturating_sub(evidence_count)
}

// ── Contradiction Detection (for insertion) ────────────────────────────

/// Result of contradiction detection during insertion.
#[derive(Debug, Clone)]
pub struct ContradictionDetection {
    /// IDs of records that contradict the new record.
    pub contradicting_ids: Vec<MemoryId>,
    /// Whether any contradictions were found.
    pub has_contradictions: bool,
}

#[derive(Clone, Copy)]
pub struct InsertionCandidateRecord<'a> {
    pub id: MemoryId,
    pub content_lower: &'a str,
    pub has_negation: bool,
    pub entities: &'a [String],
    pub similarity: f32,
}

/// Check if a new record (represented by its content and embedding)
/// contradicts any existing records.
///
/// Detection signals:
/// 1. High embedding similarity (same topic) with conflicting content
/// 2. Entity-value conflicts (same entity, different claims)
/// 3. Negation patterns
pub fn detect_contradictions_on_insert(
    content: &str,
    entities: &[String],
    similar_records: &[InsertionCandidateRecord<'_>],
    similarity_threshold: f32,
) -> ContradictionDetection {
    let mut contradicting_ids = Vec::new();
    let content_lower = content.to_lowercase();
    let new_has_negation = contains_negation(&content_lower);

    for candidate in similar_records {
        if candidate.similarity < similarity_threshold {
            continue;
        }

        let existing_content = candidate.content_lower;
        let existing_has_negation = candidate.has_negation;

        // Signal 1: Same topic (high cosine sim) + one has negation, the other doesn't.
        let negation_conflict = new_has_negation != existing_has_negation
            && content_similarity_simple(&content_lower, existing_content) > 0.3;

        // Signal 2: Entity-value conflicts — same entities but different values.
        let entity_conflict = if !entities.is_empty() {
            let shared_entities: Vec<&String> = entities
                .iter()
                .filter(|e| candidate.entities.iter().any(|ee| ee == *e))
                .collect();
            // If they share entities but have different content, likely a conflict.
            !shared_entities.is_empty()
                && content_similarity_simple(&content_lower, existing_content) < 0.8
                && (new_has_negation
                    || existing_has_negation
                    || value_conflict(&content_lower, existing_content))
        } else {
            false
        };

        if negation_conflict || entity_conflict {
            contradicting_ids.push(candidate.id);
        }
    }

    ContradictionDetection {
        has_contradictions: !contradicting_ids.is_empty(),
        contradicting_ids,
    }
}

// ── TRACE / Provenance Lineage ─────────────────────────────────────────

/// Complete trace result for a memory record.
#[derive(Debug, Clone)]
pub struct TraceReport {
    /// The traced record.
    pub record: MemoryRecord,
    /// Full provenance chain.
    pub provenance: Provenance,
    /// Source episodes (for semantic/consolidated records).
    pub source_episodes: Vec<MemoryId>,
    /// Records derived FROM this record (via DerivedFrom edges).
    pub derived_records: Vec<MemoryId>,
    /// Mutation history summary.
    pub mutation_count: usize,
    /// Trust score.
    pub trust_score: f32,
    /// Textual lineage tree.
    pub lineage_tree: String,
}

/// Build a trace report for a memory record via any [`GraphStore`].
pub async fn build_trace_report(
    store: &dyn GraphStore,
    record: MemoryRecord,
    provenance: Provenance,
    source_episodes: Vec<MemoryId>,
) -> HirnResult<TraceReport> {
    let record_id = record.id();

    let derived_edges = store
        .get_edges_of_type(record_id, EdgeRelation::DerivedFrom)
        .await?;
    let derived_records: Vec<MemoryId> = derived_edges
        .iter()
        .filter(|e| e.target == record_id)
        .map(|e| e.source)
        .collect();

    let contra_edges = store
        .get_edges_of_type(record_id, EdgeRelation::Contradicts)
        .await?;
    let contradiction_count = contra_edges.len();

    let trust_score = compute_trust_score(&provenance, contradiction_count);
    let mutation_count = provenance.mutation_log.len();

    let lineage_tree =
        format_lineage_tree(record_id, &provenance, &source_episodes, &derived_records);

    Ok(TraceReport {
        record,
        provenance,
        source_episodes,
        derived_records,
        mutation_count,
        trust_score,
        lineage_tree,
    })
}

/// Format a textual lineage tree for display.
fn format_lineage_tree(
    record_id: MemoryId,
    provenance: &Provenance,
    source_episodes: &[MemoryId],
    derived_records: &[MemoryId],
) -> String {
    let mut out = String::new();
    out.push_str(&format!("Lineage for {record_id}:\n"));
    out.push_str(&format!("  Origin: {:?}\n", provenance.origin()));
    out.push_str(&format!("  Created by: {}\n", provenance.created_by));

    if !source_episodes.is_empty() {
        out.push_str("  Source episodes:\n");
        for ep in source_episodes {
            out.push_str(&format!("    <- {ep}\n"));
        }
    }

    if let Some(ref model) = provenance.extraction_model {
        out.push_str(&format!("  Extraction model: {model}\n"));
    }

    if !provenance.mutation_log.is_empty() {
        out.push_str(&format!(
            "  Mutations ({}):\n",
            provenance.mutation_log.len()
        ));
        for m in &provenance.mutation_log {
            out.push_str(&format!(
                "    [{:?}] {}: {} -> {} ({})\n",
                m.trigger, m.field, m.old_value, m.new_value, m.reason
            ));
        }
    }

    if !derived_records.is_empty() {
        out.push_str("  Derived records:\n");
        for d in derived_records {
            out.push_str(&format!("    -> {d}\n"));
        }
    }

    out
}

// ── Helpers ────────────────────────────────────────────────────────────

/// Extract the primary text content from a memory record (any layer).
pub fn record_content_str(record: &MemoryRecord) -> &str {
    match record {
        MemoryRecord::Episodic(e) => &e.content,
        MemoryRecord::Semantic(s) => &s.description,
        MemoryRecord::Working(w) => &w.content,
        MemoryRecord::Procedural(p) => &p.description,
    }
}

/// Simple negation detection: checks for common negation patterns.
///
/// **Limitation (F-48):** This is surface-level pattern matching, not semantic
/// entailment. For example, "the project succeeded" vs "the project failed" won't
/// be caught unless negation markers are present. Full semantic contradiction
/// detection would require an LLM or NLI (Natural Language Inference) model.
/// The current approach works well for explicit negations and numerical conflicts
/// but misses implicit contradictions via paraphrase.
pub(crate) fn contains_negation(text: &str) -> bool {
    let negation_patterns = [
        "not ",
        "n't ",
        "never ",
        "no ",
        "doesn't ",
        "didn't ",
        "isn't ",
        "wasn't ",
        "aren't ",
        "won't ",
        "cannot ",
        "can't ",
        "shouldn't ",
        "wouldn't ",
        "hasn't ",
        "haven't ",
        "weren't ",
        "couldn't ",
        "needn't ",
        "shan't ",
        "nor ",
        "neither ",
        "nowhere ",
        "nothing ",
        "nobody ",
        "hardly ",
        "barely ",
        "scarcely ",
        "seldom ",
        "rarely ",
        "however ",
        "actually ",
        "instead ",
        "contrary ",
        "incorrect ",
        "false ",
        "wrong ",
        "failed ",
        "impossible ",
        "unlike ",
        "rather than ",
        "on the contrary",
        "slower ", // contextual negation for performance claims
    ];
    negation_patterns.iter().any(|pat| text.contains(pat))
}

/// Simple content similarity based on word overlap (Jaccard on word sets).
fn content_similarity_simple(a: &str, b: &str) -> f32 {
    let words_a: HashSet<&str> = a.split_whitespace().collect();
    let words_b: HashSet<&str> = b.split_whitespace().collect();
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

/// Detect value conflicts between two content strings.
///
/// Looks for numeric values in the context of the same entity/topic.
fn value_conflict(a: &str, b: &str) -> bool {
    // Extract numbers from both strings.
    let nums_a = extract_numbers(a);
    let nums_b = extract_numbers(b);

    // If both have numbers and they differ, likely a value conflict.
    if !nums_a.is_empty() && !nums_b.is_empty() {
        // Check if any numbers differ significantly.
        for na in &nums_a {
            for nb in &nums_b {
                if (na - nb).abs() > f64::EPSILON {
                    return true;
                }
            }
        }
    }

    false
}

/// Extract numeric values from text.
fn extract_numbers(text: &str) -> Vec<f64> {
    let mut numbers = Vec::new();
    for word in text.split_whitespace() {
        // Strip common suffixes like "GB", "MB", etc.
        let cleaned = word
            .trim_end_matches(|c: char| c.is_alphabetic())
            .trim_end_matches('%');
        if let Ok(n) = cleaned.parse::<f64>() {
            numbers.push(n);
        }
    }
    numbers
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::provenance::{EvidenceRef, Provenance};
    use hirn_core::timestamp::Timestamp;
    use hirn_core::types::MutationTrigger;

    // ── Trust Scoring Tests ────────────────────────────────────────────

    #[test]
    fn direct_observation_high_trust() {
        let p = Provenance::direct(hirn_core::types::AgentId::new("test").unwrap());
        let score = compute_trust_score(&p, 0);
        assert!(score >= 0.95, "score={score}");
    }

    #[test]
    fn llm_extraction_lower_trust() {
        let p = Provenance::with_origin(
            Origin::LlmExtraction,
            hirn_core::types::AgentId::new("test").unwrap(),
        );
        let score = compute_trust_score(&p, 0);
        assert!((score - 0.7).abs() < 0.05, "score={score}");
    }

    #[test]
    fn consolidation_with_diverse_sources_high_trust() {
        let agent = hirn_core::types::AgentId::new("test").unwrap();
        let mut p = Provenance::with_origin(Origin::Consolidation, agent);
        for i in 0..5 {
            p.confidence_basis.push(EvidenceRef {
                source_id: MemoryId::new(),
                description: format!("source {i}"),
            });
        }
        let score = compute_trust_score(&p, 0);
        assert!(score > 0.8, "score={score}");
    }

    #[test]
    fn consolidation_with_single_source_low_trust() {
        let agent = hirn_core::types::AgentId::new("test").unwrap();
        let mut p = Provenance::with_origin(Origin::Consolidation, agent);
        p.confidence_basis.push(EvidenceRef {
            source_id: MemoryId::new(),
            description: "only source".to_string(),
        });
        let score = compute_trust_score(&p, 0);
        assert!(score < 0.7, "score={score}");
    }

    #[test]
    fn mutations_without_evidence_reduce_trust() {
        let agent = hirn_core::types::AgentId::new("test").unwrap();
        let mut p = Provenance::direct(agent);
        // Add 3 mutations without evidence.
        for i in 0..3 {
            p.record_mutation(hirn_core::provenance::Mutation {
                timestamp: Timestamp::now(),
                trigger: MutationTrigger::Reconsolidation,
                field: "description".to_string(),
                old_value: format!("old {i}"),
                new_value: format!("new {i}"),
                reason: "test".to_string(),
            });
        }
        let score = compute_trust_score(&p, 0);
        // 1.0 - 3*0.05 = 0.85
        assert!(score < 1.0, "score={score}");
        assert!(score > 0.7, "score={score}");
    }

    #[test]
    fn mutations_with_evidence_maintain_trust() {
        let agent = hirn_core::types::AgentId::new("test").unwrap();
        let mut p = Provenance::direct(agent);
        // Add evidence for each mutation.
        for i in 0..3 {
            p.confidence_basis.push(EvidenceRef {
                source_id: MemoryId::new(),
                description: format!("evidence {i}"),
            });
            p.record_mutation(hirn_core::provenance::Mutation {
                timestamp: Timestamp::now(),
                trigger: MutationTrigger::Reconsolidation,
                field: "description".to_string(),
                old_value: format!("old {i}"),
                new_value: format!("new {i}"),
                reason: "supported update".to_string(),
            });
        }
        let score = compute_trust_score(&p, 0);
        // No unsupported mutations → trust maintained.
        assert!(score >= 0.95, "score={score}");
    }

    // ── Contradiction Detection Tests ──────────────────────────────────

    #[test]
    fn negation_detection() {
        assert!(contains_negation("hnsw is not faster"));
        assert!(contains_negation("it doesn't work"));
        assert!(contains_negation("system never recovered"));
        assert!(!contains_negation("system is fast"));
    }

    #[test]
    fn value_conflict_detection() {
        assert!(value_conflict(
            "system uses 10gb ram",
            "system uses 5gb ram"
        ));
        assert!(!value_conflict(
            "system uses 10gb ram",
            "system uses 10gb ram"
        ));
    }

    #[test]
    fn content_similarity_identical() {
        let sim = content_similarity_simple("hello world test", "hello world test");
        assert!((sim - 1.0).abs() < f64::EPSILON as f32);
    }

    #[test]
    fn content_similarity_different() {
        let sim = content_similarity_simple("hello world", "foo bar baz");
        assert!(sim < 0.1);
    }

    // ── Causal Relevance Scoring Tests ─────────────────────────────────

    fn make_link(weight: f32, strength: Option<f32>, confidence: Option<f32>) -> CausalLink {
        CausalLink {
            source: MemoryId::new(),
            target: MemoryId::new(),
            weight,
            edge_id: MemoryId::new(),
            strength,
            confidence,
            evidence_count: None,
            provenance: None,
            mechanism: None,
        }
    }

    #[test]
    fn causal_relevance_empty_chains() {
        let result = CausalChainResult {
            chains: vec![],
            cycles_detected: false,
        };
        assert!((causal_relevance(&result)).abs() < f32::EPSILON);
    }

    #[test]
    fn causal_relevance_uses_strength_and_confidence() {
        let link = make_link(0.5, Some(0.9), Some(0.8));
        let result = CausalChainResult {
            chains: vec![CausalChain { links: vec![link] }],
            cycles_detected: false,
        };
        let score = causal_relevance(&result);
        // 0.9 * 0.8 * ln(2) ≈ 0.72 * 0.693 ≈ 0.499
        let expected = 0.9 * 0.8 * (2.0_f32).ln();
        assert!(
            (score - expected).abs() < 0.01,
            "score={score}, expected={expected}"
        );
    }

    #[test]
    fn causal_relevance_falls_back_to_weight() {
        let link = make_link(0.6, None, None);
        let result = CausalChainResult {
            chains: vec![CausalChain { links: vec![link] }],
            cycles_detected: false,
        };
        let score = causal_relevance(&result);
        assert!((score - 0.6).abs() < 0.01, "score={score}");
    }

    #[test]
    fn causal_relevance_takes_max_across_chains() {
        let weak = make_link(0.2, None, None);
        let strong = make_link(0.0, Some(0.95), Some(0.95));
        let result = CausalChainResult {
            chains: vec![
                CausalChain { links: vec![weak] },
                CausalChain {
                    links: vec![strong],
                },
            ],
            cycles_detected: false,
        };
        let score = causal_relevance(&result);
        // max(0.2, 0.95*0.95*ln(2)) = max(0.2, 0.625) = 0.625
        assert!(score > 0.5, "score={score}");
    }

    #[test]
    fn causal_relevance_averages_links_in_chain() {
        let l1 = make_link(0.0, Some(1.0), Some(1.0)); // 1.0 * ln(2) ≈ 0.693
        let l2 = make_link(0.0, Some(0.5), Some(0.5)); // 0.25 * ln(2) ≈ 0.173
        let result = CausalChainResult {
            chains: vec![CausalChain {
                links: vec![l1, l2],
            }],
            cycles_detected: false,
        };
        let score = causal_relevance(&result);
        // avg(0.693, 0.173) ≈ 0.433
        let expected = f32::midpoint(1.0 * 1.0 * (2.0_f32).ln(), 0.5 * 0.5 * (2.0_f32).ln());
        assert!(
            (score - expected).abs() < 0.01,
            "score={score}, expected={expected}"
        );
    }

    #[test]
    fn link_score_strength_only() {
        let link = make_link(0.3, Some(0.8), None);
        assert!((link_score(&link) - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn link_score_confidence_only() {
        let link = make_link(0.3, None, Some(0.7));
        assert!((link_score(&link) - 0.7).abs() < f32::EPSILON);
    }
}
