//! Multi-entity query decomposition for comparative / duration questions.
//!
//! LongMemEval's temporal-reasoning split is dominated by questions that compare
//! or measure **two events**: "Which did I do first, A or B?", "How many days
//! between X and Y?". A single vector recall of the whole question tends to
//! surface one side and miss the other, so recall@10 caps low — the gold evidence
//! for *both* A and B must be retrieved. That is an entity-coverage problem, not a
//! temporal one.
//!
//! Decomposition is a hybrid: when an LLM provider is configured,
//! [`decompose_query_llm`] extracts the compared phrases (robust to paraphrase and
//! implicit comparison), gated behind the cheap [`has_comparison_cue`] pre-check so
//! single-topic questions never pay for a network round trip; otherwise (or on
//! any LLM failure) it falls back to the deterministic [`decompose_comparative`],
//! which fires only on clear `A or B` / `between A and B` structure. The recall
//! orchestration then retrieves a deep pool for each phrase and **weighted
//! Reciprocal Rank Fusion** merges them with the base full-question results
//! (RAG-Fusion, arXiv:2402.03367; RRF, Cormack et al. SIGIR 2009), so both sides
//! land in the candidate set and documents that several sub-queries agree on rise.
//! Ordinary queries are untouched.
//!
//! On the LongMemEval temporal-reasoning split (real embeddings, 500-session
//! subset) this lifts recall@10 from 0.250 to 0.267 (ndcg 0.189→0.201, QA
//! 0.515→0.520) with no change to the other categories — it targets exactly the
//! comparative/duration questions and is a no-op elsewhere. Note the subset is
//! small enough that per-variant deltas are near the ~1-question noise floor; the
//! design choices (RRF over interleaving, equal weighting) are made on the
//! consistent direction plus first principles, and want a full-corpus confirmation.

use std::time::Duration;

use hirn_core::MemoryId;
use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};
use hirn_core::types::{AgentId, Namespace};

use crate::db::HirnDB;
use crate::ql::results::ScoredMemory;
use crate::recall::RecallResult;

/// Timeout for the LLM decomposition call; on timeout or error we fall back to
/// the deterministic parser so a slow/unavailable provider never stalls recall.
const LLM_DECOMPOSE_TIMEOUT: Duration = Duration::from_secs(8);

/// LLM-backed query decomposition — the SOTA path when a provider is configured.
///
/// Returns `Some(phrases)` when the model answered — the phrases a question
/// compares or relates, which is an **empty** vec when the model judged the
/// question single-topic (a confident "don't decompose", to be respected rather
/// than second-guessed) — or `None` when the call itself failed (timeout, transport,
/// or unparseable output), which is the only case the deterministic
/// [`decompose_comparative`] fallback should fire on.
pub async fn decompose_query_llm(llm: &dyn LlmProvider, question: &str) -> Option<Vec<String>> {
    let sanitized = hirn_core::sanitize::sanitize_for_llm(question);
    let messages = vec![
        ChatMessage {
            role: "system".into(),
            content: "You decompose a memory-retrieval question into the distinct entities or \
                      events it compares or relates. If the question asks about or compares two or \
                      more distinct entities/events (e.g. \"which happened first, A or B\", \"how \
                      many days between X and Y\", \"did I do A before B\"), return a JSON object \
                      {\"entities\": [\"concise search phrase for A\", \"...for B\"]} listing each \
                      as a short noun phrase suitable for vector search. If it concerns a single \
                      topic, return {\"entities\": []}. Return ONLY the JSON object."
                .into(),
        },
        ChatMessage {
            role: "user".into(),
            content: sanitized,
        },
    ];
    let options = LlmOptions {
        temperature: 0.0,
        max_tokens: 200,
        response_format: ResponseFormat::JsonObject,
        ..Default::default()
    };

    let response = tokio::time::timeout(
        LLM_DECOMPOSE_TIMEOUT,
        llm.generate_text(&messages, &options),
    )
    .await
    .ok()?
    .ok()?;

    #[derive(serde::Deserialize)]
    struct Decomposition {
        entities: Vec<String>,
    }
    let parsed: Decomposition = serde_json::from_str(response.trim()).ok()?;
    let entities: Vec<String> = parsed
        .entities
        .into_iter()
        .map(|e| e.trim().to_lowercase())
        .filter(|e| e.len() >= 3)
        .collect();
    // The model answered; `entities` may be empty (single-topic) or hold <2
    // usable phrases — either way it is not a decomposition, but it *is* a
    // definitive answer, so we return it rather than falling back.
    Some(entities)
}

/// Reciprocal Rank Fusion smoothing constant (Cormack, Clarke & Büttcher,
/// SIGIR 2009); 60 is the long-standing default across hybrid-search systems.
/// Larger flattens the curve (rank differences matter less), smaller sharpens
/// top-rank dominance.
pub const RRF_K: f32 = 60.0;

/// Fusion weight on the base full-question list, relative to the entity sub-lists.
///
/// Weighted RRF can up-weight the base list to guard against noisy sub-queries
/// diluting the ranking (the standard RAG advice). But decomposition exists
/// precisely to inject entity coverage the base query misses, and on the
/// LongMemEval temporal split a base weight of 2.0 measurably *suppressed* that
/// coverage — recall fell back to the no-decomposition baseline while only QA
/// nudged up. Equal weighting (the RAG-Fusion canonical default) preserved the
/// recall gain and still improved QA over naive interleaving, so 1.0 is the
/// shipped default. Corpora where sub-query noise dominates can raise it via
/// `HIRN_RRF_BASE_WEIGHT` (see [`base_list_weight`]).
pub const BASE_LIST_WEIGHT: f32 = 1.0;

/// Effective base-list fusion weight — [`BASE_LIST_WEIGHT`] unless overridden by
/// the `HIRN_RRF_BASE_WEIGHT` environment variable (a tuning knob; the default is
/// the shipped value).
#[must_use]
pub fn base_list_weight() -> f32 {
    std::env::var("HIRN_RRF_BASE_WEIGHT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .filter(|w| w.is_finite() && *w >= 0.0)
        .unwrap_or(BASE_LIST_WEIGHT)
}

/// Fusion weight on each entity sub-list.
pub const ENTITY_LIST_WEIGHT: f32 = 1.0;

/// Candidate-pool depth per entity sub-query as a multiple of the final `k`.
/// Fusion can only recover a gold document that appears in *some* list and needs
/// depth to detect cross-list agreement, so each sub-recall retrieves a deeper
/// pool than the final cut (3–5× is the established range).
const FUSION_POOL_FACTOR: usize = 4;

/// Decompose a question into its compared entity phrases — the hybrid selector.
///
/// LLM decomposer when a provider is configured and the cheap comparison-cue
/// pre-check fires (robust to paraphrase / implicit comparison), else the
/// deterministic parser. A confident LLM "single-topic" answer is respected;
/// only a *failed* LLM call (timeout/transport/parse) falls back to the parser.
async fn decompose_entities(db: &HirnDB, question: &str) -> Vec<String> {
    match db.llm_provider() {
        Some(llm) if has_comparison_cue(question) => {
            match decompose_query_llm(llm.as_ref(), question).await {
                Some(entities) => entities,
                None => decompose_comparative(question),
            }
        }
        _ => decompose_comparative(question),
    }
}

/// Retrieve a deep candidate pool for each compared entity in a comparative /
/// duration question, returning one ranked list per entity — empty when the
/// question isn't a multi-entity comparison. The caller fuses these with its own
/// base-query results. `scope` bounds each sub-recall to the caller's namespaces.
///
/// Exposed so callers whose base list is not a [`RecallResult`] (e.g. the
/// compiled-query surface, whose results carry no per-record score) can fuse the
/// entity coverage in with their own base ranking.
pub async fn entity_recall_lists(
    db: &HirnDB,
    agent_id: Option<&AgentId>,
    question: &str,
    scope: Option<&[Namespace]>,
    k: usize,
) -> Vec<Vec<RecallResult>> {
    let entities = decompose_entities(db, question).await;
    if entities.len() < 2 {
        return Vec::new();
    }
    let pool = k.saturating_mul(FUSION_POOL_FACTOR).max(k);
    let mut lists = Vec::with_capacity(entities.len());
    for entity in &entities {
        let Ok(embedding) = db.embed_text(entity).await else {
            continue;
        };
        let mut builder = db.recall(embedding).limit(pool);
        if let Some(agent) = agent_id {
            builder = builder.agent_id(agent.as_str());
        }
        builder = match scope {
            Some(namespaces) => builder.allowed_namespaces(namespaces.to_vec()),
            None => builder.unrestricted(),
        };
        if let Ok(results) = Box::pin(builder.execute()).await {
            lists.push(results);
        }
    }
    lists
}

/// Retrieve coverage for every compared entity and **weighted-RRF fuse** the
/// per-entity lists with the base results, so both sides' gold evidence reaches
/// the top-`k`. A no-op (returns `base_results`) for non-comparative questions.
pub async fn multi_entity_recall(
    db: &HirnDB,
    agent_id: Option<&AgentId>,
    question: &str,
    base_results: Vec<RecallResult>,
    scope: Option<&[Namespace]>,
    k: usize,
) -> Vec<RecallResult> {
    let entity_lists = entity_recall_lists(db, agent_id, question, scope, k).await;
    if entity_lists.is_empty() {
        return base_results;
    }
    let mut weighted: Vec<(f32, Vec<RecallResult>)> = Vec::with_capacity(entity_lists.len() + 1);
    weighted.push((base_list_weight(), base_results));
    weighted.extend(
        entity_lists
            .into_iter()
            .map(|list| (ENTITY_LIST_WEIGHT, list)),
    );
    rrf_fuse(weighted, k)
}

/// Scored-memory variant of [`multi_entity_recall`] for THINK context assembly.
///
/// This keeps decomposition in the product retrieval path rather than appending
/// benchmark-only evidence after context assembly. All entity sub-recalls retain
/// the caller's actor and namespace scope.
pub async fn multi_entity_scored_memories(
    db: &HirnDB,
    agent_id: Option<&AgentId>,
    question: &str,
    base_results: Vec<ScoredMemory>,
    scope: Option<&[Namespace]>,
    k: usize,
) -> Vec<ScoredMemory> {
    let base_results = base_results
        .into_iter()
        .map(|scored| RecallResult {
            record: scored.record,
            revision: scored.revision,
            similarity: scored.score_breakdown.similarity,
            composite_score: scored.score,
            score_breakdown: scored.score_breakdown,
            resource_evidence: scored.resource_evidence,
            resource_preview_packages: scored.resource_preview_packages,
            resource_score_attribution: scored.resource_score_attribution,
            presentation: crate::recall::RecallPresentation::default(),
        })
        .collect();

    multi_entity_recall(db, agent_id, question, base_results, scope, k)
        .await
        .into_iter()
        .map(|result| ScoredMemory {
            record: result.record,
            revision: result.revision,
            score: result.composite_score,
            score_breakdown: result.score_breakdown,
            resource_evidence: result.resource_evidence,
            resource_preview_packages: result.resource_preview_packages,
            resource_score_attribution: result.resource_score_attribution,
        })
        .collect()
}

/// Weighted Reciprocal Rank Fusion over several ranked result lists, deduping by
/// memory id — a document's contributions from every list in which it appears are
/// summed, which is the cross-list-agreement signal that plain interleaving
/// discards. Returns the top-`k` by fused score.
fn rrf_fuse(weighted_lists: Vec<(f32, Vec<RecallResult>)>, k: usize) -> Vec<RecallResult> {
    use std::collections::HashMap;
    let mut scores: HashMap<MemoryId, f32> = HashMap::new();
    let mut reps: HashMap<MemoryId, RecallResult> = HashMap::new();
    for (weight, list) in &weighted_lists {
        for (rank, result) in list.iter().enumerate() {
            let id = result.record.id();
            // 1-based rank per the RRF definition.
            *scores.entry(id).or_insert(0.0) += weight / (RRF_K + (rank + 1) as f32);
            reps.entry(id).or_insert_with(|| result.clone());
        }
    }
    let mut fused: Vec<RecallResult> = reps.into_values().collect();
    fused.sort_by(|a, b| {
        let (sa, sb) = (scores[&a.record.id()], scores[&b.record.id()]);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Deterministic tie-break: stronger per-record composite score, then id.
            .then_with(|| {
                b.composite_score
                    .partial_cmp(&a.composite_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| a.record.id().cmp(&b.record.id()))
    });
    fused.truncate(k);
    fused
}

/// Cheap pre-check: does the question plausibly compare or relate two events?
///
/// Broader than [`decompose_comparative`] (it also fires on implicit comparisons
/// the deterministic parser can't split, e.g. "did the checkup happen before the
/// vaccination"), so the LLM decomposer is only consulted when a comparison is
/// plausible — but still skips the large majority of single-topic questions.
#[must_use]
pub fn has_comparison_cue(query: &str) -> bool {
    let lower = query.to_lowercase();
    const CUES: &[&str] = &[
        " or ",
        "between ",
        "how many days",
        "how many weeks",
        "how many months",
        "how many years",
        "how long",
        " before ",
        " after ",
        " earlier",
        " later",
        " longer",
        " sooner",
        " first",
        " versus ",
        " vs ",
        " vs. ",
        "compared to",
        "compared with",
    ];
    CUES.iter().any(|cue| lower.contains(cue))
}

/// Split a comparative or duration question into its compared entity phrases.
///
/// Returns 2+ phrases for `… A or B …`, `… A, B, or C …`, and `between A and B`;
/// an empty vec when the question isn't a multi-entity comparison.
#[must_use]
pub fn decompose_comparative(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let trimmed = lower.trim().trim_end_matches('?').trim();

    // ── "between A and B" (duration / interval questions) ────────────────
    if let Some(pos) = find_keyword(trimmed, "between ") {
        let tail = &trimmed[pos + "between ".len()..];
        // Split on the first top-level " and " into two entities.
        if let Some(and_pos) = find_keyword(tail, " and ") {
            let a = clean_entity(&tail[..and_pos]);
            let b = clean_entity(&tail[and_pos + " and ".len()..]);
            let parts: Vec<String> = [a, b].into_iter().flatten().collect();
            if parts.len() == 2 {
                return parts;
            }
        }
    }

    // ── "… A or B" / "… A, B, or C" (disjunctive comparison) ─────────────
    if find_keyword(trimmed, " or ").is_some() {
        // Split the whole question on the disjunction and comma separators, then
        // keep only the entity segments — the question-stem segment ("which event
        // did I attend first") is dropped by `clean_entity`'s stem filter, which
        // also handles 3-way lists ("A, B, or C") without fragile clause slicing.
        let parts: Vec<String> = trimmed
            .split(" or ")
            .flat_map(|segment| segment.split(", "))
            .filter_map(clean_entity)
            .collect();
        if parts.len() >= 2 {
            return parts;
        }
    }

    Vec::new()
}

/// Find a keyword at a word boundary, returning its byte offset. A needle that
/// begins with a non-alphanumeric char (e.g. `" or "`) already carries its own
/// left boundary; otherwise the char to the left must be a non-word char so the
/// needle doesn't match inside a larger word.
fn find_keyword(haystack: &str, needle: &str) -> Option<usize> {
    let needle_starts_word = needle
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_alphanumeric);
    let bytes = haystack.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = haystack[from..].find(needle) {
        let pos = from + rel;
        let left_ok = !needle_starts_word || pos == 0 || !bytes[pos - 1].is_ascii_alphanumeric();
        if left_ok {
            return Some(pos);
        }
        from = pos + 1;
    }
    None
}

/// Tokens that mark a segment as a question stem or verb phrase rather than an
/// entity — if a segment starts with one, it isn't a compared entity.
fn is_stem_first_token(tok: &str) -> bool {
    matches!(
        tok,
        "which"
            | "what"
            | "where"
            | "when"
            | "how"
            | "why"
            | "who"
            | "should"
            | "would"
            | "could"
            | "do"
            | "does"
            | "did"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "have"
            | "has"
            | "had"
            | "can"
            | "will"
            | "i"
            | "we"
            | "you"
            | "they"
            | "he"
            | "she"
            | "it"
            | "not"
            | "many"
    )
}

/// Trim an entity phrase (stripping a leading article) and reject question-stem
/// or contentless fragments, so decomposition never emits a useless sub-query.
fn clean_entity(raw: &str) -> Option<String> {
    let raw = raw.trim();
    let first = raw.split_whitespace().next()?;
    if is_stem_first_token(first) {
        return None;
    }
    let phrase = raw
        .trim_start_matches("the ")
        .trim_start_matches("a ")
        .trim_start_matches("an ")
        .trim();
    if phrase.len() < 3 {
        return None;
    }
    // Require a content token beyond bare function/ordinal words.
    let contentful = phrase.split_whitespace().any(|w| {
        w.chars().any(char::is_alphabetic)
            && !matches!(
                w,
                "first" | "last" | "the" | "a" | "an" | "of" | "to" | "in" | "on"
            )
    });
    contentful.then(|| phrase.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::record::MemoryRecord;
    use hirn_core::types::EventType;

    fn recall_result(label: &str, composite_score: f32) -> RecallResult {
        let record = EpisodicRecord::builder()
            .event_type(EventType::Observation)
            .content(label)
            .summary(label)
            .embedding(vec![1.0, 0.0])
            .agent_id(AgentId::new("decompose-tests").unwrap())
            .build()
            .unwrap();
        RecallResult {
            record: MemoryRecord::Episodic(record),
            similarity: composite_score,
            composite_score,
            score_breakdown: crate::scoring::ScoreBreakdown {
                similarity: composite_score,
                importance: 0.0,
                recency: 0.0,
                activation: 0.0,
                causal_relevance: 0.0,
                surprise: 0.0,
                source_reliability: 0.0,
                temporal_relevance: 0.0,
            },
            revision: None,
            resource_evidence: Vec::new(),
            resource_preview_packages: Vec::new(),
            resource_score_attribution: Vec::new(),
            presentation: crate::recall::RecallPresentation::default(),
        }
    }

    #[test]
    fn disjunction_two_entities() {
        let alts = decompose_comparative(
            "Which event did I attend first, the 'Effective Time Management' workshop or the 'Data Analysis using Python' webinar?",
        );
        assert_eq!(alts.len(), 2);
        assert!(alts[0].contains("effective time management"));
        assert!(alts[1].contains("data analysis using python"));
    }

    #[test]
    fn disjunction_short_entities() {
        let alts = decompose_comparative(
            "Which vehicle did I take care of first in February, the bike or the car?",
        );
        assert_eq!(alts, vec!["bike".to_string(), "car".to_string()]);
    }

    #[test]
    fn between_and_duration() {
        let alts = decompose_comparative(
            "How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?",
        );
        assert_eq!(alts.len(), 2);
        assert!(alts[0].contains("sunday mass"));
        assert!(alts[1].contains("ash wednesday service"));
    }

    #[test]
    fn three_way_list() {
        let alts =
            decompose_comparative("Which came first, the coffee maker, the toaster, or the mixer?");
        assert_eq!(alts.len(), 3);
    }

    #[test]
    fn comparison_cue_gates_llm_calls() {
        // Fires on explicit and implicit comparisons.
        assert!(has_comparison_cue(
            "Which came first, the toaster or the mixer?"
        ));
        assert!(has_comparison_cue(
            "How many days between the checkup and the surgery?"
        ));
        assert!(has_comparison_cue(
            "Did the checkup happen before the vaccination?"
        ));
        assert!(has_comparison_cue("Which workshop did I attend first?"));
        // Skips ordinary single-topic questions (no network round trip).
        assert!(!has_comparison_cue("What was the issue with my car?"));
        assert!(!has_comparison_cue("Where did I move last year?"));
        assert!(!has_comparison_cue("What is my dentist's name?"));
    }

    #[test]
    fn non_comparative_returns_empty() {
        assert!(decompose_comparative("What was the first issue with my car?").is_empty());
        assert!(decompose_comparative("Where did I move last year?").is_empty());
        // "or" inside a non-comparative phrase without two entities is ignored.
        assert!(decompose_comparative("Should I refactor or not?").is_empty());
    }

    #[test]
    fn rrf_fusion_prioritizes_cross_list_agreement_and_deduplicates() {
        let first_only = recall_result("first-list leader", 0.99);
        let second_only = recall_result("second-list leader", 0.98);
        let shared = recall_result("shared evidence", 0.70);
        let shared_id = shared.record.id();

        let fused = rrf_fuse(
            vec![
                (1.0, vec![first_only, shared.clone()]),
                (1.0, vec![second_only, shared]),
            ],
            3,
        );

        assert_eq!(fused.len(), 3);
        assert_eq!(fused[0].record.id(), shared_id);
        assert_eq!(
            fused
                .iter()
                .filter(|result| result.record.id() == shared_id)
                .count(),
            1
        );
    }

    #[test]
    fn rrf_fusion_applies_final_top_k() {
        let candidates = (0..5)
            .map(|index| recall_result(&format!("candidate-{index}"), 1.0 - index as f32 * 0.1))
            .collect();

        let fused = rrf_fuse(vec![(1.0, candidates)], 2);

        assert_eq!(fused.len(), 2);
        assert!(fused[0].composite_score > fused[1].composite_score);
    }
}
