use std::sync::Arc;

use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};
use hirn_core::nlu::{Classification, ClassificationTask, DecisionSource, LabelSpec};
use hirn_provider::HybridClassifier;

use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Concept Extraction
// ═══════════════════════════════════════════════════════════════════════════

/// A concept extracted from a narrative thread.
#[derive(Debug, Clone)]
pub struct ExtractedConcept {
    pub concept_name: String,
    pub description: String,
    pub knowledge_type: KnowledgeType,
    pub confidence: f32,
    pub source_episode_ids: Vec<MemoryId>,
    pub contradiction_ids: Vec<MemoryId>,
    pub embedding: Option<Vec<f32>>,
}

/// F-047 FIX: Extract semantic concepts from narrative threads.
///
/// When an `LlmProvider` is available, uses structured LLM extraction for
/// richer concept names, descriptions, and knowledge type classification.
/// Falls back to deterministic heuristic extraction when no LLM is provided
/// or when the LLM call fails.
pub async fn extract_concepts(
    threads: &[NarrativeThread],
    db: &HirnDB,
    llm: Option<&Arc<dyn LlmProvider>>,
    llm_timeout: std::time::Duration,
) -> Vec<ExtractedConcept> {
    if let Some(llm) = llm {
        match llm_extract_concepts(llm, threads, db, llm_timeout).await {
            Ok(concepts) => return concepts,
            Err(e) => {
                tracing::warn!("LLM concept extraction failed, falling back to heuristic: {e}");
            }
        }
    }
    heuristic_extract_concepts(threads, db).await
}

/// JSON schema for one extracted concept.
///
/// The `knowledge_type` enum is derived from [`KNOWLEDGE_TYPE_TASK`] so the
/// extraction path and the classification path can never drift apart.
fn concept_schema() -> String {
    let types: Vec<&str> = KNOWLEDGE_TYPE_TASK.labels.iter().map(|l| l.name).collect();
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["concept_name", "description", "knowledge_type", "confidence"],
        "properties": {
            "concept_name": {"type": "string"},
            "description": {"type": "string"},
            "knowledge_type": {"type": "string", "enum": types},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    })
    .to_string()
}

/// LLM-powered concept extraction. Sends thread descriptions to the LLM and
/// parses structured JSON responses for concept name, description, and type.
async fn llm_extract_concepts(
    llm: &Arc<dyn LlmProvider>,
    threads: &[NarrativeThread],
    db: &HirnDB,
    llm_timeout: std::time::Duration,
) -> HirnResult<Vec<ExtractedConcept>> {
    let mut concepts = Vec::new();

    for thread in threads {
        let description = build_thread_description_deduped(
            thread,
            db.embedder().as_deref(),
            db.nlu_config().summary_dedup_threshold,
        )
        .await;
        // Scope the parking_lot guard so it is dropped before the .await —
        // parking_lot guards are !Send, which would make this future !Send.
        let contradiction_ids = find_contradictions_in_thread(thread, db.graph_store()).await?;

        let sanitized_title = hirn_core::sanitize::sanitize_for_llm(&thread.title);
        let sanitized_desc = hirn_core::sanitize::sanitize_for_llm(
            &description.chars().take(2000).collect::<String>(),
        );
        let prompt = format!(
            "Extract the single most important concept from the following narrative thread.\n\
             Respond with a JSON object (no markdown fences) with exactly these fields:\n\
             - \"concept_name\": a short canonical name (2-5 words)\n\
             - \"description\": a one-sentence description of the concept\n\
             - \"knowledge_type\": one of \"propositional\", \"prescriptive\", or \"taxonomic\"\n\
             - \"confidence\": a float between 0.0 and 1.0 indicating extraction confidence\n\n\
             Thread title: {}\n\
             Thread content ({} episodes):\n{}",
            sanitized_title,
            thread.record_ids.len(),
            sanitized_desc,
        );

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a knowledge extraction engine. Output valid JSON only."
                    .to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];

        // A pinned schema rather than a free JSON object: the knowledge-type
        // enum is the same label set the classifier task uses, so a schema-
        // enforcing provider cannot answer with a type that does not exist.
        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: 256,
            response_format: ResponseFormat::JsonSchema(concept_schema()),
            ..Default::default()
        };

        let response =
            super::generate_text_with_timeout(llm.as_ref(), &messages, &options, llm_timeout)
                .await?;

        // Parse the JSON response.
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&response) {
            let concept_name = parsed["concept_name"]
                .as_str()
                .unwrap_or(&thread.title)
                .to_string();
            let desc = parsed["description"]
                .as_str()
                .unwrap_or(&description)
                .to_string();
            let kt = knowledge_type_from_label(
                parsed["knowledge_type"].as_str().unwrap_or("propositional"),
            );
            let confidence = parsed["confidence"]
                .as_f64()
                .map(|c| (c as f32).clamp(0.1, 1.0))
                .unwrap_or(0.7);

            // Apply contradiction penalty.
            let penalty = if contradiction_ids.is_empty() {
                0.0
            } else {
                0.15 * contradiction_ids.len() as f32
            };

            concepts.push(ExtractedConcept {
                concept_name,
                description: desc,
                knowledge_type: kt,
                confidence: (confidence - penalty).clamp(0.1, 1.0),
                source_episode_ids: thread.record_ids.clone(),
                contradiction_ids,
                embedding: thread.embedding.clone(),
            });
        } else {
            // JSON parse failed — fall back to heuristic for this thread.
            tracing::debug!(
                "LLM returned non-JSON for thread '{}', using heuristic",
                thread.title
            );
            let fallback = heuristic_extract_single(thread, db.graph_store(), db).await?;
            concepts.push(fallback);
        }
    }

    Ok(concepts)
}

/// Heuristic concept extraction (original deterministic logic).
async fn heuristic_extract_concepts(
    threads: &[NarrativeThread],
    db: &HirnDB,
) -> Vec<ExtractedConcept> {
    let mut concepts = Vec::new();
    for t in threads {
        match heuristic_extract_single(t, db.graph_store(), db).await {
            Ok(c) => concepts.push(c),
            Err(e) => {
                tracing::warn!("heuristic extraction failed for thread '{}': {e}", t.title);
            }
        }
    }
    concepts
}

async fn heuristic_extract_single(
    thread: &NarrativeThread,
    store: &dyn crate::graph_store::GraphStore,
    db: &HirnDB,
) -> HirnResult<ExtractedConcept> {
    let concept_name = thread.title.clone();
    let description = build_thread_description_deduped(
        thread,
        db.embedder().as_deref(),
        db.nlu_config().summary_dedup_threshold,
    )
    .await;
    // Knowledge typing is a semantic judgment even on the no-LLM extraction
    // path: the embedding router can decide it without a generation call.
    let knowledge_type = classify_knowledge_type(&db.nlu_classifier(), thread).await;

    let evidence_count = thread.record_ids.len();
    let base_confidence = match evidence_count {
        1 => 0.3,
        2..=3 => 0.5,
        4..=7 => 0.7,
        _ => 0.85,
    };

    let contradiction_ids = find_contradictions_in_thread(thread, store).await?;
    let contradiction_penalty = if contradiction_ids.is_empty() {
        0.0
    } else {
        0.15 * contradiction_ids.len() as f32
    };
    let confidence = (base_confidence - contradiction_penalty).clamp(0.1, 1.0);

    Ok(ExtractedConcept {
        concept_name,
        description,
        knowledge_type,
        confidence,
        source_episode_ids: thread.record_ids.clone(),
        contradiction_ids,
        embedding: thread.embedding.clone(),
    })
}

/// Build a coherent description from a thread's summaries and content.
///
/// Deduplicates by exact text only. Two summaries that say the same thing in
/// different words both survive, padding the description with redundancy —
/// prefer [`build_thread_description_deduped`] where an embedder is available.
pub(super) fn build_thread_description(thread: &NarrativeThread) -> String {
    let summaries = non_empty_summaries(thread);
    if summaries.is_empty() {
        return fallback_description(thread);
    }

    let mut unique: Vec<&str> = Vec::new();
    for summary in &summaries {
        if !unique.iter().any(|u| u == summary) {
            unique.push(summary);
        }
    }

    join_description(unique)
}

/// Build a description with **semantic** summary deduplication.
///
/// Exact-match deduplication only catches summaries that are byte-identical;
/// consolidation routinely produces "the team chose Postgres" and "Postgres was
/// selected by the team" for the same thread, and both used to land in the
/// concept description. Summaries are embedded once and any whose cosine
/// similarity to an already-kept summary exceeds `threshold` is dropped.
///
/// The exact-match check runs first as a collision-safe fast path, so identical
/// text is dropped even when the embedder is unavailable or returns degenerate
/// vectors. Any embedding failure degrades to
/// [`build_thread_description`] rather than failing consolidation.
pub(super) async fn build_thread_description_deduped(
    thread: &NarrativeThread,
    embedder: Option<&dyn hirn_core::embed::Embedder>,
    threshold: f32,
) -> String {
    let summaries = non_empty_summaries(thread);
    if summaries.is_empty() {
        return fallback_description(thread);
    }

    let Some(embedder) = embedder else {
        return build_thread_description(thread);
    };

    // Exact-match pass first: cheaper, and it shrinks the embedding batch.
    let mut exact_unique: Vec<&str> = Vec::new();
    for summary in &summaries {
        if !exact_unique.iter().any(|u| u == summary) {
            exact_unique.push(summary);
        }
    }
    if exact_unique.len() < 2 {
        return join_description(exact_unique);
    }

    let embeddings = match embedder.embed(&exact_unique).await {
        Ok(embeddings) if embeddings.len() == exact_unique.len() => embeddings,
        Ok(_) | Err(_) => {
            tracing::debug!(
                thread = %thread.title,
                "summary embedding unavailable; falling back to exact-match dedup"
            );
            return join_description(exact_unique);
        }
    };

    let mut kept: Vec<(&str, &[f32])> = Vec::with_capacity(exact_unique.len());
    for (summary, embedding) in exact_unique.iter().zip(embeddings.iter()) {
        let duplicate = kept.iter().any(|(_, kept_vector)| {
            hirn_core::nlu::cosine_similarity(&embedding.vector, kept_vector) >= threshold
        });
        if !duplicate {
            kept.push((summary, &embedding.vector));
        }
    }

    join_description(kept.into_iter().map(|(summary, _)| summary).collect())
}

fn non_empty_summaries(thread: &NarrativeThread) -> Vec<&str> {
    thread
        .summaries
        .iter()
        .filter(|s| !s.is_empty())
        .map(String::as_str)
        .collect()
}

fn fallback_description(thread: &NarrativeThread) -> String {
    thread
        .contents
        .iter()
        .take(5)
        .map(String::as_str)
        .collect::<Vec<&str>>()
        .join(". ")
}

fn join_description(summaries: Vec<&str>) -> String {
    summaries
        .into_iter()
        .take(10)
        .collect::<Vec<&str>>()
        .join(". ")
}

/// The knowledge-typing decision surface.
pub const KNOWLEDGE_TYPE_TASK: ClassificationTask = ClassificationTask {
    name: "knowledge_type",
    instruction: "Classify what kind of knowledge a body of related notes carries. Judge what \
                  the notes are doing, not which words appear: a rule can be stated without \
                  the word \"should\", and a note can mention deploying without being an \
                  instruction.",
    labels: &[
        LabelSpec {
            name: "propositional",
            description: "States facts, observations, or what happened. The default when the \
                          notes are neither instructions nor a classification scheme.",
            exemplars: &[
                "the outage started at 14:02 and lasted nineteen minutes",
                "revenue grew for three consecutive quarters",
            ],
        },
        LabelSpec {
            name: "prescriptive",
            description: "Tells someone what to do: rules, procedures, policies, conventions, \
                          or best practices.",
            exemplars: &[
                "always run the migration against staging before production",
                "rotate credentials every ninety days; never commit them to the repo",
                "the review checklist: lint, test, then a second pair of eyes",
            ],
        },
        LabelSpec {
            name: "taxonomic",
            description: "Organizes things into categories, types, or a hierarchy — what \
                          something is an instance of, or how a domain is subdivided.",
            exemplars: &[
                "a semantic record is a kind of memory record, alongside episodic and procedural",
                "our services fall into three tiers: edge, core, and batch",
            ],
        },
    ],
    default_label: "propositional",
};

/// Map a classifier label onto a [`KnowledgeType`].
fn knowledge_type_from_label(label: &str) -> KnowledgeType {
    match label {
        "prescriptive" => KnowledgeType::Prescriptive,
        "taxonomic" => KnowledgeType::Taxonomic,
        // Anything unrecognized takes the conservative type rather than
        // guessing a more specific one.
        _ => KnowledgeType::Propositional,
    }
}

/// Classify a thread's knowledge type.
///
/// The classifier chain sees the whole thread and judges what the notes are
/// doing; [`infer_knowledge_type`] is the provider-free floor.
pub(super) async fn classify_knowledge_type(
    classifier: &HybridClassifier,
    thread: &NarrativeThread,
) -> KnowledgeType {
    let context = thread.contents.join("\n");
    let decision = classifier
        .decide(&KNOWLEDGE_TYPE_TASK, &context, Some(&thread.title), || {
            Classification::new(
                knowledge_type_label(infer_knowledge_type(thread)),
                1.0,
                DecisionSource::Heuristic,
                Some("cue fallback: no model-backed backend produced a decision".to_owned()),
            )
        })
        .await;
    knowledge_type_from_label(&decision.label)
}

/// The task label for a knowledge type.
const fn knowledge_type_label(knowledge_type: KnowledgeType) -> &'static str {
    match knowledge_type {
        KnowledgeType::Prescriptive => "prescriptive",
        KnowledgeType::Taxonomic => "taxonomic",
        _ => "propositional",
    }
}

/// Provider-free knowledge typing from cue words.
///
/// Word-level tokenization avoids substring false positives (e.g. "category"
/// inside "subcategorize"); multi-word cues are matched as contiguous
/// sequences. It cannot see a rule phrased without a modal verb or a hierarchy
/// described without taxonomy vocabulary, and it is English-only — which is
/// why [`classify_knowledge_type`] prefers a model and falls back here.
///
/// Both cue classes require two independent hits, so a single incidental word
/// cannot retype a thread; failing that, the conservative `Propositional` type
/// applies.
pub(super) fn infer_knowledge_type(thread: &NarrativeThread) -> KnowledgeType {
    let all_content: String = thread.contents.join(" ").to_lowercase();
    let words: Vec<&str> = all_content.split_whitespace().collect();
    let joined = words.join(" "); // normalized whitespace for phrase matching

    // Prescriptive: instructions, rules, best practices.
    let prescriptive_signals = [
        "should",
        "must",
        "always",
        "never",
        "best practice",
        "rule",
        "recommend",
        "configure",
        "set up",
        "deploy",
    ];
    let prescriptive_score: usize = prescriptive_signals
        .iter()
        .filter(|&signal| {
            if signal.contains(' ') {
                // Multi-word phrase: check in normalized joined string
                joined.contains(signal)
            } else {
                // Single word: check word boundaries
                words
                    .iter()
                    .any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *signal)
            }
        })
        .count();

    // Taxonomic: categorization, hierarchy, types.
    let taxonomic_signals = [
        "type of",
        "kind of",
        "category",
        "classify",
        "hierarchy",
        "subtypes",
        "belongs to",
        "instance of",
        "is a",
    ];
    let taxonomic_score: usize = taxonomic_signals
        .iter()
        .filter(|&signal| {
            if signal.contains(' ') {
                joined.contains(signal)
            } else {
                words
                    .iter()
                    .any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *signal)
            }
        })
        .count();

    if prescriptive_score >= 2 {
        KnowledgeType::Prescriptive
    } else if taxonomic_score >= 2 {
        KnowledgeType::Taxonomic
    } else {
        KnowledgeType::Propositional
    }
}

async fn find_contradictions_in_thread(
    thread: &NarrativeThread,
    store: &dyn crate::graph_store::GraphStore,
) -> HirnResult<Vec<MemoryId>> {
    let ids: HashSet<MemoryId> = thread.record_ids.iter().copied().collect();
    let mut contradictions = Vec::new();

    for &id in &thread.record_ids {
        let edges = store
            .get_edges_of_type(id, EdgeRelation::Contradicts)
            .await?;
        for edge in edges {
            if ids.contains(&edge.target) && !contradictions.contains(&edge.target) {
                contradictions.push(edge.target);
            }
        }
    }

    Ok(contradictions)
}
