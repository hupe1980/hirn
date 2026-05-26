use std::sync::Arc;

use hirn_core::embed::{ChatMessage, LlmOptions, LlmProvider, ResponseFormat};

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
        let description = build_thread_description(thread);
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

        let options = LlmOptions {
            temperature: 0.0,
            max_tokens: 256,
            response_format: ResponseFormat::JsonObject,
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
            let kt = match parsed["knowledge_type"].as_str().unwrap_or("propositional") {
                "prescriptive" => KnowledgeType::Prescriptive,
                "taxonomic" => KnowledgeType::Taxonomic,
                _ => KnowledgeType::Propositional,
            };
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
            let fallback = heuristic_extract_single(thread, db.graph_store()).await?;
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
        match heuristic_extract_single(t, db.graph_store()).await {
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
) -> HirnResult<ExtractedConcept> {
    let concept_name = thread.title.clone();
    let description = build_thread_description(thread);
    let knowledge_type = infer_knowledge_type(thread);

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
pub(super) fn build_thread_description(thread: &NarrativeThread) -> String {
    // Use summaries for concise description. Filter empty summaries.
    let summaries: Vec<&str> = thread
        .summaries
        .iter()
        .filter(|s| !s.is_empty())
        .map(String::as_str)
        .collect();

    if summaries.is_empty() {
        // Fall back to content.
        let contents: Vec<&str> = thread.contents.iter().take(5).map(String::as_str).collect();
        return contents.join(". ");
    }

    // Deduplicate similar summaries.
    let mut unique_summaries: Vec<&str> = Vec::new();
    for s in &summaries {
        if !unique_summaries.iter().any(|u| u == s) {
            unique_summaries.push(s);
        }
    }

    // Take top summaries and join.
    unique_summaries
        .into_iter()
        .take(10)
        .collect::<Vec<&str>>()
        .join(". ")
}

/// Infer knowledge type from thread content using word-boundary matching.
///
/// Uses word-level tokenization to avoid false positives from substring
/// matches (e.g., "category" inside "subcategorize"). Multi-word phrases
/// are matched as contiguous sequences.
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
