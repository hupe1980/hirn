//! Query-adaptive view routing (MAGMA-style, arXiv:2601.03236).
//!
//! MAGMA's insight: a single fused retrieval pipeline under-serves
//! structure-sensitive questions. Instead, classify a query's *intent* and route
//! it to the memory view(s) that match — a "when/before/after" question to the
//! **temporal** view, a "why/because" question to the **causal** view, a
//! "who/what/which" question to the **entity** view, and everything else to the
//! **semantic** (embedding) view — with per-view weights rather than one-size
//! fusion.
//!
//! This module is the deterministic classifier at the heart of that router:
//! pure string analysis producing normalized [`ViewWeights`], fully unit-testable
//! without a model. hirn already *has* all four views (hybrid embedding recall,
//! the [`temporal`](crate::retrieval) timeline, causal edges/chains, and the
//! entity/property graph); this decides which to lean on per query.

/// The four memory views a query can be routed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewKind {
    /// Embedding-similarity recall (the default, general-purpose view).
    Semantic,
    /// Temporal ordering / timeline reasoning ("when", "before", "how long").
    Temporal,
    /// Causal edges and chains ("why", "because", "led to").
    Causal,
    /// Entity/factoid lookup ("who", "which", "where").
    Entity,
}

impl ViewKind {
    /// Stable machine-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Temporal => "temporal",
            Self::Causal => "causal",
            Self::Entity => "entity",
        }
    }
}

/// Normalized per-view routing weights (sum ≈ 1.0) plus the dominant view.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewWeights {
    pub semantic: f32,
    pub temporal: f32,
    pub causal: f32,
    pub entity: f32,
}

impl ViewWeights {
    /// The view with the greatest weight (ties resolve to the more specific
    /// structured view: Temporal > Causal > Entity > Semantic).
    #[must_use]
    pub fn primary(&self) -> ViewKind {
        // `>=` so a structured view wins ties against Semantic; the iteration
        // order (Entity → Causal → Temporal) resolves structured-vs-structured
        // ties toward the more specific view (Temporal > Causal > Entity).
        let mut best = (ViewKind::Semantic, self.semantic);
        for (kind, w) in [
            (ViewKind::Entity, self.entity),
            (ViewKind::Causal, self.causal),
            (ViewKind::Temporal, self.temporal),
        ] {
            if w >= best.1 {
                best = (kind, w);
            }
        }
        best.0
    }

    fn normalized(semantic: f32, temporal: f32, causal: f32, entity: f32) -> Self {
        let total = semantic + temporal + causal + entity;
        if total <= f32::EPSILON {
            return Self {
                semantic: 1.0,
                temporal: 0.0,
                causal: 0.0,
                entity: 0.0,
            };
        }
        Self {
            semantic: semantic / total,
            temporal: temporal / total,
            causal: causal / total,
            entity: entity / total,
        }
    }
}

/// Count how many of `cues` appear as whole words in `tokens`.
fn cue_hits(tokens: &[&str], cues: &[&str]) -> u32 {
    tokens
        .iter()
        .filter(|t| cues.contains(&t.to_lowercase().as_str()))
        .count() as u32
}

/// Whether any multi-word phrase in `phrases` occurs in `lower`.
fn phrase_hits(lower: &str, phrases: &[&str]) -> u32 {
    phrases.iter().filter(|p| lower.contains(**p)).count() as u32
}

/// Temporal single-word cues.
const TEMPORAL_WORDS: &[&str] = &[
    "when",
    "before",
    "after",
    "during",
    "then",
    "first",
    "last",
    "earliest",
    "latest",
    "ago",
    "since",
    "until",
    "till",
    "timeline",
    "chronological",
    "chronology",
    "order",
    "sequence",
    "date",
    "day",
    "week",
    "month",
    "year",
    "yesterday",
    "today",
    "recently",
    "earlier",
    "later",
    "prior",
    "subsequent",
    "meanwhile",
    "eventually",
    "initially",
    "finally",
];
/// Temporal phrases.
const TEMPORAL_PHRASES: &[&str] = &[
    "how long",
    "what time",
    "in what order",
    "leading up to",
    "up to now",
    "over time",
    "at the time",
    "how often",
    "how many times",
];

/// Causal single-word cues.
const CAUSAL_WORDS: &[&str] = &[
    "why",
    "because",
    "cause",
    "caused",
    "causes",
    "causing",
    "reason",
    "reasons",
    "effect",
    "effects",
    "consequence",
    "consequences",
    "result",
    "results",
    "resulted",
    "trigger",
    "triggered",
    "leads",
    "led",
    "enables",
    "prevents",
    "impact",
    "influence",
    "affect",
    "affected",
];
/// Causal phrases.
const CAUSAL_PHRASES: &[&str] = &[
    "why did",
    "why does",
    "why is",
    "led to",
    "leads to",
    "result of",
    "due to",
    "because of",
    "as a result",
    "caused by",
    "reason for",
    "what if",
    "would have",
    "responsible for",
];

/// Entity/factoid single-word cues.
const ENTITY_WORDS: &[&str] = &["who", "whom", "whose", "which", "where", "what"];
/// Entity phrases.
const ENTITY_PHRASES: &[&str] = &["who is", "who was", "what is", "which one", "how many"];

/// Classify a query into normalized [`ViewWeights`].
///
/// A base semantic weight is always present (embedding recall is the fallback);
/// temporal / causal / entity weights accrue from whole-word and phrase cue
/// hits. The result is normalized so the weights sum to ~1.0.
#[must_use]
pub fn classify_query(query: &str) -> ViewWeights {
    let lower = query.to_lowercase();
    let tokens: Vec<&str> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    // Phrases weigh double a single-word cue (stronger signal).
    let temporal = cue_hits(&tokens, TEMPORAL_WORDS) as f32
        + 2.0 * phrase_hits(&lower, TEMPORAL_PHRASES) as f32;
    let causal =
        cue_hits(&tokens, CAUSAL_WORDS) as f32 + 2.0 * phrase_hits(&lower, CAUSAL_PHRASES) as f32;
    let entity =
        cue_hits(&tokens, ENTITY_WORDS) as f32 + 2.0 * phrase_hits(&lower, ENTITY_PHRASES) as f32;

    // Semantic is the always-on base so a cue-free query routes to embedding
    // recall; structured cues bias away from it.
    let semantic = 1.0;

    ViewWeights::normalized(semantic, temporal, causal, entity)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_query_routes_semantic() {
        let w = classify_query("kubernetes deployment strategies");
        assert_eq!(w.primary(), ViewKind::Semantic);
        assert!(w.semantic > 0.9);
    }

    #[test]
    fn temporal_query_routes_temporal() {
        let w = classify_query("when did we first deploy the service and how long until launch");
        assert_eq!(w.primary(), ViewKind::Temporal);
        assert!(w.temporal > w.causal && w.temporal > w.entity);
    }

    #[test]
    fn causal_query_routes_causal() {
        let w = classify_query("why did the deploy fail and what led to the outage");
        assert_eq!(w.primary(), ViewKind::Causal);
        assert!(w.causal > w.temporal);
    }

    #[test]
    fn entity_query_routes_entity() {
        let w = classify_query("who approved the migration");
        assert_eq!(w.primary(), ViewKind::Entity);
    }

    #[test]
    fn phrases_outweigh_single_words() {
        // "how long" (temporal phrase) should dominate a lone entity "what".
        let w = classify_query("what is the timeline and how long did it take");
        assert_eq!(w.primary(), ViewKind::Temporal);
    }

    #[test]
    fn weights_are_normalized() {
        let w = classify_query("why did revenue fall after the launch");
        let sum = w.semantic + w.temporal + w.causal + w.entity;
        assert!((sum - 1.0).abs() < 1e-5, "weights sum to 1, got {sum}");
    }

    #[test]
    fn empty_query_is_semantic() {
        let w = classify_query("");
        assert_eq!(w.primary(), ViewKind::Semantic);
        assert_eq!(w.semantic, 1.0);
    }

    #[test]
    fn substrings_do_not_false_match() {
        // "whenever" / "because" boundaries: whole-word cue matching only.
        let w = classify_query("whichever whenever");
        // "whichever"/"whenever" are not the whole words "which"/"when".
        assert_eq!(w.primary(), ViewKind::Semantic);
    }
}
