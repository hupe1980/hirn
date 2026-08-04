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
//! # Routing is a model decision, not a word list
//!
//! Intent is a semantic property. "How much time passed between the two
//! releases?" is temporal with no temporal cue word in it; "what triggered the
//! regression?" is causal though "what" is an entity cue; "wann haben wir
//! deployed?" is temporal in a language no English cue list covers. Routing is
//! therefore decided by [`QUERY_INTENT_TASK`] through the configured
//! [`HybridClassifier`] — structured LLM first, embedding exemplar router
//! second.
//!
//! [`classify_query_heuristic`] remains as the provider-free floor: whole-word
//! and token-sequence cue matching, good enough that a deployment with no
//! provider still routes recognizable English questions, and never the primary
//! decision surface. Which path decided is reported on
//! [`QueryRoute::source`] and counted in `hirn_nlu_decisions_total`.

use hirn_core::nlu::{Classification, ClassificationTask, DecisionSource, LabelSpec};
use hirn_provider::HybridClassifier;

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
    /// Stable machine-readable label. Matches the task's label names, so a
    /// classification result maps back to a view without a lookup table.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Temporal => "temporal",
            Self::Causal => "causal",
            Self::Entity => "entity",
        }
    }

    /// Parse a classifier label back into a view.
    #[must_use]
    pub fn parse(label: &str) -> Option<Self> {
        match label {
            "semantic" => Some(Self::Semantic),
            "temporal" => Some(Self::Temporal),
            "causal" => Some(Self::Causal),
            "entity" => Some(Self::Entity),
            _ => None,
        }
    }
}

/// The routing decision surface.
///
/// Exemplars are chosen to cover exactly what a cue list cannot: implicit
/// intent with no cue word, cue words that point at the wrong view, passive
/// voice, and non-English phrasing.
pub const QUERY_INTENT_TASK: ClassificationTask = ClassificationTask {
    name: "query_intent",
    instruction: "Decide which memory view best answers a user's question about their own \
                  stored memories. Judge what the question is really asking for, not which \
                  words it happens to contain: a question can ask about time without using a \
                  time word, and a question starting with \"what\" can be asking for a cause. \
                  Questions in any language are classified the same way.",
    labels: &[
        LabelSpec {
            name: "semantic",
            description: "General recall of content or topics; no ordering, cause, or specific \
                          entity is being asked for. This is the default when nothing more \
                          specific fits.",
            exemplars: &[
                "what did we discuss about kubernetes deployments",
                "remind me what I decided about the caching strategy",
                "anything about onboarding documentation",
            ],
        },
        LabelSpec {
            name: "temporal",
            description: "Asks about when something happened, the order of events, durations, \
                          or how things changed over time.",
            exemplars: &[
                "when did we first deploy the service",
                "how much time passed between the two releases",
                "walk me through what happened leading up to the outage",
                "which came first, the migration or the incident",
            ],
        },
        LabelSpec {
            name: "causal",
            description: "Asks why something happened, what brought it about, or what followed \
                          from it.",
            exemplars: &[
                "why did the deploy fail",
                "what triggered the latency regression",
                "what would have happened if we had not rolled back",
                "the outage — what was behind it",
            ],
        },
        LabelSpec {
            name: "entity",
            description: "Asks for a specific person, place, thing, or value — a factoid \
                          lookup rather than a topic.",
            exemplars: &[
                "who approved the migration",
                "which database are we using for the event log",
                "where is the staging cluster hosted",
                "the person who wrote the auth middleware",
            ],
        },
    ],
    default_label: "semantic",
};

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

    /// Weight for one view.
    #[must_use]
    pub fn weight_for(&self, view: ViewKind) -> f32 {
        match view {
            ViewKind::Semantic => self.semantic,
            ViewKind::Temporal => self.temporal,
            ViewKind::Causal => self.causal,
            ViewKind::Entity => self.entity,
        }
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

    /// Build weights from a classification decision.
    ///
    /// A backend that produces a full distribution (the embedding router)
    /// supplies all four weights directly. A point decision (the LLM) gives
    /// the chosen view its calibrated confidence and assigns the remaining
    /// mass to `semantic` — the general-purpose view is the right place for
    /// residual uncertainty, since it is what a wrong structured route would
    /// have wanted anyway. Residual mass on a `semantic` decision is spread
    /// evenly over the structured views.
    #[must_use]
    pub fn from_classification(decision: &Classification) -> Self {
        let scored = |view: ViewKind| decision.score_for(view.label());
        let distribution = [
            scored(ViewKind::Semantic),
            scored(ViewKind::Temporal),
            scored(ViewKind::Causal),
            scored(ViewKind::Entity),
        ];
        // A full distribution means every view was scored.
        if distribution.iter().filter(|w| **w > 0.0).count() > 1 {
            return Self::normalized(
                distribution[0],
                distribution[1],
                distribution[2],
                distribution[3],
            );
        }

        let chosen = ViewKind::parse(&decision.label).unwrap_or(ViewKind::Semantic);
        let confidence = decision.confidence.clamp(0.0, 1.0);
        let residual = 1.0 - confidence;
        let mut weights = Self {
            semantic: 0.0,
            temporal: 0.0,
            causal: 0.0,
            entity: 0.0,
        };
        match chosen {
            ViewKind::Semantic => {
                weights.semantic = confidence;
                weights.temporal = residual / 3.0;
                weights.causal = residual / 3.0;
                weights.entity = residual / 3.0;
            }
            ViewKind::Temporal => {
                weights.temporal = confidence;
                weights.semantic = residual;
            }
            ViewKind::Causal => {
                weights.causal = confidence;
                weights.semantic = residual;
            }
            ViewKind::Entity => {
                weights.entity = confidence;
                weights.semantic = residual;
            }
        }
        Self::normalized(
            weights.semantic,
            weights.temporal,
            weights.causal,
            weights.entity,
        )
    }
}

/// A routing decision plus its provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QueryRoute {
    pub weights: ViewWeights,
    pub primary: ViewKind,
    /// Which backend decided — `Heuristic` means the cue fallback ran.
    pub source: DecisionSource,
    /// Calibrated confidence in the primary view.
    pub confidence: f32,
}

/// Route a query to a memory view.
///
/// Runs the configured classifier chain and falls back to
/// [`classify_query_heuristic`] when every backend abstains. Never fails: a
/// query is always routed somewhere, because the semantic view answers
/// anything.
pub async fn route_query(classifier: &HybridClassifier, query: &str) -> QueryRoute {
    let decision = classifier
        .decide(&QUERY_INTENT_TASK, query, None, || {
            heuristic_classification(query)
        })
        .await;

    let weights = ViewWeights::from_classification(&decision);
    QueryRoute {
        weights,
        primary: weights.primary(),
        source: decision.source,
        confidence: decision.confidence,
    }
}

/// The deterministic fallback expressed as a [`Classification`].
fn heuristic_classification(query: &str) -> Classification {
    let weights = classify_query_heuristic(query);
    let primary = weights.primary();
    Classification::new(
        primary.label(),
        weights.weight_for(primary),
        DecisionSource::Heuristic,
        Some("cue fallback: no model-backed backend produced a decision".to_owned()),
    )
    .with_scores(vec![
        ("semantic".to_owned(), weights.semantic),
        ("temporal".to_owned(), weights.temporal),
        ("causal".to_owned(), weights.causal),
        ("entity".to_owned(), weights.entity),
    ])
}

// ── Deterministic fallback ───────────────────────────────────────────────

/// Count how many of `cues` appear as whole words in `tokens`.
fn cue_hits(tokens: &[String], cues: &[&str]) -> u32 {
    tokens.iter().filter(|t| cues.contains(&t.as_str())).count() as u32
}

/// Count multi-word cues that appear as a contiguous **token sequence**.
///
/// Token-sequence matching rather than substring search: `lower.contains()`
/// counted "how long" inside "show longer results" and "what if" inside
/// "somewhat iffy", turning incidental character overlap into evidence of
/// semantic intent.
fn phrase_hits(tokens: &[String], phrases: &[&str]) -> u32 {
    phrases
        .iter()
        .filter(|phrase| {
            let words: Vec<&str> = phrase.split_whitespace().collect();
            !words.is_empty()
                && tokens
                    .windows(words.len())
                    .any(|window| window.iter().zip(&words).all(|(t, w)| t == w))
        })
        .count() as u32
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
/// Temporal multi-word cues.
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
/// Causal multi-word cues.
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
/// Entity multi-word cues.
const ENTITY_PHRASES: &[&str] = &["who is", "who was", "what is", "which one", "how many"];

/// Provider-free cue routing: the floor beneath [`route_query`].
///
/// A base semantic weight is always present (embedding recall answers
/// anything); temporal / causal / entity weights accrue from whole-word and
/// token-sequence cue hits. The result is normalized so the weights sum to
/// ~1.0.
///
/// This path is English-only and blind to paraphrase, implicit intent, and
/// scope by construction. It exists so hirn keeps routing with no provider
/// configured — expanding these lists is not how routing quality improves;
/// configuring a classifier backend is.
#[must_use]
pub fn classify_query_heuristic(query: &str) -> ViewWeights {
    let tokens: Vec<String> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(str::to_lowercase)
        .collect();

    // Multi-word cues weigh double a single-word cue (stronger signal).
    let temporal = cue_hits(&tokens, TEMPORAL_WORDS) as f32
        + 2.0 * phrase_hits(&tokens, TEMPORAL_PHRASES) as f32;
    let causal =
        cue_hits(&tokens, CAUSAL_WORDS) as f32 + 2.0 * phrase_hits(&tokens, CAUSAL_PHRASES) as f32;
    let entity =
        cue_hits(&tokens, ENTITY_WORDS) as f32 + 2.0 * phrase_hits(&tokens, ENTITY_PHRASES) as f32;

    // Semantic is the always-on base so a cue-free query routes to embedding
    // recall; structured cues bias away from it.
    let semantic = 1.0;

    ViewWeights::normalized(semantic, temporal, causal, entity)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use async_trait::async_trait;
    use hirn_core::HirnResult;
    use hirn_core::nlu::{NluBudget, TextClassifier};

    use super::*;

    // ── Task definition ──────────────────────────────────────────────────

    #[test]
    fn task_is_well_formed_and_maps_onto_views() {
        assert!(QUERY_INTENT_TASK.is_well_formed());
        for label in QUERY_INTENT_TASK.labels {
            assert!(
                ViewKind::parse(label.name).is_some(),
                "label {} has no view",
                label.name
            );
        }
        for view in [
            ViewKind::Semantic,
            ViewKind::Temporal,
            ViewKind::Causal,
            ViewKind::Entity,
        ] {
            assert!(QUERY_INTENT_TASK.contains(view.label()));
        }
    }

    // ── Model-backed routing ─────────────────────────────────────────────

    struct StubClassifier {
        label: &'static str,
        confidence: f32,
        scores: Option<Vec<(String, f32)>>,
    }

    #[async_trait]
    impl TextClassifier for StubClassifier {
        async fn classify(
            &self,
            _task: &ClassificationTask,
            _text: &str,
            _context: Option<&str>,
            _budget: &NluBudget,
        ) -> HirnResult<Option<Classification>> {
            let decision =
                Classification::new(self.label, self.confidence, DecisionSource::Model, None);
            Ok(Some(match self.scores.clone() {
                Some(scores) => decision.with_scores(scores),
                None => decision,
            }))
        }

        fn backend_id(&self) -> &str {
            "stub"
        }

        fn source(&self) -> DecisionSource {
            DecisionSource::Model
        }
    }

    fn model_chain(label: &'static str, confidence: f32) -> HybridClassifier {
        HybridClassifier::new().with_backend(Arc::new(StubClassifier {
            label,
            confidence,
            scores: None,
        }))
    }

    #[tokio::test]
    async fn model_routes_implicit_temporal_intent_with_no_cue_word() {
        // No cue word appears — the fallback would route this to semantic.
        let query = "how much elapsed between the two releases";
        assert_eq!(
            classify_query_heuristic(query).primary(),
            ViewKind::Semantic,
            "precondition: the cue fallback cannot see this intent"
        );

        let route = route_query(&model_chain("temporal", 0.9), query).await;
        assert_eq!(route.primary, ViewKind::Temporal);
        assert_eq!(route.source, DecisionSource::Model);
    }

    #[tokio::test]
    async fn model_overrides_a_misleading_cue_word() {
        // "what" is an entity cue, but the question asks for a cause.
        let query = "what set off the latency regression";
        assert_eq!(classify_query_heuristic(query).primary(), ViewKind::Entity);

        let route = route_query(&model_chain("causal", 0.88), query).await;
        assert_eq!(route.primary, ViewKind::Causal);
    }

    #[tokio::test]
    async fn model_routes_non_english_queries() {
        // German: no English cue word, so the fallback routes semantic.
        let query = "wann haben wir den dienst zum ersten mal ausgeliefert";
        assert_eq!(
            classify_query_heuristic(query).primary(),
            ViewKind::Semantic
        );

        let route = route_query(&model_chain("temporal", 0.85), query).await;
        assert_eq!(route.primary, ViewKind::Temporal);
        assert_eq!(route.source, DecisionSource::Model);
    }

    #[tokio::test]
    async fn model_routes_passive_voice() {
        let query = "the migration was approved by whom";
        let route = route_query(&model_chain("entity", 0.8), query).await;
        assert_eq!(route.primary, ViewKind::Entity);
    }

    #[tokio::test]
    async fn no_provider_falls_back_to_cue_routing() {
        let route = route_query(&HybridClassifier::new(), "why did the deploy fail").await;
        assert_eq!(route.primary, ViewKind::Causal);
        assert_eq!(route.source, DecisionSource::Heuristic);
    }

    #[tokio::test]
    async fn low_confidence_model_decision_falls_back() {
        // Below the default 0.55 gate: the chain must not act on it.
        let route = route_query(&model_chain("causal", 0.2), "when did we ship").await;
        assert_eq!(route.source, DecisionSource::Heuristic);
        assert_eq!(route.primary, ViewKind::Temporal);
    }

    #[tokio::test]
    async fn distribution_backends_supply_all_four_weights() {
        let chain = HybridClassifier::new().with_backend(Arc::new(StubClassifier {
            label: "temporal",
            confidence: 0.6,
            scores: Some(vec![
                ("semantic".into(), 0.1),
                ("temporal".into(), 0.6),
                ("causal".into(), 0.2),
                ("entity".into(), 0.1),
            ]),
        }));
        let route = route_query(&chain, "when did it happen").await;
        assert_eq!(route.primary, ViewKind::Temporal);
        assert!((route.weights.causal - 0.2).abs() < 1e-5);
        let sum = route.weights.semantic
            + route.weights.temporal
            + route.weights.causal
            + route.weights.entity;
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn point_decisions_assign_residual_mass_to_semantic() {
        let decision = Classification::new("causal", 0.7, DecisionSource::Model, None);
        let weights = ViewWeights::from_classification(&decision);
        assert!((weights.causal - 0.7).abs() < 1e-5);
        assert!((weights.semantic - 0.3).abs() < 1e-5);
        assert_eq!(weights.primary(), ViewKind::Causal);

        // A confident semantic decision spreads its residual over the
        // structured views instead of doubling back onto itself.
        let semantic = Classification::new("semantic", 0.7, DecisionSource::Model, None);
        let weights = ViewWeights::from_classification(&semantic);
        assert!((weights.semantic - 0.7).abs() < 1e-5);
        assert!((weights.temporal - 0.1).abs() < 1e-5);
        assert_eq!(weights.primary(), ViewKind::Semantic);
    }

    #[test]
    fn unknown_label_degrades_to_semantic() {
        let decision = Classification::new("nonsense", 0.9, DecisionSource::Model, None);
        let weights = ViewWeights::from_classification(&decision);
        assert_eq!(weights.primary(), ViewKind::Semantic);
    }

    // ── Deterministic fallback ───────────────────────────────────────────

    #[test]
    fn plain_query_routes_semantic() {
        let w = classify_query_heuristic("kubernetes deployment strategies");
        assert_eq!(w.primary(), ViewKind::Semantic);
        assert!(w.semantic > 0.9);
    }

    #[test]
    fn temporal_query_routes_temporal() {
        let w = classify_query_heuristic(
            "when did we first deploy the service and how long until launch",
        );
        assert_eq!(w.primary(), ViewKind::Temporal);
        assert!(w.temporal > w.causal && w.temporal > w.entity);
    }

    #[test]
    fn causal_query_routes_causal() {
        let w = classify_query_heuristic("why did the deploy fail and what led to the outage");
        assert_eq!(w.primary(), ViewKind::Causal);
        assert!(w.causal > w.temporal);
    }

    #[test]
    fn entity_query_routes_entity() {
        let w = classify_query_heuristic("who approved the migration");
        assert_eq!(w.primary(), ViewKind::Entity);
    }

    #[test]
    fn phrases_outweigh_single_words() {
        // "how long" (temporal phrase) should dominate a lone entity "what".
        let w = classify_query_heuristic("what is the timeline and how long did it take");
        assert_eq!(w.primary(), ViewKind::Temporal);
    }

    #[test]
    fn weights_are_normalized() {
        let w = classify_query_heuristic("why did revenue fall after the launch");
        let sum = w.semantic + w.temporal + w.causal + w.entity;
        assert!((sum - 1.0).abs() < 1e-5, "weights sum to 1, got {sum}");
    }

    #[test]
    fn empty_query_is_semantic() {
        let w = classify_query_heuristic("");
        assert_eq!(w.primary(), ViewKind::Semantic);
        assert_eq!(w.semantic, 1.0);
    }

    #[test]
    fn substrings_do_not_false_match() {
        // "whenever" / "because" boundaries: whole-word cue matching only.
        let w = classify_query_heuristic("whichever whenever");
        assert_eq!(w.primary(), ViewKind::Semantic);
    }

    #[test]
    fn multi_word_cues_match_tokens_not_substrings() {
        // "show longer" contains the characters of "how long"; token-sequence
        // matching must not count that as temporal evidence.
        let spurious = classify_query_heuristic("show longer output");
        assert_eq!(spurious.primary(), ViewKind::Semantic);
        assert_eq!(spurious.temporal, 0.0);

        // The real phrase still counts.
        let genuine = classify_query_heuristic("how long did the rollout take");
        assert!(genuine.temporal > 0.0);
    }

    #[test]
    fn phrase_matching_is_punctuation_insensitive() {
        // Tokenization drops punctuation, so a comma inside a cue phrase does
        // not hide it.
        let w = classify_query_heuristic("why, did the deploy fail");
        assert_eq!(w.primary(), ViewKind::Causal);
    }
}
