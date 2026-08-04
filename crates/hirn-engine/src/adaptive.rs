//! Adaptive retrieval strategy (Jeong et al., NAACL 2024).
//!
//! Classifies query complexity and routes to the matching retrieval strategy:
//!
//! | Complexity | Strategy                                    |
//! |------------|---------------------------------------------|
//! | Simple     | Local only (HNSW + spreading activation)    |
//! | Moderate   | Hybrid (local + community global)           |
//! | Complex    | Full pipeline: RAPTOR + community + local   |
//!
//! Two kinds of signal feed the decision, and they are deliberately kept
//! apart:
//!
//! - **Structural** (deterministic): clause counts, `INVOLVING` arity,
//!   `EXPAND GRAPH`, and `FOLLOW CAUSES`. These are facts about the compiled
//!   plan, not inferences about language — a query that expands the graph
//!   three hops needs traversal however it is phrased. They set a *floor* on
//!   the depth and can only raise it.
//! - **Linguistic** (model-backed): whether the question itself is a factoid
//!   lookup, a multi-faceted question, or an analytical/comparative one. This
//!   is decided by [`QUERY_COMPLEXITY_TASK`] through the configured
//!   classifier chain; "which of these two approaches aged better" is
//!   analytical with none of the cue phrases a word list would look for.
//!
//! [`classify_query_heuristic`] is the provider-free floor for the linguistic
//! half: token-count buckets and interrogative cue phrases. It is English-only
//! and blind to paraphrase, so it is the fallback, never the primary.
//!
//! Reference: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented
//!             Large Language Models through Question Complexity"
//!             (Jeong et al., NAACL 2024)

use hirn_core::nlu::{Classification, ClassificationTask, DecisionSource, LabelSpec};
use hirn_provider::HybridClassifier;
use hirn_query::ast::RetrievalMode;

/// Query complexity level determined by the adaptive classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueryComplexity {
    /// Factoid / keyword lookups — vector search is sufficient.
    Simple,
    /// Multi-faceted queries — benefit from both local and global retrieval.
    Moderate,
    /// Analytical / comparative / multi-hop — need the full retrieval pipeline.
    Complex,
}

impl QueryComplexity {
    /// Stable machine-readable label, matching the task's label names.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Simple => "simple",
            Self::Moderate => "moderate",
            Self::Complex => "complex",
        }
    }

    /// Parse a classifier label back into a complexity level.
    #[must_use]
    pub fn parse(label: &str) -> Option<Self> {
        match label {
            "simple" => Some(Self::Simple),
            "moderate" => Some(Self::Moderate),
            "complex" => Some(Self::Complex),
            _ => None,
        }
    }

    /// The retrieval strategy this level calls for.
    #[must_use]
    pub const fn retrieval_mode(self) -> RetrievalMode {
        match self {
            Self::Simple => RetrievalMode::Local,
            Self::Moderate => RetrievalMode::Hybrid,
            Self::Complex => RetrievalMode::Raptor,
        }
    }
}

/// The query-complexity decision surface.
///
/// Exemplars cover the analytical questions that carry no cue phrase and the
/// simple ones that happen to contain one.
pub const QUERY_COMPLEXITY_TASK: ClassificationTask = ClassificationTask {
    name: "query_complexity",
    instruction: "Decide how much retrieval work a question needs. Judge the reasoning the \
                  question demands, not its length or its opening word: a short question can \
                  require comparing several sources, and a long one can be a single lookup.",
    labels: &[
        LabelSpec {
            name: "simple",
            description: "A single fact or definition; one relevant memory answers it.",
            exemplars: &[
                "what is JWT",
                "what port does the gateway listen on",
                "who is on call this week",
            ],
        },
        LabelSpec {
            name: "moderate",
            description: "Needs several related memories pulled together, but no comparison \
                          or multi-step reasoning.",
            exemplars: &[
                "how does authentication work with OAuth tokens",
                "what have we tried for the flaky integration test",
                "describe the current deployment process",
            ],
        },
        LabelSpec {
            name: "complex",
            description: "Analytical, comparative, or multi-hop: needs evidence from many \
                          places reasoned over together.",
            exemplars: &[
                "compare the trade-offs between JWT and session-based authentication",
                "which of the two caching approaches aged better and why",
                "summarize everything that shaped the storage rewrite",
            ],
        },
    ],
    default_label: "simple",
};

/// Structural facts about a compiled query that set a floor on retrieval depth.
///
/// These come from the plan, not from the question's wording, so they stay
/// deterministic: a query that asks to expand the graph or follow causes needs
/// the machinery to do so regardless of how a model reads the text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuralSignals {
    pub involving_count: usize,
    pub where_count: usize,
    pub has_temporal: bool,
    pub has_expand: bool,
    pub has_follow_causes: bool,
}

impl StructuralSignals {
    /// The minimum complexity the plan structure alone demands.
    #[must_use]
    pub fn floor(&self) -> QueryComplexity {
        // Graph expansion and causal traversal are not stylistic: the plan
        // cannot execute them under the Local strategy.
        if self.has_expand || self.has_follow_causes || self.involving_count > 2 {
            return QueryComplexity::Complex;
        }
        if self.has_temporal || self.involving_count > 0 || self.where_count > 0 {
            return QueryComplexity::Moderate;
        }
        QueryComplexity::Simple
    }
}

/// Classify a query and return the recommended [`RetrievalMode`].
///
/// The linguistic half is decided by the classifier chain (falling back to
/// [`classify_query_heuristic`]); the structural half sets a floor. The result
/// is the greater of the two, so a model that reads a graph-expanding query as
/// "simple" cannot strand it in a strategy that has no traversal.
pub async fn route_query_complexity(
    classifier: &HybridClassifier,
    query: &str,
    structure: StructuralSignals,
) -> RetrievalMode {
    let decision = classifier
        .decide(&QUERY_COMPLEXITY_TASK, query, None, || {
            let complexity = classify_query_heuristic(
                query,
                structure.involving_count,
                structure.where_count,
                structure.has_temporal,
                structure.has_expand,
                structure.has_follow_causes,
            );
            Classification::new(
                complexity.label(),
                1.0,
                DecisionSource::Heuristic,
                Some("cue fallback: no model-backed backend produced a decision".to_owned()),
            )
        })
        .await;

    let linguistic = QueryComplexity::parse(&decision.label).unwrap_or(QueryComplexity::Simple);
    linguistic.max(structure.floor()).retrieval_mode()
}

/// Provider-free complexity routing.
///
/// Retained so a deployment with no classifier still routes; see
/// [`route_query_complexity`] for the model-backed path.
pub fn classify_and_route(
    query: &str,
    involving_count: usize,
    where_count: usize,
    has_temporal: bool,
    has_expand: bool,
    has_follow_causes: bool,
) -> RetrievalMode {
    classify_query_heuristic(
        query,
        involving_count,
        where_count,
        has_temporal,
        has_expand,
        has_follow_causes,
    )
    .retrieval_mode()
}

/// Provider-free complexity classification from cue phrases and counts.
///
/// English-only and blind to paraphrase by construction — the fallback beneath
/// [`route_query_complexity`], not the primary decision surface.
pub fn classify_query_heuristic(
    query: &str,
    involving_count: usize,
    where_count: usize,
    has_temporal: bool,
    has_expand: bool,
    has_follow_causes: bool,
) -> QueryComplexity {
    let mut score: u32 = 0;

    // Signal 1: Token count (whitespace-split approximation).
    let token_count = query.split_whitespace().count();
    if token_count >= 20 {
        score += 3;
    } else if token_count >= 10 {
        score += 2;
    } else if token_count >= 4 {
        score += 1;
    }

    // Signal 2: Clause count — each additional clause adds complexity.
    score += (where_count as u32).min(3);
    if involving_count > 2 {
        score += 2;
    } else if involving_count > 0 {
        score += 1;
    }

    // Signal 3: Complex question words / analytical patterns.
    let lower = query.to_lowercase();
    let complex_patterns = [
        "compare",
        "contrast",
        "why",
        "how does",
        "what caused",
        "relationship between",
        "difference between",
        "trade-off",
        "pros and cons",
        "implications of",
        "summarize all",
        "overview of",
        "explain the",
        "analyze",
    ];
    let moderate_patterns = [
        "how", "what are", "describe", "list", "when did", "where", "who", "which",
    ];

    // Match phrases on word boundaries, not raw substrings: "how" must not
    // fire inside "shower"/"however", "who" inside "whole", "where" inside
    // "everywhere" — substring hits routed benign queries to heavier
    // retrieval modes.
    let contains_phrase = |phrase: &str| -> bool {
        lower.match_indices(phrase).any(|(start, matched)| {
            let before_ok = start == 0
                || lower[..start]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !c.is_alphanumeric());
            let end = start + matched.len();
            let after_ok = end == lower.len()
                || lower[end..]
                    .chars()
                    .next()
                    .is_none_or(|c| !c.is_alphanumeric());
            before_ok && after_ok
        })
    };

    let complex_hits = complex_patterns
        .iter()
        .filter(|p| contains_phrase(p))
        .count();
    let moderate_hits = moderate_patterns
        .iter()
        .filter(|p| contains_phrase(p))
        .count();

    score += (complex_hits as u32) * 2;
    score += (moderate_hits as u32).min(2);

    // Signal 4: Temporal scope adds moderate complexity.
    if has_temporal {
        score += 2;
    }

    // Signal 5: Expand / follow_causes demand graph traversal.
    if has_expand {
        score += 3;
    }
    if has_follow_causes {
        score += 3;
    }

    // Route based on aggregate score.
    if score >= 6 {
        QueryComplexity::Complex
    } else if score >= 3 {
        QueryComplexity::Moderate
    } else {
        QueryComplexity::Simple
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substring_homonyms_do_not_inflate_complexity() {
        // "shower" contains "how", "whole" contains "who", "nowhere" contains
        // "where" — none may count as an interrogative pattern hit.
        let a = classify_query_heuristic(
            "the shower in the whole nowhere annex",
            0,
            0,
            false,
            false,
            false,
        );
        let b = classify_query_heuristic("the sink in the main annex", 0, 0, false, false, false);
        assert_eq!(a, b, "embedded substrings must not change routing");
    }

    #[test]
    fn simple_factoid_query() {
        let c = classify_query_heuristic("what is JWT", 0, 0, false, false, false);
        assert_eq!(c, QueryComplexity::Simple);
    }

    #[test]
    fn moderate_query_with_entity() {
        let c = classify_query_heuristic(
            "how does authentication work with OAuth tokens",
            1,
            0,
            false,
            false,
            false,
        );
        assert_eq!(c, QueryComplexity::Moderate);
    }

    #[test]
    fn complex_analytical_query() {
        let c = classify_query_heuristic(
            "compare the trade-off between JWT and session-based authentication across all services",
            3,
            1,
            false,
            true,
            false,
        );
        assert_eq!(c, QueryComplexity::Complex);
    }

    #[test]
    fn temporal_adds_complexity() {
        let c =
            classify_query_heuristic("what happened with deployments", 0, 0, true, false, false);
        assert_eq!(c, QueryComplexity::Moderate);
    }

    #[test]
    fn follow_causes_is_complex() {
        let c = classify_query_heuristic("why did the service fail", 0, 0, false, false, true);
        assert_eq!(c, QueryComplexity::Complex);
    }

    #[test]
    fn classify_and_route_simple() {
        let mode = classify_and_route("hello", 0, 0, false, false, false);
        assert_eq!(mode, RetrievalMode::Local);
    }

    #[test]
    fn task_is_well_formed_and_maps_onto_levels() {
        assert!(QUERY_COMPLEXITY_TASK.is_well_formed());
        for label in QUERY_COMPLEXITY_TASK.labels {
            assert!(
                QueryComplexity::parse(label.name).is_some(),
                "{}",
                label.name
            );
        }
    }

    #[test]
    fn structural_floor_ignores_wording() {
        let expand = StructuralSignals {
            has_expand: true,
            ..Default::default()
        };
        assert_eq!(expand.floor(), QueryComplexity::Complex);

        let temporal = StructuralSignals {
            has_temporal: true,
            ..Default::default()
        };
        assert_eq!(temporal.floor(), QueryComplexity::Moderate);

        assert_eq!(
            StructuralSignals::default().floor(),
            QueryComplexity::Simple
        );
    }

    struct StubClassifier(&'static str);

    #[async_trait::async_trait]
    impl hirn_core::nlu::TextClassifier for StubClassifier {
        async fn classify(
            &self,
            _task: &ClassificationTask,
            _text: &str,
            _context: Option<&str>,
            _budget: &hirn_core::nlu::NluBudget,
        ) -> hirn_core::HirnResult<Option<Classification>> {
            Ok(Some(Classification::new(
                self.0,
                0.9,
                DecisionSource::Model,
                None,
            )))
        }

        fn backend_id(&self) -> &str {
            "stub"
        }

        fn source(&self) -> DecisionSource {
            DecisionSource::Model
        }
    }

    fn model_chain(label: &'static str) -> HybridClassifier {
        HybridClassifier::new().with_backend(std::sync::Arc::new(StubClassifier(label)))
    }

    #[tokio::test]
    async fn model_sees_analytical_intent_without_cue_phrases() {
        // No "compare"/"trade-off"/"why": the cue fallback reads this as a
        // simple lookup and would route it to vector search alone.
        let query = "which of the two caching approaches aged better";
        assert_eq!(
            classify_query_heuristic(query, 0, 0, false, false, false),
            QueryComplexity::Simple
        );

        let mode =
            route_query_complexity(&model_chain("complex"), query, StructuralSignals::default())
                .await;
        assert_eq!(mode, RetrievalMode::Raptor);
    }

    #[tokio::test]
    async fn structure_raises_but_never_lowers_the_model_decision() {
        // A model reading a graph-expanding query as trivial must not strand
        // it in a strategy with no traversal.
        let mode = route_query_complexity(
            &model_chain("simple"),
            "anything",
            StructuralSignals {
                has_expand: true,
                ..Default::default()
            },
        )
        .await;
        assert_eq!(mode, RetrievalMode::Raptor);

        // Conversely, plain structure does not cap an analytical question.
        let mode = route_query_complexity(
            &model_chain("complex"),
            "anything",
            StructuralSignals::default(),
        )
        .await;
        assert_eq!(mode, RetrievalMode::Raptor);
    }

    #[tokio::test]
    async fn no_provider_uses_the_cue_fallback() {
        let mode = route_query_complexity(
            &HybridClassifier::new(),
            "compare the trade-off between JWT and session-based authentication across services",
            StructuralSignals {
                involving_count: 3,
                where_count: 1,
                has_expand: true,
                ..Default::default()
            },
        )
        .await;
        assert_eq!(mode, RetrievalMode::Raptor);
    }

    #[test]
    fn classify_and_route_complex() {
        let mode = classify_and_route(
            "compare all authentication strategies and their trade-offs",
            2,
            1,
            true,
            true,
            false,
        );
        assert_eq!(mode, RetrievalMode::Raptor);
    }
}
