//! Adaptive retrieval strategy (Jeong et al., NAACL 2024).
//!
//! Classifies query complexity using lightweight heuristics and routes to
//! the optimal retrieval strategy:
//!
//! | Complexity | Strategy                                    |
//! |------------|---------------------------------------------|
//! | Simple     | Local only (HNSW + spreading activation)    |
//! | Moderate   | Hybrid (local + community global)           |
//! | Complex    | Full pipeline: RAPTOR + community + local   |
//!
//! The classifier uses five orthogonal signals:
//! 1. **Token count** — longer queries tend to be more complex.
//! 2. **Clause count** — more WHERE/INVOLVING/TEMPORAL clauses = more complex.
//! 3. **Question words** — "why", "how", "compare" suggest analytical queries.
//! 4. **Entity count** — multi-entity queries benefit from graph traversal.
//! 5. **Temporal scope** — temporal constraints suggest moderate complexity.
//!
//! Reference: "Adaptive-RAG: Learning to Adapt Retrieval-Augmented
//!             Large Language Models through Question Complexity"
//!             (Jeong et al., NAACL 2024)

use hirn_query::ast::RetrievalMode;

/// Query complexity level determined by the adaptive classifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryComplexity {
    /// Factoid / keyword lookups — vector search is sufficient.
    Simple,
    /// Multi-faceted queries — benefit from both local and global retrieval.
    Moderate,
    /// Analytical / comparative / multi-hop — need the full retrieval pipeline.
    Complex,
}

/// Classify query complexity and return the recommended `RetrievalMode`.
///
/// This is a deterministic, rule-based classifier inspired by Adaptive-RAG.
/// It avoids the cost of an LLM call for routing while still achieving good
/// strategy selection for most queries.
pub fn classify_and_route(
    query: &str,
    involving_count: usize,
    where_count: usize,
    has_temporal: bool,
    has_expand: bool,
    has_follow_causes: bool,
) -> RetrievalMode {
    let complexity = classify_query(
        query,
        involving_count,
        where_count,
        has_temporal,
        has_expand,
        has_follow_causes,
    );

    match complexity {
        QueryComplexity::Simple => RetrievalMode::Local,
        QueryComplexity::Moderate => RetrievalMode::Hybrid,
        QueryComplexity::Complex => RetrievalMode::Raptor,
    }
}

/// Classify query complexity into Simple / Moderate / Complex.
pub fn classify_query(
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
        let a = classify_query(
            "the shower in the whole nowhere annex",
            0,
            0,
            false,
            false,
            false,
        );
        let b = classify_query("the sink in the main annex", 0, 0, false, false, false);
        assert_eq!(a, b, "embedded substrings must not change routing");
    }

    #[test]
    fn simple_factoid_query() {
        let c = classify_query("what is JWT", 0, 0, false, false, false);
        assert_eq!(c, QueryComplexity::Simple);
    }

    #[test]
    fn moderate_query_with_entity() {
        let c = classify_query(
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
        let c = classify_query(
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
        let c = classify_query("what happened with deployments", 0, 0, true, false, false);
        assert_eq!(c, QueryComplexity::Moderate);
    }

    #[test]
    fn follow_causes_is_complex() {
        let c = classify_query("why did the service fail", 0, 0, false, false, true);
        assert_eq!(c, QueryComplexity::Complex);
    }

    #[test]
    fn classify_and_route_simple() {
        let mode = classify_and_route("hello", 0, 0, false, false, false);
        assert_eq!(mode, RetrievalMode::Local);
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
