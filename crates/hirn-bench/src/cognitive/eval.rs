//! Scoring functions for cognitive benchmark evaluation.
//!
//! Two primary metrics:
//! - **Containment**: Is any expected answer a substring of the retrieved context?
//! - **Token F1**: Token-level overlap between expected answer and context.

/// Graduated containment: checks whether expected answers appear in the context.
///
/// Returns 1.0 for exact substring match (case-insensitive). Otherwise falls
/// back to word-level overlap: the fraction of answer words found in the context.
/// When multiple expected answers are provided, the best (max) score is returned.
pub fn containment(context: &str, expected_answers: &[String]) -> f64 {
    let ctx_lower = context.to_lowercase();
    let ctx_words: Vec<&str> = tokenize(&ctx_lower);
    let mut best = 0.0_f64;
    for answer in expected_answers {
        let ans_lower = answer.to_lowercase();
        // Exact substring match → 1.0
        if ctx_lower.contains(&ans_lower) {
            return 1.0;
        }
        // Word-level overlap: fraction of answer words found in context.
        let ans_words: Vec<&str> = tokenize(&ans_lower);
        if ans_words.is_empty() {
            continue;
        }
        let hits = ans_words.iter().filter(|w| ctx_words.contains(w)).count();
        let score = hits as f64 / ans_words.len() as f64;
        best = best.max(score);
    }
    best
}

/// Compute token-level F1 between the best-matching expected answer and the context.
///
/// Tokenizes by whitespace, computes precision (answer tokens found in context)
/// and recall (answer tokens covered), then returns the harmonic mean.
/// Returns the best F1 across all expected answers.
pub fn token_f1(context: &str, expected_answers: &[String]) -> f64 {
    let ctx_tokens: Vec<&str> = tokenize(context);
    let mut best = 0.0_f64;
    for answer in expected_answers {
        let ans_tokens: Vec<&str> = tokenize(answer);
        if ans_tokens.is_empty() {
            continue;
        }
        let hits = ans_tokens.iter().filter(|t| ctx_tokens.contains(t)).count();
        let precision = if ans_tokens.is_empty() {
            0.0
        } else {
            hits as f64 / ans_tokens.len() as f64
        };
        let recall = if ctx_tokens.is_empty() {
            0.0
        } else {
            hits as f64 / ctx_tokens.len() as f64
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        best = best.max(f1);
    }
    best
}

/// Simple whitespace tokenizer with lowercasing and punctuation stripping.
fn tokenize(text: &str) -> Vec<&str> {
    text.split_whitespace()
        .map(|t| t.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|t| !t.is_empty())
        .collect()
}

/// Compute MRR from a ranked list of recall results.
///
/// Each result is checked against expected answers. The reciprocal rank of the
/// first matching result is returned (1-indexed). Returns 0.0 if no match found.
pub fn mrr(results: &[String], expected_answers: &[String]) -> f64 {
    for (i, content) in results.iter().enumerate() {
        let lower = content.to_lowercase();
        if expected_answers
            .iter()
            .any(|a| lower.contains(&a.to_lowercase()))
        {
            return 1.0 / (i as f64 + 1.0);
        }
    }
    0.0
}

/// Compute nDCG@K from a ranked list of recall results.
///
/// Binary relevance: a result is relevant if it contains any expected answer.
pub fn ndcg_at_k(results: &[String], expected_answers: &[String], k: usize) -> f64 {
    let top_k: Vec<&String> = results.iter().take(k).collect();
    if top_k.is_empty() || expected_answers.is_empty() {
        return 0.0;
    }

    // Compute DCG
    let dcg: f64 = top_k
        .iter()
        .enumerate()
        .map(|(i, content)| {
            let lower = content.to_lowercase();
            let rel = if expected_answers
                .iter()
                .any(|a| lower.contains(&a.to_lowercase()))
            {
                1.0
            } else {
                0.0
            };
            rel / (i as f64 + 2.0_f64).log2()
        })
        .sum();

    // Count how many actually match (for ideal DCG)
    let n_relevant = top_k
        .iter()
        .filter(|content| {
            let lower = content.to_lowercase();
            expected_answers
                .iter()
                .any(|a| lower.contains(&a.to_lowercase()))
        })
        .count();

    let ideal_k = k.min(n_relevant);
    if ideal_k == 0 {
        return 0.0;
    }
    let ideal_dcg: f64 = (0..ideal_k)
        .map(|i| 1.0 / (i as f64 + 2.0_f64).log2())
        .sum();

    if ideal_dcg == 0.0 {
        return 0.0;
    }
    dcg / ideal_dcg
}

/// Check if context contains any of the forbidden answers (for negative queries).
///
/// Returns `true` if any forbidden answer is found (a false positive).
pub fn has_false_positive(context: &str, forbidden_answers: &[String]) -> bool {
    let ctx_lower = context.to_lowercase();
    forbidden_answers
        .iter()
        .any(|a| ctx_lower.contains(&a.to_lowercase()))
}

/// Cosine similarity between two embedding vectors (F-40).
///
/// Returns a value in [-1, 1]. Returns 0.0 when either vector is zero-length
/// or has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let x = *x as f64;
        let y = *y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// Compute semantic similarity between a query embedding and a set of
/// expected-answer embeddings (F-40).
///
/// Returns the maximum cosine similarity across all expected answers. This
/// captures cases where word-overlap metrics (containment, token_f1) miss
/// paraphrased or synonym-rich matches.
pub fn semantic_similarity(
    context_embedding: Option<&[f32]>,
    expected_embeddings: &[Vec<f32>],
) -> f64 {
    let ctx = match context_embedding {
        Some(e) if !e.is_empty() => e,
        _ => return 0.0,
    };
    expected_embeddings
        .iter()
        .map(|e| cosine_similarity(ctx, e))
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn containment_exact() {
        assert_eq!(
            containment("The deadline is March 15th", &["March 15th".to_string()]),
            1.0
        );
    }

    #[test]
    fn containment_case_insensitive() {
        assert_eq!(
            containment("the DEADLINE is march 15th", &["March 15th".to_string()]),
            1.0
        );
    }

    #[test]
    fn containment_not_found() {
        assert_eq!(
            containment("The deadline is March 15th", &["April 1st".to_string()]),
            0.0
        );
    }

    #[test]
    fn containment_partial_word_overlap() {
        // "PostgreSQL 16" not an exact substring of "PostgreSQL version 16"
        // but 2/2 answer words found in context → 1.0 word overlap
        let score = containment("PostgreSQL version 16", &["PostgreSQL 16".to_string()]);
        assert!(score > 0.9, "expected high word overlap, got {score}");
    }

    #[test]
    fn containment_low_word_overlap() {
        // Only 1 out of 3 answer words found
        let score = containment(
            "The project deadline is tomorrow",
            &["server migration deadline".to_string()],
        );
        assert!(score > 0.0 && score < 0.5, "got {score}");
    }

    #[test]
    fn containment_multiple_answers() {
        assert_eq!(
            containment(
                "The deadline is March 15th",
                &["April 1st".to_string(), "March 15th".to_string()]
            ),
            1.0
        );
    }

    #[test]
    fn token_f1_perfect() {
        // "March 15th" (2 tokens) inside "the deadline is March 15th" (5 tokens).
        // precision = 2/2 = 1.0 (all answer tokens found in context)
        // recall = 2/5 = 0.4 (fraction of context tokens that are answer tokens)
        // F1 = 2 * 1.0 * 0.4 / 1.4 ≈ 0.571
        let f1 = token_f1("the deadline is March 15th", &["March 15th".to_string()]);
        assert!(f1 > 0.5 && f1 < 0.6, "expected F1 ≈ 0.57, got {f1}");
    }

    #[test]
    fn token_f1_exact_match() {
        // When context == answer, precision = recall = 1.0, F1 = 1.0.
        let f1 = token_f1("March 15th", &["March 15th".to_string()]);
        assert!((f1 - 1.0).abs() < 1e-6, "expected F1 = 1.0, got {f1}");
    }

    #[test]
    fn token_f1_partial() {
        let f1 = token_f1(
            "project deadline and release",
            &["March 15th deadline".to_string()],
        );
        // "deadline" is a hit, "March" and "15th" are not
        assert!(f1 > 0.0 && f1 < 1.0, "expected partial F1, got {f1}");
    }

    #[test]
    fn token_f1_no_match() {
        let f1 = token_f1("completely unrelated content", &["March 15th".to_string()]);
        assert_eq!(f1, 0.0);
    }

    #[test]
    fn token_f1_empty() {
        assert_eq!(token_f1("some context", &[]), 0.0);
        assert_eq!(token_f1("", &["answer".to_string()]), 0.0);
    }

    #[test]
    fn mrr_first_result() {
        let results = vec!["relevant content here".into(), "other stuff".into()];
        let answers = vec!["relevant".to_string()];
        assert!((mrr(&results, &answers) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_second_result() {
        let results = vec!["unrelated stuff".into(), "relevant content".into()];
        let answers = vec!["relevant".to_string()];
        assert!((mrr(&results, &answers) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn mrr_no_match() {
        let results = vec!["unrelated".into(), "nothing".into()];
        let answers = vec!["missing".to_string()];
        assert!((mrr(&results, &answers)).abs() < f64::EPSILON);
    }

    #[test]
    fn ndcg_at_k_perfect() {
        let results = vec!["relevant a".into(), "relevant b".into()];
        let answers = vec!["relevant".to_string()];
        let score = ndcg_at_k(&results, &answers, 2);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn ndcg_at_k_empty() {
        let results = vec!["unrelated".into()];
        let answers = vec!["missing".to_string()];
        assert!(ndcg_at_k(&results, &answers, 2).abs() < f64::EPSILON);
    }

    #[test]
    fn false_positive_detected() {
        assert!(has_false_positive(
            "The SSN is 123-45-6789",
            &["123-45-6789".to_string()]
        ));
    }

    #[test]
    fn false_positive_not_detected() {
        assert!(!has_false_positive(
            "Normal meeting info",
            &["123-45-6789".to_string()]
        ));
    }
}
