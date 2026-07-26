//! Subject-Verb-Object event extraction (Chronos).
//!
//! Regex-based SVO extraction used by the imperative write path
//! ([`write_path`](../../../hirn_engine/db/write_path)) to derive structured
//! events from incoming memory content for temporal queries like
//! "what happened in March?".
//!
//! This was previously fronted by a `SvoExtractionExec` DataFusion operator
//! that was never emitted into any compiled plan (R-20b). The operator was
//! retired; the pure extraction logic (`extract_svo_regex`) lives on because
//! the write path calls it directly.

/// A single extracted SVO event.
#[derive(Debug, Clone)]
pub struct SvoEvent {
    pub subject: String,
    pub verb: String,
    pub object: String,
    pub time_start: Option<String>,
    pub time_end: Option<String>,
    pub location: Option<String>,
    pub confidence: f32,
}

/// Extract SVO events using regex patterns (fallback mode).
///
/// Recognizes common English SVO patterns with optional temporal markers.
pub fn extract_svo_regex(text: &str, confidence_threshold: f32) -> Vec<SvoEvent> {
    let mut events = Vec::new();

    // Simple sentence-splitting heuristic.
    let sentences: Vec<&str> = text
        .split(['.', '!', '?'])
        .filter(|s| s.split_whitespace().count() >= 3)
        .collect();

    for sentence in sentences {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.len() < 3 {
            continue;
        }

        // Basic SVO extraction: first capitalized word as subject,
        // first verb-like word, rest as object.
        let subject = extract_subject(&words);
        let (verb, verb_idx) = extract_verb(&words);
        let object = extract_object(&words, verb_idx);
        let time = extract_temporal(sentence);

        if !subject.is_empty() && !verb.is_empty() && !object.is_empty() {
            let confidence = compute_confidence(&subject, &verb, &object);
            if confidence >= confidence_threshold {
                events.push(SvoEvent {
                    subject,
                    verb,
                    object,
                    time_start: time.clone(),
                    time_end: time,
                    location: None,
                    confidence,
                });
            }
        }
    }

    events
}

/// Extract subject: first capitalized word or proper noun.
fn extract_subject(words: &[&str]) -> String {
    // Skip leading adverbs/prepositions.
    for word in words {
        let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.is_empty() {
            continue;
        }
        // Capitalized word or pronoun.
        if trimmed.chars().next().is_some_and(|c| c.is_uppercase())
            || matches!(
                trimmed.to_lowercase().as_str(),
                "i" | "he" | "she" | "they" | "we" | "it"
            )
        {
            return trimmed.to_string();
        }
        // First non-skip word as subject.
        if !matches!(
            trimmed.to_lowercase().as_str(),
            "the" | "a" | "an" | "on" | "in" | "at" | "then" | "also" | "however"
        ) {
            return trimmed.to_string();
        }
    }
    String::new()
}

/// Extract verb: common action words.
fn extract_verb(words: &[&str]) -> (String, usize) {
    let verb_suffixes = ["ed", "ing", "es", "ied"];
    let common_verbs = [
        "is",
        "was",
        "are",
        "were",
        "has",
        "had",
        "have",
        "will",
        "can",
        "could",
        "should",
        "would",
        "do",
        "does",
        "did",
        "said",
        "went",
        "made",
        "got",
        "took",
        "came",
        "gave",
        "knew",
        "thought",
        "told",
        "found",
        "put",
        "ran",
        "set",
        "met",
        "created",
        "deployed",
        "updated",
        "deleted",
        "sent",
        "bought",
        "sold",
        "moved",
        "started",
        "stopped",
        "finished",
        "completed",
        "began",
        "decided",
        "agreed",
        "mentioned",
        "discussed",
        "scheduled",
        "planned",
        "launched",
        "released",
        "fixed",
        "resolved",
        "discovered",
    ];

    for (i, word) in words.iter().enumerate() {
        let lower = word.to_lowercase();
        let trimmed = lower.trim_matches(|c: char| !c.is_alphanumeric());
        if common_verbs.contains(&trimmed) {
            return (trimmed.to_string(), i);
        }
        for suffix in &verb_suffixes {
            if trimmed.ends_with(suffix) && trimmed.len() > suffix.len() + 1 {
                return (trimmed.to_string(), i);
            }
        }
    }
    (String::new(), 0)
}

/// Extract object: words after the verb.
fn extract_object(words: &[&str], verb_idx: usize) -> String {
    if verb_idx + 1 >= words.len() {
        return String::new();
    }
    words[verb_idx + 1..]
        .iter()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '-'))
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Extract temporal markers from text.
fn extract_temporal(text: &str) -> Option<String> {
    let lower = text.to_lowercase();

    // Month patterns.
    let months = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    for month in &months {
        if lower.contains(month) {
            // Try to find "Month Day" or "Month Day, Year" pattern.
            if let Some(pos) = lower.find(month) {
                // `pos` is a byte offset into `lower`, not `text`
                // (`to_lowercase()` can change byte lengths, e.g. U+212A→'k'), so
                // slicing `text` could land mid-code-point and panic on the write
                // path. Slice `lower` at the known-good boundary `pos` and take a
                // char-bounded window instead of a raw byte range.
                let after: String = lower[pos..].chars().take(month.len() + 15).collect();
                return Some(after.trim().to_string());
            }
        }
    }

    // Date patterns: YYYY-MM-DD.
    for word in lower.split_whitespace() {
        if word.len() >= 8 && word.chars().filter(|c| *c == '-').count() == 2 {
            let parts: Vec<&str> = word.split('-').collect();
            if parts.len() == 3
                && parts[0].len() == 4
                && parts[0].chars().all(|c| c.is_ascii_digit())
            {
                return Some(word.to_string());
            }
        }
    }

    // Relative time patterns.
    let relative = [
        "yesterday",
        "today",
        "last week",
        "last month",
        "this morning",
    ];
    for pattern in &relative {
        if lower.contains(pattern) {
            return Some(pattern.to_string());
        }
    }

    None
}

/// Compute confidence based on extraction quality.
fn compute_confidence(subject: &str, verb: &str, object: &str) -> f32 {
    let mut score: f32 = 0.6; // base confidence for regex extraction

    // Boost for proper nouns (capitalized subject).
    if subject.chars().next().is_some_and(|c| c.is_uppercase()) {
        score += 0.1;
    }

    // Boost for recognized verbs.
    if verb.len() > 2 {
        score += 0.1;
    }

    // Boost for longer objects (more specific).
    if object.split_whitespace().count() >= 2 {
        score += 0.1;
    }

    score.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_svo_alice_deployed() {
        let events = extract_svo_regex("Alice deployed the new release on March 15th.", 0.5);
        assert!(!events.is_empty());
        let e = &events[0];
        assert_eq!(e.subject, "Alice");
        assert_eq!(e.verb, "deployed");
        assert!(e.object.contains("release") || e.object.contains("new"));
        assert!(e.time_start.is_some());
    }

    #[test]
    fn extract_svo_no_temporal() {
        let events = extract_svo_regex("The cat sat on the mat.", 0.5);
        // May or may not extract depending on patterns.
        for e in &events {
            assert!(e.time_start.is_none());
        }
    }

    #[test]
    fn extract_svo_empty_text() {
        let events = extract_svo_regex("", 0.5);
        assert!(events.is_empty());
    }

    #[test]
    fn extract_svo_too_short() {
        let events = extract_svo_regex("Hi.", 0.5);
        assert!(events.is_empty());
    }

    #[test]
    fn extract_svo_multiple_sentences() {
        let events = extract_svo_regex(
            "Alice deployed the release on March 15th. Bob fixed the login bug yesterday.",
            0.5,
        );
        assert!(!events.is_empty());
    }

    #[test]
    fn temporal_extraction_iso_date() {
        let t = extract_temporal("Meeting on 2026-03-15 at noon.");
        assert!(t.is_some());
        assert!(t.unwrap().contains("2026-03-15"));
    }

    #[test]
    fn temporal_extraction_month_name() {
        let t = extract_temporal("The event happened in March 2026.");
        assert!(t.is_some());
    }

    #[test]
    fn temporal_extraction_relative() {
        let t = extract_temporal("I saw this yesterday at the park.");
        assert!(t.is_some());
        assert_eq!(t.unwrap(), "yesterday");
    }
}
