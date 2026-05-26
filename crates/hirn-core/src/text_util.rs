//! Lightweight text utilities shared across crates.

/// Truncate `s` to at most `max_len` **characters** at a word boundary.
///
/// The returned string is always ≤ `max_len` characters (N-M17 fix: the
/// ellipsis is counted against `max_len`, so the usable content is truncated
/// to at most `max_len - 3` characters before appending `...`).
///
/// If the string is ≤ `max_len` characters it is returned unchanged.
/// `max_len < 3` is treated as 3 so the ellipsis always fits.
///
/// Safe for multi-byte UTF-8 — truncation is character-based, not byte-based.
pub fn truncate_at_word_boundary(s: &str, max_len: usize) -> String {
    let max_len = max_len.max(3);
    if s.chars().count() <= max_len {
        return s.to_string();
    }
    // We need to fit "content..." in max_len chars, so content ≤ max_len - 3.
    let content_limit = max_len - 3;
    // Find the byte offset of the content_limit-th character.
    let byte_offset = s
        .char_indices()
        .nth(content_limit)
        .map_or(s.len(), |(idx, _)| idx);
    let prefix = &s[..byte_offset];
    match prefix.rfind(' ') {
        Some(pos) => format!("{}...", &s[..pos]),
        None => format!("{prefix}..."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_string_unchanged() {
        assert_eq!(truncate_at_word_boundary("short", 80), "short");
    }

    #[test]
    fn long_string_truncated_at_word() {
        let result = truncate_at_word_boundary("hello world this is a long text", 15);
        assert!(result.ends_with("..."));
        // Result must be ≤ max_len chars (N-M17).
        assert!(
            result.chars().count() <= 15,
            "result length {} > 15",
            result.chars().count()
        );
    }

    #[test]
    fn no_spaces_truncates_at_max() {
        // max_len=10 → content_limit=7, then "..." → total 10 chars.
        let result = truncate_at_word_boundary("abcdefghijklmnopqrstuvwxyz", 10);
        assert_eq!(result, "abcdefg...");
        assert_eq!(result.chars().count(), 10);
    }

    #[test]
    fn result_never_exceeds_max_len() {
        // Invariant: result.chars().count() ≤ max_len for all inputs.
        for max_len in [3, 5, 10, 20] {
            let long = "a".repeat(100);
            let result = truncate_at_word_boundary(&long, max_len);
            assert!(
                result.chars().count() <= max_len,
                "max_len={max_len} but result has {} chars",
                result.chars().count()
            );
        }
    }

    #[test]
    fn multi_byte_utf8_does_not_panic() {
        // 'é' is 2 bytes — truncation must be character-count-based.
        let result = truncate_at_word_boundary("café élève über", 6);
        assert!(result.ends_with("..."));
        assert!(
            result.chars().count() <= 6,
            "result {} exceeds max_len 6",
            result.chars().count()
        );
    }

    #[test]
    fn emoji_does_not_panic() {
        let result = truncate_at_word_boundary("hello 🌍🌎🌏 world", 8);
        assert!(result.ends_with("..."));
        assert!(result.chars().count() <= 8);
    }

    #[test]
    fn exact_boundary_no_truncation() {
        assert_eq!(truncate_at_word_boundary("hello", 5), "hello");
    }
}
