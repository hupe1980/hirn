//! Tokenizer contracts and zero-dependency fallback implementations.
//!
//! `hirn-core` owns the abstraction used by the engine for token budgeting.
//! Concrete model-backed tokenizers live in `hirn-provider`.

use crate::embed::TokenCounter;
use crate::error::{HirnError, HirnResult};

/// Pluggable tokenizer providing encode, decode, truncate, and count operations.
///
/// Implementations must be `Send + Sync` for use in async contexts.
/// Object-safe: can be used as `Box<dyn Tokenizer>` or `Arc<dyn Tokenizer>`.
pub trait Tokenizer: TokenCounter + Send + Sync {
    /// Truncate `text` to at most `max_tokens` tokens, returning the truncated string.
    fn truncate(&self, text: &str, max_tokens: usize) -> String;

    /// Encode `text` into token IDs.
    fn encode(&self, text: &str) -> Vec<usize>;

    /// Decode token IDs back to a string.
    fn decode(&self, tokens: &[usize]) -> HirnResult<String>;

    /// Stable model identifier (for example `"cl100k_base"` or `"estimate"`).
    fn model_id(&self) -> &str;

    /// Maximum number of tokens the underlying model supports.
    fn max_tokens(&self) -> usize;
}

/// Character-estimate tokenizer: `ceil(len / 4)`. Zero dependencies, always available.
///
/// Useful as a fallback when no real tokenizer is configured.
#[derive(Debug, Clone, Copy)]
pub struct EstimatingTokenizer;

impl TokenCounter for EstimatingTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.len().div_ceil(4)
    }
}

impl Tokenizer for EstimatingTokenizer {
    fn truncate(&self, text: &str, max_tokens: usize) -> String {
        let max_chars = max_tokens * 4;
        if text.len() <= max_chars {
            return text.to_string();
        }

        let mut end = max_chars;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text[..end].to_string()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.as_bytes()
            .chunks(4)
            .enumerate()
            .map(|(index, _)| index)
            .collect()
    }

    fn decode(&self, _tokens: &[usize]) -> HirnResult<String> {
        Err(HirnError::InvalidInput(
            "EstimatingTokenizer cannot decode token IDs".to_string(),
        ))
    }

    fn model_id(&self) -> &str {
        "estimate"
    }

    fn max_tokens(&self) -> usize {
        128_000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedTokenizer;

    impl TokenCounter for FixedTokenizer {
        fn count_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    impl Tokenizer for FixedTokenizer {
        fn truncate(&self, text: &str, max_tokens: usize) -> String {
            text.split_whitespace()
                .take(max_tokens)
                .collect::<Vec<_>>()
                .join(" ")
        }

        fn encode(&self, text: &str) -> Vec<usize> {
            (0..text.split_whitespace().count()).collect()
        }

        fn decode(&self, tokens: &[usize]) -> HirnResult<String> {
            Ok(tokens
                .iter()
                .map(|token| token.to_string())
                .collect::<Vec<_>>()
                .join(" "))
        }

        fn model_id(&self) -> &str {
            "fixed"
        }

        fn max_tokens(&self) -> usize {
            64
        }
    }

    #[test]
    fn estimating_count_tokens() {
        let tok = EstimatingTokenizer;
        assert_eq!(crate::embed::TokenCounter::count_tokens(&tok, ""), 0);
        assert_eq!(crate::embed::TokenCounter::count_tokens(&tok, "a"), 1);
        assert_eq!(crate::embed::TokenCounter::count_tokens(&tok, "abcd"), 1);
        assert_eq!(crate::embed::TokenCounter::count_tokens(&tok, "abcde"), 2);
    }

    #[test]
    fn estimating_truncate_preserves_char_boundary() {
        let tok = EstimatingTokenizer;
        let truncated = tok.truncate("Gruezi mitenand", 2);
        assert_eq!(truncated, "Gruezi m");
        assert!(truncated.is_char_boundary(truncated.len()));
    }

    #[test]
    fn estimating_encode_and_decode_behavior() {
        let tok = EstimatingTokenizer;
        assert_eq!(tok.encode("abcdefghij"), vec![0, 1, 2]);
        assert!(tok.decode(&[0, 1, 2]).is_err());
    }

    #[test]
    fn estimating_metadata_is_stable() {
        let tok = EstimatingTokenizer;
        assert_eq!(tok.model_id(), "estimate");
        assert_eq!(tok.max_tokens(), 128_000);
    }

    #[test]
    fn tokenizer_trait_is_also_a_token_counter() {
        let tokenizer: Box<dyn Tokenizer> = Box::new(FixedTokenizer);
        let fixed = FixedTokenizer;
        let counter: &dyn TokenCounter = &fixed;

        assert_eq!(counter.count_tokens("one two three"), 3);
        assert_eq!(tokenizer.truncate("one two three", 2), "one two");
        assert_eq!(tokenizer.decode(&[1, 2]).unwrap(), "1 2");
    }
}
