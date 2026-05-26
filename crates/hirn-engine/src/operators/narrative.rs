//! Narrative assembly operator.
//!
//! Packs memory content into a single narrative `RecordBatch` that fits
//! within a given token budget. Used to construct LLM-ready context windows.

use std::sync::Arc;

use arrow_array::Array;
use arrow_array::RecordBatch;
use arrow_array::builder::StringBuilder;
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;

use hirn_core::embed::TokenCounter;
use hirn_core::error::HirnResult;

use super::{OpContext, Operator};

/// Operator that assembles input content into a token-budgeted narrative.
///
/// Reads the `content` column from input batches in order, concatenating
/// until the token budget is exhausted. Produces a single-row `RecordBatch`
/// with a `narrative` column containing the assembled text.
pub struct NarrativeAssemble {
    /// Maximum number of tokens in the assembled narrative.
    pub max_tokens: usize,
    /// Token counter implementation.
    pub token_counter: Arc<dyn TokenCounter>,
}

#[async_trait]
impl Operator for NarrativeAssemble {
    async fn execute(
        &self,
        input: Vec<RecordBatch>,
        _ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>> {
        let mut narrative = String::new();
        let mut used_tokens: usize = 0;

        'outer: for batch in &input {
            let content_col = match batch.column_by_name("content") {
                Some(c) => c,
                None => continue,
            };
            let str_arr = content_col.as_string::<i32>();
            for i in 0..str_arr.len() {
                if str_arr.is_null(i) {
                    continue;
                }
                let text = str_arr.value(i);
                let tokens = self.token_counter.count_tokens(text);
                if used_tokens + tokens > self.max_tokens {
                    // Try to fit a truncated version.
                    let remaining = self.max_tokens.saturating_sub(used_tokens);
                    if remaining > 0 {
                        let truncated = truncate_to_tokens(text, remaining, &*self.token_counter);
                        if !truncated.is_empty() {
                            if !narrative.is_empty() {
                                narrative.push_str("\n\n");
                            }
                            narrative.push_str(&truncated);
                        }
                    }
                    break 'outer;
                }
                if !narrative.is_empty() {
                    narrative.push_str("\n\n");
                    // Account for separator tokens.
                    let sep_tokens = self.token_counter.count_tokens("\n\n");
                    used_tokens += sep_tokens;
                }
                narrative.push_str(text);
                used_tokens += tokens;
            }
        }

        let schema = Arc::new(Schema::new(vec![Field::new(
            "narrative",
            DataType::Utf8,
            false,
        )]));

        if narrative.is_empty() {
            return Ok(vec![RecordBatch::new_empty(schema)]);
        }

        let mut builder = StringBuilder::new();
        builder.append_value(&narrative);
        let batch = RecordBatch::try_new(schema, vec![Arc::new(builder.finish())])
            .map_err(|e| hirn_core::error::HirnError::storage(e))?;
        Ok(vec![batch])
    }
}

/// Truncate text to fit approximately `max_tokens` tokens by binary search
/// on character boundaries.
fn truncate_to_tokens(text: &str, max_tokens: usize, counter: &dyn TokenCounter) -> String {
    if counter.count_tokens(text) <= max_tokens {
        return text.to_string();
    }
    // Binary search for the longest prefix that fits.
    let chars: Vec<char> = text.chars().collect();
    let mut lo = 0usize;
    let mut hi = chars.len();
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        let prefix: String = chars[..mid].iter().collect();
        if counter.count_tokens(&prefix) <= max_tokens {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if lo == 0 {
        return String::new();
    }
    chars[..lo].iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::embed::CharEstimateCounter;

    #[test]
    fn truncate_respects_budget() {
        let counter = CharEstimateCounter;
        // CharEstimateCounter: ceil(len / 4)
        let text = "a]".repeat(40); // 80 chars → 20 tokens
        let result = truncate_to_tokens(&text, 10, &counter);
        let tokens = counter.count_tokens(&result);
        assert!(tokens <= 10, "tokens={tokens}");
    }

    #[test]
    fn truncate_empty_budget() {
        let counter = CharEstimateCounter;
        let result = truncate_to_tokens("hello world", 0, &counter);
        assert!(result.is_empty());
    }
}
