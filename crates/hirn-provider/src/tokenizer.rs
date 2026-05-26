//! Concrete tokenizer implementations for hirn.
//!
//! `hirn-core` owns the tokenizer trait and heuristic fallback.
//! `hirn-provider` owns real model-backed tokenizers.

use std::sync::Arc;

#[cfg(feature = "tiktoken")]
use std::sync::OnceLock;

#[cfg(feature = "tiktoken")]
use hirn_core::embed::TokenCounter;
use hirn_core::tokenizer::{EstimatingTokenizer, Tokenizer};
use hirn_core::{HirnError, HirnResult};

/// Supported tiktoken tokenizer models.
#[cfg(feature = "tiktoken")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerModel {
    /// `cl100k_base` — used by GPT-4, GPT-3.5-turbo, and text-embedding-3 models.
    Cl100kBase,
    /// `o200k_base` — used by GPT-4o and o-series models.
    O200kBase,
}

#[cfg(feature = "tiktoken")]
impl TokenizerModel {
    /// Parse a model name from provider or config input.
    pub fn parse(value: &str) -> HirnResult<Self> {
        match value {
            "cl100k_base" | "cl100k" => Ok(Self::Cl100kBase),
            "o200k_base" | "o200k" => Ok(Self::O200kBase),
            other => Err(HirnError::config(format!(
                "unknown tiktoken model '{other}' (expected: cl100k_base, o200k_base)"
            ))),
        }
    }
}

/// A tokenizer backed by a tiktoken `CoreBPE`.
#[cfg(feature = "tiktoken")]
pub struct TiktokenTokenizer {
    bpe: Arc<tiktoken_rs::CoreBPE>,
    model: TokenizerModel,
}

#[cfg(feature = "tiktoken")]
static CL100K_BPE: OnceLock<Result<Arc<tiktoken_rs::CoreBPE>, String>> = OnceLock::new();
#[cfg(feature = "tiktoken")]
static O200K_BPE: OnceLock<Result<Arc<tiktoken_rs::CoreBPE>, String>> = OnceLock::new();

#[cfg(feature = "tiktoken")]
fn cached_bpe(model: TokenizerModel) -> Result<Arc<tiktoken_rs::CoreBPE>, HirnError> {
    let result = match model {
        TokenizerModel::Cl100kBase => CL100K_BPE.get_or_init(|| {
            tiktoken_rs::cl100k_base()
                .map(Arc::new)
                .map_err(|err| err.to_string())
        }),
        TokenizerModel::O200kBase => O200K_BPE.get_or_init(|| {
            tiktoken_rs::o200k_base()
                .map(Arc::new)
                .map_err(|err| err.to_string())
        }),
    };

    match result {
        Ok(bpe) => Ok(Arc::clone(bpe)),
        Err(err) => Err(HirnError::storage(format!("tokenizer init failed: {err}"))),
    }
}

#[cfg(feature = "tiktoken")]
impl TiktokenTokenizer {
    /// Create a tokenizer for the given model.
    pub fn new(model: TokenizerModel) -> Result<Self, HirnError> {
        let bpe = cached_bpe(model)?;
        Ok(Self { bpe, model })
    }

    /// The model this tokenizer uses.
    pub const fn model(&self) -> TokenizerModel {
        self.model
    }
}

#[cfg(feature = "tiktoken")]
impl TokenCounter for TiktokenTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        self.bpe.encode_ordinary(text).len()
    }
}

#[cfg(feature = "tiktoken")]
impl Tokenizer for TiktokenTokenizer {
    fn truncate(&self, text: &str, max_tokens: usize) -> String {
        let tokens: Vec<u32> = self.bpe.encode_ordinary(text);
        if tokens.len() <= max_tokens {
            return text.to_string();
        }

        self.bpe
            .decode(tokens[..max_tokens].to_vec())
            .unwrap_or_default()
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.bpe
            .encode_ordinary(text)
            .into_iter()
            .map(|token| token as usize)
            .collect()
    }

    fn decode(&self, tokens: &[usize]) -> HirnResult<String> {
        let token_ids: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
        self.bpe
            .decode(token_ids)
            .map_err(|err| HirnError::storage(format!("tiktoken decode failed: {err}")))
    }

    fn model_id(&self) -> &str {
        match self.model {
            TokenizerModel::Cl100kBase => "cl100k_base",
            TokenizerModel::O200kBase => "o200k_base",
        }
    }

    fn max_tokens(&self) -> usize {
        match self.model {
            TokenizerModel::Cl100kBase => 8_192,
            TokenizerModel::O200kBase => 128_000,
        }
    }
}

/// Best-effort default tokenizer for the engine.
///
/// Prefers a real `cl100k_base` tokenizer and falls back to the heuristic
/// estimator if model initialization fails.
#[must_use]
pub fn default_tokenizer() -> Arc<dyn Tokenizer> {
    #[cfg(feature = "tiktoken")]
    {
        TiktokenTokenizer::new(TokenizerModel::Cl100kBase)
            .map(|tokenizer| Arc::new(tokenizer) as Arc<dyn Tokenizer>)
            .unwrap_or_else(|_| Arc::new(EstimatingTokenizer))
    }

    #[cfg(not(feature = "tiktoken"))]
    {
        Arc::new(EstimatingTokenizer)
    }
}

/// Build a tokenizer from provider-facing configuration.
pub fn build_tokenizer(
    tokenizer_type: &str,
    model: Option<&str>,
    _max_length: Option<usize>,
) -> HirnResult<Arc<dyn Tokenizer>> {
    #[cfg(not(any(feature = "tiktoken", feature = "hf-tokenizer")))]
    let _ = model;

    match tokenizer_type {
        "estimating" => Ok(Arc::new(EstimatingTokenizer)),
        #[cfg(feature = "tiktoken")]
        "tiktoken" => {
            let model = TokenizerModel::parse(model.unwrap_or("cl100k_base"))?;
            let tokenizer = TiktokenTokenizer::new(model)?;
            Ok(Arc::new(tokenizer))
        }
        #[cfg(not(feature = "tiktoken"))]
        "tiktoken" => Err(HirnError::config(
            "tiktoken tokenizer requires the 'tiktoken' feature",
        )),
        #[cfg(feature = "hf-tokenizer")]
        "huggingface" => {
            let model = model
                .ok_or_else(|| HirnError::config("'model' required for huggingface tokenizer"))?;
            let mut tokenizer = HuggingFaceTokenizer::from_pretrained(model)?;
            if let Some(max_length) = _max_length {
                tokenizer = tokenizer.with_max_length(max_length);
            }
            Ok(Arc::new(tokenizer))
        }
        #[cfg(not(feature = "hf-tokenizer"))]
        "huggingface" => Err(HirnError::config(
            "huggingface tokenizer requires the 'hf-tokenizer' feature",
        )),
        other => Err(HirnError::config(format!(
            "unknown tokenizer type '{other}'"
        ))),
    }
}

/// A tokenizer backed by any HuggingFace `tokenizers` model.
#[cfg(feature = "hf-tokenizer")]
pub struct HuggingFaceTokenizer {
    inner: tokenizers::Tokenizer,
    model_id: String,
    max_length: usize,
}

#[cfg(feature = "hf-tokenizer")]
impl std::fmt::Debug for HuggingFaceTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HuggingFaceTokenizer")
            .field("model_id", &self.model_id)
            .field("max_length", &self.max_length)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "hf-tokenizer")]
impl HuggingFaceTokenizer {
    /// Load a tokenizer from the HuggingFace Hub.
    pub fn from_pretrained(model_id: &str) -> Result<Self, HirnError> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|err| HirnError::provider(format!("HF API init failed: {err}")))?;
        let repo = api.model(model_id.to_owned());
        let tokenizer_path = repo.get("tokenizer.json").map_err(|err| {
            HirnError::provider(format!(
                "failed to download tokenizer for '{model_id}': {err}"
            ))
        })?;
        Self::from_file(tokenizer_path, model_id)
    }

    /// Load a tokenizer from a local `tokenizer.json` file.
    pub fn from_file(path: impl AsRef<std::path::Path>, model_id: &str) -> Result<Self, HirnError> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|err| {
            HirnError::provider(format!("failed to load tokenizer from file: {err}"))
        })?;
        let max_length = inner.get_truncation().map(|t| t.max_length).unwrap_or(512);
        Ok(Self {
            inner,
            model_id: model_id.to_owned(),
            max_length,
        })
    }

    /// Override the maximum token length.
    #[must_use]
    pub const fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }
}

#[cfg(feature = "hf-tokenizer")]
impl TokenCounter for HuggingFaceTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        self.inner
            .encode(text, false)
            .map(|encoding| encoding.get_ids().len())
            .unwrap_or(0)
    }
}

#[cfg(feature = "hf-tokenizer")]
impl Tokenizer for HuggingFaceTokenizer {
    fn truncate(&self, text: &str, max_tokens: usize) -> String {
        let encoding = match self.inner.encode(text, false) {
            Ok(encoding) => encoding,
            Err(_) => return text.to_string(),
        };
        let ids = encoding.get_ids();
        if ids.len() <= max_tokens {
            return text.to_string();
        }

        self.inner
            .decode(&ids[..max_tokens], true)
            .unwrap_or_else(|_| text.to_string())
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner
            .encode(text, false)
            .map(|encoding| {
                encoding
                    .get_ids()
                    .iter()
                    .map(|&token| token as usize)
                    .collect()
            })
            .unwrap_or_default()
    }

    fn decode(&self, tokens: &[usize]) -> HirnResult<String> {
        let token_ids: Vec<u32> = tokens.iter().map(|&token| token as u32).collect();
        self.inner
            .decode(&token_ids, true)
            .map_err(|err| HirnError::storage(format!("HuggingFace decode failed: {err}")))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn max_tokens(&self) -> usize {
        self.max_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_counts_tokens() {
        let tokenizer = TiktokenTokenizer::new(TokenizerModel::Cl100kBase).unwrap();
        assert_eq!(tokenizer.count_tokens("hello"), 1);
        assert_eq!(tokenizer.count_tokens("hello world"), 2);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_round_trip() {
        let tokenizer = TiktokenTokenizer::new(TokenizerModel::Cl100kBase).unwrap();
        let encoded = tokenizer.encode("hello world");
        let decoded = tokenizer.decode(&encoded).unwrap();
        assert_eq!(decoded, "hello world");
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_truncate_respects_budget() {
        let tokenizer = TiktokenTokenizer::new(TokenizerModel::Cl100kBase).unwrap();
        let truncated = tokenizer.truncate("hello world test", 2);

        assert_eq!(tokenizer.count_tokens(&truncated), 2);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_metadata_is_stable() {
        let cl = TiktokenTokenizer::new(TokenizerModel::Cl100kBase).unwrap();
        let o2 = TiktokenTokenizer::new(TokenizerModel::O200kBase).unwrap();

        assert_eq!(cl.model_id(), "cl100k_base");
        assert_eq!(o2.model_id(), "o200k_base");
        assert_eq!(cl.max_tokens(), 8_192);
        assert_eq!(o2.max_tokens(), 128_000);
    }

    #[test]
    fn default_tokenizer_is_available() {
        let tokenizer = default_tokenizer();
        assert!(tokenizer.count_tokens("hello world") > 0);
    }

    #[cfg(not(feature = "tiktoken"))]
    #[test]
    fn default_tokenizer_falls_back_to_estimate_without_tiktoken() {
        let tokenizer = default_tokenizer();

        assert_eq!(tokenizer.model_id(), "estimate");
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn build_tokenizer_creates_tiktoken_instances() {
        let tokenizer = build_tokenizer("tiktoken", Some("cl100k_base"), None).unwrap();

        assert_eq!(tokenizer.model_id(), "cl100k_base");
        assert_eq!(tokenizer.count_tokens("hello world"), 2);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn build_tokenizer_rejects_unknown_tiktoken_models() {
        let error = match build_tokenizer("tiktoken", Some("bad-model"), None) {
            Ok(_) => panic!("expected bad-model to be rejected"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("unknown tiktoken model 'bad-model'")
        );
    }

    #[cfg(not(feature = "tiktoken"))]
    #[test]
    fn build_tokenizer_requires_tiktoken_feature_for_real_tiktoken() {
        let error = match build_tokenizer("tiktoken", Some("cl100k_base"), None) {
            Ok(_) => panic!("expected missing tiktoken feature to be rejected"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("requires the 'tiktoken' feature")
        );
    }

    #[cfg(feature = "hf-tokenizer")]
    mod hf_tests {
        use std::io::Write;

        use super::*;

        fn sample_tokenizer_json() -> String {
            serde_json::json!({
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [
                    {
                        "id": 0,
                        "content": "[UNK]",
                        "single_word": false,
                        "lstrip": false,
                        "rstrip": false,
                        "normalized": true,
                        "special": true
                    }
                ],
                "normalizer": { "type": "Sequence", "normalizers": [] },
                "pre_tokenizer": { "type": "Whitespace" },
                "post_processor": null,
                "decoder": { "type": "WordPiece", "prefix": "##", "cleanup": true },
                "model": {
                    "type": "WordPiece",
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100,
                    "vocab": {
                        "[UNK]": 0,
                        "hello": 1,
                        "world": 2,
                        "test": 3,
                        "this": 4,
                        "is": 5,
                        "a": 6,
                        "tokenizer": 7
                    }
                }
            })
            .to_string()
        }

        fn write_sample_tokenizer() -> tempfile::TempPath {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(sample_tokenizer_json().as_bytes()).unwrap();
            file.into_temp_path()
        }

        #[test]
        fn from_file_loads_hf_tokenizer() {
            let path = write_sample_tokenizer();
            let tokenizer = HuggingFaceTokenizer::from_file(&path, "test-model").unwrap();

            assert_eq!(tokenizer.model_id(), "test-model");
            assert_eq!(tokenizer.count_tokens("hello world"), 2);
        }

        #[test]
        fn hf_truncation_respects_budget() {
            let path = write_sample_tokenizer();
            let tokenizer = HuggingFaceTokenizer::from_file(&path, "test-model")
                .unwrap()
                .with_max_length(2);
            let truncated = tokenizer.truncate("hello world test", 2);

            assert_eq!(tokenizer.count_tokens(&truncated), 2);
        }
    }
}
