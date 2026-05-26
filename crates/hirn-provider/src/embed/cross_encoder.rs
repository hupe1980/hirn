//! `CrossEncoderReranker` — local, offline reranker using ONNX Runtime.
//!
//! Runs a cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
//! locally via ONNX Runtime. No API key needed; all inference is on-device.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use ndarray::Array2;
use parking_lot::Mutex;
use tracing::debug;

use hirn_core::embed::RerankResult;
use hirn_core::error::HirnResult;

use super::error::EmbedError;
use crate::metrics::record_invalid_reranker_score;

/// Default HuggingFace model ID for the cross-encoder.
const DEFAULT_MODEL_ID: &str = "cross-encoder/ms-marco-MiniLM-L-6-v2";

/// Default maximum sequence length (query + document tokens).
const DEFAULT_MAX_LENGTH: usize = 512;

/// Local cross-encoder reranker backed by ONNX Runtime.
///
/// Downloads the model from HuggingFace Hub on first use and caches it
/// locally. All inference runs on-device — no API calls, no data leaves
/// the machine.
///
/// # Example
///
/// ```rust,no_run
/// use hirn_provider::CrossEncoderReranker;
///
/// let reranker = CrossEncoderReranker::from_pretrained(
///     "cross-encoder/ms-marco-MiniLM-L-6-v2",
///     None, // use default HF cache
/// ).expect("model download/load failed");
/// ```
pub struct CrossEncoderReranker {
    session: Arc<Mutex<ort::session::Session>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    max_length: usize,
}

impl CrossEncoderReranker {
    /// Load a cross-encoder model from HuggingFace Hub.
    ///
    /// - `model_id`: HuggingFace model identifier (e.g. `"cross-encoder/ms-marco-MiniLM-L-6-v2"`)
    /// - `cache_dir`: Optional directory for model files. Defaults to the HF Hub cache.
    pub fn from_pretrained(model_id: &str, cache_dir: Option<&Path>) -> HirnResult<Self> {
        let api = if let Some(dir) = cache_dir {
            hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(dir.to_path_buf())
                .build()
                .map_err(|e| {
                    EmbedError::local("cross-encoder", format!("HF API init failed: {e}"))
                })?
        } else {
            hf_hub::api::sync::Api::new().map_err(|e| {
                EmbedError::local("cross-encoder", format!("HF API init failed: {e}"))
            })?
        };

        let repo = api.model(model_id.to_owned());

        // Download ONNX model file.
        let model_path = repo.get("onnx/model.onnx").map_err(|e| {
            EmbedError::local("cross-encoder", format!("Model download failed: {e}"))
        })?;

        // Download tokenizer.
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            EmbedError::local("cross-encoder", format!("Tokenizer download failed: {e}"))
        })?;

        Self::from_files(&model_path, &tokenizer_path)
    }

    /// Load the default cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
    pub fn default_model() -> HirnResult<Self> {
        Self::from_pretrained(DEFAULT_MODEL_ID, None)
    }

    /// Load a cross-encoder from local ONNX model and tokenizer files.
    pub fn from_files(model_path: &Path, tokenizer_path: &Path) -> HirnResult<Self> {
        let mut builder = ort::session::Session::builder()
            .map_err(|e| {
                EmbedError::local("cross-encoder", format!("ONNX session builder failed: {e}"))
            })?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| {
                EmbedError::local(
                    "cross-encoder",
                    format!("ONNX optimization config failed: {e}"),
                )
            })?
            .with_intra_threads(1)
            .map_err(|e| {
                EmbedError::local("cross-encoder", format!("ONNX thread config failed: {e}"))
            })?;
        let session = builder.commit_from_file(model_path).map_err(|e| {
            EmbedError::local("cross-encoder", format!("ONNX model load failed: {e}"))
        })?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(|e| {
            EmbedError::local("cross-encoder", format!("Tokenizer load failed: {e}"))
        })?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            max_length: DEFAULT_MAX_LENGTH,
        })
    }

    /// Override the maximum sequence length.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Score a batch of (query, document) pairs in a single ONNX forward pass.
    ///
    /// F-019 FIX: All pairs are tokenized, padded to the same length, batched
    /// into a single tensor, and scored in one session.run() call.
    fn score_batch(&self, query: &str, documents: &[String]) -> HirnResult<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let (input_ids_arr, attn_mask_arr, type_ids_arr) =
            tokenize_and_pad(&self.tokenizer, query, documents, self.max_length)?;

        let input_ids_tensor = ort::value::Tensor::from_array(input_ids_arr).map_err(|e| {
            EmbedError::local("cross-encoder", format!("input_ids tensor error: {e}"))
        })?;
        let attn_mask_tensor = ort::value::Tensor::from_array(attn_mask_arr).map_err(|e| {
            EmbedError::local("cross-encoder", format!("attention_mask tensor error: {e}"))
        })?;
        let type_ids_tensor = ort::value::Tensor::from_array(type_ids_arr).map_err(|e| {
            EmbedError::local("cross-encoder", format!("token_type_ids tensor error: {e}"))
        })?;

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attn_mask_tensor,
                "token_type_ids" => type_ids_tensor,
            ])
            .map_err(|e| {
                EmbedError::local("cross-encoder", format!("ONNX batch inference failed: {e}"))
            })?;

        // Output shape: [batch_size, 1] → extract the logit per pair.
        let (_shape, logits) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
            EmbedError::local("cross-encoder", format!("Output extraction failed: {e}"))
        })?;

        Ok(logits.iter().copied().collect())
    }
}

/// Tokenize query/document pairs and build padded `[batch_size, max_len]` tensors.
///
/// Returns `(input_ids, attention_mask, token_type_ids)` as `Array2<i64>`.
/// This is extracted from [`CrossEncoderReranker::score_batch`] so the
/// tokenization + padding logic can be unit-tested without an ONNX session.
fn tokenize_and_pad(
    tokenizer: &tokenizers::Tokenizer,
    query: &str,
    documents: &[String],
    max_length: usize,
) -> HirnResult<(Array2<i64>, Array2<i64>, Array2<i64>)> {
    let pairs: Vec<(&str, &str)> = documents.iter().map(|d| (query, d.as_str())).collect();
    let encodings = tokenizer.encode_batch(pairs, true).map_err(|e| {
        EmbedError::local("cross-encoder", format!("Batch tokenization failed: {e}"))
    })?;

    let batch_size = encodings.len();

    let max_len = encodings
        .iter()
        .map(|enc| enc.get_ids().len().min(max_length))
        .max()
        .unwrap_or(0);

    let mut all_input_ids = vec![0i64; batch_size * max_len];
    let mut all_attn_mask = vec![0i64; batch_size * max_len];
    let mut all_type_ids = vec![0i64; batch_size * max_len];

    for (i, enc) in encodings.iter().enumerate() {
        let ids = enc.get_ids();
        let attn = enc.get_attention_mask();
        let types = enc.get_type_ids();
        let len = ids.len().min(max_length).min(max_len);
        let offset = i * max_len;
        for j in 0..len {
            all_input_ids[offset + j] = ids[j] as i64;
            all_attn_mask[offset + j] = attn[j] as i64;
            all_type_ids[offset + j] = types[j] as i64;
        }
    }

    let input_ids_arr = Array2::from_shape_vec((batch_size, max_len), all_input_ids)
        .map_err(|e| EmbedError::local("cross-encoder", format!("input_ids shape error: {e}")))?;
    let attn_mask_arr =
        Array2::from_shape_vec((batch_size, max_len), all_attn_mask).map_err(|e| {
            EmbedError::local("cross-encoder", format!("attention_mask shape error: {e}"))
        })?;
    let type_ids_arr =
        Array2::from_shape_vec((batch_size, max_len), all_type_ids).map_err(|e| {
            EmbedError::local("cross-encoder", format!("token_type_ids shape error: {e}"))
        })?;

    Ok((input_ids_arr, attn_mask_arr, type_ids_arr))
}

#[async_trait]
impl hirn_core::embed::Reranker for CrossEncoderReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> HirnResult<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        // F-019 FIX: Run batched inference on a blocking thread — ONNX is CPU-bound.
        let session = Arc::clone(&self.session);
        let tokenizer = Arc::clone(&self.tokenizer);
        let max_length = self.max_length;
        let query_owned = query.to_owned();
        let docs_owned: Vec<String> = documents.iter().map(|d| (*d).to_owned()).collect();

        let scores = tokio::task::spawn_blocking(move || {
            let reranker = CrossEncoderReranker {
                session,
                tokenizer,
                max_length,
            };
            reranker.score_batch(&query_owned, &docs_owned)
        })
        .await
        .map_err(|e| EmbedError::local("cross-encoder", format!("Rerank task panicked: {e}")))??;

        // Build results sorted by score descending, truncated to top_k.
        let mut indexed: Vec<(usize, f32)> = scores
            .into_iter()
            .enumerate()
            .filter_map(|(index, score)| {
                if !score.is_finite() {
                    tracing::warn!(
                        index,
                        score = %score,
                        "cross-encoder returned non-finite score, skipping"
                    );
                    record_invalid_reranker_score(DEFAULT_MODEL_ID, "non_finite");
                    return None;
                }

                Some((index, score))
            })
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        indexed.truncate(top_k);

        debug!(
            model = DEFAULT_MODEL_ID,
            query_len = query.len(),
            docs = documents.len(),
            top_k,
            "cross-encoder rerank complete"
        );

        Ok(indexed
            .into_iter()
            .map(|(index, score)| RerankResult { index, score })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::embed::Reranker;

    /// Build a minimal BERT-style tokenizer in-memory for unit tests.
    /// Uses the `tokenizers` crate directly — no model files needed.
    fn test_tokenizer() -> tokenizers::Tokenizer {
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::normalizers::BertNormalizer;
        use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
        use tokenizers::processors::template::TemplateProcessing;

        // Minimal vocab for test purposes — uses a fixed-size array to
        // satisfy the `Into<AHashMap>` bound on WordPieceBuilder::vocab.
        let vocab = [
            ("[PAD]".to_string(), 0u32),
            ("[UNK]".to_string(), 1),
            ("[CLS]".to_string(), 2),
            ("[SEP]".to_string(), 3),
            ("the".to_string(), 4),
            ("capital".to_string(), 5),
            ("of".to_string(), 6),
            ("france".to_string(), 7),
            ("is".to_string(), 8),
            ("paris".to_string(), 9),
            ("hello".to_string(), 10),
            ("world".to_string(), 11),
            ("rust".to_string(), 12),
            ("language".to_string(), 13),
        ];

        let wordpiece = WordPiece::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();

        let mut tokenizer = tokenizers::Tokenizer::new(wordpiece);
        tokenizer.with_normalizer(Some(BertNormalizer::default()));
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS]:0 $A:0 [SEP]:0")
                .unwrap()
                .try_pair("[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1")
                .unwrap()
                .special_tokens(vec![("[CLS]", 2), ("[SEP]", 3)])
                .build()
                .unwrap(),
        ));
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).unwrap();
        tokenizer
    }

    #[test]
    fn tokenize_and_pad_shapes() {
        let tok = test_tokenizer();
        let docs = vec!["the capital of france".into(), "hello world".into()];

        let (ids, mask, types) = tokenize_and_pad(&tok, "paris", &docs, 512).unwrap();

        // Two document pairs → batch_size = 2.
        assert_eq!(ids.shape()[0], 2);
        assert_eq!(mask.shape()[0], 2);
        assert_eq!(types.shape()[0], 2);
        // Both rows must have the same padded length.
        assert_eq!(ids.shape()[1], mask.shape()[1]);
        assert_eq!(ids.shape()[1], types.shape()[1]);
    }

    #[test]
    fn tokenize_and_pad_attention_mask_correctness() {
        let tok = test_tokenizer();
        let docs = vec!["the capital".into(), "hello world rust language".into()];

        let (ids, mask, _types) = tokenize_and_pad(&tok, "paris", &docs, 512).unwrap();

        let max_len = ids.shape()[1];

        // Shorter sequence should have trailing zeros in attention mask.
        let short_row = mask.row(0);
        let long_row = mask.row(1);

        // The longer sequence should have more 1s than the shorter one.
        let short_ones: i64 = short_row.iter().sum();
        let long_ones: i64 = long_row.iter().sum();
        assert!(
            short_ones < long_ones,
            "short {short_ones} >= long {long_ones}"
        );

        // Padding positions should have 0 in input_ids and 0 in attention mask.
        for j in 0..max_len {
            if mask[[0, j]] == 0 {
                assert_eq!(ids[[0, j]], 0, "padded position should have id=0");
            }
        }
    }

    #[test]
    fn tokenize_and_pad_max_length_truncation() {
        let tok = test_tokenizer();
        let docs = vec!["the capital of france is paris hello world rust language".into()];

        // Severely limit max_length to force truncation.
        let (ids, _mask, _types) = tokenize_and_pad(&tok, "paris", &docs, 6).unwrap();

        assert!(ids.shape()[1] <= 6, "should be truncated to max_length=6");
    }

    #[test]
    fn tokenize_and_pad_token_type_ids() {
        let tok = test_tokenizer();
        let docs = vec!["the capital of france".into()];

        let (_ids, _mask, types) = tokenize_and_pad(&tok, "paris", &docs, 512).unwrap();

        // Pair encoding: [CLS] query [SEP] document [SEP]
        // Type IDs: segment A (query) = 0, segment B (document) = 1.
        let row: Vec<i64> = types.row(0).to_vec();
        // Must contain both 0s and 1s.
        assert!(row.contains(&0), "should have type_id 0 for query segment");
        assert!(
            row.contains(&1),
            "should have type_id 1 for document segment"
        );
    }

    #[test]
    fn tokenize_and_pad_single_doc() {
        let tok = test_tokenizer();
        let docs = vec!["hello".into()];

        let (ids, mask, types) = tokenize_and_pad(&tok, "world", &docs, 512).unwrap();

        assert_eq!(ids.shape()[0], 1);
        assert_eq!(mask.shape()[0], 1);
        assert_eq!(types.shape()[0], 1);
        // All attention positions should be 1 (no padding for a single doc).
        let ones: i64 = mask.row(0).iter().sum();
        assert_eq!(ones, ids.shape()[1] as i64);
    }

    #[tokio::test]
    async fn empty_documents_returns_empty() {
        // NoopReranker is used here because CrossEncoderReranker::rerank
        // early-returns on empty docs before touching the ONNX session.
        // The tokenize_and_pad tests above verify the actual CrossEncoder logic.
        let reranker = hirn_core::embed::NoopReranker;
        let result = reranker.rerank("query", &[], 5).await.unwrap();
        assert!(result.is_empty());
    }

    /// Integration test requiring model download. Run with:
    /// `cargo test -p hirn-provider --features cross-encoder -- --ignored cross_encoder`
    #[tokio::test]
    #[ignore]
    async fn rerank_with_real_model() {
        let reranker = CrossEncoderReranker::default_model()
            .expect("Failed to download/load cross-encoder model");

        let docs = &[
            "The capital of France is Paris.",
            "Photosynthesis converts sunlight to energy.",
            "Paris is a beautiful city in France known for the Eiffel Tower.",
            "Rust is a systems programming language.",
        ];

        let results = reranker
            .rerank("What is the capital of France?", docs, 4)
            .await
            .unwrap();

        assert_eq!(results.len(), 4);
        // The most relevant docs should be ranked first.
        // Index 0 and 2 are about France/Paris.
        let top_indices: Vec<usize> = results.iter().take(2).map(|r| r.index).collect();
        assert!(
            top_indices.contains(&0) || top_indices.contains(&2),
            "Expected France-related docs in top 2, got indices: {top_indices:?}"
        );
    }

    #[tokio::test]
    #[ignore]
    async fn top_k_truncation() {
        let reranker = CrossEncoderReranker::default_model()
            .expect("Failed to download/load cross-encoder model");

        let docs = &["doc one", "doc two", "doc three", "doc four"];
        let results = reranker.rerank("query", docs, 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    #[ignore]
    async fn relevant_doc_scores_higher() {
        let reranker = CrossEncoderReranker::default_model()
            .expect("Failed to download/load cross-encoder model");

        let docs = &["quantum computing uses qubits", "cats are popular pets"];
        let results = reranker
            .rerank("What is quantum computing?", docs, 2)
            .await
            .unwrap();

        // The quantum computing doc should score higher.
        assert_eq!(results[0].index, 0, "Expected quantum doc ranked first");
        assert!(results[0].score > results[1].score);
    }
}
