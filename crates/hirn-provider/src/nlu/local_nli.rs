//! `LocalNli` — on-device 3-class NLI via ONNX Runtime.
//!
//! For contradiction, polarity, and negation-scope decisions at write-path
//! volume, a remote call per pair is not affordable and, in privacy-bound
//! deployments, not permitted. A small NLI cross-encoder
//! (e.g. `cross-encoder/nli-deberta-v3-small`) runs on-device in single-digit
//! milliseconds and is a far better polarity signal than any negation word
//! list.
//!
//! **Label order is read from the model's own `config.json`.** NLI checkpoints
//! disagree about head order — DeBERTa-MNLI emits
//! `[contradiction, entailment, neutral]`, other checkpoints emit
//! `[entailment, neutral, contradiction]` — and a hard-coded guess silently
//! inverts entailment and contradiction. Loading a model whose label mapping
//! cannot be established is an error, not a default.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::HirnResult;
use hirn_core::nlu::{DecisionSource, NliJudgment, NliLabel, NliModel, NluBudget};
use ndarray::Array2;
use parking_lot::Mutex;

use crate::embed::error::EmbedError;

/// Default HuggingFace NLI checkpoint.
const DEFAULT_MODEL_ID: &str = "cross-encoder/nli-deberta-v3-small";

/// Default maximum premise + hypothesis sequence length.
const DEFAULT_MAX_LENGTH: usize = 256;

/// Local ONNX natural-language-inference model.
pub struct LocalNli {
    session: Arc<Mutex<ort::session::Session>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    /// Output index of each NLI label, resolved from the model's `id2label`.
    label_order: Arc<[NliLabel; 3]>,
    max_length: usize,
    model_id: String,
}

impl LocalNli {
    /// Load the default NLI checkpoint from the HuggingFace Hub.
    pub fn default_model() -> HirnResult<Self> {
        Self::from_pretrained(DEFAULT_MODEL_ID, None)
    }

    /// Load an NLI checkpoint from the HuggingFace Hub.
    ///
    /// Downloads `onnx/model.onnx`, `tokenizer.json`, and `config.json`; the
    /// last supplies the `id2label` mapping that pins the head order.
    pub fn from_pretrained(model_id: &str, cache_dir: Option<&Path>) -> HirnResult<Self> {
        let api = if let Some(dir) = cache_dir {
            hf_hub::api::sync::ApiBuilder::new()
                .with_cache_dir(dir.to_path_buf())
                .build()
                .map_err(|e| EmbedError::local("local-nli", format!("HF API init failed: {e}")))?
        } else {
            hf_hub::api::sync::Api::new()
                .map_err(|e| EmbedError::local("local-nli", format!("HF API init failed: {e}")))?
        };

        let repo = api.model(model_id.to_owned());
        let model_path = repo
            .get("onnx/model.onnx")
            .map_err(|e| EmbedError::local("local-nli", format!("Model download failed: {e}")))?;
        let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
            EmbedError::local("local-nli", format!("Tokenizer download failed: {e}"))
        })?;
        let config_path = repo
            .get("config.json")
            .map_err(|e| EmbedError::local("local-nli", format!("Config download failed: {e}")))?;

        let config = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbedError::local("local-nli", format!("Config read failed: {e}")))?;
        let label_order = parse_label_order(&config)?;

        Self::from_files(model_id, &model_path, &tokenizer_path, label_order)
    }

    /// Load an NLI model from local files with an explicit head order.
    pub fn from_files(
        model_id: &str,
        model_path: &Path,
        tokenizer_path: &Path,
        label_order: [NliLabel; 3],
    ) -> HirnResult<Self> {
        let session = ort::session::Session::builder()
            .map_err(|e| EmbedError::local("local-nli", format!("session builder failed: {e}")))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| {
                EmbedError::local("local-nli", format!("optimization config failed: {e}"))
            })?
            .with_intra_threads(1)
            .map_err(|e| EmbedError::local("local-nli", format!("thread config failed: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| EmbedError::local("local-nli", format!("model load failed: {e}")))?;

        let mut tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| EmbedError::local("local-nli", format!("tokenizer load failed: {e}")))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: DEFAULT_MAX_LENGTH,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                ..Default::default()
            }))
            .map_err(|e| {
                EmbedError::local("local-nli", format!("truncation config failed: {e}"))
            })?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            label_order: Arc::new(label_order),
            max_length: DEFAULT_MAX_LENGTH,
            model_id: format!("local-nli/{model_id}"),
        })
    }

    /// Run one premise/hypothesis pair through the model.
    fn infer(&self, premise: &str, hypothesis: &str) -> HirnResult<[f32; 3]> {
        let encoding = self
            .tokenizer
            .encode((premise, hypothesis), true)
            .map_err(|e| EmbedError::local("local-nli", format!("tokenization failed: {e}")))?;

        let len = encoding.get_ids().len().min(self.max_length);
        let ids: Vec<i64> = encoding.get_ids()[..len]
            .iter()
            .map(|i| *i as i64)
            .collect();
        let mask: Vec<i64> = encoding.get_attention_mask()[..len]
            .iter()
            .map(|m| *m as i64)
            .collect();
        let types: Vec<i64> = encoding.get_type_ids()[..len]
            .iter()
            .map(|t| *t as i64)
            .collect();

        let shape = (1, len);
        let ids = Array2::from_shape_vec(shape, ids)
            .map_err(|e| EmbedError::local("local-nli", format!("input_ids shape error: {e}")))?;
        let mask = Array2::from_shape_vec(shape, mask).map_err(|e| {
            EmbedError::local("local-nli", format!("attention_mask shape error: {e}"))
        })?;
        let types = Array2::from_shape_vec(shape, types).map_err(|e| {
            EmbedError::local("local-nli", format!("token_type_ids shape error: {e}"))
        })?;

        let ids = ort::value::Tensor::from_array(ids)
            .map_err(|e| EmbedError::local("local-nli", format!("input_ids tensor error: {e}")))?;
        let mask = ort::value::Tensor::from_array(mask).map_err(|e| {
            EmbedError::local("local-nli", format!("attention_mask tensor error: {e}"))
        })?;
        let types = ort::value::Tensor::from_array(types).map_err(|e| {
            EmbedError::local("local-nli", format!("token_type_ids tensor error: {e}"))
        })?;

        let mut session = self.session.lock();
        let outputs = session
            .run(ort::inputs![
                "input_ids" => ids,
                "attention_mask" => mask,
                "token_type_ids" => types,
            ])
            .map_err(|e| EmbedError::local("local-nli", format!("inference failed: {e}")))?;

        let (_shape, logits) = outputs[0].try_extract_tensor::<f32>().map_err(|e| {
            EmbedError::local("local-nli", format!("output extraction failed: {e}"))
        })?;

        // A head that is not 3-class is not an NLI model; refusing beats
        // silently reading entailment off the wrong index.
        if logits.len() != 3 {
            return Err(EmbedError::local(
                "local-nli",
                format!(
                    "model returned {} logits; expected a 3-class NLI head",
                    logits.len()
                ),
            )
            .into());
        }

        Ok(softmax3([logits[0], logits[1], logits[2]]))
    }
}

/// Numerically stable 3-way softmax.
fn softmax3(logits: [f32; 3]) -> [f32; 3] {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return [1.0 / 3.0; 3];
    }
    let exps = logits.map(|l| (l - max).exp());
    let total: f32 = exps.iter().sum();
    if total <= f32::EPSILON {
        return [1.0 / 3.0; 3];
    }
    exps.map(|e| e / total)
}

/// Read the head order from a HuggingFace `config.json`'s `id2label`.
///
/// # Errors
/// Returns an error when `id2label` is absent, is not exactly three entries,
/// or names a label that is not an NLI class — every one of which would
/// otherwise mean reading entailment off the wrong output index.
fn parse_label_order(config_json: &str) -> HirnResult<[NliLabel; 3]> {
    let config: serde_json::Value = serde_json::from_str(config_json)
        .map_err(|e| EmbedError::local("local-nli", format!("config.json is not JSON: {e}")))?;
    let id2label = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            EmbedError::local(
                "local-nli",
                "config.json has no id2label map; cannot establish NLI head order",
            )
        })?;

    let mut by_index: HashMap<usize, NliLabel> = HashMap::new();
    for (index, label) in id2label {
        let index: usize = index.parse().map_err(|_| {
            EmbedError::local(
                "local-nli",
                format!("id2label key {index:?} is not an output index"),
            )
        })?;
        let name = label.as_str().unwrap_or_default();
        let label = NliLabel::parse(name).ok_or_else(|| {
            EmbedError::local(
                "local-nli",
                format!("id2label value {name:?} is not an NLI class"),
            )
        })?;
        by_index.insert(index, label);
    }

    if by_index.len() != 3 {
        return Err(EmbedError::local(
            "local-nli",
            format!("id2label has {} entries; expected 3", by_index.len()),
        )
        .into());
    }

    let order = [
        *by_index.get(&0).ok_or_else(|| missing_index(0))?,
        *by_index.get(&1).ok_or_else(|| missing_index(1))?,
        *by_index.get(&2).ok_or_else(|| missing_index(2))?,
    ];

    // All three classes must be present exactly once.
    if order[0] == order[1] || order[1] == order[2] || order[0] == order[2] {
        return Err(EmbedError::local(
            "local-nli",
            "id2label repeats an NLI class; head order is ambiguous",
        )
        .into());
    }

    Ok(order)
}

fn missing_index(index: usize) -> hirn_core::HirnError {
    EmbedError::local(
        "local-nli",
        format!("id2label is missing output index {index}"),
    )
    .into()
}

#[async_trait]
impl NliModel for LocalNli {
    async fn judge(
        &self,
        premise: &str,
        hypothesis: &str,
        budget: &NluBudget,
    ) -> HirnResult<Option<NliJudgment>> {
        if premise.trim().is_empty() || hypothesis.trim().is_empty() {
            return Ok(None);
        }

        let premise: String = premise.chars().take(budget.max_input_chars).collect();
        let hypothesis: String = hypothesis.chars().take(budget.max_input_chars).collect();

        // ONNX inference is CPU-bound: run it off the async runtime so a batch
        // of judgments cannot stall unrelated tasks on the same worker.
        let session = Arc::clone(&self.session);
        let tokenizer = Arc::clone(&self.tokenizer);
        let label_order = Arc::clone(&self.label_order);
        let max_length = self.max_length;
        let model_id = self.model_id.clone();

        let inference = tokio::task::spawn_blocking(move || {
            let model = Self {
                session,
                tokenizer,
                label_order,
                max_length,
                model_id,
            };
            let probabilities = model.infer(&premise, &hypothesis)?;
            Ok::<_, hirn_core::HirnError>((probabilities, *model.label_order))
        });

        let (probabilities, label_order) =
            match tokio::time::timeout(budget.timeout, inference).await {
                Ok(Ok(Ok(result))) => result,
                Ok(Ok(Err(error))) => return Err(error),
                Ok(Err(join_error)) => {
                    return Err(hirn_core::HirnError::provider(format!(
                        "local NLI inference task failed: {join_error}"
                    )));
                }
                Err(_elapsed) => {
                    super::metrics::record_abstain(
                        "nli_entailment",
                        DecisionSource::LocalModel,
                        "timeout",
                    );
                    return Ok(None);
                }
            };

        // Map head indices onto canonical (entailment, neutral, contradiction)
        // order before anything downstream reads the distribution.
        let mut canonical = [0.0f32; 3];
        for (index, label) in label_order.iter().enumerate() {
            let slot = match label {
                NliLabel::Entailment => 0,
                NliLabel::Neutral => 1,
                NliLabel::Contradiction => 2,
            };
            canonical[slot] = probabilities[index];
        }

        let (best, confidence) = canonical
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(index, probability)| (index, *probability))
            .expect("three-class distribution");
        let label = match best {
            0 => NliLabel::Entailment,
            1 => NliLabel::Neutral,
            _ => NliLabel::Contradiction,
        };

        Ok(Some(NliJudgment {
            label,
            confidence,
            distribution: Some(canonical),
            source: DecisionSource::LocalModel,
        }))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_deberta_mnli_head_order() {
        let config = r#"{"id2label":{"0":"contradiction","1":"entailment","2":"neutral"}}"#;
        assert_eq!(
            parse_label_order(config).unwrap(),
            [
                NliLabel::Contradiction,
                NliLabel::Entailment,
                NliLabel::Neutral
            ]
        );
    }

    #[test]
    fn parses_alternate_head_order() {
        let config = r#"{"id2label":{"0":"entailment","1":"neutral","2":"contradiction"}}"#;
        assert_eq!(
            parse_label_order(config).unwrap(),
            [
                NliLabel::Entailment,
                NliLabel::Neutral,
                NliLabel::Contradiction
            ]
        );
    }

    #[test]
    fn rejects_configs_that_cannot_pin_head_order() {
        // No mapping at all — guessing here silently inverts entailment.
        assert!(parse_label_order(r#"{"architectures":["DebertaV2"]}"#).is_err());
        // Two-class head (a relevance model, not NLI).
        assert!(parse_label_order(r#"{"id2label":{"0":"entailment","1":"neutral"}}"#).is_err());
        // Unknown class name.
        assert!(
            parse_label_order(r#"{"id2label":{"0":"entailment","1":"neutral","2":"LABEL_2"}}"#)
                .is_err()
        );
        // Duplicate class.
        assert!(
            parse_label_order(r#"{"id2label":{"0":"neutral","1":"neutral","2":"contradiction"}}"#)
                .is_err()
        );
        // Non-numeric index.
        assert!(
            parse_label_order(
                r#"{"id2label":{"a":"entailment","1":"neutral","2":"contradiction"}}"#
            )
            .is_err()
        );
        // Not JSON.
        assert!(parse_label_order("not json").is_err());
    }

    #[test]
    fn softmax_is_normalized_and_stable() {
        let probabilities = softmax3([2.0, 1.0, 0.5]);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probabilities[0] > probabilities[1]);
        // Extreme logits must not produce NaN.
        let extreme = softmax3([1e30, -1e30, 0.0]);
        assert!(extreme.iter().all(|p| p.is_finite()));
        assert!((extreme.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        // A degenerate all-infinite input degrades to uniform.
        let degenerate = softmax3([f32::NEG_INFINITY; 3]);
        assert!(degenerate.iter().all(|p| (p - 1.0 / 3.0).abs() < 1e-6));
    }
}
