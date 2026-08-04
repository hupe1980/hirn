//! Natural-language-understanding backends.
//!
//! Concrete implementations of the [`hirn_core::nlu`] contracts:
//!
//! | Backend | Contract | Cost | Strength |
//! |---|---|---|---|
//! | [`LlmTextClassifier`] | [`TextClassifier`](hirn_core::nlu::TextClassifier) | one generation call | nuanced intent, scope, any language |
//! | [`ExemplarRouter`] | [`TextClassifier`](hirn_core::nlu::TextClassifier) | one embedding call | paraphrase-robust routing, cheap |
//! | [`HybridClassifier`] | [`TextClassifier`](hirn_core::nlu::TextClassifier) | chain | ordered fallback with measured fallback rate |
//! | [`LlmNli`] | [`NliModel`](hirn_core::nlu::NliModel) | one call per pair | entailment for contradiction/polarity |
//! | [`LocalNli`] | [`NliModel`](hirn_core::nlu::NliModel) | local ONNX | write-path volume, no data egress |
//! | [`LlmEventExtractor`] | [`EventExtractor`](hirn_core::nlu::EventExtractor) | one call per record | passive voice, negation-aware SVO |
//! | [`LlmEntityExtractor`] | [`EntityExtractor`](hirn_core::embed::EntityExtractor) | one call per record | typed, case-independent NER |
//! | [`LlmPreferenceExtractor`] | [`PreferenceExtractor`](hirn_core::preference::PreferenceExtractor) | one call per message | indirect and non-English preference phrasing |
//! | [`LlmTemporalExtractor`] | [`TemporalExtractor`] | one call per record | event time, precision, and ongoing/completed/timeless state |
//!
//! Assemble them with [`HybridClassifier`]; see its docs for the fallback
//! contract every hirn decision path follows.

pub(crate) mod metrics;

mod entity_extractor;
mod event_extractor;
mod exemplar_router;
mod hybrid;
mod llm_classifier;
mod nli;
mod preference_extractor;
mod temporal_extractor;

#[cfg(feature = "cross-encoder")]
mod local_nli;

pub use entity_extractor::LlmEntityExtractor;
pub use event_extractor::LlmEventExtractor;
pub use exemplar_router::{DEFAULT_EXEMPLAR_TEMPERATURE, ExemplarRouter};
pub use hybrid::HybridClassifier;
pub use llm_classifier::LlmTextClassifier;
pub use metrics::{
    NLU_ABSTENTIONS_TOTAL, NLU_CONFIDENCE, NLU_DECISION_SECONDS, NLU_DECISIONS_TOTAL,
};
pub use nli::{LlmNli, NLI_TASK, nli_input};
pub use preference_extractor::LlmPreferenceExtractor;
pub use temporal_extractor::{LlmTemporalExtractor, TemporalExtractor};

#[cfg(feature = "cross-encoder")]
pub use local_nli::LocalNli;
