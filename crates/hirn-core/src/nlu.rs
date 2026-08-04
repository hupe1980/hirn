//! Natural-language understanding: model-backed decision contracts.
//!
//! Not to be confused with [`crate::semantic`], which is the *semantic memory
//! layer* (concept records and their revisions). This module is about
//! **understanding text**: the contract every meaning-dependent decision in
//! hirn runs through, whatever memory layer it serves.
//!
//! Cognitive decisions that depend on *meaning* — what a query is asking for,
//! how evidence relates to a belief, what kind of knowledge a thread carries,
//! whether two statements conflict — cannot be settled by lowercasing a string
//! and scanning it for cue words. Cue lists miss paraphrase, implicit intent,
//! scoped negation, passive voice, and every language they were not written
//! for, and they grow into unmeasurable policy that nobody can calibrate.
//!
//! This module defines the contract those decisions run through:
//!
//! - [`ClassificationTask`] — a named decision surface with typed labels,
//!   natural-language label descriptions, and exemplars. One task definition
//!   drives *both* the LLM prompt/JSON schema and the embedding exemplar
//!   router, so the two backends can never disagree about the label set.
//! - [`TextClassifier`] — the backend-agnostic classification trait.
//! - [`NliModel`] — entailment/contradiction judgment for relation, polarity,
//!   and negation-scope decisions.
//! - [`EventExtractor`] — typed subject/verb/object extraction.
//! - [`Classification`] / [`DecisionSource`] — the result carries *which*
//!   backend decided and how confident it is, so fallback rate and calibration
//!   are measurable rather than assumed.
//!
//! # The hybrid policy
//!
//! Nothing here says "LLM everywhere". The contract is deliberately layered:
//!
//! 1. **Embeddings** for cheap similarity, deduplication, and exemplar routing
//!    where a calibrated distance is meaningful.
//! 2. **Temperature-zero structured LLM output** for nuanced intent and typed
//!    extraction, constrained by [`ClassificationTask::json_schema`].
//! 3. **Local NLI/NER models** for high-volume relation, contradiction, and
//!    negation-scope work where latency or privacy precludes a remote model.
//! 4. **Deterministic code** for protocol syntax, schema validation, security
//!    boundaries, exact identifiers, explicit user options, and the fail-safe
//!    fallback that keeps the system working with no provider configured.
//!
//! Every consumer of this module keeps a provider-free fallback. The fallback
//! is a *floor*, not the primary decision surface, and each decision records
//! [`Classification::source`] so the fallback rate is a metric.
//!
//! # Safety properties
//!
//! - Model input is sanitized with [`crate::sanitize::sanitize_for_llm`] by
//!   [`ClassificationTask::user_prompt`] before it reaches a provider.
//! - [`ClassificationTask::parse_response`] is strict: an unknown label, an
//!   out-of-range confidence, or malformed JSON yields `None`. A confused model
//!   can never widen a decision — callers fall back instead.
//! - Confidence is calibrated ([`Calibration`]) and gated
//!   ([`NluBudget::min_confidence`]) before a decision is acted on.
//!
//! # Calibrating confidence
//!
//! A gate is only meaningful once a reported 0.8 means "right about 80% of the
//! time". [`Calibration::evaluate`] measures that against labeled samples
//! (expected calibration error, Brier score, a reliability diagram), and
//! [`Calibration::fit`] returns the affine map that best predicts observed
//! correctness:
//!
//! ```
//! # use hirn_core::nlu::{Calibration, CalibrationSample};
//! # fn demo(samples: &[CalibrationSample]) {
//! let deployed = Calibration::default();
//! let before = deployed.evaluate(samples);
//!
//! if let Some(fitted) = deployed.fit(samples) {
//!     let after = fitted.evaluate(samples);
//!     // Only adopt a fit that actually improves calibration.
//!     if after.expected_calibration_error < before.expected_calibration_error {
//!         // …write `fitted` into `nlu.llm_calibration`.
//!     }
//! }
//! # }
//! ```

use std::fmt;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::HirnResult;

// ── Task definition ──────────────────────────────────────────────────────

/// One label of a [`ClassificationTask`].
///
/// `description` is written for a model to read — it appears verbatim in the
/// LLM prompt. `exemplars` are short, representative inputs used by the
/// embedding router as label centroids; they are also shown to the LLM as
/// few-shot anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelSpec {
    /// Stable machine-readable label (snake_case; appears in the JSON schema).
    pub name: &'static str,
    /// One-line natural-language definition shown to the model.
    pub description: &'static str,
    /// Representative inputs used as embedding centroids and few-shot anchors.
    pub exemplars: &'static [&'static str],
}

/// A named semantic decision surface.
///
/// A task is defined once as a `const` and shared by every backend, so the LLM
/// prompt, the JSON schema, the embedding exemplar centroids, and the
/// deterministic fallback all agree on the same label set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassificationTask {
    /// Stable task identifier; used as a metrics label and exemplar cache key.
    pub name: &'static str,
    /// Instruction shown to the model describing what to decide.
    pub instruction: &'static str,
    /// The complete label set. Must be non-empty.
    pub labels: &'static [LabelSpec],
    /// Label used when no backend produces an accepted decision. Must be one
    /// of `labels`; this is the conservative, no-op choice for the task.
    pub default_label: &'static str,
}

impl ClassificationTask {
    /// Whether `label` is part of this task's label set (exact match).
    #[must_use]
    pub fn contains(&self, label: &str) -> bool {
        self.labels.iter().any(|l| l.name == label)
    }

    /// Resolve a model-produced string to a canonical label.
    ///
    /// Accepts surrounding whitespace and any letter case; rejects everything
    /// else. Deliberately *not* fuzzy: "probably reinforces" is a parse
    /// failure, not a `reinforces` decision.
    #[must_use]
    pub fn resolve(&self, raw: &str) -> Option<&'static str> {
        let trimmed = raw.trim();
        self.labels
            .iter()
            .find(|l| l.name.eq_ignore_ascii_case(trimmed))
            .map(|l| l.name)
    }

    /// The label to fall back to. Panics only on a malformed `const` task,
    /// which the [`Self::is_well_formed`] test guard catches at build time.
    #[must_use]
    pub fn default_label(&self) -> &'static str {
        debug_assert!(
            self.contains(self.default_label),
            "task {} default_label {} is not in its label set",
            self.name,
            self.default_label
        );
        self.default_label
    }

    /// Structural self-check used by tests to validate `const` task definitions.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        if self.labels.is_empty() || !self.contains(self.default_label) {
            return false;
        }
        // Label names must be unique and non-empty.
        for (i, label) in self.labels.iter().enumerate() {
            if label.name.is_empty() || label.description.is_empty() {
                return false;
            }
            if self.labels[..i].iter().any(|l| l.name == label.name) {
                return false;
            }
        }
        true
    }

    /// Strict JSON Schema for a single decision, passed to the provider as
    /// [`crate::embed::ResponseFormat::JsonSchema`].
    ///
    /// The `label` field is an enum over exactly this task's labels, so a
    /// schema-enforcing provider cannot return an unknown label at all, and a
    /// non-enforcing provider is caught by [`Self::parse_response`].
    #[must_use]
    pub fn json_schema(&self) -> String {
        let labels: Vec<&str> = self.labels.iter().map(|l| l.name).collect();
        let schema = serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["label", "confidence", "rationale"],
            "properties": {
                "label": {
                    "type": "string",
                    "enum": labels,
                    "description": self.instruction,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Probability that the chosen label is correct.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One short sentence justifying the label.",
                },
            },
        });
        schema.to_string()
    }

    /// System prompt: the task instruction plus the label definitions.
    #[must_use]
    pub fn system_prompt(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str(self.instruction);
        out.push_str(
            "\n\nAnswer with a single JSON object and nothing else: \
             {\"label\": <one label below>, \"confidence\": <0.0-1.0>, \
             \"rationale\": <one short sentence>}.\n\
             Judge meaning, not wording: paraphrase, implicit intent, passive \
             voice, and any language count the same as literal phrasing. \
             Report honest confidence — use a low value when the input is \
             ambiguous rather than guessing a specific label.\n\nLabels:\n",
        );
        for label in self.labels {
            out.push_str("- \"");
            out.push_str(label.name);
            out.push_str("\": ");
            out.push_str(label.description);
            if let Some(example) = label.exemplars.first() {
                out.push_str(" (e.g. \"");
                out.push_str(example);
                out.push('"');
                out.push(')');
            }
            out.push('\n');
        }
        out
    }

    /// User prompt for one input, with optional extra context.
    ///
    /// Both `text` and `context` pass through
    /// [`crate::sanitize::sanitize_for_llm`], and each is truncated to
    /// `max_chars` on a character boundary so a hostile or oversized record
    /// cannot blow the token budget.
    #[must_use]
    pub fn user_prompt(&self, text: &str, context: Option<&str>, max_chars: usize) -> String {
        let clean = |value: &str| -> String {
            crate::sanitize::sanitize_for_llm(value)
                .chars()
                .take(max_chars)
                .collect()
        };
        match context {
            Some(context) if !context.trim().is_empty() => {
                format!("Context:\n{}\n\nInput:\n{}", clean(context), clean(text))
            }
            _ => format!("Input:\n{}", clean(text)),
        }
    }

    /// Strictly parse a model response into a [`Classification`].
    ///
    /// Accepts a bare JSON object, or one wrapped in a ```` ```json ```` fence
    /// (some providers add fences even under a JSON response format). Returns
    /// `None` — never a guess — when the payload is not valid JSON, the label
    /// is not in this task's set, or `confidence` is missing or outside
    /// `[0, 1]`. Callers treat `None` as "this backend abstained" and fall
    /// through to the next one.
    #[must_use]
    pub fn parse_response(&self, raw: &str, source: DecisionSource) -> Option<Classification> {
        let payload = extract_json_object(raw)?;
        let value: serde_json::Value = serde_json::from_str(payload).ok()?;
        let label = self.resolve(value.get("label")?.as_str()?)?;
        let confidence = value.get("confidence")?.as_f64()? as f32;
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return None;
        }
        let rationale = value
            .get("rationale")
            .and_then(serde_json::Value::as_str)
            .map(|r| r.trim().chars().take(400).collect::<String>())
            .filter(|r| !r.is_empty());

        Some(Classification {
            label: label.to_string(),
            confidence,
            rationale,
            source,
            scores: vec![(label.to_string(), confidence)],
        })
    }
}

/// Find the outermost JSON object in a model response.
///
/// Handles bare objects, fenced blocks, and leading/trailing prose. Brace
/// counting is string-aware so a `{` inside a quoted rationale does not
/// unbalance the scan.
fn extract_json_object(raw: &str) -> Option<&str> {
    let bytes = raw.as_bytes();
    let start = raw.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&raw[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

// ── Decision result ──────────────────────────────────────────────────────

/// Which backend produced a decision.
///
/// Recorded on every [`Classification`] so fallback rate, per-backend
/// accuracy, and calibration can be measured instead of assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionSource {
    /// A structured LLM call.
    Model,
    /// Embedding similarity against labeled exemplars.
    Embedding,
    /// A local NLI / NER model.
    LocalModel,
    /// The provider-free deterministic fallback.
    Heuristic,
}

impl DecisionSource {
    /// Stable machine-readable label (metrics dimension).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Embedding => "embedding",
            Self::LocalModel => "local_model",
            Self::Heuristic => "heuristic",
        }
    }

    /// Whether this source is a model-backed primary (as opposed to the
    /// deterministic floor).
    #[must_use]
    pub const fn is_model_backed(self) -> bool {
        !matches!(self, Self::Heuristic)
    }
}

impl fmt::Display for DecisionSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One classification decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    /// The chosen label; always a member of the task's label set.
    pub label: String,
    /// Calibrated confidence in `[0, 1]`.
    pub confidence: f32,
    /// One-sentence justification when the backend supplied one.
    pub rationale: Option<String>,
    /// Which backend decided.
    pub source: DecisionSource,
    /// Per-label scores when the backend produces a distribution (embedding
    /// routing, local models). Single-entry for point decisions.
    pub scores: Vec<(String, f32)>,
}

impl Classification {
    /// A point decision with no distribution.
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        confidence: f32,
        source: DecisionSource,
        rationale: Option<String>,
    ) -> Self {
        let label = label.into();
        let confidence = confidence.clamp(0.0, 1.0);
        Self {
            scores: vec![(label.clone(), confidence)],
            label,
            confidence,
            rationale,
            source,
        }
    }

    /// Whether this decision clears an acceptance gate.
    #[must_use]
    pub fn accepted(&self, min_confidence: f32) -> bool {
        self.confidence >= min_confidence
    }

    /// Score for `label`, or `0.0` when the backend did not score it.
    #[must_use]
    pub fn score_for(&self, label: &str) -> f32 {
        self.scores
            .iter()
            .find(|(name, _)| name == label)
            .map_or(0.0, |(_, score)| *score)
    }

    /// Attach a per-label distribution, sorting it strongest-first.
    #[must_use]
    pub fn with_scores(mut self, mut scores: Vec<(String, f32)>) -> Self {
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        self.scores = scores;
        self
    }
}

// ── Calibration and budget ───────────────────────────────────────────────

/// Confidence calibration applied to a backend's raw score.
///
/// Raw scores are not probabilities: LLM self-reported confidence is
/// systematically over-confident, and a cosine-similarity softmax is peaked or
/// flat depending on the embedding model. `temperature` reshapes a
/// distribution before the argmax is scored; `scale` and `floor` map the raw
/// score onto the calibrated range that the acceptance gate is tuned against.
///
/// Defaults are identity (`temperature = 1`, `scale = 1`, `floor = 0`) so an
/// uncalibrated deployment behaves exactly like the raw backend; deployments
/// fit `scale` from a labeled sample.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Calibration {
    /// Softmax temperature for distribution-producing backends. `> 1` flattens
    /// (less confident), `< 1` sharpens. Must be `> 0`.
    pub temperature: f32,
    /// Multiplicative shrink applied to the winning score.
    pub scale: f32,
    /// Additive floor so a calibrated score never collapses to zero.
    pub floor: f32,
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            scale: 1.0,
            floor: 0.0,
        }
    }
}

impl Calibration {
    /// Apply the affine calibration to a raw `[0, 1]` score.
    #[must_use]
    pub fn apply(&self, raw: f32) -> f32 {
        if !raw.is_finite() {
            return self.floor.clamp(0.0, 1.0);
        }
        (raw.clamp(0.0, 1.0) * self.scale + self.floor).clamp(0.0, 1.0)
    }

    /// Temperature-scaled softmax over raw scores.
    ///
    /// Returns an empty vector for empty input. Guards against a non-positive
    /// or non-finite temperature by falling back to `1.0`, and subtracts the
    /// max before exponentiating so large logits cannot overflow.
    #[must_use]
    pub fn softmax(&self, scores: &[f32]) -> Vec<f32> {
        if scores.is_empty() {
            return Vec::new();
        }
        let temperature = if self.temperature.is_finite() && self.temperature > 0.0 {
            self.temperature
        } else {
            1.0
        };
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return vec![1.0 / scores.len() as f32; scores.len()];
        }
        let exps: Vec<f32> = scores
            .iter()
            .map(|s| ((s - max) / temperature).exp())
            .collect();
        let total: f32 = exps.iter().sum();
        if total <= f32::EPSILON {
            return vec![1.0 / scores.len() as f32; scores.len()];
        }
        exps.into_iter().map(|e| e / total).collect()
    }

    /// Measure how well this calibration's outputs match observed correctness.
    ///
    /// Confidence is only meaningful if a reported 0.8 means "right about 80%
    /// of the time". Until that is measured, a confidence gate is an arbitrary
    /// number: tightening `min_confidence` against uncalibrated scores raises
    /// the fallback rate without improving quality.
    ///
    /// `samples` are `(raw_confidence, was_correct)` pairs collected from a
    /// labeled evaluation set. Scores are passed through [`Self::apply`] first,
    /// so the report describes the calibration as deployed.
    #[must_use]
    pub fn evaluate(&self, samples: &[CalibrationSample]) -> CalibrationReport {
        let usable: Vec<(f32, bool)> = samples
            .iter()
            .filter(|s| s.confidence.is_finite())
            .map(|s| (self.apply(s.confidence), s.correct))
            .collect();

        if usable.is_empty() {
            return CalibrationReport::default();
        }

        let n = usable.len() as f32;
        let mean_confidence = usable.iter().map(|(c, _)| *c).sum::<f32>() / n;
        let accuracy = usable.iter().filter(|(_, ok)| *ok).count() as f32 / n;
        let brier = usable
            .iter()
            .map(|(c, ok)| {
                let outcome = if *ok { 1.0 } else { 0.0 };
                (c - outcome) * (c - outcome)
            })
            .sum::<f32>()
            / n;

        // Expected calibration error over equal-width confidence bins:
        // the sample-weighted mean gap between a bin's mean confidence and its
        // observed accuracy.
        let mut bins = vec![CalibrationBin::default(); CALIBRATION_BINS];
        for (confidence, correct) in &usable {
            let index = ((confidence * CALIBRATION_BINS as f32) as usize).min(CALIBRATION_BINS - 1);
            let bin = &mut bins[index];
            bin.count += 1;
            bin.mean_confidence += confidence;
            if *correct {
                bin.accuracy += 1.0;
            }
        }
        let mut expected_calibration_error = 0.0f32;
        for bin in &mut bins {
            if bin.count == 0 {
                continue;
            }
            let count = bin.count as f32;
            bin.mean_confidence /= count;
            bin.accuracy /= count;
            expected_calibration_error += (count / n) * (bin.mean_confidence - bin.accuracy).abs();
        }

        CalibrationReport {
            samples: usable.len(),
            accuracy,
            mean_confidence,
            expected_calibration_error,
            brier_score: brier,
            bins,
        }
    }

    /// Fit `scale` and `floor` from labeled samples by least squares.
    ///
    /// The executable form of "fit the calibration against a labeled sample":
    /// it regresses observed correctness on raw confidence and returns the
    /// affine map that best predicts it, leaving `temperature` untouched (that
    /// shapes a distribution *before* the argmax, which outcome labels say
    /// nothing about).
    ///
    /// Returns `None` when no map is safe to deploy; use [`Self::fit_report`]
    /// when you need to know *why*.
    ///
    /// The fit is linear, not isotonic or Platt: it corrects the systematic
    /// over-confidence these backends exhibit without pretending to model a
    /// non-monotone response curve. Verify with [`Self::evaluate`] before and
    /// after — a fit that does not lower expected calibration error should not
    /// be deployed.
    #[must_use]
    pub fn fit(&self, samples: &[CalibrationSample]) -> Option<Self> {
        match self.fit_report(samples) {
            CalibrationFit::Fitted(calibration) => Some(calibration),
            _ => None,
        }
    }

    /// [`Self::fit`] with the reason a fit was refused.
    ///
    /// Two refusals matter, and both are cases where the arithmetic happily
    /// produces a number that must not be deployed:
    ///
    /// - **Too few samples.** A map fitted from a handful of observations looks
    ///   authoritative and is noise.
    /// - **A degenerate slope.** When confidence barely varies, or barely
    ///   relates to correctness, least squares collapses to a *constant*
    ///   predictor: `scale ≈ 0`, `floor ≈ observed accuracy`. That genuinely
    ///   minimizes squared error and genuinely lowers expected calibration
    ///   error — and writing it into config reports the same confidence for
    ///   every decision, which makes
    ///   [`NluBudget::min_confidence`] inert and lets every decision through
    ///   the gate unchallenged. A backend whose confidence carries no
    ///   information should not have that confidence laundered into a constant
    ///   that always passes; collect a sample with more spread, or stop gating
    ///   on that backend.
    #[must_use]
    pub fn fit_report(&self, samples: &[CalibrationSample]) -> CalibrationFit {
        let usable: Vec<(f32, f32)> = samples
            .iter()
            .filter(|s| s.confidence.is_finite())
            .map(|s| {
                (
                    s.confidence.clamp(0.0, 1.0),
                    if s.correct { 1.0 } else { 0.0 },
                )
            })
            .collect();
        if usable.len() < MIN_CALIBRATION_SAMPLES {
            return CalibrationFit::TooFewSamples {
                have: usable.len(),
                need: MIN_CALIBRATION_SAMPLES,
            };
        }

        let n = usable.len() as f32;
        let mean_x = usable.iter().map(|(x, _)| *x).sum::<f32>() / n;
        let mean_y = usable.iter().map(|(_, y)| *y).sum::<f32>() / n;
        let variance = usable
            .iter()
            .map(|(x, _)| (x - mean_x) * (x - mean_x))
            .sum::<f32>()
            / n;

        if variance <= f32::EPSILON {
            return CalibrationFit::Degenerate {
                slope: 0.0,
                observed_accuracy: mean_y,
            };
        }

        let covariance = usable
            .iter()
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum::<f32>()
            / n;
        let slope = covariance / variance;

        // A slope at or below the floor — including a negative one, where
        // confidence is anti-correlated with correctness — carries no usable
        // discrimination. Neither clamping it to zero nor inverting it would be
        // honest: the sample says this confidence does not separate right from
        // wrong.
        if slope < MIN_CALIBRATION_SLOPE {
            return CalibrationFit::Degenerate {
                slope,
                observed_accuracy: mean_y,
            };
        }

        // `NluConfig::validate` requires both in [0, 1].
        CalibrationFit::Fitted(Self {
            temperature: self.temperature,
            scale: slope.clamp(0.0, 1.0),
            floor: (mean_y - slope * mean_x).clamp(0.0, 1.0),
        })
    }
}

/// Outcome of fitting a calibration from labeled samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum CalibrationFit {
    /// A map that preserves a usable confidence gate.
    Fitted(Calibration),
    /// Not enough labeled observations to fit anything but noise.
    TooFewSamples { have: usize, need: usize },
    /// Least squares collapsed to a constant predictor. Deploying it would
    /// report one confidence for every decision and make the gate inert.
    Degenerate { slope: f32, observed_accuracy: f32 },
}

impl CalibrationFit {
    /// A one-line explanation suitable for an operator-facing report.
    #[must_use]
    pub fn explain(&self) -> String {
        match self {
            Self::Fitted(calibration) => format!(
                "fitted scale={:.4} floor={:.4}",
                calibration.scale, calibration.floor
            ),
            Self::TooFewSamples { have, need } => {
                format!("not fitted: {have} samples, {need} required")
            }
            Self::Degenerate {
                slope,
                observed_accuracy,
            } => format!(
                "not fitted: slope {slope:.4} below {MIN_CALIBRATION_SLOPE:.2} — confidence \
                 does not separate correct from incorrect on this sample (accuracy \
                 {observed_accuracy:.4}); a constant map would disable the gate"
            ),
        }
    }

    /// The fitted calibration, if one was produced.
    #[must_use]
    pub const fn calibration(&self) -> Option<&Calibration> {
        match self {
            Self::Fitted(calibration) => Some(calibration),
            _ => None,
        }
    }
}

/// Number of equal-width bins used for expected calibration error.
const CALIBRATION_BINS: usize = 10;

/// Minimum labeled samples required before [`Calibration::fit`] will fit.
pub const MIN_CALIBRATION_SAMPLES: usize = 30;

/// Minimum regression slope a fit must have to be deployable.
///
/// Below this, the map is effectively constant: it reports the same confidence
/// whatever the backend said, so the acceptance gate stops discriminating and
/// admits everything.
pub const MIN_CALIBRATION_SLOPE: f32 = 0.05;

/// One labeled observation: what the backend reported, and whether it was right.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// The backend's raw (pre-calibration) confidence.
    pub confidence: f32,
    /// Whether the decision matched the label.
    pub correct: bool,
}

impl CalibrationSample {
    /// A labeled observation.
    #[must_use]
    pub const fn new(confidence: f32, correct: bool) -> Self {
        Self {
            confidence,
            correct,
        }
    }
}

/// One reliability-diagram bin.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub count: usize,
    /// Mean calibrated confidence of the bin's samples.
    pub mean_confidence: f32,
    /// Observed accuracy of the bin's samples.
    pub accuracy: f32,
}

/// How well calibrated a backend's confidence is on a labeled sample.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub samples: usize,
    /// Fraction of decisions that were correct.
    pub accuracy: f32,
    /// Mean calibrated confidence. Well above `accuracy` means over-confident.
    pub mean_confidence: f32,
    /// Sample-weighted mean gap between per-bin confidence and accuracy.
    /// Lower is better; `0.0` is perfect calibration.
    pub expected_calibration_error: f32,
    /// Mean squared error against the 0/1 outcome. Captures sharpness as well
    /// as calibration, so a model that is always 50%-confident scores poorly
    /// here even though its calibration error can be near zero.
    pub brier_score: f32,
    /// Reliability diagram, one entry per equal-width confidence bin.
    pub bins: Vec<CalibrationBin>,
}

impl CalibrationReport {
    /// Whether confidence systematically exceeds observed accuracy by more
    /// than `tolerance` — the over-confidence that makes a confidence gate
    /// admit decisions it should have deferred.
    #[must_use]
    pub fn is_overconfident(&self, tolerance: f32) -> bool {
        self.samples > 0 && (self.mean_confidence - self.accuracy) > tolerance
    }
}

/// Time, token, and confidence budget for one semantic decision.
///
/// Every model-backed path is bounded: a provider that hangs, rambles, or
/// answers with low confidence must degrade to the next backend rather than
/// stall the caller or push an unreliable decision downstream.
///
/// **`min_confidence` is the weakest of those bounds.** It only fires when a
/// backend *reports* doubt, and measurement shows LLM backends often do not:
/// on hirn's query-intent set every `gpt-4o-mini` decision landed in
/// `[0.80, 0.90]`, including the one that was wrong. Treat the timeout, the
/// schema constraint, and the strict label parse as the load-bearing
/// protections; treat the confidence gate as a cheap extra filter that catches
/// a backend which does express uncertainty, not as a defence against a
/// confidently wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NluBudget {
    /// Hard wall-clock deadline for one provider call.
    #[serde(with = "duration_millis")]
    pub timeout: Duration,
    /// Completion-token ceiling for one provider call.
    pub max_tokens: u32,
    /// Calibrated confidence a decision must reach to be acted on. Below this,
    /// the caller falls through to the next backend.
    pub min_confidence: f32,
    /// Maximum characters of input text passed to a provider.
    pub max_input_chars: usize,
}

impl Default for NluBudget {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(2),
            max_tokens: 200,
            min_confidence: 0.55,
            max_input_chars: 2_000,
        }
    }
}

impl NluBudget {
    /// Validate the budget's invariants.
    ///
    /// # Errors
    /// Returns [`crate::HirnError::InvalidInput`] when the timeout is zero,
    /// the token ceiling is zero, the input ceiling is zero, or
    /// `min_confidence` is outside `[0, 1]`.
    pub fn validate(&self) -> HirnResult<()> {
        if self.timeout.is_zero() {
            return Err(crate::HirnError::InvalidInput(
                "semantic budget timeout must be non-zero".into(),
            ));
        }
        if self.max_tokens == 0 {
            return Err(crate::HirnError::InvalidInput(
                "semantic budget max_tokens must be non-zero".into(),
            ));
        }
        if self.max_input_chars == 0 {
            return Err(crate::HirnError::InvalidInput(
                "semantic budget max_input_chars must be non-zero".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(crate::HirnError::InvalidInput(
                "semantic budget min_confidence must be in [0.0, 1.0]".into(),
            ));
        }
        Ok(())
    }
}

/// Serde helper: `Duration` as whole milliseconds.
mod duration_millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(value: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(value.as_millis().min(u128::from(u64::MAX)) as u64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        Ok(Duration::from_millis(u64::deserialize(d)?))
    }
}

// ── Classifier trait ─────────────────────────────────────────────────────

/// A backend that decides a [`ClassificationTask`] for one input.
///
/// Implementations must be bounded by the supplied [`NluBudget`] and must
/// return `Ok(None)` — not an error and not a guess — when they cannot produce
/// a decision they stand behind (malformed output, unknown label, confidence
/// below the gate). `Err` is reserved for transport-level failure the caller
/// may want to log; both cases fall through to the next backend.
#[async_trait]
pub trait TextClassifier: Send + Sync {
    /// Classify `text` under `task`, optionally with extra `context`.
    async fn classify(
        &self,
        task: &ClassificationTask,
        text: &str,
        context: Option<&str>,
        budget: &NluBudget,
    ) -> HirnResult<Option<Classification>>;

    /// Stable identifier of the deciding backend (model id or router name).
    fn backend_id(&self) -> &str;

    /// Which source this backend reports its decisions as.
    fn source(&self) -> DecisionSource;
}

// ── NLI ──────────────────────────────────────────────────────────────────

/// Natural-language-inference relation between a premise and a hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NliLabel {
    /// The premise supports the hypothesis.
    Entailment,
    /// The premise neither supports nor conflicts with the hypothesis.
    Neutral,
    /// The premise conflicts with the hypothesis.
    Contradiction,
}

impl NliLabel {
    /// Stable machine-readable label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Entailment => "entailment",
            Self::Neutral => "neutral",
            Self::Contradiction => "contradiction",
        }
    }

    /// Parse a canonical NLI label (case-insensitive).
    #[must_use]
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "entailment" | "entails" => Some(Self::Entailment),
            "neutral" => Some(Self::Neutral),
            "contradiction" | "contradicts" => Some(Self::Contradiction),
            _ => None,
        }
    }
}

impl fmt::Display for NliLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One NLI judgment with calibrated confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NliJudgment {
    pub label: NliLabel,
    /// Calibrated probability of `label`.
    pub confidence: f32,
    /// Full three-way distribution `(entailment, neutral, contradiction)` when
    /// the model produces one.
    pub distribution: Option<[f32; 3]>,
    pub source: DecisionSource,
}

impl NliJudgment {
    /// A judgment with no distribution.
    #[must_use]
    pub fn point(label: NliLabel, confidence: f32, source: DecisionSource) -> Self {
        Self {
            label,
            confidence: confidence.clamp(0.0, 1.0),
            distribution: None,
            source,
        }
    }

    /// Whether this judgment clears an acceptance gate.
    #[must_use]
    pub fn accepted(&self, min_confidence: f32) -> bool {
        self.confidence >= min_confidence
    }
}

/// A model that judges entailment between two statements.
///
/// This is the semantic replacement for negation-marker mismatch as a
/// contradiction signal: negation-word presence is a *candidate generator*,
/// entailment is the decision.
#[async_trait]
pub trait NliModel: Send + Sync {
    /// Judge how `premise` relates to `hypothesis`.
    ///
    /// Returns `Ok(None)` when the model cannot produce a judgment within
    /// budget or its output is unusable.
    async fn judge(
        &self,
        premise: &str,
        hypothesis: &str,
        budget: &NluBudget,
    ) -> HirnResult<Option<NliJudgment>>;

    /// Stable model identifier.
    fn model_id(&self) -> &str;
}

// ── Typed event extraction ───────────────────────────────────────────────

/// A typed subject–verb–object event with optional temporal and spatial scope.
///
/// This is the structured-extraction counterpart to regex SVO scraping: the
/// same shape, but produced by a model that understands passive voice
/// ("the release was deployed by Alice"), coreference, and clause scope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredEvent {
    pub subject: String,
    pub verb: String,
    pub object: String,
    /// Start of the event's temporal scope, as written in the source text.
    pub time_start: Option<String>,
    /// End of the event's temporal scope, as written in the source text.
    pub time_end: Option<String>,
    pub location: Option<String>,
    /// Extraction confidence in `[0, 1]`.
    pub confidence: f32,
    /// Whether the source asserts the event did *not* happen. Regex extraction
    /// cannot represent this, which is why negated clauses previously entered
    /// the event store as positive assertions.
    pub negated: bool,
    pub source: DecisionSource,
}

impl StructuredEvent {
    /// Whether the event has the three fields that make it storable.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        !self.subject.trim().is_empty()
            && !self.verb.trim().is_empty()
            && !self.object.trim().is_empty()
    }
}

/// A backend that extracts typed events from unstructured text.
#[async_trait]
pub trait EventExtractor: Send + Sync {
    /// Extract events from `text` within `budget`.
    async fn extract_events(
        &self,
        text: &str,
        budget: &NluBudget,
    ) -> HirnResult<Vec<StructuredEvent>>;

    /// Stable model identifier.
    fn model_id(&self) -> &str;

    /// Which source this backend reports its extractions as.
    fn source(&self) -> DecisionSource;
}

// ── Similarity helpers ───────────────────────────────────────────────────

/// Cosine similarity of two vectors, or `0.0` when either is empty or the
/// lengths differ.
///
/// Shared by the embedding router, reflection gating, and summary dedup so
/// every embedding-similarity threshold in the system is measured the same way.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= f32::EPSILON || norm_b <= f32::EPSILON {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TASK: ClassificationTask = ClassificationTask {
        name: "test_task",
        instruction: "Decide whether the input is a greeting or a farewell.",
        labels: &[
            LabelSpec {
                name: "greeting",
                description: "The speaker is starting a conversation.",
                exemplars: &["hello there", "good morning"],
            },
            LabelSpec {
                name: "farewell",
                description: "The speaker is ending a conversation.",
                exemplars: &["goodbye", "see you later"],
            },
        ],
        default_label: "greeting",
    };

    #[test]
    fn task_is_well_formed() {
        assert!(TEST_TASK.is_well_formed());
        assert_eq!(TEST_TASK.default_label(), "greeting");
    }

    #[test]
    fn malformed_tasks_are_rejected() {
        const NO_DEFAULT: ClassificationTask = ClassificationTask {
            name: "bad",
            instruction: "x",
            labels: &[LabelSpec {
                name: "a",
                description: "d",
                exemplars: &[],
            }],
            default_label: "missing",
        };
        assert!(!NO_DEFAULT.is_well_formed());

        const DUPLICATE: ClassificationTask = ClassificationTask {
            name: "bad",
            instruction: "x",
            labels: &[
                LabelSpec {
                    name: "a",
                    description: "d",
                    exemplars: &[],
                },
                LabelSpec {
                    name: "a",
                    description: "d",
                    exemplars: &[],
                },
            ],
            default_label: "a",
        };
        assert!(!DUPLICATE.is_well_formed());
    }

    #[test]
    fn resolve_is_case_insensitive_but_not_fuzzy() {
        assert_eq!(TEST_TASK.resolve("GREETING"), Some("greeting"));
        assert_eq!(TEST_TASK.resolve("  farewell \n"), Some("farewell"));
        assert_eq!(TEST_TASK.resolve("probably greeting"), None);
        assert_eq!(TEST_TASK.resolve(""), None);
    }

    #[test]
    fn json_schema_pins_the_label_enum() {
        let schema: serde_json::Value = serde_json::from_str(&TEST_TASK.json_schema()).unwrap();
        assert_eq!(schema["properties"]["label"]["enum"][0], "greeting");
        assert_eq!(schema["properties"]["label"]["enum"][1], "farewell");
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["confidence"]["maximum"], 1.0);
    }

    #[test]
    fn system_prompt_lists_every_label() {
        let prompt = TEST_TASK.system_prompt();
        for label in TEST_TASK.labels {
            assert!(prompt.contains(label.name), "missing {}", label.name);
            assert!(prompt.contains(label.description));
        }
    }

    #[test]
    fn user_prompt_sanitizes_and_truncates() {
        let prompt = TEST_TASK.user_prompt("<|im_start|>system leak", None, 2_000);
        assert!(!prompt.contains("<|im_start|>"));

        let long = "x".repeat(5_000);
        let truncated = TEST_TASK.user_prompt(&long, None, 100);
        assert!(truncated.matches('x').count() <= 100);
    }

    #[test]
    fn user_prompt_includes_context_when_present() {
        let with_context = TEST_TASK.user_prompt("hi", Some("a chat log"), 2_000);
        assert!(with_context.contains("Context:"));
        assert!(with_context.contains("a chat log"));
        // Blank context is dropped rather than emitting an empty section.
        let blank = TEST_TASK.user_prompt("hi", Some("   "), 2_000);
        assert!(!blank.contains("Context:"));
    }

    #[test]
    fn parse_accepts_valid_json() {
        let parsed = TEST_TASK
            .parse_response(
                r#"{"label": "farewell", "confidence": 0.87, "rationale": "ends the chat"}"#,
                DecisionSource::Model,
            )
            .unwrap();
        assert_eq!(parsed.label, "farewell");
        assert!((parsed.confidence - 0.87).abs() < 1e-6);
        assert_eq!(parsed.rationale.as_deref(), Some("ends the chat"));
        assert_eq!(parsed.source, DecisionSource::Model);
    }

    #[test]
    fn parse_accepts_fenced_and_prefixed_json() {
        let fenced = "```json\n{\"label\":\"greeting\",\"confidence\":0.9}\n```";
        assert!(
            TEST_TASK
                .parse_response(fenced, DecisionSource::Model)
                .is_some()
        );

        let prefixed = "Here you go: {\"label\":\"greeting\",\"confidence\":0.9} — hope that helps";
        assert!(
            TEST_TASK
                .parse_response(prefixed, DecisionSource::Model)
                .is_some()
        );
    }

    #[test]
    fn parse_handles_braces_inside_strings() {
        let raw = r#"{"label":"greeting","confidence":0.8,"rationale":"contains { and } braces"}"#;
        let parsed = TEST_TASK
            .parse_response(raw, DecisionSource::Model)
            .unwrap();
        assert_eq!(parsed.rationale.as_deref(), Some("contains { and } braces"));
    }

    #[test]
    fn parse_rejects_malformed_output() {
        // Not JSON at all.
        assert!(
            TEST_TASK
                .parse_response("greeting!", DecisionSource::Model)
                .is_none()
        );
        // Unknown label.
        assert!(
            TEST_TASK
                .parse_response(
                    r#"{"label":"shrug","confidence":0.9}"#,
                    DecisionSource::Model
                )
                .is_none()
        );
        // Missing confidence.
        assert!(
            TEST_TASK
                .parse_response(r#"{"label":"greeting"}"#, DecisionSource::Model)
                .is_none()
        );
        // Out-of-range confidence.
        assert!(
            TEST_TASK
                .parse_response(
                    r#"{"label":"greeting","confidence":1.4}"#,
                    DecisionSource::Model
                )
                .is_none()
        );
        // Truncated JSON (provider hit the token ceiling mid-object).
        assert!(
            TEST_TASK
                .parse_response(r#"{"label":"greeting","confid"#, DecisionSource::Model)
                .is_none()
        );
        // Empty response.
        assert!(
            TEST_TASK
                .parse_response("", DecisionSource::Model)
                .is_none()
        );
    }

    #[test]
    fn calibration_defaults_to_identity() {
        let calibration = Calibration::default();
        assert!((calibration.apply(0.73) - 0.73).abs() < 1e-6);
    }

    #[test]
    fn calibration_shrinks_overconfidence() {
        let calibration = Calibration {
            temperature: 1.0,
            scale: 0.8,
            floor: 0.05,
        };
        assert!((calibration.apply(1.0) - 0.85).abs() < 1e-6);
        assert!((calibration.apply(0.0) - 0.05).abs() < 1e-6);
        // Clamped into range regardless of the raw value.
        assert!((0.0..=1.0).contains(&calibration.apply(f32::NAN)));
    }

    #[test]
    fn softmax_is_normalized_and_temperature_sensitive() {
        let sharp = Calibration {
            temperature: 0.1,
            ..Default::default()
        };
        let flat = Calibration {
            temperature: 10.0,
            ..Default::default()
        };
        let scores = [0.9f32, 0.6, 0.3];

        let sharp_probs = sharp.softmax(&scores);
        let flat_probs = flat.softmax(&scores);

        for probs in [&sharp_probs, &flat_probs] {
            let total: f32 = probs.iter().sum();
            assert!((total - 1.0).abs() < 1e-5, "softmax must normalize");
        }
        assert!(
            sharp_probs[0] > flat_probs[0],
            "low temperature must sharpen the winner"
        );
    }

    #[test]
    fn softmax_handles_degenerate_input() {
        let calibration = Calibration::default();
        assert!(calibration.softmax(&[]).is_empty());
        // A non-positive temperature must not produce NaN.
        let bad = Calibration {
            temperature: 0.0,
            ..Default::default()
        };
        let probs = bad.softmax(&[1.0, 2.0]);
        assert!(probs.iter().all(|p| p.is_finite()));
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    // ── Calibration measurement and fitting ──────────────────────────────

    /// Deterministic sample set: `confidence` reported for every item, and the
    /// first `accuracy * n` of them correct.
    fn samples_at(confidence: f32, accuracy: f32, n: usize) -> Vec<CalibrationSample> {
        let correct_count = (accuracy * n as f32).round() as usize;
        (0..n)
            .map(|i| CalibrationSample::new(confidence, i < correct_count))
            .collect()
    }

    #[test]
    fn evaluate_reports_perfect_calibration_as_zero_error() {
        // Two confidence levels, each matching its observed accuracy exactly.
        let mut samples = samples_at(0.9, 0.9, 100);
        samples.extend(samples_at(0.5, 0.5, 100));

        let report = Calibration::default().evaluate(&samples);
        assert_eq!(report.samples, 200);
        assert!(
            report.expected_calibration_error < 0.02,
            "ECE {} should be ~0 for a calibrated backend",
            report.expected_calibration_error
        );
        assert!(!report.is_overconfident(0.05));
    }

    #[test]
    fn evaluate_detects_overconfidence() {
        // Claims 0.95, right 60% of the time.
        let report = Calibration::default().evaluate(&samples_at(0.95, 0.6, 100));
        assert!(report.is_overconfident(0.1));
        assert!(
            (report.expected_calibration_error - 0.35).abs() < 0.02,
            "ECE {} should be ~|0.95 - 0.60|",
            report.expected_calibration_error
        );
        assert!((report.accuracy - 0.6).abs() < 1e-5);
    }

    #[test]
    fn evaluate_handles_empty_and_non_finite_input() {
        let report = Calibration::default().evaluate(&[]);
        assert_eq!(report.samples, 0);
        assert!(!report.is_overconfident(0.0));

        let junk = [
            CalibrationSample::new(f32::NAN, true),
            CalibrationSample::new(f32::INFINITY, false),
        ];
        assert_eq!(Calibration::default().evaluate(&junk).samples, 0);
    }

    #[test]
    fn fit_reduces_calibration_error_on_overconfident_samples() {
        // Two confidence levels, both over-stated by a consistent margin.
        let mut samples = samples_at(0.9, 0.6, 100);
        samples.extend(samples_at(0.6, 0.4, 100));

        let uncalibrated = Calibration::default();
        let before = uncalibrated.evaluate(&samples);
        let fitted = uncalibrated.fit(&samples).expect("enough samples to fit");
        let after = fitted.evaluate(&samples);

        assert!(
            after.expected_calibration_error < before.expected_calibration_error,
            "fit must improve calibration: {} -> {}",
            before.expected_calibration_error,
            after.expected_calibration_error
        );
        assert!(after.expected_calibration_error < 0.05);
        assert!(fitted.scale <= 1.0 && fitted.floor >= 0.0);
    }

    #[test]
    fn fit_preserves_temperature() {
        let base = Calibration {
            temperature: 0.07,
            scale: 1.0,
            floor: 0.0,
        };
        // Needs confidence spread, or the fit is refused as degenerate.
        let mut samples = samples_at(0.9, 0.9, 60);
        samples.extend(samples_at(0.4, 0.3, 60));
        let fitted = base.fit(&samples).unwrap();
        assert!(
            (fitted.temperature - 0.07).abs() < 1e-6,
            "temperature shapes the pre-argmax distribution and is not fit from outcomes"
        );
    }

    #[test]
    fn fit_refuses_to_learn_from_too_few_samples() {
        // A discriminating sample, just short of the minimum.
        let mut few = samples_at(0.9, 0.9, (MIN_CALIBRATION_SAMPLES - 1) / 2);
        few.extend(samples_at(0.4, 0.3, (MIN_CALIBRATION_SAMPLES - 1) / 2));
        assert!(few.len() < MIN_CALIBRATION_SAMPLES);
        assert!(
            Calibration::default().fit(&few).is_none(),
            "fitting on noise is worse than the identity default"
        );

        // The same shape, above the minimum, does fit.
        let mut enough = samples_at(0.9, 0.9, MIN_CALIBRATION_SAMPLES);
        enough.extend(samples_at(0.4, 0.3, MIN_CALIBRATION_SAMPLES));
        assert!(Calibration::default().fit(&enough).is_some());
    }

    #[test]
    fn fit_refuses_a_constant_map_that_would_disable_the_gate() {
        // No spread in the input: least squares collapses to "always predict
        // the base rate". That minimizes squared error and lowers calibration
        // error — and reports the same confidence for every decision, so
        // `min_confidence` would stop rejecting anything.
        let samples = samples_at(0.9, 0.4, 100);
        assert!(Calibration::default().fit(&samples).is_none());

        match Calibration::default().fit_report(&samples) {
            CalibrationFit::Degenerate {
                slope,
                observed_accuracy,
            } => {
                assert!(slope < MIN_CALIBRATION_SLOPE);
                assert!((observed_accuracy - 0.4).abs() < 0.02);
            }
            other => panic!("expected a degenerate fit, got {other:?}"),
        }
    }

    #[test]
    fn fit_refuses_an_anti_correlated_signal() {
        // Higher confidence, lower accuracy. Inverting it would be dishonest,
        // and clamping it to a constant would disable the gate — so neither.
        let mut samples = samples_at(0.9, 0.2, 100);
        samples.extend(samples_at(0.3, 0.8, 100));
        assert!(Calibration::default().fit(&samples).is_none());
        assert!(matches!(
            Calibration::default().fit_report(&samples),
            CalibrationFit::Degenerate { slope, .. } if slope < 0.0
        ));
    }

    #[test]
    fn a_near_perfect_backend_cannot_be_calibrated_into_always_passing() {
        // The exact distribution the live query-intent routing evaluation
        // produced against gpt-4o-mini: 35 model-backed decisions, 34 correct,
        // confidence almost always 0.9. Least squares answers `scale = 0,
        // floor = 1.0` — it lowers expected calibration error *and* reports
        // full confidence for every decision, which would make
        // `min_confidence` reject nothing. That must never be recommended.
        let mut samples = vec![CalibrationSample::new(0.9, true); 30];
        samples.extend(vec![CalibrationSample::new(0.8, true); 4]);
        samples.push(CalibrationSample::new(0.9, false));

        let report = Calibration::default().fit_report(&samples);
        assert!(
            report.calibration().is_none(),
            "a constant-confidence map must never be recommended: {}",
            report.explain()
        );
        assert!(report.explain().contains("disable the gate"));

        // The measurement itself is still available and still useful: the
        // backend is mildly *under*-confident here (0.89 stated, 0.97 actual).
        let measured = Calibration::default().evaluate(&samples);
        assert!(measured.accuracy > measured.mean_confidence);
        assert!(!measured.is_overconfident(0.0));
    }

    #[test]
    fn fit_report_explains_a_short_sample() {
        let report = Calibration::default().fit_report(&samples_at(0.9, 0.5, 5));
        assert!(matches!(
            report,
            CalibrationFit::TooFewSamples {
                have: 5,
                need: MIN_CALIBRATION_SAMPLES
            }
        ));
        assert!(report.explain().contains("5 samples"));
    }

    #[test]
    fn fitted_output_always_stays_in_range() {
        let mut samples = samples_at(1.0, 1.0, 60);
        samples.extend(samples_at(0.0, 0.0, 60));
        let fitted = Calibration::default().fit(&samples).unwrap();
        for raw in [0.0, 0.25, 0.5, 0.75, 1.0, f32::NAN] {
            let calibrated = fitted.apply(raw);
            assert!(
                (0.0..=1.0).contains(&calibrated),
                "apply({raw}) = {calibrated} out of range"
            );
        }
    }

    #[test]
    fn budget_validation_rejects_degenerate_values() {
        assert!(NluBudget::default().validate().is_ok());

        let zero_timeout = NluBudget {
            timeout: Duration::ZERO,
            ..Default::default()
        };
        assert!(zero_timeout.validate().is_err());

        let bad_confidence = NluBudget {
            min_confidence: 1.5,
            ..Default::default()
        };
        assert!(bad_confidence.validate().is_err());

        let zero_tokens = NluBudget {
            max_tokens: 0,
            ..Default::default()
        };
        assert!(zero_tokens.validate().is_err());

        let zero_chars = NluBudget {
            max_input_chars: 0,
            ..Default::default()
        };
        assert!(zero_chars.validate().is_err());
    }

    #[test]
    fn budget_round_trips_through_serde() {
        let budget = NluBudget {
            timeout: Duration::from_millis(1_500),
            max_tokens: 64,
            min_confidence: 0.6,
            max_input_chars: 512,
        };
        let json = serde_json::to_string(&budget).unwrap();
        let back: NluBudget = serde_json::from_str(&json).unwrap();
        assert_eq!(budget, back);
    }

    #[test]
    fn classification_gate_and_scores() {
        let decision =
            Classification::new("greeting", 0.4, DecisionSource::Embedding, None).with_scores(
                vec![("greeting".to_string(), 0.4), ("farewell".to_string(), 0.6)],
            );
        assert!(!decision.accepted(0.55));
        assert!(decision.accepted(0.3));
        // Scores sort strongest-first.
        assert_eq!(decision.scores[0].0, "farewell");
        assert!((decision.score_for("greeting") - 0.4).abs() < 1e-6);
        assert!(decision.score_for("missing").abs() < 1e-6);
    }

    #[test]
    fn decision_source_labels_are_stable() {
        assert_eq!(DecisionSource::Model.as_str(), "model");
        assert_eq!(DecisionSource::Embedding.as_str(), "embedding");
        assert_eq!(DecisionSource::LocalModel.as_str(), "local_model");
        assert_eq!(DecisionSource::Heuristic.as_str(), "heuristic");
        assert!(DecisionSource::Model.is_model_backed());
        assert!(!DecisionSource::Heuristic.is_model_backed());
    }

    #[test]
    fn nli_label_parsing_round_trips() {
        for label in [
            NliLabel::Entailment,
            NliLabel::Neutral,
            NliLabel::Contradiction,
        ] {
            assert_eq!(NliLabel::parse(label.as_str()), Some(label));
            assert_eq!(NliLabel::parse(&label.as_str().to_uppercase()), Some(label));
        }
        assert_eq!(NliLabel::parse("maybe"), None);
    }

    #[test]
    fn structured_event_completeness() {
        let complete = StructuredEvent {
            subject: "Alice".into(),
            verb: "deployed".into(),
            object: "the release".into(),
            time_start: None,
            time_end: None,
            location: None,
            confidence: 0.9,
            negated: false,
            source: DecisionSource::Model,
        };
        assert!(complete.is_complete());

        let blank_object = StructuredEvent {
            object: "  ".into(),
            ..complete.clone()
        };
        assert!(!blank_object.is_complete());
    }

    #[test]
    fn cosine_similarity_basics() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        // Degenerate inputs are 0.0, never NaN.
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
    }
}
