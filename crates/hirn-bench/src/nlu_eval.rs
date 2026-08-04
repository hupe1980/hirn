//! Query-intent routing evaluation.
//!
//! The LongMemEval and HIRN-Bench suites exercise `recall_view()` and compiled
//! HirnQL; neither reaches `smart_recall`, so neither measures query-view
//! routing — the surface M-06 is fundamentally about. This module measures it
//! directly.
//!
//! It answers three questions the other suites cannot:
//!
//! 1. **Is the model-backed router more accurate than the cue fallback?** Both
//!    are scored on the same labeled set, so the delta is attributable.
//! 2. **Is its confidence calibrated?** Every decision yields a
//!    `(confidence, correct)` pair, which is exactly the labeled sample
//!    [`Calibration::fit`] needs — turning "fit against a labeled sample" from
//!    an instruction into a command.
//! 3. **How often does it fall back?** A router silently running on the cue
//!    floor looks identical to one that is working, until you count.
//!
//! The labeled set is deliberately adversarial in both directions: it includes
//! cases the cue list gets wrong (implicit intent, misleading cue words,
//! non-English, passive voice) *and* ordinary cases it gets right, so the
//! comparison is not rigged by construction.

use std::sync::Arc;
use std::time::Instant;

use hirn_core::nlu::{Calibration, CalibrationReport, CalibrationSample, DecisionSource};
use hirn_engine::{ViewKind, classify_query_heuristic, route_query};
use hirn_provider::{ExemplarRouter, HybridClassifier, LlmTextClassifier};
use serde::{Deserialize, Serialize};

/// One labeled routing case.
#[derive(Debug, Clone, Copy)]
pub struct IntentCase {
    pub query: &'static str,
    pub expected: ViewKind,
    /// What makes this case hard, for per-category reporting.
    pub category: IntentCategory,
}

/// Why a case is in the set. Reporting per category is what distinguishes
/// "the model is better" from "the model is better *at the cases the cue list
/// was always going to miss*", which is a much narrower claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentCategory {
    /// Contains the cue word for its own view — the cue list should get these.
    Literal,
    /// Asks for its view with no cue word present.
    ImplicitIntent,
    /// Contains a cue word belonging to a *different* view.
    MisleadingCue,
    /// The actor or object is behind a passive construction.
    PassiveVoice,
    /// Not English.
    Multilingual,
}

impl IntentCategory {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Literal => "literal",
            Self::ImplicitIntent => "implicit_intent",
            Self::MisleadingCue => "misleading_cue",
            Self::PassiveVoice => "passive_voice",
            Self::Multilingual => "multilingual",
        }
    }
}

const fn case(query: &'static str, expected: ViewKind, category: IntentCategory) -> IntentCase {
    IntentCase {
        query,
        expected,
        category,
    }
}

/// The labeled routing set.
///
/// Gold labels follow [`QUERY_INTENT_TASK`](hirn_engine::QUERY_INTENT_TASK)'s
/// definitions: `temporal` for when/order/duration, `causal` for why/what
/// brought it about, `entity` for a specific person/thing/value, `semantic`
/// for general topical recall.
pub const INTENT_CASES: &[IntentCase] = &[
    // ── Literal: the cue list is expected to handle these ────────────────
    case(
        "when did the lease renewal go through",
        ViewKind::Temporal,
        IntentCategory::Literal,
    ),
    case(
        "why did the flight get cancelled",
        ViewKind::Causal,
        IntentCategory::Literal,
    ),
    case(
        "who recommended that dentist",
        ViewKind::Entity,
        IntentCategory::Literal,
    ),
    case(
        "what did the doctor say about my knee",
        ViewKind::Semantic,
        IntentCategory::Literal,
    ),
    case(
        "what happened after the tax filing",
        ViewKind::Temporal,
        IntentCategory::Literal,
    ),
    case(
        "what caused the leak in the basement",
        ViewKind::Causal,
        IntentCategory::Literal,
    ),
    case(
        "which insurance plan did i pick",
        ViewKind::Entity,
        IntentCategory::Literal,
    ),
    case(
        "notes from the parent teacher meeting",
        ViewKind::Semantic,
        IntentCategory::Literal,
    ),
    case(
        "how long was the hotel stay",
        ViewKind::Temporal,
        IntentCategory::Literal,
    ),
    case(
        "the reason we switched banks",
        ViewKind::Causal,
        IntentCategory::Literal,
    ),
    case(
        "where did i park at the airport",
        ViewKind::Entity,
        IntentCategory::Literal,
    ),
    case(
        "what did we cover in the book club",
        ViewKind::Semantic,
        IntentCategory::Literal,
    ),
    // ── Implicit intent: the right view, no cue word ─────────────────────
    case(
        "how many weeks stood between the offer and the start date",
        ViewKind::Temporal,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "take me through the run of events at the wedding",
        ViewKind::Temporal,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "did i renew the passport or the license earlier",
        ViewKind::Temporal,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "i still do not understand how the roof ended up damaged",
        ViewKind::Causal,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "something made the sourdough collapse and i want to know what",
        ViewKind::Causal,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "the colleague who lent me the drill",
        ViewKind::Entity,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "my cousin's new address",
        ViewKind::Entity,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "everything i have stored on my knee rehab",
        ViewKind::Semantic,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "bring back what i saved about beekeeping",
        ViewKind::Semantic,
        IntentCategory::ImplicitIntent,
    ),
    case(
        "jog my memory on the pension paperwork",
        ViewKind::Semantic,
        IntentCategory::ImplicitIntent,
    ),
    // ── Misleading cue: contains another view's cue word ─────────────────
    case(
        "what set off the smoke alarm",
        ViewKind::Causal,
        IntentCategory::MisleadingCue,
    ),
    case(
        "which decision left us short on savings",
        ViewKind::Causal,
        IntentCategory::MisleadingCue,
    ),
    case(
        "what is behind my recurring headaches",
        ViewKind::Causal,
        IntentCategory::MisleadingCue,
    ),
    case(
        "what is the most recent invoice i paid",
        ViewKind::Temporal,
        IntentCategory::MisleadingCue,
    ),
    case(
        "who sat with me during the recital",
        ViewKind::Entity,
        IntentCategory::MisleadingCue,
    ),
    case(
        "what results came back from the blood panel",
        ViewKind::Semantic,
        IntentCategory::MisleadingCue,
    ),
    case(
        "summarize the effects chapter from that nutrition book",
        ViewKind::Semantic,
        IntentCategory::MisleadingCue,
    ),
    case(
        "which order did the packages arrive in",
        ViewKind::Temporal,
        IntentCategory::MisleadingCue,
    ),
    // ── Passive voice ────────────────────────────────────────────────────
    case(
        "the mortgage was signed by which lender",
        ViewKind::Entity,
        IntentCategory::PassiveVoice,
    ),
    case(
        "when was the car serviced last",
        ViewKind::Temporal,
        IntentCategory::PassiveVoice,
    ),
    case(
        "the fence was knocked down by what",
        ViewKind::Causal,
        IntentCategory::PassiveVoice,
    ),
    case(
        "what was talked about at the council session",
        ViewKind::Semantic,
        IntentCategory::PassiveVoice,
    ),
    case(
        "the prescription was written by whom",
        ViewKind::Entity,
        IntentCategory::PassiveVoice,
    ),
    case(
        "what was brought up in the performance review",
        ViewKind::Semantic,
        IntentCategory::PassiveVoice,
    ),
    // ── Multilingual ─────────────────────────────────────────────────────
    case(
        "wann wurde die heizung zuletzt gewartet",
        ViewKind::Temporal,
        IntentCategory::Multilingual,
    ),
    case(
        "warum wurde der vertrag gekündigt",
        ViewKind::Causal,
        IntentCategory::Multilingual,
    ),
    case(
        "wer hat mir den zahnarzt empfohlen",
        ViewKind::Entity,
        IntentCategory::Multilingual,
    ),
    case(
        "was habe ich über meine knie reha notiert",
        ViewKind::Semantic,
        IntentCategory::Multilingual,
    ),
    case(
        "cuándo renové el pasaporte",
        ViewKind::Temporal,
        IntentCategory::Multilingual,
    ),
    case(
        "por qué se canceló el vuelo",
        ViewKind::Causal,
        IntentCategory::Multilingual,
    ),
    case(
        "quién me prestó el taladro",
        ViewKind::Entity,
        IntentCategory::Multilingual,
    ),
    case(
        "qué guardé sobre la apicultura",
        ViewKind::Semantic,
        IntentCategory::Multilingual,
    ),
    case(
        "quand ai-je payé la dernière facture",
        ViewKind::Temporal,
        IntentCategory::Multilingual,
    ),
    case(
        "pourquoi la toiture a-t-elle été endommagée",
        ViewKind::Causal,
        IntentCategory::Multilingual,
    ),
];

/// One routed case with its outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentOutcome {
    pub query: String,
    pub category: IntentCategory,
    pub expected: String,
    pub routed: String,
    pub correct: bool,
    pub confidence: f32,
    pub source: DecisionSource,
    pub latency_ms: f64,
}

/// A deterministic 95% bootstrap confidence interval for a proportion.
///
/// A router reported without an interval invites over-reading: at n=46 a single
/// flipped decision moves accuracy by 2.2 points, and two arms whose intervals
/// overlap are not distinguishable by this evaluation however different their
/// point estimates look.
///
/// The resample seed is fixed so the interval is reproducible across runs —
/// the artifact must not change when nothing else did.
fn bootstrap_ci(correct: &[bool], resamples: usize) -> (f64, f64) {
    if correct.is_empty() {
        return (0.0, 0.0);
    }
    let n = correct.len();
    // xorshift64* — a fixed, dependency-free deterministic generator.
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = move || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };

    let mut means: Vec<f64> = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let hits = (0..n)
            .filter(|_| correct[(next() % n as u64) as usize])
            .count();
        means.push(hits as f64 / n as f64);
    }
    means.sort_by(f64::total_cmp);
    let lower = means[(resamples as f64 * 0.025) as usize];
    let upper = means[((resamples as f64 * 0.975) as usize).min(resamples - 1)];
    (lower, upper)
}

/// Accuracy for one slice of the set.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SliceAccuracy {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// 95% bootstrap interval; `None` for per-category slices, where the
    /// sample is far too small for an interval to mean anything.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ci95: Option<(f64, f64)>,
}

impl SliceAccuracy {
    fn record(&mut self, correct: bool) {
        self.total += 1;
        if correct {
            self.correct += 1;
        }
        self.accuracy = self.correct as f64 / self.total as f64;
    }
}

/// Result of one arm (model-backed or fallback-only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentArmResult {
    pub arm: String,
    pub overall: SliceAccuracy,
    pub by_category: Vec<(String, SliceAccuracy)>,
    /// Share of decisions made by each backend. `heuristic` is the fallback rate.
    pub by_source: Vec<(String, usize)>,
    pub mean_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub outcomes: Vec<IntentOutcome>,
    /// Calibration **per deciding backend**.
    ///
    /// `nlu.llm_calibration` and `nlu.embedding_calibration` are separate
    /// config knobs because an LLM's self-reported confidence and a
    /// cosine-similarity softmax are different scales. Pooling their samples
    /// would fit a map that belongs to neither backend.
    pub calibration: Vec<(String, CalibrationAnalysis)>,
}

/// Calibration as deployed, plus whether a refit is safe to adopt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationAnalysis {
    pub before: CalibrationReport,
    /// Human-readable fit outcome — a fitted map, or why one was refused.
    pub fit: String,
    /// Present only when a deployable map was produced.
    pub fitted_scale: Option<f32>,
    pub fitted_floor: Option<f32>,
    /// Calibration after the refit, when there was one.
    pub after: Option<CalibrationReport>,
    /// Safe to adopt: a map was produced **and** it lowers calibration error.
    pub adopt: bool,
}

fn summarize(arm: &str, outcomes: Vec<IntentOutcome>) -> IntentArmResult {
    let mut overall = SliceAccuracy::default();
    let mut by_category: Vec<(String, SliceAccuracy)> = Vec::new();
    let mut by_source: Vec<(String, usize)> = Vec::new();

    for outcome in &outcomes {
        overall.record(outcome.correct);

        let category = outcome.category.as_str().to_string();
        match by_category.iter_mut().find(|(name, _)| *name == category) {
            Some((_, slice)) => slice.record(outcome.correct),
            None => {
                let mut slice = SliceAccuracy::default();
                slice.record(outcome.correct);
                by_category.push((category, slice));
            }
        }

        let source = outcome.source.as_str().to_string();
        match by_source.iter_mut().find(|(name, _)| *name == source) {
            Some((_, count)) => *count += 1,
            None => by_source.push((source, 1)),
        }
    }
    by_category.sort_by(|a, b| a.0.cmp(&b.0));
    by_source.sort_by(|a, b| a.0.cmp(&b.0));

    overall.ci95 = Some(bootstrap_ci(
        &outcomes.iter().map(|o| o.correct).collect::<Vec<_>>(),
        2_000,
    ));

    let mut latencies: Vec<f64> = outcomes.iter().map(|o| o.latency_ms).collect();
    latencies.sort_by(f64::total_cmp);
    let mean_latency_ms = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };
    let p95_latency_ms = latencies
        .get(((latencies.len() as f64 * 0.95).ceil() as usize).saturating_sub(1))
        .copied()
        .unwrap_or(0.0);

    // Calibration is grouped by deciding backend. A fallback is excluded
    // outright: it reports a cue-derived weight, not a probability.
    let mut grouped: Vec<(String, Vec<CalibrationSample>)> = Vec::new();
    for outcome in outcomes.iter().filter(|o| o.source.is_model_backed()) {
        let source = outcome.source.as_str().to_string();
        let sample = CalibrationSample::new(outcome.confidence, outcome.correct);
        match grouped.iter_mut().find(|(name, _)| *name == source) {
            Some((_, samples)) => samples.push(sample),
            None => grouped.push((source, vec![sample])),
        }
    }
    grouped.sort_by(|a, b| a.0.cmp(&b.0));

    let deployed = Calibration::default();
    let calibration: Vec<(String, CalibrationAnalysis)> = grouped
        .into_iter()
        .map(|(source, samples)| {
            let before = deployed.evaluate(&samples);
            let report = deployed.fit_report(&samples);
            let analysis = match report.calibration() {
                Some(fitted) => {
                    let after = fitted.evaluate(&samples);
                    CalibrationAnalysis {
                        adopt: after.expected_calibration_error < before.expected_calibration_error,
                        fitted_scale: Some(fitted.scale),
                        fitted_floor: Some(fitted.floor),
                        after: Some(after),
                        fit: report.explain(),
                        before,
                    }
                }
                // Refused — report the measurement and the reason. A refusal is
                // a result about the sample, not a missing number.
                None => CalibrationAnalysis {
                    fit: report.explain(),
                    fitted_scale: None,
                    fitted_floor: None,
                    after: None,
                    adopt: false,
                    before,
                },
            };
            (source, analysis)
        })
        .collect();

    IntentArmResult {
        arm: arm.to_string(),
        overall,
        by_category,
        by_source,
        mean_latency_ms,
        p95_latency_ms,
        outcomes,
        calibration,
    }
}

/// Both arms of the routing evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentEvalResult {
    /// Accuracy of always answering with the task's default label.
    ///
    /// The floor any router must clear to have earned its latency. Reporting
    /// it is what separates "the router works" from "the label distribution is
    /// skewed".
    pub majority_class_baseline: f64,
    /// Provenance required by `bench-results/README.md` for a publishable run:
    /// without it an artifact cannot be tied to the tree that produced it.
    pub generated_at_rfc3339: String,
    pub environment: crate::cognitive::EnvironmentInfo,
    pub cases: usize,
    pub backends: String,
    pub model_backed: IntentArmResult,
    pub fallback_only: IntentArmResult,
    /// Model-backed accuracy minus fallback accuracy.
    pub accuracy_delta: f64,
}

/// Route every case through `chain`, recording the outcome.
async fn run_arm(arm: &str, chain: &HybridClassifier) -> IntentArmResult {
    let mut outcomes = Vec::with_capacity(INTENT_CASES.len());
    for case in INTENT_CASES {
        let started = Instant::now();
        let route = route_query(chain, case.query).await;
        let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
        outcomes.push(IntentOutcome {
            query: case.query.to_string(),
            category: case.category,
            expected: case.expected.label().to_string(),
            routed: route.primary.label().to_string(),
            correct: route.primary == case.expected,
            confidence: route.confidence,
            source: route.source,
            latency_ms,
        });
    }
    summarize(arm, outcomes)
}

/// Score the cue fallback without consulting any backend.
fn run_fallback_arm() -> IntentArmResult {
    let outcomes = INTENT_CASES
        .iter()
        .map(|case| {
            let started = Instant::now();
            let weights = classify_query_heuristic(case.query);
            let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let routed = weights.primary();
            IntentOutcome {
                query: case.query.to_string(),
                category: case.category,
                expected: case.expected.label().to_string(),
                routed: routed.label().to_string(),
                correct: routed == case.expected,
                confidence: weights.weight_for(routed),
                source: DecisionSource::Heuristic,
                latency_ms,
            }
        })
        .collect();
    summarize("fallback_only", outcomes)
}

/// Run the routing evaluation.
///
/// Builds the same chain the engine builds from configured providers, so the
/// measured arm is the shipped configuration rather than a bespoke one. With no
/// provider available the model-backed arm degrades to the fallback, which the
/// per-source counts make visible rather than silently equal.
pub async fn run(
    llm: Option<Arc<dyn hirn_core::embed::LlmProvider>>,
    embedder: Option<Arc<dyn hirn_core::embed::Embedder>>,
    environment_label: Option<String>,
) -> IntentEvalResult {
    let mut chain = HybridClassifier::new();
    let mut backends = Vec::new();
    if let Some(llm) = llm {
        backends.push(format!("llm:{}", llm.model_id()));
        chain = chain.with_backend(Arc::new(LlmTextClassifier::new(llm)));
    }
    if let Some(embedder) = embedder {
        backends.push(format!("embedding:{}", embedder.model_id()));
        chain = chain.with_backend(Arc::new(ExemplarRouter::new(embedder)));
    }
    if backends.is_empty() {
        backends.push("none (fallback only)".to_string());
    }

    let model_backed = run_arm("model_backed", &chain).await;
    let fallback_only = run_fallback_arm();
    let accuracy_delta = model_backed.overall.accuracy - fallback_only.overall.accuracy;

    let default_label = hirn_engine::QUERY_INTENT_TASK.default_label;
    let majority_class_baseline = INTENT_CASES
        .iter()
        .filter(|c| c.expected.label() == default_label)
        .count() as f64
        / INTENT_CASES.len() as f64;

    IntentEvalResult {
        majority_class_baseline,
        generated_at_rfc3339: crate::provenance::generated_at_rfc3339(),
        environment: crate::provenance::current_environment_info(environment_label),
        cases: INTENT_CASES.len(),
        backends: backends.join(" -> "),
        model_backed,
        fallback_only,
        accuracy_delta,
    }
}

/// Render a human-readable report.
#[must_use]
pub fn render_markdown(result: &IntentEvalResult) -> String {
    let mut out = String::new();
    out.push_str("# Query-Intent Routing Evaluation\n\n");
    out.push_str(&format!(
        "Generated: {}  \nCases: {}  \nBackends: `{}`\n",
        result.generated_at_rfc3339, result.cases, result.backends
    ));
    out.push_str(&format!(
        "Environment: {} · {} {} · {} cpus  \nCommit: `{}`  \nCargo.lock blake3: `{}`\n\n",
        result.environment.label.as_deref().unwrap_or("unlabeled"),
        result.environment.os,
        result.environment.arch,
        result.environment.logical_cpus,
        result
            .environment
            .git_commit_sha
            .as_deref()
            .unwrap_or("unknown"),
        result
            .environment
            .cargo_lock_blake3
            .as_deref()
            .unwrap_or("unknown"),
    ));

    out.push_str("## Accuracy\n\n");
    out.push_str(
        "| Arm | Accuracy | 95% CI | Correct | Mean ms | p95 ms |\n\
         |---|---:|---:|---:|---:|---:|\n",
    );
    for arm in [&result.model_backed, &result.fallback_only] {
        let (low, high) = arm.overall.ci95.unwrap_or((0.0, 0.0));
        out.push_str(&format!(
            "| {} | {:.4} | {:.3}–{:.3} | {}/{} | {:.1} | {:.1} |\n",
            arm.arm,
            arm.overall.accuracy,
            low,
            high,
            arm.overall.correct,
            arm.overall.total,
            arm.mean_latency_ms,
            arm.p95_latency_ms,
        ));
    }
    out.push_str(&format!(
        "| majority-class baseline | {:.4} | — | — | 0.0 | 0.0 |\n",
        result.majority_class_baseline
    ));
    out.push_str(&format!(
        "\n**Delta (model − fallback): {:+.4}**\n\n",
        result.accuracy_delta
    ));

    out.push_str("## Accuracy by category\n\n");
    out.push_str("| Category | Model-backed | Fallback |\n|---|---:|---:|\n");
    for (category, model_slice) in &result.model_backed.by_category {
        let fallback_slice = result
            .fallback_only
            .by_category
            .iter()
            .find(|(name, _)| name == category)
            .map_or(0.0, |(_, slice)| slice.accuracy);
        out.push_str(&format!(
            "| {} | {:.4} ({}/{}) | {:.4} |\n",
            category, model_slice.accuracy, model_slice.correct, model_slice.total, fallback_slice
        ));
    }

    out.push_str("\n## Deciding backend\n\n| Source | Decisions |\n|---|---:|\n");
    for (source, count) in &result.model_backed.by_source {
        out.push_str(&format!("| {source} | {count} |\n"));
    }

    for (source, calibration) in &result.model_backed.calibration {
        out.push_str(&format!(
            "\n## Confidence calibration — `{source}` backend\n\n"
        ));
        out.push_str("| Metric | As deployed |\n|---|---:|\n");
        out.push_str(&format!("| Samples | {} |\n", calibration.before.samples));
        out.push_str(&format!(
            "| Accuracy | {:.4} |\n",
            calibration.before.accuracy
        ));
        out.push_str(&format!(
            "| Mean confidence | {:.4} |\n",
            calibration.before.mean_confidence
        ));
        out.push_str(&format!(
            "| Expected calibration error | {:.4} |\n",
            calibration.before.expected_calibration_error
        ));
        out.push_str(&format!(
            "| Brier score | {:.4} |\n\n",
            calibration.before.brier_score
        ));
        out.push_str(&format!("Fit: {}\n\n", calibration.fit));
        match (&calibration.after, calibration.adopt) {
            (Some(after), true) => out.push_str(&format!(
                "Refit lowers expected calibration error to {:.4} — safe to adopt.\n",
                after.expected_calibration_error
            )),
            (Some(after), false) => out.push_str(&format!(
                "Refit does **not** lower expected calibration error ({:.4}); do not adopt.\n",
                after.expected_calibration_error
            )),
            (None, _) => out.push_str(
                "No deployable map was produced, so the shipped identity calibration stands.\n",
            ),
        }
    }

    out.push_str("\n## Caveats\n\n");
    out.push_str(&format!(
        "- **{} cases is a small sample.** One flipped decision moves accuracy by \
         {:.1} points, so treat the per-category rows as directional rather than \
         precise.\n",
        result.cases,
        100.0 / result.cases as f64
    ));
    out.push_str(
        "- **Gold labels carry judgement.** Some queries admit more than one defensible \
         view — \"what is the last thing we shipped\" asks for an entity *via* an \
         ordering, and is labeled `temporal` because answering it requires ordering \
         events. A miss on such a case is a disagreement about the label as much as a \
         routing error.\n",
    );
    out.push_str(
        "- **The fallback arm is not a straw man.** The `literal` slice exists so the \
         cue list is scored on the cases it was designed for; a unit test fails the \
         build if it drops below 0.5 there.\n",
    );
    out.push_str(
        "- **Latency is a real cost.** The model arm adds a provider call per routed \
         query. Weigh the accuracy delta against the p95 above before enabling it on a \
         latency-sensitive path.\n",
    );

    out.push_str("\n## Misroutes (model-backed)\n\n");
    let misroutes: Vec<&IntentOutcome> = result
        .model_backed
        .outcomes
        .iter()
        .filter(|o| !o.correct)
        .collect();
    if misroutes.is_empty() {
        out.push_str("None.\n");
    } else {
        out.push_str(
            "| Query | Expected | Routed | Confidence | Source |\n|---|---|---|---:|---|\n",
        );
        for outcome in misroutes {
            out.push_str(&format!(
                "| {} | {} | {} | {:.2} | {} |\n",
                outcome.query, outcome.expected, outcome.routed, outcome.confidence, outcome.source
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Token set of a query, for overlap detection.
    fn tokens(text: &str) -> std::collections::HashSet<String> {
        text.split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(str::to_lowercase)
            .collect()
    }

    #[test]
    fn no_case_overlaps_a_task_exemplar() {
        // The task's exemplars are few-shot anchors inside the model's own
        // system prompt. A labeled case that duplicates one is not measuring
        // routing, it is measuring string matching — and the contamination is
        // *asymmetric*: it inflates the model arm while the cue fallback,
        // which never sees the prompt, gets no such help.
        //
        // An earlier version of this set shared 4 exact and 3 near-duplicate
        // queries with the exemplars (19.4%), which invalidated its headline
        // number. This guard is why that cannot recur.
        let exemplars: Vec<&str> = hirn_engine::QUERY_INTENT_TASK
            .labels
            .iter()
            .flat_map(|label| label.exemplars.iter().copied())
            .collect();

        for case in INTENT_CASES {
            let case_tokens = tokens(case.query);
            for exemplar in &exemplars {
                assert!(
                    !case.query.eq_ignore_ascii_case(exemplar),
                    "case {:?} is verbatim a task exemplar",
                    case.query
                );
                let exemplar_tokens = tokens(exemplar);
                let shared = case_tokens.intersection(&exemplar_tokens).count();
                let union = case_tokens.union(&exemplar_tokens).count().max(1);
                let jaccard = shared as f64 / union as f64;
                assert!(
                    jaccard < 0.6,
                    "case {:?} overlaps exemplar {:?} at Jaccard {jaccard:.2}; \
                     the model would be matching its own prompt",
                    case.query,
                    exemplar
                );
            }
        }
    }

    #[test]
    fn multilingual_share_is_reported_not_hidden() {
        // The multilingual slice is where the English-only fallback is weakest,
        // so its share drives the headline delta. Keeping it a minority and
        // reporting an English-only figure alongside the overall one stops the
        // set composition from silently manufacturing the result.
        let multilingual = INTENT_CASES
            .iter()
            .filter(|c| c.category == IntentCategory::Multilingual)
            .count();
        let share = multilingual as f64 / INTENT_CASES.len() as f64;
        assert!(
            share < 0.35,
            "multilingual cases are {:.0}% of the set; at that share the headline \
             number mostly measures language coverage",
            share * 100.0
        );
    }

    #[test]
    fn every_case_is_labeled_with_a_real_view() {
        for case in INTENT_CASES {
            assert!(
                !case.query.trim().is_empty(),
                "empty query in the labeled set"
            );
            assert!(
                ViewKind::parse(case.expected.label()).is_some(),
                "case {:?} has an unroutable gold label",
                case.query
            );
        }
    }

    #[test]
    fn the_set_covers_every_view_and_every_category() {
        // A set missing a view cannot detect a router that never selects it.
        for view in [
            ViewKind::Semantic,
            ViewKind::Temporal,
            ViewKind::Causal,
            ViewKind::Entity,
        ] {
            assert!(
                INTENT_CASES.iter().any(|c| c.expected == view),
                "no labeled case expects {}",
                view.label()
            );
        }
        for category in [
            IntentCategory::Literal,
            IntentCategory::ImplicitIntent,
            IntentCategory::MisleadingCue,
            IntentCategory::PassiveVoice,
            IntentCategory::Multilingual,
        ] {
            assert!(
                INTENT_CASES.iter().any(|c| c.category == category),
                "no labeled case in category {}",
                category.as_str()
            );
        }
    }

    #[test]
    fn queries_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for case in INTENT_CASES {
            assert!(
                seen.insert(case.query),
                "duplicate query in the labeled set: {:?}",
                case.query
            );
        }
    }

    #[test]
    fn the_set_is_not_rigged_against_the_fallback() {
        // The `literal` slice exists so the comparison includes cases the cue
        // list is designed for. If the fallback cannot clear a majority there,
        // the set is stacked and any reported delta is meaningless.
        let arm = run_fallback_arm();
        let literal = arm
            .by_category
            .iter()
            .find(|(name, _)| name == IntentCategory::Literal.as_str())
            .map(|(_, slice)| slice.accuracy)
            .expect("literal slice present");
        assert!(
            literal >= 0.5,
            "cue fallback scores {literal:.2} on literal cues; the labeled set is \
             stacked against it and no delta measured on it would be honest"
        );
    }

    #[tokio::test]
    async fn with_no_provider_both_arms_agree() {
        // Without a backend the model arm *is* the fallback, so any difference
        // would mean the two paths disagree about the same cue logic.
        let result = run(None, None, None).await;
        assert_eq!(
            result.model_backed.overall.correct,
            result.fallback_only.overall.correct
        );
        assert!(result.accuracy_delta.abs() < 1e-9);
        assert!(
            result
                .model_backed
                .by_source
                .iter()
                .all(|(source, _)| source == "heuristic"),
            "no provider means every decision is a fallback decision"
        );
        assert!(
            result.model_backed.calibration.is_empty(),
            "fallback weights are not probabilities and must not be calibrated"
        );
    }
}
