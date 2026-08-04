+++
title = "Language Understanding"
description = "How hirn decides meaning-dependent questions — query routing, belief revision, knowledge typing, contradiction, and extraction — through a calibrated model chain with a deterministic fallback."
weight = 5
+++

# Language Understanding

{% experimental() %}
This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.
{% end %}


A cognitive database has to answer questions *about meaning* before it can answer
questions about data: is this query asking when something happened, or why? Does
this new evidence support the belief it lands next to, or refute it? Are these two
records in conflict? This page describes the layer that makes those decisions.

## The problem with cue lists

Every one of those decisions can be approximated with a word list, and hirn used to
approximate all of them that way: a query containing "when" routes temporal, a record
containing "not" contradicts one that doesn't, a thread containing "should" is
prescriptive.

Word lists fail in ways that compound:

| Failure | Example | What the cue list does |
|---|---|---|
| Implicit intent | "how much time passed between the two releases" | no temporal cue word → routed to plain recall |
| Misleading cue | "**what** triggered the regression" | "what" is an entity cue → routed to factoid lookup |
| Scoped negation | "the pipeline is **not** unstable" vs "the pipeline is stable" | negation mismatch → filed as a contradiction between two statements that agree |
| Negation-free conflict | "the migration succeeded" vs "the migration was rolled back" | no cue on either side → conflict missed entirely |
| Passive voice | "the release was deployed by Alice" | subject extracted as "the release" |
| Other languages | "wann haben wir das ausgeliefert" | no English cue → routed to plain recall |

The failures are not individually fatal, and that is what makes them dangerous: each
one is a small quality loss in a different subsystem, none of them raises an error,
and the usual remedy — adding another word to the list — makes the policy larger,
more language-specific, and no more measurable.

hirn's position is that these are **model decisions with a deterministic floor**, not
string operations.

---

## The contract

[`hirn_core::nlu`](https://docs.rs/hirn-core/latest/hirn_core/nlu/) defines one contract
that every meaning-dependent decision runs through.

A **`ClassificationTask`** is a named decision surface: typed labels, a
natural-language description of each label written for a model to read, and
exemplars. Each task is declared once as a `const` and drives *every* backend — the
LLM prompt, the JSON schema, the embedding router's centroids, and the fallback all
read the same label set, so they cannot drift apart.

```rust
pub const QUERY_INTENT_TASK: ClassificationTask = ClassificationTask {
    name: "query_intent",
    instruction: "Decide which memory view best answers a user's question …",
    labels: &[
        LabelSpec {
            name: "temporal",
            description: "Asks about when something happened, the order of events, \
                          durations, or how things changed over time.",
            exemplars: &[
                "when did we first deploy the service",
                "how much time passed between the two releases",
            ],
        },
        // …
    ],
    default_label: "semantic",
};
```

A **`Classification`** is the result: the label, a *calibrated* confidence, an optional
rationale, the per-label distribution where the backend produces one, and —
critically — a `DecisionSource` recording which backend decided.

---

## The backend chain

`HybridClassifier` runs backends in order and stops at the first one that produces a
decision it stands behind:

| Order | Backend | Cost | Sees |
|---|---|---|---|
| 1 | `LlmTextClassifier` | one generation call, temperature 0, JSON-schema constrained | scope, implicit intent, any language |
| 2 | `ExemplarRouter` | one embedding call | paraphrase, cross-lingual with a multilingual embedder |
| 3 | caller's fallback | free | English cue words |

A backend **abstains** — it does not guess — when it times out, emits output that
does not match the task schema, returns a label outside the task's set, or lands
below the confidence gate. The chain moves on. If every backend abstains, the
caller's deterministic fallback decides, and the result is recorded as
`source = heuristic` whatever the fallback claims about itself.

{% important() %}
**The confidence gate is the weakest of those four checks — do not lean on it.**
Measured on the routing set, all 36 `gpt-4o-mini` decisions landed between 0.80 and
0.90: none fell below the default 0.55 gate, and the one wrong route was made at 0.90.
A model that does not express doubt cannot be filtered by a doubt threshold. What
actually protects this path is the strict schema parse, the unknown-label rejection,
and the timeout paths. If you need a defence against a *confidently wrong* decision,
it has to be a second opinion — the entailment review that
[reflection](#contradiction-is-treated-as-destructive) applies to contradictions is
that pattern — not a higher threshold.
{% end %}


That last point is the design's load-bearing property: **hirn works with no provider
configured**, and the rate at which it is running that way is a metric rather than an
assumption.

```rust
let decision = classifier
    .decide(&QUERY_INTENT_TASK, query, None, || {
        // Deterministic floor. Always available, never the primary.
        heuristic_classification(query)
    })
    .await;
```

### Entailment

Contradiction, polarity, and negation scope go through `NliModel` rather than a
classifier, because the question is asymmetric: does *this* rule out *that*?

- `LlmNli` — any configured classifier judging the entailment task.
- `LocalNli` — an on-device ONNX 3-class NLI cross-encoder (feature `cross-encoder`),
  for write-path volume or deployments where data cannot leave the machine.

`LocalNli` reads the head order from the checkpoint's own `config.json`: NLI models
disagree about whether index 0 means `contradiction` or `entailment`, and a hard-coded
guess silently inverts every judgment. A checkpoint whose label mapping cannot be
established fails to load rather than loading wrong.

### Typed extraction

- `LlmEventExtractor` — subject/verb/object with resolved passive voice and an
  explicit `negated` flag. "We never shipped v2" no longer enters the event store as a
  shipping event.
- `LlmEntityExtractor` — typed, case-independent NER with typed relations, falling
  back to `RegexEntityExtractor` internally so extraction degrades rather than stops.
- `LlmTemporalExtractor` — the write-time temporal envelope: *when* the event happened
  (distinct from when it was recorded), *how precisely* the text pins it, and *what
  temporal state* it asserts (ongoing / completed / planned / timeless). Precision is
  carried explicitly because "in March" parses to midnight on 1 March, and without the
  paired precision a ranker treats that fabricated instant as evidence. State is what
  keeps "I live in Berlin" from decaying — see
  [Cognitive Model](@/docs/concepts/cognitive-model.md) for how retrieval uses it.
- `LlmPreferenceExtractor` — first-person preferences from any phrasing, producing the
  same typed `PreferenceEvidence` envelope the cue matcher writes. It can also answer
  "no preference here", which a cue matcher structurally cannot: "no cue matched" and
  "the speaker stated no preference" are the same outcome to a word list, and different
  outcomes to a reader. A confident *no* therefore suppresses the cue fallback, so
  "my manager said they prefer dark mode" no longer writes a preference to the
  speaker's profile.

---

## Where it is used

| Decision | Task | Fallback |
|---|---|---|
| Query-view routing (`smart_recall`) | `query_intent` | whole-word + token-sequence cue matching |
| Belief revision (`reflect`) | `reflection_outcome` | negation-cue mismatch + antonym lexicon |
| Knowledge typing (consolidation) | `knowledge_type` | cue words, two-hit minimum |
| Retrieval-depth routing | `query_complexity` | token buckets + interrogative cues |
| Contradiction on insert | entailment | surface signals, recorded as unreviewed |
| SVO extraction (write path) | typed extraction | regex extractor |
| Entity extraction | typed extraction | regex extractor |
| Preference extraction (write path) | typed extraction | first-person cue phrases |
| Temporal envelope (write path) | typed extraction | deterministic date parser (no state axis) |

### Contradiction is treated as destructive

Two paths deliberately hold contradiction to a higher bar than other outcomes,
because halving a belief's credence and writing a `Contradicts` edge are not
reversible by later evidence arriving in a different order:

1. **Reflection's deterministic floor can never return `Contradicts`.** A
   negation-cue mismatch caps out at the reversible `Weakens` step. Asserting a
   contradiction requires a model that judged entailment, at or above
   `nlu.contradiction_min_confidence` (default 0.70), and it must survive review by
   the entailment model where one is configured.
2. **Insert-time contradiction separates nomination from decision.** Cheap surface
   signals nominate candidate pairs; the entailment model decides. With no model
   configured the nominations still stand — offline deployments keep working — but
   each edge records `contradiction_decided_by` in its metadata, so an unreviewed
   surface signal and a model-confirmed conflict never look alike downstream.

### Structure stays deterministic

Model-backed does not mean model-everywhere. Retrieval-depth routing keeps its
structural signals — `INVOLVING` arity, `EXPAND GRAPH`, `FOLLOW CAUSES`, clause
counts — as deterministic code, because those are facts about the compiled plan
rather than inferences about language. They set a *floor* the model can raise but
not lower: a query that expands the graph three hops needs traversal however a model
reads its wording.

The same principle applies throughout. Protocol syntax, schema validation, security
boundaries, exact identifiers, and explicit user options are decided by code. Only
questions whose answer depends on what the text *means* go to a model.

---

## Configuration

```toml
[nlu]
# Master switch. `false` pins every decision to its deterministic fallback
# regardless of which providers are configured.
enabled = true

# Which backends participate, in chain order.
llm_primary = true
embedding_router = true

# Typed SVO extraction on the write path. Correct on passive voice and
# negation, but adds a provider call per ingested record.
typed_event_extraction = false

# Typed preference extraction on the write path. Reads indirect and
# non-English phrasing the cue list cannot, and can answer "no preference
# here" — but adds a provider call per ingested message.
typed_preference_extraction = false

# Write-time temporal envelope: event time, precision, and
# ongoing/completed/planned/timeless state. This is what makes recency decay
# state-aware — without it every record ages uniformly and a timeless fact
# loses ground to a recent irrelevant note. One provider call per record.
typed_temporal_extraction = false

# Maximum concurrent write-path extraction calls during batch ingest. Typed
# extraction is one provider round-trip per record; issued sequentially, a
# 10k-record batch spends over an hour in provider latency.
extraction_concurrency = 8

# Contradiction is destructive, so it carries a stricter bar than the
# general decision gate.
contradiction_min_confidence = 0.70

# Cosine similarity above which two thread summaries are duplicates.
summary_dedup_threshold = 0.95

[nlu.budget]
timeout = 2000          # milliseconds, per provider call
max_tokens = 200
min_confidence = 0.55   # below this, fall through to the next backend
max_input_chars = 2000

# Calibration. Defaults are identity for the LLM (so an uncalibrated
# deployment behaves like the raw backend) and a cosine-scale softmax
# temperature for the embedding router.
[nlu.llm_calibration]
temperature = 1.0
scale = 1.0
floor = 0.0

[nlu.embedding_calibration]
temperature = 0.07
scale = 1.0
floor = 0.0
```

Registering a provider at runtime rebuilds the chain immediately:

```rust
db.set_llm_provider(llm);           // upgrades every decision to model-backed
db.set_nli_model(Arc::new(local));  // explicit entailment model wins over the chain
```

### Calibrating confidence

LLM self-reported confidence is systematically over-confident, and a cosine-similarity
softmax is peaked or flat depending on the embedding model. `min_confidence` only means
something once a reported 0.8 corresponds to being right about 80% of the time —
until then, tightening the gate raises the fallback rate without improving quality.

`Calibration` measures and fits this rather than leaving it to judgement:

```rust
use hirn_core::nlu::{Calibration, CalibrationSample};

// (raw confidence, was the decision correct?) from a labeled evaluation set.
let samples: Vec<CalibrationSample> = /* … */;

let deployed = Calibration::default();
let before = deployed.evaluate(&samples);
println!(
    "ECE {:.3}  Brier {:.3}  accuracy {:.3}  mean confidence {:.3}",
    before.expected_calibration_error, before.brier_score,
    before.accuracy, before.mean_confidence,
);

if let Some(fitted) = deployed.fit(&samples) {
    let after = fitted.evaluate(&samples);
    if after.expected_calibration_error < before.expected_calibration_error {
        // Write `fitted.scale` / `fitted.floor` into `[nlu.llm_calibration]`.
    }
}
```

- **`evaluate`** reports expected calibration error (sample-weighted gap between
  per-bin confidence and accuracy), Brier score, and a reliability diagram
  (`bins`). `is_overconfident(tolerance)` answers the one question the gate
  depends on.
- **`fit`** regresses correctness on raw confidence and returns the affine map that
  best predicts it. It leaves `temperature` alone — that shapes the distribution
  *before* the argmax, which outcome labels say nothing about.
- **`fit` refuses fewer than 30 samples** (`MIN_CALIBRATION_SAMPLES`). A map fitted
  from a handful of observations looks authoritative and is noise; the identity
  default is the better choice.
- A signal whose confidence is *anti*-correlated with correctness clamps to
  `scale = 0` rather than being inverted into a usable one — that would be reading
  information out of a backend that has none.

Always compare `evaluate` before and after: a fit that does not lower expected
calibration error should not be deployed.

Do the same per task. Routing and belief revision have different label sets, different
prompts, and different difficulty; one global number will fit neither well.

---

## Measured routing quality

`hirn-bench nlu-routing` scores the model-backed router and the deterministic cue
fallback on the same labeled set — the surface LongMemEval and HIRN-Bench do not reach,
because they exercise `recall_view()` and compiled HirnQL rather than `smart_recall`.

```bash
cargo run -p hirn-bench -- nlu-routing \
  --environment-label local \
  --json-output bench-results/nlu-routing.json \
  --markdown-output bench-results/nlu-routing.md
```

46 labeled queries, `gpt-4o-mini` → `text-embedding-3-small`:

| Arm | Accuracy | 95% CI | p95 latency |
|---|---:|---:|---:|
| Model-backed chain | **0.9783** (45/46) | 0.935–1.000 | 1060 ms |
| Cue fallback | 0.4348 (20/46) | 0.283–0.587 | ~0 ms |
| Majority-class baseline | 0.2609 | — | 0 ms |

| Category | Model | Fallback |
|---|---:|---:|
| Literal cue words | 1.0000 | 0.8333 |
| Implicit intent (no cue word) | 1.0000 | 0.4000 |
| Misleading cue word | 0.8750 | 0.1250 |
| Passive voice | 1.0000 | 0.5000 |
| Non-English | 1.0000 | 0.2000 |

Read two rows first. The `literal` row is the control: the cue list scores 0.83 on the
cases it was built for, so the set is not stacked against it. The **majority-class
baseline** row is the one that matters: at 0.4348 the cue fallback is only 17 points above
answering "semantic" every single time. That, not the model's 0.978, is the argument for
model-backed routing.

{% warning() %}
**The labeled set must never overlap the task's exemplars.** An earlier version of this
evaluation shared 4 verbatim queries and 3 near-duplicates with `QUERY_INTENT_TASK`'s
few-shot exemplars — the model was matching its own system prompt, and because the cue
fallback never sees that prompt, the contamination inflated only the model arm. A build
guard now fails on any verbatim match or Jaccard ≥ 0.6 against an exemplar.
{% end %}


The cost is real and should be weighed deliberately: **~0.9 s median added latency** and
one provider call per routed query. If that is not acceptable for your workload, run the
embedding router alone (`llm_primary = false`) or keep routing off and accept the cue
floor — both are configurations, not failures.

The evaluation also emits the labeled `(confidence, correct)` samples that calibration
needs, so `--json-output` doubles as a calibration input.

What it measured about calibration is worth stating, because it contradicts the
assumption the defaults were written under: over these decisions the LLM backend is
mildly **under**-confident (0.972 accuracy at 0.889 mean confidence, ECE 0.083). Fitting
is nonetheless *refused* — the confidence barely separates correct from incorrect on a
sample this accurate, so least squares collapses to a constant map that would report full
confidence for everything and disable `min_confidence` entirely. `fit_report` says so
rather than handing back a number that looks fitted.

Two caveats the artifact repeats and you should carry into any claim made from it: 36
cases is small (one flip moves accuracy ~2.8 points), and some gold labels are
judgement calls — "what is the last thing we shipped" asks for an entity *via* an
ordering, and is labeled `temporal` because answering it requires ordering events.

## Observability

| Metric | Labels | Meaning |
|---|---|---|
| `hirn_nlu_decisions_total` | `task`, `source` | decisions by deciding backend; `source="heuristic"` **is the fallback rate** |
| `hirn_nlu_abstentions_total` | `task`, `backend`, `reason` | why a backend declined (`timeout`, `malformed_output`, `low_confidence`, `provider_error`, …) |
| `hirn_nlu_decision_seconds` | `task` | end-to-end latency including the fallback chain |
| `hirn_nlu_confidence` | `task`, `source` | calibrated confidence of accepted decisions |

What to watch:

- **Fallback rate climbing** — a provider is failing, and quality has quietly reverted
  to the deterministic floor. This is the failure mode a hybrid design is most likely
  to hide, which is why it is the first metric.
- **`reason="malformed_output"`** — the provider is ignoring the JSON schema. Check
  whether it supports structured output at all.
- **`reason="low_confidence"` dominating** — either `min_confidence` is set above what
  your calibration supports, or the task's exemplars do not cover your traffic.
- **`reason="timeout"`** — raise `nlu.budget.timeout` or move the task to a local
  model; the write path is the usual victim.

Per-decision provenance is also visible in the API surface: `smart_recall` returns
`route_source` and `route_confidence`, and each `ReflectionUpdate` carries
`decided_by` and `confidence`.

---

## Extending it

Adding a new semantic decision means declaring a task, not writing a matcher:

```rust
const MY_TASK: ClassificationTask = ClassificationTask {
    name: "my_decision",
    instruction: "…what to decide, and to judge meaning over wording",
    labels: &[/* name, description, exemplars */],
    default_label: "the conservative, no-op choice",
};

let decision = db.nlu_classifier()
    .decide(&MY_TASK, text, None, || my_deterministic_fallback())
    .await;
```

Write exemplars for the cases a word list gets wrong — implicit intent, misleading
cue words, scoped negation, passive voice, a non-English phrasing — because those are
what the exemplars have to teach the router that the fallback cannot already do. Make
`default_label` the choice that changes nothing, so an abstention is inert.

Then register it in `hirn_engine::nlu_tasks()`. That registry is what the
`nlu_task_registry` guard iterates, and it enforces the properties a malformed task
would otherwise degrade on silently rather than fail to compile:

- labels are unique, non-empty, and each carries a description;
- `default_label` is a member of the label set;
- **every label has exemplars** — one without them can never win the embedding
  router's argmax, so the router would route around it while still reporting a
  confident decision over the rest;
- task names are unique, since `task.name` is both the `hirn_nlu_*` metric label and
  the exemplar cache key — two tasks sharing one would blend their metrics and serve
  one task the other's cached centroids;
- the generated JSON schema pins the label enum and rejects unknown fields;
- the system prompt names every label and its definition;
- malformed output (unknown label, out-of-range confidence, prose) abstains.

The registry is also what a calibration pass iterates: calibration is fitted per task,
because routing and belief revision differ in label set, prompt, and difficulty.

---

## See also

- [Cognitive Model](@/docs/concepts/cognitive-model.md) — the memory tiers these decisions serve
- [Causal Reasoning](@/docs/concepts/causal.md) — contradiction edges and causal traversal
- [Write-Path Intelligence](@/docs/concepts/write-path.md) — where extraction runs
- [Observability](@/docs/operations/observability.md) — the full metric catalogue
