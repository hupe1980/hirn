# Query-Intent Routing Evaluation

Generated: 2026-08-03T22:35:46.986640+00:00  
Cases: 46  
Backends: `llm:gpt-4o-mini -> embedding:text-embedding-3-small`
Environment: local-dev-m4 · macos aarch64 · 14 cpus  
Commit: `07476b5fbc2320e6c5471b6c895924cef31a8d97`  
Cargo.lock blake3: `b94c9adfe44c8c8dcbe34b4cc655d3b64c27d3a1cb4963e84798ece0d5921667`

## Accuracy

| Arm | Accuracy | 95% CI | Correct | Mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|
| model_backed | 0.9783 | 0.935–1.000 | 45/46 | 880.5 | 1060.0 |
| fallback_only | 0.4348 | 0.283–0.587 | 20/46 | 0.0 | 0.0 |
| majority-class baseline | 0.2609 | — | — | 0.0 | 0.0 |

**Delta (model − fallback): +0.5435**

## Accuracy by category

| Category | Model-backed | Fallback |
|---|---:|---:|
| implicit_intent | 1.0000 (10/10) | 0.4000 |
| literal | 1.0000 (12/12) | 0.8333 |
| misleading_cue | 0.8750 (7/8) | 0.1250 |
| multilingual | 1.0000 (10/10) | 0.2000 |
| passive_voice | 1.0000 (6/6) | 0.5000 |

## Deciding backend

| Source | Decisions |
|---|---:|
| model | 46 |

## Confidence calibration — `model` backend

| Metric | As deployed |
|---|---:|
| Samples | 46 |
| Accuracy | 0.9783 |
| Mean confidence | 0.8783 |
| Expected calibration error | 0.1000 |
| Brier score | 0.0296 |

Fit: fitted scale=1.0000 floor=0.1000

Refit lowers expected calibration error to 0.0000 — safe to adopt.

## Caveats

- **46 cases is a small sample.** One flipped decision moves accuracy by 2.2 points, so treat the per-category rows as directional rather than precise.
- **Gold labels carry judgement.** Some queries admit more than one defensible view — "what is the last thing we shipped" asks for an entity *via* an ordering, and is labeled `temporal` because answering it requires ordering events. A miss on such a case is a disagreement about the label as much as a routing error.
- **The fallback arm is not a straw man.** The `literal` slice exists so the cue list is scored on the cases it was designed for; a unit test fails the build if it drops below 0.5 there.
- **Latency is a real cost.** The model arm adds a provider call per routed query. Weigh the accuracy delta against the p95 above before enabling it on a latency-sensitive path.

## Misroutes (model-backed)

| Query | Expected | Routed | Confidence | Source |
|---|---|---|---:|---|
| what is the most recent invoice i paid | temporal | entity | 0.80 | model |
