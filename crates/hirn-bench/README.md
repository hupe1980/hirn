# hirn-bench

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Benchmark framework for the hirn cognitive memory database. Implements evaluation suites from the cognitive memory research literature plus synthetic performance benchmarks.

The cognitive benchmark artifact now publishes execution metadata with generated-at plus repo/dependency provenance (`git_commit_sha`, `cargo_lock_blake3`), per-query latency p50/p95/p99, token-cost estimates, executable `full-context` / `iterative-retrieval` baseline rows, and reproducibility drift summaries for repeated runs.

The advanced benchmark artifact covers the new offline cognition layer directly: explanation quality, dream hypothesis precision/recall, reconcile accuracy, planning usefulness, and latency or spend envelopes for the deterministic Story 3.2 operator surfaces, with the same generated-at and workspace provenance metadata.

Checked-in publishable output belongs under [bench-results/README.md](../../bench-results/README.md) at the workspace root. Use that directory for nightly or manual evidence you want reviewed in git; keep ad hoc local smoke output out of version control.

## Benchmark Suites

| Suite | Domain | Source |
|-------|--------|--------|
| **LoCoMo-Plus** | Long-context conversational memory | LoCoMo (NeurIPS 2024) |
| **LongMemEval** | Long-term memory evaluation | LongMemEval benchmark |
| **AMemGym** | Agent memory gymnasium | AMemGym framework |
| **CLadder** | Causal reasoning ladder | CLadder (Pearl's hierarchy) |
| **ActMemEval** | Active memory evaluation | ActMemEval benchmark |

## Advanced Offline Cognition

`hirn-bench` now ships a dedicated advanced suite for the Story 3.2 surfaces that do not fit the H1-H6 retrieval schema cleanly:

| Surface | What It Measures |
|---------|------------------|
| **Explanation quality** | Retrieval and write-path explanation completeness and fidelity |
| **Dream hypothesis** | Offline hypothesis provenance coverage and provisional quality gates |
| **Reconcile accuracy** | Deterministic conflict proposal correctness without premature head mutation |
| **Planning usefulness** | Goal-conditioned agenda quality, evidence coverage, and gap detection |

## Synthetic Benchmarks

Performance regression tests:

```bash
cargo bench -p hirn-bench
```

- **Store throughput:** 1000 records, measure ops/sec
- **Recall latency:** p50/p95/p99 over 1000 queries
- **Consolidation throughput:** Full pipeline timing
- **Graph activation:** Spreading activation timing over 10K-node graphs
- **Concurrent load:** mixed remember/recall latency envelopes under parallel writers/readers

## Architecture

```
hirn-bench/src/
├── advanced/     — Advanced offline cognition suite and tracker
├── cognitive/    — Cognitive benchmark implementations
├── dataset.rs    — Dataset loading and management
├── load.rs       — Concurrent mixed remember/recall load benchmark
├── runner.rs     — Benchmark runner framework
├── metrics.rs    — Metrics collection and reporting
├── output.rs     — Result formatting and export
├── compare.rs    — Cross-run comparison
└── storage.rs    — Storage backend for benchmark data
```

## Pre-computed Embeddings

Benchmark embeddings are pre-computed and stored in `embeddings/` at the workspace root. This is the canonical cache directory for real-embedding benchmark runs and is the path used by the benchmark CLI, the docs, and nightly automation.

## Running

```bash
# Criterion microbenchmarks
cargo bench -p hirn-engine

# Cognitive H1-H6 with canonical real embedding caches
cargo run -p hirn-bench -- cognitive --benchmark all --embeddings embeddings/all_embeddings.json --embedding-model-label text-embedding-3-small --runs 2 --repro-threshold-percent 15 --environment-label macos-local --format markdown --output bench-results/cognitive.md --json-output bench-results/cognitive.json

# Focused H2 temporal-contradiction slice (micro-benchmark)
cargo run -p hirn-bench -- cognitive --benchmark h2-temporal-contradiction --embeddings embeddings/all_embeddings.json --embedding-model-label text-embedding-3-small --runs 10 --repro-threshold-percent 15 --environment-label macos-local --format markdown --output bench-results/cognitive-h2-temporal-contradiction.md --json-output bench-results/cognitive-h2-temporal-contradiction.json

# External benchmark adapters with cached embeddings
cargo run -p hirn-bench -- external --format-name locomo --auto-download --embeddings embeddings/locomo_embeddings.json --embedding-model-label text-embedding-3-small --runs 2 --repro-threshold-percent 15

cargo run -p hirn-bench -- external --format-name dmr --auto-download --embeddings embeddings/dmr_embeddings.json --embedding-model-label text-embedding-3-small --runs 2 --repro-threshold-percent 15

cargo run -p hirn-bench -- external --format-name longmemeval --auto-download --embeddings embeddings/longmemeval_embeddings.json --embedding-model-label text-embedding-3-small --runs 2 --repro-threshold-percent 15

cargo run -p hirn-bench -- external --format-name beam --data-dir /path/to/BEAM/chats/100K --embeddings embeddings/beam_embeddings.json --embedding-model-label text-embedding-3-small --runs 2 --repro-threshold-percent 15

# Advanced offline cognition suite
cargo run -p hirn-bench -- advanced --benchmark all --format markdown --output bench-results/advanced.md --json-output bench-results/advanced.json --tracker bench-results/advanced-history.json

# Concurrent load envelope
cargo run -p hirn-bench -- load --writers 4 --readers 8 --writes-per-writer 50 --reads-per-reader 100 --format markdown --output bench-results/load.md --json-output bench-results/load.json

# JSON output
cargo run -p hirn-bench -- cognitive --benchmark all --format json --output results.json

# Fast pseudo-embedding smoke path
cargo run -p hirn-bench -- cognitive --benchmark all --no-baselines
```

LoCoMo auto-download uses the canonical upstream GitHub repository `snap-research/locomo` and downloads `data/locomo10.json` directly. The loader also accepts that raw file layout through `--data-dir`, so a checked-out upstream repo or local mirror works without repacking.

DMR auto-download is intentionally disabled until a verified public canonical dataset source is configured. Use `--data-dir` with a local mirror containing `dialogs.json`.

LongMemEval is downloaded from the dataset repo's published raw files rather than the rows API, because HuggingFace does not expose a working rows endpoint for that corpus. Set `HF_TOKEN` (or the deprecated `HUGGING_FACE_HUB_TOKEN`) or run `huggingface-cli login` if your environment requires authenticated access. The public files are large, so prefer a warm local cache or `--data-dir` for repeated runs.

BEAM (Tavakoli et al., ICLR 2026, arXiv:2510.27246) has no auto-download; clone `github.com/mohammadtavakoli78/BEAM` (or mirror the HuggingFace datasets `Mohammadta/BEAM` / `Mohammadta/BEAM-10M`) and pass `--data-dir` at a tier directory or a directory of conversation folders containing `chat.json` plus `probing_questions/probing_questions.json`.

By default, `external` runs now enforce safety caps to avoid laptop memory exhaustion:
- `--max-sessions 500`
- `--max-records 10000`
- `--max-queries 200`

When any limit drops data, the loader logs a warning and the emitted artifacts carry a machine-readable `truncated {sessions, records, queries}` note; runs also publish tokens/query (mean/p50/p95 plus the estimator label) and per-suite `oracle_assisted` flags.

Use `--full-corpus` only when you intentionally want an unrestricted run and have enough RAM. For stricter smoke checks, lower limits explicitly, for example:

```bash
cargo run -p hirn-bench -- external --format-name longmemeval --auto-download --runs 1 --no-baselines --max-sessions 50 --max-records 1000 --max-queries 20
```

## LLM Reader & Judge (opt-in)

By default every run is **retrieval-only**: no chat-completion network calls are made and the published scores are retrieval metrics (containment, token F1, recall accuracy, MRR, nDCG). The `external` subcommand can additionally run an LLM QA reader and an LLM judge:

LongMemEval uses each question's published `haystack_sessions` as a pre-retrieval
visibility boundary. Both THINK context assembly and RECALL—including temporal
grounding on the direct surface and query decomposition on both surfaces—remain
inside that boundary. The active surface is recorded as
`per-query-haystack`. The reader prompt strategy is versioned and published as
`reader_prompt_strategy`; the current `evidence-notes-v1` strategy reconciles
updates and temporal relations internally while returning only a short final answer.

The current checked-in full oracle result is **0.6500 `official_reader_accuracy`**
over 500 questions, with 0.7913 retrieval containment and 27/30 correct abstentions.
It is reviewable as
[Markdown](../../bench-results/longmemeval-oracle-product-temporal-reader-v2.md) and
[JSON](../../bench-results/longmemeval-oracle-product-temporal-reader-v2.json). It does not
establish SOTA: preference following scores 0.2000 and temporal reasoning 0.3985.

| Flag | Default | Meaning |
|------|---------|---------|
| `--reader <MODEL>` | disabled | Generate an answer per query from the SAME retrieved context the harness scores (e.g. `gpt-4o`). Requires `OPENAI_API_KEY` (or a `.env` file, same convention as `precompute`). |
| `--judge <MODEL>` | disabled | Judge the generated answers (requires `--reader`). LongMemEval runs use the official question-type-aware GPT-4o judge prompts from the LongMemEval repo, including the abstention variant for `_abs` question ids. BEAM runs use a gold-answer-cited yes/no judge (`beam-reader-judged`); other formats get a generic gold-answer-cited judge. |
| `--reader-base-url <URL>` | `$OPENAI_BASE_URL`, else `https://api.openai.com/v1` | Any OpenAI-compatible chat-completions endpoint. |
| `--reader-temperature <F>` | `0.0` | Keep 0.0 for deterministic, publishable runs. |
| `--reader-concurrency <N>` | `4` | Concurrent reader/judge requests (each retried with exponential backoff). |
| `--seed <U64>` | unset | Recorded in provenance. No sampling/subsetting path currently consumes it (the crate uses no RNG); the flag pins provenance for future sampling. |
| `--expect-dataset-hash <HEX>` | unset | Fail fast unless the combined blake3 of the loaded dataset files matches. |

### Honest metrics — never conflate

- `containment` / `token_f1` / `recall_accuracy` are **retrieval-only** scores over the assembled context. They are cheap, deterministic, and NOT comparable to published end-to-end QA numbers.
- `official_reader_accuracy` is **LLM-judged end-to-end QA accuracy** over answers the reader generated from the retrieved context. Only this number is comparable to LongMemEval/BEAM paper results, and only when produced with the official reader/judge models (`gpt-4o`).
- Tokens are reported separately and mean different things:
  - `context_tokens_per_query_{mean,p50,p95}` — retrieval-context size (estimator-based, `tiktoken-rs/cl100k_base`).
  - `tokens_per_query_{mean,p50,p95}` — context plus RECALL result contents returned to a hypothetical reader (estimator-based).
  - `reader_prompt_tokens_per_query_{mean,p50,p95}` / `reader_completion_tokens_per_query_{mean,p50,p95}` — **EXACT** `usage` values from the chat-completions API. Publishable cost per query = reader prompt + completion tokens.

### Dataset pinning

Every `external` run checksums (blake3) the format's known dataset files present under the data directory (e.g. `longmemeval_oracle`/`longmemeval_s`/`longmemeval_m`, every BEAM `chat.json` + `probing_questions.json`) and publishes them in the artifact metadata as `dataset_files` plus a combined `dataset_hash_blake3` (blake3 over the sorted `path\n<hex>\n` lines). Re-run with `--expect-dataset-hash <combined-hash>` to fail fast on any drift. Record the hash printed by your first verified run; upstream files are large and hashes depend on the exact snapshot, so no universal known-good hash is checked in here.

LongMemEval auto-download is pinned to HuggingFace revision `2ec2a557f339b6c0369619b1ed5793734cc87533` of `xiaowu0162/longmemeval` (the `main` commit as of 2025-09-19); the revision is recorded in artifact metadata as `dataset_revision`. Override with `HIRN_BENCH_LME_REVISION=<sha>` when you deliberately target a different snapshot.

### Reproduction runbook: LongMemEval with the official GPT-4o reader

```bash
# 1. Acquire the dataset (pinned HF revision; needs HF_TOKEN if gated for you)
cargo run -p hirn-bench -- precompute-external --format-name longmemeval --auto-download \
  --output embeddings/longmemeval_oracle_embeddings.json

# 2. Full 500-query oracle run with official per-query haystacks, reader, and judge
cargo run --release -p hirn-bench -- external --format-name longmemeval \
  --data-dir "$LONGMEMEVAL_ORACLE_DIR" \
  --embeddings embeddings/longmemeval_oracle_embeddings.json \
  --embedding-model-label text-embedding-3-small --dims 1536 --token-budget 4096 --k 10 \
  --retrieval-profile minimal --execution-surface compiled-hirnql \
  --full-corpus --runs 1 --no-baselines --seed 0 \
  --reader gpt-4o --judge gpt-4o --reader-temperature 0.0 --reader-concurrency 4 \
  --expect-dataset-hash ff7ed687a502556b330b41fee915854b7b944c950fb54c6715a7cb28a1fa9034 \
  --environment-label "macOS arm64; official per-query haystack; evidence-notes-v1; hydrated product temporal+RRF" \
  --format markdown --output bench-results/longmemeval-oracle-product-temporal-reader-v2.md \
  --json-output bench-results/longmemeval-oracle-product-temporal-reader-v2.json

# 3. Re-runs: pin the dataset bytes with the hash printed in step 2
#    (also published as metadata.dataset_hash_blake3 in the JSON artifact)
#    ... --expect-dataset-hash <hash-from-step-2>
```

Cost estimate before running: expected reader spend ≈ `queries × (avg context tokens + question + answer)` — read `context_tokens_per_query_mean` from a prior retrieval-only run. Example: 500 questions × (~4 100 prompt + ~50 completion) tokens ≈ 2.1M prompt + 25K completion tokens per reader pass, plus one judge call per question (≈ question + gold + answer, typically < 300 tokens each). The judge uses temperature 0.0 unconditionally.

### Reproduction runbook: BEAM-10M

BEAM has no auto-download. Acquire the corpus, then point `--data-dir` at the 10M tier:

```bash
git clone https://github.com/mohammadtavakoli78/BEAM   # or HF: Mohammadta/BEAM-10M

# Precompute embeddings for the tier (spend guard: raise --max-api-texts deliberately)
cargo run -p hirn-bench -- precompute-external --format-name beam --data-dir BEAM/chats/10M \
  --output embeddings/beam10m_embeddings.json --max-api-texts 200000

# Full run: full corpus, reader + judge, pinned seed; record the printed dataset hash
cargo run --release -p hirn-bench -- external --format-name beam --data-dir BEAM/chats/10M \
  --embeddings embeddings/beam10m_embeddings.json --embedding-model-label text-embedding-3-small \
  --full-corpus --runs 1 --seed 0 \
  --reader gpt-4o --judge gpt-4o --reader-temperature 0.0 \
  --format markdown --output bench-results/beam10m.md --json-output bench-results/beam10m.json

# Re-runs: add --expect-dataset-hash <hash printed above>
```

Memory behaviour at 10M tokens: `chat.json` batches are streamed out of the file one batch at a time (never a whole-file `Vec` of batches), so the loader's peak overhead beyond the retained dataset is one batch. The retained dataset (all turns of the conversation) is inherently in-memory — a 10M-token conversation is roughly 40–60 MB of text, comfortably within a dev machine. Ingest into the store is batch-bounded (2 000–5 000 records per flush at this scale), so ingest memory does not grow with corpus size. Results are labeled reader-judged (`beam-reader-judged`), not the official BEAM rubric pipeline.

Scores from BEAM runs are per-question averages over the probing questions of the conversations under `--data-dir`; keep tiers separate (one run per tier) so 100K/1M/10M numbers stay comparable.

## Advanced Offline Cognition Workflow

Use this workflow when validating the offline cognition layer end to end:

```bash
# 1. Run the full advanced suite, publish paired artifacts, and update regression history
cargo run -p hirn-bench -- advanced --benchmark all --format markdown --output bench-results/advanced.md --json-output bench-results/advanced.json --tracker bench-results/advanced-history.json

# 2. Compare a candidate run against a checked-in or prior baseline artifact
cargo run -p hirn-bench -- bench-compare --baseline bench-results/advanced-baseline.json --current bench-results/advanced.json --threshold 5.0
```

Enable these advanced operators in production when you need audited explanation surfaces or scheduled offline cognition windows and you can afford the added review surface.

Do not enable them on latency-critical paths, during uncontrolled provider spend conditions, or when you do not have a quarantine or review workflow for generated cognition.
