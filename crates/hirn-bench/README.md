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

By default, `external` runs now enforce safety caps to avoid laptop memory exhaustion:
- `--max-sessions 500`
- `--max-records 10000`
- `--max-queries 200`

Use `--full-corpus` only when you intentionally want an unrestricted run and have enough RAM. For stricter smoke checks, lower limits explicitly, for example:

```bash
cargo run -p hirn-bench -- external --format-name longmemeval --auto-download --runs 1 --no-baselines --max-sessions 50 --max-records 1000 --max-queries 20
```

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
