# Cognitive Memory Benchmark Report

**Run ID:** 01KSHBFNQ08Q9WEG67QPTTAXNZ
**Total time:** 43.00s
**Final Score:** 77.1%
**Geometric Mean:** 77.1%
**Min Suite Score:** 77.1%
**All Competitive:** ✓

## Run Metadata

| Field | Value |
|-------|:------|
| Generated at | 2026-05-26T05:20:09.815637+00:00 |
| Dataset source | external:locomo:auto-download |
| Corpus embedding source | cache:embeddings/locomo_embeddings.json |
| Corpus embedding model | text-embedding-3-small |
| Query embedding source | cache |
| Query embedding model | text-embedding-3-small |
| Embedding dims | 1536 |
| Token budget | 4096 |
| Top-K | 10 |
| Retrieval profile | minimal |
| Execution surface | compiled-hirnql |
| Query-text hybrid | disabled |
| Active retrieval surfaces | enabled: graph, compiled-hirnql; disabled: hybrid, multivector, reranker, tokenizer, quality-gate, iterative-retrieval; notes: compiled_hirnql=true via plain THINK/RECALL execution with diagnostics; quality_gate=false for benchmark minimal-profile parity, while iterative retrieval remains off because benchmark queries use local THINK mode, cache-backed benchmark embedder installed for query-time parity with ingest, minimal profile keeps provider-backed retrieval extras disabled |
| Runs | 3 |
| Environment label | release-20260526 |
| Environment image | - |
| Platform | macos/aarch64 |
| Logical CPUs | 14 |
| Git commit | 2b0787f12360e43a0e77e2651445bf47bac3860b |
| Cargo.lock blake3 | c6ddbe1cb1bcc0178f4b200f3bf8d70d8a59da909f943a0f942812a765dc384d |
| Baseline strategies | full-context, iterative-retrieval |

## Summary

| Benchmark | Containment | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | SOTA Target | Status |
|-----------|------------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|:------------|:-------|
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | 0.7708 | 0.5617 | 0.4410 | 0.5291 | 0.0000 | 34.5 ms | 47.4 ms | 50.8 ms | 635892 | - | - |

## Strategy Comparisons

### LoCoMo (/Users/hupe/.cache/hirn-bench/locomo)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 0.7708 | 0.0029 | 0.5617 | 0.4410 | 0.5291 | 0.0000 | 34.5 ms | 47.4 ms | 50.8 ms | 635892 | - | - | - | 3 runs, max 4.77% (similar) |
| full-context | 0.6199 | 0.0017 | 0.1483 | 0.0109 | 0.0084 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 817715 | +0.1509 | +47.3 ms | -181823 | 3 runs, max 70.65% (drift) |
| iterative-retrieval | 0.6304 | 0.0114 | 0.2725 | 0.1253 | 0.2213 | 0.0000 | 5.7 ms | 6.8 ms | 7.2 ms | 105943 | +0.1404 | +40.6 ms | +529949 | 3 runs, max 7.08% (similar) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 34.5 ms | 47.4 ms | 50.8 ms | 36.8 ms |
| evaluation | 0.1 ms | 0.2 ms | 0.2 ms | 0.1 ms |
| end-to-end | 34.6 ms | 47.4 ms | 50.9 ms | 36.9 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.2 ms | 0.2 ms | 0.2 ms |
| execute-plan | 17.2 ms | 25.1 ms | 26.9 ms | 18.8 ms |
| decode | 2.8 ms | 3.5 ms | 4.1 ms | 2.8 ms |
| assemble | 12.8 ms | 17.4 ms | 20.7 ms | 13.6 ms |
| total | 33.3 ms | 46.3 ms | 49.5 ms | 35.6 ms |

## Reproducibility

| Benchmark | Strategy | Runs | Threshold | Max drift | Mean drift | Status |
|-----------|----------|-----:|----------:|----------:|-----------:|:-------|
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | hirn | 3 | 15.00% | 4.77% | 0.94% | materially similar |
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | full-context | 3 | 15.00% | 70.65% | 5.47% | drift exceeds threshold |
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | iterative-retrieval | 3 | 15.00% | 7.08% | 1.32% | materially similar |

## Reference Baselines (RFC §10)

| Benchmark | System | Score | Source |
|-----------|--------|------:|--------|
| h1-retrieval | Vector DB + RAG (estimated) | 75.0% | Estimated: cosine-recall baseline without reranking |
| h1-retrieval | Zep/Graphiti | 94.8% | Zep DMR benchmark (2024) |
| h1-retrieval | MemGPT/Letta | 93.4% | MemGPT DMR benchmark (2024) |
| h2-temporal | Vector DB + RAG (estimated) | 50.0% | Estimated: no temporal filtering or recency weighting |
| h2-temporal | TraceMem | 72.0% | Maharana et al. 2024 — LoCoMo temporal F1 (Table 3) |
| h3-graph | Vector DB + RAG (estimated) | 40.0% | Estimated: no graph traversal or causal reasoning |
| h3-graph | ActMem | 68.0% | Plausible causal baseline from graph-based LLM memory (2024) |
| h4-agent | Vector DB + RAG (estimated) | 60.0% | Estimated: single-namespace, no isolation |
| h5-action | Vector DB + RAG (estimated) | 55.0% | Estimated: no action/tool memory subsystem |
| h6-safety | Vector DB + RAG (estimated) | 50.0% | Estimated: no adversarial robustness measures |

## LoCoMo (/Users/hupe/.cache/hirn-bench/locomo)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| adversarial | 1.0000 | 0.0000 | 0.4787 | 0.2544 | 0.3805 | 0.0000 | 47 |
| multi-hop | 0.3375 | 0.0017 | 0.3542 | 0.2851 | 0.2757 | 0.0000 | 32 |
| single-hop | 0.9172 | 0.0050 | 0.5986 | 0.4725 | 0.5895 | 0.0000 | 71 |
| temporal | 0.7477 | 0.0033 | 0.8649 | 0.7590 | 0.8615 | 0.0000 | 37 |
| world-knowledge | 0.2747 | 0.0029 | 0.3077 | 0.4231 | 0.4133 | 0.0000 | 13 |

