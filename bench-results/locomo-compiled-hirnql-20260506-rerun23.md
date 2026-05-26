# Cognitive Memory Benchmark Report

**Run ID:** 01KQZKPNF7AG0XPC7RNHA77D3B
**Total time:** 179.39s
**Final Score:** 88.1%
**Geometric Mean:** 88.1%
**Min Suite Score:** 88.1%
**All Competitive:** ✓

## Run Metadata

| Field | Value |
|-------|:------|
| Generated at | 2026-05-06T21:44:10.497368+00:00 |
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
| Runs | 1 |
| Environment label | - |
| Environment image | - |
| Platform | macos/aarch64 |
| Logical CPUs | 14 |
| Git commit | 32c5d7e56d3b2600f86d81ff0a847ec3367d1b82 |
| Cargo.lock blake3 | c6a4ce0eb42c159dfb440b29bdfdaa7e542014553cf39f57a2923165bdcf1d90 |
| Baseline strategies | full-context, iterative-retrieval |

## Summary

| Benchmark | Containment | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | SOTA Target | Status |
|-----------|------------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|:------------|:-------|
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | 0.8811 | 0.6156 | 0.4845 | 0.5774 | 0.0000 | 59.9 ms | 127.7 ms | 176.6 ms | 6316855 | - | - |

## Strategy Comparisons

### LoCoMo (/Users/hupe/.cache/hirn-bench/locomo)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 0.8811 | 0.0028 | 0.6156 | 0.4845 | 0.5774 | 0.0000 | 59.9 ms | 127.7 ms | 176.6 ms | 6316855 | - | - | - | single run |
| full-context | 0.5002 | 0.0012 | 0.1063 | 0.0131 | 0.0125 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 8117822 | +0.3809 | +127.6 ms | -1800967 | single run |
| iterative-retrieval | 0.6015 | 0.0149 | 0.2536 | 0.1663 | 0.2657 | 0.0000 | 47.8 ms | 55.6 ms | 59.8 ms | 798359 | +0.2796 | +72.1 ms | +5518496 | single run |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 59.9 ms | 127.7 ms | 176.6 ms | 69.4 ms |
| evaluation | 1.2 ms | 1.4 ms | 1.5 ms | 1.2 ms |
| end-to-end | 61.0 ms | 128.8 ms | 177.7 ms | 70.6 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.2 ms | 0.3 ms | 0.3 ms | 0.2 ms |
| physical-plan | 0.5 ms | 0.7 ms | 0.7 ms | 0.5 ms |
| execute-plan | 13.9 ms | 28.5 ms | 31.9 ms | 16.2 ms |
| decode | 4.3 ms | 5.5 ms | 6.1 ms | 4.4 ms |
| assemble | 39.1 ms | 93.7 ms | 141.8 ms | 46.1 ms |
| total | 58.2 ms | 125.9 ms | 174.8 ms | 67.7 ms |

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
| adversarial | 1.0000 | 0.0000 | 0.4473 | 0.2423 | 0.3343 | 0.0000 | 446 |
| multi-hop | 0.7767 | 0.0044 | 0.4941 | 0.5075 | 0.4671 | 0.0000 | 282 |
| single-hop | 0.9402 | 0.0038 | 0.7120 | 0.5580 | 0.6906 | 0.0000 | 841 |
| temporal | 0.7754 | 0.0024 | 0.7817 | 0.6638 | 0.7881 | 0.0000 | 321 |
| world-knowledge | 0.4722 | 0.0036 | 0.3530 | 0.2985 | 0.3354 | 0.0000 | 96 |

