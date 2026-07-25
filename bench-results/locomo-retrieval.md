# Cognitive Memory Benchmark Report

**Run ID:** 01KYADDY11X4ESVT25YWBVM1VJ
**Total time:** 124.99s
**Final Score:** 81.5%
**Geometric Mean:** 81.5%
**Min Suite Score:** 81.5%
**All Competitive:** ✓

## Run Metadata

| Field | Value |
|-------|:------|
| Generated at | 2026-07-24T15:58:06.361769+00:00 |
| Dataset source | external:locomo |
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
| Git commit | e2218090719dff118a0e367519052604d3aa3c55 |
| Cargo.lock blake3 | 62e23a51f0f626200c2e9470a8424171e52fa2310315a057c8588e8a6be83d68 |
| Baseline strategies | full-context, iterative-retrieval |
| Seed | 0 |
| Dataset hash (blake3) | 1d1332346b5b05d68e225b87b74b737ef6c61bfdebf0cf156f64af41a26bd5e5 |

### Dataset Files

| File | Blake3 |
|------|:-------|
| locomo10.json | 282cde5689a523eb2bf58d37d95c1f1fece99bb687d3ddae7918311b93b04249 |

## Summary

| Benchmark | Containment | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Tokens / query | Total tokens | SOTA Target | Status |
|-----------|------------:|------------:|----:|-----:|----:|----:|----:|----:|---------------:|-------------:|:------------|:-------|
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | 0.8146 | 0.6155 | 0.4842 | 0.5772 | 0.0000 | 37.6 ms | 61.1 ms | 134.8 ms | 3972 | 6464289 | - | - |

Tokens / query counts the tokens returned to the (hypothetical) reader per executed query — assembled THINK context plus RECALL result contents — using estimator `tiktoken-rs/cl100k_base`.

## Strategy Comparisons

### LoCoMo (/Users/hupe/.cache/hirn-bench/locomo)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Tokens / query | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|---------------:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 0.8146 | 0.0030 | 0.6155 | 0.4842 | 0.5772 | 0.0000 | 37.6 ms | 61.1 ms | 134.8 ms | 3972 (p50 4048 / p95 4746) | 6464289 | - | - | - | single run |
| full-context | 0.5002 | 0.0012 | 0.1063 | 0.0131 | 0.0125 | 0.0000 | 1.2 ms | 1.3 ms | 1.4 ms | 8151 (p50 8158 / p95 8158) | 8117822 | +0.3144 | +59.8 ms | -1653533 | single run |
| iterative-retrieval | 0.6015 | 0.0149 | 0.2536 | 0.1663 | 0.2657 | 0.0000 | 6.9 ms | 8.1 ms | 8.6 ms | 1855 (p50 1926 / p95 2620) | 798359 | +0.2131 | +53.0 ms | +5665930 | single run |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 37.6 ms | 61.1 ms | 134.8 ms | 43.1 ms |
| evaluation | 0.1 ms | 0.1 ms | 0.2 ms | 0.1 ms |
| end-to-end | 37.7 ms | 61.2 ms | 135.0 ms | 43.2 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.2 ms | 0.2 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.3 ms | 0.4 ms | 0.2 ms |
| execute-plan | 18.2 ms | 29.2 ms | 43.8 ms | 20.6 ms |
| decode | 3.2 ms | 4.6 ms | 17.6 ms | 3.7 ms |
| assemble | 14.3 ms | 25.9 ms | 68.3 ms | 16.8 ms |
| total | 36.1 ms | 59.6 ms | 132.4 ms | 41.5 ms |

## Reference Baselines (RFC §10)

| Benchmark | System | Score | Source |
|-----------|--------|------:|--------|
| h1-retrieval | Vector DB + RAG (estimated) | 75.0% | Estimated: cosine-recall baseline without reranking |
| h2-temporal | Vector DB + RAG (estimated) | 50.0% | Estimated: no temporal filtering or recency weighting |
| h3-graph | Vector DB + RAG (estimated) | 40.0% | Estimated: no graph traversal or causal reasoning |
| h4-agent | Vector DB + RAG (estimated) | 60.0% | Estimated: single-namespace, no isolation |
| h5-action | Vector DB + RAG (estimated) | 55.0% | Estimated: no action/tool memory subsystem |
| h6-safety | Vector DB + RAG (estimated) | 50.0% | Estimated: no adversarial robustness measures |

## LoCoMo (/Users/hupe/.cache/hirn-bench/locomo)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| adversarial | 1.0000 | 0.0000 | 0.4473 | 0.2412 | 0.3335 | 0.0000 | 446 |
| multi-hop | 0.3429 | 0.0038 | 0.4936 | 0.5075 | 0.4666 | 0.0000 | 282 |
| single-hop | 0.9474 | 0.0043 | 0.7120 | 0.5580 | 0.6906 | 0.0000 | 841 |
| temporal | 0.7810 | 0.0027 | 0.7817 | 0.6638 | 0.7880 | 0.0000 | 321 |
| world-knowledge | 0.2885 | 0.0042 | 0.3530 | 0.2985 | 0.3354 | 0.0000 | 96 |

