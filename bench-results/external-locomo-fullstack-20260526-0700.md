# Cognitive Memory Benchmark Report

**Run ID:** 01KSHE5KP63PV3FJC605YS7RAM
**Total time:** 48.22s
**Final Score:** 77.1%
**Geometric Mean:** 77.1%
**Min Suite Score:** 77.1%
**All Competitive:** ✓

## Run Metadata

| Field | Value |
|-------|:------|
| Generated at | 2026-05-26T06:07:21.814461+00:00 |
| Dataset source | external:locomo:auto-download |
| Corpus embedding source | cache:embeddings/locomo_embeddings.json |
| Corpus embedding model | text-embedding-3-small |
| Query embedding source | cache |
| Query embedding model | text-embedding-3-small |
| Embedding dims | 1536 |
| Token budget | 4096 |
| Top-K | 10 |
| Retrieval profile | ablation |
| Execution surface | compiled-hirnql |
| Query-text hybrid | enabled |
| Active retrieval surfaces | enabled: hybrid, graph, reranker, compiled-hirnql, quality-gate; disabled: multivector, tokenizer, iterative-retrieval; notes: compiled_hirnql=true via plain THINK/RECALL execution with diagnostics; quality_gate reflects the compiled read path, while iterative retrieval remains off because benchmark queries use local THINK mode, cache-backed benchmark embedder installed for query-time parity with ingest, default embedder does not support multivector late interaction |
| Runs | 3 |
| Environment label | release-20260526-fullstack |
| Environment image | - |
| Platform | macos/aarch64 |
| Logical CPUs | 14 |
| Git commit | 2b0787f12360e43a0e77e2651445bf47bac3860b |
| Cargo.lock blake3 | c6ddbe1cb1bcc0178f4b200f3bf8d70d8a59da909f943a0f942812a765dc384d |
| Baseline strategies | full-context, iterative-retrieval |

## Summary

| Benchmark | Containment | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | SOTA Target | Status |
|-----------|------------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|:------------|:-------|
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | 0.7711 | 0.5617 | 0.4410 | 0.5291 | 0.0000 | 53.9 ms | 68.1 ms | 77.8 ms | 635746 | - | - |

## Strategy Comparisons

### LoCoMo (/Users/hupe/.cache/hirn-bench/locomo)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 0.7711 | 0.0028 | 0.5617 | 0.4410 | 0.5291 | 0.0000 | 53.9 ms | 68.1 ms | 77.8 ms | 635746 | - | - | - | 3 runs, max 13.30% (similar) |
| full-context | 0.6224 | 0.0017 | 0.1433 | 0.0101 | 0.0063 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 817714 | +0.1488 | +68.1 ms | -181968 | 3 runs, max 32.59% (drift) |
| iterative-retrieval | 0.6354 | 0.0114 | 0.2725 | 0.1253 | 0.2213 | 0.0000 | 6.2 ms | 7.5 ms | 8.3 ms | 105685 | +0.1358 | +60.6 ms | +530061 | 3 runs, max 8.02% (similar) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 53.9 ms | 68.1 ms | 77.8 ms | 56.0 ms |
| evaluation | 0.1 ms | 0.2 ms | 0.2 ms | 0.1 ms |
| end-to-end | 54.1 ms | 68.2 ms | 77.9 ms | 56.2 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.2 ms | 0.3 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.3 ms | 0.6 ms | 0.2 ms |
| execute-plan | 34.8 ms | 44.1 ms | 47.7 ms | 36.3 ms |
| decode | 3.1 ms | 4.1 ms | 4.7 ms | 3.2 ms |
| assemble | 14.3 ms | 19.2 ms | 23.6 ms | 14.9 ms |
| total | 52.7 ms | 66.7 ms | 73.3 ms | 54.8 ms |

## Reproducibility

| Benchmark | Strategy | Runs | Threshold | Max drift | Mean drift | Status |
|-----------|----------|-----:|----------:|----------:|-----------:|:-------|
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | hirn | 3 | 15.00% | 13.30% | 1.69% | materially similar |
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | full-context | 3 | 15.00% | 32.59% | 6.14% | drift exceeds threshold |
| LoCoMo (/Users/hupe/.cache/hirn-bench/locomo) | iterative-retrieval | 3 | 15.00% | 8.02% | 1.65% | materially similar |

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
| adversarial | 1.0000 | 0.0000 | 0.4688 | 0.2491 | 0.3725 | 0.0000 | 48 |
| multi-hop | 0.3396 | 0.0017 | 0.3542 | 0.2851 | 0.2757 | 0.0000 | 32 |
| single-hop | 0.9160 | 0.0051 | 0.6071 | 0.4792 | 0.5980 | 0.0000 | 70 |
| temporal | 0.7477 | 0.0033 | 0.8649 | 0.7590 | 0.8615 | 0.0000 | 37 |
| world-knowledge | 0.2747 | 0.0029 | 0.3077 | 0.4231 | 0.4133 | 0.0000 | 13 |

