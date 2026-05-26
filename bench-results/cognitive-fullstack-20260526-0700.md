# Cognitive Memory Benchmark Report

**Run ID:** 01KSHE2BSJQ54K9V467NAWFM44
**Total time:** 4.53s
**Final Score:** 100.0%
**Geometric Mean:** 100.0%
**Min Suite Score:** 100.0%
**All Competitive:** ✓

## Run Metadata

| Field | Value |
|-------|:------|
| Generated at | 2026-05-26T06:03:09.921299+00:00 |
| Dataset source | synthetic |
| Corpus embedding source | cache:embeddings/all_embeddings.json |
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
| Synthetic scale | 1 |
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
| H1-Retrieval (synthetic) | 1.0000 | 1.0000 | 0.9333 | 0.9287 | 0.0000 | 6.1 ms | 16.8 ms | 16.8 ms | 9245 | precision@10 ≥ 0.95 | ✓ |
| H2-Temporal (synthetic) | 1.0000 | 1.0000 | 0.9615 | 0.9678 | 1.0000 | 6.2 ms | 9.7 ms | 9.7 ms | 4944 | temporal accuracy ≥ 0.90 | ✓ |
| H3-Graph (synthetic) | 1.0000 | 1.0000 | 0.6314 | 0.7462 | 0.0000 | 6.2 ms | 8.1 ms | 8.1 ms | 8915 | spreading activation paths ≥ 0.95 | ✓ |
| H4-Agent (synthetic) | 1.0000 | 1.0000 | 0.9583 | 0.9777 | 0.0000 | 5.4 ms | 7.5 ms | 7.5 ms | 1736 | consolidation quality ≥ 0.85 | ✓ |
| H5-Action (synthetic) | 1.0000 | 1.0000 | 1.0000 | 0.9866 | 0.0000 | 6.1 ms | 8.3 ms | 8.3 ms | 5881 | noise rejection ≥ 0.90, quality acceptance ≥ 0.95 | ✓ |
| H6-Safety (synthetic) | 1.0000 | 1.0000 | 0.9583 | 0.9777 | 0.0000 | 5.8 ms | 8.3 ms | 8.3 ms | 6191 | cross-modal / adversarial precision ≥ 0.80 | ✓ |

## Strategy Comparisons

### H1-Retrieval (synthetic)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 1.0000 | 0.0226 | 1.0000 | 0.9333 | 0.9287 | 0.0000 | 6.1 ms | 16.8 ms | 16.8 ms | 9245 | - | - | - | 3 runs, max 96.29% (drift) |
| full-context | 1.0000 | 0.0160 | 1.0000 | 0.5074 | 0.6272 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 9054 | +0.0000 | +16.8 ms | +191 | 3 runs, max 8.57% (similar) |
| iterative-retrieval | 0.9333 | 0.0932 | 1.0000 | 0.7317 | 0.7917 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 1540 | +0.0667 | +16.8 ms | +7705 | 3 runs, max 35.65% (drift) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 6.1 ms | 16.8 ms | 16.8 ms | 6.8 ms |
| evaluation | 0.0 ms | 0.1 ms | 0.1 ms | 0.0 ms |
| end-to-end | 6.2 ms | 16.9 ms | 16.9 ms | 6.8 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.3 ms | 0.3 ms | 0.1 ms |
| physical-plan | 0.1 ms | 0.3 ms | 0.3 ms | 0.2 ms |
| execute-plan | 1.1 ms | 2.5 ms | 2.5 ms | 1.2 ms |
| decode | 1.7 ms | 2.4 ms | 2.4 ms | 1.8 ms |
| assemble | 2.8 ms | 3.4 ms | 3.4 ms | 2.9 ms |
| total | 5.9 ms | 8.4 ms | 8.4 ms | 6.1 ms |

### H2-Temporal (synthetic)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 1.0000 | 0.0178 | 1.0000 | 0.9615 | 0.9678 | 1.0000 | 6.2 ms | 9.7 ms | 9.7 ms | 4944 | - | - | - | 3 runs, max 10.71% (similar) |
| full-context | 1.0000 | 0.0143 | 1.0000 | 0.2561 | 0.3127 | 1.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 5299 | +0.0000 | +9.7 ms | -355 | 3 runs, max 33.33% (drift) |
| iterative-retrieval | 0.7308 | 0.0562 | 0.8462 | 0.5769 | 0.6454 | 1.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 1393 | +0.2692 | +9.6 ms | +3551 | 3 runs, max 28.82% (drift) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 6.2 ms | 9.7 ms | 9.7 ms | 6.5 ms |
| evaluation | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| end-to-end | 6.2 ms | 9.7 ms | 9.7 ms | 6.5 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.2 ms | 0.2 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.3 ms | 0.3 ms | 0.2 ms |
| execute-plan | 1.6 ms | 3.8 ms | 3.8 ms | 1.6 ms |
| decode | 1.7 ms | 2.7 ms | 2.7 ms | 1.8 ms |
| assemble | 2.4 ms | 3.3 ms | 3.3 ms | 2.5 ms |
| total | 6.0 ms | 9.4 ms | 9.4 ms | 6.3 ms |

### H3-Graph (synthetic)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 1.0000 | 0.0186 | 1.0000 | 0.6314 | 0.7462 | 0.0000 | 6.2 ms | 8.1 ms | 8.1 ms | 8915 | - | - | - | 3 runs, max 11.85% (similar) |
| full-context | 1.0000 | 0.0181 | 1.0000 | 0.3202 | 0.3573 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 7553 | +0.0000 | +8.1 ms | +1362 | 3 runs, max 24.49% (drift) |
| iterative-retrieval | 0.8910 | 0.1046 | 0.8462 | 0.5239 | 0.6207 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 1269 | +0.1090 | +8.0 ms | +7646 | 3 runs, max 152.16% (drift) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 6.2 ms | 8.1 ms | 8.1 ms | 6.3 ms |
| evaluation | 0.0 ms | 0.1 ms | 0.1 ms | 0.0 ms |
| end-to-end | 6.2 ms | 8.1 ms | 8.1 ms | 6.4 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.2 ms | 0.2 ms | 0.2 ms |
| execute-plan | 1.1 ms | 2.4 ms | 2.4 ms | 1.2 ms |
| decode | 1.7 ms | 1.9 ms | 1.9 ms | 1.7 ms |
| assemble | 2.8 ms | 3.1 ms | 3.1 ms | 2.9 ms |
| total | 6.0 ms | 7.7 ms | 7.7 ms | 6.1 ms |

### H4-Agent (synthetic)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 1.0000 | 0.1054 | 1.0000 | 0.9583 | 0.9777 | 0.0000 | 5.4 ms | 7.5 ms | 7.5 ms | 1736 | - | - | - | 3 runs, max 18.41% (drift) |
| full-context | 1.0000 | 0.0266 | 1.0000 | 0.3808 | 0.4600 | 1.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 6303 | +0.0000 | +7.5 ms | -4567 | 3 runs, max 1.61% (similar) |
| iterative-retrieval | 0.8333 | 0.1065 | 0.9167 | 0.5813 | 0.6544 | 1.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 1626 | +0.1667 | +7.5 ms | +110 | 3 runs, max 3.42% (similar) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 5.4 ms | 7.5 ms | 7.5 ms | 5.6 ms |
| evaluation | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| end-to-end | 5.4 ms | 7.6 ms | 7.6 ms | 5.6 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.3 ms | 0.3 ms | 0.2 ms |
| execute-plan | 1.1 ms | 2.7 ms | 2.7 ms | 1.4 ms |
| decode | 1.7 ms | 2.2 ms | 2.2 ms | 1.7 ms |
| assemble | 2.0 ms | 2.3 ms | 2.3 ms | 2.0 ms |
| total | 5.3 ms | 7.4 ms | 7.4 ms | 5.5 ms |

### H5-Action (synthetic)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 1.0000 | 0.0243 | 1.0000 | 1.0000 | 0.9866 | 0.0000 | 6.1 ms | 8.3 ms | 8.3 ms | 5881 | - | - | - | 3 runs, max 12.86% (similar) |
| full-context | 1.0000 | 0.0159 | 1.0000 | 0.3172 | 0.4669 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 6713 | +0.0000 | +8.2 ms | -832 | 3 runs, max 11.75% (similar) |
| iterative-retrieval | 0.7500 | 0.1094 | 0.9167 | 0.7458 | 0.7706 | 0.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 1308 | +0.2500 | +8.2 ms | +4573 | 3 runs, max 11.64% (similar) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 6.1 ms | 8.3 ms | 8.3 ms | 6.2 ms |
| evaluation | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| end-to-end | 6.1 ms | 8.3 ms | 8.3 ms | 6.2 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.2 ms | 0.2 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.3 ms | 0.3 ms | 0.2 ms |
| execute-plan | 1.1 ms | 2.4 ms | 2.4 ms | 1.2 ms |
| decode | 1.7 ms | 2.2 ms | 2.2 ms | 1.7 ms |
| assemble | 2.6 ms | 3.2 ms | 3.2 ms | 2.7 ms |
| total | 5.9 ms | 8.0 ms | 8.0 ms | 6.0 ms |

### H6-Safety (synthetic)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 1.0000 | 0.0420 | 1.0000 | 0.9583 | 0.9777 | 0.0000 | 5.8 ms | 8.3 ms | 8.3 ms | 6191 | - | - | - | 3 runs, max 11.96% (similar) |
| full-context | 1.0000 | 0.0229 | 1.0000 | 0.2663 | 0.3834 | 1.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 5936 | +0.0000 | +8.3 ms | +255 | 3 runs, max 8.06% (similar) |
| iterative-retrieval | 1.0000 | 0.1523 | 1.0000 | 0.9583 | 0.9547 | 1.0000 | 0.0 ms | 0.0 ms | 0.0 ms | 1060 | +0.0000 | +8.3 ms | +5131 | 3 runs, max 3.12% (similar) |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 5.8 ms | 8.3 ms | 8.3 ms | 6.1 ms |
| evaluation | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| end-to-end | 5.8 ms | 8.3 ms | 8.3 ms | 6.1 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| physical-plan | 0.1 ms | 0.2 ms | 0.2 ms | 0.2 ms |
| execute-plan | 1.1 ms | 2.8 ms | 2.8 ms | 1.3 ms |
| decode | 1.7 ms | 2.7 ms | 2.7 ms | 1.8 ms |
| assemble | 2.3 ms | 3.1 ms | 3.1 ms | 2.4 ms |
| total | 5.6 ms | 8.0 ms | 8.0 ms | 5.9 ms |

## Reproducibility

| Benchmark | Strategy | Runs | Threshold | Max drift | Mean drift | Status |
|-----------|----------|-----:|----------:|----------:|-----------:|:-------|
| H1-Retrieval (synthetic) | hirn | 3 | 15.00% | 96.29% | 24.21% | drift exceeds threshold |
| H1-Retrieval (synthetic) | full-context | 3 | 15.00% | 8.57% | 1.09% | materially similar |
| H1-Retrieval (synthetic) | iterative-retrieval | 3 | 15.00% | 35.65% | 5.50% | drift exceeds threshold |
| H2-Temporal (synthetic) | hirn | 3 | 15.00% | 10.71% | 2.18% | materially similar |
| H2-Temporal (synthetic) | full-context | 3 | 15.00% | 33.33% | 5.16% | drift exceeds threshold |
| H2-Temporal (synthetic) | iterative-retrieval | 3 | 15.00% | 28.82% | 8.71% | drift exceeds threshold |
| H3-Graph (synthetic) | hirn | 3 | 15.00% | 11.85% | 1.74% | materially similar |
| H3-Graph (synthetic) | full-context | 3 | 15.00% | 24.49% | 4.08% | drift exceeds threshold |
| H3-Graph (synthetic) | iterative-retrieval | 3 | 15.00% | 152.16% | 18.09% | drift exceeds threshold |
| H4-Agent (synthetic) | hirn | 3 | 15.00% | 18.41% | 5.63% | drift exceeds threshold |
| H4-Agent (synthetic) | full-context | 3 | 15.00% | 1.61% | 0.34% | materially similar |
| H4-Agent (synthetic) | iterative-retrieval | 3 | 15.00% | 3.42% | 0.58% | materially similar |
| H5-Action (synthetic) | hirn | 3 | 15.00% | 12.86% | 3.73% | materially similar |
| H5-Action (synthetic) | full-context | 3 | 15.00% | 11.75% | 1.21% | materially similar |
| H5-Action (synthetic) | iterative-retrieval | 3 | 15.00% | 11.64% | 2.08% | materially similar |
| H6-Safety (synthetic) | hirn | 3 | 15.00% | 11.96% | 1.76% | materially similar |
| H6-Safety (synthetic) | full-context | 3 | 15.00% | 8.06% | 1.40% | materially similar |
| H6-Safety (synthetic) | iterative-retrieval | 3 | 15.00% | 3.12% | 0.62% | materially similar |

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

## H1-Retrieval (synthetic)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| disambiguation | 1.0000 | 0.0168 | 1.0000 | 1.0000 | 0.9264 | 0.0000 | 5 |
| entity-query | 1.0000 | 0.0371 | 1.0000 | 0.8333 | 0.8890 | 0.0000 | 3 |
| fact-retrieval | 1.0000 | 0.0197 | 1.0000 | 0.9000 | 0.9262 | 0.0000 | 5 |
| fuzzy-query | 1.0000 | 0.0224 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2 |
| negative-retrieval | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |

## H2-Temporal (synthetic)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| event-ordering | 1.0000 | 0.0259 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 3 |
| knowledge-update | 1.0000 | 0.0168 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 5 |
| recency | 1.0000 | 0.0198 | 1.0000 | 0.7500 | 0.8155 | 0.0000 | 2 |
| session-continuity | 1.0000 | 0.0099 | 1.0000 | 1.0000 | 0.9833 | 0.0000 | 3 |
| temporal-contradiction | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 2 |

## H3-Graph (synthetic)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| causal-chain | 1.0000 | 0.0195 | 1.0000 | 0.5167 | 0.6795 | 0.0000 | 5 |
| causal-trap | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |
| contradiction | 1.0000 | 0.0138 | 1.0000 | 0.5417 | 0.6488 | 0.0000 | 3 |
| multi-hop | 1.0000 | 0.0206 | 1.0000 | 0.8000 | 0.8714 | 0.0000 | 5 |

## H4-Agent (synthetic)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| collaboration | 1.0000 | 0.1448 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 5 |
| cross-tenant | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3 |
| isolation | 1.0000 | 0.1168 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 3 |
| unauthorized-access | 1.0000 | 0.0476 | 1.0000 | 0.8750 | 0.9332 | 0.0000 | 4 |

## H5-Action (synthetic)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| config-decision | 1.0000 | 0.0303 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 4 |
| multi-step-plan | 1.0000 | 0.0275 | 1.0000 | 1.0000 | 0.9799 | 0.0000 | 4 |
| negative-retrieval | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2 |
| tool-selection | 1.0000 | 0.0151 | 1.0000 | 1.0000 | 0.9799 | 0.0000 | 4 |

## H6-Safety (synthetic)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| adversarial-robustness | 1.0000 | 0.0298 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 3 |
| conflict-resolution | 1.0000 | 0.0361 | 1.0000 | 0.8333 | 0.9109 | 0.0000 | 3 |
| injection-defense | 1.0000 | 0.0435 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 4 |
| normal-recall | 1.0000 | 0.0359 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2 |
| pii-handling | 1.0000 | 0.0741 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2 |
| pii-leakage | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1 |

