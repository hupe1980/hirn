# Cognitive Memory Benchmark Report

**Run ID:** 01KZ5T5FR8CH5A33VBD59BZYGQ
**Total time:** 350.72s
**Final Score:** 78.7%
**Geometric Mean:** 78.7%
**Min Suite Score:** 78.7%
**Competitive Gates:** not configured

## Run Metadata

| Field | Value |
|-------|:------|
| Generated at | 2026-08-04T07:27:31.080103+00:00 |
| Dataset source | external:longmemeval |
| Corpus embedding source | cache:embeddings/longmemeval_oracle_embeddings.json |
| Corpus embedding model | text-embedding-3-small |
| Query embedding source | cache |
| Query embedding model | text-embedding-3-small |
| Embedding dims | 1536 |
| Token budget | 4096 |
| Top-K | 10 |
| Retrieval profile | minimal |
| Execution surface | compiled-hirnql |
| Query-text hybrid | disabled |
| Active retrieval surfaces | enabled: per-query-haystack, graph, compiled-hirnql; disabled: hybrid, multivector, reranker, tokenizer, quality-gate, iterative-retrieval; notes: typed_temporal_extraction=false (no temporal envelope — every record ages uniformly (H-01 control arm)), nlu_enabled=true (model-backed routing/relation decisions with deterministic fallback), compiled_hirnql=true via plain THINK/RECALL execution with diagnostics; quality_gate=false for benchmark minimal-profile parity, while iterative retrieval remains off because benchmark queries use local THINK mode, cache-backed benchmark embedder installed for query-time parity with ingest, ambient LLM enabled for query decomposition, minimal profile keeps provider-backed retrieval extras disabled, retrieval and THINK context are isolated to each query's official haystack |
| Runs | 1 |
| Environment label | - |
| Environment image | - |
| Platform | macos/aarch64 |
| Logical CPUs | 14 |
| Git commit | 07476b5fbc2320e6c5471b6c895924cef31a8d97 |
| Cargo.lock blake3 | 40dc8f7052baacf3fb8102b767f759d2e65f879c286ecec7789413bcfb6e43c4 |
| Baseline strategies | disabled |
| Seed | 0 |
| Dataset hash (blake3) | ff7ed687a502556b330b41fee915854b7b944c950fb54c6715a7cb28a1fa9034 |
| Reader model | gpt-4o |
| Judge model | gpt-4o |

### Dataset Files

| File | Blake3 |
|------|:-------|
| longmemeval_oracle | c7b0a5dc8cbb1170629c43e06f0c7029375324acb7c5bd7d73a7959be386d37a |

## Summary

| Benchmark | Containment | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Tokens / query | Total tokens | SOTA Target | Status |
|-----------|------------:|------------:|----:|-----:|----:|----:|----:|----:|---------------:|-------------:|:------------|:-------|
| LongMemEval (/private/tmp/hirn-lme-oracle) | 0.7868 | 0.4560 | 0.3352 | 0.3576 | 0.0000 | 674.6 ms | 1137.8 ms | 2173.9 ms | 4942 | 1316267 | - | - |

Tokens / query counts the tokens returned to the (hypothetical) reader per executed query — assembled THINK context plus RECALL result contents — using estimator `tiktoken-rs/cl100k_base`.

## Reader-Judged Results (LLM QA)

`official_reader_accuracy` is LLM-judged end-to-end QA accuracy over answers the reader generated from the retrieved context. It is a DIFFERENT measurement from the retrieval-only `containment` column above — never compare the two directly.

| Benchmark | Reader | Prompt strategy | Judge | Protocol | official_reader_accuracy | containment (retrieval-only) | Judged | Abstention correct | Reader prompt tok/query (mean, p50/p95) | Reader completion tok/query (mean, p50/p95) |
|-----------|--------|-----------------|-------|----------|-------------------------:|-----------------------------:|-------:|-------------------:|----------------------------------------:|--------------------------------------------:|
| LongMemEval (/private/tmp/hirn-lme-oracle) | gpt-4o | evidence-notes-v2 | gpt-4o | longmemeval-official | 0.7040 | 0.7868 | 500 | 27/30 | 2945 (p50 3284 / p95 3831) | 20 (p50 14 / p95 59) |

Reader-judged accuracy by category (LongMemEval (/private/tmp/hirn-lme-oracle)):

| Category | Accuracy | Queries |
|----------|---------:|--------:|
| knowledge-update | 0.7436 | 78 |
| multi-session | 0.6541 | 133 |
| single-session-assistant | 0.9643 | 56 |
| single-session-preference | 0.3667 | 30 |
| single-session-user | 0.9571 | 70 |
| temporal-reasoning | 0.5639 | 133 |

Reader token counts are EXACT `usage` values from the chat-completions API (publishable cost = reader prompt + completion tokens); `context_tokens_per_query_*` and `tokens_per_query_*` remain estimator-based retrieval-side sizes.

## Strategy Comparisons

### LongMemEval (/private/tmp/hirn-lme-oracle)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Tokens / query | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|---------------:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 0.7868 | 0.0086 | 0.4560 | 0.3352 | 0.3576 | 0.0000 | 674.6 ms | 1137.8 ms | 2173.9 ms | 4942 (p50 5234 / p95 6513) | 1316267 | - | - | - | single run |

Executable baselines were disabled for this run.

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 674.6 ms | 1137.8 ms | 2173.9 ms | 584.1 ms |
| evaluation | 0.1 ms | 0.2 ms | 0.4 ms | 0.1 ms |
| end-to-end | 674.7 ms | 1138.0 ms | 2174.0 ms | 584.3 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.2 ms | 0.3 ms | 0.2 ms |
| physical-plan | 0.3 ms | 0.5 ms | 0.6 ms | 0.3 ms |
| execute-plan | 3.4 ms | 7.6 ms | 15.6 ms | 3.9 ms |
| decode | 545.4 ms | 927.2 ms | 1995.0 ms | 408.5 ms |
| assemble | 9.8 ms | 24.0 ms | 47.4 ms | 11.3 ms |
| total | 672.6 ms | 1135.9 ms | 2171.9 ms | 582.1 ms |

## Reference Baselines (RFC §10)

| Benchmark | System | Score | Source |
|-----------|--------|------:|--------|
| h1-retrieval | Vector DB + RAG (estimated) | 75.0% | Estimated: cosine-recall baseline without reranking |
| h2-temporal | Vector DB + RAG (estimated) | 50.0% | Estimated: no temporal filtering or recency weighting |
| h3-graph | Vector DB + RAG (estimated) | 40.0% | Estimated: no graph traversal or causal reasoning |
| h4-agent | Vector DB + RAG (estimated) | 60.0% | Estimated: single-namespace, no isolation |
| h5-action | Vector DB + RAG (estimated) | 55.0% | Estimated: no action/tool memory subsystem |
| h6-safety | Vector DB + RAG (estimated) | 50.0% | Estimated: no adversarial robustness measures |

## LongMemEval (/private/tmp/hirn-lme-oracle)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| knowledge-update | 0.9277 | 0.0029 | 0.7308 | 0.4314 | 0.4971 | 0.0000 | 78 |
| multi-session | 0.5967 | 0.0034 | 0.3534 | 0.2179 | 0.2490 | 0.0000 | 133 |
| single-session-assistant | 0.9528 | 0.0184 | 0.5536 | 0.3984 | 0.4390 | 0.0000 | 56 |
| single-session-preference | 0.6573 | 0.0494 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 30 |
| single-session-user | 0.9422 | 0.0057 | 0.8143 | 0.7786 | 0.7691 | 0.0000 | 70 |
| temporal-reasoning | 0.7717 | 0.0053 | 0.2707 | 0.2119 | 0.2143 | 0.0000 | 133 |

