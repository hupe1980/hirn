# Cognitive Memory Benchmark Report

**Run ID:** 01KYAC9PMGHPX5QD93APW2M07E
**Total time:** 78.10s
**Final Score:** 82.0%
**Geometric Mean:** 82.0%
**Min Suite Score:** 82.0%
**All Competitive:** ✓

## Run Metadata

| Field | Value |
|-------|:------|
| Generated at | 2026-07-24T15:42:07.381511+00:00 |
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
| LongMemEval (/private/tmp/claude-501/-Users-hupe-Workspaces-hupe1980-hirn/16e76fa0-5dd5-4b00-b31a-48a92f03b6ac/scratchpad/lme-oracle) | 0.8200 | 0.4340 | 0.2815 | 0.3132 | 0.0000 | 47.0 ms | 91.2 ms | 95.9 ms | 5310 | 1630452 | - | - |

Tokens / query counts the tokens returned to the (hypothetical) reader per executed query — assembled THINK context plus RECALL result contents — using estimator `tiktoken-rs/cl100k_base`.

## Reader-Judged Results (LLM QA)

`official_reader_accuracy` is LLM-judged end-to-end QA accuracy over answers the reader generated from the retrieved context. It is a DIFFERENT measurement from the retrieval-only `containment` column above — never compare the two directly.

| Benchmark | Reader | Judge | Protocol | official_reader_accuracy | containment (retrieval-only) | Judged | Abstention correct | Reader prompt tok/query (mean, p50/p95) | Reader completion tok/query (mean, p50/p95) |
|-----------|--------|-------|----------|-------------------------:|-----------------------------:|-------:|-------------------:|----------------------------------------:|--------------------------------------------:|
| LongMemEval (/private/tmp/claude-501/-Users-hupe-Workspaces-hupe1980-hirn/16e76fa0-5dd5-4b00-b31a-48a92f03b6ac/scratchpad/lme-oracle) | gpt-4o | gpt-4o | longmemeval-official | 0.4740 | 0.8200 | 500 | 30/30 | 3245 (p50 3267 / p95 3497) | 12 (p50 4 / p95 40) |

Reader-judged accuracy by category (LongMemEval (/private/tmp/claude-501/-Users-hupe-Workspaces-hupe1980-hirn/16e76fa0-5dd5-4b00-b31a-48a92f03b6ac/scratchpad/lme-oracle)):

| Category | Accuracy | Queries |
|----------|---------:|--------:|
| knowledge-update | 0.6410 | 78 |
| multi-session | 0.3910 | 133 |
| single-session-assistant | 0.7679 | 56 |
| single-session-preference | 0.1000 | 30 |
| single-session-user | 0.7429 | 70 |
| temporal-reasoning | 0.2782 | 133 |

Reader token counts are EXACT `usage` values from the chat-completions API (publishable cost = reader prompt + completion tokens); `context_tokens_per_query_*` and `tokens_per_query_*` remain estimator-based retrieval-side sizes.

## Strategy Comparisons

### LongMemEval (/private/tmp/claude-501/-Users-hupe-Workspaces-hupe1980-hirn/16e76fa0-5dd5-4b00-b31a-48a92f03b6ac/scratchpad/lme-oracle)

| Strategy | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Exec p50 | Exec p95 | Exec p99 | Tokens / query | Total tokens | Delta containment | Delta Exec p95 | Delta tokens | Reproducibility |
|----------|------------:|---------:|------------:|----:|-----:|----:|----:|----:|----:|---------------:|-------------:|------------------:|----------:|-------------:|:----------------|
| hirn | 0.8200 | 0.0054 | 0.4340 | 0.2815 | 0.3132 | 0.0000 | 47.0 ms | 91.2 ms | 95.9 ms | 5310 (p50 5292 / p95 6622) | 1630452 | - | - | - | single run |
| full-context | 0.4193 | 0.0027 | 0.1460 | 0.0868 | 0.0939 | 0.0000 | 1.2 ms | 1.2 ms | 1.3 ms | 8081 (p50 8116 / p95 8116) | 2029341 | +0.4007 | +89.9 ms | -398889 | single run |
| iterative-retrieval | 0.5387 | 0.0109 | 0.3100 | 0.2070 | 0.2341 | 0.0000 | 23.3 ms | 31.0 ms | 32.9 ms | 6200 (p50 6370 / p95 8081) | 673442 | +0.2812 | +60.2 ms | +957010 | single run |

Strategy note (full-context): Concatenate the entire history until the token budget is exhausted
Strategy note (iterative-retrieval): Lexical multi-hop retrieval with keyword expansion and no graph, policy, or temporal reasoning

Benchmark latencies:
| Component | p50 | p95 | p99 | mean |
|-----------|----:|----:|----:|-----:|
| execution | 47.0 ms | 91.2 ms | 95.9 ms | 51.5 ms |
| evaluation | 0.1 ms | 0.2 ms | 0.3 ms | 0.1 ms |
| end-to-end | 47.2 ms | 91.3 ms | 96.1 ms | 51.7 ms |

Compiled phase timings:
| Phase | p50 | p95 | p99 | mean |
|-------|----:|----:|----:|-----:|
| embed | 0.0 ms | 0.0 ms | 0.0 ms | 0.0 ms |
| optimize | 0.1 ms | 0.1 ms | 0.1 ms | 0.1 ms |
| physical-plan | 0.2 ms | 0.2 ms | 0.2 ms | 0.2 ms |
| execute-plan | 27.4 ms | 71.3 ms | 75.8 ms | 31.7 ms |
| decode | 4.0 ms | 4.4 ms | 4.8 ms | 4.0 ms |
| assemble | 13.4 ms | 16.9 ms | 19.2 ms | 13.6 ms |
| total | 45.1 ms | 89.3 ms | 94.1 ms | 49.7 ms |

## Reference Baselines (RFC §10)

| Benchmark | System | Score | Source |
|-----------|--------|------:|--------|
| h1-retrieval | Vector DB + RAG (estimated) | 75.0% | Estimated: cosine-recall baseline without reranking |
| h2-temporal | Vector DB + RAG (estimated) | 50.0% | Estimated: no temporal filtering or recency weighting |
| h3-graph | Vector DB + RAG (estimated) | 40.0% | Estimated: no graph traversal or causal reasoning |
| h4-agent | Vector DB + RAG (estimated) | 60.0% | Estimated: single-namespace, no isolation |
| h5-action | Vector DB + RAG (estimated) | 55.0% | Estimated: no action/tool memory subsystem |
| h6-safety | Vector DB + RAG (estimated) | 50.0% | Estimated: no adversarial robustness measures |

## LongMemEval (/private/tmp/claude-501/-Users-hupe-Workspaces-hupe1980-hirn/16e76fa0-5dd5-4b00-b31a-48a92f03b6ac/scratchpad/lme-oracle)

| Category | Containment | Token F1 | Recall Acc. | MRR | nDCG | FPR | Queries |
|----------|------------:|---------:|------------:|----:|-----:|----:|--------:|
| knowledge-update | 0.9134 | 0.0026 | 0.6410 | 0.3782 | 0.4254 | 0.0000 | 78 |
| multi-session | 0.6825 | 0.0030 | 0.3985 | 0.2123 | 0.2666 | 0.0000 | 133 |
| single-session-assistant | 0.8862 | 0.0040 | 0.5536 | 0.3960 | 0.4323 | 0.0000 | 56 |
| single-session-preference | 0.7336 | 0.0359 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 30 |
| single-session-user | 0.9664 | 0.0026 | 0.7571 | 0.5721 | 0.5923 | 0.0000 | 70 |
| temporal-reasoning | 0.8171 | 0.0048 | 0.2256 | 0.1562 | 0.1678 | 0.0000 | 133 |

