# Explanation Surfaces

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

hirn exposes structured explanations because state-of-the-art memory systems need more than ranked results. Operators, evaluators, and downstream agents need to know why a memory was returned, why it was suppressed, and why a write was accepted, rejected, or downgraded.

## What Is Exposed

hirn currently ships three primary explanation surfaces:

- `RecallBuilder::execute_with_explanation()`
- `ThinkBuilder::execute_with_explanation()`
- `EpisodicView::remember_with_explanation()`

Each surface is designed to be useful for both operator-facing UIs and benchmark harnesses without leaking policy-redacted payload details.

## Which Surface Answers Which Question

| Question | Best Surface | Fields to inspect first |
|----------|--------------|-------------------------|
| Why did this result rank here? | Recall explanation | `score_breakdown`, `suppression`, `policy`, `diagnostics` |
| Why did this context pack drop or keep a memory? | Think explanation | `records_included_count`, `records_excluded_count`, `conflict_group_count`, embedded retrieval explanation |
| Why did this write take the slow path or trigger consolidation? | Write-path explanation | `rpe`, `prospective_indexing`, `svo_extraction`, `interference` |
| Why are scores or raw text missing? | Recall or think explanation | `raw_text_redacted`, `ranking_details_redacted`, `policy` |

If you are building an operator console, start with the smallest explanation surface that answers the question. Use metrics or event history only after the request-scoped explanation says the issue is broader than one query or write.

## Retrieval Explanation

`execute_with_explanation()` on recall returns the normal retrieval results plus a `RetrievalExplanation`.

That explanation includes:

- `diagnostics`: query id, time, scanned records, threshold filtering, competitive inhibition, limit truncation, and raw-text redaction counts
- `scoring_weights`: the configured weights used to compute the composite score
- `policy`: a `RetrievalPolicySummary` describing namespace restriction scope and whether raw-text redaction was applied
- `suppression`: candidate count plus how many results were filtered or truncated
- `results`: one explanation entry per returned memory

Each `RetrievedRecordExplanation` contains:

- `memory_id`, `layer`, and optional revision reference
- `composite_score` when ranking details are allowed
- `score_breakdown` when ranking details are allowed
- `raw_text_redacted` and `ranking_details_redacted`
- `resource_evidence_count`

The score breakdown mirrors the live scoring pipeline: similarity, recency decay, importance, activation, causal relevance, surprise, and source reliability.

## Think Explanation

`ThinkBuilder::execute_with_explanation()` returns the assembled context plus a `ThinkExplanation`.

It embeds the full retrieval explanation and adds context-budget details:

- `token_budget`
- `token_count`
- `records_included_count`
- `records_excluded_count`
- `conflict_group_count`
- `query_time_ms`

This is the surface to use when you need to explain why a context pack included one memory and excluded another under a bounded token budget.

## Write-Path Explanation

`remember_with_explanation()` turns the write path into an observable decision surface.

On success it returns `(MemoryId, RememberExplanation)`. On failure it returns `RememberFailure`, which preserves the same explanation payload together with the error.

`RememberExplanation` includes:

- `status`: accepted, rejected, deferred, merged, or failed
- `actor_id`, `namespace`, and `bypass_admission`
- `memory_id` when the write committed
- `admission` details and consulted controllers when admission logic ran
- `embedding` disposition: provided, generated, pending retry, or missing
- `rpe` summary: enabled, score, max similarity, threshold, and fast/slow-path routing
- `text_retention`
- `resources_extracted`
- `prospective_indexing` and `svo_extraction` status/counts
- `interference` disposition, including consolidation triggers when applicable
- `arrival_sequence`
- `error` on failure paths

This is the easiest way to audit why one write stayed on the fast path while another triggered slow-path enrichment or consolidation.

## Common Interpretation Patterns

- A result with no score breakdown is usually a policy-redaction outcome, not a broken ranker.
- A high candidate count with strong suppression often means thresholding, truncation, or competitive inhibition did its job.
- A `RememberFailure` is useful precisely because it preserves the explanation payload; it lets you separate admission rejection, provider degradation, and post-admission failure instead of collapsing them into one generic error path.

## Redaction Rules

Explanations are policy-aware.

- if raw content is redacted, ranking details are redacted too
- callers still get diagnostics, suppression counts, and policy-scope information
- explanations advertise that redaction happened instead of pretending the score data does not exist

This matters for multi-agent or multi-tenant deployments where operators need auditability without leaking protected content.

## Resource-Aware Retrieval

Retrieval results can include resource evidence summaries. Explanations intentionally stay lightweight and pair with the resource surfaces rather than duplicating hydrated payloads.

- retrieval explanations expose `resource_evidence_count`
- JSON recall outputs can include `resource_evidence`, `resource_hydration_available`, and `resource_preview_packages`
- explicit hydration happens separately through `fetch_resource(..., HydrationMode::{MetadataOnly, Preview, Full})`

That split keeps explanation payloads cheap while preserving a clean path to previews or full evidence hydration.

## Rust Examples

Recall with explanation:

```rust
let (results, explanation) = db
    .recall_view()
    .query(query_embedding)
    .limit(5)
    .execute_with_explanation()
    .await?;

println!("candidates: {}", explanation.suppression.candidate_count);
println!("redacted: {}", explanation.raw_text_redacted_results);
```

Think with explanation:

```rust
let (context, explanation) = db
    .recall_view()
    .think(query_embedding)
    .limit(8)
    .budget(120)
    .execute_with_explanation()
    .await?;

println!("included: {}", explanation.records_included_count);
println!("excluded: {}", explanation.records_excluded_count);
println!("tokens: {}", context.token_count);
```

Write path with explanation:

```rust
let (id, explanation) = db
    .episodic()
    .remember_with_explanation(record)
    .await?;

println!("stored: {id}");
println!("status: {:?}", explanation.status);
println!("fast path: {}", explanation.rpe.is_some_and(|rpe| rpe.is_fast_path));
```

## When To Use Them

- use retrieval explanations in ranking audits and benchmark harnesses
- use think explanations in context-packing UIs and token-budget tuning
- use write-path explanations when debugging admission, RPE routing, or interference-driven consolidation

Related docs:

- [documentation-map.md](documentation-map.md)
- [getting-started.md](getting-started.md)
- [hirnql-reference.md](hirnql-reference.md)
- [offline-intelligence.md](offline-intelligence.md)
- [benchmarks.md](benchmarks.md)