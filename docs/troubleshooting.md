# Troubleshooting

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

This guide covers the operator-facing failures that are surfaced by the live codebase today: config validation errors, daemon HTTP status mapping, rate limiting, forwarding timeouts, Cedar denials, provider-side partial failures, offline job stalls, explanation redaction surprises, and resource hydration confusion.

If you are not sure which guide you should be using yet, start with [documentation-map.md](documentation-map.md).

## Quick Triage

| Symptom | Where It Shows Up | Retryable | First Move |
|---------|-------------------|-----------|------------|
| `invalid config: field ...` | Startup, config load, builder construction | No | Fix the field value and restart |
| `access denied: ...` | API response, library error | No | Run `EXPLAIN POLICY` and verify token scopes |
| `not found: ...` | API response, library error | No | Check the memory id, namespace, and realm |
| `already exists: ...` | API response, library error | No | Treat the request as idempotent or remove the duplicate |
| `rate limited: ...` or HTTP `429` | `hirnd` | Yes | Back off and retry later |
| `provider error: ...` | Write path, retrieval, provider setup | Yes | Check credentials, network reachability, and model selection |
| `partial embedding failure: X/Y embeddings succeeded, Z failed` | Batched embedding work | Partial | Retry only the failed subset |
| `timeout: ...` | Provider calls, forwarded writes, long-running operations | Yes | Retry with an idempotency key and inspect upstream latency |
| `limit exceeded: ...` | Graph fan-out, metadata growth, watch limits | Usually no | Reduce load or tune the relevant limit |

## HTTP Status Mapping

The daemon maps a small set of `HirnError` variants to explicit HTTP status codes and treats everything else as a server error.

| HTTP Status | Error Shape | Typical Cause |
|-------------|-------------|---------------|
| `400 Bad Request` | `invalid input: ...` | malformed ids, invalid HirnQL, bad request bodies |
| `403 Forbidden` | `access denied: ...` | Cedar or token-scope denial |
| `404 Not Found` | `not found: ...` | missing memory, realm, or resource |
| `409 Conflict` | `already exists: ...` | duplicate creates or conflicting writes |
| `429 Too Many Requests` | `{class} rate limit exceeded — try again later` | authenticated actor exceeded a route-class throttle budget |
| `500 Internal Server Error` | everything else | provider failure, timeout, limit breach, unexpected runtime failure |

Daemon error bodies also expose a `retryable` flag. Today that is set for HTTP `5xx` responses and explicit daemon throttling responses.

## Rate Limits And Forwarded Writes

`hirnd` enforces route-class throttles by authenticated actor. When a request exceeds the budget, the daemon returns HTTP `429` with a retryable payload.

What to check:

- Repeated write bursts from the same `realm + agent_id` pair.
- Batch-heavy requests sent in tight loops. Large `/v1/execute` and `/v1/consolidate` requests count once per request, not once per inner statement, but rapid retries still consume budget.
- Realm-owner forwarding. Follower nodes proxy writes to the owner, so end-to-end latency includes forwarding.

Operational advice:

- Retry throttled requests with backoff.
- Preserve idempotency keys for client retries so a timed-out forwarded write can be replayed safely.
- Expect forwarded-write client timeouts at about 30 seconds by default; sustained timeouts usually mean owner-node overload or a slow provider dependency.

## Cedar And Token-Scope Denials

If you receive `access denied: ...` or a `403` from `hirnd`, do not start by changing headers. The authoritative decision comes from Cedar policy and authenticated token metadata, not from client-supplied namespace hints.

Recommended checks:

1. Run `EXPLAIN POLICY` for the principal, action, and resource in question.
2. Confirm the token is allowed to perform the requested operation.
3. Confirm the token is allowed to access the requested namespace or realm.
4. Review the relevant policy file under `brain/policies/`.

Useful commands:

```sql
SHOW POLICIES
EXPLAIN POLICY FOR AGENT "researcher" ON REALM "production" ACTION remember
```

Related docs:

- [docs/cedar-guide.md](cedar-guide.md)
- [docs/cedar-patterns.md](cedar-patterns.md)

## Provider Errors, Timeouts, And Partial Embedding Failure

The current provider stack preserves partial progress instead of discarding completed work.

What the error shapes mean:

- `provider error: ...` means the upstream embedder, reranker, or LLM failed.
- `timeout: ...` means the operation exceeded a deadline and can be retried.
- `partial embedding failure: X/Y embeddings succeeded, Z failed` means successful embeddings were preserved and only a subset failed.

Recovery procedure:

1. Inspect provider credentials, base URLs, and model identifiers.
2. Retry only the failed subset when partial embedding failure is reported.
3. Check whether the system degraded gracefully by storing records without embeddings; recall quality may dip temporarily even when writes succeed.
4. If failures repeat, reduce concurrent load and verify provider-specific rate limits.

## Offline Jobs Stuck, Skipped, Or Waiting For Review

The offline scheduler is explicit by design, so most offline cognition issues are visible if you separate queueing, execution, and review.

Common symptoms:

- queue depth rises while completion stays flat
- jobs are skipped repeatedly instead of failing
- a job finishes but the expected cognition never becomes active
- rollback of an approved output is refused

First moves:

1. Check `hirn_offline_job_queue_depth`, `hirn_offline_job_running`, `hirn_offline_job_failed_total`, and `hirn_offline_job_skipped_total` in [observability.md](observability.md).
2. Inspect one exemplar job with `offline_job_status()` or `inspect_offline_job()` before changing any scheduler budget.
3. Confirm the target is narrow enough; oversized `topic`, `goal`, or `temporal_window` scopes often manifest as skipped or budget-capped work rather than a clean failure.
4. Separate review blockage from execution failure. A quarantined result exists, but it is still not an active truth.

If review or rollback behavior is the surprise, read [offline-intelligence.md](offline-intelligence.md) and [security.md](security.md) together. The behavior is intentionally conservative.

## Explanation Redaction And "Missing" Score Details

If a recall or think result is returned but score details or raw text are absent, assume policy redaction before assuming a ranking bug.

What to check:

1. Use `execute_with_explanation()` or `remember_with_explanation()` and inspect the redaction flags and policy summary.
2. Confirm whether the caller is allowed to read raw content or only the existence of the record.
3. Treat missing ranking details and missing raw text as the same class of symptom: policy hid the sensitive payload but preserved enough metadata for auditability.

Related doc:

- [explanation-surfaces.md](explanation-surfaces.md)

## Resource Hydration And Evidence Gaps

Resource-backed evidence is intentionally split into summary, preview, and full hydration surfaces. Most confusion comes from asking one layer for data that only exists in another.

Typical symptoms:

- `resource_evidence` is present but no raw payload is returned
- preview hydration succeeds but full hydration does not
- a derived artifact is expected but not available

Recovery procedure:

1. Inspect the evidence summary first: role, resource id, and available artifact kinds.
2. Choose the lowest hydration mode that answers the question: `MetadataOnly`, then `Preview`, then `Full`.
3. If `Full` fails but preview succeeds, check policy and raw-content permissions rather than assuming the resource is missing.
4. Distinguish source resources from derived artifacts. A source image, its caption, and its thumbnail are different provenance surfaces.

Related docs:

- [getting-started.md](getting-started.md)
- [explanation-surfaces.md](explanation-surfaces.md)

## Config Validation Errors

`HirnConfig` rejects invalid values at construction or deserialization time with field-specific messages.

Common cases:

| Field | Valid Range / Rule |
|-------|---------------------|
| `rpe_fast_path_threshold` | must be in `[0.0, 2.0]` |
| `quality_gate_threshold` | must be in `[0.0, 1.0]` |
| `svo_confidence_threshold` | must be in `[0.0, 1.0]` |
| `interference_consolidation_threshold` | must be `>= 0.0` |
| `consolidation_causal_window` | capped at `10_000` |
| `prospective_indexing_templates` | every template must contain `{content}` |

If the daemon fails during startup, fix the invalid field first. These are not runtime warnings; they are hard configuration errors.

Related doc:

- [docs/performance-tuning.md](performance-tuning.md)

## Semantic Revision Validation And Repair

Revision-native semantic storage is append-only. A healthy chain has one stable `logical_memory_id`, immutable `revision_id` values that match the physical record ID, contiguous versions, and a runtime head cache that agrees with storage.

Library-level validation surface:

```rust
let report = db.admin().validate_semantic_revisions().await?;
if !report.is_clean {
	for issue in &report.issues {
		eprintln!("{issue}");
	}
}

let repair = db.admin().repair_semantic_revisions().await?;
eprintln!("repaired: {:?}", repair.repaired);
eprintln!("failed: {:?}", repair.failed);
```

What validation checks today:

- revision IDs map back to the immutable row ID that stores them
- each logical chain has a version-1 `Create` root and contiguous versions thereafter
- duplicate `revision_id` or duplicate version claims are surfaced explicitly
- cached semantic heads match the authoritative head derived from storage

Repair expectations:

- `repair_semantic_revisions()` rebuilds the runtime semantic head cache from authoritative storage
- missing or stale cached heads are safe to repair automatically
- structural corruption inside the semantic dataset is report-only; hirn will not rewrite or guess a repaired chain in place
- if the repair report still contains failures, treat the listed logical IDs as corrupted and rebuild or purge those chains intentionally

Performance envelope:

- validation performs a full scan of the semantic dataset, so cost grows with total revision count, not only live heads
- repair performs the same scan and then replaces the in-memory head cache for structurally clean chains
- this is an admin/CI operation, not a per-request hot-path check

## Graph And Watch Limits

`limit exceeded: ...` usually means one of the hot-path safety guards fired.

Typical sources:

- automatic edge creation exceeded `max_auto_edges_per_record`
- a graph node hit the per-node fan-out cap
- edge metadata exceeded the hot-tier metadata budget
- watch subscribers exceeded the configured buffer/stream limits

Recovery procedure:

1. Reduce the write burst or graph fan-out created by the current workload.
2. Re-check graph and activation tuning before raising limits.
3. Avoid encoding large opaque payloads into edge metadata.

## Keep Nearby

- [observability.md](observability.md)
- [offline-intelligence.md](offline-intelligence.md)
- [explanation-surfaces.md](explanation-surfaces.md)
- [documentation-map.md](documentation-map.md)

## Escalation Checklist

Before treating an issue as a bug, capture:

- the exact error string
- whether the error was marked retryable
- realm, namespace, and agent identity
- whether the request hit a follower or the realm owner
- whether the failure was library-local, HTTP, gRPC, or MCP
- provider model name and feature flags, if a provider was involved