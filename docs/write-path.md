# Write-Path Intelligence

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> RPE-gated admission, prospective indexing, SVO extraction, and adaptive consolidation.

hirn's write path is not just "store a vector." Every incoming memory passes through an intelligent admission pipeline that decides how much processing to invest based on novelty.

---

## Architecture Overview

```
  ┌──────────┐
  │ remember │
  └────┬─────┘
       │
  ┌────▼──────────────┐
  │  Embed content     │
  └────┬──────────────┘
       │
  ┌────▼──────────────┐
  │  RPE Score         │  reward prediction error
  │  (novelty check)   │
  └────┬──────────────┘
       │
       ├── RPE < 0.3 ──► Fast Path
       │                  • Heuristic importance
       │                  • Skip LLM / PI / SVO
       │                  • Store + graph edges
       │
       └── RPE ≥ 0.3 ──► Slow Path
                          • Full importance scoring
                          • Prospective indexing
                          • SVO extraction
                          • Interference check
                          • Store + graph edges
```

---

## RPE-Gated Admission

**Reward Prediction Error (RPE)** measures how surprising a new memory is relative to existing knowledge. Routine, repetitive content skips expensive processing; novel content gets the full pipeline.

### Computation

1. Embed the incoming content
2. Vector search against episodic, semantic, and procedural datasets
3. Find the maximum similarity: `max_sim`
4. Compute distance: `distance = 1 - max_sim`
5. Z-score against historical population (Welford's online algorithm)
6. `RPE = distance × (1 + z_score)`, clamped to [0, 2]

### Fast Path (RPE < threshold)

- **Importance**: heuristic `0.3 + 0.2 × rpe_score`
- **Skipped**: LLM analysis, prospective indexing, SVO extraction
- **Kept**: embedding, timestamp, metadata, auto-detected similarity edges

### Slow Path (RPE ≥ threshold)

Full pipeline: entity extraction, importance scoring, prospective indexing (if enabled), SVO extraction (if enabled), interference checking, graph edge creation.

### Configuration

```toml
rpe_enabled = true                # Enable RPE routing (default: false)
rpe_fast_path_threshold = 0.3     # D-MEM §4.2 default
rpe_similarity_search_limit = 5   # Neighbors to check
```

---

## Prospective Indexing (Kumiho)

At write time, hirn generates anticipated future questions that the memory could answer. These questions are embedded and stored in the `prospective_implications` dataset. At recall time, queries can match against these prospective embeddings for faster retrieval.

### How It Works

1. Truncate content to 80 characters at word boundary
2. Apply configurable template strings with `{content}` placeholder
3. Batch embed all generated questions
4. Store in `prospective_implications` with FK to source memory

### Configuration

```toml
prospective_indexing_enabled = true
prospective_indexing_num_questions = 5
prospective_indexing_timeout_secs = 5

# Custom templates (default: 5 heuristic what/when/who/outcome/why)
prospective_indexing_templates = [
    "What is known about {content}?",
    "When did {content} happen?",
    "Who was involved in {content}?",
    "What was the outcome of {content}?",
    "Why is {content} important?",
]
```

### HirnQL

```sql
RECALL "what happened at the meeting?" WITH PROSPECTIVE ON;
RECALL "project status" WITH PROSPECTIVE OFF;
```

When `PROSPECTIVE ON`, the logical plan inserts `ProspectiveSearch`, which compiles to `ProspectiveShortCircuitExec`. That operator checks `prospective_implications` first and can skip the full vector scan if a high-confidence prospective hit is found.

---

## SVO Event Extraction (Chronos)

Subject-Verb-Object events are extracted from incoming memories and indexed by calendar time. This enables structured temporal queries like "what happened in March?"

### Extraction

- **Regex fallback** (always available): basic SVO patterns
- Events stored in `svo_events` dataset with:
  - `source_memory_id` FK
  - Subject, verb, object text
  - `time_start` and `time_end` (BTree-indexed)
  - Confidence score

### Configuration

```toml
svo_extraction_enabled = true
svo_confidence_threshold = 0.5   # Minimum confidence to store
```

### HirnQL

```sql
RECALL EVENTS WHERE time >= '2024-03-01' AND time <= '2024-03-31';
```

---

## Interference-Driven Consolidation

When multiple writes create overlapping or contradictory memories, hirn tracks cumulative interference and triggers automatic consolidation when a threshold is exceeded.

### Detection

- **Near-duplicate** (similarity > 0.95): duplicate content detected
- **Supersession**: temporal + entity overlap suggests newer version
- **Contradiction**: NLI-based conflict detection (placeholder)

### Trigger

Cumulative interference accumulates across writes. When it exceeds the threshold, consolidation fires for the affected namespace(s). A cooldown prevents runaway re-triggering.

### Configuration

```toml
interference_consolidation_threshold = 0.3   # Cumulative score to trigger
interference_consolidation_cooldown_secs = 300   # 5-minute cooldown
```

---

## Provider Fallback

The write path degrades gracefully when providers are unavailable:

- **Embedder down**: Store without embedding, log warning. Memory not lost.
- **LLM down during slow path**: Fall back to fast path (heuristic importance, skip LLM analysis).
- **Batch embed failure**: Continue without embeddings for the batch (not batch-fatal).

The `hirn_provider_fallback_total` metric tracks fallback events.

---

## Resource-Backed Modality Routing And Fallbacks

The multimodal ingest path is explicit about which modalities already have first-class resource routing and which degrade paths are in force when richer modality tooling is not available yet.

| Input surface | Resource path today | Derived artifact target | Current fallback behavior |
| --- | --- | --- | --- |
| Image `MemoryContent` | Blob-backed `ResourceObject` with MIME metadata | `Caption` plus explicit fallback `OcrText` from the supplied description text, plus binary `Thumbnail` derived from the source image bytes | If the image bytes exist but the description is empty, the resource still persists and storage records `GenerationFailure` with `intended_kind=caption`; if the image bytes cannot be decoded, storage records `GenerationFailure` with `intended_kind=thumbnail`; the current `OcrText` artifact remains a deterministic text-surrogate fallback rather than real vision OCR. |
| Audio `MemoryContent` | Blob-backed `ResourceObject` with `duration_ms` and optional `channel_count` metadata | `Transcript` from the supplied transcript text | If audio bytes exist but the transcript is empty, the resource still persists and storage records `GenerationFailure` with `intended_kind=transcript`; dedicated speech-to-text fallback is not implemented yet. |
| Video `MemoryContent` | Blob-backed `ResourceObject` with MIME metadata | `Preview` from the supplied transcript/description surrogate | If the video bytes exist but both transcript and description are empty, the resource still persists and storage records `GenerationFailure` with `intended_kind=preview`; thumbnail extraction and richer video/audio enrichment are not implemented yet. |
| Document `MemoryContent` | Blob-backed `ResourceObject` with MIME metadata | `Preview` from the supplied extracted text | If the document bytes exist but the extracted text surrogate is empty, the resource still persists and storage records `GenerationFailure` with `intended_kind=preview`; richer document structure extraction is not implemented yet. |
| External `MemoryContent` | Non-blob `ResourceObject` with `ResourceLocation::External { uri }`, optional MIME/checksum, and typed `fetch_policy` / `stale_at` metadata mirrored into resource metadata | `Preview` from the captured title/snippet/URI surrogate | No live fetch or refresh execution happens on ingest yet; callers can store both local `file://...` handles and remote URLs consistently, and preview artifacts are derived from the captured surrogate text rather than network-fetched content. |
| Tool output `MemoryContent` | Blob-backed `ResourceObject` stamped with `EvidenceRole::Output`, `tool_name`, optional `schema`, and optional `invocation_id` metadata | `Preview` from the tool-name/output surrogate | Tool outputs embed through the text route using the tool name plus output payload, while the stored resource remains typed as text or structured based on MIME/schema; no separate tool-runtime owner executes or refreshes outputs after ingest yet. |
| Code `MemoryContent` | Blob-backed `ResourceObject` with `language` and optional `ast_hash` metadata | `SyntaxSummary` from the source text | If the code source is empty, no resource/artifact is synthesized and the placeholder content remains inline because there is no meaningful source payload to persist. |
| Structured `MemoryContent` | Blob-backed `ResourceObject` with `schema` metadata | `SchemaSummary` from deterministic JSON serialization | Serialization failures surface as write errors; there is no silent fallback that drops the resource. |
| Attachment-style blob extraction | Blob-backed `ResourceObject` whenever a non-empty binary attachment reaches the extraction path | `Preview` from the supplied text surrogate | If the surrogate text is empty, the resource still persists and storage records `GenerationFailure` with `intended_kind=preview`. |
| Composite `MemoryContent` | Each part is routed independently through the same modality owners, then collapsed into one weighted, L2-normalized aggregate embedding bundle | Per-part artifact kind | Parts with empty payloads or unsupported shapes pass through unchanged instead of being force-coerced into generic text. |

Live evidence links now preserve which provenance surface a memory is actually referencing instead of flattening everything back to one parent resource row. Source links are tagged as `observed_resource`; derived artifacts such as OCR text, transcripts, and thumbnails surface as `generated_artifact`; and semantically transformed preview/summarization artifacts such as `Preview`, `Caption`, `SyntaxSummary`, and `SchemaSummary` surface as `transformed_summary`. Image ingest is the clearest example today: one resource-backed image can emit a source resource link plus a caption summary link, an OCR-derived artifact link, and a binary thumbnail artifact. The direct engine remember path no longer size-gates non-empty image, audio, or video payloads before resourceization, so tiny binary inputs now reach the same evidence/provenance surface as larger blobs.

Composite embedding policy is now explicit instead of implied. The current runtime emits one aggregate bundle per composite input by taking per-part modality embeddings, applying `CompositeEmbeddingPolicy::WeightedMeanNormalized`, and L2-normalizing the result. Default modality weights are `1.0` for text, image, audio, video, code, document, and structured parts; callers can override them when configuring multimodal routing. Nested composites still collapse nested branches through their text surrogate inside that single aggregate bundle.

Standalone PDF-like blobs without an existing `multi_content` source are promoted into primary `Document` content before persistence; document blobs remain `Attachment` evidence only when another primary modality is already present or when the MIME is not recognized as a first-class document source.

## Cross-Modal Retrieval Contract

Current cross-modal recall is **derived-text retrieval over modality-specific surrogates**, not raw-binary shared-space ANN over image or audio bytes.

- Recall embeds the caller query text once and searches the single `embedding` column stored on memory rows.
- Image memories are indexed from `description`, audio from `transcript`, video from the transcript/description surrogate, document from `extracted_text`, code from `source`, and structured content from deterministic JSON text.
- Tool-output memories are indexed from `tool_name` plus the raw output payload, with structured-vs-text modality inferred from the optional MIME/schema metadata rather than a separate recall-only modality.
- `derived_artifacts` power hydration, previews, and evidence packaging, but they are not separately vector-indexed recall targets today.
- Modality-specific provider slots can still use different models per route, but cross-modal text recall only works when those models remain dimension-compatible and semantically aligned with the query text space.

Resource-adjacent scalar indexing is now configurable instead of being hard-coded to one dataset-wide recipe.

- `HirnConfig::resource_index_policy` adds extra modality-scoped secondary indices on `resources` by prefixing every configured rule with the physical `modality` column.
- `HirnConfig::derived_artifact_index_policy` adds extra artifact-kind-scoped secondary indices on `derived_artifacts` by prefixing every configured rule with the physical `kind` column.
- These policies tune lookup and filtering paths for resource metadata and derived-artifact fetches; they do not change the current single-embedding recall contract.
- `HirnDB::open_with_config(...)` now runs storage dataset/index bootstrap directly, so these policies take effect on engine startup instead of relying on out-of-band storage initialization.

Example configuration:

```rust
use hirn::HirnConfig;
use hirn::resource::{
    DerivedArtifactIndexPolicy, DerivedArtifactIndexRule, DerivedArtifactKind,
    ModalityProfile, ResourceIndexPolicy, ResourceIndexRule, SecondaryIndexType,
};

let config = HirnConfig::builder()
    .resource_index_policy(
        ResourceIndexPolicy::default().with_rule(
            ResourceIndexRule::new(ModalityProfile::Document, SecondaryIndexType::Bitmap)
                .with_column("mime_type"),
        ),
    )
    .derived_artifact_index_policy(
        DerivedArtifactIndexPolicy::default().with_rule(
            DerivedArtifactIndexRule::new(
                DerivedArtifactKind::Transcript,
                SecondaryIndexType::Bitmap,
            )
            .with_column("modality"),
        ),
    )
    .build()?;
```

## Operator Controls For Resource Memory

The resource path has three operator-owned control planes. Keep them aligned when enabling new modalities in production:

1. **Governance**: use `resource_retention_policy` and `resource_quota_policy` to decide which active resource heads may persist and when payload-bearing resources must redact or purge.
2. **Lookup cost**: use `resource_index_policy` and `derived_artifact_index_policy` to accelerate the modality/kind filters your recall and hydration paths actually use.
3. **Provider routing**: install a routed `MultiModalEmbedder` when different modalities need different embedding providers; `set_embedder(...)` still works, but it wraps one text embedder into the default multimodal router so every non-text route falls back to its textual surrogate.

```rust
use std::sync::Arc;

use hirn::HirnConfig;
use hirn::resource::{
    ModalityProfile, ResourceQuotaPolicy, ResourceQuotaRule, ResourceQuotaScope,
    ResourceRetentionAction, ResourceRetentionPolicy, ResourceRetentionRule,
};
use hirn_provider::{MultiModalEmbedder, PseudoEmbedder};

let config = HirnConfig::builder()
    .resource_retention_policy(
        ResourceRetentionPolicy::default().with_rule(
            ResourceRetentionRule::new(ResourceRetentionAction::Redact)
                .modality(ModalityProfile::Image)
                .classification("restricted"),
        ),
    )
    .resource_quota_policy(
        ResourceQuotaPolicy::default().with_rule(
            ResourceQuotaRule::new(ResourceQuotaScope::Realm)
                .max_total_bytes(10 * 1024 * 1024 * 1024),
        ),
    )
    .build()?;

let multimodal = MultiModalEmbedder::new(Arc::new(PseudoEmbedder::new(384)))
    .with_image_embedder(Arc::new(PseudoEmbedder::new(384)))
    .with_audio_embedder(Arc::new(PseudoEmbedder::new(384)))
    .with_document_embedder(Arc::new(PseudoEmbedder::new(384)));

db.set_multimodal_embedder(Arc::new(multimodal));
```

Hydration stays authorization-sensitive even after the provider side is configured: `HydrationMode::MetadataOnly` and `HydrationMode::Preview` require `Recall`, while `HydrationMode::Full` additionally requires `RecallRawText` for the resource namespace.

Current gaps are deliberate and should be treated as product limitations, not implicit behavior:

- Images now store explicit `Caption`, fallback `OcrText`, and binary `Thumbnail` artifacts. Real OCR extraction is not implemented yet; the current `OcrText` artifact is still generated from the supplied description text rather than vision OCR.
- Richer document structure extraction beyond MIME type plus preview text is not implemented yet.
- Video now has a dedicated `MemoryContent::Video` route, but richer enrichment beyond transcript/description surrogate indexing plus preview-or-failure artifact semantics is not implemented yet.
- Richer audio enrichment beyond transcript plus `duration_ms` / `channel_count` resource metadata is not implemented yet.
- Additional multi-bundle composite emission is not implemented yet; the current contract is one documented weighted aggregate bundle per composite input.

---

## Batch Writes

`batch_remember()` provides the same write-path intelligence as `remember()`:

- Per-record RPE gating (fast/slow path routing)
- Slow-path prospective indexing and SVO extraction (after Lance append)
- Interference tracking across the batch
- TemporalNext edges between consecutive records
- Single Lance `append()` for the batch (not per-record)

### Graph Cleanup

If the batch Lance append fails, orphaned graph nodes are cleaned up (best-effort).

---

## API Examples

### Rust

```rust
use hirn::HirnConfig;

let config = HirnConfig::builder()
    .rpe_enabled(true)
    .rpe_fast_path_threshold(0.3)
    .prospective_indexing_enabled(true)
    .prospective_indexing_num_questions(5)
    .svo_extraction_enabled(true)
    .interference_consolidation_threshold(0.3)
    .build()?;

let db = hirn::open(config).await?;
db.remember("Alice deployed v2.3 on March 15th").await?;
```

### Custom Prospective Templates

```rust
let config = HirnConfig::builder()
    .prospective_indexing_enabled(true)
    .prospective_indexing_templates(vec![
        "What is {content}?".into(),
        "Why does {content} matter?".into(),
        "How is {content} related to the project?".into(),
    ])
    .build()?;
```
