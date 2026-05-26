use std::collections::BTreeMap;
use std::io::Cursor;

use arrow_array::{Array, BinaryArray};
use image::ImageFormat;

use hirn_core::metadata::Metadata;
use hirn_core::types::{AgentId, Namespace};
use hirn_core::{
    DerivedArtifact, DerivedArtifactKind, EvidenceRole, HydrationMode, LogicalResourceId,
    ModalityProfile, ResourceGovernanceState, ResourceId, ResourceLocation, ResourceObject,
    ResourceQuotaPolicy, ResourceQuotaScope, ResourceRetentionAction, ResourceRetentionPolicy,
    ResourceRevisionId, RevisionOperation, Timestamp,
};

use crate::HirnDbError;
use crate::datasets::{derived_artifact as artifact_ds, resource_blob as blob_ds, resource_object};
use crate::mutation_envelope_ops::{
    MutationEnvelopeRecord, MutationEnvelopeState, list_pending_mutation_envelopes,
    update_mutation_envelope_state,
};
use crate::store::{PhysicalStore, ScanOptions};

pub const RESOURCE_HEAD_TRANSITION_KIND: &str = "resource_head_transition";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct ResourceHeadTransitionEnvelope {
    current_id: ResourceId,
    successor_id: ResourceId,
    successor_created_at_ms: i64,
}

/// Patch-style update describing a superseding resource revision.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResourceSupersession {
    pub reason: Option<String>,
    pub modality: Option<ModalityProfile>,
    pub mime_type: Option<String>,
    pub display_name: Option<String>,
    pub checksum: Option<String>,
    pub size_bytes: Option<u64>,
    pub location: Option<ResourceLocation>,
    pub metadata: Option<Metadata>,
}

/// Lineage-preserving governance update for a resource head.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResourceGovernanceUpdate {
    pub reason: Option<String>,
    pub placeholder_display_name: Option<String>,
}

/// Hydrated resource payload returned from storage fetch operations.
#[derive(Debug, Clone, PartialEq)]
pub struct HydratedResource {
    pub resource: ResourceObject,
    pub artifacts: Vec<DerivedArtifact>,
    pub blob: Option<Vec<u8>>,
}

/// Outcome summary for an operator-triggered retention pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceRetentionApplyResult {
    pub scanned_active_heads: usize,
    pub governed_resources: usize,
    pub redacted_resources: usize,
    pub purged_resources: usize,
    pub skipped_resources: usize,
}

/// Persist a resource object plus its blob payload when applicable.
///
/// When a checksum is present, storage deduplicates within the same namespace
/// and returns the existing resource head instead of creating a duplicate.
pub async fn persist_resource(
    store: &dyn PhysicalStore,
    resource: ResourceObject,
    blob: Option<Vec<u8>>,
) -> Result<ResourceObject, HirnDbError> {
    persist_resource_inner(store, resource, blob, None).await
}

/// Persist a resource while enforcing a configured quota policy.
pub async fn persist_resource_with_quota_policy(
    store: &dyn PhysicalStore,
    resource: ResourceObject,
    blob: Option<Vec<u8>>,
    quota_policy: &ResourceQuotaPolicy,
) -> Result<ResourceObject, HirnDbError> {
    persist_resource_inner(store, resource, blob, Some(quota_policy)).await
}

/// Build a canonical blob-backed resource object for live ingest paths.
pub fn build_configured_blob_resource<F>(
    namespace: Namespace,
    owner_agent_id: AgentId,
    modality: ModalityProfile,
    mime_type: Option<&str>,
    data: &[u8],
    configure: F,
) -> Result<ResourceObject, HirnDbError>
where
    F: FnOnce(
        hirn_core::resource::ResourceObjectBuilder,
    ) -> hirn_core::resource::ResourceObjectBuilder,
{
    let checksum = format!("blake3:{}", blake3::hash(data).to_hex());
    let mut builder = ResourceObject::builder()
        .modality(modality)
        .checksum(checksum)
        .size_bytes(data.len() as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .owner_agent_id(owner_agent_id)
        .namespace(namespace);
    if let Some(mime_type) = mime_type {
        builder = builder.mime_type(mime_type);
    }

    configure(builder)
        .build()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))
}

/// Add standardized audio transport metadata to a resource builder.
pub fn configure_audio_resource_builder(
    builder: hirn_core::resource::ResourceObjectBuilder,
    duration_ms: u64,
    channel_count: Option<u16>,
) -> hirn_core::resource::ResourceObjectBuilder {
    let mut builder = builder.metadata_entry(
        "duration_ms",
        i64::try_from(duration_ms).unwrap_or(i64::MAX),
    );
    if let Some(channel_count) = channel_count {
        builder = builder.metadata_entry("channel_count", i64::from(channel_count));
    }
    builder
}

async fn persist_resource_inner(
    store: &dyn PhysicalStore,
    resource: ResourceObject,
    blob: Option<Vec<u8>>,
    quota_policy: Option<&ResourceQuotaPolicy>,
) -> Result<ResourceObject, HirnDbError> {
    let mut resource = resource;
    let blob = prepare_blob_payload(store, None, &mut resource, blob).await?;

    if let Some(checksum) = resource.checksum.as_deref()
        && let Some(existing) =
            find_live_resource_by_checksum(store, resource.namespace, checksum).await?
    {
        return Ok(existing);
    }

    if let Some(quota_policy) = quota_policy {
        enforce_resource_quota_policy(store, &resource, None, quota_policy).await?;
    }

    append_resource_revision(store, &resource, blob).await?;

    Ok(resource)
}

/// List all revisions in the lineage containing the provided resource revision.
pub async fn list_resource_revisions(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
) -> Result<Vec<ResourceObject>, HirnDbError> {
    let Some(resource) = get_resource_raw(store, resource_id).await? else {
        return Ok(Vec::new());
    };

    list_resource_revisions_for_logical_id(store, resource.logical_resource_id).await
}

/// Resolve the active resource head for the lineage containing the provided revision.
pub async fn get_resource_head(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
) -> Result<Option<ResourceObject>, HirnDbError> {
    let revisions = list_resource_revisions(store, resource_id).await?;
    Ok(select_active_resource_head(&revisions))
}

/// Append a superseding resource revision while preserving historical lookups.
pub async fn supersede_resource(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    supersession: ResourceSupersession,
    blob: Option<Vec<u8>>,
) -> Result<ResourceObject, HirnDbError> {
    supersede_resource_inner(store, resource_id, supersession, blob, None).await
}

/// Append a superseding resource revision while enforcing a configured quota policy.
pub async fn supersede_resource_with_quota_policy(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    supersession: ResourceSupersession,
    blob: Option<Vec<u8>>,
    quota_policy: &ResourceQuotaPolicy,
) -> Result<ResourceObject, HirnDbError> {
    supersede_resource_inner(store, resource_id, supersession, blob, Some(quota_policy)).await
}

async fn supersede_resource_inner(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    supersession: ResourceSupersession,
    blob: Option<Vec<u8>>,
    quota_policy: Option<&ResourceQuotaPolicy>,
) -> Result<ResourceObject, HirnDbError> {
    let Some(current) = get_resource_head(store, resource_id).await? else {
        return Err(HirnDbError::InvalidArgument(format!(
            "resource not found: {resource_id}"
        )));
    };

    let now = Timestamp::now();
    let mut successor = build_successor_revision(
        &current,
        normalize_optional_string(supersession.reason),
        now,
    );

    if let Some(modality) = supersession.modality {
        successor.modality = modality;
    }
    if let Some(mime_type) = supersession.mime_type {
        successor.mime_type = Some(mime_type);
    }
    if let Some(display_name) = supersession.display_name {
        successor.display_name = Some(display_name);
    }
    if let Some(checksum) = supersession.checksum {
        successor.checksum = Some(checksum);
    }
    if let Some(size_bytes) = supersession.size_bytes {
        successor.size_bytes = size_bytes;
    }
    if let Some(location) = supersession.location {
        successor.location = location;
    }
    if let Some(metadata) = supersession.metadata {
        successor.metadata = metadata;
    }

    let blob = prepare_blob_payload(store, Some(&current), &mut successor, blob).await?;
    if let Some(quota_policy) = quota_policy {
        enforce_resource_quota_policy(store, &successor, Some(&current), quota_policy).await?;
    }
    let envelope = build_resource_head_transition_envelope(&current, &successor)?;
    crate::mutation_envelope_ops::append_mutation_envelope(store, &envelope).await?;
    if let Err(error) = append_resource_revision(store, &successor, blob).await {
        let _ = mark_resource_head_transition_failed(store, &envelope.id, &error).await;
        return Err(error);
    }

    let mut updated_current = current.clone();
    updated_current.superseded_by = Some(successor.id);
    updated_current.updated_at = now;
    if let Err(error) = upsert_resource_revision(store, &updated_current).await {
        rollback_resource_revision(store, &successor).await;
        let _ = mark_resource_head_transition_failed(store, &envelope.id, &error).await;
        return Err(error);
    }

    update_mutation_envelope_state(store, &envelope.id, MutationEnvelopeState::Applied, None)
        .await?;

    Ok(successor)
}

pub async fn reconcile_resource_head_mutations(
    store: &dyn PhysicalStore,
) -> Result<usize, HirnDbError> {
    let envelopes =
        list_pending_mutation_envelopes(store, Some(RESOURCE_HEAD_TRANSITION_KIND)).await?;
    let mut reconciled = 0usize;

    for envelope in envelopes {
        match reconcile_single_resource_head_transition(store, &envelope).await {
            Ok(true) => reconciled += 1,
            Ok(false) => {}
            Err(error) => {
                let _ = mark_resource_head_transition_failed(store, &envelope.id, &error).await;
            }
        }
    }

    Ok(reconciled)
}

/// Reconcile resources that were left in `storage_ready = false` staging state by a
/// previous crash or interrupted blob write.
///
/// For each staging record the reconciler checks whether the accompanying blob row
/// exists in `_resource_blobs`.  If the blob is present the record is finalized by
/// flipping `storage_ready = true` via `merge_insert`.  If the blob is absent the
/// partial record is deleted (preventing a dangling metadata row that would never
/// become visible).
///
/// Called once during `HirnDB::open_with_config()` after dataset initialization.
pub async fn reconcile_pending_resource_blob_staging(
    store: &dyn PhysicalStore,
) -> Result<usize, HirnDbError> {
    // Scan for all resource rows — filter must not be pushed to the physical store
    // because `get_resource_raw` already filters on `storage_ready = true`.  We
    // need the raw rows regardless of that flag.
    let filter = "storage_ready = false".to_string();
    let batches = store
        .scan(
            resource_object::DATASET_NAME,
            ScanOptions {
                filter: Some(filter),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await?;

    let mut staging_records: Vec<ResourceObject> = Vec::new();
    for batch in &batches {
        staging_records.extend(
            resource_object::from_batch(batch)?
                .into_iter()
                .filter(|r| !r.storage_ready),
        );
    }

    if staging_records.is_empty() {
        return Ok(0);
    }

    let mut reconciled = 0_usize;
    for mut resource in staging_records {
        let blob_index = match resource.location {
            ResourceLocation::Blob { blob_index } => blob_index,
            _ => {
                // Metadata-only staging record with no blob — finalize it directly.
                resource.storage_ready = true;
                let _ = upsert_resource_revision(store, &resource).await;
                reconciled += 1;
                continue;
            }
        };

        // Check whether the blob was already written.
        match load_resource_blob_unchecked(store, resource.id, blob_index).await {
            Ok(_) => {
                // Blob is present — finalize the metadata record.
                resource.storage_ready = true;
                match upsert_resource_revision(store, &resource).await {
                    Ok(()) => {
                        tracing::debug!(
                            resource_id = %resource.id,
                            "reconciled staging resource: blob present, finalized"
                        );
                        reconciled += 1;
                    }
                    Err(error) => {
                        tracing::warn!(
                            resource_id = %resource.id,
                            %error,
                            "reconcile: failed to finalize staged resource"
                        );
                    }
                }
            }
            Err(_) => {
                // Blob is absent — the write was interrupted before the blob was
                // persisted.  Delete the dangling staging row.
                let filter = format!("id = '{}'", resource.id);
                if let Err(error) = store.delete(resource_object::DATASET_NAME, &filter).await {
                    tracing::warn!(
                        resource_id = %resource.id,
                        %error,
                        "reconcile: failed to delete dangling staging resource row"
                    );
                } else {
                    tracing::debug!(
                        resource_id = %resource.id,
                        "reconcile: deleted dangling staging resource (no blob found)"
                    );
                    reconciled += 1;
                }
            }
        }
    }

    Ok(reconciled)
}

fn build_resource_head_transition_envelope(
    current: &ResourceObject,
    successor: &ResourceObject,
) -> Result<MutationEnvelopeRecord, HirnDbError> {
    let payload = ResourceHeadTransitionEnvelope {
        current_id: current.id,
        successor_id: successor.id,
        successor_created_at_ms: successor.created_at.timestamp_ms(),
    };

    let payload = serde_json::to_vec(&payload).map_err(|error| {
        HirnDbError::InvalidArgument(format!("resource head envelope serialize: {error}"))
    })?;

    Ok(MutationEnvelopeRecord::pending(
        format!("resource-head:{}", successor.id),
        RESOURCE_HEAD_TRANSITION_KIND,
        payload,
    ))
}

async fn reconcile_single_resource_head_transition(
    store: &dyn PhysicalStore,
    envelope: &MutationEnvelopeRecord,
) -> Result<bool, HirnDbError> {
    let payload: ResourceHeadTransitionEnvelope = serde_json::from_slice(&envelope.payload)
        .map_err(|error| {
            HirnDbError::InvalidArgument(format!("resource head envelope deserialize: {error}"))
        })?;

    let current = get_resource_raw(store, payload.current_id).await?;
    let successor = get_resource_raw(store, payload.successor_id).await?;

    match (current, successor) {
        (Some(current), Some(successor)) if current.superseded_by == Some(successor.id) => {
            update_mutation_envelope_state(
                store,
                &envelope.id,
                MutationEnvelopeState::Applied,
                None,
            )
            .await?;
            Ok(false)
        }
        (Some(mut current), Some(successor)) => {
            current.superseded_by = Some(successor.id);
            current.updated_at = Timestamp::from_millis(
                u64::try_from(payload.successor_created_at_ms).map_err(|_| {
                    HirnDbError::InvalidArgument(
                        "resource head envelope successor_created_at_ms was negative".into(),
                    )
                })?,
            );
            upsert_resource_revision(store, &current).await?;
            update_mutation_envelope_state(
                store,
                &envelope.id,
                MutationEnvelopeState::Applied,
                None,
            )
            .await?;
            Ok(true)
        }
        (Some(mut current), None) => {
            if current.superseded_by == Some(payload.successor_id) {
                current.superseded_by = None;
                upsert_resource_revision(store, &current).await?;
            }
            mark_resource_head_transition_failed(
                store,
                &envelope.id,
                &HirnDbError::InvalidArgument(format!(
                    "resource head recovery missing successor revision: {}",
                    payload.successor_id
                )),
            )
            .await?;
            Ok(true)
        }
        (None, Some(successor)) => {
            rollback_resource_revision(store, &successor).await;
            mark_resource_head_transition_failed(
                store,
                &envelope.id,
                &HirnDbError::InvalidArgument(format!(
                    "resource head recovery missing current revision: {}",
                    payload.current_id
                )),
            )
            .await?;
            Ok(true)
        }
        (None, None) => {
            mark_resource_head_transition_failed(
                store,
                &envelope.id,
                &HirnDbError::InvalidArgument(format!(
                    "resource head recovery missing both revisions: {} -> {}",
                    payload.current_id, payload.successor_id
                )),
            )
            .await?;
            Ok(false)
        }
    }
}

async fn mark_resource_head_transition_failed(
    store: &dyn PhysicalStore,
    envelope_id: &str,
    error: &HirnDbError,
) -> Result<(), HirnDbError> {
    update_mutation_envelope_state(
        store,
        envelope_id,
        MutationEnvelopeState::Failed,
        Some(error.to_string()),
    )
    .await
}

/// Create a lineage-preserving redacted placeholder and block future payload hydration.
pub async fn redact_resource(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    update: ResourceGovernanceUpdate,
) -> Result<ResourceObject, HirnDbError> {
    govern_resource(
        store,
        resource_id,
        ResourceGovernanceState::Redacted,
        update,
    )
    .await
}

/// Create a lineage-preserving purged placeholder and block future payload hydration.
pub async fn purge_resource(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    update: ResourceGovernanceUpdate,
) -> Result<ResourceObject, HirnDbError> {
    govern_resource(store, resource_id, ResourceGovernanceState::Purged, update).await
}

/// Apply an operator-configured retention policy to active resource heads.
pub async fn apply_resource_retention_policy(
    store: &dyn PhysicalStore,
    policy: &ResourceRetentionPolicy,
) -> Result<ResourceRetentionApplyResult, HirnDbError> {
    if policy.is_empty() {
        return Ok(ResourceRetentionApplyResult::default());
    }

    let mut result = ResourceRetentionApplyResult::default();
    for resource in list_active_resource_heads(store).await? {
        result.scanned_active_heads += 1;

        let Some(action) = policy.strongest_action_for(&resource) else {
            continue;
        };

        if resource.governance_state == governance_state_for_action(action)
            || resource.governance_state == ResourceGovernanceState::Purged
        {
            result.skipped_resources += 1;
            continue;
        }

        let update = ResourceGovernanceUpdate {
            reason: Some(format!("retention policy {}", action.as_str())),
            placeholder_display_name: None,
        };
        match action {
            ResourceRetentionAction::Redact => {
                redact_resource(store, resource.id, update).await?;
                result.redacted_resources += 1;
            }
            ResourceRetentionAction::Purge => {
                purge_resource(store, resource.id, update).await?;
                result.purged_resources += 1;
            }
        }
        result.governed_resources += 1;
    }

    Ok(result)
}

/// Persist a derived artifact for an existing resource.
pub async fn persist_derived_artifact(
    store: &dyn PhysicalStore,
    artifact: DerivedArtifact,
) -> Result<(), HirnDbError> {
    let batch = artifact_ds::to_batch(std::slice::from_ref(&artifact))?;
    store.append(artifact_ds::DATASET_NAME, batch).await
}

/// Inputs available to the shared derived-artifact planner.
#[derive(Debug, Clone, Copy, Default)]
pub struct DerivedArtifactInput<'a> {
    pub text_content: &'a str,
    pub blob_bytes: Option<&'a [u8]>,
    pub mime_type: Option<&'a str>,
}

impl<'a> DerivedArtifactInput<'a> {
    #[must_use]
    pub const fn new(text_content: &'a str) -> Self {
        Self {
            text_content,
            blob_bytes: None,
            mime_type: None,
        }
    }

    #[must_use]
    pub const fn with_blob(mut self, blob_bytes: &'a [u8], mime_type: Option<&'a str>) -> Self {
        self.blob_bytes = Some(blob_bytes);
        self.mime_type = mime_type;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DefaultTextArtifactPlan {
    kind: DerivedArtifactKind,
    record_failure: bool,
}

impl DefaultTextArtifactPlan {
    const fn new(kind: DerivedArtifactKind, record_failure: bool) -> Self {
        Self {
            kind,
            record_failure,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DefaultBlobArtifactPlan {
    kind: DerivedArtifactKind,
    record_failure: bool,
}

impl DefaultBlobArtifactPlan {
    const fn new(kind: DerivedArtifactKind, record_failure: bool) -> Self {
        Self {
            kind,
            record_failure,
        }
    }
}

const IMAGE_SOURCE_TEXT_ARTIFACTS: [DefaultTextArtifactPlan; 2] = [
    DefaultTextArtifactPlan::new(DerivedArtifactKind::Caption, true),
    DefaultTextArtifactPlan::new(DerivedArtifactKind::OcrText, false),
];
const IMAGE_SOURCE_BLOB_ARTIFACTS: [DefaultBlobArtifactPlan; 1] = [DefaultBlobArtifactPlan::new(
    DerivedArtifactKind::Thumbnail,
    true,
)];
const AUDIO_SOURCE_TEXT_ARTIFACTS: [DefaultTextArtifactPlan; 1] = [DefaultTextArtifactPlan::new(
    DerivedArtifactKind::Transcript,
    true,
)];
const CODE_SOURCE_TEXT_ARTIFACTS: [DefaultTextArtifactPlan; 1] = [DefaultTextArtifactPlan::new(
    DerivedArtifactKind::SyntaxSummary,
    true,
)];
const STRUCTURED_SOURCE_TEXT_ARTIFACTS: [DefaultTextArtifactPlan; 1] =
    [DefaultTextArtifactPlan::new(
        DerivedArtifactKind::SchemaSummary,
        true,
    )];
const PREVIEW_TEXT_ARTIFACTS: [DefaultTextArtifactPlan; 1] = [DefaultTextArtifactPlan::new(
    DerivedArtifactKind::Preview,
    true,
)];
const NO_TEXT_ARTIFACTS: [DefaultTextArtifactPlan; 0] = [];
const NO_BLOB_ARTIFACTS: [DefaultBlobArtifactPlan; 0] = [];
const THUMBNAIL_MAX_DIMENSION_PX: u32 = 256;

/// Persist deterministic derived artifacts from already-available ingest inputs.
pub async fn persist_default_derived_artifacts(
    store: &dyn PhysicalStore,
    resource: &ResourceObject,
    role: EvidenceRole,
    input: DerivedArtifactInput<'_>,
) -> Result<Vec<DerivedArtifact>, HirnDbError> {
    let text_plans = default_text_artifact_plan(resource.modality, role);
    let blob_plans = default_blob_artifact_plan(resource.modality, role);
    if text_plans.is_empty() && blob_plans.is_empty() {
        return Ok(Vec::new());
    }

    let mut known_artifacts = list_derived_artifacts(store, resource.id).await?;
    let text_content = input.text_content.trim();
    let mut created = Vec::new();

    for plan in text_plans {
        if known_artifacts
            .iter()
            .any(|artifact| artifact.kind == plan.kind)
            || known_artifacts
                .iter()
                .any(|artifact| artifact_failure_matches(artifact, plan.kind))
        {
            continue;
        }

        let artifact = if text_content.is_empty() {
            if !plan.record_failure {
                continue;
            }

            build_generation_failure_artifact(resource, role, plan.kind, "source text was empty")?
        } else {
            let mut builder = DerivedArtifact::builder()
                .resource_id(resource.id)
                .kind(plan.kind)
                .modality(ModalityProfile::Text)
                .text_content(text_content)
                .namespace(resource.namespace);

            if resource.modality == ModalityProfile::Image
                && role == EvidenceRole::Source
                && plan.kind == DerivedArtifactKind::OcrText
            {
                builder = builder
                    .metadata_entry("generation_strategy", "text_surrogate_fallback")
                    .metadata_entry("fallback_source", "image_description");
            }

            builder
                .build()
                .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?
        };

        persist_derived_artifact(store, artifact.clone()).await?;
        known_artifacts.push(artifact.clone());
        created.push(artifact);
    }

    for plan in blob_plans {
        if known_artifacts
            .iter()
            .any(|artifact| artifact.kind == plan.kind)
            || known_artifacts
                .iter()
                .any(|artifact| artifact_failure_matches(artifact, plan.kind))
        {
            continue;
        }

        let artifact = match input.blob_bytes {
            Some(blob_bytes) if !blob_bytes.is_empty() => {
                match build_binary_derived_artifact(
                    resource,
                    plan.kind,
                    blob_bytes,
                    input.mime_type,
                    &known_artifacts,
                ) {
                    Ok((artifact, blob_bytes)) => {
                        persist_derived_artifact_with_blob(store, artifact.clone(), blob_bytes)
                            .await?;
                        artifact
                    }
                    Err(error) if plan.record_failure => {
                        let failure =
                            build_generation_failure_artifact(resource, role, plan.kind, &error)?;
                        persist_derived_artifact(store, failure.clone()).await?;
                        failure
                    }
                    Err(_) => continue,
                }
            }
            _ if plan.record_failure => {
                let failure = build_generation_failure_artifact(
                    resource,
                    role,
                    plan.kind,
                    "source blob was unavailable",
                )?;
                persist_derived_artifact(store, failure.clone()).await?;
                failure
            }
            _ => continue,
        };

        known_artifacts.push(artifact.clone());
        created.push(artifact);
    }

    Ok(created)
}

/// Map a derived artifact kind onto the evidence role callers should expose.
#[must_use]
pub const fn derived_artifact_evidence_role(kind: DerivedArtifactKind) -> EvidenceRole {
    match kind {
        DerivedArtifactKind::Preview | DerivedArtifactKind::Thumbnail => EvidenceRole::Preview,
        DerivedArtifactKind::OcrText
        | DerivedArtifactKind::Transcript
        | DerivedArtifactKind::Caption
        | DerivedArtifactKind::SyntaxSummary
        | DerivedArtifactKind::SchemaSummary
        | DerivedArtifactKind::GenerationFailure => EvidenceRole::Derived,
    }
}

/// Build explicit evidence links for derived artifacts created during ingest.
#[must_use]
pub fn evidence_links_for_derived_artifacts(
    artifacts: &[DerivedArtifact],
    part_index: Option<u32>,
) -> Vec<hirn_core::EvidenceLink> {
    artifacts
        .iter()
        .filter(|artifact| artifact.kind != DerivedArtifactKind::GenerationFailure)
        .map(|artifact| {
            let mut link = hirn_core::EvidenceLink::new(
                artifact.resource_id,
                derived_artifact_evidence_role(artifact.kind),
            )
            .with_artifact(artifact.id)
            .with_provenance(artifact.kind.evidence_provenance())
            .with_description(artifact.kind.as_str());
            if let Some(part_index) = part_index {
                link = link.with_part_index(part_index);
            }
            link
        })
        .collect()
}

/// Stable checksum for text-backed resources whose semantics depend on more than raw bytes.
#[must_use]
pub fn text_backed_resource_checksum(discriminator: &str, payload: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(discriminator.as_bytes());
    hasher.update(&[0]);
    hasher.update(payload);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn artifact_failure_matches(
    artifact: &DerivedArtifact,
    intended_kind: DerivedArtifactKind,
) -> bool {
    artifact.kind == DerivedArtifactKind::GenerationFailure
        && matches!(
            artifact.metadata.get("intended_kind"),
            Some(hirn_core::metadata::MetadataValue::String(value)) if value == intended_kind.as_str()
        )
}

fn build_generation_failure_artifact(
    resource: &ResourceObject,
    role: EvidenceRole,
    intended_kind: DerivedArtifactKind,
    reason: &str,
) -> Result<DerivedArtifact, HirnDbError> {
    DerivedArtifact::builder()
        .resource_id(resource.id)
        .kind(DerivedArtifactKind::GenerationFailure)
        .modality(ModalityProfile::Text)
        .text_content(format!(
            "{} generation failed: {reason}",
            intended_kind.as_str()
        ))
        .metadata_entry("intended_kind", intended_kind.as_str().to_string())
        .metadata_entry("failure_reason", reason.to_string())
        .metadata_entry("source_role", role.as_str().to_string())
        .namespace(resource.namespace)
        .build()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))
}

fn build_binary_derived_artifact(
    resource: &ResourceObject,
    kind: DerivedArtifactKind,
    blob_bytes: &[u8],
    mime_type: Option<&str>,
    existing_artifacts: &[DerivedArtifact],
) -> Result<(DerivedArtifact, Vec<u8>), String> {
    match kind {
        DerivedArtifactKind::Thumbnail => {
            let (thumbnail_bytes, width, height) = generate_thumbnail_bytes(blob_bytes, mime_type)?;
            let blob_index = next_derived_artifact_blob_index(existing_artifacts);
            let mut builder = DerivedArtifact::builder()
                .resource_id(resource.id)
                .kind(DerivedArtifactKind::Thumbnail)
                .modality(ModalityProfile::Image)
                .mime_type("image/png")
                .blob_index(blob_index)
                .checksum(format!(
                    "blake3:{}",
                    blake3::hash(&thumbnail_bytes).to_hex()
                ))
                .metadata_entry("generation_strategy", "downscaled_source_image")
                .metadata_entry("max_dimension_px", i64::from(THUMBNAIL_MAX_DIMENSION_PX))
                .metadata_entry("width_px", i64::from(width))
                .metadata_entry("height_px", i64::from(height))
                .namespace(resource.namespace);
            if let Some(mime_type) = mime_type {
                builder = builder.metadata_entry("source_mime_type", mime_type.to_string());
            }
            let artifact = builder.build().map_err(|error| error.to_string())?;
            Ok((artifact, thumbnail_bytes))
        }
        other => Err(format!(
            "unsupported binary derived artifact kind: {}",
            other.as_str()
        )),
    }
}

fn generate_thumbnail_bytes(
    blob_bytes: &[u8],
    mime_type: Option<&str>,
) -> Result<(Vec<u8>, u32, u32), String> {
    let image = if let Some(format) = mime_type.and_then(image_format_from_mime_type) {
        image::load_from_memory_with_format(blob_bytes, format)
    } else {
        image::load_from_memory(blob_bytes)
    }
    .map_err(|error| format!("failed to decode image for thumbnail generation: {error}"))?;

    let thumbnail = image.thumbnail(THUMBNAIL_MAX_DIMENSION_PX, THUMBNAIL_MAX_DIMENSION_PX);
    let width = thumbnail.width();
    let height = thumbnail.height();
    let mut encoded = Cursor::new(Vec::new());
    thumbnail
        .write_to(&mut encoded, ImageFormat::Png)
        .map_err(|error| format!("failed to encode thumbnail image: {error}"))?;
    Ok((encoded.into_inner(), width, height))
}

fn image_format_from_mime_type(mime_type: &str) -> Option<ImageFormat> {
    match mime_type {
        "image/png" => Some(ImageFormat::Png),
        "image/jpeg" | "image/jpg" => Some(ImageFormat::Jpeg),
        "image/gif" => Some(ImageFormat::Gif),
        "image/webp" => Some(ImageFormat::WebP),
        "image/bmp" => Some(ImageFormat::Bmp),
        "image/tiff" => Some(ImageFormat::Tiff),
        _ => None,
    }
}

const fn next_derived_artifact_blob_index(existing_artifacts: &[DerivedArtifact]) -> u32 {
    let mut next = 1;
    let mut idx = 0;
    while idx < existing_artifacts.len() {
        if let Some(blob_index) = existing_artifacts[idx].blob_index
            && blob_index >= next
        {
            next = blob_index + 1;
        }
        idx += 1;
    }
    next
}

async fn persist_derived_artifact_with_blob(
    store: &dyn PhysicalStore,
    artifact: DerivedArtifact,
    blob_bytes: Vec<u8>,
) -> Result<(), HirnDbError> {
    let blob_index = artifact.blob_index.ok_or_else(|| {
        HirnDbError::InvalidArgument("blob-backed derived artifact requires blob_index".into())
    })?;
    let row = blob_ds::ResourceBlobRow {
        resource_id: artifact.resource_id,
        blob_index,
        data: blob_bytes,
    };
    let batch = blob_ds::to_batch(std::slice::from_ref(&row))?;
    store.append(blob_ds::DATASET_NAME, batch).await?;
    if let Err(error) = persist_derived_artifact(store, artifact).await {
        let filter = format!(
            "resource_id = '{}' AND blob_index = {}",
            row.resource_id, row.blob_index
        );
        let _ = store.delete(blob_ds::DATASET_NAME, &filter).await;
        return Err(error);
    }
    Ok(())
}

/// Fetch resource metadata only.
pub async fn get_resource(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
) -> Result<Option<ResourceObject>, HirnDbError> {
    let Some(resource) = get_resource_raw(store, resource_id).await? else {
        return Ok(None);
    };

    sanitize_resource_for_effective_head(store, resource)
        .await
        .map(Some)
}

async fn get_resource_raw(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
) -> Result<Option<ResourceObject>, HirnDbError> {
    let filter = format!("id = '{}'", resource_id);
    let batches = store
        .scan(
            resource_object::DATASET_NAME,
            ScanOptions {
                filter: Some(filter),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: Some(1),
                offset: None,
            },
        )
        .await?;

    for batch in &batches {
        let mut decoded = resource_object::from_batch(batch)?;
        if let Some(resource) = decoded.pop().filter(ResourceObject::is_storage_ready) {
            return Ok(Some(resource));
        }
    }

    Ok(None)
}

/// Load the blob payload for a resource-backed artifact.
pub async fn load_resource_blob(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    blob_index: u32,
) -> Result<Vec<u8>, HirnDbError> {
    let Some(resource) = get_resource_raw(store, resource_id).await? else {
        return Err(HirnDbError::BlobError {
            dataset: resource_object::DATASET_NAME.to_string(),
            details: format!("resource not found or not visible: {resource_id}"),
        });
    };

    if effective_head_for_logical_id(store, resource.logical_resource_id)
        .await?
        .is_some_and(|head| head.governance_state.hides_payload())
    {
        return Err(HirnDbError::BlobError {
            dataset: blob_ds::DATASET_NAME.to_string(),
            details: format!(
                "resource payload unavailable: {resource_id} is governed by the active head"
            ),
        });
    }

    load_resource_blob_unchecked(store, resource_id, blob_index).await
}

async fn load_resource_blob_unchecked(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    blob_index: u32,
) -> Result<Vec<u8>, HirnDbError> {
    let filter = format!(
        "resource_id = '{}' AND blob_index = {}",
        resource_id, blob_index
    );
    let batches = store
        .scan(
            blob_ds::DATASET_NAME,
            ScanOptions {
                filter: Some(filter),
                exact_filter: None,
                columns: Some(vec!["data".to_string()]),
                order_by: None,
                limit: Some(1),
                offset: None,
            },
        )
        .await?;

    let batch = batches.first().ok_or_else(|| HirnDbError::BlobError {
        dataset: blob_ds::DATASET_NAME.to_string(),
        details: format!("resource blob not found: {resource_id}:{blob_index}"),
    })?;
    if batch.num_rows() == 0 {
        return Err(HirnDbError::BlobError {
            dataset: blob_ds::DATASET_NAME.to_string(),
            details: format!("resource blob not found: {resource_id}:{blob_index}"),
        });
    }

    let array = batch
        .column_by_name("data")
        .ok_or_else(|| HirnDbError::InvalidArgument("missing data column".into()))?
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| {
            HirnDbError::InvalidArgument("resource blob data column wrong type".into())
        })?;
    if array.is_null(0) {
        return Err(HirnDbError::BlobError {
            dataset: blob_ds::DATASET_NAME.to_string(),
            details: format!("resource blob was null: {resource_id}:{blob_index}"),
        });
    }

    Ok(array.value(0).to_vec())
}

/// List derived artifacts for a resource.
pub async fn list_derived_artifacts(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
) -> Result<Vec<DerivedArtifact>, HirnDbError> {
    let Some(resource) = get_resource_raw(store, resource_id).await? else {
        return Ok(Vec::new());
    };

    if effective_head_for_logical_id(store, resource.logical_resource_id)
        .await?
        .is_some_and(|head| head.governance_state.hides_payload())
    {
        return Ok(Vec::new());
    }

    let filter = format!("resource_id = '{}'", resource_id);
    let batches = store
        .scan(
            artifact_ds::DATASET_NAME,
            ScanOptions {
                filter: Some(filter),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await?;

    let mut decoded = Vec::new();
    for batch in &batches {
        decoded.extend(artifact_ds::from_batch(batch)?);
    }
    Ok(decoded)
}

/// Fetch a resource with explicit metadata/preview/full hydration semantics.
pub async fn fetch_resource(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    hydration_mode: HydrationMode,
) -> Result<Option<HydratedResource>, HirnDbError> {
    let Some(resource) = get_resource(store, resource_id).await? else {
        return Ok(None);
    };

    let artifacts = if matches!(hydration_mode, HydrationMode::Preview | HydrationMode::Full) {
        list_derived_artifacts(store, resource_id).await?
    } else {
        Vec::new()
    };

    let blob = if matches!(hydration_mode, HydrationMode::Full) {
        match resource.location {
            ResourceLocation::Blob { blob_index } => {
                Some(load_resource_blob(store, resource_id, blob_index).await?)
            }
            ResourceLocation::Inline | ResourceLocation::External { .. } => None,
        }
    } else {
        None
    };

    Ok(Some(HydratedResource {
        resource,
        artifacts,
        blob,
    }))
}

async fn find_live_resource_by_checksum(
    store: &dyn PhysicalStore,
    namespace: Namespace,
    checksum: &str,
) -> Result<Option<ResourceObject>, HirnDbError> {
    let escaped_checksum = checksum.replace('\'', "''");
    let escaped_namespace = namespace.as_str().replace('\'', "''");
    let filter = format!(
        "checksum = '{}' AND namespace = '{}'",
        escaped_checksum, escaped_namespace
    );

    let batches = store
        .scan(
            resource_object::DATASET_NAME,
            ScanOptions {
                filter: Some(filter),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: Some(1),
                offset: None,
            },
        )
        .await?;

    let mut matches = Vec::new();
    for batch in &batches {
        matches.extend(
            resource_object::from_batch(batch)?
                .into_iter()
                .filter(ResourceObject::is_storage_ready),
        );
    }

    Ok(select_live_resource_match(&matches))
}

async fn list_resource_revisions_for_logical_id(
    store: &dyn PhysicalStore,
    logical_resource_id: LogicalResourceId,
) -> Result<Vec<ResourceObject>, HirnDbError> {
    let escaped_logical_id = logical_resource_id.to_string().replace('\'', "''");
    let filter = format!("logical_resource_id = '{}'", escaped_logical_id);
    let batches = store
        .scan(
            resource_object::DATASET_NAME,
            ScanOptions {
                filter: Some(filter),
                exact_filter: None,
                columns: None,
                order_by: None,
                limit: None,
                offset: None,
            },
        )
        .await?;

    let mut revisions = Vec::new();
    for batch in &batches {
        revisions.extend(
            resource_object::from_batch(batch)?
                .into_iter()
                .filter(ResourceObject::is_storage_ready),
        );
    }
    revisions.sort_by(|left, right| {
        left.version
            .cmp(&right.version)
            .then_with(|| left.created_at.millis().cmp(&right.created_at.millis()))
    });
    Ok(revisions)
}

async fn effective_head_for_logical_id(
    store: &dyn PhysicalStore,
    logical_resource_id: LogicalResourceId,
) -> Result<Option<ResourceObject>, HirnDbError> {
    let revisions = list_resource_revisions_for_logical_id(store, logical_resource_id).await?;
    Ok(select_active_resource_head(&revisions))
}

async fn sanitize_resource_for_effective_head(
    store: &dyn PhysicalStore,
    resource: ResourceObject,
) -> Result<ResourceObject, HirnDbError> {
    let head = effective_head_for_logical_id(store, resource.logical_resource_id).await?;
    Ok(apply_effective_head_governance(resource, head.as_ref()))
}

fn apply_effective_head_governance(
    mut resource: ResourceObject,
    head: Option<&ResourceObject>,
) -> ResourceObject {
    let Some(head) = head.filter(|head| head.governance_state.hides_payload()) else {
        return resource;
    };

    resource.governance_state = head.governance_state;
    resource.governance_reason = head.governance_reason.clone();
    resource.governed_at = head.governed_at;
    resource.location = ResourceLocation::Inline;
    resource.checksum = None;
    resource.size_bytes = 0;
    resource.mime_type = None;
    resource.display_name = Some(
        head.display_name
            .clone()
            .filter(|name| !name.trim().is_empty())
            .unwrap_or_else(|| head.governance_state.placeholder_display_name().to_string()),
    );
    resource
}

fn select_active_resource_head(revisions: &[ResourceObject]) -> Option<ResourceObject> {
    revisions
        .iter()
        .filter(|resource| resource.is_storage_ready() && resource.superseded_by.is_none())
        .max_by_key(|resource| resource.version)
        .cloned()
        .or_else(|| {
            revisions
                .iter()
                .filter(|resource| resource.is_storage_ready())
                .max_by_key(|resource| resource.version)
                .cloned()
        })
}

fn select_live_resource_match(revisions: &[ResourceObject]) -> Option<ResourceObject> {
    revisions
        .iter()
        .filter(|resource| resource.is_storage_ready() && resource.superseded_by.is_none())
        .max_by_key(|resource| resource.version)
        .cloned()
}

fn build_successor_revision(
    current: &ResourceObject,
    reason: Option<String>,
    now: Timestamp,
) -> ResourceObject {
    let mut successor = current.clone();
    let successor_id = ResourceId::new();
    successor.id = successor_id;
    successor.logical_resource_id = current.logical_resource_id;
    successor.revision_id = ResourceRevisionId::from_resource_id(successor_id);
    successor.version = current.version + 1;
    successor.revision_operation = RevisionOperation::Supersede;
    successor.revision_reason = reason;
    successor.revision_causation_id = Some(current.id);
    successor.superseded_by = None;
    successor.created_at = now;
    successor.updated_at = now;
    successor
}

async fn govern_resource(
    store: &dyn PhysicalStore,
    resource_id: ResourceId,
    state: ResourceGovernanceState,
    update: ResourceGovernanceUpdate,
) -> Result<ResourceObject, HirnDbError> {
    let Some(current) = get_resource_head(store, resource_id).await? else {
        return Err(HirnDbError::InvalidArgument(format!(
            "resource not found: {resource_id}"
        )));
    };
    if current.governance_state == state {
        return Ok(current);
    }

    let now = Timestamp::now();
    let reason = normalize_optional_string(update.reason)
        .or_else(|| Some(format!("resource {}", state.as_str())));
    let mut successor = build_successor_revision(&current, reason.clone(), now);
    successor.governance_state = state;
    successor.governance_reason = reason;
    successor.governed_at = Some(now);
    successor.display_name = Some(
        normalize_optional_string(update.placeholder_display_name)
            .unwrap_or_else(|| state.placeholder_display_name().to_string()),
    );
    successor.mime_type = None;
    successor.checksum = None;
    successor.size_bytes = 0;
    successor.location = ResourceLocation::Inline;

    append_resource_revision(store, &successor, None).await?;

    let mut updated_current = current.clone();
    updated_current.superseded_by = Some(successor.id);
    updated_current.updated_at = now;
    if let Err(error) = upsert_resource_revision(store, &updated_current).await {
        rollback_resource_revision(store, &successor).await;
        return Err(error);
    }

    let _ = delete_lineage_payloads_and_artifacts(store, current.logical_resource_id).await;

    Ok(successor)
}

async fn list_active_resource_heads(
    store: &dyn PhysicalStore,
) -> Result<Vec<ResourceObject>, HirnDbError> {
    let batches = store
        .scan(resource_object::DATASET_NAME, ScanOptions::default())
        .await?;
    let mut grouped: BTreeMap<LogicalResourceId, Vec<ResourceObject>> = BTreeMap::new();
    for batch in &batches {
        for resource in resource_object::from_batch(batch)? {
            if !resource.is_storage_ready() {
                continue;
            }
            grouped
                .entry(resource.logical_resource_id)
                .or_default()
                .push(resource);
        }
    }

    Ok(grouped
        .into_values()
        .filter_map(|revisions| select_active_resource_head(&revisions))
        .collect())
}

const fn governance_state_for_action(action: ResourceRetentionAction) -> ResourceGovernanceState {
    match action {
        ResourceRetentionAction::Redact => ResourceGovernanceState::Redacted,
        ResourceRetentionAction::Purge => ResourceGovernanceState::Purged,
    }
}

async fn delete_lineage_payloads_and_artifacts(
    store: &dyn PhysicalStore,
    logical_resource_id: LogicalResourceId,
) -> Result<(), HirnDbError> {
    let resource_ids = list_resource_revisions_for_logical_id(store, logical_resource_id)
        .await?
        .into_iter()
        .map(|resource| resource.id)
        .collect::<Vec<_>>();
    if resource_ids.is_empty() {
        return Ok(());
    }

    delete_rows_for_resource_ids(store, blob_ds::DATASET_NAME, "resource_id", &resource_ids)
        .await?;
    delete_rows_for_resource_ids(
        store,
        artifact_ds::DATASET_NAME,
        "resource_id",
        &resource_ids,
    )
    .await
}

async fn delete_rows_for_resource_ids(
    store: &dyn PhysicalStore,
    dataset: &str,
    column: &str,
    resource_ids: &[ResourceId],
) -> Result<(), HirnDbError> {
    if resource_ids.is_empty() {
        return Ok(());
    }

    let filter = resource_ids
        .iter()
        .map(|resource_id| {
            format!(
                "{column} = '{}'",
                resource_id.to_string().replace('\'', "''")
            )
        })
        .collect::<Vec<_>>()
        .join(" OR ");
    store.delete(dataset, &filter).await.map(|_| ())
}

async fn append_resource_revision(
    store: &dyn PhysicalStore,
    resource: &ResourceObject,
    blob: Option<Vec<u8>>,
) -> Result<(), HirnDbError> {
    let mut staged_resource = resource.clone();
    let requires_finalize = matches!(
        (&staged_resource.location, blob.as_ref()),
        (ResourceLocation::Blob { .. }, Some(_))
    );
    if requires_finalize {
        staged_resource.storage_ready = false;
    }

    let batch = resource_object::to_batch(std::slice::from_ref(&staged_resource))?;
    store.append(resource_object::DATASET_NAME, batch).await?;

    if let (ResourceLocation::Blob { blob_index }, Some(blob_bytes)) =
        (&staged_resource.location, blob)
    {
        let row = blob_ds::ResourceBlobRow {
            resource_id: staged_resource.id,
            blob_index: *blob_index,
            data: blob_bytes,
        };
        let batch = blob_ds::to_batch(std::slice::from_ref(&row))?;
        if let Err(error) = store.append(blob_ds::DATASET_NAME, batch).await {
            rollback_resource_revision(store, &staged_resource).await;
            return Err(error);
        }
    }

    if requires_finalize {
        staged_resource.storage_ready = true;
        if let Err(error) = upsert_resource_revision(store, &staged_resource).await {
            rollback_resource_revision(store, &staged_resource).await;
            return Err(error);
        }
    }

    Ok(())
}

async fn upsert_resource_revision(
    store: &dyn PhysicalStore,
    resource: &ResourceObject,
) -> Result<(), HirnDbError> {
    let batch = resource_object::to_batch(std::slice::from_ref(resource))?;
    store
        .merge_insert(resource_object::DATASET_NAME, &["id"], batch)
        .await
}

async fn rollback_resource_revision(store: &dyn PhysicalStore, resource: &ResourceObject) {
    let resource_filter = format!("id = '{}'", resource.id);
    let _ = store
        .delete(resource_object::DATASET_NAME, &resource_filter)
        .await;

    if matches!(resource.location, ResourceLocation::Blob { .. }) {
        let blob_filter = format!("resource_id = '{}'", resource.id);
        let _ = store.delete(blob_ds::DATASET_NAME, &blob_filter).await;
    }
}

async fn prepare_blob_payload(
    store: &dyn PhysicalStore,
    current: Option<&ResourceObject>,
    resource: &mut ResourceObject,
    blob: Option<Vec<u8>>,
) -> Result<Option<Vec<u8>>, HirnDbError> {
    match (&resource.location, blob) {
        (ResourceLocation::Blob { .. }, Some(blob_bytes)) => {
            sync_blob_metadata(resource, &blob_bytes)?;
            Ok(Some(blob_bytes))
        }
        (ResourceLocation::Blob { .. }, None) => {
            let Some(current) = current else {
                return Err(HirnDbError::InvalidArgument(
                    "blob-backed resource requires payload bytes".into(),
                ));
            };
            let ResourceLocation::Blob { blob_index } = current.location else {
                return Err(HirnDbError::InvalidArgument(
                    "cannot supersede a non-blob resource without new payload bytes".into(),
                ));
            };
            let blob_bytes = load_resource_blob_unchecked(store, current.id, blob_index).await?;
            sync_blob_metadata(resource, &blob_bytes)?;
            Ok(Some(blob_bytes))
        }
        (ResourceLocation::Inline | ResourceLocation::External { .. }, Some(_)) => {
            Err(HirnDbError::InvalidArgument(
                "only ResourceLocation::Blob may carry persisted payload bytes".into(),
            ))
        }
        (ResourceLocation::Inline | ResourceLocation::External { .. }, None) => Ok(None),
    }
}

fn sync_blob_metadata(resource: &mut ResourceObject, blob: &[u8]) -> Result<(), HirnDbError> {
    if resource.checksum.is_none() {
        resource.checksum = Some(format!("blake3:{}", blake3::hash(blob).to_hex()));
    }
    resource.size_bytes = blob.len() as u64;
    Ok(())
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResourceQuotaUsage {
    active_resources: usize,
    total_bytes: u64,
}

async fn enforce_resource_quota_policy(
    store: &dyn PhysicalStore,
    new_head: &ResourceObject,
    replaced_head: Option<&ResourceObject>,
    quota_policy: &ResourceQuotaPolicy,
) -> Result<(), HirnDbError> {
    if quota_policy.is_empty() {
        return Ok(());
    }

    let active_heads = list_active_resource_heads(store).await?;
    for rule in quota_policy.rules_for(new_head) {
        let usage = quota_usage_for_scope(&active_heads, rule.scope);
        let projected_active_resources =
            usage.active_resources + usize::from(replaced_head.is_none());
        if let Some(max_active_resources) = rule.max_active_resources
            && projected_active_resources > max_active_resources
        {
            return Err(HirnDbError::LimitExceeded(format!(
                "resource quota exceeded for {}: projected {} active resources exceeds limit {}",
                quota_scope_label(rule.scope),
                projected_active_resources,
                max_active_resources,
            )));
        }

        let replaced_bytes = replaced_head.map_or(0, |head| head.size_bytes);
        let projected_total_bytes = usage
            .total_bytes
            .saturating_sub(replaced_bytes)
            .saturating_add(new_head.size_bytes);
        if let Some(max_total_bytes) = rule.max_total_bytes
            && projected_total_bytes > max_total_bytes
        {
            return Err(HirnDbError::LimitExceeded(format!(
                "resource quota exceeded for {}: projected {} bytes exceeds limit {}",
                quota_scope_label(rule.scope),
                projected_total_bytes,
                max_total_bytes,
            )));
        }
    }

    Ok(())
}

fn quota_usage_for_scope(
    active_heads: &[ResourceObject],
    scope: ResourceQuotaScope,
) -> ResourceQuotaUsage {
    active_heads
        .iter()
        .filter(|resource| scope.matches(resource))
        .fold(ResourceQuotaUsage::default(), |mut usage, resource| {
            usage.active_resources += 1;
            usage.total_bytes = usage.total_bytes.saturating_add(resource.size_bytes);
            usage
        })
}

fn quota_scope_label(scope: ResourceQuotaScope) -> String {
    match scope {
        ResourceQuotaScope::Realm => "realm".to_string(),
        ResourceQuotaScope::Namespace(namespace) => format!("namespace `{}`", namespace.as_str()),
        ResourceQuotaScope::Agent(agent_id) => format!("agent `{}`", agent_id.as_str()),
    }
}

const fn default_text_artifact_plan(
    modality: ModalityProfile,
    role: EvidenceRole,
) -> &'static [DefaultTextArtifactPlan] {
    match role {
        EvidenceRole::Source => match modality {
            ModalityProfile::Image => &IMAGE_SOURCE_TEXT_ARTIFACTS,
            ModalityProfile::Audio => &AUDIO_SOURCE_TEXT_ARTIFACTS,
            ModalityProfile::Code => &CODE_SOURCE_TEXT_ARTIFACTS,
            ModalityProfile::Structured => &STRUCTURED_SOURCE_TEXT_ARTIFACTS,
            ModalityProfile::Text => &NO_TEXT_ARTIFACTS,
            _ => &PREVIEW_TEXT_ARTIFACTS,
        },
        EvidenceRole::Attachment
        | EvidenceRole::Proof
        | EvidenceRole::Output
        | EvidenceRole::Preview
        | EvidenceRole::Derived => &PREVIEW_TEXT_ARTIFACTS,
    }
}

const fn default_blob_artifact_plan(
    modality: ModalityProfile,
    role: EvidenceRole,
) -> &'static [DefaultBlobArtifactPlan] {
    match role {
        EvidenceRole::Source => match modality {
            ModalityProfile::Image => &IMAGE_SOURCE_BLOB_ARTIFACTS,
            _ => &NO_BLOB_ARTIFACTS,
        },
        EvidenceRole::Attachment
        | EvidenceRole::Proof
        | EvidenceRole::Output
        | EvidenceRole::Preview
        | EvidenceRole::Derived => &NO_BLOB_ARTIFACTS,
    }
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::RecordBatch;
    use async_trait::async_trait;
    use datafusion::catalog::TableProvider;
    use hirn_core::{
        DerivedArtifact, DerivedArtifactKind, ModalityProfile, ResourceGovernanceState,
        ResourceLocation, ResourceObject, ResourceQuotaPolicy, ResourceQuotaRule,
        ResourceQuotaScope, ResourceRetentionAction, ResourceRetentionPolicy,
        ResourceRetentionRule, RevisionState,
    };

    use crate::HirnDbError;
    use crate::datasets::{resource_blob as blob_ds, resource_object};
    use crate::memory_store::MemoryStore;
    use crate::mutation_envelope_ops::{MutationEnvelopeState, get_mutation_envelope};
    use crate::policy_store::{CURRENT_PRINCIPAL, NamespacePolicy, PolicyEnforcedStore};
    use crate::store::{
        ColumnTransform, CompactOptions, CompactResult, DatasetInfo, FtsSearchOptions,
        HybridSearchOptions, IndexConfig, MultivectorSearchOptions, PhysicalStore, ScanOptions,
        VectorSearchOptions, VersionTag,
    };

    struct FaultInjectingStore {
        inner: MemoryStore,
        fail_blob_append: bool,
        fail_resource_merge_insert: bool,
    }

    #[async_trait]
    impl PhysicalStore for FaultInjectingStore {
        async fn append(&self, dataset: &str, batch: RecordBatch) -> Result<(), HirnDbError> {
            if self.fail_blob_append && dataset == blob_ds::DATASET_NAME {
                return Err(HirnDbError::Unsupported(
                    "simulated blob append failure".to_string(),
                ));
            }
            self.inner.append(dataset, batch).await
        }

        async fn append_batches(
            &self,
            dataset: &str,
            batches: Vec<RecordBatch>,
        ) -> Result<(), HirnDbError> {
            for batch in batches {
                self.append(dataset, batch).await?;
            }
            Ok(())
        }

        async fn scan(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.scan(dataset, opts).await
        }

        async fn scan_stream(
            &self,
            dataset: &str,
            opts: ScanOptions,
        ) -> Result<crate::store::RecordBatchStream, HirnDbError> {
            self.inner.scan_stream(dataset, opts).await
        }

        async fn delete(&self, dataset: &str, predicate: &str) -> Result<u64, HirnDbError> {
            self.inner.delete(dataset, predicate).await
        }

        async fn merge_insert(
            &self,
            dataset: &str,
            on: &[&str],
            batch: RecordBatch,
        ) -> Result<(), HirnDbError> {
            if self.fail_resource_merge_insert && dataset == resource_object::DATASET_NAME {
                return Err(HirnDbError::Unsupported(
                    "simulated resource finalize failure".to_string(),
                ));
            }
            self.inner.merge_insert(dataset, on, batch).await
        }

        async fn update_where(
            &self,
            dataset: &str,
            filter: &str,
            updates: &[(&str, &str)],
        ) -> Result<u64, HirnDbError> {
            self.inner.update_where(dataset, filter, updates).await
        }

        async fn count(&self, dataset: &str, filter: Option<&str>) -> Result<u64, HirnDbError> {
            self.inner.count(dataset, filter).await
        }

        async fn vector_search(
            &self,
            dataset: &str,
            opts: VectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.vector_search(dataset, opts).await
        }

        async fn vector_search_many(
            &self,
            dataset: &str,
            queries: Vec<VectorSearchOptions>,
        ) -> Result<Vec<Vec<RecordBatch>>, HirnDbError> {
            self.inner.vector_search_many(dataset, queries).await
        }

        async fn fts_search(
            &self,
            dataset: &str,
            opts: FtsSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.fts_search(dataset, opts).await
        }

        async fn hybrid_search(
            &self,
            dataset: &str,
            opts: HybridSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.hybrid_search(dataset, opts).await
        }

        async fn multivector_search(
            &self,
            dataset: &str,
            opts: MultivectorSearchOptions,
        ) -> Result<Vec<RecordBatch>, HirnDbError> {
            self.inner.multivector_search(dataset, opts).await
        }

        async fn create_index(
            &self,
            dataset: &str,
            config: IndexConfig,
        ) -> Result<(), HirnDbError> {
            self.inner.create_index(dataset, config).await
        }

        async fn optimize_indices(&self, dataset: &str) -> Result<(), HirnDbError> {
            self.inner.optimize_indices(dataset).await
        }

        async fn compact(
            &self,
            dataset: &str,
            opts: CompactOptions,
        ) -> Result<CompactResult, HirnDbError> {
            self.inner.compact(dataset, opts).await
        }

        async fn version(&self, dataset: &str) -> Result<u64, HirnDbError> {
            self.inner.version(dataset).await
        }

        async fn tag(&self, dataset: &str, tag: &str) -> Result<(), HirnDbError> {
            self.inner.tag(dataset, tag).await
        }

        async fn checkout(&self, dataset: &str, version: u64) -> Result<(), HirnDbError> {
            self.inner.checkout(dataset, version).await
        }

        async fn list_tags(&self, dataset: &str) -> Result<Vec<VersionTag>, HirnDbError> {
            self.inner.list_tags(dataset).await
        }

        async fn list_datasets(&self) -> Result<Vec<DatasetInfo>, HirnDbError> {
            self.inner.list_datasets().await
        }

        async fn exists(&self, dataset: &str) -> Result<bool, HirnDbError> {
            self.inner.exists(dataset).await
        }

        async fn list_namespaces(&self) -> Result<Vec<String>, HirnDbError> {
            self.inner.list_namespaces().await
        }

        async fn create_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.create_namespace(name).await
        }

        async fn drop_namespace(&self, name: &str) -> Result<(), HirnDbError> {
            self.inner.drop_namespace(name).await
        }

        async fn add_columns(
            &self,
            dataset: &str,
            transforms: Vec<ColumnTransform>,
        ) -> Result<(), HirnDbError> {
            self.inner.add_columns(dataset, transforms).await
        }

        async fn drop_columns(&self, dataset: &str, columns: &[&str]) -> Result<(), HirnDbError> {
            self.inner.drop_columns(dataset, columns).await
        }

        async fn table_provider(&self, dataset: &str) -> Option<Arc<dyn TableProvider>> {
            self.inner.table_provider(dataset).await
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_resource_deduplicates_by_checksum_within_namespace() {
        let store = MemoryStore::new();
        let blob = vec![1_u8; 2048];

        let first = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .mime_type("image/png")
            .checksum("blake3:dedup")
            .size_bytes(blob.len() as u64)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let second = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .mime_type("image/png")
            .checksum("blake3:dedup")
            .size_bytes(blob.len() as u64)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();

        let persisted_first = persist_resource(&store, first, Some(blob.clone()))
            .await
            .unwrap();
        let persisted_second = persist_resource(&store, second, Some(blob)).await.unwrap();

        assert_eq!(persisted_first.id, persisted_second.id);

        let resources = store
            .scan(resource_object::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap();
        let blobs = store
            .scan(blob_ds::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap();

        assert_eq!(resources.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
        assert_eq!(blobs.iter().map(|b| b.num_rows()).sum::<usize>(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_resource_rolls_back_when_blob_append_fails() {
        let store = FaultInjectingStore {
            inner: MemoryStore::new(),
            fail_blob_append: true,
            fail_resource_merge_insert: false,
        };
        let blob = vec![4_u8; 128];
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource_id = resource.id;

        let error = persist_resource(&store, resource, Some(blob))
            .await
            .unwrap_err();

        assert!(matches!(error, HirnDbError::Unsupported(_)));
        assert!(get_resource(&store, resource_id).await.unwrap().is_none());
        assert!(
            get_resource_head(&store, resource_id)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            store
                .scan(resource_object::DATASET_NAME, ScanOptions::default())
                .await
                .unwrap()
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>(),
            0
        );
        assert_eq!(
            store
                .scan(blob_ds::DATASET_NAME, ScanOptions::default())
                .await
                .unwrap()
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>(),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_resource_rolls_back_when_visibility_finalize_fails() {
        let store = FaultInjectingStore {
            inner: MemoryStore::new(),
            fail_blob_append: false,
            fail_resource_merge_insert: true,
        };
        let blob = vec![6_u8; 128];
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource_id = resource.id;

        let error = persist_resource(&store, resource, Some(blob))
            .await
            .unwrap_err();

        assert!(matches!(error, HirnDbError::Unsupported(_)));
        assert!(get_resource(&store, resource_id).await.unwrap().is_none());
        assert!(
            get_resource_head(&store, resource_id)
                .await
                .unwrap()
                .is_none()
        );
        assert_eq!(
            store
                .scan(resource_object::DATASET_NAME, ScanOptions::default())
                .await
                .unwrap()
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>(),
            0
        );
        assert_eq!(
            store
                .scan(blob_ds::DATASET_NAME, ScanOptions::default())
                .await
                .unwrap()
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>(),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_resource_dedup_does_not_cross_namespaces() {
        let store = MemoryStore::new();
        let blob = vec![5_u8; 64];

        let alpha = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .checksum("blake3:isolation")
            .size_bytes(blob.len() as u64)
            .namespace(Namespace::new("alpha").unwrap())
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let beta = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .checksum("blake3:isolation")
            .size_bytes(blob.len() as u64)
            .namespace(Namespace::new("beta").unwrap())
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();

        let alpha = persist_resource(&store, alpha, Some(blob.clone()))
            .await
            .unwrap();
        let beta = persist_resource(&store, beta, Some(blob)).await.unwrap();

        assert_ne!(alpha.id, beta.id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn fetch_resource_respects_hydration_mode() {
        let store = MemoryStore::new();
        let blob = vec![9_u8; 512];
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Audio)
            .checksum("blake3:preview")
            .size_bytes(blob.len() as u64)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource = persist_resource(&store, resource, Some(blob.clone()))
            .await
            .unwrap();

        let mut artifact = DerivedArtifact::builder()
            .resource_id(resource.id)
            .kind(DerivedArtifactKind::Transcript)
            .modality(ModalityProfile::Text)
            .text_content("preview transcript")
            .build()
            .unwrap();
        artifact.created_at = hirn_core::Timestamp::from_millis(artifact.created_at.millis());
        persist_derived_artifact(&store, artifact.clone())
            .await
            .unwrap();

        let metadata_only = fetch_resource(&store, resource.id, HydrationMode::MetadataOnly)
            .await
            .unwrap()
            .unwrap();
        assert!(metadata_only.artifacts.is_empty());
        assert!(metadata_only.blob.is_none());

        let preview = fetch_resource(&store, resource.id, HydrationMode::Preview)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(preview.artifacts, vec![artifact.clone()]);
        assert!(preview.blob.is_none());

        let full = fetch_resource(&store, resource.id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(full.artifacts, vec![artifact]);
        assert_eq!(full.blob, Some(blob));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_default_derived_artifacts_adds_caption_ocr_and_thumbnail_for_images() {
        let store = MemoryStore::new();
        let source_image = image::DynamicImage::new_rgba8(4, 4);
        let mut encoded = Cursor::new(Vec::new());
        source_image
            .write_to(&mut encoded, ImageFormat::Png)
            .unwrap();
        let blob = encoded.into_inner();
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .mime_type("image/png")
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource = persist_resource_with_quota_policy(
            &store,
            resource,
            Some(blob.clone()),
            &ResourceQuotaPolicy::default(),
        )
        .await
        .unwrap();
        let created = persist_default_derived_artifacts(
            &store,
            &resource,
            EvidenceRole::Source,
            DerivedArtifactInput::new("diagram of the auth handshake")
                .with_blob(&blob, Some("image/png")),
        )
        .await
        .unwrap();
        let links = evidence_links_for_derived_artifacts(&created, Some(0));

        let artifacts = list_derived_artifacts(&store, resource.id).await.unwrap();
        assert_eq!(artifacts.len(), 3);
        assert_eq!(artifacts[0].kind, DerivedArtifactKind::Caption);
        assert_eq!(
            artifacts[0].text_content.as_deref(),
            Some("diagram of the auth handshake")
        );
        assert_eq!(artifacts[1].kind, DerivedArtifactKind::OcrText);
        assert_eq!(
            artifacts[1].text_content.as_deref(),
            Some("diagram of the auth handshake")
        );
        assert_eq!(artifacts[2].kind, DerivedArtifactKind::Thumbnail);
        assert_eq!(artifacts[2].mime_type.as_deref(), Some("image/png"));
        assert_eq!(artifacts[2].blob_index, Some(1));
        assert!(artifacts[2].text_content.is_none());
        assert!(matches!(
            artifacts[1].metadata.get("generation_strategy"),
            Some(hirn_core::metadata::MetadataValue::String(value)) if value == "text_surrogate_fallback"
        ));
        assert!(matches!(
            artifacts[1].metadata.get("fallback_source"),
            Some(hirn_core::metadata::MetadataValue::String(value)) if value == "image_description"
        ));
        assert_eq!(links.len(), 3);
        assert_eq!(links[0].role, EvidenceRole::Derived);
        assert_eq!(links[0].provenance.as_str(), "transformed_summary");
        assert_eq!(links[1].role, EvidenceRole::Derived);
        assert_eq!(links[1].provenance.as_str(), "generated_artifact");
        assert_eq!(links[2].role, EvidenceRole::Preview);
        assert_eq!(links[2].provenance.as_str(), "generated_artifact");

        let thumbnail_blob = load_resource_blob(&store, resource.id, 1).await.unwrap();
        let thumbnail =
            image::load_from_memory_with_format(&thumbnail_blob, ImageFormat::Png).unwrap();
        assert!(thumbnail.width() <= THUMBNAIL_MAX_DIMENSION_PX);
        assert!(thumbnail.height() <= THUMBNAIL_MAX_DIMENSION_PX);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_default_derived_artifacts_records_generation_failure_for_empty_inputs() {
        let store = MemoryStore::new();
        let blob = vec![7_u8; 48];
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .mime_type("image/png")
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource = persist_resource_with_quota_policy(
            &store,
            resource,
            Some(blob.clone()),
            &ResourceQuotaPolicy::default(),
        )
        .await
        .unwrap();
        let created = persist_default_derived_artifacts(
            &store,
            &resource,
            EvidenceRole::Source,
            DerivedArtifactInput::new(""),
        )
        .await
        .unwrap();

        let artifacts = list_derived_artifacts(&store, resource.id).await.unwrap();
        assert_eq!(artifacts.len(), 2);
        assert!(
            artifacts
                .iter()
                .all(|artifact| artifact.kind == DerivedArtifactKind::GenerationFailure)
        );
        assert!(artifacts.iter().any(|artifact| {
            artifact.text_content.as_deref()
                == Some("caption generation failed: source text was empty")
        }));
        assert!(artifacts.iter().any(|artifact| {
            artifact.text_content.as_deref()
                == Some("thumbnail generation failed: source blob was unavailable")
        }));
        assert!(evidence_links_for_derived_artifacts(&created, Some(0)).is_empty());

        let hydrated = fetch_resource(&store, resource.id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(hydrated.blob, Some(blob));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn supersede_resource_promotes_new_head_and_preserves_history() {
        let store = MemoryStore::new();
        let original_blob = vec![7_u8; 32];
        let original = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .mime_type("image/png")
            .display_name("frame-v1.png")
            .checksum(format!("blake3:{}", blake3::hash(&original_blob).to_hex()))
            .size_bytes(original_blob.len() as u64)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let original = persist_resource(&store, original, Some(original_blob.clone()))
            .await
            .unwrap();

        let successor_blob = vec![8_u8; 48];
        let successor = supersede_resource(
            &store,
            original.id,
            ResourceSupersession {
                reason: Some("cropped and re-encoded".into()),
                display_name: Some("frame-v2.png".into()),
                checksum: Some(format!("blake3:{}", blake3::hash(&successor_blob).to_hex())),
                ..ResourceSupersession::default()
            },
            Some(successor_blob.clone()),
        )
        .await
        .unwrap();

        let active_head = get_resource_head(&store, original.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(active_head.id, successor.id);
        assert_eq!(
            active_head.logical_resource_id,
            original.logical_resource_id
        );
        assert_eq!(active_head.version, 2);
        assert_eq!(active_head.revision_operation, RevisionOperation::Supersede);
        assert_eq!(
            active_head.revision_reason.as_deref(),
            Some("cropped and re-encoded")
        );
        assert_eq!(active_head.revision_causation_id, Some(original.id));
        assert_eq!(active_head.display_name.as_deref(), Some("frame-v2.png"));

        let historical = get_resource(&store, original.id).await.unwrap().unwrap();
        assert_eq!(historical.superseded_by, Some(successor.id));
        assert_eq!(
            historical.revision_state_against(&active_head),
            RevisionState::Superseded
        );

        let revisions = list_resource_revisions(&store, original.id).await.unwrap();
        assert_eq!(revisions.len(), 2);
        assert_eq!(revisions[0].id, original.id);
        assert_eq!(revisions[1].id, successor.id);

        let historical_fetch = fetch_resource(&store, original.id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(historical_fetch.blob, Some(original_blob));

        let head_fetch = fetch_resource(&store, successor.id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(head_fetch.blob, Some(successor_blob));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn reconcile_resource_head_mutations_repairs_missing_backlink() {
        let store = MemoryStore::new();
        let original_blob = vec![1_u8; 16];
        let original = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .display_name("draft-v1.txt")
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let original = persist_resource(&store, original, Some(original_blob))
            .await
            .unwrap();

        let successor_blob = vec![2_u8; 24];
        let now = Timestamp::now();
        let mut successor =
            build_successor_revision(&original, Some("published revision".to_string()), now);
        successor.display_name = Some("draft-v2.txt".into());
        successor.location = ResourceLocation::Blob { blob_index: 0 };
        successor.size_bytes = successor_blob.len() as u64;
        successor.checksum = Some(format!("blake3:{}", blake3::hash(&successor_blob).to_hex()));

        let envelope = build_resource_head_transition_envelope(&original, &successor).unwrap();
        crate::mutation_envelope_ops::append_mutation_envelope(&store, &envelope)
            .await
            .unwrap();
        append_resource_revision(&store, &successor, Some(successor_blob))
            .await
            .unwrap();

        let reconciled = reconcile_resource_head_mutations(&store).await.unwrap();
        assert_eq!(reconciled, 1);

        let original_after = get_resource(&store, original.id).await.unwrap().unwrap();
        assert_eq!(original_after.superseded_by, Some(successor.id));

        let envelope_after = get_mutation_envelope(&store, &envelope.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(envelope_after.state, MutationEnvelopeState::Applied);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn supersede_resource_can_preserve_existing_blob_without_new_payload() {
        let store = MemoryStore::new();
        let blob = vec![3_u8; 24];
        let original = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .display_name("brief-v1.pdf")
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let original = persist_resource(&store, original, Some(blob.clone()))
            .await
            .unwrap();

        let successor = supersede_resource(
            &store,
            original.id,
            ResourceSupersession {
                reason: Some("metadata refresh".into()),
                display_name: Some("brief-v2.pdf".into()),
                ..ResourceSupersession::default()
            },
            None,
        )
        .await
        .unwrap();

        let hydrated = fetch_resource(&store, successor.id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(hydrated.blob, Some(blob));
        assert_eq!(
            hydrated.resource.display_name.as_deref(),
            Some("brief-v2.pdf")
        );
        assert_eq!(hydrated.resource.revision_causation_id, Some(original.id));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_resource_does_not_deduplicate_to_superseded_checksum_match() {
        let store = MemoryStore::new();
        let original_blob = vec![1_u8; 16];
        let original = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .checksum(format!("blake3:{}", blake3::hash(&original_blob).to_hex()))
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let original = persist_resource(&store, original, Some(original_blob.clone()))
            .await
            .unwrap();

        supersede_resource(
            &store,
            original.id,
            ResourceSupersession {
                checksum: Some("blake3:replacement".into()),
                ..ResourceSupersession::default()
            },
            Some(vec![2_u8; 20]),
        )
        .await
        .unwrap();

        let replacement_candidate = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .checksum(format!("blake3:{}", blake3::hash(&original_blob).to_hex()))
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let replacement = persist_resource(&store, replacement_candidate, Some(original_blob))
            .await
            .unwrap();

        assert_ne!(replacement.id, original.id);
        assert_ne!(
            replacement.logical_resource_id,
            original.logical_resource_id
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn load_resource_blob_requires_visible_resource_metadata() {
        struct TestPolicy;

        #[async_trait::async_trait]
        impl NamespacePolicy for TestPolicy {
            async fn allowed_namespaces(&self, principal: &str) -> Option<Vec<String>> {
                match principal {
                    "allowed" => Some(vec!["default".to_string()]),
                    "blocked" => Some(vec!["blocked".to_string()]),
                    _ => Some(Vec::new()),
                }
            }
        }

        let store = PolicyEnforcedStore::new(MemoryStore::new(), Arc::new(TestPolicy));
        let blob = vec![4_u8; 64];
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource = CURRENT_PRINCIPAL
            .scope("allowed".to_string(), async {
                persist_resource(&store, resource, Some(blob.clone())).await
            })
            .await
            .unwrap();

        let visible = CURRENT_PRINCIPAL
            .scope("allowed".to_string(), async {
                load_resource_blob(&store, resource.id, 0).await
            })
            .await
            .unwrap();
        assert_eq!(visible, blob);

        let denied = CURRENT_PRINCIPAL
            .scope("blocked".to_string(), async {
                load_resource_blob(&store, resource.id, 0).await
            })
            .await
            .unwrap_err();
        assert!(matches!(denied, HirnDbError::BlobError { .. }));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn redact_resource_blocks_payload_hydration_and_keeps_placeholder_head() {
        let store = MemoryStore::new();
        let blob = vec![6_u8; 96];
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .mime_type("application/pdf")
            .display_name("roadmap.pdf")
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource = persist_resource(&store, resource, Some(blob.clone()))
            .await
            .unwrap();

        let artifact = DerivedArtifact::builder()
            .resource_id(resource.id)
            .kind(DerivedArtifactKind::Preview)
            .modality(ModalityProfile::Text)
            .text_content("preview text")
            .build()
            .unwrap();
        persist_derived_artifact(&store, artifact).await.unwrap();

        let redacted = redact_resource(
            &store,
            resource.id,
            ResourceGovernanceUpdate {
                reason: Some("contains sensitive evidence".into()),
                placeholder_display_name: Some("redacted evidence".into()),
            },
        )
        .await
        .unwrap();

        assert_eq!(redacted.governance_state, ResourceGovernanceState::Redacted);
        assert_eq!(redacted.display_name.as_deref(), Some("redacted evidence"));

        let historical = get_resource(&store, resource.id).await.unwrap().unwrap();
        assert_eq!(
            historical.governance_state,
            ResourceGovernanceState::Redacted
        );
        assert_eq!(
            historical.display_name.as_deref(),
            Some("redacted evidence")
        );
        assert!(historical.mime_type.is_none());
        assert_eq!(historical.size_bytes, 0);

        let preview = fetch_resource(&store, resource.id, HydrationMode::Preview)
            .await
            .unwrap()
            .unwrap();
        assert!(preview.artifacts.is_empty());
        assert!(preview.blob.is_none());

        let full = fetch_resource(&store, redacted.id, HydrationMode::Full)
            .await
            .unwrap()
            .unwrap();
        assert!(full.artifacts.is_empty());
        assert!(full.blob.is_none());

        let blob_err = load_resource_blob(&store, resource.id, 0)
            .await
            .unwrap_err();
        assert!(matches!(blob_err, HirnDbError::BlobError { .. }));

        let remaining_blobs = store
            .scan(blob_ds::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap();
        let remaining_artifacts = store
            .scan(artifact_ds::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap();
        assert_eq!(
            remaining_blobs
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>(),
            0
        );
        assert_eq!(
            remaining_artifacts
                .iter()
                .map(|batch| batch.num_rows())
                .sum::<usize>(),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn purge_resource_marks_lineage_as_purged() {
        let store = MemoryStore::new();
        let blob = vec![2_u8; 40];
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Image)
            .display_name("frame.png")
            .location(ResourceLocation::Blob { blob_index: 0 })
            .build()
            .unwrap();
        let resource = persist_resource(&store, resource, Some(blob))
            .await
            .unwrap();

        let purged = purge_resource(&store, resource.id, ResourceGovernanceUpdate::default())
            .await
            .unwrap();
        assert_eq!(purged.governance_state, ResourceGovernanceState::Purged);
        assert_eq!(purged.display_name.as_deref(), Some("purged resource"));

        let head = get_resource_head(&store, resource.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(head.id, purged.id);
        assert_eq!(head.governance_state, ResourceGovernanceState::Purged);

        let historical = get_resource(&store, resource.id).await.unwrap().unwrap();
        assert_eq!(historical.governance_state, ResourceGovernanceState::Purged);
        assert!(historical.mime_type.is_none());
        assert_eq!(historical.size_bytes, 0);

        let revisions = list_resource_revisions(&store, resource.id).await.unwrap();
        assert_eq!(revisions.len(), 2);
        assert_eq!(revisions[0].id, resource.id);
        assert_eq!(revisions[1].id, purged.id);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn retention_policy_targets_modality_and_classification() {
        let store = MemoryStore::new();

        let image_restricted = persist_resource(
            &store,
            ResourceObject::builder()
                .modality(ModalityProfile::Image)
                .metadata_entry("classification", "restricted")
                .location(ResourceLocation::Blob { blob_index: 0 })
                .build()
                .unwrap(),
            Some(vec![1_u8; 24]),
        )
        .await
        .unwrap();
        let image_public = persist_resource(
            &store,
            ResourceObject::builder()
                .modality(ModalityProfile::Image)
                .metadata_entry("classification", "public")
                .location(ResourceLocation::Blob { blob_index: 0 })
                .build()
                .unwrap(),
            Some(vec![2_u8; 24]),
        )
        .await
        .unwrap();
        let document_restricted = persist_resource(
            &store,
            ResourceObject::builder()
                .modality(ModalityProfile::Document)
                .metadata_entry("classification", "restricted")
                .location(ResourceLocation::Blob { blob_index: 0 })
                .build()
                .unwrap(),
            Some(vec![3_u8; 24]),
        )
        .await
        .unwrap();
        let document_public = persist_resource(
            &store,
            ResourceObject::builder()
                .modality(ModalityProfile::Document)
                .metadata_entry("classification", "public")
                .location(ResourceLocation::Blob { blob_index: 0 })
                .build()
                .unwrap(),
            Some(vec![4_u8; 24]),
        )
        .await
        .unwrap();

        let policy = ResourceRetentionPolicy::default()
            .with_rule(
                ResourceRetentionRule::new(ResourceRetentionAction::Redact)
                    .classification("restricted"),
            )
            .with_rule(
                ResourceRetentionRule::new(ResourceRetentionAction::Purge)
                    .modality(ModalityProfile::Image),
            );

        let result = apply_resource_retention_policy(&store, &policy)
            .await
            .unwrap();
        assert_eq!(result.scanned_active_heads, 4);
        assert_eq!(result.governed_resources, 3);
        assert_eq!(result.redacted_resources, 1);
        assert_eq!(result.purged_resources, 2);
        assert_eq!(result.skipped_resources, 0);

        let image_restricted = get_resource_head(&store, image_restricted.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            image_restricted.governance_state,
            ResourceGovernanceState::Purged
        );

        let image_public = get_resource_head(&store, image_public.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            image_public.governance_state,
            ResourceGovernanceState::Purged
        );

        let document_restricted = get_resource_head(&store, document_restricted.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            document_restricted.governance_state,
            ResourceGovernanceState::Redacted
        );

        let document_public = get_resource_head(&store, document_public.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            document_public.governance_state,
            ResourceGovernanceState::Active
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn persist_resource_with_quota_policy_blocks_namespace_limit() {
        let store = MemoryStore::new();
        let namespace = Namespace::new("quota-ns").unwrap();
        let policy = ResourceQuotaPolicy::default().with_rule(
            ResourceQuotaRule::new(ResourceQuotaScope::Namespace(namespace))
                .max_active_resources(1),
        );

        let first = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .namespace(namespace)
            .build()
            .unwrap();
        persist_resource_with_quota_policy(&store, first, Some(vec![1_u8; 16]), &policy)
            .await
            .unwrap();

        let second = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .namespace(namespace)
            .build()
            .unwrap();
        let error =
            persist_resource_with_quota_policy(&store, second, Some(vec![2_u8; 16]), &policy)
                .await
                .unwrap_err();

        assert!(
            matches!(error, HirnDbError::LimitExceeded(message) if message.contains("namespace `quota-ns`") && message.contains("active resources"))
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn supersede_resource_with_quota_policy_reuses_the_active_head_slot() {
        let store = MemoryStore::new();
        let namespace = Namespace::new("quota-replace").unwrap();
        let policy = ResourceQuotaPolicy::default().with_rule(
            ResourceQuotaRule::new(ResourceQuotaScope::Namespace(namespace))
                .max_active_resources(1),
        );

        let original = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .namespace(namespace)
            .build()
            .unwrap();
        let original =
            persist_resource_with_quota_policy(&store, original, Some(vec![1_u8; 16]), &policy)
                .await
                .unwrap();

        let successor = supersede_resource_with_quota_policy(
            &store,
            original.id,
            ResourceSupersession {
                display_name: Some("replacement.pdf".into()),
                ..Default::default()
            },
            Some(vec![2_u8; 24]),
            &policy,
        )
        .await
        .unwrap();

        let head = get_resource_head(&store, original.id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(head.id, successor.id);
        assert_eq!(head.display_name.as_deref(), Some("replacement.pdf"));
        let revisions = list_resource_revisions(&store, original.id).await.unwrap();
        assert_eq!(revisions.len(), 2);
    }
}
