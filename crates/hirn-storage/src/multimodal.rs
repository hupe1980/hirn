//! Multimodal ingest pipeline.
//!
//! Accepts raw content (text, image, audio, video, PDF) alongside binary blobs,
//! embeds via the supplied [`Embedder`], and stores binary payloads as
//! first-class resources linked from episodic provenance.

use std::borrow::Cow;
use std::sync::Arc;

use hirn_core::content::{CompositeEmbeddingPolicy, MemoryContent};
use hirn_core::embed::Embedder;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::resource::{
    EvidenceLink, EvidenceRole, ModalityProfile, ResourceLocation, ResourceObject,
    ResourceQuotaPolicy,
};
use hirn_core::types::{AgentId, Namespace};

use crate::HirnDbError;
use crate::datasets::episodic as ep_ds;
use crate::resource_ops::{
    DerivedArtifactInput, build_configured_blob_resource, configure_audio_resource_builder,
    evidence_links_for_derived_artifacts, persist_default_derived_artifacts,
    persist_resource_with_quota_policy, text_backed_resource_checksum,
};
use crate::store::PhysicalStore;

/// Configuration for multimodal ingest: which embedder to use per MIME type.
#[derive(Clone)]
pub struct MultimodalIngestConfig {
    /// Embedder for text-like content (also used as fallback).
    pub text_embedder: Arc<dyn Embedder>,
    /// Optional embedder for image MIME types (e.g. CLIP/SigLIP).
    /// Falls back to `text_embedder` if `None`.
    pub image_embedder: Option<Arc<dyn Embedder>>,
    /// Optional embedder for audio MIME types.
    /// Falls back to `text_embedder` if `None`.
    pub audio_embedder: Option<Arc<dyn Embedder>>,
    /// Optional embedder for video transcript/scene-description surrogates.
    /// Falls back to `text_embedder` if `None`.
    pub video_embedder: Option<Arc<dyn Embedder>>,
    /// Optional embedder for code content.
    /// Falls back to `text_embedder` if `None`.
    pub code_embedder: Option<Arc<dyn Embedder>>,
    /// Optional embedder for document content.
    /// Falls back to `text_embedder` if `None`.
    pub document_embedder: Option<Arc<dyn Embedder>>,
    /// Policy used when collapsing composite content into one aggregate bundle.
    pub composite_policy: CompositeEmbeddingPolicy,
    /// Embedding dimensions (must match the embedder output).
    pub embedding_dims: usize,
    /// Resource quota policy applied to extracted first-class resources.
    pub resource_quota_policy: ResourceQuotaPolicy,
}

/// A single multimodal input to ingest.
pub struct MultimodalInput {
    /// Human-readable text content (caption, transcript, extracted text).
    pub content: String,
    /// Optional structured multimodal content.
    pub multi_content: Option<MemoryContent>,
    /// Optional raw binary blob (image bytes, audio bytes, PDF bytes).
    /// This is persisted as a resource link rather than inline episodic bytes.
    pub blob: Option<Vec<u8>>,
    /// MIME type of the blob (e.g. `image/png`, `audio/wav`, `application/pdf`).
    pub blob_mime: Option<String>,
    /// Agent performing the ingest.
    pub agent_id: AgentId,
    /// Namespace owning both the episodic row and any persisted resources.
    pub namespace: Namespace,
}

/// Select the appropriate embedder based on MIME type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EmbeddingRoute {
    Text,
    Image,
    Audio,
    Video,
    Code,
    Document,
    Composite,
}

fn select_embedder(config: &MultimodalIngestConfig, route: EmbeddingRoute) -> Arc<dyn Embedder> {
    match route {
        EmbeddingRoute::Image => config
            .image_embedder
            .clone()
            .unwrap_or_else(|| config.text_embedder.clone()),
        EmbeddingRoute::Audio => config
            .audio_embedder
            .clone()
            .unwrap_or_else(|| config.text_embedder.clone()),
        EmbeddingRoute::Video => config
            .video_embedder
            .clone()
            .unwrap_or_else(|| config.text_embedder.clone()),
        EmbeddingRoute::Code => config
            .code_embedder
            .clone()
            .unwrap_or_else(|| config.text_embedder.clone()),
        EmbeddingRoute::Document => config
            .document_embedder
            .clone()
            .unwrap_or_else(|| config.text_embedder.clone()),
        EmbeddingRoute::Text | EmbeddingRoute::Composite => config.text_embedder.clone(),
    }
}

fn route_for_input(input: &MultimodalInput) -> EmbeddingRoute {
    match input.multi_content.as_ref() {
        Some(MemoryContent::Image { .. }) => EmbeddingRoute::Image,
        Some(MemoryContent::Audio { .. }) => EmbeddingRoute::Audio,
        Some(MemoryContent::Video { .. }) => EmbeddingRoute::Video,
        Some(MemoryContent::Code { .. }) => EmbeddingRoute::Code,
        Some(MemoryContent::Document { .. }) => EmbeddingRoute::Document,
        Some(MemoryContent::Composite(_)) => EmbeddingRoute::Composite,
        Some(MemoryContent::Text(_))
        | Some(MemoryContent::ToolOutput { .. })
        | Some(MemoryContent::External { .. })
        | Some(MemoryContent::Structured { .. }) => EmbeddingRoute::Text,
        None => match input.blob_mime.as_deref() {
            Some(mime) if mime.starts_with("image/") => EmbeddingRoute::Image,
            Some(mime) if mime.starts_with("audio/") => EmbeddingRoute::Audio,
            Some(mime) if mime.starts_with("video/") => EmbeddingRoute::Video,
            Some(mime) if is_primary_document_mime(mime) => EmbeddingRoute::Document,
            _ => EmbeddingRoute::Text,
        },
    }
}

fn embedding_text_for_input(input: &MultimodalInput) -> Cow<'_, str> {
    input.multi_content.as_ref().map_or_else(
        || Cow::Borrowed(input.content.as_str()),
        MemoryContent::text_for_embedding,
    )
}

fn normalize_input(mut input: MultimodalInput) -> MultimodalInput {
    if input.multi_content.is_none()
        && let (Some(blob_data), Some(blob_mime_value)) =
            (input.blob.clone(), input.blob_mime.clone())
    {
        if blob_mime_value.starts_with("video/") {
            input.multi_content = Some(MemoryContent::Video {
                data: blob_data,
                mime_type: blob_mime_value,
                transcript: String::new(),
                description: input.content.clone(),
            });
        } else if is_primary_document_mime(blob_mime_value.as_str()) {
            input.multi_content = Some(MemoryContent::Document {
                data: blob_data,
                mime_type: blob_mime_value,
                extracted_text: input.content.clone(),
            });
        }
    }
    input
}

async fn embed_route_batch(
    embedder: Arc<dyn Embedder>,
    batch: Vec<(usize, String)>,
    results: &mut [Option<Vec<f32>>],
) -> Result<(), HirnDbError> {
    if batch.is_empty() {
        return Ok(());
    }

    let refs: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();
    let embeddings = embedder
        .embed(&refs)
        .await
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
    if embeddings.len() != batch.len() {
        return Err(HirnDbError::InvalidArgument(format!(
            "embedder returned {} vectors for {} inputs",
            embeddings.len(),
            batch.len()
        )));
    }

    for ((idx, _), embedding) in batch.into_iter().zip(embeddings) {
        results[idx] = Some(embedding.vector);
    }

    Ok(())
}

async fn embed_text_with_route(
    config: &MultimodalIngestConfig,
    route: EmbeddingRoute,
    text: &str,
) -> Result<Vec<f32>, HirnDbError> {
    let embeddings = select_embedder(config, route)
        .embed(&[text])
        .await
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
    embeddings
        .into_iter()
        .next()
        .map(|embedding| embedding.vector)
        .ok_or_else(|| HirnDbError::InvalidArgument("embedder returned empty result".into()))
}

async fn embed_composite_content(
    config: &MultimodalIngestConfig,
    parts: &[MemoryContent],
) -> Result<Vec<f32>, HirnDbError> {
    if parts.is_empty() {
        return embed_text_with_route(config, EmbeddingRoute::Text, "").await;
    }

    let mut embeddings = Vec::with_capacity(parts.len());
    for part in parts {
        let embedding = match part {
            MemoryContent::Text(text) => {
                embed_text_with_route(config, EmbeddingRoute::Text, text.as_str()).await?
            }
            MemoryContent::Image { description, .. } => {
                embed_text_with_route(config, EmbeddingRoute::Image, description.as_str()).await?
            }
            MemoryContent::Audio { transcript, .. } => {
                embed_text_with_route(config, EmbeddingRoute::Audio, transcript.as_str()).await?
            }
            MemoryContent::Video { .. } => {
                let text = part.text_for_embedding();
                embed_text_with_route(config, EmbeddingRoute::Video, text.as_ref()).await?
            }
            MemoryContent::Code { source, .. } => {
                embed_text_with_route(config, EmbeddingRoute::Code, source.as_str()).await?
            }
            MemoryContent::Document { extracted_text, .. } => {
                embed_text_with_route(config, EmbeddingRoute::Document, extracted_text.as_str())
                    .await?
            }
            MemoryContent::External { .. } => {
                let text = part.text_for_embedding();
                embed_text_with_route(config, EmbeddingRoute::Text, text.as_ref()).await?
            }
            MemoryContent::ToolOutput { .. } => {
                let text = part.text_for_embedding();
                embed_text_with_route(config, EmbeddingRoute::Text, text.as_ref()).await?
            }
            MemoryContent::Structured { data, .. } => {
                let json = data.to_string();
                embed_text_with_route(config, EmbeddingRoute::Text, json.as_str()).await?
            }
            MemoryContent::Composite(_) => {
                let text = part.text_for_embedding();
                embed_text_with_route(config, EmbeddingRoute::Text, text.as_ref()).await?
            }
        };
        embeddings.push(embedding);
    }

    let dims = embeddings[0].len();
    let mut avg = vec![0.0f32; dims];
    let mut total_weight = 0.0f32;
    for (part, embedding) in parts.iter().zip(&embeddings) {
        if embedding.len() != dims {
            return Err(HirnDbError::InvalidArgument(format!(
                "composite part embedding dimension mismatch: expected {dims}, got {}",
                embedding.len()
            )));
        }

        let weight = config.composite_policy.weight_for(part);
        if weight <= 0.0 {
            continue;
        }

        for (idx, value) in embedding.iter().enumerate() {
            avg[idx] += value * weight;
        }
        total_weight += weight;
    }

    if total_weight <= 0.0 {
        return embed_text_with_route(config, EmbeddingRoute::Text, "").await;
    }

    for value in &mut avg {
        *value /= total_weight;
    }

    let norm: f32 = avg.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut avg {
            *value /= norm;
        }
    }

    Ok(avg)
}

async fn embed_content_with_config(
    config: &MultimodalIngestConfig,
    content: &MemoryContent,
) -> Result<Vec<f32>, HirnDbError> {
    match content {
        MemoryContent::Text(text) => {
            embed_text_with_route(config, EmbeddingRoute::Text, text.as_str()).await
        }
        MemoryContent::Image { description, .. } => {
            embed_text_with_route(config, EmbeddingRoute::Image, description.as_str()).await
        }
        MemoryContent::Audio { transcript, .. } => {
            embed_text_with_route(config, EmbeddingRoute::Audio, transcript.as_str()).await
        }
        MemoryContent::Video { .. } => {
            let text = content.text_for_embedding();
            embed_text_with_route(config, EmbeddingRoute::Video, text.as_ref()).await
        }
        MemoryContent::Code { source, .. } => {
            embed_text_with_route(config, EmbeddingRoute::Code, source.as_str()).await
        }
        MemoryContent::Document { extracted_text, .. } => {
            embed_text_with_route(config, EmbeddingRoute::Document, extracted_text.as_str()).await
        }
        MemoryContent::External { .. } => {
            let text = content.text_for_embedding();
            embed_text_with_route(config, EmbeddingRoute::Text, text.as_ref()).await
        }
        MemoryContent::ToolOutput { .. } => {
            let text = content.text_for_embedding();
            embed_text_with_route(config, EmbeddingRoute::Text, text.as_ref()).await
        }
        MemoryContent::Structured { data, .. } => {
            let json = data.to_string();
            embed_text_with_route(config, EmbeddingRoute::Text, json.as_str()).await
        }
        MemoryContent::Composite(parts) => embed_composite_content(config, parts).await,
    }
}

struct ResourceizedContent {
    content: MemoryContent,
    evidence_links: Vec<EvidenceLink>,
}

fn modality_for_mime(mime: Option<&str>) -> ModalityProfile {
    match mime {
        Some(mime) if mime.starts_with("image/") => ModalityProfile::Image,
        Some(mime) if mime.starts_with("audio/") => ModalityProfile::Audio,
        Some(mime) if mime.starts_with("video/") => ModalityProfile::Video,
        Some("application/json") => ModalityProfile::Structured,
        Some("application/pdf") => ModalityProfile::Document,
        Some(mime) if mime.starts_with("text/") => ModalityProfile::Text,
        Some(_) | None => ModalityProfile::Document,
    }
}

fn root_binary_payload(content: &MemoryContent) -> Option<&[u8]> {
    match content {
        MemoryContent::Image { data, .. } => Some(data),
        MemoryContent::Audio { data, .. } => Some(data),
        MemoryContent::Video { data, .. } => Some(data),
        MemoryContent::Document { data, .. } => Some(data),
        _ => None,
    }
}

fn video_surrogate_text(transcript: &str, description: &str) -> String {
    match (transcript.trim().is_empty(), description.trim().is_empty()) {
        (false, true) => transcript.to_string(),
        (true, false) => description.to_string(),
        (true, true) => String::new(),
        (false, false) => format!("{transcript}\n{description}"),
    }
}

fn external_surrogate_text(title: &str, snippet: &str, uri: &str) -> String {
    let title = title.trim();
    let snippet = snippet.trim();
    let uri = uri.trim();

    match (title.is_empty(), snippet.is_empty(), uri.is_empty()) {
        (false, true, true) => title.to_string(),
        (true, false, true) => snippet.to_string(),
        (true, true, false) => uri.to_string(),
        (false, false, true) => format!("{title}\n{snippet}"),
        (false, true, false) => format!("{title}\n{uri}"),
        (true, false, false) => format!("{snippet}\n{uri}"),
        (false, false, false) => format!("{title}\n{snippet}\n{uri}"),
        (true, true, true) => String::new(),
    }
}

fn is_primary_document_mime(mime: &str) -> bool {
    matches!(
        mime,
        "application/pdf"
            | "application/msword"
            | "application/rtf"
            | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            | "application/vnd.oasis.opendocument.text"
    )
}

#[derive(Clone, Copy)]
struct ResourcePersistenceContext<'a> {
    store: &'a dyn PhysicalStore,
    namespace: Namespace,
    owner_agent_id: AgentId,
    quota_policy: &'a ResourceQuotaPolicy,
}

struct CodeResourceInput {
    source: String,
    language: String,
    ast_hash: Option<String>,
    part_index: u32,
}

struct ExternalResourceInput {
    uri: String,
    title: String,
    snippet: String,
    mime_type: Option<String>,
    checksum: Option<String>,
    fetch_policy: hirn_core::content::ExternalFetchPolicy,
    stale_at: Option<hirn_core::Timestamp>,
    part_index: u32,
}

struct ToolOutputResourceInput {
    tool_name: String,
    output: String,
    mime_type: Option<String>,
    schema: Option<String>,
    invocation_id: Option<String>,
    checksum: Option<String>,
    part_index: u32,
}

struct AttachmentResourceInput<'a, F>
where
    F: FnOnce(
        hirn_core::resource::ResourceObjectBuilder,
    ) -> hirn_core::resource::ResourceObjectBuilder,
{
    modality: ModalityProfile,
    role: EvidenceRole,
    mime_type: Option<&'a str>,
    data: &'a [u8],
    description: &'a str,
    part_index: Option<u32>,
    configure: F,
}

fn structured_artifact_text(schema: &str, data: &serde_json::Value) -> String {
    if schema.trim().is_empty() {
        data.to_string()
    } else {
        format!("[{schema}] {data}")
    }
}

async fn persist_code_resource(
    context: ResourcePersistenceContext<'_>,
    input: CodeResourceInput,
) -> Result<(MemoryContent, Vec<EvidenceLink>), HirnDbError> {
    let CodeResourceInput {
        source,
        language,
        ast_hash,
        part_index,
    } = input;
    let blob = source.as_bytes().to_vec();
    let mut builder = ResourceObject::builder()
        .modality(ModalityProfile::Code)
        .mime_type("text/plain")
        .checksum(text_backed_resource_checksum(
            &format!("code:{language}"),
            &blob,
        ))
        .size_bytes(blob.len() as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .owner_agent_id(context.owner_agent_id)
        .namespace(context.namespace)
        .metadata_entry("language", language.clone());
    if let Some(ast_hash) = ast_hash.as_ref() {
        builder = builder.metadata_entry("ast_hash", ast_hash.clone());
    }

    let resource = builder
        .build()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
    let resource = persist_resource_with_quota_policy(
        context.store,
        resource,
        Some(blob),
        context.quota_policy,
    )
    .await?;
    let artifacts = persist_default_derived_artifacts(
        context.store,
        &resource,
        EvidenceRole::Source,
        DerivedArtifactInput::new(&source),
    )
    .await?;
    let mut evidence_links =
        vec![EvidenceLink::new(resource.id, EvidenceRole::Source).with_part_index(part_index)];
    evidence_links.extend(evidence_links_for_derived_artifacts(
        &artifacts,
        Some(part_index),
    ));

    Ok((
        MemoryContent::Code {
            source: String::new(),
            language,
            ast_hash,
        },
        evidence_links,
    ))
}

async fn persist_structured_resource(
    store: &dyn PhysicalStore,
    namespace: Namespace,
    owner_agent_id: AgentId,
    schema: String,
    data: serde_json::Value,
    quota_policy: &ResourceQuotaPolicy,
    part_index: u32,
) -> Result<(MemoryContent, Vec<EvidenceLink>), HirnDbError> {
    let blob = serde_json::to_vec(&data)
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
    let artifact_text = structured_artifact_text(&schema, &data);
    let resource = ResourceObject::builder()
        .modality(ModalityProfile::Structured)
        .mime_type("application/json")
        .checksum(text_backed_resource_checksum(
            &format!("structured:{schema}"),
            &blob,
        ))
        .size_bytes(blob.len() as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .owner_agent_id(owner_agent_id)
        .namespace(namespace)
        .metadata_entry("schema", schema.clone())
        .build()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
    let resource =
        persist_resource_with_quota_policy(store, resource, Some(blob), quota_policy).await?;
    let artifacts = persist_default_derived_artifacts(
        store,
        &resource,
        EvidenceRole::Source,
        DerivedArtifactInput::new(&artifact_text),
    )
    .await?;
    let mut evidence_links =
        vec![EvidenceLink::new(resource.id, EvidenceRole::Source).with_part_index(part_index)];
    evidence_links.extend(evidence_links_for_derived_artifacts(
        &artifacts,
        Some(part_index),
    ));

    Ok((
        MemoryContent::Structured {
            schema,
            data: serde_json::Value::Null,
        },
        evidence_links,
    ))
}

async fn persist_external_resource(
    context: ResourcePersistenceContext<'_>,
    input: ExternalResourceInput,
) -> Result<(MemoryContent, Vec<EvidenceLink>), HirnDbError> {
    let ExternalResourceInput {
        uri,
        title,
        snippet,
        mime_type,
        checksum,
        fetch_policy,
        stale_at,
        part_index,
    } = input;
    let artifact_text = external_surrogate_text(&title, &snippet, &uri);
    let mut builder = ResourceObject::builder()
        .modality(ModalityProfile::External)
        .location(ResourceLocation::External { uri: uri.clone() })
        .owner_agent_id(context.owner_agent_id)
        .namespace(context.namespace)
        .metadata_entry("fetch_policy", fetch_policy.as_str());
    if let Some(mime_type) = mime_type.as_ref() {
        builder = builder.mime_type(mime_type.clone());
    }
    if let Some(checksum) = checksum.as_ref() {
        builder = builder.checksum(checksum.clone());
    }
    if !title.trim().is_empty() {
        builder = builder.display_name(title.clone());
    }
    if !snippet.trim().is_empty() {
        builder = builder.metadata_entry("snippet", snippet.clone());
    }
    if let Some(stale_at) = stale_at {
        builder = builder.metadata_entry("stale_at", stale_at.to_string());
    }

    let resource = builder
        .build()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
    let resource =
        persist_resource_with_quota_policy(context.store, resource, None, context.quota_policy)
            .await?;
    let artifacts = persist_default_derived_artifacts(
        context.store,
        &resource,
        EvidenceRole::Source,
        DerivedArtifactInput::new(&artifact_text),
    )
    .await?;

    let description = if !title.trim().is_empty() {
        title.clone()
    } else if !snippet.trim().is_empty() {
        snippet.clone()
    } else {
        uri.clone()
    };

    let mut evidence_links = vec![
        EvidenceLink::new(resource.id, EvidenceRole::Source)
            .with_part_index(part_index)
            .with_description(description.clone()),
    ];
    evidence_links.extend(evidence_links_for_derived_artifacts(
        &artifacts,
        Some(part_index),
    ));

    Ok((
        MemoryContent::External {
            uri,
            title,
            snippet,
            mime_type,
            checksum,
            fetch_policy,
            stale_at,
        },
        evidence_links,
    ))
}

async fn persist_tool_output_resource(
    context: ResourcePersistenceContext<'_>,
    input: ToolOutputResourceInput,
) -> Result<(MemoryContent, Vec<EvidenceLink>), HirnDbError> {
    let ToolOutputResourceInput {
        tool_name,
        output,
        mime_type,
        schema,
        invocation_id,
        checksum,
        part_index,
    } = input;
    let content = MemoryContent::ToolOutput {
        tool_name: tool_name.clone(),
        output: output.clone(),
        mime_type: mime_type.clone(),
        schema: schema.clone(),
        invocation_id: invocation_id.clone(),
        checksum: checksum.clone(),
    };
    let artifact_text = content.text_for_embedding().into_owned();
    let modality = content.modality_profile();
    let blob = output.into_bytes();
    let checksum = checksum.unwrap_or_else(|| {
        text_backed_resource_checksum(
            &format!(
                "tool_output:{}:{}",
                tool_name,
                schema.as_deref().unwrap_or_default()
            ),
            &blob,
        )
    });

    let mut builder = ResourceObject::builder()
        .modality(modality)
        .size_bytes(blob.len() as u64)
        .location(ResourceLocation::Blob { blob_index: 0 })
        .checksum(checksum.clone())
        .owner_agent_id(context.owner_agent_id)
        .namespace(context.namespace)
        .metadata_entry("content_kind", "tool_output")
        .metadata_entry("tool_name", tool_name.clone());
    if let Some(mime_type) = mime_type.as_ref() {
        builder = builder.mime_type(mime_type.clone());
    }
    if !tool_name.trim().is_empty() {
        builder = builder.display_name(tool_name.clone());
    }
    if let Some(schema) = schema.as_ref() {
        builder = builder.metadata_entry("schema", schema.clone());
    }
    if let Some(invocation_id) = invocation_id.as_ref() {
        builder = builder.metadata_entry("invocation_id", invocation_id.clone());
    }

    let resource = builder
        .build()
        .map_err(|error| HirnDbError::InvalidArgument(error.to_string()))?;
    let resource = persist_resource_with_quota_policy(
        context.store,
        resource,
        Some(blob),
        context.quota_policy,
    )
    .await?;
    let artifacts = persist_default_derived_artifacts(
        context.store,
        &resource,
        EvidenceRole::Output,
        DerivedArtifactInput::new(&artifact_text),
    )
    .await?;

    let mut link = EvidenceLink::new(resource.id, EvidenceRole::Output).with_part_index(part_index);
    if !tool_name.trim().is_empty() {
        link = link.with_description(tool_name.clone());
    }
    let mut evidence_links = vec![link];
    evidence_links.extend(evidence_links_for_derived_artifacts(
        &artifacts,
        Some(part_index),
    ));

    Ok((
        MemoryContent::ToolOutput {
            tool_name,
            output: String::new(),
            mime_type,
            schema,
            invocation_id,
            checksum: Some(checksum),
        },
        evidence_links,
    ))
}

async fn persist_attachment_resource(
    context: ResourcePersistenceContext<'_>,
    input: AttachmentResourceInput<
        '_,
        impl FnOnce(
            hirn_core::resource::ResourceObjectBuilder,
        ) -> hirn_core::resource::ResourceObjectBuilder,
    >,
) -> Result<Vec<EvidenceLink>, HirnDbError> {
    let AttachmentResourceInput {
        modality,
        role,
        mime_type,
        data,
        description,
        part_index,
        configure,
    } = input;
    let resource = build_configured_blob_resource(
        context.namespace,
        context.owner_agent_id,
        modality,
        mime_type,
        data,
        configure,
    )?;
    let resource = persist_resource_with_quota_policy(
        context.store,
        resource,
        Some(data.to_vec()),
        context.quota_policy,
    )
    .await?;
    let artifacts = persist_default_derived_artifacts(
        context.store,
        &resource,
        role,
        DerivedArtifactInput::new(description).with_blob(data, mime_type),
    )
    .await?;

    let mut link = EvidenceLink::new(resource.id, role);
    if let Some(part_index) = part_index {
        link = link.with_part_index(part_index);
    }
    if !description.is_empty() {
        link = link.with_description(description);
    }
    let mut evidence_links = vec![link];
    evidence_links.extend(evidence_links_for_derived_artifacts(&artifacts, part_index));
    Ok(evidence_links)
}

async fn resourceize_multi_content(
    store: &dyn PhysicalStore,
    namespace: Namespace,
    owner_agent_id: AgentId,
    quota_policy: &ResourceQuotaPolicy,
    content: MemoryContent,
) -> Result<ResourceizedContent, HirnDbError> {
    let context = ResourcePersistenceContext {
        store,
        namespace,
        owner_agent_id,
        quota_policy,
    };

    match content {
        MemoryContent::Image {
            data,
            mime_type,
            description,
        } if !data.is_empty() => {
            let evidence_links = persist_attachment_resource(
                context,
                AttachmentResourceInput {
                    modality: ModalityProfile::Image,
                    role: EvidenceRole::Source,
                    mime_type: Some(mime_type.as_str()),
                    data: &data,
                    description: &description,
                    part_index: Some(0),
                    configure: |builder| builder,
                },
            )
            .await?;
            Ok(ResourceizedContent {
                content: MemoryContent::Image {
                    data: Vec::new(),
                    mime_type,
                    description,
                },
                evidence_links,
            })
        }
        MemoryContent::Audio {
            data,
            transcript,
            duration_ms,
            channel_count,
        } if !data.is_empty() => {
            let evidence_links = persist_attachment_resource(
                context,
                AttachmentResourceInput {
                    modality: ModalityProfile::Audio,
                    role: EvidenceRole::Source,
                    mime_type: None,
                    data: &data,
                    description: &transcript,
                    part_index: Some(0),
                    configure: |builder| {
                        configure_audio_resource_builder(builder, duration_ms, channel_count)
                    },
                },
            )
            .await?;
            Ok(ResourceizedContent {
                content: MemoryContent::Audio {
                    data: Vec::new(),
                    transcript,
                    duration_ms,
                    channel_count,
                },
                evidence_links,
            })
        }
        MemoryContent::Video {
            data,
            mime_type,
            transcript,
            description,
        } if !data.is_empty() => {
            let surrogate = video_surrogate_text(&transcript, &description);
            let evidence_links = persist_attachment_resource(
                context,
                AttachmentResourceInput {
                    modality: ModalityProfile::Video,
                    role: EvidenceRole::Source,
                    mime_type: Some(mime_type.as_str()),
                    data: &data,
                    description: &surrogate,
                    part_index: Some(0),
                    configure: |builder| builder,
                },
            )
            .await?;
            Ok(ResourceizedContent {
                content: MemoryContent::Video {
                    data: Vec::new(),
                    mime_type,
                    transcript,
                    description,
                },
                evidence_links,
            })
        }
        MemoryContent::Document {
            data,
            mime_type,
            extracted_text,
        } if !data.is_empty() => {
            let evidence_links = persist_attachment_resource(
                context,
                AttachmentResourceInput {
                    modality: ModalityProfile::Document,
                    role: EvidenceRole::Source,
                    mime_type: Some(mime_type.as_str()),
                    data: &data,
                    description: &extracted_text,
                    part_index: Some(0),
                    configure: |builder| builder,
                },
            )
            .await?;
            Ok(ResourceizedContent {
                content: MemoryContent::Document {
                    data: Vec::new(),
                    mime_type,
                    extracted_text,
                },
                evidence_links,
            })
        }
        MemoryContent::Code {
            source,
            language,
            ast_hash,
        } if !source.is_empty() => {
            let (content, evidence_links) = persist_code_resource(
                context,
                CodeResourceInput {
                    source,
                    language,
                    ast_hash,
                    part_index: 0,
                },
            )
            .await?;
            Ok(ResourceizedContent {
                content,
                evidence_links,
            })
        }
        MemoryContent::Structured { schema, data } => {
            let (content, evidence_links) = persist_structured_resource(
                store,
                namespace,
                owner_agent_id,
                schema,
                data,
                quota_policy,
                0,
            )
            .await?;
            Ok(ResourceizedContent {
                content,
                evidence_links,
            })
        }
        MemoryContent::External {
            uri,
            title,
            snippet,
            mime_type,
            checksum,
            fetch_policy,
            stale_at,
        } => {
            let (content, evidence_links) = persist_external_resource(
                context,
                ExternalResourceInput {
                    uri,
                    title,
                    snippet,
                    mime_type,
                    checksum,
                    fetch_policy,
                    stale_at,
                    part_index: 0,
                },
            )
            .await?;
            Ok(ResourceizedContent {
                content,
                evidence_links,
            })
        }
        MemoryContent::ToolOutput {
            tool_name,
            output,
            mime_type,
            schema,
            invocation_id,
            checksum,
        } if !output.is_empty() => {
            let (content, evidence_links) = persist_tool_output_resource(
                context,
                ToolOutputResourceInput {
                    tool_name,
                    output,
                    mime_type,
                    schema,
                    invocation_id,
                    checksum,
                    part_index: 0,
                },
            )
            .await?;
            Ok(ResourceizedContent {
                content,
                evidence_links,
            })
        }
        MemoryContent::Composite(parts) => {
            let mut resourceized_parts = Vec::with_capacity(parts.len());
            let mut evidence_links = Vec::new();

            for (idx, part) in parts.into_iter().enumerate() {
                match part {
                    MemoryContent::Image {
                        data,
                        mime_type,
                        description,
                    } if !data.is_empty() => {
                        let links = persist_attachment_resource(
                            context,
                            AttachmentResourceInput {
                                modality: ModalityProfile::Image,
                                role: EvidenceRole::Source,
                                mime_type: Some(mime_type.as_str()),
                                data: &data,
                                description: &description,
                                part_index: Some(idx as u32),
                                configure: |builder| builder,
                            },
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(MemoryContent::Image {
                            data: Vec::new(),
                            mime_type,
                            description,
                        });
                    }
                    MemoryContent::Audio {
                        data,
                        transcript,
                        duration_ms,
                        channel_count,
                    } if !data.is_empty() => {
                        let links = persist_attachment_resource(
                            context,
                            AttachmentResourceInput {
                                modality: ModalityProfile::Audio,
                                role: EvidenceRole::Source,
                                mime_type: None,
                                data: &data,
                                description: &transcript,
                                part_index: Some(idx as u32),
                                configure: |builder| {
                                    configure_audio_resource_builder(
                                        builder,
                                        duration_ms,
                                        channel_count,
                                    )
                                },
                            },
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(MemoryContent::Audio {
                            data: Vec::new(),
                            transcript,
                            duration_ms,
                            channel_count,
                        });
                    }
                    MemoryContent::Video {
                        data,
                        mime_type,
                        transcript,
                        description,
                    } if !data.is_empty() => {
                        let surrogate = video_surrogate_text(&transcript, &description);
                        let links = persist_attachment_resource(
                            context,
                            AttachmentResourceInput {
                                modality: ModalityProfile::Video,
                                role: EvidenceRole::Source,
                                mime_type: Some(mime_type.as_str()),
                                data: &data,
                                description: &surrogate,
                                part_index: Some(idx as u32),
                                configure: |builder| builder,
                            },
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(MemoryContent::Video {
                            data: Vec::new(),
                            mime_type,
                            transcript,
                            description,
                        });
                    }
                    MemoryContent::Document {
                        data,
                        mime_type,
                        extracted_text,
                    } if !data.is_empty() => {
                        let links = persist_attachment_resource(
                            context,
                            AttachmentResourceInput {
                                modality: ModalityProfile::Document,
                                role: EvidenceRole::Source,
                                mime_type: Some(mime_type.as_str()),
                                data: &data,
                                description: &extracted_text,
                                part_index: Some(idx as u32),
                                configure: |builder| builder,
                            },
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(MemoryContent::Document {
                            data: Vec::new(),
                            mime_type,
                            extracted_text,
                        });
                    }
                    MemoryContent::Code {
                        source,
                        language,
                        ast_hash,
                    } if !source.is_empty() => {
                        let (content, links) = persist_code_resource(
                            context,
                            CodeResourceInput {
                                source,
                                language,
                                ast_hash,
                                part_index: idx as u32,
                            },
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(content);
                    }
                    MemoryContent::Structured { schema, data } => {
                        let (content, links) = persist_structured_resource(
                            store,
                            namespace,
                            owner_agent_id,
                            schema,
                            data,
                            quota_policy,
                            idx as u32,
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(content);
                    }
                    MemoryContent::External {
                        uri,
                        title,
                        snippet,
                        mime_type,
                        checksum,
                        fetch_policy,
                        stale_at,
                    } => {
                        let (content, links) = persist_external_resource(
                            context,
                            ExternalResourceInput {
                                uri,
                                title,
                                snippet,
                                mime_type,
                                checksum,
                                fetch_policy,
                                stale_at,
                                part_index: idx as u32,
                            },
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(content);
                    }
                    MemoryContent::ToolOutput {
                        tool_name,
                        output,
                        mime_type,
                        schema,
                        invocation_id,
                        checksum,
                    } if !output.is_empty() => {
                        let (content, links) = persist_tool_output_resource(
                            context,
                            ToolOutputResourceInput {
                                tool_name,
                                output,
                                mime_type,
                                schema,
                                invocation_id,
                                checksum,
                                part_index: idx as u32,
                            },
                        )
                        .await?;
                        evidence_links.extend(links);
                        resourceized_parts.push(content);
                    }
                    other => resourceized_parts.push(other),
                }
            }

            Ok(ResourceizedContent {
                content: MemoryContent::Composite(resourceized_parts),
                evidence_links,
            })
        }
        other => Ok(ResourceizedContent {
            content: other,
            evidence_links: Vec::new(),
        }),
    }
}

async fn build_multimodal_record(
    store: &dyn PhysicalStore,
    resource_quota_policy: &ResourceQuotaPolicy,
    input: MultimodalInput,
    embedding: Vec<f32>,
) -> Result<EpisodicRecord, HirnDbError> {
    let MultimodalInput {
        content,
        multi_content,
        blob,
        blob_mime,
        agent_id,
        namespace,
    } = input;

    let duplicate_root_blob = blob
        .as_deref()
        .zip(multi_content.as_ref().and_then(root_binary_payload))
        .is_some_and(|(blob_data, multi_blob)| blob_data == multi_blob);
    let mut builder = EpisodicRecord::builder()
        .content(&content)
        .agent_id(agent_id)
        .namespace(namespace)
        .embedding(embedding);
    let mut evidence_links = Vec::new();

    if let Some(multi_content) = multi_content {
        let resourceized = resourceize_multi_content(
            store,
            namespace,
            agent_id,
            resource_quota_policy,
            multi_content,
        )
        .await?;
        builder = builder.multi_content(resourceized.content);
        evidence_links.extend(resourceized.evidence_links);
    }

    if let Some(blob_data) = blob
        && !duplicate_root_blob
    {
        let context = ResourcePersistenceContext {
            store,
            namespace,
            owner_agent_id: agent_id,
            quota_policy: resource_quota_policy,
        };
        let links = persist_attachment_resource(
            context,
            AttachmentResourceInput {
                modality: modality_for_mime(blob_mime.as_deref()),
                role: EvidenceRole::Attachment,
                mime_type: blob_mime.as_deref(),
                data: &blob_data,
                description: &content,
                part_index: None,
                configure: |builder| builder,
            },
        )
        .await?;
        evidence_links.extend(links);
    }

    for evidence_link in evidence_links {
        builder = builder.evidence_link(evidence_link);
    }

    builder
        .build()
        .map_err(|e| HirnDbError::InvalidArgument(e.to_string()))
}

/// Ingest a single multimodal input into the episodic dataset.
///
/// Computes an embedding for the text content, persists any binary payloads as
/// linked resources, and appends the result as one episodic row.
pub async fn ingest_multimodal(
    store: &dyn PhysicalStore,
    config: &MultimodalIngestConfig,
    input: MultimodalInput,
) -> Result<EpisodicRecord, HirnDbError> {
    let input = normalize_input(input);
    let vector = if let Some(multi_content) = input.multi_content.as_ref() {
        embed_content_with_config(config, multi_content).await?
    } else {
        let embedding_text = embedding_text_for_input(&input);
        embed_text_with_route(config, route_for_input(&input), embedding_text.as_ref()).await?
    };

    let record =
        build_multimodal_record(store, &config.resource_quota_policy, input, vector).await?;

    // Serialize and append to the episodic dataset.
    let batch = ep_ds::to_batch(std::slice::from_ref(&record), config.embedding_dims)?;
    store.append(ep_ds::DATASET_NAME, batch).await?;

    Ok(record)
}

/// Ingest a batch of multimodal inputs into the episodic dataset.
///
/// More efficient than calling [`ingest_multimodal`] in a loop because it
/// batches the embedder calls and appends all rows in a single batch.
pub async fn ingest_multimodal_batch(
    store: &dyn PhysicalStore,
    config: &MultimodalIngestConfig,
    inputs: Vec<MultimodalInput>,
) -> Result<Vec<EpisodicRecord>, HirnDbError> {
    if inputs.is_empty() {
        return Ok(Vec::new());
    }

    let inputs: Vec<_> = inputs.into_iter().map(normalize_input).collect();
    let mut text_batch = Vec::new();
    let mut image_batch = Vec::new();
    let mut audio_batch = Vec::new();
    let mut video_batch = Vec::new();
    let mut code_batch = Vec::new();
    let mut document_batch = Vec::new();
    let mut composite_indices = Vec::new();

    for (idx, input) in inputs.iter().enumerate() {
        let text = embedding_text_for_input(input).into_owned();
        match route_for_input(input) {
            EmbeddingRoute::Text => text_batch.push((idx, text)),
            EmbeddingRoute::Image => image_batch.push((idx, text)),
            EmbeddingRoute::Audio => audio_batch.push((idx, text)),
            EmbeddingRoute::Video => video_batch.push((idx, text)),
            EmbeddingRoute::Code => code_batch.push((idx, text)),
            EmbeddingRoute::Document => document_batch.push((idx, text)),
            EmbeddingRoute::Composite => composite_indices.push(idx),
        }
    }

    let mut embeddings = vec![None; inputs.len()];
    embed_route_batch(
        select_embedder(config, EmbeddingRoute::Text),
        text_batch,
        &mut embeddings,
    )
    .await?;
    embed_route_batch(
        select_embedder(config, EmbeddingRoute::Image),
        image_batch,
        &mut embeddings,
    )
    .await?;
    embed_route_batch(
        select_embedder(config, EmbeddingRoute::Audio),
        audio_batch,
        &mut embeddings,
    )
    .await?;
    embed_route_batch(
        select_embedder(config, EmbeddingRoute::Video),
        video_batch,
        &mut embeddings,
    )
    .await?;
    embed_route_batch(
        select_embedder(config, EmbeddingRoute::Code),
        code_batch,
        &mut embeddings,
    )
    .await?;
    embed_route_batch(
        select_embedder(config, EmbeddingRoute::Document),
        document_batch,
        &mut embeddings,
    )
    .await?;

    for idx in composite_indices {
        let multi_content = inputs[idx].multi_content.as_ref().ok_or_else(|| {
            HirnDbError::InvalidArgument("composite route missing multi_content".into())
        })?;
        embeddings[idx] = Some(embed_content_with_config(config, multi_content).await?);
    }

    let mut records = Vec::with_capacity(inputs.len());
    for (input, embedding) in inputs.into_iter().zip(embeddings) {
        let vector = embedding.ok_or_else(|| {
            HirnDbError::InvalidArgument("no embedding generated for multimodal input".into())
        })?;
        let record =
            build_multimodal_record(store, &config.resource_quota_policy, input, vector).await?;
        records.push(record);
    }

    let batch = ep_ds::to_batch(&records, config.embedding_dims)?;
    store.append(ep_ds::DATASET_NAME, batch).await?;

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::{resource_blob as blob_ds, resource_object};
    use crate::memory_store::MemoryStore;
    use crate::resource_ops::{fetch_resource, list_derived_artifacts, load_resource_blob};
    use crate::store::ScanOptions;
    use async_trait::async_trait;
    use hirn_core::DerivedArtifactKind;
    use hirn_core::HirnResult;
    use hirn_core::HydrationMode;
    use hirn_core::content::{
        CompositeEmbeddingPolicy, CompositeModalityWeights, ExternalFetchPolicy,
    };
    use hirn_core::embed::Embedding;
    use hirn_core::metadata::MetadataValue;
    use hirn_core::types::AgentId;
    use serde_json::json;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct TestPseudoEmbedder {
        dimensions: usize,
    }

    impl TestPseudoEmbedder {
        const fn new(dimensions: usize) -> Self {
            Self { dimensions }
        }
    }

    #[async_trait]
    impl Embedder for TestPseudoEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            let mut out = Vec::with_capacity(texts.len());
            for text in texts {
                let mut vector = vec![0.0; self.dimensions];
                for (i, slot) in vector.iter_mut().enumerate() {
                    let mut hasher = DefaultHasher::new();
                    text.hash(&mut hasher);
                    i.hash(&mut hasher);
                    let hash = hasher.finish();
                    *slot = (hash as f64 / u64::MAX as f64) as f32;
                }
                out.push(Embedding {
                    vector,
                    model_id: "test-pseudo".to_string(),
                });
            }
            Ok(out)
        }

        fn dimensions(&self) -> usize {
            self.dimensions
        }

        fn model_id(&self) -> &str {
            "test-pseudo"
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    fn pseudo_embedder() -> Arc<dyn Embedder> {
        Arc::new(TestPseudoEmbedder::new(4))
    }
    fn test_config() -> MultimodalIngestConfig {
        MultimodalIngestConfig {
            text_embedder: pseudo_embedder(),
            image_embedder: None,
            audio_embedder: None,
            video_embedder: None,
            code_embedder: None,
            document_embedder: None,
            composite_policy: CompositeEmbeddingPolicy::default(),
            embedding_dims: 4,
            resource_quota_policy: ResourceQuotaPolicy::default(),
        }
    }

    struct CountingEmbedder {
        model_id: &'static str,
        calls: Arc<AtomicUsize>,
    }

    struct VectorEmbedder {
        model_id: &'static str,
        vector: Vec<f32>,
    }

    #[async_trait]
    impl Embedder for CountingEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: vec![1.0; 4],
                    model_id: self.model_id.to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            self.model_id
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    #[async_trait]
    impl Embedder for VectorEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|_| Embedding {
                    vector: self.vector.clone(),
                    model_id: self.model_id.to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            self.vector.len()
        }

        fn model_id(&self) -> &str {
            self.model_id
        }

        fn max_input_tokens(&self) -> usize {
            8192
        }
    }

    fn assert_vector_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "component {idx} mismatch: expected {expected}, got {actual}"
            );
        }
    }

    fn test_namespace() -> Namespace {
        Namespace::new("media").unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_text_only() {
        let store = MemoryStore::new();
        let config = test_config();

        let input = MultimodalInput {
            content: "hello world".into(),
            multi_content: None,
            blob: None,
            blob_mime: None,
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.content, "hello world");
        assert!(rec.embedding.is_some());
        assert!(rec.provenance.evidence_links.is_empty());

        // Verify it was stored.
        let results = store
            .scan(ep_ds::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap();
        let decoded: Vec<_> = results
            .iter()
            .flat_map(|b| ep_ds::from_batch(b).unwrap())
            .collect();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].content, "hello world");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_image_with_caption() {
        let store = MemoryStore::new();
        let config = test_config();

        let img_data = vec![0x89, 0x50, 0x4E, 0x47];
        let input = MultimodalInput {
            content: "a cat sitting on a mat".into(),
            multi_content: Some(MemoryContent::Image {
                data: img_data.clone(),
                mime_type: "image/png".into(),
                description: "a cat sitting on a mat".into(),
            }),
            blob: Some(img_data.clone()),
            blob_mime: Some("image/png".into()),
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert!(rec.embedding.is_some());
        assert_eq!(rec.provenance.evidence_links.len(), 3);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[0].part_index, Some(0));
        assert!(
            rec.provenance.evidence_links[1..]
                .iter()
                .all(|link| link.role == EvidenceRole::Derived)
        );
        assert!(
            rec.provenance.evidence_links[1..]
                .iter()
                .all(|link| link.part_index == Some(0))
        );
        let artifacts =
            list_derived_artifacts(&store, rec.provenance.evidence_links[0].resource_id)
                .await
                .unwrap();
        assert_eq!(artifacts.len(), 3);
        assert_eq!(artifacts[0].kind, DerivedArtifactKind::Caption);
        assert_eq!(
            artifacts[0].text_content.as_deref(),
            Some("a cat sitting on a mat")
        );
        assert_eq!(artifacts[1].kind, DerivedArtifactKind::OcrText);
        assert_eq!(
            artifacts[1].text_content.as_deref(),
            Some("a cat sitting on a mat")
        );
        assert_eq!(artifacts[2].kind, DerivedArtifactKind::GenerationFailure);
        let resource_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        assert_eq!(resource_blob, img_data);

        let results = store
            .scan(ep_ds::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap();
        let decoded: Vec<_> = results
            .iter()
            .flat_map(|b| ep_ds::from_batch(b).unwrap())
            .collect();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].provenance.evidence_links.len(), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_image_without_caption_records_generation_failure() {
        let store = MemoryStore::new();
        let config = test_config();

        let img_data = vec![0x89, 0x50, 0x4E, 0x47];
        let input = MultimodalInput {
            content: "image missing caption source text".into(),
            multi_content: Some(MemoryContent::Image {
                data: img_data.clone(),
                mime_type: "image/png".into(),
                description: String::new(),
            }),
            blob: Some(img_data.clone()),
            blob_mime: Some("image/png".into()),
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 1);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        match rec.multi_content.as_ref() {
            Some(MemoryContent::Image {
                data,
                mime_type,
                description,
            }) => {
                assert!(data.is_empty());
                assert_eq!(mime_type, "image/png");
                assert!(description.is_empty());
            }
            other => panic!("expected Image multi_content, got {other:?}"),
        }
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[0].part_index, Some(0));
        let artifacts =
            list_derived_artifacts(&store, rec.provenance.evidence_links[0].resource_id)
                .await
                .unwrap();
        assert_eq!(artifacts.len(), 2);
        assert!(
            artifacts
                .iter()
                .all(|artifact| artifact.kind == DerivedArtifactKind::GenerationFailure)
        );
        assert!(
            artifacts
                .iter()
                .any(|artifact| artifact.text_content.as_deref()
                    == Some("caption generation failed: source text was empty"))
        );
        assert!(matches!(
            artifacts
                .iter()
                .find(|artifact| artifact.text_content.as_deref()
                    == Some("caption generation failed: source text was empty"))
                .and_then(|artifact| artifact.metadata.get("intended_kind")),
            Some(MetadataValue::String(value)) if value == "caption"
        ));
        let resource_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        assert_eq!(resource_blob, img_data);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_audio_with_transcript() {
        let store = MemoryStore::new();
        let config = test_config();

        let audio_data = vec![0x52, 0x49, 0x46, 0x46]; // "RIFF" header
        let input = MultimodalInput {
            content: "meeting about auth system".into(),
            multi_content: Some(MemoryContent::Audio {
                data: audio_data.clone(),
                transcript: "meeting about auth system".into(),
                duration_ms: 60_000,
                channel_count: Some(2),
            }),
            blob: Some(audio_data.clone()),
            blob_mime: Some("audio/wav".into()),
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 2);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[1].role, EvidenceRole::Derived);
        let artifacts =
            list_derived_artifacts(&store, rec.provenance.evidence_links[0].resource_id)
                .await
                .unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, DerivedArtifactKind::Transcript);
        assert_eq!(
            artifacts[0].text_content.as_deref(),
            Some("meeting about auth system")
        );
        let hydrated = fetch_resource(
            &store,
            rec.provenance.evidence_links[0].resource_id,
            HydrationMode::MetadataOnly,
        )
        .await
        .unwrap()
        .unwrap();
        assert!(matches!(
            hydrated.resource.metadata.get("duration_ms"),
            Some(MetadataValue::Int(value)) if *value == 60_000
        ));
        assert!(matches!(
            hydrated.resource.metadata.get("channel_count"),
            Some(MetadataValue::Int(value)) if *value == 2
        ));
        let resource_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        assert_eq!(resource_blob, audio_data);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_code_with_syntax_summary() {
        let store = MemoryStore::new();
        let config = test_config();

        let source = "fn main() { println!(\"hello\"); }";
        let input = MultimodalInput {
            content: "rust hello world example".into(),
            multi_content: Some(MemoryContent::Code {
                source: source.into(),
                language: "rust".into(),
                ast_hash: Some("ast123".into()),
            }),
            blob: None,
            blob_mime: None,
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 2);
        assert_eq!(rec.provenance.evidence_links[0].part_index, Some(0));
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[1].role, EvidenceRole::Derived);
        assert_eq!(rec.provenance.evidence_links[1].part_index, Some(0));
        let artifacts =
            list_derived_artifacts(&store, rec.provenance.evidence_links[0].resource_id)
                .await
                .unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, DerivedArtifactKind::SyntaxSummary);
        assert_eq!(artifacts[0].text_content.as_deref(), Some(source));
        let resource_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        assert_eq!(resource_blob, source.as_bytes());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_structured_with_schema_summary() {
        let store = MemoryStore::new();
        let config = test_config();

        let data = json!({"service": "auth", "status": "green"});
        let input = MultimodalInput {
            content: "auth service health snapshot".into(),
            multi_content: Some(MemoryContent::Structured {
                schema: "health.v1".into(),
                data: data.clone(),
            }),
            blob: None,
            blob_mime: None,
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 2);
        assert_eq!(rec.provenance.evidence_links[0].part_index, Some(0));
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[1].role, EvidenceRole::Derived);
        assert_eq!(rec.provenance.evidence_links[1].part_index, Some(0));
        let artifacts =
            list_derived_artifacts(&store, rec.provenance.evidence_links[0].resource_id)
                .await
                .unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, DerivedArtifactKind::SchemaSummary);
        assert_eq!(
            artifacts[0].text_content.as_deref(),
            Some("[health.v1] {\"service\":\"auth\",\"status\":\"green\"}")
        );
        let resource_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        let restored: serde_json::Value = serde_json::from_slice(&resource_blob).unwrap();
        assert_eq!(restored, data);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_batch() {
        let store = MemoryStore::new();
        let config = test_config();

        let inputs = vec![
            MultimodalInput {
                content: "first item".into(),
                multi_content: None,
                blob: None,
                blob_mime: None,
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "second item with blob".into(),
                multi_content: None,
                blob: Some(vec![1, 2, 3]),
                blob_mime: Some("application/octet-stream".into()),
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
        ];

        let records = ingest_multimodal_batch(&store, &config, inputs)
            .await
            .unwrap();
        assert_eq!(records.len(), 2);

        let results = store
            .scan(ep_ds::DATASET_NAME, ScanOptions::default())
            .await
            .unwrap();
        let decoded: Vec<_> = results
            .iter()
            .flat_map(|b| ep_ds::from_batch(b).unwrap())
            .collect();
        assert_eq!(decoded.len(), 2);
        assert!(decoded[0].provenance.evidence_links.is_empty());
        assert_eq!(decoded[1].provenance.evidence_links.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_pdf_as_blob() {
        let store = MemoryStore::new();
        let config = test_config();

        let pdf_data = b"%PDF-1.4 fake pdf content".to_vec();
        let input = MultimodalInput {
            content: "extracted text from the PDF document".into(),
            multi_content: None,
            blob: Some(pdf_data.clone()),
            blob_mime: Some("application/pdf".into()),
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 2);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[1].role, EvidenceRole::Preview);
        let artifacts =
            list_derived_artifacts(&store, rec.provenance.evidence_links[0].resource_id)
                .await
                .unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, DerivedArtifactKind::Preview);
        assert_eq!(
            artifacts[0].text_content.as_deref(),
            Some("extracted text from the PDF document")
        );
        let resource_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        assert_eq!(resource_blob, pdf_data);

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
    async fn ingest_mixed_source_and_attachment_resources() {
        let store = MemoryStore::new();
        let config = test_config();

        let image_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D];
        let attachment_data = b"%PDF-1.4 attachment".to_vec();
        let input = MultimodalInput {
            content: "diagram with attached spec".into(),
            multi_content: Some(MemoryContent::Image {
                data: image_data.clone(),
                mime_type: "image/png".into(),
                description: "architecture diagram".into(),
            }),
            blob: Some(attachment_data.clone()),
            blob_mime: Some("application/pdf".into()),
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 5);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[0].part_index, Some(0));
        assert!(
            rec.provenance.evidence_links[1..3]
                .iter()
                .all(|link| link.role == EvidenceRole::Derived)
        );
        let attachment_link = rec
            .provenance
            .evidence_links
            .iter()
            .find(|link| link.role == EvidenceRole::Attachment)
            .expect("attachment evidence link should be present");
        assert_eq!(attachment_link.part_index, None);

        let source_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        let attachment_blob = load_resource_blob(&store, attachment_link.resource_id, 0)
            .await
            .unwrap();
        assert_eq!(source_blob, image_data);
        assert_eq!(attachment_blob, attachment_data);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_external_reference_persists_external_resource_and_preview() {
        let store = MemoryStore::new();
        let config = test_config();

        let input = MultimodalInput {
            content: "release evidence".into(),
            multi_content: Some(MemoryContent::External {
                uri: "https://example.com/releases/42".into(),
                title: "release dashboard".into(),
                snippet: "green rollout completed".into(),
                mime_type: Some("text/html".into()),
                checksum: Some("sha256:release-42".into()),
                fetch_policy: ExternalFetchPolicy::IfStale,
                stale_at: None,
            }),
            blob: None,
            blob_mime: None,
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 2);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[1].role, EvidenceRole::Preview);
        match rec.multi_content.as_ref() {
            Some(MemoryContent::External {
                uri,
                title,
                snippet,
                mime_type,
                checksum,
                fetch_policy,
                ..
            }) => {
                assert_eq!(uri, "https://example.com/releases/42");
                assert_eq!(title, "release dashboard");
                assert_eq!(snippet, "green rollout completed");
                assert_eq!(mime_type.as_deref(), Some("text/html"));
                assert_eq!(checksum.as_deref(), Some("sha256:release-42"));
                assert_eq!(*fetch_policy, ExternalFetchPolicy::IfStale);
            }
            other => panic!("expected External multi_content, got {other:?}"),
        }

        let hydrated = fetch_resource(
            &store,
            rec.provenance.evidence_links[0].resource_id,
            HydrationMode::Preview,
        )
        .await
        .unwrap()
        .expect("external resource should exist");
        assert!(matches!(
            hydrated.resource.location,
            ResourceLocation::External { .. }
        ));
        assert_eq!(
            hydrated.resource.display_name.as_deref(),
            Some("release dashboard")
        );
        assert_eq!(
            hydrated.resource.metadata.get("fetch_policy"),
            Some(&MetadataValue::String("if_stale".into()))
        );
        assert!(hydrated.blob.is_none());
        assert_eq!(hydrated.artifacts.len(), 1);
        assert_eq!(hydrated.artifacts[0].kind, DerivedArtifactKind::Preview);
        assert_eq!(
            hydrated.artifacts[0].text_content.as_deref(),
            Some("release dashboard\ngreen rollout completed\nhttps://example.com/releases/42")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_tool_output_persists_output_resource_and_preview() {
        let store = MemoryStore::new();
        let config = test_config();

        let input = MultimodalInput {
            content: "deployment tool output".into(),
            multi_content: Some(MemoryContent::ToolOutput {
                tool_name: "terraform".into(),
                output: r#"{"applied":true}"#.into(),
                mime_type: Some("application/json".into()),
                schema: Some("terraform/apply.v1".into()),
                invocation_id: Some("apply-42".into()),
                checksum: None,
            }),
            blob: None,
            blob_mime: None,
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 2);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Output);
        assert_eq!(rec.provenance.evidence_links[1].role, EvidenceRole::Preview);
        match rec.multi_content.as_ref() {
            Some(MemoryContent::ToolOutput {
                tool_name,
                output,
                mime_type,
                schema,
                invocation_id,
                ..
            }) => {
                assert_eq!(tool_name, "terraform");
                assert!(output.is_empty());
                assert_eq!(mime_type.as_deref(), Some("application/json"));
                assert_eq!(schema.as_deref(), Some("terraform/apply.v1"));
                assert_eq!(invocation_id.as_deref(), Some("apply-42"));
            }
            other => panic!("expected ToolOutput multi_content, got {other:?}"),
        }

        let hydrated = fetch_resource(
            &store,
            rec.provenance.evidence_links[0].resource_id,
            HydrationMode::Preview,
        )
        .await
        .unwrap()
        .expect("tool output resource should exist");
        assert_eq!(hydrated.resource.modality, ModalityProfile::Structured);
        assert_eq!(hydrated.resource.display_name.as_deref(), Some("terraform"));
        assert_eq!(
            hydrated.resource.metadata.get("content_kind"),
            Some(&MetadataValue::String("tool_output".into()))
        );
        assert_eq!(
            hydrated.resource.metadata.get("tool_name"),
            Some(&MetadataValue::String("terraform".into()))
        );
        assert_eq!(hydrated.artifacts.len(), 1);
        assert_eq!(hydrated.artifacts[0].kind, DerivedArtifactKind::Preview);
        assert_eq!(
            hydrated.artifacts[0].text_content.as_deref(),
            Some("terraform\n{\"applied\":true}")
        );

        let blob = load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
            .await
            .unwrap();
        assert_eq!(blob, br#"{"applied":true}"#);
    }

    #[test]
    fn select_embedder_prefers_primary_multi_content_route() {
        let text = pseudo_embedder();
        let image: Arc<dyn Embedder> = Arc::new(CountingEmbedder {
            model_id: "image",
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let document: Arc<dyn Embedder> = Arc::new(CountingEmbedder {
            model_id: "document",
            calls: Arc::new(AtomicUsize::new(0)),
        });
        let config = MultimodalIngestConfig {
            text_embedder: text,
            image_embedder: Some(image),
            audio_embedder: None,
            video_embedder: None,
            code_embedder: None,
            document_embedder: Some(document),
            composite_policy: CompositeEmbeddingPolicy::default(),
            embedding_dims: 4,
            resource_quota_policy: ResourceQuotaPolicy::default(),
        };
        let input = MultimodalInput {
            content: "diagram with attached spec".into(),
            multi_content: Some(MemoryContent::Image {
                data: vec![1, 2, 3],
                mime_type: "image/png".into(),
                description: "architecture diagram".into(),
            }),
            blob: Some(b"%PDF-1.4 attachment".to_vec()),
            blob_mime: Some("application/pdf".into()),
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let embedder = select_embedder(&config, route_for_input(&input));
        assert_eq!(embedder.model_id(), "image");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_multimodal_batch_uses_route_specific_embedders() {
        let store = MemoryStore::new();
        let text_calls = Arc::new(AtomicUsize::new(0));
        let image_calls = Arc::new(AtomicUsize::new(0));
        let audio_calls = Arc::new(AtomicUsize::new(0));
        let video_calls = Arc::new(AtomicUsize::new(0));
        let code_calls = Arc::new(AtomicUsize::new(0));
        let document_calls = Arc::new(AtomicUsize::new(0));
        let config = MultimodalIngestConfig {
            text_embedder: Arc::new(CountingEmbedder {
                model_id: "text",
                calls: Arc::clone(&text_calls),
            }),
            image_embedder: Some(Arc::new(CountingEmbedder {
                model_id: "image",
                calls: Arc::clone(&image_calls),
            })),
            audio_embedder: Some(Arc::new(CountingEmbedder {
                model_id: "audio",
                calls: Arc::clone(&audio_calls),
            })),
            video_embedder: Some(Arc::new(CountingEmbedder {
                model_id: "video",
                calls: Arc::clone(&video_calls),
            })),
            code_embedder: Some(Arc::new(CountingEmbedder {
                model_id: "code",
                calls: Arc::clone(&code_calls),
            })),
            document_embedder: Some(Arc::new(CountingEmbedder {
                model_id: "document",
                calls: Arc::clone(&document_calls),
            })),
            composite_policy: CompositeEmbeddingPolicy::default(),
            embedding_dims: 4,
            resource_quota_policy: ResourceQuotaPolicy::default(),
        };
        let inputs = vec![
            MultimodalInput {
                content: "plain text".into(),
                multi_content: None,
                blob: None,
                blob_mime: None,
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "image caption".into(),
                multi_content: Some(MemoryContent::Image {
                    data: vec![1, 2, 3],
                    mime_type: "image/png".into(),
                    description: "image caption".into(),
                }),
                blob: None,
                blob_mime: Some("image/png".into()),
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "meeting transcript".into(),
                multi_content: Some(MemoryContent::Audio {
                    data: vec![0x52, 0x49],
                    transcript: "meeting transcript".into(),
                    duration_ms: 1_000,
                    channel_count: Some(1),
                }),
                blob: None,
                blob_mime: Some("audio/wav".into()),
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "rust helper".into(),
                multi_content: Some(MemoryContent::Code {
                    source: "fn main() {}".into(),
                    language: "rust".into(),
                    ast_hash: None,
                }),
                blob: None,
                blob_mime: None,
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "video summary".into(),
                multi_content: Some(MemoryContent::Video {
                    data: vec![0, 0, 1, 0xBA],
                    mime_type: "video/mp4".into(),
                    transcript: "deployment dashboard walkthrough".into(),
                    description: "screen capture with release status".into(),
                }),
                blob: None,
                blob_mime: Some("video/mp4".into()),
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "document text".into(),
                multi_content: None,
                blob: Some(b"%PDF-1.4 doc".to_vec()),
                blob_mime: Some("application/pdf".into()),
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "release evidence".into(),
                multi_content: Some(MemoryContent::External {
                    uri: "https://example.com/releases/42".into(),
                    title: "release dashboard".into(),
                    snippet: "green rollout completed".into(),
                    mime_type: Some("text/html".into()),
                    checksum: None,
                    fetch_policy: ExternalFetchPolicy::OnDemand,
                    stale_at: None,
                }),
                blob: None,
                blob_mime: None,
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
            MultimodalInput {
                content: "tool output".into(),
                multi_content: Some(MemoryContent::ToolOutput {
                    tool_name: "terraform".into(),
                    output: r#"{"applied":true}"#.into(),
                    mime_type: Some("application/json".into()),
                    schema: Some("terraform/apply.v1".into()),
                    invocation_id: None,
                    checksum: None,
                }),
                blob: None,
                blob_mime: None,
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            },
        ];

        let records = ingest_multimodal_batch(&store, &config, inputs)
            .await
            .unwrap();
        assert_eq!(records.len(), 8);
        assert_eq!(text_calls.load(Ordering::SeqCst), 1);
        assert_eq!(image_calls.load(Ordering::SeqCst), 1);
        assert_eq!(audio_calls.load(Ordering::SeqCst), 1);
        assert_eq!(video_calls.load(Ordering::SeqCst), 1);
        assert_eq!(code_calls.load(Ordering::SeqCst), 1);
        assert_eq!(document_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_video_with_preview_artifact_and_placeholder_blob() {
        let store = MemoryStore::new();
        let config = test_config();

        let video_data = vec![0, 0, 1, 0xBA, 0x44, 0x00];
        let input = MultimodalInput {
            content: "incident walk-through recording".into(),
            multi_content: Some(MemoryContent::Video {
                data: video_data.clone(),
                mime_type: "video/mp4".into(),
                transcript: "incident walk-through recording".into(),
                description: "screen capture of the deployment timeline".into(),
            }),
            blob: Some(video_data.clone()),
            blob_mime: Some("video/mp4".into()),
            agent_id: AgentId::well_known("test-agent"),
            namespace: test_namespace(),
        };

        let rec = ingest_multimodal(&store, &config, input).await.unwrap();
        assert_eq!(rec.provenance.evidence_links.len(), 2);
        assert_eq!(rec.provenance.evidence_links[0].role, EvidenceRole::Source);
        assert_eq!(rec.provenance.evidence_links[1].role, EvidenceRole::Preview);
        match rec.multi_content.as_ref() {
            Some(MemoryContent::Video {
                data,
                mime_type,
                transcript,
                description,
            }) => {
                assert!(data.is_empty());
                assert_eq!(mime_type, "video/mp4");
                assert_eq!(transcript, "incident walk-through recording");
                assert_eq!(description, "screen capture of the deployment timeline");
            }
            other => panic!("expected Video multi_content, got {other:?}"),
        }
        let artifacts =
            list_derived_artifacts(&store, rec.provenance.evidence_links[0].resource_id)
                .await
                .unwrap();
        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].kind, DerivedArtifactKind::Preview);
        assert_eq!(
            artifacts[0].text_content.as_deref(),
            Some("incident walk-through recording\nscreen capture of the deployment timeline")
        );
        let resource_blob =
            load_resource_blob(&store, rec.provenance.evidence_links[0].resource_id, 0)
                .await
                .unwrap();
        assert_eq!(resource_blob, video_data);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn ingest_multimodal_batch_embeds_composite_with_documented_weighted_policy() {
        let store = MemoryStore::new();
        let config = MultimodalIngestConfig {
            text_embedder: Arc::new(VectorEmbedder {
                model_id: "text",
                vector: vec![1.0, 0.0, 0.0, 0.0],
            }),
            image_embedder: Some(Arc::new(VectorEmbedder {
                model_id: "image",
                vector: vec![0.0, 1.0, 0.0, 0.0],
            })),
            audio_embedder: Some(Arc::new(VectorEmbedder {
                model_id: "audio",
                vector: vec![0.0, 0.0, 1.0, 0.0],
            })),
            video_embedder: None,
            code_embedder: None,
            document_embedder: None,
            composite_policy: CompositeEmbeddingPolicy::WeightedMeanNormalized(
                CompositeModalityWeights {
                    text: 2.0,
                    image: 1.0,
                    audio: 1.0,
                    ..Default::default()
                },
            ),
            embedding_dims: 4,
            resource_quota_policy: ResourceQuotaPolicy::default(),
        };

        let records = ingest_multimodal_batch(
            &store,
            &config,
            vec![MultimodalInput {
                content: "composite bundle".into(),
                multi_content: Some(MemoryContent::Composite(vec![
                    MemoryContent::Text("summary".into()),
                    MemoryContent::Image {
                        data: vec![1, 2, 3],
                        mime_type: "image/png".into(),
                        description: "diagram".into(),
                    },
                    MemoryContent::Audio {
                        data: vec![0x52, 0x49],
                        transcript: "meeting".into(),
                        duration_ms: 1_000,
                        channel_count: Some(1),
                    },
                ])),
                blob: None,
                blob_mime: None,
                agent_id: AgentId::well_known("test-agent"),
                namespace: test_namespace(),
            }],
        )
        .await
        .unwrap();

        let embedding = records[0]
            .embedding
            .as_ref()
            .expect("composite batch record should be embedded");
        assert_vector_close(embedding, &[0.81649655, 0.40824828, 0.40824828, 0.0]);
    }
}
