use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use dashmap::DashMap;
use serde_json::Value;

use hirn_core::content::MemoryContent;
use hirn_core::episodic::EpisodicRecord;
use hirn_core::resource::{
    EvidenceLink, EvidenceRole, ModalityProfile, ResourceLocation, ResourceObject,
    ResourceQuotaPolicy,
};
use hirn_core::revision::LogicalMemoryId;
use hirn_core::semantic::SemanticRecord;
use hirn_core::types::{AgentId, Namespace};
use hirn_core::{HirnError, HirnResult};
use hirn_storage::PhysicalStore;
use hirn_storage::{configure_audio_resource_builder, evidence_links_for_derived_artifacts};

fn structured_artifact_text(schema: &str, data: &Value) -> String {
    if schema.trim().is_empty() {
        data.to_string()
    } else {
        format!("[{schema}] {data}")
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

pub(crate) struct ExtractedResources {
    pub(crate) content: MemoryContent,
    pub(crate) evidence_links: Vec<EvidenceLink>,
}

pub(crate) struct StorageRuntime {
    path: PathBuf,
    storage: Arc<dyn PhysicalStore>,
    resource_quota_policy: ResourceQuotaPolicy,
    fts_initialized: AtomicBool,
    episodic_heads: DashMap<LogicalMemoryId, EpisodicRecord>,
    semantic_heads: DashMap<LogicalMemoryId, SemanticRecord>,
}

impl StorageRuntime {
    pub(crate) fn new(
        path: PathBuf,
        storage: Arc<dyn PhysicalStore>,
        resource_quota_policy: ResourceQuotaPolicy,
    ) -> Self {
        Self {
            path,
            storage,
            resource_quota_policy,
            fts_initialized: AtomicBool::new(false),
            episodic_heads: DashMap::new(),
            semantic_heads: DashMap::new(),
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn storage_backend(&self) -> &dyn PhysicalStore {
        self.storage.as_ref()
    }

    pub(crate) fn storage_arc(&self) -> Arc<dyn PhysicalStore> {
        Arc::clone(&self.storage)
    }

    pub(crate) fn cached_semantic_head(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> Option<SemanticRecord> {
        self.semantic_heads
            .get(&logical_memory_id)
            .map(|entry| entry.value().clone())
    }

    pub(crate) fn cached_episodic_head(
        &self,
        logical_memory_id: LogicalMemoryId,
    ) -> Option<EpisodicRecord> {
        self.episodic_heads
            .get(&logical_memory_id)
            .map(|entry| entry.value().clone())
    }

    pub(crate) fn cache_episodic_head(&self, record: EpisodicRecord) {
        self.episodic_heads.insert(record.logical_memory_id, record);
    }

    pub(crate) fn evict_episodic_head(&self, logical_memory_id: LogicalMemoryId) {
        self.episodic_heads.remove(&logical_memory_id);
    }

    pub(crate) fn cache_semantic_head(&self, record: SemanticRecord) {
        self.semantic_heads.insert(record.logical_memory_id, record);
    }

    pub(crate) fn evict_semantic_head(&self, logical_memory_id: LogicalMemoryId) {
        self.semantic_heads.remove(&logical_memory_id);
    }

    pub(crate) fn cached_semantic_heads_snapshot(
        &self,
    ) -> std::collections::HashMap<LogicalMemoryId, SemanticRecord> {
        self.semantic_heads
            .iter()
            .map(|entry| (*entry.key(), entry.value().clone()))
            .collect()
    }

    pub(crate) fn replace_semantic_heads(&self, records: impl IntoIterator<Item = SemanticRecord>) {
        self.semantic_heads.clear();
        for record in records {
            self.semantic_heads.insert(record.logical_memory_id, record);
        }
    }

    pub(crate) fn file_size_bytes(&self) -> u64 {
        std::fs::metadata(&self.path)
            .map(|meta| meta.len())
            .unwrap_or(0)
    }

    pub(crate) fn fts_initialized(&self) -> bool {
        self.fts_initialized.load(Ordering::Relaxed)
    }

    pub(crate) async fn ensure_fts_indexes(&self) -> HirnResult<()> {
        if self.fts_initialized() {
            return Ok(());
        }

        let fts_targets: &[(&str, &str)] = &[
            (hirn_storage::datasets::episodic::DATASET_NAME, "content"),
            (
                hirn_storage::datasets::semantic::DATASET_NAME,
                "description",
            ),
            (
                hirn_storage::datasets::procedural::DATASET_NAME,
                "description",
            ),
        ];

        for &(dataset, column) in fts_targets {
            if !self.storage.exists(dataset).await.unwrap_or(false) {
                continue;
            }

            let config = hirn_storage::store::IndexConfig {
                columns: vec![column.to_owned()],
                index_type: hirn_storage::store::IndexType::Bm25,
                replace: false,
                params: Default::default(),
            };

            match self.storage.create_index(dataset, config).await {
                Ok(()) => {
                    tracing::info!(dataset, column, "FTS index created");
                }
                Err(error) => {
                    tracing::debug!(dataset, column, error = %error, "FTS index creation skipped");
                }
            }
        }

        self.fts_initialized.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) async fn create_vector_indexes(
        &self,
        index_type: hirn_storage::store::IndexType,
        params: Option<hirn_storage::store::IndexParams>,
    ) -> HirnResult<()> {
        self.apply_vector_indexes(index_type, params, false).await
    }

    pub(crate) async fn rebuild_vector_indexes(
        &self,
        index_type: hirn_storage::store::IndexType,
        params: Option<hirn_storage::store::IndexParams>,
    ) -> HirnResult<()> {
        self.apply_vector_indexes(index_type, params, true).await
    }

    async fn apply_vector_indexes(
        &self,
        index_type: hirn_storage::store::IndexType,
        params: Option<hirn_storage::store::IndexParams>,
        replace: bool,
    ) -> HirnResult<()> {
        let targets: &[&str] = &[
            hirn_storage::datasets::episodic::DATASET_NAME,
            hirn_storage::datasets::semantic::DATASET_NAME,
            hirn_storage::datasets::procedural::DATASET_NAME,
        ];

        for &dataset in targets {
            if !self.storage.exists(dataset).await.unwrap_or(false) {
                continue;
            }

            let count = self.storage.count(dataset, None).await.unwrap_or(0);
            if count == 0 {
                continue;
            }

            let config = hirn_storage::store::IndexConfig {
                columns: vec!["embedding".to_owned()],
                index_type: index_type.clone(),
                replace,
                params: params.clone().unwrap_or_default(),
            };

            match self.storage.create_index(dataset, config).await {
                Ok(()) => {
                    tracing::info!(dataset, "vector index created");
                }
                Err(error) => {
                    tracing::debug!(dataset, error = %error, "vector index creation skipped");
                }
            }
        }

        Ok(())
    }

    pub(crate) async fn extract_and_store_resources(
        &self,
        namespace: Namespace,
        owner_agent_id: AgentId,
        content: &MemoryContent,
    ) -> HirnResult<ExtractedResources> {
        match content {
            MemoryContent::Image {
                data,
                mime_type,
                description,
            } if !data.is_empty() => {
                let evidence_links = self
                    .store_resource_blob(
                        namespace,
                        owner_agent_id,
                        ModalityProfile::Image,
                        EvidenceRole::Source,
                        Some(mime_type.as_str()),
                        data,
                        0,
                        description,
                        |builder| builder,
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: MemoryContent::Image {
                        data: Vec::new(),
                        mime_type: mime_type.clone(),
                        description: description.clone(),
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
                let evidence_links = self
                    .store_resource_blob(
                        namespace,
                        owner_agent_id,
                        ModalityProfile::Audio,
                        EvidenceRole::Source,
                        None,
                        data,
                        0,
                        transcript,
                        |builder| {
                            configure_audio_resource_builder(builder, *duration_ms, *channel_count)
                        },
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: MemoryContent::Audio {
                        data: Vec::new(),
                        transcript: transcript.clone(),
                        duration_ms: *duration_ms,
                        channel_count: *channel_count,
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
                let surrogate = video_surrogate_text(transcript, description);
                let evidence_links = self
                    .store_resource_blob(
                        namespace,
                        owner_agent_id,
                        ModalityProfile::Video,
                        EvidenceRole::Source,
                        Some(mime_type.as_str()),
                        data,
                        0,
                        &surrogate,
                        |builder| builder,
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: MemoryContent::Video {
                        data: Vec::new(),
                        mime_type: mime_type.clone(),
                        transcript: transcript.clone(),
                        description: description.clone(),
                    },
                    evidence_links,
                })
            }
            MemoryContent::Document {
                data,
                mime_type,
                extracted_text,
            } if !data.is_empty() => {
                let evidence_links = self
                    .store_resource_blob(
                        namespace,
                        owner_agent_id,
                        ModalityProfile::Document,
                        EvidenceRole::Source,
                        Some(mime_type.as_str()),
                        data,
                        0,
                        extracted_text,
                        |builder| builder,
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: MemoryContent::Document {
                        data: Vec::new(),
                        mime_type: mime_type.clone(),
                        extracted_text: extracted_text.clone(),
                    },
                    evidence_links,
                })
            }
            MemoryContent::Code {
                source,
                language,
                ast_hash,
            } if !source.is_empty() => {
                let evidence_links = self
                    .store_text_resource_blob(
                        namespace,
                        owner_agent_id,
                        ModalityProfile::Code,
                        EvidenceRole::Source,
                        Some("text/plain"),
                        source.as_bytes(),
                        0,
                        source,
                        |builder| {
                            let mut builder = builder.metadata_entry("language", language.clone());
                            if let Some(ast_hash) = ast_hash.as_ref() {
                                builder = builder.metadata_entry("ast_hash", ast_hash.clone());
                            }
                            builder.checksum(hirn_storage::text_backed_resource_checksum(
                                &format!("code:{language}"),
                                source.as_bytes(),
                            ))
                        },
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: MemoryContent::Code {
                        source: String::new(),
                        language: language.clone(),
                        ast_hash: ast_hash.clone(),
                    },
                    evidence_links,
                })
            }
            MemoryContent::Structured { schema, data } => {
                let blob = serde_json::to_vec(data)
                    .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
                let artifact_text = structured_artifact_text(schema, data);
                let evidence_links = self
                    .store_text_resource_blob(
                        namespace,
                        owner_agent_id,
                        ModalityProfile::Structured,
                        EvidenceRole::Source,
                        Some("application/json"),
                        &blob,
                        0,
                        &artifact_text,
                        |builder| {
                            builder.metadata_entry("schema", schema.clone()).checksum(
                                hirn_storage::text_backed_resource_checksum(
                                    &format!("structured:{schema}"),
                                    &blob,
                                ),
                            )
                        },
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: MemoryContent::Structured {
                        schema: schema.clone(),
                        data: Value::Null,
                    },
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
                let artifact_text = external_surrogate_text(title, snippet, uri);
                let description = if !title.trim().is_empty() {
                    title.clone()
                } else if !snippet.trim().is_empty() {
                    snippet.clone()
                } else {
                    uri.clone()
                };
                let evidence_links = self
                    .store_external_resource_reference(
                        namespace,
                        owner_agent_id,
                        uri,
                        title,
                        snippet,
                        mime_type.as_deref(),
                        checksum.as_deref(),
                        *fetch_policy,
                        *stale_at,
                        0,
                        &artifact_text,
                        &description,
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: content.clone(),
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
                let artifact_text = content.text_for_embedding().into_owned();
                let evidence_links = self
                    .store_tool_output_resource(
                        namespace,
                        owner_agent_id,
                        tool_name,
                        output,
                        mime_type.as_deref(),
                        schema.as_deref(),
                        invocation_id.as_deref(),
                        checksum.as_deref(),
                        0,
                        &artifact_text,
                    )
                    .await?;
                Ok(ExtractedResources {
                    content: MemoryContent::ToolOutput {
                        tool_name: tool_name.clone(),
                        output: String::new(),
                        mime_type: mime_type.clone(),
                        schema: schema.clone(),
                        invocation_id: invocation_id.clone(),
                        checksum: checksum.clone(),
                    },
                    evidence_links,
                })
            }
            MemoryContent::Composite(parts) => {
                let mut new_parts = Vec::with_capacity(parts.len());
                let mut evidence_links = Vec::new();
                for (idx, part) in parts.iter().enumerate() {
                    match part {
                        MemoryContent::Image {
                            data,
                            mime_type,
                            description,
                        } if !data.is_empty() => {
                            let links = self
                                .store_resource_blob(
                                    namespace,
                                    owner_agent_id,
                                    ModalityProfile::Image,
                                    EvidenceRole::Source,
                                    Some(mime_type.as_str()),
                                    data,
                                    idx as u32,
                                    description,
                                    |builder| builder,
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(MemoryContent::Image {
                                data: Vec::new(),
                                mime_type: mime_type.clone(),
                                description: description.clone(),
                            });
                        }
                        MemoryContent::Audio {
                            data,
                            transcript,
                            duration_ms,
                            channel_count,
                        } if !data.is_empty() => {
                            let links = self
                                .store_resource_blob(
                                    namespace,
                                    owner_agent_id,
                                    ModalityProfile::Audio,
                                    EvidenceRole::Source,
                                    None,
                                    data,
                                    idx as u32,
                                    transcript,
                                    |builder| {
                                        configure_audio_resource_builder(
                                            builder,
                                            *duration_ms,
                                            *channel_count,
                                        )
                                    },
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(MemoryContent::Audio {
                                data: Vec::new(),
                                transcript: transcript.clone(),
                                duration_ms: *duration_ms,
                                channel_count: *channel_count,
                            });
                        }
                        MemoryContent::Video {
                            data,
                            mime_type,
                            transcript,
                            description,
                        } if !data.is_empty() => {
                            let surrogate = video_surrogate_text(transcript, description);
                            let links = self
                                .store_resource_blob(
                                    namespace,
                                    owner_agent_id,
                                    ModalityProfile::Video,
                                    EvidenceRole::Source,
                                    Some(mime_type.as_str()),
                                    data,
                                    idx as u32,
                                    &surrogate,
                                    |builder| builder,
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(MemoryContent::Video {
                                data: Vec::new(),
                                mime_type: mime_type.clone(),
                                transcript: transcript.clone(),
                                description: description.clone(),
                            });
                        }
                        MemoryContent::Document {
                            data,
                            mime_type,
                            extracted_text,
                        } if !data.is_empty() => {
                            let links = self
                                .store_resource_blob(
                                    namespace,
                                    owner_agent_id,
                                    ModalityProfile::Document,
                                    EvidenceRole::Source,
                                    Some(mime_type.as_str()),
                                    data,
                                    idx as u32,
                                    extracted_text,
                                    |builder| builder,
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(MemoryContent::Document {
                                data: Vec::new(),
                                mime_type: mime_type.clone(),
                                extracted_text: extracted_text.clone(),
                            });
                        }
                        MemoryContent::Code {
                            source,
                            language,
                            ast_hash,
                        } if !source.is_empty() => {
                            let links = self
                                .store_text_resource_blob(
                                    namespace,
                                    owner_agent_id,
                                    ModalityProfile::Code,
                                    EvidenceRole::Source,
                                    Some("text/plain"),
                                    source.as_bytes(),
                                    idx as u32,
                                    source,
                                    |builder| {
                                        let mut builder =
                                            builder.metadata_entry("language", language.clone());
                                        if let Some(ast_hash) = ast_hash.as_ref() {
                                            builder = builder
                                                .metadata_entry("ast_hash", ast_hash.clone());
                                        }
                                        builder.checksum(
                                            hirn_storage::text_backed_resource_checksum(
                                                &format!("code:{language}"),
                                                source.as_bytes(),
                                            ),
                                        )
                                    },
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(MemoryContent::Code {
                                source: String::new(),
                                language: language.clone(),
                                ast_hash: ast_hash.clone(),
                            });
                        }
                        MemoryContent::Structured { schema, data } => {
                            let blob = serde_json::to_vec(data)
                                .map_err(|error| HirnError::InvalidInput(error.to_string()))?;
                            let artifact_text = structured_artifact_text(schema, data);
                            let links = self
                                .store_text_resource_blob(
                                    namespace,
                                    owner_agent_id,
                                    ModalityProfile::Structured,
                                    EvidenceRole::Source,
                                    Some("application/json"),
                                    &blob,
                                    idx as u32,
                                    &artifact_text,
                                    |builder| {
                                        builder.metadata_entry("schema", schema.clone()).checksum(
                                            hirn_storage::text_backed_resource_checksum(
                                                &format!("structured:{schema}"),
                                                &blob,
                                            ),
                                        )
                                    },
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(MemoryContent::Structured {
                                schema: schema.clone(),
                                data: Value::Null,
                            });
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
                            let artifact_text = external_surrogate_text(title, snippet, uri);
                            let description = if !title.trim().is_empty() {
                                title.clone()
                            } else if !snippet.trim().is_empty() {
                                snippet.clone()
                            } else {
                                uri.clone()
                            };
                            let links = self
                                .store_external_resource_reference(
                                    namespace,
                                    owner_agent_id,
                                    uri,
                                    title,
                                    snippet,
                                    mime_type.as_deref(),
                                    checksum.as_deref(),
                                    *fetch_policy,
                                    *stale_at,
                                    idx as u32,
                                    &artifact_text,
                                    &description,
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(part.clone());
                        }
                        MemoryContent::ToolOutput {
                            tool_name,
                            output,
                            mime_type,
                            schema,
                            invocation_id,
                            checksum,
                        } if !output.is_empty() => {
                            let artifact_text = part.text_for_embedding().into_owned();
                            let links = self
                                .store_tool_output_resource(
                                    namespace,
                                    owner_agent_id,
                                    tool_name,
                                    output,
                                    mime_type.as_deref(),
                                    schema.as_deref(),
                                    invocation_id.as_deref(),
                                    checksum.as_deref(),
                                    idx as u32,
                                    &artifact_text,
                                )
                                .await?;
                            evidence_links.extend(links);
                            new_parts.push(MemoryContent::ToolOutput {
                                tool_name: tool_name.clone(),
                                output: String::new(),
                                mime_type: mime_type.clone(),
                                schema: schema.clone(),
                                invocation_id: invocation_id.clone(),
                                checksum: checksum.clone(),
                            });
                        }
                        other => new_parts.push(other.clone()),
                    }
                }
                Ok(ExtractedResources {
                    content: MemoryContent::Composite(new_parts),
                    evidence_links,
                })
            }
            other => Ok(ExtractedResources {
                content: other.clone(),
                evidence_links: Vec::new(),
            }),
        }
    }

    pub(crate) async fn load_resource_blob(
        &self,
        evidence_links: &[EvidenceLink],
        blob_index: u32,
    ) -> HirnResult<Vec<u8>> {
        let resource_id = evidence_links
            .iter()
            .find(|link| link.part_index == Some(blob_index) && link.artifact_id.is_none())
            .map(|link| link.resource_id)
            .ok_or_else(|| HirnError::NotFound(format!("resource blob slot {blob_index}")))?;

        hirn_storage::load_resource_blob(self.storage.as_ref(), resource_id, 0)
            .await
            .map_err(HirnError::storage)
    }

    pub(crate) async fn hydrate_content_resources(
        &self,
        content: &MemoryContent,
        evidence_links: &[EvidenceLink],
    ) -> HirnResult<MemoryContent> {
        match content {
            MemoryContent::Image {
                data,
                mime_type,
                description,
            } if data.is_empty() => match self.load_resource_blob(evidence_links, 0).await {
                Ok(blob) => Ok(MemoryContent::Image {
                    data: blob,
                    mime_type: mime_type.clone(),
                    description: description.clone(),
                }),
                Err(_) => Ok(content.clone()),
            },
            MemoryContent::Audio {
                data,
                transcript,
                duration_ms,
                channel_count,
            } if data.is_empty() => match self.load_resource_blob(evidence_links, 0).await {
                Ok(blob) => Ok(MemoryContent::Audio {
                    data: blob,
                    transcript: transcript.clone(),
                    duration_ms: *duration_ms,
                    channel_count: *channel_count,
                }),
                Err(_) => Ok(content.clone()),
            },
            MemoryContent::Video {
                data,
                mime_type,
                transcript,
                description,
            } if data.is_empty() => match self.load_resource_blob(evidence_links, 0).await {
                Ok(blob) => Ok(MemoryContent::Video {
                    data: blob,
                    mime_type: mime_type.clone(),
                    transcript: transcript.clone(),
                    description: description.clone(),
                }),
                Err(_) => Ok(content.clone()),
            },
            MemoryContent::Document {
                data,
                mime_type,
                extracted_text,
            } if data.is_empty() => match self.load_resource_blob(evidence_links, 0).await {
                Ok(blob) => Ok(MemoryContent::Document {
                    data: blob,
                    mime_type: mime_type.clone(),
                    extracted_text: extracted_text.clone(),
                }),
                Err(_) => Ok(content.clone()),
            },
            MemoryContent::Code {
                source,
                language,
                ast_hash,
            } if source.is_empty() => match self.load_resource_blob(evidence_links, 0).await {
                Ok(blob) => match String::from_utf8(blob) {
                    Ok(source) => Ok(MemoryContent::Code {
                        source,
                        language: language.clone(),
                        ast_hash: ast_hash.clone(),
                    }),
                    Err(_) => Ok(content.clone()),
                },
                Err(_) => Ok(content.clone()),
            },
            MemoryContent::Structured { schema, data } if data.is_null() => {
                match self.load_resource_blob(evidence_links, 0).await {
                    Ok(blob) => match serde_json::from_slice(&blob) {
                        Ok(data) => Ok(MemoryContent::Structured {
                            schema: schema.clone(),
                            data,
                        }),
                        Err(_) => Ok(content.clone()),
                    },
                    Err(_) => Ok(content.clone()),
                }
            }
            MemoryContent::ToolOutput {
                tool_name,
                output,
                mime_type,
                schema,
                invocation_id,
                checksum,
            } if output.is_empty() => match self.load_resource_blob(evidence_links, 0).await {
                Ok(blob) => match String::from_utf8(blob) {
                    Ok(output) => Ok(MemoryContent::ToolOutput {
                        tool_name: tool_name.clone(),
                        output,
                        mime_type: mime_type.clone(),
                        schema: schema.clone(),
                        invocation_id: invocation_id.clone(),
                        checksum: checksum.clone(),
                    }),
                    Err(_) => Ok(content.clone()),
                },
                Err(_) => Ok(content.clone()),
            },
            MemoryContent::Composite(parts) => {
                let mut restored = Vec::with_capacity(parts.len());
                for (idx, part) in parts.iter().enumerate() {
                    match part {
                        MemoryContent::Image {
                            data,
                            mime_type,
                            description,
                        } if data.is_empty() => {
                            match self.load_resource_blob(evidence_links, idx as u32).await {
                                Ok(blob) => restored.push(MemoryContent::Image {
                                    data: blob,
                                    mime_type: mime_type.clone(),
                                    description: description.clone(),
                                }),
                                Err(_) => restored.push(part.clone()),
                            }
                        }
                        MemoryContent::Audio {
                            data,
                            transcript,
                            duration_ms,
                            channel_count,
                        } if data.is_empty() => {
                            match self.load_resource_blob(evidence_links, idx as u32).await {
                                Ok(blob) => restored.push(MemoryContent::Audio {
                                    data: blob,
                                    transcript: transcript.clone(),
                                    duration_ms: *duration_ms,
                                    channel_count: *channel_count,
                                }),
                                Err(_) => restored.push(part.clone()),
                            }
                        }
                        MemoryContent::Video {
                            data,
                            mime_type,
                            transcript,
                            description,
                        } if data.is_empty() => {
                            match self.load_resource_blob(evidence_links, idx as u32).await {
                                Ok(blob) => restored.push(MemoryContent::Video {
                                    data: blob,
                                    mime_type: mime_type.clone(),
                                    transcript: transcript.clone(),
                                    description: description.clone(),
                                }),
                                Err(_) => restored.push(part.clone()),
                            }
                        }
                        MemoryContent::Document {
                            data,
                            mime_type,
                            extracted_text,
                        } if data.is_empty() => {
                            match self.load_resource_blob(evidence_links, idx as u32).await {
                                Ok(blob) => restored.push(MemoryContent::Document {
                                    data: blob,
                                    mime_type: mime_type.clone(),
                                    extracted_text: extracted_text.clone(),
                                }),
                                Err(_) => restored.push(part.clone()),
                            }
                        }
                        MemoryContent::Code {
                            source,
                            language,
                            ast_hash,
                        } if source.is_empty() => {
                            match self.load_resource_blob(evidence_links, idx as u32).await {
                                Ok(blob) => match String::from_utf8(blob) {
                                    Ok(source) => restored.push(MemoryContent::Code {
                                        source,
                                        language: language.clone(),
                                        ast_hash: ast_hash.clone(),
                                    }),
                                    Err(_) => restored.push(part.clone()),
                                },
                                Err(_) => restored.push(part.clone()),
                            }
                        }
                        MemoryContent::Structured { schema, data } if data.is_null() => {
                            match self.load_resource_blob(evidence_links, idx as u32).await {
                                Ok(blob) => match serde_json::from_slice(&blob) {
                                    Ok(data) => restored.push(MemoryContent::Structured {
                                        schema: schema.clone(),
                                        data,
                                    }),
                                    Err(_) => restored.push(part.clone()),
                                },
                                Err(_) => restored.push(part.clone()),
                            }
                        }
                        MemoryContent::ToolOutput {
                            tool_name,
                            output,
                            mime_type,
                            schema,
                            invocation_id,
                            checksum,
                        } if output.is_empty() => {
                            match self.load_resource_blob(evidence_links, idx as u32).await {
                                Ok(blob) => match String::from_utf8(blob) {
                                    Ok(output) => restored.push(MemoryContent::ToolOutput {
                                        tool_name: tool_name.clone(),
                                        output,
                                        mime_type: mime_type.clone(),
                                        schema: schema.clone(),
                                        invocation_id: invocation_id.clone(),
                                        checksum: checksum.clone(),
                                    }),
                                    Err(_) => restored.push(part.clone()),
                                },
                                Err(_) => restored.push(part.clone()),
                            }
                        }
                        other => restored.push(other.clone()),
                    }
                }
                Ok(MemoryContent::Composite(restored))
            }
            other => Ok(other.clone()),
        }
    }

    async fn store_resource_blob<F>(
        &self,
        namespace: Namespace,
        owner_agent_id: AgentId,
        modality: ModalityProfile,
        role: EvidenceRole,
        mime_type: Option<&str>,
        data: &[u8],
        part_index: u32,
        description: &str,
        configure: F,
    ) -> HirnResult<Vec<EvidenceLink>>
    where
        F: FnOnce(
            hirn_core::resource::ResourceObjectBuilder,
        ) -> hirn_core::resource::ResourceObjectBuilder,
    {
        let resource = hirn_storage::build_configured_blob_resource(
            namespace,
            owner_agent_id,
            modality,
            mime_type,
            data,
            configure,
        )
        .map_err(HirnError::storage)?;
        let resource = hirn_storage::persist_resource_with_quota_policy(
            self.storage.as_ref(),
            resource,
            Some(data.to_vec()),
            &self.resource_quota_policy,
        )
        .await?;
        let artifacts = hirn_storage::persist_default_derived_artifacts(
            self.storage.as_ref(),
            &resource,
            role,
            hirn_storage::DerivedArtifactInput::new(description).with_blob(data, mime_type),
        )
        .await?;

        let mut evidence_links = vec![
            EvidenceLink::new(resource.id, role)
                .with_part_index(part_index)
                .with_description(description),
        ];
        evidence_links.extend(evidence_links_for_derived_artifacts(
            &artifacts,
            Some(part_index),
        ));
        Ok(evidence_links)
    }

    async fn store_text_resource_blob<F>(
        &self,
        namespace: Namespace,
        owner_agent_id: AgentId,
        modality: ModalityProfile,
        role: EvidenceRole,
        mime_type: Option<&str>,
        data: &[u8],
        part_index: u32,
        artifact_text: &str,
        configure: F,
    ) -> HirnResult<Vec<EvidenceLink>>
    where
        F: FnOnce(
            hirn_core::resource::ResourceObjectBuilder,
        ) -> hirn_core::resource::ResourceObjectBuilder,
    {
        let mut builder = ResourceObject::builder()
            .modality(modality)
            .size_bytes(data.len() as u64)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .owner_agent_id(owner_agent_id)
            .namespace(namespace);
        if let Some(mime_type) = mime_type {
            builder = builder.mime_type(mime_type);
        }
        let resource = configure(builder).build()?;
        let resource = hirn_storage::persist_resource_with_quota_policy(
            self.storage.as_ref(),
            resource,
            Some(data.to_vec()),
            &self.resource_quota_policy,
        )
        .await?;
        let artifacts = hirn_storage::persist_default_derived_artifacts(
            self.storage.as_ref(),
            &resource,
            role,
            hirn_storage::DerivedArtifactInput::new(artifact_text),
        )
        .await?;

        let mut evidence_links =
            vec![EvidenceLink::new(resource.id, role).with_part_index(part_index)];
        evidence_links.extend(evidence_links_for_derived_artifacts(
            &artifacts,
            Some(part_index),
        ));
        Ok(evidence_links)
    }

    async fn store_external_resource_reference(
        &self,
        namespace: Namespace,
        owner_agent_id: AgentId,
        uri: &str,
        title: &str,
        snippet: &str,
        mime_type: Option<&str>,
        checksum: Option<&str>,
        fetch_policy: hirn_core::content::ExternalFetchPolicy,
        stale_at: Option<hirn_core::Timestamp>,
        part_index: u32,
        artifact_text: &str,
        description: &str,
    ) -> HirnResult<Vec<EvidenceLink>> {
        let mut builder = ResourceObject::builder()
            .modality(ModalityProfile::External)
            .location(ResourceLocation::External {
                uri: uri.to_string(),
            })
            .owner_agent_id(owner_agent_id)
            .namespace(namespace)
            .metadata_entry("fetch_policy", fetch_policy.as_str());
        if let Some(mime_type) = mime_type {
            builder = builder.mime_type(mime_type);
        }
        if let Some(checksum) = checksum {
            builder = builder.checksum(checksum);
        }
        if !title.trim().is_empty() {
            builder = builder.display_name(title);
        }
        if !snippet.trim().is_empty() {
            builder = builder.metadata_entry("snippet", snippet.to_string());
        }
        if let Some(stale_at) = stale_at {
            builder = builder.metadata_entry("stale_at", stale_at.to_string());
        }
        let resource = builder.build()?;
        let resource = hirn_storage::persist_resource_with_quota_policy(
            self.storage.as_ref(),
            resource,
            None,
            &self.resource_quota_policy,
        )
        .await?;
        let artifacts = hirn_storage::persist_default_derived_artifacts(
            self.storage.as_ref(),
            &resource,
            EvidenceRole::Source,
            hirn_storage::DerivedArtifactInput::new(artifact_text),
        )
        .await?;

        let mut evidence_links = vec![
            EvidenceLink::new(resource.id, EvidenceRole::Source)
                .with_part_index(part_index)
                .with_description(description),
        ];
        evidence_links.extend(evidence_links_for_derived_artifacts(
            &artifacts,
            Some(part_index),
        ));
        Ok(evidence_links)
    }

    async fn store_tool_output_resource(
        &self,
        namespace: Namespace,
        owner_agent_id: AgentId,
        tool_name: &str,
        output: &str,
        mime_type: Option<&str>,
        schema: Option<&str>,
        invocation_id: Option<&str>,
        checksum: Option<&str>,
        part_index: u32,
        artifact_text: &str,
    ) -> HirnResult<Vec<EvidenceLink>> {
        let content = MemoryContent::ToolOutput {
            tool_name: tool_name.to_string(),
            output: output.to_string(),
            mime_type: mime_type.map(str::to_string),
            schema: schema.map(str::to_string),
            invocation_id: invocation_id.map(str::to_string),
            checksum: checksum.map(str::to_string),
        };
        let modality = content.modality_profile();
        let blob = output.as_bytes();
        let checksum = checksum.map(str::to_string).unwrap_or_else(|| {
            hirn_storage::text_backed_resource_checksum(
                &format!("tool_output:{}:{}", tool_name, schema.unwrap_or_default()),
                blob,
            )
        });

        let mut builder = ResourceObject::builder()
            .modality(modality)
            .size_bytes(blob.len() as u64)
            .location(ResourceLocation::Blob { blob_index: 0 })
            .checksum(checksum)
            .owner_agent_id(owner_agent_id)
            .namespace(namespace)
            .metadata_entry("content_kind", "tool_output")
            .metadata_entry("tool_name", tool_name.to_string());
        if let Some(mime_type) = mime_type {
            builder = builder.mime_type(mime_type);
        }
        if !tool_name.trim().is_empty() {
            builder = builder.display_name(tool_name);
        }
        if let Some(schema) = schema {
            builder = builder.metadata_entry("schema", schema.to_string());
        }
        if let Some(invocation_id) = invocation_id {
            builder = builder.metadata_entry("invocation_id", invocation_id.to_string());
        }

        let resource = builder.build()?;
        let resource = hirn_storage::persist_resource_with_quota_policy(
            self.storage.as_ref(),
            resource,
            Some(blob.to_vec()),
            &self.resource_quota_policy,
        )
        .await?;
        let artifacts = hirn_storage::persist_default_derived_artifacts(
            self.storage.as_ref(),
            &resource,
            EvidenceRole::Output,
            hirn_storage::DerivedArtifactInput::new(artifact_text),
        )
        .await?;

        let mut link =
            EvidenceLink::new(resource.id, EvidenceRole::Output).with_part_index(part_index);
        if !tool_name.trim().is_empty() {
            link = link.with_description(tool_name);
        }
        let mut evidence_links = vec![link];
        evidence_links.extend(evidence_links_for_derived_artifacts(
            &artifacts,
            Some(part_index),
        ));
        Ok(evidence_links)
    }
}

impl Deref for StorageRuntime {
    type Target = Arc<dyn PhysicalStore>;

    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use hirn_core::content::{ExternalFetchPolicy, MemoryContent};
    use hirn_core::metadata::MetadataValue;
    use hirn_core::{HydrationMode, Timestamp};
    use hirn_storage::fetch_resource;
    use hirn_storage::memory_store::MemoryStore;

    #[tokio::test(flavor = "multi_thread")]
    async fn ensure_fts_indexes_marks_runtime_initialized() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let runtime = StorageRuntime::new(
            dir.path().join("db"),
            Arc::new(MemoryStore::new()),
            ResourceQuotaPolicy::default(),
        );

        assert!(!runtime.fts_initialized());
        runtime
            .ensure_fts_indexes()
            .await
            .expect("FTS initialization should succeed");
        assert!(runtime.fts_initialized());

        runtime
            .ensure_fts_indexes()
            .await
            .expect("FTS initialization should stay idempotent");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn blob_round_trip_restores_large_image_data() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let runtime = StorageRuntime::new(
            dir.path().join("db"),
            Arc::new(MemoryStore::new()),
            ResourceQuotaPolicy::default(),
        );
        let blob_data: Vec<u8> = (0..2048).map(|value| (value % 251) as u8).collect();
        let content = MemoryContent::Image {
            data: blob_data.clone(),
            mime_type: "image/png".into(),
            description: "storage-runtime".into(),
        };

        let extracted = runtime
            .extract_and_store_resources(
                Namespace::default(),
                hirn_core::types::AgentId::well_known("storage-runtime-test"),
                &content,
            )
            .await
            .expect("blob extraction should succeed");

        match &extracted.content {
            MemoryContent::Image { data, .. } => assert!(data.is_empty()),
            _ => panic!("expected image placeholder"),
        }
        let source_link = extracted
            .evidence_links
            .iter()
            .find(|link| link.artifact_id.is_none())
            .expect("source image link should be present");
        assert_eq!(source_link.part_index, Some(0));
        assert!(
            extracted
                .evidence_links
                .iter()
                .any(|link| link.artifact_id.is_some())
        );

        let restored = runtime
            .hydrate_content_resources(&extracted.content, &extracted.evidence_links)
            .await
            .expect("blob restoration should succeed");

        match restored {
            MemoryContent::Image { data, .. } => assert_eq!(data, blob_data),
            _ => panic!("expected restored image"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn blob_round_trip_restores_large_video_data() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let runtime = StorageRuntime::new(
            dir.path().join("db"),
            Arc::new(MemoryStore::new()),
            ResourceQuotaPolicy::default(),
        );
        let blob_data: Vec<u8> = (0..4096).map(|value| (value % 239) as u8).collect();
        let content = MemoryContent::Video {
            data: blob_data.clone(),
            mime_type: "video/mp4".into(),
            transcript: "release review recording".into(),
            description: "screen capture of rollout dashboard".into(),
        };

        let extracted = runtime
            .extract_and_store_resources(
                Namespace::default(),
                hirn_core::types::AgentId::well_known("storage-runtime-test"),
                &content,
            )
            .await
            .expect("video extraction should succeed");

        match &extracted.content {
            MemoryContent::Video { data, .. } => assert!(data.is_empty()),
            _ => panic!("expected video placeholder"),
        }
        let source_link = extracted
            .evidence_links
            .iter()
            .find(|link| link.artifact_id.is_none())
            .expect("source video link should be present");
        assert_eq!(source_link.part_index, Some(0));
        assert!(
            extracted
                .evidence_links
                .iter()
                .any(|link| link.artifact_id.is_some())
        );

        let restored = runtime
            .hydrate_content_resources(&extracted.content, &extracted.evidence_links)
            .await
            .expect("video restoration should succeed");

        match restored {
            MemoryContent::Video { data, .. } => assert_eq!(data, blob_data),
            _ => panic!("expected restored video"),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn external_reference_extracts_to_external_resource_without_hydration() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let runtime = StorageRuntime::new(
            dir.path().join("db"),
            Arc::new(MemoryStore::new()),
            ResourceQuotaPolicy::default(),
        );
        let content = MemoryContent::External {
            uri: "https://example.com/releases/42".into(),
            title: "release dashboard".into(),
            snippet: "green rollout completed".into(),
            mime_type: Some("text/html".into()),
            checksum: Some("sha256:release-42".into()),
            fetch_policy: ExternalFetchPolicy::IfStale,
            stale_at: Some(Timestamp::from_millis(1_712_345_678_000)),
        };

        let extracted = runtime
            .extract_and_store_resources(
                Namespace::default(),
                hirn_core::types::AgentId::well_known("storage-runtime-test"),
                &content,
            )
            .await
            .expect("external extraction should succeed");

        assert_eq!(extracted.content, content);
        let source_link = extracted
            .evidence_links
            .iter()
            .find(|link| link.artifact_id.is_none())
            .expect("source external link should be present");
        assert_eq!(source_link.part_index, Some(0));
        let source_resource_id = source_link.resource_id;

        let hydrated = runtime
            .hydrate_content_resources(&extracted.content, &extracted.evidence_links)
            .await
            .expect("external hydration should be a no-op");
        assert_eq!(hydrated, content);

        let resource = fetch_resource(
            runtime.storage_backend(),
            source_resource_id,
            HydrationMode::Preview,
        )
        .await
        .expect("fetch should succeed")
        .expect("resource should exist");
        assert!(matches!(
            resource.resource.location,
            ResourceLocation::External { .. }
        ));
        assert_eq!(
            resource.resource.metadata.get("fetch_policy"),
            Some(&MetadataValue::String("if_stale".into()))
        );
        assert!(resource.blob.is_none());
        assert!(!resource.artifacts.is_empty());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn tool_output_round_trip_restores_placeholder_output() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let runtime = StorageRuntime::new(
            dir.path().join("db"),
            Arc::new(MemoryStore::new()),
            ResourceQuotaPolicy::default(),
        );
        let content = MemoryContent::ToolOutput {
            tool_name: "terraform".into(),
            output: r#"{"applied":true}"#.into(),
            mime_type: Some("application/json".into()),
            schema: Some("terraform/apply.v1".into()),
            invocation_id: Some("apply-42".into()),
            checksum: Some("sha256:apply".into()),
        };

        let extracted = runtime
            .extract_and_store_resources(
                Namespace::default(),
                hirn_core::types::AgentId::well_known("storage-runtime-test"),
                &content,
            )
            .await
            .expect("tool output extraction should succeed");

        match &extracted.content {
            MemoryContent::ToolOutput { output, .. } => assert!(output.is_empty()),
            _ => panic!("expected tool output placeholder"),
        }
        let output_link = extracted
            .evidence_links
            .iter()
            .find(|link| link.artifact_id.is_none())
            .expect("source tool output link should be present");
        assert_eq!(output_link.role, EvidenceRole::Output);
        let output_resource_id = output_link.resource_id;

        let restored = runtime
            .hydrate_content_resources(&extracted.content, &extracted.evidence_links)
            .await
            .expect("tool output hydration should succeed");

        assert_eq!(restored, content);

        let resource = fetch_resource(
            runtime.storage_backend(),
            output_resource_id,
            HydrationMode::Preview,
        )
        .await
        .expect("fetch should succeed")
        .expect("resource should exist");
        assert_eq!(resource.resource.display_name.as_deref(), Some("terraform"));
        assert_eq!(
            resource.resource.metadata.get("content_kind"),
            Some(&MetadataValue::String("tool_output".into()))
        );
        assert_eq!(resource.artifacts.len(), 1);
        assert_eq!(
            resource.artifacts[0].kind,
            hirn_core::DerivedArtifactKind::Preview
        );
    }
}
