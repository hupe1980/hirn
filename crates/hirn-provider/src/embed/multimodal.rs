//! Multi-modal embedding router.
//!
//! Routes `MemoryContent` to the appropriate embedder slot per modality.
//! All current routes hand textual surrogates to the underlying `Embedder`
//! trait: text directly, code via source, audio via transcript, image via
//! description, video via transcript plus description, document via extracted
//! text, external references via title/snippet/URI, and structured data via JSON.
//! Dedicated modality embedders therefore specialize those surrogate inputs,
//! not raw binary payload bytes.

use std::sync::Arc;

use async_trait::async_trait;
use hirn_core::content::{CompositeEmbeddingPolicy, MemoryContent};
use hirn_core::embed::{Embedder, Embedding, MultivectorEmbedding};
use hirn_core::{HirnError, HirnResult};

/// Routes embedding requests to the correct embedder per modality.
///
/// By default, all modalities extract their text representation
/// (via [`MemoryContent::text_for_embedding`]) and delegate to the
/// text embedder. Dedicated modality embedders can be registered for
/// image descriptions, audio transcripts, video surrogates, code, or document
/// text without changing the shared `Embedder` contract. External references
/// reuse the text embedder via their text surrogate.
#[allow(clippy::struct_field_names)]
#[derive(Clone)]
pub struct MultiModalEmbedder {
    /// Default text embedder (handles text, code, audio-transcript, structured).
    text_embedder: Arc<dyn Embedder>,
    /// Optional embedder for image description surrogates.
    image_embedder: Option<Arc<dyn Embedder>>,
    /// Optional audio embedder for transcript-oriented audio indexing.
    audio_embedder: Option<Arc<dyn Embedder>>,
    /// Optional video embedder for transcript/scene-description surrogates.
    video_embedder: Option<Arc<dyn Embedder>>,
    /// Optional code embedder (e.g. `CodeBERT`).
    code_embedder: Option<Arc<dyn Embedder>>,
    /// Optional document embedder for extracted document text.
    document_embedder: Option<Arc<dyn Embedder>>,
    /// Policy used when collapsing composite content into one aggregate bundle.
    composite_policy: CompositeEmbeddingPolicy,
}

impl MultiModalEmbedder {
    /// Create a multi-modal embedder backed by the given text embedder.
    #[must_use]
    pub fn new(text_embedder: Arc<dyn Embedder>) -> Self {
        Self {
            text_embedder,
            image_embedder: None,
            audio_embedder: None,
            video_embedder: None,
            code_embedder: None,
            document_embedder: None,
            composite_policy: CompositeEmbeddingPolicy::default(),
        }
    }

    /// Register a dedicated image-description embedder.
    #[must_use]
    pub fn with_image_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.image_embedder = Some(embedder);
        self
    }

    /// Register a specialized audio embedder.
    #[must_use]
    pub fn with_audio_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.audio_embedder = Some(embedder);
        self
    }

    /// Register a specialized video embedder.
    #[must_use]
    pub fn with_video_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.video_embedder = Some(embedder);
        self
    }

    /// Register a specialized code embedder (e.g. `CodeBERT`).
    #[must_use]
    pub fn with_code_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.code_embedder = Some(embedder);
        self
    }

    /// Register a specialized document embedder.
    #[must_use]
    pub fn with_document_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.document_embedder = Some(embedder);
        self
    }

    /// Set the policy used to collapse composite content into one aggregate
    /// embedding bundle.
    #[must_use]
    pub fn with_composite_policy(mut self, policy: CompositeEmbeddingPolicy) -> Self {
        self.composite_policy = policy;
        self
    }

    /// Clone this router while mapping each configured provider through the
    /// supplied wrapper builder.
    #[must_use]
    pub fn map_embedders<F>(&self, mut mapper: F) -> Self
    where
        F: FnMut(Arc<dyn Embedder>) -> Arc<dyn Embedder>,
    {
        Self {
            text_embedder: mapper(Arc::clone(&self.text_embedder)),
            image_embedder: self
                .image_embedder
                .as_ref()
                .map(|embedder| mapper(Arc::clone(embedder))),
            audio_embedder: self
                .audio_embedder
                .as_ref()
                .map(|embedder| mapper(Arc::clone(embedder))),
            video_embedder: self
                .video_embedder
                .as_ref()
                .map(|embedder| mapper(Arc::clone(embedder))),
            code_embedder: self
                .code_embedder
                .as_ref()
                .map(|embedder| mapper(Arc::clone(embedder))),
            document_embedder: self
                .document_embedder
                .as_ref()
                .map(|embedder| mapper(Arc::clone(embedder))),
            composite_policy: self.composite_policy.clone(),
        }
    }

    /// Embed a single [`MemoryContent`], routing to the correct embedder.
    pub async fn embed_content(&self, content: &MemoryContent) -> HirnResult<Embedding> {
        match content {
            MemoryContent::Image { description, .. } => {
                let embedder = self.image_embedder.as_ref().unwrap_or(&self.text_embedder);
                embed_one(embedder.as_ref(), description).await
            }
            MemoryContent::Code { source, .. } => {
                let embedder = self.code_embedder.as_ref().unwrap_or(&self.text_embedder);
                embed_one(embedder.as_ref(), source).await
            }
            MemoryContent::Audio { transcript, .. } => {
                let embedder = self.audio_embedder.as_ref().unwrap_or(&self.text_embedder);
                embed_one(embedder.as_ref(), transcript).await
            }
            MemoryContent::Video { .. } => {
                let embedder = self.video_embedder.as_ref().unwrap_or(&self.text_embedder);
                let text = content.text_for_embedding();
                embed_one(embedder.as_ref(), text.as_ref()).await
            }
            MemoryContent::Document { extracted_text, .. } => {
                let embedder = self
                    .document_embedder
                    .as_ref()
                    .unwrap_or(&self.text_embedder);
                embed_one(embedder.as_ref(), extracted_text).await
            }
            MemoryContent::External { .. } => {
                let text = content.text_for_embedding();
                embed_one(self.text_embedder.as_ref(), text.as_ref()).await
            }
            MemoryContent::ToolOutput { .. } => {
                let text = content.text_for_embedding();
                embed_one(self.text_embedder.as_ref(), text.as_ref()).await
            }
            MemoryContent::Text(t) => embed_one(self.text_embedder.as_ref(), t).await,
            MemoryContent::Structured { data, .. } => {
                let json_str = data.to_string();
                embed_one(self.text_embedder.as_ref(), &json_str).await
            }
            MemoryContent::Composite(parts) => self.embed_composite(parts).await,
        }
    }

    /// Embed a batch of [`MemoryContent`] items, grouping by modality
    /// to minimise the number of underlying embed calls.
    ///
    /// Items that map to the same embedder (text, image, code) are batched
    /// into a single `embed()` call. Composite items are handled individually
    /// because they require per-part averaging.
    pub async fn embed_contents(&self, contents: &[&MemoryContent]) -> HirnResult<Vec<Embedding>> {
        // Classify each item into a modality bucket.
        #[derive(Clone, Copy)]
        enum Modality {
            Text,
            Image,
            Audio,
            Video,
            Code,
            Document,
            Composite,
        }

        let mut classifications: Vec<(Modality, String)> = Vec::with_capacity(contents.len());
        let mut composite_indices: Vec<usize> = Vec::new();

        for (i, content) in contents.iter().enumerate() {
            match content {
                MemoryContent::Image { description, .. } => {
                    classifications.push((Modality::Image, description.clone()));
                }
                MemoryContent::Code { source, .. } => {
                    classifications.push((Modality::Code, source.clone()));
                }
                MemoryContent::Audio { transcript, .. } => {
                    classifications.push((Modality::Audio, transcript.clone()));
                }
                MemoryContent::Video { .. } => {
                    classifications
                        .push((Modality::Video, content.text_for_embedding().into_owned()));
                }
                MemoryContent::Document { extracted_text, .. } => {
                    classifications.push((Modality::Document, extracted_text.clone()));
                }
                MemoryContent::External { .. } => {
                    classifications
                        .push((Modality::Text, content.text_for_embedding().into_owned()));
                }
                MemoryContent::ToolOutput { .. } => {
                    classifications
                        .push((Modality::Text, content.text_for_embedding().into_owned()));
                }
                MemoryContent::Text(t) => {
                    classifications.push((Modality::Text, t.clone()));
                }
                MemoryContent::Structured { data, .. } => {
                    classifications.push((Modality::Text, data.to_string()));
                }
                MemoryContent::Composite(_) => {
                    classifications.push((Modality::Composite, String::new()));
                    composite_indices.push(i);
                }
            }
        }

        // Collect indices per batch-embedder.
        let mut text_batch: Vec<(usize, String)> = Vec::new();
        let mut image_batch: Vec<(usize, String)> = Vec::new();
        let mut audio_batch: Vec<(usize, String)> = Vec::new();
        let mut video_batch: Vec<(usize, String)> = Vec::new();
        let mut code_batch: Vec<(usize, String)> = Vec::new();
        let mut document_batch: Vec<(usize, String)> = Vec::new();

        for (i, (modality, text)) in classifications.into_iter().enumerate() {
            match modality {
                Modality::Text => text_batch.push((i, text)),
                Modality::Image => image_batch.push((i, text)),
                Modality::Audio => audio_batch.push((i, text)),
                Modality::Video => video_batch.push((i, text)),
                Modality::Code => code_batch.push((i, text)),
                Modality::Document => document_batch.push((i, text)),
                Modality::Composite => {} // handled separately
            }
        }

        let mut results: Vec<Option<Embedding>> = vec![None; contents.len()];

        // Batch-embed each modality group.
        if !text_batch.is_empty() {
            let refs: Vec<&str> = text_batch.iter().map(|(_, s)| s.as_str()).collect();
            let embeddings = self.text_embedder.embed(&refs).await?;
            assign_batch_embeddings(
                &mut results,
                text_batch,
                embeddings,
                self.text_embedder.model_id(),
            )?;
        }

        if !image_batch.is_empty() {
            let embedder = self.image_embedder.as_ref().unwrap_or(&self.text_embedder);
            let refs: Vec<&str> = image_batch.iter().map(|(_, s)| s.as_str()).collect();
            let embeddings = embedder.embed(&refs).await?;
            assign_batch_embeddings(&mut results, image_batch, embeddings, embedder.model_id())?;
        }

        if !audio_batch.is_empty() {
            let embedder = self.audio_embedder.as_ref().unwrap_or(&self.text_embedder);
            let refs: Vec<&str> = audio_batch.iter().map(|(_, s)| s.as_str()).collect();
            let embeddings = embedder.embed(&refs).await?;
            assign_batch_embeddings(&mut results, audio_batch, embeddings, embedder.model_id())?;
        }

        if !video_batch.is_empty() {
            let embedder = self.video_embedder.as_ref().unwrap_or(&self.text_embedder);
            let refs: Vec<&str> = video_batch.iter().map(|(_, s)| s.as_str()).collect();
            let embeddings = embedder.embed(&refs).await?;
            assign_batch_embeddings(&mut results, video_batch, embeddings, embedder.model_id())?;
        }

        if !code_batch.is_empty() {
            let embedder = self.code_embedder.as_ref().unwrap_or(&self.text_embedder);
            let refs: Vec<&str> = code_batch.iter().map(|(_, s)| s.as_str()).collect();
            let embeddings = embedder.embed(&refs).await?;
            assign_batch_embeddings(&mut results, code_batch, embeddings, embedder.model_id())?;
        }

        if !document_batch.is_empty() {
            let embedder = self
                .document_embedder
                .as_ref()
                .unwrap_or(&self.text_embedder);
            let refs: Vec<&str> = document_batch.iter().map(|(_, s)| s.as_str()).collect();
            let embeddings = embedder.embed(&refs).await?;
            assign_batch_embeddings(
                &mut results,
                document_batch,
                embeddings,
                embedder.model_id(),
            )?;
        }

        // Handle composite items individually (each requires averaging).
        for idx in composite_indices {
            if let MemoryContent::Composite(parts) = contents[idx] {
                results[idx] = Some(self.embed_composite(parts).await?);
            }
        }

        results
            .into_iter()
            .enumerate()
            .map(|(idx, embedding)| {
                embedding.ok_or_else(|| {
                    HirnError::ProviderError(format!(
                        "multimodal embedding missing result for content index {idx}"
                    ))
                })
            })
            .collect()
    }

    /// Embed composite content as one weighted, normalized aggregate bundle.
    ///
    /// The current policy is explicit so callers can reason about how mixed
    /// modalities influence the stored representation.
    async fn embed_composite(&self, parts: &[MemoryContent]) -> HirnResult<Embedding> {
        if parts.is_empty() {
            return embed_one(self.text_embedder.as_ref(), "").await;
        }

        // Extract text representation from each part and embed via the
        // appropriate per-modality embedder. For nested composites,
        // flatten to text_for_embedding to avoid infinite recursion.
        let mut embeddings = Vec::with_capacity(parts.len());
        for part in parts {
            let emb = match part {
                MemoryContent::Image { description, .. } => {
                    let embedder = self.image_embedder.as_ref().unwrap_or(&self.text_embedder);
                    embed_one(embedder.as_ref(), description).await?
                }
                MemoryContent::Code { source, .. } => {
                    let embedder = self.code_embedder.as_ref().unwrap_or(&self.text_embedder);
                    embed_one(embedder.as_ref(), source).await?
                }
                MemoryContent::Audio { transcript, .. } => {
                    let embedder = self.audio_embedder.as_ref().unwrap_or(&self.text_embedder);
                    embed_one(embedder.as_ref(), transcript).await?
                }
                MemoryContent::Video { .. } => {
                    let embedder = self.video_embedder.as_ref().unwrap_or(&self.text_embedder);
                    let text = part.text_for_embedding();
                    embed_one(embedder.as_ref(), text.as_ref()).await?
                }
                MemoryContent::Document { extracted_text, .. } => {
                    let embedder = self
                        .document_embedder
                        .as_ref()
                        .unwrap_or(&self.text_embedder);
                    embed_one(embedder.as_ref(), extracted_text).await?
                }
                MemoryContent::External { .. } => {
                    let text = part.text_for_embedding();
                    embed_one(self.text_embedder.as_ref(), text.as_ref()).await?
                }
                MemoryContent::ToolOutput { .. } => {
                    let text = part.text_for_embedding();
                    embed_one(self.text_embedder.as_ref(), text.as_ref()).await?
                }
                other => {
                    let text = other.text_for_embedding();
                    embed_one(self.text_embedder.as_ref(), &text).await?
                }
            };
            embeddings.push(emb);
        }

        // Collapse part embeddings according to the configured composite policy.
        let dims = embeddings[0].vector.len();
        let mut avg = vec![0.0f32; dims];
        let mut total_weight = 0.0f32;
        for (part, emb) in parts.iter().zip(&embeddings) {
            if emb.vector.len() != dims {
                return Err(HirnError::InvalidInput(format!(
                    "composite part embedding dimension mismatch: expected {dims}, got {}",
                    emb.vector.len()
                )));
            }

            let weight = self.composite_policy.weight_for(part);
            if weight <= 0.0 {
                continue;
            }

            for (i, v) in emb.vector.iter().enumerate() {
                avg[i] += v * weight;
            }
            total_weight += weight;
        }

        if total_weight <= 0.0 {
            return embed_one(self.text_embedder.as_ref(), "").await;
        }

        for v in &mut avg {
            *v /= total_weight;
        }

        // L2-normalize the averaged vector.
        let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut avg {
                *v /= norm;
            }
        }

        Ok(Embedding {
            vector: avg,
            model_id: embeddings[0].model_id.clone(),
        })
    }

    /// Return the embedder that would be selected for a given modality.
    pub fn embedder_for_modality(&self, modality: &str) -> &dyn Embedder {
        match modality {
            "image" => self
                .image_embedder
                .as_deref()
                .unwrap_or_else(|| self.text_embedder.as_ref()),
            "audio" => self
                .audio_embedder
                .as_deref()
                .unwrap_or_else(|| self.text_embedder.as_ref()),
            "video" => self
                .video_embedder
                .as_deref()
                .unwrap_or_else(|| self.text_embedder.as_ref()),
            "code" => self
                .code_embedder
                .as_deref()
                .unwrap_or_else(|| self.text_embedder.as_ref()),
            "document" => self
                .document_embedder
                .as_deref()
                .unwrap_or_else(|| self.text_embedder.as_ref()),
            _ => self.text_embedder.as_ref(),
        }
    }
}

async fn embed_one(embedder: &dyn Embedder, text: &str) -> HirnResult<Embedding> {
    let results = embedder.embed(&[text]).await?;
    let count = results.len();
    if count != 1 {
        return Err(HirnError::ProviderError(format!(
            "embedder `{}` returned {count} embeddings for one input",
            embedder.model_id()
        )));
    }
    let mut results = results;
    Ok(results.remove(0))
}

fn assign_batch_embeddings(
    results: &mut [Option<Embedding>],
    batch: Vec<(usize, String)>,
    embeddings: Vec<Embedding>,
    model_id: &str,
) -> HirnResult<()> {
    if embeddings.len() != batch.len() {
        return Err(HirnError::ProviderError(format!(
            "embedder `{model_id}` returned {} embeddings for {} inputs",
            embeddings.len(),
            batch.len()
        )));
    }

    for ((idx, _), embedding) in batch.into_iter().zip(embeddings) {
        results[idx] = Some(embedding);
    }
    Ok(())
}

#[async_trait]
impl Embedder for MultiModalEmbedder {
    async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
        self.text_embedder.embed(texts).await
    }

    fn dimensions(&self) -> usize {
        self.text_embedder.dimensions()
    }

    fn model_id(&self) -> &str {
        self.text_embedder.model_id()
    }

    fn max_input_tokens(&self) -> usize {
        self.text_embedder.max_input_tokens()
    }

    async fn embed_multivec(&self, texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
        self.text_embedder.embed_multivec(texts).await
    }

    fn supports_multivec(&self) -> bool {
        self.text_embedder.supports_multivec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PseudoEmbedder;
    use hirn_core::content::CompositeModalityWeights;

    struct FixedVectorEmbedder {
        model_id: &'static str,
        vector: Vec<f32>,
    }

    struct LengthEmbedder {
        model_id: &'static str,
    }

    #[async_trait]
    impl Embedder for FixedVectorEmbedder {
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

    #[async_trait]
    impl Embedder for LengthEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: vec![text.len() as f32],
                    model_id: self.model_id.to_string(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            1
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

    fn pseudo() -> Arc<dyn Embedder> {
        Arc::new(PseudoEmbedder::new(64))
    }

    #[tokio::test]
    async fn routes_text_to_text_embedder() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Text("hello world".into());
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
        assert_eq!(result.model_id, "pseudo-3gram-hash");
    }

    #[tokio::test]
    async fn routes_image_to_text_embedder_by_default() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Image {
            data: vec![1, 2, 3],
            mime_type: "image/png".into(),
            description: "login page screenshot".into(),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn routes_code_to_text_embedder_by_default() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Code {
            source: "fn sort(arr: &mut [i32]) { arr.sort(); }".into(),
            language: "rust".into(),
            ast_hash: None,
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn routes_audio_to_text_embedder() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Audio {
            data: vec![0xFF, 0xFB],
            transcript: "meeting about auth".into(),
            duration_ms: 5000,
            channel_count: Some(1),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn routes_video_to_text_embedder() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Video {
            data: vec![0x00, 0x00, 0x00, 0x20],
            mime_type: "video/mp4".into(),
            transcript: "launch keynote transcript".into(),
            description: "stage demo with dashboard overlays".into(),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn routes_document_to_text_embedder() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Document {
            data: b"%PDF-1.4 fake".to_vec(),
            mime_type: "application/pdf".into(),
            extracted_text: "design review packet".into(),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn routes_external_to_text_embedder() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::External {
            uri: "https://example.com/run/42".into(),
            title: "deployment log".into(),
            snippet: "green rollout completed".into(),
            mime_type: Some("text/html".into()),
            checksum: None,
            fetch_policy: hirn_core::content::ExternalFetchPolicy::OnDemand,
            stale_at: None,
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn routes_tool_output_to_text_embedder() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::ToolOutput {
            tool_name: "terraform".into(),
            output: r#"{"applied":true}"#.into(),
            mime_type: Some("application/json".into()),
            schema: Some("terraform/apply.v1".into()),
            invocation_id: Some("apply-42".into()),
            checksum: None,
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn routes_structured_to_text_embedder() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Structured {
            schema: "event/v1".into(),
            data: serde_json::json!({"action": "login", "user": "alice"}),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn embeds_composite_as_normalized_weighted_mean() {
        let mm = MultiModalEmbedder::new(pseudo());
        let content = MemoryContent::Composite(vec![
            MemoryContent::Text("caption".into()),
            MemoryContent::Image {
                data: vec![1],
                mime_type: "image/jpeg".into(),
                description: "photo".into(),
            },
        ]);
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
        // Verify it's normalized.
        let norm: f32 = result.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn composite_policy_weights_modalities() {
        let mm = MultiModalEmbedder::new(Arc::new(FixedVectorEmbedder {
            model_id: "text",
            vector: vec![1.0, 0.0, 0.0, 0.0],
        }))
        .with_image_embedder(Arc::new(FixedVectorEmbedder {
            model_id: "image",
            vector: vec![0.0, 1.0, 0.0, 0.0],
        }))
        .with_audio_embedder(Arc::new(FixedVectorEmbedder {
            model_id: "audio",
            vector: vec![0.0, 0.0, 1.0, 0.0],
        }))
        .with_video_embedder(Arc::new(FixedVectorEmbedder {
            model_id: "video",
            vector: vec![0.0, 0.0, 0.0, 1.0],
        }))
        .with_composite_policy(CompositeEmbeddingPolicy::WeightedMeanNormalized(
            CompositeModalityWeights {
                text: 2.0,
                image: 1.0,
                audio: 1.0,
                video: 1.0,
                ..Default::default()
            },
        ));

        let content = MemoryContent::Composite(vec![
            MemoryContent::Text("summary".into()),
            MemoryContent::Image {
                data: vec![1],
                mime_type: "image/png".into(),
                description: "diagram".into(),
            },
            MemoryContent::Audio {
                data: vec![2],
                transcript: "meeting".into(),
                duration_ms: 100,
                channel_count: None,
            },
            MemoryContent::Video {
                data: vec![3],
                mime_type: "video/mp4".into(),
                transcript: String::new(),
                description: "walkthrough".into(),
            },
        ]);

        let result = mm.embed_content(&content).await.unwrap();
        assert_vector_close(&result.vector, &[0.755929, 0.3779645, 0.3779645, 0.3779645]);
    }

    #[tokio::test]
    async fn specialized_image_embedder_is_used() {
        let text_emb = Arc::new(PseudoEmbedder::new(64));
        let image_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(64));
        let mm = MultiModalEmbedder::new(text_emb).with_image_embedder(image_emb);

        let content = MemoryContent::Image {
            data: vec![1, 2, 3],
            mime_type: "image/png".into(),
            description: "test image".into(),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn specialized_code_embedder_is_used() {
        let text_emb = Arc::new(PseudoEmbedder::new(64));
        let code_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(64));
        let mm = MultiModalEmbedder::new(text_emb).with_code_embedder(code_emb);

        let content = MemoryContent::Code {
            source: "def sort(a): a.sort()".into(),
            language: "python".into(),
            ast_hash: None,
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 64);
    }

    #[tokio::test]
    async fn specialized_audio_embedder_is_used() {
        let text_emb = Arc::new(PseudoEmbedder::new(64));
        let audio_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(32));
        let mm = MultiModalEmbedder::new(text_emb).with_audio_embedder(audio_emb);

        let content = MemoryContent::Audio {
            data: vec![0xFF, 0xFB],
            transcript: "meeting transcript".into(),
            duration_ms: 5000,
            channel_count: Some(2),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 32);
    }

    #[tokio::test]
    async fn specialized_video_embedder_is_used() {
        let text_emb = Arc::new(PseudoEmbedder::new(64));
        let video_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(40));
        let mm = MultiModalEmbedder::new(text_emb).with_video_embedder(video_emb);

        let content = MemoryContent::Video {
            data: vec![0x00, 0x00, 0x00, 0x20],
            mime_type: "video/mp4".into(),
            transcript: "screen capture".into(),
            description: "deployment walkthrough".into(),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 40);
    }

    #[tokio::test]
    async fn specialized_document_embedder_is_used() {
        let text_emb = Arc::new(PseudoEmbedder::new(64));
        let document_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(48));
        let mm = MultiModalEmbedder::new(text_emb).with_document_embedder(document_emb);

        let content = MemoryContent::Document {
            data: b"%PDF-1.4".to_vec(),
            mime_type: "application/pdf".into(),
            extracted_text: "design review packet".into(),
        };
        let result = mm.embed_content(&content).await.unwrap();
        assert_eq!(result.vector.len(), 48);
    }

    #[tokio::test]
    async fn modality_specific_embedders_receive_textual_surrogates() {
        let mm = MultiModalEmbedder::new(Arc::new(LengthEmbedder { model_id: "text" }))
            .with_image_embedder(Arc::new(LengthEmbedder { model_id: "image" }))
            .with_audio_embedder(Arc::new(LengthEmbedder { model_id: "audio" }))
            .with_video_embedder(Arc::new(LengthEmbedder { model_id: "video" }))
            .with_code_embedder(Arc::new(LengthEmbedder { model_id: "code" }))
            .with_document_embedder(Arc::new(LengthEmbedder {
                model_id: "document",
            }));

        let image = MemoryContent::Image {
            data: vec![1, 2, 3],
            mime_type: "image/png".into(),
            description: "orbit map".into(),
        };
        let audio = MemoryContent::Audio {
            data: vec![0xFF, 0xFB],
            transcript: "launch audio".into(),
            duration_ms: 5000,
            channel_count: Some(1),
        };
        let code = MemoryContent::Code {
            source: "fn deploy() {}".into(),
            language: "rust".into(),
            ast_hash: None,
        };
        let video = MemoryContent::Video {
            data: vec![1, 2, 3, 4],
            mime_type: "video/mp4".into(),
            transcript: "launch demo".into(),
            description: "camera pan".into(),
        };
        let document = MemoryContent::Document {
            data: b"%PDF-1.4".to_vec(),
            mime_type: "application/pdf".into(),
            extracted_text: "incident runbook".into(),
        };
        let external = MemoryContent::External {
            uri: "https://example.com/runbook".into(),
            title: "runbook".into(),
            snippet: "incident runbook".into(),
            mime_type: Some("text/html".into()),
            checksum: None,
            fetch_policy: hirn_core::content::ExternalFetchPolicy::OnDemand,
            stale_at: None,
        };
        let tool_output = MemoryContent::ToolOutput {
            tool_name: "kubectl".into(),
            output: r#"{"ready":true}"#.into(),
            mime_type: Some("application/json".into()),
            schema: Some("cluster/status.v1".into()),
            invocation_id: None,
            checksum: None,
        };

        assert_eq!(mm.embed_content(&image).await.unwrap().vector, vec![9.0]);
        assert_eq!(mm.embed_content(&audio).await.unwrap().vector, vec![12.0]);
        assert_eq!(mm.embed_content(&code).await.unwrap().vector, vec![14.0]);
        assert_eq!(mm.embed_content(&video).await.unwrap().vector, vec![22.0]);
        assert_eq!(
            mm.embed_content(&document).await.unwrap().vector,
            vec![16.0]
        );
        assert_eq!(
            mm.embed_content(&external).await.unwrap().vector,
            vec![external.text_for_embedding().len() as f32]
        );
        assert_eq!(
            mm.embed_content(&tool_output).await.unwrap().vector,
            vec![tool_output.text_for_embedding().len() as f32]
        );
    }

    #[tokio::test]
    async fn malformed_embedder_result_is_error_not_panic() {
        struct EmptyEmbedder;

        #[async_trait]
        impl Embedder for EmptyEmbedder {
            async fn embed(&self, _: &[&str]) -> HirnResult<Vec<Embedding>> {
                Ok(Vec::new())
            }

            fn dimensions(&self) -> usize {
                1
            }

            fn model_id(&self) -> &str {
                "empty"
            }

            fn max_input_tokens(&self) -> usize {
                8192
            }
        }

        let mm = MultiModalEmbedder::new(Arc::new(EmptyEmbedder));
        let err = mm
            .embed_content(&MemoryContent::Text("hello".into()))
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("returned 0 embeddings for one input")
        );
    }

    #[tokio::test]
    async fn embed_contents_batch() {
        let mm = MultiModalEmbedder::new(pseudo());
        let text = MemoryContent::Text("hello".into());
        let code = MemoryContent::Code {
            source: "x = 1".into(),
            language: "python".into(),
            ast_hash: None,
        };
        let results = mm.embed_contents(&[&text, &code]).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].vector.len(), 64);
        assert_eq!(results[1].vector.len(), 64);
    }

    #[tokio::test]
    async fn embed_contents_mixed_modalities_preserves_order() {
        let mm = MultiModalEmbedder::new(pseudo());
        let items = [
            MemoryContent::Text("first".into()),
            MemoryContent::Image {
                data: vec![1],
                mime_type: "image/png".into(),
                description: "second".into(),
            },
            MemoryContent::Code {
                source: "third".into(),
                language: "rust".into(),
                ast_hash: None,
            },
            MemoryContent::Audio {
                data: vec![],
                transcript: "fourth".into(),
                duration_ms: 100,
                channel_count: None,
            },
            MemoryContent::Video {
                data: vec![],
                mime_type: "video/mp4".into(),
                transcript: "fifth-a".into(),
                description: "fifth-b".into(),
            },
            MemoryContent::Document {
                data: b"%PDF-1.4".to_vec(),
                mime_type: "application/pdf".into(),
                extracted_text: "sixth".into(),
            },
            MemoryContent::External {
                uri: "https://example.com/seventh".into(),
                title: "seventh".into(),
                snippet: "external summary".into(),
                mime_type: Some("text/html".into()),
                checksum: None,
                fetch_policy: hirn_core::content::ExternalFetchPolicy::OnDemand,
                stale_at: None,
            },
            MemoryContent::ToolOutput {
                tool_name: "tool".into(),
                output: "eighth".into(),
                mime_type: Some("text/plain".into()),
                schema: None,
                invocation_id: None,
                checksum: None,
            },
            MemoryContent::Composite(vec![
                MemoryContent::Text("ninth-a".into()),
                MemoryContent::Text("ninth-b".into()),
            ]),
        ];
        let refs: Vec<&MemoryContent> = items.iter().collect();
        let results = mm.embed_contents(&refs).await.unwrap();
        assert_eq!(results.len(), 9);
        // Each embedding produced the correct dimensions.
        for (i, emb) in results.iter().enumerate() {
            assert_eq!(emb.vector.len(), 64, "item {i} has wrong dimension");
        }
        // Composite should be normalised.
        let norm: f32 = results[8].vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "composite should be normalized");
    }

    #[tokio::test]
    async fn embedder_for_modality_routing() {
        let text_emb = Arc::new(PseudoEmbedder::new(64));
        let image_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(128));
        let audio_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(96));
        let video_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(88));
        let document_emb: Arc<dyn Embedder> = Arc::new(PseudoEmbedder::new(80));
        let mm = MultiModalEmbedder::new(text_emb)
            .with_image_embedder(image_emb)
            .with_audio_embedder(audio_emb)
            .with_video_embedder(video_emb)
            .with_document_embedder(document_emb);

        assert_eq!(mm.embedder_for_modality("text").dimensions(), 64);
        assert_eq!(mm.embedder_for_modality("image").dimensions(), 128);
        assert_eq!(mm.embedder_for_modality("code").dimensions(), 64);
        assert_eq!(mm.embedder_for_modality("audio").dimensions(), 96);
        assert_eq!(mm.embedder_for_modality("video").dimensions(), 88);
        assert_eq!(mm.embedder_for_modality("document").dimensions(), 80);
        assert_eq!(mm.embedder_for_modality("external").dimensions(), 64);
    }

    #[tokio::test]
    async fn implements_embedder_trait() {
        let mm = MultiModalEmbedder::new(pseudo());
        assert_eq!(mm.dimensions(), 64);
        assert_eq!(mm.model_id(), "pseudo-3gram-hash");
        let results = mm.embed(&["test text"]).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn implements_embedder_multivec_passthrough() {
        let mm = MultiModalEmbedder::new(pseudo());
        assert!(mm.supports_multivec());
        let results = mm.embed_multivec(&["test text"]).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].vectors.is_empty());
    }
}
