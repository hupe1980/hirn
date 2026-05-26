use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::resource::ModalityProfile;
use crate::timestamp::Timestamp;

/// Refresh strategy for external references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExternalFetchPolicy {
    /// Fetch only when a caller explicitly requests hydration or refresh.
    #[default]
    OnDemand,
    /// Refresh only after the reference is marked stale.
    IfStale,
    /// Treat the reference as immutable metadata.
    Never,
}

impl ExternalFetchPolicy {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OnDemand => "on_demand",
            Self::IfStale => "if_stale",
            Self::Never => "never",
        }
    }

    pub fn parse(value: &str) -> Result<Self, crate::HirnError> {
        match value {
            "on_demand" => Ok(Self::OnDemand),
            "if_stale" => Ok(Self::IfStale),
            "never" => Ok(Self::Never),
            _ => Err(crate::HirnError::InvalidInput(format!(
                "unknown external fetch policy: {value}"
            ))),
        }
    }
}

/// Multi-modal memory content.
///
/// Each variant represents a different modality of information that can be
/// stored in a memory record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryContent {
    /// Plain text content (the default, backward-compatible variant).
    Text(String),

    /// Image binary data with metadata.
    Image {
        /// Raw image bytes (PNG, JPEG, etc.).
        data: Vec<u8>,
        /// MIME type, e.g. `"image/png"`, `"image/jpeg"`.
        mime_type: String,
        /// Human-readable description of the image.
        description: String,
    },

    /// Source code with language metadata.
    Code {
        /// The source code text.
        source: String,
        /// Programming language identifier, e.g. `"rust"`, `"python"`.
        language: String,
        /// Optional AST hash for structural deduplication.
        ast_hash: Option<String>,
    },

    /// Audio binary data with transcript.
    Audio {
        /// Raw audio bytes.
        data: Vec<u8>,
        /// Transcription of the audio content.
        transcript: String,
        /// Duration in milliseconds.
        duration_ms: u64,
        /// Number of channels when known.
        #[serde(default)]
        channel_count: Option<u16>,
    },

    /// Video binary data with transcript and visual description surrogates.
    Video {
        /// Raw video bytes.
        data: Vec<u8>,
        /// MIME type, e.g. `"video/mp4"`.
        mime_type: String,
        /// Speech or subtitle transcript extracted from the video when available.
        transcript: String,
        /// Frame-level or scene-level summary used when transcript is missing or incomplete.
        description: String,
    },

    /// Document binary data with extracted text.
    Document {
        /// Raw document bytes.
        data: Vec<u8>,
        /// MIME type, e.g. `"application/pdf"`.
        mime_type: String,
        /// Extracted text or document surrogate used for embedding/preview.
        extracted_text: String,
    },

    /// External resource reference with cached text surrogates.
    External {
        /// Canonical remote or local URI.
        uri: String,
        /// Human-readable title used in previews and recall.
        title: String,
        /// Short snippet or summary captured at ingest time.
        snippet: String,
        /// Optional MIME type when known without fetching the payload.
        #[serde(default)]
        mime_type: Option<String>,
        /// Optional strong checksum supplied by the caller.
        #[serde(default)]
        checksum: Option<String>,
        /// Refresh policy for external hydration owners.
        #[serde(default)]
        fetch_policy: ExternalFetchPolicy,
        /// Timestamp after which the reference should be considered stale.
        #[serde(default)]
        stale_at: Option<Timestamp>,
    },

    /// Resource-backed output emitted by a tool or procedure step.
    ToolOutput {
        /// Tool or function name that produced the output.
        tool_name: String,
        /// Raw textual payload captured from the tool.
        output: String,
        /// Optional MIME type for the serialized payload.
        #[serde(default)]
        mime_type: Option<String>,
        /// Optional schema hint for structured outputs.
        #[serde(default)]
        schema: Option<String>,
        /// Optional tool invocation or call identifier.
        #[serde(default)]
        invocation_id: Option<String>,
        /// Optional strong checksum supplied by the caller.
        #[serde(default)]
        checksum: Option<String>,
    },

    /// Structured data (JSON) with a schema identifier.
    Structured {
        /// Schema name or URI describing the shape of the data.
        schema: String,
        /// The structured data itself.
        data: serde_json::Value,
    },

    /// Composite content combining multiple modalities.
    Composite(Vec<Self>),
}

/// Per-modality weights used when collapsing composite content into one
/// aggregate embedding bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompositeModalityWeights {
    pub text: f32,
    pub image: f32,
    pub audio: f32,
    pub video: f32,
    pub code: f32,
    pub document: f32,
    pub external: f32,
    pub structured: f32,
}

impl Default for CompositeModalityWeights {
    fn default() -> Self {
        Self {
            text: 1.0,
            image: 1.0,
            audio: 1.0,
            video: 1.0,
            code: 1.0,
            document: 1.0,
            external: 1.0,
            structured: 1.0,
        }
    }
}

impl CompositeModalityWeights {
    #[must_use]
    pub fn weight_for(&self, content: &MemoryContent) -> f32 {
        match content {
            MemoryContent::Text(_) => self.text,
            MemoryContent::Image { .. } => self.image,
            MemoryContent::Audio { .. } => self.audio,
            MemoryContent::Video { .. } => self.video,
            MemoryContent::Code { .. } => self.code,
            MemoryContent::Document { .. } => self.document,
            MemoryContent::External { .. } => self.external,
            MemoryContent::ToolOutput {
                mime_type, schema, ..
            } => {
                if tool_output_modality_profile(mime_type.as_deref(), schema.as_deref())
                    == ModalityProfile::Structured
                {
                    self.structured
                } else {
                    self.text
                }
            }
            MemoryContent::Structured { .. } => self.structured,
            MemoryContent::Composite(_) => self.structured,
        }
    }
}

fn video_surrogate_text<'a>(transcript: &'a str, description: &'a str) -> Cow<'a, str> {
    match (transcript.trim().is_empty(), description.trim().is_empty()) {
        (false, true) => Cow::Borrowed(transcript),
        (true, false) => Cow::Borrowed(description),
        (true, true) => Cow::Borrowed(""),
        (false, false) => Cow::Owned(format!("{transcript}\n{description}")),
    }
}

fn external_surrogate_text<'a>(title: &'a str, snippet: &'a str, uri: &'a str) -> Cow<'a, str> {
    let title = title.trim();
    let snippet = snippet.trim();
    let uri = uri.trim();

    match (title.is_empty(), snippet.is_empty(), uri.is_empty()) {
        (false, true, true) => Cow::Borrowed(title),
        (true, false, true) => Cow::Borrowed(snippet),
        (true, true, false) => Cow::Borrowed(uri),
        (false, false, true) => Cow::Owned(format!("{title}\n{snippet}")),
        (false, true, false) => Cow::Owned(format!("{title}\n{uri}")),
        (true, false, false) => Cow::Owned(format!("{snippet}\n{uri}")),
        (false, false, false) => Cow::Owned(format!("{title}\n{snippet}\n{uri}")),
        (true, true, true) => Cow::Borrowed(""),
    }
}

fn tool_output_surrogate_text<'a>(tool_name: &'a str, output: &'a str) -> Cow<'a, str> {
    let tool_name = tool_name.trim();
    let output = output.trim();

    match (tool_name.is_empty(), output.is_empty()) {
        (false, true) => Cow::Borrowed(tool_name),
        (true, false) => Cow::Borrowed(output),
        (false, false) => Cow::Owned(format!("{tool_name}\n{output}")),
        (true, true) => Cow::Borrowed(""),
    }
}

fn tool_output_modality_profile(mime_type: Option<&str>, schema: Option<&str>) -> ModalityProfile {
    if schema.is_some_and(|value| !value.trim().is_empty())
        || mime_type.is_some_and(is_structured_tool_output_mime)
    {
        ModalityProfile::Structured
    } else {
        ModalityProfile::Text
    }
}

fn is_structured_tool_output_mime(mime_type: &str) -> bool {
    let mime_type = mime_type.trim();
    mime_type.eq_ignore_ascii_case("application/json")
        || mime_type.eq_ignore_ascii_case("text/json")
        || mime_type.ends_with("+json")
}

/// Policy describing how composite content should collapse into one stored
/// embedding bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompositeEmbeddingPolicy {
    /// Weighted mean across part embeddings, followed by L2 normalization.
    WeightedMeanNormalized(CompositeModalityWeights),
}

impl Default for CompositeEmbeddingPolicy {
    fn default() -> Self {
        Self::WeightedMeanNormalized(CompositeModalityWeights::default())
    }
}

impl CompositeEmbeddingPolicy {
    #[must_use]
    pub fn weight_for(&self, content: &MemoryContent) -> f32 {
        match self {
            Self::WeightedMeanNormalized(weights) => weights.weight_for(content).max(0.0),
        }
    }
}

impl MemoryContent {
    /// Returns the modality name as a static string.
    #[must_use]
    pub const fn modality(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Image { .. } => "image",
            Self::Code { .. } => "code",
            Self::Audio { .. } => "audio",
            Self::Video { .. } => "video",
            Self::Document { .. } => "document",
            Self::External { .. } => "external",
            Self::ToolOutput { .. } => "tool_output",
            Self::Structured { .. } => "structured",
            Self::Composite(_) => "composite",
        }
    }

    /// Returns the typed modality profile for this content.
    #[must_use]
    pub fn modality_profile(&self) -> ModalityProfile {
        match self {
            Self::Text(_) => ModalityProfile::Text,
            Self::Image { .. } => ModalityProfile::Image,
            Self::Code { .. } => ModalityProfile::Code,
            Self::Audio { .. } => ModalityProfile::Audio,
            Self::Video { .. } => ModalityProfile::Video,
            Self::Document { .. } => ModalityProfile::Document,
            Self::External { .. } => ModalityProfile::External,
            Self::ToolOutput {
                mime_type, schema, ..
            } => tool_output_modality_profile(mime_type.as_deref(), schema.as_deref()),
            Self::Structured { .. } => ModalityProfile::Structured,
            Self::Composite(_) => ModalityProfile::Composite,
        }
    }

    /// Returns the text representation used for embedding.
    ///
    /// For non-text modalities, this extracts the text-like component
    /// (description, transcript, source code, JSON serialization).
    /// Returns a borrowed reference when possible (Text, Image, Code, Audio)
    /// to avoid unnecessary allocations.
    #[must_use]
    pub fn text_for_embedding(&self) -> Cow<'_, str> {
        match self {
            Self::Text(t) => Cow::Borrowed(t),
            Self::Image { description, .. } => Cow::Borrowed(description),
            Self::Code { source, .. } => Cow::Borrowed(source),
            Self::Audio { transcript, .. } => Cow::Borrowed(transcript),
            Self::Video {
                transcript,
                description,
                ..
            } => video_surrogate_text(transcript, description),
            Self::Document { extracted_text, .. } => Cow::Borrowed(extracted_text),
            Self::External {
                uri,
                title,
                snippet,
                ..
            } => external_surrogate_text(title, snippet, uri),
            Self::ToolOutput {
                tool_name, output, ..
            } => tool_output_surrogate_text(tool_name, output),
            Self::Structured { data, .. } => Cow::Owned(data.to_string()),
            Self::Composite(parts) => Cow::Owned(
                parts
                    .iter()
                    .map(|p| p.text_for_embedding())
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
        }
    }

    /// Returns the size in bytes of the binary data, if any.
    #[must_use]
    #[allow(clippy::match_same_arms)]
    pub fn blob_size(&self) -> usize {
        match self {
            Self::Text(t) => t.len(),
            Self::Image { data, .. } => data.len(),
            Self::Code { source, .. } => source.len(),
            Self::Audio { data, .. } => data.len(),
            Self::Video { data, .. } => data.len(),
            Self::Document { data, .. } => data.len(),
            Self::External { .. } => 0,
            Self::ToolOutput { output, .. } => output.len(),
            Self::Structured { data, .. } => data.to_string().len(),
            Self::Composite(parts) => parts.iter().map(Self::blob_size).sum(),
        }
    }

    /// Returns `true` if this is a `Text` variant.
    #[must_use]
    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Extracts the text content if this is a `Text` variant.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(t),
            _ => None,
        }
    }
}

impl From<String> for MemoryContent {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for MemoryContent {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

impl std::fmt::Display for MemoryContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text(t) => write!(f, "{t}"),
            Self::Image {
                description,
                mime_type,
                ..
            } => {
                write!(f, "[image/{mime_type}] {description}")
            }
            Self::Code {
                language, source, ..
            } => {
                write!(
                    f,
                    "[code/{language}] {}",
                    source.chars().take(80).collect::<String>()
                )
            }
            Self::Audio {
                transcript,
                duration_ms,
                channel_count,
                ..
            } => {
                if let Some(channel_count) = channel_count {
                    write!(f, "[audio/{duration_ms}ms/{channel_count}ch] {transcript}")
                } else {
                    write!(f, "[audio/{duration_ms}ms] {transcript}")
                }
            }
            Self::Video {
                mime_type,
                transcript,
                description,
                ..
            } => {
                let surrogate = video_surrogate_text(transcript, description);
                write!(
                    f,
                    "[video/{mime_type}] {}",
                    surrogate.chars().take(80).collect::<String>()
                )
            }
            Self::Document {
                mime_type,
                extracted_text,
                ..
            } => {
                write!(
                    f,
                    "[document/{mime_type}] {}",
                    extracted_text.chars().take(80).collect::<String>()
                )
            }
            Self::External {
                uri,
                title,
                snippet,
                mime_type,
                ..
            } => {
                let surrogate = external_surrogate_text(title, snippet, uri);
                if let Some(mime_type) = mime_type {
                    write!(
                        f,
                        "[external/{mime_type}] {}",
                        surrogate.chars().take(80).collect::<String>()
                    )
                } else {
                    write!(
                        f,
                        "[external] {}",
                        surrogate.chars().take(80).collect::<String>()
                    )
                }
            }
            Self::ToolOutput {
                tool_name,
                output,
                mime_type,
                ..
            } => {
                let surrogate = tool_output_surrogate_text(tool_name, output);
                if let Some(mime_type) = mime_type {
                    write!(
                        f,
                        "[tool_output/{mime_type}] {}",
                        surrogate.chars().take(80).collect::<String>()
                    )
                } else {
                    write!(
                        f,
                        "[tool_output] {}",
                        surrogate.chars().take(80).collect::<String>()
                    )
                }
            }
            Self::Structured { schema, data } => {
                write!(f, "[structured/{schema}] {data}")
            }
            Self::Composite(parts) => {
                write!(f, "[composite/{}]", parts.len())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_modality() {
        let c = MemoryContent::Text("hello".into());
        assert_eq!(c.modality(), "text");
        assert!(c.is_text());
        assert_eq!(c.as_text(), Some("hello"));
    }

    #[test]
    fn image_modality() {
        let c = MemoryContent::Image {
            data: vec![0x89, 0x50, 0x4E, 0x47],
            mime_type: "image/png".into(),
            description: "a test image".into(),
        };
        assert_eq!(c.modality(), "image");
        assert_eq!(c.modality_profile(), ModalityProfile::Image);
        assert!(!c.is_text());
        assert_eq!(c.text_for_embedding(), "a test image");
        assert_eq!(c.blob_size(), 4);
    }

    #[test]
    fn code_modality() {
        let c = MemoryContent::Code {
            source: "fn main() {}".into(),
            language: "rust".into(),
            ast_hash: Some("abc123".into()),
        };
        assert_eq!(c.modality(), "code");
        assert_eq!(c.text_for_embedding(), "fn main() {}");
    }

    #[test]
    fn audio_modality() {
        let c = MemoryContent::Audio {
            data: vec![0xFF, 0xFB],
            transcript: "hello world".into(),
            duration_ms: 5000,
            channel_count: Some(2),
        };
        assert_eq!(c.modality(), "audio");
        assert_eq!(c.text_for_embedding(), "hello world");
        assert_eq!(c.blob_size(), 2);
    }

    #[test]
    fn video_modality() {
        let c = MemoryContent::Video {
            data: vec![0x00, 0x00, 0x00, 0x18],
            mime_type: "video/mp4".into(),
            transcript: "quarterly update walkthrough".into(),
            description: "screen recording of dashboard".into(),
        };
        assert_eq!(c.modality(), "video");
        assert_eq!(c.modality_profile(), ModalityProfile::Video);
        assert_eq!(
            c.text_for_embedding(),
            "quarterly update walkthrough\nscreen recording of dashboard"
        );
        assert_eq!(c.blob_size(), 4);
    }

    #[test]
    fn structured_modality() {
        let c = MemoryContent::Structured {
            schema: "event/v1".into(),
            data: serde_json::json!({"key": "value"}),
        };
        assert_eq!(c.modality(), "structured");
        assert!(c.text_for_embedding().contains("key"));
    }

    #[test]
    fn document_modality() {
        let c = MemoryContent::Document {
            data: b"%PDF-1.4".to_vec(),
            mime_type: "application/pdf".into(),
            extracted_text: "architecture design review".into(),
        };
        assert_eq!(c.modality(), "document");
        assert_eq!(c.modality_profile(), ModalityProfile::Document);
        assert_eq!(c.text_for_embedding(), "architecture design review");
        assert_eq!(c.blob_size(), 8);
    }

    #[test]
    fn external_modality() {
        let c = MemoryContent::External {
            uri: "file:///tmp/report.pdf".into(),
            title: "incident report".into(),
            snippet: "rotated keys and closed incident".into(),
            mime_type: Some("application/pdf".into()),
            checksum: Some("sha256:abc".into()),
            fetch_policy: ExternalFetchPolicy::IfStale,
            stale_at: Some(Timestamp::from_millis(1_712_345_678_000)),
        };
        assert_eq!(c.modality(), "external");
        assert_eq!(c.modality_profile(), ModalityProfile::External);
        assert_eq!(
            c.text_for_embedding(),
            "incident report\nrotated keys and closed incident\nfile:///tmp/report.pdf"
        );
        assert_eq!(c.blob_size(), 0);
    }

    #[test]
    fn tool_output_modality_uses_payload_shape() {
        let text_output = MemoryContent::ToolOutput {
            tool_name: "shell".into(),
            output: "build finished".into(),
            mime_type: Some("text/plain".into()),
            schema: None,
            invocation_id: Some("run-1".into()),
            checksum: None,
        };
        assert_eq!(text_output.modality(), "tool_output");
        assert_eq!(text_output.modality_profile(), ModalityProfile::Text);
        assert_eq!(text_output.text_for_embedding(), "shell\nbuild finished");

        let structured_output = MemoryContent::ToolOutput {
            tool_name: "kubectl".into(),
            output: r#"{"status":"ok"}"#.into(),
            mime_type: Some("application/json".into()),
            schema: Some("k8s/status.v1".into()),
            invocation_id: None,
            checksum: Some("sha256:status".into()),
        };
        assert_eq!(
            structured_output.modality_profile(),
            ModalityProfile::Structured
        );
    }

    #[test]
    fn composite_modality() {
        let c = MemoryContent::Composite(vec![
            MemoryContent::Text("caption".into()),
            MemoryContent::Image {
                data: vec![1, 2, 3],
                mime_type: "image/jpeg".into(),
                description: "photo".into(),
            },
        ]);
        assert_eq!(c.modality(), "composite");
        let embed_text = c.text_for_embedding();
        assert!(embed_text.contains("caption"));
        assert!(embed_text.contains("photo"));
    }

    #[test]
    fn from_string() {
        let c: MemoryContent = "hello".into();
        assert_eq!(c, MemoryContent::Text("hello".into()));
    }

    #[test]
    fn serde_round_trip_text() {
        let c = MemoryContent::Text("hello".into());
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_image() {
        let c = MemoryContent::Image {
            data: vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            mime_type: "image/png".into(),
            description: "test png".into(),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_code() {
        let c = MemoryContent::Code {
            source: "def hello(): pass".into(),
            language: "python".into(),
            ast_hash: None,
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_audio() {
        let c = MemoryContent::Audio {
            data: vec![0xFF, 0xFB, 0x90, 0x00],
            transcript: "meeting notes".into(),
            duration_ms: 60000,
            channel_count: Some(1),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_video() {
        let c = MemoryContent::Video {
            data: vec![0x00, 0x00, 0x00, 0x20],
            mime_type: "video/mp4".into(),
            transcript: "launch demo".into(),
            description: "camera pan across booth".into(),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_document() {
        let c = MemoryContent::Document {
            data: b"%PDF-1.4 sample".to_vec(),
            mime_type: "application/pdf".into(),
            extracted_text: "sample pdf".into(),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_external() {
        let c = MemoryContent::External {
            uri: "https://example.com/run/42".into(),
            title: "deployment log".into(),
            snippet: "green rollout completed".into(),
            mime_type: Some("text/html".into()),
            checksum: Some("sha256:abc".into()),
            fetch_policy: ExternalFetchPolicy::IfStale,
            stale_at: Some(Timestamp::from_millis(1_712_345_678_000)),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_tool_output() {
        let c = MemoryContent::ToolOutput {
            tool_name: "terraform".into(),
            output: r#"{"applied":true}"#.into(),
            mime_type: Some("application/json".into()),
            schema: Some("terraform/apply.v1".into()),
            invocation_id: Some("apply-42".into()),
            checksum: Some("sha256:apply".into()),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_structured() {
        let c = MemoryContent::Structured {
            schema: "config/v2".into(),
            data: serde_json::json!({"port": 8080, "host": "localhost"}),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn serde_round_trip_composite() {
        let c = MemoryContent::Composite(vec![
            MemoryContent::Text("description".into()),
            MemoryContent::Image {
                data: vec![1, 2, 3],
                mime_type: "image/jpeg".into(),
                description: "photo".into(),
            },
        ]);
        let json = serde_json::to_string(&c).unwrap();
        let back: MemoryContent = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn composite_policy_weights_by_modality() {
        let policy = CompositeEmbeddingPolicy::WeightedMeanNormalized(CompositeModalityWeights {
            text: 2.0,
            image: 3.0,
            ..Default::default()
        });

        assert_eq!(policy.weight_for(&MemoryContent::Text("x".into())), 2.0);
        assert_eq!(
            policy.weight_for(&MemoryContent::Image {
                data: vec![1],
                mime_type: "image/png".into(),
                description: "img".into(),
            }),
            3.0
        );
    }

    #[test]
    fn bincode_round_trip() {
        let c = MemoryContent::Image {
            data: vec![0x89, 0x50, 0x4E, 0x47],
            mime_type: "image/png".into(),
            description: "test".into(),
        };
        let bytes = bincode::serialize(&c).unwrap();
        let back: MemoryContent = bincode::deserialize(&bytes).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn display_text() {
        let c = MemoryContent::Text("hello world".into());
        assert_eq!(format!("{c}"), "hello world");
    }

    #[test]
    fn display_image() {
        let c = MemoryContent::Image {
            data: vec![],
            mime_type: "image/png".into(),
            description: "screenshot".into(),
        };
        assert!(format!("{c}").contains("screenshot"));
    }

    #[test]
    fn display_video_prefers_available_surrogate_text() {
        let c = MemoryContent::Video {
            data: vec![],
            mime_type: "video/mp4".into(),
            transcript: String::new(),
            description: "operator walk-through".into(),
        };
        assert!(format!("{c}").contains("operator walk-through"));
    }

    #[test]
    fn display_external_prefers_surrogate_text() {
        let c = MemoryContent::External {
            uri: "https://example.com/logs/1".into(),
            title: "release log".into(),
            snippet: "completed successfully".into(),
            mime_type: Some("text/html".into()),
            checksum: None,
            fetch_policy: ExternalFetchPolicy::OnDemand,
            stale_at: None,
        };
        let rendered = format!("{c}");
        assert!(rendered.contains("release log"));
        assert!(rendered.contains("completed successfully"));
    }

    #[test]
    fn display_tool_output_uses_tool_name_and_output() {
        let c = MemoryContent::ToolOutput {
            tool_name: "shell".into(),
            output: "deploy succeeded".into(),
            mime_type: None,
            schema: None,
            invocation_id: None,
            checksum: None,
        };
        let rendered = format!("{c}");
        assert!(rendered.contains("shell"));
        assert!(rendered.contains("deploy succeeded"));
    }
}
