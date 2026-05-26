use std::fmt;

use serde::{Deserialize, Serialize};

use crate::content::MemoryContent;
use crate::error::HirnError;
use crate::id::next_monotonic_ulid;
use crate::metadata::{Metadata, MetadataValue};
use crate::revision::{RevisionOperation, RevisionState};
use crate::timestamp::Timestamp;
use crate::types::{AgentId, Namespace};

/// Time-sortable, globally unique resource identifier wrapping a ULID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResourceId(ulid::Ulid);

impl ResourceId {
    /// Create a new `ResourceId` with the current timestamp.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(next_monotonic_ulid())
    }

    /// Create a `ResourceId` from an existing ULID.
    #[must_use]
    pub const fn from_ulid(ulid: ulid::Ulid) -> Self {
        Self(ulid)
    }

    /// Get the inner ULID.
    #[must_use]
    pub const fn as_ulid(&self) -> ulid::Ulid {
        self.0
    }

    /// Parse a `ResourceId` from a ULID string.
    pub fn parse(s: &str) -> Result<Self, crate::HirnError> {
        ulid::Ulid::from_string(s)
            .map(Self)
            .map_err(|e| crate::HirnError::InvalidInput(format!("invalid resource id '{s}': {e}")))
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Stable identity for a resource across all of its revisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LogicalResourceId(ulid::Ulid);

impl LogicalResourceId {
    /// Create a new logical resource identifier.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(next_monotonic_ulid())
    }

    /// Derive a stable logical identifier from a resource ID.
    #[must_use]
    pub const fn from_resource_id(id: ResourceId) -> Self {
        Self(id.as_ulid())
    }

    /// Parse a `LogicalResourceId` from a ULID string.
    pub fn parse(s: &str) -> Result<Self, crate::HirnError> {
        ulid::Ulid::from_string(s).map(Self).map_err(|e| {
            crate::HirnError::InvalidInput(format!("invalid logical resource id '{s}': {e}"))
        })
    }
}

impl fmt::Display for LogicalResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Immutable identifier for a specific resource revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResourceRevisionId(ulid::Ulid);

impl ResourceRevisionId {
    /// Create a new resource revision identifier.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(next_monotonic_ulid())
    }

    /// Derive a revision identifier from a resource ID.
    #[must_use]
    pub const fn from_resource_id(id: ResourceId) -> Self {
        Self(id.as_ulid())
    }

    /// Parse a `ResourceRevisionId` from a ULID string.
    pub fn parse(s: &str) -> Result<Self, crate::HirnError> {
        ulid::Ulid::from_string(s).map(Self).map_err(|e| {
            crate::HirnError::InvalidInput(format!("invalid resource revision id '{s}': {e}"))
        })
    }
}

impl fmt::Display for ResourceRevisionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Immutable identifier for a derived artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DerivedArtifactId(ulid::Ulid);

impl DerivedArtifactId {
    /// Create a new derived artifact identifier.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(next_monotonic_ulid())
    }

    /// Parse a `DerivedArtifactId` from a ULID string.
    pub fn parse(s: &str) -> Result<Self, crate::HirnError> {
        ulid::Ulid::from_string(s).map(Self).map_err(|e| {
            crate::HirnError::InvalidInput(format!("invalid derived artifact id '{s}': {e}"))
        })
    }
}

impl fmt::Display for DerivedArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// High-level modality classification for a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ModalityProfile {
    #[default]
    Text,
    Image,
    Audio,
    Code,
    Structured,
    Document,
    Video,
    Composite,
    External,
}

impl ModalityProfile {
    /// Stable wire representation for the modality.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Code => "code",
            Self::Structured => "structured",
            Self::Document => "document",
            Self::Video => "video",
            Self::Composite => "composite",
            Self::External => "external",
        }
    }

    /// Parse a modality from its stable string form.
    pub fn parse(value: &str) -> Result<Self, HirnError> {
        match value {
            "text" => Ok(Self::Text),
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "code" => Ok(Self::Code),
            "structured" => Ok(Self::Structured),
            "document" => Ok(Self::Document),
            "video" => Ok(Self::Video),
            "composite" => Ok(Self::Composite),
            "external" => Ok(Self::External),
            _ => Err(HirnError::InvalidInput(format!(
                "unknown modality profile: {value}"
            ))),
        }
    }
}

impl fmt::Display for ModalityProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&MemoryContent> for ModalityProfile {
    fn from(value: &MemoryContent) -> Self {
        value.modality_profile()
    }
}

/// Explicit resource hydration mode for recall and fetch surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum HydrationMode {
    #[default]
    MetadataOnly,
    Preview,
    Full,
}

impl HydrationMode {
    /// Stable wire representation for the hydration mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MetadataOnly => "metadata_only",
            Self::Preview => "preview",
            Self::Full => "full",
        }
    }

    /// Parse a hydration mode from its stable string form.
    pub fn parse(value: &str) -> Result<Self, HirnError> {
        match value {
            "metadata" | "metadata_only" => Ok(Self::MetadataOnly),
            "preview" => Ok(Self::Preview),
            "full" => Ok(Self::Full),
            _ => Err(HirnError::InvalidInput(format!(
                "unknown hydration mode: {value}"
            ))),
        }
    }
}

/// How a resource payload is stored or resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceLocation {
    /// The payload is intentionally not materialized separately.
    Inline,
    /// The payload lives in the resource blob dataset at the given slot.
    Blob { blob_index: u32 },
    /// The payload lives behind an external URI.
    External { uri: String },
}

/// Typed relationship from a memory to a resource object or one of its artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EvidenceRole {
    #[default]
    Source,
    Attachment,
    Proof,
    Output,
    Preview,
    Derived,
}

impl EvidenceRole {
    /// Stable wire representation for the evidence role.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Attachment => "attachment",
            Self::Proof => "proof",
            Self::Output => "output",
            Self::Preview => "preview",
            Self::Derived => "derived",
        }
    }

    /// Parse an evidence role from its stable string form.
    pub fn parse(value: &str) -> Result<Self, HirnError> {
        match value {
            "source" => Ok(Self::Source),
            "attachment" => Ok(Self::Attachment),
            "proof" => Ok(Self::Proof),
            "output" => Ok(Self::Output),
            "preview" => Ok(Self::Preview),
            "derived" => Ok(Self::Derived),
            _ => Err(HirnError::InvalidInput(format!(
                "unknown evidence role: {value}"
            ))),
        }
    }
}

/// Provenance class for what an evidence link actually references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum EvidenceProvenance {
    #[default]
    ObservedResource,
    GeneratedArtifact,
    TransformedSummary,
}

impl EvidenceProvenance {
    /// Stable wire representation for the provenance class.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ObservedResource => "observed_resource",
            Self::GeneratedArtifact => "generated_artifact",
            Self::TransformedSummary => "transformed_summary",
        }
    }

    /// Parse a provenance class from its stable string form.
    pub fn parse(value: &str) -> Result<Self, HirnError> {
        match value {
            "observed_resource" => Ok(Self::ObservedResource),
            "generated_artifact" => Ok(Self::GeneratedArtifact),
            "transformed_summary" => Ok(Self::TransformedSummary),
            _ => Err(HirnError::InvalidInput(format!(
                "unknown evidence provenance: {value}"
            ))),
        }
    }
}

/// Typed link from a memory record to a resource object or derived artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceLink {
    pub resource_id: ResourceId,
    pub artifact_id: Option<DerivedArtifactId>,
    pub role: EvidenceRole,
    #[serde(default)]
    pub provenance: EvidenceProvenance,
    #[serde(default)]
    pub part_index: Option<u32>,
    pub description: Option<String>,
}

impl EvidenceLink {
    /// Create a direct evidence link to a resource.
    #[must_use]
    pub const fn new(resource_id: ResourceId, role: EvidenceRole) -> Self {
        Self {
            resource_id,
            artifact_id: None,
            role,
            provenance: EvidenceProvenance::ObservedResource,
            part_index: None,
            description: None,
        }
    }

    /// Attach a derived artifact to this evidence link.
    #[must_use]
    pub const fn with_artifact(mut self, artifact_id: DerivedArtifactId) -> Self {
        self.artifact_id = Some(artifact_id);
        self
    }

    /// Mark what kind of provenance surface this link references.
    #[must_use]
    pub const fn with_provenance(mut self, provenance: EvidenceProvenance) -> Self {
        self.provenance = provenance;
        self
    }

    /// Attach the content part index this evidence link hydrates.
    #[must_use]
    pub const fn with_part_index(mut self, part_index: u32) -> Self {
        self.part_index = Some(part_index);
        self
    }

    /// Attach an optional human-readable description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Kind of derived artifact materialized from a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DerivedArtifactKind {
    #[default]
    Preview,
    OcrText,
    Transcript,
    Caption,
    Thumbnail,
    SyntaxSummary,
    SchemaSummary,
    GenerationFailure,
}

impl DerivedArtifactKind {
    /// Stable wire representation for the artifact kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Preview => "preview",
            Self::OcrText => "ocr_text",
            Self::Transcript => "transcript",
            Self::Caption => "caption",
            Self::Thumbnail => "thumbnail",
            Self::SyntaxSummary => "syntax_summary",
            Self::SchemaSummary => "schema_summary",
            Self::GenerationFailure => "generation_failure",
        }
    }

    /// Parse an artifact kind from its stable string form.
    pub fn parse(value: &str) -> Result<Self, HirnError> {
        match value {
            "preview" => Ok(Self::Preview),
            "ocr_text" => Ok(Self::OcrText),
            "transcript" => Ok(Self::Transcript),
            "caption" => Ok(Self::Caption),
            "thumbnail" => Ok(Self::Thumbnail),
            "syntax_summary" => Ok(Self::SyntaxSummary),
            "schema_summary" => Ok(Self::SchemaSummary),
            "generation_failure" => Ok(Self::GenerationFailure),
            _ => Err(HirnError::InvalidInput(format!(
                "unknown derived artifact kind: {value}"
            ))),
        }
    }

    /// Whether this artifact can satisfy preview-oriented hydration and packaging.
    #[must_use]
    pub const fn is_previewable(self) -> bool {
        matches!(
            self,
            Self::Preview
                | Self::OcrText
                | Self::Transcript
                | Self::Caption
                | Self::Thumbnail
                | Self::SyntaxSummary
                | Self::SchemaSummary
        )
    }

    /// Provenance class callers should expose for this artifact kind.
    #[must_use]
    pub const fn evidence_provenance(self) -> EvidenceProvenance {
        match self {
            Self::OcrText | Self::Transcript | Self::Thumbnail | Self::GenerationFailure => {
                EvidenceProvenance::GeneratedArtifact
            }
            Self::Preview | Self::Caption | Self::SyntaxSummary | Self::SchemaSummary => {
                EvidenceProvenance::TransformedSummary
            }
        }
    }
}

/// Governance state for a logical resource lineage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ResourceGovernanceState {
    #[default]
    Active,
    Redacted,
    Purged,
}

impl ResourceGovernanceState {
    /// Stable wire representation for the governance state.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Redacted => "redacted",
            Self::Purged => "purged",
        }
    }

    /// Parse a governance state from its stable string form.
    pub fn parse(value: &str) -> Result<Self, HirnError> {
        match value {
            "active" => Ok(Self::Active),
            "redacted" => Ok(Self::Redacted),
            "purged" => Ok(Self::Purged),
            _ => Err(HirnError::InvalidInput(format!(
                "unknown resource governance state: {value}"
            ))),
        }
    }

    /// Whether payload- and artifact-bearing hydration should be blocked.
    #[must_use]
    pub const fn hides_payload(self) -> bool {
        !matches!(self, Self::Active)
    }

    /// Default placeholder label for governed resources.
    #[must_use]
    pub const fn placeholder_display_name(self) -> &'static str {
        match self {
            Self::Active => "resource",
            Self::Redacted => "redacted resource",
            Self::Purged => "purged resource",
        }
    }
}

/// Governance action applied by a retention rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum ResourceRetentionAction {
    #[default]
    Redact,
    Purge,
}

impl ResourceRetentionAction {
    /// Stable wire representation for the action.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Redact => "redact",
            Self::Purge => "purge",
        }
    }

    /// Parse a retention action from its stable string form.
    pub fn parse(value: &str) -> Result<Self, HirnError> {
        match value {
            "redact" => Ok(Self::Redact),
            "purge" => Ok(Self::Purge),
            _ => Err(HirnError::InvalidInput(format!(
                "unknown resource retention action: {value}"
            ))),
        }
    }

    /// Severity ordering used when multiple rules match the same resource.
    #[must_use]
    pub const fn severity(self) -> u8 {
        match self {
            Self::Redact => 1,
            Self::Purge => 2,
        }
    }
}

/// A single operator-configured retention rule for resource governance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ResourceRetentionRule {
    pub action: ResourceRetentionAction,
    pub namespaces: Vec<Namespace>,
    pub modalities: Vec<ModalityProfile>,
    pub classifications: Vec<String>,
}

impl ResourceRetentionRule {
    /// Create a new retention rule with the provided governance action.
    #[must_use]
    pub const fn new(action: ResourceRetentionAction) -> Self {
        Self {
            action,
            namespaces: Vec::new(),
            modalities: Vec::new(),
            classifications: Vec::new(),
        }
    }

    /// Restrict the rule to a specific namespace.
    #[must_use]
    pub fn namespace(mut self, namespace: Namespace) -> Self {
        self.namespaces.push(namespace);
        self
    }

    /// Restrict the rule to a specific modality.
    #[must_use]
    pub fn modality(mut self, modality: ModalityProfile) -> Self {
        if !self.modalities.contains(&modality) {
            self.modalities.push(modality);
        }
        self
    }

    /// Restrict the rule to a specific classification metadata label.
    #[must_use]
    pub fn classification(mut self, classification: impl Into<String>) -> Self {
        let classification = classification.into().trim().to_string();
        if !classification.is_empty()
            && !self
                .classifications
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&classification))
        {
            self.classifications.push(classification);
        }
        self
    }

    /// Validate the rule before use.
    pub fn validate(&self) -> Result<(), HirnError> {
        if self.namespaces.is_empty()
            && self.modalities.is_empty()
            && self.classifications.is_empty()
        {
            return Err(HirnError::InvalidInput(
                "resource retention rule must target at least one namespace, modality, or classification"
                    .into(),
            ));
        }
        if self
            .classifications
            .iter()
            .any(|classification| classification.trim().is_empty())
        {
            return Err(HirnError::InvalidInput(
                "resource retention classifications must be non-empty".into(),
            ));
        }
        Ok(())
    }

    /// Whether this rule matches the provided resource head.
    #[must_use]
    pub fn matches(&self, resource: &ResourceObject) -> bool {
        let namespace_match =
            self.namespaces.is_empty() || self.namespaces.contains(&resource.namespace);
        let modality_match =
            self.modalities.is_empty() || self.modalities.contains(&resource.modality);
        let classification_match = if self.classifications.is_empty() {
            true
        } else {
            resource.classification().is_some_and(|classification| {
                self.classifications
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(classification))
            })
        };

        namespace_match && modality_match && classification_match
    }
}

/// Operator-configured resource retention policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ResourceRetentionPolicy {
    pub rules: Vec<ResourceRetentionRule>,
}

impl ResourceRetentionPolicy {
    /// Append a rule to the policy.
    #[must_use]
    pub fn with_rule(mut self, rule: ResourceRetentionRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Validate all configured rules.
    pub fn validate(&self) -> Result<(), HirnError> {
        for rule in &self.rules {
            rule.validate()?;
        }
        Ok(())
    }

    /// Whether the policy contains no rules.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Select the strongest action for a resource when multiple rules match.
    #[must_use]
    pub fn strongest_action_for(
        &self,
        resource: &ResourceObject,
    ) -> Option<ResourceRetentionAction> {
        self.rules
            .iter()
            .filter(|rule| rule.matches(resource))
            .map(|rule| rule.action)
            .max_by_key(|action| action.severity())
    }
}

/// Scope a quota applies to when admitting resource writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceQuotaScope {
    /// Aggregate all active resources in the current store/realm.
    Realm,
    /// Aggregate active resources inside one namespace.
    Namespace(Namespace),
    /// Aggregate active resources owned by one agent.
    Agent(AgentId),
}

impl ResourceQuotaScope {
    /// Whether the scope includes the provided resource.
    #[must_use]
    pub fn matches(&self, resource: &ResourceObject) -> bool {
        match self {
            Self::Realm => true,
            Self::Namespace(namespace) => *namespace == resource.namespace,
            Self::Agent(agent_id) => {
                matches!(resource.owner_agent_id, Some(owner) if owner == *agent_id)
            }
        }
    }
}

/// A single quota rule evaluated before persisting a resource write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceQuotaRule {
    pub scope: ResourceQuotaScope,
    pub max_active_resources: Option<usize>,
    pub max_total_bytes: Option<u64>,
}

impl ResourceQuotaRule {
    /// Create a new quota rule for the provided scope.
    #[must_use]
    pub const fn new(scope: ResourceQuotaScope) -> Self {
        Self {
            scope,
            max_active_resources: None,
            max_total_bytes: None,
        }
    }

    /// Set the maximum number of active resource heads allowed in this scope.
    #[must_use]
    pub const fn max_active_resources(mut self, max_active_resources: usize) -> Self {
        self.max_active_resources = Some(max_active_resources);
        self
    }

    /// Set the maximum total active payload bytes allowed in this scope.
    #[must_use]
    pub const fn max_total_bytes(mut self, max_total_bytes: u64) -> Self {
        self.max_total_bytes = Some(max_total_bytes);
        self
    }

    /// Validate the rule before use.
    pub fn validate(&self) -> Result<(), HirnError> {
        if self.max_active_resources.is_none() && self.max_total_bytes.is_none() {
            return Err(HirnError::InvalidInput(
                "resource quota rule must configure max_active_resources or max_total_bytes".into(),
            ));
        }
        if self.max_active_resources == Some(0) {
            return Err(HirnError::InvalidInput(
                "resource quota max_active_resources must be > 0".into(),
            ));
        }
        if self.max_total_bytes == Some(0) {
            return Err(HirnError::InvalidInput(
                "resource quota max_total_bytes must be > 0".into(),
            ));
        }
        Ok(())
    }

    /// Whether this rule applies to the provided resource.
    #[must_use]
    pub fn matches(&self, resource: &ResourceObject) -> bool {
        self.scope.matches(resource)
    }
}

/// Operator-configured quota policy for first-class resources.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ResourceQuotaPolicy {
    pub rules: Vec<ResourceQuotaRule>,
}

impl ResourceQuotaPolicy {
    /// Append a quota rule to the policy.
    #[must_use]
    pub fn with_rule(mut self, rule: ResourceQuotaRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Validate all configured quota rules.
    pub fn validate(&self) -> Result<(), HirnError> {
        for rule in &self.rules {
            rule.validate()?;
        }
        Ok(())
    }

    /// Whether the policy contains no quota rules.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Iterate quota rules that apply to the provided resource.
    pub fn rules_for<'a>(
        &'a self,
        resource: &'a ResourceObject,
    ) -> impl Iterator<Item = &'a ResourceQuotaRule> + 'a {
        self.rules.iter().filter(|rule| rule.matches(resource))
    }
}

/// Scalar secondary index families supported for resource-adjacent datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SecondaryIndexType {
    #[default]
    BTree,
    Bitmap,
}

/// Per-modality secondary index rule for the `resources` dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ResourceIndexRule {
    /// Modality whose access pattern this rule is tuning.
    pub modality: ModalityProfile,
    /// Index family to create for the configured composite columns.
    pub index_type: SecondaryIndexType,
    /// Additional columns appended after the `modality` prefix.
    pub columns: Vec<String>,
}

impl ResourceIndexRule {
    /// Create a new rule for the given modality.
    #[must_use]
    pub const fn new(modality: ModalityProfile, index_type: SecondaryIndexType) -> Self {
        Self {
            modality,
            index_type,
            columns: Vec::new(),
        }
    }

    /// Add an additional column after the `modality` prefix.
    #[must_use]
    pub fn with_column(mut self, column: impl Into<String>) -> Self {
        self.columns.push(column.into());
        self
    }

    /// Validate the configured columns.
    pub fn validate(&self) -> Result<(), HirnError> {
        for column in &self.columns {
            if column.trim().is_empty() {
                return Err(HirnError::InvalidConfig {
                    field: "resource_index_policy.columns".into(),
                    value: column.clone(),
                    reason: "index column names must be non-empty".into(),
                });
            }
            if !matches!(
                column.as_str(),
                "logical_resource_id"
                    | "revision_id"
                    | "mime_type"
                    | "display_name"
                    | "checksum"
                    | "size_bytes"
                    | "owner_agent_id"
                    | "governance_state"
                    | "namespace"
                    | "created_at_ms"
                    | "updated_at_ms"
            ) {
                return Err(HirnError::InvalidConfig {
                    field: "resource_index_policy.columns".into(),
                    value: column.clone(),
                    reason: "unsupported resources index column".into(),
                });
            }
        }

        Ok(())
    }

    /// Physical composite columns to index for this rule.
    #[must_use]
    pub fn scoped_columns(&self) -> Vec<String> {
        let mut columns = vec!["modality".to_string()];
        for column in &self.columns {
            if column != "modality" && !columns.contains(column) {
                columns.push(column.clone());
            }
        }
        columns
    }
}

/// Policy describing additional resource secondary indices keyed by modality.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ResourceIndexPolicy {
    pub rules: Vec<ResourceIndexRule>,
}

impl ResourceIndexPolicy {
    /// Add a rule to the policy.
    #[must_use]
    pub fn with_rule(mut self, rule: ResourceIndexRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Validate all configured rules.
    pub fn validate(&self) -> Result<(), HirnError> {
        for rule in &self.rules {
            rule.validate()?;
        }
        Ok(())
    }

    /// Whether the policy contains no extra rules.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// Per-kind secondary index rule for the `derived_artifacts` dataset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DerivedArtifactIndexRule {
    /// Artifact kind whose access pattern this rule is tuning.
    pub kind: DerivedArtifactKind,
    /// Index family to create for the configured composite columns.
    pub index_type: SecondaryIndexType,
    /// Additional columns appended after the `kind` prefix.
    pub columns: Vec<String>,
}

impl DerivedArtifactIndexRule {
    /// Create a new rule for the given artifact kind.
    #[must_use]
    pub const fn new(kind: DerivedArtifactKind, index_type: SecondaryIndexType) -> Self {
        Self {
            kind,
            index_type,
            columns: Vec::new(),
        }
    }

    /// Add an additional column after the `kind` prefix.
    #[must_use]
    pub fn with_column(mut self, column: impl Into<String>) -> Self {
        self.columns.push(column.into());
        self
    }

    /// Validate the configured columns.
    pub fn validate(&self) -> Result<(), HirnError> {
        for column in &self.columns {
            if column.trim().is_empty() {
                return Err(HirnError::InvalidConfig {
                    field: "derived_artifact_index_policy.columns".into(),
                    value: column.clone(),
                    reason: "index column names must be non-empty".into(),
                });
            }
            if !matches!(
                column.as_str(),
                "resource_id"
                    | "modality"
                    | "mime_type"
                    | "checksum"
                    | "namespace"
                    | "created_at_ms"
            ) {
                return Err(HirnError::InvalidConfig {
                    field: "derived_artifact_index_policy.columns".into(),
                    value: column.clone(),
                    reason: "unsupported derived_artifacts index column".into(),
                });
            }
        }

        Ok(())
    }

    /// Physical composite columns to index for this rule.
    #[must_use]
    pub fn scoped_columns(&self) -> Vec<String> {
        let mut columns = vec!["kind".to_string()];
        for column in &self.columns {
            if column != "kind" && !columns.contains(column) {
                columns.push(column.clone());
            }
        }
        columns
    }
}

/// Policy describing additional derived-artifact secondary indices keyed by artifact kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DerivedArtifactIndexPolicy {
    pub rules: Vec<DerivedArtifactIndexRule>,
}

impl DerivedArtifactIndexPolicy {
    /// Add a rule to the policy.
    #[must_use]
    pub fn with_rule(mut self, rule: DerivedArtifactIndexRule) -> Self {
        self.rules.push(rule);
        self
    }

    /// Validate all configured rules.
    pub fn validate(&self) -> Result<(), HirnError> {
        for rule in &self.rules {
            rule.validate()?;
        }
        Ok(())
    }

    /// Whether the policy contains no extra rules.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

/// First-class resource object stored independently from memory rows.
const fn resource_storage_ready_default() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceObject {
    pub id: ResourceId,
    pub logical_resource_id: LogicalResourceId,
    pub revision_id: ResourceRevisionId,
    pub version: u32,
    pub revision_operation: RevisionOperation,
    pub revision_reason: Option<String>,
    pub revision_causation_id: Option<ResourceId>,
    pub superseded_by: Option<ResourceId>,
    pub modality: ModalityProfile,
    pub mime_type: Option<String>,
    pub display_name: Option<String>,
    pub checksum: Option<String>,
    pub size_bytes: u64,
    pub location: ResourceLocation,
    pub metadata: Metadata,
    #[serde(default = "resource_storage_ready_default")]
    pub storage_ready: bool,
    pub owner_agent_id: Option<AgentId>,
    pub governance_state: ResourceGovernanceState,
    pub governance_reason: Option<String>,
    pub governed_at: Option<Timestamp>,
    pub namespace: Namespace,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl ResourceObject {
    /// Create a new builder for this resource type.
    #[must_use]
    pub fn builder() -> ResourceObjectBuilder {
        ResourceObjectBuilder::default()
    }

    /// Whether this revision is still the active resource head.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        self.superseded_by.is_none()
    }

    /// Whether the revision is visible to read paths.
    #[must_use]
    pub const fn is_storage_ready(&self) -> bool {
        self.storage_ready
    }

    /// Computed state for this revision within the context of a resource head.
    #[must_use]
    pub fn revision_state_against(&self, head: &Self) -> RevisionState {
        if self.revision_id == head.revision_id {
            RevisionState::Active
        } else {
            RevisionState::Superseded
        }
    }

    /// Optional classification label carried in resource metadata.
    #[must_use]
    pub fn classification(&self) -> Option<&str> {
        match self.metadata.get("classification") {
            Some(MetadataValue::String(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    /// Owning agent for agent-scoped resource governance and quotas.
    #[must_use]
    pub const fn owner_agent_id(&self) -> Option<AgentId> {
        self.owner_agent_id
    }
}

/// Builder for [`ResourceObject`].
#[derive(Debug, Default)]
pub struct ResourceObjectBuilder {
    modality: Option<ModalityProfile>,
    mime_type: Option<String>,
    display_name: Option<String>,
    checksum: Option<String>,
    size_bytes: Option<u64>,
    location: Option<ResourceLocation>,
    metadata: Metadata,
    owner_agent_id: Option<AgentId>,
    namespace: Option<Namespace>,
}

impl ResourceObjectBuilder {
    /// Set the modality of the resource.
    #[must_use]
    pub const fn modality(mut self, modality: ModalityProfile) -> Self {
        self.modality = Some(modality);
        self
    }

    /// Set the MIME type associated with the resource.
    #[must_use]
    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// Set the human-readable display name.
    #[must_use]
    pub fn display_name(mut self, display_name: impl Into<String>) -> Self {
        self.display_name = Some(display_name.into());
        self
    }

    /// Set a content-address or strong checksum string.
    #[must_use]
    pub fn checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    /// Set the payload size in bytes.
    #[must_use]
    pub const fn size_bytes(mut self, size_bytes: u64) -> Self {
        self.size_bytes = Some(size_bytes);
        self
    }

    /// Set the storage location for the resource payload.
    #[must_use]
    pub fn location(mut self, location: ResourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Insert a metadata entry.
    #[must_use]
    pub fn metadata_entry(
        mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the owning agent for this resource.
    #[must_use]
    pub const fn owner_agent_id(mut self, owner_agent_id: AgentId) -> Self {
        self.owner_agent_id = Some(owner_agent_id);
        self
    }

    /// Set the namespace for this resource.
    #[must_use]
    pub fn namespace(mut self, namespace: Namespace) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Build the resource object.
    pub fn build(self) -> Result<ResourceObject, HirnError> {
        let modality = self
            .modality
            .ok_or_else(|| HirnError::InvalidInput("resource modality is required".into()))?;
        let location = self
            .location
            .ok_or_else(|| HirnError::InvalidInput("resource location is required".into()))?;

        if let ResourceLocation::External { uri } = &location
            && uri.trim().is_empty()
        {
            return Err(HirnError::InvalidInput(
                "resource external URI must be non-empty".into(),
            ));
        }

        let now = Timestamp::now();
        let id = ResourceId::new();

        Ok(ResourceObject {
            id,
            logical_resource_id: LogicalResourceId::from_resource_id(id),
            revision_id: ResourceRevisionId::from_resource_id(id),
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            superseded_by: None,
            modality,
            mime_type: self.mime_type,
            display_name: self.display_name,
            checksum: self.checksum,
            size_bytes: self.size_bytes.unwrap_or(0),
            location,
            metadata: self.metadata,
            storage_ready: resource_storage_ready_default(),
            owner_agent_id: self.owner_agent_id,
            governance_state: ResourceGovernanceState::Active,
            governance_reason: None,
            governed_at: None,
            namespace: self.namespace.unwrap_or_default(),
            created_at: now,
            updated_at: now,
        })
    }
}

/// Derived artifact generated from a resource object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivedArtifact {
    pub id: DerivedArtifactId,
    pub resource_id: ResourceId,
    pub kind: DerivedArtifactKind,
    pub modality: ModalityProfile,
    pub mime_type: Option<String>,
    pub text_content: Option<String>,
    pub blob_index: Option<u32>,
    pub checksum: Option<String>,
    pub metadata: Metadata,
    pub namespace: Namespace,
    pub created_at: Timestamp,
}

impl DerivedArtifact {
    /// Create a new builder for this artifact type.
    #[must_use]
    pub fn builder() -> DerivedArtifactBuilder {
        DerivedArtifactBuilder::default()
    }
}

/// Builder for [`DerivedArtifact`].
#[derive(Debug, Default)]
pub struct DerivedArtifactBuilder {
    resource_id: Option<ResourceId>,
    kind: Option<DerivedArtifactKind>,
    modality: Option<ModalityProfile>,
    mime_type: Option<String>,
    text_content: Option<String>,
    blob_index: Option<u32>,
    checksum: Option<String>,
    metadata: Metadata,
    namespace: Option<Namespace>,
}

impl DerivedArtifactBuilder {
    /// Set the parent resource ID.
    #[must_use]
    pub const fn resource_id(mut self, resource_id: ResourceId) -> Self {
        self.resource_id = Some(resource_id);
        self
    }

    /// Set the derived artifact kind.
    #[must_use]
    pub const fn kind(mut self, kind: DerivedArtifactKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Set the artifact modality.
    #[must_use]
    pub const fn modality(mut self, modality: ModalityProfile) -> Self {
        self.modality = Some(modality);
        self
    }

    /// Set the MIME type associated with the artifact.
    #[must_use]
    pub fn mime_type(mut self, mime_type: impl Into<String>) -> Self {
        self.mime_type = Some(mime_type.into());
        self
    }

    /// Set the textual payload for this artifact.
    #[must_use]
    pub fn text_content(mut self, text_content: impl Into<String>) -> Self {
        self.text_content = Some(text_content.into());
        self
    }

    /// Reference a binary preview or artifact blob.
    #[must_use]
    pub const fn blob_index(mut self, blob_index: u32) -> Self {
        self.blob_index = Some(blob_index);
        self
    }

    /// Set a checksum string for the artifact payload.
    #[must_use]
    pub fn checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    /// Insert a metadata entry.
    #[must_use]
    pub fn metadata_entry(
        mut self,
        key: impl Into<String>,
        value: impl Into<MetadataValue>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Set the namespace for this artifact.
    #[must_use]
    pub fn namespace(mut self, namespace: Namespace) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Build the artifact record.
    pub fn build(self) -> Result<DerivedArtifact, HirnError> {
        let resource_id = self
            .resource_id
            .ok_or_else(|| HirnError::InvalidInput("artifact resource_id is required".into()))?;
        let kind = self
            .kind
            .ok_or_else(|| HirnError::InvalidInput("artifact kind is required".into()))?;
        let modality = self
            .modality
            .ok_or_else(|| HirnError::InvalidInput("artifact modality is required".into()))?;

        if self.text_content.is_none() && self.blob_index.is_none() {
            return Err(HirnError::InvalidInput(
                "artifact requires text_content or blob_index".into(),
            ));
        }

        Ok(DerivedArtifact {
            id: DerivedArtifactId::new(),
            resource_id,
            kind,
            modality,
            mime_type: self.mime_type,
            text_content: self.text_content,
            blob_index: self.blob_index,
            checksum: self.checksum,
            metadata: self.metadata,
            namespace: self.namespace.unwrap_or_default(),
            created_at: Timestamp::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_id_round_trip() {
        let id = ResourceId::new();
        let parsed = ResourceId::parse(&id.to_string()).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn modality_profile_from_memory_content() {
        let content = MemoryContent::Image {
            data: vec![1, 2, 3],
            mime_type: "image/png".into(),
            description: "preview".into(),
        };
        assert_eq!(ModalityProfile::from(&content), ModalityProfile::Image);
    }

    #[test]
    fn resource_quota_policy_matches_agent_and_namespace_scopes() {
        let agent = AgentId::well_known("quota-agent");
        let namespace = Namespace::new("quota-ns").unwrap();
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .location(ResourceLocation::Inline)
            .namespace(namespace)
            .owner_agent_id(agent)
            .build()
            .unwrap();

        let policy = ResourceQuotaPolicy::default()
            .with_rule(
                ResourceQuotaRule::new(ResourceQuotaScope::Agent(agent)).max_active_resources(2),
            )
            .with_rule(
                ResourceQuotaRule::new(ResourceQuotaScope::Namespace(namespace))
                    .max_total_bytes(1024),
            );

        let matched = policy.rules_for(&resource).collect::<Vec<_>>();
        assert_eq!(matched.len(), 2);
        assert!(
            matched
                .iter()
                .any(|rule| matches!(rule.scope, ResourceQuotaScope::Agent(_)))
        );
        assert!(
            matched
                .iter()
                .any(|rule| matches!(rule.scope, ResourceQuotaScope::Namespace(_)))
        );
    }

    #[test]
    fn resource_quota_rule_requires_a_limit() {
        let rule = ResourceQuotaRule::new(ResourceQuotaScope::Realm);
        assert!(rule.validate().is_err());
    }

    #[test]
    fn build_resource_object() {
        let resource = ResourceObject::builder()
            .modality(ModalityProfile::Document)
            .mime_type("application/pdf")
            .display_name("design-doc.pdf")
            .checksum("blake3:abc")
            .size_bytes(2048)
            .location(ResourceLocation::External {
                uri: "https://example.invalid/design-doc.pdf".into(),
            })
            .build()
            .unwrap();

        assert_eq!(resource.version, 1);
        assert_eq!(resource.modality, ModalityProfile::Document);
        assert!(resource.is_live());
        assert!(resource.is_storage_ready());
    }

    #[test]
    fn build_derived_artifact_requires_payload() {
        let resource_id = ResourceId::new();
        let result = DerivedArtifact::builder()
            .resource_id(resource_id)
            .kind(DerivedArtifactKind::Caption)
            .modality(ModalityProfile::Text)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn evidence_link_keeps_artifact_reference() {
        let resource_id = ResourceId::new();
        let artifact_id = DerivedArtifactId::new();
        let link = EvidenceLink::new(resource_id, EvidenceRole::Preview)
            .with_artifact(artifact_id)
            .with_provenance(EvidenceProvenance::TransformedSummary)
            .with_part_index(2)
            .with_description("thumbnail");

        assert_eq!(link.resource_id, resource_id);
        assert_eq!(link.artifact_id, Some(artifact_id));
        assert_eq!(link.role, EvidenceRole::Preview);
        assert_eq!(link.provenance, EvidenceProvenance::TransformedSummary);
        assert_eq!(link.part_index, Some(2));
    }

    #[test]
    fn derived_artifact_kind_maps_to_provenance_class() {
        assert_eq!(
            DerivedArtifactKind::OcrText.evidence_provenance(),
            EvidenceProvenance::GeneratedArtifact
        );
        assert_eq!(
            DerivedArtifactKind::Caption.evidence_provenance(),
            EvidenceProvenance::TransformedSummary
        );
        assert_eq!(
            DerivedArtifactKind::Preview.evidence_provenance(),
            EvidenceProvenance::TransformedSummary
        );
    }

    #[test]
    fn resource_index_policy_rejects_unknown_columns() {
        let policy = ResourceIndexPolicy::default().with_rule(
            ResourceIndexRule::new(ModalityProfile::Document, SecondaryIndexType::BTree)
                .with_column("unsupported"),
        );

        assert!(policy.validate().is_err());
    }

    #[test]
    fn derived_artifact_index_rule_scopes_columns_by_kind() {
        let rule = DerivedArtifactIndexRule::new(
            DerivedArtifactKind::Transcript,
            SecondaryIndexType::Bitmap,
        )
        .with_column("modality")
        .with_column("namespace");

        assert_eq!(rule.scoped_columns(), vec!["kind", "modality", "namespace"]);
    }
}
