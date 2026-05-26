use serde::{Deserialize, Serialize};

use crate::error::{HirnError, HirnResult};
use crate::id::MemoryId;
use crate::metadata::Metadata;
use crate::provenance::Provenance;
use crate::resource::EvidenceLink;
use crate::revision::{LogicalMemoryId, RevisionId, RevisionOperation, RevisionState};
use crate::timestamp::Timestamp;
use crate::types::{AgentId, Namespace, Origin};

/// A single step in a procedural workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionStep {
    /// Human-readable description of what this step does.
    pub description: String,
    /// Tool or function name to invoke (if applicable).
    pub tool: Option<String>,
    /// Parameters or arguments for the tool.
    pub parameters: Metadata,
}

/// Result of executing a single [`ActionStep`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Index of the step within the procedure.
    pub step_index: usize,
    /// Whether the step succeeded.
    pub success: bool,
    /// Output produced by the step (tool output, log message, etc.).
    pub output: String,
}

/// Result of executing an entire [`ProceduralRecord`].
#[derive(Debug, Clone)]
pub struct ProcedureResult {
    /// ID of the executed procedure.
    pub procedure_id: MemoryId,
    /// Whether all steps completed successfully.
    pub success: bool,
    /// Per-step results, in order.
    pub step_results: Vec<StepResult>,
}

/// Trait for dispatching tool invocations from procedural action steps.
///
/// Implement this to connect hirn's procedural memory to an actual
/// tool runtime (function-calling agents, MCP servers, shell commands, etc.).
///
/// # Example
///
/// ```rust,ignore
/// struct MyToolRuntime;
///
/// impl ToolExecutor for MyToolRuntime {
///     async fn execute_step(&self, step: &ActionStep) -> HirnResult<StepResult> {
///         // dispatch to your tool registry
///     }
/// }
/// ```
pub trait ToolExecutor: Send + Sync {
    /// Execute a single action step and return its result.
    ///
    /// Implementations should:
    /// 1. Resolve `step.tool` to an actual callable
    /// 2. Pass `step.parameters` as arguments
    /// 3. Return success/failure with output
    fn execute_step(
        &self,
        step: &ActionStep,
    ) -> impl std::future::Future<Output = HirnResult<StepResult>> + Send;
}

/// A procedural memory record — a learned skill, tool-use pattern, or multi-step workflow.
///
/// Inspired by `CoALA` (arXiv:2309.02427) and AWM (arXiv:2409.07429).
/// Procedural memory captures *how* to do something, as opposed to episodic
/// (what happened) or semantic (what is known).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProceduralRecord {
    pub id: MemoryId,
    pub logical_memory_id: LogicalMemoryId,
    pub revision_id: RevisionId,
    /// Short name for this procedure (e.g., "deploy-to-staging").
    pub name: String,
    /// Natural-language description of the skill/workflow.
    pub description: String,
    /// Ordered sequence of action steps.
    pub steps: Vec<ActionStep>,
    /// Preconditions that must hold for this procedure to be applicable.
    pub preconditions: Vec<String>,
    /// Embedding of the description for vector retrieval.
    pub embedding: Option<Vec<f32>>,
    /// How often this procedure has been successfully invoked.
    pub success_count: u64,
    /// How often this procedure has been invoked (success + failure).
    pub invocation_count: u64,
    /// Success rate = `success_count` / `invocation_count` (cached).
    pub success_rate: f32,
    /// Source episodes that contributed to learning this procedure.
    pub source_episodes: Vec<MemoryId>,
    /// Event/observed time for this revision.
    pub observed_at: Timestamp,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub last_accessed: Timestamp,
    pub access_count: u64,
    /// Monotonic revision number within a logical memory chain.
    pub version: u32,
    /// Operation that produced this immutable revision.
    pub revision_operation: RevisionOperation,
    /// Optional human-readable reason for the revision.
    pub revision_reason: Option<String>,
    /// Optional revision or memory that caused this revision to be written.
    pub revision_causation_id: Option<MemoryId>,
    /// ID of the revision that superseded this one, if any.
    pub superseded_by: Option<MemoryId>,
    pub provenance: Provenance,
    pub metadata: Metadata,
    pub namespace: Namespace,
    pub archived: bool,
}

impl ProceduralRecord {
    /// Create a new builder for this record type.
    #[must_use]
    pub fn builder() -> ProceduralRecordBuilder {
        ProceduralRecordBuilder::default()
    }

    /// Record an access: bump count and update timestamp.
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Timestamp::now();
    }

    /// Record a successful invocation.
    ///
    /// F-81: Uses an exponential moving average (EMA) with α=0.1 so recent
    /// outcomes are weighted more heavily than ancient history. The all-time
    /// `success_count`/`invocation_count` ratio is still available via those
    /// fields for long-term analytics.
    pub fn record_success(&mut self) {
        self.invocation_count += 1;
        self.success_count += 1;
        // EMA: rate = α·outcome + (1-α)·rate, α = 0.1
        self.success_rate = 0.1_f32
            .mul_add(1.0, 0.9 * self.success_rate)
            .clamp(0.0, 1.0);
        self.updated_at = Timestamp::now();
    }

    /// Record a failed invocation.
    pub fn record_failure(&mut self) {
        self.invocation_count += 1;
        // EMA: rate = α·outcome + (1-α)·rate, outcome = 0.0
        self.success_rate = (0.9 * self.success_rate).clamp(0.0, 1.0);
        self.updated_at = Timestamp::now();
    }

    /// Whether this revision is a retraction/tombstone.
    #[must_use]
    pub const fn is_retracted(&self) -> bool {
        matches!(self.revision_operation, RevisionOperation::Retract)
    }

    /// Whether this revision should participate in current-state recall.
    #[must_use]
    pub const fn is_live(&self) -> bool {
        !self.archived && !self.is_retracted()
    }

    /// Computed state for this revision within the context of a logical chain head.
    #[must_use]
    pub fn revision_state_against(&self, head: &Self) -> RevisionState {
        if self.revision_id == head.revision_id {
            if head.is_live() {
                RevisionState::Active
            } else {
                RevisionState::Retracted
            }
        } else {
            RevisionState::Superseded
        }
    }
}

#[derive(Debug, Default)]
pub struct ProceduralRecordBuilder {
    name: Option<String>,
    description: Option<String>,
    steps: Vec<ActionStep>,
    preconditions: Vec<String>,
    embedding: Option<Vec<f32>>,
    source_episodes: Vec<MemoryId>,
    agent_id: Option<AgentId>,
    namespace: Option<Namespace>,
    evidence_links: Vec<EvidenceLink>,
    metadata: Metadata,
}

impl ProceduralRecordBuilder {
    /// Set the procedure name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the procedure description.
    #[must_use]
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set the ordered action steps.
    #[must_use]
    pub fn steps(mut self, steps: Vec<ActionStep>) -> Self {
        self.steps = steps;
        self
    }

    /// Set preconditions that must hold before execution.
    #[must_use]
    pub fn preconditions(mut self, preconditions: Vec<String>) -> Self {
        self.preconditions = preconditions;
        self
    }

    /// Set a pre-computed embedding vector.
    #[must_use]
    pub fn embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    /// Set the episodic memories this procedure was derived from.
    #[must_use]
    pub fn source_episodes(mut self, ids: Vec<MemoryId>) -> Self {
        self.source_episodes = ids;
        self
    }

    /// Set the agent that created this record.
    #[must_use]
    pub fn agent_id(mut self, agent_id: AgentId) -> Self {
        self.agent_id = Some(agent_id);
        self
    }

    /// Set the namespace for this record.
    #[must_use]
    pub fn namespace(mut self, namespace: Namespace) -> Self {
        self.namespace = Some(namespace);
        self
    }

    /// Add a typed resource evidence link to this record's provenance.
    #[must_use]
    pub fn evidence_link(mut self, evidence_link: EvidenceLink) -> Self {
        self.evidence_links.push(evidence_link);
        self
    }

    /// Insert a key-value metadata entry.
    #[must_use]
    pub fn metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<crate::metadata::MetadataValue>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Validate and build the procedural record.
    pub fn build(self) -> Result<ProceduralRecord, HirnError> {
        let name = self.name.filter(|n| !n.trim().is_empty()).ok_or_else(|| {
            HirnError::InvalidInput("procedural record requires non-empty name".into())
        })?;
        let description = self
            .description
            .filter(|d| !d.trim().is_empty())
            .ok_or_else(|| {
                HirnError::InvalidInput("procedural record requires non-empty description".into())
            })?;
        let agent_id = self
            .agent_id
            .ok_or_else(|| HirnError::InvalidInput("procedural record requires agent_id".into()))?;

        let now = Timestamp::now();
        let namespace = self
            .namespace
            .unwrap_or_else(|| Namespace::private_for(&agent_id));
        let id = MemoryId::new();
        let mut provenance = Provenance::with_origin(Origin::DirectObservation, agent_id);
        provenance.evidence_links = self.evidence_links;

        Ok(ProceduralRecord {
            id,
            logical_memory_id: LogicalMemoryId::from_memory_id(id),
            revision_id: RevisionId::from_memory_id(id),
            name,
            description,
            steps: self.steps,
            preconditions: self.preconditions,
            embedding: self.embedding,
            success_count: 0,
            invocation_count: 0,
            success_rate: 0.0,
            source_episodes: self.source_episodes,
            observed_at: now,
            created_at: now,
            updated_at: now,
            last_accessed: now,
            access_count: 0,
            version: 1,
            revision_operation: RevisionOperation::Create,
            revision_reason: None,
            revision_causation_id: None,
            superseded_by: None,
            provenance,
            metadata: self.metadata,
            namespace,
            archived: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_procedural_record() {
        let agent = AgentId::new("agent_a").unwrap();
        let record = ProceduralRecord::builder()
            .name("deploy-to-staging")
            .description("Deploy the current branch to the staging environment")
            .steps(vec![ActionStep {
                description: "Run tests".into(),
                tool: Some("cargo_test".into()),
                parameters: Metadata::new(),
            }])
            .preconditions(vec!["branch is clean".into()])
            .agent_id(agent)
            .build()
            .unwrap();

        assert_eq!(record.name, "deploy-to-staging");
        assert!(record.success_rate.abs() < f32::EPSILON);
        assert_eq!(record.steps.len(), 1);
    }

    #[test]
    fn success_rate_tracking() {
        let agent = AgentId::new("agent_a").unwrap();
        let mut record = ProceduralRecord::builder()
            .name("test-proc")
            .description("A test procedure")
            .agent_id(agent)
            .build()
            .unwrap();

        record.record_success();
        record.record_success();
        record.record_failure();
        assert_eq!(record.invocation_count, 3);
        assert_eq!(record.success_count, 2);
        // F-81: EMA with α=0.1: 0.0 → 0.1 → 0.19 → 0.171
        assert!((record.success_rate - 0.171).abs() < 0.001);
    }

    #[test]
    fn rejects_empty_name() {
        let agent = AgentId::new("agent_a").unwrap();
        let result = ProceduralRecord::builder()
            .name("")
            .description("desc")
            .agent_id(agent)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_attaches_evidence_links_to_provenance() {
        let agent = AgentId::new("agent_a").unwrap();
        let link = EvidenceLink::new(
            crate::resource::ResourceId::new(),
            crate::resource::EvidenceRole::Output,
        );
        let record = ProceduralRecord::builder()
            .name("proc")
            .description("desc")
            .agent_id(agent)
            .evidence_link(link.clone())
            .build()
            .unwrap();

        assert_eq!(record.provenance.evidence_links, vec![link]);
    }
}
