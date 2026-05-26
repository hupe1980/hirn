use std::fmt;

use hirn_core::types::{AgentId, Namespace};
use hirn_core::{HirnError, MemoryId, TextRetention};

use crate::admission::AdmissionDecision;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RememberStatus {
    Accepted,
    Rejected,
    Deferred,
    Merged,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDisposition {
    Provided,
    Generated,
    PendingRetry,
    Missing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WritePathOperationStatus {
    Disabled,
    SkippedFastPath,
    Unavailable,
    Applied,
}

#[derive(Debug, Clone)]
pub struct AdmissionExplanation {
    pub decision: AdmissionDecision,
    pub controllers_consulted: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RpeExplanation {
    pub enabled: bool,
    pub score: Option<f32>,
    pub max_similarity: Option<f32>,
    pub threshold: f32,
    pub is_fast_path: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WritePathOperationExplanation {
    pub status: WritePathOperationStatus,
    pub count: usize,
}

impl WritePathOperationExplanation {
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            status: WritePathOperationStatus::Disabled,
            count: 0,
        }
    }

    #[must_use]
    pub const fn skipped_fast_path() -> Self {
        Self {
            status: WritePathOperationStatus::SkippedFastPath,
            count: 0,
        }
    }

    #[must_use]
    pub const fn unavailable() -> Self {
        Self {
            status: WritePathOperationStatus::Unavailable,
            count: 0,
        }
    }

    #[must_use]
    pub const fn applied(count: usize) -> Self {
        Self {
            status: WritePathOperationStatus::Applied,
            count,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterferenceDisposition {
    None,
    TriggerConsolidation {
        namespaces: Vec<Namespace>,
        backlog_score: f32,
        cause: &'static str,
    },
    Suppressed {
        reason: &'static str,
        backlog_score: f32,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct InterferenceExplanation {
    pub score: f32,
    pub disposition: InterferenceDisposition,
}

#[derive(Debug, Clone)]
pub struct RememberExplanation {
    pub status: RememberStatus,
    pub actor_id: AgentId,
    pub namespace: Namespace,
    pub bypass_admission: bool,
    pub memory_id: Option<MemoryId>,
    pub admission: Option<AdmissionExplanation>,
    pub embedding: EmbeddingDisposition,
    pub rpe: Option<RpeExplanation>,
    pub text_retention: TextRetention,
    pub resources_extracted: bool,
    pub prospective_indexing: WritePathOperationExplanation,
    pub svo_extraction: WritePathOperationExplanation,
    pub interference: Option<InterferenceExplanation>,
    pub arrival_sequence: Option<u64>,
    pub error: Option<String>,
}

impl RememberExplanation {
    #[must_use]
    pub(crate) fn new(
        actor_id: AgentId,
        namespace: Namespace,
        bypass_admission: bool,
        embedding: EmbeddingDisposition,
        text_retention: TextRetention,
    ) -> Self {
        Self {
            status: RememberStatus::Accepted,
            actor_id,
            namespace,
            bypass_admission,
            memory_id: None,
            admission: None,
            embedding,
            rpe: None,
            text_retention,
            resources_extracted: false,
            prospective_indexing: WritePathOperationExplanation::disabled(),
            svo_extraction: WritePathOperationExplanation::disabled(),
            interference: None,
            arrival_sequence: None,
            error: None,
        }
    }
}

#[derive(Debug)]
pub struct RememberFailure {
    pub error: HirnError,
    pub explanation: RememberExplanation,
}

impl RememberFailure {
    #[must_use]
    pub(crate) fn new(error: HirnError, explanation: RememberExplanation) -> Self {
        Self { error, explanation }
    }
}

impl fmt::Display for RememberFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for RememberFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}
