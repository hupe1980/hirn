use serde::{Deserialize, Serialize};

/// Logical kind of a record stored in quarantine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuarantinedRecordKind {
    Episodic,
    Semantic,
}
