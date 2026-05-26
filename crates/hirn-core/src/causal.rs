//! Rich causal edge metadata.
//!
//! `CausalEdge` captures strength, confidence, evidence count, confounders,
//! provenance, and mechanism for `Causes`/`CausedBy` edges in the property
//! graph. Non-causal edges carry no `CausalEdge`.

use serde::{Deserialize, Serialize};

use crate::id::MemoryId;

/// Rich metadata for a causal graph edge.
///
/// Attached to `Causes` and `CausedBy` [`EdgeRelation`](crate::types::EdgeRelation) variants.
/// Non-causal edges do not carry this data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CausalEdge {
    /// Causal effect magnitude in `[0.0, 1.0]`.
    pub strength: f32,
    /// Certainty in the causal claim `[0.0, 1.0]`.
    pub confidence: f32,
    /// Number of observations supporting this edge.
    pub evidence_count: u32,
    /// Known confounding variables.
    #[serde(default)]
    pub confounders: Vec<String>,
    /// Provenance memory IDs that justify this edge.
    #[serde(default)]
    pub provenance: Vec<MemoryId>,
    /// Described causal mechanism (free text).
    #[serde(default)]
    pub mechanism: String,
}

impl CausalEdge {
    /// Create a causal edge with default metadata, deriving strength from an
    /// existing edge weight.
    #[must_use]
    pub fn from_weight(weight: f32) -> Self {
        Self {
            strength: weight.clamp(0.0, 1.0),
            confidence: 0.5,
            evidence_count: 1,
            confounders: Vec::new(),
            provenance: Vec::new(),
            mechanism: String::new(),
        }
    }

    /// Compute `strength × confidence × ln(1 + evidence_count)`.
    ///
    /// This is the per-link causal relevance score used in the composite
    /// scoring formula (ε weight).
    #[must_use]
    pub fn relevance_score(&self) -> f32 {
        self.strength * self.confidence * (1.0 + self.evidence_count as f32).ln()
    }
}

impl Default for CausalEdge {
    fn default() -> Self {
        Self {
            strength: 0.5,
            confidence: 0.5,
            evidence_count: 1,
            confounders: Vec::new(),
            provenance: Vec::new(),
            mechanism: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_weight_clamps_and_defaults() {
        let ce = CausalEdge::from_weight(0.8);
        assert!((ce.strength - 0.8).abs() < f32::EPSILON);
        assert!((ce.confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(ce.evidence_count, 1);
        assert!(ce.confounders.is_empty());
        assert!(ce.provenance.is_empty());
        assert!(ce.mechanism.is_empty());

        // Clamps above 1.0.
        let ce2 = CausalEdge::from_weight(1.5);
        assert!((ce2.strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn relevance_score_formula() {
        let ce = CausalEdge {
            strength: 0.8,
            confidence: 0.9,
            evidence_count: 5,
            ..Default::default()
        };
        // 0.8 * 0.9 * ln(6) ≈ 0.72 * 1.7918 ≈ 1.290
        let expected = 0.8 * 0.9 * (6.0_f32).ln();
        let actual = ce.relevance_score();
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn relevance_score_single_evidence() {
        let ce = CausalEdge::default();
        // 0.5 * 0.5 * ln(2) ≈ 0.25 * 0.693 ≈ 0.173
        let expected = 0.5 * 0.5 * (2.0_f32).ln();
        let actual = ce.relevance_score();
        assert!(
            (actual - expected).abs() < 1e-5,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn serde_round_trip() {
        let ce = CausalEdge {
            strength: 0.7,
            confidence: 0.85,
            evidence_count: 3,
            confounders: vec!["age".into(), "diet".into()],
            provenance: vec![MemoryId::new(), MemoryId::new()],
            mechanism: "dopamine pathway".into(),
        };
        let json = serde_json::to_string(&ce).unwrap();
        let ce2: CausalEdge = serde_json::from_str(&json).unwrap();
        assert_eq!(ce, ce2);
    }

    #[test]
    fn default_values() {
        let ce = CausalEdge::default();
        assert!((ce.strength - 0.5).abs() < f32::EPSILON);
        assert!((ce.confidence - 0.5).abs() < f32::EPSILON);
        assert_eq!(ce.evidence_count, 1);
    }
}
