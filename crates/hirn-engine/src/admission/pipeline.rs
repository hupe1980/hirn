//! Admission pipeline: ordered chain of controllers with short-circuit.

use hirn_core::HirnResult;

use super::{AdmissionController, AdmissionDecision, ControllerVerdict, MemoryCandidate};

/// Result of running the full admission pipeline.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The final decision.
    pub decision: AdmissionDecision,
    /// Verdicts from every controller that was consulted (in order).
    pub verdicts: Vec<ControllerVerdict>,
}

/// Ordered pipeline of admission controllers.
///
/// Evaluates each controller in sequence. Short-circuits on the first
/// non-Accept decision (Reject, Defer, or Merge).
pub struct AdmissionPipeline {
    controllers: Vec<Box<dyn AdmissionController>>,
}

impl AdmissionPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            controllers: Vec::new(),
        }
    }

    /// Add a controller to the end of the pipeline.
    pub fn add(&mut self, controller: impl AdmissionController + 'static) {
        self.controllers.push(Box::new(controller));
    }

    /// Convenience builder — add a controller and return self.
    pub fn with(mut self, controller: impl AdmissionController + 'static) -> Self {
        self.add(controller);
        self
    }

    /// Number of controllers in the pipeline.
    pub fn len(&self) -> usize {
        self.controllers.len()
    }

    /// Whether the pipeline has no controllers.
    pub fn is_empty(&self) -> bool {
        self.controllers.is_empty()
    }

    /// Evaluate a candidate through all controllers.
    ///
    /// Short-circuits on the first non-Accept decision.
    pub async fn evaluate(&self, candidate: &MemoryCandidate) -> HirnResult<PipelineResult> {
        let mut verdicts = Vec::with_capacity(self.controllers.len());
        let mut final_importance_override: Option<f32> = None;

        for controller in &self.controllers {
            let decision = controller.evaluate(candidate).await?;
            let name = controller.name().to_string();

            match &decision {
                AdmissionDecision::Accept {
                    importance_override,
                } => {
                    // Track the latest importance override (last one wins).
                    if importance_override.is_some() {
                        final_importance_override = *importance_override;
                    }
                    verdicts.push(ControllerVerdict {
                        controller: name,
                        decision,
                    });
                    // Continue to the next controller.
                }
                _ => {
                    // Short-circuit: Reject, Defer, or Merge.
                    let short_circuit = decision.clone();
                    verdicts.push(ControllerVerdict {
                        controller: name,
                        decision,
                    });
                    return Ok(PipelineResult {
                        decision: short_circuit,
                        verdicts,
                    });
                }
            }
        }

        // All controllers accepted.
        Ok(PipelineResult {
            decision: AdmissionDecision::Accept {
                importance_override: final_importance_override,
            },
            verdicts,
        })
    }
}

impl Default for AdmissionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::admission::MemoryCandidate;
    use hirn_core::id::MemoryId;
    use hirn_core::metadata::Metadata;
    use hirn_core::types::{AgentId, Namespace};

    fn test_candidate() -> MemoryCandidate {
        MemoryCandidate {
            id: MemoryId::new(),
            content: "test memory".to_string(),
            entities: vec![],
            embedding: None,
            agent_id: AgentId::new("test").unwrap(),
            namespace: Namespace::shared(),
            importance: 0.5,
            surprise: 0.5,
            metadata: Metadata::default(),
        }
    }

    /// Always-accept controller.
    struct AcceptAll;

    #[async_trait::async_trait]
    impl AdmissionController for AcceptAll {
        fn name(&self) -> &str {
            "accept_all"
        }
        async fn evaluate(&self, _: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
            Ok(AdmissionDecision::Accept {
                importance_override: None,
            })
        }
    }

    /// Always-reject controller.
    struct RejectAll {
        reason: String,
    }

    #[async_trait::async_trait]
    impl AdmissionController for RejectAll {
        fn name(&self) -> &str {
            "reject_all"
        }
        async fn evaluate(&self, _: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
            Ok(AdmissionDecision::Reject {
                reason: self.reason.clone(),
            })
        }
    }

    /// Accept with importance override.
    struct OverrideImportance(f32);

    #[async_trait::async_trait]
    impl AdmissionController for OverrideImportance {
        fn name(&self) -> &str {
            "override_importance"
        }
        async fn evaluate(&self, _: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
            Ok(AdmissionDecision::Accept {
                importance_override: Some(self.0),
            })
        }
    }

    /// Defer controller.
    struct DeferUntil(i64);

    #[async_trait::async_trait]
    impl AdmissionController for DeferUntil {
        fn name(&self) -> &str {
            "defer"
        }
        async fn evaluate(&self, _: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
            Ok(AdmissionDecision::Defer { until: self.0 })
        }
    }

    /// Merge controller.
    struct MergeInto(MemoryId);

    #[async_trait::async_trait]
    impl AdmissionController for MergeInto {
        fn name(&self) -> &str {
            "merge"
        }
        async fn evaluate(&self, _: &MemoryCandidate) -> HirnResult<AdmissionDecision> {
            Ok(AdmissionDecision::Merge { target: self.0 })
        }
    }

    #[tokio::test]
    async fn single_accept_controller() {
        let pipeline = AdmissionPipeline::new().with(AcceptAll);
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        assert!(result.decision.is_accept());
        assert_eq!(result.verdicts.len(), 1);
        assert_eq!(result.verdicts[0].controller, "accept_all");
    }

    #[tokio::test]
    async fn single_reject_controller() {
        let pipeline = AdmissionPipeline::new().with(RejectAll {
            reason: "too boring".into(),
        });
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        assert!(result.decision.is_reject());
        if let AdmissionDecision::Reject { reason } = &result.decision {
            assert_eq!(reason, "too boring");
        }
    }

    #[tokio::test]
    async fn pipeline_accept_then_reject_short_circuits() {
        let pipeline = AdmissionPipeline::new().with(AcceptAll).with(RejectAll {
            reason: "blocked".into(),
        });
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        assert!(result.decision.is_reject());
        assert_eq!(result.verdicts.len(), 2, "both controllers consulted");
    }

    #[tokio::test]
    async fn pipeline_all_accept() {
        let pipeline = AdmissionPipeline::new().with(AcceptAll).with(AcceptAll);
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        assert!(result.decision.is_accept());
        assert_eq!(result.verdicts.len(), 2);
    }

    #[tokio::test]
    async fn defer_short_circuits() {
        let pipeline = AdmissionPipeline::new()
            .with(AcceptAll)
            .with(DeferUntil(99999));
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        assert!(matches!(
            result.decision,
            AdmissionDecision::Defer { until: 99999 }
        ));
    }

    #[tokio::test]
    async fn merge_short_circuits() {
        let target = MemoryId::new();
        let pipeline = AdmissionPipeline::new()
            .with(AcceptAll)
            .with(MergeInto(target));
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        if let AdmissionDecision::Merge { target: t } = result.decision {
            assert_eq!(t, target);
        } else {
            panic!("expected Merge decision");
        }
    }

    #[tokio::test]
    async fn verdict_log_shows_all_consulted() {
        let pipeline = AdmissionPipeline::new()
            .with(AcceptAll)
            .with(OverrideImportance(0.9))
            .with(RejectAll {
                reason: "nope".into(),
            });
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        // Short-circuits at third controller.
        assert_eq!(result.verdicts.len(), 3);
        assert_eq!(result.verdicts[0].controller, "accept_all");
        assert_eq!(result.verdicts[1].controller, "override_importance");
        assert_eq!(result.verdicts[2].controller, "reject_all");
    }

    #[tokio::test]
    async fn empty_pipeline_accepts() {
        let pipeline = AdmissionPipeline::new();
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        assert!(result.decision.is_accept());
        assert!(result.verdicts.is_empty());
    }

    #[tokio::test]
    async fn importance_override_from_last_controller() {
        let pipeline = AdmissionPipeline::new()
            .with(OverrideImportance(0.3))
            .with(OverrideImportance(0.9));
        let result = pipeline.evaluate(&test_candidate()).await.unwrap();
        if let AdmissionDecision::Accept {
            importance_override,
        } = result.decision
        {
            assert_eq!(importance_override, Some(0.9));
        } else {
            panic!("expected Accept");
        }
    }
}
