use std::sync::Arc;

use parking_lot::Mutex;

use hirn_core::episodic::EpisodicRecord;
use hirn_core::tokenizer::Tokenizer;
use hirn_core::types::AgentId;
use hirn_core::{HirnConfig, HirnResult};
use hirn_storage::PhysicalStore;

use crate::admission::{AdmissionPipeline, MemoryCandidate, PipelineResult};
use crate::security::{CorruptionDefense, CorruptionDefenseConfig};

pub(crate) struct AdmissionRuntime {
    corruption_defense: Mutex<CorruptionDefense>,
    // Arc so the write path can hold a reservation guard (commit/release)
    // that outlives the borrow of the runtime.
    admission_pipeline: Option<Arc<AdmissionPipeline>>,
}

impl AdmissionRuntime {
    pub(crate) fn new() -> Self {
        Self {
            corruption_defense: Mutex::new(CorruptionDefense::default()),
            admission_pipeline: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_corruption_defense(config: CorruptionDefenseConfig) -> Self {
        Self {
            corruption_defense: Mutex::new(CorruptionDefense::new(config)),
            admission_pipeline: None,
        }
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: AdmissionPipeline) {
        self.admission_pipeline = Some(Arc::new(pipeline));
    }

    pub(crate) fn setup_default_pipeline(
        &mut self,
        config: &HirnConfig,
        storage: Arc<dyn PhysicalStore>,
        tokenizer: Arc<dyn Tokenizer>,
    ) {
        if !config.admission_enabled {
            return;
        }

        use crate::admission::controllers::{
            duplicate::{DuplicateAction, DuplicateDetector},
            poisoning::PoisoningGate,
            rate_limiter::RateLimiter,
            surprise::SurpriseGate,
            token_budget::TokenBudgetGate,
            trust::TrustGate,
        };
        use hirn_core::config::{AdmissionDuplicateAction, AdmissionPoisoningAction};

        let action = match config.admission_duplicate_action {
            AdmissionDuplicateAction::Merge => DuplicateAction::Merge,
            AdmissionDuplicateAction::Reject => DuplicateAction::Reject,
        };

        let mut pipeline = AdmissionPipeline::new();

        // Cheap checks first: trust (in-process scoring + tiny agent-table
        // scan) and the pure-CPU poisoning scan run before the
        // embedding/vector-search controllers.
        if config.admission_min_trust > 0.0 || config.admission_trust_quarantine_below.is_some() {
            pipeline.add(TrustGate::new(
                storage.clone(),
                config.admission_min_trust,
                config.admission_trust_quarantine_below,
            ));
        }
        if config.admission_poisoning_action != AdmissionPoisoningAction::Off {
            pipeline.add(PoisoningGate::new(config.admission_poisoning_action));
        }

        let pipeline = pipeline
            .with(SurpriseGate::new(
                storage.clone(),
                "episodic",
                config.admission_surprise_threshold,
            ))
            .with(DuplicateDetector::new(
                storage.clone(),
                "episodic",
                1.0 - config.admission_duplicate_threshold,
                action,
            ))
            .with(TokenBudgetGate::new_cognitive(
                storage,
                tokenizer,
                config.admission_token_budget_limit as usize,
            ))
            .with(RateLimiter::new(config.admission_rate_limit as u64, 60));

        self.admission_pipeline = Some(Arc::new(pipeline));
    }

    pub(crate) fn admission_pipeline(&self) -> Option<&AdmissionPipeline> {
        self.admission_pipeline.as_deref()
    }

    /// Owned handle for reservation guards on the write path.
    pub(crate) fn admission_pipeline_arc(&self) -> Option<Arc<AdmissionPipeline>> {
        self.admission_pipeline.clone()
    }

    pub(crate) async fn evaluate_record(
        &self,
        record: &EpisodicRecord,
    ) -> HirnResult<Option<PipelineResult>> {
        let Some(pipeline) = self.admission_pipeline.as_ref() else {
            return Ok(None);
        };

        let candidate = MemoryCandidate::from_record(record);
        pipeline.evaluate(&candidate).await.map(Some)
    }

    pub(crate) fn rate_limit_config(&self, agent_id: &AgentId) -> Option<CorruptionDefenseConfig> {
        let defense = self.corruption_defense.lock();
        if defense.is_rate_limited(agent_id) {
            Some(defense.config().clone())
        } else {
            None
        }
    }

    pub(crate) fn record_quarantine(&self, agent_id: &AgentId) -> Option<CorruptionDefenseConfig> {
        let mut defense = self.corruption_defense.lock();
        if defense.record_quarantine(agent_id) {
            Some(defense.config().clone())
        } else {
            None
        }
    }

    pub(crate) fn clear_agent(&self, agent_id: &AgentId) {
        self.corruption_defense.lock().clear_agent(agent_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hirn_core::EstimatingTokenizer;
    use hirn_core::types::AgentId;
    use hirn_storage::memory_store::MemoryStore;

    #[test]
    fn default_runtime_has_no_pipeline() {
        let runtime = AdmissionRuntime::new();
        assert!(runtime.admission_pipeline().is_none());
    }

    #[test]
    fn setup_default_pipeline_installs_all_default_controllers() {
        let mut runtime = AdmissionRuntime::new();
        let mut config = HirnConfig::default();
        config.admission_enabled = true;

        runtime.setup_default_pipeline(
            &config,
            Arc::new(MemoryStore::new()),
            Arc::new(EstimatingTokenizer),
        );

        // Trust and poisoning gates stay uninstalled with default config
        // (admission_min_trust = 0.0, poisoning action = off).
        assert_eq!(
            runtime.admission_pipeline().map(|pipeline| pipeline.len()),
            Some(4)
        );
    }

    #[test]
    fn setup_default_pipeline_installs_trust_and_poisoning_gates_when_configured() {
        let mut runtime = AdmissionRuntime::new();
        let mut config = HirnConfig::default();
        config.admission_enabled = true;
        config.admission_min_trust = 0.4;
        config.admission_poisoning_action = hirn_core::config::AdmissionPoisoningAction::Audit;

        runtime.setup_default_pipeline(
            &config,
            Arc::new(MemoryStore::new()),
            Arc::new(EstimatingTokenizer),
        );

        assert_eq!(
            runtime.admission_pipeline().map(|pipeline| pipeline.len()),
            Some(6)
        );
    }

    #[test]
    fn quarantine_tier_alone_installs_trust_gate() {
        let mut runtime = AdmissionRuntime::new();
        let mut config = HirnConfig::default();
        config.admission_enabled = true;
        config.admission_trust_quarantine_below = Some(0.5);

        runtime.setup_default_pipeline(
            &config,
            Arc::new(MemoryStore::new()),
            Arc::new(EstimatingTokenizer),
        );

        assert_eq!(
            runtime.admission_pipeline().map(|pipeline| pipeline.len()),
            Some(5)
        );
    }

    #[test]
    fn record_quarantine_enters_rate_limited_state() {
        let runtime = AdmissionRuntime::with_corruption_defense(CorruptionDefenseConfig {
            max_quarantines_per_window: 0,
            window_seconds: 300,
        });
        let agent_id = AgentId::new("admission-test").unwrap();

        let config = runtime.record_quarantine(&agent_id);
        assert!(config.is_some());
        assert!(runtime.rate_limit_config(&agent_id).is_some());

        runtime.clear_agent(&agent_id);
        assert!(runtime.rate_limit_config(&agent_id).is_none());
    }
}
