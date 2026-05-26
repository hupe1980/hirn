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
    admission_pipeline: Option<AdmissionPipeline>,
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
        self.admission_pipeline = Some(pipeline);
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
            rate_limiter::RateLimiter,
            surprise::SurpriseGate,
            token_budget::TokenBudgetGate,
        };

        let action = match config.admission_duplicate_action.as_str() {
            "merge" => DuplicateAction::Merge,
            _ => DuplicateAction::Reject,
        };
        let pipeline = AdmissionPipeline::new()
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
            .with(TokenBudgetGate::new(
                storage,
                tokenizer,
                "episodic",
                config.admission_token_budget_limit as usize,
            ))
            .with(RateLimiter::new(config.admission_rate_limit as u64, 60));

        self.admission_pipeline = Some(pipeline);
    }

    pub(crate) fn admission_pipeline(&self) -> Option<&AdmissionPipeline> {
        self.admission_pipeline.as_ref()
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

        assert_eq!(
            runtime.admission_pipeline().map(|pipeline| pipeline.len()),
            Some(4)
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
