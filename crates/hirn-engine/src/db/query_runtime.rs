use std::sync::Arc;

use datafusion::prelude::SessionContext;
use hirn_core::HirnConfig;
use hirn_core::HirnResult;
use hirn_core::embed::Embedder;
use hirn_core::tokenizer::Tokenizer;
use hirn_storage::PhysicalStore;

use crate::cached_graph_store::CachedGraphStore;

pub(crate) struct QueryRuntime {
    session: SessionContext,
    query_pipeline: hirn_query::QueryPipeline,
    plan_cache: Arc<hirn_query::PlanCache>,
}

impl QueryRuntime {
    pub(crate) fn new(
        cached_graph: &CachedGraphStore,
        config: &HirnConfig,
        storage: Arc<dyn PhysicalStore>,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> HirnResult<Self> {
        let session = Self::build_session();
        Self::register_session_extensions(
            &session,
            cached_graph,
            config,
            storage,
            None,
            tokenizer,
        )?;

        let plan_cache = Arc::new(hirn_query::PlanCache::new(1024));
        let query_pipeline = {
            let plan_cache = Arc::clone(&plan_cache);
            hirn_query::QueryPipeline::new(hirn_query::AnalyzeContext::default())
                .with_cache(plan_cache)
        };

        Ok(Self {
            session,
            query_pipeline,
            plan_cache,
        })
    }

    pub(crate) fn session(&self) -> &SessionContext {
        &self.session
    }

    pub(crate) fn query_pipeline(&self) -> &hirn_query::QueryPipeline {
        &self.query_pipeline
    }

    pub(crate) fn plan_cache(&self) -> &Arc<hirn_query::PlanCache> {
        &self.plan_cache
    }

    pub(crate) fn register_runtime_state(
        &self,
        cached_graph: &CachedGraphStore,
        config: &HirnConfig,
        storage: Arc<dyn PhysicalStore>,
        embedder: Option<Arc<dyn Embedder>>,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> HirnResult<()> {
        Self::register_session_extensions(
            &self.session,
            cached_graph,
            config,
            storage,
            embedder,
            tokenizer,
        )
    }

    fn build_session() -> SessionContext {
        use datafusion::execution::SessionStateBuilder;

        let mut builder = SessionStateBuilder::new_with_default_features()
            .with_query_planner(Arc::new(hirn_exec::HirnQueryPlanner));
        for rule in hirn_exec::rules::all_rules() {
            builder = builder.with_physical_optimizer_rule(rule);
        }
        let state = builder.build();
        SessionContext::new_with_state(state)
    }

    fn register_session_extensions(
        session: &SessionContext,
        cached_graph: &CachedGraphStore,
        config: &HirnConfig,
        storage: Arc<dyn PhysicalStore>,
        embedder: Option<Arc<dyn Embedder>>,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> HirnResult<()> {
        hirn_exec::HirnSessionExt::new(
            cached_graph.hot_arc() as Arc<dyn std::any::Any + Send + Sync>,
            Arc::new(config.clone()),
            embedder,
        )
        .with_graph_read_runtime(
            Arc::new(cached_graph.clone()) as Arc<dyn hirn_exec::GraphReadRuntime>
        )
        .with_storage(storage)
        .with_tokenizer(tokenizer)
        .register(session)
        .map_err(|e| {
            hirn_core::HirnError::storage(format!("Failed to register HirnSessionExt: {e}"))
        })?;

        hirn_exec::udfs::register_all_udfs(session);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use hirn_core::embed::{Embedder, Embedding, MultivectorEmbedding};
    use hirn_core::tokenizer::EstimatingTokenizer;
    use hirn_storage::memory_store::MemoryStore;

    struct TestEmbedder;

    #[async_trait]
    impl Embedder for TestEmbedder {
        async fn embed(&self, texts: &[&str]) -> HirnResult<Vec<Embedding>> {
            Ok(texts
                .iter()
                .map(|text| Embedding {
                    vector: vec![text.len() as f32; 4],
                    model_id: self.model_id().to_owned(),
                })
                .collect())
        }

        fn dimensions(&self) -> usize {
            4
        }

        fn model_id(&self) -> &str {
            "query-runtime-test"
        }

        fn max_input_tokens(&self) -> usize {
            8_192
        }

        async fn embed_multivec(&self, texts: &[&str]) -> HirnResult<Vec<MultivectorEmbedding>> {
            Ok(texts
                .iter()
                .map(|_| MultivectorEmbedding {
                    vectors: vec![vec![1.0; 4]],
                    model_id: self.model_id().to_owned(),
                })
                .collect())
        }

        fn supports_multivec(&self) -> bool {
            true
        }
    }

    #[test]
    fn new_runtime_registers_session_extension_and_empty_cache() {
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let graph_runtime = crate::db::graph_runtime::GraphRuntime::new(Arc::clone(&storage));
        let config = HirnConfig::default();
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);

        let runtime =
            QueryRuntime::new(graph_runtime.cached_graph(), &config, storage, tokenizer).unwrap();

        let ext = hirn_exec::HirnSessionExt::get(runtime.session())
            .expect("session extension should be registered");
        assert!(ext.storage().is_some());
        assert!(ext.graph_read_runtime().is_some());
        assert!(ext.embedder().is_none());
        assert!(ext.tokenizer().is_some());
        assert!(runtime.plan_cache().is_empty());
    }

    #[test]
    fn register_runtime_state_updates_session_extension() {
        let storage: Arc<dyn PhysicalStore> = Arc::new(MemoryStore::new());
        let graph_runtime = crate::db::graph_runtime::GraphRuntime::new(Arc::clone(&storage));
        let config = HirnConfig::default();
        let tokenizer: Arc<dyn Tokenizer> = Arc::new(EstimatingTokenizer);

        let runtime = QueryRuntime::new(
            graph_runtime.cached_graph(),
            &config,
            Arc::clone(&storage),
            Arc::clone(&tokenizer),
        )
        .unwrap();

        runtime
            .register_runtime_state(
                graph_runtime.cached_graph(),
                &config,
                storage,
                Some(Arc::new(TestEmbedder)),
                tokenizer,
            )
            .unwrap();

        let ext = hirn_exec::HirnSessionExt::get(runtime.session())
            .expect("session extension should remain registered");
        assert!(ext.graph_read_runtime().is_some());
        let embedder = ext.embedder().expect("embedder should be present");
        assert_eq!(embedder.model_id(), "query-runtime-test");
        assert!(ext.tokenizer().is_some());
    }
}
