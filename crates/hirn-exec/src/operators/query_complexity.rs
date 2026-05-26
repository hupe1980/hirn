//! `QueryComplexityExec` — classifies query complexity for depth scheduling.
//!
//! Classification rules (all thresholds configurable):
//! - Token count > 50: +1 point
//! - Temporal keywords present: +1 point
//! - Entity count > 3: +1 point
//! - Multi-hop requested (EXPAND GRAPH DEPTH > 1): +1 point
//! - Causal query (FOLLOW CAUSES): +1 point
//! - Iterative mode: +1 point
//!
//! Simple (0 pts) / Medium (1–2 pts) / Complex (3+ pts).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use arrow_array::{RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::Result;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties};

/// Complexity level for depth scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    Simple,
    Medium,
    Complex,
}

impl fmt::Display for Complexity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Simple => write!(f, "Simple"),
            Self::Medium => write!(f, "Medium"),
            Self::Complex => write!(f, "Complex"),
        }
    }
}

/// Configuration for complexity classification thresholds.
#[derive(Debug, Clone)]
pub struct ComplexityConfig {
    /// Token count threshold (default: 50).
    pub token_threshold: usize,
    /// Entity count threshold (default: 3).
    pub entity_threshold: usize,
    /// Medium threshold: >= this many points (default: 1).
    pub medium_threshold: u32,
    /// Complex threshold: >= this many points (default: 3).
    pub complex_threshold: u32,
}

impl Default for ComplexityConfig {
    fn default() -> Self {
        Self {
            token_threshold: 50,
            entity_threshold: 3,
            medium_threshold: 1,
            complex_threshold: 3,
        }
    }
}

/// Features extracted from a query for complexity classification.
#[derive(Debug, Clone, Default)]
pub struct QueryFeatures {
    /// Approximate token count of the query text.
    pub token_count: usize,
    /// Whether temporal keywords are present (AFTER, BEFORE, BETWEEN, AS OF).
    pub has_temporal: bool,
    /// Number of entities referenced (INVOLVING clause count).
    pub entity_count: usize,
    /// Graph expansion depth (0 = no expansion).
    pub graph_depth: u32,
    /// Whether FOLLOW CAUSES is present.
    pub has_causal: bool,
    /// Whether iterative mode is requested.
    pub is_iterative: bool,
}

impl QueryFeatures {
    /// Classify query complexity based on features and config.
    pub fn classify(&self, config: &ComplexityConfig) -> (Complexity, u32) {
        let mut points: u32 = 0;
        if self.token_count > config.token_threshold {
            points += 1;
        }
        if self.has_temporal {
            points += 1;
        }
        if self.entity_count > config.entity_threshold {
            points += 1;
        }
        if self.graph_depth > 1 {
            points += 1;
        }
        if self.has_causal {
            points += 1;
        }
        if self.is_iterative {
            points += 1;
        }

        let complexity = if points >= config.complex_threshold {
            Complexity::Complex
        } else if points >= config.medium_threshold {
            Complexity::Medium
        } else {
            Complexity::Simple
        };

        (complexity, points)
    }
}

/// Output schema: `query_complexity (Utf8)`, `complexity_points (UInt32)`.
fn output_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("query_complexity", DataType::Utf8, false),
        Field::new("complexity_points", DataType::UInt32, false),
    ]))
}

/// DataFusion operator that classifies query complexity for depth scheduling.
///
/// This is a leaf operator (no children) — it computes classification from
/// `QueryFeatures` provided at construction time.
#[derive(Debug)]
pub struct QueryComplexityExec {
    features: QueryFeatures,
    config: ComplexityConfig,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl QueryComplexityExec {
    pub fn new(features: QueryFeatures, config: ComplexityConfig) -> Self {
        let schema = output_schema();
        let properties = PlanProperties::new(
            datafusion_physical_expr::EquivalenceProperties::new(schema.clone()),
            datafusion_physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Self {
            features,
            config,
            schema,
            properties,
        }
    }

    pub fn features(&self) -> &QueryFeatures {
        &self.features
    }

    pub fn config(&self) -> &ComplexityConfig {
        &self.config
    }
}

impl DisplayAs for QueryComplexityExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (complexity, points) = self.features.classify(&self.config);
        write!(
            f,
            "QueryComplexityExec: complexity={complexity}, points={points}"
        )
    }
}

impl ExecutionPlan for QueryComplexityExec {
    fn name(&self) -> &str {
        "QueryComplexityExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(Self::new(
            self.features.clone(),
            self.config.clone(),
        )))
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let (complexity, points) = self.features.classify(&self.config);

        let batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![complexity.to_string()])),
                Arc::new(UInt32Array::from(vec![points])),
            ],
        )?;

        let schema = self.schema.clone();
        let stream = futures::stream::once(async move { Ok(batch) });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_query() {
        let features = QueryFeatures {
            token_count: 5,
            ..Default::default()
        };
        let config = ComplexityConfig::default();
        let (complexity, points) = features.classify(&config);
        assert_eq!(complexity, Complexity::Simple);
        assert_eq!(points, 0);
    }

    #[test]
    fn medium_query_temporal() {
        let features = QueryFeatures {
            token_count: 10,
            has_temporal: true,
            ..Default::default()
        };
        let config = ComplexityConfig::default();
        let (complexity, points) = features.classify(&config);
        assert_eq!(complexity, Complexity::Medium);
        assert_eq!(points, 1);
    }

    #[test]
    fn medium_query_graph_depth() {
        let features = QueryFeatures {
            graph_depth: 2,
            ..Default::default()
        };
        let config = ComplexityConfig::default();
        let (complexity, points) = features.classify(&config);
        assert_eq!(complexity, Complexity::Medium);
        assert_eq!(points, 1);
    }

    #[test]
    fn complex_query_all_features() {
        let features = QueryFeatures {
            token_count: 60,
            has_temporal: true,
            entity_count: 5,
            graph_depth: 3,
            has_causal: true,
            is_iterative: true,
        };
        let config = ComplexityConfig::default();
        let (complexity, points) = features.classify(&config);
        assert_eq!(complexity, Complexity::Complex);
        assert_eq!(points, 6);
    }

    #[test]
    fn complex_query_three_features() {
        let features = QueryFeatures {
            has_temporal: true,
            entity_count: 5,
            has_causal: true,
            ..Default::default()
        };
        let config = ComplexityConfig::default();
        let (complexity, points) = features.classify(&config);
        assert_eq!(complexity, Complexity::Complex);
        assert_eq!(points, 3);
    }

    #[test]
    fn custom_thresholds() {
        let features = QueryFeatures {
            token_count: 30,
            has_temporal: true,
            ..Default::default()
        };
        let config = ComplexityConfig {
            token_threshold: 20,
            complex_threshold: 2,
            ..Default::default()
        };
        let (complexity, points) = features.classify(&config);
        assert_eq!(complexity, Complexity::Complex);
        assert_eq!(points, 2);
    }

    #[test]
    fn classification_sub_millisecond() {
        let features = QueryFeatures {
            token_count: 100,
            has_temporal: true,
            entity_count: 10,
            graph_depth: 5,
            has_causal: true,
            is_iterative: true,
        };
        let config = ComplexityConfig::default();
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            std::hint::black_box(features.classify(&config));
        }
        let elapsed = start.elapsed();
        // 10K classifications should take well under 1ms total.
        assert!(elapsed.as_millis() < 10, "too slow: {elapsed:?}");
    }

    #[tokio::test]
    async fn execute_produces_batch() {
        let features = QueryFeatures {
            has_temporal: true,
            has_causal: true,
            entity_count: 5,
            ..Default::default()
        };
        let exec = QueryComplexityExec::new(features, ComplexityConfig::default());
        let ctx = Arc::new(TaskContext::default());
        let mut stream = exec.execute(0, ctx).unwrap();

        use futures::StreamExt;
        let batch = stream.next().await.unwrap().unwrap();
        assert_eq!(batch.num_rows(), 1);

        let complexity = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(complexity.value(0), "Complex");

        let points = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(points.value(0), 3);
    }
}
