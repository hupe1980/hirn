//! `composite_score` UDF — weighted scoring formula for memory ranking.
//!
//! `composite_score(similarity, importance, age_hours, activation, causal, surprise) → Float32`

use std::any::Any;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, Float32Array};
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use arrow_schema::DataType;

/// Default scoring weights (should sum to ~1.0).
const DEFAULT_SIMILARITY_W: f32 = 0.35;
const DEFAULT_IMPORTANCE_W: f32 = 0.20;
const DEFAULT_RECENCY_W: f32 = 0.20;
const DEFAULT_ACTIVATION_W: f32 = 0.10;
const DEFAULT_CAUSAL_W: f32 = 0.05;
const DEFAULT_SURPRISE_W: f32 = 0.10;

/// Temporal decay half-life in hours (for age → recency score conversion).
const DECAY_HALF_LIFE_HOURS: f64 = 168.0; // 1 week

/// Precomputed decay rate: `ln(2) / half_life`. Used in the inner loop to avoid
/// a division per row.
const DECAY_RATE: f64 = std::f64::consts::LN_2 / DECAY_HALF_LIFE_HOURS;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CompositeScoreUdf {
    signature: Signature,
}

impl Default for CompositeScoreUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeScoreUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::Float32, // similarity
                    DataType::Float32, // importance
                    DataType::Float64, // age_hours
                    DataType::Float32, // activation
                    DataType::Float32, // causal
                    DataType::Float32, // surprise
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for CompositeScoreUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "composite_score"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let num_rows = args.number_rows;
        let arrays: Vec<ArrayRef> = args
            .args
            .iter()
            .map(|a| a.to_array(num_rows))
            .collect::<Result<Vec<_>>>()?;

        let similarity = arrays[0].as_primitive::<Float32Type>();
        let importance = arrays[1].as_primitive::<Float32Type>();
        let age_hours = arrays[2].as_primitive::<arrow_array::types::Float64Type>();
        let activation = arrays[3].as_primitive::<Float32Type>();
        let causal = arrays[4].as_primitive::<Float32Type>();
        let surprise = arrays[5].as_primitive::<Float32Type>();

        let len = similarity.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if similarity.is_null(i)
                || importance.is_null(i)
                || age_hours.is_null(i)
                || activation.is_null(i)
                || causal.is_null(i)
                || surprise.is_null(i)
            {
                results.push(None);
                continue;
            }

            let sim = similarity.value(i);
            let imp = importance.value(i);
            let age = age_hours.value(i);
            let act = activation.value(i);
            let caus = causal.value(i);
            let surp = surprise.value(i);

            // Exponential decay: recency = exp(-decay_rate * age)
            let recency = ((-DECAY_RATE * age).exp()) as f32;

            let score = DEFAULT_SIMILARITY_W * sim
                + DEFAULT_IMPORTANCE_W * imp
                + DEFAULT_RECENCY_W * recency
                + DEFAULT_ACTIVATION_W * act
                + DEFAULT_CAUSAL_W * caus
                + DEFAULT_SURPRISE_W * surp;

            results.push(Some(score));
        }

        let result = Float32Array::from(results);
        Ok(ColumnarValue::Array(Arc::new(result)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Float64Array;
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;

    fn make_args(
        sim: &[f32],
        imp: &[f32],
        age: &[f64],
        act: &[f32],
        caus: &[f32],
        surp: &[f32],
    ) -> ScalarFunctionArgs {
        ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Float32Array::from(sim.to_vec()))),
                ColumnarValue::Array(Arc::new(Float32Array::from(imp.to_vec()))),
                ColumnarValue::Array(Arc::new(Float64Array::from(age.to_vec()))),
                ColumnarValue::Array(Arc::new(Float32Array::from(act.to_vec()))),
                ColumnarValue::Array(Arc::new(Float32Array::from(caus.to_vec()))),
                ColumnarValue::Array(Arc::new(Float32Array::from(surp.to_vec()))),
            ],
            number_rows: sim.len(),
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        }
    }

    #[test]
    fn known_inputs() {
        let udf = CompositeScoreUdf::new();
        let args = make_args(&[0.9], &[0.8], &[0.0], &[0.5], &[0.3], &[0.1]);
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let vals = arr.as_primitive::<Float32Type>();
        // recency = exp(0) = 1.0
        // score = 0.35*0.9 + 0.20*0.8 + 0.20*1.0 + 0.10*0.5 + 0.05*0.3 + 0.10*0.1
        //       = 0.315 + 0.16 + 0.20 + 0.05 + 0.015 + 0.01 = 0.75
        let expected = 0.315 + 0.16 + 0.20 + 0.05 + 0.015 + 0.01;
        assert!(
            (vals.value(0) - expected).abs() < 1e-5,
            "got {}",
            vals.value(0)
        );
    }

    #[test]
    fn null_handling() {
        let udf = CompositeScoreUdf::new();
        let sim = Float32Array::from(vec![Some(0.9), None]);
        let imp = Float32Array::from(vec![Some(0.8), Some(0.5)]);
        let age = Float64Array::from(vec![Some(0.0), Some(1.0)]);
        let act = Float32Array::from(vec![Some(0.5), Some(0.3)]);
        let caus = Float32Array::from(vec![Some(0.3), Some(0.2)]);
        let surp = Float32Array::from(vec![Some(0.1), Some(0.1)]);
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(sim)),
                ColumnarValue::Array(Arc::new(imp)),
                ColumnarValue::Array(Arc::new(age)),
                ColumnarValue::Array(Arc::new(act)),
                ColumnarValue::Array(Arc::new(caus)),
                ColumnarValue::Array(Arc::new(surp)),
            ],
            number_rows: 2,
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let vals = arr.as_primitive::<Float32Type>();
        assert!(!vals.is_null(0));
        assert!(vals.is_null(1));
    }

    #[test]
    fn batch_of_1000() {
        let udf = CompositeScoreUdf::new();
        let n = 1000;
        let sim: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let imp: Vec<f32> = vec![0.5; n];
        let age: Vec<f64> = vec![24.0; n];
        let act: Vec<f32> = vec![0.3; n];
        let caus: Vec<f32> = vec![0.1; n];
        let surp: Vec<f32> = vec![0.05; n];
        let args = make_args(&sim, &imp, &age, &act, &caus, &surp);
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        assert_eq!(arr.len(), n);
        assert_eq!(arr.null_count(), 0);
    }

    /// Throughput test: 100K rows exercising the hot scoring path.
    /// Verifies correctness at scale — monotonically increasing similarity
    /// should produce monotonically increasing scores (when all other inputs
    /// are constant).
    #[test]
    fn throughput_100k_rows() {
        let udf = CompositeScoreUdf::new();
        let n = 100_000;
        let sim: Vec<f32> = (0..n).map(|i| i as f32 / n as f32).collect();
        let imp: Vec<f32> = vec![0.5; n];
        let age: Vec<f64> = vec![48.0; n]; // 2 days old
        let act: Vec<f32> = vec![0.4; n];
        let caus: Vec<f32> = vec![0.2; n];
        let surp: Vec<f32> = vec![0.1; n];
        let args = make_args(&sim, &imp, &age, &act, &caus, &surp);
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a,
            _ => panic!("expected array"),
        };
        let vals = arr.as_primitive::<Float32Type>();
        assert_eq!(vals.len(), n);
        assert_eq!(vals.null_count(), 0);

        // With constant inputs except increasing similarity, scores must
        // be monotonically non-decreasing.
        for i in 1..n {
            assert!(
                vals.value(i) >= vals.value(i - 1),
                "score at {} ({}) < score at {} ({})",
                i,
                vals.value(i),
                i - 1,
                vals.value(i - 1)
            );
        }
    }
}
