//! `rpe_score` UDF — reward prediction error for admission gating.
//!
//! `rpe_score(max_similarity: Float32, novelty_zscore: Float32) → Float32`

use std::any::Any;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::Float32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use arrow_schema::DataType;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct RpeScoreUdf {
    signature: Signature,
}

impl Default for RpeScoreUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl RpeScoreUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float32, DataType::Float32],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for RpeScoreUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "rpe_score"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let num_rows = args.number_rows;
        let arrays: Vec<_> = args
            .args
            .iter()
            .map(|a| a.to_array(num_rows))
            .collect::<Result<Vec<_>>>()?;

        let max_sim = arrays[0].as_primitive::<Float32Type>();
        let novelty = arrays[1].as_primitive::<Float32Type>();

        let len = max_sim.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if max_sim.is_null(i) || novelty.is_null(i) {
                results.push(None);
                continue;
            }

            let sim = max_sim.value(i);
            let nov = novelty.value(i);

            // RPE = (1.0 - max_similarity) × (1.0 + novelty_zscore), clamped to [0.0, 2.0]
            let rpe = ((1.0 - sim) * (1.0 + nov)).clamp(0.0, 2.0);
            results.push(Some(rpe));
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;

    fn invoke(sim: &[f32], nov: &[f32]) -> Float32Array {
        let udf = RpeScoreUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Float32Array::from(sim.to_vec()))),
                ColumnarValue::Array(Arc::new(Float32Array::from(nov.to_vec()))),
            ],
            number_rows: sim.len(),
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        match result {
            ColumnarValue::Array(a) => a.as_any().downcast_ref::<Float32Array>().unwrap().clone(),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn high_similarity_low_novelty_low_rpe() {
        let vals = invoke(&[0.95], &[0.1]);
        // RPE = (1 - 0.95) * (1 + 0.1) = 0.05 * 1.1 = 0.055
        assert!(vals.value(0) < 0.1, "got {}", vals.value(0));
    }

    #[test]
    fn low_similarity_high_novelty_high_rpe() {
        let vals = invoke(&[0.1], &[1.5]);
        // RPE = (1 - 0.1) * (1 + 1.5) = 0.9 * 2.5 = 2.25 → clamped to 2.0
        assert!((vals.value(0) - 2.0).abs() < 1e-6, "got {}", vals.value(0));
    }

    #[test]
    fn boundary_values() {
        let vals = invoke(&[0.0, 1.0], &[0.0, 0.0]);
        // RPE(0, 0) = 1.0 * 1.0 = 1.0
        assert!((vals.value(0) - 1.0).abs() < 1e-6);
        // RPE(1, 0) = 0.0 * 1.0 = 0.0
        assert!((vals.value(1) - 0.0).abs() < 1e-6);
    }
}
