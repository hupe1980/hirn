//! `temporal_decay` UDF — Ebbinghaus-modulated decay scoring.
//!
//! `temporal_decay(age_hours, base_rate, access_freq, importance) → Float32`

use std::any::Any;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use arrow_array::{ArrayRef, Float32Array};
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use arrow_schema::DataType;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TemporalDecayUdf {
    signature: Signature,
}

impl Default for TemporalDecayUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalDecayUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::Float64, // age_hours
                    DataType::Float64, // base_rate
                    DataType::UInt64,  // access_freq
                    DataType::Float32, // importance
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for TemporalDecayUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "temporal_decay"
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

        let age_hours = arrays[0].as_primitive::<arrow_array::types::Float64Type>();
        let base_rate = arrays[1].as_primitive::<arrow_array::types::Float64Type>();
        let access_freq = arrays[2].as_primitive::<arrow_array::types::UInt64Type>();
        let importance = arrays[3].as_primitive::<Float32Type>();

        let len = age_hours.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if age_hours.is_null(i)
                || base_rate.is_null(i)
                || access_freq.is_null(i)
                || importance.is_null(i)
            {
                results.push(None);
                continue;
            }

            let age = age_hours.value(i);
            let base = base_rate.value(i);
            let freq = access_freq.value(i) as f64;
            let imp = importance.value(i) as f64;

            // FadeMem formula: decay_rate = base × (1 / (1 + importance)) × (1 / (1 + access_freq))
            let decay_rate = base * (1.0 / (1.0 + imp)) * (1.0 / (1.0 + freq));
            let score = (-decay_rate * age).exp() as f32;

            results.push(Some(score));
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, UInt64Array};
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;

    fn invoke(age: &[f64], base: &[f64], freq: &[u64], imp: &[f32]) -> Float32Array {
        let udf = TemporalDecayUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Float64Array::from(age.to_vec()))),
                ColumnarValue::Array(Arc::new(Float64Array::from(base.to_vec()))),
                ColumnarValue::Array(Arc::new(UInt64Array::from(freq.to_vec()))),
                ColumnarValue::Array(Arc::new(Float32Array::from(imp.to_vec()))),
            ],
            number_rows: age.len(),
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
    fn recently_accessed_high_importance_near_one() {
        let vals = invoke(&[1.0], &[0.1], &[10], &[0.9]);
        // decay_rate = 0.1 * (1/(1+0.9)) * (1/(1+10)) ≈ 0.00478
        // score = exp(-0.00478 * 1.0) ≈ 0.9952
        assert!(vals.value(0) > 0.99, "got {}", vals.value(0));
    }

    #[test]
    fn old_low_importance_near_zero() {
        let vals = invoke(&[10000.0], &[0.1], &[0], &[0.0]);
        // decay_rate = 0.1 * 1.0 * 1.0 = 0.1
        // score = exp(-0.1 * 10000) ≈ 0.0
        assert!(vals.value(0) < 0.01, "got {}", vals.value(0));
    }

    #[test]
    fn access_freq_modulates() {
        let low_freq = invoke(&[100.0], &[0.1], &[0], &[0.5]);
        let high_freq = invoke(&[100.0], &[0.1], &[50], &[0.5]);
        // Higher access frequency → lower decay rate → higher score
        assert!(high_freq.value(0) > low_freq.value(0));
    }

    #[test]
    fn zero_age_returns_one() {
        let vals = invoke(&[0.0], &[0.5], &[5], &[0.5]);
        assert!((vals.value(0) - 1.0).abs() < 1e-6);
    }
}
