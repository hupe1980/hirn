//! Remaining 3 UDFs: `surprise_score`, `fade_mem_decay`, `causal_relevance`.

use std::any::Any;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::Float32Array;
use arrow_array::cast::AsArray;
use arrow_array::types::Float32Type;
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use arrow_schema::DataType;

// ── surprise_score ──────────────────────────────────────────────────

/// `surprise_score(kl_divergence: Float32) → Float32` — sigmoid transform.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SurpriseScoreUdf {
    signature: Signature,
}

impl Default for SurpriseScoreUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl SurpriseScoreUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float32], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SurpriseScoreUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "surprise_score"
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

        let kl = arrays[0].as_primitive::<Float32Type>();
        let len = kl.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if kl.is_null(i) {
                results.push(None);
            } else {
                // Sigmoid: 1 / (1 + exp(-kl))
                let val = kl.value(i);
                let score = 1.0 / (1.0 + (-val).exp());
                results.push(Some(score));
            }
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
    }
}

// ── fade_mem_decay ──────────────────────────────────────────────────

/// `fade_mem_decay(base: Float64, importance: Float32, access_freq: UInt64) → Float32`
///
/// Formula: `base × (1/(1+importance)) × (1/(1+access_freq))`
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct FadeMemDecayUdf {
    signature: Signature,
}

impl Default for FadeMemDecayUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl FadeMemDecayUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float64, DataType::Float32, DataType::UInt64],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for FadeMemDecayUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "fade_mem_decay"
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

        let base = arrays[0].as_primitive::<arrow_array::types::Float64Type>();
        let importance = arrays[1].as_primitive::<Float32Type>();
        let access_freq = arrays[2].as_primitive::<arrow_array::types::UInt64Type>();

        let len = base.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if base.is_null(i) || importance.is_null(i) || access_freq.is_null(i) {
                results.push(None);
                continue;
            }

            let b = base.value(i);
            let imp = importance.value(i) as f64;
            let freq = access_freq.value(i) as f64;

            let decay = b * (1.0 / (1.0 + imp)) * (1.0 / (1.0 + freq));
            results.push(Some(decay as f32));
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
    }
}

// ── causal_relevance ────────────────────────────────────────────────

/// `causal_relevance(strength: Float32, confidence: Float32, evidence_count: UInt32) → Float32`
///
/// Formula: `strength × confidence × log(1 + evidence_count)`
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CausalRelevanceUdf {
    signature: Signature,
}

impl Default for CausalRelevanceUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalRelevanceUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::Float32, DataType::Float32, DataType::UInt32],
                Volatility::Immutable,
            ),
        }
    }
}

impl ScalarUDFImpl for CausalRelevanceUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "causal_relevance"
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

        let strength = arrays[0].as_primitive::<Float32Type>();
        let confidence = arrays[1].as_primitive::<Float32Type>();
        let evidence = arrays[2].as_primitive::<arrow_array::types::UInt32Type>();

        let len = strength.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if strength.is_null(i) || confidence.is_null(i) || evidence.is_null(i) {
                results.push(None);
                continue;
            }

            let s = strength.value(i);
            let c = confidence.value(i);
            let e = evidence.value(i) as f32;

            let score = s * c * (1.0 + e).ln();
            results.push(Some(score));
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, UInt32Array, UInt64Array};
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;

    // ── surprise_score tests ──

    #[test]
    fn surprise_sigmoid() {
        let udf = SurpriseScoreUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(Float32Array::from(vec![
                0.0, 2.0, -2.0,
            ])))],
            number_rows: 3,
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a.as_primitive::<Float32Type>().clone(),
            _ => panic!("expected array"),
        };
        assert!((arr.value(0) - 0.5).abs() < 1e-5); // sigmoid(0) = 0.5
        assert!(arr.value(1) > 0.8); // sigmoid(2) ≈ 0.88
        assert!(arr.value(2) < 0.2); // sigmoid(-2) ≈ 0.12
    }

    #[test]
    fn surprise_null() {
        let udf = SurpriseScoreUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(Float32Array::from(vec![
                None,
            ])))],
            number_rows: 1,
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a.as_primitive::<Float32Type>().clone(),
            _ => panic!("expected array"),
        };
        assert!(arr.is_null(0));
    }

    // ── fade_mem_decay tests ──

    #[test]
    fn fade_mem_decay_known() {
        let udf = FadeMemDecayUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Float64Array::from(vec![0.1]))),
                ColumnarValue::Array(Arc::new(Float32Array::from(vec![0.5]))),
                ColumnarValue::Array(Arc::new(UInt64Array::from(vec![10]))),
            ],
            number_rows: 1,
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a.as_primitive::<Float32Type>().clone(),
            _ => panic!("expected array"),
        };
        // 0.1 * (1/1.5) * (1/11) ≈ 0.00606
        assert!((arr.value(0) - 0.00606).abs() < 0.001);
    }

    #[test]
    fn fade_mem_decay_null() {
        let udf = FadeMemDecayUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Float64Array::from(vec![None]))),
                ColumnarValue::Array(Arc::new(Float32Array::from(vec![Some(0.5)]))),
                ColumnarValue::Array(Arc::new(UInt64Array::from(vec![Some(10)]))),
            ],
            number_rows: 1,
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a.as_primitive::<Float32Type>().clone(),
            _ => panic!("expected array"),
        };
        assert!(arr.is_null(0));
    }

    // ── causal_relevance tests ──

    #[test]
    fn causal_relevance_known() {
        let udf = CausalRelevanceUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Float32Array::from(vec![0.8]))),
                ColumnarValue::Array(Arc::new(Float32Array::from(vec![0.9]))),
                ColumnarValue::Array(Arc::new(UInt32Array::from(vec![5]))),
            ],
            number_rows: 1,
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a.as_primitive::<Float32Type>().clone(),
            _ => panic!("expected array"),
        };
        // 0.8 * 0.9 * ln(6) ≈ 0.72 * 1.7918 ≈ 1.2901
        let expected = 0.8 * 0.9 * 6.0_f32.ln();
        assert!(
            (arr.value(0) - expected).abs() < 0.001,
            "got {}",
            arr.value(0)
        );
    }

    #[test]
    fn causal_relevance_null() {
        let udf = CausalRelevanceUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![
                ColumnarValue::Array(Arc::new(Float32Array::from(vec![Some(0.8)]))),
                ColumnarValue::Array(Arc::new(Float32Array::from(vec![None]))),
                ColumnarValue::Array(Arc::new(UInt32Array::from(vec![Some(5)]))),
            ],
            number_rows: 1,
            return_field: Arc::new(Field::new("result", DataType::Float32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        let arr = match result {
            ColumnarValue::Array(a) => a.as_primitive::<Float32Type>().clone(),
            _ => panic!("expected array"),
        };
        assert!(arr.is_null(0));
    }
}
