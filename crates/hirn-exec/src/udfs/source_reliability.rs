//! `source_reliability` UDF — score memory provenance.
//!
//! `source_reliability(source_type: Utf8) → Float32`

use std::any::Any;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::Float32Array;
use arrow_array::cast::AsArray;
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use arrow_schema::DataType;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct SourceReliabilityUdf {
    signature: Signature,
}

impl Default for SourceReliabilityUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceReliabilityUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Utf8], Volatility::Immutable),
        }
    }
}

fn reliability_score(source_type: &str) -> f32 {
    match source_type {
        "direct_observation" => 1.0,
        "agent_generated" => 0.8,
        "inferred" => 0.6,
        "cross_agent" => 0.5,
        _ => 0.4,
    }
}

impl ScalarUDFImpl for SourceReliabilityUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "source_reliability"
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

        let source_types = arrays[0].as_string::<i32>();
        let len = source_types.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if source_types.is_null(i) {
                results.push(None);
            } else {
                results.push(Some(reliability_score(source_types.value(i))));
            }
        }

        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;

    fn invoke(types: Vec<Option<&str>>) -> Float32Array {
        let udf = SourceReliabilityUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(StringArray::from(
                types.clone(),
            )))],
            number_rows: types.len(),
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
    fn known_source_types() {
        let vals = invoke(vec![
            Some("direct_observation"),
            Some("agent_generated"),
            Some("inferred"),
            Some("cross_agent"),
        ]);
        assert!((vals.value(0) - 1.0).abs() < 1e-6);
        assert!((vals.value(1) - 0.8).abs() < 1e-6);
        assert!((vals.value(2) - 0.6).abs() < 1e-6);
        assert!((vals.value(3) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn unknown_returns_default() {
        let vals = invoke(vec![Some("unknown_type")]);
        assert!((vals.value(0) - 0.4).abs() < 1e-6);
    }

    #[test]
    fn null_returns_null() {
        let vals = invoke(vec![None]);
        assert!(vals.is_null(0));
    }
}
