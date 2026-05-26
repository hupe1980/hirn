//! `token_count` UDF — tokenize text and return approximate token count.
//!
//! `token_count(text: Utf8) → UInt32`

use std::any::Any;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, UInt32Array};
use datafusion_common::Result;
use datafusion_expr::{ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility};

use arrow_schema::DataType;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TokenCountUdf {
    signature: Signature,
}

impl Default for TokenCountUdf {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenCountUdf {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Utf8], Volatility::Immutable),
        }
    }
}

/// Estimate token count using the simple heuristic: ~4 chars per token.
/// This avoids the overhead of loading a full BPE tokenizer for every row.
fn estimate_tokens(text: &str) -> u32 {
    // A commonly used heuristic: ~4 characters per token for English text.
    // This matches OpenAI's guidance and is good enough for budget calculations.
    let char_count = text.len();
    ((char_count as f64 / 4.0).ceil()) as u32
}

impl ScalarUDFImpl for TokenCountUdf {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "token_count"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt32)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let num_rows = args.number_rows;
        let arrays: Vec<ArrayRef> = args
            .args
            .iter()
            .map(|a| a.to_array(num_rows))
            .collect::<Result<Vec<_>>>()?;

        let text_arr = arrays[0].as_string::<i32>();
        let len = text_arr.len();
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            if text_arr.is_null(i) {
                results.push(None);
            } else {
                results.push(Some(estimate_tokens(text_arr.value(i))));
            }
        }

        Ok(ColumnarValue::Array(Arc::new(UInt32Array::from(results))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::StringArray;
    use arrow_schema::Field;
    use datafusion_common::config::ConfigOptions;

    fn invoke(texts: Vec<Option<&str>>) -> UInt32Array {
        let udf = TokenCountUdf::new();
        let args = ScalarFunctionArgs {
            args: vec![ColumnarValue::Array(Arc::new(StringArray::from(
                texts.clone(),
            )))],
            number_rows: texts.len(),
            return_field: Arc::new(Field::new("result", DataType::UInt32, true)),
            arg_fields: vec![],
            config_options: Arc::new(ConfigOptions::new()),
        };
        let result = udf.invoke_with_args(args).unwrap();
        match result {
            ColumnarValue::Array(a) => a.as_any().downcast_ref::<UInt32Array>().unwrap().clone(),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn known_text() {
        let vals = invoke(vec![Some("Hello, world! This is a test.")]);
        // 29 chars / 4 ≈ 8 tokens
        assert!(vals.value(0) > 0);
        assert!(vals.value(0) < 20);
    }

    #[test]
    fn empty_string() {
        let vals = invoke(vec![Some("")]);
        assert_eq!(vals.value(0), 0);
    }

    #[test]
    fn null_returns_null() {
        let vals = invoke(vec![None]);
        assert!(vals.is_null(0));
    }
}
