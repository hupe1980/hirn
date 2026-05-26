//! Policy filtering operator.
//!
//! Filters input batches to include only rows whose `namespace` column
//! is in the set of namespaces allowed for the current principal.

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use async_trait::async_trait;

use hirn_core::error::HirnResult;
use hirn_storage::NamespacePolicy;

use std::collections::HashSet;
use std::sync::Arc;

use super::{OpContext, Operator};

/// Operator that filters rows by namespace based on Cedar policy.
///
/// If the `OpContext` has no principal, all rows pass through (permissive).
/// If the principal has no policy restrictions, all rows pass through.
pub struct PolicyFilter {
    pub policy: Arc<dyn NamespacePolicy>,
}

#[async_trait]
impl Operator for PolicyFilter {
    async fn execute(
        &self,
        input: Vec<RecordBatch>,
        ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>> {
        let principal = match &ctx.principal {
            Some(p) => p.as_str(),
            None => return Ok(input), // No principal → permissive.
        };

        let allowed = match self.policy.allowed_namespaces(principal).await {
            Some(ns) => ns.into_iter().collect::<HashSet<String>>(),
            None => return Ok(input), // No restrictions → pass all.
        };

        let mut out = Vec::new();
        for batch in &input {
            if let Some(filtered) = filter_batch_by_namespace(batch, &allowed)? {
                if filtered.num_rows() > 0 {
                    out.push(filtered);
                }
            }
        }
        Ok(out)
    }
}

/// Filter a single batch: keep only rows where `namespace` ∈ `allowed`.
/// Returns `None` if the batch has no `namespace` column (passes through).
fn filter_batch_by_namespace(
    batch: &RecordBatch,
    allowed: &HashSet<String>,
) -> HirnResult<Option<RecordBatch>> {
    let ns_col = match batch.column_by_name("namespace") {
        Some(c) => c,
        None => return Ok(Some(batch.clone())), // No namespace column → pass.
    };

    let str_arr = ns_col.as_string::<i32>();
    let mut keep = Vec::with_capacity(batch.num_rows());
    for i in 0..str_arr.len() {
        if !str_arr.is_null(i) && allowed.contains(str_arr.value(i)) {
            keep.push(i as u32);
        }
    }

    let indices = arrow_array::UInt32Array::from(keep);
    let columns: Vec<_> = batch
        .columns()
        .iter()
        .map(|col| arrow_select::take::take(col.as_ref(), &indices, None))
        .collect::<Result<_, _>>()
        .map_err(|e| hirn_core::error::HirnError::storage(e))?;

    let filtered = RecordBatch::try_new(batch.schema(), columns)
        .map_err(|e| hirn_core::error::HirnError::storage(e))?;
    Ok(Some(filtered))
}
