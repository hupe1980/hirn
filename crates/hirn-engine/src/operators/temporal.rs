//! Temporal expansion operator.
//!
//! Given input batches containing a `created_at_ms` column, retrieves
//! additional memories within a ± time window around each timestamp.

use arrow_array::cast::AsArray;
use arrow_array::{Array, RecordBatch};
use async_trait::async_trait;
use futures::TryStreamExt;

use hirn_core::error::{HirnError, HirnResult};
use hirn_storage::store::ScanOptions;

use super::{OpContext, Operator};

/// Operator that expands results with temporally adjacent memories.
///
/// For each unique `created_at_ms` value in the input, scans the given
/// dataset for memories within `window_ms` milliseconds in either direction.
pub struct TemporalExpand {
    /// Dataset to scan for temporal neighbours.
    pub dataset: String,
    /// Time window in milliseconds (applied in both directions).
    pub window_ms: i64,
}

#[async_trait]
impl Operator for TemporalExpand {
    async fn execute(
        &self,
        input: Vec<RecordBatch>,
        ctx: &OpContext,
    ) -> HirnResult<Vec<RecordBatch>> {
        let timestamps = extract_timestamps(&input)?;
        if timestamps.is_empty() {
            return Ok(input);
        }

        // Compute the overall [min - window, max + window] range.
        let min_ts = timestamps.iter().copied().min().unwrap_or(0);
        let max_ts = timestamps.iter().copied().max().unwrap_or(0);
        let lo = min_ts.saturating_sub(self.window_ms);
        let hi = max_ts.saturating_add(self.window_ms);

        let filter = format!("created_at_ms >= {lo} AND created_at_ms <= {hi}");
        let mut expanded = ctx
            .store
            .scan_stream(
                &self.dataset,
                ScanOptions {
                    filter: Some(filter),
                    exact_filter: None,
                    columns: None,
                    order_by: None,
                    limit: None,
                    offset: None,
                },
            )
            .await
            .map_err(|e| HirnError::storage(e))?;

        // Merge input + expanded (caller can deduplicate later).
        let mut out = input;
        while let Some(batch) = expanded
            .try_next()
            .await
            .map_err(|e| HirnError::storage(e))?
        {
            out.push(batch);
        }
        Ok(out)
    }
}

/// Extract `created_at_ms` values from all input batches.
fn extract_timestamps(batches: &[RecordBatch]) -> HirnResult<Vec<i64>> {
    let mut ts = Vec::new();
    for batch in batches {
        if let Some(col) = batch.column_by_name("created_at_ms") {
            let arr = col.as_primitive::<arrow_array::types::Int64Type>();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    ts.push(arr.value(i));
                }
            }
        }
    }
    Ok(ts)
}
