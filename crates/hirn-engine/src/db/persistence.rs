use super::*;

impl HirnDB {
    /// Returns the configured distance metric.
    ///
    /// `hirn_core::DistanceMetric` and `hirn_storage::store::DistanceMetric` are
    /// now the same type (re-export), so no conversion is needed.
    pub(crate) fn distance_metric(&self) -> hirn_storage::store::DistanceMetric {
        self.config.metric
    }
}
