/// Incremental population statistics using Welford's online algorithm.
///
/// Tracks mean and variance of a stream of `f64` observations with O(1) memory.
/// Used for RPE z-score computation across writes and batch scoring.
///
/// Z-score is computed against the historical population **excluding** the
/// current sample (jackknife-style), then the current distance is added for
/// future z-scores. This prevents mathematical circularity where a value's
/// novelty score would be influenced by its own inclusion in the population.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WelfordStats {
    count: u64,
    mean: f64,
    m2: f64, // sum of squared deviations (Welford's online algorithm)
}

impl Default for WelfordStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }
}

impl WelfordStats {
    /// Create new empty statistics accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of observations recorded so far.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Current running mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Record a new observation using Welford's online update.
    ///
    /// Non-finite values (NaN, ±∞) are silently ignored to prevent poisoning
    /// the running mean/variance (N-M06).
    pub fn update(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Merge another accumulator into this one without replaying samples.
    pub fn merge(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean += delta * (other.count as f64 / combined_count as f64);
        self.m2 += other.m2
            + delta * delta * (self.count as f64 * other.count as f64 / combined_count as f64);
        self.count = combined_count;
    }

    /// Sample variance (Bessel's correction). Returns 1.0 if fewer than 2 samples.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 1.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Sample standard deviation.
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Compute z-score of `value` against the running population.
    /// Returns 0.0 when fewer than 2 samples have been recorded.
    pub fn z_score(&self, value: f64) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let variance = self.m2 / (self.count - 1) as f64;
        let stddev = variance.sqrt();
        // Guard against FP underflow: Welford's M2 accumulation can drift
        // slightly negative under extreme precision loss, producing NaN sqrt.
        // Epsilon 1e-10 treats near-zero stddev as "no variance observed."
        if stddev < 1e-10 {
            return 0.0;
        }
        (value - self.mean) / stddev
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let stats = WelfordStats::new();
        assert_eq!(stats.count(), 0);
        assert!((stats.mean() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn welford_known_distribution() {
        let mut stats = WelfordStats::new();
        stats.update(2.0);
        stats.update(4.0);
        stats.update(4.0);
        stats.update(4.0);
        stats.update(5.0);
        stats.update(5.0);
        stats.update(7.0);
        stats.update(9.0);

        assert!((stats.mean() - 5.0).abs() < 0.01);
        // Sample variance (Bessel's correction): m2 / (n-1) ≈ 4.571
        assert!((stats.variance() - 4.571).abs() < 0.01);
        assert!((stats.stddev() - 2.138).abs() < 0.01);
    }

    #[test]
    fn z_score_single_observation() {
        let mut stats = WelfordStats::new();
        stats.update(5.0);
        // Single observation → returns 0.0
        assert!((stats.z_score(5.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn z_score_known_values() {
        let mut stats = WelfordStats::new();
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.update(v);
        }
        // mean=5.0, stddev≈2.138
        // z(5.0) = (5-5)/2.138 = 0
        assert!((stats.z_score(5.0)).abs() < 0.01);
        // z(7.0) = (7-5)/2.138 ≈ 0.935
        assert!((stats.z_score(7.0) - 0.935).abs() < 0.05);
    }

    #[test]
    fn z_score_zero_variance() {
        let mut stats = WelfordStats::new();
        stats.update(5.0);
        stats.update(5.0);
        stats.update(5.0);
        // All identical → zero variance → z_score returns 0.0
        assert!((stats.z_score(5.0)).abs() < f64::EPSILON);
        assert!((stats.z_score(10.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_preserves_distribution() {
        let values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

        let mut full = WelfordStats::new();
        for value in values {
            full.update(value);
        }

        let mut left = WelfordStats::new();
        for value in [2.0, 4.0, 4.0, 4.0] {
            left.update(value);
        }

        let mut right = WelfordStats::new();
        for value in [5.0, 5.0, 7.0, 9.0] {
            right.update(value);
        }

        left.merge(&right);

        assert_eq!(left.count(), full.count());
        assert!((left.mean() - full.mean()).abs() < 1e-10);
        assert!((left.variance() - full.variance()).abs() < 1e-10);
    }
}
