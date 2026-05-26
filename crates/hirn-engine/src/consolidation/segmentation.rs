use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Episode Segmentation
// ═══════════════════════════════════════════════════════════════════════════

/// A segment of topically-related, temporally-contiguous episodic records.
#[derive(Debug, Clone)]
pub struct EpisodeSegment {
    /// Records in this segment, ordered by timestamp.
    pub records: Vec<EpisodicRecord>,
    /// Start time of the segment.
    pub start_time: Timestamp,
    /// End time of the segment.
    pub end_time: Timestamp,
    /// Entities that appear most in this segment.
    pub dominant_entities: Vec<String>,
    /// Mean embedding across all records with embeddings.
    pub topic_embedding: Option<Vec<f32>>,
}

impl EpisodeSegment {
    /// Build a segment from a non-empty list of records.
    /// Returns `None` if `records` is empty.
    pub(super) fn from_records(records: Vec<EpisodicRecord>) -> Option<Self> {
        let start_time = records.first()?.timestamp;
        let end_time = records.last()?.timestamp;

        // Count entities.
        let mut entity_counts: HashMap<String, usize> = HashMap::new();
        for r in &records {
            for e in &r.entities {
                *entity_counts.entry(e.name.clone()).or_default() += 1;
            }
        }
        let mut dominant: Vec<(String, usize)> = entity_counts.into_iter().collect();
        dominant.sort_by_key(|item| std::cmp::Reverse(item.1));
        let dominant_entities: Vec<String> =
            dominant.into_iter().take(5).map(|(name, _)| name).collect();

        // Compute mean embedding.
        let embeddings: Vec<&Vec<f32>> = records
            .iter()
            .filter_map(|r| r.embedding.as_ref())
            .collect();
        let topic_embedding = if embeddings.is_empty() {
            None
        } else {
            let dims = embeddings[0].len();
            let mut mean = vec![0.0f32; dims];
            for emb in &embeddings {
                for (i, v) in emb.iter().enumerate() {
                    mean[i] += v;
                }
            }
            let n = embeddings.len() as f32;
            for v in &mut mean {
                *v /= n;
            }
            // Normalize.
            let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut mean {
                    *v /= norm;
                }
            }
            Some(mean)
        };

        Some(Self {
            records,
            start_time,
            end_time,
            dominant_entities,
            topic_embedding,
        })
    }
}

/// Adaptive Bayesian surprise-based segmentation inspired by EM-LLM
/// (Fountas et al., ICLR 2025, arXiv:2407.09450).
///
/// Uses three complementary signals with **adaptive thresholds**:
/// 1. **Topic boundary**: cosine dissimilarity exceeds adaptive threshold
///    T = μ_{recent} + γ · σ_{recent}  (Bayesian surprise)
/// 2. **Surprise spike**: record surprise exceeds adaptive threshold
/// 3. **Temporal gap**: large time gaps between consecutive records
///
/// The adaptive window (`segmentation_lookback`) controls how many recent
/// observations inform the running mean and standard deviation. This replaces
/// fixed thresholds with data-driven boundaries that self-tune to the stream.
pub fn segment_episodes(
    records: &[EpisodicRecord],
    config: &ConsolidationConfig,
) -> Vec<EpisodeSegment> {
    if records.is_empty() {
        return Vec::new();
    }
    if records.len() == 1 {
        return EpisodeSegment::from_records(vec![records[0].clone()])
            .into_iter()
            .collect();
    }

    let lookback = config.segmentation_lookback;
    let gamma = config.segmentation_gamma;

    // Collect per-transition dissimilarity and surprise signals.
    let mut dissimilarities: Vec<f64> = Vec::with_capacity(records.len());
    let mut surprises: Vec<f64> = Vec::with_capacity(records.len());

    // Index 0 has no predecessor — use sentinel values.
    dissimilarities.push(0.0);
    surprises.push(records[0].surprise as f64);

    for i in 1..records.len() {
        let prev = &records[i - 1];
        let curr = &records[i];

        let dissim = if let (Some(emb_prev), Some(emb_curr)) = (&prev.embedding, &curr.embedding) {
            let sim = 1.0 - lance_linalg::distance::cosine_distance(emb_prev, emb_curr);
            (1.0 - sim) as f64
        } else {
            0.0
        };

        dissimilarities.push(dissim);
        surprises.push(curr.surprise as f64);
    }

    // Find boundary indices (index i means a boundary BEFORE record i).
    let mut boundaries: HashSet<usize> = HashSet::new();

    for i in 1..records.len() {
        let prev = &records[i - 1];
        let curr = &records[i];

        // Signal 1: Adaptive topic boundary.
        // Threshold = μ + γ·σ over a sliding window of recent dissimilarities.
        let topic_threshold = adaptive_threshold(
            &dissimilarities,
            i,
            lookback,
            gamma,
            config.topic_similarity_threshold as f64, // fallback floor
        );
        if dissimilarities[i] > topic_threshold {
            boundaries.insert(i);
        }

        // Signal 2: Adaptive surprise threshold.
        let surprise_threshold = adaptive_threshold(
            &surprises,
            i,
            lookback,
            gamma,
            config.surprise_threshold as f64, // fallback floor
        );
        if surprises[i] > surprise_threshold {
            boundaries.insert(i);
        }

        // Signal 3: Temporal gap (kept as fixed threshold — time gaps are absolute).
        let gap_secs = curr
            .timestamp
            .as_datetime()
            .signed_duration_since(prev.timestamp.as_datetime())
            .num_seconds();
        if gap_secs > config.temporal_gap_seconds {
            boundaries.insert(i);
        }
    }

    // Split into segments at boundaries.
    let mut segments = Vec::new();
    let mut start = 0;
    let mut sorted_boundaries: Vec<usize> = boundaries.into_iter().collect();
    sorted_boundaries.sort_unstable();

    for boundary in sorted_boundaries {
        if boundary > start {
            segments.extend(EpisodeSegment::from_records(
                records[start..boundary].to_vec(),
            ));
        }
        start = boundary;
    }
    // Last segment.
    if start < records.len() {
        segments.extend(EpisodeSegment::from_records(records[start..].to_vec()));
    }

    segments
}

/// Compute an adaptive threshold T = μ + γ·σ over a sliding window of recent
/// signal values. Falls back to `floor` when insufficient data.
///
/// Reference: EM-LLM Bayesian surprise threshold (Fountas et al., ICLR 2025).
fn adaptive_threshold(
    signal: &[f64],
    current_idx: usize,
    lookback: usize,
    gamma: f64,
    floor: f64,
) -> f64 {
    // Need at least 2 observations for meaningful statistics.
    if current_idx < 2 {
        return floor;
    }

    let window_start = current_idx.saturating_sub(lookback);
    let window = &signal[window_start..current_idx];

    if window.is_empty() {
        return floor;
    }

    let n = window.len() as f64;
    let mean = window.iter().sum::<f64>() / n;
    let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    // T = μ + γ·σ, but never below the configured floor.
    (mean + gamma * stddev).max(floor)
}
