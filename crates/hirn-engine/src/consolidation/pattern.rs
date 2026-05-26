use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Pattern Detection
// ═══════════════════════════════════════════════════════════════════════════

/// A detected recurring pattern across segments.
#[derive(Debug, Clone)]
pub struct Pattern {
    /// Entities involved in this pattern.
    pub entities: Vec<String>,
    /// Number of segments this pattern appears in.
    pub frequency: usize,
    /// Indices of segments containing this pattern.
    pub segment_indices: Vec<usize>,
    /// Diversity score: unique time spans covered.
    pub diversity_score: f64,
    /// Representative embedding (mean of segment embeddings where pattern appears).
    pub representative_embedding: Option<Vec<f32>>,
}

/// A temporal pattern: recurring topic over time.
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// The dominant entity or topic.
    pub topic: String,
    /// Segment indices where this topic recurs.
    pub occurrences: Vec<usize>,
    /// Estimated period in seconds (if periodic), None if irregular.
    pub period_seconds: Option<i64>,
    /// First occurrence timestamp.
    pub first_occurrence: Timestamp,
    /// Last occurrence timestamp.
    pub last_occurrence: Timestamp,
}

/// A recurring causal chain pattern.
#[derive(Debug, Clone)]
pub struct CausalPattern {
    /// Entity names forming the causal chain.
    pub chain: Vec<String>,
    /// Number of times this chain was observed.
    pub occurrences: usize,
    /// Confidence based on consistency.
    pub confidence: f32,
}

/// All detected patterns from a set of segments.
#[derive(Debug, Clone)]
pub struct DetectedPatterns {
    pub entity_patterns: Vec<Pattern>,
    pub temporal_patterns: Vec<TemporalPattern>,
    pub causal_patterns: Vec<CausalPattern>,
}

/// Detect patterns across episode segments.
pub async fn detect_patterns(
    segments: &[EpisodeSegment],
    config: &ConsolidationConfig,
    db: &HirnDB,
) -> DetectedPatterns {
    let entity_patterns = detect_entity_patterns(segments, config);
    let temporal_patterns = detect_temporal_patterns(segments, config);
    let causal_patterns = detect_causal_patterns(segments, db).await;

    DetectedPatterns {
        entity_patterns,
        temporal_patterns,
        causal_patterns,
    }
}

/// Detect entity frequency and co-occurrence patterns.
pub(super) fn detect_entity_patterns(
    segments: &[EpisodeSegment],
    config: &ConsolidationConfig,
) -> Vec<Pattern> {
    // Count entity appearances across segments.
    let mut entity_segments: HashMap<String, Vec<usize>> = HashMap::new();
    for (seg_idx, seg) in segments.iter().enumerate() {
        let mut seen_entities: HashSet<String> = HashSet::new();
        for rec in &seg.records {
            for ent in &rec.entities {
                seen_entities.insert(ent.name.clone());
            }
        }
        for entity in seen_entities {
            entity_segments.entry(entity).or_default().push(seg_idx);
        }
    }

    // Detect co-occurrence patterns (entities that consistently appear together).
    let mut co_occurrence: HashMap<(String, String), Vec<usize>> = HashMap::new();
    for (seg_idx, seg) in segments.iter().enumerate() {
        let mut seg_entities: HashSet<String> = HashSet::new();
        for rec in &seg.records {
            for ent in &rec.entities {
                seg_entities.insert(ent.name.clone());
            }
        }
        let mut entities: Vec<String> = seg_entities.into_iter().collect();
        entities.sort();
        for i in 0..entities.len() {
            for j in (i + 1)..entities.len() {
                let pair = (entities[i].clone(), entities[j].clone());
                co_occurrence.entry(pair).or_default().push(seg_idx);
            }
        }
    }

    let mut patterns = Vec::new();

    // Single-entity patterns.
    for (entity, seg_indices) in &entity_segments {
        if seg_indices.len() >= config.min_pattern_frequency {
            let diversity = compute_diversity(segments, seg_indices);
            let embedding = compute_pattern_embedding(segments, seg_indices);
            patterns.push(Pattern {
                entities: vec![entity.clone()],
                frequency: seg_indices.len(),
                segment_indices: seg_indices.clone(),
                diversity_score: diversity,
                representative_embedding: embedding,
            });
        }
    }

    // Co-occurrence patterns.
    for ((e1, e2), seg_indices) in &co_occurrence {
        if seg_indices.len() >= config.min_pattern_frequency {
            let diversity = compute_diversity(segments, seg_indices);
            let embedding = compute_pattern_embedding(segments, seg_indices);
            patterns.push(Pattern {
                entities: vec![e1.clone(), e2.clone()],
                frequency: seg_indices.len(),
                segment_indices: seg_indices.clone(),
                diversity_score: diversity,
                representative_embedding: embedding,
            });
        }
    }

    // Sort by frequency × diversity (descending).
    patterns.sort_by(|a, b| {
        let score_a = a.frequency as f64 * a.diversity_score;
        let score_b = b.frequency as f64 * b.diversity_score;
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    patterns
}

/// Compute diversity score for a pattern across segments.
/// Measures the time span covered relative to total time span.
fn compute_diversity(segments: &[EpisodeSegment], seg_indices: &[usize]) -> f64 {
    if seg_indices.len() <= 1 || segments.is_empty() {
        return 1.0;
    }

    // Safety: seg_indices.len() > 1 guarantees first()/last() succeed.
    // Bounds-check indices against segments to guard against corruption.
    let &first_idx = seg_indices.first().unwrap();
    let &last_idx = seg_indices.last().unwrap();
    if first_idx >= segments.len() || last_idx >= segments.len() {
        return 1.0;
    }

    let first_seg_time = segments[first_idx].start_time.as_datetime();
    let last_seg_time = segments[last_idx].end_time.as_datetime();

    let total_first = segments.first().unwrap().start_time.as_datetime();
    let total_last = segments.last().unwrap().end_time.as_datetime();

    let pattern_span = last_seg_time
        .signed_duration_since(first_seg_time)
        .num_seconds() as f64;
    let total_span = total_last
        .signed_duration_since(total_first)
        .num_seconds()
        .max(1) as f64;

    (pattern_span / total_span).clamp(0.0, 1.0)
}

/// Compute a representative embedding for a pattern.
fn compute_pattern_embedding(
    segments: &[EpisodeSegment],
    seg_indices: &[usize],
) -> Option<Vec<f32>> {
    let embeddings: Vec<&Vec<f32>> = seg_indices
        .iter()
        .filter_map(|&idx| segments.get(idx))
        .filter_map(|seg| seg.topic_embedding.as_ref())
        .collect();

    if embeddings.is_empty() {
        return None;
    }

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
    let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut mean {
            *v /= norm;
        }
    }
    Some(mean)
}

/// Detect temporal patterns (recurring topics over time).
fn detect_temporal_patterns(
    segments: &[EpisodeSegment],
    config: &ConsolidationConfig,
) -> Vec<TemporalPattern> {
    // Group segments by dominant entity.
    let mut topic_occurrences: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, seg) in segments.iter().enumerate() {
        for entity in &seg.dominant_entities {
            topic_occurrences
                .entry(entity.clone())
                .or_default()
                .push(idx);
        }
    }

    let mut patterns = Vec::new();

    for (topic, occurrences) in &topic_occurrences {
        if occurrences.len() < config.min_pattern_frequency {
            continue;
        }

        let first = segments[occurrences[0]].start_time;
        let Some(&last_idx) = occurrences.last() else {
            continue;
        };
        let last = segments[last_idx].start_time;

        // Estimate period if we have enough data points.
        let period = if occurrences.len() >= 3 {
            estimate_period(segments, occurrences)
        } else {
            None
        };

        patterns.push(TemporalPattern {
            topic: topic.clone(),
            occurrences: occurrences.clone(),
            period_seconds: period,
            first_occurrence: first,
            last_occurrence: last,
        });
    }

    patterns
}

/// Estimate period of recurring pattern using median inter-occurrence interval.
fn estimate_period(segments: &[EpisodeSegment], occurrences: &[usize]) -> Option<i64> {
    if occurrences.len() < 3 {
        return None;
    }

    let mut intervals: Vec<i64> = Vec::new();
    for i in 1..occurrences.len() {
        let prev_time = segments[occurrences[i - 1]].start_time.as_datetime();
        let curr_time = segments[occurrences[i]].start_time.as_datetime();
        let interval = curr_time.signed_duration_since(prev_time).num_seconds();
        intervals.push(interval);
    }

    // Compute median interval.
    intervals.sort_unstable();
    let median = intervals[intervals.len() / 2];

    // Check if intervals are reasonably consistent (coefficient of variation < 0.5).
    let mean = intervals.iter().sum::<i64>() as f64 / intervals.len() as f64;
    if mean <= 0.0 {
        return None;
    }
    let variance = intervals
        .iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / intervals.len() as f64;
    let cv = variance.sqrt() / mean;

    if cv < 0.5 {
        Some(median)
    } else {
        None // Too irregular to be a pattern.
    }
}

/// Detect recurring causal chain patterns using graph edges.
async fn detect_causal_patterns(segments: &[EpisodeSegment], db: &HirnDB) -> Vec<CausalPattern> {
    let store = db.graph_store();
    let mut chain_counts: HashMap<Vec<String>, usize> = HashMap::new();

    for seg in segments {
        // For each record in the segment, follow causal edges to find chains.
        for rec in &seg.records {
            let causes_edges = store
                .get_edges_of_type(rec.id, EdgeRelation::Causes)
                .await
                .unwrap_or_default();
            if causes_edges.is_empty() {
                continue;
            }

            // Build chain from this record following Causes edges.
            let mut chain = vec![dominant_entity_name(rec)];
            let mut current = rec.id;
            let mut visited = HashSet::new();
            visited.insert(current);

            loop {
                let edges = store
                    .get_edges_of_type(current, EdgeRelation::Causes)
                    .await
                    .unwrap_or_default();
                let next = edges.into_iter().find(|e| !visited.contains(&e.target));
                match next {
                    Some(edge) => {
                        visited.insert(edge.target);
                        // Try to find entity name from target record.
                        let target_name = seg
                            .records
                            .iter()
                            .find(|r| r.id == edge.target)
                            .map_or_else(|| format!("{}", edge.target), dominant_entity_name);
                        chain.push(target_name);
                        current = edge.target;
                    }
                    None => break,
                }
            }

            if chain.len() >= 2 {
                *chain_counts.entry(chain).or_default() += 1;
            }
        }
    }

    chain_counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(chain, occurrences)| CausalPattern {
            confidence: (occurrences as f32 / 10.0).clamp(0.0, 1.0),
            chain,
            occurrences,
        })
        .collect()
}

fn dominant_entity_name(rec: &EpisodicRecord) -> String {
    rec.entities.first().map_or_else(
        || rec.content.chars().take(30).collect(),
        |e| e.name.clone(),
    )
}
