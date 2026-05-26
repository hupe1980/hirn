use super::*;

// ═══════════════════════════════════════════════════════════════════════════
// Narrative Thread Formation
// ═══════════════════════════════════════════════════════════════════════════

/// A narrative thread — a coherent "story" spanning multiple segments.
#[derive(Debug, Clone)]
pub struct NarrativeThread {
    /// Auto-generated title from dominant entities/topics.
    pub title: String,
    /// Segments composing this thread.
    pub segment_indices: Vec<usize>,
    /// All record IDs in this thread (ordered by time).
    pub record_ids: Vec<MemoryId>,
    /// All content from records in this thread.
    pub contents: Vec<String>,
    /// All summaries from records in this thread.
    pub summaries: Vec<String>,
    /// Timeline: start to end.
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    /// Key entities participating in this thread.
    pub entities: Vec<String>,
    /// Sub-threads (if hierarchical splitting detected).
    pub sub_threads: Vec<Self>,
    /// Mean embedding for the thread.
    pub embedding: Option<Vec<f32>>,
}

/// F-021 FIX: Precomputed condensed distance matrix for O(N²) pairwise similarity.
/// Stores upper-triangular entries in row-major order:
///   index(i,j) = i * n - i*(i+1)/2 + j - i - 1   for i < j
struct CondensedMatrix {
    data: Vec<f32>,
    n: usize,
}

impl CondensedMatrix {
    fn new(segments: &[EpisodeSegment]) -> Self {
        let n = segments.len();
        let size = n * (n - 1) / 2;
        let mut data = Vec::with_capacity(size);

        // Pre-compute entity sets once to avoid re-allocating HashSets.
        let entity_sets: Vec<HashSet<&str>> = segments
            .iter()
            .map(|s| s.dominant_entities.iter().map(String::as_str).collect())
            .collect();

        for i in 0..n {
            for j in (i + 1)..n {
                let embedding_sim =
                    match (&segments[i].topic_embedding, &segments[j].topic_embedding) {
                        (Some(ea), Some(eb)) => {
                            1.0 - lance_linalg::distance::cosine_distance(ea, eb)
                        }
                        _ => 0.0,
                    };
                let intersection = entity_sets[i].intersection(&entity_sets[j]).count();
                let union = entity_sets[i].union(&entity_sets[j]).count();
                let entity_sim = if union > 0 {
                    intersection as f32 / union as f32
                } else {
                    0.0
                };
                data.push(embedding_sim * 0.6 + entity_sim * 0.4);
            }
        }

        Self { data, n }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f32 {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        self.data[a * self.n - a * (a + 1) / 2 + b - a - 1]
    }
}

/// F-021 FIX: Union-Find with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return false;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
        true
    }
}

/// Form narrative threads from segments using single-linkage clustering
/// with a precomputed condensed distance matrix.
///
/// **F-021 FIX:** Replaced O(N⁴) hierarchical agglomerative clustering with
/// O(N² log N) sorted-edge single-linkage via union-find. The pairwise
/// similarity matrix is computed once (O(N²·D)), sorted once (O(N² log N)),
/// then edges are greedily merged above the threshold.
pub fn form_narrative_threads(
    segments: &[EpisodeSegment],
    _patterns: &DetectedPatterns,
    config: &ConsolidationConfig,
) -> Vec<NarrativeThread> {
    if segments.is_empty() {
        return Vec::new();
    }
    if segments.len() == 1 {
        return vec![thread_from_segment_group(segments, &[0])];
    }

    let n = segments.len();
    let matrix = CondensedMatrix::new(segments);

    // Collect all (similarity, i, j) edges and sort descending.
    let mut edges: Vec<(f32, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let sim = matrix.get(i, j);
            if sim >= config.thread_similarity_threshold {
                edges.push((sim, i, j));
            }
        }
    }
    edges.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Single-linkage merge via union-find.
    let mut uf = UnionFind::new(n);
    for &(_sim, i, j) in &edges {
        uf.union(i, j);
    }

    // Collect clusters from union-find roots.
    let mut cluster_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        cluster_map.entry(uf.find(i)).or_default().push(i);
    }

    // Convert clusters to narrative threads.
    let mut threads: Vec<NarrativeThread> = Vec::new();
    for (_root, cluster) in &cluster_map {
        let mut thread = thread_from_segment_group(segments, cluster);

        // Sub-thread detection for large clusters.
        if cluster.len() >= 4 {
            let sub_threads = detect_sub_threads(segments, cluster, config, &matrix);
            if sub_threads.len() > 1 {
                thread.sub_threads = sub_threads;
            }
        }

        threads.push(thread);
    }

    // Sort threads by start time.
    threads.sort_by_key(|thread| thread.start_time);

    threads
}

/// Create a narrative thread from a group of segment indices.
fn thread_from_segment_group(segments: &[EpisodeSegment], indices: &[usize]) -> NarrativeThread {
    let mut all_records: Vec<&EpisodicRecord> = Vec::new();
    let mut all_entities: HashMap<String, usize> = HashMap::new();
    let mut embeddings: Vec<&Vec<f32>> = Vec::new();

    for &idx in indices {
        let seg = &segments[idx];
        for rec in &seg.records {
            all_records.push(rec);
            for ent in &rec.entities {
                *all_entities.entry(ent.name.clone()).or_default() += 1;
            }
        }
        if let Some(ref emb) = seg.topic_embedding {
            embeddings.push(emb);
        }
    }

    // Sort records by timestamp.
    all_records.sort_by_key(|r| r.timestamp);

    // Get top entities for title.
    let mut entity_list: Vec<(String, usize)> = all_entities.into_iter().collect();
    entity_list.sort_by_key(|item| std::cmp::Reverse(item.1));
    let top_entities: Vec<String> = entity_list
        .iter()
        .take(3)
        .map(|(name, _)| name.clone())
        .collect();

    let title = if top_entities.is_empty() {
        "Unnamed Thread".to_string()
    } else {
        top_entities.join(", ")
    };

    let entities: Vec<String> = entity_list.into_iter().map(|(name, _)| name).collect();

    // Mean embedding.
    let embedding = if embeddings.is_empty() {
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
        let norm: f32 = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut mean {
                *v /= norm;
            }
        }
        Some(mean)
    };

    let (start_time, end_time) = match (all_records.first(), all_records.last()) {
        (Some(first), Some(last)) => (first.timestamp, last.timestamp),
        _ => (Timestamp::default(), Timestamp::default()),
    };

    NarrativeThread {
        title,
        segment_indices: indices.to_vec(),
        record_ids: all_records.iter().map(|r| r.id).collect(),
        contents: all_records.iter().map(|r| r.content.clone()).collect(),
        summaries: all_records.iter().map(|r| r.summary.clone()).collect(),
        start_time,
        end_time,
        entities,
        sub_threads: Vec::new(),
        embedding,
    }
}

/// Detect sub-threads within a cluster by re-clustering with a tighter threshold.
///
/// **F-021 FIX:** Reuses the precomputed `CondensedMatrix` instead of
/// recomputing O(C²·|A|·|B|) similarities. Union-find single-linkage
/// on within-cluster pairs.
fn detect_sub_threads(
    segments: &[EpisodeSegment],
    cluster: &[usize],
    config: &ConsolidationConfig,
    matrix: &CondensedMatrix,
) -> Vec<NarrativeThread> {
    let tighter_threshold = config.thread_similarity_threshold + 0.15;
    let m = cluster.len();

    // Collect within-cluster edges that exceed the tighter threshold.
    let mut edges: Vec<(f32, usize, usize)> = Vec::new();
    for ci in 0..m {
        for cj in (ci + 1)..m {
            let sim = matrix.get(cluster[ci], cluster[cj]);
            if sim >= tighter_threshold {
                edges.push((sim, ci, cj));
            }
        }
    }
    edges.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut uf = UnionFind::new(m);
    for &(_sim, i, j) in &edges {
        uf.union(i, j);
    }

    let mut sub_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for ci in 0..m {
        sub_map.entry(uf.find(ci)).or_default().push(cluster[ci]);
    }

    sub_map
        .into_values()
        .map(|c| thread_from_segment_group(segments, &c))
        .collect()
}
