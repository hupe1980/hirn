//! ANN-vs-flat parity suite for `LancePhysicalStore`.
//!
//! This suite is the gate the Lance upgrade needs: it proves that once the
//! IVF/ANN vector index activates, recall stays high AND the index is built with
//! the dataset's *actual* distance metric.
//!
//! It guards two fixes:
//!   * **R-22** — the vector index must be partitioned with the dataset's search
//!     metric (cosine/dot/L2), not a hardcoded L2. An L2-partitioned index
//!     mis-buckets cosine/dot queries and silently loses recall (or returns the
//!     wrong top-1) once the dataset is large enough to bypass the flat path.
//!   * **R-38** — an explicitly-`create_index`-built index is honored even below
//!     the flat-scan row threshold, so a forced index is actually exercised.
//!
//! The flat path (exact brute force) and the indexed (ANN) path must agree:
//!   * `ann_parity_cosine_recall_at_10_above_threshold` (>50k, `#[ignore]`) —
//!     auto-ANN activates; recall@10 vs an in-test exact cosine ground truth.
//!   * `ann_index_honors_metric_cosine_vs_l2` (fast) — end-to-end proof that a
//!     cosine index returns the angle-nearest vector and an L2 index the
//!     Euclidean-nearest vector for the SAME data.
//!   * `ann_and_flat_agree_below_threshold` (fast) — flat and forced-index
//!     searches agree on top-1 and overlap heavily on top-10.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, RecordBatch, UInt64Array, builder::Float32Builder,
};
use arrow_schema::{DataType, Field, Schema};
use hirn_storage::lance_store::LancePhysicalStore;
use hirn_storage::namespace::NamespaceConfig;
use hirn_storage::store::*;
use tempfile::TempDir;

// ── Test-support helpers (self-contained, no external deps) ──

/// Create a Lance-backed store rooted in a fresh tmpdir, configured to build
/// vector indexes with `metric`.
async fn setup_store(metric: DistanceMetric) -> (TempDir, LancePhysicalStore) {
    let tmpdir = TempDir::new().unwrap();
    let root = tmpdir.path().to_str().unwrap().to_string();
    let ns = NamespaceConfig::local(&root).connect().await.unwrap();
    let store = LancePhysicalStore::new(root, ns).with_vector_index_metric(metric);
    (tmpdir, store)
}

/// Deterministic pseudo-random embedding of length `dim`, seeded by `seed`.
///
/// A splitmix-seeded LCG produces reproducible f32 values in `[-1, 1)` with no
/// external RNG dependency. Distinct seeds yield distinct vectors.
fn pseudo_embedding(seed: u64, dim: usize) -> Vec<f32> {
    // splitmix64 to decorrelate adjacent seeds, then a classic LCG stream.
    let mut state = seed
        .wrapping_add(1)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Take 31 high bits → [0, 1) → [-1, 1).
        let bits = (state >> 33) as u32;
        let unit = bits as f32 / (1u32 << 31) as f32;
        out.push(unit * 2.0 - 1.0);
    }
    out
}

/// Build an `id: UInt64` + `embedding: FixedSizeList<Float32; dim>` batch.
fn vector_batch(ids: &[u64], embeddings: &[Vec<f32>], dim: usize) -> RecordBatch {
    assert_eq!(ids.len(), embeddings.len());
    let id_array = UInt64Array::from(ids.to_vec());
    let mut builder = Float32Builder::with_capacity(ids.len() * dim);
    for emb in embeddings {
        assert_eq!(emb.len(), dim, "every embedding must have dim {dim}");
        for &v in emb {
            builder.append_value(v);
        }
    }
    let values = builder.finish();
    let embedding_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values),
        None,
    )
    .unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::UInt64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), dim as i32),
            false,
        ),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array) as ArrayRef,
            Arc::new(embedding_array) as ArrayRef,
        ],
    )
    .unwrap()
}

/// Extract the `id` column values from search-result batches, in result order
/// (nearest first).
fn result_ids(batches: &[RecordBatch]) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|batch| {
            let ids = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            (0..batch.num_rows())
                .map(|i| ids.value(i))
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Cosine distance (`1 - cos`), matching the store's flat-path semantics
/// (smaller = more similar; a zero-norm operand yields 1.0).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na * nb)
}

/// Exact top-`k` ids by cosine distance over `(id, vector)` corpus (ground truth).
fn brute_force_top_k_cosine(query: &[f32], corpus: &[(u64, Vec<f32>)], k: usize) -> Vec<u64> {
    let mut scored: Vec<(u64, f32)> = corpus
        .iter()
        .map(|(id, v)| (*id, cosine_distance(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

// ── Test 1: >50k auto-ANN recall gate (explicit, #[ignore]) ──

/// **Explicit large-scale gate — run with:**
/// `cargo test -p hirn-storage --test ann_parity -- --ignored`
///
/// Inserting >50k rows into a default-vector-indexed dataset ("semantic")
/// crosses `FLAT_VECTOR_CACHE_MAX_ROWS`, so the auto-ANN index activates and
/// `vector_search` uses the IVF/ANN path instead of the exact flat scan. Because
/// the store is configured for cosine, the R-22 fix must build the index with
/// the cosine metric; otherwise recall collapses. We assert recall@10 ≥ 0.80
/// against an in-test exact-cosine ground truth, averaged over a query sample.
///
/// Marked `#[ignore]` because it inserts 60k rows and builds a real ANN index
/// (order of a minute); the two fast tests below run on every `cargo test`.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "large-scale >50k auto-ANN recall gate; run with --ignored"]
async fn ann_parity_cosine_recall_at_10_above_threshold() {
    const ROWS: usize = 60_000; // > FLAT_VECTOR_CACHE_MAX_ROWS (50k)
    const DIM: usize = 32;
    const QUERIES: usize = 50;
    const K: usize = 10;
    const RECALL_FLOOR: f32 = 0.80;

    // "semantic" is in DEFAULT_VECTOR_INDEXED_DATASETS → auto-ANN on append.
    let (_dir, store) = setup_store(DistanceMetric::Cosine).await;

    // Build the corpus deterministically and keep it in memory for ground truth.
    let corpus: Vec<(u64, Vec<f32>)> = (0..ROWS as u64)
        .map(|i| (i, pseudo_embedding(i, DIM)))
        .collect();
    let ids: Vec<u64> = corpus.iter().map(|(id, _)| *id).collect();
    let embeddings: Vec<Vec<f32>> = corpus.iter().map(|(_, v)| v.clone()).collect();

    store
        .append("semantic", vector_batch(&ids, &embeddings, DIM))
        .await
        .unwrap();
    assert_eq!(store.count("semantic", None).await.unwrap(), ROWS as u64);

    // The auto-ANN index must have activated — otherwise we'd be measuring the
    // exact flat path (recall 1.0) and the gate would be meaningless.
    assert!(
        store
            .vector_index_exists("semantic", "embedding")
            .await
            .unwrap(),
        "crossing 50k rows must auto-build the ANN index (else ANN path is untested)"
    );

    let mut recall_sum = 0.0_f32;
    for q in 0..QUERIES {
        let query = pseudo_embedding(10_000_000 + q as u64, DIM);
        let truth: std::collections::HashSet<u64> =
            brute_force_top_k_cosine(&query, &corpus, K).into_iter().collect();

        let results = store
            .vector_search(
                "semantic",
                VectorSearchOptions {
                    column: "embedding".to_string(),
                    query: query.clone(),
                    metric: DistanceMetric::Cosine,
                    limit: K,
                    filter: None,
                    // Boost recall: probe many IVF partitions and exact-re-rank a
                    // wide candidate set. ANN is approximate; these keep it honest.
                    nprobes: Some(64),
                    refine_factor: Some(16),
                },
            )
            .await
            .unwrap();

        let got: std::collections::HashSet<u64> = result_ids(&results).into_iter().collect();
        let hits = truth.intersection(&got).count();
        recall_sum += hits as f32 / K as f32;
    }

    let recall = recall_sum / QUERIES as f32;
    println!("ann_parity_cosine_recall_at_10_above_threshold: recall@10 = {recall:.4} over {QUERIES} queries, {ROWS} rows");
    assert!(
        recall >= RECALL_FLOOR,
        "recall@10 = {recall:.4} fell below the {RECALL_FLOOR} floor; the ANN index may be \
         mis-partitioned (R-22) or under-probed"
    );
}

// ── Test 2: index metric honored end-to-end (cosine vs L2) ──

/// End-to-end proof that the ANN index metric matches the search metric (R-22).
///
/// Two stores hold the SAME vectors; one is configured cosine, one L2, and both
/// force an explicit `create_index` (honored below 50k thanks to R-38). The
/// corpus is constructed so the cosine-nearest and L2-nearest neighbours of the
/// query are DIFFERENT rows:
///   * candidate A = query direction scaled up 8x — cosine-identical (angle 0)
///     but L2-far;
///   * candidate B = query nudged slightly off-axis — L2-near but a different
///     angle;
///   * fillers are orthogonal to the query's axes — far in both metrics.
///
/// A cosine index must return A first; an L2 index must return B first. The
/// R-22 bug (index hardcoded to L2) would make the cosine store partition under
/// L2 and return the wrong top-1.
#[tokio::test(flavor = "multi_thread")]
async fn ann_index_honors_metric_cosine_vs_l2() {
    const DIM: usize = 32;
    const FILLERS: usize = 1_024;
    const ID_A: u64 = 90_001; // cosine-nearest (scaled query)
    const ID_B: u64 = 90_002; // L2-nearest (slightly off-axis)

    // Query points along axis 0 only.
    let mut query = vec![0.0_f32; DIM];
    query[0] = 1.0;

    // A: same direction, magnitude 8 → cosine dist 0, L2 dist^2 = 49.
    let mut vec_a = vec![0.0_f32; DIM];
    vec_a[0] = 8.0;

    // B: near query in L2 (dist^2 = 0.04) but a distinct angle.
    let mut vec_b = vec![0.0_f32; DIM];
    vec_b[0] = 1.0;
    vec_b[1] = 0.2;

    // Fillers live entirely in axes >= 2, so they are orthogonal to the query
    // (cosine dist ~1) and at L2 dist^2 >= 1 + |filler|^2 — never beating A or B.
    let mut ids = Vec::with_capacity(FILLERS + 2);
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(FILLERS + 2);
    for i in 0..FILLERS as u64 {
        let mut raw = pseudo_embedding(i, DIM);
        raw[0] = 0.0;
        raw[1] = 0.0;
        ids.push(i);
        embeddings.push(raw);
    }
    ids.push(ID_A);
    embeddings.push(vec_a);
    ids.push(ID_B);
    embeddings.push(vec_b);

    let batch = vector_batch(&ids, &embeddings, DIM);
    // Small explicit IVF so training is stable on ~1k rows.
    let index_cfg = IndexConfig::vector("embedding").with_params(IndexParams {
        num_partitions: Some(16),
        ..IndexParams::default()
    });

    // ── Cosine store: top-1 must be A (angle-aligned). ──
    let (_dir_c, cosine_store) = setup_store(DistanceMetric::Cosine).await;
    cosine_store
        .append("vecs", batch.clone())
        .await
        .unwrap();
    cosine_store
        .create_index("vecs", index_cfg.clone())
        .await
        .unwrap();
    assert!(
        cosine_store
            .vector_index_exists("vecs", "embedding")
            .await
            .unwrap(),
        "explicit index below 50k must be honored (R-38)"
    );
    let cosine_results = cosine_store
        .vector_search(
            "vecs",
            VectorSearchOptions {
                column: "embedding".to_string(),
                query: query.clone(),
                metric: DistanceMetric::Cosine,
                limit: 5,
                filter: None,
                nprobes: Some(16),
                refine_factor: Some(10),
            },
        )
        .await
        .unwrap();
    let cosine_top = result_ids(&cosine_results);
    assert_eq!(
        cosine_top.first().copied(),
        Some(ID_A),
        "cosine index must return the angle-aligned vector (A) first, got {cosine_top:?}"
    );

    // ── L2 store: top-1 must be B (Euclidean-nearest). ──
    let (_dir_l, l2_store) = setup_store(DistanceMetric::L2).await;
    l2_store.append("vecs", batch).await.unwrap();
    l2_store.create_index("vecs", index_cfg).await.unwrap();
    let l2_results = l2_store
        .vector_search(
            "vecs",
            VectorSearchOptions {
                column: "embedding".to_string(),
                query: query.clone(),
                metric: DistanceMetric::L2,
                limit: 5,
                filter: None,
                nprobes: Some(16),
                refine_factor: Some(10),
            },
        )
        .await
        .unwrap();
    let l2_top = result_ids(&l2_results);
    assert_eq!(
        l2_top.first().copied(),
        Some(ID_B),
        "L2 index must return the Euclidean-nearest vector (B) first, got {l2_top:?}"
    );

    // The two correctly-metricked indexes disagree on top-1 — exactly the signal
    // the R-22 bug would erase by forcing both to L2 partitions.
    assert_ne!(
        cosine_top.first(),
        l2_top.first(),
        "cosine and L2 top-1 must differ for this corpus"
    );
}

// ── Test 3: ANN and flat agree below the threshold ──

/// On a few-thousand-row cosine dataset, the forced-index (ANN) path must agree
/// with the exact flat path: identical top-1 and heavy top-10 overlap. Guards
/// that turning the index on does not change correctness near the boundary.
#[tokio::test(flavor = "multi_thread")]
async fn ann_and_flat_agree_below_threshold() {
    const ROWS: usize = 3_000; // below FLAT_VECTOR_CACHE_MAX_ROWS → flat is exact
    const DIM: usize = 32;
    const QUERIES: usize = 20;
    const K: usize = 10;
    const OVERLAP_FLOOR: f32 = 0.80;

    let ids: Vec<u64> = (0..ROWS as u64).collect();
    let embeddings: Vec<Vec<f32>> = (0..ROWS as u64).map(|i| pseudo_embedding(i, DIM)).collect();
    let batch = vector_batch(&ids, &embeddings, DIM);

    // Flat store: no index → exact brute-force path (ground truth).
    let (_dir_f, flat_store) = setup_store(DistanceMetric::Cosine).await;
    flat_store.append("vecs", batch.clone()).await.unwrap();
    assert!(
        !flat_store
            .vector_index_exists("vecs", "embedding")
            .await
            .unwrap(),
        "flat store must have no vector index"
    );

    // Indexed store: same vectors, forced ANN index (honored below 50k, R-38).
    let (_dir_i, indexed_store) = setup_store(DistanceMetric::Cosine).await;
    indexed_store.append("vecs", batch).await.unwrap();
    indexed_store
        .create_index(
            "vecs",
            IndexConfig::vector("embedding").with_params(IndexParams {
                num_partitions: Some(16),
                ..IndexParams::default()
            }),
        )
        .await
        .unwrap();
    assert!(
        indexed_store
            .vector_index_exists("vecs", "embedding")
            .await
            .unwrap(),
        "indexed store must use the ANN path"
    );

    let mk_opts = |query: Vec<f32>| VectorSearchOptions {
        column: "embedding".to_string(),
        query,
        metric: DistanceMetric::Cosine,
        limit: K,
        filter: None,
        nprobes: Some(64),
        refine_factor: Some(16),
    };

    let mut overlap_sum = 0.0_f32;
    for q in 0..QUERIES {
        let query = pseudo_embedding(20_000_000 + q as u64, DIM);

        let flat = result_ids(&flat_store.vector_search("vecs", mk_opts(query.clone())).await.unwrap());
        let ann = result_ids(&indexed_store.vector_search("vecs", mk_opts(query.clone())).await.unwrap());

        assert!(!flat.is_empty() && !ann.is_empty(), "both paths must return results");
        assert_eq!(
            flat.first(),
            ann.first(),
            "flat and ANN must agree on top-1 for query {q}: flat={flat:?} ann={ann:?}"
        );

        let flat_set: std::collections::HashSet<u64> = flat.into_iter().collect();
        let ann_set: std::collections::HashSet<u64> = ann.into_iter().collect();
        overlap_sum += flat_set.intersection(&ann_set).count() as f32 / K as f32;
    }

    let overlap = overlap_sum / QUERIES as f32;
    println!("ann_and_flat_agree_below_threshold: mean top-10 overlap = {overlap:.4} over {QUERIES} queries");
    assert!(
        overlap >= OVERLAP_FLOOR,
        "mean top-10 overlap {overlap:.4} fell below {OVERLAP_FLOOR}; ANN diverges from flat"
    );
}
