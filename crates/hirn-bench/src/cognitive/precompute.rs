//! Precompute embeddings for all synthetic datasets using OpenAI API.
//!
//! Generates all unique texts (turn contents and query questions), calls the
//! OpenAI embedding API once, and saves the results as a cache file that
//! the benchmark runner can load for reproducible, real-embedding evaluations.

use std::path::Path;

use super::Benchmark;
use super::dataset_embedding_texts;
use super::openai;
use super::synthetic;

/// Precompute embeddings for all (or selected) benchmarks and save to `output_dir`.
///
/// Creates one file per benchmark: `dmr_embeddings.json`, `locomo_embeddings.json`, etc.
/// Also creates `all_embeddings.json` combining everything.
pub fn precompute(
    benchmarks: &[Benchmark],
    api_key: &str,
    output_dir: &Path,
    model_config: &openai::EmbeddingModelConfig,
    max_api_texts: usize,
) -> Result<PrecomputeReport, String> {
    std::fs::create_dir_all(output_dir).map_err(|e| format!("failed to create output dir: {e}"))?;

    let mut total_texts = 0;
    let mut total_tokens_estimate = 0;
    let mut all_cache = openai::EmbeddingCache::new();

    for &bench in benchmarks {
        let dataset = synthetic::generate(bench);

        let texts = dataset_embedding_texts(&dataset);

        let text_list: Vec<String> = texts.into_iter().collect();
        let text_refs: Vec<&str> = text_list.iter().map(|s| s.as_str()).collect();

        eprintln!(
            "  {}: {} unique texts to embed",
            bench.name(),
            text_refs.len()
        );

        // Call OpenAI API.
        let embeddings = openai::batch_embed(api_key, &text_refs, max_api_texts, model_config)?;

        if embeddings.len() != text_refs.len() {
            return Err(format!(
                "{}: expected {} embeddings, got {}",
                bench.name(),
                text_refs.len(),
                embeddings.len()
            ));
        }

        // Build cache for this benchmark.
        let mut bench_cache = openai::EmbeddingCache::new();
        for (text, emb) in text_list.into_iter().zip(embeddings) {
            let tokens = text.split_whitespace().count(); // rough estimate
            total_tokens_estimate += tokens;
            bench_cache.insert(text.clone(), emb.clone());
            all_cache.insert(text, emb);
        }

        total_texts += bench_cache.len();

        // Save per-benchmark cache.
        let bench_path = output_dir.join(format!("{}_embeddings.json", bench.name()));
        openai::save_cache(&bench_path, &bench_cache)?;
        eprintln!("  → saved {}", bench_path.display());
    }

    // Save combined cache.
    let all_path = output_dir.join("all_embeddings.json");
    openai::save_cache(&all_path, &all_cache)?;
    eprintln!("  → saved {}", all_path.display());

    Ok(PrecomputeReport {
        total_texts,
        total_tokens_estimate,
        model: model_config.model.clone(),
        dims: model_config.dims,
        benchmarks: benchmarks.len(),
    })
}

/// Precompute embeddings for an external dataset (LoCoMo, DMR, LongMemEval).
///
/// Loads the dataset, extracts all unique texts (formatted the same way the
/// runner ingests them), calls the OpenAI embedding API, and saves the cache.
pub fn precompute_external(
    dataset: &super::CognitiveDataset,
    api_key: &str,
    output_path: &Path,
    model_config: &openai::EmbeddingModelConfig,
    max_api_texts: usize,
) -> Result<PrecomputeReport, String> {
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("failed to create output dir: {e}"))?;
    }

    let texts = dataset_embedding_texts(dataset);

    let text_list: Vec<String> = texts.into_iter().collect();
    let text_refs: Vec<&str> = text_list.iter().map(|s| s.as_str()).collect();

    eprintln!(
        "  {}: {} unique texts to embed",
        dataset.name,
        text_refs.len()
    );

    // Check for existing cache and skip already-embedded texts.
    let existing_cache = if output_path.exists() {
        openai::load_cache(output_path).unwrap_or_default()
    } else {
        openai::EmbeddingCache::new()
    };
    let mut cache: openai::EmbeddingCache = text_refs
        .iter()
        .filter_map(|text| {
            existing_cache
                .get(*text)
                .cloned()
                .map(|embedding| ((*text).to_string(), embedding))
        })
        .collect();

    let new_texts: Vec<&str> = text_refs
        .iter()
        .filter(|t| !cache.contains_key(**t))
        .copied()
        .collect();

    if new_texts.is_empty() {
        eprintln!(
            "  All {} texts already cached, nothing to embed.",
            text_refs.len()
        );
        openai::save_cache(output_path, &cache)?;
        return Ok(PrecomputeReport {
            total_texts: cache.len(),
            total_tokens_estimate: 0,
            model: model_config.model.clone(),
            dims: model_config.dims,
            benchmarks: 1,
        });
    }

    eprintln!(
        "  {} already cached, {} new texts to embed",
        cache.len(),
        new_texts.len()
    );

    let embeddings = openai::batch_embed(api_key, &new_texts, max_api_texts, model_config)?;

    if embeddings.len() != new_texts.len() {
        return Err(format!(
            "expected {} embeddings, got {}",
            new_texts.len(),
            embeddings.len()
        ));
    }

    let mut total_tokens_estimate = 0;
    for (text, emb) in new_texts.into_iter().zip(embeddings) {
        total_tokens_estimate += text.split_whitespace().count();
        cache.insert(text.to_string(), emb);
    }

    openai::save_cache(output_path, &cache)?;
    eprintln!(
        "  → saved {} embeddings to {}",
        cache.len(),
        output_path.display()
    );

    Ok(PrecomputeReport {
        total_texts: cache.len(),
        total_tokens_estimate,
        model: model_config.model.clone(),
        dims: model_config.dims,
        benchmarks: 1,
    })
}

/// Summary of a precompute run.
pub struct PrecomputeReport {
    pub total_texts: usize,
    pub total_tokens_estimate: usize,
    pub model: String,
    pub dims: usize,
    pub benchmarks: usize,
}
