//! OpenAI embedding client for precomputing real embeddings.

use std::collections::HashMap;

/// Default embedding model.
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";
/// Default embedding dimensions for `text-embedding-3-small`.
pub const DEFAULT_EMBEDDING_DIMS: usize = 1536;
/// OpenAI batch limit is 2048 inputs; we stay well under.
const BATCH_SIZE: usize = 100;

/// Configuration for the embedding provider.
#[derive(Debug, Clone)]
pub struct EmbeddingModelConfig {
    pub model: String,
    pub dims: usize,
}

impl Default for EmbeddingModelConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            dims: DEFAULT_EMBEDDING_DIMS,
        }
    }
}

/// Precomputed embeddings: text → vector.
pub type EmbeddingCache = HashMap<String, Vec<f32>>;

/// Embed a batch of texts via the OpenAI API.
///
/// Uses the model specified in `model_config`. Returns vectors in the same
/// order as input texts. Batches requests to stay within API limits.
///
/// `max_texts` limits the number of texts to prevent accidental unbounded spend.
pub fn batch_embed(
    api_key: &str,
    texts: &[&str],
    max_texts: usize,
    model_config: &EmbeddingModelConfig,
) -> Result<Vec<Vec<f32>>, String> {
    if texts.len() > max_texts {
        return Err(format!(
            "Refusing to embed {} texts (limit: {}). Use --max-texts to increase.",
            texts.len(),
            max_texts
        ));
    }
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for chunk in texts.chunks(BATCH_SIZE) {
        let body = serde_json::json!({
            "model": model_config.model,
            "input": chunk,
        });

        let response: serde_json::Value = ureq::post("https://api.openai.com/v1/embeddings")
            .header("Authorization", &format!("Bearer {api_key}"))
            .send_json(&body)
            .map_err(|e| format!("OpenAI API request failed: {e}"))?
            .body_mut()
            .read_json()
            .map_err(|e| format!("Failed to parse OpenAI response: {e}"))?;

        if let Some(err) = response.get("error") {
            return Err(format!("OpenAI API error: {err}"));
        }

        let data = response["data"]
            .as_array()
            .ok_or("missing 'data' in OpenAI response")?;

        // API returns embeddings sorted by index, but be safe.
        let mut indexed: Vec<(usize, Vec<f32>)> = data
            .iter()
            .map(|item| {
                let idx = item["index"].as_u64().unwrap_or(0) as usize;
                let embedding: Vec<f32> = item["embedding"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                    .collect();
                (idx, embedding)
            })
            .collect();
        indexed.sort_by_key(|(idx, _)| *idx);

        for (_, emb) in indexed {
            all_embeddings.push(emb);
        }
    }

    Ok(all_embeddings)
}

/// Returns the default model name used for embeddings.
pub fn model_name() -> &'static str {
    DEFAULT_EMBEDDING_MODEL
}

/// Returns the default embedding dimensions.
pub fn embedding_dims() -> usize {
    DEFAULT_EMBEDDING_DIMS
}

/// Load an embedding cache from a bincode file (or fall back to JSON for migration).
/// Bincode format: serialized `EmbeddingCache` (`HashMap<String, Vec<f32>>`).
/// JSON format: `{"texts": {"text1": [0.1, 0.2, ...], ...}, "model": "...", "dims": 1536}`
pub fn load_cache(path: &std::path::Path) -> Result<EmbeddingCache, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("failed to read embedding cache {}: {e}", path.display()))?;

    // Try bincode first.
    if let Ok(cache) = bincode::deserialize::<EmbeddingCache>(&bytes) {
        return Ok(cache);
    }

    // Fall back to JSON for backward compatibility / migration.
    let content = std::str::from_utf8(&bytes)
        .map_err(|e| format!("embedding cache is neither valid bincode nor UTF-8: {e}"))?;
    let parsed: serde_json::Value = serde_json::from_str(content)
        .map_err(|e| format!("failed to parse embedding cache as JSON: {e}"))?;

    let texts = parsed["texts"]
        .as_object()
        .ok_or("missing 'texts' in embedding cache")?;

    let mut cache = HashMap::with_capacity(texts.len());
    for (key, val) in texts {
        let embedding: Vec<f32> = val
            .as_array()
            .ok_or_else(|| format!("embedding for '{key}' is not an array"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        cache.insert(key.clone(), embedding);
    }

    Ok(cache)
}

/// Save an embedding cache as JSON (matches `load_cache_from_file` format).
pub fn save_cache(path: &std::path::Path, cache: &EmbeddingCache) -> Result<(), String> {
    // F-85: Use JSON format consistently with load_cache.
    let texts: serde_json::Map<String, serde_json::Value> = cache
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                serde_json::Value::Array(v.iter().map(|&f| serde_json::json!(f)).collect()),
            )
        })
        .collect();
    let json = serde_json::json!({ "texts": texts });
    let content = serde_json::to_string_pretty(&json)
        .map_err(|e| format!("failed to serialize cache as JSON: {e}"))?;
    std::fs::write(path, content)
        .map_err(|e| format!("failed to write cache to {}: {e}", path.display()))?;

    Ok(())
}
