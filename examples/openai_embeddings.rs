//! # OpenAI Embeddings — Real Vector Search with hirn
//!
//! This example connects hirn to OpenAI's embedding API for production-quality
//! semantic search. It demonstrates:
//!
//! 1. Generating real embeddings via the OpenAI API
//! 2. Storing memories with production-grade vectors
//! 3. Semantic recall with meaningful similarity scores
//! 4. Building LLM context with `think()`
//!
//! ## Prerequisites
//!
//! Set the `OPENAI_API_KEY` environment variable:
//! ```bash
//! export OPENAI_API_KEY="sk-..."
//! ```
//!
//! Run with:
//! ```bash
//! cargo run --example openai_embeddings -p hirn --features openai
//! ```
//!
//! ## Note
//!
//! This example uses `text-embedding-3-small` (1536 dimensions) which costs
//! ~$0.02 per 1M tokens. The example uses ~500 tokens total (~$0.00001).

use hirn::prelude::*;
use hirn_storage::{HirnDb, HirnDbConfig};
use std::io::Read;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| "OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=\"sk-...\"")?;

    let dir = tempfile::tempdir()?;
    let path = dir.path().join("openai_brain");

    // text-embedding-3-small produces 1536-dimensional vectors
    let config = HirnConfig::builder()
        .db_path(&path)
        .embedding_dimensions(1536)
        .build()?;
    let storage_config = HirnDbConfig::local(path.join("lance").to_str().unwrap());
    let storage = HirnDb::open(storage_config).await?.store_arc();
    let brain = Hirn::open_with_config(config, storage).await?;

    let agent = AgentId::new("researcher").expect("non-empty agent id");
    brain.register_agent(&agent, "AI Researcher").await?;
    println!("✓ Opened database with 1536-dim embeddings");

    // ── Knowledge base ──────────────────────────────────────────────────
    let knowledge = [
        (
            "HNSW (Hierarchical Navigable Small World) achieves sub-linear \
             search time by building a multi-layer proximity graph. Key \
             parameters: M controls graph connectivity, ef controls search \
             beam width. Higher M = better recall but more memory.",
            EventType::Observation,
            0.90,
        ),
        (
            "Product quantization (PQ) compresses vectors by splitting them \
             into subvectors and quantizing each independently. Achieves 4-32x \
             compression with moderate recall loss. Best combined with IVF \
             for large-scale deployments.",
            EventType::Observation,
            0.85,
        ),
        (
            "Benchmark: HNSW with M=16, ef=200 achieves 99.2% recall@10 on \
             SIFT-1M dataset with 0.5ms query latency. Brute force: 45ms. \
             IVF-PQ: 2ms at 94% recall.",
            EventType::Experiment,
            0.88,
        ),
        (
            "For RAG applications, text-embedding-3-small (1536 dims) offers \
             the best cost-performance trade-off. text-embedding-3-large (3072 \
             dims) gives 2% better accuracy but doubles storage and compute.",
            EventType::Decision,
            0.82,
        ),
        (
            "Cosine similarity is preferred over L2 distance for normalized \
             text embeddings because it's invariant to vector magnitude, \
             focusing purely on directional similarity.",
            EventType::Observation,
            0.75,
        ),
    ];

    // ── Generate embeddings and store ────────────────────────────────────
    println!("→ Generating embeddings via OpenAI API...");

    let texts: Vec<&str> = knowledge.iter().map(|(text, _, _)| *text).collect();
    let embeddings = batch_embed(&api_key, &texts)?;

    for (i, ((content, event_type, importance), embedding)) in
        knowledge.iter().zip(embeddings.iter()).enumerate()
    {
        let episode = EpisodicRecord::builder()
            .content(*content)
            .event_type(*event_type)
            .agent_id(agent.clone())
            .importance(*importance)
            .embedding(embedding.clone())
            .build()?;

        brain.episodic().remember(episode).await?;
        println!(
            "  [{}/{}] Stored: {:.60}...",
            i + 1,
            knowledge.len(),
            content
        );
    }
    println!(
        "✓ Stored {} memories with real embeddings\n",
        knowledge.len()
    );

    // ── Semantic search ─────────────────────────────────────────────────
    let queries = [
        "What's the fastest algorithm for nearest neighbor search?",
        "How can I reduce memory usage for vector storage?",
        "Which embedding model should I use for my RAG app?",
    ];

    for query in &queries {
        println!("── Query: \"{query}\" ──");

        let query_embedding = embed(&api_key, query)?;

        let results = brain
            .recall_view()
            .query(query_embedding.clone())
            .activation(ActivationMode::Spreading)
            .limit(3)
            .execute()
            .await?;

        for (i, r) in results.iter().enumerate() {
            let content = match &r.record {
                MemoryRecord::Episodic(ep) => &ep.content,
                _ => continue,
            };
            println!(
                "  #{}: score={:.3} sim={:.3} | {:.70}...",
                i + 1,
                r.composite_score,
                r.similarity,
                content,
            );
        }

        // ── Think — assemble LLM context ────────────────────────────────
        let context = brain
            .recall_view()
            .think(query_embedding)
            .budget(1024)
            .execute()
            .await?;

        println!(
            "  → Think: {} tokens, {} records included\n",
            context.token_count,
            context.records_included.len()
        );
    }

    println!("✓ OpenAI embeddings demo complete!");
    Ok(())
}

/// Embed a single text using OpenAI's API.
fn embed(api_key: &str, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let embeddings = batch_embed(api_key, &[text])?;
    Ok(embeddings.into_iter().next().unwrap())
}

/// Batch-embed multiple texts in a single API call.
fn batch_embed(api_key: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let body = serde_json::json!({
        "model": "text-embedding-3-small",
        "input": texts,
    });

    // Use std::net for a minimal HTTP client (no extra dependencies)
    use std::io::Write;
    use std::net::TcpStream;

    let host = "api.openai.com";
    let body_bytes = serde_json::to_vec(&body)?;

    let request = format!(
        "POST /v1/embeddings HTTP/1.1\r\n\
         Host: {host}\r\n\
         Authorization: Bearer {api_key}\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body_bytes.len()
    );

    // Connect with TLS
    let tcp = TcpStream::connect((host, 443))?;
    let connector = native_tls::TlsConnector::new()?;
    let mut stream = connector.connect(host, tcp)?;

    stream.write_all(request.as_bytes())?;
    stream.write_all(&body_bytes)?;

    let mut response = String::new();
    stream.read_to_string(&mut response)?;

    // Parse HTTP response — skip headers
    let body_start = response.find("\r\n\r\n").ok_or("invalid HTTP response")? + 4;
    let response_body = &response[body_start..];

    // Handle chunked transfer encoding
    let json_str = if response.contains("Transfer-Encoding: chunked") {
        decode_chunked(response_body)?
    } else {
        response_body.to_string()
    };

    let parsed: serde_json::Value = serde_json::from_str(&json_str)?;

    if let Some(err) = parsed.get("error") {
        return Err(format!("OpenAI API error: {err}").into());
    }

    let data = parsed["data"]
        .as_array()
        .ok_or("missing 'data' in response")?;

    let mut embeddings = Vec::with_capacity(data.len());
    for item in data {
        let embedding: Vec<f32> = item["embedding"]
            .as_array()
            .ok_or("missing embedding")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect();
        embeddings.push(embedding);
    }

    Ok(embeddings)
}

/// Decode HTTP chunked transfer encoding.
fn decode_chunked(body: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut decoded = String::new();
    let mut remaining = body;

    loop {
        // Find chunk size line
        let newline = remaining.find("\r\n").ok_or("malformed chunk")?;
        let size_str = remaining[..newline].trim();
        let size = usize::from_str_radix(size_str, 16)?;

        if size == 0 {
            break;
        }

        let chunk_start = newline + 2;
        let chunk_end = chunk_start + size;
        if chunk_end > remaining.len() {
            // Incomplete chunk — take what we have
            decoded.push_str(&remaining[chunk_start..]);
            break;
        }
        decoded.push_str(&remaining[chunk_start..chunk_end]);
        remaining = &remaining[chunk_end + 2..]; // skip \r\n after chunk
    }

    Ok(decoded)
}
