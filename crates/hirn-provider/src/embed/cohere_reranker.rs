//! `CohereReranker` — production reranker using the Cohere Rerank v3.5 API.

use async_trait::async_trait;
use hirn_core::embed::{RerankResult, Reranker};
use hirn_core::error::HirnResult;
use reqwest::Client;
use secrecy::{ExposeSecret, SecretString};
use std::time::Duration;
use tracing::{debug, warn};

use crate::metrics::record_invalid_reranker_score;

use super::error::{EmbedError, parse_retry_after};

/// Default request timeout for HTTP calls.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
/// Default connection timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Reranker backed by Cohere's Rerank API.
///
/// Uses the [Cohere Rerank v3.5](https://docs.cohere.com/reference/rerank) model
/// which supports multilingual queries and up to 4096-token context per document.
///
/// # Example
///
/// ```rust,no_run
/// use hirn_provider::CohereReranker;
///
/// let reranker = CohereReranker::new("your-api-key")
///     .expect("cohere client should initialize");
/// ```
#[derive(Debug)]
pub struct CohereReranker {
    client: Client,
    api_key: SecretString,
    model: String,
    base_url: String,
}

impl CohereReranker {
    /// Create a new Cohere reranker with the given API key.
    ///
    /// Defaults to the `rerank-v3.5` model.
    pub fn new(api_key: impl Into<String>) -> HirnResult<Self> {
        Ok(Self {
            client: super::build_http_client(
                "cohere",
                Client::builder()
                    .timeout(REQUEST_TIMEOUT)
                    .connect_timeout(CONNECT_TIMEOUT),
            )?,
            api_key: SecretString::from(api_key.into()),
            model: "rerank-v3.5".to_owned(),
            base_url: "https://api.cohere.com/v2".to_owned(),
        })
    }

    /// Create from the `COHERE_API_KEY` environment variable.
    ///
    /// Returns `None` if the variable is not set.
    pub fn from_env() -> HirnResult<Option<Self>> {
        std::env::var("COHERE_API_KEY")
            .ok()
            .map(Self::new)
            .transpose()
    }

    /// Override the model (e.g. `rerank-english-v3.0`).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the base URL (for proxies / testing).
    ///
    /// Secret-bearing Cohere traffic requires HTTPS unless the endpoint is
    /// loopback HTTP for local development or tests.
    pub fn with_base_url(mut self, url: impl Into<String>) -> HirnResult<Self> {
        let url = url.into();
        crate::transport::validate_secret_bearing_base_url("cohere", &url)?;
        self.base_url = url;
        Ok(self)
    }
}

// ── Cohere Rerank API types ────────────────────────────────────────

#[derive(serde::Serialize)]
struct RerankRequest<'a> {
    model: &'a str,
    query: &'a str,
    documents: Vec<&'a str>,
    top_n: usize,
}

#[derive(serde::Deserialize)]
struct RerankResponse {
    results: Vec<RerankHit>,
}

#[derive(serde::Deserialize)]
struct RerankHit {
    index: usize,
    relevance_score: f32,
}

impl CohereReranker {
    fn into_result(hit: RerankHit, num_docs: usize, provider: &str) -> Option<RerankResult> {
        if hit.index >= num_docs {
            warn!(
                index = hit.index,
                num_docs, "cohere returned out-of-bounds index, skipping"
            );
            record_invalid_reranker_score(provider, "out_of_bounds");
            return None;
        }

        if !hit.relevance_score.is_finite() {
            warn!(
                index = hit.index,
                score = %hit.relevance_score,
                "cohere returned non-finite score, skipping"
            );
            record_invalid_reranker_score(provider, "non_finite");
            return None;
        }

        Some(RerankResult {
            index: hit.index,
            score: hit.relevance_score,
        })
    }
}

// ── Reranker trait impl ────────────────────────────────────────────

#[async_trait]
impl Reranker for CohereReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> HirnResult<Vec<RerankResult>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }

        let top_n = top_k.min(documents.len());
        let body = RerankRequest {
            model: &self.model,
            query,
            documents: documents.to_vec(),
            top_n,
        };

        let resp = self
            .client
            .post(format!("{}/rerank", self.base_url))
            .header(
                "Authorization",
                format!("Bearer {}", self.api_key.expose_secret()),
            )
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbedError::from_reqwest(&self.model, e))?;

        let status = resp.status();
        if !status.is_success() {
            let status_code = status.as_u16();
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(parse_retry_after);
            let body_text = resp.text().await.unwrap_or_default();
            return Err(
                EmbedError::from_status(&self.model, status_code, body_text, retry_after).into(),
            );
        }

        let parsed: RerankResponse =
            resp.json().await.map_err(|e| EmbedError::InvalidResponse {
                provider: self.model.clone(),
                details: format!("Cohere rerank parse error: {e}"),
            })?;

        debug!(
            model = %self.model,
            query_len = query.len(),
            docs = documents.len(),
            returned = parsed.results.len(),
            "cohere rerank complete"
        );

        let num_docs = documents.len();

        Ok(parsed
            .results
            .into_iter()
            .filter_map(|hit| Self::into_result(hit, num_docs, &self.model))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::any::TypeId;

    use super::*;
    use hirn_core::embed::Reranker;
    use metrics_util::debugging::{DebugValue, DebuggingRecorder};
    use metrics_util::{CompositeKey, MetricKind};

    type Snap = Vec<(
        CompositeKey,
        Option<metrics::Unit>,
        Option<metrics::SharedString>,
        DebugValue,
    )>;

    fn counter_with_labels(snap: &Snap, name: &str, labels: &[(&str, &str)]) -> u64 {
        snap.iter()
            .filter(|(key, _, _, _)| {
                key.kind() == MetricKind::Counter
                    && key.key().name() == name
                    && labels.iter().all(|(label_key, label_value)| {
                        key.key()
                            .labels()
                            .any(|label| label.key() == *label_key && label.value() == *label_value)
                    })
            })
            .map(|(_, _, _, value)| match value {
                DebugValue::Counter(count) => *count,
                _ => 0,
            })
            .sum()
    }

    #[tokio::test]
    async fn empty_documents_returns_empty() {
        let reranker = CohereReranker::new("fake-key").expect("cohere client should initialize");
        let result = reranker.rerank("query", &[], 5).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn remote_plaintext_base_url_is_rejected() {
        let result = CohereReranker::new("fake-key")
            .expect("cohere client should initialize")
            .with_base_url("http://example.com/v2");
        assert!(result.is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mock_server_rerank() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let base_url = format!("http://127.0.0.1:{port}");

        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _n = stream.read(&mut buf).await.unwrap();

            let body = serde_json::json!({
                "results": [
                    { "index": 2, "relevance_score": 0.95 },
                    { "index": 0, "relevance_score": 0.80 },
                    { "index": 1, "relevance_score": 0.30 }
                ]
            });
            let body_str = body.to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                body_str.len(),
                body_str
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        });

        let reranker = CohereReranker::new("test-key")
            .expect("cohere client should initialize")
            .with_base_url(base_url)
            .expect("loopback http endpoint should be accepted");

        let docs = &["first doc", "second doc", "third doc"];
        let result = reranker.rerank("test query", docs, 3).await.unwrap();

        assert_eq!(result.len(), 3);
        assert_eq!(result[0].index, 2); // "third doc" ranked first
        assert!(result[0].score > result[1].score);
        assert!(result[1].score > result[2].score);

        handle.await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn api_error_returns_err() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let base_url = format!("http://127.0.0.1:{port}");

        let handle = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await.unwrap();

            let body = r#"{"message":"invalid api key"}"#;
            let response = format!(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            stream.shutdown().await.unwrap();
        });

        let reranker = CohereReranker::new("bad-key")
            .expect("cohere client should initialize")
            .with_base_url(base_url)
            .expect("loopback http endpoint should be accepted");
        let result = reranker.rerank("query", &["doc"], 1).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("auth failed") || err_msg.contains("access denied"),
            "should mention auth failure: {err_msg}"
        );

        handle.await.unwrap();
    }

    #[test]
    fn invalid_hits_are_dropped_and_counted() {
        let provider = "rerank-v3.5";
        let recorder = DebuggingRecorder::new();
        let snapshotter = recorder.snapshotter();

        let out_of_bounds = metrics::with_local_recorder(&recorder, || {
            CohereReranker::into_result(
                RerankHit {
                    index: 5,
                    relevance_score: 0.9,
                },
                2,
                provider,
            )
        });
        assert!(out_of_bounds.is_none());

        let non_finite = metrics::with_local_recorder(&recorder, || {
            CohereReranker::into_result(
                RerankHit {
                    index: 0,
                    relevance_score: f32::NAN,
                },
                2,
                provider,
            )
        });
        assert!(non_finite.is_none());

        let valid = CohereReranker::into_result(
            RerankHit {
                index: 1,
                relevance_score: 0.75,
            },
            2,
            provider,
        )
        .expect("valid hit should be preserved");
        assert_eq!(valid.index, 1);
        assert!((valid.score - 0.75).abs() < f32::EPSILON);

        let snap = snapshotter.snapshot().into_vec();
        assert_eq!(
            counter_with_labels(
                &snap,
                crate::metrics::RERANKER_INVALID_SCORES_TOTAL,
                &[("provider", provider), ("reason", "out_of_bounds")],
            ),
            1
        );
        assert_eq!(
            counter_with_labels(
                &snap,
                crate::metrics::RERANKER_INVALID_SCORES_TOTAL,
                &[("provider", provider), ("reason", "non_finite")],
            ),
            1
        );
    }

    #[test]
    fn llm_reexport_points_to_canonical_cohere_reranker() {
        assert_eq!(
            TypeId::of::<crate::embed::CohereReranker>(),
            TypeId::of::<crate::llm::CohereReranker>()
        );
    }
}
