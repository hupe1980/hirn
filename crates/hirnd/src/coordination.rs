use axum::Json;
use axum::http::{HeaderMap, StatusCode};
use metrics::counter;
use serde::Serialize;
use serde_json::Value;

use crate::http::{CachedJsonResponse, ErrorResponse, HttpState};
use crate::raft::NodeId;

type HttpError = (StatusCode, Json<ErrorResponse>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RealmWriteOwner {
    pub(crate) node_id: NodeId,
    pub(crate) addr: String,
}

pub(crate) struct ForwardedWriteResponse {
    pub(crate) owner: RealmWriteOwner,
    pub(crate) response: CachedJsonResponse,
}

const FORWARDED_WRITE_HEADERS: &[&str] = &[
    "x-agent-id",
    "x-trace-id",
    "x-idempotency-key",
    "authorization",
];

pub(crate) struct CoordinationRuntime;

impl CoordinationRuntime {
    pub(crate) fn local_node_id(state: &HttpState) -> Option<NodeId> {
        state.raft.as_ref().map(|raft| raft.metrics().borrow().id)
    }

    pub(crate) async fn current_realm_owner(state: &HttpState, realm: &str) -> Option<NodeId> {
        let sm = state.raft_state_machine.as_ref()?;
        sm.realm_owner(realm).await
    }

    /// Returns the remote owner when this request should be forwarded.
    pub(crate) async fn realm_write_owner(
        state: &HttpState,
        realm: &str,
    ) -> Result<Option<RealmWriteOwner>, HttpError> {
        let (Some(my_id), Some(sm)) = (Self::local_node_id(state), &state.raft_state_machine)
        else {
            return Ok(None);
        };

        let Some(owner_node_id) = sm.realm_owner(realm).await else {
            return Ok(None);
        };
        if owner_node_id == my_id {
            return Ok(None);
        }

        let Some(owner_addr) = sm.node_addr(owner_node_id).await else {
            return Err((
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::with_retryable(
                    format!(
                        "realm '{realm}' is assigned to owner node {owner_node_id} but no owner address is registered"
                    ),
                    true,
                )),
            ));
        };

        Ok(Some(RealmWriteOwner {
            node_id: owner_node_id,
            addr: owner_addr,
        }))
    }

    /// Forward a realm-owned write to the current owner node.
    ///
    /// When the realm has no assigned owner, or the daemon is running without
    /// cluster metadata, this returns `Ok(None)` and the caller must execute the
    /// request locally. Transport failures are surfaced as retryable gateway
    /// errors. Forwarded owner responses preserve their status codes and, for
    /// error bodies, gain a `retryable` flag when the owner did not provide one.
    pub(crate) async fn try_forward_write(
        state: &HttpState,
        headers: &HeaderMap,
        path: &str,
        body: &[u8],
    ) -> Result<Option<ForwardedWriteResponse>, HttpError> {
        let realm = headers
            .get("x-realm-id")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse::new("missing X-Realm-ID header")),
                )
            })?;

        let Some(owner) = Self::realm_write_owner(state, realm).await? else {
            // Unassigned realms and standalone mode fall back to local execution.
            return Ok(None);
        };

        let parsed = Self::build_forward_url(&owner.addr, path)?;

        let mut request = state
            .forward_client
            .post(parsed)
            .header("x-realm-id", realm);
        for key in FORWARDED_WRITE_HEADERS {
            if let Some(value) = headers.get(*key) {
                request = request.header(*key, value.as_bytes());
            }
        }

        let response = request
            .header("content-type", "application/json")
            .body(body.to_vec())
            .send()
            .await
            .map_err(|error| {
                (
                    StatusCode::BAD_GATEWAY,
                    Json(ErrorResponse::with_retryable(
                        format!("failed to forward to owner node: {error}"),
                        true,
                    )),
                )
            })?;

        let status =
            StatusCode::from_u16(response.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        let response_body = response.bytes().await.map_err(|error| {
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::with_retryable(
                    format!("failed to read forwarded response: {error}"),
                    true,
                )),
            )
        })?;

        let response_body = if status.is_client_error() || status.is_server_error() {
            Self::annotate_forwarded_error(status, response_body.as_ref())
        } else {
            response_body.to_vec()
        };

        counter!(
            "hirnd_forwarded_requests_total",
            "path" => path.to_owned(),
            "realm" => realm.to_owned()
        )
        .increment(1);

        Ok(Some(ForwardedWriteResponse {
            owner,
            response: CachedJsonResponse::from_parts(status, response_body),
        }))
    }

    pub(crate) async fn forward_json_write<T: Serialize + Sync>(
        state: &HttpState,
        headers: &HeaderMap,
        path: &str,
        body: &T,
    ) -> Result<Option<ForwardedWriteResponse>, HttpError> {
        let body = serde_json::to_vec(body).map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::with_retryable(
                    format!("failed to serialize forwarded request: {error}"),
                    false,
                )),
            )
        })?;

        Self::try_forward_write(state, headers, path, &body).await
    }

    fn build_forward_url(owner_addr: &str, path: &str) -> Result<reqwest::Url, HttpError> {
        let base = reqwest::Url::parse(owner_addr).map_err(|_| {
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::with_retryable(
                    "invalid owner node address; expected an explicit http:// or https:// URL",
                    false,
                )),
            )
        })?;

        match base.scheme() {
            "https" => {
                // N-H04: SSRF guard — reject cloud metadata IP ranges even over HTTPS.
                Self::reject_ssrf_target(&base)?;
            }
            "http" if Self::is_loopback_http_endpoint(&base) => {
                tracing::warn!(
                    owner_addr,
                    "using plaintext loopback owner forwarding endpoint"
                );
            }
            "http" => {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    Json(ErrorResponse::with_retryable(
                        "owner node forwarding requires HTTPS; only loopback HTTP is allowed for local development",
                        false,
                    )),
                ));
            }
            _ => {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    Json(ErrorResponse::with_retryable(
                        "invalid owner node address scheme",
                        false,
                    )),
                ));
            }
        }

        base.join(path).map_err(|_| {
            (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::with_retryable(
                    "invalid forwarded request path",
                    false,
                )),
            )
        })
    }

    /// Reject SSRF targets: cloud metadata endpoints, RFC1918, link-local,
    /// loopback, CGNAT (100.64/10), and known cloud metadata hostnames.
    ///
    /// Only checks hostnames that are IP literals or well-known metadata names;
    /// DNS-resolved IPs are validated at the reqwest client level via the
    /// `no_proxy` / custom DNS resolver if configured.
    fn reject_ssrf_target(url: &reqwest::Url) -> Result<(), HttpError> {
        let host = url.host_str().unwrap_or("");

        // Block known cloud metadata hostnames regardless of scheme.
        if matches!(
            host,
            "169.254.169.254"
                | "metadata.google.internal"
                | "metadata.goog"
                | "fd69::1"
                | "100.100.100.200" // Alibaba Cloud ECS metadata
        ) {
            return Err((
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::with_retryable(
                    "forwarding to cloud metadata endpoints is not allowed",
                    false,
                )),
            ));
        }

        // Block IP literals in forbidden ranges.
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            if Self::is_forbidden_ip(ip) {
                return Err((
                    StatusCode::BAD_GATEWAY,
                    Json(ErrorResponse::with_retryable(
                        "forwarding to reserved/private IP ranges is not allowed",
                        false,
                    )),
                ));
            }
        }

        Ok(())
    }

    /// Returns `true` for IP addresses that must not be forwarded to:
    /// loopback, link-local (169.254/16, fe80::/10), RFC1918 private
    /// (10/8, 172.16/12, 192.168/16), CGNAT (100.64/10), unique-local
    /// IPv6 (fc00::/7), and multicast.
    fn is_forbidden_ip(ip: std::net::IpAddr) -> bool {
        use std::net::{IpAddr, Ipv4Addr};
        match ip {
            IpAddr::V4(v4) => {
                v4.is_loopback()
                    || v4.is_link_local()
                    || v4.is_private()
                    || v4.is_broadcast()
                    // CGNAT: 100.64.0.0/10
                    || (u32::from(v4) >> 22 == u32::from(Ipv4Addr::new(100, 64, 0, 0)) >> 22)
                    // Documentation ranges: 192.0.2/24, 198.51.100/24, 203.0.113/24
                    || matches!(
                        v4.octets(),
                        [192, 0, 2, _] | [198, 51, 100, _] | [203, 0, 113, _]
                    )
            }
            IpAddr::V6(v6) => {
                let octets = v6.octets();
                v6.is_loopback()
                    || v6.is_multicast()
                    // Link-local: fe80::/10
                    || (octets[0] == 0xfe && (octets[1] & 0xc0) == 0x80)
                    // Unique-local: fc00::/7
                    || (octets[0] & 0xfe == 0xfc)
                    // Documentation range: 2001:db8::/32 (RFC 3849)
                    || (octets[0] == 0x20 && octets[1] == 0x01
                        && octets[2] == 0x0d && octets[3] == 0xb8)
                    // Check IPv4-mapped IPv6 addresses
                    || matches!(v6.to_ipv4_mapped(), Some(v4) if Self::is_forbidden_ip(IpAddr::V4(v4)))
            }
        }
    }

    fn is_loopback_http_endpoint(url: &reqwest::Url) -> bool {
        matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"))
    }

    fn annotate_forwarded_error(status: StatusCode, response_body: &[u8]) -> Vec<u8> {
        let retryable = Self::is_retryable_status(status);
        let annotated = match serde_json::from_slice::<Value>(response_body) {
            Ok(Value::Object(mut object)) => {
                object
                    .entry("retryable".to_owned())
                    .or_insert(Value::Bool(retryable));
                Value::Object(object)
            }
            Ok(other) => serde_json::json!({
                "error": other,
                "retryable": retryable,
            }),
            Err(_) => serde_json::json!({
                "error": String::from_utf8_lossy(response_body),
                "retryable": retryable,
            }),
        };

        serde_json::to_vec(&annotated).unwrap_or_else(|_| {
            format!(
                r#"{{"error":"failed to encode forwarded error response","retryable":{retryable}}}"#
            )
            .into_bytes()
        })
    }

    fn is_retryable_status(status: StatusCode) -> bool {
        status == StatusCode::REQUEST_TIMEOUT
            || status == StatusCode::TOO_MANY_REQUESTS
            || status.is_server_error()
    }
}

#[cfg(test)]
mod tests {
    use super::CoordinationRuntime;
    use axum::http::StatusCode;

    #[test]
    fn build_forward_url_accepts_loopback_http() {
        let parsed =
            CoordinationRuntime::build_forward_url("http://127.0.0.1:8080", "/v1/remember")
                .expect("loopback http should be accepted for local development");
        assert_eq!(parsed.as_str(), "http://127.0.0.1:8080/v1/remember");
    }

    #[test]
    fn build_forward_url_rejects_plain_hosts() {
        let err = CoordinationRuntime::build_forward_url("127.0.0.1:8080", "/v1/remember")
            .expect_err("owner addresses must include an explicit scheme");
        assert_eq!(err.0, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn build_forward_url_rejects_remote_http() {
        let err = CoordinationRuntime::build_forward_url("http://example.com", "/v1/remember")
            .expect_err("remote plaintext forwarding must be rejected");
        assert_eq!(err.0, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn build_forward_url_rejects_non_http_schemes() {
        let err = CoordinationRuntime::build_forward_url("ftp://127.0.0.1:8080", "/v1/remember")
            .expect_err("non-http schemes must be rejected");
        assert_eq!(err.0, StatusCode::BAD_GATEWAY);
    }

    // N-H04: SSRF tests — cloud metadata endpoints must be blocked even over HTTPS.

    #[test]
    fn build_forward_url_rejects_aws_imds() {
        let err = CoordinationRuntime::build_forward_url(
            "https://169.254.169.254/latest/meta-data/",
            "/v1/remember",
        )
        .expect_err("AWS IMDS must be rejected");
        assert_eq!(err.0, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn build_forward_url_rejects_gcp_metadata() {
        let err = CoordinationRuntime::build_forward_url(
            "https://metadata.google.internal/computeMetadata/v1/",
            "/v1/remember",
        )
        .expect_err("GCP metadata endpoint must be rejected");
        assert_eq!(err.0, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn build_forward_url_rejects_private_ip() {
        let err = CoordinationRuntime::build_forward_url("https://10.0.0.1:443", "/v1/remember")
            .expect_err("RFC1918 private IPs must be rejected");
        assert_eq!(err.0, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn build_forward_url_rejects_cgnat_ip() {
        let err = CoordinationRuntime::build_forward_url("https://100.64.0.1:443", "/v1/remember")
            .expect_err("CGNAT range must be rejected");
        assert_eq!(err.0, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn build_forward_url_accepts_public_https() {
        // Public IP should be accepted for HTTPS forwarding.
        let _ = CoordinationRuntime::build_forward_url("https://192.0.2.1:443", "/v1/remember")
            .expect_err("documentation range 192.0.2/24 should be rejected");
        // A genuinely public IP (not in any reserved range) must be accepted.
        let parsed = CoordinationRuntime::build_forward_url("https://1.2.3.4:443", "/v1/remember")
            .expect("public IP over HTTPS should be accepted");
        assert!(parsed.as_str().starts_with("https://1.2.3.4"));
    }
}
