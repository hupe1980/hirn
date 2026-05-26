use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::Response;
use hirn::prelude::{AgentId, Namespace};
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use metrics::counter;
use serde::{Deserialize, Serialize};
use subtle::ConstantTimeEq;

use crate::config::{AuthConfig, TokenConfig};

const INTERNAL_REQUEST_HEADERS: &[&str] = &[
    "x-hirnd-expected-owner-id",
    "x-client-cert-cn",
    "x-token-namespaces",
    "x-token-operations",
];

/// Hash an API key to a fixed 32-byte digest for constant-time comparison.
///
/// Using blake3 normalizes all key lengths so that `ct_eq` on the digests
/// does not leak the expected key length via response timing (N-H05).
fn hash_api_key(key: &str) -> [u8; 32] {
    *blake3::hash(key.as_bytes()).as_bytes()
}

/// Resolved identity from an API key: realm + agent_id.
#[derive(Debug, Clone)]
pub struct KeyIdentity {
    pub realm: String,
    pub agent_id: String,
}

/// Operations a token is allowed to perform.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operation {
    Read,
    Write,
    Admin,
}

/// JWT claims carried in a token-scoped session token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClaims {
    /// Realm this token is scoped to.
    pub realm: String,
    /// Agent identity.
    pub agent_id: String,
    /// Namespace allowlist. Empty = private + shared only.
    #[serde(default)]
    pub namespaces: Vec<String>,
    /// Allowed operations. Empty = all.
    #[serde(default)]
    pub operations: Vec<Operation>,
    /// Issued-at (seconds since epoch).
    pub iat: u64,
    /// Expiry (seconds since epoch).
    pub exp: u64,
}

/// Resolved identity from either an API key or JWT token.
#[derive(Debug, Clone)]
pub struct ResolvedIdentity {
    pub realm: String,
    pub agent_id: String,
    /// Namespace restrictions from token (empty = unrestricted / default).
    pub namespaces: Vec<String>,
    /// Operation restrictions from token (empty = unrestricted).
    pub operations: Vec<Operation>,
}

pub(crate) fn token_allows_operation(
    allowed_operations: &[Operation],
    required: &Operation,
) -> bool {
    allowed_operations.is_empty() || allowed_operations.contains(required)
}

pub(crate) fn token_allows_namespace(
    agent_id: &AgentId,
    allowed_namespaces: &[String],
    namespace: Option<&str>,
) -> bool {
    let private_namespace = Namespace::private_for(agent_id);
    let shared_namespace = Namespace::shared();
    let default_namespace = Namespace::default_ns();
    let private_name = private_namespace.as_str();
    let shared_name = shared_namespace.as_str();
    let default_name = default_namespace.as_str();

    let effective_namespace = match namespace {
        Some(ns) if ns != default_name => ns,
        _ => private_name,
    };

    if allowed_namespaces.is_empty() {
        return effective_namespace == private_name || effective_namespace == shared_name;
    }

    allowed_namespaces
        .iter()
        .any(|allowed| match allowed.as_str() {
            "private" | "default" => effective_namespace == private_name,
            "shared" => effective_namespace == shared_name,
            other => effective_namespace == other,
        })
}

/// Shared authentication state.
pub struct AuthState {
    /// Maps blake3(API key) digest → (realm, agent_id).
    /// Pre-hashing at construction time means `validate()` performs O(n)
    /// fixed-length digest comparisons with no length side-channel (N-H05).
    keys: Option<HashMap<[u8; 32], KeyIdentity>>,
    /// Maps client certificate CN → (realm, agent_id) for mTLS authentication.
    client_certs: HashMap<String, KeyIdentity>,
    /// Token signing/verification config.
    token_config: Option<TokenConfig>,
    /// Whether explicit insecure development mode permits unauthenticated requests.
    allow_unauthenticated: bool,
}

impl AuthState {
    pub fn new(auth_config: Option<&AuthConfig>, token_config: Option<&TokenConfig>) -> Self {
        Self::with_posture(auth_config, token_config, false)
    }

    pub fn insecure_dev_mode(
        auth_config: Option<&AuthConfig>,
        token_config: Option<&TokenConfig>,
    ) -> Self {
        Self::with_posture(auth_config, token_config, true)
    }

    fn with_posture(
        auth_config: Option<&AuthConfig>,
        token_config: Option<&TokenConfig>,
        allow_unauthenticated: bool,
    ) -> Self {
        let client_certs = auth_config
            .map(|c| {
                c.client_certs
                    .iter()
                    .map(|(cn, kc)| {
                        (
                            cn.clone(),
                            KeyIdentity {
                                realm: kc.realm.clone(),
                                agent_id: kc.agent_id.clone(),
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();

        Self {
            keys: auth_config.map(|c| {
                c.api_keys
                    .iter()
                    .map(|(key, kc)| {
                        (
                            hash_api_key(key),
                            KeyIdentity {
                                realm: kc.realm.clone(),
                                agent_id: kc.agent_id.clone(),
                            },
                        )
                    })
                    .collect()
            }),
            client_certs,
            token_config: token_config.cloned(),
            allow_unauthenticated,
        }
    }

    /// Validate an API key using constant-time comparison to prevent timing
    /// side-channel attacks.
    ///
    /// **N-H05 fix:** both the candidate and stored keys are hashed to a
    /// fixed 32-byte blake3 digest before comparison, so `ct_eq` always
    /// compares equal-length values and the response time does not reveal
    /// expected key length. All stored-key digests are iterated regardless
    /// of match to avoid early-exit timing differences.
    pub fn validate(&self, key: &str) -> Option<&KeyIdentity> {
        self.keys.as_ref().and_then(|keys| {
            let candidate_hash = hash_api_key(key);
            let mut found: Option<&KeyIdentity> = None;
            for (stored_hash, identity) in keys {
                // ct_eq on [u8; 32]: always 32 bytes, constant time.
                if stored_hash.ct_eq(&candidate_hash).into() {
                    found = Some(identity);
                }
            }
            found
        })
    }

    /// Validate a client certificate CN. Returns the associated realm + agent_id if mapped.
    pub fn validate_client_cert(&self, cn: &str) -> Option<&KeyIdentity> {
        self.client_certs.get(cn)
    }

    /// Whether auth is enabled.
    pub fn is_enabled(&self) -> bool {
        self.keys.is_some()
    }

    /// Whether explicit insecure development posture allows unauthenticated requests.
    pub fn allows_unauthenticated(&self) -> bool {
        self.allow_unauthenticated
    }

    /// Whether token issuance is enabled.
    pub fn tokens_enabled(&self) -> bool {
        self.token_config.is_some()
    }

    /// Issue a JWT token for the given identity with optional namespace/operation scoping.
    pub fn issue_token(
        &self,
        identity: &KeyIdentity,
        namespaces: Vec<String>,
        operations: Vec<Operation>,
        ttl_override: Option<u64>,
    ) -> Result<String, String> {
        let tc = self
            .token_config
            .as_ref()
            .ok_or("token issuance not configured")?;

        let now = jsonwebtoken::get_current_timestamp();
        let ttl = ttl_override.unwrap_or(tc.ttl_secs);

        let claims = TokenClaims {
            realm: identity.realm.clone(),
            agent_id: identity.agent_id.clone(),
            namespaces,
            operations,
            iat: now,
            exp: now + ttl,
        };

        encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(tc.secret.as_bytes()),
        )
        .map_err(|e| format!("failed to encode token: {e}"))
    }

    /// Validate a JWT token. Returns the decoded claims if valid.
    pub fn validate_token(&self, token: &str) -> Result<TokenClaims, TokenError> {
        let tc = self
            .token_config
            .as_ref()
            .ok_or(TokenError::NotConfigured)?;

        let mut validation = Validation::default();
        validation.set_required_spec_claims(&["exp", "iat"]);
        // N-M01 fix: leeway covers only clock skew between client and server.
        // rotation_grace_secs is NOT applied as universal leeway because that
        // would silently accept tokens expired by up to rotation_grace_secs
        // from ANY key, widening the acceptance window dangerously.
        // API key rotation is managed at the key-store level (add new key,
        // issue new tokens, remove old key after drain); JWT leeway is for
        // clock skew only.
        validation.leeway = tc.clock_skew_leeway_secs;

        let data = decode::<TokenClaims>(
            token,
            &DecodingKey::from_secret(tc.secret.as_bytes()),
            &validation,
        )
        .map_err(|e| match e.kind() {
            jsonwebtoken::errors::ErrorKind::ExpiredSignature => TokenError::Expired,
            _ => TokenError::Invalid(e.to_string()),
        })?;

        Ok(data.claims)
    }
}

#[derive(Debug)]
pub enum TokenError {
    Expired,
    Invalid(String),
    NotConfigured,
}

/// Axum middleware layer for API key and JWT authentication.
pub async fn auth_middleware(
    state: axum::extract::State<Arc<AuthState>>,
    mut request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let client_cn = request
        .headers()
        .get("x-client-cert-cn")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_owned());
    for header in INTERNAL_REQUEST_HEADERS {
        if request.headers_mut().remove(*header).is_some() {
            counter!(
                "hirnd_internal_metadata_strips_total",
                "interface" => "http",
                "header" => *header,
            )
            .increment(1);
        }
    }

    if !state.is_enabled() {
        if !state.allows_unauthenticated() {
            tracing::warn!(
                "HTTP auth rejected: auth is not configured and insecure_dev_mode is disabled"
            );
            return Err(StatusCode::UNAUTHORIZED);
        }
        return Ok(next.run(request).await);
    }

    // ── mTLS: check client certificate CN (injected by serve_http_tls) ──
    // F-17: Internal forwarding headers were already stripped above.

    if let Some(cn) = client_cn.as_deref() {
        if let Some(ki) = state.validate_client_cert(cn) {
            let identity = ResolvedIdentity {
                realm: ki.realm.clone(),
                agent_id: ki.agent_id.clone(),
                namespaces: vec![],
                operations: vec![],
            };

            // Inject realm and agent_id as headers for downstream handlers
            request.headers_mut().insert(
                "x-realm-id",
                identity
                    .realm
                    .parse()
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            );
            request.headers_mut().insert(
                "x-agent-id",
                identity
                    .agent_id
                    .parse()
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            );

            return Ok(next.run(request).await);
        }
        // CN not in mapping — fall through to Bearer token auth
    }

    // ── Bearer token auth (JWT or API key) ──

    // Extract Bearer token from Authorization header
    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    let bearer = match auth_header {
        Some(h) if h.starts_with("Bearer ") => &h[7..],
        _ => {
            tracing::warn!("HTTP auth failed: missing or invalid authorization header");
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    // Try JWT token first (if tokens are configured), then fall back to API key
    let identity = if state.tokens_enabled() {
        match state.validate_token(bearer) {
            Ok(claims) => {
                // Store token restrictions as daemon-authored headers for downstream enforcement.
                let ns_json = serde_json::to_string(&claims.namespaces)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                request.headers_mut().insert(
                    "x-token-namespaces",
                    ns_json
                        .parse()
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                );
                let ops_json = serde_json::to_string(&claims.operations)
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
                request.headers_mut().insert(
                    "x-token-operations",
                    ops_json
                        .parse()
                        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                );
                ResolvedIdentity {
                    realm: claims.realm,
                    agent_id: claims.agent_id,
                    namespaces: claims.namespaces,
                    operations: claims.operations,
                }
            }
            Err(TokenError::Expired) => {
                tracing::warn!("HTTP auth failed: token expired");
                return Err(StatusCode::UNAUTHORIZED);
            }
            Err(TokenError::NotConfigured) => {
                // Shouldn't happen since we checked tokens_enabled
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
            Err(TokenError::Invalid(_)) => {
                // Not a valid JWT — try as API key
                match state.validate(bearer) {
                    Some(ki) => ResolvedIdentity {
                        realm: ki.realm.clone(),
                        agent_id: ki.agent_id.clone(),
                        namespaces: vec![],
                        operations: vec![],
                    },
                    None => {
                        tracing::warn!("HTTP auth failed: invalid JWT and invalid API key");
                        return Err(StatusCode::UNAUTHORIZED);
                    }
                }
            }
        }
    } else {
        // No token config — API key only
        match state.validate(bearer) {
            Some(ki) => ResolvedIdentity {
                realm: ki.realm.clone(),
                agent_id: ki.agent_id.clone(),
                namespaces: vec![],
                operations: vec![],
            },
            None => {
                tracing::warn!("HTTP auth failed: invalid API key");
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
    };

    // Inject realm and agent_id as headers for downstream handlers
    request.headers_mut().insert(
        "x-realm-id",
        identity
            .realm
            .parse()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    );
    request.headers_mut().insert(
        "x-agent-id",
        identity
            .agent_id
            .parse()
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
    );

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use axum::middleware;
    use axum::routing::get;
    use tower::ServiceExt;

    async fn ok() -> &'static str {
        "ok"
    }

    #[tokio::test]
    async fn auth_disabled_without_insecure_dev_mode_returns_unauthorized() {
        let router = Router::new()
            .route("/ok", get(ok))
            .layer(middleware::from_fn_with_state(
                Arc::new(AuthState::new(None, None)),
                auth_middleware,
            ));

        let response = router
            .oneshot(Request::builder().uri("/ok").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
