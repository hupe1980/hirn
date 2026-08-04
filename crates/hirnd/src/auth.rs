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
    "x-hirnd-issuer-kid",
];

/// Daemon-authored header carrying the fingerprint of the credential that
/// authenticated the request (see [`credential_kid`]). Stripped from all
/// inbound requests and re-injected by [`auth_middleware`], so downstream
/// handlers (token issuance) can bind minted JWTs to their issuing credential.
pub const ISSUER_KID_HEADER: &str = "x-hirnd-issuer-kid";

/// Hash an API key to a fixed 32-byte digest for constant-time comparison.
///
/// Using blake3 normalizes all key lengths so that `ct_eq` on the digests
/// does not leak the expected key length via response timing (N-H05).
fn hash_api_key(key: &str) -> [u8; 32] {
    *blake3::hash(key.as_bytes()).as_bytes()
}

/// Compute a short, stable fingerprint ("kid") for an authenticating
/// credential, used as the `iss_kid` claim in minted JWTs.
///
/// `kind` domain-separates the credential class (`"key"` for API keys,
/// `"cn"` for mTLS client-certificate CNs) so an API key that happens to
/// equal a certificate CN cannot produce a colliding kid. The output is the
/// first 16 bytes (32 hex chars) of a domain-separated blake3 hash — a
/// one-way fingerprint, so the kid can be logged and shipped in tokens
/// without revealing the credential.
pub fn credential_kid(kind: &str, credential: &str) -> String {
    let mut hasher = blake3::Hasher::new_derive_key("hirnd credential kid v1");
    hasher.update(kind.as_bytes());
    hasher.update(&[0]);
    hasher.update(credential.as_bytes());
    hasher.finalize().to_hex()[..32].to_string()
}

/// Resolved identity from an API key: realm + agent_id.
#[derive(Debug, Clone)]
pub struct KeyIdentity {
    pub realm: String,
    pub agent_id: String,
}

/// Operations a token is allowed to perform.
///
/// Operations form a privilege hierarchy: `Admin ⊇ Write ⊇ Read`. A token
/// granted a higher operation implicitly permits the lower ones (an `[Admin]`
/// token may Read and Write); see [`Operation::rank`] and
/// [`token_allows_operation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Operation {
    Read,
    Write,
    Admin,
}

impl Operation {
    /// Privilege rank used for hierarchical implication: `Read < Write < Admin`.
    /// A granted operation permits every required operation of equal-or-lower
    /// rank.
    #[must_use]
    pub const fn rank(self) -> u8 {
        match self {
            Self::Read => 0,
            Self::Write => 1,
            Self::Admin => 2,
        }
    }

    /// Whether holding `self` implies permission to perform `required`
    /// (i.e. `self` is at least as privileged).
    #[must_use]
    pub const fn implies(self, required: Operation) -> bool {
        self.rank() >= required.rank()
    }
}

/// Issuer claim embedded in every hirnd-minted JWT.
pub const TOKEN_ISSUER: &str = "hirnd";

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
    /// Issuer — always [`TOKEN_ISSUER`]; rejected otherwise.
    pub iss: String,
    /// Audience — the realm the token is scoped to. Bound to `realm` at
    /// issuance and re-checked at validation so a token cannot be replayed
    /// against a different audience.
    pub aud: String,
    /// Issued-at (seconds since epoch).
    pub iat: u64,
    /// Expiry (seconds since epoch).
    pub exp: u64,
    /// Unique token id (ULID), minted at issuance. Enables per-token
    /// revocation before `exp`. Required — tokens without a `jti` are
    /// rejected at validation.
    pub jti: String,
    /// Fingerprint of the credential that (transitively) issued this token —
    /// see [`credential_kid`]. Tokens minted by a restricted token inherit
    /// the parent's kid, so revoking the root credential (e.g. a rotated-out
    /// API key) invalidates the whole issuance tree. `None` only for tokens
    /// minted through paths where no issuing credential is known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub iss_kid: Option<String>,
}

/// How long (seconds) past a token's `exp` its revocation entry is retained,
/// covering the validator's clock-skew leeway before the entry is pruned.
const REVOCATION_PRUNE_LEEWAY_SECS: u64 = 300;

/// In-process revocation list for hirnd-minted JWTs.
///
/// Two revocation axes:
/// - **per-token** (`jti`): entries carry the token's `exp` and are pruned
///   once the token would have expired anyway, so the list is naturally
///   bounded by the number of live revoked tokens.
/// - **per-issuer** (`iss_kid`): revoking an issuing credential's kid
///   rejects every token whose `iss_kid` matches and whose `iat` is not
///   strictly newer than the revocation, so rotating an API key kills all
///   outstanding tokens it issued while tokens minted after a later re-add
///   of the credential remain valid.
///
/// **Scope: node-local.** Entries live only in this process. In a Raft
/// cluster each node must be told about revocations separately (or tokens
/// simply age out at `exp`); propagating revocations through the Raft log
/// is intentionally out of scope for this layer.
#[derive(Default)]
pub struct RevocationList {
    inner: parking_lot::RwLock<RevocationInner>,
}

#[derive(Default)]
struct RevocationInner {
    /// (realm, jti) → token `exp` (seconds since epoch).
    jtis: HashMap<(String, String), u64>,
    /// (realm, issuer kid) → revocation timestamp (seconds since epoch).
    issuers: HashMap<(String, String), u64>,
}

impl RevocationList {
    fn now() -> u64 {
        jsonwebtoken::get_current_timestamp()
    }

    /// Revoke a single token by its `jti`. `exp` is the token's expiry;
    /// the entry is pruned once the token would have expired on its own.
    pub fn revoke_jti(&self, realm: impl Into<String>, jti: impl Into<String>, exp: u64) {
        let now = Self::now();
        let mut inner = self.inner.write();
        // Amortized TTL-bound: drop entries for tokens that are already
        // expired (plus leeway) on every insert.
        inner
            .jtis
            .retain(|_, entry_exp| entry_exp.saturating_add(REVOCATION_PRUNE_LEEWAY_SECS) > now);
        inner.jtis.insert((realm.into(), jti.into()), exp);
    }

    /// Revoke every outstanding token issued by the credential with this kid.
    pub fn revoke_issuer(&self, realm: impl Into<String>, kid: impl Into<String>) {
        self.inner
            .write()
            .issuers
            .insert((realm.into(), kid.into()), Self::now());
    }

    /// Whether this `jti` has been revoked (and would not have expired anyway).
    pub fn is_jti_revoked(&self, realm: &str, jti: &str) -> bool {
        self.inner
            .read()
            .jtis
            .contains_key(&(realm.to_owned(), jti.to_owned()))
    }

    /// Whether tokens issued by `kid` at `iat` are revoked. Tokens minted
    /// strictly after the revocation timestamp are accepted again (the
    /// credential was re-added / re-trusted).
    pub fn is_issuer_revoked(&self, realm: &str, kid: &str, iat: u64) -> bool {
        self.inner
            .read()
            .issuers
            .get(&(realm.to_owned(), kid.to_owned()))
            .is_some_and(|revoked_at| *revoked_at >= iat)
    }

    /// Number of live per-token revocation entries (test observability).
    #[cfg(test)]
    fn jti_entries(&self) -> usize {
        self.inner.read().jtis.len()
    }
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

/// Whether a token's operation allowlist permits `required`, applying the
/// `Admin ⊇ Write ⊇ Read` hierarchy: an empty allowlist means "all
/// operations", and any granted operation of equal-or-higher rank than
/// `required` satisfies it (so `[Admin]` permits Read and Write).
pub(crate) fn token_allows_operation(
    allowed_operations: &[Operation],
    required: &Operation,
) -> bool {
    allowed_operations.is_empty()
        || allowed_operations
            .iter()
            .any(|allowed| allowed.implies(*required))
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

/// Resolve a token namespace claim into concrete engine namespaces.
///
/// An empty claim retains the token default of private + shared. Aliases are
/// normalized before the scope is intersected with engine authorization by
/// `HirnDB::as_agent_with_namespaces`.
pub(crate) fn token_namespace_scope(
    agent_id: &AgentId,
    allowed_namespaces: &[String],
) -> Result<Vec<Namespace>, String> {
    let private = Namespace::private_for(agent_id);
    let shared = Namespace::shared();
    if allowed_namespaces.is_empty() {
        return Ok(vec![private, shared]);
    }

    let mut resolved = Vec::with_capacity(allowed_namespaces.len());
    for allowed in allowed_namespaces {
        let namespace = match allowed.as_str() {
            "private" | "default" => private,
            "shared" => shared,
            other => Namespace::new(other)
                .map_err(|error| format!("invalid token namespace '{other}': {error}"))?,
        };
        if !resolved.contains(&namespace) {
            resolved.push(namespace);
        }
    }
    Ok(resolved)
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
    /// Node-local JWT revocation list (per-jti and per-issuer-kid).
    revocations: RevocationList,
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
            revocations: RevocationList::default(),
        }
    }

    /// The node-local JWT revocation list.
    pub fn revocations(&self) -> &RevocationList {
        &self.revocations
    }

    /// Revoke every outstanding JWT issued by the given API key.
    ///
    /// Call this when an API key is rotated out or removed so already-minted
    /// tokens die with the key instead of remaining valid until `exp`.
    pub fn revoke_api_key(&self, realm: &str, key: &str) {
        self.revocations
            .revoke_issuer(realm, credential_kid("key", key));
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

    /// Configured default token TTL in seconds (0 when tokens are disabled).
    pub fn token_ttl_secs(&self) -> u64 {
        self.token_config.as_ref().map_or(0, |tc| tc.ttl_secs)
    }

    /// Issue a JWT token for the given identity with optional namespace/operation scoping.
    ///
    /// `iss_kid` is the fingerprint of the credential that authenticated the
    /// issuance request (see [`credential_kid`]); it is embedded in the token
    /// so revoking that credential also revokes this token.
    pub fn issue_token(
        &self,
        identity: &KeyIdentity,
        namespaces: Vec<String>,
        operations: Vec<Operation>,
        ttl_override: Option<u64>,
        iss_kid: Option<String>,
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
            iss: TOKEN_ISSUER.to_owned(),
            aud: identity.realm.clone(),
            iat: now,
            exp: now + ttl,
            jti: ulid::Ulid::new().to_string(),
            iss_kid,
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
        validation.set_required_spec_claims(&["exp", "iat", "iss", "aud"]);
        validation.set_issuer(&[TOKEN_ISSUER]);
        // The audience is the token's own realm, which is only known after
        // decoding — disable the library's audience-list check and compare
        // aud against the realm claim below instead.
        validation.validate_aud = false;
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

        if data.claims.aud != data.claims.realm {
            return Err(TokenError::Invalid(
                "token audience does not match its realm".to_owned(),
            ));
        }

        // Revocation checks: the token itself, then its issuing credential.
        if self
            .revocations
            .is_jti_revoked(&data.claims.realm, &data.claims.jti)
        {
            return Err(TokenError::Revoked);
        }
        if let Some(kid) = data.claims.iss_kid.as_deref() {
            if self
                .revocations
                .is_issuer_revoked(&data.claims.realm, kid, data.claims.iat)
            {
                return Err(TokenError::Revoked);
            }
        }

        Ok(data.claims)
    }

    /// Decode a token for revocation purposes: the signature, issuer, and
    /// claim shape are verified, but expiry is **not** — an operator must be
    /// able to revoke a token (e.g. to persist an issuer-kid revocation)
    /// even when it has just expired. Never use this for authentication.
    pub fn decode_for_revocation(&self, token: &str) -> Result<TokenClaims, TokenError> {
        let tc = self
            .token_config
            .as_ref()
            .ok_or(TokenError::NotConfigured)?;

        let mut validation = Validation::default();
        validation.set_required_spec_claims(&["iss"]);
        validation.set_issuer(&[TOKEN_ISSUER]);
        validation.validate_exp = false;
        validation.validate_aud = false;

        decode::<TokenClaims>(
            token,
            &DecodingKey::from_secret(tc.secret.as_bytes()),
            &validation,
        )
        .map(|data| data.claims)
        .map_err(|e| TokenError::Invalid(e.to_string()))
    }

    /// Resolve a bearer credential to an identity using the same acceptance
    /// rules as the HTTP middleware: a JWT is tried first when token sessions
    /// are configured, then the credential is treated as an API key.
    pub fn resolve_bearer(&self, bearer: &str) -> Result<BearerIdentity, String> {
        if self.tokens_enabled() {
            match self.validate_token(bearer) {
                Ok(claims) => {
                    return Ok(BearerIdentity {
                        realm: claims.realm,
                        agent_id: claims.agent_id,
                        namespaces: Some(claims.namespaces),
                        operations: claims.operations,
                    });
                }
                Err(TokenError::Expired) => return Err("token expired".to_owned()),
                Err(TokenError::Revoked) => return Err("token revoked".to_owned()),
                // Not a valid JWT — fall through to API key lookup.
                Err(TokenError::Invalid(_) | TokenError::NotConfigured) => {}
            }
        }

        self.validate(bearer)
            .map(|ki| BearerIdentity {
                realm: ki.realm.clone(),
                agent_id: ki.agent_id.clone(),
                namespaces: None,
                operations: vec![],
            })
            .ok_or_else(|| "credential is not a valid API key or token".to_owned())
    }
}

/// Identity resolved from a bearer credential (JWT or API key).
///
/// Unlike [`ResolvedIdentity`], namespace scoping keeps the distinction
/// between "no restriction" (API key → `None`) and "token with an explicit —
/// possibly empty — allowlist" (`Some(list)`), matching how the HTTP layer
/// only enforces namespace scope for token-authenticated requests.
#[derive(Debug, Clone)]
pub struct BearerIdentity {
    pub realm: String,
    pub agent_id: String,
    /// `None` = unrestricted (API key); `Some(list)` = token allowlist where
    /// an empty list means private + shared namespaces only.
    pub namespaces: Option<Vec<String>>,
    /// Operation restrictions (empty = all operations).
    pub operations: Vec<Operation>,
}

#[derive(Debug)]
pub enum TokenError {
    Expired,
    /// The token (or its issuing credential) has been revoked before expiry.
    Revoked,
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

            // Bind tokens minted by this request to the client certificate.
            request.headers_mut().insert(
                ISSUER_KID_HEADER,
                credential_kid("cn", cn)
                    .parse()
                    .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
            );

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
                // Tokens minted by this token inherit the root credential's
                // kid, so revoking the root kills the whole issuance tree.
                if let Some(ref kid) = claims.iss_kid {
                    request.headers_mut().insert(
                        ISSUER_KID_HEADER,
                        kid.parse().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                    );
                }
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
            Err(TokenError::Revoked) => {
                tracing::warn!("HTTP auth failed: token revoked");
                return Err(StatusCode::UNAUTHORIZED);
            }
            Err(TokenError::NotConfigured) => {
                // Shouldn't happen since we checked tokens_enabled
                return Err(StatusCode::INTERNAL_SERVER_ERROR);
            }
            Err(TokenError::Invalid(_)) => {
                // Not a valid JWT — try as API key
                match state.validate(bearer) {
                    Some(ki) => {
                        // Compute before headers_mut(): `bearer` borrows the request.
                        let kid = credential_kid("key", bearer);
                        let identity = ResolvedIdentity {
                            realm: ki.realm.clone(),
                            agent_id: ki.agent_id.clone(),
                            namespaces: vec![],
                            operations: vec![],
                        };
                        request.headers_mut().insert(
                            ISSUER_KID_HEADER,
                            kid.parse().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                        );
                        identity
                    }
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
            Some(ki) => {
                // Compute before headers_mut(): `bearer` borrows the request.
                let kid = credential_kid("key", bearer);
                let identity = ResolvedIdentity {
                    realm: ki.realm.clone(),
                    agent_id: ki.agent_id.clone(),
                    namespaces: vec![],
                    operations: vec![],
                };
                request.headers_mut().insert(
                    ISSUER_KID_HEADER,
                    kid.parse().map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?,
                );
                identity
            }
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

    fn token_config() -> TokenConfig {
        TokenConfig {
            secret: zeroize::Zeroizing::new("0123456789abcdef0123456789abcdef".to_owned()),
            ttl_secs: 3600,
            rotation_grace_secs: 0,
            clock_skew_leeway_secs: 30,
        }
    }

    fn auth_state_with_tokens() -> AuthState {
        AuthState::new(None, Some(&token_config()))
    }

    fn identity() -> KeyIdentity {
        KeyIdentity {
            realm: "default".to_owned(),
            agent_id: "agent-a".to_owned(),
        }
    }

    #[test]
    fn token_namespace_scope_normalizes_without_widening() {
        let agent = AgentId::new("agent-a").unwrap();
        let defaults = token_namespace_scope(&agent, &[]).unwrap();
        assert_eq!(
            defaults,
            vec![Namespace::private_for(&agent), Namespace::shared()]
        );

        let claimed = vec![
            "private".to_owned(),
            "default".to_owned(),
            "shared".to_owned(),
            "team-a".to_owned(),
        ];
        let resolved = token_namespace_scope(&agent, &claimed).unwrap();
        assert_eq!(
            resolved,
            vec![
                Namespace::private_for(&agent),
                Namespace::shared(),
                Namespace::new("team-a").unwrap(),
            ]
        );
        assert!(token_namespace_scope(&agent, &["bad namespace".to_owned()]).is_err());
    }

    #[test]
    fn issued_tokens_carry_issuer_and_realm_audience() {
        let state = auth_state_with_tokens();
        let token = state
            .issue_token(&identity(), vec![], vec![Operation::Read], None, None)
            .unwrap();

        let claims = state.validate_token(&token).unwrap();
        assert_eq!(claims.iss, TOKEN_ISSUER);
        assert_eq!(claims.aud, "default");
        assert_eq!(claims.aud, claims.realm);
        assert_eq!(claims.operations, vec![Operation::Read]);
        assert!(!claims.jti.is_empty(), "every token must carry a jti");
    }

    #[test]
    fn issued_tokens_have_unique_jtis() {
        let state = auth_state_with_tokens();
        let a = state
            .issue_token(&identity(), vec![], vec![], None, None)
            .unwrap();
        let b = state
            .issue_token(&identity(), vec![], vec![], None, None)
            .unwrap();
        let jti_a = state.validate_token(&a).unwrap().jti;
        let jti_b = state.validate_token(&b).unwrap().jti;
        assert_ne!(jti_a, jti_b);
    }

    #[test]
    fn revoked_jti_is_rejected_and_others_unaffected() {
        let state = auth_state_with_tokens();
        let revoked = state
            .issue_token(&identity(), vec![], vec![], None, None)
            .unwrap();
        let untouched = state
            .issue_token(&identity(), vec![], vec![], None, None)
            .unwrap();

        // Both valid (unexpired) before revocation.
        let claims = state.validate_token(&revoked).unwrap();
        state
            .revocations()
            .revoke_jti(&claims.realm, claims.jti, claims.exp);

        // Unexpired but revoked → rejected.
        assert!(matches!(
            state.validate_token(&revoked),
            Err(TokenError::Revoked)
        ));
        // Unrevoked token keeps working.
        state.validate_token(&untouched).unwrap();
    }

    #[test]
    fn expired_revocation_entries_are_pruned() {
        let list = RevocationList::default();
        let now = jsonwebtoken::get_current_timestamp();

        // Entry for a token that expired long ago (past the prune leeway).
        list.revoke_jti("realm-a", "stale-jti", now - 10_000);
        assert_eq!(list.jti_entries(), 1);

        // The next insert prunes the stale entry.
        list.revoke_jti("realm-a", "live-jti", now + 3600);
        assert_eq!(list.jti_entries(), 1);
        assert!(list.is_jti_revoked("realm-a", "live-jti"));
        assert!(!list.is_jti_revoked("realm-a", "stale-jti"));
    }

    #[test]
    fn revocations_are_isolated_by_realm() {
        let list = RevocationList::default();
        let exp = jsonwebtoken::get_current_timestamp() + 3600;
        list.revoke_jti("realm-a", "shared-jti", exp);
        list.revoke_issuer("realm-a", "shared-kid");

        assert!(list.is_jti_revoked("realm-a", "shared-jti"));
        assert!(!list.is_jti_revoked("realm-b", "shared-jti"));
        assert!(list.is_issuer_revoked("realm-a", "shared-kid", 0));
        assert!(!list.is_issuer_revoked("realm-b", "shared-kid", 0));
    }

    #[test]
    fn issuer_kid_revocation_kills_all_its_tokens() {
        let state = auth_state_with_tokens();
        let kid = credential_kid("key", "the-api-key");
        let other_kid = credential_kid("key", "another-api-key");

        let t1 = state
            .issue_token(&identity(), vec![], vec![], None, Some(kid.clone()))
            .unwrap();
        let t2 = state
            .issue_token(&identity(), vec![], vec![], None, Some(kid.clone()))
            .unwrap();
        let t3 = state
            .issue_token(&identity(), vec![], vec![], None, Some(other_kid))
            .unwrap();

        state.validate_token(&t1).unwrap();
        state.validate_token(&t2).unwrap();

        state.revoke_api_key("default", "the-api-key");

        assert!(matches!(
            state.validate_token(&t1),
            Err(TokenError::Revoked)
        ));
        assert!(matches!(
            state.validate_token(&t2),
            Err(TokenError::Revoked)
        ));
        // Tokens from a different issuing credential are unaffected.
        state.validate_token(&t3).unwrap();
    }

    #[test]
    fn tokens_minted_after_issuer_revocation_are_accepted() {
        let state = auth_state_with_tokens();
        let kid = credential_kid("key", "rotating-key");
        state.revocations().revoke_issuer("default", kid.clone());

        // A token whose iat is strictly after the revocation timestamp
        // (credential re-added later) validates again.
        let now = jsonwebtoken::get_current_timestamp();
        let token = encode_raw_claims(&serde_json::json!({
            "realm": "default",
            "agent_id": "agent-a",
            "iss": TOKEN_ISSUER,
            "aud": "default",
            "iat": now + 60,
            "exp": now + 3600,
            "jti": "post-revocation-token",
            "iss_kid": kid,
        }));
        state.validate_token(&token).unwrap();
    }

    #[test]
    fn tokens_without_jti_are_rejected() {
        let state = auth_state_with_tokens();
        let now = jsonwebtoken::get_current_timestamp();
        // Correctly signed, correct iss/aud, but no jti (legacy shape).
        let token = encode_raw_claims(&serde_json::json!({
            "realm": "default",
            "agent_id": "agent-a",
            "iss": TOKEN_ISSUER,
            "aud": "default",
            "iat": now,
            "exp": now + 3600,
        }));

        assert!(matches!(
            state.validate_token(&token),
            Err(TokenError::Invalid(_))
        ));
    }

    #[test]
    fn decode_for_revocation_accepts_expired_tokens() {
        let state = auth_state_with_tokens();
        let now = jsonwebtoken::get_current_timestamp();
        let token = encode_raw_claims(&serde_json::json!({
            "realm": "default",
            "agent_id": "agent-a",
            "iss": TOKEN_ISSUER,
            "aud": "default",
            "iat": now - 7200,
            "exp": now - 3600,
            "jti": "expired-jti",
        }));

        // validate_token refuses it, but the revocation path can still read it.
        assert!(matches!(
            state.validate_token(&token),
            Err(TokenError::Expired)
        ));
        let claims = state.decode_for_revocation(&token).unwrap();
        assert_eq!(claims.jti, "expired-jti");
    }

    #[test]
    fn credential_kid_is_stable_and_domain_separated() {
        assert_eq!(credential_kid("key", "abc"), credential_kid("key", "abc"));
        assert_ne!(credential_kid("key", "abc"), credential_kid("cn", "abc"));
        assert_ne!(credential_kid("key", "abc"), credential_kid("key", "abd"));
        assert_eq!(credential_kid("key", "abc").len(), 32);
    }

    fn encode_raw_claims(claims: &serde_json::Value) -> String {
        encode(
            &Header::default(),
            claims,
            &EncodingKey::from_secret(token_config().secret.as_bytes()),
        )
        .unwrap()
    }

    #[test]
    fn tokens_without_issuer_or_audience_are_rejected() {
        let state = auth_state_with_tokens();
        let now = jsonwebtoken::get_current_timestamp();
        // Legacy claim shape: correctly signed, but missing iss/aud.
        let token = encode_raw_claims(&serde_json::json!({
            "realm": "default",
            "agent_id": "agent-a",
            "iat": now,
            "exp": now + 3600,
        }));

        assert!(matches!(
            state.validate_token(&token),
            Err(TokenError::Invalid(_))
        ));
    }

    #[test]
    fn tokens_with_wrong_issuer_are_rejected() {
        let state = auth_state_with_tokens();
        let now = jsonwebtoken::get_current_timestamp();
        let token = encode_raw_claims(&serde_json::json!({
            "realm": "default",
            "agent_id": "agent-a",
            "iss": "someone-else",
            "aud": "default",
            "iat": now,
            "exp": now + 3600,
        }));

        assert!(matches!(
            state.validate_token(&token),
            Err(TokenError::Invalid(_))
        ));
    }

    #[test]
    fn tokens_with_mismatched_audience_are_rejected() {
        let state = auth_state_with_tokens();
        let now = jsonwebtoken::get_current_timestamp();
        let token = encode_raw_claims(&serde_json::json!({
            "realm": "default",
            "agent_id": "agent-a",
            "iss": TOKEN_ISSUER,
            "aud": "other-realm",
            "iat": now,
            "exp": now + 3600,
        }));

        assert!(matches!(
            state.validate_token(&token),
            Err(TokenError::Invalid(_))
        ));
    }

    #[test]
    fn admin_token_implies_write_and_read() {
        // R-72: Admin ⊇ Write ⊇ Read. An [Admin] token must satisfy Read/Write.
        let admin = [Operation::Admin];
        assert!(token_allows_operation(&admin, &Operation::Read));
        assert!(token_allows_operation(&admin, &Operation::Write));
        assert!(token_allows_operation(&admin, &Operation::Admin));

        let write = [Operation::Write];
        assert!(token_allows_operation(&write, &Operation::Read));
        assert!(token_allows_operation(&write, &Operation::Write));
        assert!(!token_allows_operation(&write, &Operation::Admin));

        let read = [Operation::Read];
        assert!(token_allows_operation(&read, &Operation::Read));
        assert!(!token_allows_operation(&read, &Operation::Write));
        assert!(!token_allows_operation(&read, &Operation::Admin));
    }

    #[test]
    fn empty_operations_allow_everything() {
        assert!(token_allows_operation(&[], &Operation::Read));
        assert!(token_allows_operation(&[], &Operation::Write));
        assert!(token_allows_operation(&[], &Operation::Admin));
    }

    #[test]
    fn operation_implication_is_transitive() {
        assert!(Operation::Admin.implies(Operation::Read));
        assert!(Operation::Admin.implies(Operation::Write));
        assert!(Operation::Write.implies(Operation::Read));
        assert!(!Operation::Read.implies(Operation::Write));
        assert!(!Operation::Write.implies(Operation::Admin));
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
