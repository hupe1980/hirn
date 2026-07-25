use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Top-level server configuration, loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server bind address (default: "127.0.0.1:3000")
    #[serde(default = "default_bind")]
    pub bind: String,

    /// Path to the hirn data directory (databases stored per realm)
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,

    /// Logging configuration
    #[serde(default)]
    pub log: LogConfig,

    /// TLS configuration (optional)
    pub tls: Option<TlsConfig>,

    /// Authentication configuration (optional)
    pub auth: Option<AuthConfig>,

    /// Token-scoped sessions configuration (optional; requires auth)
    pub token: Option<TokenConfig>,

    /// Explicit insecure development posture.
    ///
    /// When `true`, hirnd permits unauthenticated local development flows.
    /// Clients must still provide explicit realm and agent identity on request
    /// surfaces that require them. Production deployments should leave this
    /// disabled.
    #[serde(default)]
    pub insecure_dev_mode: bool,

    /// Metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// gRPC-specific settings
    #[serde(default)]
    pub grpc: GrpcConfig,

    /// Route-class throttling for authenticated and auth-adjacent endpoints.
    #[serde(default)]
    pub throttle: ThrottleConfig,

    /// Watch streaming settings
    #[serde(default)]
    pub watch: WatchConfig,

    /// hirn engine configuration overrides
    #[serde(default)]
    pub engine: EngineConfig,

    /// MCP (Model Context Protocol) Streamable HTTP server settings
    #[serde(default)]
    pub mcp: McpConfig,

    /// Sleep-time consolidation (idle-time cognitive maintenance) settings
    #[serde(default)]
    pub sleep: SleepConfig,

    /// Remote storage backend configuration (S3, GCS, Azure).
    /// When set, realms use the remote object store instead of local filesystem.
    pub storage: Option<StorageBackendConfig>,

    /// Raft cluster configuration (optional; enables multi-node mode)
    #[serde(default)]
    pub raft: Option<RaftConfig>,

    /// DynamoDB coordination for serverless deployments (optional). When set
    /// and Raft is not configured, the sleep-time consolidation pass is gated
    /// on a DynamoDB conditional-write lease instead of the Raft lease.
    #[cfg(feature = "serverless")]
    #[serde(default)]
    pub dynamo: Option<crate::dynamo::DynamoConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    /// Log level: "trace", "debug", "info", "warn", "error"
    #[serde(default = "default_log_level")]
    pub level: String,

    /// Output JSON format logs
    #[serde(default)]
    pub json: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to TLS certificate (PEM)
    pub cert_path: PathBuf,

    /// Path to TLS private key (PEM)
    pub key_path: PathBuf,

    /// Path to CA certificate (PEM) for client certificate authentication (mTLS).
    /// When set, the server requires clients to present a certificate signed by this CA.
    pub client_ca_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// API keys mapped to realm/agent: `{ "key" = { realm = "default", agent_id = "agent_a" } }`
    #[serde(default)]
    pub api_keys: std::collections::HashMap<String, KeyConfig>,

    /// Client certificate CN → realm/agent mapping for mTLS authentication.
    /// When a client presents a valid certificate, the CN is looked up here.
    #[serde(default)]
    pub client_certs: std::collections::HashMap<String, KeyConfig>,
}

/// Configuration for a single API key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyConfig {
    /// Realm this key belongs to (isolation boundary)
    #[serde(default = "default_realm")]
    pub realm: String,
    /// Agent identity associated with this key
    pub agent_id: String,
}

/// Configuration for token-scoped sessions (JWT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenConfig {
    /// HMAC secret used to sign/verify JWTs.
    ///
    /// Supports environment variable expansion: if the value starts with
    /// `$` (e.g. `$HIRN_TOKEN_SECRET`), it is resolved from the environment
    /// at load time. A file reference `file:///run/secrets/hirn_jwt` reads
    /// the secret from a file.
    ///
    /// The secret is stored in a `Zeroizing<String>` so it is wiped from
    /// memory on drop (F-12).
    #[serde(
        deserialize_with = "resolve_secret",
        serialize_with = "serialize_secret_redacted"
    )]
    pub secret: zeroize::Zeroizing<String>,
    /// Token time-to-live in seconds (default: 3600 = 1 hour).
    #[serde(default = "default_token_ttl")]
    pub ttl_secs: u64,
    /// Grace period in seconds during which a rotated-out key remains valid (default: 0).
    #[serde(default)]
    pub rotation_grace_secs: u64,
    /// Clock-skew leeway in seconds added to token expiration checks (default: 30).
    /// Compensates for clock drift between issuer and validator (F-11).
    #[serde(default = "default_clock_skew_leeway")]
    pub clock_skew_leeway_secs: u64,
}

fn default_clock_skew_leeway() -> u64 {
    30
}

fn resolve_secret_value(raw: String) -> Result<String, String> {
    if let Some(var_name) = raw.strip_prefix('$') {
        std::env::var(var_name).map_err(|_| format!("environment variable {var_name} is not set"))
    } else if let Some(path) = raw.strip_prefix("file://") {
        std::fs::read_to_string(path)
            .map(|s| s.trim().to_string())
            .map_err(|e| format!("cannot read secret file {path}: {e}"))
    } else {
        Ok(raw)
    }
}

/// Deserializer that resolves `$ENV_VAR` or `file:///path` references.
fn resolve_secret<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<zeroize::Zeroizing<String>, D::Error> {
    let raw: String = serde::Deserialize::deserialize(deserializer)?;
    let resolved = resolve_secret_value(raw).map_err(serde::de::Error::custom)?;
    Ok(zeroize::Zeroizing::new(resolved))
}

fn resolve_optional_secret<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<zeroize::Zeroizing<String>>, D::Error> {
    let raw: Option<String> = serde::Deserialize::deserialize(deserializer)?;
    raw.map(|value| resolve_secret_value(value).map(zeroize::Zeroizing::new))
        .transpose()
        .map_err(serde::de::Error::custom)
}

fn serialize_secret_redacted<S: serde::Serializer>(
    _secret: &zeroize::Zeroizing<String>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    // Never write the resolved plaintext secret back to the config file.
    // This prevents accidental leaking of secrets that were originally
    // referenced via $ENV_VAR or file:// paths.
    serializer.serialize_str("<REDACTED — set via $ENV_VAR or file://>")
}

fn serialize_optional_secret_redacted<S: serde::Serializer>(
    secret: &Option<zeroize::Zeroizing<String>>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    match secret {
        Some(_) => serializer.serialize_some("<REDACTED — set via $ENV_VAR or file://>"),
        None => serializer.serialize_none(),
    }
}

fn default_realm() -> String {
    "default".to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable Prometheus metrics endpoint
    #[serde(default = "default_true")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcConfig {
    /// Per-request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RateLimitBudget {
    /// Maximum requests allowed within the sliding window.
    pub max_requests: usize,
    /// Sliding-window duration in seconds.
    pub window_secs: u64,
}

impl RateLimitBudget {
    pub fn validate(&self, field: &str) -> Result<(), String> {
        if self.max_requests == 0 {
            return Err(format!("{field}.max_requests must be > 0"));
        }
        if self.window_secs == 0 {
            return Err(format!("{field}.window_secs must be > 0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThrottleConfig {
    /// Whether route-class throttling is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum number of tracked actors before stale entries are evicted.
    #[serde(default = "default_throttle_max_entries")]
    pub max_entries: usize,
    /// Budget for auth-adjacent endpoints such as token issuance and credential checks.
    #[serde(default = "default_auth_rate_limit_budget")]
    pub auth: RateLimitBudget,
    /// Default posture for read-heavy endpoints (`recall`, `think`, `inspect`, `watch`).
    #[serde(default = "default_read_rate_limit_budget")]
    pub read: RateLimitBudget,
    /// Default posture for write endpoints (`remember`, `forget`, `connect`, updates).
    #[serde(default = "default_write_rate_limit_budget")]
    pub write: RateLimitBudget,
    /// Default posture for admin endpoints (`consolidate`, snapshots, namespace admin).
    #[serde(default = "default_admin_rate_limit_budget")]
    pub admin: RateLimitBudget,
}

impl ThrottleConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.max_entries == 0 {
            return Err("throttle.max_entries must be > 0".to_string());
        }
        self.auth.validate("throttle.auth")?;
        self.read.validate("throttle.read")?;
        self.write.validate("throttle.write")?;
        self.admin.validate("throttle.admin")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchConfig {
    /// Broadcast channel buffer capacity (number of events).
    /// When a subscriber falls behind by more than this many events,
    /// the oldest events are dropped.
    #[serde(default = "default_watch_buffer")]
    pub buffer_size: usize,
}

/// MCP (Model Context Protocol) Streamable HTTP server configuration.
///
/// The MCP surface authenticates **per request**: every call must carry an
/// `Authorization: Bearer` credential (API key or JWT) resolved against the
/// same `[auth]` / `[token]` machinery as the HTTP API, and the credential's
/// realm selects the tenant database. There is no listener-level credential.
/// The transport validates the `Host` header natively (loopback-only by
/// default) to block DNS-rebinding attacks (RUSTSEC-2026-0189), and serves
/// TLS when the daemon-wide `[tls]` section is configured.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Enable the MCP Streamable HTTP server (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Bind address for the MCP server.
    /// Defaults to `127.0.0.1` (localhost only) for security.
    /// Set to `0.0.0.0` to expose on all interfaces — this additionally
    /// requires `allowed_hosts` so `Host`-header validation keeps working.
    #[serde(default = "default_mcp_bind")]
    pub bind: String,
    /// `Host` authorities accepted by the transport (DNS-rebinding
    /// protection). Empty (the default) accepts loopback hosts only
    /// (`localhost`, `127.0.0.1`, `::1`). Public deployments must list the
    /// hostnames clients connect to, e.g. `["mcp.example.com"]`; entries
    /// without a port match any port.
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    /// Browser origins accepted by the transport, matched per RFC 6454
    /// `(scheme, host, port)`, e.g. `["https://app.example.com"]`. Empty
    /// (the default) disables `Origin` validation — non-browser MCP clients
    /// don't send one; requests without an `Origin` header always pass.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            bind: default_mcp_bind(),
            allowed_hosts: Vec::new(),
            allowed_origins: Vec::new(),
        }
    }
}

/// Sleep-time consolidation configuration.
///
/// When the daemon has seen no authenticated request for `idle_after_secs`,
/// a background scheduler runs one budgeted maintenance pass per open realm
/// (consolidation pipeline + bounded offline cognition jobs), spaced at
/// least `min_pass_interval_secs` apart. See `docs/deployment.md`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepConfig {
    /// Enable idle-time sleep passes (default: true).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Seconds without any authenticated request before the daemon counts
    /// as idle (default: 300).
    #[serde(default = "default_sleep_idle_after")]
    pub idle_after_secs: u64,
    /// How often (seconds) the scheduler wakes to evaluate idleness
    /// (default: 60).
    #[serde(default = "default_sleep_check_interval")]
    pub check_interval_secs: u64,
    /// Minimum seconds between the end of one sleep pass and the start of
    /// the next (default: 3600).
    #[serde(default = "default_sleep_min_pass_interval")]
    pub min_pass_interval_secs: u64,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            idle_after_secs: default_sleep_idle_after(),
            check_interval_secs: default_sleep_check_interval(),
            min_pass_interval_secs: default_sleep_min_pass_interval(),
        }
    }
}

impl SleepConfig {
    /// Validate sleep scheduling parameters for logical consistency.
    /// A disabled section is accepted as-is, since none of its values are used.
    pub fn validate(&self) -> Result<(), String> {
        if !self.enabled {
            return Ok(());
        }
        if self.check_interval_secs == 0 {
            return Err("sleep.check_interval_secs must be > 0".to_string());
        }
        if self.idle_after_secs == 0 {
            return Err("sleep.idle_after_secs must be > 0".to_string());
        }
        if self.min_pass_interval_secs == 0 {
            return Err("sleep.min_pass_interval_secs must be > 0".to_string());
        }
        if self.idle_after_secs < self.check_interval_secs {
            return Err(format!(
                "sleep.idle_after_secs ({}) must be >= sleep.check_interval_secs ({}) — \
                 otherwise the idle window is finer than the scheduler can observe",
                self.idle_after_secs, self.check_interval_secs
            ));
        }
        Ok(())
    }
}

const fn default_sleep_idle_after() -> u64 {
    300
}

const fn default_sleep_check_interval() -> u64 {
    60
}

const fn default_sleep_min_pass_interval() -> u64 {
    3600
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    pub embedding_dimensions: Option<u32>,
    pub token_budget: Option<u32>,
    pub working_memory_token_limit: Option<u32>,
    pub decay_lambda: Option<f64>,
    pub archive_threshold: Option<f32>,
    pub max_episodic_entries: Option<u32>,
    /// Enable the engine's offline cognition scheduler (dream/reconcile/
    /// planning jobs). Off by default in the engine; the daemon's sleep-time
    /// pass can only enqueue offline jobs when this is set.
    pub offline_scheduler_enabled: Option<bool>,
}

/// Remote storage backend configuration.
///
/// When configured, realm databases use the specified object store URI
/// (e.g. `s3://bucket/hirn-data`) instead of local filesystem storage.
/// All hirnd nodes in the cluster share the same remote storage, enabling
/// horizontal scaling with Lance's MVCC consistency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageBackendConfig {
    /// Object store root URI: `s3://bucket/path`, `gs://bucket/path`,
    /// `az://container/path`, or a local path.
    pub uri: String,
    /// Additional storage properties passed to the object store connector.
    /// Examples: `{ "storage.region" = "us-east-1", "storage.endpoint" = "http://minio:9000" }`
    #[serde(default)]
    pub properties: std::collections::HashMap<String, String>,
}

impl ServerConfig {
    /// Validate the configuration for logical consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.token.is_some() && self.auth.is_none() {
            return Err(
                "token-scoped sessions require auth configuration; configure [auth] or remove [token]"
                    .to_string(),
            );
        }

        if !self.insecure_dev_mode && self.auth.is_none() {
            return Err("auth must be configured unless insecure_dev_mode = true".to_string());
        }

        if !self.insecure_dev_mode
            && self
                .auth
                .as_ref()
                .is_some_and(|auth| auth.api_keys.is_empty() && auth.client_certs.is_empty())
        {
            return Err(
                "auth must define at least one API key or client certificate unless insecure_dev_mode = true"
                    .to_string(),
            );
        }

        // CN→identity mapping (auth.client_certs) is only safe under mandatory
        // mTLS. Without a client CA the `x-client-cert-cn` identity comes from a
        // client-supplied header that any caller can forge, granting the mapped
        // identity with no credential. Require tls.client_ca_path so the CN can
        // only ever be a certificate the server itself verified.
        if self
            .auth
            .as_ref()
            .is_some_and(|auth| !auth.client_certs.is_empty())
            && self
                .tls
                .as_ref()
                .and_then(|tls| tls.client_ca_path.as_ref())
                .is_none()
        {
            return Err(
                "auth.client_certs (CN→identity mapping) requires tls.client_ca_path so client \
                 certificates are verified by mandatory mTLS; otherwise the CN header is forgeable"
                    .to_string(),
            );
        }

        // JWT secret minimum length (OWASP recommendation: ≥ 32 bytes for HMAC-SHA256).
        if let Some(ref token) = self.token {
            if token.secret.len() < 32 {
                return Err(
                    "token.secret must be at least 32 characters for adequate HMAC security"
                        .to_string(),
                );
            }
        }

        self.throttle.validate()?;
        self.sleep.validate()?;

        // MCP authenticates per request against [auth]/[token] (covered by
        // the global auth requirement above). The transport's Host-header
        // allowlist defaults to loopback only, so a non-loopback bind without
        // an explicit allowlist would reject every request — catch the
        // misconfiguration at startup.
        if self.mcp.enabled && self.mcp.allowed_hosts.is_empty() {
            let is_loopback = self
                .mcp
                .bind
                .parse::<std::net::IpAddr>()
                .map(|ip| ip.is_loopback())
                .unwrap_or(false);
            if !is_loopback {
                return Err(format!(
                    "mcp.bind = \"{}\" is not a loopback address; set mcp.allowed_hosts to the \
                     hostnames MCP clients will use (Host-header validation defaults to loopback \
                     only to prevent DNS rebinding), or bind to 127.0.0.1",
                    self.mcp.bind
                ));
            }
        }

        // Raft configuration validation.
        if let Some(ref raft) = self.raft {
            if !self.insecure_dev_mode
                && raft.transport_profile == ClusterTransportProfile::DevLocal
            {
                return Err(
                    "raft.transport_profile = dev-local requires insecure_dev_mode = true; use prod-tls or prod-mtls for production"
                        .to_string(),
                );
            }
            if raft.transport_profile != ClusterTransportProfile::DevLocal
                && raft.transport_secret.is_none()
            {
                return Err(
                    "raft.transport_secret must be configured for prod-tls/prod-mtls transport"
                        .to_string(),
                );
            }
            if !self.insecure_dev_mode && raft.transport_secret.is_none() {
                return Err(
                    "raft.transport_secret must be configured unless insecure_dev_mode = true"
                        .to_string(),
                );
            }
            if raft.transport_profile == ClusterTransportProfile::ProdMtls
                && self
                    .tls
                    .as_ref()
                    .and_then(|tls| tls.client_ca_path.as_ref())
                    .is_none()
            {
                return Err(
                    "raft.transport_profile = prod-mtls requires tls.client_ca_path so raft endpoints require client certificates"
                        .to_string(),
                );
            }
            raft.validate()?;
        }

        // Storage backend validation.
        if let Some(ref storage) = self.storage {
            storage.validate()?;
        }

        Ok(())
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
            data_dir: default_data_dir(),
            log: LogConfig::default(),
            tls: None,
            auth: None,
            token: None,
            insecure_dev_mode: false,
            metrics: MetricsConfig::default(),
            grpc: GrpcConfig::default(),
            throttle: ThrottleConfig::default(),
            watch: WatchConfig::default(),
            engine: EngineConfig::default(),
            mcp: McpConfig::default(),
            sleep: SleepConfig::default(),
            storage: None,
            raft: None,
            #[cfg(feature = "serverless")]
            dynamo: None,
        }
    }
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            json: false,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
        }
    }
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_timeout(),
        }
    }
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self {
            enabled: default_true(),
            max_entries: default_throttle_max_entries(),
            auth: default_auth_rate_limit_budget(),
            read: default_read_rate_limit_budget(),
            write: default_write_rate_limit_budget(),
            admin: default_admin_rate_limit_budget(),
        }
    }
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            buffer_size: default_watch_buffer(),
        }
    }
}

fn default_bind() -> String {
    "127.0.0.1:3000".to_owned()
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("hirn_data")
}

fn default_log_level() -> String {
    "info".to_owned()
}

const fn default_true() -> bool {
    true
}

const fn default_timeout() -> u64 {
    30
}

const fn default_token_ttl() -> u64 {
    3600
}

const fn default_watch_buffer() -> usize {
    1024
}

const fn default_throttle_max_entries() -> usize {
    10_000
}

const fn default_auth_rate_limit_budget() -> RateLimitBudget {
    RateLimitBudget {
        max_requests: 10,
        window_secs: 60,
    }
}

const fn default_read_rate_limit_budget() -> RateLimitBudget {
    RateLimitBudget {
        max_requests: 240,
        window_secs: 60,
    }
}

const fn default_write_rate_limit_budget() -> RateLimitBudget {
    RateLimitBudget {
        max_requests: 60,
        window_secs: 60,
    }
}

const fn default_admin_rate_limit_budget() -> RateLimitBudget {
    RateLimitBudget {
        max_requests: 10,
        window_secs: 60,
    }
}

fn default_mcp_bind() -> String {
    "127.0.0.1".to_owned()
}

/// Raft cluster configuration for multi-node hirnd deployments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftConfig {
    /// Unique node ID in the cluster (must be unique per node).
    pub node_id: u64,
    /// Advertised address for this node (other nodes connect here).
    /// Must be an explicit `http://` or `https://` URL. Production profiles
    /// require HTTPS; `dev-local` permits loopback HTTP only.
    pub advertise_addr: String,
    /// Peer addresses for cluster bootstrap (empty = single-node auto-init).
    /// Format: [{ id = 1, addr = "https://node-1.example:3000" }, ...]
    #[serde(default)]
    pub peers: Vec<RaftPeer>,
    /// Cluster transport posture for Raft RPCs and forwarded owner addresses.
    #[serde(default)]
    pub transport_profile: ClusterTransportProfile,
    /// Heartbeat interval in milliseconds (default: 150).
    #[serde(default = "default_heartbeat")]
    pub heartbeat_interval_ms: u64,
    /// Minimum election timeout in milliseconds (default: 300).
    #[serde(default = "default_election_min")]
    pub election_timeout_min_ms: u64,
    /// Maximum election timeout in milliseconds (default: 500).
    #[serde(default = "default_election_max")]
    pub election_timeout_max_ms: u64,
    /// Shared secret for internal `/raft/*` transport authentication.
    #[serde(
        default,
        deserialize_with = "resolve_optional_secret",
        serialize_with = "serialize_optional_secret_redacted"
    )]
    pub transport_secret: Option<zeroize::Zeroizing<String>>,
    /// Directory for durable Raft log storage.
    ///
    /// When set, `hirnd` opens a `DurableLogStore` at `<data_dir>/raft-log.redb`
    /// so that votes and committed log entries survive process restarts.
    ///
    /// **Required unless `insecure_dev_mode = true`.** Production deployments
    /// without `data_dir` will fail to start. Single-node and multi-node clusters
    /// both need a durable log to maintain Raft safety across restarts.
    #[serde(default)]
    pub data_dir: Option<std::path::PathBuf>,
}

/// A peer node in the Raft cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftPeer {
    /// Node ID.
    pub id: u64,
    /// Network address ("host:port").
    pub addr: String,
}

/// Transport posture for cluster-internal addresses.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub enum ClusterTransportProfile {
    /// Local development only. Allows loopback HTTP and HTTPS.
    #[default]
    DevLocal,
    /// Production TLS. Requires HTTPS addresses.
    ProdTls,
    /// Production mTLS. Requires HTTPS addresses and TLS client CA config.
    ProdMtls,
}

impl ClusterTransportProfile {
    /// Validate a cluster node URL according to this transport profile.
    pub fn validate_endpoint(self, field: &str, value: &str) -> Result<(), String> {
        if value.is_empty() {
            return Err(format!("{field} must not be empty"));
        }

        let url = reqwest::Url::parse(value).map_err(|_| {
            format!("{field} must be an explicit http:// or https:// URL, got '{value}'")
        })?;
        if url.host_str().is_none() {
            return Err(format!("{field} must include a host"));
        }

        match self {
            Self::DevLocal => match url.scheme() {
                "https" => Ok(()),
                "http" if is_loopback_http_endpoint(&url) => Ok(()),
                "http" => Err(format!(
                    "{field} uses remote plaintext HTTP; dev-local permits only loopback HTTP"
                )),
                scheme => Err(format!(
                    "{field} has unsupported URL scheme '{scheme}'; expected http or https"
                )),
            },
            Self::ProdTls | Self::ProdMtls => {
                if url.scheme() == "https" {
                    Ok(())
                } else {
                    Err(format!(
                        "{field} must use https:// when raft.transport_profile = {}",
                        self.as_str()
                    ))
                }
            }
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::DevLocal => "dev-local",
            Self::ProdTls => "prod-tls",
            Self::ProdMtls => "prod-mtls",
        }
    }
}

fn is_loopback_http_endpoint(url: &reqwest::Url) -> bool {
    matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "::1"))
}

const fn default_heartbeat() -> u64 {
    150
}

const fn default_election_min() -> u64 {
    300
}

const fn default_election_max() -> u64 {
    500
}

impl RaftConfig {
    /// Validate Raft configuration for logical consistency.
    pub fn validate(&self) -> Result<(), String> {
        if self.node_id == 0 {
            return Err("raft.node_id must be > 0".to_string());
        }
        if self.advertise_addr.is_empty() {
            return Err("raft.advertise_addr must not be empty".to_string());
        }
        self.transport_profile
            .validate_endpoint("raft.advertise_addr", &self.advertise_addr)?;
        if self.heartbeat_interval_ms == 0 {
            return Err("raft.heartbeat_interval_ms must be > 0".to_string());
        }
        if self.election_timeout_min_ms <= self.heartbeat_interval_ms {
            return Err(format!(
                "raft.election_timeout_min_ms ({}) must be > heartbeat_interval_ms ({})",
                self.election_timeout_min_ms, self.heartbeat_interval_ms
            ));
        }
        if self.election_timeout_max_ms < self.election_timeout_min_ms {
            return Err(format!(
                "raft.election_timeout_max_ms ({}) must be >= election_timeout_min_ms ({})",
                self.election_timeout_max_ms, self.election_timeout_min_ms
            ));
        }
        if self
            .transport_secret
            .as_ref()
            .is_some_and(|secret| secret.len() < 32)
        {
            return Err(
                "raft.transport_secret must be at least 32 characters for adequate transport authentication"
                    .to_string(),
            );
        }
        for peer in &self.peers {
            self.transport_profile
                .validate_endpoint(&format!("raft.peers[id={}].addr", peer.id), &peer.addr)?;
        }
        Ok(())
    }
}

impl StorageBackendConfig {
    /// Validate storage backend configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.uri.is_empty() {
            return Err("storage.uri must not be empty".to_string());
        }
        // Ensure URI has a recognized scheme.
        let valid_prefixes = ["s3://", "gs://", "az://", "/", "./"];
        if !valid_prefixes.iter().any(|p| self.uri.starts_with(p)) {
            return Err(format!(
                "storage.uri '{}' has unrecognized scheme — expected s3://, gs://, az://, or a local path",
                self.uri
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::*;

    fn auth_config() -> AuthConfig {
        AuthConfig {
            api_keys: HashMap::from([(
                "test-key".to_owned(),
                KeyConfig {
                    realm: "default".to_owned(),
                    agent_id: "agent".to_owned(),
                },
            )]),
            client_certs: HashMap::new(),
        }
    }

    fn base_raft_config() -> RaftConfig {
        RaftConfig {
            node_id: 1,
            advertise_addr: "http://127.0.0.1:3000".to_owned(),
            peers: Vec::new(),
            transport_profile: ClusterTransportProfile::DevLocal,
            heartbeat_interval_ms: 150,
            election_timeout_min_ms: 300,
            election_timeout_max_ms: 500,
            transport_secret: None,
            data_dir: None,
        }
    }

    #[test]
    fn raft_dev_local_rejects_remote_plaintext_http() {
        let mut raft = base_raft_config();
        raft.advertise_addr = "http://example.com:3000".to_owned();

        let err = raft.validate().unwrap_err();
        assert!(
            err.contains("remote plaintext HTTP"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn raft_requires_explicit_cluster_url() {
        let mut raft = base_raft_config();
        raft.advertise_addr = "127.0.0.1:3000".to_owned();

        let err = raft.validate().unwrap_err();
        assert!(
            err.contains("explicit http:// or https:// URL"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn production_raft_rejects_dev_local_profile() {
        let mut config = ServerConfig::default();
        config.auth = Some(auth_config());
        let mut raft = base_raft_config();
        raft.transport_secret = Some(zeroize::Zeroizing::new(
            "0123456789abcdef0123456789abcdef".to_owned(),
        ));
        config.raft = Some(raft);

        let err = config.validate().unwrap_err();
        assert!(
            err.contains("prod-tls or prod-mtls"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn production_raft_tls_profile_requires_https() {
        let mut config = ServerConfig::default();
        config.auth = Some(auth_config());
        let mut raft = base_raft_config();
        raft.transport_profile = ClusterTransportProfile::ProdTls;
        raft.transport_secret = Some(zeroize::Zeroizing::new(
            "0123456789abcdef0123456789abcdef".to_owned(),
        ));
        config.raft = Some(raft);

        let err = config.validate().unwrap_err();
        assert!(err.contains("must use https://"), "unexpected error: {err}");
    }

    #[test]
    fn production_raft_mtls_requires_client_ca() {
        let mut config = ServerConfig::default();
        config.auth = Some(auth_config());
        let mut raft = base_raft_config();
        raft.transport_profile = ClusterTransportProfile::ProdMtls;
        raft.advertise_addr = "https://node-1.example:3000".to_owned();
        raft.transport_secret = Some(zeroize::Zeroizing::new(
            "0123456789abcdef0123456789abcdef".to_owned(),
        ));
        config.raft = Some(raft);

        let err = config.validate().unwrap_err();
        assert!(
            err.contains("tls.client_ca_path"),
            "unexpected error: {err}"
        );

        config.tls = Some(TlsConfig {
            cert_path: PathBuf::from("server.crt"),
            key_path: PathBuf::from("server.key"),
            client_ca_path: Some(PathBuf::from("ca.crt")),
        });
        config.validate().unwrap();
    }

    #[test]
    fn sleep_defaults_are_valid() {
        let sleep = SleepConfig::default();
        assert!(sleep.enabled);
        assert_eq!(sleep.idle_after_secs, 300);
        assert_eq!(sleep.check_interval_secs, 60);
        assert_eq!(sleep.min_pass_interval_secs, 3600);
        sleep.validate().unwrap();
    }

    #[test]
    fn sleep_rejects_zero_intervals() {
        for field in ["idle_after", "check_interval", "min_pass_interval"] {
            let mut sleep = SleepConfig::default();
            match field {
                "idle_after" => sleep.idle_after_secs = 0,
                "check_interval" => sleep.check_interval_secs = 0,
                _ => sleep.min_pass_interval_secs = 0,
            }
            let err = sleep.validate().unwrap_err();
            assert!(err.contains("must be > 0"), "unexpected error: {err}");
        }
    }

    #[test]
    fn sleep_idle_window_must_cover_check_interval() {
        let mut sleep = SleepConfig::default();
        sleep.idle_after_secs = 30;
        sleep.check_interval_secs = 60;

        let err = sleep.validate().unwrap_err();
        assert!(
            err.contains("sleep.idle_after_secs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn disabled_sleep_section_skips_interval_validation() {
        let sleep = SleepConfig {
            enabled: false,
            idle_after_secs: 0,
            check_interval_secs: 0,
            min_pass_interval_secs: 0,
        };
        sleep.validate().unwrap();
    }

    #[test]
    fn server_validation_covers_sleep_section() {
        let mut config = ServerConfig::default();
        config.auth = Some(auth_config());
        config.sleep.idle_after_secs = 0;

        let err = config.validate().unwrap_err();
        assert!(
            err.contains("sleep.idle_after_secs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn mcp_non_loopback_bind_requires_allowed_hosts() {
        let mut config = ServerConfig::default();
        config.auth = Some(auth_config());
        config.mcp.bind = "0.0.0.0".to_owned();

        let err = config.validate().unwrap_err();
        assert!(err.contains("mcp.allowed_hosts"), "unexpected error: {err}");

        // An explicit Host allowlist makes the non-loopback bind valid.
        config.mcp.allowed_hosts = vec!["mcp.example.com".to_owned()];
        config.validate().unwrap();

        // Disabling MCP also clears the requirement.
        config.mcp.allowed_hosts.clear();
        config.mcp.enabled = false;
        config.validate().unwrap();
    }

    #[test]
    fn mcp_loopback_bind_needs_no_allowed_hosts() {
        let mut config = ServerConfig::default();
        config.auth = Some(auth_config());
        assert_eq!(config.mcp.bind, "127.0.0.1");
        assert!(config.mcp.allowed_hosts.is_empty());
        config.validate().unwrap();
    }

    #[test]
    fn sleep_section_deserializes_with_partial_toml() {
        let config: ServerConfig = toml::from_str(
            r#"
            insecure_dev_mode = true

            [sleep]
            idle_after_secs = 120
            "#,
        )
        .unwrap();

        assert!(config.sleep.enabled);
        assert_eq!(config.sleep.idle_after_secs, 120);
        assert_eq!(config.sleep.check_interval_secs, 60);
        assert_eq!(config.sleep.min_pass_interval_secs, 3600);
    }
}
