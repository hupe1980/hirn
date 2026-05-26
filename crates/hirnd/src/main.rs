use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand};
use hirnd::auth::AuthState;
use hirnd::config::ServerConfig;
use hirnd::grpc::HirnGrpcService;
use hirnd::http::HttpState;
use hirnd::mcp::HirnMcpService;
use hirnd::proto::hirn_service_server::HirnServiceServer;
use hirnd::watch::WatchEvent;
use rmcp::transport::sse_server::SseServer;
use tokio::net::TcpListener;
use tokio::signal;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

/// Default embedding dimensions. Used as fallback when no explicit value is
/// configured (matches common models like `text-embedding-3-small` at 768-d).
const DEFAULT_EMBEDDING_DIMS: usize = hirnd::DEFAULT_EMBEDDING_DIMS;

#[derive(Parser)]
#[command(
    name = "hirnd",
    about = "hirn standalone server — cognitive memory daemon"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Path to the TOML configuration file
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Path to the data directory (overrides config)
    #[arg(short, long)]
    data: Option<PathBuf>,

    /// Bind address (overrides config)
    #[arg(short, long)]
    bind: Option<String>,

    /// Enable explicit insecure development mode.
    #[arg(long)]
    insecure_dev_mode: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a self-signed TLS certificate and key
    GenerateCert {
        /// Output path for the certificate PEM file
        #[arg(long, default_value = "cert.pem")]
        cert: PathBuf,
        /// Output path for the private key PEM file
        #[arg(long, default_value = "key.pem")]
        key: PathBuf,
    },
    /// Add an API key to the configuration file
    AddKey {
        /// Path to the TOML configuration file
        #[arg(long)]
        config: PathBuf,
        /// Realm the key belongs to
        #[arg(long)]
        realm: String,
        /// Agent identity for this key
        #[arg(long)]
        agent: String,
        // NOTE: The key value is intentionally NOT accepted as a CLI argument to prevent
        // it from appearing in process listings and shell history (N-H06).
        // Pass the key via the HIRND_API_KEY environment variable, or omit it for a
        // randomly-generated key.
    },
    /// Rotate an API key: replace old key with a new one
    RotateKey {
        /// Path to the TOML configuration file
        #[arg(long)]
        config: PathBuf,
        // NOTE: Key values are intentionally NOT accepted as CLI arguments to prevent
        // them from appearing in process listings and shell history (N-H06).
        // Pass keys via HIRND_OLD_KEY and HIRND_NEW_KEY environment variables.
        // HIRND_NEW_KEY is optional — a random key is generated if unset.
    },
    /// Check database integrity
    Check {
        /// Path to the database file
        #[arg(long)]
        data: PathBuf,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// Repair database issues
    Repair {
        /// Path to the database file
        #[arg(long)]
        data: PathBuf,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// Create a named snapshot (tags all datasets at their current version)
    Snapshot {
        /// Path to the database directory (realm root)
        #[arg(long)]
        data: PathBuf,
        /// Snapshot name
        #[arg(long)]
        name: String,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// List all available snapshots
    ListSnapshots {
        /// Path to the database directory (realm root)
        #[arg(long)]
        data: PathBuf,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// Roll back all datasets to a named snapshot
    Rollback {
        /// Path to the database directory (realm root)
        #[arg(long)]
        data: PathBuf,
        /// Snapshot name to roll back to
        #[arg(long)]
        name: String,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// Validate a TOML configuration file
    ValidateConfig {
        /// Path to the TOML configuration file
        config: PathBuf,
    },
    /// Show database information and statistics
    Info {
        /// Path to the database directory (realm root)
        #[arg(long)]
        data: PathBuf,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// Optimize all datasets (compact, prune old versions, re-index)
    Optimize {
        /// Path to the database directory (realm root)
        #[arg(long)]
        data: PathBuf,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// Export database contents to JSON
    Export {
        /// Path to the database directory (realm root)
        #[arg(long)]
        data: PathBuf,
        /// Output file path
        #[arg(long)]
        output: PathBuf,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
    /// Import database contents from JSON
    Import {
        /// Path to the JSON input file
        #[arg(long)]
        input: PathBuf,
        /// Target database directory (realm root)
        #[arg(long)]
        data: PathBuf,
        /// Embedding dimensions (default: 768)
        #[arg(long, default_value_t = DEFAULT_EMBEDDING_DIMS)]
        embedding_dimensions: usize,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Handle subcommands
    if let Some(cmd) = cli.command {
        match cmd {
            Commands::GenerateCert { cert, key } => {
                hirnd::tls::generate_self_signed_cert(&cert, &key)?;
                println!("Certificate written to {}", cert.display());
                println!("Private key written to {}", key.display());
            }
            Commands::AddKey {
                config,
                realm,
                agent,
            } => {
                // Read key from env var to avoid process-listing exposure (N-H06).
                let key_value = std::env::var("HIRND_API_KEY")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(generate_api_key);
                let mut server_config = load_config(&config)?;
                let auth = server_config
                    .auth
                    .get_or_insert_with(|| hirnd::config::AuthConfig {
                        api_keys: std::collections::HashMap::new(),
                        client_certs: std::collections::HashMap::new(),
                    });
                auth.api_keys.insert(
                    key_value.clone(),
                    hirnd::config::KeyConfig {
                        realm: realm.clone(),
                        agent_id: agent.clone(),
                    },
                );
                write_config(&config, &server_config)?;
                println!("Added key for realm '{realm}', agent '{agent}'");
                println!("Key: {key_value}");
            }
            Commands::RotateKey { config } => {
                // Read keys from env vars to avoid process-listing exposure (N-H06).
                let old_key = std::env::var("HIRND_OLD_KEY").map_err(
                    |_| "HIRND_OLD_KEY environment variable is required for key rotation",
                )?;
                let new_key_value = std::env::var("HIRND_NEW_KEY")
                    .ok()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(generate_api_key);
                let mut server_config = load_config(&config)?;
                let auth = server_config
                    .auth
                    .as_mut()
                    .ok_or("no [auth] section in config")?;
                let identity = auth
                    .api_keys
                    .remove(&old_key)
                    .ok_or_else(|| "old key not found in config".to_string())?;
                let realm = identity.realm.clone();
                let agent = identity.agent_id.clone();
                auth.api_keys.insert(new_key_value.clone(), identity);
                write_config(&config, &server_config)?;
                println!("Rotated key for realm '{realm}', agent '{agent}'");
                println!("Old key removed, new key: {new_key_value}");
            }
            Commands::Check {
                data,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let report = hirn_engine::integrity::check_integrity(storage.as_ref())
                    .await
                    .map_err(|e| format!("integrity check failed: {e}"))?;
                if report.is_clean {
                    println!("OK — database is clean");
                } else {
                    println!("Issues found:");
                    for issue in &report.issues {
                        println!("  [{:?}] {}", issue.kind, issue.description);
                    }
                    std::process::exit(1);
                }
            }
            Commands::Repair {
                data,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let report = hirn_engine::integrity::repair(storage.as_ref())
                    .await
                    .map_err(|e| format!("repair failed: {e}"))?;
                if report.repaired.is_empty() && report.failed.is_empty() {
                    println!("No repairs needed — database is clean");
                } else {
                    for msg in &report.repaired {
                        println!("  Repaired: {msg}");
                    }
                    for msg in &report.failed {
                        eprintln!("  FAILED: {msg}");
                    }
                    if !report.failed.is_empty() {
                        std::process::exit(1);
                    }
                }
            }
            Commands::Snapshot {
                data,
                name,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let report = hirn_engine::backup::create_snapshot(storage.as_ref(), &name)
                    .await
                    .map_err(|e| format!("snapshot failed: {e}"))?;
                println!(
                    "Snapshot '{}' created: {} datasets tagged",
                    name, report.datasets_tagged
                );
            }
            Commands::ListSnapshots {
                data,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let snapshots = hirn_engine::backup::list_snapshots(storage.as_ref())
                    .await
                    .map_err(|e| format!("list snapshots failed: {e}"))?;
                if snapshots.is_empty() {
                    println!("No snapshots found");
                } else {
                    for snap in &snapshots {
                        println!("  {} ({} datasets)", snap.name, snap.versions.len());
                        for (ds, ver) in &snap.versions {
                            println!("    {ds}: v{ver}");
                        }
                    }
                }
            }
            Commands::Rollback {
                data,
                name,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let report = hirn_engine::backup::rollback(storage.as_ref(), &name)
                    .await
                    .map_err(|e| format!("rollback failed: {e}"))?;
                println!(
                    "Rolled back to '{}': {} datasets restored",
                    name, report.datasets_rolled_back
                );
            }
            Commands::ValidateConfig { config } => {
                let content = std::fs::read_to_string(&config).map_err(|e| {
                    format!("failed to read config file '{}': {e}", config.display())
                })?;
                let server_config: ServerConfig = toml::from_str(&content)
                    .map_err(|e| format!("invalid config file '{}': {e}", config.display()))?;
                // Also validate the engine config by building a HirnConfig
                let mut hirn_config =
                    hirn_core::HirnConfig::builder().db_path("/tmp/validate-config-dummy");
                if let Some(dims) = server_config.engine.embedding_dimensions {
                    hirn_config = hirn_config.embedding_dimensions(dims);
                }
                if let Some(budget) = server_config.engine.token_budget {
                    hirn_config = hirn_config.token_budget(budget);
                }
                if let Some(limit) = server_config.engine.working_memory_token_limit {
                    hirn_config = hirn_config.working_memory_token_limit(limit);
                }
                if let Some(lambda) = server_config.engine.decay_lambda {
                    hirn_config = hirn_config.decay_lambda(lambda);
                }
                if let Some(thresh) = server_config.engine.archive_threshold {
                    hirn_config = hirn_config.archive_threshold(thresh);
                }
                if let Some(max) = server_config.engine.max_episodic_entries {
                    hirn_config = hirn_config.max_episodic_entries(max);
                }
                hirn_config
                    .build()
                    .map_err(|e| format!("engine config validation failed: {e}"))?
                    .validate()
                    .map_err(|e| format!("engine config validation failed: {e}"))?;
                println!("Configuration is valid");
            }
            Commands::Info {
                data,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let db =
                    hirn_engine::HirnDB::open(&data.join("default").join("lance_brain"), storage)
                        .await
                        .map_err(|e| format!("failed to open database: {e}"))?;
                let stats = db
                    .admin()
                    .stats()
                    .await
                    .map_err(|e| format!("failed to get stats: {e}"))?;
                println!("Database Information");
                println!("  Path:             {}", data.display());
                println!("  File size:        {} bytes", stats.file_size_bytes);
                println!("  Working memory:   {} entries", stats.working_count);
                println!("  Episodic memory:  {} records", stats.episodic_count);
                println!("  Semantic memory:  {} records", stats.semantic_count);
                println!("  Total records:    {}", stats.total_count);
                println!("  Graph edges:      {}", stats.edge_count);
            }
            Commands::Optimize {
                data,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let datasets = storage
                    .list_datasets()
                    .await
                    .map_err(|e| format!("failed to list datasets: {e}"))?;
                for ds in &datasets {
                    storage
                        .compact(&ds.name, Default::default())
                        .await
                        .map_err(|e| format!("failed to optimize dataset '{}': {e}", ds.name))?;
                    println!("  Optimized: {}", ds.name);
                }
                println!("Optimization complete ({} datasets)", datasets.len());
            }
            Commands::Export {
                data,
                output,
                embedding_dimensions: _,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let mut file = std::fs::File::create(&output).map_err(|e| {
                    format!("failed to create output file '{}': {e}", output.display())
                })?;
                let report = hirn_engine::export::export(storage.as_ref(), &mut file)
                    .await
                    .map_err(|e| format!("export failed: {e}"))?;
                println!(
                    "Export complete: {} episodic, {} semantic, {} working, {} agents, {} namespaces ({} bytes)",
                    report.episodic_count,
                    report.semantic_count,
                    report.working_count,
                    report.agent_count,
                    report.namespace_count,
                    report.bytes_written,
                );
                println!("Written to {}", output.display());
            }
            Commands::Import {
                input,
                data,
                embedding_dimensions,
            } => {
                let storage = open_storage_for_path(&data).await?;
                let mut file = std::fs::File::open(&input)
                    .map_err(|e| format!("failed to open input file '{}': {e}", input.display()))?;
                let report =
                    hirn_engine::export::import(&mut file, storage.as_ref(), embedding_dimensions)
                        .await
                        .map_err(|e| format!("import failed: {e}"))?;
                println!(
                    "Import complete: {} episodic, {} semantic, {} working, {} agents, {} namespaces",
                    report.episodic_count,
                    report.semantic_count,
                    report.working_count,
                    report.agent_count,
                    report.namespace_count,
                );
                println!("Imported to {}", data.display());
            }
        }
        return Ok(());
    }

    // Load configuration
    let mut config = if let Some(ref config_path) = cli.config {
        let content = std::fs::read_to_string(config_path).map_err(|e| {
            format!(
                "failed to read config file '{}': {e}",
                config_path.display()
            )
        })?;
        toml::from_str::<ServerConfig>(&content)
            .map_err(|e| format!("invalid config file '{}': {e}", config_path.display()))?
    } else {
        ServerConfig::default()
    };

    // CLI overrides
    if let Some(data) = cli.data {
        config.data_dir = data;
    }
    if let Some(bind) = cli.bind {
        config.bind = bind;
    }
    if cli.insecure_dev_mode {
        config.insecure_dev_mode = true;
    }

    // Validate configuration
    config
        .validate()
        .map_err(|e| format!("config validation failed: {e}"))?;

    // Initialize logging
    init_logging(&config);

    // Create the realm manager (one HirnDB per realm)
    let realm_manager = if let Some(ref storage) = config.storage {
        Arc::new(hirnd::realm::RealmManager::with_storage_backend(
            config.data_dir.clone(),
            config.engine.clone(),
            storage.clone(),
        ))
    } else {
        Arc::new(hirnd::realm::RealmManager::new(
            config.data_dir.clone(),
            config.engine.clone(),
        ))
    };

    // Pre-open the default realm
    realm_manager
        .get("default")
        .await
        .map_err(|e| format!("failed to open default realm database: {e}"))?;

    info!(data_dir = %config.data_dir.display(), "realm manager initialized");

    let start_time = Instant::now();
    let (watch_tx, _) = broadcast::channel::<WatchEvent>(config.watch.buffer_size);

    // Parse bind address
    let addr: std::net::SocketAddr = config
        .bind
        .parse()
        .map_err(|e| format!("invalid bind address '{}': {e}", config.bind))?;

    // ── Metrics ──
    let metrics_handle = if config.metrics.enabled {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        metrics::set_global_recorder(recorder)
            .map_err(|e| format!("failed to install Prometheus recorder: {e}"))?;
        info!("Prometheus metrics enabled");
        Some(handle)
    } else {
        None
    };

    // Load TLS if configured
    let tls_acceptor = if let Some(ref tls_config) = config.tls {
        let acceptor = hirnd::tls::load_tls(tls_config)
            .map_err(|e| format!("failed to load TLS config: {e}"))?;
        info!("TLS enabled");
        Some(acceptor)
    } else {
        None
    };

    // ── HTTP server ──
    let auth_state = Arc::new(if config.insecure_dev_mode {
        AuthState::insecure_dev_mode(config.auth.as_ref(), config.token.as_ref())
    } else {
        AuthState::new(config.auth.as_ref(), config.token.as_ref())
    });
    let ready = Arc::new(AtomicBool::new(false));

    // ── Raft consensus (optional) ──
    let (raft_node, raft_sm) = if let Some(ref raft_config) = config.raft {
        info!(
            node_id = raft_config.node_id,
            advertise_addr = %raft_config.advertise_addr,
            peers = raft_config.peers.len(),
            "initializing Raft consensus"
        );
        let mut openraft_config = hirnd::raft::default_raft_config();
        openraft_config.heartbeat_interval = raft_config.heartbeat_interval_ms;
        openraft_config.election_timeout_min = raft_config.election_timeout_min_ms;
        openraft_config.election_timeout_max = raft_config.election_timeout_max_ms;
        let openraft_config = Arc::new(
            openraft_config
                .validate()
                .map_err(|e| format!("invalid raft config: {e}"))?,
        );

        // R-6: refuse to start without a durable log unless insecure_dev_mode
        // is explicitly set.  A missing data_dir in production silently loses
        // Raft votes and committed log entries across restarts, which can elect
        // two leaders for the same term — a Raft safety violation.
        if raft_config.data_dir.is_none() && !config.insecure_dev_mode {
            return Err("raft.data_dir must be set for production deployments. \
                Without a durable log store, a process restart will lose votes and \
                committed log entries (Raft safety violation). \
                Set `raft.data_dir` to a writable directory, or set \
                `insecure_dev_mode = true` to explicitly opt into the volatile \
                in-memory log (development/testing only)."
                .into());
        }

        let log_store_result: Result<hirnd::raft::DurableLogStore, String> =
            if let Some(data_dir) = &raft_config.data_dir {
                let log_path = data_dir.join("raft-log.redb");
                info!(path = %log_path.display(), "opening durable Raft log store");
                hirnd::raft::DurableLogStore::open(&log_path)
            } else {
                // insecure_dev_mode = true, no data_dir: volatile in-memory store.
                warn!(
                    "insecure_dev_mode: using non-durable in-memory Raft log store. \
                    Votes and committed log entries are lost on restart. \
                    Do not use in production."
                );
                Err("insecure_dev_mode: no data_dir".to_string())
            };

        let raft = match log_store_result {
            Ok(log_store) => {
                let state_machine = Arc::new(hirnd::raft::HirnStateMachine::new());
                let network = hirnd::raft::network::HirnRaftNetworkFactory::new(
                    raft_config
                        .transport_secret
                        .as_ref()
                        .map(|secret| secret.as_str()),
                )?;
                let r = hirnd::raft::new_raft(
                    raft_config.node_id,
                    Arc::clone(&openraft_config),
                    log_store,
                    Arc::clone(&state_machine),
                    network,
                )
                .await
                .map_err(|e| format!("failed to create Raft node: {e}"))?;
                (r, state_machine)
            }
            Err(_) => {
                // insecure_dev_mode + no data_dir: fall back to volatile in-memory store.
                let dev_log_store = hirnd::raft::DevMemLogStore::new();
                let state_machine = Arc::new(hirnd::raft::HirnStateMachine::new());
                let network = hirnd::raft::network::HirnRaftNetworkFactory::new(
                    raft_config
                        .transport_secret
                        .as_ref()
                        .map(|secret| secret.as_str()),
                )?;
                let r = hirnd::raft::new_raft_dev(
                    raft_config.node_id,
                    Arc::clone(&openraft_config),
                    dev_log_store,
                    Arc::clone(&state_machine),
                    network,
                )
                .await
                .map_err(|e| format!("failed to create Raft node (dev): {e}"))?;
                (r, state_machine)
            }
        };
        let (raft, state_machine) = raft;

        // Auto-init single-node cluster when no peers are configured.
        if raft_config.peers.is_empty() {
            info!("no peers configured — auto-initializing single-node cluster");
            let mut members = std::collections::BTreeMap::new();
            members.insert(
                raft_config.node_id,
                openraft::BasicNode {
                    addr: raft_config.advertise_addr.clone(),
                },
            );
            if let Err(e) = raft.initialize(members).await {
                // InitializeError::NotAllowed means already initialized — that's fine.
                info!("raft init result (may already be initialized): {e}");
            }
        }

        (Some(raft), Some(state_machine))
    } else {
        (None, None)
    };

    let shared_rate_limiter = Arc::new(hirnd::throttle::RateLimiter::from_config(&config.throttle));
    let raft_transport_secret = config.raft.as_ref().and_then(|raft| {
        raft.transport_secret
            .as_ref()
            .map(|secret| Arc::<str>::from(secret.as_str()))
    });
    let allow_insecure_raft_transport = config.insecure_dev_mode
        && config.raft.as_ref().is_some_and(|raft| {
            raft.transport_profile == hirnd::config::ClusterTransportProfile::DevLocal
        });

    let http_state = Arc::new(HttpState {
        realms: Arc::clone(&realm_manager),
        auth_state: Arc::clone(&auth_state),
        start_time,
        watch_tx: watch_tx.clone(),
        metrics_enabled: config.metrics.enabled,
        metrics_handle,
        rate_limiter: Arc::clone(&shared_rate_limiter),
        ready: Arc::clone(&ready),
        raft: raft_node,
        raft_state_machine: raft_sm,
        raft_transport_secret,
        allow_insecure_raft_transport,
        forward_client: hirnd::http::default_forward_client()?,
        idempotency_cache: Arc::new(hirnd::http::IdempotencyCache::default()),
    });

    let http_router = hirnd::http::router(http_state, Arc::clone(&auth_state));
    let http_listener = TcpListener::bind(addr).await?;
    info!(bind = %addr, tls = tls_acceptor.is_some(), "HTTP server listening");

    // ── gRPC server (on port + 1) ──
    // F-16 FIX: Validate port arithmetic to prevent u16 overflow.
    let grpc_port = addr.port().checked_add(1).ok_or_else(|| {
        format!(
            "gRPC port overflow: base port {} + 1 exceeds u16::MAX",
            addr.port()
        )
    })?;
    let grpc_addr = std::net::SocketAddr::new(addr.ip(), grpc_port);
    let grpc_service = HirnGrpcService::new(
        Arc::clone(&realm_manager),
        watch_tx.clone(),
        Arc::clone(&shared_rate_limiter),
    );
    let grpc_interceptor = hirnd::grpc::grpc_auth_interceptor(auth_state);

    let grpc_timeout = Duration::from_secs(config.grpc.timeout_secs);
    let grpc_server = if let Some(ref tls_config) = config.tls {
        let cert_pem = std::fs::read(&tls_config.cert_path)?;
        let key_pem = std::fs::read(&tls_config.key_path)?;
        let identity = tonic::transport::Identity::from_pem(cert_pem, key_pem);
        let tls = tonic::transport::ServerTlsConfig::new().identity(identity);
        tonic::transport::Server::builder()
            .timeout(grpc_timeout)
            .tls_config(tls)
            .map_err(|e| format!("gRPC TLS config error: {e}"))?
            .add_service(HirnServiceServer::with_interceptor(
                grpc_service,
                grpc_interceptor,
            ))
            .serve(grpc_addr)
    } else {
        tonic::transport::Server::builder()
            .timeout(grpc_timeout)
            .add_service(HirnServiceServer::with_interceptor(
                grpc_service,
                grpc_interceptor,
            ))
            .serve(grpc_addr)
    };

    info!(bind = %grpc_addr, "gRPC server listening");

    // ── MCP SSE server (on port + 2) ──
    // F-05 FIX: MCP binds to configured address (default: 127.0.0.1 / localhost only).
    // For production with network exposure, use a TLS-terminating reverse proxy with auth.
    let mcp_ct = if config.mcp.enabled {
        let mcp_ip: std::net::IpAddr = config
            .mcp
            .bind
            .parse()
            .map_err(|e| format!("invalid mcp.bind address '{}': {e}", config.mcp.bind))?;
        let mcp_port = addr.port().checked_add(2).ok_or_else(|| {
            format!(
                "MCP port overflow: base port {} + 2 exceeds u16::MAX",
                addr.port()
            )
        })?;
        let mcp_addr = std::net::SocketAddr::new(mcp_ip, mcp_port);
        let mcp_db = realm_manager
            .get("default")
            .await
            .map_err(|e| format!("default realm must be open for MCP server: {e}"))?;
        let mcp_watch_tx = watch_tx.clone();
        let mcp_server = SseServer::serve(mcp_addr).await?;
        let ct = mcp_server.with_service(move || {
            HirnMcpService::new(
                Arc::clone(&mcp_db),
                mcp_watch_tx.clone(),
                "default".to_string(),
            )
        });
        // F-12 FIX: Warn about plaintext SSE transport and missing auth.
        tracing::warn!(
            bind = %mcp_addr,
            "MCP SSE transport is plaintext (no TLS); place behind a TLS-terminating \
             reverse proxy (e.g. nginx, Caddy, Envoy) for production deployments"
        );
        if !mcp_ip.is_loopback() {
            tracing::warn!(
                bind = %mcp_addr,
                "MCP SSE server is exposed on a non-loopback interface without built-in auth; \
                 use a reverse proxy with authentication in production"
            );
        }
        info!(bind = %mcp_addr, "MCP SSE server listening");
        Some(ct)
    } else {
        info!("MCP SSE server disabled");
        None
    };

    // ── HTTP server with optional TLS ──
    let http_tls = tls_acceptor.clone();
    let http_future = async move {
        if let Some(acceptor) = http_tls {
            hirnd::http::serve_http_tls(http_listener, http_router, acceptor).await
        } else {
            axum::serve(http_listener, http_router)
                .with_graceful_shutdown(shutdown_signal())
                .await
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
        }
    };

    // ── Startup complete — mark as ready ──
    ready.store(true, Ordering::Release);
    info!("server ready");

    // Run HTTP, gRPC, and MCP concurrently
    tokio::select! {
        result = http_future => {
            if let Err(e) = result {
                error!(error = %e, "HTTP server error");
            }
        }
        result = grpc_server => {
            if let Err(e) = result {
                error!(error = %e, "gRPC server error");
            }
        }
    }
    if let Some(ct) = mcp_ct {
        ct.cancel();
    }

    info!("server shutdown complete");
    Ok(())
}

/// Open a LanceDB storage backend for a data directory (used by CLI check/repair).
async fn open_storage_for_path(
    data: &std::path::Path,
) -> Result<Arc<dyn hirn_storage::PhysicalStore>, String> {
    let lance_path = data.join("default").join("lance_brain");
    let storage_cfg = hirn_storage::HirnDbConfig::local(lance_path.to_string_lossy());
    let hirn_storage = hirn_storage::HirnDb::open(storage_cfg)
        .await
        .map_err(|e| format!("failed to open storage at {}: {e}", lance_path.display()))?;
    Ok(hirn_storage.store_arc())
}

fn init_logging(config: &ServerConfig) {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&config.log.level));

    // If OTEL_EXPORTER_OTLP_ENDPOINT is set, add an OpenTelemetry layer.
    let otel_layer = if std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").is_ok() {
        match init_otel_tracer() {
            Ok(layer) => Some(layer),
            Err(e) => {
                eprintln!("WARNING: Failed to initialize OpenTelemetry: {e}");
                None
            }
        }
    } else {
        None
    };

    // OTel layer must be added first (directly to Registry) since it's typed as Layer<Registry>.
    let registry = tracing_subscriber::registry()
        .with(otel_layer)
        .with(env_filter);

    if config.log.json {
        registry
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        registry.with(tracing_subscriber::fmt::layer()).init();
    }
}

/// Initialize an OpenTelemetry OTLP tracing layer.
///
/// Reads `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://localhost:4317`)
/// and `OTEL_SERVICE_NAME` (default: `hirnd`).
fn init_otel_tracer() -> Result<
    tracing_opentelemetry::OpenTelemetryLayer<
        tracing_subscriber::Registry,
        opentelemetry_sdk::trace::SdkTracer,
    >,
    Box<dyn std::error::Error>,
> {
    use opentelemetry::trace::TracerProvider as _;
    use opentelemetry_sdk::trace::SdkTracerProvider;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .build()?;

    let provider = SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .build();

    let tracer =
        provider.tracer(std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "hirnd".to_string()));

    // Keep provider alive — register as global.
    opentelemetry::global::set_tracer_provider(provider);

    Ok(tracing_opentelemetry::layer().with_tracer(tracer))
}

async fn shutdown_signal() {
    // F-15 FIX: Use proper error handling instead of expect() in signal handlers.
    let ctrl_c = async {
        if let Err(e) = signal::ctrl_c().await {
            tracing::error!("failed to install Ctrl+C handler: {e}");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match signal::unix::signal(signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                sig.recv().await;
            }
            Err(e) => {
                tracing::error!("failed to install SIGTERM handler: {e}");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    info!("shutdown signal received, gracefully shutting down...");
}

/// Load a `ServerConfig` from a TOML file.
fn load_config(path: &PathBuf) -> Result<ServerConfig, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read config '{}': {e}", path.display()))?;
    let config: ServerConfig = toml::from_str(&content)
        .map_err(|e| format!("invalid config '{}': {e}", path.display()))?;
    Ok(config)
}

/// Write a `ServerConfig` back to a TOML file.
fn write_config(path: &PathBuf, config: &ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    let content =
        toml::to_string_pretty(config).map_err(|e| format!("failed to serialize config: {e}"))?;
    std::fs::write(path, content)
        .map_err(|e| format!("failed to write config '{}': {e}", path.display()))?;
    Ok(())
}

/// Generate a random API key (32 hex characters) using cryptographically secure randomness.
fn generate_api_key() -> String {
    use std::fmt::Write;
    let mut bytes = [0u8; 16];
    getrandom::fill(&mut bytes).expect("OS RNG unavailable");
    let mut hex = String::with_capacity(32);
    for byte in bytes {
        write!(hex, "{byte:02x}").ok();
    }
    hex
}
