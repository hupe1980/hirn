use std::sync::Arc;

use hirn::prelude::*;
use hirn_core::types::NamespaceKind;
use hirn_engine::HirnDB;
use metrics::counter;
use tokio::sync::broadcast;
use tonic::metadata::MetadataMap;
use tonic::{Request, Response, Status};

use crate::auth::{
    AuthState, Operation, ResolvedIdentity, TokenError, token_allows_namespace,
    token_allows_operation,
};
use crate::convert;
use crate::proto;
use crate::proto::hirn_service_server::HirnService;
use crate::realm::RealmManager;
use crate::throttle::{RateLimitClass, RateLimiter};
use crate::watch::WatchEvent;
use crate::watch::WatchNamespaceScope;

const INTERNAL_GRPC_METADATA_HEADERS: &[&str] = &["x-token-namespaces", "x-token-operations"];

/// Shared server state holding the hirn engine and watch broadcaster.
pub struct HirnGrpcService {
    realms: Arc<RealmManager>,
    watch_tx: broadcast::Sender<WatchEvent>,
    rate_limiter: Arc<RateLimiter>,
}

impl HirnGrpcService {
    #[must_use]
    pub fn new(
        realms: Arc<RealmManager>,
        watch_tx: broadcast::Sender<WatchEvent>,
        rate_limiter: Arc<RateLimiter>,
    ) -> Self {
        Self {
            realms,
            watch_tx,
            rate_limiter,
        }
    }

    async fn get_db(&self, metadata: &MetadataMap) -> Result<Arc<HirnDB>, Status> {
        realm_db(&self.realms, metadata).await
    }

    async fn ensure_read_consistency(&self, _metadata: &MetadataMap) -> Result<(), Status> {
        Ok(())
    }
}

pub fn grpc_auth_interceptor(
    auth_state: Arc<AuthState>,
) -> impl tonic::service::Interceptor + Clone {
    move |mut request: Request<()>| {
        for header in INTERNAL_GRPC_METADATA_HEADERS {
            if request.metadata_mut().remove(*header).is_some() {
                counter!(
                    "hirnd_internal_metadata_strips_total",
                    "interface" => "grpc",
                    "header" => *header,
                )
                .increment(1);
            }
        }

        if !auth_state.is_enabled() {
            if !auth_state.allows_unauthenticated() {
                tracing::warn!(
                    "gRPC auth rejected: auth is not configured and insecure_dev_mode is disabled"
                );
                return Err(Status::unauthenticated("authentication required"));
            }
            return Ok(request);
        }

        let auth_header = request
            .metadata()
            .get("authorization")
            .and_then(|value| value.to_str().ok());
        let bearer = match auth_header {
            Some(header) if header.starts_with("Bearer ") => &header[7..],
            _ => {
                tracing::warn!("gRPC auth failed: missing or invalid authorization metadata");
                return Err(Status::unauthenticated(
                    "missing or invalid authorization metadata",
                ));
            }
        };

        let identity = if auth_state.tokens_enabled() {
            match auth_state.validate_token(bearer) {
                Ok(claims) => {
                    let ns_json = serde_json::to_string(&claims.namespaces)
                        .map_err(|_| Status::internal("failed to encode token namespaces"))?;
                    request.metadata_mut().insert(
                        "x-token-namespaces",
                        ns_json
                            .parse()
                            .map_err(|_| Status::internal("failed to attach token namespaces"))?,
                    );
                    let ops_json = serde_json::to_string(&claims.operations)
                        .map_err(|_| Status::internal("failed to encode token operations"))?;
                    request.metadata_mut().insert(
                        "x-token-operations",
                        ops_json
                            .parse()
                            .map_err(|_| Status::internal("failed to attach token operations"))?,
                    );
                    ResolvedIdentity {
                        realm: claims.realm,
                        agent_id: claims.agent_id,
                        namespaces: claims.namespaces,
                        operations: claims.operations,
                    }
                }
                Err(TokenError::Expired) => {
                    tracing::warn!("gRPC auth failed: token expired");
                    return Err(Status::unauthenticated("token expired"));
                }
                Err(TokenError::NotConfigured) => {
                    return Err(Status::internal("token validation misconfigured"));
                }
                Err(TokenError::Invalid(_)) => match auth_state.validate(bearer) {
                    Some(identity) => ResolvedIdentity {
                        realm: identity.realm.clone(),
                        agent_id: identity.agent_id.clone(),
                        namespaces: vec![],
                        operations: vec![],
                    },
                    None => {
                        tracing::warn!("gRPC auth failed: invalid bearer credential");
                        return Err(Status::unauthenticated("invalid bearer credential"));
                    }
                },
            }
        } else {
            match auth_state.validate(bearer) {
                Some(identity) => ResolvedIdentity {
                    realm: identity.realm.clone(),
                    agent_id: identity.agent_id.clone(),
                    namespaces: vec![],
                    operations: vec![],
                },
                None => {
                    tracing::warn!("gRPC auth failed: invalid bearer credential");
                    return Err(Status::unauthenticated("invalid bearer credential"));
                }
            }
        };

        request.metadata_mut().insert(
            "x-realm-id",
            identity
                .realm
                .parse()
                .map_err(|_| Status::internal("failed to encode realm metadata"))?,
        );
        request.metadata_mut().insert(
            "x-agent-id",
            identity
                .agent_id
                .parse()
                .map_err(|_| Status::internal("failed to encode agent metadata"))?,
        );

        Ok(request)
    }
}

fn parse_token_operations(metadata: &MetadataMap) -> Result<Option<Vec<Operation>>, Status> {
    let Some(raw) = metadata.get("x-token-operations") else {
        return Ok(None);
    };

    let raw = raw
        .to_str()
        .map_err(|_| Status::internal("invalid x-token-operations metadata"))?;
    let operations = serde_json::from_str(raw)
        .map_err(|_| Status::internal("invalid x-token-operations metadata"))?;
    Ok(Some(operations))
}

fn extract_realm_id(metadata: &MetadataMap) -> Result<String, Status> {
    metadata
        .get("x-realm-id")
        .ok_or_else(|| Status::invalid_argument("missing x-realm-id metadata"))?
        .to_str()
        .map_err(|_| Status::invalid_argument("x-realm-id metadata is not valid UTF-8"))
        .map(|value| value.to_string())
}

fn extract_agent_id(metadata: &MetadataMap) -> Result<AgentId, Status> {
    let value = metadata
        .get("x-agent-id")
        .ok_or_else(|| Status::invalid_argument("missing x-agent-id metadata"))?
        .to_str()
        .map_err(|_| Status::invalid_argument("x-agent-id metadata is not valid UTF-8"))?;
    AgentId::new(value).map_err(|e| Status::invalid_argument(format!("invalid agent_id: {e}")))
}

fn parse_token_namespaces(metadata: &MetadataMap) -> Result<Option<Vec<String>>, Status> {
    let Some(raw) = metadata.get("x-token-namespaces") else {
        return Ok(None);
    };

    let raw = raw
        .to_str()
        .map_err(|_| Status::internal("invalid x-token-namespaces metadata"))?;
    let namespaces = serde_json::from_str(raw)
        .map_err(|_| Status::internal("invalid x-token-namespaces metadata"))?;
    Ok(Some(namespaces))
}

fn enforce_rate_limit(
    rate_limiter: &RateLimiter,
    metadata: &MetadataMap,
    class: RateLimitClass,
) -> Result<(), Status> {
    let realm = extract_realm_id(metadata)?;
    let agent_id = extract_agent_id(metadata)?;

    if rate_limiter.check_agent(class, &realm, agent_id.as_str()) {
        return Ok(());
    }

    Err(Status::resource_exhausted(format!(
        "{} rate limit exceeded",
        class.as_str()
    )))
}

fn watch_namespace_scope(
    metadata: &MetadataMap,
    agent_id: &AgentId,
    requested_namespace: Option<String>,
) -> Result<WatchNamespaceScope, Status> {
    match parse_token_namespaces(metadata)? {
        Some(allowed_namespaces) => Ok(WatchNamespaceScope::token_scoped(
            agent_id,
            requested_namespace,
            allowed_namespaces,
        )),
        None => Ok(WatchNamespaceScope::unrestricted(requested_namespace)),
    }
}

fn check_operation(metadata: &MetadataMap, required: &Operation) -> Result<(), Status> {
    if let Some(operations) = parse_token_operations(metadata)? {
        if !token_allows_operation(&operations, required) {
            return Err(Status::permission_denied(format!(
                "token does not permit {required:?} operations"
            )));
        }
    }

    Ok(())
}

fn check_namespace(
    metadata: &MetadataMap,
    agent_id: &AgentId,
    namespace: Option<&str>,
) -> Result<(), Status> {
    if let Some(ns) = namespace {
        Namespace::new(ns)
            .map_err(|e| Status::invalid_argument(format!("invalid namespace: {e}")))?;
    }

    if let Some(namespaces) = parse_token_namespaces(metadata)? {
        if !token_allows_namespace(agent_id, &namespaces, namespace) {
            return Err(Status::permission_denied(format!(
                "token does not permit access to namespace '{}'",
                namespace.unwrap_or("default")
            )));
        }
    }

    Ok(())
}

fn execute_statement_operation(stmt: &hirn_engine::ql::ast::Statement) -> Operation {
    match stmt {
        hirn_engine::ql::ast::Statement::Explain(explain) if explain.analyze => {
            execute_statement_operation(&explain.inner)
        }
        hirn_engine::ql::ast::Statement::Correct(_)
        | hirn_engine::ql::ast::Statement::Supersede(_)
        | hirn_engine::ql::ast::Statement::MergeMemory(_)
        | hirn_engine::ql::ast::Statement::Retract(_) => Operation::Write,
        hirn_engine::ql::ast::Statement::Grant(_)
        | hirn_engine::ql::ast::Statement::Revoke(_)
        | hirn_engine::ql::ast::Statement::SetTierPolicy(_)
        | hirn_engine::ql::ast::Statement::CreateRealm(_)
        | hirn_engine::ql::ast::Statement::DropRealm(_) => Operation::Admin,
        _ => Operation::Read,
    }
}

fn execute_statement_namespace(stmt: &hirn_engine::ql::ast::Statement) -> Option<&str> {
    match stmt {
        hirn_engine::ql::ast::Statement::Explain(explain) => {
            execute_statement_namespace(&explain.inner)
        }
        hirn_engine::ql::ast::Statement::Recall(stmt) => stmt.namespace.as_deref(),
        hirn_engine::ql::ast::Statement::RecallEvents(stmt) => stmt.namespace.as_deref(),
        hirn_engine::ql::ast::Statement::Think(stmt) => stmt.namespace.as_deref(),
        hirn_engine::ql::ast::Statement::Traverse(stmt) => stmt.namespace.as_deref(),
        hirn_engine::ql::ast::Statement::History(stmt) => stmt.namespace.as_deref(),
        hirn_engine::ql::ast::Statement::ExplainCauses(stmt) => stmt.namespace.as_deref(),
        hirn_engine::ql::ast::Statement::WhatIf(stmt) => stmt.namespace.as_deref(),
        hirn_engine::ql::ast::Statement::Counterfactual(stmt) => stmt.namespace.as_deref(),
        _ => None,
    }
}

/// Resolve the realm database from metadata.
async fn realm_db(
    realms: &RealmManager,
    metadata: &tonic::metadata::MetadataMap,
) -> Result<Arc<HirnDB>, Status> {
    let realm_id = extract_realm_id(metadata)?;
    realms.get(&realm_id).await.map_err(|e| {
        tracing::error!(realm = %realm_id, error = %e, "realm database lookup failed");
        Status::internal("realm not found or unavailable")
    })
}

/// Map `HirnError` to gRPC `Status`.
///
/// Client-facing errors (not_found, invalid_argument, etc.) include the
/// specific message. Internal/storage/config errors are logged server-side
/// but return a generic message to avoid leaking implementation details.
fn map_err(e: HirnError) -> Status {
    match &e {
        HirnError::NotFound(_) => Status::not_found(e.to_string()),
        HirnError::AlreadyExists(_) => Status::already_exists(e.to_string()),
        HirnError::InvalidInput(_) => Status::invalid_argument(e.to_string()),
        HirnError::AccessDenied(_) => Status::permission_denied(e.to_string()),
        HirnError::Quarantined(_) => Status::failed_precondition(e.to_string()),
        HirnError::FileLocked => Status::unavailable("resource is temporarily locked"),
        _ => {
            tracing::error!(error = %e, "internal gRPC error");
            Status::internal("internal error")
        }
    }
}

/// Ensure the agent is registered and return a namespace-scoped context.
async fn agent_context<'a>(
    db: &'a HirnDB,
    agent_id: &AgentId,
) -> Result<hirn_engine::AgentContext<'a>, Status> {
    db.ensure_agent(agent_id).await.map_err(map_err)?;
    db.as_agent(agent_id).await.map_err(map_err)
}

fn parse_namespace_arg(namespace: Option<&str>) -> Result<Option<Namespace>, Status> {
    namespace
        .map(Namespace::new)
        .transpose()
        .map_err(|e| Status::invalid_argument(format!("invalid namespace: {e}")))
}

fn build_agent_recall<'a>(
    ctx: &'a hirn_engine::AgentContext<'a>,
    inner: proto::RecallRequest,
) -> Result<hirn_engine::agent_context::AgentRecallBuilder<'a>, Status> {
    let mut builder = ctx.recall(inner.query_embedding);

    if inner.limit > 0 {
        builder = builder.limit(inner.limit as usize);
    }
    if inner.threshold > 0.0 {
        builder = builder.threshold(inner.threshold);
    }
    if let Some(namespace) = parse_namespace_arg(inner.namespace.as_deref())? {
        builder = builder.namespace(namespace);
    }
    if let Some(snapshot) = inner.snapshot.as_ref() {
        builder = builder.snapshot(convert::recall_snapshot_from_proto(snapshot).map_err(map_err)?);
    }

    Ok(builder)
}

#[tonic::async_trait]
impl HirnService for HirnGrpcService {
    // ── Remember ─────────────────────────────────────────────

    async fn remember(
        &self,
        request: Request<proto::RememberRequest>,
    ) -> Result<Response<proto::RememberResponse>, Status> {
        let metadata = request.metadata().clone();
        let agent = extract_agent_id(&metadata)?;
        check_operation(&metadata, &Operation::Write)?;
        enforce_rate_limit(&self.rate_limiter, &metadata, RateLimitClass::Write)?;
        let db = self.get_db(&metadata).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();

        let (id, layer, watch_entities, watch_importance, watch_namespace) = match inner.record {
            Some(proto::remember_request::Record::Episodic(ep)) => {
                check_namespace(
                    &metadata,
                    &agent,
                    if ep.namespace.is_empty() {
                        None
                    } else {
                        Some(ep.namespace.as_str())
                    },
                )?;
                let agent_id = agent.clone();
                let w_entities: Vec<String> = ep.entities.iter().map(|e| e.name.clone()).collect();
                let w_importance = ep.importance;
                let w_namespace = if ep.namespace.is_empty() {
                    Namespace::default()
                } else {
                    Namespace::new(&ep.namespace)
                        .map_err(|e| Status::invalid_argument(format!("invalid namespace: {e}")))?
                };

                let mut builder = EpisodicRecord::builder()
                    .content(&ep.content)
                    .agent_id(agent_id)
                    .event_type(convert::event_type_from_proto(ep.event_type))
                    .importance(ep.importance)
                    .surprise(ep.surprise);

                if !ep.summary.is_empty() {
                    builder = builder.summary(&ep.summary);
                }
                if !ep.embedding.is_empty() {
                    builder = builder.embedding(ep.embedding);
                }
                if !ep.namespace.is_empty() {
                    if let Ok(ns) = Namespace::new(&ep.namespace) {
                        builder = builder.namespace(ns);
                    }
                }
                for entity in &ep.entities {
                    builder = builder.entity(&entity.name, &entity.role);
                }
                let metadata = convert::metadata_from_proto(&ep.metadata);
                for (k, v) in metadata {
                    builder = builder.metadata_entry(k, v);
                }
                let record = builder.build().map_err(|e| {
                    Status::invalid_argument(format!("failed to build episodic record: {e}"))
                })?;
                let id = ctx.remember(record).await.map_err(map_err)?;
                (id, Layer::Episodic, w_entities, w_importance, w_namespace)
            }
            Some(proto::remember_request::Record::Semantic(sem)) => {
                check_namespace(
                    &metadata,
                    &agent,
                    if sem.namespace.is_empty() {
                        None
                    } else {
                        Some(sem.namespace.as_str())
                    },
                )?;
                let agent_id = agent.clone();
                let w_importance = sem.confidence;
                let w_namespace = if sem.namespace.is_empty() {
                    Namespace::default()
                } else {
                    Namespace::new(&sem.namespace)
                        .map_err(|e| Status::invalid_argument(format!("invalid namespace: {e}")))?
                };

                let mut builder = SemanticRecord::builder()
                    .concept(&sem.concept)
                    .description(&sem.description)
                    .agent_id(agent_id)
                    .knowledge_type(convert::knowledge_type_from_proto(sem.knowledge_type))
                    .confidence(sem.confidence);

                if !sem.embedding.is_empty() {
                    builder = builder.embedding(sem.embedding);
                }
                if !sem.namespace.is_empty() {
                    if let Ok(ns) = Namespace::new(&sem.namespace) {
                        builder = builder.namespace(ns);
                    }
                }
                let record = builder.build().map_err(|e| {
                    Status::invalid_argument(format!("failed to build semantic record: {e}"))
                })?;
                let id = ctx.store_semantic(record).await.map_err(map_err)?;
                (id, Layer::Semantic, vec![], w_importance, w_namespace)
            }
            Some(proto::remember_request::Record::Working(wm)) => {
                let agent_id = AgentId::new(&wm.agent_id).unwrap_or_else(|_| agent.clone());

                let mut builder = WorkingMemoryEntry::builder()
                    .content(&wm.content)
                    .agent_id(agent_id)
                    .relevance_score(wm.relevance_score)
                    .token_count(wm.token_count);

                if let Some(p) = proto::Priority::try_from(wm.priority).ok() {
                    builder = builder.priority(convert::priority_from_proto(p as i32));
                }
                let entry = builder.build().map_err(|e| {
                    Status::invalid_argument(format!("failed to build working memory entry: {e}"))
                })?;
                // Working memory is not namespace-scoped.
                let id = db.working().focus(entry).await.map_err(map_err)?;
                (id, Layer::Working, vec![], 0.0, Namespace::default())
            }
            None => {
                return Err(Status::invalid_argument("record is required"));
            }
        };

        // Broadcast watch event
        let _ = self.watch_tx.send(WatchEvent::Created {
            id: id.clone(),
            layer: layer.clone(),
            entities: watch_entities,
            importance: watch_importance,
            namespace: watch_namespace,
        });

        Ok(Response::new(proto::RememberResponse {
            id: Some(convert::memory_id_to_proto(&id)),
            layer: convert::layer_to_proto(&layer),
        }))
    }

    // ── Recall ───────────────────────────────────────────────

    async fn recall(
        &self,
        request: Request<proto::RecallRequest>,
    ) -> Result<Response<proto::RecallResponse>, Status> {
        let metadata = request.metadata().clone();
        self.ensure_read_consistency(&metadata).await?;
        let agent = extract_agent_id(&metadata)?;
        check_operation(&metadata, &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, &metadata, RateLimitClass::Read)?;
        let db = self.get_db(&metadata).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();
        check_namespace(&metadata, &agent, inner.namespace.as_deref())?;

        if inner.query_embedding.is_empty() {
            return Err(Status::invalid_argument("query_embedding is required"));
        }

        let builder = build_agent_recall(&ctx, inner)?;
        let results = builder.execute().await.map_err(map_err)?;

        let proto_results: Vec<proto::RecallResult> = results
            .iter()
            .map(convert::recall_result_to_proto)
            .collect();

        Ok(Response::new(proto::RecallResponse {
            results: proto_results,
        }))
    }

    // ── RecallStream (server-streaming) ──────────────────────

    type RecallStreamStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<proto::RecallResult, Status>> + Send>,
    >;

    async fn recall_stream(
        &self,
        request: Request<proto::RecallRequest>,
    ) -> Result<Response<Self::RecallStreamStream>, Status> {
        let metadata = request.metadata().clone();
        self.ensure_read_consistency(&metadata).await?;
        let agent = extract_agent_id(&metadata)?;
        check_operation(&metadata, &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, &metadata, RateLimitClass::Read)?;
        let db = self.get_db(&metadata).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();
        check_namespace(&metadata, &agent, inner.namespace.as_deref())?;

        if inner.query_embedding.is_empty() {
            return Err(Status::invalid_argument("query_embedding is required"));
        }

        let builder = build_agent_recall(&ctx, inner)?;
        let results = builder.execute().await.map_err(map_err)?;

        let stream = async_stream::stream! {
            for r in results {
                yield Ok(convert::recall_result_to_proto(&r));
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    // ── Think ────────────────────────────────────────────────

    async fn think(
        &self,
        request: Request<proto::ThinkRequest>,
    ) -> Result<Response<proto::ThinkResponse>, Status> {
        let metadata = request.metadata().clone();
        self.ensure_read_consistency(&metadata).await?;
        let agent = extract_agent_id(&metadata)?;
        check_operation(&metadata, &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, &metadata, RateLimitClass::Read)?;
        let db = self.get_db(&metadata).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();
        check_namespace(&metadata, &agent, inner.namespace.as_deref())?;

        if inner.query_embedding.is_empty() {
            return Err(Status::invalid_argument("query_embedding is required"));
        }

        let mut builder = ctx.think(inner.query_embedding);

        if inner.limit > 0 {
            builder = builder.limit(inner.limit as usize);
        }
        if let Some(ns_str) = &inner.namespace {
            if let Ok(ns) = Namespace::new(ns_str) {
                builder = builder.namespace(ns);
            }
        }
        if inner.budget > 0 {
            builder = builder.budget(inner.budget as usize);
        }

        let result = builder.execute().await.map_err(map_err)?;

        Ok(Response::new(proto::ThinkResponse {
            context: result.context,
            token_count: result.token_count as u32,
            records_included: result
                .records_included
                .iter()
                .map(convert::memory_id_to_proto)
                .collect(),
            records_excluded_count: result.records_excluded_count as u32,
            contradictions: result
                .contradictions
                .iter()
                .map(convert::conflict_pair_to_proto)
                .collect(),
            query_time_ms: result.query_time_ms,
            score_distribution: Some(proto::ScoreDistribution {
                min: result.score_distribution.min,
                max: result.score_distribution.max,
                mean: result.score_distribution.mean,
            }),
            conflict_groups: result
                .conflict_groups
                .iter()
                .map(convert::conflict_group_to_proto)
                .collect(),
        }))
    }

    // ── Forget ───────────────────────────────────────────────

    async fn forget(
        &self,
        request: Request<proto::ForgetRequest>,
    ) -> Result<Response<proto::ForgetResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Write)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Write,
        )?;
        let db = self.get_db(request.metadata()).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();

        let id = inner
            .id
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("id is required"))?;
        let memory_id = convert::memory_id_from_proto(id).map_err(map_err)?;

        let mode = proto::ForgetMode::try_from(inner.mode).unwrap_or(proto::ForgetMode::Archive);
        match mode {
            proto::ForgetMode::Purge => match ctx.delete_episode(memory_id).await {
                Ok(()) => {}
                Err(_) => {
                    ctx.purge_semantic(memory_id).await.map_err(map_err)?;
                }
            },
            _ => {
                ctx.archive_episode(memory_id).await.map_err(map_err)?;
            }
        }

        Ok(Response::new(proto::ForgetResponse {}))
    }

    // ── Focus ────────────────────────────────────────────────

    async fn focus(
        &self,
        request: Request<proto::FocusRequest>,
    ) -> Result<Response<proto::FocusResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Write)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Write,
        )?;
        let db = self.get_db(request.metadata()).await?;
        db.ensure_agent(&agent).await.map_err(map_err)?;
        let inner = request.into_inner();

        let wm = inner
            .entry
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("entry is required"))?;

        let mut builder = WorkingMemoryEntry::builder()
            .content(&wm.content)
            .agent_id(agent)
            .relevance_score(wm.relevance_score)
            .token_count(wm.token_count);

        if let Ok(p) = proto::Priority::try_from(wm.priority) {
            builder = builder.priority(convert::priority_from_proto(p as i32));
        }

        let entry = builder.build().map_err(|e| {
            Status::invalid_argument(format!("failed to build working memory entry: {e}"))
        })?;
        let id = db.working().focus(entry).await.map_err(map_err)?;

        Ok(Response::new(proto::FocusResponse {
            id: Some(convert::memory_id_to_proto(&id)),
        }))
    }

    // ── Defocus ──────────────────────────────────────────────

    async fn defocus(
        &self,
        request: Request<proto::DefocusRequest>,
    ) -> Result<Response<proto::DefocusResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Write)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Write,
        )?;
        let db = self.get_db(request.metadata()).await?;
        db.ensure_agent(&agent).await.map_err(map_err)?;
        let inner = request.into_inner();

        let id = inner
            .id
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("id is required"))?;
        let memory_id = convert::memory_id_from_proto(id).map_err(map_err)?;
        db.working().defocus(memory_id).await.map_err(map_err)?;

        Ok(Response::new(proto::DefocusResponse {}))
    }

    // ── LinkMemories ───────────────────────────────────────

    async fn link_memories(
        &self,
        request: Request<proto::ConnectRequest>,
    ) -> Result<Response<proto::ConnectResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Write)?;
        let db = self.get_db(request.metadata()).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();

        let source = inner
            .source
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("source is required"))?;
        let target = inner
            .target
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("target is required"))?;

        let source_id = convert::memory_id_from_proto(source).map_err(map_err)?;
        let target_id = convert::memory_id_from_proto(target).map_err(map_err)?;
        let relation = convert::edge_relation_from_proto(inner.relation);
        let weight = if inner.weight > 0.0 {
            inner.weight
        } else {
            1.0
        };
        let metadata = convert::metadata_from_proto(&inner.metadata);

        let edge_id = ctx
            .connect_with(source_id, target_id, relation, weight, metadata)
            .await
            .map_err(map_err)?;

        Ok(Response::new(proto::ConnectResponse {
            edge_id: Some(convert::memory_id_to_proto(&edge_id)),
        }))
    }

    // ── Inspect ──────────────────────────────────────────────

    async fn inspect(
        &self,
        request: Request<proto::InspectRequest>,
    ) -> Result<Response<proto::InspectResponse>, Status> {
        self.ensure_read_consistency(request.metadata()).await?;
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, request.metadata(), RateLimitClass::Read)?;
        let db = self.get_db(request.metadata()).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();

        let id = inner
            .id
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("id is required"))?;
        let memory_id = convert::memory_id_from_proto(id).map_err(map_err)?;

        let result = ctx.inspect(memory_id).await.map_err(map_err)?;

        Ok(Response::new(proto::InspectResponse {
            record: Some(convert::memory_record_to_proto(&result.record)),
            importance: result.importance,
            access_count: result.access_count,
            last_accessed: Some(convert::timestamp_to_proto(&result.last_accessed)),
            neighbors: result
                .neighbors
                .iter()
                .map(|n| proto::NeighborInfo {
                    edge: Some(proto::GraphEdge {
                        id: Some(convert::memory_id_to_proto(&n.edge.id)),
                        source: Some(convert::memory_id_to_proto(&n.edge.source)),
                        target: Some(convert::memory_id_to_proto(&n.edge.target)),
                        relation: convert::edge_relation_to_proto(&n.edge.relation),
                        weight: n.edge.weight,
                        co_retrieval_count: n.edge.co_retrieval_count,
                        created_at: Some(convert::timestamp_to_proto(&n.edge.created_at)),
                        updated_at: Some(convert::timestamp_to_proto(&n.edge.updated_at)),
                        valid_from: n.edge.valid_from.map(|ts| ts.to_string()),
                        valid_until: n.edge.valid_until.map(|ts| ts.to_string()),
                        metadata: convert::metadata_to_proto(&n.edge.metadata),
                    }),
                    neighbor_id: Some(convert::memory_id_to_proto(&n.neighbor_id)),
                })
                .collect(),
            trust_score: result.trust_score,
            semantic_revision: result
                .semantic_revision
                .as_ref()
                .map(convert::semantic_revision_summary_to_proto),
            conflict_groups: result
                .conflict_groups
                .iter()
                .map(convert::conflict_group_to_proto)
                .collect(),
        }))
    }

    // ── Trace ────────────────────────────────────────────────

    async fn trace(
        &self,
        request: Request<proto::TraceRequest>,
    ) -> Result<Response<proto::TraceResponse>, Status> {
        self.ensure_read_consistency(request.metadata()).await?;
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, request.metadata(), RateLimitClass::Read)?;
        let db = self.get_db(request.metadata()).await?;
        let ctx = agent_context(&db, &agent).await?;
        let inner = request.into_inner();

        let id = inner
            .id
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("id is required"))?;
        let memory_id = convert::memory_id_from_proto(id).map_err(map_err)?;

        let result = ctx.trace(memory_id).await.map_err(map_err)?;

        Ok(Response::new(proto::TraceResponse {
            record: Some(convert::memory_record_to_proto(&result.record)),
            source_episodes: result
                .source_episodes
                .iter()
                .map(convert::memory_id_to_proto)
                .collect(),
            derived_records: result
                .derived_records
                .iter()
                .map(convert::memory_id_to_proto)
                .collect(),
            mutation_count: result.mutation_count as u32,
            trust_score: result.trust_score,
            lineage_tree: result.lineage_tree,
            semantic_revision: result
                .semantic_revision
                .as_ref()
                .map(convert::semantic_revision_summary_to_proto),
            conflict_groups: result
                .conflict_groups
                .iter()
                .map(convert::conflict_group_to_proto)
                .collect(),
        }))
    }

    // ── Execute (HirnQL) ─────────────────────────────────────

    async fn execute(
        &self,
        request: Request<proto::ExecuteRequest>,
    ) -> Result<Response<proto::ExecuteResponse>, Status> {
        let metadata = request.metadata().clone();
        let agent = extract_agent_id(&metadata)?;
        let inner = request.into_inner();

        if inner.query.is_empty() {
            return Err(Status::invalid_argument("query is required"));
        }

        let stmt = hirn_engine::ql::parser::parse(&inner.query)
            .map_err(|e| Status::invalid_argument(format!("HirnQL parse error: {e}")))?;
        let required_op = execute_statement_operation(&stmt);
        check_operation(&metadata, &required_op)?;
        let rate_limit_class = match required_op {
            Operation::Read => RateLimitClass::Read,
            Operation::Write => RateLimitClass::Write,
            Operation::Admin => RateLimitClass::Admin,
        };
        enforce_rate_limit(&self.rate_limiter, &metadata, rate_limit_class)?;
        if let Some(namespace) = execute_statement_namespace(&stmt) {
            check_namespace(&metadata, &agent, Some(namespace))?;
        }
        if required_op == Operation::Read {
            self.ensure_read_consistency(&metadata).await?;
        }

        let db = self.get_db(&metadata).await?;
        let ctx = agent_context(&db, &agent).await?;

        let result = ctx.execute_ql(&inner.query).await.map_err(map_err)?;

        let response =
            convert::query_result_to_execute_response(&result).map_err(Status::unimplemented)?;

        Ok(Response::new(response))
    }

    // ── Consolidate ──────────────────────────────────────────

    async fn consolidate(
        &self,
        request: Request<proto::ConsolidateRequest>,
    ) -> Result<Response<proto::ConsolidateResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Admin)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Admin,
        )?;
        let db = self.get_db(request.metadata()).await?;
        db.ensure_agent(&agent).await.map_err(map_err)?;
        let inner = request.into_inner();

        let mut builder = db.admin().consolidate();

        if let Some(t) = inner.topic_threshold {
            builder = builder.topic_threshold(t);
        }
        if let Some(s) = inner.surprise_threshold {
            builder = builder.surprise_threshold(s);
        }
        if let Some(g) = inner.temporal_gap_secs {
            builder = builder.temporal_gap(g);
        }
        builder = builder.archive(inner.archive);

        let result = builder.execute().await.map_err(map_err)?;

        Ok(Response::new(proto::ConsolidateResponse {
            records_processed: result.records_processed as u32,
            segments_created: result.segments_created as u32,
            patterns_detected: result.patterns_detected as u32,
            threads_formed: result.threads_formed as u32,
            concepts_extracted: result.concepts_extracted as u32,
            provenance_edges_created: result.provenance_edges_created as u32,
            episodes_archived: result.episodes_archived as u32,
            execution_time_ms: result.execution_time_ms,
        }))
    }

    // ── CreateNamespace ──────────────────────────────────────

    async fn create_namespace(
        &self,
        request: Request<proto::CreateNamespaceRequest>,
    ) -> Result<Response<proto::CreateNamespaceResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Admin)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Admin,
        )?;
        let db = self.get_db(request.metadata()).await?;
        db.ensure_agent(&agent).await.map_err(map_err)?;
        let inner = request.into_inner();

        if inner.name.is_empty() {
            return Err(Status::invalid_argument("name is required"));
        }

        let kind = match inner.kind.as_str() {
            "team" | "Team" => NamespaceKind::Team,
            "shared" | "Shared" => NamespaceKind::Shared,
            _ => NamespaceKind::Private,
        };

        let members: Vec<AgentId> = inner
            .member_agent_ids
            .iter()
            .filter_map(|s| AgentId::new(s).ok())
            .collect();

        db.namespaces()
            .create(&inner.name, kind, members)
            .await
            .map_err(map_err)?;

        Ok(Response::new(proto::CreateNamespaceResponse {}))
    }

    // ── ShareMemory ──────────────────────────────────────────

    async fn share_memory(
        &self,
        request: Request<proto::ShareMemoryRequest>,
    ) -> Result<Response<proto::ShareMemoryResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Write)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Write,
        )?;
        let db = self.get_db(request.metadata()).await?;
        let inner = request.into_inner();

        let id = inner
            .id
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("id is required"))?;
        let memory_id = convert::memory_id_from_proto(id).map_err(map_err)?;

        let agent_id = if inner.agent_id.is_empty() {
            agent
        } else {
            AgentId::new(&inner.agent_id)
                .map_err(|e| Status::invalid_argument(format!("invalid agent_id: {e}")))?
        };

        let target_ns = Namespace::new(&inner.target_namespace)
            .map_err(|e| Status::invalid_argument(format!("invalid target_namespace: {e}")))?;

        let ctx = agent_context(&db, &agent_id).await?;
        let new_id = ctx
            .share_memory(memory_id, &target_ns)
            .await
            .map_err(map_err)?;

        Ok(Response::new(proto::ShareMemoryResponse {
            new_id: Some(convert::memory_id_to_proto(&new_id)),
        }))
    }

    // ── Stats ────────────────────────────────────────────────

    async fn stats(
        &self,
        request: Request<proto::StatsRequest>,
    ) -> Result<Response<proto::StatsResponse>, Status> {
        self.ensure_read_consistency(request.metadata()).await?;
        check_operation(request.metadata(), &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, request.metadata(), RateLimitClass::Read)?;
        let db = self.get_db(request.metadata()).await?;
        let stats = db.admin().stats().await.map_err(map_err)?;

        Ok(Response::new(proto::StatsResponse {
            working_count: stats.working_count,
            episodic_count: stats.episodic_count,
            semantic_count: stats.semantic_count,
            total_count: stats.total_count,
            file_size_bytes: stats.file_size_bytes,
        }))
    }

    // ── Watch (server-streaming) ─────────────────────────────

    type WatchStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<proto::WatchEvent, Status>> + Send>,
    >;

    async fn watch(
        &self,
        request: Request<proto::WatchRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let metadata = request.metadata().clone();
        let agent = extract_agent_id(&metadata)?;
        check_operation(&metadata, &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, &metadata, RateLimitClass::Read)?;
        let db = self.get_db(&metadata).await?;
        db.ensure_agent(&agent).await.map_err(map_err)?;
        let inner = request.into_inner();
        if let Some(namespace) = inner.namespace.as_deref() {
            check_namespace(&metadata, &agent, Some(namespace))?;
        }

        let layer_filter = inner.layer_filter.and_then(convert::layer_from_proto);
        let entities: Vec<String> = inner.entities;
        let min_importance = inner.min_importance;
        let namespace_scope = watch_namespace_scope(&metadata, &agent, inner.namespace.clone())?;

        let mut rx = self.watch_tx.subscribe();

        let stream = async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(event) => {
                        if let Some(proto_event) = event.to_proto(&layer_filter, &entities, min_importance, &namespace_scope) {
                            yield Ok(proto_event);
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("watch subscriber lagged, dropped {n} events");
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }

    // ── CreateSnapshot ─────────────────────────────────────────

    async fn create_snapshot(
        &self,
        request: Request<proto::CreateSnapshotRequest>,
    ) -> Result<Response<proto::CreateSnapshotResponse>, Status> {
        check_operation(request.metadata(), &Operation::Admin)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Admin,
        )?;
        let db = self.get_db(request.metadata()).await?;
        let inner = request.into_inner();

        let report = hirn_engine::backup::create_snapshot(db.storage_backend(), &inner.name)
            .await
            .map_err(|e| Status::internal(format!("snapshot failed: {e}")))?;

        Ok(Response::new(proto::CreateSnapshotResponse {
            datasets_tagged: report.datasets_tagged as u32,
        }))
    }

    // ── ListSnapshots ────────────────────────────────────────

    async fn list_snapshots(
        &self,
        request: Request<proto::ListSnapshotsRequest>,
    ) -> Result<Response<proto::ListSnapshotsResponse>, Status> {
        check_operation(request.metadata(), &Operation::Admin)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Admin,
        )?;
        let db = self.get_db(request.metadata()).await?;

        let snapshots = hirn_engine::backup::list_snapshots(db.storage_backend())
            .await
            .map_err(|e| Status::internal(format!("list snapshots failed: {e}")))?;

        let infos = snapshots
            .into_iter()
            .map(|s| proto::SnapshotInfo {
                name: s.name,
                versions: s.versions.into_iter().collect(),
            })
            .collect();

        Ok(Response::new(proto::ListSnapshotsResponse {
            snapshots: infos,
        }))
    }

    // ── Rollback ─────────────────────────────────────────────

    async fn rollback(
        &self,
        request: Request<proto::RollbackRequest>,
    ) -> Result<Response<proto::RollbackResponse>, Status> {
        check_operation(request.metadata(), &Operation::Admin)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Admin,
        )?;
        let db = self.get_db(request.metadata()).await?;
        let inner = request.into_inner();

        let report = hirn_engine::backup::rollback(db.storage_backend(), &inner.name)
            .await
            .map_err(|e| Status::internal(format!("rollback failed: {e}")))?;

        Ok(Response::new(proto::RollbackResponse {
            datasets_rolled_back: report.datasets_rolled_back as u32,
        }))
    }

    // ── UpdateMemory ─────────────────────────────────────────

    async fn update_memory(
        &self,
        request: Request<proto::UpdateMemoryRequest>,
    ) -> Result<Response<proto::UpdateMemoryResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Write)?;
        enforce_rate_limit(
            &self.rate_limiter,
            request.metadata(),
            RateLimitClass::Write,
        )?;
        let db = self.get_db(request.metadata()).await?;
        let inner = request.into_inner();

        let id = inner
            .id
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("id is required"))?;
        let memory_id = convert::memory_id_from_proto(id).map_err(map_err)?;

        let metadata = if inner.metadata.is_empty() {
            None
        } else {
            Some(convert::metadata_from_proto(&inner.metadata))
        };

        let toolkit = hirn_engine::MemoryToolkit::new(db);
        toolkit
            .update(
                agent,
                hirn_engine::UpdateRequest {
                    id: memory_id,
                    content: inner.content,
                    metadata,
                    importance: inner.importance,
                },
            )
            .await
            .map_err(map_err)?;

        Ok(Response::new(proto::UpdateMemoryResponse {}))
    }

    // ── ToolkitIntrospect ────────────────────────────────────

    async fn toolkit_introspect(
        &self,
        request: Request<proto::ToolkitIntrospectRequest>,
    ) -> Result<Response<proto::ToolkitIntrospectResponse>, Status> {
        let agent = extract_agent_id(request.metadata())?;
        check_operation(request.metadata(), &Operation::Read)?;
        enforce_rate_limit(&self.rate_limiter, request.metadata(), RateLimitClass::Read)?;
        let db = self.get_db(request.metadata()).await?;
        let inner = request.into_inner();

        let id = inner
            .id
            .map(|ref id| convert::memory_id_from_proto(id))
            .transpose()
            .map_err(map_err)?;

        let toolkit = hirn_engine::MemoryToolkit::new(db);
        let result = toolkit.introspect(agent, id).await.map_err(map_err)?;

        Ok(Response::new(proto::ToolkitIntrospectResponse {
            total_memories: result.total_memories,
            episodic_count: result.episodic_count,
            semantic_count: result.semantic_count,
            procedural_count: result.procedural_count,
            working_count: result.working_count,
            edge_count: result.edge_count,
            edges: result
                .edges
                .iter()
                .map(|e| proto::EdgeInfo {
                    source: Some(convert::memory_id_to_proto(&e.source)),
                    target: Some(convert::memory_id_to_proto(&e.target)),
                    relation: convert::edge_relation_to_proto(&e.relation),
                    weight: e.weight,
                })
                .collect(),
        }))
    }
}
