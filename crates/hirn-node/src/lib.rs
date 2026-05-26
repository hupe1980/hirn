//! Node.js bindings for the hirn cognitive memory database.
//!
//! This module exposes hirn's core API to Node.js via napi-rs.
//!
//! **Threading model (F-72):** The core `hirn::HirnDB` API is fully async,
//! backed by LanceDB. Methods like `remember()`, `recall()`, `think()`
//! execute as async operations on the tokio runtime.

use std::sync::Arc;

use napi_derive::napi;

use hirn::prelude::*;
use hirn::{
    ProviderRegistry, Tokenizer, inspected_result_to_json, trace_result_to_json,
    traced_result_to_json,
};
use hirn_storage::{HirnDb, HirnDbConfig};

// ─── Event Mapping ───────────────────────────────────────────

/// A watch event emitted by the database.
#[napi(object)]
pub struct JsWatchEvent {
    /// Event type: "created" | "archived" | "consolidated"
    pub event_type: String,
    /// Memory ID (for created/archived events). Null for consolidated.
    pub id: Option<String>,
    /// Memory layer (for created events). Null otherwise.
    pub layer: Option<String>,
    /// Content preview (for created events). Null otherwise.
    pub content_preview: Option<String>,
    /// Number of records processed (for consolidated events). Null otherwise.
    pub records_processed: Option<i64>,
}

impl From<MemoryEvent> for JsWatchEvent {
    fn from(event: MemoryEvent) -> Self {
        let event_type = event.event_type().to_string();
        match event {
            MemoryEvent::EpisodeCreated {
                id,
                content_preview,
            } => Self {
                event_type,
                id: Some(id.to_string()),
                layer: Some("episodic".to_owned()),
                content_preview: Some(content_preview),
                records_processed: None,
            },
            MemoryEvent::SemanticCreated { id, concept_name } => Self {
                event_type,
                id: Some(id.to_string()),
                layer: Some("semantic".to_owned()),
                content_preview: Some(concept_name),
                records_processed: None,
            },
            MemoryEvent::ProceduralCreated { id, procedure_name } => Self {
                event_type,
                id: Some(id.to_string()),
                layer: Some("procedural".to_owned()),
                content_preview: Some(procedure_name),
                records_processed: None,
            },
            MemoryEvent::Consolidated { records_processed } => Self {
                event_type,
                id: None,
                layer: None,
                content_preview: None,
                records_processed: Some(i64::try_from(records_processed).unwrap_or(i64::MAX)),
            },
            _ => Self {
                event_type,
                id: None,
                layer: None,
                content_preview: None,
                records_processed: None,
            },
        }
    }
}

/// A watch stream that yields memory events.
///
/// Call `next()` repeatedly to receive events. Returns `null` when the
/// database is closed or the stream is unsubscribed.
///
/// ```js
/// const stream = db.watch();
/// const event = await stream.next();
/// // event: { eventType: "created", id: "...", layer: "Episodic", ... }
/// stream.unsubscribe();
/// ```
#[napi]
pub struct WatchStream {
    rx: Arc<parking_lot::Mutex<Option<tokio::sync::mpsc::Receiver<MemoryEvent>>>>,
    /// Set to `true` by `unsubscribe()` to signal that `next()` should bail out
    /// after its `recv().await` returns. Without this flag, the take/restore
    /// pattern on `rx` cannot distinguish "we took it ourselves" from
    /// "unsubscribe() set it to None".
    cancelled: Arc<std::sync::atomic::AtomicBool>,
    filter_layer: Option<String>,
}

#[napi]
impl WatchStream {
    /// Wait for the next event. Returns `null` if the stream is closed.
    #[napi]
    pub async fn next(&self, filter_type: Option<String>) -> napi::Result<Option<JsWatchEvent>> {
        let filter_layer = self.filter_layer.clone();
        loop {
            // Temporarily take the receiver so we can call recv().await without
            // holding the Mutex across the await point.
            let mut rx = {
                let mut guard = self.rx.lock();
                match guard.take() {
                    Some(rx) => rx,
                    None => return Ok(None), // stream closed
                }
            };

            let event = rx.recv().await;

            // Check if unsubscribe() was called while we awaited.
            if self.cancelled.load(std::sync::atomic::Ordering::Acquire) {
                // Don't restore the receiver — the stream is done.
                return Ok(None);
            }

            // Put the receiver back for the next call to next().
            {
                let mut guard = self.rx.lock();
                *guard = Some(rx);
            }

            match event {
                Some(event) => {
                    // Apply layer filter
                    if let Some(ref layer) = filter_layer {
                        let event_layer = event.layer().map(|l| format!("{l:?}"));
                        if event_layer.as_deref() != Some(layer.as_str()) {
                            continue;
                        }
                    }
                    // Apply type filter
                    if let Some(ref ft) = filter_type {
                        if event.event_type() != ft.as_str() {
                            continue;
                        }
                    }
                    return Ok(Some(JsWatchEvent::from(event)));
                }
                None => {
                    // Channel closed (sender dropped).
                    let mut guard = self.rx.lock();
                    *guard = None;
                    return Ok(None);
                }
            }
        }
    }

    /// Unsubscribe from the event stream.
    #[napi]
    pub fn unsubscribe(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::Release);
        let mut guard = self.rx.lock();
        *guard = None;
    }
}

// ─── Error Mapping ───────────────────────────────────────────

fn to_napi_err(e: hirn::HirnError) -> napi::Error {
    let status = match &e {
        hirn::HirnError::NotFound(_) => napi::Status::GenericFailure,
        hirn::HirnError::InvalidInput(_) => napi::Status::InvalidArg,
        _ => napi::Status::GenericFailure,
    };
    napi::Error::new(status, e.to_string())
}

// ─── Helpers ─────────────────────────────────────────────────

/// Convert JS number array (f64) to Vec<f32> for embeddings.
///
/// Note: f64→f32 cast may lose precision for values outside f32 range.
/// Rejects NaN and infinity inputs that would silently pass through the cast.
fn to_f32_vec(arr: &[f64]) -> napi::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(arr.len());
    for (i, &x) in arr.iter().enumerate() {
        if !x.is_finite() {
            return Err(napi::Error::from_reason(format!(
                "embedding[{i}] is not finite: {x}"
            )));
        }
        let y = x as f32;
        // F-68: Detect f64→f32 overflow (finite f64 outside f32 range → ±inf).
        if !y.is_finite() {
            return Err(napi::Error::from_reason(format!(
                "embedding[{i}] overflows f32 range: {x}"
            )));
        }
        out.push(y);
    }
    Ok(out)
}

/// Parse a ULID string into a `MemoryId`.
fn parse_memory_id(s: &str) -> napi::Result<MemoryId> {
    ulid::Ulid::from_string(s)
        .map(MemoryId::from_ulid)
        .map_err(|e| napi::Error::new(napi::Status::InvalidArg, format!("invalid memory id: {e}")))
}

/// Parse an agent ID string.
fn parse_agent_id(s: &str) -> napi::Result<AgentId> {
    if s.is_empty() {
        return Err(napi::Error::new(
            napi::Status::InvalidArg,
            "agent_id must not be empty",
        ));
    }

    AgentId::new(s)
        .map_err(|e| napi::Error::new(napi::Status::InvalidArg, format!("invalid agent_id: {e}")))
}

fn parse_optional_recall_snapshot(
    value: Option<&str>,
    snapshot_kind: Option<&str>,
) -> napi::Result<Option<RecallSnapshot>> {
    let Some(value) = value else {
        return match snapshot_kind {
            Some(kind) => Err(napi::Error::new(
                napi::Status::InvalidArg,
                format!("snapshot_kind '{kind}' requires an as_of value"),
            )),
            None => Ok(None),
        };
    };

    let snapshot_kind = snapshot_kind.unwrap_or("observed").to_ascii_lowercase();
    match snapshot_kind.as_str() {
        "observed" => Timestamp::parse_date_or_rfc3339(value)
            .map(RecallSnapshot::observed)
            .ok_or_else(|| {
                napi::Error::new(
                    napi::Status::InvalidArg,
                    format!(
                        "invalid observed timestamp '{value}' (expected YYYY-MM-DD or RFC 3339)"
                    ),
                )
            })
            .map(Some),
        "recorded" => Timestamp::parse_date_or_rfc3339(value)
            .map(RecallSnapshot::recorded)
            .ok_or_else(|| {
                napi::Error::new(
                    napi::Status::InvalidArg,
                    format!(
                        "invalid recorded timestamp '{value}' (expected YYYY-MM-DD or RFC 3339)"
                    ),
                )
            })
            .map(Some),
        "revision" => RevisionId::parse(value)
            .map(RecallSnapshot::revision)
            .map(Some)
            .map_err(|error| {
                napi::Error::new(
                    napi::Status::InvalidArg,
                    format!("invalid revision id '{value}': {error}"),
                )
            }),
        other => Err(napi::Error::new(
            napi::Status::InvalidArg,
            format!("invalid snapshot_kind '{other}' (expected observed, recorded, or revision)"),
        )),
    }
}

fn parse_optional_observed_at(value: Option<&str>) -> napi::Result<Option<Timestamp>> {
    value
        .map(|raw| {
            Timestamp::parse_date_or_rfc3339(raw).ok_or_else(|| {
                napi::Error::new(
                    napi::Status::InvalidArg,
                    format!("invalid observed_at '{raw}' (expected YYYY-MM-DD or RFC 3339)"),
                )
            })
        })
        .transpose()
}

fn parse_optional_evidence_count(value: Option<i64>) -> napi::Result<Option<u32>> {
    value
        .map(|raw| {
            if raw < 0 {
                return Err(napi::Error::new(
                    napi::Status::InvalidArg,
                    "evidence_count must be a non-negative integer",
                ));
            }
            u32::try_from(raw).map_err(|_| {
                napi::Error::new(napi::Status::InvalidArg, "evidence_count exceeds u32 range")
            })
        })
        .transpose()
}

fn runtime_handle() -> tokio::runtime::Handle {
    tokio::runtime::Handle::try_current().unwrap_or_else(|_| {
        static RT: std::sync::LazyLock<tokio::runtime::Runtime> =
            std::sync::LazyLock::new(|| tokio::runtime::Runtime::new().expect("tokio runtime"));
        RT.handle().clone()
    })
}

fn resolve_registry_tokenizer(tokenizer_name: &str) -> napi::Result<Arc<dyn Tokenizer>> {
    let registry = ProviderRegistry::from_env();
    registry.tokenizer_by_name(tokenizer_name).ok_or_else(|| {
        napi::Error::new(
            napi::Status::InvalidArg,
            format!("unknown Rust tokenizer '{tokenizer_name}'"),
        )
    })
}

fn authoritative_working_token_count(
    db: &hirn::HirnDB,
    content: &str,
    hinted_count: Option<u32>,
) -> u32 {
    let authoritative_count = db.tokenizer().count_tokens(content) as u32;
    match hinted_count {
        Some(count) if count == authoritative_count => count,
        _ => authoritative_count,
    }
}

// ─── QueryResult to JSON ─────────────────────────────────────

fn query_result_to_json(result: &hirn::ql::QueryResult) -> serde_json::Value {
    use hirn::ql::QueryResult;

    if let Some(json) = hirn::ql::revision_query_result_to_json(result) {
        return json;
    }

    match result {
        QueryResult::Records(r) => {
            let records = r
                .records
                .iter()
                .map(|scored_memory| {
                    serde_json::json!({
                        "record": serde_json::to_value(&scored_memory.record).unwrap_or(serde_json::Value::Null),
                        "revision": serde_json::to_value(scored_memory.revision).unwrap_or(serde_json::Value::Null),
                        "score": scored_memory.score,
                        "score_breakdown": {
                            "similarity": scored_memory.score_breakdown.similarity,
                            "importance": scored_memory.score_breakdown.importance,
                            "recency": scored_memory.score_breakdown.recency,
                            "activation": scored_memory.score_breakdown.activation,
                            "causal_relevance": scored_memory.score_breakdown.causal_relevance,
                            "surprise": scored_memory.score_breakdown.surprise,
                            "source_reliability": scored_memory.score_breakdown.source_reliability,
                        },
                    })
                })
                .collect::<Vec<_>>();
            let conflicts = serde_json::to_value(&r.conflicts).unwrap_or(serde_json::Value::Null);
            let conflict_groups =
                serde_json::to_value(&r.conflict_groups).unwrap_or(serde_json::Value::Null);

            serde_json::json!({
                "type": "records",
                "records": records,
                "records_returned": r.records_returned,
                "records_scanned": r.records_scanned,
                "query_time_ms": r.query_time_ms,
                "context": r.context,
                "conflicts": conflicts,
                "conflict_groups": conflict_groups,
            })
        }
        QueryResult::Created(c) => serde_json::json!({
            "type": "created",
            "id": c.id.to_string(),
            "layer": format!("{:?}", c.layer),
        }),
        QueryResult::Forgotten(f) => serde_json::json!({
            "type": "forgotten",
            "target": f.target,
        }),
        QueryResult::Inspected(i) => inspected_result_to_json(i),
        QueryResult::Traced(t) => traced_result_to_json(t),
        QueryResult::Consolidated(c) => serde_json::json!({
            "type": "consolidated",
            "records_processed": c.records_processed,
        }),
        QueryResult::WatchAck(w) => serde_json::json!({
            "type": "watch_ack",
            "message": w.message,
        }),
        QueryResult::Aggregated(a) => serde_json::json!({
            "type": "aggregated",
            "group_field": a.group_field,
            "function": format!("{}", a.function),
            "groups": a.groups.iter().map(|g| serde_json::json!({
                "key": g.key,
                "value": g.value,
            })).collect::<Vec<_>>(),
            "query_time_ms": a.query_time_ms,
            "formatted": a.formatted,
        }),
        QueryResult::ExplainPlan(e) => {
            let mut result = serde_json::json!({
                "type": "explain",
                "plan_text": e.plan_text,
                "has_actual_results": e.actual_result.is_some(),
            });
            if let Some(ref diag) = e.diagnostics {
                result["diagnostics"] = serde_json::json!({
                    "query_id": diag.query_id.as_ref().map(|id| id.to_string()),
                    "authorize_us": diag.authorize_us,
                    "optimize_ms": diag.optimize_ms,
                    "physical_plan_ms": diag.physical_plan_ms,
                    "execute_plan_ms": diag.execute_plan_ms,
                    "vector_search_ms": diag.vector_search_ms,
                    "graph_expand_ms": diag.graph_expand_ms,
                    "rerank_ms": diag.rerank_ms,
                    "assemble_ms": diag.assemble_ms,
                    "total_ms": diag.total_ms,
                });
            }
            result
        }
        QueryResult::Policy(p) => serde_json::json!({
            "type": "policy",
            "message": p.message,
            "policies": p.policies.iter().map(|(name, text)| serde_json::json!({
                "name": name,
                "text": text,
            })).collect::<Vec<_>>(),
        }),
        QueryResult::SvoEvents(e) => serde_json::json!({
            "type": "svo_events",
            "events_returned": e.events_returned,
            "events": e.events.iter().map(|ev| serde_json::json!({
                "source_memory_id": ev.source_memory_id,
                "subject": ev.subject,
                "verb": ev.verb,
                "object": ev.object,
                "time_start": ev.time_start,
                "time_end": ev.time_end,
                "confidence": ev.confidence,
            })).collect::<Vec<_>>(),
        }),
        QueryResult::Causal(c) => serde_json::json!({
            "type": "causal",
            "kind": c.kind.to_string(),
            "query_time_ms": c.query_time_ms,
            "rows": c.rows.iter().map(|r| {
                r.columns.iter().map(|(k, v)| (k.clone(), serde_json::Value::String(v.clone()))).collect::<serde_json::Map<String, serde_json::Value>>()
            }).collect::<Vec<_>>(),
        }),
        QueryResult::Corrected(_)
        | QueryResult::Superseded(_)
        | QueryResult::Merged(_)
        | QueryResult::Retracted(_)
        | QueryResult::History(_) => unreachable!("handled by revision_query_result_to_json"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use hirn::prelude::{AgentId, SemanticRecord};
    use hirn::ql::QueryResult;
    use hirn::semantic::SemanticSupersession;
    use hirn_storage::{HirnDb, HirnDbConfig, PhysicalStore};

    fn agent() -> AgentId {
        AgentId::new("test_agent").unwrap()
    }

    async fn temp_db() -> (hirn::Hirn, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("node_binding_test");
        let lance_path = dir.path().join("lance");

        let storage_config = HirnDbConfig::local(lance_path.to_str().unwrap());
        let backend: Arc<dyn PhysicalStore> =
            HirnDb::open(storage_config).await.unwrap().store_arc();
        let config = hirn::HirnConfig::builder()
            .db_path(&db_path)
            .working_memory_token_limit(2000)
            .build()
            .unwrap();
        let db = hirn::Hirn::open_with_config(config, backend).await.unwrap();
        (db, dir)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn records_json_includes_revision_metadata() {
        let (db, _dir) = temp_db().await;
        let about = "canonical lease policy";
        let embedding = db.embed_text(about).await.unwrap();
        db.semantic()
            .store(
                SemanticRecord::builder()
                    .concept("cache_policy")
                    .description(about)
                    .embedding(embedding)
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        let result = db
            .ql()
            .execute(&format!(r#"RECALL semantic ABOUT "{about}" LIMIT 10"#))
            .await
            .unwrap();
        let QueryResult::Records(_) = result else {
            panic!("expected Records query result");
        };

        let json_val = query_result_to_json(&result);

        assert_eq!(json_val["type"], "records");
        assert_eq!(json_val["records_returned"], 1);
        assert_eq!(json_val["records"][0]["revision"]["state"], "Active");
        assert!(json_val["records"][0]["record"].is_object());
        assert_eq!(
            json_val["records"][0]["revision"]["logical_memory_id"],
            json_val["records"][0]["record"]["Semantic"]["logical_memory_id"]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn execute_history_query_returns_revision_history_json() {
        let (db, _dir) = temp_db().await;
        db.register_agent(&agent(), "Test Agent").await.unwrap();
        let bridge = Hirn {
            db: Some(Arc::new(db)),
        };
        let db = bridge.db().unwrap().clone();
        let ctx = db.as_agent(&agent()).await.unwrap();

        let id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("node_history_binding")
                    .description("initial history policy")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let original = db
            .semantic()
            .history(id)
            .await
            .unwrap()
            .into_iter()
            .next()
            .expect("initial semantic revision");

        let corrected = db
            .semantic()
            .correct(
                id,
                hirn::semantic::SemanticUpdate {
                    description: Some("updated history policy".into()),
                    reason: Some("binding regression".into()),
                    ..hirn::semantic::SemanticUpdate::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();

        let result = bridge
            .execute(
                agent().to_string(),
                format!(r#"HISTORY LOGICAL "{}""#, original.logical_memory_id),
            )
            .await
            .unwrap();

        assert_eq!(result.r#type, "history");
        assert_eq!(
            result.data["semantic_revision"]["logical_memory_id"],
            original.logical_memory_id.to_string()
        );
        assert_eq!(result.data["semantic_revision"]["revision_count"], 2);
        assert_eq!(
            result.data["semantic_revision"]["current_revision_id"],
            corrected.revision_id.to_string()
        );
        assert_eq!(result.data["items"].as_array().unwrap().len(), 2);
        assert_eq!(
            result.data["items"][1]["record"]["description"],
            "updated history policy"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_inspect_returns_semantic_revision_and_conflict_groups() {
        let (db, _dir) = temp_db().await;
        db.register_agent(&agent(), "Test Agent").await.unwrap();
        let bridge = Hirn {
            db: Some(Arc::new(db)),
        };
        let db = bridge.db().unwrap().clone();
        let ctx = db.as_agent(&agent()).await.unwrap();

        let left_id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("node_inspect_left")
                    .description("rollout is safe")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let right_id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("node_inspect_right")
                    .description("rollout is unsafe")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                left_id,
                right_id,
                hirn::prelude::EdgeRelation::Contradicts,
                0.91,
                Default::default(),
            )
            .await
            .unwrap();

        let left_head_id = db
            .semantic()
            .history(left_id)
            .await
            .unwrap()
            .into_iter()
            .last()
            .expect("connect-era left head")
            .id;

        let result = bridge
            .inspect(agent().to_string(), left_head_id.to_string())
            .await
            .unwrap();

        assert_eq!(result.r#type, "inspected");
        assert_eq!(result.data["layer"], "Semantic");
        assert!(result.data["neighbor_count"].as_u64().unwrap() >= 1);
        assert_eq!(result.data["semantic_revision"]["logical_state"], "Active");
        assert_eq!(result.data["conflict_groups"].as_array().unwrap().len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_trace_returns_semantic_revision_and_conflict_groups() {
        let (db, _dir) = temp_db().await;
        db.register_agent(&agent(), "Test Agent").await.unwrap();
        let bridge = Hirn {
            db: Some(Arc::new(db)),
        };
        let db = bridge.db().unwrap().clone();
        let ctx = db.as_agent(&agent()).await.unwrap();

        let left_id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("node_trace_left")
                    .description("rollout is safe")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let right_id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("node_trace_right")
                    .description("rollout is unsafe")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                left_id,
                right_id,
                hirn::prelude::EdgeRelation::Contradicts,
                0.91,
                Default::default(),
            )
            .await
            .unwrap();

        let left_head_id = db
            .semantic()
            .history(left_id)
            .await
            .unwrap()
            .into_iter()
            .last()
            .expect("connect-era left head")
            .id;

        let result = bridge
            .trace(agent().to_string(), left_head_id.to_string())
            .await
            .unwrap();

        assert_eq!(result.r#type, "traced");
        assert_eq!(result.data["layer"], "Semantic");
        assert!(result.data["source_episodes"].is_array());
        assert!(result.data["derived_records"].is_array());
        assert!(result.data["semantic_revision"].is_object());
        assert_eq!(result.data["semantic_revision"]["logical_state"], "Active");
        assert_eq!(result.data["conflict_groups"].as_array().unwrap().len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn execute_trace_query_returns_rich_trace_json() {
        let (db, _dir) = temp_db().await;
        db.register_agent(&agent(), "Test Agent").await.unwrap();
        let bridge = Hirn {
            db: Some(Arc::new(db)),
        };
        let db = bridge.db().unwrap().clone();
        let ctx = db.as_agent(&agent()).await.unwrap();

        let left_id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("node_execute_trace_left")
                    .description("rollout is safe")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let right_id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("node_execute_trace_right")
                    .description("rollout is unsafe")
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();

        db.graph_view()
            .connect_with(
                left_id,
                right_id,
                hirn::prelude::EdgeRelation::Contradicts,
                0.91,
                Default::default(),
            )
            .await
            .unwrap();

        let left_head_id = db
            .semantic()
            .history(left_id)
            .await
            .unwrap()
            .into_iter()
            .last()
            .expect("connect-era left head")
            .id;

        let result = bridge
            .execute(agent().to_string(), format!(r#"TRACE "{}""#, left_head_id))
            .await
            .unwrap();

        assert_eq!(result.r#type, "traced");
        assert_eq!(result.data["layer"], "Semantic");
        assert!(result.data["source_episodes"].is_array());
        assert!(result.data["derived_records"].is_array());
        assert_eq!(result.data["semantic_revision"]["logical_state"], "Active");
        assert_eq!(result.data["conflict_groups"].as_array().unwrap().len(), 1);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_recall_supports_explicit_snapshots_and_preserves_current_default() {
        let (db, _dir) = temp_db().await;
        db.register_agent(&agent(), "Test Agent").await.unwrap();
        let bridge = Hirn {
            db: Some(Arc::new(db)),
        };
        let db = bridge.db().unwrap().clone();
        let ctx = db.as_agent(&agent()).await.unwrap();

        let original_about = "lease authority";
        let current_about = "lease authority v2";
        let original_embedding = db.embed_text(original_about).await.unwrap();
        let current_embedding = db.embed_text(current_about).await.unwrap();
        let id = ctx
            .store_semantic(
                SemanticRecord::builder()
                    .concept("lease_policy")
                    .description(original_about)
                    .embedding(original_embedding.clone())
                    .agent_id(agent())
                    .build()
                    .unwrap(),
            )
            .await
            .unwrap();
        let original = db.semantic().get(id).await.unwrap();
        let logical_memory_id = original.logical_memory_id.to_string();
        let original_revision_id = original.revision_id.to_string();
        let observed_at = Timestamp::from_millis(original.created_at.millis() + 2 * 60 * 60 * 1000);

        let current_revision = db
            .semantic()
            .supersede(
                id,
                SemanticSupersession {
                    description: Some(current_about.to_owned()),
                    reason: Some("cutover".to_owned()),
                    observed_at: Some(observed_at),
                    ..SemanticSupersession::with_metadata(agent(), id)
                },
            )
            .await
            .unwrap();
        let recorded_cutoff = current_revision.created_at.to_string();

        let current = bridge
            .recall(
                agent().to_string(),
                current_embedding
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
                Some(10),
                None,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(current.len(), 1);
        assert_eq!(
            current[0].logical_memory_id.as_deref(),
            Some(logical_memory_id.as_str())
        );
        assert_eq!(current[0].revision_state.as_deref(), Some("Active"));
        assert_ne!(
            current[0].revision_id.as_deref(),
            Some(original_revision_id.as_str())
        );

        let historical = bridge
            .recall(
                agent().to_string(),
                original_embedding
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
                Some(10),
                None,
                Some(original.created_at.to_string()),
                None,
            )
            .await
            .unwrap();
        assert_eq!(historical.len(), 1);
        assert_eq!(
            historical[0].revision_id.as_deref(),
            Some(original_revision_id.as_str())
        );
        assert_eq!(historical[0].revision_state.as_deref(), Some("Active"));

        let recorded = bridge
            .recall(
                agent().to_string(),
                current_embedding
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
                Some(10),
                None,
                Some(recorded_cutoff),
                Some("recorded".to_string()),
            )
            .await
            .unwrap();
        assert_eq!(recorded.len(), 1);
        assert_ne!(
            recorded[0].revision_id.as_deref(),
            Some(original_revision_id.as_str())
        );

        let revision_snapshot = bridge
            .recall(
                agent().to_string(),
                original_embedding
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
                Some(10),
                None,
                Some(original_revision_id.clone()),
                Some("revision".to_string()),
            )
            .await
            .unwrap();
        assert_eq!(revision_snapshot.len(), 1);
        assert_eq!(
            revision_snapshot[0].revision_id.as_deref(),
            Some(original_revision_id.as_str())
        );
    }
}

/// Database statistics.
#[napi(object)]
pub struct JsStats {
    pub working_count: i64,
    pub episodic_count: i64,
    pub semantic_count: i64,
    pub total_count: i64,
    pub file_size_bytes: i64,
}

/// A single recall result.
#[napi(object)]
pub struct JsRecallResult {
    pub id: String,
    pub layer: String,
    pub similarity: f64,
    pub composite_score: f64,
    pub activation: f64,
    pub importance: f64,
    pub recency: f64,
    pub causal_relevance: f64,
    pub surprise: f64,
    pub source_reliability: f64,
    pub logical_memory_id: Option<String>,
    pub revision_id: Option<String>,
    pub revision_state: Option<String>,
}

/// Think result — assembled context for an LLM prompt.
#[napi(object)]
pub struct JsContext {
    pub context: String,
    pub token_count: i64,
    pub records_included: Vec<String>,
    pub query_time_ms: f64,
}

/// Result of a HirnQL execute / inspect / trace operation.
#[napi(object)]
pub struct JsQueryResult {
    pub r#type: String,
    /// The full result data as a JSON-compatible object.
    pub data: serde_json::Value,
}

fn recall_result_to_js(result: &hirn::query::RecallResult) -> JsRecallResult {
    let revision = result.revision.as_ref();
    JsRecallResult {
        id: result.record.id().to_string(),
        layer: format!("{:?}", result.record.layer()),
        similarity: f64::from(result.similarity),
        composite_score: f64::from(result.composite_score),
        activation: f64::from(result.score_breakdown.activation),
        importance: f64::from(result.score_breakdown.importance),
        recency: f64::from(result.score_breakdown.recency),
        causal_relevance: f64::from(result.score_breakdown.causal_relevance),
        surprise: f64::from(result.score_breakdown.surprise),
        source_reliability: f64::from(result.score_breakdown.source_reliability),
        logical_memory_id: revision.map(|r| r.logical_memory_id.to_string()),
        revision_id: revision.map(|r| r.revision_id.to_string()),
        revision_state: revision.map(|r| format!("{:?}", r.state)),
    }
}

// ─── Hirn class ──────────────────────────────────────────────

/// Internal native bridge for the Node.js bindings.
///
/// The public package root exposes the high-level `Memory` API instead.
///
/// ```js
/// const { HirnBridge } = require('./bridge');
/// const db = HirnBridge.open('path/to.hirn');
/// try {
///   db.registerAgent('agent-1', 'My Agent');
///   const id = db.remember('agent-1', 'Hello world', { embedding: new Array(768).fill(0.1) });
/// } finally {
///   db.close();
/// }
/// ```
#[napi]
pub struct Hirn {
    db: Option<Arc<hirn::HirnDB>>,
}

#[napi]
impl Hirn {
    /// Open a hirn database at the given path.
    ///
    /// @param path - File system path to the database file.
    /// @param embeddingDimensions - Dimension of embedding vectors (default: 768).
    /// @param tokenBudget - Token budget for context assembly (default: 4096).
    /// @param tokenizerName - Optional Rust tokenizer registry name.
    #[napi(factory)]
    pub fn open(
        path: String,
        embedding_dimensions: Option<u32>,
        token_budget: Option<u32>,
        tokenizer_name: Option<String>,
    ) -> napi::Result<Self> {
        let dims = embedding_dimensions.unwrap_or(768);
        let budget = token_budget.unwrap_or(4096);
        let config = HirnConfig::builder()
            .db_path(&path)
            .embedding_dimensions(dims)
            .token_budget(budget)
            .build()
            .map_err(to_napi_err)?;
        let rt = runtime_handle();
        let lance_dir = std::path::Path::new(&path).join("lance");
        let storage_config = HirnDbConfig::local(lance_dir.to_string_lossy());
        let storage: Arc<dyn hirn_storage::PhysicalStore> = rt
            .block_on(HirnDb::open(storage_config))
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, format!("storage: {e}")))?
            .store_arc();
        let mut db = rt
            .block_on(hirn::HirnDB::open_with_config(config, storage))
            .map_err(to_napi_err)?;
        if let Some(tokenizer_name) = tokenizer_name.as_deref() {
            db.set_tokenizer(resolve_registry_tokenizer(tokenizer_name)?);
        }
        Ok(Self {
            db: Some(Arc::new(db)),
        })
    }

    /// Close the database. Should be called when done.
    #[napi]
    pub fn close(&mut self) -> napi::Result<()> {
        self.db = None;
        Ok(())
    }

    fn db(&self) -> napi::Result<&Arc<hirn::HirnDB>> {
        self.db
            .as_ref()
            .ok_or_else(|| napi::Error::new(napi::Status::GenericFailure, "database is closed"))
    }

    /// Register an agent.
    ///
    /// @param agentId - Unique agent identifier.
    /// @param displayName - Human-readable name for the agent.
    #[napi]
    pub async fn register_agent(&self, agent_id: String, display_name: String) -> napi::Result<()> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        db.register_agent(&aid, &display_name)
            .await
            .map_err(to_napi_err)
    }

    /// Store an episodic memory.
    ///
    /// @param agentId - The agent who owns this memory.
    /// @param content - Text content of the memory.
    /// @param options - Optional: { embedding?: number[], importance?: number }
    /// @returns The ULID string of the new memory.
    #[napi]
    pub async fn remember(
        &self,
        agent_id: String,
        content: String,
        embedding: Option<Vec<f64>>,
        importance: Option<f64>,
    ) -> napi::Result<String> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let imp = importance.unwrap_or(0.5) as f32;

        let mut builder = EpisodicRecord::builder()
            .agent_id(aid.clone())
            .content(&content)
            .importance(imp);

        if let Some(emb) = &embedding {
            builder = builder.embedding(to_f32_vec(emb)?);
        }

        let record = builder.build().map_err(to_napi_err)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let id = ctx.remember(record).await.map_err(to_napi_err)?;
        Ok(id.to_string())
    }

    /// Recall memories by vector similarity.
    ///
    /// @param agentId - The agent performing the recall.
    /// @param query - Query embedding vector.
    /// @param limit - Maximum number of results (default: 10).
    /// @param threshold - Minimum similarity threshold.
    /// @param asOf - Optional snapshot value (`YYYY-MM-DD`, RFC 3339, or revision ULID).
    /// @param snapshotKind - Optional snapshot kind: `observed`, `recorded`, or `revision`.
    /// @returns Array of recall results.
    #[napi]
    pub async fn recall(
        &self,
        agent_id: String,
        query: Vec<f64>,
        limit: Option<u32>,
        threshold: Option<f64>,
        as_of: Option<String>,
        snapshot_kind: Option<String>,
    ) -> napi::Result<Vec<JsRecallResult>> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let emb = to_f32_vec(&query)?;
        let snapshot = parse_optional_recall_snapshot(as_of.as_deref(), snapshot_kind.as_deref())?;

        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let mut builder = ctx.recall(emb).limit(limit.unwrap_or(10) as usize);
        if let Some(t) = threshold {
            builder = builder.threshold(t as f32);
        }
        if let Some(snapshot) = snapshot {
            builder = builder.snapshot(snapshot);
        }

        let results = builder.execute().await.map_err(to_napi_err)?;
        Ok(results.iter().map(recall_result_to_js).collect())
    }

    /// Assemble context for an LLM prompt.
    ///
    /// @param agentId - The agent performing the think.
    /// @param query - Query embedding vector.
    /// @param budget - Token budget (default: 4096).
    /// @returns Context object with the assembled context string.
    #[napi]
    pub async fn think(
        &self,
        agent_id: String,
        query: Vec<f64>,
        budget: Option<u32>,
    ) -> napi::Result<JsContext> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let emb = to_f32_vec(&query)?;

        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let result = ctx
            .think(emb)
            .budget(budget.unwrap_or(4096) as usize)
            .execute()
            .await
            .map_err(to_napi_err)?;

        Ok(JsContext {
            context: result.context,
            token_count: i64::try_from(result.token_count).unwrap_or(i64::MAX),
            records_included: result
                .records_included
                .iter()
                .map(|id| id.to_string())
                .collect(),
            query_time_ms: result.query_time_ms,
        })
    }

    /// Forget (archive) a memory by its ULID string ID.
    ///
    /// @param agentId - The agent performing the forget.
    /// @param id - ULID string of the memory to forget.
    #[napi]
    pub async fn forget(&self, agent_id: String, id: String) -> napi::Result<()> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mid = parse_memory_id(&id)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        ctx.archive_episode(mid).await.map_err(to_napi_err)
    }

    /// Execute a HirnQL query.
    ///
    /// @param agentId - The agent executing the query.
    /// @param query - HirnQL query string.
    /// @returns QueryResult with the result type and data.
    #[napi]
    pub async fn execute(&self, agent_id: String, query: String) -> napi::Result<JsQueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let result = ctx.execute_ql(&query).await.map_err(to_napi_err)?;

        let json_val = query_result_to_json(&result);
        let result_type = json_val["type"].as_str().unwrap_or("unknown").to_string();
        Ok(JsQueryResult {
            r#type: result_type,
            data: json_val,
        })
    }

    /// Inspect a memory record.
    ///
    /// @param agentId - The agent inspecting the record.
    /// @param id - ULID string of the memory to inspect.
    /// @returns QueryResult with inspection details.
    #[napi]
    pub async fn inspect(&self, agent_id: String, id: String) -> napi::Result<JsQueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mid = parse_memory_id(&id)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let result = ctx.inspect(mid).await.map_err(to_napi_err)?;

        let json_val = inspected_result_to_json(&result);
        let result_type = json_val["type"].as_str().unwrap_or("unknown").to_string();
        Ok(JsQueryResult {
            r#type: result_type,
            data: json_val,
        })
    }

    /// Trace the provenance of a memory record.
    ///
    /// @param agentId - The agent tracing the record.
    /// @param id - ULID string of the memory to trace.
    /// @returns QueryResult with trace/provenance details.
    #[napi]
    pub async fn trace(&self, agent_id: String, id: String) -> napi::Result<JsQueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mid = parse_memory_id(&id)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let result = ctx.trace(mid).await.map_err(to_napi_err)?;

        let json_val = trace_result_to_json(&result);
        Ok(JsQueryResult {
            r#type: "traced".to_string(),
            data: json_val,
        })
    }

    /// Get database statistics.
    ///
    /// @returns Stats object with record counts and file size.
    #[napi]
    pub fn stats(&self) -> napi::Result<JsStats> {
        let db = self.db()?;
        let rt = runtime_handle();
        let s = rt.block_on(db.admin().stats()).map_err(to_napi_err)?;
        Ok(JsStats {
            // F-73: Safe u64→i64 conversion (saturate at i64::MAX).
            working_count: i64::try_from(s.working_count).unwrap_or(i64::MAX),
            episodic_count: i64::try_from(s.episodic_count).unwrap_or(i64::MAX),
            semantic_count: i64::try_from(s.semantic_count).unwrap_or(i64::MAX),
            total_count: i64::try_from(s.total_count).unwrap_or(i64::MAX),
            file_size_bytes: i64::try_from(s.file_size_bytes).unwrap_or(i64::MAX),
        })
    }

    // ── F-66: Additional memory layer APIs ─────────────────────────

    /// Store a semantic record (concept / fact).
    ///
    /// @param agentId - The agent who owns this record.
    /// @param concept - Concept name.
    /// @param description - Textual description.
    /// @param embedding - Optional embedding vector.
    /// @param confidence - Confidence score 0.0–1.0 (default: 0.5).
    /// @returns The ULID string of the new semantic record.
    #[napi]
    pub async fn store_semantic(
        &self,
        agent_id: String,
        concept: String,
        description: String,
        embedding: Option<Vec<f64>>,
        confidence: Option<f64>,
    ) -> napi::Result<String> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mut builder = SemanticRecord::builder()
            .agent_id(aid)
            .namespace(Namespace::private_for(&aid))
            .concept(&concept)
            .description(&description)
            .confidence(confidence.unwrap_or(0.5) as f32);
        if let Some(emb) = &embedding {
            builder = builder.embedding(to_f32_vec(emb)?);
        }
        let record = builder.build().map_err(to_napi_err)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let id = ctx.store_semantic(record).await.map_err(to_napi_err)?;
        Ok(id.to_string())
    }

    /// Apply a semantic correction revision directly through the semantic view API.
    #[napi]
    pub async fn correct_semantic(
        &self,
        agent_id: String,
        id: String,
        description: Option<String>,
        confidence: Option<f64>,
        evidence_count: Option<i64>,
        reason: Option<String>,
        observed_at: Option<String>,
        caused_by: Option<String>,
    ) -> napi::Result<JsQueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mid = parse_memory_id(&id)?;
        let causation_id = caused_by
            .as_deref()
            .map(parse_memory_id)
            .transpose()?
            .unwrap_or(mid);
        let observed_at = parse_optional_observed_at(observed_at.as_deref())?;
        let evidence_count = parse_optional_evidence_count(evidence_count)?;

        let prior = db.semantic().get(mid).await.map_err(to_napi_err)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let corrected = ctx
            .correct_semantic(
                mid,
                hirn::semantic::SemanticUpdate {
                    description,
                    confidence: confidence.map(|v| v as f32),
                    evidence_count,
                    reason,
                    actor_id: aid,
                    observed_at,
                    causation_id,
                },
            )
            .await
            .map_err(to_napi_err)?;

        let data = serde_json::json!({
            "type": "corrected",
            "logical_memory_id": corrected.logical_memory_id.to_string(),
            "prior_revision_id": prior.revision_id.to_string(),
            "new_revision_id": corrected.revision_id.to_string(),
        });

        Ok(JsQueryResult {
            r#type: "corrected".to_string(),
            data,
        })
    }

    /// Apply a semantic supersession revision directly through the semantic view API.
    #[napi]
    pub async fn supersede_semantic(
        &self,
        agent_id: String,
        id: String,
        description: Option<String>,
        confidence: Option<f64>,
        evidence_count: Option<i64>,
        reason: Option<String>,
        observed_at: Option<String>,
        caused_by: Option<String>,
    ) -> napi::Result<JsQueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mid = parse_memory_id(&id)?;
        let causation_id = caused_by
            .as_deref()
            .map(parse_memory_id)
            .transpose()?
            .unwrap_or(mid);
        let observed_at = parse_optional_observed_at(observed_at.as_deref())?;
        let evidence_count = parse_optional_evidence_count(evidence_count)?;

        let prior = db.semantic().get(mid).await.map_err(to_napi_err)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let superseded = ctx
            .supersede_semantic(
                mid,
                hirn::semantic::SemanticSupersession {
                    description,
                    confidence: confidence.map(|v| v as f32),
                    evidence_count,
                    reason: reason.clone(),
                    actor_id: aid,
                    observed_at,
                    causation_id,
                },
            )
            .await
            .map_err(to_napi_err)?;

        let data = serde_json::json!({
            "type": "superseded",
            "logical_memory_id": superseded.logical_memory_id.to_string(),
            "prior_revision_id": prior.revision_id.to_string(),
            "new_revision_id": superseded.revision_id.to_string(),
            "reason": reason,
        });

        Ok(JsQueryResult {
            r#type: "superseded".to_string(),
            data,
        })
    }

    /// Apply a semantic retraction revision directly through the semantic view API.
    #[napi]
    pub async fn retract_semantic(
        &self,
        agent_id: String,
        id: String,
        reason: Option<String>,
        observed_at: Option<String>,
        caused_by: Option<String>,
    ) -> napi::Result<JsQueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mid = parse_memory_id(&id)?;
        let causation_id = caused_by
            .as_deref()
            .map(parse_memory_id)
            .transpose()?
            .unwrap_or(mid);
        let observed_at = parse_optional_observed_at(observed_at.as_deref())?;

        let prior = db.semantic().get(mid).await.map_err(to_napi_err)?;
        let ctx = db.as_agent(&aid).await.map_err(to_napi_err)?;
        let retracted = ctx
            .retract_semantic(
                mid,
                hirn::semantic::SemanticRetraction {
                    reason: reason.clone(),
                    actor_id: aid,
                    observed_at,
                    causation_id,
                },
            )
            .await
            .map_err(to_napi_err)?;

        let data = serde_json::json!({
            "type": "retracted",
            "logical_memory_id": retracted.logical_memory_id.to_string(),
            "prior_revision_id": prior.revision_id.to_string(),
            "tombstone_revision_id": retracted.revision_id.to_string(),
            "reason": reason,
        });

        Ok(JsQueryResult {
            r#type: "retracted".to_string(),
            data,
        })
    }

    /// Store a procedural record (skill / action sequence).
    ///
    /// @param agentId - The agent who owns this record.
    /// @param name - Short skill name.
    /// @param description - Textual description.
    /// @param embedding - Optional embedding vector.
    /// @returns The ULID string of the new procedural record.
    #[napi]
    pub async fn store_procedural(
        &self,
        agent_id: String,
        name: String,
        description: String,
        embedding: Option<Vec<f64>>,
    ) -> napi::Result<String> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let mut builder = ProceduralRecord::builder()
            .agent_id(aid)
            .name(&name)
            .description(&description);
        if let Some(emb) = &embedding {
            builder = builder.embedding(to_f32_vec(emb)?);
        }
        let record = builder.build().map_err(to_napi_err)?;
        let id = db.procedural().store(record).await.map_err(to_napi_err)?;
        Ok(id.to_string())
    }

    /// Add an entry to working memory.
    ///
    /// @param agentId - The agent who owns the entry.
    /// @param content - Text content for the working memory slot.
    /// @param tokenCount - Optional client-side estimate. The Rust tokenizer
    /// remains authoritative and revalidates the effective token count.
    /// @returns The ULID string of the new working memory entry.
    #[napi]
    pub async fn focus(
        &self,
        agent_id: String,
        content: String,
        token_count: Option<u32>,
    ) -> napi::Result<String> {
        let db = self.db()?;
        let aid = parse_agent_id(&agent_id)?;
        let effective_token_count = authoritative_working_token_count(db, &content, token_count);
        let entry = WorkingMemoryEntry::builder()
            .agent_id(aid)
            .content(&content)
            .token_count(effective_token_count)
            .build()
            .map_err(to_napi_err)?;
        let id = db.working().focus(entry).await.map_err(to_napi_err)?;
        Ok(id.to_string())
    }

    /// Remove an entry from working memory.
    ///
    /// @param id - ULID string of the working memory entry.
    #[napi]
    pub async fn defocus(&self, id: String) -> napi::Result<()> {
        let db = self.db()?;
        let mid = parse_memory_id(&id)?;
        db.working().defocus(mid).await.map_err(to_napi_err)
    }

    /// Run the consolidation pipeline.
    ///
    /// @returns Number of records processed.
    #[napi]
    pub async fn consolidate(&self) -> napi::Result<i64> {
        let db = self.db()?;
        let report = db
            .admin()
            .consolidate()
            .execute()
            .await
            .map_err(to_napi_err)?;
        Ok(i64::try_from(report.records_processed).unwrap_or(i64::MAX))
    }

    /// Connect two memories with a graph edge.
    ///
    /// @param source - ULID string of the source memory.
    /// @param target - ULID string of the target memory.
    #[napi]
    pub async fn connect(&self, source: String, target: String) -> napi::Result<()> {
        let db = self.db()?;
        let src = parse_memory_id(&source)?;
        let tgt = parse_memory_id(&target)?;
        db.graph_view()
            .connect(src, tgt)
            .await
            .map_err(to_napi_err)?;
        Ok(())
    }

    /// Subscribe to memory events (create, archive, consolidate).
    ///
    /// @param filterLayer - Optional layer filter: "Episodic", "Semantic", "Working".
    /// @returns A WatchStream whose `next()` method yields events.
    #[napi]
    pub async fn watch(&self, filter_layer: Option<String>) -> napi::Result<WatchStream> {
        let db = self.db()?;
        let mut broadcast_rx = db.subscribe();
        let (async_tx, async_rx) = tokio::sync::mpsc::channel(4096);

        // Bridge the lock-free broadcast receiver into a bounded mpsc channel
        // that WatchStream::next() can consume.  Using tokio::spawn avoids
        // blocking any Tokio worker thread.  On broadcast lag the subscriber
        // emits a warning and continues; on mpsc backpressure (consumer too
        // slow) the task exits cleanly.
        tokio::spawn(async move {
            loop {
                match broadcast_rx.recv().await {
                    Ok(event) => {
                        if async_tx.send(event).await.is_err() {
                            break; // WatchStream dropped
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!(
                            skipped = n,
                            "hirn-watch-bridge: broadcast lagged, events dropped"
                        );
                        // Continue — stay subscribed rather than disconnecting.
                    }
                }
            }
        });

        Ok(WatchStream {
            rx: Arc::new(parking_lot::Mutex::new(Some(async_rx))),
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            filter_layer,
        })
    }
}
