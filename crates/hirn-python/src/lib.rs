//! Python bindings for the hirn cognitive memory database.
//!
//! This module exposes hirn's core API to Python via PyO3.
//! The native module is named `hirn._hirn` and re-exported from
//! the `hirn` Python package.

use std::sync::Arc;

use numpy::PyArrayMethods;
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use hirn::prelude::*;
use hirn::{
    ProviderRegistry, Tokenizer, inspected_result_to_json, trace_result_to_json,
    traced_result_to_json,
};
use hirn_storage::{HirnDb, HirnDbConfig};

// ─── Error Mapping ───────────────────────────────────────────

pyo3::create_exception!(hirn, HirnError, pyo3::exceptions::PyException);
pyo3::create_exception!(hirn, NotFoundError, HirnError);
pyo3::create_exception!(hirn, QueryError, HirnError);

fn to_py_err(e: hirn::HirnError) -> PyErr {
    match &e {
        hirn::HirnError::NotFound(_) => NotFoundError::new_err(e.to_string()),
        hirn::HirnError::InvalidInput(_) => QueryError::new_err(e.to_string()),
        _ => HirnError::new_err(e.to_string()),
    }
}

// ─── Helper: extract embedding from Python ───────────────────

/// Accept a Python list of floats or a numpy array and return a `Vec<f32>`.
fn extract_embedding(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Keep Python-list inputs dependency-free even when numpy is unavailable.
    if obj.is_instance_of::<PyList>() {
        let list: Vec<f32> = obj.extract()?;
        if list.is_empty() {
            return Err(PyErr::new::<QueryError, _>("embedding must not be empty"));
        }
        return Ok(list);
    }

    let object_type = obj.get_type();
    let type_module = object_type
        .getattr("__module__")
        .ok()
        .and_then(|value| value.extract::<String>().ok());

    // Try numpy f32 array first (zero-copy).
    if type_module.as_deref() == Some("numpy") {
        if let Ok(arr) = obj.cast::<numpy::PyArray1<f32>>() {
            let readonly = arr.try_readonly()?;
            return Ok(readonly.as_slice()?.to_vec());
        }
        // F-71: Try numpy f64 array (numpy's default dtype) with safe conversion.
        if let Ok(arr) = obj.cast::<numpy::PyArray1<f64>>() {
            let readonly = arr.try_readonly()?;
            let slice = readonly.as_slice()?;
            let mut out = Vec::with_capacity(slice.len());
            for (i, &x) in slice.iter().enumerate() {
                if !x.is_finite() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "embedding[{i}] is not finite: {x}"
                    )));
                }
                let y = x as f32;
                if !y.is_finite() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "embedding[{i}] overflows f32 range: {x}"
                    )));
                }
                out.push(y);
            }
            return Ok(out);
        }
    }
    // Fall back to Python sequence
    let list: Vec<f32> = obj.extract()?;
    if list.is_empty() {
        return Err(PyErr::new::<QueryError, _>("embedding must not be empty"));
    }
    let _ = py; // used for numpy context
    Ok(list)
}

/// Parse a ULID string into a `MemoryId`.
///
/// Raises `ValueError` (not `QueryError`) because an invalid identifier is a
/// caller error, not a query execution failure (N-L10).
fn parse_memory_id(s: &str) -> PyResult<MemoryId> {
    ulid::Ulid::from_string(s)
        .map(MemoryId::from_ulid)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid memory id: {e}"))
        })
}

/// Parse an agent ID string.
///
/// Raises `ValueError` for invalid identifiers (N-L10).
fn parse_agent_id(s: &str) -> PyResult<AgentId> {
    AgentId::new(s).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid agent_id: {e}"))
    })
}

fn parse_optional_recall_snapshot(
    value: Option<&str>,
    snapshot_kind: Option<&str>,
) -> PyResult<Option<RecallSnapshot>> {
    let Some(value) = value else {
        return match snapshot_kind {
            Some(kind) => Err(QueryError::new_err(format!(
                "snapshot_kind '{kind}' requires an as_of value"
            ))),
            None => Ok(None),
        };
    };

    let snapshot_kind = snapshot_kind.unwrap_or("observed").to_ascii_lowercase();
    match snapshot_kind.as_str() {
        "observed" => Timestamp::parse_date_or_rfc3339(value)
            .map(RecallSnapshot::observed)
            .ok_or_else(|| {
                QueryError::new_err(format!(
                    "invalid observed timestamp '{value}' (expected YYYY-MM-DD or RFC 3339)"
                ))
            })
            .map(Some),
        "recorded" => Timestamp::parse_date_or_rfc3339(value)
            .map(RecallSnapshot::recorded)
            .ok_or_else(|| {
                QueryError::new_err(format!(
                    "invalid recorded timestamp '{value}' (expected YYYY-MM-DD or RFC 3339)"
                ))
            })
            .map(Some),
        "revision" => RevisionId::parse(value)
            .map(RecallSnapshot::revision)
            .map(Some)
            .map_err(|error| {
                // Invalid revision IDs are caller errors — raise ValueError (N-L10).
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "invalid revision id '{value}': {error}"
                ))
            }),
        other => Err(QueryError::new_err(format!(
            "invalid snapshot_kind '{other}' (expected observed, recorded, or revision)"
        ))),
    }
}

/// Create a LanceDB storage backend from a brain directory path.
async fn open_lance_storage(
    path: &str,
) -> Result<Arc<dyn hirn_storage::PhysicalStore>, hirn_storage::HirnDbError> {
    let lance_dir = std::path::Path::new(path).join("lance");
    let config = HirnDbConfig::local(lance_dir.to_string_lossy());
    let hirn_storage = HirnDb::open(config).await?;
    Ok(hirn_storage.store_arc())
}

/// Block on a future using the pyo3 async runtime (for sync Python API).
fn block_on<F: std::future::Future>(f: F) -> F::Output {
    pyo3_async_runtimes::tokio::get_runtime().block_on(f)
}

fn resolve_registry_tokenizer(tokenizer_name: &str) -> PyResult<Arc<dyn Tokenizer>> {
    let registry = ProviderRegistry::from_env();
    registry
        .tokenizer_by_name(tokenizer_name)
        .ok_or_else(|| QueryError::new_err(format!("unknown Rust tokenizer '{tokenizer_name}'")))
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

fn json_to_pyobj(py: Python<'_>, val: &serde_json::Value) -> PyResult<Py<PyAny>> {
    json_to_pyobj_inner(py, val, 0)
}

/// Maximum nesting depth for JSON→Python conversion (prevents stack overflow).
const JSON_MAX_DEPTH: usize = 64;

fn json_to_pyobj_inner(
    py: Python<'_>,
    val: &serde_json::Value,
    depth: usize,
) -> PyResult<Py<PyAny>> {
    if depth > JSON_MAX_DEPTH {
        return Err(PyErr::new::<PyRuntimeError, _>(
            "JSON nesting depth exceeds maximum (64)",
        ));
    }
    match val {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_pyobj_inner(py, item, depth + 1)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_pyobj_inner(py, v, depth + 1)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

// ─── Python Types ────────────────────────────────────────────

/// Database statistics.
#[pyclass(from_py_object)]
#[derive(Clone)]

struct Stats {
    #[pyo3(get)]
    working_count: u64,
    #[pyo3(get)]
    episodic_count: u64,
    #[pyo3(get)]
    semantic_count: u64,
    #[pyo3(get)]
    total_count: u64,
    #[pyo3(get)]
    file_size_bytes: u64,
}

#[pymethods]
impl Stats {
    fn __repr__(&self) -> String {
        format!(
            "Stats(total={}, working={}, episodic={}, semantic={}, size={})",
            self.total_count,
            self.working_count,
            self.episodic_count,
            self.semantic_count,
            self.file_size_bytes,
        )
    }
}

/// A memory record returned from inspect/trace.
#[pyclass(from_py_object)]
#[derive(Clone)]

struct MemoryRecord {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    layer: String,
    #[pyo3(get)]
    content: String,
    json_val: serde_json::Value,
}

#[pymethods]
impl MemoryRecord {
    #[getter]
    fn json(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_pyobj(py, &self.json_val)
    }

    fn __repr__(&self) -> String {
        format!("MemoryRecord(id='{}', layer='{}')", self.id, self.layer)
    }
}

/// A single recall result.
#[pyclass(from_py_object)]
#[derive(Clone)]
struct RecallResult {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    layer: String,
    #[pyo3(get)]
    similarity: f32,
    #[pyo3(get)]
    composite_score: f32,
    #[pyo3(get)]
    activation: f32,
    #[pyo3(get)]
    importance: f32,
    #[pyo3(get)]
    recency: f32,
    #[pyo3(get)]
    causal_relevance: f32,
    #[pyo3(get)]
    surprise: f32,
    #[pyo3(get)]
    source_reliability: f32,
    #[pyo3(get)]
    logical_memory_id: Option<String>,
    #[pyo3(get)]
    revision_id: Option<String>,
    #[pyo3(get)]
    revision_state: Option<String>,
}

#[pymethods]
impl RecallResult {
    fn __repr__(&self) -> String {
        format!(
            "RecallResult(id='{}', similarity={:.4}, score={:.4})",
            self.id, self.similarity, self.composite_score,
        )
    }
}

fn recall_result_from_runtime(result: &hirn::query::RecallResult) -> RecallResult {
    let revision = result.revision.as_ref();
    RecallResult {
        id: result.record.id().to_string(),
        layer: format!("{:?}", result.record.layer()),
        similarity: result.similarity,
        composite_score: result.composite_score,
        activation: result.score_breakdown.activation,
        importance: result.score_breakdown.importance,
        recency: result.score_breakdown.recency,
        causal_relevance: result.score_breakdown.causal_relevance,
        surprise: result.score_breakdown.surprise,
        source_reliability: result.score_breakdown.source_reliability,
        logical_memory_id: revision.map(|r| r.logical_memory_id.to_string()),
        revision_id: revision.map(|r| r.revision_id.to_string()),
        revision_state: revision.map(|r| format!("{:?}", r.state)),
    }
}

/// Think result — assembled context for an LLM prompt.
#[pyclass(from_py_object)]
#[derive(Clone)]

struct Context {
    #[pyo3(get)]
    context: String,
    #[pyo3(get)]
    token_count: usize,
    records_included_ids: Vec<String>,
    #[pyo3(get)]
    query_time_ms: f64,
}

#[pymethods]
impl Context {
    #[getter]
    fn records_included(&self) -> Vec<String> {
        self.records_included_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Context(tokens={}, records={}, time={:.2}ms)",
            self.token_count,
            self.records_included_ids.len(),
            self.query_time_ms,
        )
    }
}

/// Result of a HirnQL execute operation.
#[pyclass(from_py_object)]
#[derive(Clone)]

struct QueryResult {
    result_type: String,
    json_val: serde_json::Value,
}

#[pymethods]
impl QueryResult {
    #[getter]
    fn r#type(&self) -> &str {
        &self.result_type
    }

    #[getter]
    fn json(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        json_to_pyobj(py, &self.json_val)
    }

    fn __repr__(&self) -> String {
        format!("QueryResult(type='{}')", self.result_type)
    }
}

// ─── HirnBridge class ────────────────────────────────────────

/// Internal native bridge for the Python bindings.
///
/// The public package root exposes the high-level ``Memory`` API instead.
///
/// Use as a context manager::
///
///     from hirn._hirn import HirnBridge
///
///     with HirnBridge.open("path/to.hirn") as h:
///         h.remember("agent", "content", embedding=[0.1] * 64)
#[pyclass(name = "HirnBridge")]
struct Hirn {
    db: Option<Arc<hirn::HirnDB>>,
}

impl Hirn {
    fn db(&self) -> PyResult<&Arc<hirn::HirnDB>> {
        self.db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))
    }
}

#[pymethods]
impl Hirn {
    /// Open a hirn database at the given path.
    ///
    /// Args:
    ///     path: File system path to the brain directory.
    ///     embedding_dimensions: Dimension of embedding vectors (default: 768).
    ///     token_budget: Token budget for context assembly (default: 4096).
    ///     tokenizer_name: Optional Rust tokenizer registry name. This selects
    ///         an already-registered Rust tokenizer; Python does not become the
    ///         authoritative tokenizer for engine budgeting.
    #[staticmethod]
    #[pyo3(signature = (path, *, embedding_dimensions=768, token_budget=4096, tokenizer_name=None))]
    fn open(
        path: &str,
        embedding_dimensions: u32,
        token_budget: u32,
        tokenizer_name: Option<&str>,
    ) -> PyResult<Self> {
        let config = HirnConfig::builder()
            .db_path(path)
            .embedding_dimensions(embedding_dimensions)
            .token_budget(token_budget)
            .build()
            .map_err(to_py_err)?;
        let storage = block_on(open_lance_storage(path))
            .map_err(|e| PyRuntimeError::new_err(format!("storage: {e}")))?;
        let db = block_on(hirn::HirnDB::open_with_config(config, storage)).map_err(to_py_err)?;
        if let Some(tokenizer_name) = tokenizer_name {
            db.set_tokenizer(resolve_registry_tokenizer(tokenizer_name)?);
        }
        Ok(Self {
            db: Some(Arc::new(db)),
        })
    }

    /// Close the database. Called automatically when used as a context manager.
    fn close(&mut self) -> PyResult<()> {
        self.db = None;
        Ok(())
    }

    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close()?;
        Ok(false)
    }

    /// Register an agent.
    ///
    /// Args:
    ///     agent_id: Unique agent identifier.
    ///     display_name: Human-readable name for the agent.
    fn register_agent(&self, agent_id: &str, display_name: &str) -> PyResult<()> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        block_on(db.register_agent(&aid, display_name)).map_err(to_py_err)
    }

    /// Store an episodic memory.
    ///
    /// Args:
    ///     agent_id: The agent who owns this memory.
    ///     content: Text content of the memory.
    ///     embedding: Optional embedding vector (list or numpy array).
    ///     importance: Importance score 0.0–1.0 (default: 0.5).
    ///
    /// Returns:
    ///     The ULID string of the new memory.
    #[pyo3(signature = (agent_id, content, *, embedding=None, importance=0.5))]
    fn remember(
        &self,
        py: Python<'_>,
        agent_id: &str,
        content: &str,
        embedding: Option<&Bound<'_, PyAny>>,
        importance: f32,
    ) -> PyResult<String> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;

        let mut builder = EpisodicRecord::builder()
            .agent_id(aid.clone())
            .content(content)
            .importance(importance);

        if let Some(emb_obj) = embedding {
            let emb = extract_embedding(py, emb_obj)?;
            builder = builder.embedding(emb);
        }

        let record = builder.build().map_err(to_py_err)?;
        let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
        let id = block_on(ctx.remember(record)).map_err(to_py_err)?;
        Ok(id.to_string())
    }

    /// Recall memories by vector similarity.
    ///
    /// Args:
    ///     agent_id: The agent performing the recall.
    ///     query: Query embedding vector (list or numpy array).
    ///     limit: Maximum number of results (default: 10).
    ///     threshold: Minimum similarity threshold (default: None).
    ///     as_of: Optional snapshot value (`YYYY-MM-DD`, RFC 3339, or revision ULID).
    ///     snapshot_kind: Optional snapshot kind: `observed`, `recorded`, or `revision`.
    ///
    /// Returns:
    ///     List of RecallResult objects.
    #[pyo3(signature = (agent_id, query, *, limit=10, threshold=None, as_of=None, snapshot_kind=None))]
    fn recall(
        &self,
        py: Python<'_>,
        agent_id: &str,
        query: &Bound<'_, PyAny>,
        limit: usize,
        threshold: Option<f32>,
        as_of: Option<&str>,
        snapshot_kind: Option<&str>,
    ) -> PyResult<Vec<RecallResult>> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let emb = extract_embedding(py, query)?;
        let snapshot = parse_optional_recall_snapshot(as_of, snapshot_kind)?;

        let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
        let mut builder = ctx.recall(emb).limit(limit);
        if let Some(t) = threshold {
            builder = builder.threshold(t);
        }
        if let Some(snapshot) = snapshot {
            builder = builder.snapshot(snapshot);
        }

        let results = block_on(builder.execute()).map_err(to_py_err)?;
        Ok(results.iter().map(recall_result_from_runtime).collect())
    }

    /// Assemble context for an LLM prompt.
    ///
    /// Args:
    ///     agent_id: The agent performing the think.
    ///     query: Query embedding vector (list or numpy array).
    ///     budget: Token budget (default: 4096).
    ///
    /// Returns:
    ///     Context object with the assembled context string.
    #[pyo3(signature = (agent_id, query, *, budget=4096))]
    fn think(
        &self,
        py: Python<'_>,
        agent_id: &str,
        query: &Bound<'_, PyAny>,
        budget: usize,
    ) -> PyResult<Context> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let emb = extract_embedding(py, query)?;

        let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
        let result = block_on(ctx.think(emb).budget(budget).execute()).map_err(to_py_err)?;

        Ok(Context {
            context: result.context,
            token_count: result.token_count,
            records_included_ids: result
                .records_included
                .iter()
                .map(|id| id.to_string())
                .collect(),
            query_time_ms: result.query_time_ms,
        })
    }

    /// Forget (archive) a memory by its ULID string ID.
    ///
    /// Args:
    ///     agent_id: The agent performing the forget.
    ///     id: ULID string of the memory to forget.
    fn forget(&self, agent_id: &str, id: &str) -> PyResult<()> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let mid = parse_memory_id(id)?;
        let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
        block_on(ctx.archive_episode(mid)).map_err(to_py_err)
    }

    /// Execute a HirnQL query.
    ///
    /// Args:
    ///     agent_id: The agent executing the query.
    ///     query: HirnQL query string.
    ///
    /// Returns:
    ///     QueryResult with the result as a JSON-accessible dict.
    fn execute(&self, agent_id: &str, query: &str) -> PyResult<QueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
        let result = block_on(ctx.execute_ql(query)).map_err(to_py_err)?;

        let json_val = query_result_to_json(&result);
        let result_type = json_val["type"].as_str().unwrap_or("unknown").to_string();
        Ok(QueryResult {
            result_type,
            json_val,
        })
    }

    /// Inspect a memory record.
    ///
    /// Args:
    ///     agent_id: The agent inspecting the record.
    ///     id: ULID string of the memory to inspect.
    ///
    /// Returns:
    ///     QueryResult with inspection details.
    fn inspect(&self, agent_id: &str, id: &str) -> PyResult<QueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let mid = parse_memory_id(id)?;
        let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
        let result = block_on(ctx.inspect(mid)).map_err(to_py_err)?;

        let json_val = inspected_result_to_json(&result);
        let result_type = json_val["type"].as_str().unwrap_or("unknown").to_string();
        Ok(QueryResult {
            result_type,
            json_val,
        })
    }

    /// Trace the provenance of a memory record.
    ///
    /// Args:
    ///     agent_id: The agent tracing the record.
    ///     id: ULID string of the memory to trace.
    ///
    /// Returns:
    ///     QueryResult with trace/provenance details.
    fn trace(&self, agent_id: &str, id: &str) -> PyResult<QueryResult> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let mid = parse_memory_id(id)?;
        let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
        let result = block_on(ctx.trace(mid)).map_err(to_py_err)?;

        let json_val = trace_result_to_json(&result);
        Ok(QueryResult {
            result_type: "traced".to_string(),
            json_val,
        })
    }

    /// Get database statistics.
    ///
    /// Returns:
    ///     Stats object with record counts and file size.
    fn stats(&self) -> PyResult<Stats> {
        let db = self.db()?;
        let s = block_on(db.admin().stats()).map_err(to_py_err)?;
        Ok(Stats {
            working_count: s.working_count,
            episodic_count: s.episodic_count,
            semantic_count: s.semantic_count,
            total_count: s.total_count,
            file_size_bytes: s.file_size_bytes,
        })
    }

    // ── F-66: Additional memory layer APIs ─────────────────────────

    /// Store a semantic record.
    ///
    /// Args:
    ///     agent_id: The agent who owns this record.
    ///     concept: Concept name (e.g. "photosynthesis").
    ///     description: Textual description of the concept.
    ///     embedding: Optional embedding vector (list or numpy array).
    ///     confidence: Confidence score 0.0–1.0 (default: 0.5).
    ///
    /// Returns:
    ///     The ULID string of the new semantic record.
    #[pyo3(signature = (agent_id, concept, description, *, embedding=None, confidence=0.5))]
    fn store_semantic(
        &self,
        py: Python<'_>,
        agent_id: &str,
        concept: &str,
        description: &str,
        embedding: Option<&Bound<'_, PyAny>>,
        confidence: f32,
    ) -> PyResult<String> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let mut builder = SemanticRecord::builder()
            .agent_id(aid)
            .concept(concept)
            .description(description)
            .confidence(confidence);
        if let Some(emb_obj) = embedding {
            builder = builder.embedding(extract_embedding(py, emb_obj)?);
        }
        let record = builder.build().map_err(to_py_err)?;
        let id = block_on(db.semantic().store(record)).map_err(to_py_err)?;
        Ok(id.to_string())
    }

    /// Store a procedural record (skill / action sequence).
    ///
    /// Args:
    ///     agent_id: The agent who owns this record.
    ///     name: Short skill name.
    ///     description: Textual description of the procedure.
    ///     embedding: Optional embedding vector (list or numpy array).
    ///
    /// Returns:
    ///     The ULID string of the new procedural record.
    #[pyo3(signature = (agent_id, name, description, *, embedding=None))]
    fn store_procedural(
        &self,
        py: Python<'_>,
        agent_id: &str,
        name: &str,
        description: &str,
        embedding: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let mut builder = ProceduralRecord::builder()
            .agent_id(aid)
            .name(name)
            .description(description);
        if let Some(emb_obj) = embedding {
            builder = builder.embedding(extract_embedding(py, emb_obj)?);
        }
        let record = builder.build().map_err(to_py_err)?;
        let id = block_on(db.procedural().store(record)).map_err(to_py_err)?;
        Ok(id.to_string())
    }

    /// Add an entry to working memory.
    ///
    /// Args:
    ///     agent_id: The agent who owns the entry.
    ///     content: Text content for the working memory slot.
    ///     token_count: Optional client-side estimate. The Rust tokenizer
    ///         remains authoritative and revalidates the effective token count.
    ///
    /// Returns:
    ///     The ULID string of the new working memory entry.
    #[pyo3(signature = (agent_id, content, *, token_count=None))]
    fn focus(&self, agent_id: &str, content: &str, token_count: Option<u32>) -> PyResult<String> {
        let db = self.db()?;
        let aid = parse_agent_id(agent_id)?;
        let effective_token_count = authoritative_working_token_count(db, content, token_count);
        let entry = WorkingMemoryEntry::builder()
            .agent_id(aid)
            .content(content)
            .token_count(effective_token_count)
            .build()
            .map_err(to_py_err)?;
        let id = block_on(db.working().focus(entry)).map_err(to_py_err)?;
        Ok(id.to_string())
    }

    /// Remove an entry from working memory.
    ///
    /// Args:
    ///     id: ULID string of the working memory entry to remove.
    fn defocus(&self, id: &str) -> PyResult<()> {
        let db = self.db()?;
        let mid = parse_memory_id(id)?;
        block_on(db.working().defocus(mid)).map_err(to_py_err)
    }

    /// Run the consolidation pipeline.
    fn consolidate(&self) -> PyResult<u64> {
        let db = self.db()?;
        let report = block_on(db.admin().consolidate().execute()).map_err(to_py_err)?;
        Ok(report.records_processed as u64)
    }

    /// Connect two memories with a graph edge.
    ///
    /// Args:
    ///     source: ULID string of the source memory.
    ///     target: ULID string of the target memory.
    ///
    /// Returns:
    ///     The edge ID string.
    fn connect(&self, source: &str, target: &str) -> PyResult<String> {
        let db = self.db()?;
        let src = parse_memory_id(source)?;
        let tgt = parse_memory_id(target)?;
        let edge_id = block_on(db.graph_view().connect(src, tgt)).map_err(to_py_err)?;
        Ok(format!("{edge_id:?}"))
    }

    /// Watch for memory events.
    ///
    /// Subscribes to the database event stream and collects events for the
    /// specified duration.
    ///
    /// Args:
    ///     duration_ms: How long to listen for events in milliseconds (default: 1000).
    ///
    /// Returns:
    ///     List of dicts, each with ``type`` and event-specific fields.
    #[pyo3(signature = (*, duration_ms=1000))]
    fn watch(&self, py: Python<'_>, duration_ms: u64) -> PyResult<Py<PyList>> {
        let db = self.db()?;
        let mut rx = db.subscribe();
        let events = py.detach(move || {
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(duration_ms);
            let mut events = Vec::new();
            while std::time::Instant::now() < deadline {
                match block_on(tokio::time::timeout(
                    std::time::Duration::from_millis(50),
                    rx.recv(),
                )) {
                    Ok(Ok(ev)) => {
                        let json = serde_json::to_value(&ev).unwrap_or_default();
                        events.push(json);
                    }
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => break,
                    Err(_timeout) => continue,
                }
            }
            events
        });
        let list = PyList::empty(py);
        for ev in events {
            let obj = json_to_pyobj(py, &ev)?;
            list.append(obj)?;
        }
        Ok(list.into())
    }
}

// ─── AsyncHirnBridge ─────────────────────────────────────────

/// Async internal native bridge for the Python bindings.
///
/// The public package root exposes the high-level ``AsyncMemory`` API instead.
///
/// Usage::
///
///     from hirn._hirn import AsyncHirnBridge
///
///     async with AsyncHirnBridge.open("path/to.hirn") as h:
///         await h.remember("agent", "content", embedding=[0.1] * 64)
#[pyclass(name = "AsyncHirnBridge")]
struct AsyncHirn {
    db: Option<Arc<hirn::HirnDB>>,
}

impl AsyncHirn {
    fn db(&self) -> PyResult<&Arc<hirn::HirnDB>> {
        self.db
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("database is closed"))
    }
}

#[pymethods]
impl AsyncHirn {
    /// Open a hirn database asynchronously.
    #[staticmethod]
    #[pyo3(signature = (path, *, embedding_dimensions=768, token_budget=4096, tokenizer_name=None))]
    fn open<'py>(
        py: Python<'py>,
        path: String,
        embedding_dimensions: u32,
        token_budget: u32,
        tokenizer_name: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let config = HirnConfig::builder()
                .db_path(&path)
                .embedding_dimensions(embedding_dimensions)
                .token_budget(token_budget)
                .build()
                .map_err(to_py_err)?;
            let storage = open_lance_storage(&path)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("storage: {e}")))?;
            let db = hirn::HirnDB::open_with_config(config, storage)
                .await
                .map_err(to_py_err)?;
            if let Some(tokenizer_name) = tokenizer_name.as_deref() {
                db.set_tokenizer(resolve_registry_tokenizer(tokenizer_name)?);
            }
            Ok(AsyncHirn {
                db: Some(Arc::new(db)),
            })
        })
    }

    /// Close the database.
    fn close<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.db = None;
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    fn __aenter__<'py>(slf: Py<Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        pyo3_async_runtimes::tokio::future_into_py(py, async move { Ok(slf) })
    }

    fn __aexit__<'py>(
        &mut self,
        py: Python<'py>,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_val: Option<&Bound<'_, PyAny>>,
        _exc_tb: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.db = None;
        pyo3_async_runtimes::tokio::future_into_py(py, async { Ok(()) })
    }

    /// Register an agent asynchronously.
    fn register_agent<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        display_name: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let aid = parse_agent_id(&agent_id)?;
            db.register_agent(&aid, &display_name)
                .await
                .map_err(to_py_err)?;
            Ok(())
        })
    }

    /// Store an episodic memory asynchronously.
    ///
    /// Args:
    ///     embedding: Optional embedding vector (list or numpy array).
    #[pyo3(signature = (agent_id, content, *, embedding=None, importance=0.5))]
    fn remember<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        content: String,
        embedding: Option<&Bound<'py, PyAny>>,
        importance: f32,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        // F-65: Accept numpy arrays (like sync Hirn) by extracting before entering async.
        let emb = embedding
            .map(|obj| extract_embedding(py, obj))
            .transpose()?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let aid = parse_agent_id(&agent_id)?;
            let mut builder = EpisodicRecord::builder()
                .agent_id(aid.clone())
                .content(&content)
                .importance(importance);
            if let Some(emb) = emb {
                builder = builder.embedding(emb);
            }
            let record = builder.build().map_err(to_py_err)?;
            let ctx = db.as_agent(&aid).await.map_err(to_py_err)?;
            let id = ctx.remember(record).await.map_err(to_py_err)?;
            Ok(id.to_string())
        })
    }

    /// Recall memories asynchronously.
    ///
    /// Args:
    ///     query: Query embedding vector (list or numpy array).
    ///     as_of: Optional snapshot value (`YYYY-MM-DD`, RFC 3339, or revision ULID).
    ///     snapshot_kind: Optional snapshot kind: `observed`, `recorded`, or `revision`.
    #[pyo3(signature = (agent_id, query, *, limit=10, threshold=None, as_of=None, snapshot_kind=None))]
    fn recall<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        query: &Bound<'py, PyAny>,
        limit: usize,
        threshold: Option<f32>,
        as_of: Option<String>,
        snapshot_kind: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        // F-65: Accept numpy arrays by extracting before entering async.
        let emb = extract_embedding(py, query)?;
        let snapshot = parse_optional_recall_snapshot(as_of.as_deref(), snapshot_kind.as_deref())?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let aid = parse_agent_id(&agent_id)?;
                let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
                let mut builder = ctx.recall(emb).limit(limit);
                if let Some(t) = threshold {
                    builder = builder.threshold(t);
                }
                if let Some(snapshot) = snapshot {
                    builder = builder.snapshot(snapshot);
                }
                let results = block_on(builder.execute()).map_err(to_py_err)?;
                Ok(results
                    .iter()
                    .map(recall_result_from_runtime)
                    .collect::<Vec<_>>())
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        })
    }

    /// Assemble context asynchronously.
    ///
    /// Args:
    ///     query: Query embedding vector (list or numpy array).
    #[pyo3(signature = (agent_id, query, *, budget=4096))]
    fn think<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        query: &Bound<'py, PyAny>,
        budget: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        // F-65: Accept numpy arrays by extracting before entering async.
        let emb = extract_embedding(py, query)?;
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let aid = parse_agent_id(&agent_id)?;
                let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
                let result =
                    block_on(ctx.think(emb).budget(budget).execute()).map_err(to_py_err)?;
                Ok(Context {
                    context: result.context,
                    token_count: result.token_count,
                    records_included_ids: result
                        .records_included
                        .iter()
                        .map(|id| id.to_string())
                        .collect(),
                    query_time_ms: result.query_time_ms,
                })
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        })
    }

    /// Forget a memory asynchronously.
    fn forget<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let aid = parse_agent_id(&agent_id)?;
            let mid = parse_memory_id(&id)?;
            let ctx = db.as_agent(&aid).await.map_err(to_py_err)?;
            ctx.archive_episode(mid).await.map_err(to_py_err)?;
            Ok(())
        })
    }

    /// Execute a HirnQL query asynchronously.
    fn execute<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        query: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let aid = parse_agent_id(&agent_id)?;
                let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
                let result = block_on(ctx.execute_ql(&query)).map_err(to_py_err)?;
                let json_val = query_result_to_json(&result);
                let result_type = json_val["type"].as_str().unwrap_or("unknown").to_string();
                Ok(QueryResult {
                    result_type,
                    json_val,
                })
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        })
    }

    /// Inspect a memory record asynchronously.
    fn inspect<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let aid = parse_agent_id(&agent_id)?;
                let mid = parse_memory_id(&id)?;
                let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
                let result = block_on(ctx.inspect(mid)).map_err(to_py_err)?;
                let json_val = inspected_result_to_json(&result);
                let result_type = json_val["type"].as_str().unwrap_or("unknown").to_string();
                Ok(QueryResult {
                    result_type,
                    json_val,
                })
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        })
    }

    /// Trace provenance asynchronously.
    fn trace<'py>(
        &self,
        py: Python<'py>,
        agent_id: String,
        id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            tokio::task::spawn_blocking(move || {
                let aid = parse_agent_id(&agent_id)?;
                let mid = parse_memory_id(&id)?;
                let ctx = block_on(db.as_agent(&aid)).map_err(to_py_err)?;
                let result = block_on(ctx.trace(mid)).map_err(to_py_err)?;
                let json_val = trace_result_to_json(&result);
                Ok(QueryResult {
                    result_type: "traced".to_string(),
                    json_val,
                })
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        })
    }

    /// Get database statistics asynchronously.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let s = db.admin().stats().await.map_err(to_py_err)?;
            Ok(Stats {
                working_count: s.working_count,
                episodic_count: s.episodic_count,
                semantic_count: s.semantic_count,
                total_count: s.total_count,
                file_size_bytes: s.file_size_bytes,
            })
        })
    }

    /// Subscribe to memory events and return a WatchStream.
    ///
    /// Usage::
    ///
    ///     stream = await db.watch()
    ///     # stream.next() to poll, stream.cancel() to stop
    fn watch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let db = self.db()?.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let rx = db.subscribe();
            Ok(WatchStream {
                rx: Arc::new(tokio::sync::Mutex::new(rx)),
                done: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            })
        })
    }
}

// ─── WatchStream ─────────────────────────────────────────────

/// Event stream for watching memory changes.
///
/// Supports Python's ``async for`` protocol via PyO3's ``__anext__`` slot.
/// Events are dicts with variant-specific fields.
///
/// Usage::
///
///     stream = await db.watch()
///     async for event in stream:
///         print(event)
///     # or: stream.cancel() to stop early
#[pyclass]
struct WatchStream {
    rx: Arc<tokio::sync::Mutex<tokio::sync::broadcast::Receiver<MemoryEvent>>>,
    done: Arc<std::sync::atomic::AtomicBool>,
}

#[pymethods]
impl WatchStream {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Yield the next event without ending the stream on idle timeouts.
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let rx = Arc::clone(&self.rx);
        let done = Arc::clone(&self.done);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            loop {
                if done.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(PyStopAsyncIteration::new_err("watch stream closed"));
                }

                let result = {
                    let mut guard = rx.lock().await;
                    tokio::time::timeout(std::time::Duration::from_millis(200), guard.recv()).await
                };

                match result {
                    Ok(Ok(event)) => {
                        return Python::attach(|py| {
                            let json = serde_json::to_value(&event).unwrap_or_default();
                            json_to_pyobj(py, &json)
                        });
                    }
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => continue,
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => {
                        done.store(true, std::sync::atomic::Ordering::Relaxed);
                        return Err(PyStopAsyncIteration::new_err("watch stream closed"));
                    }
                    Err(_timeout) => continue,
                }
            }
        })
    }

    /// Poll for the next event synchronously.
    ///
    /// Args:
    ///     timeout_ms: Maximum time to wait in milliseconds (default: 200).
    ///
    /// Returns:
    ///     Event dict or None if timeout/stream ended.
    #[pyo3(signature = (*, timeout_ms=200))]
    fn next_event(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<Py<PyAny>>> {
        enum WatchRecv {
            Event(MemoryEvent),
            Timeout,
            Disconnected,
        }

        if self.done.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(None);
        }
        let rx = Arc::clone(&self.rx);
        let recv = py.detach(move || {
            block_on(async move {
                let mut guard = rx.lock().await;
                match tokio::time::timeout(
                    std::time::Duration::from_millis(timeout_ms),
                    guard.recv(),
                )
                .await
                {
                    Ok(Ok(event)) => WatchRecv::Event(event),
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Lagged(_))) => {
                        WatchRecv::Timeout
                    }
                    Ok(Err(tokio::sync::broadcast::error::RecvError::Closed)) => {
                        WatchRecv::Disconnected
                    }
                    Err(_timeout) => WatchRecv::Timeout,
                }
            })
        });

        match recv {
            WatchRecv::Event(event) => {
                let json = serde_json::to_value(&event).unwrap_or_default();
                let obj = json_to_pyobj(py, &json)?;
                Ok(Some(obj))
            }
            WatchRecv::Timeout => Ok(None),
            WatchRecv::Disconnected => {
                self.done.store(true, std::sync::atomic::Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    /// Cancel the watch stream.
    fn cancel(&self) {
        self.done.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether the stream has ended (disconnected or cancelled).
    fn is_done(&self) -> bool {
        self.done.load(std::sync::atomic::Ordering::Relaxed)
    }
}

// ─── Module ──────────────────────────────────────────────────

#[pymodule]
fn _hirn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Hirn>()?;
    m.add_class::<AsyncHirn>()?;
    m.add_class::<WatchStream>()?;
    m.add_class::<Stats>()?;
    m.add_class::<MemoryRecord>()?;
    m.add_class::<RecallResult>()?;
    m.add_class::<Context>()?;
    m.add_class::<QueryResult>()?;
    m.add("HirnError", m.py().get_type::<HirnError>())?;
    m.add("NotFoundError", m.py().get_type::<NotFoundError>())?;
    m.add("QueryError", m.py().get_type::<QueryError>())?;
    Ok(())
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
        let db_path = dir.path().join("python_binding_test");
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

    #[test]
    fn execute_history_query_returns_revision_history_json() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (bridge, _dir, agent_id, logical_memory_id, corrected_revision_id) =
            runtime.block_on(async {
                let (db, dir) = temp_db().await;
                db.register_agent(&agent(), "Test Agent").await.unwrap();
                let bridge = Hirn {
                    db: Some(Arc::new(db)),
                };
                let db = bridge.db().unwrap().clone();
                let ctx = db.as_agent(&agent()).await.unwrap();

                let id = ctx
                    .store_semantic(
                        SemanticRecord::builder()
                            .concept("python_history_binding")
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

                (
                    bridge,
                    dir,
                    agent().to_string(),
                    original.logical_memory_id.to_string(),
                    corrected.revision_id.to_string(),
                )
            });

        let result = bridge
            .execute(
                &agent_id,
                &format!(r#"HISTORY LOGICAL "{}""#, logical_memory_id),
            )
            .unwrap();

        assert_eq!(result.result_type, "history");
        assert_eq!(
            result.json_val["semantic_revision"]["logical_memory_id"],
            logical_memory_id
        );
        assert_eq!(result.json_val["semantic_revision"]["revision_count"], 2);
        assert_eq!(
            result.json_val["semantic_revision"]["current_revision_id"],
            corrected_revision_id
        );
        assert_eq!(result.json_val["items"].as_array().unwrap().len(), 2);
        assert_eq!(
            result.json_val["items"][1]["record"]["description"],
            "updated history policy"
        );
    }

    #[test]
    fn direct_inspect_returns_semantic_revision_and_conflict_groups() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (bridge, _dir, agent_id, left_id) = runtime.block_on(async {
            let (db, dir) = temp_db().await;
            db.register_agent(&agent(), "Test Agent").await.unwrap();
            let bridge = Hirn {
                db: Some(Arc::new(db)),
            };
            let db = bridge.db().unwrap().clone();
            let ctx = db.as_agent(&agent()).await.unwrap();

            let left_id = ctx
                .store_semantic(
                    SemanticRecord::builder()
                        .concept("python_inspect_left")
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
                        .concept("python_inspect_right")
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

            (bridge, dir, agent().to_string(), left_head_id.to_string())
        });

        let result = bridge.inspect(&agent_id, &left_id).unwrap();

        assert_eq!(result.result_type, "inspected");
        assert_eq!(result.json_val["layer"], "Semantic");
        assert!(result.json_val["neighbor_count"].as_u64().unwrap() >= 1);
        assert_eq!(
            result.json_val["semantic_revision"]["logical_state"],
            "Active"
        );
        assert_eq!(
            result.json_val["conflict_groups"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn direct_trace_returns_semantic_revision_and_conflict_groups() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (bridge, _dir, agent_id, left_id) = runtime.block_on(async {
            let (db, dir) = temp_db().await;
            db.register_agent(&agent(), "Test Agent").await.unwrap();
            let bridge = Hirn {
                db: Some(Arc::new(db)),
            };
            let db = bridge.db().unwrap().clone();
            let ctx = db.as_agent(&agent()).await.unwrap();

            let left_id = ctx
                .store_semantic(
                    SemanticRecord::builder()
                        .concept("python_trace_left")
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
                        .concept("python_trace_right")
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

            (bridge, dir, agent().to_string(), left_head_id.to_string())
        });

        let result = bridge.trace(&agent_id, &left_id).unwrap();

        assert_eq!(result.result_type, "traced");
        assert_eq!(result.json_val["layer"], "Semantic");
        assert!(result.json_val["source_episodes"].is_array());
        assert!(result.json_val["derived_records"].is_array());
        assert!(result.json_val["semantic_revision"].is_object());
        assert_eq!(
            result.json_val["semantic_revision"]["logical_state"],
            "Active"
        );
        assert_eq!(
            result.json_val["conflict_groups"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn execute_trace_query_returns_rich_trace_json() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (bridge, _dir, agent_id, left_id) = runtime.block_on(async {
            let (db, dir) = temp_db().await;
            db.register_agent(&agent(), "Test Agent").await.unwrap();
            let bridge = Hirn {
                db: Some(Arc::new(db)),
            };
            let db = bridge.db().unwrap().clone();
            let ctx = db.as_agent(&agent()).await.unwrap();

            let left_id = ctx
                .store_semantic(
                    SemanticRecord::builder()
                        .concept("python_execute_trace_left")
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
                        .concept("python_execute_trace_right")
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

            (bridge, dir, agent().to_string(), left_head_id.to_string())
        });

        let result = bridge
            .execute(&agent_id, &format!(r#"TRACE "{}""#, left_id))
            .unwrap();

        assert_eq!(result.result_type, "traced");
        assert_eq!(result.json_val["layer"], "Semantic");
        assert!(result.json_val["source_episodes"].is_array());
        assert!(result.json_val["derived_records"].is_array());
        assert_eq!(
            result.json_val["semantic_revision"]["logical_state"],
            "Active"
        );
        assert_eq!(
            result.json_val["conflict_groups"].as_array().unwrap().len(),
            1
        );
    }

    #[test]
    fn direct_recall_supports_explicit_snapshots_and_preserves_current_default() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (
            bridge,
            _dir,
            agent_id,
            current_embedding,
            original_embedding,
            logical_memory_id,
            original_revision_id,
            historical_cutoff,
            recorded_cutoff,
        ) = runtime.block_on(async {
            let (db, dir) = temp_db().await;
            db.register_agent(&agent(), "Test Agent").await.unwrap();
            let bridge = Hirn {
                db: Some(Arc::new(db)),
            };
            let db = bridge.db().unwrap().clone();
            let ctx = db.as_agent(&agent()).await.unwrap();

            let original_about = "lease authority";
            let current_about = "lease authority v2";
            let agent_id = agent().to_string();
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
            let historical_cutoff = original.created_at.to_string();
            let observed_at =
                Timestamp::from_millis(original.created_at.millis() + 2 * 60 * 60 * 1000);

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

            (
                bridge,
                dir,
                agent_id,
                current_embedding,
                original_embedding,
                logical_memory_id,
                original_revision_id,
                historical_cutoff,
                recorded_cutoff,
            )
        });
        drop(runtime);

        Python::initialize();
        Python::attach(|py| {
            let current_query = pyo3::types::PyList::new(
                py,
                current_embedding.iter().map(|value| f64::from(*value)),
            )
            .unwrap();
            let current = bridge
                .recall(py, &agent_id, current_query.as_any(), 10, None, None, None)
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

            let historical_query = pyo3::types::PyList::new(
                py,
                original_embedding.iter().map(|value| f64::from(*value)),
            )
            .unwrap();
            let historical = bridge
                .recall(
                    py,
                    &agent_id,
                    historical_query.as_any(),
                    10,
                    None,
                    Some(historical_cutoff.as_str()),
                    None,
                )
                .unwrap();
            assert_eq!(historical.len(), 1);
            assert_eq!(
                historical[0].revision_id.as_deref(),
                Some(original_revision_id.as_str())
            );
            assert_eq!(historical[0].revision_state.as_deref(), Some("Active"));

            let recorded_query = pyo3::types::PyList::new(
                py,
                current_embedding.iter().map(|value| f64::from(*value)),
            )
            .unwrap();
            let recorded = bridge
                .recall(
                    py,
                    &agent_id,
                    recorded_query.as_any(),
                    10,
                    None,
                    Some(recorded_cutoff.as_str()),
                    Some("recorded"),
                )
                .unwrap();
            assert_eq!(recorded.len(), 1);
            assert_ne!(
                recorded[0].revision_id.as_deref(),
                Some(original_revision_id.as_str())
            );

            let revision_query = pyo3::types::PyList::new(
                py,
                original_embedding.iter().map(|value| f64::from(*value)),
            )
            .unwrap();
            let revision_snapshot = bridge
                .recall(
                    py,
                    &agent_id,
                    revision_query.as_any(),
                    10,
                    None,
                    Some(original_revision_id.as_str()),
                    Some("revision"),
                )
                .unwrap();
            assert_eq!(revision_snapshot.len(), 1);
            assert_eq!(
                revision_snapshot[0].revision_id.as_deref(),
                Some(original_revision_id.as_str())
            );
        });
    }

    #[test]
    fn sync_watch_collects_events() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (bridge, _dir, db) = runtime.block_on(async {
            let (db, dir) = temp_db().await;
            db.register_agent(&agent(), "Test Agent").await.unwrap();
            let db = Arc::new(db);
            (
                Hirn {
                    db: Some(Arc::clone(&db)),
                },
                dir,
                db,
            )
        });

        let sender_db = Arc::clone(&db);
        let sender = std::thread::spawn(move || {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            runtime.block_on(async move {
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                sender_db
                    .semantic()
                    .store(
                        SemanticRecord::builder()
                            .concept("watch_test")
                            .description("watch event")
                            .agent_id(agent())
                            .build()
                            .unwrap(),
                    )
                    .await
                    .unwrap();
            });
        });

        Python::initialize();
        Python::attach(|py| {
            let events = bridge.watch(py, 250).unwrap();
            let events = events.bind(py);
            assert!(!events.is_empty());
        });

        sender.join().unwrap();
    }
}
