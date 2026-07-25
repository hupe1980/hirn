//! AST types produced by the HirnQL parser.

use std::collections::HashMap;
use std::fmt;

use hirn_core::types::{EdgeRelation, Layer};

// ── Top-level statement ────────────────────────────────────────────────

/// A parsed HirnQL statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Recall(Box<RecallStmt>),
    RecallEvents(RecallEventsStmt),
    Think(Box<ThinkStmt>),
    Correct(CorrectStmt),
    Supersede(SupersedeStmt),
    MergeMemory(MergeMemoryStmt),
    Retract(RetractStmt),
    Inspect(InspectStmt),
    History(HistoryStmt),
    Trace(TraceStmt),
    Traverse(TraverseStmt),
    Explain(ExplainStmt),
    ExplainCauses(ExplainCausesStmt),
    WhatIf(WhatIfStmt),
    Counterfactual(CounterfactualStmt),
    CreateRealm(CreateRealmStmt),
    DropRealm(DropRealmStmt),
    Grant(GrantStmt),
    Revoke(RevokeStmt),
    ShowPolicies(ShowPoliciesStmt),
    ExplainPolicy(ExplainPolicyStmt),
    ShowCluster,
    SetTierPolicy(SetTierPolicyStmt),
}

// ── EXPLAIN ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ExplainStmt {
    /// Whether EXPLAIN ANALYZE was used (execute + compare estimates).
    pub analyze: bool,
    /// The inner statement to explain.
    pub inner: Box<Statement>,
}

// ── EXPLAIN CAUSES (Pearl Rung 1) ──────────────────────────────────────

/// `EXPLAIN CAUSES "event" [IN <ns>] [DEPTH N]` — find causal chains backward.
#[derive(Debug, Clone, PartialEq)]
pub struct ExplainCausesStmt {
    /// The event description to explain.
    pub target: String,
    /// Optional namespace scope.
    pub namespace: Option<String>,
    /// Max causal chain depth (default: 3).
    pub depth: Option<usize>,
}

// ── WHAT_IF (Pearl Rung 2) ─────────────────────────────────────────────

/// `WHAT_IF "intervention" THEN "outcome" [IN <ns>]` — simulate do-calculus.
#[derive(Debug, Clone, PartialEq)]
pub struct WhatIfStmt {
    /// The intervention (do-variable).
    pub intervention: String,
    /// The outcome to evaluate.
    pub outcome: String,
    /// Optional namespace scope.
    pub namespace: Option<String>,
}

// ── COUNTERFACTUAL (Pearl Rung 3) ──────────────────────────────────────

/// `COUNTERFACTUAL "antecedent" THEN "consequent" [IN <ns>]` — reason about alternative histories.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualStmt {
    /// The counterfactual antecedent (what didn't happen).
    pub antecedent: String,
    /// The consequent to evaluate.
    pub consequent: String,
    /// Optional namespace scope.
    pub namespace: Option<String>,
}

// ── RECALL ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecallSnapshotAst {
    Unqualified(String),
    Observed(String),
    Recorded(String),
    Revision(String),
}

impl fmt::Display for RecallSnapshotAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unqualified(value) => write!(f, "\"{}\"", EscapeStr(value)),
            Self::Observed(value) => write!(f, "OBSERVED \"{}\"", EscapeStr(value)),
            Self::Recorded(value) => write!(f, "RECORDED \"{}\"", EscapeStr(value)),
            Self::Revision(value) => write!(f, "REVISION \"{}\"", EscapeStr(value)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallStmt {
    pub layers: Vec<Layer>,
    pub about: StringOrParam,
    pub involving: Option<Vec<String>>,
    pub temporal: Option<TemporalClause>,
    pub as_of: Option<RecallSnapshotAst>,
    pub expand: Option<ExpandClause>,
    pub follow_causes: Option<usize>,
    pub where_clauses: Vec<WhereCondition>,
    pub subquery_filters: Vec<SubqueryFilter>,
    pub modality: Option<Vec<String>>,
    pub resource_roles: Option<Vec<String>>,
    pub hydration_modes: Option<Vec<String>>,
    pub artifact_kinds: Option<Vec<String>>,
    /// Depth scheduling: Auto (default), Full, Summary.
    pub depth_mode: Option<DepthModeAst>,
    /// WITH PROSPECTIVE ON|OFF (default: ON).
    pub with_prospective: Option<bool>,
    /// WITH MCFA_DEFENSE ON|OFF (default: OFF for recall).
    pub with_mcfa: Option<bool>,
    /// WITH CONFLICTS — include contradiction annotations.
    pub with_conflicts: bool,
    /// WITH PROVENANCE DEPTH N — expand DerivedFrom/PartOf edges (0 = no expansion).
    pub provenance_depth: Option<usize>,
    /// TOPIC "label" — scoped to a specific topic timeline.
    pub topic: Option<String>,
    pub group_by: Option<GroupByClause>,
    pub projection: Option<Vec<String>>,
    pub output_format: Option<OutputFormat>,
    pub result_format: Option<OutputFormat>,
    pub budget: Option<usize>,
    pub namespace: Option<String>,
    /// FROM REALM "a", "b" — cross-realm query (daemon-dispatched).
    pub from_realms: Option<Vec<String>>,
    pub consistency: Option<ConsistencyLevel>,
    pub limit: Option<usize>,
    /// Enable hybrid BM25+vector search.
    pub hybrid: bool,
}

// ── RECALL EVENTS (audit query) ────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct RecallEventsStmt {
    /// Entity filter: `EVENTS FOR "entity"`.
    pub entity_filter: Option<String>,
    pub where_clauses: Vec<WhereCondition>,
    pub temporal: Option<TemporalClause>,
    pub namespace: Option<String>,
    pub limit: Option<usize>,
}

// ── THINK ──────────────────────────────────────────────────────────────

/// Retrieval mode for THINK statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RetrievalMode {
    /// Standard local retrieval (HNSW + spreading activation).
    #[default]
    Local,
    /// Global retrieval via community summaries.
    Global,
    /// Both local and global, results merged.
    Hybrid,
    /// RAPTOR tree-based retrieval (Sarthi et al., 2024).
    /// Top-down traversal through hierarchical summaries.
    Raptor,
    /// Adaptive retrieval (Jeong et al., NAACL 2024).
    /// Automatically classifies query complexity and routes to the optimal strategy:
    /// simple → local only, moderate → hybrid, complex → full pipeline with RAPTOR.
    Adaptive,
    /// Iterative multi-hop retrieval — retrieve → reformulate → retrieve loop.
    Iterative,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ThinkStmt {
    pub about: StringOrParam,
    pub involving: Option<Vec<String>>,
    pub temporal: Option<TemporalClause>,
    pub expand: Option<ExpandClause>,
    pub follow_causes: Option<usize>,
    pub where_clauses: Vec<WhereCondition>,
    pub output_format: Option<OutputFormat>,
    pub budget: Option<usize>,
    pub namespace: Option<String>,
    pub consistency: Option<ConsistencyLevel>,
    pub limit: Option<usize>,
    /// Enable hybrid BM25+vector search on the local THINK branch.
    pub hybrid: bool,
    pub mode: RetrievalMode,
    /// Depth scheduling: Auto (default), Full, Summary.
    pub depth_mode: Option<DepthModeAst>,
    /// WITH PROSPECTIVE ON|OFF (default: ON).
    pub with_prospective: Option<bool>,
    /// WITH MCFA_DEFENSE ON|OFF.
    pub with_mcfa: Option<bool>,
    /// WITH PROVENANCE DEPTH N — expand DerivedFrom/PartOf edges (0 = no expansion).
    pub provenance_depth: Option<usize>,
    /// Maximum hops for iterative retrieval (default: 3).
    pub max_hops: Option<usize>,
    pub community_depth: Option<usize>,
}

// ── REMEMBER ───────────────────────────────────────────────────────────

/// Multi-modal content parsed from HirnQL CONTENT IMAGE/CODE/AUDIO/VIDEO/DOCUMENT/STRUCTURED.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModalContent {
    Image {
        data: String,
        description: String,
    },
    Code {
        source: String,
        language: String,
    },
    Audio {
        data: String,
        transcript: String,
    },
    Video {
        data: String,
        transcript: String,
        description: String,
    },
    Document {
        data: String,
        title: String,
    },
    External {
        uri: String,
        title: String,
        snippet: Option<String>,
        mime_type: Option<String>,
        checksum: Option<String>,
        fetch_policy: Option<String>,
        stale_at: Option<String>,
    },
    ToolOutput {
        output: String,
        tool: String,
        mime_type: Option<String>,
        schema: Option<String>,
        call_id: Option<String>,
        checksum: Option<String>,
    },
    Structured {
        data: String,
        schema: String,
    },
}

/// A SET assignment in semantic mutation SET clauses.
#[derive(Debug, Clone, PartialEq)]
pub struct SetAssignment {
    pub field: String,
    pub value: SetValue,
}

/// Value for a SET assignment.
#[derive(Debug, Clone, PartialEq)]
pub enum SetValue {
    Float(f64),
    Int(i64),
    String(String),
    Max(String, f64),
    Min(String, f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ForgetMode {
    #[default]
    Archive,
    Purge,
    /// Hard delete — irrecoverable.
    Hard,
}

// ── CORRECT / RETRACT ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticTargetRef {
    Memory(String),
    Logical(String),
    Revision(String),
}

impl SemanticTargetRef {
    #[must_use]
    pub fn raw_value(&self) -> &str {
        match self {
            Self::Memory(value) | Self::Logical(value) | Self::Revision(value) => value,
        }
    }

    /// Replace the inner value (preserving the variant), used by parameter
    /// binding to substitute a `$param` target with its concrete id.
    pub fn set_raw_value(&mut self, new_value: String) {
        match self {
            Self::Memory(value) | Self::Logical(value) | Self::Revision(value) => {
                *value = new_value;
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorrectStmt {
    pub target: SemanticTargetRef,
    pub updates: Vec<SetAssignment>,
    pub reason: Option<String>,
    pub observed_at: Option<String>,
    pub caused_by: Option<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SupersedeStmt {
    pub target: SemanticTargetRef,
    pub updates: Vec<SetAssignment>,
    pub reason: Option<String>,
    pub observed_at: Option<String>,
    pub caused_by: Option<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MergeMemoryStmt {
    pub sources: Vec<SemanticTargetRef>,
    pub target: SemanticTargetRef,
    pub updates: Vec<SetAssignment>,
    pub reason: Option<String>,
    pub observed_at: Option<String>,
    pub caused_by: Option<String>,
    pub namespace: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetractStmt {
    pub target: SemanticTargetRef,
    pub reason: Option<String>,
    pub observed_at: Option<String>,
    pub caused_by: Option<String>,
    pub namespace: Option<String>,
}

// ── INSPECT ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectStmt {
    pub target: SemanticTargetRef,
}

// ── HISTORY ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryStmt {
    pub target: SemanticTargetRef,
    pub namespace: Option<String>,
}

// ── TRACE ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceStmt {
    pub target: SemanticTargetRef,
}

// ── Shared clause types ────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum TemporalClause {
    After(String),
    Before(String),
    Between { start: String, end: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExpandClause {
    pub depth: usize,
    pub min_weight: Option<f32>,
    pub activation: Option<ActivationModeAst>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationModeAst {
    None,
    Static,
    Spreading,
    /// Personalized PageRank (F-057).
    Ppr,
}

/// Depth scheduling mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthModeAst {
    /// Let the engine classify automatically.
    Auto,
    /// Always run full pipeline.
    Full,
    /// Summary-only (skip graph activation).
    Summary,
}

impl fmt::Display for DepthModeAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "AUTO"),
            Self::Full => write!(f, "FULL"),
            Self::Summary => write!(f, "SUMMARY"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WhereCondition {
    pub field: String,
    pub op: ComparisonOp,
    pub value: ConditionValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Gt,
    Lt,
    Gte,
    Lte,
    Eq,
    Neq,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConditionValue {
    Float(f64),
    Int(i64),
    String(String),
    /// Unresolved parameter placeholder (e.g. `$1`, `$threshold`).
    Param(String),
}

/// A value in a free-text string position (e.g. `ABOUT`) that is either a
/// literal string or an unbound parameter placeholder.
///
/// The distinction is captured **at parse time** from the grammar (a quoted
/// `string_literal` vs a bare `parameter` token) and preserved through the
/// AST. This is what lets a quoted literal such as `ABOUT "$q"` be treated as
/// a search for the text `$q` (a `Literal`) while a bare `ABOUT $q` is a
/// placeholder (a `Param`) that must be bound before execution — mirroring
/// [`ConditionValue::Param`] for WHERE values (R-50). Without this, the old
/// `$`-prefix heuristic on a plain `String` could not tell them apart, so a
/// quoted `"$q"` was wrongly re-bound and an unbound `$q` silently reached
/// execution as a literal search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringOrParam {
    /// A literal string value (verbatim search text).
    Literal(String),
    /// An unbound parameter placeholder, stored **with** its `$` prefix
    /// (e.g. `$1`, `$query`).
    Param(String),
}

impl StringOrParam {
    /// The parameter name (with `$` prefix) if this is an unbound placeholder.
    pub fn param_name(&self) -> Option<&str> {
        match self {
            Self::Param(name) => Some(name.as_str()),
            Self::Literal(_) => None,
        }
    }

    /// The literal text if this is a `Literal`, otherwise `None`.
    pub fn as_literal(&self) -> Option<&str> {
        match self {
            Self::Literal(value) => Some(value.as_str()),
            Self::Param(_) => None,
        }
    }

    /// True when this is a `Literal` whose (trimmed) content is empty — used to
    /// detect a missing/blank `ABOUT` clause. A `Param` is never blank.
    pub fn is_blank_literal(&self) -> bool {
        matches!(self, Self::Literal(value) if value.trim().is_empty())
    }
}

impl Default for StringOrParam {
    fn default() -> Self {
        Self::Literal(String::new())
    }
}

impl From<String> for StringOrParam {
    fn from(value: String) -> Self {
        Self::Literal(value)
    }
}

impl From<&str> for StringOrParam {
    fn from(value: &str) -> Self {
        Self::Literal(value.to_string())
    }
}

impl PartialEq<str> for StringOrParam {
    fn eq(&self, other: &str) -> bool {
        matches!(self, Self::Literal(value) if value == other)
    }
}

impl PartialEq<&str> for StringOrParam {
    fn eq(&self, other: &&str) -> bool {
        matches!(self, Self::Literal(value) if value == *other)
    }
}

impl fmt::Display for StringOrParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Literals re-render as quoted+escaped so the statement re-parses
            // to an equal AST; params render as the bare `$name` token.
            Self::Literal(value) => write!(f, "\"{}\"", EscapeStr(value)),
            Self::Param(name) => write!(f, "{name}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Narrative,
    Context,
    Graph,
    CausalChain,
    Json,
    Csv,
    Structured,
}

/// GROUP BY clause with aggregation function.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupByClause {
    pub field: String,
    pub function: AggFunction,
}

/// Aggregation function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggFunction {
    Count,
    Avg,
    Sum,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsistencyLevel {
    Linearizable,
    Eventual,
    Session,
}

/// A WHERE ... IN (subquery) filter.
#[derive(Debug, Clone, PartialEq)]
pub struct SubqueryFilter {
    /// The field to match (e.g. "caused_by", "id").
    pub field: String,
    /// The inner subquery that produces IDs.
    pub subquery: Subquery,
}

/// An inner RECALL subquery (used in WHERE ... IN (...)).
#[derive(Debug, Clone, PartialEq)]
pub struct Subquery {
    pub layers: Vec<Layer>,
    pub about: StringOrParam,
    pub involving: Option<Vec<String>>,
    pub temporal: Option<TemporalClause>,
    pub limit: Option<usize>,
}

// ── Display implementations ────────────────────────────────────────────

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Recall(s) => write!(f, "{s}"),
            Self::RecallEvents(s) => write!(f, "{s}"),
            Self::Think(s) => write!(f, "{s}"),
            Self::Correct(s) => write!(f, "{s}"),
            Self::Supersede(s) => write!(f, "{s}"),
            Self::MergeMemory(s) => write!(f, "{s}"),
            Self::Retract(s) => write!(f, "{s}"),
            Self::Inspect(s) => write!(f, "{s}"),
            Self::History(s) => write!(f, "{s}"),
            Self::Trace(s) => write!(f, "{s}"),
            Self::Traverse(s) => write!(f, "{s}"),
            Self::Explain(s) => write!(f, "{s}"),
            Self::ExplainCauses(s) => write!(f, "{s}"),
            Self::WhatIf(s) => write!(f, "{s}"),
            Self::Counterfactual(s) => write!(f, "{s}"),
            Self::CreateRealm(s) => write!(f, "{s}"),
            Self::DropRealm(s) => write!(f, "{s}"),
            Self::Grant(s) => write!(f, "{s}"),
            Self::Revoke(s) => write!(f, "{s}"),
            Self::ShowPolicies(s) => write!(f, "{s}"),
            Self::ExplainPolicy(s) => write!(f, "{s}"),
            Self::ShowCluster => write!(f, "SHOW CLUSTER"),
            Self::SetTierPolicy(s) => write!(f, "{s}"),
        }
    }
}

impl fmt::Display for ExplainStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EXPLAIN")?;
        if self.analyze {
            write!(f, " ANALYZE")?;
        }
        write!(f, " {}", self.inner)
    }
}

impl fmt::Display for ExplainCausesStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EXPLAIN CAUSES \"{}\"", EscapeStr(&self.target))?;
        if let Some(ref ns) = self.namespace {
            write_namespace(f, ns)?;
        }
        if let Some(d) = self.depth {
            write!(f, " DEPTH {d}")?;
        }
        Ok(())
    }
}

impl fmt::Display for WhatIfStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WHAT_IF \"{}\" THEN \"{}\"",
            EscapeStr(&self.intervention),
            EscapeStr(&self.outcome)
        )?;
        if let Some(ref ns) = self.namespace {
            write_namespace(f, ns)?;
        }
        Ok(())
    }
}

impl fmt::Display for CounterfactualStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "COUNTERFACTUAL \"{}\" THEN \"{}\"",
            EscapeStr(&self.antecedent),
            EscapeStr(&self.consequent)
        )?;
        if let Some(ref ns) = self.namespace {
            write_namespace(f, ns)?;
        }
        Ok(())
    }
}

fn write_layer_filter(f: &mut fmt::Formatter<'_>, layers: &[Layer]) -> fmt::Result {
    for (i, l) in layers.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{}", display_layer(*l))?;
    }
    Ok(())
}

fn display_layer(l: Layer) -> &'static str {
    match l {
        Layer::Episodic => "episodic",
        Layer::Semantic => "semantic",
        Layer::Working => "working",
        Layer::Procedural => "procedural",
    }
}

fn write_string_list(f: &mut fmt::Formatter<'_>, items: &[String]) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "\"{}\"", EscapeStr(item))?;
    }
    Ok(())
}

/// Render a ` NAMESPACE "..."` clause with the namespace quoted and escaped.
///
/// `namespace_clause` accepts a quoted `string_literal`, so quoting is both
/// safe (a namespace containing spaces/quotes cannot break out and inject
/// trailing clauses when a statement is re-serialized) and round-trips.
fn write_namespace(f: &mut fmt::Formatter<'_>, ns: &str) -> fmt::Result {
    write!(f, " NAMESPACE \"{}\"", EscapeStr(ns))
}

fn write_semantic_target_list(
    f: &mut fmt::Formatter<'_>,
    items: &[SemanticTargetRef],
) -> fmt::Result {
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

/// Zero-allocation string escaper — writes `\\` and `\"` directly to the
/// formatter without heap-allocating an intermediate `String`.
struct EscapeStr<'a>(&'a str);

impl fmt::Display for EscapeStr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        let mut start = 0;
        for (i, ch) in s.char_indices() {
            let esc = match ch {
                '\\' => "\\\\",
                '"' => "\\\"",
                _ => continue,
            };
            f.write_str(&s[start..i])?;
            f.write_str(esc)?;
            start = i + ch.len_utf8();
        }
        f.write_str(&s[start..])
    }
}

/// Format a float so it always carries a decimal point. The grammar's
/// `float_literal` requires a `.`, so a whole-number float like `1.0` must not
/// serialize as `1` — otherwise it fails to re-parse (or silently re-parses as
/// an integer) on the prepared-statement round-trip path.
struct FloatLit(f64);

impl fmt::Display for FloatLit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.0;
        if v.is_finite() && v.fract() == 0.0 {
            write!(f, "{v:.1}")
        } else {
            write!(f, "{v}")
        }
    }
}

impl fmt::Display for SemanticTargetRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Memory(value) => write!(f, "\"{}\"", EscapeStr(value)),
            Self::Logical(value) => write!(f, "LOGICAL \"{}\"", EscapeStr(value)),
            Self::Revision(value) => write!(f, "REVISION \"{}\"", EscapeStr(value)),
        }
    }
}

impl fmt::Display for RecallStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Clauses MUST be emitted in exact grammar order (see `recall_stmt` in
        // hirnql.pest) so the rendered statement re-parses to an equal AST.
        write!(f, "RECALL ")?;
        write_layer_filter(f, &self.layers)?;
        write!(f, " ABOUT {}", self.about)?;
        if let Some(ref inv) = self.involving {
            write!(f, " INVOLVING ")?;
            write_string_list(f, inv)?;
        }
        if let Some(ref tc) = self.temporal {
            write!(f, " {tc}")?;
        }
        if let Some(ref snapshot) = self.as_of {
            write!(f, " AS OF {snapshot}")?;
        }
        if let Some(ref ex) = self.expand {
            write!(f, " {ex}")?;
        }
        if let Some(d) = self.follow_causes {
            write!(f, " FOLLOW CAUSES DEPTH {d}")?;
        }
        for wc in &self.where_clauses {
            write!(f, " {wc}")?;
        }
        for sf in &self.subquery_filters {
            write!(f, " WHERE {} IN ({})", sf.field, sf.subquery)?;
        }
        if let Some(ref modalities) = self.modality {
            write!(f, " MODALITY {}", modalities.join(", "))?;
        }
        if let Some(ref resource_roles) = self.resource_roles {
            write!(f, " RESOURCE_ROLE {}", resource_roles.join(", "))?;
        }
        if let Some(ref hydration_modes) = self.hydration_modes {
            write!(f, " HYDRATION {}", hydration_modes.join(", "))?;
        }
        if let Some(ref artifact_kinds) = self.artifact_kinds {
            write!(f, " ARTIFACT {}", artifact_kinds.join(", "))?;
        }
        if let Some(dm) = self.depth_mode {
            write!(f, " DEPTH {dm}")?;
        }
        if let Some(ref topic) = self.topic {
            write!(f, " TOPIC \"{}\"", EscapeStr(topic))?;
        }
        if let Some(wp) = self.with_prospective {
            write!(f, " WITH PROSPECTIVE {}", if wp { "ON" } else { "OFF" })?;
        }
        if let Some(wm) = self.with_mcfa {
            write!(f, " WITH MCFA_DEFENSE {}", if wm { "ON" } else { "OFF" })?;
        }
        if self.with_conflicts {
            write!(f, " WITH CONFLICTS")?;
        }
        if let Some(pd) = self.provenance_depth {
            write!(f, " WITH PROVENANCE DEPTH {pd}")?;
        }
        if let Some(ref gb) = self.group_by {
            write!(f, " GROUP BY {} {}", gb.field, gb.function)?;
        }
        if let Some(ref proj) = self.projection {
            write!(f, " SELECT {}", proj.join(", "))?;
        }
        if let Some(of) = self.output_format {
            write!(f, " AS {of}")?;
        }
        if let Some(ref rf) = self.result_format {
            write!(f, " FORMAT {rf}")?;
        }
        if let Some(b) = self.budget {
            write!(f, " BUDGET {b}")?;
        }
        if let Some(ref ns) = self.namespace {
            write_namespace(f, ns)?;
        }
        if let Some(ref realms) = self.from_realms {
            write!(f, " FROM REALM ")?;
            for (i, r) in realms.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "\"{}\"", EscapeStr(r))?;
            }
        }
        if let Some(c) = self.consistency {
            write!(f, " CONSISTENCY {c}")?;
        }
        if let Some(l) = self.limit {
            write!(f, " LIMIT {l}")?;
        }
        if self.hybrid {
            write!(f, " HYBRID")?;
        }
        Ok(())
    }
}

impl fmt::Display for RecallEventsStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RECALL EVENTS")?;
        if let Some(ref entity) = self.entity_filter {
            write!(f, " FOR \"{}\"", EscapeStr(entity))?;
        }
        for wc in &self.where_clauses {
            // `WhereCondition::Display` already emits the `WHERE` keyword; an
            // extra one here produced `WHERE WHERE …`, which fails to re-parse
            // on the prepared-statement round-trip path.
            write!(f, " {wc}")?;
        }
        if let Some(ref tc) = self.temporal {
            write!(f, " {tc}")?;
        }
        if let Some(ref ns) = self.namespace {
            write_namespace(f, ns)?;
        }
        if let Some(l) = self.limit {
            write!(f, " LIMIT {l}")?;
        }
        Ok(())
    }
}

impl fmt::Display for ThinkStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Exact grammar order per `think_stmt` in hirnql.pest.
        write!(f, "THINK")?;
        if self.mode == RetrievalMode::Global {
            write!(f, " GLOBAL")?;
        }
        write!(f, " ABOUT {}", self.about)?;
        if let Some(ref inv) = self.involving {
            write!(f, " INVOLVING ")?;
            write_string_list(f, inv)?;
        }
        if let Some(ref tc) = self.temporal {
            write!(f, " {tc}")?;
        }
        if let Some(ref ex) = self.expand {
            write!(f, " {ex}")?;
        }
        if let Some(d) = self.follow_causes {
            write!(f, " FOLLOW CAUSES DEPTH {d}")?;
        }
        for wc in &self.where_clauses {
            write!(f, " {wc}")?;
        }
        if let Some(dm) = self.depth_mode {
            write!(f, " DEPTH {dm}")?;
        }
        if let Some(wp) = self.with_prospective {
            write!(f, " WITH PROSPECTIVE {}", if wp { "ON" } else { "OFF" })?;
        }
        if let Some(wm) = self.with_mcfa {
            write!(f, " WITH MCFA_DEFENSE {}", if wm { "ON" } else { "OFF" })?;
        }
        if let Some(pd) = self.provenance_depth {
            write!(f, " WITH PROVENANCE DEPTH {pd}")?;
        }
        if let Some(of) = self.output_format {
            write!(f, " AS {of}")?;
        }
        if let Some(b) = self.budget {
            write!(f, " BUDGET {b}")?;
        }
        if let Some(ref ns) = self.namespace {
            write_namespace(f, ns)?;
        }
        if let Some(c) = self.consistency {
            write!(f, " CONSISTENCY {c}")?;
        }
        if let Some(l) = self.limit {
            write!(f, " LIMIT {l}")?;
        }
        match self.mode {
            RetrievalMode::Hybrid => write!(f, " MODE hybrid")?,
            RetrievalMode::Raptor => write!(f, " MODE raptor")?,
            RetrievalMode::Adaptive => write!(f, " MODE adaptive")?,
            RetrievalMode::Iterative => {
                write!(f, " MODE iterative")?;
                if let Some(mh) = self.max_hops {
                    write!(f, " MAX_HOPS {mh}")?;
                }
            }
            _ => {}
        }
        if let Some(depth) = self.community_depth {
            write!(f, " COMMUNITY_DEPTH {depth}")?;
        }
        if self.hybrid {
            write!(f, " HYBRID")?;
        }
        Ok(())
    }
}

impl fmt::Display for SetAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = {}", self.field, self.value)
    }
}

impl fmt::Display for SetValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float(v) => write!(f, "{}", FloatLit(*v)),
            Self::Int(v) => write!(f, "{v}"),
            Self::String(v) => write!(f, "\"{}\"", EscapeStr(v)),
            Self::Max(field, val) => write!(f, "MAX({field}, {})", FloatLit(*val)),
            Self::Min(field, val) => write!(f, "MIN({field}, {})", FloatLit(*val)),
        }
    }
}

impl fmt::Display for CorrectStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CORRECT {} SET ", self.target)?;
        for (i, update) in self.updates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{update}")?;
        }
        if let Some(ref reason) = self.reason {
            write!(f, " REASON \"{}\"", EscapeStr(reason))?;
        }
        if let Some(ref observed_at) = self.observed_at {
            write!(f, " OBSERVED AT \"{}\"", EscapeStr(observed_at))?;
        }
        if let Some(ref caused_by) = self.caused_by {
            write!(f, " CAUSED BY \"{}\"", EscapeStr(caused_by))?;
        }
        if let Some(ref namespace) = self.namespace {
            write_namespace(f, namespace)?;
        }
        Ok(())
    }
}

impl fmt::Display for SupersedeStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SUPERSEDE {} SET ", self.target)?;
        for (i, update) in self.updates.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{update}")?;
        }
        if let Some(ref reason) = self.reason {
            write!(f, " REASON \"{}\"", EscapeStr(reason))?;
        }
        if let Some(ref observed_at) = self.observed_at {
            write!(f, " OBSERVED AT \"{}\"", EscapeStr(observed_at))?;
        }
        if let Some(ref caused_by) = self.caused_by {
            write!(f, " CAUSED BY \"{}\"", EscapeStr(caused_by))?;
        }
        if let Some(ref namespace) = self.namespace {
            write_namespace(f, namespace)?;
        }
        Ok(())
    }
}

impl fmt::Display for MergeMemoryStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MERGE MEMORY ")?;
        write_semantic_target_list(f, &self.sources)?;
        write!(f, " INTO {}", self.target)?;
        if !self.updates.is_empty() {
            write!(f, " SET ")?;
            for (i, update) in self.updates.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{update}")?;
            }
        }
        if let Some(ref reason) = self.reason {
            write!(f, " REASON \"{}\"", EscapeStr(reason))?;
        }
        if let Some(ref observed_at) = self.observed_at {
            write!(f, " OBSERVED AT \"{}\"", EscapeStr(observed_at))?;
        }
        if let Some(ref caused_by) = self.caused_by {
            write!(f, " CAUSED BY \"{}\"", EscapeStr(caused_by))?;
        }
        if let Some(ref namespace) = self.namespace {
            write_namespace(f, namespace)?;
        }
        Ok(())
    }
}

impl fmt::Display for RetractStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RETRACT {}", self.target)?;
        if let Some(ref reason) = self.reason {
            write!(f, " REASON \"{}\"", EscapeStr(reason))?;
        }
        if let Some(ref observed_at) = self.observed_at {
            write!(f, " OBSERVED AT \"{}\"", EscapeStr(observed_at))?;
        }
        if let Some(ref caused_by) = self.caused_by {
            write!(f, " CAUSED BY \"{}\"", EscapeStr(caused_by))?;
        }
        if let Some(ref namespace) = self.namespace {
            write_namespace(f, namespace)?;
        }
        Ok(())
    }
}

impl fmt::Display for InspectStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "INSPECT {}", self.target)
    }
}

impl fmt::Display for HistoryStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HISTORY {}", self.target)?;
        if let Some(ref namespace) = self.namespace {
            write_namespace(f, namespace)?;
        }
        Ok(())
    }
}

impl fmt::Display for TraceStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TRACE {}", self.target)
    }
}

impl fmt::Display for TemporalClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::After(ts) => write!(f, "AFTER \"{}\"", EscapeStr(ts)),
            Self::Before(ts) => write!(f, "BEFORE \"{}\"", EscapeStr(ts)),
            Self::Between { start, end } => {
                write!(
                    f,
                    "BETWEEN \"{}\" AND \"{}\"",
                    EscapeStr(start),
                    EscapeStr(end)
                )
            }
        }
    }
}

impl fmt::Display for ExpandClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EXPAND GRAPH DEPTH {}", self.depth)?;
        if let Some(mw) = self.min_weight {
            write!(f, " MIN_WEIGHT {}", FloatLit(f64::from(mw)))?;
        }
        if let Some(am) = self.activation {
            write!(f, " ACTIVATION {am}")?;
        }
        Ok(())
    }
}

impl fmt::Display for ActivationModeAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Static => write!(f, "static"),
            Self::Spreading => write!(f, "spreading"),
            Self::Ppr => write!(f, "ppr"),
        }
    }
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gt => write!(f, ">"),
            Self::Lt => write!(f, "<"),
            Self::Gte => write!(f, ">="),
            Self::Lte => write!(f, "<="),
            Self::Eq => write!(f, "="),
            Self::Neq => write!(f, "!="),
        }
    }
}

impl fmt::Display for ConditionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float(value) => write!(f, "{}", FloatLit(*value)),
            Self::Int(value) => write!(f, "{value}"),
            Self::String(value) => write!(f, "\"{}\"", EscapeStr(value)),
            Self::Param(value) => write!(f, "{value}"),
        }
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Narrative => write!(f, "narrative"),
            Self::Context => write!(f, "context"),
            Self::Graph => write!(f, "graph"),
            Self::CausalChain => write!(f, "causal_chain"),
            Self::Json => write!(f, "json"),
            Self::Csv => write!(f, "csv"),
            Self::Structured => write!(f, "structured"),
        }
    }
}

impl fmt::Display for WhereCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WHERE {} {} {}", self.field, self.op, self.value)
    }
}

impl fmt::Display for ConsistencyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Linearizable => write!(f, "linearizable"),
            Self::Eventual => write!(f, "eventual"),
            Self::Session => write!(f, "session"),
        }
    }
}

impl fmt::Display for AggFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count => write!(f, "COUNT"),
            Self::Avg => write!(f, "AVG"),
            Self::Sum => write!(f, "SUM"),
            Self::Min => write!(f, "MIN"),
            Self::Max => write!(f, "MAX"),
        }
    }
}

impl fmt::Display for Subquery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RECALL ")?;
        write_layer_filter(f, &self.layers)?;
        write!(f, " ABOUT {}", self.about)?;
        if let Some(ref inv) = self.involving {
            write!(f, " INVOLVING ")?;
            write_string_list(f, inv)?;
        }
        if let Some(ref tc) = self.temporal {
            write!(f, " {tc}")?;
        }
        if let Some(n) = self.limit {
            write!(f, " LIMIT {n}")?;
        }
        Ok(())
    }
}

// ── TRAVERSE ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct TraverseStmt {
    pub from: String,
    pub via: Option<Vec<String>>,
    pub depth: usize,
    pub where_clauses: Vec<WhereCondition>,
    pub limit: Option<usize>,
    /// Namespace isolation for traversal results (F-SEC-11).
    pub namespace: Option<String>,
}
impl fmt::Display for TraverseStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TRAVERSE FROM \"{}\"", EscapeStr(&self.from))?;
        if let Some(ref via) = self.via {
            write!(f, " VIA {}", via.join(", "))?;
        }
        write!(f, " DEPTH {}", self.depth)?;
        for wc in &self.where_clauses {
            write!(f, " {wc}")?;
        }
        if let Some(ref ns) = self.namespace {
            write_namespace(f, ns)?;
        }
        if let Some(n) = self.limit {
            write!(f, " LIMIT {n}")?;
        }
        Ok(())
    }
}
/// Parse a case-insensitive edge relation name (e.g. `"related_to"`, `"causes"`) into an [`EdgeRelation`].
///
/// Uses `eq_ignore_ascii_case` for zero-allocation matching.
pub fn parse_edge_relation(s: &str) -> Option<EdgeRelation> {
    const TABLE: &[(&str, EdgeRelation)] = &[
        ("related_to", EdgeRelation::RelatedTo),
        ("relatedto", EdgeRelation::RelatedTo),
        ("causes", EdgeRelation::Causes),
        ("caused_by", EdgeRelation::CausedBy),
        ("causedby", EdgeRelation::CausedBy),
        ("derived_from", EdgeRelation::DerivedFrom),
        ("derivedfrom", EdgeRelation::DerivedFrom),
        ("contradicts", EdgeRelation::Contradicts),
        ("supports", EdgeRelation::Supports),
        ("temporal_next", EdgeRelation::TemporalNext),
        ("temporalnext", EdgeRelation::TemporalNext),
        ("part_of", EdgeRelation::PartOf),
        ("partof", EdgeRelation::PartOf),
        ("instance_of", EdgeRelation::InstanceOf),
        ("instanceof", EdgeRelation::InstanceOf),
        ("similar_to", EdgeRelation::SimilarTo),
        ("similarto", EdgeRelation::SimilarTo),
        ("inhibits", EdgeRelation::Inhibits),
        ("participates_in", EdgeRelation::ParticipatesIn),
        ("participatesin", EdgeRelation::ParticipatesIn),
    ];
    TABLE
        .iter()
        .find(|(k, _)| s.eq_ignore_ascii_case(k))
        .map(|(_, v)| *v)
}

// ── CREATE REALM / DROP REALM ─────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateRealmStmt {
    pub name: String,
    pub description: Option<String>,
}

impl fmt::Display for CreateRealmStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CREATE REALM \"{}\"", EscapeStr(&self.name))?;
        if let Some(ref desc) = self.description {
            write!(f, " DESCRIPTION \"{}\"", EscapeStr(desc))?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropRealmStmt {
    pub name: String,
    pub confirm: bool,
}

impl fmt::Display for DropRealmStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DROP REALM \"{}\"", EscapeStr(&self.name))?;
        if self.confirm {
            write!(f, " CONFIRM")?;
        }
        Ok(())
    }
}

// ── GRANT / REVOKE ────────────────────────────

/// Target of a GRANT/REVOKE: either a namespace or a realm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrantTarget {
    Namespace(String),
    Realm(String),
}

/// Principal reference: either an agent or a team.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrincipalRef {
    Agent(String),
    Team(String),
}

impl PrincipalRef {
    /// Get the principal identifier string.
    pub fn id(&self) -> &str {
        match self {
            Self::Agent(a) => a,
            Self::Team(t) => t,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrantStmt {
    pub actions: Vec<String>,
    pub target: GrantTarget,
    pub principal: PrincipalRef,
}

impl fmt::Display for GrantStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GRANT {}", self.actions.join(", "))?;
        match &self.target {
            GrantTarget::Namespace(ns) => write!(f, " ON NAMESPACE \"{}\"", EscapeStr(ns))?,
            GrantTarget::Realm(r) => write!(f, " ON REALM \"{}\"", EscapeStr(r))?,
        }
        match &self.principal {
            PrincipalRef::Agent(a) => write!(f, " TO AGENT \"{}\"", EscapeStr(a))?,
            PrincipalRef::Team(t) => write!(f, " TO TEAM \"{}\"", EscapeStr(t))?,
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevokeStmt {
    pub actions: Vec<String>,
    pub target: GrantTarget,
    pub principal: PrincipalRef,
}

impl fmt::Display for RevokeStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "REVOKE {}", self.actions.join(", "))?;
        match &self.target {
            GrantTarget::Namespace(ns) => write!(f, " ON NAMESPACE \"{}\"", EscapeStr(ns))?,
            GrantTarget::Realm(r) => write!(f, " ON REALM \"{}\"", EscapeStr(r))?,
        }
        match &self.principal {
            PrincipalRef::Agent(a) => write!(f, " FROM AGENT \"{}\"", EscapeStr(a))?,
            PrincipalRef::Team(t) => write!(f, " FROM TEAM \"{}\"", EscapeStr(t))?,
        }
        Ok(())
    }
}

// ── SHOW POLICIES / EXPLAIN POLICY ───────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShowPoliciesStmt {
    pub principal: Option<PrincipalRef>,
}

impl fmt::Display for ShowPoliciesStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SHOW POLICIES")?;
        if let Some(ref p) = self.principal {
            match p {
                PrincipalRef::Agent(a) => write!(f, " FOR AGENT \"{}\"", EscapeStr(a))?,
                PrincipalRef::Team(t) => write!(f, " FOR TEAM \"{}\"", EscapeStr(t))?,
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExplainPolicyStmt {
    pub principal: PrincipalRef,
    pub resource_type: String,
    pub resource_name: String,
    pub action: String,
}

impl fmt::Display for ExplainPolicyStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EXPLAIN POLICY FOR ")?;
        match &self.principal {
            PrincipalRef::Agent(a) => write!(f, "AGENT \"{}\"", EscapeStr(a))?,
            PrincipalRef::Team(t) => write!(f, "TEAM \"{}\"", EscapeStr(t))?,
        }
        write!(
            f,
            " ON {} \"{}\" ACTION {}",
            self.resource_type.to_uppercase(),
            EscapeStr(&self.resource_name),
            self.action
        )
    }
}

// ── SET TIER_POLICY ──────────────────────────

/// A value in a `SET TIER_POLICY` assignment.
#[derive(Debug, Clone, PartialEq)]
pub enum TierPolicyValue {
    /// A string literal, e.g., `'2h'`.
    Str(String),
    /// A floating point number, e.g., `0.7`.
    Float(f64),
    /// An integer, e.g., `3600`.
    Int(i64),
}

impl fmt::Display for TierPolicyValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Rendered double-quoted (a valid `string_literal`) with the same
            // escaping as every other string clause, so it round-trips cleanly.
            Self::Str(s) => write!(f, "\"{}\"", EscapeStr(s)),
            Self::Float(v) => write!(f, "{}", FloatLit(*v)),
            Self::Int(v) => write!(f, "{v}"),
        }
    }
}

/// `SET TIER_POLICY <field> = <value>`
#[derive(Debug, Clone, PartialEq)]
pub struct SetTierPolicyStmt {
    /// The policy field to set (e.g., `working_to_episodic_ttl`).
    pub field: String,
    /// The value to assign.
    pub value: TierPolicyValue,
}

impl fmt::Display for SetTierPolicyStmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SET TIER_POLICY {} = {}", self.field, self.value)
    }
}

/// Returns `true` if the string is a parameter placeholder (`$1`, `$name`).
pub fn is_param(s: &str) -> bool {
    s.starts_with('$')
        && s.len() > 1
        && s[1..]
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Extract all parameter placeholders from a statement.
///
/// Scans string fields that may hold `$param` references and returns
/// a deduped, sorted list of parameter names (including the `$` prefix).
pub fn collect_parameters(stmt: &Statement) -> Vec<String> {
    let mut params = Vec::new();

    fn check(s: &str, params: &mut Vec<String>) {
        if is_param(s) && !params.contains(&s.to_string()) {
            params.push(s.to_string());
        }
    }

    fn check_wheres(wcs: &[WhereCondition], params: &mut Vec<String>) {
        for wc in wcs {
            if let ConditionValue::Param(ref p) = wc.value
                && !params.contains(p)
            {
                params.push(p.clone());
            }
        }
    }

    fn check_str_param(s: &StringOrParam, params: &mut Vec<String>) {
        if let Some(name) = s.param_name()
            && !params.iter().any(|p| p == name)
        {
            params.push(name.to_string());
        }
    }

    match stmt {
        Statement::Recall(r) => {
            check_str_param(&r.about, &mut params);
            if let Some(ref topic) = r.topic {
                check(topic, &mut params);
            }
            if let Some(ref ns) = r.namespace {
                check(ns, &mut params);
            }
            for sf in &r.subquery_filters {
                check_str_param(&sf.subquery.about, &mut params);
            }
            check_wheres(&r.where_clauses, &mut params);
        }
        Statement::Think(t) => {
            check_str_param(&t.about, &mut params);
            if let Some(ref ns) = t.namespace {
                check(ns, &mut params);
            }
            check_wheres(&t.where_clauses, &mut params);
        }
        Statement::Correct(c) => {
            check(c.target.raw_value(), &mut params);
            if let Some(ref reason) = c.reason {
                check(reason, &mut params);
            }
            if let Some(ref observed_at) = c.observed_at {
                check(observed_at, &mut params);
            }
            if let Some(ref caused_by) = c.caused_by {
                check(caused_by, &mut params);
            }
        }
        Statement::Supersede(s) => {
            check(s.target.raw_value(), &mut params);
            if let Some(ref reason) = s.reason {
                check(reason, &mut params);
            }
            if let Some(ref observed_at) = s.observed_at {
                check(observed_at, &mut params);
            }
            if let Some(ref caused_by) = s.caused_by {
                check(caused_by, &mut params);
            }
        }
        Statement::MergeMemory(m) => {
            for source in &m.sources {
                check(source.raw_value(), &mut params);
            }
            check(m.target.raw_value(), &mut params);
            if let Some(ref reason) = m.reason {
                check(reason, &mut params);
            }
            if let Some(ref observed_at) = m.observed_at {
                check(observed_at, &mut params);
            }
            if let Some(ref caused_by) = m.caused_by {
                check(caused_by, &mut params);
            }
        }
        Statement::Retract(r) => {
            check(r.target.raw_value(), &mut params);
            if let Some(ref reason) = r.reason {
                check(reason, &mut params);
            }
            if let Some(ref observed_at) = r.observed_at {
                check(observed_at, &mut params);
            }
            if let Some(ref caused_by) = r.caused_by {
                check(caused_by, &mut params);
            }
        }
        Statement::Traverse(t) => {
            check(&t.from, &mut params);
            if let Some(ref ns) = t.namespace {
                check(ns, &mut params);
            }
            check_wheres(&t.where_clauses, &mut params);
        }
        Statement::Inspect(i) => check(i.target.raw_value(), &mut params),
        Statement::History(h) => check(h.target.raw_value(), &mut params),
        Statement::Trace(t) => check(t.target.raw_value(), &mut params),
        Statement::Explain(e) => return collect_parameters(&e.inner),
        Statement::RecallEvents(r) => {
            if let Some(ref entity) = r.entity_filter {
                check(entity, &mut params);
            }
            check_wheres(&r.where_clauses, &mut params);
        }
        Statement::CreateRealm(_)
        | Statement::DropRealm(_)
        | Statement::Grant(_)
        | Statement::Revoke(_)
        | Statement::ShowPolicies(_)
        | Statement::ExplainPolicy(_)
        | Statement::ShowCluster
        | Statement::SetTierPolicy(_) => {}
        Statement::ExplainCauses(e) => check(&e.target, &mut params),
        Statement::WhatIf(w) => {
            check(&w.intervention, &mut params);
            check(&w.outcome, &mut params);
        }
        Statement::Counterfactual(c) => {
            check(&c.antecedent, &mut params);
            check(&c.consequent, &mut params);
        }
    }

    params.sort();
    params
}

/// Bind parameter values into a statement **at the AST level**.
///
/// `values` maps parameter names (with `$` prefix) to their string
/// representations. Each `$param` placeholder in a string field is replaced
/// with its value, and each `ConditionValue::Param` becomes a typed value
/// (integer, float, or string, inferred from the value).
///
/// This is the safe alternative to textual substitution on the query source:
/// because values are placed into typed AST nodes (not spliced into the query
/// text), a value can never break out of a string literal to inject trailing
/// clauses, and there is no `$t` / `$t2` prefix-collision hazard.
///
/// Returns the sorted list of parameter names that had no provided value.
pub fn bind_parameters(stmt: &mut Statement, values: &HashMap<String, String>) -> Vec<String> {
    let mut missing = Vec::new();

    // Substitute a raw-`$name` string field.
    fn subst(s: &mut String, values: &HashMap<String, String>, missing: &mut Vec<String>) {
        if is_param(s) {
            match values.get(s.as_str()) {
                Some(v) => *s = v.clone(),
                None => {
                    if !missing.contains(s) {
                        missing.push(s.clone());
                    }
                }
            }
        }
    }

    // Bind a string-or-param position: an unbound `Param` becomes a `Literal`
    // once its value is supplied; a `Literal` (e.g. a quoted `"$q"`) is never
    // re-bound.
    fn subst_str(
        s: &mut StringOrParam,
        values: &HashMap<String, String>,
        missing: &mut Vec<String>,
    ) {
        if let StringOrParam::Param(name) = s {
            match values.get(name.as_str()) {
                Some(v) => *s = StringOrParam::Literal(v.clone()),
                None => {
                    if !missing.contains(name) {
                        missing.push(name.clone());
                    }
                }
            }
        }
    }

    fn subst_wheres(
        wcs: &mut [WhereCondition],
        values: &HashMap<String, String>,
        missing: &mut Vec<String>,
    ) {
        for wc in wcs {
            if let ConditionValue::Param(name) = &wc.value {
                match values.get(name) {
                    Some(v) => wc.value = bound_condition_value(v),
                    None => {
                        if !missing.contains(name) {
                            missing.push(name.clone());
                        }
                    }
                }
            }
        }
    }

    match stmt {
        Statement::Recall(r) => {
            subst_str(&mut r.about, values, &mut missing);
            if let Some(ref mut topic) = r.topic {
                subst(topic, values, &mut missing);
            }
            if let Some(ref mut ns) = r.namespace {
                subst(ns, values, &mut missing);
            }
            for sf in &mut r.subquery_filters {
                subst_str(&mut sf.subquery.about, values, &mut missing);
            }
            subst_wheres(&mut r.where_clauses, values, &mut missing);
        }
        Statement::Think(t) => {
            subst_str(&mut t.about, values, &mut missing);
            if let Some(ref mut ns) = t.namespace {
                subst(ns, values, &mut missing);
            }
            subst_wheres(&mut t.where_clauses, values, &mut missing);
        }
        Statement::Correct(c) => {
            subst_target(&mut c.target, values, &mut missing);
            subst_opt(&mut c.reason, values, &mut missing);
            subst_opt(&mut c.observed_at, values, &mut missing);
            subst_opt(&mut c.caused_by, values, &mut missing);
        }
        Statement::Supersede(s) => {
            subst_target(&mut s.target, values, &mut missing);
            subst_opt(&mut s.reason, values, &mut missing);
            subst_opt(&mut s.observed_at, values, &mut missing);
            subst_opt(&mut s.caused_by, values, &mut missing);
        }
        Statement::MergeMemory(m) => {
            for source in &mut m.sources {
                subst_target(source, values, &mut missing);
            }
            subst_target(&mut m.target, values, &mut missing);
            subst_opt(&mut m.reason, values, &mut missing);
            subst_opt(&mut m.observed_at, values, &mut missing);
            subst_opt(&mut m.caused_by, values, &mut missing);
        }
        Statement::Retract(r) => {
            subst_target(&mut r.target, values, &mut missing);
            subst_opt(&mut r.reason, values, &mut missing);
            subst_opt(&mut r.observed_at, values, &mut missing);
            subst_opt(&mut r.caused_by, values, &mut missing);
        }
        Statement::Traverse(t) => {
            subst(&mut t.from, values, &mut missing);
            if let Some(ref mut ns) = t.namespace {
                subst(ns, values, &mut missing);
            }
            subst_wheres(&mut t.where_clauses, values, &mut missing);
        }
        Statement::Inspect(i) => subst_target(&mut i.target, values, &mut missing),
        Statement::History(h) => subst_target(&mut h.target, values, &mut missing),
        Statement::Trace(t) => subst_target(&mut t.target, values, &mut missing),
        Statement::Explain(e) => return bind_parameters(&mut e.inner, values),
        Statement::RecallEvents(r) => {
            if let Some(ref mut entity) = r.entity_filter {
                subst(entity, values, &mut missing);
            }
            subst_wheres(&mut r.where_clauses, values, &mut missing);
        }
        Statement::ExplainCauses(e) => subst(&mut e.target, values, &mut missing),
        Statement::WhatIf(w) => {
            subst(&mut w.intervention, values, &mut missing);
            subst(&mut w.outcome, values, &mut missing);
        }
        Statement::Counterfactual(c) => {
            subst(&mut c.antecedent, values, &mut missing);
            subst(&mut c.consequent, values, &mut missing);
        }
        Statement::CreateRealm(_)
        | Statement::DropRealm(_)
        | Statement::Grant(_)
        | Statement::Revoke(_)
        | Statement::ShowPolicies(_)
        | Statement::ExplainPolicy(_)
        | Statement::ShowCluster
        | Statement::SetTierPolicy(_) => {}
    }

    missing.sort();
    missing
}

fn subst_opt(
    field: &mut Option<String>,
    values: &HashMap<String, String>,
    missing: &mut Vec<String>,
) {
    if let Some(s) = field {
        if is_param(s) {
            match values.get(s.as_str()) {
                Some(v) => *s = v.clone(),
                None => {
                    if !missing.contains(s) {
                        missing.push(s.clone());
                    }
                }
            }
        }
    }
}

fn subst_target(
    target: &mut SemanticTargetRef,
    values: &HashMap<String, String>,
    missing: &mut Vec<String>,
) {
    let raw = target.raw_value().to_string();
    if is_param(&raw) {
        match values.get(&raw) {
            Some(v) => target.set_raw_value(v.clone()),
            None => {
                if !missing.contains(&raw) {
                    missing.push(raw);
                }
            }
        }
    }
}

/// Infer a typed `ConditionValue` from a bound parameter's string value:
/// integer, then float, otherwise a string.
fn bound_condition_value(v: &str) -> ConditionValue {
    if let Ok(i) = v.parse::<i64>() {
        ConditionValue::Int(i)
    } else if let Ok(fl) = v.parse::<f64>() {
        ConditionValue::Float(fl)
    } else {
        ConditionValue::String(v.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn bind_parameters_substitutes_into_ast_and_resists_injection() {
        // A value crafted to break out of the string literal and inject a
        // trailing clause (`" NAMESPACE evil ...`). AST-level binding must place
        // it into the `about` node verbatim — it can never become a clause.
        let mut stmt = parse(r#"RECALL episodic ABOUT $q LIMIT 5"#).unwrap();
        let mut values = HashMap::new();
        values.insert(
            "$q".to_string(),
            r#"hello" NAMESPACE "attacker"#.to_string(),
        );
        let missing = bind_parameters(&mut stmt, &values);
        assert!(missing.is_empty(), "no missing params, got {missing:?}");

        // Re-serialize and re-parse: the injected text is still just the ABOUT
        // value, and the namespace was NOT set from the payload.
        let rendered = stmt.to_string();
        let reparsed = parse(&rendered).unwrap();
        match reparsed {
            Statement::Recall(r) => {
                assert_eq!(r.about, r#"hello" NAMESPACE "attacker"#);
                assert_eq!(r.namespace, None, "payload must not become a namespace");
                assert_eq!(r.limit, Some(5));
            }
            other => panic!("expected Recall, got {other:?}"),
        }
    }

    #[test]
    fn bind_parameters_reports_missing() {
        let mut stmt = parse(r#"RECALL episodic ABOUT $q WHERE importance > $t"#).unwrap();
        let values = HashMap::new();
        let mut missing = bind_parameters(&mut stmt, &values);
        missing.sort();
        assert_eq!(missing, vec!["$q".to_string(), "$t".to_string()]);
    }

    #[test]
    fn bind_parameters_typed_where_value() {
        let mut stmt = parse(r#"RECALL episodic ABOUT "x" WHERE importance > $t"#).unwrap();
        let mut values = HashMap::new();
        values.insert("$t".to_string(), "0.75".to_string());
        assert!(bind_parameters(&mut stmt, &values).is_empty());
        match stmt {
            Statement::Recall(r) => {
                assert_eq!(r.where_clauses[0].value, ConditionValue::Float(0.75));
            }
            _ => panic!("expected Recall"),
        }
    }

    #[test]
    fn display_recall() {
        let stmt = RecallStmt {
            layers: vec![Layer::Episodic, Layer::Semantic],
            about: "test query".into(),
            involving: None,
            temporal: None,
            expand: None,
            follow_causes: None,
            where_clauses: vec![],
            modality: None,
            resource_roles: None,
            hydration_modes: None,
            artifact_kinds: None,
            group_by: None,
            projection: None,
            output_format: None,
            result_format: None,
            as_of: None,
            subquery_filters: vec![],
            budget: None,
            namespace: None,
            consistency: None,
            limit: Some(10),
            hybrid: false,
            depth_mode: None,
            with_prospective: None,
            with_mcfa: None,
            with_conflicts: false,
            provenance_depth: None,
            topic: None,
            from_realms: None,
        };
        let s = stmt.to_string();
        assert!(s.starts_with("RECALL episodic, semantic ABOUT"));
        assert!(s.contains("LIMIT 10"));
    }

    #[test]
    fn display_all_variants() {
        // Verify every Statement variant has a working Display
        let stmts: Vec<Statement> = vec![
            Statement::Recall(Box::new(RecallStmt {
                layers: vec![Layer::Episodic],
                about: "x".into(),
                involving: None,
                temporal: None,
                expand: None,
                follow_causes: None,
                where_clauses: vec![],
                modality: None,
                resource_roles: None,
                hydration_modes: None,
                artifact_kinds: None,
                group_by: None,
                projection: None,
                output_format: None,
                result_format: None,
                as_of: None,
                subquery_filters: vec![],
                budget: None,
                namespace: None,
                consistency: None,
                limit: None,
                hybrid: false,
                depth_mode: None,
                with_prospective: None,
                with_mcfa: None,
                with_conflicts: false,
                provenance_depth: None,
                topic: None,
                from_realms: None,
            })),
            Statement::Think(Box::new(ThinkStmt {
                about: "y".into(),
                involving: None,
                temporal: None,
                expand: None,
                follow_causes: None,
                where_clauses: vec![],
                output_format: None,
                budget: Some(4096),
                namespace: None,
                consistency: None,
                limit: None,
                hybrid: false,
                mode: RetrievalMode::Local,
                community_depth: None,
                depth_mode: None,
                with_prospective: None,
                with_mcfa: None,
                provenance_depth: None,
                max_hops: None,
            })),
            Statement::Correct(CorrectStmt {
                target: SemanticTargetRef::Memory("some_id".into()),
                updates: vec![SetAssignment {
                    field: "description".into(),
                    value: SetValue::String("updated".into()),
                }],
                reason: Some("fix".into()),
                observed_at: None,
                caused_by: None,
                namespace: None,
            }),
            Statement::Supersede(SupersedeStmt {
                target: SemanticTargetRef::Memory("some_id".into()),
                updates: vec![SetAssignment {
                    field: "description".into(),
                    value: SetValue::String("replacement".into()),
                }],
                reason: Some("new authority".into()),
                observed_at: None,
                caused_by: None,
                namespace: None,
            }),
            Statement::MergeMemory(MergeMemoryStmt {
                sources: vec![
                    SemanticTargetRef::Memory("some_id".into()),
                    SemanticTargetRef::Logical("other_id".into()),
                ],
                target: SemanticTargetRef::Revision("target_id".into()),
                updates: vec![SetAssignment {
                    field: "confidence".into(),
                    value: SetValue::Float(0.92),
                }],
                reason: Some("deduplicate".into()),
                observed_at: None,
                caused_by: None,
                namespace: None,
            }),
            Statement::Retract(RetractStmt {
                target: SemanticTargetRef::Memory("some_id".into()),
                reason: Some("obsolete".into()),
                observed_at: None,
                caused_by: None,
                namespace: None,
            }),
            Statement::Inspect(InspectStmt {
                target: SemanticTargetRef::Logical("id".into()),
            }),
            Statement::History(HistoryStmt {
                target: SemanticTargetRef::Revision("id".into()),
                namespace: Some("team_a".into()),
            }),
            Statement::Trace(TraceStmt {
                target: SemanticTargetRef::Memory("id".into()),
            }),
        ];
        for s in &stmts {
            let text = s.to_string();
            assert!(!text.is_empty(), "Display for {s:?} should not be empty");
        }
    }

    #[test]
    fn parse_edge_relation_cases() {
        assert_eq!(
            parse_edge_relation("related_to"),
            Some(EdgeRelation::RelatedTo)
        );
        assert_eq!(parse_edge_relation("causes"), Some(EdgeRelation::Causes));
        assert_eq!(
            parse_edge_relation("CONTRADICTS"),
            Some(EdgeRelation::Contradicts)
        );
        assert_eq!(parse_edge_relation("unknown"), None);
    }
}
