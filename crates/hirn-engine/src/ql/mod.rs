//! HirnQL — the query language for hirn cognitive memory operations.
//!
//! Provides parsing, semantic analysis, planning, and execution of
//! declarative memory queries.
//!
//! # Compilation Pipeline
//!
//! ```text
//! HirnQL text → Parser → AST → typed_ast::analyze → LogicalPlan → Physical plan → Execute
//! ```
//!
//! All stages live in `hirn_query` (the single compiler stack); this module
//! re-exports the language surface and owns the prepared-statement front-end.
//! Compiled plans are cached by `hirn_query::PlanCache` (the single plan
//! cache, attached to the engine's `QueryPipeline`).
//!
//! # Example
//! ```ignore
//! // Preferred public entrypoint:
//! let result = db.ql().execute(r#"RECALL episodic ABOUT "test" LIMIT 5"#).await?;
//!
//! // Prepared statements bind values into the parsed AST and execute it
//! // directly on the same bridge — no re-serialization or re-parse:
//! let prepared = db.ql().prepare(r#"RECALL episodic ABOUT $1 LIMIT 5"#)?;
//! let result = db.ql().execute_prepared(&prepared, &params).await?;
//! ```

pub use hirn_query::ast;
pub use hirn_query::parser;

pub mod builder;
pub mod compiler;
pub mod context;
pub(crate) mod direct_support;
pub(crate) mod read_support;
pub(crate) mod results;

pub use ast::Statement;
pub use compiler::{CompileError, PreparedStatement, bind, prepare};
pub use hirn_query::compiler::validate::{AnalysisError, AnalysisErrorKind, validate as analyze};
pub use parser::{ParseError, parse};
pub use results::{
    AggregatedGroup, AggregatedResults, CausalQueryKind, CausalQueryResult, CausalRow,
    ConsolidatedResult, CorrectedResult, CreatedResult, ExplainResult, ForgottenResult,
    HistoryResult, MergedResult, PolicyResult, ProjectedRecord, QueryResult, RecordResults,
    RetractedResult, ScoreBreakdown, ScoredMemory, SemanticHistoryItem, SemanticRevisionEntry,
    SemanticRevisionSummary, SupersededResult, SvoEventResult, SvoEventResults, WatchAckResult,
    revision_query_result_to_json,
};
