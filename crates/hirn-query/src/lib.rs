//! `hirn-query` — HirnQL parser, AST, and compiler for the hirn cognitive
//! memory query language.
//!
//! This crate provides:
//! - **Parser:** Pest grammar + AST types
//! - **Compiler:** `TypedStatement` + DataFusion `LogicalPlan` compilation
//!
//! # Example
//! ```
//! use hirn_query::{parse, Statement};
//!
//! let stmt = parse(r#"RECALL episodic ABOUT "test" LIMIT 5"#).unwrap();
//! assert!(matches!(stmt, Statement::Recall(_)));
//! ```

pub mod compiler;
pub mod parser;

pub use compiler::pipeline::{CompiledPlan, PlanCache, QueryPipeline};
pub use compiler::plan_compiler::{ActivationRepr, HirnOp, HirnPlanNode, compile, query_hash};
pub use compiler::typed_ast::{
    AnalyzeContext, DepthMode, TypedCounterfactual, TypedExpand, TypedExplainCauses, TypedFilter,
    TypedFilterValue, TypedRecall, TypedRecallEvents, TypedStatement, TypedSubqueryFilter,
    TypedTemporalRange, TypedThink, TypedTraverse, TypedWhatIf, analyze,
};
pub use parser::ast;
pub use parser::ast::*;
pub use parser::{ParseError, QueryLimits, parse, parse_with_limits};
