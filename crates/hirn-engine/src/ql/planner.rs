//! Query planner — transforms AST into optimized execution plans.

use std::fmt;

use super::ast::*;
use crate::db::DbStats;

// ── Query plan types ───────────────────────────────────────────────────

/// An ordered execution plan derived from a parsed HirnQL statement.
#[derive(Debug, Clone, PartialEq)]
pub struct QueryPlan {
    pub steps: Vec<PlanStep>,
    pub verb: PlanVerb,
}

/// Identifies which top-level verb the plan is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanVerb {
    Recall,
    Think,
    Remember,
    Forget,
    Correct,
    Supersede,
    Merge,
    Retract,
    Connect,
    Inspect,
    History,
    Trace,
    Consolidate,
    Watch,
    Other,
}

/// A single step in the query plan, with an estimated cost.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanStep {
    pub op: PlanOp,
    pub estimated_cost: f64,
    /// Estimated rows surviving after this step (0 = unknown).
    pub estimated_cardinality: u64,
}

/// Recommended index strategy for a plan step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexHint {
    /// IVF-HNSW vector index (semantic similarity).
    IvfHnsw,
    /// BTree index (scalars, timestamps).
    BTree,
    /// Bitmap index (low-cardinality columns).
    Bitmap,
    /// Full-text search index (BM25/Tantivy).
    Fts,
    /// LabelList index (list columns like entities).
    LabelList,
    /// No specific index — sequential scan.
    SeqScan,
    /// Automatic (let LanceDB decide).
    Auto,
}

/// Operations that the executor can perform.
#[derive(Debug, Clone, PartialEq)]
pub enum PlanOp {
    /// Filter by memory layer (cheapest filter).
    LayerFilter {
        layers: Vec<hirn_core::types::Layer>,
    },
    /// Filter by namespace.
    NamespaceFilter { namespace: String },
    /// Temporal range filter on records.
    TemporalFilter { temporal: TemporalClause },
    /// Filter by importance threshold.
    ImportanceFilter { op: ComparisonOp, threshold: f64 },
    /// Filter by confidence threshold.
    ConfidenceFilter { op: ComparisonOp, threshold: f64 },
    /// Filter by entity mention (INVOLVING clause).
    EntityFilter { entities: Vec<String> },
    /// Generic WHERE condition filter.
    ConditionFilter { condition: WhereCondition },
    /// HNSW vector similarity search.
    VectorSearch {
        query: String,
        limit: usize,
        index_hint: IndexHint,
    },
    /// Graph expansion from candidate set.
    GraphExpand {
        depth: usize,
        min_weight: Option<f32>,
        activation: Option<ActivationModeAst>,
    },
    /// Spreading activation pass.
    ActivationPass {
        decay: f64,
        inhibition: f64,
        max_iter: usize,
    },
    /// Causal chain traversal.
    CausalTraverse { depth: usize },
    /// Score and rank results.
    ScoreAndRank,
    /// Aggregate results (GROUP BY).
    Aggregate {
        field: String,
        function: AggFunction,
    },
    /// Project specific fields (SELECT).
    Project { fields: Vec<String> },
    /// Format output (FORMAT json/csv).
    FormatOutput { format: OutputFormat },
    /// Assemble context within token budget.
    ContextAssemble { budget: usize, format: OutputFormat },
    /// Limit result count.
    LimitResults { n: usize },
    /// Resolve a logical or revision ID to the active semantic head.
    ResolveActiveHead { target: String },
    /// Correct a semantic record by appending a new revision.
    CorrectRecord { target: String, fields: Vec<String> },
    /// Supersede a semantic record with a new authoritative revision.
    SupersedeRecord { target: String, fields: Vec<String> },
    /// Merge one or more semantic memories into an active target chain.
    MergeMemory {
        target: String,
        sources: Vec<String>,
        fields: Vec<String>,
    },
    /// Retract a semantic record by appending a tombstone revision.
    RetractRecord { target: String },
    /// Inspect a record's metadata.
    InspectRecord { target: String },
    /// Load a semantic record's revision chain.
    HistoryRecord { target: String },
    /// Trace a record's provenance.
    TraceProvenance { target: String },
    /// Resolve a subquery and filter outer results by field membership.
    SubqueryResolve { field: String, subquery: Subquery },
    /// Time-travel snapshot filter (AS OF clause).
    TimeTravelFilter { snapshot: RecallSnapshotAst },
}

// ── Planning ───────────────────────────────────────────────────────────

/// Generate an optimized query plan from a statement.
///
/// The planner uses optional `DbStats` to reorder filter steps. If stats
/// are not available, a default ordering is used.
pub fn plan(stmt: &Statement, stats: Option<&DbStats>) -> QueryPlan {
    match stmt {
        Statement::Recall(r) => plan_recall(r, stats),
        Statement::Think(t) => plan_think(t, stats),
        Statement::Correct(c) => plan_correct(c),
        Statement::Supersede(s) => plan_supersede(s),
        Statement::MergeMemory(m) => plan_merge_memory(m),
        Statement::Retract(r) => plan_retract(r),
        Statement::Inspect(i) => plan_inspect(i),
        Statement::History(h) => plan_history(h),
        Statement::Trace(t) => plan_trace(t),
        Statement::Traverse(t) => plan_traverse(t),
        Statement::Explain(e) => plan(&e.inner, stats),
        Statement::CreateRealm(_)
        | Statement::DropRealm(_)
        | Statement::Grant(_)
        | Statement::Revoke(_)
        | Statement::ShowPolicies(_)
        | Statement::ExplainPolicy(_)
        | Statement::RecallEvents(_)
        | Statement::ShowCluster
        | Statement::SetTierPolicy(_)
        | Statement::ExplainCauses(_)
        | Statement::WhatIf(_)
        | Statement::Counterfactual(_) => QueryPlan {
            verb: PlanVerb::Other,
            steps: Vec::new(),
        },
    }
}

fn plan_recall(r: &RecallStmt, stats: Option<&DbStats>) -> QueryPlan {
    let total = stats.map(|s| s.total_count).unwrap_or(0);
    let mut remaining = total;
    let mut steps = Vec::new();

    // 1. Layer filter — always first (cheapest, uses dataset partitioning).
    let layer_cardinality = estimate_layer_cardinality(&r.layers, stats);
    remaining = remaining.min(layer_cardinality);
    steps.push(PlanStep {
        op: PlanOp::LayerFilter {
            layers: r.layers.clone(),
        },
        estimated_cost: 0.01,
        estimated_cardinality: remaining,
    });

    // 2. Namespace filter — cheap metadata check (Bitmap-indexed).
    if let Some(ref ns) = r.namespace {
        remaining = (remaining as f64 * 0.3) as u64; // ~30% selectivity heuristic
        steps.push(PlanStep {
            op: PlanOp::NamespaceFilter {
                namespace: ns.clone(),
            },
            estimated_cost: 0.02,
            estimated_cardinality: remaining,
        });
    }

    // 3. Subquery filters (WHERE field IN (RECALL ...)) — resolve inner query first.
    for sf in &r.subquery_filters {
        remaining = (remaining as f64 * 0.2) as u64; // ~20% selectivity heuristic
        steps.push(PlanStep {
            op: PlanOp::SubqueryResolve {
                field: sf.field.clone(),
                subquery: sf.subquery.clone(),
            },
            estimated_cost: 0.8, // subquery is expensive
            estimated_cardinality: remaining.max(1),
        });
    }

    // 4. Time-travel snapshot (AS OF clause).
    if let Some(ref snapshot) = r.as_of {
        remaining = (remaining as f64 * 0.6) as u64; // ~60% survive snapshot filter
        steps.push(PlanStep {
            op: PlanOp::TimeTravelFilter {
                snapshot: snapshot.clone(),
            },
            estimated_cost: 0.15,
            estimated_cardinality: remaining.max(1),
        });
    }

    // 5. Collect scalar filters with selectivity estimates, then sort.
    let mut scalar_filters: Vec<(PlanOp, f64, f64)> = Vec::new(); // (op, cost, selectivity)

    // Entity filter (INVOLVING clause) — uses LabelList index.
    if let Some(ref entities) = r.involving {
        // Entity filters are quite selective: each named entity narrows the set.
        // Heuristic: ~10% selectivity per entity, multiplicative.
        let sel = (0.1_f64).powi(entities.len() as i32).max(0.01);
        scalar_filters.push((
            PlanOp::EntityFilter {
                entities: entities.clone(),
            },
            0.08,
            sel,
        ));
    }

    if let Some(ref tc) = r.temporal {
        let sel = estimate_temporal_selectivity(tc, stats);
        scalar_filters.push((
            PlanOp::TemporalFilter {
                temporal: tc.clone(),
            },
            0.1,
            sel,
        ));
    }

    let mut general_conditions = Vec::new();
    for wc in &r.where_clauses {
        match wc.field.as_str() {
            "importance" => {
                if let ConditionValue::Float(v) = wc.value {
                    let sel = estimate_threshold_selectivity(v, &wc.op);
                    scalar_filters.push((
                        PlanOp::ImportanceFilter {
                            op: wc.op,
                            threshold: v,
                        },
                        0.05,
                        sel,
                    ));
                }
            }
            "confidence" => {
                if let ConditionValue::Float(v) = wc.value {
                    let sel = estimate_threshold_selectivity(v, &wc.op);
                    scalar_filters.push((
                        PlanOp::ConfidenceFilter {
                            op: wc.op,
                            threshold: v,
                        },
                        0.05,
                        sel,
                    ));
                }
            }
            _ => general_conditions.push(wc.clone()),
        }
    }

    // Sort scalar filters: most selective first (lowest selectivity = most rows eliminated).
    scalar_filters.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Decide: if the most selective scalar filter eliminates >70% of rows and we have
    // a large dataset, apply scalar filters BEFORE vector search.
    let has_highly_selective_scalar = scalar_filters
        .first()
        .map(|(_, _, sel)| *sel < 0.3)
        .unwrap_or(false);
    let scalar_before_vector = has_highly_selective_scalar && total > 100;

    if scalar_before_vector {
        for (op, cost, sel) in &scalar_filters {
            remaining = (remaining as f64 * sel) as u64;
            steps.push(PlanStep {
                op: op.clone(),
                estimated_cost: *cost,
                estimated_cardinality: remaining.max(1),
            });
        }
    }

    // Vector search — choose index hint based on dataset size.
    let limit = r.limit.unwrap_or(10);
    let index_hint = if total > 1000 {
        IndexHint::IvfHnsw
    } else {
        IndexHint::Auto
    };
    let vector_cardinality = limit as u64;
    steps.push(PlanStep {
        op: PlanOp::VectorSearch {
            query: r.about.clone(),
            limit,
            index_hint,
        },
        estimated_cost: if total > 1000 { 1.0 } else { 0.5 },
        estimated_cardinality: vector_cardinality,
    });
    remaining = remaining.min(vector_cardinality);

    // Add scalar filters after vector search if not already added.
    if !scalar_before_vector {
        for (op, cost, sel) in &scalar_filters {
            remaining = (remaining as f64 * sel) as u64;
            steps.push(PlanStep {
                op: op.clone(),
                estimated_cost: *cost,
                estimated_cardinality: remaining.max(1),
            });
        }
    }

    // General condition filters.
    for cond in general_conditions {
        remaining = (remaining as f64 * 0.5) as u64; // default 50% selectivity
        steps.push(PlanStep {
            op: PlanOp::ConditionFilter { condition: cond },
            estimated_cost: 0.05,
            estimated_cardinality: remaining.max(1),
        });
    }

    // Graph expansion.
    if let Some(ref ex) = r.expand {
        let graph_output = remaining * (ex.depth as u64 + 1);
        steps.push(PlanStep {
            op: PlanOp::GraphExpand {
                depth: ex.depth,
                min_weight: ex.min_weight,
                activation: ex.activation,
            },
            estimated_cost: compute_graph_cost(ex.depth),
            estimated_cardinality: graph_output,
        });
        remaining = graph_output;
    }

    // Causal traversal.
    if let Some(depth) = r.follow_causes {
        steps.push(PlanStep {
            op: PlanOp::CausalTraverse { depth },
            estimated_cost: compute_graph_cost(depth),
            estimated_cardinality: remaining,
        });
    }

    // Score and rank.
    steps.push(PlanStep {
        op: PlanOp::ScoreAndRank,
        estimated_cost: 0.1,
        estimated_cardinality: remaining,
    });

    // Aggregation (GROUP BY).
    if let Some(ref gb) = r.group_by {
        // Aggregation reduces cardinality to the number of distinct groups.
        let group_count = (remaining as f64 * 0.1).max(1.0) as u64;
        steps.push(PlanStep {
            op: PlanOp::Aggregate {
                field: gb.field.clone(),
                function: gb.function,
            },
            estimated_cost: 0.15,
            estimated_cardinality: group_count,
        });
        remaining = group_count;
    }

    // Projection (SELECT).
    if let Some(ref fields) = r.projection {
        steps.push(PlanStep {
            op: PlanOp::Project {
                fields: fields.clone(),
            },
            estimated_cost: 0.02,
            estimated_cardinality: remaining,
        });
    }

    // Output format (FORMAT json/csv).
    if let Some(fmt) = r.result_format {
        steps.push(PlanStep {
            op: PlanOp::FormatOutput { format: fmt },
            estimated_cost: 0.05,
            estimated_cardinality: remaining,
        });
    }

    // Limit.
    let final_count = remaining.min(limit as u64);
    steps.push(PlanStep {
        op: PlanOp::LimitResults { n: limit },
        estimated_cost: 0.01,
        estimated_cardinality: final_count,
    });

    QueryPlan {
        steps,
        verb: PlanVerb::Recall,
    }
}

fn plan_think(t: &ThinkStmt, stats: Option<&DbStats>) -> QueryPlan {
    // Think is like Recall but with ContextAssemble at the end.
    let recall_equivalent = RecallStmt {
        layers: vec![
            hirn_core::types::Layer::Episodic,
            hirn_core::types::Layer::Semantic,
        ],
        about: t.about.clone(),
        involving: t.involving.clone(),
        temporal: t.temporal.clone(),
        expand: t.expand.clone(),
        follow_causes: t.follow_causes,
        where_clauses: t.where_clauses.clone(),
        modality: None,
        resource_roles: None,
        hydration_modes: None,
        artifact_kinds: None,
        group_by: None,
        projection: None,
        output_format: t.output_format,
        result_format: None,
        as_of: None,
        subquery_filters: vec![],
        budget: t.budget,
        namespace: t.namespace.clone(),
        consistency: t.consistency,
        limit: t.limit,
        hybrid: false,
        depth_mode: None,
        with_prospective: None,
        with_mcfa: None,
        with_conflicts: false,
        provenance_depth: None,
        topic: None,
        from_realms: None,
    };

    let mut plan = plan_recall(&recall_equivalent, stats);
    plan.verb = PlanVerb::Think;

    // Add context assembly step before the final limit.
    let budget = t.budget.unwrap_or(4096);
    let format = t.output_format.unwrap_or(OutputFormat::Context);
    let cardinality = plan
        .steps
        .last()
        .map(|s| s.estimated_cardinality)
        .unwrap_or(0);
    let assemble = PlanStep {
        op: PlanOp::ContextAssemble { budget, format },
        estimated_cost: 0.5,
        estimated_cardinality: cardinality,
    };

    // Insert before LimitResults.
    if let Some(pos) = plan
        .steps
        .iter()
        .position(|s| matches!(s.op, PlanOp::LimitResults { .. }))
    {
        plan.steps.insert(pos, assemble);
    } else {
        plan.steps.push(assemble);
    }

    plan
}

fn plan_correct(c: &CorrectStmt) -> QueryPlan {
    QueryPlan {
        steps: vec![
            PlanStep {
                op: PlanOp::ResolveActiveHead {
                    target: c.target.to_string(),
                },
                estimated_cost: 0.1,
                estimated_cardinality: 1,
            },
            PlanStep {
                op: PlanOp::CorrectRecord {
                    target: c.target.to_string(),
                    fields: c
                        .updates
                        .iter()
                        .map(|update| update.field.clone())
                        .collect(),
                },
                estimated_cost: 0.2,
                estimated_cardinality: 1,
            },
        ],
        verb: PlanVerb::Correct,
    }
}

fn plan_supersede(s: &SupersedeStmt) -> QueryPlan {
    QueryPlan {
        steps: vec![
            PlanStep {
                op: PlanOp::ResolveActiveHead {
                    target: s.target.to_string(),
                },
                estimated_cost: 0.1,
                estimated_cardinality: 1,
            },
            PlanStep {
                op: PlanOp::SupersedeRecord {
                    target: s.target.to_string(),
                    fields: s
                        .updates
                        .iter()
                        .map(|update| update.field.clone())
                        .collect(),
                },
                estimated_cost: 0.2,
                estimated_cardinality: 1,
            },
        ],
        verb: PlanVerb::Supersede,
    }
}

fn plan_merge_memory(m: &MergeMemoryStmt) -> QueryPlan {
    QueryPlan {
        steps: vec![
            PlanStep {
                op: PlanOp::ResolveActiveHead {
                    target: m.target.to_string(),
                },
                estimated_cost: 0.1,
                estimated_cardinality: 1,
            },
            PlanStep {
                op: PlanOp::MergeMemory {
                    target: m.target.to_string(),
                    sources: m.sources.iter().map(ToString::to_string).collect(),
                    fields: m
                        .updates
                        .iter()
                        .map(|update| update.field.clone())
                        .collect(),
                },
                estimated_cost: 0.3,
                estimated_cardinality: 1,
            },
        ],
        verb: PlanVerb::Merge,
    }
}

fn plan_retract(r: &RetractStmt) -> QueryPlan {
    QueryPlan {
        steps: vec![
            PlanStep {
                op: PlanOp::ResolveActiveHead {
                    target: r.target.to_string(),
                },
                estimated_cost: 0.1,
                estimated_cardinality: 1,
            },
            PlanStep {
                op: PlanOp::RetractRecord {
                    target: r.target.to_string(),
                },
                estimated_cost: 0.2,
                estimated_cardinality: 1,
            },
        ],
        verb: PlanVerb::Retract,
    }
}

fn plan_traverse(t: &TraverseStmt) -> QueryPlan {
    let mut steps = vec![];

    // Start node lookup.
    steps.push(PlanStep {
        op: PlanOp::VectorSearch {
            query: t.from.clone(),
            limit: 1,
            index_hint: IndexHint::SeqScan,
        },
        estimated_cost: 0.5,
        estimated_cardinality: 1,
    });

    // Graph expansion.
    steps.push(PlanStep {
        op: PlanOp::GraphExpand {
            depth: t.depth,
            min_weight: None,
            activation: None,
        },
        estimated_cost: 0.5 * t.depth as f64,
        estimated_cardinality: (t.depth * 5) as u64,
    });

    // WHERE filters.
    for wc in &t.where_clauses {
        steps.push(PlanStep {
            op: PlanOp::ConditionFilter {
                condition: wc.clone(),
            },
            estimated_cost: 0.2,
            estimated_cardinality: (t.depth * 3) as u64,
        });
    }

    // LIMIT.
    if let Some(n) = t.limit {
        steps.push(PlanStep {
            op: PlanOp::LimitResults { n },
            estimated_cost: 0.0,
            estimated_cardinality: n as u64,
        });
    }

    QueryPlan {
        steps,
        verb: PlanVerb::Recall, // traverse uses recall-like verb
    }
}

fn plan_inspect(i: &InspectStmt) -> QueryPlan {
    QueryPlan {
        steps: vec![PlanStep {
            op: PlanOp::InspectRecord {
                target: i.target.to_string(),
            },
            estimated_cost: 0.1,
            estimated_cardinality: 1,
        }],
        verb: PlanVerb::Inspect,
    }
}

fn plan_history(h: &HistoryStmt) -> QueryPlan {
    let mut steps = vec![PlanStep {
        op: PlanOp::HistoryRecord {
            target: h.target.to_string(),
        },
        estimated_cost: 0.15,
        estimated_cardinality: 1,
    }];

    if let Some(namespace) = &h.namespace {
        steps.insert(
            0,
            PlanStep {
                op: PlanOp::NamespaceFilter {
                    namespace: namespace.clone(),
                },
                estimated_cost: 0.05,
                estimated_cardinality: 1,
            },
        );
    }

    QueryPlan {
        steps,
        verb: PlanVerb::History,
    }
}

fn plan_trace(t: &TraceStmt) -> QueryPlan {
    QueryPlan {
        steps: vec![PlanStep {
            op: PlanOp::TraceProvenance {
                target: t.target.to_string(),
            },
            estimated_cost: 0.1,
            estimated_cardinality: 1,
        }],
        verb: PlanVerb::Trace,
    }
}

// ── Cost estimation helpers ────────────────────────────────────────────

/// Estimate the cardinality after layer filtering.
fn estimate_layer_cardinality(layers: &[hirn_core::types::Layer], stats: Option<&DbStats>) -> u64 {
    let Some(s) = stats else { return 0 };
    let mut total = 0u64;
    for l in layers {
        match l {
            hirn_core::types::Layer::Episodic => total += s.episodic_count,
            hirn_core::types::Layer::Semantic => total += s.semantic_count,
            hirn_core::types::Layer::Working => total += s.working_count,
            hirn_core::types::Layer::Procedural => total += 0, // Not tracked in DbStats
        }
    }
    total
}

/// Estimate temporal filter selectivity (0.0 = eliminates everything, 1.0 = keeps everything).
fn estimate_temporal_selectivity(tc: &TemporalClause, stats: Option<&DbStats>) -> f64 {
    let Some(stats) = stats else { return 0.5 };
    if stats.total_count == 0 {
        return 1.0;
    }

    // Heuristic: BETWEEN is the most selective, AFTER/BEFORE are moderate.
    // With real stats we'd use time distribution; for now, use conservative estimates.
    match tc {
        TemporalClause::Between { .. } => 0.2, // ~20% of records in range
        TemporalClause::After(_) => 0.4,       // ~40% after the timestamp
        TemporalClause::Before(_) => 0.4,      // ~40% before the timestamp
    }
}

/// Estimate selectivity for threshold-based filters (importance > X, confidence > X).
fn estimate_threshold_selectivity(threshold: f64, op: &ComparisonOp) -> f64 {
    // Assume uniform distribution [0, 1] for importance/confidence.
    match op {
        ComparisonOp::Gt | ComparisonOp::Gte => (1.0 - threshold).max(0.01),
        ComparisonOp::Lt | ComparisonOp::Lte => threshold.max(0.01),
        ComparisonOp::Eq => 0.05,  // Exact match is rare
        ComparisonOp::Neq => 0.95, // Most records won't match exactly
    }
}

fn compute_graph_cost(depth: usize) -> f64 {
    // Exponential cost estimate for graph traversal.
    (depth as f64).powi(2) * 0.5
}

// ── Display ────────────────────────────────────────────────────────────

impl fmt::Display for QueryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QueryPlan ({:?}):", self.verb)?;
        for (i, step) in self.steps.iter().enumerate() {
            writeln!(
                f,
                "  Step {}: {} (est. cost: {:.3}, est. rows: {})",
                i + 1,
                step.op,
                step.estimated_cost,
                step.estimated_cardinality,
            )?;
        }
        let total: f64 = self.steps.iter().map(|s| s.estimated_cost).sum();
        writeln!(f, "  Total estimated cost: {total:.3}")?;
        Ok(())
    }
}

impl fmt::Display for PlanOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LayerFilter { layers } => {
                let names: Vec<&str> = layers
                    .iter()
                    .map(|l| match l {
                        hirn_core::types::Layer::Episodic => "episodic",
                        hirn_core::types::Layer::Semantic => "semantic",
                        hirn_core::types::Layer::Working => "working",
                        hirn_core::types::Layer::Procedural => "procedural",
                    })
                    .collect();
                write!(f, "LayerFilter({})", names.join(", "))
            }
            Self::NamespaceFilter { namespace } => write!(f, "NamespaceFilter({namespace})"),
            Self::TemporalFilter { temporal } => write!(f, "TemporalFilter({temporal})"),
            Self::ImportanceFilter { op, threshold } => {
                write!(f, "ImportanceFilter({op} {threshold})")
            }
            Self::ConfidenceFilter { op, threshold } => {
                write!(f, "ConfidenceFilter({op} {threshold})")
            }
            Self::ConditionFilter { condition } => write!(f, "ConditionFilter({condition})"),
            Self::EntityFilter { entities } => {
                write!(f, "EntityFilter({})", entities.join(", "))
            }
            Self::VectorSearch {
                query,
                limit,
                index_hint,
            } => {
                write!(
                    f,
                    "VectorSearch(\"{query}\", limit={limit}, index={index_hint:?})"
                )
            }
            Self::GraphExpand {
                depth,
                min_weight,
                activation,
            } => {
                write!(f, "GraphExpand(depth={depth}")?;
                if let Some(mw) = min_weight {
                    write!(f, ", min_weight={mw}")?;
                }
                if let Some(am) = activation {
                    write!(f, ", activation={am}")?;
                }
                write!(f, ")")
            }
            Self::ActivationPass {
                decay,
                inhibition,
                max_iter,
            } => write!(
                f,
                "ActivationPass(decay={decay}, inhibition={inhibition}, max_iter={max_iter})"
            ),
            Self::CausalTraverse { depth } => write!(f, "CausalTraverse(depth={depth})"),
            Self::ScoreAndRank => write!(f, "ScoreAndRank"),
            Self::Aggregate { field, function } => {
                write!(f, "Aggregate(GROUP BY {field} {function})")
            }
            Self::Project { fields } => write!(f, "Project({})", fields.join(", ")),
            Self::FormatOutput { format } => write!(f, "FormatOutput({format})"),
            Self::ContextAssemble { budget, format } => {
                write!(f, "ContextAssemble(budget={budget}, format={format})")
            }
            Self::LimitResults { n } => write!(f, "LimitResults({n})"),
            Self::ResolveActiveHead { target } => write!(f, "ResolveActiveHead({target})"),
            Self::CorrectRecord { target, fields } => {
                write!(f, "CorrectRecord({target}; fields={})", fields.join(", "))
            }
            Self::SupersedeRecord { target, fields } => {
                write!(f, "SupersedeRecord({target}; fields={})", fields.join(", "))
            }
            Self::MergeMemory {
                target,
                sources,
                fields,
            } => write!(
                f,
                "MergeMemory(target={target}; sources={}; fields={})",
                sources.join(", "),
                fields.join(", ")
            ),
            Self::RetractRecord { target } => write!(f, "RetractRecord({target})"),
            Self::InspectRecord { target } => write!(f, "InspectRecord({target})"),
            Self::HistoryRecord { target } => write!(f, "HistoryRecord({target})"),
            Self::TraceProvenance { target } => write!(f, "TraceProvenance({target})"),
            Self::SubqueryResolve { field, .. } => write!(f, "SubqueryResolve({field})"),
            Self::TimeTravelFilter { snapshot } => {
                write!(f, "TimeTravelFilter(AS OF {snapshot})")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ql::parser;

    fn empty_stats() -> DbStats {
        DbStats {
            working_count: 0,
            episodic_count: 0,
            semantic_count: 0,
            edge_count: 0,
            procedural_count: 0,
            total_count: 0,
            file_size_bytes: 0,
        }
    }

    fn large_stats() -> DbStats {
        DbStats {
            working_count: 10,
            episodic_count: 5000,
            semantic_count: 2000,
            edge_count: 0,
            procedural_count: 0,
            total_count: 7010,
            file_size_bytes: 100_000_000,
        }
    }

    #[test]
    fn simple_recall_plan() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let p = plan(&stmt, None);
        assert_eq!(p.verb, PlanVerb::Recall);

        let op_names: Vec<String> = p.steps.iter().map(|s| format!("{}", s.op)).collect();
        assert!(op_names[0].starts_with("LayerFilter"));
        assert!(op_names.iter().any(|o| o.starts_with("VectorSearch")));
        assert!(op_names.iter().any(|o| o.starts_with("ScoreAndRank")));
        assert!(op_names.last().unwrap().starts_with("LimitResults"));
    }

    #[test]
    fn recall_with_expand_includes_graph_step() {
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 2 ACTIVATION spreading"#,
        )
        .unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::GraphExpand { .. }))
        );
    }

    #[test]
    fn think_includes_context_assemble() {
        let stmt = parser::parse(r#"THINK ABOUT "optimize" BUDGET 4096"#).unwrap();
        let p = plan(&stmt, None);
        assert_eq!(p.verb, PlanVerb::Think);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::ContextAssemble { .. }))
        );
    }

    #[test]
    fn plan_display_readable() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test" LIMIT 5"#).unwrap();
        let p = plan(&stmt, None);
        let display = format!("{p}");
        assert!(display.contains("QueryPlan"));
        assert!(display.contains("Step 1"));
        assert!(display.contains("Total estimated cost"));
    }

    #[test]
    fn highly_selective_temporal_before_vector_with_large_stats() {
        // BETWEEN has selectivity 0.2 (< 0.3 threshold), so it's placed before vector search.
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "test" BETWEEN "2026-03-01" AND "2026-03-14" LIMIT 5"#,
        )
        .unwrap();
        let p = plan(&stmt, Some(&large_stats()));

        let temporal_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::TemporalFilter { .. }))
            .unwrap();
        let vector_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();
        assert!(
            temporal_pos < vector_pos,
            "highly selective temporal filter should run before vector search with large dataset"
        );
    }

    #[test]
    fn moderate_temporal_after_vector_with_large_stats() {
        // AFTER has selectivity 0.4 (>= 0.3 threshold), so it's placed after vector search.
        let stmt =
            parser::parse(r#"RECALL episodic ABOUT "test" AFTER "2026-03-14" LIMIT 5"#).unwrap();
        let p = plan(&stmt, Some(&large_stats()));

        let temporal_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::TemporalFilter { .. }))
            .unwrap();
        let vector_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();
        assert!(
            temporal_pos > vector_pos,
            "moderate-selectivity temporal filter should run after vector search"
        );
    }

    #[test]
    fn wide_temporal_after_vector_with_small_stats() {
        let stmt =
            parser::parse(r#"RECALL episodic ABOUT "test" AFTER "2020-01-01" LIMIT 5"#).unwrap();
        let p = plan(&stmt, Some(&empty_stats()));

        // With small stats (0 records), temporal shouldn't be prioritized.
        let temporal_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::TemporalFilter { .. }));
        let vector_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();

        if let Some(tp) = temporal_pos {
            assert!(
                tp > vector_pos,
                "temporal filter should run after vector search with small dataset"
            );
        }
    }

    #[test]
    fn layer_filter_always_first() {
        let queries = [
            r#"RECALL episodic ABOUT "x""#,
            r#"RECALL semantic ABOUT "y" LIMIT 5"#,
            r#"RECALL episodic ABOUT "z" EXPAND GRAPH DEPTH 2"#,
        ];
        for q in queries {
            let stmt = parser::parse(q).unwrap();
            let p = plan(&stmt, None);
            assert!(
                matches!(p.steps[0].op, PlanOp::LayerFilter { .. }),
                "LayerFilter must be first step for: {q}"
            );
        }
    }

    #[test]
    fn explain_no_side_effects() {
        // plan() is pure — calling it does not modify any database state.
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let p1 = plan(&stmt, None);
        let p2 = plan(&stmt, None);
        assert_eq!(p1, p2);
    }

    #[test]
    fn importance_filter_before_graph() {
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "test" EXPAND GRAPH DEPTH 2 WHERE importance > 0.5"#,
        )
        .unwrap();
        let p = plan(&stmt, None);

        let imp_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::ImportanceFilter { .. }))
            .unwrap();
        let graph_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::GraphExpand { .. }))
            .unwrap();
        assert!(
            imp_pos < graph_pos,
            "importance filter should run before graph expansion"
        );
    }

    // ── Cost-Based Optimizer tests ─────────────────────────

    #[test]
    fn entity_filter_before_vector_with_large_stats() {
        // INVOLVING "JWT" has selectivity 0.1 (< 0.3 threshold) on large dataset.
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "auth" INVOLVING "JWT" AFTER "2026-03-01" LIMIT 5"#,
        )
        .unwrap();
        let p = plan(&stmt, Some(&large_stats()));

        let entity_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::EntityFilter { .. }))
            .unwrap();
        let vector_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();
        assert!(
            entity_pos < vector_pos,
            "entity filter should be placed before vector search (high selectivity)"
        );
    }

    #[test]
    fn explain_shows_estimated_cardinalities() {
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "auth" INVOLVING "JWT" AFTER "2026-03-01" LIMIT 5"#,
        )
        .unwrap();
        let p = plan(&stmt, Some(&large_stats()));
        let display = format!("{p}");

        // Each step should include estimated rows.
        assert!(
            display.contains("est. rows"),
            "plan display should show estimated rows"
        );
        // Should have multiple steps with cardinality > 0.
        for step in &p.steps {
            assert!(
                step.estimated_cardinality > 0,
                "cardinality should be > 0: {:?}",
                step.op
            );
        }
    }

    #[test]
    fn plan_changes_with_data_distribution() {
        // With empty stats, scalar filters go after vector search.
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "test" BETWEEN "2026-03-01" AND "2026-03-14" LIMIT 5"#,
        )
        .unwrap();
        let plan_empty = plan(&stmt, Some(&empty_stats()));
        let plan_large = plan(&stmt, Some(&large_stats()));

        // With large stats, temporal (BETWEEN, selectivity 0.2) goes before vector.
        let temporal_pos_large = plan_large
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::TemporalFilter { .. }))
            .unwrap();
        let vector_pos_large = plan_large
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();
        assert!(temporal_pos_large < vector_pos_large);

        // With empty stats, temporal goes after vector.
        let temporal_pos_empty = plan_empty
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::TemporalFilter { .. }))
            .unwrap();
        let vector_pos_empty = plan_empty
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();
        assert!(temporal_pos_empty > vector_pos_empty);
    }

    #[test]
    fn ivf_hnsw_for_large_dataset() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let p = plan(&stmt, Some(&large_stats()));
        let vs = p
            .steps
            .iter()
            .find(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();
        match &vs.op {
            PlanOp::VectorSearch { index_hint, .. } => {
                assert_eq!(*index_hint, IndexHint::IvfHnsw);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn auto_index_for_small_dataset() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let p = plan(&stmt, Some(&empty_stats()));
        let vs = p
            .steps
            .iter()
            .find(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
            .unwrap();
        match &vs.op {
            PlanOp::VectorSearch { index_hint, .. } => {
                assert_eq!(*index_hint, IndexHint::Auto);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn hybrid_entity_and_vector_plan() {
        // Query with both INVOLVING (entity) and ABOUT (vector).
        let stmt =
            parser::parse(r#"RECALL episodic ABOUT "auth" INVOLVING "JWT" LIMIT 5"#).unwrap();
        let p = plan(&stmt, Some(&large_stats()));

        // Both entity filter and vector search should be present.
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::EntityFilter { .. })),
            "should have entity filter"
        );
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::VectorSearch { .. })),
            "should have vector search"
        );
    }

    // ── Aggregation & Projection plan ops ──────────────────

    #[test]
    fn group_by_produces_aggregate_step() {
        let stmt =
            parser::parse(r#"RECALL episodic ABOUT "test" GROUP BY entity_type COUNT"#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::Aggregate { .. })),
            "should have Aggregate step"
        );
        // Aggregate should come after ScoreAndRank.
        let agg_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::Aggregate { .. }))
            .unwrap();
        let rank_pos = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::ScoreAndRank))
            .unwrap();
        assert!(agg_pos > rank_pos);
    }

    #[test]
    fn select_produces_project_step() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test" SELECT id, summary, importance"#)
            .unwrap();
        let p = plan(&stmt, None);
        let proj = p
            .steps
            .iter()
            .find(|s| matches!(s.op, PlanOp::Project { .. }))
            .unwrap();
        match &proj.op {
            PlanOp::Project { fields } => {
                assert_eq!(fields, &["id", "summary", "importance"]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn format_json_produces_format_step() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test" FORMAT json"#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps.iter().any(|s| matches!(
                s.op,
                PlanOp::FormatOutput {
                    format: OutputFormat::Json
                }
            )),
            "should have FormatOutput(Json) step"
        );
    }

    #[test]
    fn format_csv_produces_format_step() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test" FORMAT csv"#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps.iter().any(|s| matches!(
                s.op,
                PlanOp::FormatOutput {
                    format: OutputFormat::Csv
                }
            )),
            "should have FormatOutput(Csv) step"
        );
    }

    #[test]
    fn no_group_by_no_aggregate_step() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            !p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::Aggregate { .. })),
            "should not have Aggregate step without GROUP BY"
        );
    }

    // ── Subqueries & Time-Travel ──

    #[test]
    fn subquery_produces_resolve_step() {
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "outage" WHERE entity IN (RECALL semantic ABOUT "services")"#,
        )
        .unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps.iter().any(
                |s| matches!(&s.op, PlanOp::SubqueryResolve { field, .. } if field == "entity")
            ),
            "should have SubqueryResolve step for entity field"
        );
    }

    #[test]
    fn as_of_produces_time_travel_step() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "deploy" AS OF "2026-03-01T12:00:00Z""#)
            .unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps.iter().any(|s| matches!(
                &s.op,
                PlanOp::TimeTravelFilter { snapshot }
                    if snapshot
                        == &RecallSnapshotAst::Unqualified(
                            "2026-03-01T12:00:00Z".to_string()
                        )
            )),
            "should have TimeTravelFilter step"
        );
    }

    #[test]
    fn recorded_as_of_produces_structured_time_travel_step() {
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "deploy" AS OF RECORDED "2026-03-01T12:00:00Z""#,
        )
        .unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps.iter().any(|s| matches!(
                &s.op,
                PlanOp::TimeTravelFilter { snapshot }
                    if snapshot
                        == &RecallSnapshotAst::Recorded(
                            "2026-03-01T12:00:00Z".to_string()
                        )
            )),
            "should have structured TimeTravelFilter step"
        );
    }

    #[test]
    fn subquery_step_before_vector_search() {
        let stmt = parser::parse(
            r#"RECALL episodic ABOUT "test" WHERE entity IN (RECALL semantic ABOUT "svc")"#,
        )
        .unwrap();
        let p = plan(&stmt, None);
        let sq_idx = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::SubqueryResolve { .. }));
        let vs_idx = p
            .steps
            .iter()
            .position(|s| matches!(s.op, PlanOp::VectorSearch { .. }));
        assert!(
            sq_idx.unwrap() < vs_idx.unwrap(),
            "SubqueryResolve should appear before VectorSearch"
        );
    }

    #[test]
    fn no_as_of_no_time_travel_step() {
        let stmt = parser::parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            !p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::TimeTravelFilter { .. })),
            "should not have TimeTravelFilter step without AS OF"
        );
    }

    // ── TRAVERSE, Batch FORGET, Upsert ──

    #[test]
    fn traverse_plan_has_graph_expand() {
        let stmt = parser::parse(r#"TRAVERSE FROM "node1" DEPTH 3"#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(&s.op, PlanOp::GraphExpand { depth: 3, .. })),
            "should have GraphExpand step with depth 3"
        );
    }

    #[test]
    fn traverse_plan_has_vector_search_for_start() {
        let stmt = parser::parse(r#"TRAVERSE FROM "node1" DEPTH 2"#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(&s.op, PlanOp::VectorSearch { .. })),
            "should have VectorSearch step for start node lookup"
        );
    }

    #[test]
    fn traverse_with_where_has_condition_filter() {
        let stmt = parser::parse(r#"TRAVERSE FROM "root" DEPTH 3 WHERE weight > 0.5"#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::ConditionFilter { .. })),
            "should have ConditionFilter step"
        );
    }

    #[test]
    fn traverse_with_limit_has_limit_step() {
        let stmt = parser::parse(r#"TRAVERSE FROM "root" DEPTH 2 LIMIT 10"#).unwrap();
        let p = plan(&stmt, None);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::LimitResults { n: 10 })),
            "should have LimitResults step"
        );
    }

    // ── EXPLAIN ──

    #[test]
    fn explain_plans_inner_recall() {
        let stmt =
            parser::parse(r#"EXPLAIN RECALL episodic ABOUT "test" WHERE importance > 0.5 LIMIT 5"#)
                .unwrap();
        let p = plan(&stmt, None);
        // EXPLAIN delegates to inner statement's plan, so verb should be Recall
        assert_eq!(p.verb, PlanVerb::Recall);
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::VectorSearch { .. }))
        );
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::ImportanceFilter { .. }))
        );
        assert!(
            p.steps
                .iter()
                .any(|s| matches!(s.op, PlanOp::LimitResults { .. }))
        );
    }

    #[test]
    fn explain_analyze_plans_same_as_explain() {
        let stmt1 = parser::parse(r#"EXPLAIN RECALL episodic ABOUT "q" LIMIT 3"#).unwrap();
        let stmt2 = parser::parse(r#"EXPLAIN ANALYZE RECALL episodic ABOUT "q" LIMIT 3"#).unwrap();
        let p1 = plan(&stmt1, None);
        let p2 = plan(&stmt2, None);
        // Plans should be identical regardless of ANALYZE flag
        assert_eq!(p1.steps.len(), p2.steps.len());
        assert_eq!(p1.verb, p2.verb);
    }
}
