//! Semantic analysis — validates a parsed AST before planning.
//!
//! Checks field names, value types, temporal format validity, and other
//! semantic constraints that go beyond what the PEG grammar can enforce.

use std::collections::HashSet;

use hirn_query::ast::*;

/// A semantic error discovered during analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalysisError {
    pub message: String,
    pub kind: AnalysisErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisErrorKind {
    /// Unknown field name in WHERE clause.
    UnknownField,
    /// Type mismatch (e.g., comparing importance with a string).
    TypeMismatch,
    /// Invalid temporal format.
    InvalidTemporal,
    /// Value out of range (e.g., importance > 1.0).
    ValueOutOfRange,
    /// Missing required clause.
    MissingRequired,
    /// Unknown relation type for CONNECT.
    UnknownRelation,
    /// Invalid layer for operation.
    InvalidLayer,
}

impl std::fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "analysis error: {}", self.message)
    }
}

impl std::error::Error for AnalysisError {}

/// Known WHERE-clause fields and their expected value types.
const NUMERIC_FIELDS: &[&str] = &[
    "importance",
    "confidence",
    "surprise",
    "access_count",
    "evidence_count",
    "relevance_score",
    "success_rate",
    "invocation_count",
    "trust",
    "episodic.access_count",
];

/// Analyze a parsed statement for semantic correctness.
///
/// Returns a list of errors (empty = valid).
pub fn analyze(stmt: &Statement) -> Vec<AnalysisError> {
    match stmt {
        Statement::Recall(r) => analyze_recall(r),
        Statement::Think(t) => analyze_think(t),
        Statement::Correct(c) => analyze_correct(c),
        Statement::Supersede(s) => analyze_supersede(s),
        Statement::MergeMemory(m) => analyze_merge_memory(m),
        Statement::Retract(r) => analyze_retract(r),
        Statement::Inspect(_) | Statement::History(_) | Statement::Trace(_) => vec![],
        Statement::Traverse(t) => analyze_traverse(t),
        Statement::Explain(e) => analyze(&e.inner),
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
        | Statement::Counterfactual(_) => vec![],
    }
}

fn semantic_target_is_empty(target: &SemanticTargetRef) -> bool {
    target.raw_value().trim().is_empty()
}

fn analyze_recall(r: &RecallStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if r.about.trim().is_empty() {
        errors.push(AnalysisError {
            message: "ABOUT clause cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if r.layers.is_empty() {
        errors.push(AnalysisError {
            message: "RECALL requires at least one layer".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(analyze_where_clauses(&r.where_clauses));
    errors.extend(analyze_temporal(r.temporal.as_ref()));
    errors.extend(analyze_expand(r.expand.as_ref()));
    errors.extend(analyze_budget(r.budget));
    errors
}

fn analyze_think(t: &ThinkStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if t.about.trim().is_empty() {
        errors.push(AnalysisError {
            message: "THINK ABOUT clause cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(analyze_where_clauses(&t.where_clauses));
    errors.extend(analyze_temporal(t.temporal.as_ref()));
    errors.extend(analyze_expand(t.expand.as_ref()));
    errors.extend(analyze_budget(t.budget));
    errors
}

fn analyze_correct(c: &CorrectStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if semantic_target_is_empty(&c.target) {
        errors.push(AnalysisError {
            message: "CORRECT target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(analyze_semantic_updates(&c.updates, "CORRECT", true));
    errors.extend(analyze_semantic_observed_at(
        c.observed_at.as_ref(),
        "CORRECT",
    ));

    errors
}

fn analyze_semantic_updates(
    updates: &[SetAssignment],
    verb: &str,
    require_updates: bool,
) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if require_updates && updates.is_empty() {
        errors.push(AnalysisError {
            message: format!("{verb} requires at least one field assignment"),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    for update in updates {
        match update.field.as_str() {
            "description" => {
                if !matches!(update.value, SetValue::String(_)) {
                    errors.push(AnalysisError {
                        message: format!("{verb} description requires a string value"),
                        kind: AnalysisErrorKind::TypeMismatch,
                    });
                }
            }
            "confidence" => {
                let value = match update.value {
                    SetValue::Float(v) => Some(v),
                    SetValue::Int(v) => Some(v as f64),
                    _ => None,
                };

                if let Some(value) = value {
                    if !(0.0..=1.0).contains(&value) {
                        errors.push(AnalysisError {
                            message: format!(
                                "{verb} confidence must be between 0.0 and 1.0, got {value}"
                            ),
                            kind: AnalysisErrorKind::ValueOutOfRange,
                        });
                    }
                } else {
                    errors.push(AnalysisError {
                        message: format!("{verb} confidence requires a numeric value"),
                        kind: AnalysisErrorKind::TypeMismatch,
                    });
                }
            }
            "evidence_count" => match update.value {
                SetValue::Int(v) if v >= 0 => {}
                SetValue::Int(v) => errors.push(AnalysisError {
                    message: format!("{verb} evidence_count must be non-negative, got {v}"),
                    kind: AnalysisErrorKind::ValueOutOfRange,
                }),
                _ => errors.push(AnalysisError {
                    message: format!("{verb} evidence_count requires a non-negative integer"),
                    kind: AnalysisErrorKind::TypeMismatch,
                }),
            },
            other => errors.push(AnalysisError {
                message: format!(
                    "unknown {verb} field '{other}' (allowed: description, confidence, evidence_count)"
                ),
                kind: AnalysisErrorKind::UnknownField,
            }),
        }
    }

    errors
}

fn analyze_semantic_observed_at(observed_at: Option<&String>, verb: &str) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if let Some(observed_at) = observed_at
        && !is_valid_temporal(observed_at)
    {
        errors.push(AnalysisError {
            message: format!("invalid {verb} OBSERVED AT temporal format: '{observed_at}'"),
            kind: AnalysisErrorKind::InvalidTemporal,
        });
    }

    errors
}

fn analyze_supersede(s: &SupersedeStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if semantic_target_is_empty(&s.target) {
        errors.push(AnalysisError {
            message: "SUPERSEDE target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    errors.extend(analyze_semantic_updates(&s.updates, "SUPERSEDE", true));
    errors.extend(analyze_semantic_observed_at(
        s.observed_at.as_ref(),
        "SUPERSEDE",
    ));

    errors
}

fn analyze_merge_memory(m: &MergeMemoryStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if m.sources.is_empty() {
        errors.push(AnalysisError {
            message: "MERGE MEMORY requires at least one source memory".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if semantic_target_is_empty(&m.target) {
        errors.push(AnalysisError {
            message: "MERGE MEMORY target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    let mut seen_sources = HashSet::new();
    for source in &m.sources {
        let normalized = source.raw_value().trim();
        if normalized.is_empty() {
            errors.push(AnalysisError {
                message: "MERGE MEMORY source cannot be empty".into(),
                kind: AnalysisErrorKind::MissingRequired,
            });
            continue;
        }

        let canonical = source.to_string();
        if !seen_sources.insert(canonical.clone()) {
            errors.push(AnalysisError {
                message: format!("MERGE MEMORY source '{}' is duplicated", source.raw_value()),
                kind: AnalysisErrorKind::ValueOutOfRange,
            });
        }

        if canonical == m.target.to_string() {
            errors.push(AnalysisError {
                message: format!(
                    "MERGE MEMORY source '{}' cannot also be the target",
                    source.raw_value()
                ),
                kind: AnalysisErrorKind::ValueOutOfRange,
            });
        }
    }

    errors.extend(analyze_semantic_updates(&m.updates, "MERGE MEMORY", false));
    errors.extend(analyze_semantic_observed_at(
        m.observed_at.as_ref(),
        "MERGE MEMORY",
    ));

    errors
}

fn analyze_retract(r: &RetractStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if semantic_target_is_empty(&r.target) {
        errors.push(AnalysisError {
            message: "RETRACT target cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if let Some(ref observed_at) = r.observed_at {
        if !is_valid_temporal(observed_at) {
            errors.push(AnalysisError {
                message: format!("invalid OBSERVED AT temporal format: '{observed_at}'"),
                kind: AnalysisErrorKind::InvalidTemporal,
            });
        }
    }

    errors
}

fn analyze_traverse(t: &TraverseStmt) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    if t.from.trim().is_empty() {
        errors.push(AnalysisError {
            message: "TRAVERSE FROM cannot be empty".into(),
            kind: AnalysisErrorKind::MissingRequired,
        });
    }

    if t.depth == 0 {
        errors.push(AnalysisError {
            message: "TRAVERSE DEPTH must be at least 1".into(),
            kind: AnalysisErrorKind::ValueOutOfRange,
        });
    }

    errors.extend(analyze_where_clauses(&t.where_clauses));
    errors
}

fn analyze_where_clauses(clauses: &[WhereCondition]) -> Vec<AnalysisError> {
    let mut errors = Vec::new();

    for wc in clauses {
        // Check that numeric fields are compared with numeric values.
        if NUMERIC_FIELDS.contains(&wc.field.as_str()) {
            if matches!(wc.value, ConditionValue::String(_)) {
                errors.push(AnalysisError {
                    message: format!("field '{}' expects a numeric value, got string", wc.field),
                    kind: AnalysisErrorKind::TypeMismatch,
                });
            }
        }

        // Check numeric range for known bounded fields.
        match wc.field.as_str() {
            "importance" | "confidence" | "trust" | "relevance_score" | "success_rate" => {
                let v = match &wc.value {
                    ConditionValue::Float(v) => Some(*v),
                    ConditionValue::Int(v) => Some(*v as f64),
                    _ => None,
                };
                if let Some(v) = v {
                    if !(0.0..=1.0).contains(&v) {
                        errors.push(AnalysisError {
                            message: format!(
                                "field '{}' threshold should be between 0.0 and 1.0, got {}",
                                wc.field, v
                            ),
                            kind: AnalysisErrorKind::ValueOutOfRange,
                        });
                    }
                }
            }
            _ => {}
        }
    }

    errors
}

fn analyze_temporal(temporal: Option<&TemporalClause>) -> Vec<AnalysisError> {
    let Some(tc) = temporal else { return vec![] };
    let mut errors = Vec::new();

    let timestamps = match tc {
        TemporalClause::After(s) => vec![s.as_str()],
        TemporalClause::Before(s) => vec![s.as_str()],
        TemporalClause::Between { start, end } => vec![start.as_str(), end.as_str()],
    };

    for ts in timestamps {
        if !is_valid_temporal(ts) {
            errors.push(AnalysisError {
                message: format!(
                    "invalid temporal value: '{ts}' (expected YYYY-MM-DD or RFC 3339)"
                ),
                kind: AnalysisErrorKind::InvalidTemporal,
            });
        }
    }

    errors
}

fn analyze_expand(expand: Option<&ExpandClause>) -> Vec<AnalysisError> {
    let Some(ex) = expand else { return vec![] };
    let mut errors = Vec::new();

    if ex.depth == 0 {
        errors.push(AnalysisError {
            message: "EXPAND GRAPH DEPTH must be at least 1".into(),
            kind: AnalysisErrorKind::ValueOutOfRange,
        });
    }

    if let Some(mw) = ex.min_weight {
        if !(0.0..=1.0).contains(&mw) {
            errors.push(AnalysisError {
                message: format!("MIN_WEIGHT must be between 0.0 and 1.0, got {mw}"),
                kind: AnalysisErrorKind::ValueOutOfRange,
            });
        }
    }

    errors
}

fn analyze_budget(budget: Option<usize>) -> Vec<AnalysisError> {
    if let Some(b) = budget {
        if b == 0 {
            return vec![AnalysisError {
                message: "BUDGET must be greater than 0".into(),
                kind: AnalysisErrorKind::ValueOutOfRange,
            }];
        }
    }
    vec![]
}

fn is_valid_temporal(s: &str) -> bool {
    use chrono::NaiveDate;
    // Accept YYYY-MM-DD.
    if NaiveDate::parse_from_str(s, "%Y-%m-%d").is_ok() {
        return true;
    }
    // Accept RFC 3339 / ISO 8601.
    if chrono::DateTime::parse_from_rfc3339(s).is_ok() {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_recall_passes() {
        let stmt = hirn_query::parse(r#"RECALL episodic ABOUT "test""#).unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    #[test]
    fn recall_with_valid_where() {
        let stmt =
            hirn_query::parse(r#"RECALL episodic ABOUT "x" WHERE importance > 0.5"#).unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    #[test]
    fn recall_with_out_of_range_importance() {
        let stmt =
            hirn_query::parse(r#"RECALL episodic ABOUT "x" WHERE importance > 2.0"#).unwrap();
        let errors = analyze(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::ValueOutOfRange);
    }

    #[test]
    fn recall_with_invalid_temporal() {
        let stmt = hirn_query::parse(r#"RECALL episodic ABOUT "x" AFTER "not-a-date""#).unwrap();
        let errors = analyze(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::InvalidTemporal);
    }

    #[test]
    fn remember_is_rejected_before_analysis() {
        let error =
            hirn_query::parse(r#"REMEMBER episode CONTENT "x" IMPORTANCE 1.5"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("REMEMBER is not supported via embedded HirnQL anymore")
        );
    }

    #[test]
    fn correct_unknown_field_is_rejected() {
        let stmt = hirn_query::parse(r#"CORRECT "x" SET unsupported = 1"#).unwrap();
        let errors = analyze(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::UnknownField);
    }

    #[test]
    fn supersede_unknown_field_is_rejected() {
        let stmt = hirn_query::parse(r#"SUPERSEDE "x" SET unsupported = 1"#).unwrap();
        let errors = analyze(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::UnknownField);
    }

    #[test]
    fn retract_invalid_observed_at_is_rejected() {
        let stmt = hirn_query::parse(r#"RETRACT "x" OBSERVED AT "not-a-date""#).unwrap();
        let errors = analyze(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::InvalidTemporal);
    }

    #[test]
    fn connect_unknown_relation() {
        let error = hirn_query::parse(r#"CONNECT "a" TO "b" AS unknown_rel"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("CONNECT is not supported via embedded HirnQL anymore")
        );
    }

    #[test]
    fn connect_valid_relation() {
        let error =
            hirn_query::parse(r#"CONNECT "a" TO "b" AS related_to WEIGHT 0.5"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("CONNECT is not supported via embedded HirnQL anymore")
        );
    }

    #[test]
    fn connect_weight_out_of_range() {
        let error = hirn_query::parse(r#"CONNECT "a" TO "b" AS causes WEIGHT 1.5"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("CONNECT is not supported via embedded HirnQL anymore")
        );
    }

    #[test]
    fn think_valid_passes() {
        let stmt = hirn_query::parse(r#"THINK ABOUT "test" BUDGET 4096"#).unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    #[test]
    fn think_global_valid() {
        let stmt = hirn_query::parse(r#"THINK GLOBAL ABOUT "test""#).unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    #[test]
    fn consolidate_valid() {
        let error = hirn_query::parse("CONSOLIDATE").unwrap_err();
        assert!(
            error
                .to_string()
                .contains("CONSOLIDATE is not supported via HirnQL anymore")
        );
    }

    #[test]
    fn watch_valid() {
        let error = hirn_query::parse(r#"WATCH ALL"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("WATCH is not supported via embedded HirnQL anymore")
        );
    }

    #[test]
    fn budget_zero_rejected() {
        let stmt = hirn_query::parse(r#"RECALL episodic ABOUT "x" BUDGET 0"#).unwrap();
        let errors = analyze(&stmt);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, AnalysisErrorKind::ValueOutOfRange);
    }

    #[test]
    fn valid_temporal_formats() {
        assert!(is_valid_temporal("2026-03-01"));
        assert!(is_valid_temporal("2026-03-01T12:00:00Z"));
        assert!(is_valid_temporal("2026-03-01T12:00:00+01:00"));
        assert!(!is_valid_temporal("not-a-date"));
        assert!(!is_valid_temporal("March 1st"));
    }

    #[test]
    fn between_with_valid_dates() {
        let stmt =
            hirn_query::parse(r#"RECALL episodic ABOUT "x" BETWEEN "2026-01-01" AND "2026-03-01""#)
                .unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    // ── TRAVERSE, Batch FORGET ──

    #[test]
    fn traverse_valid() {
        let stmt = hirn_query::parse(r#"TRAVERSE FROM "node1" DEPTH 3"#).unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    #[test]
    fn traverse_with_via_and_where() {
        let stmt =
            hirn_query::parse(r#"TRAVERSE FROM "node1" VIA causes DEPTH 2 WHERE weight > 0.5"#)
                .unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    #[test]
    fn batch_forget_valid() {
        let error =
            hirn_query::parse(r#"FORGET episodic WHERE importance < 0.1 ARCHIVE"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("FORGET is not supported via embedded HirnQL anymore")
        );
    }

    #[test]
    fn forget_hard_mode_valid() {
        let error = hirn_query::parse(r#"FORGET "id123" HARD"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("FORGET is not supported via embedded HirnQL anymore")
        );
    }

    // ── EXPLAIN ──

    #[test]
    fn explain_valid_recall_no_warnings() {
        let stmt = hirn_query::parse(r#"EXPLAIN RECALL episodic ABOUT "test""#).unwrap();
        assert!(analyze(&stmt).is_empty());
    }

    #[test]
    fn explain_analyze_delegates_to_inner() {
        // EXPLAIN ANALYZE on a query with an invalid range should still report the inner warning
        let stmt = hirn_query::parse(
            r#"EXPLAIN ANALYZE RECALL episodic ABOUT "test" WHERE importance > 2.0"#,
        )
        .unwrap();
        let warnings = analyze(&stmt);
        assert!(
            warnings
                .iter()
                .any(|w| matches!(w.kind, AnalysisErrorKind::ValueOutOfRange)),
            "should propagate inner analysis warnings: {warnings:?}"
        );
    }
}
