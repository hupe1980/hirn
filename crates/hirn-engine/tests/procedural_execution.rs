//! F-046 FIX: Integration tests for procedural execution via `ToolExecutor`.

use std::sync::Arc;

use hirn_core::HirnConfig;
use hirn_core::error::HirnResult;
use hirn_core::id::MemoryId;
use hirn_core::metadata::Metadata;
use hirn_core::procedural::{ActionStep, StepResult, ToolExecutor};
use hirn_core::revision::LogicalMemoryId;
use hirn_core::types::AgentId;
use hirn_engine::HirnDB;
use hirn_storage::memory_store::MemoryStore;

// ── Helpers ──────────────────────────────────────────────────────────────

fn agent() -> AgentId {
    AgentId::new("test_agent").unwrap()
}

async fn temp_db() -> (Arc<HirnDB>, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("proc_exec");
    let config = HirnConfig::builder()
        .db_path(&path)
        .working_memory_token_limit(100_000)
        .build()
        .unwrap();
    let db = HirnDB::open_with_config(config, Arc::new(MemoryStore::new()))
        .await
        .unwrap();
    (Arc::new(db), dir)
}

async fn current_procedure_head(
    db: &HirnDB,
    logical_memory_id: LogicalMemoryId,
) -> hirn_core::procedural::ProceduralRecord {
    db.procedural()
        .list(None)
        .await
        .unwrap()
        .into_iter()
        .find(|record| record.logical_memory_id == logical_memory_id)
        .expect("current procedural head should remain visible")
}

fn step(name: &str) -> ActionStep {
    ActionStep {
        description: format!("Run {name}"),
        tool: Some(name.to_string()),
        parameters: Metadata::default(),
    }
}

fn doc_step(description: &str) -> ActionStep {
    ActionStep {
        description: description.to_string(),
        tool: None,
        parameters: Metadata::default(),
    }
}

// ── Executor implementations ─────────────────────────────────────────────

/// Always succeeds, echoing the tool name.
struct AlwaysSucceed;

impl ToolExecutor for AlwaysSucceed {
    async fn execute_step(&self, step: &ActionStep) -> HirnResult<StepResult> {
        Ok(StepResult {
            step_index: 0, // corrected by caller
            success: true,
            output: format!("OK: {}", step.tool.as_deref().unwrap_or("none")),
        })
    }
}

/// Fails on the step whose tool name matches `fail_on`.
struct FailOnTool {
    fail_on: String,
}

impl ToolExecutor for FailOnTool {
    async fn execute_step(&self, step: &ActionStep) -> HirnResult<StepResult> {
        let tool = step.tool.as_deref().unwrap_or("");
        if tool == self.fail_on {
            Ok(StepResult {
                step_index: 0,
                success: false,
                output: format!("FAIL: {tool}"),
            })
        } else {
            Ok(StepResult {
                step_index: 0,
                success: true,
                output: format!("OK: {tool}"),
            })
        }
    }
}

/// Returns an error (simulating a transport / runtime failure).
struct ErrorExecutor;

impl ToolExecutor for ErrorExecutor {
    async fn execute_step(&self, _step: &ActionStep) -> HirnResult<StepResult> {
        Err(hirn_core::error::HirnError::Unsupported(
            "executor crashed".into(),
        ))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread")]
async fn execute_all_steps_succeed() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::procedural::ProceduralRecord::builder()
        .name("deploy")
        .description("deploy pipeline")
        .steps(vec![step("build"), step("test"), step("push")])
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.procedural().store(rec).await.unwrap();
    let logical_id = db.procedural().get(id).await.unwrap().logical_memory_id;
    let result = db.procedural().execute(id, &AlwaysSucceed).await.unwrap();

    assert!(result.success);
    assert_eq!(result.procedure_id, id);
    assert_eq!(result.step_results.len(), 3);
    for (i, sr) in result.step_results.iter().enumerate() {
        assert!(sr.success);
        assert_eq!(sr.step_index, i);
    }

    // Success tracking should have been updated.
    let updated = current_procedure_head(&db, logical_id).await;
    assert_eq!(updated.invocation_count, 1);
    assert_eq!(updated.success_count, 1);
    assert!(updated.success_rate > 0.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_short_circuits_on_failure() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::procedural::ProceduralRecord::builder()
        .name("pipeline")
        .description("multi-step pipeline")
        .steps(vec![step("lint"), step("test"), step("deploy")])
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.procedural().store(rec).await.unwrap();
    let logical_id = db.procedural().get(id).await.unwrap().logical_memory_id;

    let executor = FailOnTool {
        fail_on: "test".into(),
    };
    let result = db.procedural().execute(id, &executor).await.unwrap();

    assert!(!result.success);
    // Should have stopped after step 1 ("test"), so only 2 results (lint OK, test FAIL).
    assert_eq!(result.step_results.len(), 2);
    assert!(result.step_results[0].success); // lint
    assert!(!result.step_results[1].success); // test
    assert_eq!(result.step_results[1].step_index, 1);

    // Failure tracking should have been updated.
    let updated = current_procedure_head(&db, logical_id).await;
    assert_eq!(updated.invocation_count, 1);
    assert_eq!(updated.success_count, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_handles_executor_error() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::procedural::ProceduralRecord::builder()
        .name("crash_proc")
        .description("procedure that triggers executor error")
        .steps(vec![step("boom")])
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.procedural().store(rec).await.unwrap();
    let result = db.procedural().execute(id, &ErrorExecutor).await.unwrap();

    assert!(!result.success);
    assert_eq!(result.step_results.len(), 1);
    assert!(!result.step_results[0].success);
    assert!(result.step_results[0].output.contains("executor crashed"));
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_skips_doc_only_steps() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::procedural::ProceduralRecord::builder()
        .name("mixed")
        .description("procedure with doc-only steps")
        .steps(vec![
            doc_step("Ensure prerequisites are met"),
            step("build"),
            doc_step("Verify output"),
            step("deploy"),
        ])
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.procedural().store(rec).await.unwrap();
    let result = db.procedural().execute(id, &AlwaysSucceed).await.unwrap();

    assert!(result.success);
    assert_eq!(result.step_results.len(), 4);

    // Doc-only steps should be auto-succeeded with empty output.
    assert!(result.step_results[0].success);
    assert_eq!(result.step_results[0].output, "");
    assert_eq!(result.step_results[0].step_index, 0);

    // Tool steps should have executor output.
    assert!(result.step_results[1].success);
    assert!(result.step_results[1].output.contains("OK"));
    assert_eq!(result.step_results[1].step_index, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_empty_procedure_succeeds() {
    let (db, _dir) = temp_db().await;

    let rec = hirn_core::procedural::ProceduralRecord::builder()
        .name("noop")
        .description("procedure with no steps")
        .agent_id(agent())
        .build()
        .unwrap();

    let id = db.procedural().store(rec).await.unwrap();
    let logical_id = db.procedural().get(id).await.unwrap().logical_memory_id;
    let result = db.procedural().execute(id, &AlwaysSucceed).await.unwrap();

    assert!(result.success);
    assert!(result.step_results.is_empty());

    // Should still record a success.
    let updated = current_procedure_head(&db, logical_id).await;
    assert_eq!(updated.invocation_count, 1);
    assert_eq!(updated.success_count, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_nonexistent_procedure_errors() {
    let (db, _dir) = temp_db().await;

    let fake_id = MemoryId::new();
    let result = db.procedural().execute(fake_id, &AlwaysSucceed).await;

    assert!(result.is_err());
}
