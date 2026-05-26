//! Integration tests for Configuration & Operational CLI commands.
//!
//! Tests: validate-config, info, optimize, export, import.

use std::process::Command;
use tempfile::TempDir;

fn hirnd_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hirnd"))
}

/// Seed a database by importing a generated JSON file via the CLI.
///
/// This avoids in-process LanceDB connections which hold global locks
/// that prevent successive CLI subprocesses from opening the same database.
fn seed_database(data_dir: &std::path::Path, count: usize) {
    use hirn_core::episodic::EpisodicRecord;
    use hirn_core::types::AgentId;
    use hirn_engine::export::ExportData;

    let json_path = data_dir.join("_seed.json");

    let agent = AgentId::new("test_agent").unwrap();
    let mut records = Vec::new();
    for i in 0..count {
        let record = EpisodicRecord::builder()
            .content(&format!("event {i}"))
            .agent_id(agent.clone())
            .embedding(vec![0.1 + (i as f32 * 0.01); 768])
            .build()
            .unwrap();
        records.push(record);
    }

    let export_data = ExportData {
        version: 1,
        working: vec![],
        episodic: records,
        semantic: vec![],
        procedural: vec![],
        agents: vec![],
        namespaces: vec![],
        edges: vec![],
    };

    std::fs::create_dir_all(data_dir).unwrap();
    let json = serde_json::to_string_pretty(&export_data).unwrap();
    std::fs::write(&json_path, &json).unwrap();

    let output = hirnd_bin()
        .arg("import")
        .arg("--input")
        .arg(json_path.to_str().unwrap())
        .arg("--data")
        .arg(data_dir.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "seed import should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Clean up seed file
    let _ = std::fs::remove_file(&json_path);
}

// ── validate-config ───────────────────────────────────────────────

#[test]
fn validate_valid_config_reports_valid() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("server.toml");
    std::fs::write(
        &config_path,
        r#"
bind = "127.0.0.1:3000"
data_dir = "/tmp/hirn-test"
"#,
    )
    .unwrap();

    let output = hirnd_bin()
        .arg("validate-config")
        .arg(config_path.to_str().unwrap())
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "valid config should pass: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("valid"), "should say valid: {stdout}");
}

#[test]
fn validate_invalid_config_reports_error() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("bad.toml");
    std::fs::write(
        &config_path,
        r#"
bind = 12345
data_dir = "/tmp/hirn-test"
"#,
    )
    .unwrap();

    let output = hirnd_bin()
        .arg("validate-config")
        .arg(config_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(!output.status.success(), "invalid config should fail");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("bind") || stderr.contains("invalid"),
        "should mention the bad field: {stderr}"
    );
}

#[test]
fn validate_config_with_bad_engine_reports_error() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("bad_engine.toml");
    std::fs::write(
        &config_path,
        r#"
[engine]
embedding_dimensions = 0
"#,
    )
    .unwrap();

    let output = hirnd_bin()
        .arg("validate-config")
        .arg(config_path.to_str().unwrap())
        .output()
        .unwrap();

    assert!(!output.status.success(), "bad engine config should fail");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("embedding_dimensions"),
        "should mention embedding_dimensions: {stderr}"
    );
}

// ── info ──────────────────────────────────────────────────────────

#[test]
fn info_on_database_with_known_content_shows_correct_counts() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");

    seed_database(&data_dir, 5);

    let output = hirnd_bin()
        .arg("info")
        .arg("--data")
        .arg(data_dir.to_str().unwrap())
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "info should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("5 records"),
        "should show 5 episodic records: {stdout}"
    );
    assert!(
        stdout.contains("Total records"),
        "should show total: {stdout}"
    );
    assert!(
        stdout.contains("Graph edges"),
        "should show graph info: {stdout}"
    );
}

// ── optimize ──────────────────────────────────────────────────────

#[test]
fn optimize_runs_successfully_after_deletes() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("data");

    // Seed with 50 records via CLI import
    seed_database(&data_dir, 50);

    let output = hirnd_bin()
        .arg("optimize")
        .arg("--data")
        .arg(data_dir.to_str().unwrap())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "optimize should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        stdout.contains("ptimiz"),
        "should report optimization status: {stdout}"
    );
}

// ── export / import ───────────────────────────────────────────────

#[test]
fn export_to_json_via_cli() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("export_data");
    let json_path = tmp.path().join("dump.json");

    seed_database(&data_dir, 5);

    let output = hirnd_bin()
        .arg("export")
        .arg("--data")
        .arg(data_dir.to_str().unwrap())
        .arg("--output")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "export should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Export complete"),
        "should report completion: {stdout}"
    );
    assert!(json_path.exists(), "JSON file should be created");

    // Verify it's valid JSON
    let content = std::fs::read_to_string(&json_path).unwrap();
    let data: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert_eq!(data["version"], 1);
}

#[test]
fn import_from_json_via_cli() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("orig");
    let json_path = tmp.path().join("dump.json");
    let imported_path = tmp.path().join("imported");

    seed_database(&data_dir, 5);

    // Export
    let output = hirnd_bin()
        .arg("export")
        .arg("--data")
        .arg(data_dir.to_str().unwrap())
        .arg("--output")
        .arg(json_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Import
    let output = hirnd_bin()
        .arg("import")
        .arg("--input")
        .arg(json_path.to_str().unwrap())
        .arg("--data")
        .arg(imported_path.to_str().unwrap())
        .output()
        .unwrap();

    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(
        output.status.success(),
        "import should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Import complete"),
        "should report completion: {stdout}"
    );
    assert!(imported_path.exists(), "imported DB should be created");
}

#[test]
fn export_import_export_produces_identical_json() {
    let tmp = TempDir::new().unwrap();
    let data_dir = tmp.path().join("orig");
    let json1_path = tmp.path().join("dump1.json");
    let imported_path = tmp.path().join("imported");
    let json2_path = tmp.path().join("dump2.json");

    seed_database(&data_dir, 5);

    // Export 1
    let output = hirnd_bin()
        .arg("export")
        .arg("--data")
        .arg(data_dir.to_str().unwrap())
        .arg("--output")
        .arg(json1_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Import
    let output = hirnd_bin()
        .arg("import")
        .arg("--input")
        .arg(json1_path.to_str().unwrap())
        .arg("--data")
        .arg(imported_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Export 2
    let output = hirnd_bin()
        .arg("export")
        .arg("--data")
        .arg(imported_path.to_str().unwrap())
        .arg("--output")
        .arg(json2_path.to_str().unwrap())
        .output()
        .unwrap();
    assert!(output.status.success());

    // Compare JSON content — should have identical record counts
    let json1: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json1_path).unwrap()).unwrap();
    let json2: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&json2_path).unwrap()).unwrap();

    assert_eq!(
        json1["episodic"].as_array().unwrap().len(),
        json2["episodic"].as_array().unwrap().len(),
        "episodic count should match after roundtrip"
    );
    assert_eq!(
        json1["semantic"].as_array().unwrap().len(),
        json2["semantic"].as_array().unwrap().len(),
        "semantic count should match after roundtrip"
    );
    assert_eq!(
        json1["working"].as_array().unwrap().len(),
        json2["working"].as_array().unwrap().len(),
        "working count should match after roundtrip"
    );
}
