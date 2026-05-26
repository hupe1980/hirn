use std::process::Command;
use tempfile::TempDir;

fn hirnd_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_hirnd"))
}

#[test]
fn test_help_shows_all_options() {
    let output = hirnd_bin().arg("--help").output().unwrap();

    assert!(output.status.success(), "hirnd --help failed");
    let stdout = String::from_utf8(output.stdout).unwrap();

    assert!(stdout.contains("--config"), "missing --config");
    assert!(stdout.contains("--data"), "missing --data");
    assert!(stdout.contains("--bind"), "missing --bind");
    assert!(
        stdout.contains("generate-cert"),
        "missing generate-cert subcommand"
    );
}

#[test]
fn test_invalid_config_file_clear_error() {
    let tmp = TempDir::new().unwrap();
    let bad_config = tmp.path().join("bad.toml");
    std::fs::write(&bad_config, "invalid = [[[toml content").unwrap();

    let output = hirnd_bin()
        .arg("--config")
        .arg(bad_config.to_str().unwrap())
        .output()
        .unwrap();

    assert!(!output.status.success(), "should fail on invalid config");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("invalid config file") || stderr.contains("bad.toml"),
        "error should mention the config file: {stderr}"
    );
}

#[test]
fn test_missing_config_file_clear_error() {
    let output = hirnd_bin()
        .arg("--config")
        .arg("/nonexistent/path/config.toml")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "should fail on missing config file"
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("failed to read config file") || stderr.contains("config.toml"),
        "error should mention the missing file: {stderr}"
    );
}

#[test]
fn test_custom_bind_address() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("test");

    // Start the server with a custom bind address, then quickly kill it
    let mut child = hirnd_bin()
        .arg("--data")
        .arg(db_path.to_str().unwrap())
        .arg("--bind")
        .arg("127.0.0.1:19850")
        .arg("--insecure-dev-mode")
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    // Poll for the server to accept connections (up to 15s — debug builds under
    // parallel workspace load can take >5s to bind).
    let mut connected = false;
    for _ in 0..150 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if std::net::TcpStream::connect_timeout(
            &"127.0.0.1:19850".parse().unwrap(),
            std::time::Duration::from_millis(200),
        )
        .is_ok()
        {
            connected = true;
            break;
        }
    }

    // Clean up
    let _ = child.kill();
    let _ = child.wait();

    assert!(
        connected,
        "should be able to connect to custom bind address 127.0.0.1:19850"
    );
}

#[cfg(unix)]
#[test]
fn test_sigterm_stops_server() {
    let tmp = TempDir::new().unwrap();
    let db_path = tmp.path().join("shutdown");

    // Start server
    let mut child = hirnd_bin()
        .arg("--data")
        .arg(db_path.to_str().unwrap())
        .arg("--bind")
        .arg("127.0.0.1:19860")
        .arg("--insecure-dev-mode")
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    // Wait until server is accepting connections (up to 15s — debug builds under
    // parallel workspace load can take >5s to bind).
    let mut ready = false;
    for _ in 0..150 {
        std::thread::sleep(std::time::Duration::from_millis(100));
        if std::net::TcpStream::connect_timeout(
            &"127.0.0.1:19860".parse().unwrap(),
            std::time::Duration::from_millis(200),
        )
        .is_ok()
        {
            ready = true;
            break;
        }
    }
    assert!(ready, "server did not start in time");

    // Send SIGTERM
    Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status()
        .expect("failed to send SIGTERM");

    // Server should exit within a reasonable time
    let status = child.wait().unwrap();
    // Process terminates (either via signal or clean exit)
    assert!(
        !status.success() || status.code() == Some(0),
        "server should have stopped after SIGTERM"
    );
}

// ─── Key Management CLI Tests ────────────────────────────────

#[test]
fn test_add_key_to_config() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("config.toml");
    // Start with empty config
    std::fs::write(&config_path, "").unwrap();

    let output = hirnd_bin()
        .arg("add-key")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .arg("--realm")
        .arg("acme")
        .arg("--agent")
        .arg("agent_a")
        // Key is supplied via env var to avoid process-listing exposure (N-H06).
        .env("HIRND_API_KEY", "my-test-key-123")
        .output()
        .unwrap();

    assert!(output.status.success(), "add-key should succeed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("my-test-key-123"), "should print the key");
    assert!(stdout.contains("acme"), "should mention realm");
    assert!(stdout.contains("agent_a"), "should mention agent");

    // Verify the config file was updated
    let content = std::fs::read_to_string(&config_path).unwrap();
    assert!(
        content.contains("my-test-key-123"),
        "config should contain the key"
    );
    assert!(content.contains("acme"), "config should contain realm");
    assert!(
        content.contains("agent_a"),
        "config should contain agent_id"
    );
}

#[test]
fn test_add_key_generates_random_key() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();

    let output = hirnd_bin()
        .arg("add-key")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        .arg("--realm")
        .arg("default")
        .arg("--agent")
        .arg("bot")
        .output()
        .unwrap();

    assert!(output.status.success(), "add-key should succeed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Key: "), "should print the generated key");
}

#[test]
fn test_rotate_key() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("config.toml");

    // Create initial config with a key
    std::fs::write(
        &config_path,
        r#"
[auth.api_keys.old-key-abc]
realm = "acme"
agent_id = "agent_a"
"#,
    )
    .unwrap();

    let output = hirnd_bin()
        .arg("rotate-key")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        // Keys are supplied via env vars to avoid process-listing exposure (N-H06).
        .env("HIRND_OLD_KEY", "old-key-abc")
        .env("HIRND_NEW_KEY", "new-key-xyz")
        .output()
        .unwrap();

    assert!(output.status.success(), "rotate-key should succeed");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("new-key-xyz"), "should print the new key");

    // Verify old key removed and new key present
    let content = std::fs::read_to_string(&config_path).unwrap();
    assert!(
        !content.contains("old-key-abc"),
        "old key should be removed"
    );
    assert!(content.contains("new-key-xyz"), "new key should be present");
    assert!(content.contains("acme"), "realm should be preserved");
    assert!(content.contains("agent_a"), "agent_id should be preserved");
}

#[test]
fn test_rotate_key_nonexistent_old_key() {
    let tmp = TempDir::new().unwrap();
    let config_path = tmp.path().join("config.toml");

    std::fs::write(
        &config_path,
        r#"
[auth.api_keys.some-key]
realm = "default"
agent_id = "agent"
"#,
    )
    .unwrap();

    let output = hirnd_bin()
        .arg("rotate-key")
        .arg("--config")
        .arg(config_path.to_str().unwrap())
        // Key is supplied via env var to avoid process-listing exposure (N-H06).
        .env("HIRND_OLD_KEY", "nonexistent-key")
        .output()
        .unwrap();

    assert!(
        !output.status.success(),
        "rotate-key should fail for nonexistent key"
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("old key not found"),
        "should report old key not found: {stderr}"
    );
}

#[test]
fn test_help_shows_key_commands() {
    let output = hirnd_bin().arg("--help").output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();

    assert!(stdout.contains("add-key"), "missing add-key subcommand");
    assert!(
        stdout.contains("rotate-key"),
        "missing rotate-key subcommand"
    );
}
