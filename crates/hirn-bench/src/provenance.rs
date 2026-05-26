use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use chrono::Utc;

use crate::cognitive::EnvironmentInfo;

pub fn generated_at_rfc3339() -> String {
    Utc::now().to_rfc3339()
}

pub fn current_environment_info(label: Option<String>) -> EnvironmentInfo {
    let logical_cpus = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);

    EnvironmentInfo {
        label: label
            .or_else(|| std::env::var("RUNNER_NAME").ok())
            .or_else(|| std::env::var("HOSTNAME").ok())
            .or_else(|| std::env::var("HOST").ok()),
        image: std::env::var("ImageOS").ok(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        logical_cpus,
        git_commit_sha: current_git_commit_sha(),
        cargo_lock_blake3: cargo_lock_blake3(),
    }
}

fn current_git_commit_sha() -> Option<String> {
    if let Ok(sha) = std::env::var("GITHUB_SHA")
        .or_else(|_| std::env::var("CI_COMMIT_SHA"))
        .or_else(|_| std::env::var("BUILDKITE_COMMIT"))
    {
        let sha = sha.trim();
        if !sha.is_empty() {
            return Some(sha.to_string());
        }
    }

    let workspace_root = workspace_root()?;
    let output = Command::new("git")
        .arg("-C")
        .arg(workspace_root)
        .arg("rev-parse")
        .arg("HEAD")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let sha = String::from_utf8(output.stdout).ok()?;
    let sha = sha.trim();
    if sha.is_empty() {
        None
    } else {
        Some(sha.to_string())
    }
}

fn cargo_lock_blake3() -> Option<String> {
    let workspace_root = workspace_root()?;
    let lock_path = workspace_root.join("Cargo.lock");
    let bytes = std::fs::read(lock_path).ok()?;
    Some(blake3::hash(&bytes).to_hex().to_string())
}

fn workspace_root() -> Option<&'static Path> {
    static WORKSPACE_ROOT: OnceLock<Option<PathBuf>> = OnceLock::new();
    WORKSPACE_ROOT
        .get_or_init(|| {
            let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
            manifest_dir.parent()?.parent().map(Path::to_path_buf)
        })
        .as_deref()
}
