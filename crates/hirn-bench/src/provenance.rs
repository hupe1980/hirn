use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use chrono::Utc;

use crate::cognitive::{DatasetFileChecksum, EnvironmentInfo};

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

/// Blake3 hex digest of one file.
pub fn blake3_file_checksum(path: &Path) -> Result<String, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

/// Checksum a set of dataset files for provenance pinning.
///
/// Returns per-file checksums (paths relative to `base` where possible) plus
/// a combined blake3 digest over the sorted `path\n<hex>\n` lines. The
/// combined digest is what `--expect-dataset-hash` compares against and is
/// stable across machines as long as the relative layout and bytes match.
pub fn dataset_checksums(
    files: &[PathBuf],
    base: &Path,
) -> Result<(Vec<DatasetFileChecksum>, String), String> {
    let mut checksums = Vec::with_capacity(files.len());
    for file in files {
        let relative = file
            .strip_prefix(base)
            .unwrap_or(file)
            .to_string_lossy()
            .into_owned();
        checksums.push(DatasetFileChecksum {
            path: relative,
            blake3: blake3_file_checksum(file)?,
        });
    }

    checksums.sort_by(|left, right| left.path.cmp(&right.path));

    let mut hasher = blake3::Hasher::new();
    for checksum in &checksums {
        hasher.update(checksum.path.as_bytes());
        hasher.update(b"\n");
        hasher.update(checksum.blake3.as_bytes());
        hasher.update(b"\n");
    }

    Ok((checksums, hasher.finalize().to_hex().to_string()))
}

/// Fail-fast dataset pinning: compare the combined dataset hash against the
/// expected value from `--expect-dataset-hash`.
pub fn verify_dataset_hash(expected: &str, actual: &str) -> Result<(), String> {
    let expected = expected.trim();
    if expected.eq_ignore_ascii_case(actual) {
        Ok(())
    } else {
        Err(format!(
            "dataset hash mismatch: expected {expected}, loaded dataset hashes to {actual}; \
             refusing to run on unpinned data (re-download the pinned revision or update \
             --expect-dataset-hash deliberately)"
        ))
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_checksums_are_stable_and_order_independent() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a.json");
        let b = dir.path().join("b.json");
        std::fs::write(&a, b"alpha").unwrap();
        std::fs::write(&b, b"beta").unwrap();

        let (files_fwd, hash_fwd) = dataset_checksums(&[a.clone(), b.clone()], dir.path()).unwrap();
        let (files_rev, hash_rev) = dataset_checksums(&[b, a], dir.path()).unwrap();

        assert_eq!(hash_fwd, hash_rev);
        assert_eq!(files_fwd.len(), 2);
        assert_eq!(files_fwd[0].path, "a.json");
        assert_eq!(
            files_fwd[0].blake3,
            blake3::hash(b"alpha").to_hex().to_string()
        );
        assert_eq!(files_rev[0].path, "a.json");
    }

    #[test]
    fn dataset_checksums_change_when_content_changes() {
        let dir = tempfile::TempDir::new().unwrap();
        let a = dir.path().join("a.json");

        std::fs::write(&a, b"alpha").unwrap();
        let (_, before) = dataset_checksums(std::slice::from_ref(&a), dir.path()).unwrap();

        std::fs::write(&a, b"tampered").unwrap();
        let (_, after) = dataset_checksums(std::slice::from_ref(&a), dir.path()).unwrap();

        assert_ne!(before, after);
    }

    #[test]
    fn verify_dataset_hash_accepts_match_and_rejects_mismatch() {
        assert!(verify_dataset_hash("ABC123", "abc123").is_ok());
        assert!(verify_dataset_hash("  abc123 ", "abc123").is_ok());

        let error = verify_dataset_hash("expected00", "actual11").unwrap_err();
        assert!(error.contains("dataset hash mismatch"), "got: {error}");
        assert!(error.contains("expected00"));
        assert!(error.contains("actual11"));
    }

    #[test]
    fn blake3_file_checksum_missing_file_errors() {
        let error = blake3_file_checksum(Path::new("/nonexistent/data.json")).unwrap_err();
        assert!(error.contains("cannot read"), "got: {error}");
    }
}
