//! Load cognitive benchmark datasets from JSONL files.
//!
//! Expected directory layout:
//! ```text
//! dataset_dir/
//!   sessions.jsonl   # one JSON-serialized Session per line
//!   queries.jsonl    # one JSON-serialized QAQuery per line
//! ```

use std::io::BufRead;
use std::path::Path;

use super::{Benchmark, CognitiveDataset, QAQuery, Session};

/// Load a cognitive dataset from a directory with `sessions.jsonl` and `queries.jsonl`.
pub fn load(benchmark: Benchmark, data_dir: &Path) -> Result<CognitiveDataset, String> {
    let sessions_path = data_dir.join("sessions.jsonl");
    let queries_path = data_dir.join("queries.jsonl");

    let sessions = load_jsonl::<Session>(&sessions_path)?;
    if sessions.is_empty() {
        return Err(format!(
            "no sessions loaded from {}",
            sessions_path.display()
        ));
    }

    let queries = load_jsonl::<QAQuery>(&queries_path)?;
    if queries.is_empty() {
        return Err(format!("no queries loaded from {}", queries_path.display()));
    }

    Ok(CognitiveDataset {
        name: format!("{} ({})", benchmark.name(), data_dir.display()),
        benchmark,
        sessions,
        queries,
    })
}

fn load_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, String> {
    let file =
        std::fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let reader = std::io::BufReader::new(file);
    let mut items = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line =
            line.map_err(|e| format!("{}:{}: read error: {e}", path.display(), line_num + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let item: T = serde_json::from_str(trimmed)
            .map_err(|e| format!("{}:{}: parse error: {e}", path.display(), line_num + 1))?;
        items.push(item);
    }

    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_dataset(dir: &Path, sessions: &str, queries: &str) {
        std::fs::write(dir.join("sessions.jsonl"), sessions).unwrap();
        std::fs::write(dir.join("queries.jsonl"), queries).unwrap();
    }

    #[test]
    fn load_valid_dataset() {
        let dir = TempDir::new().unwrap();
        let sessions = r#"{"id":"s1","turns":[{"speaker":"A","content":"Hello"}]}"#;
        let queries = r#"{"id":"q1","question":"What?","expected_answers":["Hello"],"category":"test","relevant_session_ids":["s1"]}"#;
        write_dataset(dir.path(), sessions, queries);

        let ds = load(Benchmark::H1Retrieval, dir.path()).unwrap();
        assert_eq!(ds.sessions.len(), 1);
        assert_eq!(ds.queries.len(), 1);
        assert_eq!(ds.sessions[0].turns[0].content, "Hello");
    }

    #[test]
    fn load_multiple_lines() {
        let dir = TempDir::new().unwrap();
        let sessions = concat!(
            r#"{"id":"s1","turns":[{"speaker":"A","content":"One"}]}"#,
            "\n",
            r#"{"id":"s2","turns":[{"speaker":"B","content":"Two"}]}"#,
            "\n",
        );
        let queries = r#"{"id":"q1","question":"Q","expected_answers":["A"],"category":"c","relevant_session_ids":["s1"]}"#;
        write_dataset(dir.path(), sessions, queries);

        let ds = load(Benchmark::H1Retrieval, dir.path()).unwrap();
        assert_eq!(ds.sessions.len(), 2);
    }

    #[test]
    fn load_missing_file() {
        let dir = TempDir::new().unwrap();
        let result = load(Benchmark::H1Retrieval, dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn load_skips_blank_and_comment_lines() {
        let dir = TempDir::new().unwrap();
        let sessions = concat!(
            "# comment\n",
            "\n",
            r#"{"id":"s1","turns":[{"speaker":"A","content":"X"}]}"#,
            "\n",
            "\n",
        );
        let queries = r#"{"id":"q1","question":"Q","expected_answers":["X"],"category":"c","relevant_session_ids":["s1"]}"#;
        write_dataset(dir.path(), sessions, queries);

        let ds = load(Benchmark::H1Retrieval, dir.path()).unwrap();
        assert_eq!(ds.sessions.len(), 1);
    }

    #[test]
    fn roundtrip_synthetic_dataset() {
        let ds = super::super::synthetic::generate(Benchmark::H1Retrieval);
        let dir = TempDir::new().unwrap();

        // Serialize sessions and queries as JSONL.
        let sessions: String = ds
            .sessions
            .iter()
            .map(|s| serde_json::to_string(s).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        let queries: String = ds
            .queries
            .iter()
            .map(|q| serde_json::to_string(q).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        write_dataset(dir.path(), &sessions, &queries);

        let loaded = load(Benchmark::H1Retrieval, dir.path()).unwrap();
        assert_eq!(loaded.sessions.len(), ds.sessions.len());
        assert_eq!(loaded.queries.len(), ds.queries.len());
    }
}
