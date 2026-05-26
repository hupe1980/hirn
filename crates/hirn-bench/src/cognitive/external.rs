//! External benchmark adapters — convert published datasets to HIRN-Bench format.
//!
//! This module provides adapters for standard benchmarks used by competitor
//! systems, enabling direct apples-to-apples comparison.
//!
//! # Supported Formats
//!
//! - **LoCoMo** (Maharana et al., 2024): Long-conversation memory benchmark with
//!   5 question categories (single-hop, multi-hop, temporal, world-knowledge, adversarial).
//!   Reported by: SYNAPSE, TraceMem, FadeMem, Hippocampus.
//!
//! - **DMR** (Dialog Memory Retrieval): Multi-turn dialog fact retrieval.
//!   Reported by: Zep (94.8%), MemGPT (93.4%).
//!
//! # Usage
//!
//! ```text
//! # Convert a LoCoMo dataset to HIRN-Bench JSONL:
//! hirn-bench convert --format locomo --input locomo_data/ --output data/locomo/
//!
//! # Then run the standard cognitive benchmark:
//! hirn-bench cognitive --data data/locomo/ --suite h1
//! ```

use std::collections::{BTreeSet, HashMap};
use std::io::BufReader;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use serde::de::{self, DeserializeOwned, Deserializer as _, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use super::{Benchmark, CognitiveDataset, QAQuery, Session, Turn};

fn read_huggingface_token_file(path: &Path) -> Option<String> {
    let token = std::fs::read_to_string(path).ok()?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

fn default_huggingface_token_path() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("HF_TOKEN_PATH") {
        let path = path.trim();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }

    if let Ok(home) = std::env::var("HF_HOME") {
        let home = home.trim();
        if !home.is_empty() {
            return Some(PathBuf::from(home).join("token"));
        }
    }

    if let Ok(cache_home) = std::env::var("XDG_CACHE_HOME") {
        let cache_home = cache_home.trim();
        if !cache_home.is_empty() {
            return Some(PathBuf::from(cache_home).join("huggingface").join("token"));
        }
    }

    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("token"),
    )
}

fn resolve_huggingface_auth_token() -> Option<String> {
    for env_var in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(token) = std::env::var(env_var) {
            let token = token.trim();
            if !token.is_empty() {
                return Some(token.to_string());
            }
        }
    }

    default_huggingface_token_path().and_then(|path| read_huggingface_token_file(&path))
}

fn format_huggingface_request_error(repo: &str, error: ureq::Error) -> String {
    match error {
        ureq::Error::StatusCode(status) if status == 401 || status == 403 => format!(
            "HuggingFace API request failed: http status: {status}; dataset `{repo}` may require authentication. Set HF_TOKEN (or legacy HUGGING_FACE_HUB_TOKEN), authenticate with `huggingface-cli login` so the cached token is available locally, or pass --data-dir to a local dataset mirror."
        ),
        other => format!("HuggingFace API request failed: {other}"),
    }
}

fn huggingface_resolve_file_url(repo: &str, path: &str) -> String {
    format!("https://huggingface.co/datasets/{repo}/resolve/main/{path}")
}

fn github_raw_file_url(repo: &str, path: &str) -> String {
    format!("https://raw.githubusercontent.com/{repo}/main/{path}")
}

fn write_reader_to_destination<R: std::io::Read>(
    mut reader: R,
    destination: &Path,
) -> Result<(), String> {
    let partial_path = destination.with_extension("partial");
    let mut file = std::fs::File::create(&partial_path)
        .map_err(|error| format!("failed to create {}: {error}", partial_path.display()))?;

    if let Err(error) = std::io::copy(&mut reader, &mut file) {
        let _ = std::fs::remove_file(&partial_path);
        return Err(format!(
            "failed to write {}: {error}",
            partial_path.display()
        ));
    }

    std::fs::rename(&partial_path, destination).map_err(|error| {
        format!(
            "failed to move {} into place at {}: {error}",
            partial_path.display(),
            destination.display()
        )
    })
}

fn download_public_file(url: &str, destination: &Path, label: &str) -> Result<(), String> {
    let mut response = ureq::get(url)
        .header("User-Agent", "hirn-bench")
        .call()
        .map_err(|error| format!("failed to download {label}: {error}"))?;

    write_reader_to_destination(response.body_mut().as_reader(), destination)
}

fn download_huggingface_file(
    repo: &str,
    path: &str,
    destination: &Path,
    token: Option<&str>,
) -> Result<(), String> {
    let url = huggingface_resolve_file_url(repo, path);
    download_huggingface_file_from_url(&url, repo, destination, token)
}

fn download_huggingface_file_from_url(
    url: &str,
    repo: &str,
    destination: &Path,
    token: Option<&str>,
) -> Result<(), String> {
    let mut response = match token {
        Some(token) => ureq::get(url)
            .header("Authorization", &format!("Bearer {token}"))
            .call(),
        None => ureq::get(url).call(),
    }
    .map_err(|error| format_huggingface_request_error(repo, error))?;

    write_reader_to_destination(response.body_mut().as_reader(), destination)
}

fn process_json_array_reader<T, R, F>(reader: R, mut on_item: F) -> Result<usize, String>
where
    T: DeserializeOwned,
    R: std::io::Read,
    F: FnMut(T) -> Result<(), String>,
{
    struct JsonArrayProcessor<'a, T, F> {
        on_item: &'a mut F,
        count: usize,
        marker: PhantomData<T>,
    }

    impl<'de, 'a, T, F> Visitor<'de> for JsonArrayProcessor<'a, T, F>
    where
        T: DeserializeOwned,
        F: FnMut(T) -> Result<(), String>,
    {
        type Value = usize;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON array")
        }

        fn visit_seq<A>(mut self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            while let Some(item) = seq.next_element::<T>()? {
                (self.on_item)(item).map_err(de::Error::custom)?;
                self.count += 1;
            }

            Ok(self.count)
        }
    }

    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    deserializer
        .deserialize_seq(JsonArrayProcessor {
            on_item: &mut on_item,
            count: 0,
            marker: PhantomData,
        })
        .map_err(|error| format!("failed to parse JSON array: {error}"))
}

fn json_value_to_text(value: serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::Null => None,
        serde_json::Value::String(text) => Some(text),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(boolean) => Some(boolean.to_string()),
        other => serde_json::to_string(&other).ok(),
    }
}

fn deserialize_optional_stringlike<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(value.and_then(json_value_to_text))
}

fn deserialize_vec_stringlike<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let values = Vec::<serde_json::Value>::deserialize(deserializer)?;
    Ok(values.into_iter().filter_map(json_value_to_text).collect())
}

// ─── LoCoMo Adapter ─────────────────────────────────────────

/// A LoCoMo conversation from the published dataset JSON.
#[derive(Debug, Deserialize, Serialize)]
pub struct LoCoMoConversation {
    /// Conversation ID.
    pub id: String,
    /// Turns in the conversation.
    pub conversation: Vec<LoCoMoTurn>,
    /// QA pairs grouped by category.
    #[serde(default)]
    pub questions: HashMap<String, Vec<LoCoMoQuestion>>,
}

/// A single LoCoMo turn.
#[derive(Debug, Deserialize, Serialize)]
pub struct LoCoMoTurn {
    pub speaker: String,
    pub text: String,
    #[serde(default)]
    pub timestamp: Option<String>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
}

/// A LoCoMo question-answer pair.
#[derive(Debug, Deserialize, Serialize)]
pub struct LoCoMoQuestion {
    pub question: String,
    pub answer: String,
    #[serde(default)]
    pub evidence: Vec<String>,
}

/// Convert a LoCoMo dataset directory to a `CognitiveDataset`.
///
/// Expected layout:
/// ```text
/// locomo_dir/
///   conversations.json   # array of LoCoMoConversation
/// ```
pub fn load_locomo(data_dir: &Path) -> Result<CognitiveDataset, String> {
    let conv_path = data_dir.join(LOCOMO_CACHE_FILE);
    if conv_path.exists() {
        let conversations = parse_locomo_conversations(&conv_path)?;
        return Ok(build_locomo_dataset(data_dir, &conversations));
    }

    let raw_path = data_dir.join(LOCOMO_SOURCE_FILE);
    if raw_path.exists() {
        let conversations = parse_locomo_raw_conversations(&raw_path)?;
        return Ok(build_locomo_dataset(data_dir, &conversations));
    }

    Err(format!(
        "cannot read {} or {}",
        conv_path.display(),
        raw_path.display()
    ))
}

fn parse_locomo_conversations(conv_path: &Path) -> Result<Vec<LoCoMoConversation>, String> {
    let raw = std::fs::read_to_string(&conv_path)
        .map_err(|e| format!("cannot read {}: {e}", conv_path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("parse error in {}: {e}", conv_path.display()))
}

fn parse_locomo_raw_conversations(raw_path: &Path) -> Result<Vec<LoCoMoConversation>, String> {
    let raw = std::fs::read_to_string(raw_path)
        .map_err(|e| format!("cannot read {}: {e}", raw_path.display()))?;
    let raw_conversations: Vec<LoCoMoRawConversation> = serde_json::from_str(&raw)
        .map_err(|e| format!("parse error in {}: {e}", raw_path.display()))?;

    raw_conversations
        .into_iter()
        .enumerate()
        .map(|(index, conversation)| normalize_locomo_raw_conversation(index, conversation))
        .collect()
}

fn build_locomo_dataset(data_dir: &Path, conversations: &[LoCoMoConversation]) -> CognitiveDataset {
    let mut sessions = Vec::new();
    let mut queries = Vec::new();
    let mut query_idx = 0u32;

    for conv in conversations {
        let mut source_to_session = HashMap::new();
        let mut source_to_snippet = HashMap::new();
        let mut conversation_session_ids = Vec::new();

        for turn in &conv.conversation {
            let logical_session_id = turn
                .session_id
                .as_ref()
                .map(|session_id| format!("{}::{session_id}", conv.id))
                .unwrap_or_else(|| conv.id.clone());

            if sessions.last().map(|session: &Session| session.id.as_str())
                != Some(logical_session_id.as_str())
            {
                conversation_session_ids.push(logical_session_id.clone());
                sessions.push(Session {
                    id: logical_session_id.clone(),
                    turns: Vec::new(),
                });
            }

            if let Some(source_id) = &turn.source_id {
                source_to_session.insert(source_id.clone(), logical_session_id.clone());
                source_to_snippet
                    .entry(source_id.clone())
                    .or_insert_with(|| turn.text.clone());
            }

            sessions
                .last_mut()
                .expect("session pushed before appending turn")
                .turns
                .push(Turn {
                    speaker: turn.speaker.clone(),
                    content: turn.text.clone(),
                    timestamp: turn
                        .timestamp
                        .as_deref()
                        .and_then(parse_locomo_timestamp_ms),
                    timestamp_text: turn.timestamp.clone(),
                    source_id: turn.source_id.clone(),
                });
        }

        // Convert QA pairs.
        for (category, qs) in &conv.questions {
            for q in qs {
                query_idx += 1;
                let (evidence_ids, evidence_snippets) =
                    split_locomo_evidence(&q.evidence, &source_to_session, &source_to_snippet);
                let mut relevant_session_ids: Vec<String> = evidence_ids
                    .iter()
                    .filter_map(|evidence_id| source_to_session.get(evidence_id).cloned())
                    .collect();
                relevant_session_ids.sort();
                relevant_session_ids.dedup();
                if relevant_session_ids.is_empty() {
                    relevant_session_ids.clone_from(&conversation_session_ids);
                }

                queries.push(QAQuery {
                    id: format!("locomo-{}-{query_idx}", conv.id),
                    question: q.question.clone(),
                    expected_answers: vec![q.answer.clone()],
                    category: category.clone(),
                    relevant_session_ids,
                    evidence_ids,
                    evidence_snippets,
                    negative: false,
                });
            }
        }
    }

    CognitiveDataset {
        name: format!("LoCoMo ({})", data_dir.display()),
        benchmark: Benchmark::H1Retrieval,
        sessions,
        queries,
    }
}

#[derive(Debug, Deserialize)]
struct LoCoMoRawConversation {
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    sample_id: Option<String>,
    #[serde(default)]
    conversation: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    observation: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    session_summary: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    qa: Vec<LoCoMoRawQuestion>,
}

#[derive(Debug, Deserialize)]
struct LoCoMoRawQuestion {
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    question: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    answer: Option<String>,
    #[serde(default, deserialize_with = "deserialize_vec_stringlike")]
    evidence: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    category: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LoCoMoRawTurn {
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    speaker: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    dia_id: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    text: Option<String>,
}

fn normalize_locomo_raw_conversation(
    index: usize,
    raw: LoCoMoRawConversation,
) -> Result<LoCoMoConversation, String> {
    let id = raw
        .sample_id
        .unwrap_or_else(|| format!("locomo-{}", index + 1));
    let conversation =
        extract_locomo_raw_turns(&raw.conversation, &raw.observation, &raw.session_summary)?;
    let mut questions: HashMap<String, Vec<LoCoMoQuestion>> = HashMap::new();

    for question in raw.qa {
        let Some(question_text) = question.question.map(|value| value.trim().to_string()) else {
            continue;
        };
        if question_text.is_empty() {
            continue;
        }

        questions
            .entry(locomo_category_label(question.category.as_deref()))
            .or_default()
            .push(LoCoMoQuestion {
                question: question_text,
                answer: question.answer.unwrap_or_default(),
                evidence: question.evidence,
            });
    }

    Ok(LoCoMoConversation {
        id,
        conversation,
        questions,
    })
}

fn extract_locomo_raw_turns(
    conversation: &serde_json::Map<String, serde_json::Value>,
    observation: &serde_json::Map<String, serde_json::Value>,
    session_summary: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<LoCoMoTurn>, String> {
    let mut session_numbers: Vec<u32> = conversation
        .keys()
        .filter_map(|key| key.strip_prefix("session_")?.parse::<u32>().ok())
        .collect();
    session_numbers.sort_unstable();
    session_numbers.dedup();

    let mut turns = Vec::new();
    for session_number in session_numbers {
        let session_key = format!("session_{session_number}");
        let session_timestamp = conversation
            .get(&format!("{session_key}_date_time"))
            .and_then(|value| json_value_to_text(value.clone()));
        let Some(session_turns) = conversation
            .get(&session_key)
            .and_then(serde_json::Value::as_array)
        else {
            continue;
        };

        for raw_turn in session_turns {
            let raw_turn: LoCoMoRawTurn =
                serde_json::from_value(raw_turn.clone()).map_err(|error| {
                    format!("failed to parse LoCoMo turn in {session_key}: {error}")
                })?;
            let Some(speaker) = raw_turn.speaker.map(|value| value.trim().to_string()) else {
                continue;
            };
            let Some(text) = raw_turn.text.map(|value| value.trim().to_string()) else {
                continue;
            };
            if speaker.is_empty() || text.is_empty() {
                continue;
            }

            turns.push(LoCoMoTurn {
                speaker,
                text,
                timestamp: session_timestamp.clone(),
                source_id: raw_turn.dia_id,
                session_id: Some(session_key.clone()),
            });
        }

        append_locomo_observation_turns(
            &mut turns,
            &session_key,
            session_timestamp.as_deref(),
            observation.get(&format!("{session_key}_observation")),
        );
        append_locomo_summary_turn(
            &mut turns,
            &session_key,
            session_timestamp.as_deref(),
            session_summary.get(&format!("{session_key}_summary")),
        );
    }

    Ok(turns)
}

fn append_locomo_observation_turns(
    turns: &mut Vec<LoCoMoTurn>,
    session_key: &str,
    session_timestamp: Option<&str>,
    observation_value: Option<&serde_json::Value>,
) {
    let Some(observation_map) = observation_value.and_then(serde_json::Value::as_object) else {
        return;
    };

    for (speaker, facts_value) in observation_map {
        let Some(facts) = facts_value.as_array() else {
            continue;
        };

        for fact_value in facts {
            let Some(fact_parts) = fact_value.as_array() else {
                continue;
            };
            let Some(text) = fact_parts
                .first()
                .cloned()
                .and_then(json_value_to_text)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
            else {
                continue;
            };

            let source_id = fact_parts
                .get(1)
                .cloned()
                .and_then(json_value_to_text)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());

            turns.push(LoCoMoTurn {
                speaker: format!("Observation/{speaker}"),
                text,
                timestamp: session_timestamp.map(str::to_string),
                source_id,
                session_id: Some(session_key.to_string()),
            });
        }
    }
}

fn append_locomo_summary_turn(
    turns: &mut Vec<LoCoMoTurn>,
    session_key: &str,
    session_timestamp: Option<&str>,
    summary_value: Option<&serde_json::Value>,
) {
    let Some(text) = summary_value
        .cloned()
        .and_then(json_value_to_text)
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    else {
        return;
    };

    turns.push(LoCoMoTurn {
        speaker: "SessionSummary".to_string(),
        text,
        timestamp: session_timestamp.map(str::to_string),
        source_id: None,
        session_id: Some(session_key.to_string()),
    });
}

fn split_locomo_evidence(
    evidence: &[String],
    source_to_session: &HashMap<String, String>,
    source_to_snippet: &HashMap<String, String>,
) -> (Vec<String>, Vec<String>) {
    let mut evidence_ids = Vec::new();
    let mut evidence_snippets = BTreeSet::new();

    for item in evidence {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        if source_to_session.contains_key(item) || looks_like_locomo_dialog_id(item) {
            evidence_ids.push(item.to_string());
            if let Some(snippet) = source_to_snippet
                .get(item)
                .map(String::as_str)
                .map(str::trim)
                .filter(|snippet| !snippet.is_empty())
            {
                evidence_snippets.insert(snippet.to_string());
            }
        } else {
            evidence_snippets.insert(item.to_string());
        }
    }

    (evidence_ids, evidence_snippets.into_iter().collect())
}

fn looks_like_locomo_dialog_id(value: &str) -> bool {
    let Some((session, dialog)) = value.trim().split_once(':') else {
        return false;
    };

    session.strip_prefix('D').is_some_and(|digits| {
        !digits.is_empty() && digits.chars().all(|character| character.is_ascii_digit())
    }) && !dialog.is_empty()
        && dialog.chars().all(|character| character.is_ascii_digit())
}

fn parse_locomo_timestamp_ms(text: &str) -> Option<u64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S") {
        return Some(dt.and_utc().timestamp_millis() as u64);
    }

    let (time_part, date_part) = text.split_once(" on ")?;
    let time = chrono::NaiveTime::parse_from_str(time_part, "%I:%M %P")
        .or_else(|_| chrono::NaiveTime::parse_from_str(time_part, "%I:%M %p"))
        .ok()?;

    let (day, rest) = date_part.split_once(' ')?;
    let day = if day.len() == 1 {
        format!("0{day}")
    } else {
        day.to_string()
    };
    let normalized_date = format!("{day} {rest}");
    let date = chrono::NaiveDate::parse_from_str(&normalized_date, "%d %B, %Y").ok()?;

    Some(date.and_time(time).and_utc().timestamp_millis() as u64)
}

fn locomo_category_label(category: Option<&str>) -> String {
    match category.map(str::trim).filter(|value| !value.is_empty()) {
        Some("1") => "multi-hop".to_string(),
        Some("2") => "temporal".to_string(),
        Some("3") => "world-knowledge".to_string(),
        Some("4") => "single-hop".to_string(),
        Some("5") => "adversarial".to_string(),
        Some(value) => value.to_string(),
        None => "unknown".to_string(),
    }
}

// ─── DMR Adapter ─────────────────────────────────────────────

/// A DMR dialog from the published dataset.
#[derive(Debug, Deserialize, Serialize)]
pub struct DmrDialog {
    pub dialog_id: String,
    pub turns: Vec<DmrTurn>,
    pub queries: Vec<DmrQuery>,
}

/// A single DMR turn.
#[derive(Debug, Deserialize, Serialize)]
pub struct DmrTurn {
    pub speaker: String,
    pub utterance: String,
    #[serde(default)]
    pub turn_id: Option<u32>,
}

/// A DMR retrieval query.
#[derive(Debug, Deserialize, Serialize)]
pub struct DmrQuery {
    pub query: String,
    pub answer: String,
    #[serde(default)]
    pub relevant_turn_ids: Vec<u32>,
}

/// Convert a DMR dataset directory to a `CognitiveDataset`.
///
/// Expected layout:
/// ```text
/// dmr_dir/
///   dialogs.json   # array of DmrDialog
/// ```
pub fn load_dmr(data_dir: &Path) -> Result<CognitiveDataset, String> {
    let dialog_path = data_dir.join("dialogs.json");
    let raw = std::fs::read_to_string(&dialog_path)
        .map_err(|e| format!("cannot read {}: {e}", dialog_path.display()))?;
    let dialogs: Vec<DmrDialog> = serde_json::from_str(&raw)
        .map_err(|e| format!("parse error in {}: {e}", dialog_path.display()))?;

    let mut sessions = Vec::new();
    let mut queries = Vec::new();
    let mut query_idx = 0u32;

    for dialog in &dialogs {
        let turns: Vec<Turn> = dialog
            .turns
            .iter()
            .map(|t| Turn {
                speaker: t.speaker.clone(),
                content: t.utterance.clone(),
                timestamp: None,
                timestamp_text: None,
                source_id: t.turn_id.map(|turn_id| turn_id.to_string()),
            })
            .collect();

        sessions.push(Session {
            id: dialog.dialog_id.clone(),
            turns,
        });

        for q in &dialog.queries {
            query_idx += 1;
            queries.push(QAQuery {
                id: format!("dmr-{}-{query_idx}", dialog.dialog_id),
                question: q.query.clone(),
                expected_answers: vec![q.answer.clone()],
                category: "retrieval".to_string(),
                relevant_session_ids: vec![dialog.dialog_id.clone()],
                evidence_ids: q
                    .relevant_turn_ids
                    .iter()
                    .map(|turn_id| turn_id.to_string())
                    .collect(),
                evidence_snippets: Vec::new(),
                negative: false,
            });
        }
    }

    Ok(CognitiveDataset {
        name: format!("DMR ({})", data_dir.display()),
        benchmark: Benchmark::H1Retrieval,
        sessions,
        queries,
    })
}

/// Supported external benchmark formats.
#[derive(Debug, Clone, Copy)]
pub enum ExternalFormat {
    LoCoMo,
    Dmr,
    LongMemEval,
}

impl std::str::FromStr for ExternalFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "locomo" => Ok(Self::LoCoMo),
            "dmr" => Ok(Self::Dmr),
            "longmemeval" | "lme" => Ok(Self::LongMemEval),
            _ => Err(format!(
                "unknown external format: {s} (expected: locomo, dmr, longmemeval)"
            )),
        }
    }
}

/// Load an external benchmark dataset and convert it to HIRN-Bench format.
pub fn load_external(format: ExternalFormat, data_dir: &Path) -> Result<CognitiveDataset, String> {
    match format {
        ExternalFormat::LoCoMo => load_locomo(data_dir),
        ExternalFormat::Dmr => load_dmr(data_dir),
        ExternalFormat::LongMemEval => load_longmemeval(data_dir),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ExternalLoadLimits {
    pub max_sessions: usize,
    pub max_records: usize,
    pub max_queries: usize,
}

// ─── LongMemEval Adapter (F-28) ─────────────────────────────

/// A LongMemEval test case from the published dataset.
///
/// LongMemEval (Wu et al., 2024) evaluates long-term memory in 5 task types:
/// information extraction, multi-session reasoning, temporal reasoning,
/// knowledge updates, and abstention.
#[derive(Debug, Deserialize, Serialize)]
pub struct LongMemEvalCase {
    pub id: String,
    /// The conversation history (chronologically ordered).
    pub sessions: Vec<LongMemEvalSession>,
    /// Test questions.
    pub questions: Vec<LongMemEvalQuestion>,
}

/// A session within a LongMemEval test case.
#[derive(Debug, Deserialize, Serialize)]
pub struct LongMemEvalSession {
    #[serde(default)]
    pub session_id: Option<String>,
    pub turns: Vec<LongMemEvalTurn>,
}

/// A single turn in a LongMemEval session.
#[derive(Debug, Deserialize, Serialize)]
pub struct LongMemEvalTurn {
    pub role: String,
    pub content: String,
}

/// A LongMemEval question with ground truth.
#[derive(Debug, Deserialize, Serialize)]
pub struct LongMemEvalQuestion {
    pub question: String,
    pub answer: String,
    #[serde(default)]
    pub task_type: Option<String>,
    /// Whether the question should be abstained from (no answer in history).
    #[serde(default)]
    pub should_abstain: bool,
}

/// Convert a LongMemEval dataset directory to a `CognitiveDataset`.
///
/// Expected layout:
/// ```text
/// longmemeval_dir/
///   cases.json   # array of LongMemEvalCase
/// ```
pub fn load_longmemeval(data_dir: &Path) -> Result<CognitiveDataset, String> {
    load_longmemeval_with_limits(data_dir, None)
}

pub fn load_longmemeval_with_limits(
    data_dir: &Path,
    limits: Option<ExternalLoadLimits>,
) -> Result<CognitiveDataset, String> {
    let cases_path = data_dir.join("cases.json");
    if cases_path.exists() && limits.is_none() {
        let raw = std::fs::read_to_string(&cases_path)
            .map_err(|e| format!("cannot read {}: {e}", cases_path.display()))?;
        let cases: Vec<LongMemEvalCase> = serde_json::from_str(&raw)
            .map_err(|e| format!("parse error in {}: {e}", cases_path.display()))?;

        let mut sessions = Vec::new();
        let mut queries = Vec::new();
        let mut query_idx = 0u32;

        for case in &cases {
            for (si, sess) in case.sessions.iter().enumerate() {
                let sid = sess
                    .session_id
                    .clone()
                    .unwrap_or_else(|| format!("{}-s{si}", case.id));
                let turns: Vec<Turn> = sess
                    .turns
                    .iter()
                    .map(|t| Turn {
                        speaker: t.role.clone(),
                        content: t.content.clone(),
                        timestamp: None,
                        timestamp_text: None,
                        source_id: None,
                    })
                    .collect();
                sessions.push(Session { id: sid, turns });
            }

            for q in &case.questions {
                query_idx += 1;
                let category = q.task_type.clone().unwrap_or_else(|| "general".to_string());
                let session_ids: Vec<String> = case
                    .sessions
                    .iter()
                    .enumerate()
                    .map(|(si, s)| {
                        s.session_id
                            .clone()
                            .unwrap_or_else(|| format!("{}-s{si}", case.id))
                    })
                    .collect();
                queries.push(QAQuery {
                    id: format!("lme-{}-{query_idx}", case.id),
                    question: q.question.clone(),
                    expected_answers: vec![q.answer.clone()],
                    category,
                    relevant_session_ids: session_ids,
                    evidence_ids: Vec::new(),
                    evidence_snippets: Vec::new(),
                    negative: q.should_abstain,
                });
            }
        }

        return Ok(CognitiveDataset {
            name: format!("LongMemEval ({})", data_dir.display()),
            benchmark: Benchmark::H1Retrieval,
            sessions,
            queries,
        });
    }

    load_longmemeval_raw(data_dir, limits)
}

// ─── LoCoMo Dataset Download & Caching ───────────────────────

/// Canonical upstream GitHub repository for the LoCoMo benchmark.
const LOCOMO_GITHUB_REPO: &str = "snap-research/locomo";

/// Canonical LoCoMo dataset file published by the upstream repository.
const LOCOMO_SOURCE_FILE: &str = "locomo10.json";

/// Path to the canonical LoCoMo dataset file in the upstream repository.
const LOCOMO_GITHUB_PATH: &str = "data/locomo10.json";

/// Filename for the cached conversations file.
const LOCOMO_CACHE_FILE: &str = "conversations.json";

/// Marker file to indicate a successful download.
const LOCOMO_MARKER: &str = ".locomo_downloaded";

/// Download the LoCoMo dataset from the canonical upstream GitHub repository and cache it locally.
///
/// Returns the path to the cache directory containing `locomo10.json`.
///
/// If the cache directory already contains a valid download (marker file exists),
/// this function returns immediately without re-downloading.
pub fn download_locomo(cache_dir: &Path) -> Result<PathBuf, String> {
    let cache_path = cache_dir.join(LOCOMO_SOURCE_FILE);
    let marker_path = cache_dir.join(LOCOMO_MARKER);

    // Check cache.
    if marker_path.exists() && cache_path.exists() {
        eprintln!("LoCoMo dataset already cached at {}", cache_dir.display());
        return Ok(cache_dir.to_path_buf());
    }

    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("cannot create cache dir {}: {e}", cache_dir.display()))?;

    let url = github_raw_file_url(LOCOMO_GITHUB_REPO, LOCOMO_GITHUB_PATH);
    eprintln!(
        "Downloading LoCoMo dataset from GitHub ({LOCOMO_GITHUB_REPO}/{LOCOMO_GITHUB_PATH})..."
    );
    download_public_file(&url, &cache_path, "LoCoMo dataset")?;

    let conversation_count = load_locomo(cache_dir)?.sessions.len();

    // Write marker.
    std::fs::write(&marker_path, format!("{conversation_count} conversations"))
        .map_err(|e| format!("failed to write marker file: {e}"))?;

    eprintln!(
        "LoCoMo dataset cached: {} conversations at {}",
        conversation_count,
        cache_dir.display(),
    );

    Ok(cache_dir.to_path_buf())
}

/// Download (if needed) and load the LoCoMo dataset, returning a `CognitiveDataset`.
pub fn load_locomo_cached(cache_dir: &Path) -> Result<CognitiveDataset, String> {
    let data_dir = download_locomo(cache_dir)?;
    load_locomo(&data_dir)
}

// ─── DMR Dataset Download & Caching ─────────────────────────

/// Filename for the cached DMR dialogs.
const DMR_CACHE_FILE: &str = "dialogs.json";

/// Marker file to indicate a successful DMR download.
const DMR_MARKER: &str = ".dmr_downloaded";

const DMR_AUTO_DOWNLOAD_GUIDANCE: &str = "DMR auto-download is currently disabled because no verified public canonical dataset source is configured. Pass --data-dir to a local mirror containing dialogs.json.";

/// Download the DMR dataset and cache it locally when a verified local mirror already exists.
///
/// Returns the path to the cache directory containing `dialogs.json`.
///
/// If the cache directory already contains a valid download (marker file exists),
/// this function returns immediately without re-downloading.
pub fn download_dmr(cache_dir: &Path) -> Result<PathBuf, String> {
    let cache_path = cache_dir.join(DMR_CACHE_FILE);
    let marker_path = cache_dir.join(DMR_MARKER);

    // Check cache.
    if marker_path.exists() && cache_path.exists() {
        eprintln!("DMR dataset already cached at {}", cache_dir.display());
        return Ok(cache_dir.to_path_buf());
    }

    Err(DMR_AUTO_DOWNLOAD_GUIDANCE.to_string())
}

/// Download (if needed) and load the DMR dataset, returning a `CognitiveDataset`.
pub fn load_dmr_cached(cache_dir: &Path) -> Result<CognitiveDataset, String> {
    let data_dir = download_dmr(cache_dir)?;
    load_dmr(&data_dir)
}

// ─── LongMemEval Dataset Download & Caching ─────────────────

/// HuggingFace Hub repo for the LongMemEval dataset.
const LME_HF_REPO: &str = "xiaowu0162/longmemeval";

/// LongMemEval is published as direct JSON files instead of a rows-backed viewer dataset.
const LME_HF_FILES: &[&str] = &["longmemeval_oracle", "longmemeval_s", "longmemeval_m"];

/// Filename for the legacy cached LongMemEval cases.
const LME_CACHE_FILE: &str = "cases.json";

/// Marker file to indicate a successful LongMemEval download.
const LME_MARKER: &str = ".longmemeval_downloaded";

/// LongMemEval raw turn from the published HuggingFace JSON files.
#[derive(Debug, Deserialize)]
struct LongMemEvalRawTurn {
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    role: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    content: Option<String>,
}

/// LongMemEval raw question row from the published HuggingFace JSON files.
#[derive(Debug, Deserialize)]
struct LongMemEvalRawRow {
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    question_id: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    question: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    answer: Option<String>,
    #[serde(default, deserialize_with = "deserialize_optional_stringlike")]
    question_type: Option<String>,
    #[serde(default, deserialize_with = "deserialize_vec_stringlike")]
    answer_session_ids: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_vec_stringlike")]
    haystack_session_ids: Vec<String>,
    #[serde(default)]
    haystack_sessions: Vec<Vec<LongMemEvalRawTurn>>,
}

#[derive(Default)]
struct LongMemEvalSessionRegistry {
    by_source_id: HashMap<String, (String, String)>,
    by_fingerprint: HashMap<String, String>,
}

fn longmemeval_session_fingerprint(turns: &[Turn]) -> String {
    let mut fingerprint = String::new();
    for turn in turns {
        fingerprint.push_str(&turn.speaker);
        fingerprint.push('\u{1f}');
        fingerprint.push_str(&turn.content);
        fingerprint.push('\u{1e}');
    }
    fingerprint
}

fn register_longmemeval_session(
    registry: &mut LongMemEvalSessionRegistry,
    case_id: &str,
    source_id: Option<&str>,
    turns: &[Turn],
) -> (String, bool) {
    let fingerprint = longmemeval_session_fingerprint(turns);

    if let Some(source_id) = source_id.map(str::trim).filter(|id| !id.is_empty()) {
        if let Some((existing_id, existing_fingerprint)) = registry.by_source_id.get(source_id) {
            if *existing_fingerprint == fingerprint {
                return (existing_id.clone(), false);
            }
        }

        if let Some(existing_id) = registry.by_fingerprint.get(&fingerprint) {
            registry
                .by_source_id
                .insert(source_id.to_string(), (existing_id.clone(), fingerprint));
            return (existing_id.clone(), false);
        }

        let canonical_id = if registry.by_source_id.contains_key(source_id) {
            format!("{case_id}:{source_id}")
        } else {
            source_id.to_string()
        };

        registry.by_source_id.insert(
            source_id.to_string(),
            (canonical_id.clone(), fingerprint.clone()),
        );
        registry
            .by_fingerprint
            .insert(fingerprint, canonical_id.clone());
        return (canonical_id, true);
    }

    if let Some(existing_id) = registry.by_fingerprint.get(&fingerprint) {
        return (existing_id.clone(), false);
    }

    let canonical_id = format!("{case_id}-s{}", registry.by_fingerprint.len());
    registry
        .by_fingerprint
        .insert(fingerprint, canonical_id.clone());
    (canonical_id, true)
}

fn is_longmemeval_abstention(
    task_type: Option<&str>,
    answer: &str,
    answer_session_ids: &[String],
) -> bool {
    let answer = answer.trim();
    if answer.is_empty() {
        return true;
    }

    let task_type = task_type.unwrap_or("").to_ascii_lowercase();
    task_type.contains("abstain")
        || task_type.contains("unanswer")
        || (answer_session_ids.is_empty()
            && matches!(
                answer.to_ascii_lowercase().as_str(),
                "unknown" | "not mentioned"
            ))
}

fn append_longmemeval_raw_row(
    row: LongMemEvalRawRow,
    sessions: &mut Vec<Session>,
    queries: &mut Vec<QAQuery>,
    query_idx: &mut u32,
    fallback_index: usize,
    registry: &mut LongMemEvalSessionRegistry,
    current_records: &mut usize,
    limits: Option<ExternalLoadLimits>,
) -> Result<(), String> {
    let LongMemEvalRawRow {
        question_id,
        question,
        answer,
        question_type,
        answer_session_ids,
        haystack_session_ids,
        haystack_sessions,
    } = row;

    let case_id = question_id.unwrap_or_else(|| format!("case-{fallback_index}"));
    let question = question.unwrap_or_default().trim().to_string();
    if question.is_empty() {
        return Ok(());
    }

    let answer = answer.unwrap_or_default().trim().to_string();
    let task_type = question_type.and_then(|task_type| {
        let task_type = task_type.trim();
        if task_type.is_empty() {
            None
        } else {
            Some(task_type.to_string())
        }
    });

    let mut relevant_session_ids = Vec::new();
    for (session_index, raw_session) in haystack_sessions.into_iter().enumerate() {
        let turns: Vec<Turn> = raw_session
            .into_iter()
            .filter_map(|turn| {
                let content = turn.content.unwrap_or_default().trim().to_string();
                if content.is_empty() {
                    return None;
                }

                let speaker = turn
                    .role
                    .unwrap_or_else(|| "unknown".to_string())
                    .trim()
                    .to_string();

                Some(Turn {
                    speaker: if speaker.is_empty() {
                        "unknown".to_string()
                    } else {
                        speaker
                    },
                    content,
                    timestamp: None,
                    timestamp_text: None,
                    source_id: None,
                })
            })
            .collect();

        if turns.is_empty() {
            continue;
        }

        let source_id = haystack_session_ids.get(session_index).map(String::as_str);
        let (session_id, is_new) =
            register_longmemeval_session(registry, &case_id, source_id, &turns);
        if is_new {
            if let Some(limits) = limits {
                if sessions.len() >= limits.max_sessions {
                    continue;
                }

                if *current_records >= limits.max_records {
                    continue;
                }

                if *current_records + turns.len() > limits.max_records {
                    continue;
                }
            }

            sessions.push(Session {
                id: session_id.clone(),
                turns,
            });
            *current_records += sessions
                .last()
                .map(|session| session.turns.len())
                .unwrap_or_default();
        }

        if !relevant_session_ids
            .iter()
            .any(|existing| existing == &session_id)
        {
            relevant_session_ids.push(session_id);
        }
    }

    if relevant_session_ids.is_empty() {
        return Ok(());
    }

    if let Some(limits) = limits {
        if queries.len() >= limits.max_queries {
            return Ok(());
        }
    }

    *query_idx += 1;
    queries.push(QAQuery {
        id: format!("lme-{case_id}-{}", *query_idx),
        question,
        expected_answers: vec![answer.clone()],
        category: task_type.clone().unwrap_or_else(|| "general".to_string()),
        relevant_session_ids,
        evidence_ids: Vec::new(),
        evidence_snippets: Vec::new(),
        negative: is_longmemeval_abstention(task_type.as_deref(), &answer, &answer_session_ids),
    });

    Ok(())
}

fn load_longmemeval_raw(
    data_dir: &Path,
    limits: Option<ExternalLoadLimits>,
) -> Result<CognitiveDataset, String> {
    let mut sessions = Vec::new();
    let mut queries = Vec::new();
    let mut query_idx = 0u32;
    let mut raw_row_idx = 0usize;
    let mut registry = LongMemEvalSessionRegistry::default();
    let mut processed_files = 0usize;
    let mut current_records = 0usize;

    const LIMIT_REACHED: &str = "__longmemeval_limit_reached__";

    for file_name in LME_HF_FILES {
        let file_path = data_dir.join(file_name);
        if !file_path.exists() {
            continue;
        }

        processed_files += 1;
        let file = std::fs::File::open(&file_path)
            .map_err(|error| format!("cannot read {}: {error}", file_path.display()))?;
        let parse_result = process_json_array_reader::<LongMemEvalRawRow, _, _>(
            BufReader::new(file),
            |row| {
                if let Some(limits) = limits {
                    if sessions.len() >= limits.max_sessions
                        || queries.len() >= limits.max_queries
                        || current_records >= limits.max_records
                    {
                        return Err(LIMIT_REACHED.to_string());
                    }
                }

                raw_row_idx += 1;
                append_longmemeval_raw_row(
                    row,
                    &mut sessions,
                    &mut queries,
                    &mut query_idx,
                    raw_row_idx,
                    &mut registry,
                    &mut current_records,
                    limits,
                )
            },
        );

        match parse_result {
            Ok(_) => {}
            Err(error) if error.contains(LIMIT_REACHED) => break,
            Err(error) => {
                return Err(format!("parse error in {}: {error}", file_path.display()));
            }
        }

        if let Some(limits) = limits {
            if sessions.len() >= limits.max_sessions
                || queries.len() >= limits.max_queries
                || current_records >= limits.max_records
            {
                break;
            }
        }
    }

    if processed_files == 0 {
        return Err(format!(
            "cannot read {}: file not found, and no LongMemEval raw cache files were found in {}",
            data_dir.join(LME_CACHE_FILE).display(),
            data_dir.display(),
        ));
    }

    Ok(CognitiveDataset {
        name: format!("LongMemEval ({})", data_dir.display()),
        benchmark: Benchmark::H1Retrieval,
        sessions,
        queries,
    })
}

/// Download the LongMemEval dataset from HuggingFace and cache it locally.
///
/// Returns the path to the cache directory containing `cases.json`.
pub fn download_longmemeval(cache_dir: &Path) -> Result<PathBuf, String> {
    let cache_path = cache_dir.join(LME_CACHE_FILE);
    let marker_path = cache_dir.join(LME_MARKER);
    let auth_token = resolve_huggingface_auth_token();
    let raw_files_cached = LME_HF_FILES
        .iter()
        .all(|file_name| cache_dir.join(file_name).exists());

    // Check cache.
    if marker_path.exists() && (cache_path.exists() || raw_files_cached) {
        eprintln!(
            "LongMemEval dataset already cached at {}",
            cache_dir.display()
        );
        return Ok(cache_dir.to_path_buf());
    }

    std::fs::create_dir_all(cache_dir)
        .map_err(|e| format!("cannot create cache dir {}: {e}", cache_dir.display()))?;

    eprintln!("Downloading LongMemEval dataset from HuggingFace ({LME_HF_REPO})...");

    let mut available_files = 0usize;
    for file_name in LME_HF_FILES {
        let file_path = cache_dir.join(file_name);
        if file_path.exists() {
            eprintln!("  using cached {file_name}");
            available_files += 1;
            continue;
        }

        eprintln!("  downloading {file_name}");
        download_huggingface_file(LME_HF_REPO, file_name, &file_path, auth_token.as_deref())?;
        available_files += 1;
    }

    if available_files == 0 {
        return Err("no LongMemEval files retrieved from HuggingFace".to_string());
    }

    std::fs::write(&marker_path, format!("{} files", available_files))
        .map_err(|e| format!("failed to write marker file: {e}"))?;

    eprintln!(
        "LongMemEval dataset cached: {} files at {}",
        available_files,
        cache_dir.display(),
    );

    Ok(cache_dir.to_path_buf())
}

/// Download (if needed) and load the LongMemEval dataset, returning a `CognitiveDataset`.
pub fn load_longmemeval_cached(cache_dir: &Path) -> Result<CognitiveDataset, String> {
    let data_dir = download_longmemeval(cache_dir)?;
    load_longmemeval(&data_dir)
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    use super::*;

    fn single_request_server(status: u16, body: &str) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        let body = body.to_string();

        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            let mut buffer = [0u8; 8192];
            let read = stream.read(&mut buffer).unwrap();
            request.push_str(&String::from_utf8_lossy(&buffer[..read]));
            tx.send(request).unwrap();

            let status_text = match status {
                200 => "OK",
                401 => "Unauthorized",
                403 => "Forbidden",
                _ => "Error",
            };
            let response = format!(
                "HTTP/1.1 {status} {status_text}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        });

        (format!("http://{addr}"), rx)
    }

    #[test]
    fn token_file_reader_trims_whitespace() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("token");
        std::fs::write(&path, "  test-token\n").unwrap();

        assert_eq!(
            read_huggingface_token_file(&path).as_deref(),
            Some("test-token")
        );
    }

    #[test]
    fn download_huggingface_file_sends_bearer_token_when_provided() {
        let (base_url, request_rx) = single_request_server(200, r#"{"ok":true}"#);
        let repo = "owner/dataset";
        let url = format!("{base_url}/resolve/main/file.json");
        let dir = tempfile::TempDir::new().unwrap();
        let destination = dir.path().join("file.json");

        download_huggingface_file_from_url(&url, repo, &destination, Some("secret-token")).unwrap();
        assert_eq!(
            std::fs::read_to_string(&destination).unwrap(),
            r#"{"ok":true}"#
        );

        let request = request_rx.recv().unwrap().to_ascii_lowercase();
        assert!(request.contains("authorization: bearer secret-token"));
    }

    #[test]
    fn gated_download_error_points_to_auth_options() {
        let (base_url, _request_rx) = single_request_server(401, r#"{"error":"unauthorized"}"#);
        let repo = "xiaowu0162/longmemeval";
        let url = format!("{base_url}/resolve/main/file.json");
        let dir = tempfile::TempDir::new().unwrap();
        let destination = dir.path().join("file.json");

        let error = download_huggingface_file_from_url(&url, repo, &destination, None).unwrap_err();
        assert!(error.contains("dataset `xiaowu0162/longmemeval` may require authentication"));
        assert!(error.contains("HF_TOKEN"));
        assert!(error.contains("huggingface-cli login"));
        assert!(error.contains("--data-dir"));
    }

    #[test]
    fn locomo_repo_uses_canonical_github_source() {
        assert_eq!(LOCOMO_GITHUB_REPO, "snap-research/locomo");
        assert_eq!(LOCOMO_GITHUB_PATH, "data/locomo10.json");
    }

    #[test]
    fn load_locomo_supports_canonical_github_dataset() {
        let dir = tempfile::TempDir::new().unwrap();
        let raw_rows = serde_json::json!([
            {
                "sample_id": "sample-1",
                "conversation": {
                    "speaker_a": "Alice",
                    "speaker_b": "Bob",
                    "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "I moved to Seattle in 2022."},
                        {"speaker": "Bob", "dia_id": "D1:2", "text": "You started at Acme last month."}
                    ],
                    "session_1_date_time": "1:56 pm on 8 May, 2023"
                },
                "observation": {
                    "session_1_observation": {
                        "Alice": [["Alice moved to Seattle in 2022.", "D1:1"]]
                    }
                },
                "session_summary": {
                    "session_1_summary": "On 8 May, 2023 Alice told Bob she moved to Seattle in 2022."
                },
                "qa": [
                    {
                        "question": "Where did Alice move?",
                        "answer": "Seattle",
                        "evidence": ["D1:1"],
                        "category": 1
                    },
                    {
                        "question": "When did Alice move?",
                        "answer": 2022,
                        "evidence": ["D1:1"],
                        "category": 2
                    }
                ]
            }
        ]);
        std::fs::write(
            dir.path().join("locomo10.json"),
            serde_json::to_vec(&raw_rows).unwrap(),
        )
        .unwrap();

        let dataset = load_locomo(dir.path()).unwrap();
        assert_eq!(dataset.sessions.len(), 1);
        assert_eq!(dataset.sessions[0].turns.len(), 4);
        assert_eq!(dataset.queries.len(), 2);
        assert_eq!(dataset.sessions[0].id, "sample-1::session_1");
        assert_eq!(
            dataset.sessions[0].turns[0].source_id.as_deref(),
            Some("D1:1")
        );
        assert!(dataset.sessions[0].turns[0].timestamp.is_some());
        assert_eq!(
            dataset.sessions[0].turns[0].timestamp_text.as_deref(),
            Some("1:56 pm on 8 May, 2023")
        );
        assert_eq!(dataset.sessions[0].turns[2].speaker, "Observation/Alice");
        assert_eq!(
            dataset.sessions[0].turns[2].source_id.as_deref(),
            Some("D1:1")
        );
        assert_eq!(dataset.sessions[0].turns[3].speaker, "SessionSummary");

        let where_query = dataset
            .queries
            .iter()
            .find(|query| query.question == "Where did Alice move?")
            .unwrap();
        assert_eq!(where_query.expected_answers, vec!["Seattle"]);
        assert_eq!(where_query.category, "multi-hop");
        assert_eq!(where_query.evidence_ids, vec!["D1:1"]);
        assert_eq!(
            where_query.evidence_snippets,
            vec!["I moved to Seattle in 2022."]
        );
        assert_eq!(
            where_query.relevant_session_ids,
            vec!["sample-1::session_1"]
        );

        let when_query = dataset
            .queries
            .iter()
            .find(|query| query.question == "When did Alice move?")
            .unwrap();
        assert_eq!(when_query.expected_answers, vec!["2022"]);
        assert_eq!(when_query.category, "temporal");
    }

    #[test]
    fn dmr_auto_download_requires_local_mirror_when_cache_is_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        let error = download_dmr(dir.path()).unwrap_err();
        assert!(error.contains("DMR auto-download is currently disabled"));
        assert!(error.contains("--data-dir"));
        assert!(error.contains("dialogs.json"));
    }

    #[test]
    fn longmemeval_repo_uses_canonical_lowercase_slug() {
        assert_eq!(LME_HF_REPO, "xiaowu0162/longmemeval");
    }

    #[test]
    fn load_longmemeval_supports_raw_huggingface_files() {
        let dir = tempfile::TempDir::new().unwrap();
        let raw_rows = serde_json::json!([
            {
                "question_id": "q-1",
                "question": "Where does Alice work now?",
                "answer": 3,
                "question_type": "knowledge_update",
                "answer_session_ids": ["alice-1"],
                "haystack_session_ids": ["alice-0", "alice-1"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "I used to work at Google in Seattle."},
                        {"role": "assistant", "content": "That sounds exciting."}
                    ],
                    [
                        {"role": "user", "content": "I joined Microsoft last month and moved to Bellevue."}
                    ]
                ]
            },
            {
                "question_id": "q-2",
                "question": "What is Alice's salary?",
                "answer": "",
                "question_type": "abstention",
                "answer_session_ids": [],
                "haystack_session_ids": ["alice-0", "alice-1"],
                "haystack_sessions": [
                    [
                        {"role": "user", "content": "I used to work at Google in Seattle."},
                        {"role": "assistant", "content": "That sounds exciting."}
                    ],
                    [
                        {"role": "user", "content": "I joined Microsoft last month and moved to Bellevue."}
                    ]
                ]
            }
        ]);
        std::fs::write(
            dir.path().join("longmemeval_oracle"),
            serde_json::to_vec(&raw_rows).unwrap(),
        )
        .unwrap();

        let dataset = load_longmemeval(dir.path()).unwrap();
        assert_eq!(dataset.sessions.len(), 2);
        assert_eq!(dataset.queries.len(), 2);
        assert_eq!(dataset.queries[0].expected_answers, vec!["3"]);
        assert_eq!(
            dataset.queries[0].relevant_session_ids,
            vec!["alice-0", "alice-1"]
        );
        assert!(dataset.queries[1].negative);
    }
}
