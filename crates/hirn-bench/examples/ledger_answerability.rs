//! Diagnostic: can the temporal ledger *answer* the duration questions at all?
//!
//! The ledger measured a null on LongMemEval's temporal slice (+2 of 133). That
//! leaves two very different explanations, which an accuracy number cannot
//! separate:
//!
//! 1. **Presentation** — the ledger holds the right interval and the reader
//!    fails to pick it out.
//! 2. **Capability** — the right interval is not in the ledger at all, in which
//!    case no amount of prompt work can help.
//!
//! This checks (2) directly. For every duration question ("how many days
//! between X and Y") it builds the ledger from the question's own haystack and
//! asks whether the gold day-count appears among the computed pairwise
//! intervals. If it does not, the mechanism cannot answer the question however
//! it is framed.
//!
//! ```text
//! cargo run --release -p hirn-bench --example ledger_answerability -- <oracle-json>
//! ```

use hirn_core::Timestamp;
use hirn_core::temporal_ledger::build_ledger;

const SAMPLES_TO_SHOW: usize = 3;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!(
            "usage: ledger_answerability <path-to-longmemeval-oracle-json>\n\
             reports whether the gold interval appears in the computed ledger"
        );
        std::process::exit(2);
    };

    let raw = match std::fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) => {
            eprintln!("cannot read {path}: {error}");
            std::process::exit(2);
        }
    };
    let data: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(data) => data,
        Err(error) => {
            eprintln!("{path} is not valid JSON: {error}");
            std::process::exit(2);
        }
    };

    let mut duration_questions = 0usize;
    let mut with_ledger = 0usize;
    let mut answerable = 0usize;
    let mut shown = 0usize;

    for question in data.as_array().into_iter().flatten() {
        if question["question_type"].as_str() != Some("temporal-reasoning") {
            continue;
        }
        let text = question["question"].as_str().unwrap_or_default();
        let Some(gold_days) = gold_day_count(question["answer"].as_str().unwrap_or_default())
        else {
            continue;
        };
        if !is_duration_question(text) {
            continue;
        }
        duration_questions += 1;

        // Anchor each turn to its own session's date. Anchoring the whole
        // haystack on the question date collapses every "today" onto one
        // instant, which is what made the ledger emit `0 day(s)`.
        let question_date = parse_reference(question["question_date"].as_str().unwrap_or_default());
        let session_dates: Vec<Timestamp> = question["haystack_dates"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|value| {
                value
                    .as_str()
                    .map_or(question_date, |text| parse_reference(text))
            })
            .collect();

        let mut lines: Vec<(String, Timestamp)> = Vec::new();
        for (session_index, session) in question["haystack_sessions"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            let anchor = session_dates.get(session_index).copied().unwrap_or(question_date);
            for turn in session.as_array().into_iter().flatten() {
                if let Some(content) = turn["content"].as_str() {
                    lines.push((content.to_owned(), anchor));
                }
            }
        }

        let entries: Vec<(&str, Timestamp)> = lines
            .iter()
            .map(|(line, anchor)| (line.as_str(), *anchor))
            .collect();
        let ledger = build_ledger(&entries);
        if !ledger.is_useful() {
            continue;
        }
        with_ledger += 1;

        // The gold answers accept an off-by-one ("7 days. 8 days including the
        // last day is also acceptable"), so both are treated as a hit.
        let rendered = ledger.render();
        let intervals = intervals_in(&rendered);
        let hit = intervals
            .iter()
            .any(|days| *days == gold_days || *days == gold_days + 1);
        if hit {
            answerable += 1;
        } else if shown < SAMPLES_TO_SHOW {
            shown += 1;
            println!("--- MISS: {text}");
            println!("gold: {gold_days} days");
            println!("computed intervals: {intervals:?}");
            print!("{rendered}");
            println!();
        }
    }

    println!("duration questions:              {duration_questions}");
    println!("  ...with a usable ledger:       {with_ledger}");
    println!("  ...whose ledger holds the gold interval: {answerable}");
    if with_ledger > 0 {
        println!(
            "answerable share of those with a ledger: {:.1}%",
            100.0 * answerable as f64 / with_ledger as f64
        );
    }
}

fn is_duration_question(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("how many days")
        || lower.contains("how many weeks")
        || lower.contains("how many months")
        || lower.contains("how long")
}

/// Pull the leading day count out of a gold answer like "7 days. 8 days ...".
fn gold_day_count(answer: &str) -> Option<i64> {
    let lower = answer.to_ascii_lowercase();
    let index = lower.find(" day")?;
    let prefix = &lower[..index];
    let digits: String = prefix
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect();
    if digits.is_empty() {
        return None;
    }
    digits.chars().rev().collect::<String>().parse().ok()
}

/// Every day-count the rendered ledger states.
fn intervals_in(rendered: &str) -> Vec<i64> {
    let mut found = Vec::new();
    for line in rendered.lines() {
        let lower = line.to_ascii_lowercase();
        let Some(index) = lower.find(" day") else {
            continue;
        };
        let digits: String = lower[..index]
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .collect();
        if digits.is_empty() {
            continue;
        }
        if let Ok(days) = digits.chars().rev().collect::<String>().parse::<i64>() {
            found.push(days);
        }
    }
    found
}

/// Parse LongMemEval's `YYYY/MM/DD (Day) HH:MM`; the weekday is decorative.
fn parse_reference(text: &str) -> Timestamp {
    let cleaned: String = text
        .split_whitespace()
        .filter(|token| !token.starts_with('('))
        .collect::<Vec<_>>()
        .join(" ");
    chrono::NaiveDateTime::parse_from_str(&cleaned, "%Y/%m/%d %H:%M")
        .map(|date| date.and_utc().timestamp_millis())
        .ok()
        .and_then(|ms| u64::try_from(ms).ok())
        .map_or_else(Timestamp::now, Timestamp::from_millis)
}
