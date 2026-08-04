//! Diagnostic: what does the symbolic temporal ledger actually produce on a
//! real corpus?
//!
//! Written because a benchmark run cannot distinguish a *working* ledger from a
//! *garbage* one — both yield a number. Running this first caught month names
//! being read out of ordinary prose ("you **may** find that…") and 730,386-day
//! intervals reaching a block the reader is instructed to treat as exact. The
//! benchmark would have measured that bug and reported it as a result.
//!
//! ```text
//! cargo run --release -p hirn-bench --example ledger_probe -- <oracle-json>
//! ```
//!
//! Prints a sample of rendered ledgers plus the share of temporal-reasoning
//! questions that yield a usable one. A *drop* in that share is not necessarily
//! a regression: tightening what counts as a date lowers coverage and raises
//! quality, so read it alongside the rendered samples.

use hirn_core::Timestamp;
use hirn_core::temporal_ledger::build_ledger;

const SAMPLES_TO_SHOW: usize = 2;

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!(
            "usage: ledger_probe <path-to-longmemeval-oracle-json>\n\
             prints rendered ledgers plus usable-ledger coverage"
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

    let mut usable = 0usize;
    let mut total = 0usize;
    let mut shown = 0usize;

    for question in data.as_array().into_iter().flatten() {
        if question["question_type"].as_str() != Some("temporal-reasoning") {
            continue;
        }
        total += 1;

        // Approximate the retrieved context with the haystack turns.
        let lines: Vec<String> = question["haystack_sessions"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|session| session.as_array())
            .flatten()
            .filter_map(|turn| turn["content"].as_str())
            .map(str::to_owned)
            .collect();

        let reference = parse_reference(question["question_date"].as_str().unwrap_or_default());
        let entries: Vec<(&str, Timestamp)> = lines
            .iter()
            .map(|line| (line.as_str(), reference))
            .collect();
        let ledger = build_ledger(&entries);

        if ledger.is_useful() {
            usable += 1;
            if shown < SAMPLES_TO_SHOW {
                shown += 1;
                println!("--- {}", question["question"].as_str().unwrap_or_default());
                println!("gold: {}", question["answer"].as_str().unwrap_or_default());
                print!("{}", ledger.render());
                println!();
            }
        }
    }

    println!("temporal questions with a usable ledger: {usable}/{total}");
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
