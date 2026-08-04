//! Deterministic natural-language time-expression parser.
//!
//! Extracts an explicit time frame from a free-text query and maps it to a
//! half-open `[after, before)` interval in epoch milliseconds. This is the bridge
//! that lets the Allen-interval temporal-relevance term (and bi-temporal `AS OF`)
//! fire on free-text questions like "what did I buy in 2023?" or "meetings last
//! month" — without it, the temporal machinery only activates when a caller
//! supplies explicit bounds, so temporal QA never benefits.
//!
//! It is intentionally **conservative and high-precision**: it returns a frame
//! only for unambiguous, explicit temporal references (calendar years, month +
//! year, `last/this` week/month/year, `before/after/since/until` a year, year
//! ranges, `today`/`yesterday`). Anything it isn't sure about yields no frame, so
//! a non-temporal query is never accidentally constrained (which would hurt
//! recall). No LLM, no regex — pure token scanning + `chrono` calendar math, so
//! the same query always yields the same frame.

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};

/// A parsed time frame as a half-open `[after, before)` interval in epoch ms.
/// Either bound may be open (`None`). `after` is inclusive, `before` exclusive —
/// matching the recall builder's `after`/`before` semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParsedTimeFrame {
    pub after_ms: Option<i64>,
    pub before_ms: Option<i64>,
}

impl ParsedTimeFrame {
    /// Whether any bound was parsed.
    #[must_use]
    pub const fn matched(&self) -> bool {
        self.after_ms.is_some() || self.before_ms.is_some()
    }
}

fn start_of_year_ms(year: i32) -> Option<i64> {
    NaiveDate::from_ymd_opt(year, 1, 1)?
        .and_hms_opt(0, 0, 0)?
        .and_utc()
        .timestamp_millis()
        .into()
}

fn start_of_month_ms(year: i32, month: u32) -> Option<i64> {
    NaiveDate::from_ymd_opt(year, month, 1)?
        .and_hms_opt(0, 0, 0)?
        .and_utc()
        .timestamp_millis()
        .into()
}

/// First instant of the month after `(year, month)`, handling year rollover.
fn start_of_next_month_ms(year: i32, month: u32) -> Option<i64> {
    let (ny, nm) = if month >= 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    start_of_month_ms(ny, nm)
}

fn month_from_word(word: &str) -> Option<u32> {
    let m = match word {
        "january" | "jan" => 1,
        "february" | "feb" => 2,
        "march" | "mar" => 3,
        "april" | "apr" => 4,
        "may" => 5,
        "june" | "jun" => 6,
        "july" | "jul" => 7,
        "august" | "aug" => 8,
        "september" | "sep" | "sept" => 9,
        "october" | "oct" => 10,
        "november" | "nov" => 11,
        "december" | "dec" => 12,
        _ => return None,
    };
    Some(m)
}

/// Parse a bare calendar year in the plausible range for user memories.
fn parse_year(tok: &str) -> Option<i32> {
    if tok.len() != 4 || !tok.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let y: i32 = tok.parse().ok()?;
    (1970..=2099).contains(&y).then_some(y)
}

/// Extract an explicit time frame from `query`, relative to `now_ms`.
///
/// Returns an empty (unmatched) frame when no unambiguous temporal expression is
/// present.
#[must_use]
pub fn parse_time_frame(query: &str, now_ms: i64) -> ParsedTimeFrame {
    let now: DateTime<Utc> = DateTime::from_timestamp_millis(now_ms).unwrap_or_else(Utc::now);
    let lower = query.to_lowercase();
    // Split on non-alphanumeric so "2023," / "march." tokenize cleanly.
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    // ── Relative phrases (checked on the raw lowercased string) ──────────
    if lower.contains("yesterday") {
        let y = (now - Duration::days(1)).date_naive();
        let start = y
            .and_hms_opt(0, 0, 0)
            .map(|d| d.and_utc().timestamp_millis());
        let end = now
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .map(|d| d.and_utc().timestamp_millis());
        return ParsedTimeFrame {
            after_ms: start,
            before_ms: end,
        };
    }
    if lower.contains("today") {
        let start = now
            .date_naive()
            .and_hms_opt(0, 0, 0)
            .map(|d| d.and_utc().timestamp_millis());
        return ParsedTimeFrame {
            after_ms: start,
            before_ms: None,
        };
    }
    let this_year = now.year();
    if lower.contains("last year") || lower.contains("previous year") {
        return ParsedTimeFrame {
            after_ms: start_of_year_ms(this_year - 1),
            before_ms: start_of_year_ms(this_year),
        };
    }
    if lower.contains("this year") {
        return ParsedTimeFrame {
            after_ms: start_of_year_ms(this_year),
            before_ms: start_of_year_ms(this_year + 1),
        };
    }
    if lower.contains("last month") || lower.contains("previous month") {
        let (y, m) = (now.year(), now.month());
        let (py, pm) = if m <= 1 { (y - 1, 12) } else { (y, m - 1) };
        return ParsedTimeFrame {
            after_ms: start_of_month_ms(py, pm),
            before_ms: start_of_month_ms(y, m),
        };
    }
    if lower.contains("this month") {
        return ParsedTimeFrame {
            after_ms: start_of_month_ms(now.year(), now.month()),
            before_ms: start_of_next_month_ms(now.year(), now.month()),
        };
    }
    if lower.contains("last week") || lower.contains("past week") || lower.contains("previous week")
    {
        let end = now.timestamp_millis();
        let start = (now - Duration::weeks(1)).timestamp_millis();
        return ParsedTimeFrame {
            after_ms: Some(start),
            before_ms: Some(end),
        };
    }

    // ── Month + year ("march 2024", "in january 2023") ──────────────────
    for i in 0..tokens.len() {
        if let Some(month) = month_from_word(tokens[i]) {
            if let Some(year) = tokens.get(i + 1).and_then(|t| parse_year(t)) {
                return ParsedTimeFrame {
                    after_ms: start_of_month_ms(year, month),
                    before_ms: start_of_next_month_ms(year, month),
                };
            }
        }
    }

    // ── Directional / range year expressions ────────────────────────────
    // "between 2020 and 2022", "from 2019 to 2021".
    for i in 0..tokens.len() {
        if (tokens[i] == "between" || tokens[i] == "from")
            && let Some(y1) = tokens.get(i + 1).and_then(|t| parse_year(t))
            && matches!(
                tokens.get(i + 2),
                Some(&("and" | "to" | "through" | "until"))
            )
            && let Some(y2) = tokens.get(i + 3).and_then(|t| parse_year(t))
        {
            let (lo, hi) = if y1 <= y2 { (y1, y2) } else { (y2, y1) };
            return ParsedTimeFrame {
                after_ms: start_of_year_ms(lo),
                before_ms: start_of_year_ms(hi + 1),
            };
        }
    }
    // "before 2023" / "after 2021" / "since 2020" / "until 2022" / "by 2024".
    for i in 0..tokens.len() {
        let Some(year) = tokens.get(i + 1).and_then(|t| parse_year(t)) else {
            continue;
        };
        match tokens[i] {
            "before" | "until" | "by" => {
                return ParsedTimeFrame {
                    after_ms: None,
                    before_ms: start_of_year_ms(year),
                };
            }
            "after" | "since" => {
                return ParsedTimeFrame {
                    after_ms: start_of_year_ms(year),
                    before_ms: None,
                };
            }
            _ => {}
        }
    }

    // ── Bare year mention ("in 2023", "during 2024", or just "2023") ────
    for tok in &tokens {
        if let Some(year) = parse_year(tok) {
            return ParsedTimeFrame {
                after_ms: start_of_year_ms(year),
                before_ms: start_of_year_ms(year + 1),
            };
        }
    }

    ParsedTimeFrame::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ms(y: i32, m: u32, d: u32) -> i64 {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0)
            .unwrap()
            .timestamp_millis()
    }

    // A fixed "now" = 2026-07-15 for deterministic relative parsing.
    fn now() -> i64 {
        ms(2026, 7, 15)
    }

    #[test]
    fn bare_year() {
        let f = parse_time_frame("what did I buy in 2023?", now());
        assert_eq!(f.after_ms, Some(ms(2023, 1, 1)));
        assert_eq!(f.before_ms, Some(ms(2024, 1, 1)));
    }

    #[test]
    fn month_and_year() {
        let f = parse_time_frame("meetings in March 2024", now());
        assert_eq!(f.after_ms, Some(ms(2024, 3, 1)));
        assert_eq!(f.before_ms, Some(ms(2024, 4, 1)));
    }

    #[test]
    fn month_year_december_rolls_over() {
        let f = parse_time_frame("notes from December 2023", now());
        assert_eq!(f.after_ms, Some(ms(2023, 12, 1)));
        assert_eq!(f.before_ms, Some(ms(2024, 1, 1)));
    }

    #[test]
    fn last_year() {
        let f = parse_time_frame("what happened last year", now());
        assert_eq!(f.after_ms, Some(ms(2025, 1, 1)));
        assert_eq!(f.before_ms, Some(ms(2026, 1, 1)));
    }

    #[test]
    fn last_month_rolls_over_january() {
        let jan_now = ms(2026, 1, 10);
        let f = parse_time_frame("expenses last month", jan_now);
        assert_eq!(f.after_ms, Some(ms(2025, 12, 1)));
        assert_eq!(f.before_ms, Some(ms(2026, 1, 1)));
    }

    #[test]
    fn before_and_after_year() {
        let b = parse_time_frame("anything before 2022", now());
        assert_eq!(b.after_ms, None);
        assert_eq!(b.before_ms, Some(ms(2022, 1, 1)));

        let a = parse_time_frame("changes since 2021", now());
        assert_eq!(a.after_ms, Some(ms(2021, 1, 1)));
        assert_eq!(a.before_ms, None);
    }

    #[test]
    fn year_range() {
        let f = parse_time_frame("trips between 2019 and 2021", now());
        assert_eq!(f.after_ms, Some(ms(2019, 1, 1)));
        assert_eq!(f.before_ms, Some(ms(2022, 1, 1)));
    }

    #[test]
    fn non_temporal_query_yields_nothing() {
        let f = parse_time_frame("what is my favorite programming language?", now());
        assert!(!f.matched());
        // A stray non-year number must not be treated as a year.
        let f2 = parse_time_frame("how many times did I run 5 miles?", now());
        assert!(!f2.matched());
    }
}
