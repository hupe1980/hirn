//! Symbolic temporal ledger: date arithmetic computed in Rust, not by the model.
//!
//! LongMemEval's temporal-reasoning slice is dominated by two question shapes:
//!
//! - **Ordering** — "which did I attend first, the workshop or the webinar?"
//! - **Duration** — "how many days between the mass and the service?"
//!
//! Both are arithmetic over dates that appear *in the retrieved text* ("on
//! January 10th", "last Tuesday"), not in record metadata. hirn measured 0.7754
//! retrieval containment against 0.3985 answer accuracy on that slice: the
//! evidence is being found and then reasoned over incorrectly, because the
//! reader prompt asks the model to "order dated events and compute the
//! requested interval" — the documented failure mode of LLMs at date
//! arithmetic.
//!
//! This module closes that gap. It scans retrieved excerpts for date
//! expressions, resolves them against each excerpt's own reference time,
//! sorts them, computes pairwise intervals, and renders a compact ledger the
//! reader is told **not to recompute**. Everything here is deterministic and
//! unit-testable without a model.
//!
//! It deliberately does *not* try to decide which two events a question refers
//! to. That is a semantic match the model is good at and symbolic code is bad
//! at; the division of labour is: hirn supplies exact dates and exact
//! intervals, the model picks which ones the question meant.

use std::collections::HashSet;

use crate::svo_event::parse_temporal_text;
use crate::temporal::TimePrecision;
use crate::timestamp::Timestamp;

/// Maximum dated events retained in one ledger.
///
/// Bounded because the ledger is injected into a token-budgeted reader prompt,
/// and because pairwise intervals grow quadratically.
pub const MAX_LEDGER_EVENTS: usize = 12;

/// How far from the conversation a date may sit and still count as a personal
/// event, in days (25 years).
///
/// Retrieved text contains dates that parse correctly but are *content*, not
/// memories: "the attack on December 7, 1941" in a travel recommendation is a
/// real date and not something the user did. Paired against every genuine
/// event it contributed five 29,000-day intervals to one ledger — a third of
/// the block, all noise, presented with the same authority as the real ones.
///
/// This is a heuristic and it has a cost: a genuinely ancient personal date
/// ("I was born in 1961") is dropped from a present-day conversation. That
/// trade is deliberate — such dates do not appear in interval questions, while
/// historical references demonstrably do.
const PERSONAL_HORIZON_DAYS: i64 = 25 * 365;

/// Ledger size at or below which every pair of dates gets an explicit interval.
///
/// Six events is fifteen pairs — enough that "how many days between X and Y"
/// is answered directly for the common case, without quadratic growth crowding
/// a token-budgeted prompt.
const PAIRWISE_INTERVAL_LIMIT: usize = 6;

/// Maximum characters of surrounding context kept per dated event.
const SNIPPET_CHARS: usize = 110;

/// One date found in retrieved text, with the phrase around it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatedMention {
    /// Resolved absolute date.
    pub date: Timestamp,
    /// How precisely the source expression pinned it.
    pub precision: TimePrecision,
    /// The date expression exactly as written.
    pub expression: String,
    /// Surrounding text, so the reader can tell which event this is.
    pub snippet: String,
}

/// Dated events found across retrieved excerpts, plus their intervals.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TemporalLedger {
    /// Dated mentions, earliest first, deduplicated by (date, snippet).
    pub events: Vec<DatedMention>,
}

impl TemporalLedger {
    /// Whether the ledger has enough to support arithmetic.
    ///
    /// One date supports no interval and no ordering, so a single-entry ledger
    /// is not worth the tokens it would cost.
    #[must_use]
    pub fn is_useful(&self) -> bool {
        self.events.len() >= 2
    }

    /// Whole days between the earliest and latest event.
    #[must_use]
    pub fn span_days(&self) -> Option<i64> {
        let first = self.events.first()?;
        let last = self.events.last()?;
        Some(days_between(first.date, last.date))
    }

    /// Render the ledger for a reader prompt.
    ///
    /// Empty when there is nothing arithmetic to say, so callers can splice the
    /// result in unconditionally without emitting a useless header.
    #[must_use]
    pub fn render(&self) -> String {
        if !self.is_useful() {
            return String::new();
        }

        let mut out = String::with_capacity(512);
        out.push_str(
            "Computed temporal ledger (dates resolved and intervals calculated \
             from them; use these values directly and do not recompute them. \
             An interval marked \"approximate\" has an endpoint that names only \
             a month or year, so it cannot be day-exact):\n",
        );
        for event in &self.events {
            out.push_str(&format!(
                "- {} [{}] — {}\n",
                format_date(event.date),
                event.precision.as_str(),
                event.snippet
            ));
        }

        // "How many days between X and Y" rarely asks about *consecutive*
        // events, so for a small ledger every pair is listed and the model
        // never has to add intervals together. Above that the pairing grows
        // quadratically and would crowd the prompt, so it degrades to
        // consecutive gaps.
        //
        // No earliest-to-latest summary is emitted. Retrieved text legitimately
        // contains dates that are not personal events — a historical reference
        // like "the attack on December 7, 1941" parses correctly — and a span
        // line across an unrelated pair presents a meaningless 29,637-day
        // figure with the same authority as a real interval.
        out.push_str("Intervals:\n");
        if self.events.len() <= PAIRWISE_INTERVAL_LIMIT {
            for (i, from) in self.events.iter().enumerate() {
                for to in self.events.iter().skip(i + 1) {
                    out.push_str(&render_interval(from, to));
                }
            }
        } else {
            for pair in self.events.windows(2) {
                out.push_str(&render_interval(&pair[0], &pair[1]));
            }
        }
        out
    }
}

/// Render one interval, marking it approximate when its endpoints cannot
/// support a day-exact answer.
///
/// "March 2021" pins a month, not a day, so an interval touching it is
/// uncertain by up to that month's length. The surrounding block tells the
/// reader these values are exact; a month-precise endpoint silently violates
/// that, and a confidently wrong day count is worse than an acknowledged
/// approximation.
fn render_interval(from: &DatedMention, to: &DatedMention) -> String {
    let days = days_between(from.date, to.date);
    let uncertainty = interval_uncertainty_days(from.precision, to.precision);
    if uncertainty == 0 {
        format!(
            "- {} → {}: {days} day(s)\n",
            format_date(from.date),
            format_date(to.date),
        )
    } else {
        format!(
            "- {} → {}: ~{days} day(s) (approximate: ±{uncertainty} days, \
             one endpoint is only {}-precise)\n",
            format_date(from.date),
            format_date(to.date),
            coarser(from.precision, to.precision).as_str(),
        )
    }
}

/// Worst-case day error contributed by the coarser of two endpoints.
fn interval_uncertainty_days(left: TimePrecision, right: TimePrecision) -> i64 {
    precision_slack_days(left).max(precision_slack_days(right))
}

const fn precision_slack_days(precision: TimePrecision) -> i64 {
    match precision {
        TimePrecision::Instant | TimePrecision::Day => 0,
        TimePrecision::Month => 31,
        TimePrecision::Year => 365,
        // An unknown-precision date was still resolved to a specific day; it is
        // reported as day-exact rather than inventing a slack figure, because
        // the scanner only admits expressions it could pin.
        TimePrecision::Unknown => 0,
    }
}

const fn coarser(left: TimePrecision, right: TimePrecision) -> TimePrecision {
    if precision_slack_days(left) >= precision_slack_days(right) {
        left
    } else {
        right
    }
}

/// Reject dates outside a plausible conversational range.
///
/// An ambiguous token can parse to year 0023 or 9999, producing intervals in
/// the hundreds of thousands of days — not merely wrong but conspicuously wrong
/// inside a block the reader is instructed to trust.
fn is_plausible_year(ts: Timestamp) -> bool {
    use chrono::Datelike;
    (1900..=2200).contains(&ts.as_datetime().year())
}

/// Whole days from `from` to `to` (absolute).
fn days_between(from: Timestamp, to: Timestamp) -> i64 {
    let ms = (to.timestamp_ms() - from.timestamp_ms()).abs();
    ms / 86_400_000
}

/// Render a date as `YYYY-MM-DD (Day)`.
fn format_date(ts: Timestamp) -> String {
    ts.as_datetime().format("%Y-%m-%d (%a)").to_string()
}

/// Month names recognised when scanning for date expressions.
const MONTHS: &[&str] = &[
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
];

/// Multi-word relative expressions worth resolving.
const RELATIVE_PHRASES: &[&str] = &[
    "yesterday",
    "today",
    "last week",
    "last month",
    "this morning",
];

/// Scan one excerpt for date expressions and resolve them against `reference`.
///
/// `reference` should be the excerpt's own timestamp — a bare "January 10th"
/// needs a year, and a relative phrase needs an anchor. Using a global "now"
/// would misdate every historical excerpt.
#[must_use]
pub fn scan_dated_mentions(text: &str, reference: Timestamp) -> Vec<DatedMention> {
    let mut found = Vec::new();
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = text.split_whitespace().collect();

    // ISO dates and "Month D[, YYYY]" forms.
    for (index, token) in tokens.iter().enumerate() {
        let cleaned = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
        if cleaned.is_empty() {
            continue;
        }

        // `YYYY-MM-DD`
        if cleaned.len() == 10 && cleaned.matches('-').count() == 2 {
            if let Some(date) = parse_temporal_text(cleaned, reference)
                && is_plausible_year(date)
            {
                found.push(mention(
                    date,
                    TimePrecision::Day,
                    cleaned,
                    text,
                    index,
                    &tokens,
                ));
                continue;
            }
        }

        // `Month D` / `Month D, YYYY` / `Month YYYY`.
        //
        // A **bare** month name is deliberately rejected. English month names
        // collide with ordinary vocabulary — "may" and "march" are far more
        // often a modal verb and a verb than a date. Probing real conversation
        // data, an earlier version of this scanner read "you *may* find that…"
        // as May, resolved it to year 0023, and emitted a 730,386-day interval
        // into a block the reader is told is exact. A month name counts as a
        // date only when a day number or a 4-digit year corroborates it.
        if MONTHS.contains(&cleaned.to_lowercase().as_str()) {
            let Some(next) = tokens.get(index + 1) else {
                continue;
            };
            let following = next.trim_matches(|c: char| !c.is_alphanumeric());
            let numeric: String = following.chars().take_while(char::is_ascii_digit).collect();

            let (expression, precision) = if !numeric.is_empty() && numeric.len() <= 2 {
                let mut expression = format!("{cleaned} {following}");
                if let Some(year) = tokens.get(index + 2) {
                    let year = year.trim_matches(|c: char| !c.is_alphanumeric());
                    if year.len() == 4 && year.chars().all(|c| c.is_ascii_digit()) {
                        expression = format!("{cleaned} {following} {year}");
                    }
                }
                (expression, TimePrecision::Day)
            } else if following.len() == 4 && following.chars().all(|c| c.is_ascii_digit()) {
                (format!("{cleaned} {following}"), TimePrecision::Month)
            } else {
                continue;
            };

            if let Some(date) = parse_temporal_text(&expression, reference)
                && is_plausible_year(date)
            {
                found.push(mention(date, precision, &expression, text, index, &tokens));
            }
        }
    }

    // Relative phrases, matched on the lowercased text.
    for phrase in RELATIVE_PHRASES {
        if let Some(position) = lower.find(phrase)
            && let Some(date) = parse_temporal_text(phrase, reference)
        {
            let token_index = text[..position].split_whitespace().count();
            found.push(mention(
                date,
                TimePrecision::Day,
                phrase,
                text,
                token_index,
                &tokens,
            ));
        }
    }

    found
}

/// Build a mention with a snippet centred on the date expression.
fn mention(
    date: Timestamp,
    precision: TimePrecision,
    expression: &str,
    text: &str,
    token_index: usize,
    tokens: &[&str],
) -> DatedMention {
    let start = token_index.saturating_sub(9);
    let end = (token_index + 10).min(tokens.len());
    let mut snippet = tokens[start..end].join(" ");
    if snippet.chars().count() > SNIPPET_CHARS {
        snippet = snippet.chars().take(SNIPPET_CHARS).collect();
    }
    if snippet.trim().is_empty() {
        snippet = text.chars().take(SNIPPET_CHARS).collect();
    }
    DatedMention {
        date,
        precision,
        expression: expression.to_string(),
        snippet: snippet.trim().to_string(),
    }
}

/// Build a ledger from retrieved excerpts.
///
/// Each entry is `(excerpt_text, excerpt_reference_time)`. Mentions are
/// deduplicated by resolved date **and** snippet: the same date restated across
/// several turns is one event, but two different events on one date are two.
#[must_use]
pub fn build_ledger(entries: &[(&str, Timestamp)]) -> TemporalLedger {
    let mut events: Vec<DatedMention> = Vec::new();
    let mut seen: HashSet<(i64, String)> = HashSet::new();

    for (text, reference) in entries {
        for found in scan_dated_mentions(text, *reference) {
            // Drop dates outside the conversation's own lifetime — see
            // `PERSONAL_HORIZON_DAYS`.
            if days_between(found.date, *reference) > PERSONAL_HORIZON_DAYS {
                continue;
            }
            let key = (found.date.timestamp_ms(), found.snippet.to_lowercase());
            if seen.insert(key) {
                events.push(found);
            }
        }
    }

    events.sort_by_key(|event| event.date.timestamp_ms());
    // Keep the earliest events: a question about ordering or duration is
    // almost always about the dated events themselves, and truncating the tail
    // is less damaging than truncating the head of the timeline.
    events.truncate(MAX_LEDGER_EVENTS);
    TemporalLedger { events }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn january_2023() -> Timestamp {
        // 2023-01-13, the reference date used by the worked LongMemEval example.
        Timestamp::from_millis(1_673_568_000_000)
    }

    fn mention(date_ms: u64, precision: TimePrecision) -> DatedMention {
        DatedMention {
            date: Timestamp::from_millis(date_ms),
            precision,
            expression: "expr".to_string(),
            snippet: "snippet".to_string(),
        }
    }

    /// A month-precise endpoint cannot support a day-exact interval, and the
    /// surrounding block tells the reader the numbers are usable directly.
    #[test]
    fn month_precise_endpoints_render_an_approximate_interval() {
        let ledger = TemporalLedger {
            events: vec![
                mention(1_614_556_800_000, TimePrecision::Month), // 2021-03-01
                mention(1_617_235_200_000, TimePrecision::Day),   // 2021-04-01
            ],
        };
        let rendered = ledger.render();
        assert!(rendered.contains("approximate"), "{rendered}");
        assert!(rendered.contains("±31 days"), "{rendered}");
        assert!(rendered.contains("month-precise"), "{rendered}");
    }

    #[test]
    fn day_precise_endpoints_stay_exact() {
        let ledger = TemporalLedger {
            events: vec![
                mention(1_673_308_800_000, TimePrecision::Day), // 2023-01-10
                mention(1_673_913_600_000, TimePrecision::Day), // 2023-01-17
            ],
        };
        let rendered = ledger.render();
        assert!(rendered.contains("7 day(s)"), "{rendered}");
        // The header explains the convention, so only the interval lines
        // themselves are checked for hedging.
        let intervals = rendered.split("Intervals:").nth(1).unwrap();
        assert!(
            !intervals.contains("approximate"),
            "a day-precise pair must not be hedged: {intervals}"
        );
    }

    #[test]
    fn a_year_precise_endpoint_dominates_the_uncertainty() {
        assert_eq!(
            interval_uncertainty_days(TimePrecision::Year, TimePrecision::Month),
            365
        );
        assert_eq!(
            interval_uncertainty_days(TimePrecision::Day, TimePrecision::Month),
            31
        );
        assert_eq!(
            interval_uncertainty_days(TimePrecision::Day, TimePrecision::Instant),
            0
        );
    }

    #[test]
    fn resolves_a_bare_month_day_against_the_reference_year() {
        let mentions = scan_dated_mentions(
            "I attended the workshop on January 10th and it was useful",
            january_2023(),
        );
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].precision, TimePrecision::Day);
        assert!(
            format_date(mentions[0].date).starts_with("2023-01-10"),
            "got {}",
            format_date(mentions[0].date)
        );
        assert!(mentions[0].snippet.contains("workshop"));
    }

    #[test]
    fn resolves_iso_and_explicit_year_forms() {
        let iso = scan_dated_mentions("meeting on 2023-03-15 at noon", january_2023());
        assert_eq!(iso.len(), 1);
        assert!(format_date(iso[0].date).starts_with("2023-03-15"));

        let named = scan_dated_mentions("signed on March 15, 2024 finally", january_2023());
        assert_eq!(named.len(), 1);
        assert!(
            format_date(named[0].date).starts_with("2024-03-15"),
            "an explicit year must win over the reference year, got {}",
            format_date(named[0].date)
        );
    }

    #[test]
    fn relative_phrases_anchor_on_the_excerpt_not_on_now() {
        // Using a global "now" would misdate every historical excerpt.
        let mentions = scan_dated_mentions("I saw them yesterday at the park", january_2023());
        assert_eq!(mentions.len(), 1);
        assert!(
            format_date(mentions[0].date).starts_with("2023-01-12"),
            "got {}",
            format_date(mentions[0].date)
        );
    }

    #[test]
    fn text_without_dates_yields_nothing() {
        assert!(scan_dated_mentions("we talked about the project", january_2023()).is_empty());
        assert!(scan_dated_mentions("", january_2023()).is_empty());
    }

    #[test]
    fn ledger_sorts_and_computes_the_worked_example() {
        // The real LongMemEval case: workshop 10 Jan, meeting 17 Jan, gold "7 days".
        let ledger = build_ledger(&[
            (
                "I attended a workshop on Effective Communication on January 10th",
                january_2023(),
            ),
            (
                "I have my team meeting on January 17th to practice those skills",
                january_2023(),
            ),
        ]);
        assert_eq!(ledger.events.len(), 2);
        assert_eq!(
            ledger.span_days(),
            Some(7),
            "the interval the question asks for must be computed exactly"
        );

        let rendered = ledger.render();
        assert!(rendered.contains("2023-01-10"));
        assert!(rendered.contains("2023-01-17"));
        assert!(rendered.contains("7 day(s)"));
        assert!(
            rendered.contains("do not recompute"),
            "the reader must be told these are authoritative"
        );
    }

    #[test]
    fn ordering_is_answerable_from_the_rendered_order() {
        let ledger = build_ledger(&[
            ("the Data Analysis webinar was on March 3", january_2023()),
            (
                "the Time Management workshop was on March 21",
                january_2023(),
            ),
        ]);
        assert_eq!(ledger.events.len(), 2);
        assert!(
            ledger.events[0].snippet.contains("webinar"),
            "earliest event must sort first so 'which came first' is readable"
        );
    }

    #[test]
    fn a_single_date_is_not_worth_rendering() {
        let ledger = build_ledger(&[("just one date: January 10th", january_2023())]);
        assert!(!ledger.is_useful());
        assert!(
            ledger.render().is_empty(),
            "a one-event ledger supports no arithmetic and must not spend tokens"
        );
    }

    #[test]
    fn repeated_mentions_of_one_event_collapse() {
        // The worked example restates "January 10th" in five different turns.
        let entries = vec![
            ("I attended the workshop on January 10th", january_2023()),
            ("I attended the workshop on January 10th", january_2023()),
            ("the meeting is on January 17th", january_2023()),
        ];
        let ledger = build_ledger(&entries);
        assert_eq!(
            ledger.events.len(),
            2,
            "a restated date is one event, not two"
        );
    }

    #[test]
    fn two_distinct_events_on_one_date_are_both_kept() {
        let ledger = build_ledger(&[
            (
                "the dentist appointment was on January 10th",
                january_2023(),
            ),
            (
                "the car service happened on January 10th too",
                january_2023(),
            ),
        ]);
        assert_eq!(ledger.events.len(), 2);
        assert_eq!(ledger.span_days(), Some(0));
    }

    #[test]
    fn the_ledger_is_bounded() {
        let texts: Vec<String> = (1..=20)
            .map(|day| format!("event number {day} happened on March {day}"))
            .collect();
        let entries: Vec<(&str, Timestamp)> =
            texts.iter().map(|t| (t.as_str(), january_2023())).collect();
        let ledger = build_ledger(&entries);
        assert!(
            ledger.events.len() <= MAX_LEDGER_EVENTS,
            "ledger must stay within its token budget, got {}",
            ledger.events.len()
        );
        // Truncation keeps the head of the timeline.
        assert!(format_date(ledger.events[0].date).starts_with("2023-03-01"));
    }

    #[test]
    fn intervals_are_non_negative_regardless_of_input_order() {
        let ledger = build_ledger(&[
            ("later event on March 20", january_2023()),
            ("earlier event on March 1", january_2023()),
        ]);
        let rendered = ledger.render();
        assert!(
            !rendered.contains("-19 day") && !rendered.contains(": -"),
            "no negative interval: {rendered}"
        );
        assert!(rendered.contains("19 day(s)"));
        assert_eq!(ledger.span_days(), Some(19));
    }

    #[test]
    fn month_names_used_as_ordinary_words_are_not_dates() {
        // The bug this guards: probing real conversation data, "you may find
        // that you need to make adjustments" was read as May, resolved to year
        // 0023, and produced a 730,386-day interval inside a block the reader
        // is told is exact. Corroboration by a day or year is now required.
        for text in [
            "you may find that you need to make adjustments to optimize",
            "it may take some time to develop new habits",
            "Power BI may require a bit more technical knowledge",
            "we will march through the agenda quickly",
            "the august committee reviewed the proposal",
        ] {
            let mentions = scan_dated_mentions(text, january_2023());
            assert!(
                mentions.is_empty(),
                "{text:?} produced a spurious date: {mentions:?}"
            );
        }
    }

    #[test]
    fn a_corroborated_month_is_still_recognised() {
        // The fix must not throw out real dates with the false positives.
        for (text, expect) in [
            ("the appointment is on May 20th", "2023-05-20"),
            ("we met in March 2021 for the first time", "2021-03-01"),
            ("signed on August 3, 2024", "2024-08-03"),
        ] {
            let mentions = scan_dated_mentions(text, january_2023());
            assert_eq!(mentions.len(), 1, "{text:?} -> {mentions:?}");
            assert!(
                format_date(mentions[0].date).starts_with(expect),
                "{text:?} resolved to {}",
                format_date(mentions[0].date)
            );
        }
    }

    #[test]
    fn implausible_years_are_rejected() {
        // A date that parses but lands centuries away is a parser artefact,
        // and its interval would dwarf every real one in the ledger.
        assert!(is_plausible_year(Timestamp::from_millis(1_673_568_000_000)));
        // Year 9999 — the shape a parser artefact takes when it lands far from
        // the conversation. `Timestamp` is epoch-millis and unsigned, so a
        // year-23 artefact cannot be represented here; the far-future case is
        // the representable half of the same guard.
        let year_9999 = chrono::NaiveDate::from_ymd_opt(9999, 5, 20)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc()
            .timestamp_millis();
        assert!(!is_plausible_year(Timestamp::from_millis(year_9999 as u64)));
    }

    #[test]
    fn a_polluted_context_yields_no_ledger_rather_than_a_wrong_one() {
        // End-to-end: text full of modal "may" and no real dates must produce
        // nothing, not a confident-looking ledger of zero-day intervals.
        let ledger = build_ledger(&[
            (
                "you may find that this may help, and it may take time",
                january_2023(),
            ),
            (
                "Power BI may require more technical knowledge",
                january_2023(),
            ),
        ]);
        assert!(!ledger.is_useful());
        assert!(ledger.render().is_empty());
    }

    #[test]
    fn historical_references_are_excluded_from_the_ledger() {
        // Observed in real data: a travel recommendation mentioning "the attack
        // on December 7, 1941" is a correct date and not a personal event. Left
        // in, it paired against every real event and contributed five
        // 29,000-day intervals — a third of the block, all noise.
        let ledger = build_ledger(&[
            (
                "the memorial honors the lives lost during the attack on December 7, 1941",
                january_2023(),
            ),
            ("I pre-ordered the laptop on January 28th", january_2023()),
            ("it arrived on February 25th after a delay", january_2023()),
        ]);
        assert_eq!(ledger.events.len(), 2, "{:?}", ledger.events);
        let rendered = ledger.render();
        assert!(!rendered.contains("1941"), "{rendered}");
        assert!(rendered.contains("28 day(s)"), "{rendered}");
    }

    #[test]
    fn dates_within_a_lifetime_are_kept() {
        // The horizon must not throw out ordinary personal history.
        let ledger = build_ledger(&[
            ("we bought the house in March 2011", january_2023()),
            ("we sold it in March 2021", january_2023()),
        ]);
        assert_eq!(ledger.events.len(), 2);
        assert!(
            ledger.render().contains("3653 day(s)"),
            "{}",
            ledger.render()
        );
    }

    #[test]
    fn every_pair_is_listed_for_a_small_ledger() {
        // "How many days between X and Y" rarely names consecutive events, so
        // the model must not have to sum gaps.
        let ledger = build_ledger(&[
            ("first on March 1", january_2023()),
            ("second on March 5", january_2023()),
            ("third on March 20", january_2023()),
        ]);
        let rendered = ledger.render();
        assert!(rendered.contains("4 day(s)"), "1→5: {rendered}");
        assert!(rendered.contains("19 day(s)"), "1→20: {rendered}");
        assert!(rendered.contains("15 day(s)"), "5→20: {rendered}");
        assert!(
            !rendered.contains("earliest"),
            "no span summary across unrelated events: {rendered}"
        );
    }
}
