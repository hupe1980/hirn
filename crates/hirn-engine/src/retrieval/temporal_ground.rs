//! Event-relative temporal grounding — the two-hop temporal decomposition.
//!
//! Many temporal questions anchor on an *event*, not a calendar date: "what did I
//! do **before the trip**?", "**after I started the new job**, what changed?",
//! "what did the doctor say **when I visited**?". The calendar parser
//! ([`super::temporal_parse`]) correctly matches nothing here, so the temporal
//! ranking never engages — which is exactly why LongMemEval temporal-reasoning
//! stayed flat under calendar parsing alone.
//!
//! This module closes that gap deterministically:
//! 1. **Extract** the relation cue (`before`/`after`/`when`/…) and the event
//!    phrase that follows it ([`extract_event_anchor`]).
//! 2. **Resolve** the phrase to a timestamp via a first-hop recall of the anchor
//!    episode ([`ground_temporal_frame`]).
//! 3. **Frame** a soft temporal hint from the relation + anchor timestamp
//!    ([`derive_frame_from_anchor`]), which the second-hop recall uses to boost
//!    (never exclude) time-correct evidence via the Allen-interval θ term.
//!
//! No LLM: the extraction is rule-based token scanning; the resolution reuses the
//! existing embedding + recall primitives. The frame is always applied as a soft
//! ranking hint, never a hard filter (a wrong anchor must not exclude gold).

use hirn_core::types::{AgentId, Namespace};

use super::temporal_parse::{ParsedTimeFrame, parse_time_frame};
use crate::db::HirnDB;

/// How the question's evidence relates to the anchor event's time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalRelation {
    /// Evidence precedes the anchor ("before the trip").
    Before,
    /// Evidence follows the anchor ("after I moved", "since the promotion").
    After,
    /// Evidence is contemporaneous with the anchor ("when I met Sarah").
    Around,
}

/// A parsed event anchor: the relation and the free-text event phrase to resolve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventAnchor {
    pub relation: TemporalRelation,
    pub phrase: String,
}

/// Window (± ms) applied around an anchor for `Around` relations. Episodic
/// conversation evidence tends to cluster near the anchored event; three days on
/// each side captures the surrounding context without over-broadening.
pub const AROUND_WINDOW_MS: i64 = 3 * 24 * 3600 * 1000;

/// Relation cues, longest-first so multi-word cues win over their prefixes.
const CUES: &[(&str, TemporalRelation)] = &[
    ("prior to ", TemporalRelation::Before),
    ("before ", TemporalRelation::Before),
    ("after ", TemporalRelation::After),
    ("since ", TemporalRelation::After),
    ("following ", TemporalRelation::After),
    ("once ", TemporalRelation::After),
    ("when ", TemporalRelation::Around),
    ("while ", TemporalRelation::Around),
    ("during ", TemporalRelation::Around),
];

/// Words that make a phrase a *calendar* reference (handled by the calendar
/// parser) rather than an *event* reference — if the phrase starts with one, this
/// is not an event anchor.
fn phrase_starts_with_calendar_term(phrase: &str) -> bool {
    let first = phrase.split_whitespace().next().unwrap_or("");
    // A 4-digit year, or a calendar keyword.
    if first.len() == 4 && first.bytes().all(|b| b.is_ascii_digit()) {
        return true;
    }
    matches!(
        first,
        "yesterday"
            | "today"
            | "last"
            | "this"
            | "next"
            | "previous"
            | "past"
            | "january"
            | "february"
            | "march"
            | "april"
            | "may"
            | "june"
            | "july"
            | "august"
            | "september"
            | "october"
            | "november"
            | "december"
    )
}

/// Trim a candidate event phrase at the first clause boundary and strip a few
/// leading filler tokens, returning `None` if nothing contentful remains.
fn clean_phrase(raw: &str) -> Option<String> {
    // Stop at a clause/sentence boundary — the event phrase is the clause right
    // after the cue.
    let cut = raw
        .find([',', '?', '.', ';', ':', '!'])
        .unwrap_or(raw.len());
    let phrase = raw[..cut].trim();
    // Drop a trailing interrogative tail that some phrasings append without
    // punctuation ("... the trip did i ...").
    let phrase = phrase.trim();
    if phrase.is_empty() {
        return None;
    }
    // Require at least one token that isn't a bare stopword/pronoun, so "after
    // that" / "before it" don't produce a useless anchor.
    let contentful = phrase.split_whitespace().any(|w| {
        !matches!(
            w,
            "the"
                | "a"
                | "an"
                | "i"
                | "it"
                | "that"
                | "this"
                | "my"
                | "we"
                | "you"
                | "he"
                | "she"
                | "they"
                | "was"
                | "did"
                | "do"
                | "had"
                | "have"
        ) && w.chars().any(char::is_alphabetic)
    });
    if !contentful {
        return None;
    }
    Some(phrase.to_string())
}

/// Extract an event anchor from a free-text question, or `None` when the question
/// carries no event-relative temporal cue (or the cue is calendar-based).
#[must_use]
pub fn extract_event_anchor(query: &str) -> Option<EventAnchor> {
    let lower = query.to_lowercase();
    let bytes = lower.as_bytes();

    // Choose the earliest word-boundary cue occurrence.
    let mut best: Option<(usize, usize, TemporalRelation)> = None;
    for (cue, rel) in CUES {
        let mut from = 0usize;
        while let Some(rel_pos) = lower[from..].find(cue) {
            let pos = from + rel_pos;
            let boundary = pos == 0 || !bytes[pos - 1].is_ascii_alphanumeric();
            if boundary {
                let start = pos + cue.len();
                if best.is_none_or(|(bp, _, _)| pos < bp) {
                    best = Some((pos, start, *rel));
                }
                break;
            }
            from = pos + 1;
        }
    }

    let (_, phrase_start, relation) = best?;
    let phrase = clean_phrase(&lower[phrase_start..])?;
    if phrase_starts_with_calendar_term(&phrase) {
        return None; // calendar parser's job, not event grounding
    }
    Some(EventAnchor { relation, phrase })
}

/// Build a soft time frame from a resolved anchor timestamp and the relation.
#[must_use]
pub fn derive_frame_from_anchor(relation: TemporalRelation, anchor_ms: i64) -> ParsedTimeFrame {
    match relation {
        TemporalRelation::Before => ParsedTimeFrame {
            after_ms: None,
            before_ms: Some(anchor_ms),
        },
        TemporalRelation::After => ParsedTimeFrame {
            after_ms: Some(anchor_ms),
            before_ms: None,
        },
        TemporalRelation::Around => ParsedTimeFrame {
            after_ms: Some(anchor_ms - AROUND_WINDOW_MS),
            before_ms: Some(anchor_ms + AROUND_WINDOW_MS),
        },
    }
}

/// Resolve a query's temporal frame, trying the calendar parser first and then
/// event-relative grounding (a first-hop recall of the anchor episode).
///
/// Always returns a **soft** frame (to be applied as a ranking hint, never a hard
/// filter). Returns an empty frame when the query has no resolvable time context.
/// `scope` bounds the anchor recall to the caller's accessible namespaces.
pub async fn ground_temporal_frame(
    db: &HirnDB,
    agent_id: Option<&AgentId>,
    query: &str,
    scope: Option<&[Namespace]>,
    now_ms: i64,
) -> ParsedTimeFrame {
    // Explicit calendar references win — they need no anchor recall.
    let calendar = parse_time_frame(query, now_ms);
    if calendar.matched() {
        return calendar;
    }

    let Some(anchor) = extract_event_anchor(query) else {
        return ParsedTimeFrame::default();
    };

    // First hop: resolve the event phrase to an anchor episode timestamp.
    let Ok(embedding) = db.embed_text(&anchor.phrase).await else {
        return ParsedTimeFrame::default();
    };
    let mut builder = db.recall(embedding).limit(3).episodic_only();
    if let Some(agent) = agent_id {
        builder = builder.agent_id(agent.as_str());
    }
    builder = match scope {
        Some(namespaces) => builder.allowed_namespaces(namespaces.to_vec()),
        None => builder.unrestricted(),
    };

    let Ok(results) = Box::pin(builder.execute()).await else {
        return ParsedTimeFrame::default();
    };
    let Some(anchor_ms) = results.iter().find_map(|r| match &r.record {
        hirn_core::record::MemoryRecord::Episodic(e) => Some(e.timestamp.timestamp_ms()),
        _ => None,
    }) else {
        return ParsedTimeFrame::default();
    };

    derive_frame_from_anchor(anchor.relation, anchor_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_event() {
        let a = extract_event_anchor("what did I do before the trip to Japan?").unwrap();
        assert_eq!(a.relation, TemporalRelation::Before);
        assert_eq!(a.phrase, "the trip to japan");
    }

    #[test]
    fn after_event_clause() {
        let a =
            extract_event_anchor("after I started my new job, how did my routine change?").unwrap();
        assert_eq!(a.relation, TemporalRelation::After);
        assert_eq!(a.phrase, "i started my new job");
    }

    #[test]
    fn when_event_is_around() {
        let a = extract_event_anchor("what did the doctor say when I visited the clinic").unwrap();
        assert_eq!(a.relation, TemporalRelation::Around);
        assert_eq!(a.phrase, "i visited the clinic");
    }

    #[test]
    fn since_event() {
        let a = extract_event_anchor("which tools have I used since the migration").unwrap();
        assert_eq!(a.relation, TemporalRelation::After);
        assert_eq!(a.phrase, "the migration");
    }

    #[test]
    fn calendar_reference_is_not_an_event_anchor() {
        // "before 2023" is a calendar reference — defer to the calendar parser.
        assert!(extract_event_anchor("anything before 2023?").is_none());
        assert!(extract_event_anchor("meetings since last month").is_none());
    }

    #[test]
    fn contentless_anchor_rejected() {
        assert!(extract_event_anchor("what happened after that?").is_none());
        assert!(extract_event_anchor("what did I say before it").is_none());
    }

    #[test]
    fn non_temporal_query_has_no_anchor() {
        assert!(extract_event_anchor("what is my favorite color?").is_none());
    }

    #[test]
    fn derive_frames() {
        let anchor = 1_000_000_000_000;
        assert_eq!(
            derive_frame_from_anchor(TemporalRelation::Before, anchor),
            ParsedTimeFrame {
                after_ms: None,
                before_ms: Some(anchor)
            }
        );
        assert_eq!(
            derive_frame_from_anchor(TemporalRelation::After, anchor),
            ParsedTimeFrame {
                after_ms: Some(anchor),
                before_ms: None
            }
        );
        let around = derive_frame_from_anchor(TemporalRelation::Around, anchor);
        assert_eq!(around.after_ms, Some(anchor - AROUND_WINDOW_MS));
        assert_eq!(around.before_ms, Some(anchor + AROUND_WINDOW_MS));
    }
}
