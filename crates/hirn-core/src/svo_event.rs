//! Subject–Verb–Object event record for temporal knowledge representation.
//!
//! SVO events extract structured triples from episodic memories, enabling
//! temporal queries like "what did agent X do between T1 and T2?" and
//! graph-based reasoning over causal chains.

use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use serde::{Deserialize, Serialize};

use crate::id::MemoryId;
use crate::timestamp::Timestamp;

/// A structured Subject–Verb–Object event extracted from one or more memories.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SvoEvent {
    /// Unique identifier for this event.
    pub id: MemoryId,
    /// The subject (actor/entity) of the event.
    pub subject: String,
    /// The verb (action/relation) of the event.
    pub verb: String,
    /// The object (target/entity) of the event.
    pub object: String,
    /// Raw extracted temporal start text when available.
    pub time_start_text: Option<String>,
    /// Raw extracted temporal end text when available.
    pub time_end_text: Option<String>,
    /// Normalized temporal start when the extracted text could be parsed.
    pub time_start: Option<Timestamp>,
    /// Normalized temporal end when the extracted text could be parsed.
    pub time_end: Option<Timestamp>,
    /// Extraction confidence score.
    pub confidence: f32,
    /// IDs of the source memories that produced this event.
    pub source_ids: Vec<MemoryId>,
}

impl SvoEvent {
    /// Create a new SVO event with auto-generated ID.
    #[must_use]
    pub fn new(
        subject: impl Into<String>,
        verb: impl Into<String>,
        object: impl Into<String>,
        time_start: Timestamp,
        time_end: Timestamp,
    ) -> Self {
        Self {
            id: MemoryId::new(),
            subject: subject.into(),
            verb: verb.into(),
            object: object.into(),
            time_start_text: None,
            time_end_text: None,
            time_start: Some(time_start),
            time_end: Some(time_end),
            confidence: 1.0,
            source_ids: Vec::new(),
        }
    }

    /// Create a new SVO event without normalized temporal bounds.
    #[must_use]
    pub fn new_without_time(
        subject: impl Into<String>,
        verb: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            id: MemoryId::new(),
            subject: subject.into(),
            verb: verb.into(),
            object: object.into(),
            time_start_text: None,
            time_end_text: None,
            time_start: None,
            time_end: None,
            confidence: 1.0,
            source_ids: Vec::new(),
        }
    }

    /// Create an SVO event from extraction output, preserving raw time text
    /// and normalizing timestamps when the text can be parsed.
    #[must_use]
    pub fn from_extraction(
        subject: impl Into<String>,
        verb: impl Into<String>,
        object: impl Into<String>,
        time_start_text: Option<String>,
        time_end_text: Option<String>,
        confidence: f32,
        source_ids: Vec<MemoryId>,
    ) -> Self {
        Self::new_without_time(subject, verb, object)
            .with_time_text(time_start_text, time_end_text)
            .with_confidence(confidence)
            .with_source_ids(source_ids)
    }

    /// Add source memory IDs to this event.
    #[must_use]
    pub fn with_source_ids(mut self, ids: Vec<MemoryId>) -> Self {
        self.source_ids = ids;
        self
    }

    /// Store raw temporal extraction text and normalize timestamps when possible.
    #[must_use]
    pub fn with_time_text(
        mut self,
        time_start_text: Option<String>,
        time_end_text: Option<String>,
    ) -> Self {
        let reference = Timestamp::now();
        if self.time_start.is_none() {
            self.time_start = time_start_text
                .as_deref()
                .and_then(|text| parse_temporal_text(text, reference));
        }
        if self.time_end.is_none() {
            self.time_end = time_end_text
                .as_deref()
                .and_then(|text| parse_temporal_text(text, reference))
                .or(self.time_start);
        }
        self.time_start_text = time_start_text;
        self.time_end_text = time_end_text;
        self
    }

    /// Set extraction confidence.
    #[must_use]
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Return the first source memory when one exists.
    #[must_use]
    pub fn primary_source_id(&self) -> Option<MemoryId> {
        self.source_ids.first().copied()
    }
}

/// Normalize extracted temporal text into a UTC timestamp when possible.
#[must_use]
pub fn parse_temporal_text(text: &str, reference: Timestamp) -> Option<Timestamp> {
    let trimmed = text
        .trim()
        .trim_matches(|c: char| matches!(c, '.' | ',' | ';' | '!' | '?'));
    if trimmed.is_empty() {
        return None;
    }

    if let Some(ts) = parse_iso_date(trimmed) {
        return Some(ts);
    }

    let normalized = normalize_temporal_text(trimmed);
    if let Some(ts) = parse_named_date(&normalized, reference) {
        return Some(ts);
    }

    parse_relative_date(&normalized.to_ascii_lowercase(), reference)
}

fn parse_iso_date(text: &str) -> Option<Timestamp> {
    let date = NaiveDate::parse_from_str(text, "%Y-%m-%d").ok()?;
    date_at_start_of_day(date)
}

fn parse_named_date(text: &str, reference: Timestamp) -> Option<Timestamp> {
    if let Ok(date) = NaiveDate::parse_from_str(text, "%B %d, %Y") {
        return date_at_start_of_day(date);
    }
    if let Ok(date) = NaiveDate::parse_from_str(text, "%B %d %Y") {
        return date_at_start_of_day(date);
    }

    let reference_year = reference.as_datetime().year();
    if let Ok(partial) = NaiveDate::parse_from_str(&format!("{text} {reference_year}"), "%B %d %Y")
    {
        return date_at_start_of_day(partial);
    }

    if let Ok(month_start) = NaiveDate::parse_from_str(&format!("{text} 01"), "%B %Y %d") {
        return date_at_start_of_day(month_start);
    }

    None
}

fn parse_relative_date(text: &str, reference: Timestamp) -> Option<Timestamp> {
    let reference_date = reference.as_datetime().date_naive();
    match text {
        "today" | "this morning" => date_at_start_of_day(reference_date),
        "yesterday" => date_at_start_of_day(reference_date - Duration::days(1)),
        "last week" => date_at_start_of_day(reference_date - Duration::days(7)),
        "last month" => date_at_start_of_day(reference_date - Duration::days(30)),
        _ => None,
    }
}

fn date_at_start_of_day(date: NaiveDate) -> Option<Timestamp> {
    let dt = date.and_hms_opt(0, 0, 0)?;
    Some(Timestamp::from_datetime(
        DateTime::from_naive_utc_and_offset(dt, Utc),
    ))
}

fn normalize_temporal_text(text: &str) -> String {
    text.split_whitespace()
        .map(normalize_temporal_token)
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_temporal_token(token: &str) -> String {
    let trimmed = token.trim_matches(|c: char| matches!(c, '.' | ',' | ';' | '!' | '?'));
    let lower = trimmed.to_ascii_lowercase();
    for suffix in ["st", "nd", "rd", "th"] {
        if lower.ends_with(suffix) {
            let number = &trimmed[..trimmed.len().saturating_sub(suffix.len())];
            if !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()) {
                return number.to_string();
            }
        }
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_iso_temporal_text() {
        let ts = parse_temporal_text("2026-03-15", Timestamp::now()).unwrap();
        assert_eq!(ts.to_string(), "2026-03-15T00:00:00+00:00");
    }

    #[test]
    fn parses_month_day_with_ordinal_suffix() {
        let reference = Timestamp::from_datetime(DateTime::from_naive_utc_and_offset(
            NaiveDate::from_ymd_opt(2026, 4, 17)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap(),
            Utc,
        ));
        let ts = parse_temporal_text("March 15th", reference).unwrap();
        assert_eq!(ts.to_string(), "2026-03-15T00:00:00+00:00");
    }
}
