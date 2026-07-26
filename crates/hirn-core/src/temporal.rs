//! Neuro-symbolic temporal reasoning primitives.
//!
//! A deterministic temporal calculus over [`Timestamp`]s: time intervals,
//! Allen's 13 interval relations, duration/ordering/precedence, and point-in-time
//! ("as-of") containment. This is the *symbolic* half of a TReMu-style
//! (arXiv:2502.01630) temporal reasoning pipeline — the layer that answers
//! "which happened first", "how long between", "what was true as of T", and "how
//! do these two events relate" **exactly**, so an LLM never has to do date
//! arithmetic or event ordering (its documented failure mode).
//!
//! Unlike TReMu — which has the LLM emit Python and executes it — hirn keeps the
//! symbolic layer in native Rust: fully deterministic, unit-testable without any
//! model, and with **zero code-execution injection surface**. An LLM (or the
//! HirnQL grammar) only needs to translate a question into a typed query against
//! these primitives.
//!
//! All comparisons are millisecond-resolution over [`Timestamp::timestamp_ms`].
//! An interval with `end = None` is *open-ended* (ongoing / still valid), treated
//! as extending to `+∞` for every relation.

use crate::timestamp::Timestamp;

/// Sentinel for an open-ended interval end (`+∞`) in millisecond comparisons.
const OPEN_END_MS: i64 = i64::MAX;

/// A half-open time interval `[start, end)` over event (valid) time.
///
/// `end = None` means the interval is still open (the fact is currently valid /
/// the event is ongoing). A *point* interval (`start == end`) represents an
/// instantaneous event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeInterval {
    start: Timestamp,
    end: Option<Timestamp>,
}

impl TimeInterval {
    /// Create an interval `[start, end)`.
    ///
    /// If `end` is `Some(e)` with `e < start`, the interval is normalized to an
    /// instantaneous point at `start` (a reversed interval is a caller bug; we
    /// degrade to a point rather than produce nonsense relations).
    #[must_use]
    pub fn new(start: Timestamp, end: Option<Timestamp>) -> Self {
        let end = match end {
            Some(e) if e < start => Some(start),
            other => other,
        };
        Self { start, end }
    }

    /// Create an instantaneous interval at a single instant.
    #[must_use]
    pub fn point(at: Timestamp) -> Self {
        Self {
            start: at,
            end: Some(at),
        }
    }

    /// Create an open-ended interval starting at `start` (still valid / ongoing).
    #[must_use]
    pub fn since(start: Timestamp) -> Self {
        Self { start, end: None }
    }

    /// The interval start (inclusive).
    #[must_use]
    pub fn start(&self) -> Timestamp {
        self.start
    }

    /// The interval end (exclusive), or `None` if open-ended.
    #[must_use]
    pub fn end(&self) -> Option<Timestamp> {
        self.end
    }

    /// Whether the interval is open-ended (ongoing / still valid).
    #[must_use]
    pub fn is_ongoing(&self) -> bool {
        self.end.is_none()
    }

    /// Whether the interval is instantaneous (`start == end`).
    #[must_use]
    pub fn is_point(&self) -> bool {
        self.end == Some(self.start)
    }

    #[inline]
    fn start_ms(&self) -> i64 {
        self.start.timestamp_ms()
    }

    #[inline]
    fn end_ms(&self) -> i64 {
        self.end.map_or(OPEN_END_MS, |e| e.timestamp_ms())
    }

    /// Duration in milliseconds, or `None` if the interval is open-ended.
    #[must_use]
    pub fn duration_ms(&self) -> Option<i64> {
        self.end
            .map(|e| e.timestamp_ms() - self.start.timestamp_ms())
    }

    /// Whether `instant` falls within `[start, end)` — the point-in-time
    /// ("as-of") containment test. An open-ended interval contains every instant
    /// at or after its start.
    #[must_use]
    pub fn contains(&self, instant: Timestamp) -> bool {
        let t = instant.timestamp_ms();
        t >= self.start_ms() && t < self.end_ms()
    }

    /// Classify how `self` relates to `other` under Allen's interval algebra.
    #[must_use]
    pub fn relation_to(&self, other: &TimeInterval) -> AllenRelation {
        let (a1, a2) = (self.start_ms(), self.end_ms());
        let (b1, b2) = (other.start_ms(), other.end_ms());

        if a1 == b1 && a2 == b2 {
            AllenRelation::Equals
        } else if a2 < b1 {
            AllenRelation::Before
        } else if a1 > b2 {
            AllenRelation::After
        } else if a2 == b1 {
            AllenRelation::Meets
        } else if a1 == b2 {
            AllenRelation::MetBy
        } else if a1 == b1 {
            // a2 != b2 here (Equals handled above).
            if a2 < b2 {
                AllenRelation::Starts
            } else {
                AllenRelation::StartedBy
            }
        } else if a2 == b2 {
            // a1 != b1 here.
            if a1 > b1 {
                AllenRelation::Finishes
            } else {
                AllenRelation::FinishedBy
            }
        } else if a1 < b1 && a2 > b2 {
            AllenRelation::Contains
        } else if a1 > b1 && a2 < b2 {
            AllenRelation::During
        } else if a1 < b1 {
            // Interiors overlap and a1 < b1 with a2 < b2.
            AllenRelation::Overlaps
        } else {
            // a1 > b1 && a2 > b2.
            AllenRelation::OverlappedBy
        }
    }

    /// Whether `self` wholly precedes `other` (Allen `Before` or `Meets`).
    #[must_use]
    pub fn precedes(&self, other: &TimeInterval) -> bool {
        matches!(
            self.relation_to(other),
            AllenRelation::Before | AllenRelation::Meets
        )
    }

    /// Whether `self` and `other` share any interior time (they are neither
    /// strictly before nor strictly after, and do not merely touch at an
    /// endpoint).
    #[must_use]
    pub fn overlaps_with(&self, other: &TimeInterval) -> bool {
        self.start_ms() < other.end_ms() && other.start_ms() < self.end_ms()
    }

    /// Gap in milliseconds from the end of `self` to the start of `other` when
    /// `self` precedes `other` (0 if they meet or overlap). `None` if `self` is
    /// open-ended (no finite end to measure from).
    #[must_use]
    pub fn gap_before_ms(&self, other: &TimeInterval) -> Option<i64> {
        let e = self.end?.timestamp_ms();
        Some((other.start_ms() - e).max(0))
    }
}

/// Allen's 13 mutually-exclusive, jointly-exhaustive interval relations.
///
/// Read `a.relation_to(b) == X` as "*a* is *X* *b*" — e.g. `Before` means *a*
/// ends before *b* begins; `During` means *a* is strictly contained in *b*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllenRelation {
    /// `a` ends before `b` starts (with a gap).
    Before,
    /// `a` starts after `b` ends (with a gap).
    After,
    /// `a` ends exactly when `b` starts.
    Meets,
    /// `a` starts exactly when `b` ends.
    MetBy,
    /// `a` starts before `b`, they overlap, `a` ends before `b`.
    Overlaps,
    /// Inverse of [`Overlaps`](AllenRelation::Overlaps).
    OverlappedBy,
    /// `a` and `b` start together; `a` ends first.
    Starts,
    /// `a` and `b` start together; `b` ends first.
    StartedBy,
    /// `a` is strictly inside `b`.
    During,
    /// `b` is strictly inside `a`.
    Contains,
    /// `a` and `b` end together; `a` starts later.
    Finishes,
    /// `a` and `b` end together; `b` starts later.
    FinishedBy,
    /// `a` and `b` are identical.
    Equals,
}

impl AllenRelation {
    /// The converse relation: if `a.relation_to(b) == r` then
    /// `b.relation_to(a) == r.inverse()`.
    #[must_use]
    pub fn inverse(self) -> Self {
        match self {
            Self::Before => Self::After,
            Self::After => Self::Before,
            Self::Meets => Self::MetBy,
            Self::MetBy => Self::Meets,
            Self::Overlaps => Self::OverlappedBy,
            Self::OverlappedBy => Self::Overlaps,
            Self::Starts => Self::StartedBy,
            Self::StartedBy => Self::Starts,
            Self::During => Self::Contains,
            Self::Contains => Self::During,
            Self::Finishes => Self::FinishedBy,
            Self::FinishedBy => Self::Finishes,
            Self::Equals => Self::Equals,
        }
    }

    /// A stable machine-readable label (used in query results / explanations).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::After => "after",
            Self::Meets => "meets",
            Self::MetBy => "met_by",
            Self::Overlaps => "overlaps",
            Self::OverlappedBy => "overlapped_by",
            Self::Starts => "starts",
            Self::StartedBy => "started_by",
            Self::During => "during",
            Self::Contains => "contains",
            Self::Finishes => "finishes",
            Self::FinishedBy => "finished_by",
            Self::Equals => "equals",
        }
    }
}

/// Human-readable duration between two millisecond instants (e.g. "3 days",
/// "2 hours", "45 minutes"). Coarsens to the largest whole unit; negative or
/// zero spans render as "0 seconds". Intended for temporal-answer rendering, not
/// precise formatting.
#[must_use]
pub fn humanize_duration_ms(ms: i64) -> String {
    if ms <= 0 {
        return "0 seconds".to_string();
    }
    const SEC: i64 = 1000;
    const MIN: i64 = 60 * SEC;
    const HOUR: i64 = 60 * MIN;
    const DAY: i64 = 24 * HOUR;
    const WEEK: i64 = 7 * DAY;
    const YEAR: i64 = 365 * DAY;

    let (value, unit) = if ms >= YEAR {
        (ms / YEAR, "year")
    } else if ms >= WEEK {
        (ms / WEEK, "week")
    } else if ms >= DAY {
        (ms / DAY, "day")
    } else if ms >= HOUR {
        (ms / HOUR, "hour")
    } else if ms >= MIN {
        (ms / MIN, "minute")
    } else {
        (ms / SEC, "second")
    };
    if value == 1 {
        format!("1 {unit}")
    } else {
        format!("{value} {unit}s")
    }
}

/// A dated event to place on a [`Timeline`].
///
/// An episodic event *occurred at* an instant (`occurred_at`) and remains valid
/// until `valid_until` (or indefinitely, when `None`). Ordering and inter-event
/// relations use the **occurrence instant** — a timeline answers "what happened,
/// in what order" — while the optional `valid_until` drives the bi-temporal
/// "as of" snapshot (an event is *valid as of* `t` when
/// `occurred_at <= t < valid_until`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineEvent {
    /// Stable identifier (e.g. a memory id).
    pub id: String,
    /// Human-readable label (content/summary).
    pub label: String,
    /// When the event occurred (event/valid-time start).
    pub occurred_at: Timestamp,
    /// When the event ceased to be valid, or `None` if still valid.
    pub valid_until: Option<Timestamp>,
}

impl TimelineEvent {
    /// Convenience constructor.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        occurred_at: Timestamp,
        valid_until: Option<Timestamp>,
    ) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            occurred_at,
            valid_until,
        }
    }

    /// The event's effective interval for Allen relations: an instantaneous
    /// point when there is no `valid_until`, otherwise the validity span.
    #[must_use]
    fn effective_interval(&self) -> TimeInterval {
        match self.valid_until {
            Some(vu) => TimeInterval::new(self.occurred_at, Some(vu)),
            None => TimeInterval::point(self.occurred_at),
        }
    }

    /// Whether the event was valid *as of* `t`
    /// (`occurred_at <= t < valid_until`, open-ended when `valid_until` is
    /// `None`).
    #[must_use]
    fn is_valid_at(&self, t: Timestamp) -> bool {
        self.occurred_at <= t && self.valid_until.is_none_or(|vu| vu > t)
    }
}

/// One chronologically-placed entry, annotated with how it relates to the entry
/// immediately before it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimelineEntry {
    /// Stable identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// When the event occurred.
    pub occurred_at: Timestamp,
    /// When the event ceased to be valid, or `None` if still valid.
    pub valid_until: Option<Timestamp>,
    /// Allen relation of this entry's effective interval to the previous entry's
    /// (`None` for the first entry). For point occurrences this is
    /// `Before`/`After`/`Equals`; events with a `valid_until` span can also
    /// yield `During`/`Overlaps`/etc.
    pub relation_to_prev: Option<AllenRelation>,
    /// Milliseconds between the previous entry's occurrence and this one's
    /// (`None` for the first entry). Always `>= 0` (entries are time-sorted).
    pub gap_to_prev_ms: Option<i64>,
}

/// A resolved, chronologically-ordered timeline: the deterministic answer to
/// "what happened, in what order, how far apart, and over what total span".
///
/// This is the payoff of the symbolic temporal layer — an LLM can read an
/// ordered, relation-annotated timeline instead of trying (and failing) to
/// order and date events itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timeline {
    /// Entries ordered by occurrence ascending; occurrence ties break by
    /// `valid_until` ascending (open-ended last), then input order (stable sort).
    pub entries: Vec<TimelineEntry>,
    /// Span in milliseconds from the first occurrence to the last
    /// (`None` when empty).
    pub span_ms: Option<i64>,
    /// Human-readable total span (e.g. "3 days"); "0 seconds" when empty or a
    /// single event.
    pub span_human: String,
}

impl Timeline {
    /// Build a timeline from `events`, ordered by occurrence.
    ///
    /// When `as_of` is `Some(t)`, only events that were *valid as of* `t`
    /// (`occurred_at <= t < valid_until`) are included — the point-in-time
    /// bi-temporal snapshot. Ordering is a stable sort by occurrence, breaking
    /// occurrence ties by `valid_until` ascending (open-ended sorts last).
    #[must_use]
    pub fn build(events: Vec<TimelineEvent>, as_of: Option<Timestamp>) -> Self {
        let mut events: Vec<TimelineEvent> = match as_of {
            Some(t) => events.into_iter().filter(|e| e.is_valid_at(t)).collect(),
            None => events,
        };

        events.sort_by(|a, b| {
            a.occurred_at.cmp(&b.occurred_at).then_with(|| {
                a.valid_until
                    .map_or(OPEN_END_MS, |v| v.timestamp_ms())
                    .cmp(&b.valid_until.map_or(OPEN_END_MS, |v| v.timestamp_ms()))
            })
        });

        let mut entries: Vec<TimelineEntry> = Vec::with_capacity(events.len());
        let mut prev: Option<TimelineEvent> = None;
        for e in events {
            let relation_to_prev = prev
                .as_ref()
                .map(|p| e.effective_interval().relation_to(&p.effective_interval()));
            let gap_to_prev_ms = prev
                .as_ref()
                .map(|p| (e.occurred_at.timestamp_ms() - p.occurred_at.timestamp_ms()).max(0));
            entries.push(TimelineEntry {
                id: e.id.clone(),
                label: e.label.clone(),
                occurred_at: e.occurred_at,
                valid_until: e.valid_until,
                relation_to_prev,
                gap_to_prev_ms,
            });
            prev = Some(e);
        }

        let (span_ms, span_human) = match (entries.first(), entries.last()) {
            (Some(first), Some(last)) => {
                let span =
                    (last.occurred_at.timestamp_ms() - first.occurred_at.timestamp_ms()).max(0);
                (Some(span), humanize_duration_ms(span))
            }
            _ => (None, "0 seconds".to_string()),
        };

        Self {
            entries,
            span_ms,
            span_human,
        }
    }

    /// The earliest entry, if any.
    #[must_use]
    pub fn first(&self) -> Option<&TimelineEntry> {
        self.entries.first()
    }

    /// The latest-occurring entry, if any.
    #[must_use]
    pub fn last(&self) -> Option<&TimelineEntry> {
        self.entries.last()
    }

    fn find(&self, id: &str) -> Option<&TimelineEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// The Allen relation of event `a` to event `b` (`a` rel `b`), or `None` if
    /// either id is absent from the timeline.
    #[must_use]
    pub fn relation_between(&self, a: &str, b: &str) -> Option<AllenRelation> {
        let ea = self.find(a)?;
        let eb = self.find(b)?;
        let ia = TimelineEvent::new("", "", ea.occurred_at, ea.valid_until).effective_interval();
        let ib = TimelineEvent::new("", "", eb.occurred_at, eb.valid_until).effective_interval();
        Some(ia.relation_to(&ib))
    }

    /// Absolute milliseconds between the occurrences of events `a` and `b`, or
    /// `None` if either id is absent.
    #[must_use]
    pub fn duration_between(&self, a: &str, b: &str) -> Option<i64> {
        let ea = self.find(a)?;
        let eb = self.find(b)?;
        Some((ea.occurred_at.timestamp_ms() - eb.occurred_at.timestamp_ms()).abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(ms: i64) -> Timestamp {
        Timestamp::from_millis(ms as u64)
    }

    fn ivl(start: i64, end: i64) -> TimeInterval {
        TimeInterval::new(ts(start), Some(ts(end)))
    }

    #[test]
    fn point_and_ongoing_constructors() {
        let p = TimeInterval::point(ts(100));
        assert!(p.is_point());
        assert_eq!(p.duration_ms(), Some(0));

        let o = TimeInterval::since(ts(100));
        assert!(o.is_ongoing());
        assert_eq!(o.duration_ms(), None);
    }

    #[test]
    fn reversed_interval_degrades_to_point() {
        let i = TimeInterval::new(ts(200), Some(ts(100)));
        assert!(i.is_point());
        assert_eq!(i.start(), ts(200));
    }

    #[test]
    fn allen_before_and_after() {
        let a = ivl(0, 10);
        let b = ivl(20, 30);
        assert_eq!(a.relation_to(&b), AllenRelation::Before);
        assert_eq!(b.relation_to(&a), AllenRelation::After);
        assert!(a.precedes(&b));
        assert!(!b.precedes(&a));
    }

    #[test]
    fn allen_meets_and_met_by() {
        let a = ivl(0, 10);
        let b = ivl(10, 20);
        assert_eq!(a.relation_to(&b), AllenRelation::Meets);
        assert_eq!(b.relation_to(&a), AllenRelation::MetBy);
        assert!(a.precedes(&b));
    }

    #[test]
    fn allen_overlaps() {
        let a = ivl(0, 15);
        let b = ivl(10, 25);
        assert_eq!(a.relation_to(&b), AllenRelation::Overlaps);
        assert_eq!(b.relation_to(&a), AllenRelation::OverlappedBy);
        assert!(a.overlaps_with(&b));
    }

    #[test]
    fn allen_during_and_contains() {
        let a = ivl(10, 20);
        let b = ivl(0, 30);
        assert_eq!(a.relation_to(&b), AllenRelation::During);
        assert_eq!(b.relation_to(&a), AllenRelation::Contains);
    }

    #[test]
    fn allen_starts_and_finishes() {
        let a = ivl(0, 10);
        let b = ivl(0, 20);
        assert_eq!(a.relation_to(&b), AllenRelation::Starts);
        assert_eq!(b.relation_to(&a), AllenRelation::StartedBy);

        let c = ivl(10, 30);
        let d = ivl(0, 30);
        assert_eq!(c.relation_to(&d), AllenRelation::Finishes);
        assert_eq!(d.relation_to(&c), AllenRelation::FinishedBy);
    }

    #[test]
    fn allen_equals() {
        let a = ivl(5, 15);
        let b = ivl(5, 15);
        assert_eq!(a.relation_to(&b), AllenRelation::Equals);
        assert_eq!(AllenRelation::Equals.inverse(), AllenRelation::Equals);
    }

    #[test]
    fn every_relation_inverse_is_consistent() {
        // For a spread of interval pairs, b.relation_to(a) == a.relation_to(b).inverse().
        let samples = [
            (ivl(0, 10), ivl(20, 30)),
            (ivl(0, 10), ivl(10, 20)),
            (ivl(0, 15), ivl(10, 25)),
            (ivl(10, 20), ivl(0, 30)),
            (ivl(0, 10), ivl(0, 20)),
            (ivl(10, 30), ivl(0, 30)),
            (ivl(5, 15), ivl(5, 15)),
        ];
        for (a, b) in samples {
            assert_eq!(
                b.relation_to(&a),
                a.relation_to(&b).inverse(),
                "inverse mismatch for {a:?} vs {b:?}"
            );
        }
    }

    #[test]
    fn open_ended_interval_relations() {
        let ongoing = TimeInterval::since(ts(100));
        let past = ivl(0, 50);
        assert_eq!(past.relation_to(&ongoing), AllenRelation::Before);
        // Two open-ended intervals starting together are equal (+∞ ends match).
        let other_ongoing = TimeInterval::since(ts(100));
        assert_eq!(ongoing.relation_to(&other_ongoing), AllenRelation::Equals);
    }

    #[test]
    fn as_of_containment() {
        let i = ivl(100, 200);
        assert!(!i.contains(ts(99)));
        assert!(i.contains(ts(100)));
        assert!(i.contains(ts(150)));
        assert!(!i.contains(ts(200)), "end is exclusive");

        let ongoing = TimeInterval::since(ts(100));
        assert!(ongoing.contains(ts(10_000_000)));
        assert!(!ongoing.contains(ts(99)));
    }

    #[test]
    fn gap_before() {
        let a = ivl(0, 10);
        let b = ivl(25, 30);
        assert_eq!(a.gap_before_ms(&b), Some(15));
        // Overlap / meet → clamped to 0.
        assert_eq!(a.gap_before_ms(&ivl(5, 8)), Some(0));
        // Open-ended has no finite end.
        assert_eq!(TimeInterval::since(ts(0)).gap_before_ms(&b), None);
    }

    #[test]
    fn humanize_duration_units() {
        assert_eq!(humanize_duration_ms(0), "0 seconds");
        assert_eq!(humanize_duration_ms(-5), "0 seconds");
        assert_eq!(humanize_duration_ms(1000), "1 second");
        assert_eq!(humanize_duration_ms(5000), "5 seconds");
        assert_eq!(humanize_duration_ms(60_000), "1 minute");
        assert_eq!(humanize_duration_ms(3 * 3_600_000), "3 hours");
        assert_eq!(humanize_duration_ms(2 * 86_400_000), "2 days");
        assert_eq!(humanize_duration_ms(3 * 7 * 86_400_000), "3 weeks");
        assert_eq!(humanize_duration_ms(2 * 365 * 86_400_000), "2 years");
    }

    fn ev(id: &str, occurred: i64, valid_until: Option<i64>) -> TimelineEvent {
        TimelineEvent::new(id, format!("event {id}"), ts(occurred), valid_until.map(ts))
    }

    const DAY: i64 = 86_400_000;

    #[test]
    fn timeline_orders_events_chronologically() {
        // Deliberately out of order on input.
        let events = vec![
            ev("c", 3 * DAY, Some(3 * DAY)),
            ev("a", 0, Some(0)),
            ev("b", DAY, Some(DAY)),
        ];
        let tl = Timeline::build(events, None);
        let ids: Vec<&str> = tl.entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        assert_eq!(tl.first().unwrap().id, "a");
        assert_eq!(tl.last().unwrap().id, "c");
    }

    #[test]
    fn timeline_annotates_gaps_and_relations() {
        let events = vec![ev("a", 0, Some(0)), ev("b", 3 * DAY, Some(3 * DAY))];
        let tl = Timeline::build(events, None);
        assert_eq!(tl.entries[0].relation_to_prev, None);
        assert_eq!(tl.entries[0].gap_to_prev_ms, None);
        // b is after a.
        assert_eq!(tl.entries[1].relation_to_prev, Some(AllenRelation::After));
        assert_eq!(tl.entries[1].gap_to_prev_ms, Some(3 * DAY));
    }

    #[test]
    fn timeline_span_is_earliest_to_latest() {
        let events = vec![ev("a", 0, Some(0)), ev("b", 3 * DAY, Some(3 * DAY))];
        let tl = Timeline::build(events, None);
        assert_eq!(tl.span_ms, Some(3 * DAY));
        assert_eq!(tl.span_human, "3 days");
    }

    #[test]
    fn timeline_span_uses_occurrences_even_with_ongoing_events() {
        // Span is first→last *occurrence*, independent of open-ended validity.
        let events = vec![ev("a", 0, Some(DAY)), ev("b", 2 * DAY, None)];
        let tl = Timeline::build(events, None);
        assert_eq!(tl.span_ms, Some(2 * DAY));
        assert_eq!(tl.span_human, "2 days");
        // The ongoing event carries no validity end.
        assert_eq!(tl.entries[1].valid_until, None);
    }

    #[test]
    fn timeline_empty() {
        let tl = Timeline::build(vec![], None);
        assert!(tl.entries.is_empty());
        assert_eq!(tl.span_ms, None);
        assert_eq!(tl.span_human, "0 seconds");
        assert!(tl.first().is_none());
    }

    #[test]
    fn timeline_as_of_filters_to_valid_events() {
        // a: valid [0, 2d); b: valid [d, 3d); c: valid [4d, ongoing).
        let events = vec![
            ev("a", 0, Some(2 * DAY)),
            ev("b", DAY, Some(3 * DAY)),
            ev("c", 4 * DAY, None),
        ];
        // As of day 1.5: a and b valid, c not yet.
        let tl = Timeline::build(events.clone(), Some(ts(3 * DAY / 2)));
        let ids: Vec<&str> = tl.entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids, ["a", "b"]);

        // As of day 5: only c (a and b expired).
        let tl2 = Timeline::build(events, Some(ts(5 * DAY)));
        let ids2: Vec<&str> = tl2.entries.iter().map(|e| e.id.as_str()).collect();
        assert_eq!(ids2, ["c"]);
    }

    #[test]
    fn timeline_relation_and_duration_between() {
        let events = vec![ev("a", 0, Some(DAY)), ev("b", 5 * DAY, Some(6 * DAY))];
        let tl = Timeline::build(events, None);
        assert_eq!(tl.relation_between("a", "b"), Some(AllenRelation::Before));
        assert_eq!(tl.relation_between("b", "a"), Some(AllenRelation::After));
        assert_eq!(tl.duration_between("a", "b"), Some(5 * DAY));
        assert_eq!(tl.relation_between("a", "missing"), None);
    }
}
