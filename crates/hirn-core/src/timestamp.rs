use std::fmt;

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Nanosecond-precision timestamp backed by `chrono::DateTime<Utc>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(DateTime<Utc>);

impl Timestamp {
    /// Current UTC time.
    #[must_use]
    pub fn now() -> Self {
        Self(Utc::now())
    }

    /// Create from a `chrono::DateTime<Utc>`.
    #[must_use]
    pub const fn from_datetime(dt: DateTime<Utc>) -> Self {
        Self(dt)
    }

    /// Get the inner `DateTime<Utc>`.
    #[must_use]
    pub const fn as_datetime(&self) -> DateTime<Utc> {
        self.0
    }

    /// Milliseconds since Unix epoch.
    #[must_use]
    pub const fn timestamp_ms(&self) -> i64 {
        self.0.timestamp_millis()
    }

    /// Milliseconds since Unix epoch (unsigned, saturates to 0 for pre-epoch).
    #[must_use]
    pub const fn millis(&self) -> u64 {
        let ms = self.0.timestamp_millis();
        if ms < 0 { 0 } else { ms as u64 }
    }

    /// Construct from milliseconds since Unix epoch (F-36).
    ///
    /// On overflow (a `ms` value beyond chrono's representable range) the
    /// timestamp **saturates** to [`DateTime::<Utc>::MAX_UTC`] rather than
    /// silently collapsing to the Unix epoch (1970), which would mis-date the
    /// record as ancient instead of far-future. This mirrors the TTL
    /// saturation used elsewhere. `ms / 1000` never exceeds `i64::MAX`, so the
    /// cast to signed seconds cannot itself overflow.
    #[must_use]
    pub fn from_millis(ms: u64) -> Self {
        let secs = (ms / 1000).cast_signed();
        let nanos = ((ms % 1000) * 1_000_000) as u32;
        let dt = DateTime::from_timestamp(secs, nanos).unwrap_or(DateTime::<Utc>::MAX_UTC);
        Self(dt)
    }

    /// Parse either a date-only string (`YYYY-MM-DD`) or an RFC 3339 timestamp.
    #[must_use]
    pub fn parse_date_or_rfc3339(s: &str) -> Option<Self> {
        if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
            let dt = date.and_hms_opt(0, 0, 0)?;
            return Some(Self(Utc.from_utc_datetime(&dt)));
        }

        chrono::DateTime::parse_from_rfc3339(s)
            .ok()
            .map(|dt| Self(dt.with_timezone(&Utc)))
    }
}

/// **Side-effect warning:** Captures the current wall-clock time.
/// Prefer using `Timestamp::now()` explicitly for clarity.
impl Default for Timestamp {
    fn default() -> Self {
        Self::now()
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.to_rfc3339())
    }
}

impl From<DateTime<Utc>> for Timestamp {
    fn from(dt: DateTime<Utc>) -> Self {
        Self(dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_is_recent() {
        let ts = Timestamp::now();
        let diff = Utc::now() - ts.as_datetime();
        assert!(diff.num_seconds() < 1);
    }

    #[test]
    fn ordering_is_chronological() {
        let a = Timestamp::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = Timestamp::now();
        assert!(b > a);
    }

    #[test]
    fn serde_round_trip() {
        let ts = Timestamp::now();
        let bytes = bincode::serialize(&ts).unwrap();
        let back: Timestamp = bincode::deserialize(&bytes).unwrap();
        assert_eq!(ts, back);
    }

    #[test]
    fn display_is_rfc3339() {
        let ts = Timestamp::now();
        let s = ts.to_string();
        assert!(s.contains('T'), "should be RFC 3339 format");
    }

    #[test]
    fn parse_date_or_rfc3339_accepts_both_formats() {
        let date_only = Timestamp::parse_date_or_rfc3339("2026-03-01").unwrap();
        assert_eq!(date_only.to_string(), "2026-03-01T00:00:00+00:00");

        let full = Timestamp::parse_date_or_rfc3339("2026-03-01T12:30:00Z").unwrap();
        assert_eq!(full.to_string(), "2026-03-01T12:30:00+00:00");
    }

    #[test]
    fn parse_date_or_rfc3339_rejects_invalid_input() {
        assert!(Timestamp::parse_date_or_rfc3339("not-a-date").is_none());
    }

    #[test]
    fn from_millis_round_trips_normal_value() {
        let ts = Timestamp::from_millis(1_700_000_000_000);
        assert_eq!(ts.millis(), 1_700_000_000_000);
    }

    #[test]
    fn from_millis_saturates_on_overflow() {
        use chrono::Datelike;
        // A wildly out-of-range value must saturate to the far future, not
        // collapse to the Unix epoch (1970), which would mis-date the record.
        let ts = Timestamp::from_millis(u64::MAX);
        assert_eq!(ts.as_datetime(), DateTime::<Utc>::MAX_UTC);
        assert!(
            ts.as_datetime().year() > 2100,
            "overflow must map to the far future, not 1970"
        );
    }
}
