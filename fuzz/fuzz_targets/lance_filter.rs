#![no_main]
use libfuzzer_sys::fuzz_target;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::Namespace;

/// Replicate the `build_lance_filter` logic for fuzz testing.
/// The actual function lives in hirn_engine::db::recall_exec (private).
fn build_lance_filter(
    namespace: Option<&Namespace>,
    after: Option<&Timestamp>,
    before: Option<&Timestamp>,
    time_column: &str,
) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(ns) = namespace {
        parts.push(format!("namespace = '{}'", ns.as_str()));
    }
    if let Some(ts) = after {
        parts.push(format!("{time_column} >= {}", ts.timestamp_ms()));
    }
    if let Some(ts) = before {
        parts.push(format!("{time_column} <= {}", ts.timestamp_ms()));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }

    // Use first byte to decide which parameters are present.
    let flags = data[0];
    let rest = &data[1..];

    let ns = if flags & 0x01 != 0 {
        if let Ok(s) = std::str::from_utf8(rest) {
            // Namespace::new validates input, so this may return None.
            Namespace::new(s.get(..s.len().min(64)).unwrap_or("")).ok()
        } else {
            None
        }
    } else {
        None
    };

    let after = if flags & 0x02 != 0 && rest.len() >= 8 {
        let ms = u64::from_le_bytes(rest[..8].try_into().unwrap());
        Some(Timestamp::from_millis(ms))
    } else {
        None
    };

    let before = if flags & 0x04 != 0 && rest.len() >= 16 {
        let ms = u64::from_le_bytes(rest[8..16].try_into().unwrap());
        Some(Timestamp::from_millis(ms))
    } else {
        None
    };

    let time_col = if flags & 0x08 != 0 {
        "created_at"
    } else {
        "timestamp_ms"
    };

    // Must never panic.
    let result = build_lance_filter(
        ns.as_ref(),
        after.as_ref(),
        before.as_ref(),
        time_col,
    );

    // Verify no SQL injection: result should never contain unquoted user strings.
    if let Some(filter) = &result {
        // The filter must be valid SQL fragment — no unbalanced quotes.
        let single_quotes: usize = filter.chars().filter(|&c| c == '\'').count();
        assert!(single_quotes % 2 == 0, "unbalanced quotes in filter: {filter}");
    }
});
