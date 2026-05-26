//! Property tests for hirn-core types: `MemoryId`, `Timestamp`, `Metadata`.

use proptest::prelude::*;

use hirn_core::id::MemoryId;
use hirn_core::metadata::{Metadata, MetadataValue};
use hirn_core::timestamp::Timestamp;

// ── MemoryId ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// MemoryId survives bincode serialize → deserialize.
    #[test]
    fn memory_id_bincode_round_trip(_seed in 0u64..u64::MAX) {
        let id = MemoryId::new();
        let bytes = bincode::serialize(&id).unwrap();
        let back: MemoryId = bincode::deserialize(&bytes).unwrap();
        prop_assert_eq!(id, back);
    }

    /// MemoryId survives JSON serialize → deserialize.
    #[test]
    fn memory_id_json_round_trip(_seed in 0u64..u64::MAX) {
        let id = MemoryId::new();
        let json = serde_json::to_string(&id).unwrap();
        let back: MemoryId = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(id, back);
    }

    /// MemoryId string display → parse round-trip.
    #[test]
    fn memory_id_display_parse_round_trip(_seed in 0u64..u64::MAX) {
        let id = MemoryId::new();
        let s = id.to_string();
        let back = MemoryId::parse(&s).unwrap();
        prop_assert_eq!(id, back);
    }
}

// ── Timestamp ───────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Timestamp survives bincode round-trip.
    #[test]
    fn timestamp_bincode_round_trip(ms in 0u64..4_000_000_000_000u64) {
        let ts = Timestamp::from_millis(ms);
        let bytes = bincode::serialize(&ts).unwrap();
        let back: Timestamp = bincode::deserialize(&bytes).unwrap();
        prop_assert_eq!(ts, back);
    }

    /// Timestamp survives JSON round-trip.
    #[test]
    fn timestamp_json_round_trip(ms in 0u64..4_000_000_000_000u64) {
        let ts = Timestamp::from_millis(ms);
        let json = serde_json::to_string(&ts).unwrap();
        let back: Timestamp = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(ts, back);
    }

    /// Ordering consistency: a < b in millis → Timestamp(a) < Timestamp(b).
    #[test]
    fn timestamp_ordering_consistent(a in 0u64..2_000_000_000_000u64, b in 0u64..2_000_000_000_000u64) {
        let ts_a = Timestamp::from_millis(a);
        let ts_b = Timestamp::from_millis(b);
        prop_assert_eq!(a.cmp(&b), ts_a.cmp(&ts_b));
    }

    /// Millis round-trip: from_millis(ms).millis() == ms.
    #[test]
    fn timestamp_millis_round_trip(ms in 0u64..4_000_000_000_000u64) {
        let ts = Timestamp::from_millis(ms);
        prop_assert_eq!(ts.millis(), ms);
    }
}

// ── Metadata ────────────────────────────────────────────────────────────

fn arb_metadata_value() -> impl Strategy<Value = MetadataValue> {
    prop_oneof![
        Just(MetadataValue::Null),
        any::<bool>().prop_map(MetadataValue::Bool),
        any::<i64>().prop_map(MetadataValue::Int),
        // Only finite floats within safe JSON precision range
        (-1e6f64..1e6f64).prop_map(|f| MetadataValue::Float((f * 1000.0).round() / 1000.0)),
        "[a-zA-Z0-9]{0,20}".prop_map(MetadataValue::String),
    ]
}

fn arb_metadata() -> impl Strategy<Value = Metadata> {
    prop::collection::btree_map("[a-z]{1,8}", arb_metadata_value(), 0..5)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Metadata survives bincode round-trip.
    #[test]
    fn metadata_bincode_round_trip(m in arb_metadata()) {
        let bytes = bincode::serialize(&m).unwrap();
        let back: Metadata = bincode::deserialize(&bytes).unwrap();
        prop_assert_eq!(m, back);
    }

    /// Metadata survives JSON round-trip.
    #[test]
    fn metadata_json_round_trip(m in arb_metadata()) {
        let json = serde_json::to_string(&m).unwrap();
        let back: Metadata = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(m, back);
    }
}
