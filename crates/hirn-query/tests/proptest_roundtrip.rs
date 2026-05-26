//! Property tests: parse → format → parse round-trip for HirnQL.

use proptest::prelude::*;

use hirn_query::parse;

// ── Strategies ──────────────────────────────────────────────────────────

/// Safe text content that avoids quoting/escaping issues.
fn safe_text() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z][a-zA-Z ]{0,19}")
        .unwrap()
        .prop_map(|s| s.trim().to_string())
        .prop_filter("non-empty", |s| !s.is_empty())
}

/// Random layer(s) for RECALL.
fn recall_layers() -> impl Strategy<Value = String> {
    prop::sample::subsequence(&["episodic", "semantic", "procedural"][..], 1..=3)
        .prop_map(|v| v.join(", "))
}

/// Random optional LIMIT clause.
fn opt_limit() -> impl Strategy<Value = String> {
    prop::option::of(1..100usize).prop_map(|opt| match opt {
        Some(n) => format!(" LIMIT {n}"),
        None => String::new(),
    })
}

/// Random optional NAMESPACE clause.
fn opt_namespace() -> impl Strategy<Value = String> {
    prop::option::of(prop::string::string_regex("[a-z]{3,8}").unwrap()).prop_map(|opt| match opt {
        Some(ns) => format!(" NAMESPACE {ns}"),
        None => String::new(),
    })
}

/// Strategy for RECALL queries.
fn recall_query() -> impl Strategy<Value = String> {
    (recall_layers(), safe_text(), opt_namespace(), opt_limit()).prop_map(
        |(layers, about, ns, limit)| format!("RECALL {layers} ABOUT \"{about}\"{ns}{limit}"),
    )
}

/// Random THINK mode.
fn think_mode() -> impl Strategy<Value = String> {
    prop::sample::select(&["local", "global", "hybrid", "raptor", "adaptive"][..])
        .prop_map(String::from)
}

/// Strategy for THINK queries.
fn think_query() -> impl Strategy<Value = String> {
    (safe_text(), think_mode(), opt_namespace(), opt_limit()).prop_map(
        |(about, mode, ns, limit)| {
            // "local" is the default mode (not emitted), for others add MODE clause
            if mode == "global" {
                // THINK GLOBAL has special syntax — GLOBAL before ABOUT
                return format!("THINK GLOBAL ABOUT \"{about}\"{ns}{limit}");
            }
            let mode_clause = if mode == "local" {
                String::new()
            } else {
                format!(" MODE {mode}")
            };
            format!("THINK ABOUT \"{about}\"{ns}{limit}{mode_clause}")
        },
    )
}

/// Random REMEMBER layer.
fn remember_layer() -> impl Strategy<Value = String> {
    prop::sample::select(&["episode", "semantic"][..]).prop_map(String::from)
}

/// Strategy for REMEMBER queries.
fn remember_query() -> impl Strategy<Value = String> {
    (remember_layer(), safe_text())
        .prop_map(|(layer, content)| format!("REMEMBER {layer} CONTENT \"{content}\""))
}

/// A fixed valid ULID for IDs.
const FIXED_ULID: &str = "01HQ1A2B3C4D5E6F7G8H9J0K1M";

/// Strategy for FORGET queries.
fn forget_query() -> impl Strategy<Value = String> {
    Just(format!("FORGET \"{FIXED_ULID}\""))
}

/// Strategy for CONSOLIDATE queries.
fn consolidate_query() -> impl Strategy<Value = String> {
    Just("CONSOLIDATE".to_string())
}

/// Random edge relation for TRAVERSE/CONNECT.
fn edge_relation() -> impl Strategy<Value = String> {
    prop::sample::select(
        &[
            "related_to",
            "causes",
            "caused_by",
            "derived_from",
            "contradicts",
            "supports",
            "temporal_next",
            "part_of",
            "instance_of",
            "similar_to",
            "inhibits",
            "participates_in",
        ][..],
    )
    .prop_map(String::from)
}

/// Strategy for TRAVERSE queries.
fn traverse_query() -> impl Strategy<Value = String> {
    (1..10usize, prop::option::of(edge_relation())).prop_map(|(depth, via)| {
        let via_clause = match via {
            Some(rel) => format!(" VIA {rel}"),
            None => String::new(),
        };
        format!("TRAVERSE FROM \"{FIXED_ULID}\"{via_clause} DEPTH {depth}")
    })
}

/// Strategy for CONNECT queries.
fn connect_query() -> impl Strategy<Value = String> {
    (edge_relation(), prop::option::of(1..99u32)).prop_map(|(relation, weight)| {
        let weight_clause = match weight {
            Some(w) => format!(" WEIGHT 0.{w}"),
            None => String::new(),
        };
        let ulid2 = "01HQ1A2B3C4D5E6F7G8H9J0K2N";
        format!("CONNECT \"{FIXED_ULID}\" TO \"{ulid2}\" AS {relation}{weight_clause}")
    })
}

// ── Round-trip property ─────────────────────────────────────────────────

fn assert_round_trip(input: &str) {
    let ast1 = parse(input).unwrap_or_else(|e| panic!("first parse failed for {input:?}: {e}"));
    let formatted = format!("{ast1}");
    let ast2 = parse(&formatted)
        .unwrap_or_else(|e| panic!("second parse failed for {formatted:?} (from {input:?}): {e}"));
    assert_eq!(
        ast1, ast2,
        "round-trip mismatch:\n  input:     {input:?}\n  formatted: {formatted:?}"
    );
}

// ── Proptests ───────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn recall_round_trip(query in recall_query()) {
        assert_round_trip(&query);
    }

    #[test]
    fn think_round_trip(query in think_query()) {
        assert_round_trip(&query);
    }

    #[test]
    fn remember_round_trip(query in remember_query()) {
        // REMEMBER is blocked at parse time (use direct view APIs).
        // Assert it is consistently rejected with the expected error.
        let result = parse(&query);
        prop_assert!(
            result.is_err(),
            "expected parse error for blocked verb, got Ok for: {query:?}"
        );
        let msg = result.unwrap_err().to_string();
        prop_assert!(
            msg.contains("REMEMBER"),
            "unexpected error for {query:?}: {msg}"
        );
    }

    #[test]
    fn forget_round_trip(query in forget_query()) {
        // FORGET is blocked at parse time (use direct view APIs).
        let result = parse(&query);
        prop_assert!(
            result.is_err(),
            "expected parse error for blocked verb, got Ok for: {query:?}"
        );
        let msg = result.unwrap_err().to_string();
        prop_assert!(
            msg.contains("FORGET"),
            "unexpected error for {query:?}: {msg}"
        );
    }

    #[test]
    fn consolidate_round_trip(query in consolidate_query()) {
        // CONSOLIDATE is blocked at parse time (use direct admin view APIs).
        let result = parse(&query);
        prop_assert!(
            result.is_err(),
            "expected parse error for blocked verb, got Ok for: {query:?}"
        );
        let msg = result.unwrap_err().to_string();
        prop_assert!(
            msg.contains("CONSOLIDATE"),
            "unexpected error for {query:?}: {msg}"
        );
    }

    #[test]
    fn traverse_round_trip(query in traverse_query()) {
        assert_round_trip(&query);
    }

    #[test]
    fn connect_round_trip(query in connect_query()) {
        // CONNECT is blocked at parse time (use graph view APIs).
        let result = parse(&query);
        prop_assert!(
            result.is_err(),
            "expected parse error for blocked verb, got Ok for: {query:?}"
        );
        let msg = result.unwrap_err().to_string();
        prop_assert!(
            msg.contains("CONNECT"),
            "unexpected error for {query:?}: {msg}"
        );
    }
}
