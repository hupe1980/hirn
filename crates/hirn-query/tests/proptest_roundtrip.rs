//! Property tests: parse → format → parse round-trip for HirnQL.

use proptest::prelude::*;

use hirn_query::{parse, query_hash};

// ── Strategies ──────────────────────────────────────────────────────────

/// Safe text content that avoids quoting/escaping issues.
fn safe_text() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z][a-zA-Z ]{0,19}")
        .unwrap()
        .prop_map(|s| s.trim().to_string())
        .prop_filter("non-empty", |s| !s.is_empty())
}

/// Text that DELIBERATELY includes characters requiring escaping (quotes,
/// backslashes) so the serializer's escaping is exercised by the round-trip.
fn escapable_text() -> impl Strategy<Value = String> {
    prop::string::string_regex(r#"[a-zA-Z"\\ ]{1,20}"#)
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

/// Random optional NAMESPACE clause. Emits a quoted namespace containing
/// escapable characters so the serializer's namespace escaping round-trips.
fn opt_namespace() -> impl Strategy<Value = String> {
    prop::option::of(escapable_text()).prop_map(|opt| match opt {
        Some(ns) => format!(" NAMESPACE \"{}\"", escape(&ns)),
        None => String::new(),
    })
}

/// Escape a value the same way the serializer does (for building valid inputs).
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Random optional TOPIC clause with escapable content.
fn opt_topic() -> impl Strategy<Value = String> {
    prop::option::of(escapable_text()).prop_map(|opt| match opt {
        Some(t) => format!(" TOPIC \"{}\"", escape(&t)),
        None => String::new(),
    })
}

/// Random optional DEPTH scheduling clause.
fn opt_depth() -> impl Strategy<Value = String> {
    prop::option::of(prop::sample::select(&["auto", "full", "summary"][..])).prop_map(|opt| {
        match opt {
            Some(d) => format!(" DEPTH {d}"),
            None => String::new(),
        }
    })
}

/// Random optional trailing WITH clauses + BUDGET, exercising clause ordering.
fn opt_with_and_budget() -> impl Strategy<Value = String> {
    (
        any::<bool>(),
        prop::option::of(1..1000usize),
        prop::option::of(prop::sample::select(&["ON", "OFF"][..])),
    )
        .prop_map(|(conflicts, budget, prospective)| {
            let mut s = String::new();
            if let Some(p) = prospective {
                s.push_str(&format!(" WITH PROSPECTIVE {p}"));
            }
            if conflicts {
                s.push_str(" WITH CONFLICTS");
            }
            if let Some(b) = budget {
                s.push_str(&format!(" BUDGET {b}"));
            }
            s
        })
}

/// Strategy for RECALL queries covering the ordering- and escaping-sensitive
/// clauses (about/topic/namespace with escapable text, depth, WITH clauses,
/// budget, limit) — exactly the surface the previous generator avoided.
fn recall_query() -> impl Strategy<Value = String> {
    (
        recall_layers(),
        escapable_text(),
        opt_depth(),
        opt_topic(),
        opt_with_and_budget(),
        opt_namespace(),
        opt_limit(),
    )
        .prop_map(|(layers, about, depth, topic, withs, ns, limit)| {
            format!(
                "RECALL {layers} ABOUT \"{}\"{depth}{topic}{withs}{ns}{limit}",
                escape(&about)
            )
        })
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

/// Strategy for TRAVERSE queries, now including the NAMESPACE clause that the
/// grammar gained (previously it could not round-trip a namespace).
fn traverse_query() -> impl Strategy<Value = String> {
    (
        1..10usize,
        prop::option::of(edge_relation()),
        opt_namespace(),
        opt_limit(),
    )
        .prop_map(|(depth, via, ns, limit)| {
            let via_clause = match via {
                Some(rel) => format!(" VIA {rel}"),
                None => String::new(),
            };
            format!("TRAVERSE FROM \"{FIXED_ULID}\"{via_clause} DEPTH {depth}{ns}{limit}")
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

    /// R-47: the plan-cache key must distinguish string literals that differ
    /// only by ASCII case. Whenever lowercasing the literal actually changes
    /// it, the two RECALL queries must hash to distinct cache keys — otherwise
    /// one query would be served the other's compiled plan (wrong embedding
    /// vector + FTS term). Lowercasing the *keywords* must never change the key.
    #[test]
    fn case_variant_literals_get_distinct_keys(about in "[A-Za-z][A-Za-z ]{0,19}") {
        let about = about.trim().to_string();
        prop_assume!(!about.is_empty());
        let lowered = about.to_ascii_lowercase();

        let q_upper = format!("RECALL episodic ABOUT \"{about}\"");
        let q_lower_lit = format!("RECALL episodic ABOUT \"{lowered}\"");
        if about != lowered {
            prop_assert_ne!(
                query_hash(&q_upper),
                query_hash(&q_lower_lit),
                "case-variant literal must not collide: {:?} vs {:?}",
                q_upper, q_lower_lit
            );
        }

        // Lowercasing only the KEYWORDS (literal unchanged) keeps the same key.
        let q_lower_kw = format!("recall episodic about \"{about}\"");
        prop_assert_eq!(
            query_hash(&q_upper),
            query_hash(&q_lower_kw),
            "case-variant keywords must share a key: {:?} vs {:?}",
            q_upper, q_lower_kw
        );
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
