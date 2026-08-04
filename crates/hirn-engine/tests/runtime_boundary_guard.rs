//! Architecture guard: boundary-sensitive `HirnDB` modules must reach runtime
//! state through `HirnDB`'s own helper methods, not by touching the private
//! runtime fields directly.
//!
//! Direct field access bypasses the invariants those helpers carry (cache
//! eviction, metrics, namespace scoping), and it does so invisibly — the code
//! compiles and the tests pass, and the missing bookkeeping only surfaces much
//! later as a stale cache or an unrecorded write.

/// Runtime fields that must not be reached directly from the guarded modules.
const RUNTIME_FIELDS: &[&str] = &[
    "storage_runtime",
    "graph_runtime",
    "query_runtime",
    "provider_runtime",
    "write_runtime",
    "policy_runtime",
    "admission_runtime",
    "event_runtime",
];

/// Find `self.<field>.` accesses, tolerating any whitespace (including a line
/// break) between the field and the dot. Returns the 1-based line of each hit.
///
/// The earlier check looked for the literal substring `"self.storage_runtime."`
/// and was therefore silently defeated by rustfmt: as soon as a call wrapped
/// onto the next line, `self.storage_runtime\n    .append(…)` no longer
/// contained the pattern and the guard stopped guarding. Matching across
/// arbitrary whitespace is the whole point of this scanner.
fn direct_field_accesses(source: &str, field: &str) -> Vec<usize> {
    let needle = format!("self.{field}");
    let mut hits = Vec::new();
    let mut from = 0usize;
    while let Some(offset) = source[from..].find(&needle) {
        let start = from + offset;
        let after = start + needle.len();
        from = after;

        // `self.storage_runtime_handle` is a different identifier, not a hit.
        if source[after..]
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
        {
            continue;
        }

        // A field access is only a helper-bypassing *call* when a `.` follows;
        // a bare mention (struct initializer, borrow, comment) is not.
        if source[after..].trim_start().starts_with('.') {
            hits.push(source[..start].matches('\n').count() + 1);
        }
    }
    hits
}

fn assert_no_direct_runtime_field_access(path: &str, source: &str) {
    for field in RUNTIME_FIELDS {
        let lines = direct_field_accesses(source, field);
        assert!(
            lines.is_empty(),
            "{path} should use HirnDB helper/runtime interfaces instead of direct \
             field access: found self.{field} at line(s) {lines:?}"
        );
    }
}

#[test]
fn boundary_sensitive_modules_use_hirndb_helpers() {
    let modules = [
        (
            "src/db/query_exec.rs",
            include_str!("../src/db/query_exec.rs"),
        ),
        (
            "src/db/graph_ops.rs",
            include_str!("../src/db/graph_ops.rs"),
        ),
        (
            "src/db/recall_exec.rs",
            include_str!("../src/db/recall_exec.rs"),
        ),
        ("src/db/episodic.rs", include_str!("../src/db/episodic.rs")),
        ("src/db/semantic.rs", include_str!("../src/db/semantic.rs")),
    ];

    for (path, source) in modules {
        assert_no_direct_runtime_field_access(path, source);
    }
}

#[test]
fn scanner_sees_through_a_line_break() {
    // The exact shape rustfmt produces, which the old substring check missed.
    let wrapped = "        self.storage_runtime\n            .append(NAME, batch)\n";
    assert_eq!(direct_field_accesses(wrapped, "storage_runtime"), vec![1]);

    let inline = "self.storage_runtime.append(NAME, batch)";
    assert_eq!(direct_field_accesses(inline, "storage_runtime"), vec![1]);
}

#[test]
fn scanner_does_not_flag_non_uses() {
    // Struct initialization and plain borrows are not helper-bypassing calls.
    let init = "Self { storage_runtime, embedding_dimensions }";
    assert!(direct_field_accesses(init, "storage_runtime").is_empty());

    let borrowed = "let runtime = &self.storage_runtime;";
    assert!(direct_field_accesses(borrowed, "storage_runtime").is_empty());

    // A longer identifier that merely starts with the field name.
    let other = "self.storage_runtime_handle.append(x)";
    assert!(direct_field_accesses(other, "storage_runtime").is_empty());
}
