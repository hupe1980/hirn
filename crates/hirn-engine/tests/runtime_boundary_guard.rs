fn assert_no_direct_runtime_field_access(path: &str, source: &str) {
    for forbidden in [
        "self.storage_runtime.",
        "self.graph_runtime.",
        "self.query_runtime.",
        "self.provider_runtime.",
        "self.write_runtime.",
        "self.policy_runtime.",
        "self.admission_runtime.",
        "self.event_runtime.",
    ] {
        assert!(
            !source.contains(forbidden),
            "{path} should use HirnDB helper/runtime interfaces instead of direct field access: found {forbidden}"
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
