//! Built-in admission controllers.

pub mod contradiction;
pub mod duplicate;
pub mod poisoning;
pub mod rate_limiter;
pub mod surprise;
pub mod token_budget;
pub mod trust;

use hirn_core::types::Namespace;

/// Build a `namespace = '<escaped>'` SQL filter fragment that scopes a vector
/// search to a single namespace.
///
/// Escaping (single-quote doubling) matches the read-path scoping helpers
/// (`scoped_recall_filter` / `build_namespace_filter_sql`) so admission-time
/// searches never leak candidates from a foreign tenant's namespace.
pub(crate) fn namespace_eq_filter(namespace: &Namespace) -> String {
    format!("namespace = '{}'", namespace.as_str().replace('\'', "''"))
}
