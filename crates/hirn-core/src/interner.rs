//! Lock-free string interner for zero-cost identity comparisons.
//!
//! Used by [`Namespace`](crate::types::Namespace) and [`AgentId`](crate::types::AgentId) to
//! replace heap-allocated `String` backing with `Copy` `u32` handles. Interned
//! strings are leaked (`Box::leak`) so that `resolve()` can return `&'static str`
//! without lifetime gymnastics on the lock guard.
//!
//! # Design decisions
//!
//! - **Separate interners** for `Namespace` and `AgentId` — avoids id-space
//!   collisions and keeps validation rules independent.
//! - **`DashMap`** for the forward map (string → id) — lock-free concurrent reads.
//! - **`parking_lot::RwLock<Vec>`** for the reverse map (id → string) — append-only,
//!   readers never block each other.
//! - **Leaked strings** — interned values live for `'static`. This is safe because
//!   the interner is append-only and bounded by the number of distinct namespaces
//!   and agents (typically < 1,000).
//! - **`max_entries` cap** — unbounded interning is a memory leak / DoS vector when
//!   `intern()` is reachable from untrusted input.  Callers at system boundaries
//!   must use [`StringInterner::try_intern`] which returns an error when the cap is
//!   reached rather than leaking memory indefinitely.

use std::sync::OnceLock;

use dashmap::DashMap;
use parking_lot::RwLock;

use crate::{HirnError, HirnResult};

/// Default maximum number of distinct strings that may be interned per interner.
///
/// Interned ids are `u32` handles everywhere (nothing packs them into `u16`,
/// and `Namespace`/`AgentId` serialize as strings, so the id never touches
/// disk) — the previous 65,535 cap was an arbitrary availability ceiling: once
/// hit, new agent ids / namespaces were permanently wedged for the process
/// lifetime because the interner is append-only.
///
/// The cap exists only to bound leaked memory (`Box::leak`) against
/// untrusted-input DoS, so it is set generously and can be raised further via
/// the `HIRN_INTERNER_MAX_ENTRIES` environment variable (read once, at first
/// use of the global interners).
///
/// Worst-case memory bound per interner at the default cap: each entry costs
/// roughly `2 × len` bytes (one leaked copy + one `String` key in the forward
/// map) plus ~100 B of map/vec bookkeeping. At 1,048,576 entries of typical
/// ≤ 32-byte identifiers that is ≈ 170 MB — only reachable if an operator
/// actually creates a million distinct namespaces/agents, in which case that
/// working set is expected; hostile input still hits the clean
/// [`StringInterner::try_intern`] error instead of unbounded growth.
pub const DEFAULT_INTERNER_MAX_ENTRIES: usize = 1_048_576;

/// Environment variable overriding the entry cap of the **global** namespace
/// and agent-id interners. Values are clamped to `[1, u32::MAX]`; unparsable
/// values fall back to [`DEFAULT_INTERNER_MAX_ENTRIES`].
pub const INTERNER_MAX_ENTRIES_ENV: &str = "HIRN_INTERNER_MAX_ENTRIES";

/// Resolve the cap for the global interners from the environment, falling
/// back to [`DEFAULT_INTERNER_MAX_ENTRIES`].
fn global_max_entries() -> usize {
    parse_max_entries(std::env::var(INTERNER_MAX_ENTRIES_ENV).ok().as_deref())
}

/// Parse an optional `HIRN_INTERNER_MAX_ENTRIES` value. Split out from
/// [`global_max_entries`] so the policy is testable without mutating global
/// process environment.
fn parse_max_entries(raw: Option<&str>) -> usize {
    match raw.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => match s.parse::<u64>() {
            // Ids are u32, so more than u32::MAX entries can never be handed out.
            Ok(n) if n >= 1 => usize::try_from(n.min(u32::MAX as u64)).unwrap_or(usize::MAX),
            _ => DEFAULT_INTERNER_MAX_ENTRIES,
        },
        None => DEFAULT_INTERNER_MAX_ENTRIES,
    }
}

/// A generic, thread-safe, append-only string interner.
pub struct StringInterner {
    forward: DashMap<String, u32>,
    reverse: RwLock<Vec<&'static str>>,
    /// Maximum number of distinct entries. `try_intern()` returns an error when
    /// this limit is reached; `intern()` panics (reserved for initialisation).
    max_entries: usize,
}

impl StringInterner {
    /// Create a new empty interner with the default cap (`DEFAULT_INTERNER_MAX_ENTRIES`).
    #[cfg(test)]
    fn new() -> Self {
        Self::with_max(DEFAULT_INTERNER_MAX_ENTRIES)
    }

    /// Create a new empty interner with a custom entry cap.
    ///
    /// The cap is clamped to `u32::MAX` because handles are `u32`.
    pub fn with_max(max_entries: usize) -> Self {
        Self {
            forward: DashMap::new(),
            reverse: RwLock::new(Vec::new()),
            max_entries: max_entries.min(u32::MAX as usize),
        }
    }

    /// Intern a string, returning its integer handle.
    ///
    /// # Panics
    ///
    /// Panics if the cap (`max_entries`) is reached. This method is intended only
    /// for **compile-time constant** strings interned during initialisation (e.g.
    /// `"default"`, `"system"`). For strings derived from user or network input,
    /// use [`try_intern`](Self::try_intern) instead.
    pub fn intern(&self, s: &str) -> u32 {
        self.try_intern(s).unwrap_or_else(|_| {
            panic!(
                "StringInterner capacity exceeded ({} entries): cannot intern {:?}",
                self.max_entries, s
            )
        })
    }

    /// Intern a string, returning its integer handle, or an error if the cap is
    /// reached.  This is the safe variant for use at system boundaries where
    /// the string originates from user or network input.
    pub fn try_intern(&self, s: &str) -> HirnResult<u32> {
        // Fast path: already interned (lock-free read).
        if let Some(id) = self.forward.get(s) {
            return Ok(*id);
        }

        // Slow path: acquire write lock and double-check.
        let mut reverse = self.reverse.write();
        if let Some(id) = self.forward.get(s) {
            return Ok(*id);
        }

        let current = reverse.len();
        if current >= self.max_entries {
            return Err(HirnError::InvalidInput(format!(
                "interner capacity exhausted ({} entries): refusing to intern {:?}",
                self.max_entries, s
            )));
        }

        let id = current as u32;
        let leaked: &'static str = Box::leak(s.to_string().into_boxed_str());
        reverse.push(leaked);
        self.forward.insert(s.to_string(), id);
        Ok(id)
    }

    /// Resolve an interned handle back to its string. Panics if the handle is
    /// invalid (programming error — handles are only created by `intern`).
    pub fn resolve(&self, id: u32) -> &'static str {
        let reverse = self.reverse.read();
        reverse[id as usize]
    }

    /// Resolve an interned handle back to its string. Returns `None` if the
    /// handle was never interned (e.g., came from untrusted/deserialized input).
    ///
    /// Prefer this over `resolve()` at system boundaries where `id` may be
    /// attacker-controlled or originate from a different interner instance
    /// (N-M05: stale handle → OOB panic protection).
    pub fn try_resolve(&self, id: u32) -> Option<&'static str> {
        let reverse = self.reverse.read();
        reverse.get(id as usize).copied()
    }

    /// Number of interned strings.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.reverse.read().len()
    }

    /// Whether the interner is empty.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.reverse.read().is_empty()
    }
}

// ── Global interner singletons ──────────────────────────────────────────

static NAMESPACE_INTERNER: OnceLock<StringInterner> = OnceLock::new();
static AGENT_ID_INTERNER: OnceLock<StringInterner> = OnceLock::new();

/// Returns the global namespace interner (lazily initialized).
///
/// Pre-interns `"default"` and `"shared"` on first access. The entry cap is
/// [`DEFAULT_INTERNER_MAX_ENTRIES`] unless overridden via
/// [`INTERNER_MAX_ENTRIES_ENV`].
pub fn namespace_interner() -> &'static StringInterner {
    NAMESPACE_INTERNER.get_or_init(|| {
        let interner = StringInterner::with_max(global_max_entries());
        interner.intern("default");
        interner.intern("shared");
        interner
    })
}

/// Returns the global agent-id interner (lazily initialized).
///
/// Pre-interns `"system"` on first access. The entry cap is
/// [`DEFAULT_INTERNER_MAX_ENTRIES`] unless overridden via
/// [`INTERNER_MAX_ENTRIES_ENV`].
pub fn agent_id_interner() -> &'static StringInterner {
    AGENT_ID_INTERNER.get_or_init(|| {
        let interner = StringInterner::with_max(global_max_entries());
        interner.intern("system");
        interner
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_same_string_returns_same_id() {
        let interner = StringInterner::new();
        let a = interner.intern("hello");
        let b = interner.intern("hello");
        assert_eq!(a, b);
    }

    #[test]
    fn intern_different_strings_returns_different_ids() {
        let interner = StringInterner::new();
        let a = interner.intern("hello");
        let b = interner.intern("world");
        assert_ne!(a, b);
    }

    #[test]
    fn resolve_round_trips() {
        let interner = StringInterner::new();
        let id = interner.intern("test_value");
        assert_eq!(interner.resolve(id), "test_value");
    }

    #[test]
    fn concurrent_interning_is_safe() {
        let interner = StringInterner::new();
        std::thread::scope(|s| {
            for t in 0..4 {
                let interner = &interner;
                s.spawn(move || {
                    for i in 0..250 {
                        let key = format!("thread{t}_key{i}");
                        let id = interner.intern(&key);
                        assert_eq!(interner.resolve(id), key);
                    }
                });
            }
        });
        assert_eq!(interner.len(), 1000);
    }

    #[test]
    fn concurrent_interning_same_keys() {
        let interner = StringInterner::new();
        std::thread::scope(|s| {
            for _ in 0..4 {
                let interner = &interner;
                s.spawn(move || {
                    for i in 0..100 {
                        let key = format!("shared_key_{i}");
                        interner.intern(&key);
                    }
                });
            }
        });
        // All threads interned the same 100 keys — should have exactly 100 entries.
        assert_eq!(interner.len(), 100);
    }

    #[test]
    fn namespace_interner_pre_interns_well_known() {
        let interner = namespace_interner();
        assert_eq!(interner.resolve(0), "default");
        assert_eq!(interner.resolve(1), "shared");
    }

    #[test]
    fn agent_id_interner_pre_interns_system() {
        let interner = agent_id_interner();
        assert_eq!(interner.resolve(0), "system");
    }

    #[test]
    fn try_intern_errors_cleanly_at_the_cap() {
        let interner = StringInterner::with_max(2);
        interner.try_intern("a").unwrap();
        interner.try_intern("b").unwrap();
        // Existing entries stay resolvable...
        assert_eq!(interner.try_intern("a").unwrap(), 0);
        // ...but new ones hit a clean error, not a panic or a leak.
        let err = interner.try_intern("c").unwrap_err();
        assert!(matches!(err, HirnError::InvalidInput(_)));
    }

    #[test]
    fn parse_max_entries_policy() {
        assert_eq!(parse_max_entries(None), DEFAULT_INTERNER_MAX_ENTRIES);
        assert_eq!(parse_max_entries(Some("")), DEFAULT_INTERNER_MAX_ENTRIES);
        assert_eq!(parse_max_entries(Some("  ")), DEFAULT_INTERNER_MAX_ENTRIES);
        assert_eq!(
            parse_max_entries(Some("not-a-number")),
            DEFAULT_INTERNER_MAX_ENTRIES
        );
        assert_eq!(parse_max_entries(Some("0")), DEFAULT_INTERNER_MAX_ENTRIES);
        assert_eq!(parse_max_entries(Some("1024")), 1024);
        // Clamped to the u32 handle space.
        assert_eq!(parse_max_entries(Some("99999999999999")), u32::MAX as usize);
    }

    #[test]
    fn with_max_clamps_to_u32_handle_space() {
        let interner = StringInterner::with_max(usize::MAX);
        assert_eq!(interner.max_entries, u32::MAX as usize);
    }
}
