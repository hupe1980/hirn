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
/// 65,535 (u16::MAX) is sufficient for any realistic number of namespaces or
/// agent identifiers while bounding leaked memory to < 4 MiB per interner.
pub const DEFAULT_INTERNER_MAX_ENTRIES: usize = 65_535;

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
    fn new() -> Self {
        Self::with_max(DEFAULT_INTERNER_MAX_ENTRIES)
    }

    /// Create a new empty interner with a custom entry cap.
    pub fn with_max(max_entries: usize) -> Self {
        Self {
            forward: DashMap::new(),
            reverse: RwLock::new(Vec::new()),
            max_entries,
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
/// Pre-interns `"default"` and `"shared"` on first access.
pub fn namespace_interner() -> &'static StringInterner {
    NAMESPACE_INTERNER.get_or_init(|| {
        let interner = StringInterner::new();
        interner.intern("default");
        interner.intern("shared");
        interner
    })
}

/// Returns the global agent-id interner (lazily initialized).
///
/// Pre-interns `"system"` on first access.
pub fn agent_id_interner() -> &'static StringInterner {
    AGENT_ID_INTERNER.get_or_init(|| {
        let interner = StringInterner::new();
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
}
