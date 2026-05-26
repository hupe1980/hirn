use std::fmt;
use std::sync::OnceLock;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Time-sortable, globally unique memory identifier wrapping a ULID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemoryId(ulid::Ulid);

pub(crate) fn next_monotonic_ulid() -> ulid::Ulid {
    static GENERATOR: OnceLock<Mutex<ulid::Generator>> = OnceLock::new();

    GENERATOR
        .get_or_init(|| Mutex::new(ulid::Generator::new()))
        .lock()
        .generate()
        .expect("monotonic ULID overflow")
}

impl MemoryId {
    /// Create a new `MemoryId` with the current timestamp.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self(next_monotonic_ulid())
    }

    /// Create a `MemoryId` from an existing ULID.
    #[must_use]
    pub const fn from_ulid(ulid: ulid::Ulid) -> Self {
        Self(ulid)
    }

    /// Get the inner ULID.
    #[must_use]
    pub const fn as_ulid(&self) -> ulid::Ulid {
        self.0
    }

    /// Extract the millisecond timestamp from the ULID.
    #[must_use]
    pub const fn timestamp_ms(&self) -> u64 {
        self.0.timestamp_ms()
    }

    /// Parse a `MemoryId` from a ULID string.
    pub fn parse(s: &str) -> Result<Self, crate::HirnError> {
        ulid::Ulid::from_string(s)
            .map(Self)
            .map_err(|e| crate::HirnError::InvalidInput(format!("invalid memory id '{s}': {e}")))
    }
}

// NOTE: Default impl intentionally removed (F-012).
// MemoryId::new() generates random state — use it explicitly.

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ulid::Ulid> for MemoryId {
    fn from(ulid: ulid::Ulid) -> Self {
        Self(ulid)
    }
}

// Compile-time assertions: MemoryId must be Copy + Send + Sync.
const _: () = {
    const fn assert_copy<T: Copy>() {}
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}
    assert_copy::<MemoryId>();
    assert_send::<MemoryId>();
    assert_sync::<MemoryId>();
};

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn new_ids_are_unique() {
        let a = MemoryId::new();
        let b = MemoryId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn ids_created_later_sort_after() {
        let a = MemoryId::new();
        // ULID has ms resolution—force a new ms.
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = MemoryId::new();
        assert!(b > a, "later ID should sort after earlier ID");
    }

    #[test]
    fn display_round_trip() {
        let id = MemoryId::new();
        let s = id.to_string();
        assert!(!s.is_empty());
    }

    #[test]
    fn serde_round_trip() {
        let id = MemoryId::new();
        let bytes = bincode::serialize(&id).unwrap();
        let back: MemoryId = bincode::deserialize(&bytes).unwrap();
        assert_eq!(id, back);
    }

    #[test]
    fn size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<MemoryId>(), 16);
    }

    #[test]
    fn copy_semantics() {
        let a = MemoryId::new();
        let b = a; // Copy, not move
        assert_eq!(a, b); // `a` still usable
    }

    #[test]
    fn hashmap_key() {
        let id = MemoryId::new();
        let mut map = HashMap::new();
        map.insert(id, "hello");
        assert_eq!(map.get(&id), Some(&"hello"));
        assert_eq!(map.len(), 1);

        let id2 = MemoryId::new();
        map.insert(id2, "world");
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&id), Some(&"hello"));
        assert_eq!(map.get(&id2), Some(&"world"));
    }
}
