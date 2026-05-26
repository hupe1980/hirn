//! `EmbeddingDimension` — a validated, type-safe embedding vector dimension.
//!
//! Replaces bare `usize` at all public API boundaries so that:
//! - A zero dimension is rejected at construction time, not silently propagated.
//! - A mismatch between the dimension written to storage and the dimension
//!   configured at open time produces `HirnError::DimensionMismatch` rather
//!   than a cryptic Arrow type error deep inside operator execution.
//! - Distributed deployments get a clear error message with `stored` and
//!   `configured` values when they diverge after a schema migration.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::HirnError;

/// A validated embedding-vector dimension.
///
/// Valid range: `1..=65_535`.
/// Stored as a `u32` — larger than the largest production embedding models
/// (OpenAI text-embedding-3-large: 3072 dims) while still fitting in `u16`
/// if ever needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct EmbeddingDimension(u32);

impl EmbeddingDimension {
    /// Maximum legal dimension.
    pub const MAX: u32 = 65_535;

    /// Minimum legal dimension.
    pub const MIN: u32 = 1;

    /// Create from a runtime `u32`, returning an error if the value is out of
    /// range.
    ///
    /// # Errors
    /// Returns `HirnError::InvalidConfig` when `dims == 0` or `dims > 65_535`.
    pub fn new(dims: u32) -> Result<Self, HirnError> {
        if dims < Self::MIN || dims > Self::MAX {
            return Err(HirnError::InvalidConfig {
                field: "embedding_dimensions".into(),
                value: dims.to_string(),
                reason: format!("must be in the range {}..={}", Self::MIN, Self::MAX),
            });
        }
        Ok(Self(dims))
    }

    /// Create from a compile-time constant.
    ///
    /// # Panics
    /// Panics at const-evaluation time (compile error) if `dims == 0` or
    /// `dims > 65_535`.  Intended for literals in tests and config defaults;
    /// use [`new`](Self::new) for runtime values.
    #[must_use]
    pub const fn new_const(dims: u32) -> Self {
        assert!(dims >= Self::MIN, "embedding_dimensions must be >= 1");
        assert!(dims <= Self::MAX, "embedding_dimensions must be <= 65_535");
        Self(dims)
    }

    /// Return the raw `u32` dimension value.
    #[must_use]
    #[inline]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Return the dimension as a `usize` (for Arrow / Lance APIs).
    #[must_use]
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for EmbeddingDimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<EmbeddingDimension> for usize {
    fn from(d: EmbeddingDimension) -> usize {
        d.0 as usize
    }
}

impl From<EmbeddingDimension> for u32 {
    fn from(d: EmbeddingDimension) -> u32 {
        d.0
    }
}

impl TryFrom<u32> for EmbeddingDimension {
    type Error = HirnError;
    fn try_from(n: u32) -> Result<Self, Self::Error> {
        Self::new(n)
    }
}

impl TryFrom<usize> for EmbeddingDimension {
    type Error = HirnError;
    fn try_from(n: usize) -> Result<Self, Self::Error> {
        if n > Self::MAX as usize {
            return Err(HirnError::InvalidConfig {
                field: "embedding_dimensions".into(),
                value: n.to_string(),
                reason: format!("must be in the range {}..={}", Self::MIN, Self::MAX),
            });
        }
        #[allow(clippy::cast_possible_truncation)]
        Self::new(n as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid() {
        let d = EmbeddingDimension::new(768).unwrap();
        assert_eq!(d.get(), 768);
        assert_eq!(d.as_usize(), 768);
        assert_eq!(usize::from(d), 768);
    }

    #[test]
    fn new_const_valid() {
        const D: EmbeddingDimension = EmbeddingDimension::new_const(128);
        assert_eq!(D.get(), 128);
    }

    #[test]
    fn new_zero_is_err() {
        assert!(EmbeddingDimension::new(0).is_err());
    }

    #[test]
    fn new_over_max_is_err() {
        assert!(EmbeddingDimension::new(65_536).is_err());
    }

    #[test]
    fn try_from_usize() {
        let d: EmbeddingDimension = 512usize.try_into().unwrap();
        assert_eq!(d.get(), 512);
    }

    #[test]
    fn serde_roundtrip() {
        let d = EmbeddingDimension::new_const(384);
        let json = serde_json::to_string(&d).unwrap();
        assert_eq!(json, "384");
        let back: EmbeddingDimension = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }
}
