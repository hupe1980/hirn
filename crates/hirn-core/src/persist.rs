//! Versioned envelope for bincode-serialized blob payloads.
//!
//! The primary record path persists as Arrow columns, where schema evolution
//! is handled at the Arrow layer. But several datasets carry opaque
//! bincode-serialized blobs (event payloads, quarantined records, offline-job
//! artifacts). Bincode is not self-describing and ignores `serde(default)`,
//! so adding a field to a struct silently makes every previously-persisted
//! blob unreadable — with no way to even detect *why* decoding failed.
//!
//! This module prefixes every blob with a little-endian `u16` format version:
//!
//! ```text
//! [version: u16 LE][bincode(body)]
//! ```
//!
//! Readers dispatch on the version, so a future format change becomes an
//! explicit migration instead of a decode error. Unknown (newer) versions
//! fail with a clear error rather than garbage.

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{HirnError, HirnResult};

/// The current blob format version. Bump when the serialized shape of any
/// enveloped type changes, and add a migration arm to the caller's decoder.
pub const FORMAT_VERSION: u16 = 1;

/// Serialize `value` with the current format-version prefix.
pub fn to_versioned_bytes<T: Serialize>(value: &T) -> HirnResult<Vec<u8>> {
    let body = bincode::serialize(value)
        .map_err(|e| HirnError::storage(format!("versioned serialize: {e}")))?;
    let mut out = Vec::with_capacity(2 + body.len());
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&body);
    Ok(out)
}

/// Deserialize a versioned blob written by [`to_versioned_bytes`].
///
/// Returns a clear error for truncated input or a version newer than this
/// build understands. Older versions decode through their migration arm; with
/// only one version defined so far, that arm is simply the current decoder.
pub fn from_versioned_bytes<T: DeserializeOwned>(bytes: &[u8]) -> HirnResult<T> {
    let (version, body) = split_version(bytes)?;
    match version {
        1 => bincode::deserialize(body)
            .map_err(|e| HirnError::storage(format!("versioned deserialize (v{version}): {e}"))),
        _ => Err(HirnError::storage(format!(
            "unsupported blob format version {version} (this build supports up to {FORMAT_VERSION}); \
             refusing to guess at the layout"
        ))),
    }
}

/// Split a versioned blob into its version and body without decoding.
pub fn split_version(bytes: &[u8]) -> HirnResult<(u16, &[u8])> {
    if bytes.len() < 2 {
        return Err(HirnError::storage(
            "versioned blob too short: missing the format-version prefix".to_string(),
        ));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    Ok((version, &bytes[2..]))
}

#[cfg(test)]
mod tests {
    use serde::Deserialize;

    use super::*;

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Sample {
        name: String,
        value: u32,
    }

    #[test]
    fn round_trip() {
        let s = Sample {
            name: "x".into(),
            value: 7,
        };
        let bytes = to_versioned_bytes(&s).unwrap();
        let back: Sample = from_versioned_bytes(&bytes).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn version_prefix_is_current() {
        let bytes = to_versioned_bytes(&Sample {
            name: String::new(),
            value: 0,
        })
        .unwrap();
        let (version, _) = split_version(&bytes).unwrap();
        assert_eq!(version, FORMAT_VERSION);
    }

    #[test]
    fn unknown_future_version_is_rejected_clearly() {
        let mut bytes = to_versioned_bytes(&Sample {
            name: String::new(),
            value: 0,
        })
        .unwrap();
        bytes[0] = 0xFF;
        bytes[1] = 0xFF;
        let err = from_versioned_bytes::<Sample>(&bytes).unwrap_err();
        assert!(err.to_string().contains("unsupported blob format version"));
    }

    #[test]
    fn truncated_input_is_rejected() {
        let err = from_versioned_bytes::<Sample>(&[0x01]).unwrap_err();
        assert!(err.to_string().contains("too short"));
    }

    #[test]
    fn body_change_shows_up_as_decode_error_with_version() {
        // A valid prefix over a garbage body must fail with the version in
        // the message so operators can tell schema drift from corruption.
        let bytes = [0x01, 0x00, 0xDE, 0xAD];
        let err = from_versioned_bytes::<Sample>(&bytes).unwrap_err();
        assert!(err.to_string().contains("(v1)"));
    }
}
