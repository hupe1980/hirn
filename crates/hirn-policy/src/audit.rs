//! HMAC integrity for audit trail entries.
//!
//! Provides signing and verification of audit log entries using keyed BLAKE3
//! hashes. When `event_hmac_secret` is configured, every audit entry is signed
//! at creation time and can be verified later to detect tampering.

use blake3::Hasher;

/// Compute an HMAC over the given content using a BLAKE3 keyed hash.
///
/// The key must be exactly 32 bytes. Use [`derive_key`] to convert an
/// arbitrary-length secret into a 32-byte key.
pub fn compute_hmac(key: &[u8; 32], content: &[u8]) -> Vec<u8> {
    let mut hasher = Hasher::new_keyed(key);
    hasher.update(content);
    hasher.finalize().as_bytes().to_vec()
}

/// Verify an HMAC against the given content.
///
/// Returns `true` if the HMAC matches, `false` otherwise.
/// Uses constant-time comparison to prevent timing attacks.
pub fn verify_hmac(key: &[u8; 32], content: &[u8], expected: &[u8]) -> bool {
    let computed = compute_hmac(key, content);
    constant_time_eq(&computed, expected)
}

/// Derive a 32-byte BLAKE3 key from an arbitrary-length secret.
///
/// This allows callers to use secrets of any length (e.g., from config files)
/// while producing a fixed-size key for the keyed hash.
pub fn derive_key(secret: &[u8]) -> [u8; 32] {
    blake3::derive_key("hirn-policy audit hmac v1", secret)
}

/// Serialize audit fields into a canonical byte representation for HMAC.
///
/// Fields are concatenated with length-prefixed encoding to prevent
/// ambiguity between different field combinations.
pub fn canonical_audit_bytes(fields: &[&[u8]]) -> Vec<u8> {
    let total: usize = fields.iter().map(|f| 4 + f.len()).sum();
    let mut buf = Vec::with_capacity(total);
    for field in fields {
        buf.extend_from_slice(&(field.len() as u32).to_le_bytes());
        buf.extend_from_slice(field);
    }
    buf
}

/// Constant-time byte comparison (prevents timing side-channels).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_and_verify_hmac() {
        let key = derive_key(b"test-secret");
        let content = b"audit entry content";
        let hmac = compute_hmac(&key, content);
        assert!(verify_hmac(&key, content, &hmac));
    }

    #[test]
    fn mutated_content_fails_verification() {
        let key = derive_key(b"test-secret");
        let content = b"original content";
        let hmac = compute_hmac(&key, content);
        let mutated = b"mutated content";
        assert!(!verify_hmac(&key, mutated, &hmac));
    }

    #[test]
    fn different_keys_produce_different_hmacs() {
        let key1 = derive_key(b"secret-one");
        let key2 = derive_key(b"secret-two");
        let content = b"same content";
        let hmac1 = compute_hmac(&key1, content);
        let hmac2 = compute_hmac(&key2, content);
        assert_ne!(hmac1, hmac2);
    }

    #[test]
    fn wrong_key_fails_verification() {
        let key1 = derive_key(b"correct-key");
        let key2 = derive_key(b"wrong-key");
        let content = b"audit entry";
        let hmac = compute_hmac(&key1, content);
        assert!(!verify_hmac(&key2, content, &hmac));
    }

    #[test]
    fn canonical_bytes_length_prefixed() {
        let bytes = canonical_audit_bytes(&[b"hello", b"world"]);
        // 4 + 5 + 4 + 5 = 18 bytes
        assert_eq!(bytes.len(), 18);
        // First field length = 5
        assert_eq!(&bytes[0..4], &5u32.to_le_bytes());
        assert_eq!(&bytes[4..9], b"hello");
    }

    #[test]
    fn empty_hmac_fails() {
        let key = derive_key(b"test");
        let content = b"content";
        assert!(!verify_hmac(&key, content, &[]));
    }
}
