use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use futures::TryStreamExt;

use hirn_core::audit::AuditEntry;
use hirn_core::timestamp::Timestamp;
use hirn_core::types::AgentId;
use hirn_core::{HirnError, HirnResult};
use hirn_storage::PhysicalStore;
use hirn_storage::store::{ScanOptions, ScanOrdering};

use crate::event::MemoryEvent;
use crate::policy::{Action, AuthzRequest, PolicyEngine};

/// Outcome of verifying the `_audit` dataset's tamper-evident hash chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditChainVerification {
    /// No HMAC secret is configured; audit entries are unsigned and the chain
    /// cannot be verified. Not an error — signing is opt-in.
    Unsigned,
    /// The chain is intact: every entry's tag verified, every entry links to
    /// its predecessor, and the sequence numbers are gap-free.
    Verified {
        /// Number of entries checked.
        entries: usize,
    },
}

pub(crate) struct PolicyAuthorization {
    pub(crate) audit_event: MemoryEvent,
    pub(crate) denial_error: Option<HirnError>,
}

pub(crate) struct PolicyRuntime {
    storage: Arc<dyn PhysicalStore>,
    policy_engine: Option<PolicyEngine>,
    /// Key for signing audit entries, derived from the configured HMAC secret
    /// via [`hirn_policy::audit::derive_key`]. `None` → entries are unsigned.
    audit_key: Option<[u8; 32]>,
    /// Next audit sequence number to assign.
    next_audit_seq: AtomicU64,
    /// Tag of the most recently appended audit entry — the head of the hash
    /// chain. Held behind an async mutex so signed appends serialize their
    /// sign-and-chain critical section, keeping the chain gap-free and ordered.
    chain_head: tokio::sync::Mutex<Option<String>>,
}

impl PolicyRuntime {
    /// Create an unsigned runtime over an empty (or fresh) store.
    ///
    /// Production code should use [`Self::open`], which recovers the audit
    /// sequence counter and chain head from an existing `_audit` dataset.
    #[cfg(test)]
    pub(crate) fn new(storage: Arc<dyn PhysicalStore>) -> Self {
        Self {
            storage,
            policy_engine: None,
            audit_key: None,
            next_audit_seq: AtomicU64::new(0),
            chain_head: tokio::sync::Mutex::new(None),
        }
    }

    /// Open the policy runtime, recovering audit-trail chain state from the
    /// existing `_audit` dataset so the chain continues unbroken across
    /// restarts. When `hmac_secret` is `Some`, every audit entry appended
    /// through this runtime is signed and hash-chained (tamper-evident).
    pub(crate) async fn open(
        storage: Arc<dyn PhysicalStore>,
        hmac_secret: Option<&[u8]>,
    ) -> HirnResult<Self> {
        if hmac_secret.is_none() {
            tracing::warn!(
                "event_hmac_secret is not configured; audit entries will not be tamper-evident"
            );
        }
        let audit_key = hmac_secret.map(hirn_policy::audit::derive_key);
        let (next_seq, chain_head) = Self::recover_audit_chain_state(&*storage).await?;
        Ok(Self {
            storage,
            policy_engine: None,
            audit_key,
            next_audit_seq: AtomicU64::new(next_seq),
            chain_head: tokio::sync::Mutex::new(chain_head),
        })
    }

    /// Recover the next seq and the hmac of the highest-seq audit entry (the
    /// current chain head) from the `_audit` dataset.
    async fn recover_audit_chain_state(
        storage: &dyn PhysicalStore,
    ) -> HirnResult<(u64, Option<String>)> {
        use arrow_array::Array;

        let dataset = hirn_storage::datasets::audit::DATASET_NAME;
        if !storage.exists(dataset).await.map_err(HirnError::storage)? {
            return Ok((0, None));
        }
        if storage
            .count(dataset, None)
            .await
            .map_err(HirnError::storage)?
            == 0
        {
            return Ok((0, None));
        }

        let mut batches = storage
            .scan_stream(
                dataset,
                ScanOptions {
                    columns: Some(vec!["seq".into(), "hmac".into()]),
                    filter: None,
                    exact_filter: None,
                    order_by: Some(vec![ScanOrdering::desc("seq")]),
                    limit: Some(1),
                    offset: None,
                },
            )
            .await
            .map_err(HirnError::storage)?;

        let mut next_seq = 0u64;
        let mut head = None;
        if let Some(batch) = batches.try_next().await.map_err(HirnError::storage)? {
            let seq_col = batch
                .column_by_name("seq")
                .and_then(|c| c.as_any().downcast_ref::<arrow_array::UInt64Array>())
                .ok_or_else(|| HirnError::storage("audit seq column is not UInt64"))?;
            if seq_col.len() > 0 {
                next_seq = seq_col.value(0) + 1;
            }
            if let Some(col) = batch.column_by_name("hmac") {
                if let Some(arr) = col.as_any().downcast_ref::<arrow_array::StringArray>() {
                    if arr.len() > 0 && !arr.is_null(0) {
                        head = Some(arr.value(0).to_string());
                    }
                }
            }
        }
        Ok((next_seq, head))
    }

    pub(crate) fn set_engine(&mut self, engine: PolicyEngine) {
        self.policy_engine = Some(engine);
    }

    pub(crate) fn engine(&self) -> Option<&PolicyEngine> {
        self.policy_engine.as_ref()
    }

    pub(crate) fn authorize(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> Option<PolicyAuthorization> {
        let Some(engine) = &self.policy_engine else {
            return None;
        };

        let span = tracing::info_span!(
            "recall.authorize",
            agent_id = %agent_id,
            action = %action,
            decision = tracing::field::Empty,
            policy_ids = tracing::field::Empty,
            latency_us = tracing::field::Empty,
        );

        let _guard = span.enter();

        let request = AuthzRequest {
            agent_id: agent_id.to_string(),
            action,
            realm: realm.to_string(),
            namespace: namespace.to_string(),
        };

        let authz_start = std::time::Instant::now();
        let decision = engine.authorize(&request);
        let authz_elapsed = authz_start.elapsed();

        let decision_label = if decision.allowed { "allow" } else { "deny" };
        let latency_us = authz_elapsed.as_micros() as u64;
        span.record("decision", decision_label);
        span.record("latency_us", latency_us);
        span.record("policy_ids", &format!("{:?}", decision.policy_ids));

        metrics::counter!(crate::metrics::AUTHZ_DECISIONS_TOTAL, "decision" => decision_label)
            .increment(1);
        metrics::histogram!(crate::metrics::AUTHZ_LATENCY_SECONDS)
            .record(authz_elapsed.as_secs_f64());

        let audit_event = if decision.allowed {
            MemoryEvent::AccessGranted {
                action: action.to_string(),
                realm: realm.to_string(),
                namespace: namespace.to_string(),
                policy_ids: decision.policy_ids.clone(),
            }
        } else {
            MemoryEvent::AccessDenied {
                action: action.to_string(),
                realm: realm.to_string(),
                namespace: namespace.to_string(),
                reasons: decision.reasons.clone(),
                policy_ids: decision.policy_ids.clone(),
            }
        };

        let denial_error = if decision.allowed {
            None
        } else {
            let reasons = if decision.reasons.is_empty() {
                "no matching permit policy".to_string()
            } else {
                decision.reasons.join("; ")
            };
            Some(HirnError::AccessDenied(format!(
                "{} cannot {} on {}{}: {}",
                agent_id,
                action,
                realm,
                if namespace.is_empty() {
                    String::new()
                } else {
                    format!("/{namespace}")
                },
                reasons,
            )))
        };

        Some(PolicyAuthorization {
            audit_event,
            denial_error,
        })
    }

    pub(crate) fn is_action_allowed(
        &self,
        agent_id: &str,
        action: Action,
        realm: &str,
        namespace: &str,
    ) -> bool {
        let Some(engine) = &self.policy_engine else {
            return true;
        };

        let request = AuthzRequest {
            agent_id: agent_id.to_string(),
            action,
            realm: realm.to_string(),
            namespace: namespace.to_string(),
        };

        engine.authorize(&request).allowed
    }

    pub(crate) async fn append_audit(
        &self,
        actor: Option<AgentId>,
        action: hirn_core::audit::AuditAction,
    ) -> HirnResult<()> {
        self.append_audit_entry(hirn_core::audit::AuditEntry::new(actor, action))
            .await
    }

    /// Append a pre-built audit entry, assigning its seq and — when an HMAC
    /// secret is configured — signing it and linking it into the hash chain.
    pub(crate) async fn append_audit_entry(&self, mut entry: AuditEntry) -> HirnResult<()> {
        if let Some(key) = self.audit_key.as_ref() {
            // Allocate the seq *inside* the chain-head critical section so that
            // seq order == chain-link order == persist order. Allocating the
            // seq before taking the lock lets two concurrent appends acquire
            // the lock in the opposite order to their seq numbers, which makes
            // each entry's `prev_hmac` link to the wrong predecessor and causes
            // `verify_audit_chain` (which walks seq-ascending) to report a
            // false tamper on a trail that was never touched. The head advances
            // only after the durable write so a failed write never orphans the
            // chain.
            let mut head = self.chain_head.lock().await;
            entry.seq = self.next_audit_seq.fetch_add(1, Ordering::AcqRel);
            entry.prev_hmac.clone_from(&head);
            entry.hmac = Some(Self::audit_entry_tag(key, &entry)?);
            self.persist_audit_entry(&entry).await?;
            head.clone_from(&entry.hmac);
        } else {
            entry.seq = self.next_audit_seq.fetch_add(1, Ordering::AcqRel);
            self.persist_audit_entry(&entry).await?;
        }
        Ok(())
    }

    async fn persist_audit_entry(&self, entry: &AuditEntry) -> HirnResult<()> {
        let batch = hirn_storage::datasets::audit::to_batch(std::slice::from_ref(entry))
            .map_err(HirnError::storage)?;
        self.storage
            .append(hirn_storage::datasets::audit::DATASET_NAME, batch)
            .await
            .map_err(HirnError::storage)?;
        Ok(())
    }

    /// Compute the hex-encoded keyed-hash tag for an audit entry.
    ///
    /// Covers seq, id, timestamp, actor, the previous entry's tag, and the
    /// canonical JSON action payload — the same fields the storage layer
    /// persists — so any mutation of a stored entry invalidates its tag.
    fn audit_entry_tag(key: &[u8; 32], entry: &AuditEntry) -> HirnResult<String> {
        let action_json = serde_json::to_vec(&entry.action)
            .map_err(|e| HirnError::storage(format!("audit action serialize: {e}")))?;
        let content = hirn_policy::audit::canonical_audit_bytes(&[
            &entry.seq.to_le_bytes(),
            entry.id.to_string().as_bytes(),
            &entry.timestamp.timestamp_ms().to_le_bytes(),
            entry
                .actor
                .as_ref()
                .map(|a| a.as_str())
                .unwrap_or_default()
                .as_bytes(),
            entry.prev_hmac.as_deref().unwrap_or_default().as_bytes(),
            &action_json,
        ]);
        Ok(hex_encode(&hirn_policy::audit::compute_hmac(key, &content)))
    }

    /// Verify an entry's tag against the signing key (constant-time).
    fn audit_entry_tag_valid(key: &[u8; 32], entry: &AuditEntry) -> bool {
        let Some(stored) = entry.hmac.as_deref() else {
            return false;
        };
        let Some(stored_bytes) = hex_decode(stored) else {
            return false;
        };
        let action_json = match serde_json::to_vec(&entry.action) {
            Ok(bytes) => bytes,
            Err(_) => return false,
        };
        let content = hirn_policy::audit::canonical_audit_bytes(&[
            &entry.seq.to_le_bytes(),
            entry.id.to_string().as_bytes(),
            &entry.timestamp.timestamp_ms().to_le_bytes(),
            entry
                .actor
                .as_ref()
                .map(|a| a.as_str())
                .unwrap_or_default()
                .as_bytes(),
            entry.prev_hmac.as_deref().unwrap_or_default().as_bytes(),
            &action_json,
        ]);
        hirn_policy::audit::verify_hmac(key, &content, &stored_bytes)
    }

    /// Verify the full tamper-evident audit chain: every entry's own HMAC, the
    /// `prev_hmac` linkage between consecutive entries, and gap-free `seq`
    /// contiguity. Detects mutated entries (bad tag), deleted entries (broken
    /// linkage or seq gap), and truncation.
    ///
    /// Returns [`AuditChainVerification::Unsigned`] when no HMAC secret is
    /// configured, [`AuditChainVerification::Verified`] when the chain is
    /// intact, or an error describing the first break.
    pub(crate) async fn verify_audit_chain(&self) -> HirnResult<AuditChainVerification> {
        let Some(key) = self.audit_key.as_ref() else {
            return Ok(AuditChainVerification::Unsigned);
        };

        let entries = self.audit_log(None, None).await?;
        let mut prev_seq: Option<u64> = None;
        let mut prev_tag: Option<String> = None;
        for entry in &entries {
            if let Some(ps) = prev_seq {
                if entry.seq != ps + 1 {
                    return Err(HirnError::storage(format!(
                        "audit chain seq gap: {ps} → {} (missing entries)",
                        entry.seq
                    )));
                }
            }
            if !Self::audit_entry_tag_valid(key, entry) {
                return Err(HirnError::storage(format!(
                    "audit chain: entry seq {} has an invalid or missing HMAC",
                    entry.seq
                )));
            }
            if entry.prev_hmac != prev_tag {
                return Err(HirnError::storage(format!(
                    "audit chain: entry seq {} does not link to its predecessor \
                     (expected prev_hmac {:?}, found {:?})",
                    entry.seq, prev_tag, entry.prev_hmac
                )));
            }
            prev_seq = Some(entry.seq);
            prev_tag.clone_from(&entry.hmac);
        }
        Ok(AuditChainVerification::Verified {
            entries: entries.len(),
        })
    }

    pub(crate) async fn audit_log(
        &self,
        after: Option<&Timestamp>,
        before: Option<&Timestamp>,
    ) -> HirnResult<Vec<hirn_core::audit::AuditEntry>> {
        let mut parts = Vec::new();
        if let Some(a) = after {
            parts.push(format!("timestamp_ms > {}", a.timestamp_ms()));
        }
        if let Some(b) = before {
            parts.push(format!("timestamp_ms < {}", b.timestamp_ms()));
        }
        let filter = if parts.is_empty() {
            None
        } else {
            Some(parts.join(" AND "))
        };

        let opts = hirn_storage::store::ScanOptions {
            filter,
            ..Default::default()
        };
        let batches = self
            .storage
            .scan(hirn_storage::datasets::audit::DATASET_NAME, opts)
            .await
            .map_err(|e| HirnError::storage(e))?;

        let mut result = Vec::new();
        for batch in &batches {
            let entries = hirn_storage::datasets::audit::from_batch(batch)
                .map_err(|e| HirnError::storage(e))?;
            result.extend(entries);
        }
        // Seq is the authoritative append order (and what chain verification
        // walks); timestamp breaks ties only for legacy unsigned rows.
        result.sort_by_key(|entry| (entry.seq, entry.timestamp));
        Ok(result)
    }
}

/// Hex-encode bytes (lowercase).
fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Hex-decode a lowercase/uppercase hex string. Returns `None` on malformed
/// input (odd length or non-hex characters).
fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    use hirn_storage::{HirnDb, HirnDbConfig};

    #[tokio::test(flavor = "multi_thread")]
    async fn no_engine_soft_check_allows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lance_path = dir.path().join("lance");
        let storage = HirnDb::open(HirnDbConfig::local(lance_path.to_str().expect("path")))
            .await
            .expect("open storage")
            .store_arc();
        let runtime = PolicyRuntime::new(storage);

        assert!(runtime.is_action_allowed("agent", Action::Recall, "realm", "namespace"));
        assert!(
            runtime
                .authorize("agent", Action::Recall, "realm", "namespace")
                .is_none()
        );
    }

    fn ns_action(name: &str) -> hirn_core::audit::AuditAction {
        hirn_core::audit::AuditAction::NamespaceCreated {
            namespace: name.into(),
        }
    }

    fn mem_storage() -> Arc<dyn PhysicalStore> {
        Arc::new(hirn_storage::memory_store::MemoryStore::new())
    }

    const SECRET: &[u8] = b"a-32-byte-secret-key-for-testing";

    #[tokio::test]
    async fn signed_audit_entries_form_valid_chain() {
        let storage = mem_storage();
        let runtime = PolicyRuntime::open(storage, Some(SECRET)).await.unwrap();
        for i in 0..5 {
            runtime
                .append_audit(None, ns_action(&format!("ns_{i}")))
                .await
                .unwrap();
        }

        assert_eq!(
            runtime.verify_audit_chain().await.unwrap(),
            AuditChainVerification::Verified { entries: 5 }
        );

        // Entries are seq-ordered and each (after the first) links to its
        // predecessor's tag.
        let entries = runtime.audit_log(None, None).await.unwrap();
        assert_eq!(entries.len(), 5);
        assert!(entries[0].prev_hmac.is_none());
        for (i, pair) in entries.windows(2).enumerate() {
            assert_eq!(pair[0].seq, i as u64);
            assert_eq!(pair[1].prev_hmac, pair[0].hmac);
        }
    }

    #[tokio::test]
    async fn mutating_an_entry_breaks_verification() {
        let storage = mem_storage();
        let runtime = PolicyRuntime::open(storage.clone(), Some(SECRET))
            .await
            .unwrap();
        for i in 0..3 {
            runtime
                .append_audit(None, ns_action(&format!("ns_{i}")))
                .await
                .unwrap();
        }
        runtime.verify_audit_chain().await.unwrap();

        // Rewrite the middle entry in place (an attacker with store access):
        // same seq and tags, different action payload.
        let entries = runtime.audit_log(None, None).await.unwrap();
        let mut forged = entries[1].clone();
        forged.action = ns_action("forged");
        storage
            .delete(hirn_storage::datasets::audit::DATASET_NAME, "seq = 1")
            .await
            .unwrap();
        let batch = hirn_storage::datasets::audit::to_batch(std::slice::from_ref(&forged)).unwrap();
        storage
            .append(hirn_storage::datasets::audit::DATASET_NAME, batch)
            .await
            .unwrap();

        let err = runtime.verify_audit_chain().await.unwrap_err();
        assert!(
            err.to_string().contains("invalid or missing HMAC"),
            "mutating an entry must invalidate its tag, got: {err}"
        );
    }

    #[tokio::test]
    async fn deleting_an_entry_breaks_verification() {
        let storage = mem_storage();
        let runtime = PolicyRuntime::open(storage.clone(), Some(SECRET))
            .await
            .unwrap();
        for i in 0..4 {
            runtime
                .append_audit(None, ns_action(&format!("ns_{i}")))
                .await
                .unwrap();
        }
        runtime.verify_audit_chain().await.unwrap();

        // Excise a middle entry directly from storage. Per-entry tags still
        // verify, but the chain must not.
        storage
            .delete(hirn_storage::datasets::audit::DATASET_NAME, "seq = 2")
            .await
            .unwrap();
        assert!(
            runtime.verify_audit_chain().await.is_err(),
            "removing an entry must break the hash chain"
        );
    }

    #[tokio::test]
    async fn unsigned_mode_appends_and_reports_unsigned() {
        let storage = mem_storage();
        let runtime = PolicyRuntime::open(storage, None).await.unwrap();
        runtime.append_audit(None, ns_action("ns")).await.unwrap();

        let entries = runtime.audit_log(None, None).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].hmac.is_none());
        assert_eq!(
            runtime.verify_audit_chain().await.unwrap(),
            AuditChainVerification::Unsigned
        );
    }

    #[tokio::test]
    async fn chain_continues_across_reopen() {
        let storage = mem_storage();
        {
            let runtime = PolicyRuntime::open(storage.clone(), Some(SECRET))
                .await
                .unwrap();
            for i in 0..3 {
                runtime
                    .append_audit(None, ns_action(&format!("ns_{i}")))
                    .await
                    .unwrap();
            }
        }

        // Reopen: seq counter and chain head are recovered from the max-seq
        // row, so new entries continue the existing chain unbroken.
        let runtime = PolicyRuntime::open(storage, Some(SECRET)).await.unwrap();
        for i in 3..5 {
            runtime
                .append_audit(None, ns_action(&format!("ns_{i}")))
                .await
                .unwrap();
        }
        assert_eq!(
            runtime.verify_audit_chain().await.unwrap(),
            AuditChainVerification::Verified { entries: 5 }
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn open_mode_engine_authorize_produces_audit_event() {
        let dir = tempfile::tempdir().expect("tempdir");
        let lance_path = dir.path().join("lance");
        let storage = HirnDb::open(HirnDbConfig::local(lance_path.to_str().expect("path")))
            .await
            .expect("open storage")
            .store_arc();
        let mut runtime = PolicyRuntime::new(storage);
        runtime.set_engine(PolicyEngine::open_mode());

        let decision = runtime
            .authorize("agent", Action::Recall, "realm", "namespace")
            .expect("configured engine should authorize");

        assert!(decision.denial_error.is_none());
        assert!(matches!(
            decision.audit_event,
            MemoryEvent::AccessGranted { .. }
        ));
    }
}
