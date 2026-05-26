# Encryption at Rest

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

hirn delegates encryption at rest to the underlying storage layer. **hirn does
not implement custom cryptographic primitives.** This document explains how to
enable encryption for each supported backend.

## Cloud Object Storage

### AWS S3 (Lance object_store)

| Method | Config key | Description |
|--------|-----------|-------------|
| **SSE-S3** | default | Amazon-managed keys. Enabled automatically for new buckets (2023+). |
| **SSE-KMS** | `AWS_SSE_KMS_KEY_ID` | Customer-managed KMS key. Set the env var to the KMS key ARN. |
| **SSE-C** | Supply 256-bit key in every request via `AWS_SSE_CUSTOMER_KEY` | Customer-provided key. The key is never stored by S3. |

Example — SSE-KMS:

```toml
# hirn.toml
[storage]
uri = "s3://my-bucket/hirn-brain"

# Set the KMS key via environment variable:
# export AWS_SSE_KMS_KEY_ID="arn:aws:kms:us-east-1:123456789012:key/abcd-1234"
```

### Google Cloud Storage (GCS)

| Method | How to enable |
|--------|--------------|
| **Google-managed** | On by default for all GCS objects. |
| **CMEK** | Create a Cloud KMS key and set it as the bucket's default encryption key. |

```bash
gsutil kms authorize -p <project> -k <key-resource-name>
gsutil kms encryption -k <key-resource-name> gs://my-bucket
```

### Azure Blob Storage

| Method | How to enable |
|--------|--------------|
| **Microsoft-managed** | On by default. |
| **CMK** | Configure a customer-managed key in Azure Key Vault and assign it to the storage account. |

```bash
az storage account update \
  --name <account> \
  --resource-group <rg> \
  --encryption-key-name <key-name> \
  --encryption-key-vault <vault-uri>
```

## Local / On-Premise

For local brains stored on disk (`db_path = "brain"`), use OS-level
full-disk encryption:

| OS | Technology | Command |
|----|-----------|---------|
| **macOS** | FileVault | System Settings → Privacy & Security → FileVault |
| **Linux** | LUKS/dm-crypt | `cryptsetup luksFormat /dev/sdX` |
| **Windows** | BitLocker | Settings → Privacy & Security → Device Encryption |

For containerized deployments, mount an encrypted volume into the container.

## Event Log Integrity (HMAC)

Every event in the audit log is signed with a blake3 keyed hash (HMAC) when an
HMAC secret is configured. This provides tamper evidence — not encryption — for
the audit trail.

### Signing

Events are signed automatically when appended to the event log if the
`event_hmac_secret` is set.

### Verification

External auditors can verify the full event log:

```rust
use hirn_engine::{EventLog, EventEnvelope};

// Read all events and verify each HMAC
let failures = event_log.verify_integrity(secret).await?;
assert!(failures.is_empty(), "tampered events: {:?}", failures);

// Or verify individual events:
let events = event_log.read_all().await?;
for event in &events {
    assert!(event.verify_hmac(secret));
}
```

The HMAC covers: sequence number, timestamp, realm, namespace, agent_id, and the
serialized event payload. Any modification to these fields invalidates the HMAC.

## Text Retention

The `text_retention` config controls how much raw text is persisted after
indexing:

| Value | Behavior |
|-------|----------|
| `"full"` (default) | Store full content and summary. |
| `"summary_only"` | Discard raw content after embedding; keep only the summary. |
| `"none"` | Discard all text after embedding; keep only vectors. |

```toml
# hirn.toml
text_retention = "none"  # embedding-only mode
```

Additionally, a Cedar policy can forbid specific principals from seeing raw text
at recall time:

```cedar
// Deny raw text access for agents in the restricted team
forbid(
    principal in Hirn::Team::"restricted",
    action == Hirn::Action::"recall_raw_text",
    resource
);
```

When `recall_raw_text` is denied, recall still returns embedding-matched results
but with empty text fields.
