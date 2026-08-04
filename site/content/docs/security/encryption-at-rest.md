+++
title = "Encryption at Rest"
description = "hirn delegates encryption at rest to storage and the OS — S3/GCS/Azure SSE and full-disk encryption. Application-level field encryption is roadmap, not shipped."
weight = 3
+++

# Encryption at Rest

hirn delegates encryption at rest to the underlying storage layer. **hirn does
not implement custom cryptographic primitives.** This document explains how to
enable encryption for each supported backend.

{% danger() %}
**Encryption at rest is storage- and OS-delegated only.** hirn does **not**
perform application-level encryption today — there is no in-process AES-GCM or
field-level AEAD over memory content, embeddings, or graph data before it
reaches storage. Confidentiality depends entirely on the backend you configure
below (cloud provider SSE/CMK, or OS full-disk encryption). If the storage
layer is not encrypted, data is written in the clear.
{% end %}

{% note() %}
**Field-level AEAD is on the roadmap, not shipped.** Do not design a deployment
around per-field application encryption existing yet. Where content
confidentiality is a hard requirement, combine backend encryption with
[`text_retention`](#text-retention) and Cedar `recall_raw_text` gating to
minimize what is stored and who can read it.
{% end %}

The two integrity and confidentiality controls hirn *does* provide in-process are
distinct from encryption:

| Control | What it provides | What it does **not** provide |
|---------|------------------|------------------------------|
| Event-log HMAC hash-chain | Tamper *evidence* for the audit trail | Encryption or tamper *prevention* |
| `text_retention` + `recall_raw_text` policy | Minimizing and gating raw text | Encryption of what is retained |

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

{% note() %}
The event log is not only per-event signed but also **hash-chained**: each
event folds in the previous event's tag (`prev_hmac`), so deletion or
truncation is detected in addition to mutation, and the full chain can be
verified end-to-end with `EventLog::verify_chain`. See
[Security Architecture → HMAC Integrity](@/docs/security/_index.md#hmac-integrity-hash-chained)
for the chain diagram and threat model.
{% end %}

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
