# hirn-policy

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Cedar-based authorization and policy enforcement for the hirn cognitive memory database.

## Overview

Provides RBAC + ABAC authorization using [Cedar](https://www.cedarpolicy.com/) v4.9+. Policies are evaluated at compile time via partial evaluation — the result is a plan rewrite, not a runtime gate.

## Components

### PolicyEngine

Central authorization engine:

```rust
let engine = PolicyEngine::new(schema, policies)?;

let decision = engine.authorize(AuthzRequest {
    principal: agent_id,
    action: "recall",
    resource: namespace,
    context: HashMap::new(),
})?;

match decision {
    AuthzDecision::Allow => { /* proceed */ }
    AuthzDecision::Deny(reason) => { /* reject */ }
}
```

### Pre-Mutation Enforcement

Deny happens **before** any data write — never after-the-fact.

### PolicyPushdownRule

Optimizer rule (in `hirn-exec`) that injects `namespace IN (...)` filters early in the DataFusion plan. Policies become plan rewrites, not runtime checks.

## Audit Trail

HMAC-signed audit events for tamper detection:

```rust
let audit = AuditEntry::new(agent, action, resource, decision);
let signed = audit.sign(&hmac_key);
// Verify: signed.verify(&hmac_key)
```

Audit entries stored in the `mcfa_audit_log` Lance dataset.

## Default Policies

- `DEFAULT_OPEN_POLICY` — Permits all operations (development/testing)
- `DEFAULT_SCHEMA` — Cedar entity schema defining Agent, Team, Organization, Namespace, Realm, MemoryLayer, Operation, Tool, and the live hirn action set

## Cedar Entity Model

```cedar
entity Agent;
entity Namespace;
entity Realm;

action "remember", "correct", "supersede", "merge", "retract", "purge",
       "recall", "think", "forget", "consolidate", "watch", "connect",
       "execute", "admin", "recall_raw_text", "read", "write", "delete";
```
