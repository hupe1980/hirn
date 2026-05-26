# Cedar Policy Guide

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

> How to write authorization policies for hirn using Cedar.

hirn uses [Cedar](https://www.cedarpolicy.com/) (`cedar-policy` v4.9.1) for fine-grained authorization. Cedar is an open-source policy language created by AWS and donated to the CNCF. It supports RBAC, ABAC, entity hierarchies, schema validation, and automated reasoning.

---

## Overview

Every hirn operation (remember, correct, supersede, merge, retract, purge, recall, think, forget, ...) passes through the Cedar policy engine. The engine evaluates:

1. **Who** is making the request (principal — Agent or Team)
2. **What** they want to do (action — remember, recall, etc.)
3. **Where** they want to do it (resource — Namespace or Realm)

If no `permit` policy matches, the request is **denied by default**.

---

## Schema Reference

hirn ships a default Cedar schema at `brain/policies/hirn.cedarschema`:

### Entity Types

| Entity | Parents | Attributes | Description |
|--------|---------|------------|-------------|
| `Agent` | `Team` | `reputation: Long`, `created_at: String` | A registered memory agent |
| `Team` | `Organization` | `description: String` | Group of agents |
| `Organization` | — | `description: String` | Top-level tenant |
| `Realm` | — | `description: String` | Isolation boundary |
| `Namespace` | `Realm` | `classification: String` | Memory access scope |

### Entity Hierarchy

```
Organization
  └── Team
        └── Agent

Realm
  └── Namespace
```

A policy granting access to a `Team` automatically applies to all `Agent` members. A policy granting access to a `Realm` applies to all `Namespace`s within it.

### Actions

| Action | Description | Typical Use |
|--------|-------------|-------------|
| `remember` | Store new memories | Write access |
| `correct` | Append a corrected semantic revision | Semantic edit |
| `supersede` | Advance the authoritative semantic head | Semantic edit |
| `merge` | Merge semantic logical memories into one active chain | Semantic edit |
| `retract` | Tombstone a semantic memory while preserving history | Semantic edit |
| `purge` | Permanently remove a logical memory's revisions | Destructive admin |
| `recall` | Search and retrieve memories | Read access |
| `think` | Assemble LLM context | Read access |
| `forget` | Archive or delete memories | Write access |
| `consolidate` | Run consolidation pipeline | Admin |
| `watch` | Stream real-time events | Read access |
| `connect` | Create graph edges | Write access |
| `execute` | Execute HirnQL queries | Combined |
| `admin` | Administrative operations | Admin |
| `recall_raw_text` | Recall with raw text content | Privileged read |
| `read` | Coarse-grained read permission for tools and layers | Generic read |
| `write` | Coarse-grained write permission for tools and layers | Generic write |
| `delete` | Coarse-grained delete permission for tools and layers | Generic delete |

---

## Policy File Location

Cedar policies are loaded from `brain/policies/*.cedar` at startup. You can have multiple `.cedar` files — they are merged into a single policy set.

```
brain/
└── policies/
    ├── hirn.cedarschema    # Entity schema (required)
    ├── default.cedar       # Shipped default policies
    ├── team-policies.cedar # Team-level access control
    └── security.cedar      # Security restrictions
```

---

## Writing Policies

### Basic RBAC — Team-Based Access

```cedar
// Writers can store and retrieve memories in production
permit(
    principal in Hirn::Team::"writers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"think"],
    resource in Hirn::Realm::"production"
);

// Admins have full access everywhere
permit(
    principal in Hirn::Team::"admins",
    action,
    resource
);

// Readers can only recall and think
permit(
    principal in Hirn::Team::"readers",
    action in [Hirn::Action::"recall", Hirn::Action::"think", Hirn::Action::"watch"],
    resource in Hirn::Realm::"production"
);
```

### ABAC — Attribute-Based Conditions

```cedar
// Block agents with low reputation from writing
forbid(
    principal,
    action in [Hirn::Action::"remember", Hirn::Action::"connect"],
    resource
) when { principal is Hirn::Agent && principal.reputation < 50 };

// Only allow access to confidential namespaces for high-reputation agents
forbid(
    principal,
    action,
    resource
) when {
    resource is Hirn::Namespace &&
    resource.classification == "confidential" &&
    principal is Hirn::Agent &&
    principal.reputation < 80
};
```

### Namespace Classification

```cedar
// Block all access to restricted namespaces unless admin
forbid(
    principal,
    action,
    resource
) when { resource is Hirn::Namespace && resource.classification == "restricted" }
unless { principal in Hirn::Team::"admins" };

// Allow public namespace access to everyone
permit(
    principal,
    action in [Hirn::Action::"recall", Hirn::Action::"think"],
    resource
) when { resource is Hirn::Namespace && resource.classification == "public" };
```

### Realm Isolation (Multi-Tenancy)

```cedar
// Tenant A agents can only access Tenant A realm
permit(
    principal in Hirn::Organization::"tenant-a",
    action,
    resource in Hirn::Realm::"tenant-a"
);

// Tenant B agents can only access Tenant B realm
permit(
    principal in Hirn::Organization::"tenant-b",
    action,
    resource in Hirn::Realm::"tenant-b"
);

// Explicitly deny cross-tenant access
forbid(
    principal in Hirn::Organization::"tenant-a",
    action,
    resource in Hirn::Realm::"tenant-b"
);
```

### Individual Agent Permissions

```cedar
// Grant a specific agent full access to a namespace
permit(
    principal == Hirn::Agent::"senior-researcher",
    action,
    resource == Hirn::Namespace::"experiments"
);

// Deny a specific agent from destructive and admin operations
forbid(
    principal == Hirn::Agent::"intern",
    action in [
        Hirn::Action::"retract",
        Hirn::Action::"purge",
        Hirn::Action::"admin",
        Hirn::Action::"consolidate",
        Hirn::Action::"forget"
    ],
    resource
);
```

---

## Common Patterns

For a larger operator-oriented pattern catalog, see [docs/cedar-patterns.md](cedar-patterns.md).

### Pattern 1: Read-Only Environment

```cedar
// Allow all agents to read
permit(
    principal,
    action in [Hirn::Action::"recall", Hirn::Action::"think", Hirn::Action::"watch"],
    resource in Hirn::Realm::"archive"
);

// Block all writes to archive realm
forbid(
    principal,
    action in [
        Hirn::Action::"remember",
        Hirn::Action::"correct",
        Hirn::Action::"supersede",
        Hirn::Action::"merge",
        Hirn::Action::"retract",
        Hirn::Action::"purge",
        Hirn::Action::"forget",
        Hirn::Action::"connect"
    ],
    resource in Hirn::Realm::"archive"
) unless { principal in Hirn::Team::"admins" };
```

### Pattern 2: Graduated Access

```cedar
// New agents (reputation < 20) can only read
permit(
    principal,
    action in [Hirn::Action::"recall", Hirn::Action::"think"],
    resource
) when { principal is Hirn::Agent && principal.reputation >= 0 };

// Established agents (reputation >= 50) can write and revise semantic state
permit(
    principal,
    action in [
        Hirn::Action::"remember",
        Hirn::Action::"correct",
        Hirn::Action::"supersede",
        Hirn::Action::"merge",
        Hirn::Action::"retract",
        Hirn::Action::"connect"
    ],
    resource
) when { principal is Hirn::Agent && principal.reputation >= 50 };

// Trusted agents (reputation >= 80) can do admin
permit(
    principal,
    action in [Hirn::Action::"consolidate", Hirn::Action::"admin"],
    resource
) when { principal is Hirn::Agent && principal.reputation >= 80 };
```

### Pattern 3: Development vs Production

```cedar
// Open access in development realm
permit(
    principal,
    action,
    resource in Hirn::Realm::"development"
);

// Strict access in production
permit(
    principal in Hirn::Team::"production-writers",
    action in [Hirn::Action::"remember", Hirn::Action::"recall", Hirn::Action::"think"],
    resource in Hirn::Realm::"production"
);

forbid(
    principal,
    action in [Hirn::Action::"retract", Hirn::Action::"purge", Hirn::Action::"forget", Hirn::Action::"admin"],
    resource in Hirn::Realm::"production"
) unless { principal in Hirn::Team::"admins" };
```

---

## Managing Policies via HirnQL

HirnQL provides SQL-like statements for policy management at runtime:

```sql
-- Grant specific actions to an agent on a realm
GRANT remember, correct, supersede, recall, think ON REALM "production" TO AGENT "researcher"

-- Grant admin access to a team on a namespace
GRANT admin, consolidate ON NAMESPACE "system" TO TEAM "ops"

-- Revoke permissions
REVOKE remember ON REALM "production" FROM AGENT "intern"

-- View all active policies
SHOW POLICIES

-- View policies for a specific agent
SHOW POLICIES FOR AGENT "researcher"

-- Debug: why is this allowed/denied?
EXPLAIN POLICY FOR AGENT "researcher" ON REALM "production" ACTION recall
```

---

## Feature Flag

Cedar authorization is gated by the `cedar` feature flag (default: **on**).

```toml
# With Cedar (default)
hirn = "0.1"

# Without Cedar (explicit open mode for development/testing)
hirn = { version = "0.1", default-features = false }
```

When the `cedar` feature is off, all requests are allowed without policy evaluation. Treat this as explicit development/test posture rather than a production default.

---

## PolicyEnforcedStore — Scan-Level Policy

`PolicyEnforcedStore<S: PhysicalStore>` wraps any storage backend and pushes Cedar policy decisions down to the storage scan level. Instead of filtering results after retrieval, it injects namespace predicates directly into Lance scan filters.

### How It Works

1. **Reads (scan, search):** The wrapper fetches the current principal from the task-local `CURRENT_PRINCIPAL` and calls `NamespacePolicy::allowed_namespaces(principal)`. The resulting namespace set is injected as a predicate into the scan filter, so Lance only reads matching rows.

2. **Writes (append, delete):** The target namespace is checked against the policy. If the namespace is not in the allowed set, the operation is rejected with `HirnDbError::PolicyViolation`.

3. **Fail-closed:** If no principal is set on the current task, all operations are denied.

### Usage

```rust
use hirn_storage::{PolicyEnforcedStore, NamespacePolicy};

let policy: Arc<dyn NamespacePolicy> = /* ... */;
let enforced = PolicyEnforcedStore::new(inner_store, policy);

// All operations through `enforced` are now policy-filtered.
enforced.scan("episodic", filter, projection).await?;
```

### Bitmap Index Optimization

For best performance, create a bitmap index on the `namespace` column of each dataset. This allows Lance to evaluate namespace predicates at the index level without scanning row data.

---

## Troubleshooting

### All requests denied

Check that at least one `permit` policy matches. Cedar's default is **deny**. Use `EXPLAIN POLICY` to debug:

```sql
EXPLAIN POLICY FOR AGENT "my-agent" ON REALM "production" ACTION remember
```

### Schema validation errors at startup

The Cedar schema is validated when the brain opens. Common issues:
- Typos in entity type names
- Missing action declarations
- Invalid attribute types

Check `brain/policies/hirn.cedarschema` against the reference schema.

### Policy not taking effect

Policies are loaded from `brain/policies/*.cedar` at startup. If you added a new policy file, restart the application or reload policies.

---

## Further Reading

- [Cedar Policy Language Guide](https://docs.cedarpolicy.com/policies/syntax-policy.html)
- [Cedar Schema Reference](https://docs.cedarpolicy.com/schema/human-readable-schema.html)
- [hirn Architecture — Cedar Authorization](architecture.md#cedar-authorization--audit-trail)
- [docs/cedar-patterns.md](cedar-patterns.md)
- [HirnQL Reference](hirnql-reference.md)
