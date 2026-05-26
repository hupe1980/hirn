# Cedar Policy Patterns

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

This is the operator-facing pattern library for hirn's major Cedar action types. Use it alongside the broader [docs/cedar-guide.md](cedar-guide.md).

## Writer With No Destructive Powers

Use this for application agents that should store and retrieve memory but should not delete or administer the system.

```cedar
permit(
    principal in Hirn::Team::"writers",
    action in [
        Hirn::Action::"remember",
        Hirn::Action::"recall",
        Hirn::Action::"think",
        Hirn::Action::"connect",
        Hirn::Action::"watch"
    ],
    resource in Hirn::Realm::"production"
);

forbid(
    principal in Hirn::Team::"writers",
    action in [Hirn::Action::"forget", Hirn::Action::"admin", Hirn::Action::"consolidate"],
    resource
);
```

## Read-Only Auditor

Use this for observability, audit, or support agents that must inspect memory state without mutating it.

```cedar
permit(
    principal in Hirn::Team::"auditors",
    action in [Hirn::Action::"recall", Hirn::Action::"think", Hirn::Action::"watch"],
    resource
);

forbid(
    principal in Hirn::Team::"auditors",
    action in [Hirn::Action::"remember", Hirn::Action::"connect", Hirn::Action::"forget"],
    resource
);
```

## Admin-Only Destructive Operations

Use this when ordinary writers must never archive, purge, or force consolidation.

```cedar
permit(
    principal in Hirn::Team::"admins",
    action in [
        Hirn::Action::"retract",
        Hirn::Action::"purge",
        Hirn::Action::"forget",
        Hirn::Action::"consolidate",
        Hirn::Action::"admin"
    ],
    resource
);

forbid(
    principal,
    action in [
        Hirn::Action::"retract",
        Hirn::Action::"purge",
        Hirn::Action::"forget",
        Hirn::Action::"consolidate",
        Hirn::Action::"admin"
    ],
    resource
) unless { principal in Hirn::Team::"admins" };
```

## Tenant Isolation By Realm

Use this as the base pattern for multi-tenant deployments. Keep every tenant inside its own realm and deny cross-tenant access explicitly.

```cedar
permit(
    principal in Hirn::Organization::"tenant-a",
    action,
    resource in Hirn::Realm::"tenant-a"
);

permit(
    principal in Hirn::Organization::"tenant-b",
    action,
    resource in Hirn::Realm::"tenant-b"
);

forbid(
    principal in Hirn::Organization::"tenant-a",
    action,
    resource in Hirn::Realm::"tenant-b"
);
```

## Shared Knowledge Namespace

Use this when each tenant keeps private namespaces but a shared namespace is available for common knowledge.

```cedar
permit(
    principal,
    action in [Hirn::Action::"recall", Hirn::Action::"think"],
    resource == Hirn::Namespace::"shared"
);

permit(
    principal in Hirn::Team::"curators",
    action in [Hirn::Action::"remember", Hirn::Action::"connect"],
    resource == Hirn::Namespace::"shared"
);
```

## Guard `recall_raw_text`

`recall_raw_text` is more sensitive than ordinary recall because it returns raw memory content. Keep it behind a smaller principal set.

```cedar
permit(
    principal in Hirn::Team::"incident-response",
    action in [Hirn::Action::"recall_raw_text", Hirn::Action::"recall"],
    resource in Hirn::Realm::"production"
);

forbid(
    principal,
    action == Hirn::Action::"recall_raw_text",
    resource
) unless { principal in Hirn::Team::"incident-response" };
```

## Separate `execute` From Direct Writes

`execute` can cover mutating HirnQL, so do not grant it by default just because an agent can `recall`.

```cedar
permit(
    principal in Hirn::Team::"query-users",
    action in [Hirn::Action::"execute", Hirn::Action::"recall", Hirn::Action::"think"],
    resource in Hirn::Realm::"analytics"
);

forbid(
    principal in Hirn::Team::"query-users",
    action == Hirn::Action::"execute",
    resource in Hirn::Realm::"production"
);
```

Use this pattern only when the query surface itself is intended to be available. If an agent only needs standard API calls, grant those actions directly instead.

## Reputation-Gated Writes

Use this when new or suspicious agents should be allowed to read but not mutate memory.

```cedar
permit(
    principal,
    action in [Hirn::Action::"recall", Hirn::Action::"think"],
    resource
) when { principal is Hirn::Agent && principal.reputation >= 0 };

permit(
    principal,
    action in [Hirn::Action::"remember", Hirn::Action::"connect"],
    resource
) when { principal is Hirn::Agent && principal.reputation >= 50 };

forbid(
    principal,
    action in [Hirn::Action::"remember", Hirn::Action::"connect"],
    resource
) when { principal is Hirn::Agent && principal.reputation < 50 };
```

## Production Checklist

Before shipping a policy set, verify:

1. destructive actions are explicitly narrowed
2. `execute` and `recall_raw_text` are not granted accidentally
3. realm isolation is expressed with both permits and cross-tenant forbids where needed
4. shared namespaces are intentional and separately documented
5. `EXPLAIN POLICY` is part of your operational runbook

## Related Docs

- [docs/cedar-guide.md](cedar-guide.md)
- [docs/troubleshooting.md](troubleshooting.md)