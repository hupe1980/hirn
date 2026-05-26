# Documentation Map

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

hirn now has enough depth that a flat list of documents is no longer the best entry point. Use this page when you know your goal but do not yet know which guide answers it fastest.

## Fast Routes

| I need to... | Start here | Then read |
|--------------|------------|-----------|
| build the first memory workflow | [getting-started.md](getting-started.md) | [hirnql-reference.md](hirnql-reference.md), [explanation-surfaces.md](explanation-surfaces.md) |
| understand how the engine is put together | [architecture.md](architecture.md) | [write-path.md](write-path.md), [write-guarantees.md](write-guarantees.md), [causal.md](causal.md), [glossary.md](glossary.md) |
| run heavy reasoning safely | [offline-intelligence.md](offline-intelligence.md) | [observability.md](observability.md), [benchmarks.md](benchmarks.md), [security.md](security.md) |
| explain why a result or write behaved a certain way | [explanation-surfaces.md](explanation-surfaces.md) | [observability.md](observability.md), [troubleshooting.md](troubleshooting.md) |
| operate hirn in production | [deployment.md](deployment.md) | [observability.md](observability.md), [troubleshooting.md](troubleshooting.md), [performance-tuning.md](performance-tuning.md) |
| debug Cedar or access-control behavior | [cedar-guide.md](cedar-guide.md) | [cedar-patterns.md](cedar-patterns.md), [security.md](security.md), [troubleshooting.md](troubleshooting.md) |
| work with resources and grounded evidence | [getting-started.md](getting-started.md) | [architecture.md](architecture.md), [explanation-surfaces.md](explanation-surfaces.md) |

## By Audience

### Application Builders

- Start with [getting-started.md](getting-started.md).
- Use [hirnql-reference.md](hirnql-reference.md) when you need the language surface.
- Use [explanation-surfaces.md](explanation-surfaces.md) when your UI or evaluator needs auditable reasoning.
- Use [agent-tools.md](agent-tools.md) if you are integrating tool-facing workflows or MCP surfaces.

### Platform And Operations Teams

- Start with [deployment.md](deployment.md) for topology and deployment mode.
- Use [observability.md](observability.md) for metrics, events, diagnostics, and alerting.
- Use [troubleshooting.md](troubleshooting.md) when the system is returning an error, stalling, or degrading.
- Use [performance-tuning.md](performance-tuning.md) once the system is healthy but not yet efficient enough.

### Security And Policy Owners

- Start with [security.md](security.md) for the defense model.
- Use [cedar-guide.md](cedar-guide.md) for Cedar basics and policy shape.
- Use [cedar-patterns.md](cedar-patterns.md) for operator-ready allow/deny patterns.
- Use [encryption-at-rest.md](encryption-at-rest.md) for storage-encryption posture.

## Concept Guides vs Reference Guides

These docs are meant to be read differently:

- Concept guides explain why the system is built this way: [architecture.md](architecture.md), [offline-intelligence.md](offline-intelligence.md), [causal.md](causal.md), [security.md](security.md).
- Operator guides explain how to run or debug it: [deployment.md](deployment.md), [observability.md](observability.md), [troubleshooting.md](troubleshooting.md), [performance-tuning.md](performance-tuning.md).
- Reference guides explain exact surfaces: [hirnql-reference.md](hirnql-reference.md), [cedar-guide.md](cedar-guide.md), [write-guarantees.md](write-guarantees.md), [glossary.md](glossary.md).

## Suggested Reading Sequences

### New To hirn

1. [getting-started.md](getting-started.md)
2. [documentation-map.md](documentation-map.md)
3. [architecture.md](architecture.md)

### Shipping A Production Deployment

1. [deployment.md](deployment.md)
2. [observability.md](observability.md)
3. [troubleshooting.md](troubleshooting.md)
4. [performance-tuning.md](performance-tuning.md)
5. [security.md](security.md)

### Adding Offline Cognition

1. [offline-intelligence.md](offline-intelligence.md)
2. [explanation-surfaces.md](explanation-surfaces.md)
3. [observability.md](observability.md)
4. [benchmarks.md](benchmarks.md)

### Working With Policy And Multi-Agent Isolation

1. [cedar-guide.md](cedar-guide.md)
2. [cedar-patterns.md](cedar-patterns.md)
3. [troubleshooting.md](troubleshooting.md)
4. [security.md](security.md)