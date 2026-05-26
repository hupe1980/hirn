# hirn — Python Bindings

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Brain-inspired cognitive memory database for LLMs. Native Rust performance via PyO3.

## Installation

```bash
pip install hirn

# With embedding providers:
pip install hirn[openai]          # OpenAI embeddings
pip install hirn[ollama]          # Ollama (local) embeddings
pip install hirn[sentence-transformers]  # Sentence Transformers (local)
pip install hirn[all]             # All providers
```

## Quick Start

### Level 1: Zero-Config API

```python
from hirn import Memory

# Auto-detects embeddings from OPENAI_API_KEY, OLLAMA_HOST, or sentence-transformers
mem = Memory.open("./brain")

mem.remember("User prefers dark mode")
mem.remember("Last meeting was about Q4 planning")

ctx = mem.think("What are the user's preferences?", budget=2048)
print(ctx.context)

results = mem.recall("meetings", limit=5)
for r in results:
    print(f"  [{r.similarity:.2f}] {r.id}")

mem.close()
```

### Historical Recall

```python
from hirn import Memory

mem = Memory.open("./brain")

current = mem.recall("lease authority")
observed = mem.recall("lease authority", as_of="2026-02-01T12:00:00Z")
recorded = mem.recall(
    "lease authority",
    as_of="2026-02-02T09:00:00Z",
    snapshot_kind="recorded",
)
revision = mem.recall(
    "lease authority",
    as_of="01HW7N0Z5CH9R1R7Z4S4V5Y4QF",
    snapshot_kind="revision",
)

print(current[0].logical_memory_id)
print(revision[0].revision_id)
mem.close()
```

High-level recall defaults to current-state behavior. Set `as_of` to time-travel,
and use `snapshot_kind="recorded"` or `snapshot_kind="revision"` when you
need transaction-time or exact-revision boundaries instead of observed-time
snapshots.

### Editing Semantics

High-level editing now ships as dedicated `Memory` and `AsyncMemory` methods.
They wrap the same revision-native HirnQL operations while keeping raw
`query(...)` available when you need exact clause control.

```python
from hirn import Memory

mem = Memory.open("./brain")

mem.correct(
    "01HXYZ...",
    description='lease authority "v1.1"',
    reason="editorial cleanup",
)
mem.supersede(
    "01HXYZ...",
    description="lease authority v2",
    reason="authoritative cutover",
)
mem.merge(
    ["01HSRC..."],
    "01HTARGET...",
    description="canonical lease authority",
    reason="deduplicate agents",
)
mem.retract("01HXYZ...", reason="obsolete")

# Raw HirnQL is still available for advanced clauses such as explicit
# namespace control or unsupported statements.
mem.query('FORGET "01HXYZ..." PURGE')

mem.close()
```

- `correct()`, `supersede()`, and `merge()` expose the semantic `SET` fields
  `description`, `confidence`, and `evidence_count`, plus optional
  `reason`, `observed_at`, and `caused_by` metadata.
- `SUPERSEDE` appends a new authoritative semantic revision and moves
    current-state recall to that head while preserving the older revision chain.
- `RETRACT` appends a tombstone revision so the claim disappears from default
    current-state recall but remains visible in history, audit, `INSPECT`, and
    `TRACE`.
- `FORGET ... PURGE` is the destructive removal path. Use it only when you
    truly want the memory removed from active storage rather than corrected or
    retired.

### Rust Tokenizer Selection

```python
from hirn import Memory

# Select a tokenizer already registered on the Rust side.
mem = Memory.open("./brain", tokenizer_name="estimating")
```

`tokenizer_name` is a Rust-side registry hint. It does not route token
counting through Python, and any low-level `token_count` hints remain
client-side conveniences only.

### Level 1: Async API

```python
import asyncio
from hirn import AsyncMemory

async def main():
    mem = await AsyncMemory.open("./brain")
    await mem.remember("User prefers dark mode")
    ctx = await mem.think("preferences?", budget=2048)
    print(ctx.context)
    await mem.close()

asyncio.run(main())
```

The package root intentionally exports the high-level `Memory` and
`AsyncMemory` APIs. The native PyO3 bridge stays internal to the binding.

### HirnQL

```python
mem = Memory.open("./brain")
mem.remember("The meeting is at 3pm")

result = mem.query('RECALL episodic ABOUT "meeting" LIMIT 5')
print(result.type)  # "records"
print(result.json)  # full result as dict

result = mem.query('THINK ABOUT "schedule" BUDGET 4096')
result = mem.query('EXPLAIN RECALL episodic ABOUT "meeting"')
mem.close()
```

### Batch Operations

```python
mem = Memory.open("./brain")

# Single embedding API call for all texts
ids = mem.batch_remember([
    "User prefers dark mode",
    "Last meeting was about Q4",
    "Project deadline is next Friday",
])
print(f"Stored {len(ids)} memories")
mem.close()
```

## Embedding Providers

### Auto-Detection

`Memory.open()` auto-detects the best available provider:

1. `OPENAI_API_KEY` env var → OpenAI
2. `OLLAMA_HOST` env var → Ollama
3. `sentence_transformers` importable → SentenceTransformers
4. Fallback → FakeEmbeddings (deterministic hash, for testing only)

### Explicit

```python
from hirn import Memory
from hirn.embeddings.openai import OpenAIEmbeddings
from hirn.embeddings.ollama import OllamaEmbeddings
from hirn.embeddings.sentence_transformers import SentenceTransformerEmbeddings

# OpenAI
mem = Memory.open("./brain", embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",  # default
    api_key="sk-...",                # or from OPENAI_API_KEY
))

# Ollama (local)
mem = Memory.open("./brain", embeddings=OllamaEmbeddings(
    model="nomic-embed-text",
    host="http://localhost:11434",
))

# Sentence Transformers (local, no API key needed)
mem = Memory.open("./brain", embeddings=SentenceTransformerEmbeddings(
    model="all-MiniLM-L6-v2",
    device="cpu",  # or "cuda", "mps"
))
```

### Custom

```python
from hirn.embeddings import EmbeddingFunction

class MyEmbeddings:
    @property
    def dimensions(self) -> int:
        return 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        ...  # your implementation

mem = Memory.open("./brain", embeddings=MyEmbeddings())
```

## Error Handling

```python
from hirn import HirnError, NotFoundError, QueryError

try:
    mem.query("INVALID SYNTAX")
except QueryError as e:
    print(f"Query failed: {e}")
except NotFoundError as e:
    print(f"Not found: {e}")
except HirnError as e:
    print(f"Hirn error: {e}")
```

## Type Support

Full type stubs (`.pyi`) and `py.typed` marker included for IDE autocompletion and mypy/pyright checking.

## Build from Source

```bash
pip install maturin
cd crates/hirn-python
maturin develop
```

## License

Apache-2.0
