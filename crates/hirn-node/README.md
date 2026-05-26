# hirn — Node.js Bindings

> **⚠️ Experimental:** This project is under active development. APIs, on-disk formats, and behaviour may change without notice. Not recommended for production use.

Brain-inspired cognitive memory database for LLMs. Native Rust performance via napi-rs.

## Installation

```bash
npm install @hupe1980/hirn
```

## Quick Start

### Level 1: Zero-Config API

```js
const { Memory } = require('@hupe1980/hirn');

// Auto-detects embeddings from OPENAI_API_KEY or OLLAMA_HOST
const mem = Memory.open('./brain.hirn');

try {
  await mem.remember('User prefers dark mode');
  await mem.remember('Last meeting was about Q4 planning');

  const ctx = await mem.think('What are the user\'s preferences?');
  console.log(ctx.context);

  const results = await mem.recall('meetings', { limit: 5 });
  for (const r of results) {
    console.log(`  [${r.similarity.toFixed(2)}] ${r.id}`);
  }
} finally {
  mem.close();
}
```

### Historical Recall

```js
const { Memory } = require('@hupe1980/hirn');

const mem = Memory.open('./brain.hirn');

const current = await mem.recall('lease authority');
const observed = await mem.recall('lease authority', {
  asOf: '2026-02-01T12:00:00Z',
});
const recorded = await mem.recall('lease authority', {
  asOf: '2026-02-02T09:00:00Z',
  snapshotKind: 'recorded',
});
const revision = await mem.recall('lease authority', {
  asOf: '01HW7N0Z5CH9R1R7Z4S4V5Y4QF',
  snapshotKind: 'revision',
});

console.log(current[0].logicalMemoryId);
console.log(revision[0].revisionId);
mem.close();
```

High-level recall defaults to current-state behavior. Set `asOf` to time-travel,
and use `snapshotKind: 'recorded'` or `snapshotKind: 'revision'` when you need
transaction-time or exact-revision boundaries instead of observed-time
snapshots.

### Editing Semantics

High-level semantic editing is available directly on `Memory`.

```js
const { Memory } = require('@hupe1980/hirn');

const mem = Memory.open('./brain.hirn');

await mem.correct('01HXYZ...', {
  description: 'lease authority clarified',
  evidenceCount: 2,
  reason: 'tightened wording',
});
await mem.supersede('01HXYZ...', {
  description: 'lease authority v2',
  reason: 'authoritative cutover',
});
await mem.merge(['01HAAA...', '01HBBB...'], '01HXYZ...', {
  reason: 'deduplicated overlapping claims',
});
await mem.retract('01HXYZ...', {
  reason: 'obsolete',
});

mem.close();
```

  current-state recall to that head while preserving the older revision chain.
  current-state recall but remains visible in history, audit, `INSPECT`, and
  `TRACE`.
  `EXPLAIN`, or destructive paths such as `FORGET ... PURGE`.

### Level 1: ESM Import

```js
import { Memory } from '@hupe1980/hirn';

const mem = Memory.open('./brain.hirn');
// ...
```

The package root intentionally exports the high-level `Memory` API. The
native napi bridge stays internal to the binding.

### TC39 Explicit Resource Management

```js
{
  using mem = Memory.open('./brain.hirn');
  await mem.remember('Auto-cleanup on scope exit');
} // mem.close() called automatically via Symbol.dispose
```

### HirnQL

```js
const mem = Memory.open('./brain.hirn');
await mem.remember('The meeting is at 3pm');

const result = await mem.query('RECALL episodic ABOUT "meeting" LIMIT 5');
console.log(result.type); // "records"
console.log(result.data); // full result as object

await mem.query('THINK ABOUT "schedule" BUDGET 4096');
mem.close();
```

### Rust Tokenizer Selection

```js
const { Memory } = require('@hupe1980/hirn');

const mem = Memory.open('./brain.hirn', { tokenizerName: 'estimating' });
```

`tokenizerName` selects a tokenizer already registered on the Rust side.
It does not route engine token counting through JavaScript, and any low-level
`tokenCount` hints remain client-side conveniences only.

### Batch Operations

```js
const mem = Memory.open('./brain.hirn');

// Single embedding API call for all texts
const ids = await mem.batchRemember([
  'User prefers dark mode',
  'Last meeting was about Q4',
  'Project deadline is next Friday',
]);
console.log(`Stored ${ids.length} memories`);
mem.close();
```

### Watch Events

```js
const mem = Memory.open('./brain.hirn');
const stream = await mem.watch();

await mem.remember('This will emit an event');
const event = await stream.next();
console.log(event.eventType); // "episode_created"

stream.unsubscribe();
mem.close();
```

## Embedding Providers

### Auto-Detection

`Memory.open()` auto-detects the best available provider:

1. `OPENAI_API_KEY` env var → OpenAI
2. `OLLAMA_HOST` env var → Ollama
3. Fallback → FakeEmbeddings (deterministic hash, for testing only)

### Explicit

```js
const { Memory, OpenAIEmbeddings, OllamaEmbeddings, FakeEmbeddings } = require('@hupe1980/hirn');

// OpenAI
const mem = Memory.open('./brain.hirn', {
  embeddings: new OpenAIEmbeddings({
    model: 'text-embedding-3-small', // default
    apiKey: 'sk-...',                // or from OPENAI_API_KEY
  }),
});

// Ollama (local)
const mem2 = Memory.open('./brain.hirn', {
  embeddings: new OllamaEmbeddings({
    model: 'nomic-embed-text',
    host: 'http://localhost:11434',
  }),
});

// Fake (for testing)
const mem3 = Memory.open('./brain.hirn', {
  embeddings: new FakeEmbeddings(64),
});
```

### Custom

```js
/** @implements {import('hirn').EmbeddingFunction} */
class MyEmbeddings {
  get dimensions() { return 768; }

  async embedDocuments(texts) {
    return texts.map(t => this._embed(t));
  }

  async embedQuery(text) {
    return this._embed(text);
  }

  _embed(text) {
    // your implementation
  }
}

const mem = Memory.open('./brain.hirn', { embeddings: new MyEmbeddings() });
```

## Error Handling

```js
try {
  await mem.query('INVALID SYNTAX');
} catch (err) {
  console.error(err.message); // "parse error: ..."
}
```

## TypeScript

Full TypeScript declarations (`.d.ts`) included. All types auto-complete in VS Code.

```ts
import { Memory, Stats, RecallResult, Context, EmbeddingFunction } from '@hupe1980/hirn';
```

## Platform Support

| Platform | Architecture |
|----------|-------------|
| macOS | arm64, x64 |
| Linux | x64 (glibc), arm64 (glibc) |
| Windows | x64 (MSVC) |

## Build from Source

```bash
cd crates/hirn-node
npm run build  # requires Rust toolchain
```

## License

Apache-2.0
