/**
 * Integration tests for the hirn Node.js bindings.
 *
 * Uses Node.js built-in test runner (node:test).
 */

import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory, FakeEmbeddings } = require('../index.js');
const { HirnBridge } = require('../bridge.js');

const DIM = 64;
const TEST_AGENT_ID = 'agent-1';
const ORIGINAL_ABOUT = 'lease authority';
const CURRENT_ABOUT = 'lease authority v2';

function makeEmbedding(seed = 0.1) {
  return new Array(DIM).fill(seed);
}

function estimateTokens(text) {
  return Math.ceil(Buffer.byteLength(text, 'utf8') / 4);
}

async function withDb(fn) {
  const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
  const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
  try {
    await fn(db);
  } finally {
    db.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

async function seedSemanticRevisionHistory(path, embeddings) {
  const mem = Memory.open(path, { agentId: TEST_AGENT_ID, embeddings });
  try {
    const created = await mem.query(`REMEMBER semantic CONTENT "${ORIGINAL_ABOUT}"`);
    assert.equal(created.type, 'created');
    const originalId = created.data.id;

    const originalHistory = await mem.query(`HISTORY "${originalId}"`);
    assert.equal(originalHistory.type, 'history');
    const originalCreatedAt = originalHistory.data.semantic_revision.revisions[0].created_at;
    const cutoverObservedAt = new Date(
      Date.parse(originalCreatedAt) + 2 * 60 * 60 * 1000,
    ).toISOString();

    const superseded = await mem.query(
      `SUPERSEDE "${originalId}" SET description = "${CURRENT_ABOUT}" REASON "cutover" OBSERVED AT "${cutoverObservedAt}"`,
    );
    assert.equal(superseded.type, 'superseded');

    const history = await mem.query(`HISTORY "${originalId}"`);
    assert.equal(history.type, 'history');

    const summary = history.data.semantic_revision;
    const revisions = summary.revisions;
    return {
      logicalMemoryId: summary.logical_memory_id,
      originalRevisionId: revisions[0].revision_id,
      historicalCutoff: revisions[0].created_at,
      recordedCutoff: revisions[revisions.length - 1].created_at,
    };
  } finally {
    mem.close();
  }
}

describe('package surface', () => {
  it('should not export the low-level bridge from the package root', () => {
    const root = require('../index.js');
    assert.equal(root.Hirn, undefined);
    assert.equal(root.HirnBridge, undefined);
    assert.equal(typeof HirnBridge.open, 'function');
  });
});

// ─── Open / Close ────────────────────────────────────────────

describe('HirnBridge open/close', () => {
  it('should open and close', async () => {
    await withDb((db) => {
      assert.ok(db);
    });
  });

  it('should allow double close', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    db.close();
    db.close(); // should not throw
    rmSync(dir, { recursive: true, force: true });
  });

  it('should throw on operations after close', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    db.close();
    assert.throws(() => db.stats(), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });

  it('should accept custom tokenBudget', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM, 2048);
    try {
      const s = db.stats();
      assert.equal(s.totalCount, 0);
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Register Agent ──────────────────────────────────────────

describe('registerAgent', () => {
  it('should register an agent', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
    });
  });

  it('should throw on empty agent id', async () => {
    await withDb(async (db) => {
      await assert.rejects(() => db.registerAgent('', 'Test'), /empty/);
    });
  });
});

// ─── Remember ────────────────────────────────────────────────

describe('remember', () => {
  it('should return a ULID', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const id = await db.remember('agent-1', 'Hello world', makeEmbedding());
      assert.equal(typeof id, 'string');
      assert.equal(id.length, 26);
    });
  });

  it('should store with importance', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const id = await db.remember('agent-1', 'Important!', makeEmbedding(), 0.9);
      assert.equal(id.length, 26);
    });
  });

  it('should store without embedding', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const id = await db.remember('agent-1', 'No embedding');
      assert.equal(id.length, 26);
    });
  });

  it('should store multiple items', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const ids = [];
      for (let i = 0; i < 10; i++) {
        ids.push(await db.remember('agent-1', `Memory ${i}`, makeEmbedding(0.1 * (i + 1))));
      }
      assert.equal(ids.length, 10);
      // All IDs should be unique
      assert.equal(new Set(ids).size, 10);
    });
  });
});

// ─── Recall ──────────────────────────────────────────────────

describe('recall', () => {
  it('should return an array', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const emb = makeEmbedding(0.5);
      await db.remember('agent-1', 'Memory one', emb);
      await db.remember('agent-1', 'Memory two', emb);

      // NullBackend: vector search returns empty; real recall is tested
      // via the Memory class which uses LanceDbBackend.
      const results = await db.recall('agent-1', emb, 5);
      assert.ok(Array.isArray(results));
    });
  });

  it('should return empty on empty db', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const results = await db.recall('agent-1', makeEmbedding(), 5);
      assert.deepEqual(results, []);
    });
  });

  it('should respect limit', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      for (let i = 0; i < 5; i++) {
        await db.remember('agent-1', `Memory ${i}`, makeEmbedding(0.5));
      }
      const results = await db.recall('agent-1', makeEmbedding(0.5), 2);
      assert.ok(results.length <= 2);
    });
  });

  it('should accept threshold parameter', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await db.remember('agent-1', 'Exact match', makeEmbedding(0.3));
      const results = await db.recall('agent-1', makeEmbedding(0.3), 5, 0.99);
      assert.ok(Array.isArray(results));
    });
  });
});

// ─── Think ───────────────────────────────────────────────────

describe('think', () => {
  it('should return context', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const emb = makeEmbedding(0.4);
      await db.remember('agent-1', 'Context memory', emb);

      const ctx = await db.think('agent-1', emb, 4096);
      assert.equal(typeof ctx.context, 'string');
      assert.equal(typeof ctx.tokenCount, 'number');
      assert.equal(typeof ctx.queryTimeMs, 'number');
      assert.ok(Array.isArray(ctx.recordsIncluded));
    });
  });

  it('should use default budget', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await db.remember('agent-1', 'Default budget', makeEmbedding());
      const ctx = await db.think('agent-1', makeEmbedding());
      assert.ok(ctx.context !== undefined);
    });
  });

  it('should honor tokenizerName via the Rust registry', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM, undefined, 'estimating');
    try {
      await db.registerAgent('agent-1', 'Test Agent');
      await db.remember(
        'agent-1',
        'A long context entry that makes the Rust-side estimating tokenizer easy to verify.',
        makeEmbedding(0.4),
      );

      const ctx = await db.think('agent-1', makeEmbedding(0.4), 4096);
      assert.equal(ctx.tokenCount, estimateTokens(ctx.context));
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Forget ──────────────────────────────────────────────────

describe('forget', () => {
  it('should forget a memory', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const id = await db.remember('agent-1', 'To forget', makeEmbedding());
      await db.forget('agent-1', id); // should not throw
    });
  });

  it('should reject on invalid id', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await assert.rejects(() => db.forget('agent-1', 'not-a-valid-ulid'), /invalid/i);
    });
  });
});

// ─── Execute (HirnQL) ───────────────────────────────────────

describe('execute', () => {
  it('should execute RECALL QL', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await db.remember('agent-1', 'QL test memory', makeEmbedding());

      const result = await db.execute('agent-1', 'RECALL episodic ABOUT "test" LIMIT 5');
      assert.equal(typeof result.type, 'string');
      assert.equal(result.type, 'records');
      assert.ok(result.data);
    });
  });

  it('should execute THINK QL', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await db.remember('agent-1', 'Think QL memory', makeEmbedding());

      const result = await db.execute('agent-1', 'THINK ABOUT "test" BUDGET 4096');
      assert.equal(result.type, 'records');
    });
  });

  it('should reject on invalid QL', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await assert.rejects(() => db.execute('agent-1', 'NOT VALID QL'));
    });
  });
});

// ─── Inspect ─────────────────────────────────────────────────

describe('inspect', () => {
  it('should inspect a memory', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const id = await db.remember('agent-1', 'Inspectable', makeEmbedding());
      const result = await db.inspect('agent-1', id);
      assert.equal(result.type, 'inspected');
      assert.equal(result.data.id, id);
    });
  });

  it('should reject on invalid id', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await assert.rejects(() => db.inspect('agent-1', 'bad-id'), /invalid/i);
    });
  });
});

// ─── Trace ───────────────────────────────────────────────────

describe('trace', () => {
  it('should trace a memory', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      const id = await db.remember('agent-1', 'Traceable', makeEmbedding());
      const result = await db.trace('agent-1', id);
      assert.equal(result.type, 'traced');
      assert.equal(result.data.id, id);
      assert.equal(typeof result.data.trust_score, 'number');
    });
  });
});

// ─── Stats ───────────────────────────────────────────────────

describe('stats', () => {
  it('should return stats for empty db', async () => {
    await withDb((db) => {
      const s = db.stats();
      assert.equal(s.totalCount, 0);
      assert.equal(s.episodicCount, 0);
      assert.equal(s.workingCount, 0);
      assert.equal(s.semanticCount, 0);
      assert.ok(s.fileSizeBytes >= 0);
    });
  });

  it('should count after remember', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      await db.remember('agent-1', 'Stats test', makeEmbedding());
      const s = db.stats();
      assert.equal(s.episodicCount, 1);
      assert.ok(s.totalCount >= 1);
    });
  });
});

// ─── Error handling ──────────────────────────────────────────

describe('error handling', () => {
  it('should reject remember after close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    db.close();
    await assert.rejects(() => db.remember('agent-1', 'test'), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });

  it('should reject recall after close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    db.close();
    await assert.rejects(() => db.recall('agent-1', makeEmbedding()), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });

  it('should throw on stats after close', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    db.close();
    assert.throws(() => db.stats(), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });
});

// ─── Watch ───────────────────────────────────────────────────

describe('watch', () => {
  it('should receive created event on remember', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    try {
      await db.registerAgent('agent-1', 'Test Agent');
      const stream = await db.watch();

      // Insert a memory — the event should be emitted
      const id = await db.remember('agent-1', 'Watch me');

      const event = await stream.next();
      assert.ok(event, 'expected an event');
      assert.equal(event.eventType, 'episode_created');
      assert.equal(event.id, id);
      assert.equal(event.layer, 'Episodic');
      assert.ok(event.contentPreview.includes('Watch me'));

      stream.unsubscribe();
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should receive archived event on forget', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    try {
      await db.registerAgent('agent-1', 'Test Agent');
      const id = await db.remember('agent-1', 'To be forgotten', makeEmbedding());

      // Subscribe after remember so we don't get the created event
      const stream = await db.watch();
      await db.forget('agent-1', id);

      const event = await stream.next();
      assert.ok(event, 'expected an event');
      assert.equal(event.eventType, 'archived');
      assert.equal(event.id, id);

      stream.unsubscribe();
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should return null after unsubscribe', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    try {
      const stream = await db.watch();
      stream.unsubscribe();
      const event = await stream.next();
      assert.equal(event, null);
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should filter by layer', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    try {
      await db.registerAgent('agent-1', 'Test Agent');

      // Watch only Episodic layer
      const stream = await db.watch('Episodic');

      await db.remember('agent-1', 'First memory');
      await db.remember('agent-1', 'Second memory');

      const e1 = await stream.next();
      assert.ok(e1);
      assert.equal(e1.eventType, 'episode_created');
      assert.equal(e1.layer, 'Episodic');

      stream.unsubscribe();
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should receive multiple events', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
    const db = HirnBridge.open(join(dir, 'test.hirn'), DIM);
    try {
      await db.registerAgent('agent-1', 'Test Agent');
      const stream = await db.watch();

      const id1 = await db.remember('agent-1', 'Memory one');
      const id2 = await db.remember('agent-1', 'Memory two');

      const e1 = await stream.next();
      const e2 = await stream.next();
      assert.equal(e1.eventType, 'episode_created');
      assert.equal(e1.id, id1);
      assert.equal(e2.eventType, 'episode_created');
      assert.equal(e2.id, id2);

      stream.unsubscribe();
    } finally {
      db.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Memory (zero-config API) ────────────────────────────────

describe('Memory open/close', () => {
  it('should open and close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      assert.ok(mem);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should allow double close', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    mem.close();
    mem.close(); // should not throw
    rmSync(dir, { recursive: true, force: true });
  });

  it('should throw on operations after close', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    mem.close();
    assert.throws(() => mem.stats(), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });

  it('should reject remember after close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    mem.close();
    await assert.rejects(() => mem.remember('test content'), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });

  it('should reject think after close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    mem.close();
    await assert.rejects(() => mem.think('query'), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });

  it('should reject recall after close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    mem.close();
    await assert.rejects(() => mem.recall('query'), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });

  it('should reject query after close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    mem.close();
    await assert.rejects(() => mem.query('CONSOLIDATE'), /closed/);
    rmSync(dir, { recursive: true, force: true });
  });
});

describe('Memory remember', () => {
  it('should return a ULID string', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const id = await mem.remember('The capital of France is Paris and it is known for the Eiffel Tower');
      assert.equal(typeof id, 'string');
      assert.equal(id.length, 26);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should increment episodic count', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await mem.remember('Machine learning is a subset of artificial intelligence that enables systems to learn from data');
      await mem.remember('The Great Wall of China is one of the most famous landmarks in the world spanning thousands of miles');
      const stats = mem.stats();
      assert.ok(stats.episodicCount >= 2);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('Memory think', () => {
  it('should return context', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await mem.remember('Photosynthesis is the process by which green plants convert sunlight into chemical energy using chlorophyll');
      const ctx = await mem.think('How do plants make food from sunlight?');
      assert.equal(typeof ctx.context, 'string');
      assert.equal(typeof ctx.tokenCount, 'number');
      assert.ok(Array.isArray(ctx.recordsIncluded));
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should respect budget parameter', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await mem.remember('The periodic table organizes chemical elements by their atomic number and chemical properties');
      const ctx = await mem.think('chemistry elements', { budget: 2048 });
      assert.ok(ctx.tokenCount <= 2048);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('Memory recall', () => {
  it('should return recall results', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await mem.remember('Quantum computing uses quantum-mechanical phenomena such as superposition and entanglement to process data');
      const results = await mem.recall('quantum computers');
      assert.ok(Array.isArray(results));
      if (results.length > 0) {
        assert.equal(typeof results[0].id, 'string');
        assert.equal(typeof results[0].similarity, 'number');
        assert.equal(typeof results[0].layer, 'string');
      }
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should respect limit parameter', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await mem.remember('The solar system has eight planets orbiting around the Sun in elliptical paths');
      await mem.remember('Jupiter is the largest planet in our solar system with a mass more than twice all other planets combined');
      await mem.remember('Mars is often called the Red Planet due to its reddish appearance caused by iron oxide on its surface');
      const results = await mem.recall('planets', { limit: 1 });
      assert.ok(results.length <= 1);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should support historical snapshots and revision metadata', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const path = join(dir, 'brain.hirn');
    const embeddings = new FakeEmbeddings(DIM);
    const seeded = await seedSemanticRevisionHistory(path, embeddings);
    const mem = Memory.open(path, { agentId: TEST_AGENT_ID, embeddings });
    try {
      const current = await mem.recall(CURRENT_ABOUT, {
        limit: 10,
        threshold: 0,
      });
      assert.equal(current.length, 1);
      assert.equal(current[0].logicalMemoryId, seeded.logicalMemoryId);
      assert.equal(current[0].revisionState, 'Active');
      assert.notEqual(current[0].revisionId, seeded.originalRevisionId);

      const historical = await mem.recall(ORIGINAL_ABOUT, {
        limit: 10,
        asOf: seeded.historicalCutoff,
      });
      assert.equal(historical.length, 1);
      assert.equal(historical[0].revisionId, seeded.originalRevisionId);
      assert.equal(historical[0].revisionState, 'Active');

      const recorded = await mem.recall(CURRENT_ABOUT, {
        limit: 10,
        asOf: seeded.recordedCutoff,
        snapshotKind: 'recorded',
      });
      assert.equal(recorded.length, 1);
      assert.notEqual(recorded[0].revisionId, seeded.originalRevisionId);

      const revision = await mem.recall(ORIGINAL_ABOUT, {
        limit: 10,
        asOf: seeded.originalRevisionId,
        snapshotKind: 'revision',
      });
      assert.equal(revision.length, 1);
      assert.equal(revision[0].revisionId, seeded.originalRevisionId);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('Memory query (HirnQL)', () => {
  it('should execute a RECALL query', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await mem.remember('TypeScript is a strongly typed programming language that builds on JavaScript adding static type checking');
      const result = await mem.query('RECALL episodic ABOUT "programming languages" LIMIT 5');
      assert.equal(result.type, 'records');
      assert.ok(result.data);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should execute a CONSOLIDATE query', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const result = await mem.query('CONSOLIDATE');
      assert.equal(result.type, 'consolidated');
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should reject invalid HirnQL', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await assert.rejects(() => mem.query('NOT VALID QUERY'), /parse error/);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('Memory semantic edit helpers', () => {
  it('should cover correct, supersede, and retract', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: TEST_AGENT_ID });
    try {
      const createdTarget = await mem.query('REMEMBER semantic CONTENT "lease authority"');

      const targetId = createdTarget.data.id;

      const corrected = await mem.correct(targetId, {
        description: 'canonical lease authority clarified',
        confidence: 0.7,
        evidenceCount: 2,
        reason: 'clarified wording',
      });
      assert.equal(corrected.type, 'corrected');

      const superseded = await mem.supersede(targetId, {
        description: 'canonical lease authority v2',
        reason: 'authoritative cutover',
      });
      assert.equal(superseded.type, 'superseded');

      const retracted = await mem.retract(targetId, {
        reason: 'obsolete',
      });
      assert.equal(retracted.type, 'retracted');
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should build merge HirnQL and forward agent options', async () => {
    const mem = Object.create(Memory.prototype);
    let capturedQuery = null;
    let capturedOptions = null;
    const sentinel = { type: 'merged', data: { ok: true } };

    mem.query = async (hirnql, options = {}) => {
      capturedQuery = hirnql;
      capturedOptions = options;
      return sentinel;
    };

    const result = await mem.merge(['01HSRCA', '01HSRCB'], '01HTARGET', {
      description: 'canonical lease authority',
      confidence: 0.95,
      evidenceCount: 3,
      reason: 'deduplicate agents',
      observedAt: '2026-03-01T00:00:00Z',
      causedBy: '01HCAUSE',
      agentId: 'agent-merge',
    });

    assert.equal(result, sentinel);
    assert.equal(
      capturedQuery,
      'MERGE MEMORY "01HSRCA", "01HSRCB" INTO "01HTARGET" ' +
        'SET description = "canonical lease authority", confidence = 0.95, ' +
        'evidence_count = 3 REASON "deduplicate agents" ' +
        'OBSERVED AT "2026-03-01T00:00:00Z" CAUSED BY "01HCAUSE"',
    );
    assert.deepEqual(capturedOptions, { agentId: 'agent-merge' });
  });

  it('should reject invalid helper arguments before executing HirnQL', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: TEST_AGENT_ID });
    try {
      await assert.rejects(
        () => mem.correct('01HXYZ'),
        /at least one semantic update field/
      );
      await assert.rejects(
        () => mem.merge([], '01HXYZ'),
        /sourceIds must contain at least one memory ID/
      );
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('Memory stats', () => {
  it('should return statistics', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const stats = mem.stats();
      assert.equal(typeof stats.workingCount, 'number');
      assert.equal(typeof stats.episodicCount, 'number');
      assert.equal(typeof stats.semanticCount, 'number');
      assert.equal(typeof stats.totalCount, 'number');
      assert.equal(typeof stats.fileSizeBytes, 'number');
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Memory agentId ──────────────────────────────────────────

describe('Memory agentId', () => {
  it('should open with agentId', () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: 'agent-alpha' });
    try {
      assert.ok(mem);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should remember with constructor agentId (Cedar path)', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: 'agent-alpha' });
    try {
      const id = await mem.remember('Memory with agent context for authorization testing purposes');
      assert.equal(typeof id, 'string');
      assert.equal(id.length, 26);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should remember with per-call agentId override', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const id = await mem.remember('Memory with per-call agent override for authorization', { agentId: 'agent-beta' });
      assert.equal(typeof id, 'string');
      assert.equal(id.length, 26);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should think with agentId', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: 'agent-alpha' });
    try {
      await mem.remember('Photosynthesis converts sunlight into chemical energy in plants using chlorophyll');
      const ctx = await mem.think('How do plants make energy from sunlight?', { agentId: 'agent-alpha' });
      assert.equal(typeof ctx.context, 'string');
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should recall with agentId', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: 'agent-alpha' });
    try {
      await mem.remember('The solar system has eight planets orbiting around the Sun in elliptical paths');
      const results = await mem.recall('planets', { agentId: 'agent-alpha' });
      assert.ok(Array.isArray(results));
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should query with agentId', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: 'agent-alpha' });
    try {
      await mem.remember('TypeScript is a strongly typed programming language that builds on JavaScript');
      const result = await mem.query('RECALL episodic ABOUT "programming" LIMIT 5', { agentId: 'agent-alpha' });
      assert.equal(result.type, 'records');
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should default to anonymous when no agentId', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      // No agentId → uses default "anonymous" → delegates to HirnMemory.remember
      const id = await mem.remember('Memory without any agent context uses default anonymous path');
      assert.equal(id.length, 26);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Memory batchRemember ────────────────────────────────────

describe('Memory batchRemember', () => {
  it('should store multiple memories in one call', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const ids = await mem.batchRemember([
        'Alpha particles consist of two protons and two neutrons bound together',
        'Beta decay converts a neutron into a proton electron and antineutrino',
        'Gamma rays are high-energy electromagnetic radiation from nuclear transitions',
      ]);
      assert.equal(ids.length, 3);
      for (const id of ids) {
        assert.equal(typeof id, 'string');
        assert.equal(id.length, 26);
      }
      // All IDs should be unique
      assert.equal(new Set(ids).size, 3);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should return empty array for empty input', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const ids = await mem.batchRemember([]);
      assert.deepEqual(ids, []);
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should reject non-string elements', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await assert.rejects(
        () => mem.batchRemember([42]),
        /must be a string/
      );
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should reject empty strings', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      await assert.rejects(
        () => mem.batchRemember(['valid', '   ']),
        /empty or whitespace/
      );
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Memory forget ───────────────────────────────────────────

describe('Memory forget', () => {
  it('should forget a remembered memory', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const id = await mem.remember('Temporary fact to be forgotten soon after creation');
      await mem.forget(id); // should not throw
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should forward agentId option', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'), { agentId: 'agent-f' });
    try {
      const id = await mem.remember('Fact to forget with explicit agent identifier');
      await mem.forget(id, { agentId: 'agent-f' }); // should not throw
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Memory watch ────────────────────────────────────────────

describe('Memory watch', () => {
  it('should receive events for remembered memories', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const stream = await mem.watch();
      assert.ok(stream);
      await mem.remember('Watched memory event for real-time monitoring test');

      // The stream should eventually yield an event
      const event = await stream.next();
      assert.ok(event);
      assert.equal(event.eventType, 'episode_created');

      stream.unsubscribe();
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should support layer filter', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      const stream = await mem.watch({ filterLayer: 'Episodic' });
      assert.ok(stream);
      stream.unsubscribe();
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// ─── Error types ─────────────────────────────────────────────

describe('error types', () => {
  it('should export HirnError, NotFoundError, QueryError', async () => {
    const { HirnError, NotFoundError, QueryError } = await import('../index.mjs');
    assert.ok(HirnError);
    assert.ok(NotFoundError);
    assert.ok(QueryError);

    const base = new HirnError('test');
    assert.ok(base instanceof Error);
    assert.equal(base.name, 'HirnError');

    const nf = new NotFoundError('not found');
    assert.ok(nf instanceof HirnError);
    assert.equal(nf.name, 'NotFoundError');

    const qe = new QueryError('bad query');
    assert.ok(qe instanceof HirnError);
    assert.equal(qe.name, 'QueryError');
  });
});

// ─── Concurrent operations ───────────────────────────────────

describe('concurrent operations', () => {
  it('should handle 50 concurrent Memory.remember operations', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
    const mem = Memory.open(join(dir, 'brain.hirn'));
    try {
      // Seed a first entry to initialize tables
      await mem.remember('Initialization seed entry for the concurrent operations test database');

      // Use diverse topics to avoid admission novelty rejection
      const topics = [
        'quantum computing uses qubits for exponential parallelism',
        'photosynthesis converts carbon dioxide and water into glucose',
        'neural networks are inspired by biological brain architecture',
        'blockchain provides decentralized immutable transaction ledgers',
        'CRISPR enables targeted gene editing in living organisms',
        'machine learning algorithms improve through data-driven training',
        'superconductors conduct electricity with zero resistance below critical temperature',
        'reinforcement learning agents maximize cumulative reward signals',
        'dark matter accounts for approximately 27 percent of the universe',
        'RNA interference silences gene expression post-transcriptionally',
        'gravitational waves were first detected by LIGO in 2015',
        'protein folding determines biological function from amino acid sequences',
        'cryptographic hash functions are one-way and collision resistant',
        'mitochondria are the powerhouses of eukaryotic cells',
        'transformer architecture revolutionized natural language processing',
        'plate tectonics describes the movement of lithospheric plates',
        'epigenetic modifications alter gene expression without changing DNA',
        'distributed systems achieve consensus through protocols like Raft',
        'antibodies are Y-shaped proteins that neutralize pathogens',
        'topology studies properties preserved under continuous deformations',
        'CMOS transistors form the basis of modern integrated circuits',
        'telomeres protect chromosome ends from degradation during replication',
        'convolutional neural networks excel at image recognition tasks',
        'tectonic subduction zones create deep ocean trenches and volcanoes',
        'mRNA vaccines instruct cells to produce spike protein antigens',
        'Fourier transforms decompose signals into component frequencies',
        'stem cells can differentiate into specialized cell types',
        'zero-knowledge proofs verify statements without revealing information',
        'neurotransmitters transmit signals across synaptic cleft gaps',
        'category theory provides abstract frameworks for mathematical structures',
        'enzyme catalysis accelerates biochemical reactions by lowering activation energy',
        'homomorphic encryption allows computation on encrypted data',
        'ribosomes translate messenger RNA into polypeptide chains',
        'Bayesian inference updates probability estimates from new evidence',
        'ion channels regulate electrical signaling in excitable cells',
        'formal verification proves program correctness mathematically',
        'DNA polymerase synthesizes new strands during genome replication',
        'adversarial examples exploit neural network vulnerabilities',
        'chemiosmosis couples electron transport to ATP synthesis',
        'type theory provides foundations for programming language design',
        'histone modifications regulate chromatin structure and accessibility',
        'federated learning trains models across decentralized data sources',
        'signal transduction pathways relay extracellular messages intracellularly',
        'lambda calculus is a formal system for function abstraction',
        'cytokines coordinate immune system inflammatory responses',
        'graph neural networks process non-Euclidean structured data',
        'lipid bilayers form selectively permeable biological membranes',
        'consensus algorithms ensure agreement in distributed fault-tolerant systems',
        'autophagy degrades and recycles damaged cellular components',
        'differential privacy adds calibrated noise for statistical disclosure control',
      ];

      // Run in batches of 10
      const allIds = [];
      for (let batch = 0; batch < 5; batch++) {
        const promises = [];
        for (let i = 0; i < 10; i++) {
          const idx = batch * 10 + i;
          promises.push(mem.remember(topics[idx]));
        }
        const ids = await Promise.all(promises);
        allIds.push(...ids);
      }
      assert.equal(allIds.length, 50);
      // All IDs should be unique ULIDs
      assert.equal(new Set(allIds).size, 50);
      for (const id of allIds) {
        assert.equal(typeof id, 'string');
        assert.equal(id.length, 26);
      }
    } finally {
      mem.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('should handle 50 concurrent HirnBridge.remember operations', async () => {
    await withDb(async (db) => {
      await db.registerAgent('agent-1', 'Test Agent');
      // Seed entry
      await db.remember('agent-1', 'Seed entry for initialization', makeEmbedding());

      // Run in batches of 10 with distinct embeddings
      const allIds = [];
      for (let batch = 0; batch < 5; batch++) {
        const promises = [];
        for (let i = 0; i < 10; i++) {
          const idx = batch * 10 + i;
          promises.push(db.remember('agent-1', `Concurrent L2 memory ${idx}`, makeEmbedding(0.01 * (idx + 1))));
        }
        const ids = await Promise.all(promises);
        allIds.push(...ids);
      }
      assert.equal(allIds.length, 50);
      assert.equal(new Set(allIds).size, 50);
    });
  });
});

// ─── TypeScript type check ───────────────────────────────────

describe('TypeScript types', () => {
  it('should compile a TypeScript test file with no errors', async () => {
    const { execSync } = await import('node:child_process');
    const { writeFileSync } = await import('node:fs');

    const tsFile = join(tmpdir(), 'hirn-type-check.ts');
    const tsContent = `
  import { Memory, Stats, RecallResult, RecallSnapshotKind, Context, WatchEvent, QueryResult, WatchStream, EmbeddingFunction, FakeEmbeddings, OpenAIEmbeddings, OllamaEmbeddings, detectEmbeddings, HirnError, NotFoundError, QueryError } from '../index';

async function main() {
  // Embeddings API
  const fake: EmbeddingFunction = new FakeEmbeddings(128);
  const fakeVec: number[][] = await fake.embedDocuments(['hello']);
  const fakeQ: number[] = await fake.embedQuery('hello');
  const detected: EmbeddingFunction | null = detectEmbeddings();
  const snapshotKind: RecallSnapshotKind = 'revision';

  // Memory (L1) API with options
  const mem: Memory = Memory.open('/tmp/brain.hirn', { agentId: 'agent-1', embeddings: fake });
  const memId: string = await mem.remember('content', { agentId: 'agent-override' });
  const memCtx: Context = await mem.think('query', { budget: 2048, agentId: 'agent-1' });
  const memResults: RecallResult[] = await mem.recall('query', { limit: 10, threshold: 0.25, agentId: 'agent-1', asOf: '2026-01-01T00:00:00Z', snapshotKind });
  if (memResults[0]) {
    const logicalMemoryId: string | null | undefined = memResults[0].logicalMemoryId;
    const revisionId: string | null | undefined = memResults[0].revisionId;
    const revisionState: string | null | undefined = memResults[0].revisionState;
  }
  const memQr: QueryResult = await mem.query('CONSOLIDATE', { agentId: 'agent-1' });
  const corrected: QueryResult = await mem.correct('01HXYZ', { description: 'updated', agentId: 'agent-1' });
  const superseded: QueryResult = await mem.supersede('01HXYZ', { description: 'updated v2' });
  const merged: QueryResult = await mem.merge(['01HAAA', '01HBBB'], '01HXYZ', { reason: 'dedupe' });
  const retracted: QueryResult = await mem.retract('01HXYZ', { reason: 'obsolete' });
  const memStats: Stats = mem.stats();
  mem.close();

  // Memory without options (backward compat)
  const mem2: Memory = Memory.open('/tmp/brain2.hirn');
  await mem2.remember('no agent');
  await mem2.think('query');
  await mem2.recall('query');
  await mem2.query('CONSOLIDATE');
  const batchIds: string[] = await mem2.batchRemember(['a', 'b', 'c']);
  await mem2.forget(batchIds[0]);
  const memWatch: WatchStream = await mem2.watch();
  mem2.close();

  // Error types
  const err: HirnError = new HirnError('test');
  const nf: NotFoundError = new NotFoundError('not found');
  const qe: QueryError = new QueryError('bad query');
  if (err instanceof Error) {}
  if (nf instanceof HirnError) {}
  if (qe instanceof HirnError) {}
}
`;
    writeFileSync(tsFile, tsContent);

    try {
      // Check if tsc is available
      try {
        execSync('npx tsc --version', { stdio: 'pipe' });
      } catch {
        // tsc not available, skip
        return;
      }

      const cwd = join(import.meta.url.replace('file://', ''), '..', '..');
      execSync(`npx tsc --noEmit --strict --moduleResolution node --module commonjs --target es2020 ${tsFile}`, {
        cwd: cwd.startsWith('/') ? cwd : `/${cwd}`,
        stdio: 'pipe',
      });
    } finally {
      try { rmSync(tsFile); } catch { /* ignore */ }
    }
  });
});
