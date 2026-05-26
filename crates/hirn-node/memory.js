// @ts-check
'use strict';

const { detectEmbeddings, FakeEmbeddings } = require('./embeddings');
const { wrapNativeError } = require('./errors');

function requireNonEmptyString(value, name) {
  if (typeof value !== 'string') {
    throw new TypeError(`${name} must be a string, got ${typeof value}`);
  }
  if (!value.trim()) {
    throw new Error(`${name} must not be empty or whitespace-only`);
  }
  return value;
}

function quoteHirnqlString(value) {
  const escaped = value
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\t/g, '\\t')
    .replace(/\r/g, '\\r');
  return `"${escaped}"`;
}

function appendOptionalStringClause(parts, clause, value, name) {
  if (value == null) {
    return;
  }
  parts.push(clause, quoteHirnqlString(requireNonEmptyString(value, name)));
}

function formatSemanticAssignments(options = {}, requireAny) {
  const assignments = [];
  const { description, confidence, evidenceCount } = options;

  if (description != null) {
    assignments.push(
      `description = ${quoteHirnqlString(requireNonEmptyString(description, 'description'))}`,
    );
  }

  if (confidence != null) {
    if (typeof confidence !== 'number' || Number.isNaN(confidence) || !Number.isFinite(confidence)) {
      throw new Error('confidence must be a finite number');
    }
    assignments.push(`confidence = ${confidence}`);
  }

  if (evidenceCount != null) {
    if (!Number.isInteger(evidenceCount) || evidenceCount < 0) {
      throw new Error('evidenceCount must be a non-negative integer');
    }
    assignments.push(`evidence_count = ${evidenceCount}`);
  }

  if (requireAny && assignments.length === 0) {
    throw new Error(
      'at least one semantic update field must be provided: description, confidence, or evidenceCount',
    );
  }

  return assignments;
}

function buildSemanticEditQuery(verb, memoryId, options = {}) {
  const assignments = formatSemanticAssignments(options, true);
  const parts = [
    verb,
    quoteHirnqlString(requireNonEmptyString(memoryId, 'memoryId')),
    'SET',
    assignments.join(', '),
  ];
  appendOptionalStringClause(parts, 'REASON', options.reason, 'reason');
  appendOptionalStringClause(parts, 'OBSERVED AT', options.observedAt, 'observedAt');
  appendOptionalStringClause(parts, 'CAUSED BY', options.causedBy, 'causedBy');
  return parts.join(' ');
}

function buildSemanticMergeQuery(sourceIds, targetId, options = {}) {
  if (!Array.isArray(sourceIds) || sourceIds.length === 0) {
    throw new Error('sourceIds must contain at least one memory ID');
  }

  const quotedSources = sourceIds.map((sourceId) =>
    quoteHirnqlString(requireNonEmptyString(sourceId, 'sourceIds[]')),
  );
  const parts = [
    'MERGE',
    'MEMORY',
    quotedSources.join(', '),
    'INTO',
    quoteHirnqlString(requireNonEmptyString(targetId, 'targetId')),
  ];

  const assignments = formatSemanticAssignments(options, false);
  if (assignments.length > 0) {
    parts.push('SET', assignments.join(', '));
  }

  appendOptionalStringClause(parts, 'REASON', options.reason, 'reason');
  appendOptionalStringClause(parts, 'OBSERVED AT', options.observedAt, 'observedAt');
  appendOptionalStringClause(parts, 'CAUSED BY', options.causedBy, 'causedBy');
  return parts.join(' ');
}

function buildSemanticRetractQuery(memoryId, options = {}) {
  const parts = [
    'RETRACT',
    quoteHirnqlString(requireNonEmptyString(memoryId, 'memoryId')),
  ];
  appendOptionalStringClause(parts, 'REASON', options.reason, 'reason');
  appendOptionalStringClause(parts, 'OBSERVED AT', options.observedAt, 'observedAt');
  appendOptionalStringClause(parts, 'CAUSED BY', options.causedBy, 'causedBy');
  return parts.join(' ');
}

function isAlreadyRegisteredError(error) {
  return error instanceof Error && /already registered/i.test(error.message);
}

/**
 * Zero-config memory API with pluggable embeddings.
 *
 * Combines a Rust-backed native bridge with a JavaScript-side
 * EmbeddingFunction so that remember/recall/think accept plain text.
 *
 * @example
 * ```js
 * const { Memory } = require('hirn');
 *
 * const mem = Memory.open('./brain.hirn');
 * try {
 *   const id = await mem.remember('User prefers dark mode');
 *   const ctx = await mem.think('preferences?');
 *   console.log(ctx.context);
 * } finally {
 *   mem.close();
 * }
 * ```
 */
class Memory {
  /**
  * @param {object} hirn
   * @param {import('./embeddings').EmbeddingFunction} embeddings
   * @param {string} agentId
   * @private
   */
  constructor(hirn, embeddings, agentId) {
    /** @private */ this._hirn = hirn;
    /** @private */ this._embeddings = embeddings;
    /** @private */ this._agentId = agentId;
    /** @private @type {Set<string>} */ this._registeredAgents = new Set();
  }

  /**
   * Open (or create) a brain at the given path.
   *
   * Embedding provider resolution order:
   * 1. Explicit `embeddings` option
   * 2. Auto-detect from environment (OPENAI_API_KEY, OLLAMA_HOST)
   * 3. Fall back to FakeEmbeddings
   *
   * @param {string} path - File system path to the brain directory.
   * @param {Object} [options]
   * @param {import('./embeddings').EmbeddingFunction} [options.embeddings]
   * @param {string} [options.agentId='anonymous']
   * @param {number} [options.tokenBudget=4096]
   * @param {string} [options.tokenizerName] - Rust tokenizer registry name.
   * @returns {Memory}
   */
  static open(path, options = {}) {
    const {
      embeddings: embeddingsOpt,
      agentId = 'anonymous',
      tokenBudget = 4096,
      tokenizerName,
    } = options;

    let embeddings = embeddingsOpt || detectEmbeddings();
    if (!embeddings) {
      embeddings = new FakeEmbeddings();
    }

    const { HirnBridge } = require('./bridge');
    const hirn = HirnBridge.open(path, embeddings.dimensions, tokenBudget, tokenizerName);
    return new Memory(hirn, embeddings, agentId);
  }

  /** Close the memory database. */
  close() {
    if (this._hirn) {
      this._hirn.close();
      this._hirn = null;
    }
  }

  /**
   * @param {string | null} [perCall]
   * @returns {string}
   * @private
   */
  _effectiveAgent(perCall) {
    return perCall || this._agentId;
  }

  /**
   * Wrap an async native call to re-throw as a typed HirnError.
   * @template T
   * @param {() => Promise<T>} fn
   * @returns {Promise<T>}
   * @private
   */
  async _call(fn) {
    try {
      return await fn();
    } catch (err) {
      throw wrapNativeError(err);
    }
  }

  /**
   * @param {string} agentId
   * @returns {Promise<void>}
   * @private
   */
  async _ensureAgent(agentId) {
    if (this._registeredAgents.has(agentId)) return;
    try {
      await this._hirn.registerAgent(agentId, agentId);
    } catch (err) {
      if (!isAlreadyRegisteredError(err)) {
        throw err;
      }
    }
    this._registeredAgents.add(agentId);
  }

  /**
   * Store a text memory with automatic embedding.
   *
   * @param {string} content - Text content to remember.
   * @param {Object} [options]
   * @param {string} [options.agentId]
   * @param {number} [options.importance=0.5]
   * @returns {Promise<string>} The ULID string of the new memory.
   * @throws {TypeError} If content is not a string.
   * @throws {Error} If content is empty or whitespace-only.
   */
  async remember(content, options = {}) {
    if (typeof content !== 'string') {
      throw new TypeError(`content must be a string, got ${typeof content}`);
    }
    if (!content.trim()) {
      throw new Error('content must not be empty or whitespace-only');
    }
    if (!this._hirn) throw new Error('memory is closed');
    const { agentId, importance = 0.5 } = options;
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);
    const [embedding] = await this._embeddings.embedDocuments([content]);
    return this._call(() => this._hirn.remember(aid, content, embedding, importance));
  }

  /**
   * Recall memories relevant to a query.
   *
   * @param {string} query - Natural language query.
   * @param {Object} [options]
   * @param {number} [options.limit=10]
   * @param {number} [options.threshold]
   * @param {string} [options.asOf] - Historical snapshot value (`YYYY-MM-DD`, RFC 3339, or revision ULID).
   * @param {'observed' | 'recorded' | 'revision'} [options.snapshotKind] - Snapshot selector for `asOf`.
   * @param {string} [options.agentId]
   * @returns {Promise<import('./index').RecallResult[]>}
   */
  async recall(query, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const {
      limit = 10,
      threshold,
      asOf,
      snapshotKind,
      agentId,
    } = options;
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);
    const queryVec = await this._embeddings.embedQuery(query);
    return this._call(() => this._hirn.recall(aid, queryVec, limit, threshold, asOf, snapshotKind));
  }

  /**
   * Assemble optimal LLM context for a query.
   *
   * @param {string} query - Natural language query.
   * @param {Object} [options]
   * @param {number} [options.budget=4096]
   * @param {string} [options.agentId]
   * @returns {Promise<import('./index').Context>}
   */
  async think(query, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const { budget = 4096, agentId } = options;
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);
    const queryVec = await this._embeddings.embedQuery(query);
    return this._call(() => this._hirn.think(aid, queryVec, budget));
  }

  /**
   * Append a correction revision for a semantic memory.
   *
   * @param {string} memoryId
   * @param {Object} [options]
   * @param {string} [options.description]
   * @param {number} [options.confidence]
   * @param {number} [options.evidenceCount]
   * @param {string} [options.reason]
   * @param {string} [options.observedAt]
   * @param {string} [options.causedBy]
   * @param {string} [options.agentId]
   * @returns {Promise<import('./index').QueryResult>}
   */
  async correct(memoryId, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const { agentId, ...editOptions } = options;
    const id = requireNonEmptyString(memoryId, 'memoryId');
    formatSemanticAssignments(editOptions, true);
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);

    const reason = editOptions.reason == null ? null : requireNonEmptyString(editOptions.reason, 'reason');
    const observedAt = editOptions.observedAt == null
      ? null
      : requireNonEmptyString(editOptions.observedAt, 'observedAt');
    const causedBy = editOptions.causedBy == null ? null : requireNonEmptyString(editOptions.causedBy, 'causedBy');

    return this._call(() => this._hirn.correctSemantic(
      aid,
      id,
      editOptions.description ?? null,
      editOptions.confidence ?? null,
      editOptions.evidenceCount ?? null,
      reason,
      observedAt,
      causedBy,
    ));
  }

  /**
   * Append a new authoritative semantic revision.
   *
   * @param {string} memoryId
   * @param {Object} [options]
   * @param {string} [options.description]
   * @param {number} [options.confidence]
   * @param {number} [options.evidenceCount]
   * @param {string} [options.reason]
   * @param {string} [options.observedAt]
   * @param {string} [options.causedBy]
   * @param {string} [options.agentId]
   * @returns {Promise<import('./index').QueryResult>}
   */
  async supersede(memoryId, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const { agentId, ...editOptions } = options;
    const id = requireNonEmptyString(memoryId, 'memoryId');
    formatSemanticAssignments(editOptions, true);
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);

    const reason = editOptions.reason == null ? null : requireNonEmptyString(editOptions.reason, 'reason');
    const observedAt = editOptions.observedAt == null
      ? null
      : requireNonEmptyString(editOptions.observedAt, 'observedAt');
    const causedBy = editOptions.causedBy == null ? null : requireNonEmptyString(editOptions.causedBy, 'causedBy');

    return this._call(() => this._hirn.supersedeSemantic(
      aid,
      id,
      editOptions.description ?? null,
      editOptions.confidence ?? null,
      editOptions.evidenceCount ?? null,
      reason,
      observedAt,
      causedBy,
    ));
  }

  /**
   * Merge one or more semantic memories into a canonical target.
   *
   * @param {string[]} sourceIds
   * @param {string} targetId
   * @param {Object} [options]
   * @param {string} [options.description]
   * @param {number} [options.confidence]
   * @param {number} [options.evidenceCount]
   * @param {string} [options.reason]
   * @param {string} [options.observedAt]
   * @param {string} [options.causedBy]
   * @param {string} [options.agentId]
   * @returns {Promise<import('./index').QueryResult>}
   */
  async merge(sourceIds, targetId, options = {}) {
    const { agentId, ...editOptions } = options;
    const hirnql = buildSemanticMergeQuery(sourceIds, targetId, editOptions);
    return this.query(hirnql, { agentId });
  }

  /**
   * Append a tombstone revision for a semantic memory.
   *
   * @param {string} memoryId
   * @param {Object} [options]
   * @param {string} [options.reason]
   * @param {string} [options.observedAt]
   * @param {string} [options.causedBy]
   * @param {string} [options.agentId]
   * @returns {Promise<import('./index').QueryResult>}
   */
  async retract(memoryId, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const { agentId, ...editOptions } = options;
    const id = requireNonEmptyString(memoryId, 'memoryId');
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);

    const reason = editOptions.reason == null ? null : requireNonEmptyString(editOptions.reason, 'reason');
    const observedAt = editOptions.observedAt == null
      ? null
      : requireNonEmptyString(editOptions.observedAt, 'observedAt');
    const causedBy = editOptions.causedBy == null ? null : requireNonEmptyString(editOptions.causedBy, 'causedBy');

    return this._call(() => this._hirn.retractSemantic(
      aid,
      id,
      reason,
      observedAt,
      causedBy,
    ));
  }

  /**
   * Execute a HirnQL query string.
   *
   * Use raw HirnQL here when you need exact clause control, revision-aware
   * statements not covered by the convenience helpers, or plan/explain
   * surfaces. correct(), supersede(), merge(), and retract() cover the
   * common semantic edit flows directly.
    *
   * @param {string} hirnql - HirnQL query string.
   * @param {Object} [options]
   * @param {string} [options.agentId]
   * @returns {import('./index').QueryResult}
   */
  async query(hirnql, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const { agentId } = options;
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);
    return this._call(() => this._hirn.execute(aid, hirnql));
  }

  /**
   * Get database statistics.
   * @returns {import('./index').Stats}
   */
  stats() {
    if (!this._hirn) throw new Error('memory is closed');
    return this._hirn.stats();
  }

  /**
   * Forget (archive) a memory by its ULID string.
   *
   * @param {string} memoryId - ULID string of the memory to forget.
   * @param {Object} [options]
   * @param {string} [options.agentId]
   * @returns {Promise<void>}
   */
  async forget(memoryId, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const { agentId } = options;
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);
    return this._call(() => this._hirn.forget(aid, memoryId));
  }

  /**
   * Store multiple memories with a single batch embedding call.
   *
   * This is significantly more efficient than calling {@link remember}
   * in a loop because embedding API calls are batched.
   *
   * @param {string[]} contents - Array of text contents to remember.
   * @param {Object} [options]
   * @param {string} [options.agentId]
   * @param {number} [options.importance=0.5]
   * @returns {Promise<string[]>} Array of ULID strings (same order as contents).
   * @throws {TypeError} If any element is not a string.
   * @throws {Error} If any element is empty or whitespace-only.
   */
  async batchRemember(contents, options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    if (contents.length === 0) return [];
    for (let i = 0; i < contents.length; i++) {
      if (typeof contents[i] !== 'string') {
        throw new TypeError(`contents[${i}] must be a string, got ${typeof contents[i]}`);
      }
      if (!contents[i].trim()) {
        throw new Error(`contents[${i}] must not be empty or whitespace-only`);
      }
    }
    const { agentId, importance = 0.5 } = options;
    const aid = this._effectiveAgent(agentId);
    await this._ensureAgent(aid);
    const embeddings = await this._embeddings.embedDocuments(contents);
    const ids = [];
    for (let i = 0; i < contents.length; i++) {
      ids.push(await this._call(() => this._hirn.remember(aid, contents[i], embeddings[i], importance)));
    }
    return ids;
  }

  /**
   * Support `using mem = Memory.open(...)` (TC39 Explicit Resource Management).
   */
  [Symbol.dispose]() {
    this.close();
  }

  /**
   * Subscribe to memory events (create, archive, consolidate).
   *
   * Returns a WatchStream whose `next()` method yields events.
   * Call `stream.unsubscribe()` when done.
   *
   * @param {Object} [options]
   * @param {string} [options.filterLayer] - Optional layer filter: "Episodic", "Semantic", "Working".
   * @returns {Promise<import('./index').WatchStream>}
   */
  async watch(options = {}) {
    if (!this._hirn) throw new Error('memory is closed');
    const { filterLayer } = options;
    return this._hirn.watch(filterLayer || null);
  }
}

module.exports = { Memory };
