/**
 * Node.js bindings for the hirn cognitive memory database.
 *
 * The package root intentionally exposes the high-level Memory API and
 * shared result types. The napi bridge remains internal to the binding.
 */

// ─── Error Types ─────────────────────────────────────────────

/** Base error for all hirn operations. */
export class HirnError extends Error {
  readonly name: 'HirnError';
}

/** Thrown when a memory record is not found. */
export class NotFoundError extends HirnError {
  readonly name: 'NotFoundError';
}

/** Thrown when a HirnQL query is invalid or fails. */
export class QueryError extends HirnError {
  readonly name: 'QueryError';
}

export type RecallSnapshotKind = 'observed' | 'recorded' | 'revision';

// ─── Interfaces ──────────────────────────────────────────────

/** Database statistics. */
export interface Stats {
  workingCount: number;
  episodicCount: number;
  semanticCount: number;
  totalCount: number;
  fileSizeBytes: number;
}

/** A single recall result. */
export interface RecallResult {
  /** ULID string of the record. */
  id: string;
  /** Memory layer: "Episodic", "Semantic", or "Working". */
  layer: string;
  /** Cosine similarity to the query. */
  similarity: number;
  /** Weighted composite score. */
  compositeScore: number;
  /** Graph activation score. */
  activationScore: number;
  /** Stable logical memory identity for revision-aware records. */
  logicalMemoryId?: string | null;
  /** Immutable revision identifier when the returned record is revision-native. */
  revisionId?: string | null;
  /** Revision state such as "Active", "Superseded", or "Retracted". */
  revisionState?: string | null;
}

/** Think result — assembled context for an LLM prompt. */
export interface Context {
  /** The assembled context string, ready for an LLM prompt. */
  context: string;
  /** Approximate token count. */
  tokenCount: number;
  /** ULID strings of included records. */
  recordsIncluded: string[];
  /** Execution time in milliseconds. */
  queryTimeMs: number;
}

/** A watch event emitted by the database. */
export interface WatchEvent {
  /** Event type: "created" | "archived" | "consolidated". */
  eventType: string;
  /** Memory ID (for created/archived events). Null for consolidated. */
  id?: string | null;
  /** Memory layer (for created events). Null otherwise. */
  layer?: string | null;
  /** Content preview (for created events). Null otherwise. */
  contentPreview?: string | null;
  /** Number of records processed (for consolidated events). Null otherwise. */
  recordsProcessed?: number | null;
}

/** Result of a HirnQL execute / inspect / trace operation. */
export interface QueryResult {
  /** Result type: "records", "created", "forgotten", "connected", "inspected", "traced", "consolidated", "watch_ack". */
  type: string;
  /** The full result data as a JSON-compatible object. */
  data: Record<string, unknown>;
}

export interface SemanticEditOptions {
  description?: string;
  confidence?: number;
  evidenceCount?: number;
  reason?: string;
  observedAt?: string;
  causedBy?: string;
  agentId?: string;
}

export interface SemanticRetractOptions {
  reason?: string;
  observedAt?: string;
  causedBy?: string;
  agentId?: string;
}

/**
 * A watch stream that yields memory events.
 *
 * Call `next()` repeatedly to receive events. Returns `null` when the
 * database is closed or the stream is unsubscribed.
 */
export class WatchStream {
  /**
   * Wait for the next event. Returns `null` if the stream is closed.
   *
   * @param filterType - Optional event type filter: "created", "archived", "consolidated".
   */
  next(filterType?: string | null): Promise<WatchEvent | null>;

  /** Unsubscribe from the event stream. */
  unsubscribe(): void;
}

/**
 * Zero-config memory API with pluggable embeddings.
 *
 * Combines a Rust-backed engine with a JavaScript-side embedding function.
 *
 * @example
 * ```ts
 * import { Memory } from '@hupe1980/hirn';
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
 *
 * With explicit embeddings:
 * ```ts
 * import { Memory, OpenAIEmbeddings } from '@hupe1980/hirn';
 *
 * const mem = Memory.open('./brain.hirn', {
 *   embeddings: new OpenAIEmbeddings(),
 * });
 * ```
 */
export class Memory {
  /**
   * Open (or create) a brain at the given path.
   *
   * Embedding provider resolution order:
   * 1. Explicit `embeddings` option
   * 2. Auto-detect from environment (OPENAI_API_KEY, OLLAMA_HOST)
   * 3. Fall back to FakeEmbeddings
   *
   * @param path - File system path to the brain directory.
   * @param options - Optional configuration.
   */
  static open(path: string, options?: {
    embeddings?: EmbeddingFunction;
    agentId?: string;
    tokenBudget?: number;
    /** Rust tokenizer registry name. JS token counting does not become authoritative. */
    tokenizerName?: string;
  }): Memory;

  /** Close the memory database. Should be called when done. */
  close(): void;

  /**
   * Store a text memory with automatic embedding.
   *
   * @param content - Text content to remember.
   * @param options - Optional per-call settings.
   * @returns The ULID string of the new memory.
   */
  remember(content: string, options?: {
    agentId?: string;
    importance?: number;
  }): Promise<string>;

  /**
   * Recall memories relevant to a natural language query.
   *
   * @param query - Natural language query.
   * @param options - Optional per-call settings.
   * @returns Array of recall results.
   */
  recall(query: string, options?: {
    limit?: number;
    threshold?: number;
    asOf?: string;
    snapshotKind?: RecallSnapshotKind;
    agentId?: string;
  }): Promise<RecallResult[]>;

  /**
   * Assemble optimal LLM context for a query under a token budget.
   *
   * @param query - Natural language query.
   * @param options - Optional per-call settings.
   * @returns Context with the assembled string.
   */
  think(query: string, options?: {
    budget?: number;
    agentId?: string;
  }): Promise<Context>;

  /**
   * Append a correction revision for a semantic memory.
   *
   * @param memoryId - Semantic memory ULID.
   * @param options - Semantic edit options.
   */
  correct(memoryId: string, options?: SemanticEditOptions): Promise<QueryResult>;

  /**
   * Append a new authoritative semantic revision.
   *
   * @param memoryId - Semantic memory ULID.
   * @param options - Semantic edit options.
   */
  supersede(memoryId: string, options?: SemanticEditOptions): Promise<QueryResult>;

  /**
   * Merge one or more semantic memories into a canonical target.
   *
   * @param sourceIds - Semantic memory ULIDs to merge.
   * @param targetId - Canonical target semantic memory ULID.
   * @param options - Optional semantic edit metadata for the merged head.
   */
  merge(sourceIds: string[], targetId: string, options?: SemanticEditOptions): Promise<QueryResult>;

  /**
   * Append a tombstone revision for a semantic memory.
   *
   * @param memoryId - Semantic memory ULID.
   * @param options - Retraction metadata.
   */
  retract(memoryId: string, options?: SemanticRetractOptions): Promise<QueryResult>;

  /**
   * Execute a HirnQL query string.
   *
   * Use raw HirnQL here when you need exact clause control, revision-aware
   * statements not covered by the convenience helpers, or plan/explain
   * surfaces.
   *
   * @param hirnql - HirnQL query string.
   * @param options - Optional per-call settings.
   * @returns QueryResult with the result type and data.
   */
  query(hirnql: string, options?: {
    agentId?: string;
  }): Promise<QueryResult>;

  /**
   * Forget (archive) a memory by its ULID string.
   *
   * @param memoryId - ULID string of the memory to forget.
   * @param options - Optional per-call settings.
   */
  forget(memoryId: string, options?: {
    agentId?: string;
  }): Promise<void>;

  /**
   * Store multiple memories with a single batch embedding call.
   *
   * This is significantly more efficient than calling `remember()`
   * in a loop because embedding API calls are batched.
   *
   * @param contents - Array of text contents to remember.
   * @param options - Optional per-call settings.
   * @returns Array of ULID strings (same order as contents).
   */
  batchRemember(contents: string[], options?: {
    agentId?: string;
    importance?: number;
  }): Promise<string[]>;

  /**
   * Get database statistics.
   *
   * @returns Stats with record counts and file size.
   */
  stats(): Stats;

  /**
   * Subscribe to memory events (create, archive, consolidate).
   *
   * Returns a WatchStream whose `next()` method yields events.
   * Call `stream.unsubscribe()` when done.
   *
   * @param options - Optional per-call settings.
   * @returns A WatchStream for receiving events.
   */
  watch(options?: {
    filterLayer?: string;
  }): Promise<WatchStream>;

  /**
   * Support `using mem = Memory.open(...)` (TC39 Explicit Resource Management).
   */
  [Symbol.dispose](): void;
}

/** Embedding function interface. Implement this to use a custom embedding provider. */
export interface EmbeddingFunction {
  /** The dimensionality of embedding vectors. */
  readonly dimensions: number;
  /** Embed a batch of documents. */
  embedDocuments(texts: string[]): Promise<number[][]>;
  /** Embed a single query string. */
  embedQuery(text: string): Promise<number[]>;
}

/** Deterministic hash-based embeddings for testing. */
export class FakeEmbeddings implements EmbeddingFunction {
  readonly dimensions: number;
  constructor(dimensions?: number);
  embedDocuments(texts: string[]): Promise<number[][]>;
  embedQuery(text: string): Promise<number[]>;
}

/** OpenAI embedding function. Requires the `openai` npm package. */
export class OpenAIEmbeddings implements EmbeddingFunction {
  readonly dimensions: number;
  constructor(options?: { model?: string; dimensions?: number; apiKey?: string; maxBatchSize?: number });
  embedDocuments(texts: string[]): Promise<number[][]>;
  embedQuery(text: string): Promise<number[]>;
}

/** Ollama embedding function. Requires the `ollama` npm package. */
export class OllamaEmbeddings implements EmbeddingFunction {
  readonly dimensions: number;
  constructor(options?: { model?: string; dimensions?: number; host?: string });
  embedDocuments(texts: string[]): Promise<number[][]>;
  embedQuery(text: string): Promise<number[]>;
}

/** Auto-detect an embedding function from the environment. */
export function detectEmbeddings(): EmbeddingFunction | null;
