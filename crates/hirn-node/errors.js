// @ts-check
'use strict';

/**
 * Base error for all hirn operations.
 */
class HirnError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = 'HirnError';
  }
}

/**
 * Thrown when a memory record is not found.
 */
class NotFoundError extends HirnError {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = 'NotFoundError';
    /** Stable machine-readable discriminator. */
    this.code = 'NOT_FOUND';
  }
}

/**
 * Thrown when a HirnQL query is invalid or fails.
 */
class QueryError extends HirnError {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = 'QueryError';
    /** Stable machine-readable discriminator. */
    this.code = 'QUERY';
  }
}

/**
 * Wrap a native napi Error into the appropriate HirnError subclass.
 *
 * The Rust binding prefixes messages with stable discriminators
 * (`NOT_FOUND:`, `QUERY:`) because napi's `err.code` mirrors the napi status
 * and cannot carry custom values. Caller input errors keep the napi
 * `InvalidArg` status. Message-pattern matching remains as a fallback.
 *
 * @param {Error} err - The original error from the Rust binding.
 * @returns {HirnError}
 */
function wrapNativeError(err) {
  const msg = err.message || String(err);
  if (/^NOT_FOUND:/.test(msg) || /not found/i.test(msg)) {
    return new NotFoundError(msg);
  }
  if (/^QUERY:/.test(msg) || /parse|syntax|query|hirnql|compile/i.test(msg)) {
    return new QueryError(msg);
  }
  return new HirnError(msg);
}

module.exports = { HirnError, NotFoundError, QueryError, wrapNativeError };
