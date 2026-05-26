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
  }
}

/**
 * Wrap a native napi Error into the appropriate HirnError subclass.
 *
 * @param {Error} err - The original error from the Rust binding.
 * @returns {HirnError}
 */
function wrapNativeError(err) {
  const msg = err.message || String(err);
  if (/not found/i.test(msg)) {
    return new NotFoundError(msg);
  }
  if (/parse|syntax|query|hirnql|compile/i.test(msg)) {
    return new QueryError(msg);
  }
  return new HirnError(msg);
}

module.exports = { HirnError, NotFoundError, QueryError, wrapNativeError };
