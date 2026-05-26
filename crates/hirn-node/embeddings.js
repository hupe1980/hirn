// @ts-check
'use strict';

/**
 * Embedding function interface for hirn.
 *
 * @typedef {Object} EmbeddingFunction
 * @property {number} dimensions - The dimensionality of embedding vectors.
 * @property {(texts: string[]) => Promise<number[][]>} embedDocuments - Embed a batch of documents.
 * @property {(text: string) => Promise<number[]>} embedQuery - Embed a single query string.
 */

/**
 * Deterministic hash-based embeddings for testing.
 * Not semantically meaningful — use only for tests and development.
 */
class FakeEmbeddings {
  /**
   * @param {number} [dimensions=64]
   */
  constructor(dimensions = 64) {
    this.dimensions = dimensions;
  }

  /**
   * @param {string[]} texts
   * @returns {Promise<number[][]>}
   */
  async embedDocuments(texts) {
    return texts.map((t) => this._hashEmbed(t));
  }

  /**
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedQuery(text) {
    return this._hashEmbed(text);
  }

  /**
   * @param {string} text
   * @returns {number[]}
   */
  _hashEmbed(text) {
    const { createHash } = require('crypto');
    const result = [];
    let counter = 0;
    while (result.length < this.dimensions) {
      const hash = createHash('sha256')
        .update(`${counter}:${text}`)
        .digest();
      // Read 4-byte floats from the 32-byte hash
      for (let i = 0; i + 3 < hash.length && result.length < this.dimensions; i += 4) {
        const val = hash.readFloatLE(i);
        // Replace NaN/Inf with 0.0 (some byte patterns produce non-finite floats)
        result.push(Number.isFinite(val) ? val : 0);
      }
      counter++;
    }
    // Normalize
    const norm = Math.sqrt(result.reduce((sum, x) => sum + x * x, 0));
    if (norm > 0) {
      for (let i = 0; i < result.length; i++) {
        result[i] /= norm;
      }
    }
    return result;
  }
}

/**
 * OpenAI embedding function.
 * Requires the `openai` npm package.
 */
class OpenAIEmbeddings {
  /**
   * @param {Object} [options]
   * @param {string} [options.model='text-embedding-3-small']
   * @param {number} [options.dimensions]
   * @param {string} [options.apiKey]
   * @param {number} [options.maxBatchSize=2048]
   */
  constructor(options = {}) {
    const { model = 'text-embedding-3-small', dimensions, apiKey, maxBatchSize = 2048 } = options;
    this._model = model;
    this._maxBatchSize = maxBatchSize;

    const dimensionMap = {
      'text-embedding-3-small': 1536,
      'text-embedding-3-large': 3072,
      'text-embedding-ada-002': 1536,
    };
    this.dimensions = dimensions || dimensionMap[model] || 1536;

    try {
      const OpenAI = require('openai');
      this._client = new OpenAI.default({
        apiKey: apiKey || process.env.OPENAI_API_KEY,
      });
    } catch {
      throw new Error(
        "OpenAI embeddings require the 'openai' package. Install it with: npm install openai"
      );
    }
  }

  /**
   * @param {string[]} texts
   * @returns {Promise<number[][]>}
   */
  async embedDocuments(texts) {
    if (texts.length <= this._maxBatchSize) {
      return this._embedBatch(texts);
    }
    // Chunk large inputs to stay within API limits
    const result = [];
    for (let i = 0; i < texts.length; i += this._maxBatchSize) {
      const chunk = texts.slice(i, i + this._maxBatchSize);
      const batch = await this._embedBatch(chunk);
      result.push(...batch);
    }
    return result;
  }

  /**
   * @param {string[]} texts
   * @returns {Promise<number[][]>}
   * @private
   */
  async _embedBatch(texts) {
    const response = await this._client.embeddings.create({
      model: this._model,
      input: texts,
    });
    if (response.data.length !== texts.length) {
      throw new Error(
        `OpenAI returned ${response.data.length} embeddings for ${texts.length} texts`
      );
    }
    return response.data.map((d) => d.embedding);
  }

  /**
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedQuery(text) {
    const result = await this._embedBatch([text]);
    return result[0];
  }
}

/**
 * Ollama embedding function.
 * Requires the `ollama` npm package.
 */
class OllamaEmbeddings {
  /**
   * @param {Object} [options]
   * @param {string} [options.model='nomic-embed-text']
   * @param {number} [options.dimensions=768]
   * @param {string} [options.host]
   */
  constructor(options = {}) {
    const { model = 'nomic-embed-text', dimensions = 768, host } = options;
    this._model = model;
    this.dimensions = dimensions;

    try {
      const { Ollama } = require('ollama');
      this._client = new Ollama({ host: host || process.env.OLLAMA_HOST });
    } catch {
      throw new Error(
        "Ollama embeddings require the 'ollama' package. Install it with: npm install ollama"
      );
    }
  }

  /**
   * @param {string[]} texts
   * @returns {Promise<number[][]>}
   */
  async embedDocuments(texts) {
    const response = await this._client.embed({
      model: this._model,
      input: texts,
    });
    if (!response.embeddings || response.embeddings.length !== texts.length) {
      throw new Error(
        `Ollama returned ${response.embeddings?.length ?? 0} embeddings for ${texts.length} texts`
      );
    }
    return response.embeddings;
  }

  /**
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedQuery(text) {
    const result = await this.embedDocuments([text]);
    return result[0];
  }
}

/**
 * Auto-detect an embedding function from the environment.
 * @returns {EmbeddingFunction | null}
 */
function detectEmbeddings() {
  if (process.env.OPENAI_API_KEY) {
    try {
      return new OpenAIEmbeddings();
    } catch {
      // openai package not installed
    }
  }
  if (process.env.OLLAMA_HOST) {
    try {
      return new OllamaEmbeddings();
    } catch {
      // ollama package not installed
    }
  }
  return null;
}

module.exports = {
  FakeEmbeddings,
  OpenAIEmbeddings,
  OllamaEmbeddings,
  detectEmbeddings,
};
