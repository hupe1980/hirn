// ESM wrapper for hirn native bindings.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const native = require('./index.js');

export const {
  Memory,
  HirnError,
  NotFoundError,
  QueryError,
  FakeEmbeddings,
  OpenAIEmbeddings,
  OllamaEmbeddings,
  detectEmbeddings,
} = native;
