// @ts-check
'use strict';

const fs = require('fs');
const path = require('path');

const triples = {
  'darwin-arm64': {
    out: 'hirn.darwin-arm64.node',
    sourceNames: ['libhirn_node.dylib'],
  },
  'darwin-x64': {
    out: 'hirn.darwin-x64.node',
    sourceNames: ['libhirn_node.dylib'],
  },
  'linux-x64': {
    out: 'hirn.linux-x64-gnu.node',
    sourceNames: ['libhirn_node.so'],
  },
  'linux-arm64': {
    out: 'hirn.linux-arm64-gnu.node',
    sourceNames: ['libhirn_node.so'],
  },
  'win32-x64': {
    out: 'hirn.win32-x64-msvc.node',
    sourceNames: ['hirn_node.dll', 'libhirn_node.dll'],
  },
};

const key = `${process.platform}-${process.arch}`;
const triple = triples[key];

if (!triple) {
  console.warn(`No local staging rule for platform ${key}; skipping .node staging.`);
  process.exit(0);
}

const repoRoot = path.resolve(__dirname, '..', '..', '..');
const crateDir = path.resolve(__dirname, '..');
const releaseDir = path.join(repoRoot, 'target', 'release');

let sourcePath = null;
for (const sourceName of triple.sourceNames) {
  const candidate = path.join(releaseDir, sourceName);
  if (fs.existsSync(candidate)) {
    sourcePath = candidate;
    break;
  }
}

if (!sourcePath) {
  console.error(
    `Could not find native output in ${releaseDir}. Looked for: ${triple.sourceNames.join(', ')}`,
  );
  process.exit(1);
}

const outPath = path.join(crateDir, triple.out);
fs.copyFileSync(sourcePath, outPath);
console.log(`Staged native module: ${path.basename(sourcePath)} -> ${triple.out}`);
