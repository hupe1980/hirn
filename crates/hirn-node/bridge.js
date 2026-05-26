// @ts-check
'use strict';

const { existsSync } = require('fs');
const { join } = require('path');

const { platform, arch } = process;

let nativeBinding = null;
let loadError = null;

const triples = {
  'darwin-arm64': 'hirn.darwin-arm64.node',
  'darwin-x64': 'hirn.darwin-x64.node',
  'linux-x64': 'hirn.linux-x64-gnu.node',
  'linux-arm64': 'hirn.linux-arm64-gnu.node',
  'win32-x64': 'hirn.win32-x64-msvc.node',
};

const key = `${platform}-${arch}`;
const file = triples[key];

if (file) {
  const localPath = join(__dirname, file);
  try {
    if (existsSync(localPath)) {
      nativeBinding = require(localPath);
    } else {
      nativeBinding = require(`hirn-${key}`);
    }
  } catch (error) {
    loadError = error;
  }
} else {
  loadError = new Error(`Unsupported platform: ${key}`);
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError;
  }
  throw new Error(`Failed to load native binding for ${key}`);
}

const { Hirn: HirnBridge } = nativeBinding;

module.exports = {
  HirnBridge,
};