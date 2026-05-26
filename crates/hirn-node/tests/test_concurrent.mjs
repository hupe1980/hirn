// Quick concurrent test
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { Memory } = require('../index.js');
const { HirnBridge } = require('../bridge.js');

async function main() {
  console.log('Testing 50 concurrent HirnBridge.remember...');
  const dir = mkdtempSync(join(tmpdir(), 'hirn-test-'));
  const db = HirnBridge.open(join(dir, 'test.hirn'), 64);
  await db.registerAgent('agent-1', 'Test Agent');
  await db.remember('agent-1', 'seed', new Array(64).fill(0.1));
  
  const allIds = [];
  for (let batch = 0; batch < 5; batch++) {
    const promises = [];
    for (let i = 0; i < 10; i++) {
      const idx = batch * 10 + i;
      promises.push(db.remember('agent-1', 'mem ' + idx, new Array(64).fill(0.01 * (idx+1))));
    }
    const ids = await Promise.all(promises);
    allIds.push(...ids);
    console.log('  batch', batch, ':', ids.length, 'ids');
  }
  console.log('Total:', allIds.length, 'unique:', new Set(allIds).size);
  db.close();
  rmSync(dir, { recursive: true, force: true });

  console.log('\nTesting 50 concurrent Memory.remember...');
  const dir2 = mkdtempSync(join(tmpdir(), 'hirn-mem-'));
  const mem = await Memory.open(join(dir2, 'brain.hirn'));
  await mem.remember('Seed entry to initialize tables');
  
  const topics = [
    'quantum computing uses qubits for exponential parallelism',
    'photosynthesis converts carbon dioxide and water into glucose',
    'neural networks are inspired by biological brain architecture',
    'blockchain provides decentralized immutable transaction ledgers',
    'CRISPR enables targeted gene editing in living organisms',
    'machine learning algorithms improve through data-driven training',
    'superconductors conduct electricity with zero resistance',
    'reinforcement learning agents maximize cumulative reward signals',
    'dark matter accounts for approximately 27 percent of the universe',
    'RNA interference silences gene expression post-transcriptionally',
    'gravitational waves were first detected by LIGO in 2015',
    'protein folding determines biological function from amino acids',
    'cryptographic hash functions are one-way and collision resistant',
    'mitochondria are the powerhouses of eukaryotic cells',
    'transformer architecture revolutionized natural language processing',
    'plate tectonics describes the movement of lithospheric plates',
    'epigenetic modifications alter gene expression without changing DNA',
    'distributed systems achieve consensus through protocols like Raft',
    'antibodies are Y-shaped proteins that neutralize pathogens',
    'topology studies properties preserved under continuous deformations',
    'CMOS transistors form the basis of modern integrated circuits',
    'telomeres protect chromosome ends from degradation',
    'convolutional neural networks excel at image recognition tasks',
    'tectonic subduction zones create deep ocean trenches',
    'mRNA vaccines instruct cells to produce spike protein antigens',
    'Fourier transforms decompose signals into component frequencies',
    'stem cells can differentiate into specialized cell types',
    'zero-knowledge proofs verify statements without revealing data',
    'neurotransmitters transmit signals across synaptic cleft gaps',
    'category theory provides abstract frameworks for mathematics',
    'enzyme catalysis accelerates biochemical reactions',
    'homomorphic encryption allows computation on encrypted data',
    'ribosomes translate messenger RNA into polypeptide chains',
    'Bayesian inference updates probability from new evidence',
    'ion channels regulate electrical signaling in excitable cells',
    'formal verification proves program correctness mathematically',
    'DNA polymerase synthesizes new strands during replication',
    'adversarial examples exploit neural network vulnerabilities',
    'chemiosmosis couples electron transport to ATP synthesis',
    'type theory provides foundations for programming languages',
    'histone modifications regulate chromatin structure',
    'federated learning trains models across decentralized sources',
    'signal transduction relays extracellular messages inside cells',
    'lambda calculus is a formal system for function abstraction',
    'cytokines coordinate immune system inflammatory responses',
    'graph neural networks process non-Euclidean structured data',
    'lipid bilayers form selectively permeable biological membranes',
    'consensus algorithms ensure agreement in distributed systems',
    'autophagy degrades and recycles damaged cellular components',
    'differential privacy adds calibrated noise for statistics',
  ];
  const memIds = [];
  for (let batch = 0; batch < 5; batch++) {
    const promises = [];
    for (let i = 0; i < 10; i++) {
      const idx = batch * 10 + i;
      promises.push(mem.remember(topics[idx]));
    }
    const ids = await Promise.all(promises);
    memIds.push(...ids);
    console.log('  batch', batch, ':', ids.length, 'ids');
  }
  console.log('Total:', memIds.length, 'unique:', new Set(memIds).size);
  mem.close();
  rmSync(dir2, { recursive: true, force: true });
  console.log('ALL OK');
}

main().catch(e => { console.error(e); process.exit(1); });
