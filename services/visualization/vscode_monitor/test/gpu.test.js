// services/visualization/vscode_monitor/test/gpu.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const G = require('../gpu');

test('parses nvidia-smi csv rows', () => {
  const g = G.parse('0, NVIDIA GeForce RTX 5090, 8, 758, 32607, 43\n');
  assert.deepEqual(g, [{ index: 0, name: 'NVIDIA GeForce RTX 5090', util: 8, memUsed: 758, memTotal: 32607, temp: 43 }]);
});

test('an error message or empty output gives no GPU, not a crash', () => {
  assert.deepEqual(G.parse('NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver.'), []);
  assert.deepEqual(G.parse(''), []);
  assert.deepEqual(G.parse(undefined), []);
});

test('history keeps the last samples', () => {
  const h = new G.History(3);
  for (const u of [1, 2, 3, 4]) h.add([{ index: 0, util: u, memUsed: 50, memTotal: 100 }]);
  assert.deepEqual(h.of(0).util, [2, 3, 4]);
  assert.deepEqual(h.of(0).mem, [50, 50, 50]);
});
