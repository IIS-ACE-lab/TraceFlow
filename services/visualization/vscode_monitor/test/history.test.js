// services/visualization/vscode_monitor/test/history.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const H = require('../history');

function npy(rows) {
  const cols = rows[0].length;
  let header = `{'descr': '<f8', 'fortran_order': False, 'shape': (${rows.length}, ${cols}), }`;
  while ((10 + header.length + 1) % 64) header += ' ';
  header += '\n';
  const data = Buffer.alloc(rows.length * cols * 8);
  rows.flat().forEach((v, i) => data.writeDoubleLE(v, i * 8));
  const pre = Buffer.alloc(10);
  pre.write('\x93NUMPY', 0, 'latin1'); pre[6] = 1; pre[7] = 0; pre.writeUInt16LE(header.length, 8);
  return Buffer.concat([pre, Buffer.from(header, 'latin1'), data]);
}

test('npyLastRow reads the endpoint of a rank curve', () => {
  assert.deepEqual(H.npyLastRow(npy([[10, 120], [20, 80], [5000, 3]])), [5000, 3]);
  assert.equal(H.npyLastRow(Buffer.from('not numpy')), null);
});

test('rankByEpoch orders the epochs and skips other files', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  fs.writeFileSync(path.join(dir, 'ep0002.npy'), npy([[1, 50], [2, 20]]));
  fs.writeFileSync(path.join(dir, 'ep0001.npy'), npy([[1, 90], [2, 60]]));
  fs.writeFileSync(path.join(dir, 'note.txt'), 'x');
  assert.deepEqual(H.rankByEpoch(dir), [{ epoch: 1, rank: 60 }, { epoch: 2, rank: 20 }]);
  assert.deepEqual(H.rankByEpoch(path.join(dir, 'missing')), []);
});

test('JsonlReader returns complete new lines only', () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'tf-')), 'h.jsonl');
  fs.writeFileSync(file, '{"epoch": 1}\n{"epoch": 2');
  const r = new H.JsonlReader(file);
  assert.deepEqual(r.readNew(), [{ epoch: 1 }]);
  fs.appendFileSync(file, '}\n{"epoch": 3}\n');
  assert.deepEqual(r.readNew(), [{ epoch: 2 }, { epoch: 3 }]);
  assert.deepEqual(r.readNew(), []);
  assert.equal(r.rewound, false);
  fs.writeFileSync(file, '{"epoch": 9}\n');                                 // rewritten, shorter
  assert.deepEqual(r.readNew(), [{ epoch: 9 }]);
  assert.equal(r.rewound, true);
});

test('sparkPath spans the box', () => {
  assert.equal(H.sparkPath([1, 3], 10, 4), 'M0.0,4.0L10.0,0.0');
  assert.equal(H.sparkPath([], 10, 4), '');
});

test('rankByEpoch reads each epoch file once with a cache', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  fs.writeFileSync(path.join(dir, 'ep0001.npy'), npy([[1, 9]]));
  const cache = new Map();
  assert.deepEqual(H.rankByEpoch(dir, cache), [{ epoch: 1, rank: 9 }]);
  fs.writeFileSync(path.join(dir, 'ep0001.npy'), 'overwritten');           // not read again
  fs.writeFileSync(path.join(dir, 'ep0002.npy'), npy([[1, 4]]));
  assert.deepEqual(H.rankByEpoch(dir, cache), [{ epoch: 1, rank: 9 }, { epoch: 2, rank: 4 }]);
});

test('arrayDirs labels folds and runCurveDir picks the last validation rank curve', () => {
  const m = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  for (const f of ['fold_01', 'fold_02']) {
    fs.mkdirSync(path.join(m, 'cross_validation', f, 'arrays', 'validation', 'mean_attack_rank_curve'), { recursive: true });
    fs.mkdirSync(path.join(m, 'cross_validation', f, 'scalars'), { recursive: true });
  }
  fs.mkdirSync(path.join(m, 'arrays', 'test', 'sr_curve'), { recursive: true });
  assert.deepEqual(H.arrayDirs(m).map(d => d.label),
    ['fold_01 · validation/mean_attack_rank_curve', 'fold_02 · validation/mean_attack_rank_curve', 'run · test/sr_curve']);
  assert.equal(H.runCurveDir(m), path.join(m, 'cross_validation', 'fold_02', 'arrays', 'validation', 'mean_attack_rank_curve'));
  assert.equal(H.runCurveDir(path.join(m, 'missing')), null);
});

test('scalarSeries keeps numeric fields but the epoch', () => {
  assert.deepEqual(H.scalarSeries([{ epoch: 1, loss: 2, name: 'x' }, { epoch: 2, loss: 1, acc: 0.5 }]), { loss: [2, 1], acc: [0.5] });
});

test('historyFiles finds the scalar histories only', () => {
  const m = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  fs.mkdirSync(path.join(m, 'cross_validation', 'fold_01', 'scalars'), { recursive: true });
  fs.mkdirSync(path.join(m, 'key_recovery'), { recursive: true });
  fs.writeFileSync(path.join(m, 'cross_validation', 'fold_01', 'scalars', 'x_metrics_train_history.jsonl'), '');
  fs.writeFileSync(path.join(m, 'key_recovery', 'validation_history.jsonl'), '');
  assert.deepEqual(H.historyFiles(m), [path.join(m, 'cross_validation', 'fold_01', 'scalars', 'x_metrics_train_history.jsonl')]);
});
