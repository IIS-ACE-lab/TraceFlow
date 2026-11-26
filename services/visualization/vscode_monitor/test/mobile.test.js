// services/visualization/vscode_monitor/test/mobile.test.js
'use strict';
// The status page (services/visualization/mobile/status.js): its cards are the panel's own state.js read of the
// heartbeats and the registry: the light, the phase, progress, ETA, the counts and the running run's metrics.
const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../state');
const V = require('../../mobile/status');

const NOW = 1_800_000_000_000;
const hb = extra => ({ status: 'running', written_ns: (NOW - 3000) * 1e6, started_ns: (NOW - 3600e3) * 1e6, progress: 0.5,
  model_name: 'tulip_hold', model_id: 'ASCAD_MLPbase', dataset: 'ascad', phase: 'fold', fold: 3, n_folds: 10, epoch: 45,
  fold_epochs: 200, model_hash: 'h1', batches: ['fig20'], trial: 2, trials: 5, ...extra });
const registry = { h1: { model_name: 'tulip_hold', model_id: 'ASCAD_MLPbase', env: 'dev', status: 'IN_PROGRESS' },
  h0: { model_name: 'done_one', env: 'dev', status: 'DONE' }, h9: { model_name: 'bad', env: 'dev', status: 'FAILED' } };

test('a running project: green light, its run, phase, progress, ETA, counts and latest metrics', () => {
  const live = { hash: 'h1', source: 'fold 3', epoch: 45, metrics: [{ name: 'mean_accuracy', values: [
    { process: 'train', label: 'train', value: 0.0061, improved: true }, { process: 'validation', label: 'validation (CV)', value: 0.0044, improved: false }] }] };
  const html = V.view({ projects: [{ name: 'DLSCA_ASCAD', heartbeats: { dev: hb() }, registry, queues: [], live }], gpus: [
    { index: 0, name: 'RTX 5090', util: 87, memUsed: 20480, memTotal: 32607, temp: 64 }], commands: [] }, S, NOW);
  assert.match(html, /<b>GPU0<\/b> RTX 5090 · 87% · 20\.0 \/ 31\.8 GB · 64°C/);
  assert.match(html, /<i class="lamp green"><\/i>DLSCA_ASCAD <small class="env">dev<\/small><span class="state">running<\/span>/);
  assert.match(html, /<b>tulip_hold<\/b> <small>ASCAD_MLPbase · ascad<\/small>/);
  assert.match(html, /fold 3\/10 · epoch 45\/200 · fig20 · trial 2\/5/);
  assert.match(html, /<i style-width="50"><\/i>/);
  assert.match(html, /<div class="pct">50%<\/div><table class="eta"><tr><th>ETA<\/th><td>run: \d\d:\d\d UTC <small>\(in 1\.0 h\)<\/small><\/td><td>batch: \d\d:\d\d UTC <small>\(in 7\.0 h\)<\/small><\/td><\/tr><\/table>/);
  // no queue, no sweep: the batch is the run's own line, trials 3-5 of its 5, each as long as this run (2 h)
  assert.match(html, /<b>1<\/b> running<\/span><span class="done"><b>1<\/b> done<\/span><span class="failed"><b>1<\/b> failed/);
  // as the panel's run card: the head row (fold and epoch, each process), then each metric's values under their process
  assert.match(html, /<div class="live"><table class="rows"><tr class="cols"><th>fold 3 · epoch 45<\/th><th class="col">train<\/th><th class="col">validation \(CV\)<\/th><\/tr>/);
  assert.match(html, /<tr><th>mean_accuracy<\/th><td>0\.006100<i class="dot better"><\/i><\/td><td>0\.004400<i class="dot worse"><\/i><\/td><\/tr>/);
});

test('a running project while its test attacks: the orange attack bar, attack i of n, the test values so far, the key-recovery line', () => {
  const live = { hash: 'h1', source: null, epoch: 20, attack: { attack: 37, attacks: 100, running: true },
    key_recovery: { source: 'test', recovered: true, true_key_rank: 1, top_guess: '0x00', ge: 1, sr1: 1 },
    metrics: [{ name: 'traces_to_ge_1', values: [{ process: 'test', label: 'test', value: null, improved: null, live: true, note: 'not reached yet' }] }] };
  const html = V.view({ projects: [{ name: 'DLSCA_ASCAD', heartbeats: { dev: hb() }, registry, queues: [], live }], gpus: [
    { index: 0, name: 'RTX 5090', util: 87, memUsed: 20480, memTotal: 32607, temp: 64 }], commands: [] }, S, NOW);
  assert.match(html, /<div class="live"><div class="attack"><span>attack<\/span><div class="abar running"><i style-width="37"><\/i><\/div><b>attack 37 of 100<\/b><\/div>/);
  assert.match(html, /<th class="col test">test<\/th><\/tr>/);                          // the test column, orange
  assert.match(html, /<td class="test">not reached yet<i class="dot same"><\/i><\/td>/);
  assert.match(html, /<div class="key">key recovery · test: recovered · true key rank 1 · top guess 0x00 · GE 1 · SR@1 1.00<\/div>/);
});

test('a heartbeat late or stopped turns amber or red; a finished run shows what it was; the newest running env is followed', () => {
  const late = V.view({ projects: [{ name: 'P', heartbeats: { dev: hb({ written_ns: (NOW - 30e3) * 1e6 }) }, registry: {}, queues: [] }] }, S, NOW);
  assert.match(late, /lamp amber[^]*late 30 s/);
  const dead = V.view({ projects: [{ name: 'P', heartbeats: { dev: hb({ written_ns: (NOW - 120e3) * 1e6 }) }, registry: {}, queues: [] }] }, S, NOW);
  assert.match(dead, /lamp red[^]*no beat for 120 s/);
  const done = V.view({ projects: [{ name: 'P', heartbeats: { dev: hb({ status: 'done' }) }, registry: {}, queues: [] }] }, S, NOW);
  assert.match(done, /lamp grey[^]*last: <b>tulip_hold<\/b> · done/);
  const two = { dev: hb({ status: 'done', written_ns: NOW * 1e6 }), prod: hb({ model_name: 'prod_run' }) };
  assert.equal(V.followed(two).env, 'prod', 'a running heartbeat before a newer finished one');
  assert.match(V.view({ projects: [] }, S, NOW), /No project in master\.yaml/);
});

test('the ETA is the panel\'s table: the run\'s and the batch\'s ends in master.yaml\'s timezone, ≤ under early stopping', () => {
  const trials = [1, 2, 3, 4, 5].map(position => ({ position, epochs: 200, passes: 11 }));
  const queues = [{ key: 'fig20', commands: [{ MODEL_ID: 'ASCAD_MLPbase', DATASET: 'ascad' }, { MODEL_ID: 'X', DATASET: 'ascad' }] }];
  const data = h => ({ timezone: 'Asia/Taipei', commands: [], projects: [{ name: 'P', heartbeats: { dev: h }, registry, queues, trials }] });
  const html = V.view(data(hb()), S, NOW);
  // run: half done in an hour, an hour left; batch: this trial's rest, trials 3-5 and one more command's whole sweep
  const e = S.eta({ hb: hb(), nowMs: NOW, trials, position: 2, commandsLeft: 1 });
  assert.match(html, new RegExp(`<th>ETA</th><td>run: ${V.endAt(e.runEndMs, 'Asia/Taipei', NOW)} <small>\\(in 1\\.0 h\\)</small></td>` +
    `<td>batch: ${V.endAt(e.batchEndMs, 'Asia/Taipei', NOW)} <small>\\(in ${(e.batchLeftS / 3600).toFixed(1)} h\\)</small></td>`));
  assert.match(V.endAt(e.batchEndMs, 'Asia/Taipei', NOW), / Taipei$/);
  assert.match(V.view(data(hb({ early_stopping: true })), S, NOW), /<table class="eta" title="at the latest: early stopping may end runs sooner"><tr><th>ETA ≤<\/th>/);
});
