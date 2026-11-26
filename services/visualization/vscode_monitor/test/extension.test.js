// services/visualization/vscode_monitor/test/extension.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

// extension.js needs `vscode`; these tests touch only its file reading, so an empty stub is enough.
const load = Module._load;
Module._load = function (request, ...rest) {
  if (request === 'vscode') return { workspace: { getConfiguration: () => ({ get: () => undefined }) }, Uri: { file: p => ({ fsPath: p }) } };
  return load.call(this, request, ...rest);
};
const E = require('../extension');

function project(tmp) {
  const proj = path.join(tmp, 'projects', 'P');
  fs.mkdirSync(path.join(proj, 'artifacts', 'dev'), { recursive: true });
  fs.mkdirSync(path.join(proj, 'graphs', 'jobs', 'job_002_fig', 'sub'), { recursive: true });
  fs.writeFileSync(path.join(proj, 'graphs', 'jobs', 'job_002_fig', 'sub', 'a.pdf'), '');
  return {
    name: 'P', path: proj, config_dir: path.join(proj, 'config'), configs: [],
    registry: path.join(proj, 'artifacts', 'run_registry.json'),
    heartbeat: path.join(proj, 'artifacts', 'dev', 'heartbeat.json'),       // where the tests write a beat
    artifacts: path.join(proj, 'artifacts'),
    graph_jobs_dir: path.join(proj, 'graphs', 'jobs'),
    queues: [{ key: 'fig2', commands: [{}] }],
    graph_jobs: [{ id: 2, name: 'fig', enabled: true, dir: 'job_002_fig' }, { id: 3, name: 'g', enabled: true, dir: 'job_003_g' }],
    errors: [],
  };
}

test('a registry run without a heartbeat is shown, grey, with its trial', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  fs.writeFileSync(p.registry, JSON.stringify({
    abcd: { model_name: 'm', status: 'in_progress', batches: ['fig2'], trial_index: 6, total_trials: 14, path: 'P/artifacts/dev/generated_models/m' },
  }));
  const panel = new E.Panel({});
  panel.layout = { projects: [p] };
  panel.pollFigures();
  const s = panel.projectState(p, Date.now());
  assert.equal(s.light.color, 'grey');
  assert.match(s.run.phase, /no heartbeat/);
  assert.deepEqual([s.run.trial, s.run.trials, s.run.queue], [1, 14, 'fig2']);   // first started of fig2's 14 planned
  assert.equal(s.queues[0].state, 'running');
  assert.deepEqual(s.figures.map(f => f.drawn), [true, false]);
  assert.equal(s.figures[0].filePaths[0], path.join(p.graph_jobs_dir, 'job_002_fig', 'sub', 'a.pdf'));
});

test('a beating heartbeat wins, and a half-written registry keeps the last good copy', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  fs.writeFileSync(p.registry, JSON.stringify({ abcd: { model_name: 'm', status: 'DONE', batches: ['fig2'] },
                                               ef01: { model_name: 'n', status: 'in_progress' } }));
  fs.writeFileSync(p.heartbeat, JSON.stringify({ status: 'running', written_ns: Date.now() * 1e6, model_name: 'n', model_hash: 'ef01', progress: 0.5, phase: 'refit', epoch: 3, refit_epochs: 10 }));
  const panel = new E.Panel({});
  const first = panel.projectState(p, Date.now());
  assert.equal(first.light.color, 'green');
  assert.equal(first.run.hash, 'ef01');
  assert.equal(first.run.percent, 50);
  fs.writeFileSync(p.registry, '{"abcd": {"stat');
  fs.utimesSync(p.registry, new Date(), new Date(Date.now() + 5000));
  assert.equal(panel.projectState(p, Date.now()).tiles.done.length, 1);
});

test('a run under test stays under Running with its card (attack bar, live block); the card goes when it is done', () => {
  // train_and_test writes DONE before the test; the heartbeat beats on in phase test until the run ends
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  fs.writeFileSync(p.registry, JSON.stringify({ ef01: { model_name: 'pepper_basket', status: 'DONE', env: 'dev',
                                                        path: 'P/artifacts/dev/generated_models/pepper_basket' } }));
  fs.writeFileSync(p.heartbeat, JSON.stringify({ status: 'running', written_ns: Date.now() * 1e6, model_name: 'pepper_basket',
                                                 model_hash: 'ef01', phase: 'test', progress: 0.95 }));
  // the test's progress, read directly by the extension (no Python call: a one-second test still shows)
  fs.writeFileSync(path.join(p.artifacts, 'dev', 'attack_progress.json'), JSON.stringify(
    { model_name: 'pepper_basket', model_hash: 'ef01', attack: 37, attacks: 100, running: true, _tfseal: 'x' }));
  const panel = new E.Panel({});
  const asked = [];
  panel.liveMetrics = (proj, hash) => { asked.push(hash); return { run: 'pepper_basket', metrics: [] }; };
  const s = panel.projectState(p, Date.now());
  assert.deepEqual(s.tiles.running.map(r => r.hash), ['ef01']);
  assert.deepEqual(s.tiles.done, []);
  assert.deepEqual(s.run.attack, { attack: 37, attacks: 100, running: true,     // the bar in the phase's place
                                   startedMs: null, writtenMs: null });           // no times in this file: no test speed
  assert.deepEqual(s.run.live, { run: 'pepper_basket', metrics: [] });           // the live block, in the card
  assert.ok(asked.includes('ef01'));
  fs.writeFileSync(path.join(p.artifacts, 'dev', 'attack_progress.json'), JSON.stringify(
    { model_name: 'pepper_basket', model_hash: 'ef01', attack: 100, attacks: 100, running: false }));
  fs.writeFileSync(p.heartbeat, JSON.stringify({ status: 'done', written_ns: Date.now() * 1e6, model_name: 'pepper_basket',
                                                 model_hash: 'ef01', phase: 'test', progress: 1 }));
  const after = panel.projectState(p, Date.now());
  assert.deepEqual(after.tiles.done.map(r => r.hash), ['ef01']);                  // the beat ended: done
  assert.equal(after.run, null);                                                  // and its card is gone
  assert.equal(after.lastRun, undefined);
});

test('a running heartbeat of a deleted run, or with no TraceFlow process left, is stale', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  fs.writeFileSync(p.registry, JSON.stringify({ abcd: { model_name: 'm', status: 'DONE', env: 'dev', started_at: '2026-10-01' } }));
  fs.writeFileSync(p.heartbeat, JSON.stringify({ status: 'running', written_ns: Date.now() * 1e6, model_name: 'flock_fusion', model_hash: 'ef01' }));
  const panel = new E.Panel({});
  const s = panel.projectState(p, Date.now());
  assert.equal(s.run, null);
  assert.equal(s.light.label, 'stale heartbeat: flock_fusion is not in the run registry');
  assert.doesNotMatch(s.envSource, /running run/);
  fs.writeFileSync(p.registry, JSON.stringify({ ef01: { model_name: 'flock_fusion', status: 'in_progress', env: 'dev' } }));
  fs.utimesSync(p.registry, new Date(), new Date(Date.now() + 5000));
  Object.assign(panel, { busy: [], polledBusy: true });            // terminated: nothing runs on the machine
  const t = panel.projectState(p, Date.now());
  assert.match(t.run.phase, /stale heartbeat: flock_fusion: no TraceFlow process runs/);
  assert.equal(t.run.alive.alive, false);
});

test('Housekeeping PARAM: rows whose snapshot holds the value; a key no snapshot has is an error', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  const runs = { h1: ['alpha', 800], h2: ['beta', 75] };
  fs.writeFileSync(p.registry, JSON.stringify(Object.fromEntries(Object.entries(runs).map(([h, [n]]) =>
    [h, { model_name: n, status: 'DONE', env: 'dev', started_at: '2026-10-01' }]))));
  for (const [n, epochs] of Object.values(runs)) {
    const dir = path.join(p.artifacts, 'dev', 'generated_models', n, 'config');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${n}_params.yaml`), `trainer:\n  epochs: ${epochs}\ndata:\n- name: ascad\n`);
  }
  const store = { 'traceflow.deleteFilters': { P: { PARAM: 'trainer.epochs=800' } } };
  const panel = new E.Panel({ workspaceState: { get: k => store[k], update: async (k, v) => { store[k] = v; } } });
  Object.assign(panel, { root: path.resolve(__dirname, '../../../..'), python: E.findPython() });
  let ticked;
  const done = () => new Promise(r => { ticked = r; });
  panel.tick = () => ticked && ticked();
  let wait = done();
  assert.match(panel.projectState(p, Date.now()).housekeeping.note, /reading PARAM/);
  await wait;
  let hk = panel.projectState(p, Date.now()).housekeeping;
  assert.deepEqual([hk.runs.map(r => r.name), hk.note, hk.error], [['alpha'], 'PARAM: trainer.epochs=800', '']);
  store['traceflow.deleteFilters'].P.PARAM = 'trainer.epochs.75; data.name=ASCAD';      // dotted form, case ignored
  wait = done();
  panel.projectState(p, Date.now());
  await wait;
  assert.deepEqual(panel.projectState(p, Date.now()).housekeeping.runs.map(r => r.name), ['beta']);
  store['traceflow.deleteFilters'].P.PARAM = 'trainer.epoch=3';
  wait = done();
  panel.projectState(p, Date.now());
  await wait;
  hk = panel.projectState(p, Date.now()).housekeeping;
  assert.deepEqual([hk.runs.length, hk.error], [0, 'PARAM trainer.epoch: no snapshot has this key (2 read)']);
});

test('missing registry is an error line, not a crash', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  const s = new E.Panel({}).projectState(p, Date.now());
  assert.deepEqual([s.tiles.done.length, s.run], [0, null]);
  assert.match(s.errors[0], /no run registry yet/);
});

test('inside keeps opened paths in the repository', () => {
  assert.equal(E.inside('/r', '/r/projects/P/x.pdf'), true);
  assert.equal(E.inside('/r', '/r/../etc/passwd'), false);
  assert.equal(E.inside('/r', '/rx/a'), false);
});

test('the panel shows the ENV of the beating run, whatever the setting default', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  fs.mkdirSync(path.join(p.artifacts, 'prod'), { recursive: true });
  fs.writeFileSync(path.join(p.artifacts, 'prod', 'heartbeat.json'),
    JSON.stringify({ status: 'running', written_ns: Date.now() * 1e6, model_name: 'pr', model_hash: 'ab12', progress: 0.3 }));
  const s = new E.Panel({}).projectState(p, Date.now());
  assert.deepEqual([s.env, s.envSource, s.run.name, s.light.color], ['prod', 'the running run pr', 'pr', 'green']);
});

test('a Done run\'s metric menu is reused from workspaceState while its metric files are unchanged', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  const run = name => path.join(p.path, 'artifacts', 'dev', 'generated_models', name, 'metrics', 'attack_metrics', 'scalars');
  for (const n of ['m', 'n']) { fs.mkdirSync(run(n), { recursive: true }); fs.writeFileSync(path.join(run(n), `${n}_metrics_test_history.jsonl`), '{}\n'); }
  const store = {};
  const panel = new E.Panel({ workspaceState: { get: k => store[k], update: async (k, v) => { store[k] = v; } } });
  panel.followed = () => ({ env: 'dev' });
  panel.root = null;                                           // no python here: nothing is read, only reused
  const rows = [{ source: 'attack_metrics', process: 'test', metric: 'accuracy', kind: 'per_epoch' }];
  store['traceflow.metricChoices'] = { 'P|dev|m': { sig: panel.metricsSig(p, 'm'), rows }, 'P|dev|n': { sig: '0:0:0', rows } };
  const known = panel.choicesFor(p, ['m', 'n']);
  assert.deepEqual(known.m, rows);                             // fingerprint holds: the stored menu, at once
  assert.equal(known.n, undefined);                            // files changed since: read again (not reused)
  const before = panel.metricsSig(p, 'm');
  fs.writeFileSync(path.join(run('m'), 'm_metrics_retest_history.jsonl'), '{}\n');   // a Retest adds a record
  assert.notEqual(panel.metricsSig(p, 'm'), before);
});

test('a graph job of one run is found in that run\'s graphs folder, one of several runs in graphs/jobs/', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  p.graph_jobs = [{ id: 2, name: 'fig', enabled: true, dir: 'job_002_fig', runs: ['a', 'b'] },
    { id: 3, name: 'g', enabled: true, dir: 'job_003_g', runs: ['m'] }];
  p.model_graphs = 'artifacts/{{env}}/generated_models/{{model_name}}/graphs';
  const runFig = path.join(p.path, 'artifacts', 'dev', 'generated_models', 'm', 'graphs', 'job_003_g', 'model_overlay');
  fs.mkdirSync(runFig, { recursive: true });
  fs.writeFileSync(path.join(runFig, 'b.pdf'), '');
  fs.mkdirSync(path.join(p.graph_jobs_dir, 'job_002_fig'), { recursive: true });
  const panel = new E.Panel({});
  panel.layout = { projects: [p] };
  panel.followed = () => ({ env: 'dev' });
  panel.pollFigures();
  const dirs = panel.jobDirs.P;
  assert.equal(dirs[2].root, p.graph_jobs_dir);
  assert.equal(dirs[3].root, path.join(p.path, 'artifacts', 'dev', 'generated_models', 'm', 'graphs'));
  assert.equal(dirs[3].dir, 'job_003_g');
  assert.ok(panel.isFigureDir(p, path.join(dirs[3].root, 'job_003_g')));                  // Delete may remove it
  assert.ok(!panel.isFigureDir(p, path.join(p.path, 'artifacts', 'dev', 'generated_models', 'm')));   // never the run
  assert.ok(!panel.isFigureDir(p, p.graph_jobs_dir));
});

test('a job\'s figures are its own folder only: one left by a deleted job of the same id is not shown', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  p.graph_jobs = [{ id: 3, name: 'rank', enabled: true, dir: 'job_003_rank', runs: ['a', 'b'] },
    { id: 4, name: 'acc_1', enabled: true, dir: 'job_004_acc_1', runs: ['a', 'b'] }];
  for (const [dir, file] of [['job_003_acc', 'old_accuracy.pdf'], ['job_004_acc', 'old.pdf'], ['job_004_acc_1', 'new.pdf']]) {
    fs.mkdirSync(path.join(p.graph_jobs_dir, dir), { recursive: true });
    fs.writeFileSync(path.join(p.graph_jobs_dir, dir, file), '');
  }
  const panel = new E.Panel({});
  panel.layout = { projects: [p] };
  panel.followed = () => ({ env: 'dev' });
  panel.pollFigures();
  assert.equal(panel.jobDirs.P[3], undefined);                       // job 3 rank is not drawn yet
  assert.deepEqual([panel.jobDirs.P[4].dir, panel.jobDirs.P[4].files], ['job_004_acc_1', ['new.pdf']]);
  assert.equal(panel.newestFigure(p, 3, 0, 'dev'), null);
  assert.equal(panel.newestFigure(p, 4, 0, 'dev'), path.join(p.graph_jobs_dir, 'job_004_acc_1', 'new.pdf'));
});

test('a facet is drawn when its exact folder under graphs/facets/ holds a figure; Delete may remove that folder only', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = project(tmp);
  p.graphs_dir = path.join(p.path, 'graphs');
  p.graph_facets = [{ id: 1, name: 'two', dir: 'facet_001_two' }, { id: 2, name: 'three', dir: 'facet_002_three' }];
  const facets = path.join(p.graphs_dir, 'facets');
  fs.mkdirSync(path.join(facets, 'facet_001_two'), { recursive: true });
  fs.writeFileSync(path.join(facets, 'facet_001_two', 'two.pdf'), '');
  fs.mkdirSync(path.join(facets, 'facet_002_old'), { recursive: true });          // another name: not facet 2's
  const panel = new E.Panel({});
  panel.layout = { projects: [p] };
  panel.followed = () => ({ env: 'dev' });
  panel.pollFigures();
  assert.deepEqual(Object.keys(panel.facetDirs.P), ['1']);
  assert.deepEqual(panel.facetDirs.P[1].files, ['two.pdf']);
  assert.ok(panel.isFigureDir(p, path.join(facets, 'facet_001_two')));
  assert.ok(!panel.isFigureDir(p, facets) && !panel.isFigureDir(p, path.join(facets, '_deleted', 'facet_001_two_2026-10-18')));
});

test('staged edits are written so every path still names what it named when staged', () => {
  const order = E.orderEdits([
    { op: 'delete', path: ['L', 3, 'Conv1d', 'foo'] }, { op: 'delete', path: ['L', 1] }, { op: 'add', path: ['L'], value: '{}' },
    { op: 'set', path: ['L', 4, 'Conv1d', 'bar'], value: '2' }, { op: 'remove', path: ['L', 3] }, { op: 'put', path: [], key: 'k', value: '1' },
  ]).map(e => `${e.op} ${e.path.join('.')}`);
  assert.deepEqual(order, ['set L.4.Conv1d.bar', 'put ', 'add L', 'delete L.3.Conv1d.foo', 'remove L.3', 'delete L.1']);
});

test('the planner page loads menus.js, planner_core.js, planner.js and planner.css under the panel\'s CSP', () => {
  const panel = new E.Panel({ extensionPath: path.join(__dirname, '..') });
  const webview = { cspSource: 'CSP', asWebviewUri: u => ({ toString: () => `res:${path.basename(u.fsPath)}` }) };
  const html = panel.html(webview, 'planner', 'planner');
  for (const f of ['menus.js', 'planner_core.js', 'planner.js', 'planner.css']) assert.match(html, new RegExp(`res:${f.replace('.', '\\.')}`));
  assert.match(html, /style-src CSP; script-src 'nonce-[0-9a-f]{32}'/);
  assert.match(panel.html(webview, 'main'), /res:panel\.js/);
});

test('staged arrange is written first and replaces the staged edits inside its list', () => {
  const L = ['model', 2, 'architecture_args', 'layers'];
  const order = E.orderEdits([
    { op: 'set', path: [...L, 1, 'Conv1d', 'kernel_size'], value: '5' }, { op: 'add', path: L, value: '{"Flatten": {}}' },
    { op: 'delete', path: ['model', 3] }, { op: 'arrange', path: L, value: '[{"from": 1}, {"from": 0}]' },
    { op: 'set', path: ['seed'], value: '1' }, { op: 'set', path: ['model', 2, 'architecture_args', 'layers2'], value: '1' },
  ]).map(e => `${e.op} ${e.path.join('.')}`);
  assert.deepEqual(order, ['arrange model.2.architecture_args.layers', 'set seed', 'set model.2.architecture_args.layers2', 'delete model.3']);
  assert.ok(E.APPLY_OPS.includes('arrange'));
});

test('Housekeeping menus: every value the registry\'s runs hold, beside params.yaml\'s; a model params.yaml dropped still filters', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = { ...project(tmp), model_ids: ['MLP'], datasets: ['ascad'] };
  fs.writeFileSync(p.registry, JSON.stringify({
    h1: { model_name: 'alpha', model_id: 'MLP', status: 'DONE', env: 'dev', batches: ['fig24'], started_at: '2026-10-01' },
    h2: { model_name: 'beta', model_id: 'OLD_CNN', status: 'FAILED', env: 'staging', batches: ['fig2', 'fig24'], started_at: '2026-10-02' },
  }));
  const store = { 'traceflow.deleteFilters': {} };
  const panel = new E.Panel({ workspaceState: { get: k => store[k], update: async (k, v) => { store[k] = v; } } });
  let hk = panel.projectState(p, Date.now()).housekeeping;
  assert.deepEqual(JSON.parse(JSON.stringify(hk.options)), {
    ENV: ['dev', 'staging', 'prod'], MODEL_ID: ['MLP', 'OLD_CNN'], DATASET: ['ascad'],
    STATUS: ['done', 'failed'], BATCH: ['fig2', 'fig24'] });
  store['traceflow.deleteFilters'] = { P: { MODEL_ID: 'OLD_CNN' } };          // no longer in params.yaml
  hk = panel.projectState(p, Date.now()).housekeeping;
  assert.deepEqual([hk.error, hk.runs.map(r => r.name)], ['', ['beta']]);
  panel.stop();
});

test('a queue row counts only the runs the queue makes: never what params.yaml plans now, never another queue\'s sweep (fig10: 6, not 30; fig5: 8, not 40)', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = { ...project(tmp), queues: [{ key: 'fig10', commands: [{}, {}, {}, {}, {}, {}] },
    { key: 'fig5', commands: Array.from({ length: 8 }, () => ({})) }] };
  fs.mkdirSync(p.config_dir, { recursive: true });
  fs.writeFileSync(path.join(p.config_dir, 'params.yaml'), 'seed: 1\n');
  fs.writeFileSync(p.registry, JSON.stringify({ c8802d52: { model_name: 'ketchup_computer', status: 'DONE', env: 'dev', total_trials: 5,
    started_at: '2026-10-15T20:16:17+08:00', batches: ['fig10_optimizers', 'fig10'],
    batch_ids: ['fig10_optimizers-20261015T120942Z', 'fig10-20261019T232656Z'] } }));
  const python = path.join(tmp, 'python');                       // layout.py plan: params.yaml holds another queue's 5 optimizers
  fs.writeFileSync(python, '#!/bin/sh\necho \'{"per_command": 5}\'\n');
  fs.chmodSync(python, 0o755);
  const panel = new E.Panel({});
  Object.assign(panel, { root: tmp, python });
  panel.tick = () => {};
  for (let i = 0; i < 5; i++) {                                  // polls, as the panel makes them
    panel.projectState(p, Date.now());
    await new Promise(r => setTimeout(r, 50));
  }
  const [q10, q5] = panel.projectState(p, Date.now()).queues;
  assert.deepEqual([q10.planned, q10.done, q10.state], [6, 1, 'incomplete']);
  assert.deepEqual([q5.planned, q5.done, q5.state], [8, 0, 'not started']);
  panel.stop();
});

function plannerHost() {
  const panel = new E.Panel({ extensionPath: path.join(__dirname, '..') });
  panel.layout = { projects: [{ name: 'P', configs: [{ path: '/w/params.yaml' }] }] };
  panel.describeProject = async () => ({ framework: 'pytorch', input: [1, 700], models: [{ model_id: 'A', architecture_args: { layers: [] } }] });
  panel.menusLoaded = async () => ({ params: { models: { layers: { Conv1d: {} } } } });   // the layer forms, once choices.py answered
  const tab = () => {
    const tabPanel = { posted: [], disposed: false, title: '', reveal() {}, dispose() { this.disposed = true; }, onDidDispose() {},
      webview: { cspSource: 'CSP', asWebviewUri: u => ({ toString: () => `res:${path.basename(u.fsPath)}` }),
        onDidReceiveMessage(fn) { tabPanel.receive = fn; } } };
    tabPanel.webview.postMessage = m => { tabPanel.posted.push(m); return Promise.resolve(true); };
    return tabPanel;
  };
  return { panel, tab };
}

test('a planner tab VS Code brings back after a reload is wired again and told to restore its design', async () => {
  const { panel, tab } = plannerHost();
  const t = tab();
  await panel.restorePlanner(t, { project: 'P', file: '/w/params.yaml', modelId: 'A' });
  assert.equal(panel.planners.P.panel, t);
  assert.match(t.webview.html, /res:planner\.js/);
  assert.deepEqual(t.posted, [], 'nothing is sent before the page says it is ready (a message to a page still loading is lost)');
  await t.receive({ type: 'ready' });
  const init = t.posted.find(m => m.type === 'init');
  assert.equal(init.restore, true);
  assert.equal(init.file, '/w/params.yaml');
  assert.equal(init.models[0].model_id, 'A');
  assert.deepEqual(init.forms, { layers: { Conv1d: {} } }, 'the layer forms, waited for (a reload restores the tab before choices.py answers)');
  t.posted.length = 0;
  await t.receive({ type: 'ready' });                                       // the page loaded again: it restores what it kept
  assert.equal(t.posted.find(m => m.type === 'init').restore, true);
  t.posted.length = 0;
  await panel.initPlanner(panel.planners.P, '/w/params.yaml', 'A', false);   // Visualise on a tab already ready: sent at once
  assert.equal(t.posted.find(m => m.type === 'init').restore, false);
  const gone = tab();
  await panel.restorePlanner(gone, { project: 'NOPE', file: '/w/params.yaml' });
  assert.ok(gone.disposed, 'a tab of a project that is gone is closed');
});

test('a planner stage waits until the panel shows that params.yaml form, then follows its tree', () => {
  const { panel } = plannerHost();
  const got = [];
  panel.post = (target, m) => got.push([target, m]);
  panel.postStage('/w/params.yaml', { type: 'stage', model_id: 'A' });
  assert.deepEqual(got, []);
  panel.view = { id: 'view' };
  panel.openTrees.add('/w/params.yaml');
  panel.flushStages('/w/params.yaml');
  assert.deepEqual(got.map(([, m]) => m.model_id), ['A']);
  panel.flushStages('/w/params.yaml');
  assert.equal(got.length, 1);
  panel.postStage('/w/params.yaml', { type: 'stage', model_id: 'B' });
  assert.deepEqual(got.map(([, m]) => m.model_id), ['A', 'B']);
});

test('menus() passes the blueprint of choices static to the panel (the keys a file lacks are drawn from it)', () => {
  const panel = new E.Panel({ extensionPath: path.join(__dirname, '..') });
  panel.menuRules = { rules: { r: 1 }, graph: { g: 1 }, params: { p: 1 }, blueprint: { files: { 'params.yaml': {} } }, errors: [] };
  const m = panel.menus();
  assert.deepEqual([m.rules, m.graph, m.params, m.blueprint], [{ r: 1 }, { g: 1 }, { p: 1 }, { files: { 'params.yaml': {} } }]);
  assert.match(panel.build(), /^0\.1\.\d+ [0-9a-f]{10}$/);
});
