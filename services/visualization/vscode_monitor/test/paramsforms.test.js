// services/visualization/vscode_monitor/test/paramsforms.test.js
'use strict';
// The params.yaml form (the framework's deep_learning block only, [+], a list keeps one value, true / false
// in any case, a folder button on every address, the monitor menu from metrics_to_log, the loss by
// framework) and the FIND MODEL form of model_filter.yaml (every field the runs hold, ticked to a condition).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const { execFileSync } = require('child_process');
const E = require('./extension_stub');
const M = require('../webview/menus');

const ROOT = path.resolve(__dirname, '../../../..');
const PY = { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], maxBuffer: 64 * 1024 * 1024,
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '', PYTHONPATH: [ROOT, path.join(ROOT, 'services/side_channel_core'), path.join(ROOT, 'services/side_channel_core/src')].join(':') } };
const lastJson = out => JSON.parse(out.trim().split('\n').pop());
const STATIC = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.choices', 'static'], PY));

function render(file, open, messages = [], extraState = {}, project = {}, top = {}, menus = {}) {   // top: beside projects in the state (master)
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const regions = {};
  const on = {};                                        // the webview's own listeners (change, click)
  const app = {
    set innerHTML(html) { for (const m of html.matchAll(/id="([^"]+)"/g)) regions[m[1]] = { innerHTML: '', contains: () => false }; },
    querySelectorAll: sel => (sel === '[id]' ? Object.entries(regions).map(([id, el]) => Object.assign(el, { id })) : []),
    addEventListener: (type, fn) => { on[type] = fn; },
  };
  const listeners = {};
  const sent = [];
  const k = p => `n:${file}:${JSON.stringify(p)}`;
  const state = { open: { 'p:P': true, 'c:P': true, [`cf:${file}`]: true, ...Object.fromEntries(open.map(p => [k(p), true])) }, ...extraState };
  const ctx = {
    acquireVsCodeApi: () => ({ postMessage: m => sent.push(m), getState: () => state, setState: () => {} }),
    document: { body: { dataset: { mode: 'main' } }, getElementById: () => app, activeElement: null },
    window: { addEventListener: (type, fn) => { listeners[type] = fn; }, TFMenus: M },
    console,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../webview/panel.js'), 'utf8'), ctx);
  listeners.message({ data: { type: 'tree', file, sha256: 's', tree, error: '', note: '' } });
  listeners.message({ data: { type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], ...top,
    menus: { rules: STATIC.rules, graph: STATIC.graph, params: STATIC.params, refs: {}, ...menus }, projects: [{
      name: 'P', env: 'dev', envSource: 'test', path: path.dirname(file), configDir: path.dirname(file), graphJobsDir: path.dirname(file),
      session: null, runSession: null, runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'idle' }, run: null, eta: null,
      tiles: { queued: [], running: [], done: [], failed: [] }, tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} },
      queues: [], figures: [], facets: [], configs: [{ file: path.basename(file), path: file }], errors: [], graphJobs: [], ...project }] } });
  for (const m of messages) listeners.message({ data: m });
  const now = () => Object.values(regions).map(r => r.innerHTML).join('\n');
  return { html: now(), sent, on, now, post: m => listeners.message({ data: m }) };
}
const at = p => `data-path="${JSON.stringify(p).replace(/"/g, '&quot;').replace(/[[\]]/g, '\\$&')}"`;

const PARAMS = `deep_learning:
  framework: keras
  pytorch:
    weight_init: default
  keras:
    weight_init: glorot_uniform
sca:
  leakage_model: [HW, ID]
  rank_base: 1
  candidate_ranking:
    mode: [none]
perform_attack:
  attack_step: [1]
validation:
  rank_step: [1]
data:
  - name: ascad
    type: ASCADf
    debug_h5_inspect: True
    database: "services/side_channel_core/data/ASCAD/x.h5"
    multilabel: 0
trainer:
  monitor_metric: guessing_entropy
  epochs: [20, 50]
  loss:
    type: NLLLoss
  optimizer:
    - type: RMSprop
      learning_rate: [1.0e-5]
metrics_to_log:
  - mean_attack_rank_curve
  - mean_accuracy
  - mean_fit_time_{s}
paths:
  model_graphs: artifacts/{{env}}/generated_models/{{model_name}}/graphs
model:
  - model_id: M_MLP
    architecture: MLP
    architecture_args:
      layers:
        - Flatten: {}
        - Linear:
            in_features: 700
            out_features: auto
  - model_id: M_WRONG
    architecture: MLP
    architecture_args:
      layers:
        - Conv1d:
            in_channels: 1
            out_channels: 4
            kernel_size: 3
  - model_id: M_NEW
    architecture: CNN
    architecture_args:
      layers: []
`;

test('params.yaml: the framework\'s block alone, [+], one value kept, True read, folder buttons, the monitor and the loss menus', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-pf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PARAMS);
  const { html } = render(file, [['deep_learning'], ['sca'], ['sca', 'candidate_ranking'], ['perform_attack'], ['validation'], ['data'], ['data', 0], ['trainer'], ['trainer', 'loss'], ['paths']]);
  // deep_learning: framework keras shows its keras block, not the pytorch one
  assert.match(html, /<summary[^>]*>keras <small>1<\/small><\/summary>/);
  assert.doesNotMatch(html, /<summary[^>]*>pytorch /);
  // [+] is a button; it opens one control for the new value. candidate_ranking.mode: one choice, no [+]
  const plus = p => new RegExp(`<button class="addopen" data-addopen="${JSON.stringify(p).replace(/"/g, '&quot;').replace(/[[\]]/g, '\\$&')}"[^>]*>\\+</button>`);
  for (const p of [['sca', 'leakage_model'], ['perform_attack', 'attack_step'], ['validation', 'rank_step'], ['trainer', 'epochs']]) assert.match(html, plus(p));
  assert.doesNotMatch(html, plus(['sca', 'candidate_ranking', 'mode']));
  assert.match(html, new RegExp(`<select class="val" data-op="set" ${at(['sca', 'candidate_ranking', 'mode', 0])}`));
  assert.doesNotMatch(html, /data-op="add"/);                                                  // nothing open yet
  // rank_base and multilabel are no longer settings: an older file naming them shows them read-only, with why
  assert.match(html, /<label[^>]*>rank_base<\/label><span class="readonly"[^>]*>1<\/span><small class="note hint">not a setting any more: TraceFlow ranks the keys from 1<\/small>/);
  assert.match(html, /<label[^>]*>multilabel<\/label><span class="readonly"[^>]*>0<\/span><small class="note hint">not a setting any more: single-label is the only mode<\/small>/);
  assert.doesNotMatch(html, new RegExp(`data-path="[^"]*rank_base`));
  // attack_step's one value has no ×, epochs' two have theirs
  assert.doesNotMatch(html, new RegExp(`data-op="remove" ${at(['perform_attack', 'attack_step', 0])}`));
  assert.match(html, new RegExp(`data-op="remove" ${at(['trainer', 'epochs', 0])}`));
  assert.match(html, new RegExp(`data-op="remove" ${at(['trainer', 'epochs', 1])}`));
  // debug_h5_inspect: True is true
  assert.match(html, new RegExp(`<select class="val" data-op="set" ${at(['data', 0, 'debug_h5_inspect'])}[^>]*>(?:(?!</select>)[^])*<option value="true" selected>true</option>`));
  // addresses: the folder button at the right end of the box
  assert.match(html, new RegExp(`${at(['data', 0, 'database'])} value="services/side_channel_core/data/ASCAD/x.h5"><button class="pathpick" data-pickpath="[^"]*" data-base="root" data-start="services/side_channel_core/data" data-kind="file"`));
  // paths: read only in the panel (Open in editor under its line); no folder button there
  assert.match(html, /<label[^>]*>model_graphs<\/label><span class="readonly"[^>]*>artifacts\/\{\{env\}\}\/generated_models\/\{\{model_name\}\}\/graphs<\/span>/);
  assert.doesNotMatch(html, /data-pickpath="\[&quot;paths&quot;/);
  // monitor_metric: none, the monitors of metrics_to_log, validation_objective; the file's guessing_entropy is not one
  const mon = html.match(new RegExp(`<select[^>]*${at(['trainer', 'monitor_metric'])}[^>]*>([^]*?)</select><small class="note hint"[^>]*>early stopping and best-validation checkpoints</small>`));
  assert.ok(mon, 'monitor menu with its hint');
  assert.deepEqual([...mon[1].matchAll(/<option value="([^"]*)"/g)].map(m => m[1]),
    ['guessing_entropy', 'none', 'mean_accuracy', 'accuracy', 'mean_fit_time_s', 'validation_objective']);
  assert.match(mon[1], /guessing_entropy \(not an option\)/);
  // the loss under keras: CrossEntropyLoss only (NLLLoss is PyTorch's)
  const loss = html.match(new RegExp(`<select[^>]*${at(['trainer', 'loss', 'type'])}[^>]*>([^]*?)</select>`))[1];
  // the file's NLLLoss and the other PyTorch losses stay in the menu, saying where they are; only CrossEntropyLoss is choosable
  assert.deepEqual([...loss.matchAll(/<option value="([^"]*)"(?![^>]*disabled)/g)].map(m => m[1]), ['NLLLoss', 'CrossEntropyLoss']);
  assert.match(loss, /<option value="NLLLoss" selected title="available in pytorch, not in the framework this run uses \(deep_learning.framework\)">NLLLoss \(available in pytorch\)/);
  assert.match(loss, /<option value="FocalLoss" disabled title="available in pytorch[^"]*">FocalLoss \(available in pytorch\)/);
});

test('EXPERIMENT SETTINGS: a key search beside Config folder; Enter shows the matching keys under their parent panels, × ends it', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-search-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PARAMS);
  const r = render(file, [], [], { open: { 'p:P': true, 'c:P': true } });     // the form itself closed
  assert.match(r.now(), /<button data-reveal="[^"]*"[^>]*>Config folder<\/button><span class="cfgsearch"><svg class="searchicon"[^]*?<input data-cfgsearch="P" value="" placeholder="search keys"/);
  assert.doesNotMatch(r.now(), /data-cfgsearchclear/, 'no × before a search');
  const key = (value, k = 'Enter') => r.on.keydown({ target: { dataset: { cfgsearch: 'P' }, value }, key: k, preventDefault: () => {} });
  key('WEIGHT_init');                                                          // any case, part or whole of a key
  let html = r.now();
  assert.match(html, /Experiment Settings <small>1 match of “WEIGHT_init”<\/small>/, 'the keras block\'s (the framework read)');
  assert.match(html, /<details class="cfg" data-key="cf:[^"]*params\.yaml" data-file="[^"]*" open><summary[^>]*>params\.yaml<small class="cfg-role">experiment parameters<\/small><small> 1 match<\/small>/);
  assert.match(html, /<div class="cfgsearching">/);
  assert.match(html, /data-path="\[&quot;deep_learning&quot;,&quot;keras&quot;,&quot;weight_init&quot;\]"/, 'the match, its parents opened');
  assert.doesNotMatch(html, /leakage_model|rank_step|attack_step/, 'keys that do not match are out of sight');
  assert.match(html, /<input data-cfgsearch="P" value="WEIGHT_init"[^>]*><button class="x" data-cfgsearchclear="P"/, 'the search stays, with its ×');
  key('no_such_key');
  assert.match(r.now(), /no key holds “no_such_key”/);
  const click = d => r.on.click({ target: { closest: sel => (sel === 'button' ? { dataset: d, closest: () => null } : null) } });
  click({ cfgsearchclear: 'P' });
  html = r.now();
  assert.match(html, /Experiment Settings <small>1 files<\/small>/, 'back to the full version');
  assert.match(html, /<details class="cfg" data-key="cf:[^"]*params\.yaml" data-file="[^"]*"><summary[^>]*>params\.yaml/, 'its form closed again, as it was');
  assert.doesNotMatch(html, /data-cfgsearchclear|cfgsearching/);
  assert.ok(r.sent.some(m => m.type === 'tree' && m.open === false), 'the form the search alone opened is let go');
});

test('EXPERIMENT SETTINGS: a search kept across a reload asks for every file again (no file stays at searching…)', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-search-reload-'));
  const file = path.join(dir, 'params.yaml');
  const other = path.join(dir, 'experiment_queue.yaml');
  fs.writeFileSync(file, PARAMS);
  fs.writeFileSync(other, 'experiments:\n  - label: a\n    class_weight: none\n');
  // the webview state survives a reload: the search is in it, the trees are not, and no Enter is pressed again
  const r = render(file, [], [], { open: { 'p:P': true, 'c:P': true }, cfgSearch: { P: 'weight' } },
    { configs: [{ file: 'params.yaml', path: file }, { file: 'experiment_queue.yaml', path: other }] });
  assert.match(r.now(), /experiment_queue\.yaml(<small class="cfg-role">[^<]*<\/small>)?<small> searching…<\/small>/, 'the file whose tree has not arrived');
  assert.ok(r.sent.some(m => m.type === 'tree' && m.file === other && m.open !== false), 'its tree asked for on the redraw');
  assert.equal(r.sent.filter(m => m.type === 'tree' && m.file === other && m.open !== false).length, 1, 'asked once');
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', other], PY)).tree;
  r.post({ type: 'tree', file: other, sha256: 's2', tree, error: '', note: '' });
  assert.doesNotMatch(r.now(), /searching…/);
  assert.match(r.now(), /Experiment Settings <small>2 matches of “weight”<\/small>/, 'both files searched');
});

test('FIND MODEL: every field the runs hold, grouped by where the filter finds it, ticked to a condition', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-pf-'));
  const file = path.join(dir, 'model_filter.yaml');
  fs.writeFileSync(file, 'model_filter:\n  match: all\n  status: DONE\n  where:\n    - path: batches\n      contains: fig26\n');
  const data = { runs: 3, operators: [], where: [{ path: 'batches', contains: 'fig26' }], fields: [
    { path: 'batches', source: 'registry', kind: 'list', runs: 3, varies: true, values: ['fig20', 'fig26'], more: 0 },
    { path: 'trainer.epochs', source: 'sweep', kind: 'number', runs: 3, varies: true, values: [20, 50, 100], more: 0 },
    { path: 'data.name', source: 'snapshot', kind: 'text', runs: 3, varies: true, values: ['ascad'], more: 0 }] };
  const first = render(file, [['model_filter']]);
  assert.ok(first.sent.some(m => m.type === 'filter-fields' && m.project === 'P'));           // asked once opened
  assert.match(first.html, /reading the fields your runs hold…/);
  const { html } = render(file, [['model_filter']], [{ type: 'filter-fields', project: 'P', data }]);
  assert.match(html, /<button data-findmodels="P" data-file="[^"]*model_filter.yaml"[^>]*>Find Models \(1 ticked\)<\/button>/);
  assert.doesNotMatch(html, /Find models<\/button>/);                                          // the old button: gone
  assert.match(html, /<summary[^>]*>Registry <small[^>]*>1 fields, 1 differ between runs · 1 ticked<\/small><\/summary>/);
  assert.match(html, /<div class="field mfrow on"><input type="checkbox" data-mfcheck="batches" checked><span class="readonly mf-path"[^>]*>batches<\/span><select class="val mf-op" data-mfop="0"[^>]*><option value="contains" selected[^>]*>contains<\/option><option value="count"[^>]*>count<\/option><option value="exists"[^>]*>exists<\/option><\/select><select class="val" data-mfsel="1" data-mfi="0"><option value="0">fig20<\/option><option value="1" selected>fig26<\/option><\/select><\/div>/);
  assert.match(html, /<div class="field mfrow varies"><input type="checkbox" data-mfcheck="trainer.epochs"><span class="readonly mf-path"[^>]*>trainer.epochs<\/span><small class="note mf-vals">20, 50, 100<\/small><\/div>/);
  assert.match(html, /<summary[^>]*>Snapshot <small[^>]*>1 fields, 1 differ between runs<\/small><\/summary><\/details>/);  // closed: not drawn
});


test('params.yaml: [+] opens one control, a menu of the options left (a box where none)', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-pf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PARAMS);
  const open = [['sca'], ['perform_attack']];
  const menu = render(file, open, [], { adding: { [JSON.stringify(['sca', 'leakage_model'])]: true } }).html;
  const m = menu.match(new RegExp(`<select class="add" data-op="add" ${at(['sca', 'leakage_model'])}[^>]*>([^]*?)</select>`));
  assert.ok(m, 'the menu of the new value');
  assert.deepEqual([...m[1].matchAll(/<option value="([^"]*)">([^<]*)/g)].map(x => x[2]), ['choose…', 'MSB']);   // HW, ID are there
  const box = render(file, open, [], { adding: { [JSON.stringify(['perform_attack', 'attack_step'])]: true } }).html;
  assert.match(box, new RegExp(`<input class="add" data-op="add" ${at(['perform_attack', 'attack_step'])} placeholder="the new value">`));
  assert.doesNotMatch(box, /data-addopen="\[&quot;perform_attack&quot;,&quot;attack_step&quot;\]"/);
  // the opened control and its × in one frame, on a line of its own (owner 2026-10-27: not a squeezed 5em box)
  assert.match(box, new RegExp(`<span class="addbox"><input class="add" data-op="add" ${at(['perform_attack', 'attack_step'])}[^>]*><button class="x" data-addclose="1"`));
  assert.match(menu, new RegExp(`<span class="addbox"><select class="add" data-op="add" ${at(['sca', 'leakage_model'])}`));
});


test('params.yaml: optimizers by framework, Add an Optimizer, metrics_to_log a menu of entries and the always-shown pick menu with [+]', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-pf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PARAMS);
  const open = [['trainer'], ['trainer', 'optimizer'], ['trainer', 'optimizer', 0], ['metrics_to_log']];
  const keras = STATIC.rules['params.yaml'].find(r => r.p === 'trainer.optimizer.*.type').by.map.keras;
  const opts = (html, p) => [...(html.match(new RegExp(`<select[^>]*${at(p)}[^>]*>([^]*?)</select>`)) || [, ''])[1].matchAll(/<option value="([^"]*)"/g)].map(m => m[1]);
  const { html } = render(file, open);
  // framework keras: its optimizers to choose, then PyTorch's own ones disabled, saying where they are
  const pytorchOnly = STATIC.rules['params.yaml'].find(r => r.p === 'trainer.optimizer.*.type').by.map.pytorch.filter(o => !keras.includes(o));
  assert.deepEqual(opts(html, ['trainer', 'optimizer', 0, 'type']), [...keras, ...pytorchOnly]);
  assert.ok(keras.includes('Lion') && !keras.includes('ASGD') && pytorchOnly.includes('ASGD'));
  assert.match(html, /<option value="ASGD" disabled title="available in pytorch, not in the framework this run uses \(deep_learning.framework\)">ASGD \(available in pytorch\)<\/option>/);
  // each optimizer is its own block: #i, its type and learning rates; no × on the only one; Add an Optimizer writes block #1
  assert.match(html, /<summary[^>]*>#0 <small>type: RMSprop · learning_rate 1.0e-5<\/small><\/summary>/);
  assert.match(html, /<button data-addoptimizer="1" data-path="\[&quot;trainer&quot;,&quot;optimizer&quot;\]"[^>]*>Add an Optimizer<\/button>/);
  assert.doesNotMatch(html, /data-addopt="1"/);                                                 // no type-first menu any more
  assert.match(html, new RegExp(`<select class="val" data-op="set" ${at(['metrics_to_log', 1])}`));     // an entry: a menu
  // under the entries: the menu of every metric not listed yet, always shown, and [+]
  const pick = html.match(/<select class="add wide" data-metricpick="\[&quot;metrics_to_log&quot;\]"[^>]*>([^]*?)<\/select><button class="qadd" data-metricadd="\[&quot;metrics_to_log&quot;\]"/);
  assert.ok(pick, 'the metric menu and [+]');
  const left = [...pick[1].matchAll(/<option value="([^"]+)"/g)].map(m => m[1]);
  assert.ok(left.includes('guessing_entropy') && left.includes('fit_time_{s}'));
  assert.ok(!left.includes('mean_accuracy') && !left.includes('mean_fit_time_{s}'));        // listed already
  assert.doesNotMatch(html, /data-addopen="\[&quot;metrics_to_log&quot;\]"|data-addclose="1"/);   // no [+] control to open or close: the menu is always there
});


const MODELS = `deep_learning:
  framework: FRAMEWORK
model:
  - model_id: MY_MLP
    architecture: MLP
    architecture_args:
      layers:
        - Flatten: {}
        - Linear:
            in_features: 700
            out_features: auto
  - model_id: WRONG_LABEL
    architecture: MLP
    architecture_args:
      layers:
        - Conv1d:
            in_channels: 1
            out_channels: 4
            kernel_size: 3
  - model_id: EMPTY
    architecture: CNN
    architecture_args:
      layers: []
`;

test('models: Add a Model, the architecture its layers make, each layer with its shape, ▲ ▼ ×, and [+] by group', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-pf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, MODELS.replace('FRAMEWORK', 'pytorch'));
  const L = i => ['model', i, 'architecture_args', 'layers'];
  const open = [['model'], ['model', 0], ['model', 0, 'architecture_args'], L(0), [...L(0), 1], ['model', 1], ['model', 2], ['model', 2, 'architecture_args'], L(2)];
  const check = { model_id: 'MY_MLP', dataset: 'ascad', input: [1, 700], params: 179456, error: null, at: null,
    layers: [{ index: 0, type: 'Flatten', out: [700] }, { index: 1, type: 'Linear', out: [256] }] };
  const { html } = render(file, open, [{ type: 'model-check', project: 'P', model_id: 'MY_MLP', data: check }],
    { adding: { [JSON.stringify(L(2))]: true } });
  assert.match(html, /<div class="buttons"><button data-addmodel="1"[^>]*>Add a Model<\/button><button data-planmodel="1"[^>]*>Plan a Model<\/button><button data-importmodel="1"[^>]*>Import a Model<\/button><\/div>/);
  // architecture: shown once a model has layers (and what they make); a menu while it has none
  assert.match(html, /<label[^>]*>architecture<\/label><span class="readonly"[^>]*>MLP<\/span><small class="note hint">its layers make a MLP<\/small>/);
  assert.match(html, /<span class="readonly"[^>]*>MLP<\/span><small class="note hint err">its layers make a CNN<\/small><button data-path="[^"]*" data-setto="CNN"[^>]*>Set CNN<\/button>/);
  const menu = html.match(new RegExp(`<select class="val" data-op="set" ${at(['model', 2, 'architecture'])}[^>]*>([^]*?)</select>`));
  assert.ok(menu && /<option value="CNN" selected>CNN<\/option>/.test(menu[1]) && /ResNet50_1D \(named\)/.test(menu[1]));
  // the layers: the check's shapes, ▲ ▼ ×, a layer opened shows its class's arguments
  assert.match(html, /<button data-checkmodel="MY_MLP"[^>]*>Check the model<\/button><small class="note">input \(1, 700\) from ascad · 179,456 parameters<\/small>/);
  assert.match(html, /<summary[^>]*>#0 Flatten <small><\/small><span class="shape"[^>]*>→ \(700\)<\/span><span class="end"><button data-movelayer="1"[^>]*>▼<\/button><button class="x" data-op="delete"[^>]*>×<\/button><\/span><\/summary>/);
  assert.match(html, /<summary[^>]*>#1 Linear <small>in_features 700 · out_features auto<\/small><span class="shape"[^>]*>→ \(256\)<\/span><span class="end"><button data-movelayer="-1"[^>]*>▲<\/button><button class="x"/);
  assert.match(html, /<label[^>]*>out_features<\/label><input class="val" data-op="set"[^>]*value="auto"[^>]*list="[^"]*"><datalist[^>]*><option value="auto">/);
  assert.match(html, /<label[^>]*>bias<\/label><select class="val absent" data-draft[^>]*><option value="true" selected>True<\/option>/);
  assert.match(html, /<label[^>]*>activation<\/label><select class="val absent" data-draft/);
  // [+] of the empty model: every layer core.model builds, by group (pytorch)
  const add = html.match(/<select class="add" data-addlayer="1"[^>]*>([^]*?)<\/select>/)[1];
  assert.deepEqual([...add.matchAll(/<optgroup label="([^"]+)"/g)].map(m => m[1]),
    ['Convolution', 'Pooling', 'Dense', 'Normalization', 'Dropout', 'Shape', 'Recurrent', 'Attention', 'Merge']);
  assert.match(add, /<option value="LSTM" disabled title="returns \(output/);          // core.model.LAYER_LIST_LIMITS: listed, not chosen
  // under keras: only the layers Keras builds, no named networks
  fs.writeFileSync(file, MODELS.replace('FRAMEWORK', 'keras'));
  const k = render(file, open, [], { adding: { [JSON.stringify(L(2))]: true } }).html;
  const kadd = k.match(/<select class="add" data-addlayer="1"[^>]*>([^]*?)<\/select>/)[1];
  assert.deepEqual([...kadd.matchAll(/<option value="([A-Za-z0-9]+)">/g)].map(m => m[1]).sort(),
    ['BatchNorm1d', 'Conv1d', 'Dropout', 'Flatten', 'Linear', 'Pooling', 'ZeroPad1d']);
  assert.doesNotMatch(k.match(new RegExp(`<select class="val" data-op="set" ${at(['model', 2, 'architecture'])}[^>]*>([^]*?)</select>`))[1], /named/);
});


test('models: Add a Model, the architecture its layers make, each layer\'s line and arguments, [+] by group, the check\'s shapes', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-pf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PARAMS);
  const L = i => ['model', i, 'architecture_args', 'layers'];
  const open = [['model'], ['model', 0], ['model', 0, 'architecture_args'], L(0), [...L(0), 1], ['model', 1], ['model', 1, 'architecture_args'],
    ['model', 2], ['model', 2, 'architecture_args'], L(2)];
  const check = { model_id: 'M_MLP', input: [1, 700], dataset: 'ascad', params: 179456, error: null, at: null,
    layers: [{ index: 0, type: 'Flatten', out: [700] }, { index: 1, type: 'Linear', out: [256] }] };
  const { html } = render(file, open, [{ type: 'model-check', project: 'P', model_id: 'M_MLP', data: check }]);
  const model = i => html.slice(html.indexOf(`data-key="n:${file}:[&quot;model&quot;,${i}]"`), i < 2 ? html.indexOf(`data-key="n:${file}:[&quot;model&quot;,${i + 1}]"`) : undefined);
  assert.match(html, /<div class="buttons"><button data-addmodel="1"[^>]*>Add a Model<\/button><button data-planmodel="1"[^>]*>Plan a Model<\/button><button data-importmodel="1"[^>]*>Import a Model<\/button><\/div>/);
  // the architecture: what the layers make, shown; flagged with a fix where the file says otherwise; a menu with no layers
  assert.match(model(0), /<label[^>]*>architecture<\/label><span class="readonly"[^>]*>MLP<\/span><small class="note hint">its layers make a MLP<\/small>/);
  assert.match(model(1), /<small class="note hint err">its layers make a CNN<\/small><button data-path="[^"]*architecture[^"]*" data-setto="CNN"[^>]*>Set CNN<\/button>/);
  const kinds = [...model(2).match(/<label[^>]*>architecture<\/label><select[^>]*>([^]*?)<\/select>/)[1].matchAll(/<option value="([^"]*)"/g)].map(m => m[1]);
  assert.deepEqual(kinds, ['CNN', 'HYBRID', 'MLP', 'TRANSFORMER']);                      // keras: no named network
  // layers: a line each, its values, its shape after the check, ▲ ▼ × at its right end
  const m0 = model(0);
  assert.match(m0, /Check the model<\/button><small class="note">input \(1, 700\) from ascad · 179,456 parameters<\/small>/);
  assert.match(m0, /<summary[^>]*>#0 Flatten <small><\/small><span class="shape"[^>]*>→ \(700\)<\/span><span class="end"><button data-movelayer="1"[^>]*>▼<\/button><button class="x" data-op="delete"/);
  assert.match(m0, /<summary[^>]*>#1 Linear <small>in_features 700 · out_features auto<\/small><span class="shape"[^>]*>→ \(256\)<\/span><span class="end"><button data-movelayer="-1"[^>]*>▲<\/button><button class="x"/);
  // an opened layer: every argument of its class; out_features offers auto; activation a menu
  const lin = m0.slice(m0.indexOf('#1 Linear'));
  for (const k of ['in_features', 'out_features', 'bias', 'activation']) assert.match(lin, new RegExp(`<label[^>]*>${k}</label>`));
  assert.match(lin, /<datalist id="[^"]*"><option value="auto">/);
  assert.match(lin, /<label[^>]*>bias<\/label><select class="val absent" data-draft[^>]*><option value="true" selected>True<\/option>/);
  // a new layer model: no layers yet, [+]; opened, the layers keras builds, by group
  assert.match(model(2), /<summary[^>]*>layers <small>0 layers<\/small><\/summary>[^]*?<button class="addopen" data-addopen="[^"]*"[^>]*>\+<\/button>/);
  const adding = render(file, open, [], { adding: { [JSON.stringify(L(2))]: true } }).html;
  const menu = adding.match(/<select class="add" data-addlayer="1"[^>]*>([^]*?)<\/select>/)[1];
  assert.deepEqual([...menu.matchAll(/<optgroup label="([^"]*)">/g)].map(m => m[1]), ['Convolution', 'Pooling', 'Dense', 'Normalization', 'Dropout', 'Shape']);
  const offered = [...menu.matchAll(/<option value="([^"]+)">/g)].map(m => m[1]);
  assert.deepEqual(offered.sort(), STATIC.params.models.keras.layers.slice().sort());           // keras: what it builds
});

test('menus offer only what the system builds: ConvTranspose padding, activations, schedulers by framework, attack bytes by data', () => {
  const L = STATIC.params.models.layers;
  const arg = (layer, k) => L[layer].args.find(a => a.k === k);
  assert.deepEqual([arg('Conv1d', 'padding').o, arg('ConvTranspose1d', 'padding').o], [['same', 'valid'], []]);   // ConvTranspose: numbers only
  assert.deepEqual(arg('ConvTranspose1d', 'padding_mode').o, ['zeros']);
  assert.ok(!arg('Linear', 'activation').o.includes('threshold') && arg('Linear', 'activation').o.includes('relu'));   // threshold needs its arguments
  const rule = p => STATIC.rules['params.yaml'].find(r => r.p === p);
  const sched = rule('trainer.optimizer.*.scheduler.type').by.map;
  assert.ok(sched.keras.includes('ExponentialDecay') && !sched.keras.includes('StepLR') && sched.pytorch.includes('StepLR'));
  assert.equal(rule('sca.attack_byte').ref, 'attack_bytes');
});

test('params.yaml: × on each model and data line while another is left; during a run a note, the form stays editable', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data:\n  - name: ascad\n    type: ASCADf\nmodel:\n  - model_id: A\n    architecture: MLP\n  - model_id: B\n    architecture: MLP\n');
  const { html } = render(file, [['data'], ['model']]);
  assert.match(html, new RegExp(`data-delentry="model" data-name="A" ${at(['model', 0])}`));
  assert.match(html, new RegExp(`data-delentry="model" data-name="B" ${at(['model', 1])}`));
  assert.doesNotMatch(html, /data-delentry="data"/);                // the last data entry keeps no ×
  assert.doesNotMatch(html, /class="note running"/);
  const why = 'A run of P is ongoing (make experiment PROJECT=P QUEUE=q): it reads the config frozen at its start, so changes apply to the next run.';
  const running = render(file, [['data'], ['model']], [], {}, { runNotes: { [file]: why } }).html;
  assert.match(running, /<p class="note running">A run of P is ongoing \(make experiment PROJECT=P QUEUE=q\): .*changes apply to the next run\.<\/p>/);
  assert.match(running, /<button class="x" data-delentry="model"/);
  assert.doesNotMatch(running, /<(input|select|button) disabled[^>]*data-(op|delentry|addopen|addmodel|addoptimizer|movelayer|pick)=/);  // nothing disabled
});

test('a value changed back drops its change: the bar goes; Reset undoes the changes, Cancel undoes them and closes the form', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'trainer:\n  monitor_metric: guessing_entropy\n  epochs: [20, 50]\n  learning_rate: 1e-05\n');
  const r = render(file, [['trainer']]);
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: sel => (sel === 'details[data-key]' ? { dataset: { key: `cf:${file}` } } : null) };
  const change = (p, value) => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(p) }, value, blur: () => {},
    closest: sel => (sel === '.form' ? form : null) } });
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button'
    ? { dataset: d, closest: s => (s === '.form' ? form : null) } : null) } });
  const bar = () => /class="pending"/.test(r.now());

  change(['trainer', 'monitor_metric'], 'accuracy');
  assert.match(r.now(), /1 unsaved change<\/span>.*data-apply="1".*>OK<.*data-discard="1".*>Cancel<.*data-reset="1".*>Reset</s);
  change(['trainer', 'monitor_metric'], 'guessing_entropy');           // changed back
  assert.ok(!bar(), 'the bar goes once the value is the file\'s again');
  change(['trainer', 'epochs', 1], '50.0');                             // the same number, written another way
  change(['trainer', 'learning_rate'], '0.00001');
  assert.ok(!bar());

  change(['trainer', 'monitor_metric'], 'accuracy');
  change(['trainer', 'epochs', 0], '30');
  assert.match(r.now(), /2 unsaved changes/);
  click({ reset: '1' });
  assert.ok(!bar(), 'Reset undoes the changes');
  assert.match(r.now(), /<div class="form" data-file="[^"]*params.yaml"/);   // the form stays open
  assert.ok(!r.sent.some(m => m.type === 'tree' && m.open === false));

  change(['trainer', 'monitor_metric'], 'accuracy');
  click({ discard: '1' });
  assert.ok(!bar(), 'Cancel undoes the changes');
  assert.doesNotMatch(r.now(), /<div class="form" data-file="[^"]*params.yaml"/);   // and closes the form
  assert.ok(r.sent.some(m => m.type === 'tree' && m.file === file && m.open === false));
});

test('QUEUES: Set Params and Set Queue open that file\'s form under Experiment Settings', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  const queue = path.join(dir, 'experiment_queue.yaml');
  fs.writeFileSync(file, 'seed: 42\n');
  fs.writeFileSync(queue, 'fig2:\n  - make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=A\n');
  const configs = [{ file: 'params.yaml', path: file }, { file: 'experiment_queue.yaml', path: queue }];
  const r = render(file, [], [], { open: { 'p:P': true, 'q:P': true } }, { configs });
  assert.match(r.html, /<button data-setconfig="[^"]*params.yaml" data-project="P"[^>]*>Set Params<\/button><button data-setconfig="[^"]*experiment_queue.yaml" data-project="P"[^>]*>Set Queue<\/button>/);
  assert.match(r.html, /<div class="buttons"><button data-setconfig=/);           // no Open in editor before them
  assert.doesNotMatch(r.html, /<div class="form" data-file="[^"]*params.yaml"/);       // Experiment Settings closed
  r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button'
    ? { dataset: { setconfig: file, project: 'P' }, closest: () => null } : null) } });
  assert.match(r.now(), /<details class="cfg" data-key="cf:[^"]*params.yaml" data-file="[^"]*params.yaml" open>/);
  assert.match(r.now(), /<div class="form" data-file="[^"]*params.yaml"/);           // its form, drawn
});

test('params.yaml data: Add a Dataset asks its name in the panel (checked as one types: form, not taken), then the blueprint\'s entry', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data:\n  - name: ascad\n    type: ASCADf\n  - name: ascad_d50\n    type: ASCADf\n');
  const r = render(file, [['data']]);
  assert.match(r.html, new RegExp(`<div class="buttons"><button data-adddataset="1" ${at(['data'])}[^>]*>Add a Dataset</button></div>`));
  assert.doesNotMatch(r.html, /data-askrow/);
  const form = { dataset: { file, sha: 's', project: 'P' } };
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button'
    ? { dataset: d, closest: s => (s === '.form' ? form : null) } : null) } });
  click({ adddataset: '1', path: JSON.stringify(['data']) });          // the row opens under the button
  assert.match(r.now(), /<div class="field ask" data-askrow="dataset"><label>data name<\/label><input class="val" data-askinput="dataset" value="" placeholder="DATASET= of make orchestrate[^"]*"[^>]*><button data-askok="dataset" disabled[^>]*>Add<\/button><button class="x" data-askclose="1"/);
  // typing: checked at once, the row's message and Add follow (no redraw)
  const msg = { textContent: '' };
  const ok = { disabled: true };
  const cls = new Set();
  const type = value => {
    const el = { dataset: { askinput: 'dataset' }, value, classList: { toggle: (c, on) => (on ? cls.add(c) : cls.delete(c)) },
      closest: s => (s === '[data-askrow]' ? { querySelector: q => (q === '.askerr' ? msg : ok) } : null) };
    r.on.input({ target: el });
    return el;
  };
  type('ascad_d50');
  assert.deepEqual([msg.textContent, ok.disabled, cls.has('bad')], ['ascad_d50 is a data entry already', true, true]);
  type('ascad d100');
  assert.equal(msg.textContent, 'letters, digits, _ . - (no spaces)');
  const el = type('ascad_d100');
  assert.deepEqual([msg.textContent, ok.disabled, cls.has('bad')], ['', false, false]);
  r.on.keydown({ key: 'Enter', preventDefault: () => {}, target: el });  // Enter adds it
  const sent = r.sent.filter(m => m.type === 'edit');
  assert.equal(sent.length, 1);
  assert.equal(JSON.stringify([sent[0].op, sent[0].path]), '["add",["data"]]');
  const entry = JSON.parse(sent[0].value);
  assert.equal(entry.name, 'ascad_d100');
  assert.deepEqual(Object.keys(entry), ['name', ...Object.keys(STATIC.params.data_template).filter(k => k !== 'name')]);
  assert.equal(entry.type, null);
  assert.doesNotMatch(r.now(), /data-askrow/);                         // the row closes
  r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['data', 0, 'type']) }, value: 'ASCADr', blur: () => {},
    closest: sel => (sel === '.form' ? form : null) } });
  click({ adddataset: '1', path: JSON.stringify(['data']) });
  type('ascad_d200');
  click({ askok: 'dataset' });                                         // a staged change: refused
  assert.equal(r.sent.filter(m => m.type === 'edit').length, 1);
  assert.match(r.now(), /refused: OK, Cancel or Reset the staged changes first, then add the dataset/);
});

test('Housekeeping: ENV, MODEL_ID, DATASET, STATUS, BATCH are menus of the runs\' values; NAME and PARAM are typed', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'seed: 42\n');
  const housekeeping = { runs: [], total: 2, error: '', note: '', filters: { MODEL_ID: 'OLD_CNN', BATCH: 'fig2,fig24' },
    options: { ENV: ['dev', 'staging', 'prod'], MODEL_ID: ['MLP', 'OLD_CNN'], DATASET: ['ascad'], STATUS: ['done', 'failed'], BATCH: ['fig2', 'fig24'] } };
  const { html } = render(file, [], [], { open: { 'p:P': true, 'h:P': true } }, { housekeeping });
  for (const k of ['ENV', 'MODEL_ID', 'DATASET', 'STATUS', 'BATCH']) assert.match(html, new RegExp(`<select data-delfilter="P\\|${k}"><option value=""`));
  for (const k of ['NAME', 'PARAM']) assert.match(html, new RegExp(`<input data-delfilter="P\\|${k}"`));
  assert.match(html, /<select data-delfilter="P\|MODEL_ID"><option value="">any<\/option><option value="MLP">MLP<\/option><option value="OLD_CNN" selected>OLD_CNN<\/option><\/select>/);
  // a typed list of alternatives (an older filter) stays chosen, as its own entry
  assert.match(html, /<select data-delfilter="P\|BATCH"><option value="">any<\/option><option value="fig2,fig24" selected>fig2,fig24<\/option>/);
  assert.match(html, /<select data-delfilter="P\|ENV"><option value="" selected>any<\/option>/);
});

test('sca.candidate_ranking: the modes say what they rank; top_k only with mode top_k, asked when the file lacks it', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'sca:\n  candidate_ranking:\n    mode: [none]\n    top_k: [5]\n');
  const r = render(file, [['sca'], ['sca', 'candidate_ranking']]);
  assert.match(r.html, /<option value="all">all \(rank all 256 key candidates\)<\/option>/);
  assert.match(r.html, /<option value="none" selected>none \(rank no candidates\)<\/option>/);
  assert.match(r.html, /<option value="top_k">top_k \(rank the k best candidates\)<\/option>/);
  assert.doesNotMatch(r.html, /title="sca › candidate_ranking › top_k"/);      // mode none: top_k is not read
  const form = { dataset: { file, sha: 's', project: 'P' } };
  r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['sca', 'candidate_ranking', 'mode', 0]) }, value: 'top_k',
    blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  assert.match(r.now(), /title="sca › candidate_ranking › top_k"/);

  fs.writeFileSync(file, 'sca:\n  candidate_ranking:\n    mode: [top_k]\n');                // no top_k yet
  const q = render(file, [['sca'], ['sca', 'candidate_ranking']]);
  assert.match(q.html, /<input class="val bad" data-draft="\[&quot;sca&quot;,&quot;candidate_ranking&quot;,&quot;top_k&quot;\]" data-t="list"[^>]*placeholder="required with mode top_k, e.g. 5">/);

  fs.writeFileSync(file, 'sca:\n  candidate_ranking:\n    mode: []\n    top_k: []\n');      // the blueprint's
  const b = render(file, [['sca'], ['sca', 'candidate_ranking']]);
  assert.match(b.html, new RegExp(`data-addopen="${JSON.stringify(['sca', 'candidate_ranking', 'mode']).replace(/"/g, '&quot;').replace(/[[\]]/g, '\\$&')}"`));
});

test('metrics_to_log: a metric that takes a value shows it, its default for a bare name; the pick menu shows the value box of one chosen, [+] adds both', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'metrics_to_log:\n  - mean_accuracy\n  - rank_k_success\n  - top_k_accuracy:\n      k: 7\n');
  const r = render(file, [['metrics_to_log']]);
  const mo = STATIC.params.metric_options;
  assert.deepEqual([mo.top_k_accuracy, mo.rank_k_success, mo.traces_to_ge_k, mo.ge_at_traces],
    [{ option: 'k', default: 5 }, { option: 'k', default: 1 }, { option: 'k', default: 1 }, { option: 'traces', default: 2000 }]);
  assert.match(r.html, new RegExp(`data-metricopt="rank_k_success" data-option="k" ${at(['metrics_to_log', 1])} value="" placeholder="1 \\(default\\)"`));
  assert.match(r.html, /<span class="readonly mname"[^>]*>top_k_accuracy<\/span><span class="mopt">k <input class="val" type="number" min="1" data-op="set" data-path="\[&quot;metrics_to_log&quot;,2,&quot;top_k_accuracy&quot;,&quot;k&quot;\]" value="7"/);
  assert.doesNotMatch(r.html, /data-metricopt="mean_accuracy"/);                 // takes no value

  const form = { dataset: { file, sha: 's', project: 'P' } };
  const ev = (dataset, value) => r.on.change({ target: { dataset, value, blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  // the menu of every metric not yet listed, always shown, with [+] (off until one is chosen)
  assert.match(r.now(), /<span class="mpick"><select class="add wide" data-metricpick="\[&quot;metrics_to_log&quot;\]"[^>]*><option value="">choose a metric…<\/option><option value="accuracy">accuracy<\/option>/);
  assert.doesNotMatch(r.now().match(/<select class="add wide" data-metricpick[^]*?<\/select>/)[0], /"mean_accuracy"|"rank_k_success"|"top_k_accuracy"/, 'listed ones left out');
  assert.match(r.now(), /<button class="qadd" data-metricadd="\[&quot;metrics_to_log&quot;\]" data-path="\[&quot;metrics_to_log&quot;\]" disabled[^>]*>\+<\/button>/);
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button'
    ? { dataset: d, closest: s => (s === '.form' ? form : null) } : null) } });
  ev({ metricpick: '["metrics_to_log"]' }, 'traces_to_ge_k');                     // chosen: [+] is on
  assert.match(r.now(), /<option value="traces_to_ge_k" selected>/);
  assert.match(r.now(), /data-metricadd="\[&quot;metrics_to_log&quot;\]" data-path="\[&quot;metrics_to_log&quot;\]" title="add traces_to_ge_k \(k 1\) to metrics_to_log \(OK writes it\)">\+<\/button>/);
  // a metric with a value: its box beside the menu, the default in it; [+] adds the metric with the value typed
  assert.match(r.now(), /<option value="traces_to_ge_k" selected>[^]*?<\/select><span class="mopt">k <input class="val" type="number" min="1" data-metricpickopt="\[&quot;metrics_to_log&quot;\]" value="1" size="7" title="traces_to_ge_k: its k \(default 1\)"><\/span><button class="qadd" data-metricadd/);
  assert.doesNotMatch(r.now(), /class="pending"/);
  ev({ metricpickopt: '["metrics_to_log"]' }, '3');
  assert.match(r.now(), /data-metricpickopt="\[&quot;metrics_to_log&quot;\]" value="3"/);
  click({ metricadd: '["metrics_to_log"]', path: '["metrics_to_log"]' });
  // staged: a row like a written one (its name, its value, ×), not a chip; the line's count includes it
  assert.match(r.now(), /<span class="readonly mname" title="to be written by OK">traces_to_ge_k<\/span><span class="mopt">k <input class="val dirty" type="number" min="1" data-stagedopt="add\|[^"]*" data-option="k" value="3"[^>]*><\/span><button class="x" data-unstage="add\|[^"]*"[^>]*>×<\/button>/);
  assert.doesNotMatch(r.now(), /data-metricask=|data-metricpickopt=|class="item added/, 'the pick menu is clear again');
  assert.match(r.now(), /<summary[^>]*>metrics_to_log <small>4<\/small><\/summary>/);
  ev({ metricpick: '["metrics_to_log"]' }, 'guessing_entropy');                  // one without a value: added at once
  click({ metricadd: '["metrics_to_log"]', path: '["metrics_to_log"]' });
  const row = r.now().match(/<select class="val dirty" data-stagedadd="(add\|[^"]*)"[^>]*>([^]*?)<\/select>/);
  assert.ok(row, 'a staged bare metric: a menu row');
  assert.match(row[2], /<option value="guessing_entropy" selected>/);
  assert.doesNotMatch(row[2], /"mean_accuracy"/, 'the listed ones are not offered');
  assert.doesNotMatch(r.now().match(/<select class="add wide" data-metricpick[^]*?<\/select>/)[0], /"guessing_entropy"/, 'an added one leaves the menu');
  assert.match(r.now(), /<summary[^>]*>metrics_to_log <small>5<\/small><\/summary>/);
  ev({ stagedadd: row[1].replace(/&quot;/g, '"') }, 'success_rate');             // the row's menu changes the addition (the id as the DOM gives it)
  assert.match(r.now(), /<option value="success_rate" selected>/);
  assert.doesNotMatch(r.now(), /<option value="guessing_entropy" selected>/);
  ev({ metricopt: 'rank_k_success', option: 'k', path: JSON.stringify(['metrics_to_log', 1]) }, '2');
  assert.match(r.now(), /<span class="readonly mname"[^>]*>rank_k_success<\/span><span class="mopt">k <input class="val dirty"[^>]*value="2"/);
  assert.match(r.now(), /3 unsaved changes/);
  const okMsg = () => r.sent.filter(m => m.type === 'apply').at(-1);
  r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: { apply: '1' }, closest: s => (s === '.form' ? form : null) } : null) } });
  assert.equal(JSON.stringify(okMsg().edits.filter(e => e.op === 'add').map(e => e.value)), JSON.stringify(['{"traces_to_ge_k":{"k":3}}', 'success_rate']));
});

test('MASTER CONTROL: master.yaml\'s form; folders and files read only, run-hash keys marked, × on a project but the default', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'master.yaml');
  fs.writeFileSync(file, 'secrets:\n  db_env_file: "secrets/db.env"\ndistributed:\n  enabled: False\nproject:\n' +
    '  - name: "A"\n    description: "first"\n    root: "A/"\n    params: "config/params.yaml"\n    scripts:\n      train: "src/sca/train.py"\n' +
    '    visualization:\n      clearml:\n        enabled: false\n        scope: "global"\n' +
    '  - name: "B"\n    root: "B/"\n    params: "config/params.yaml"\ndefault_project: "A"\ntimezone:\n  current_location: "Taipei"\n');
  const open = { 'p:P': true, [`cf:${file}`]: true, [`n:${file}:["project"]`]: true, [`n:${file}:["project",0]`]: true,
    [`n:${file}:["project",0,"scripts"]`]: true, [`n:${file}:["project",0,"visualization"]`]: true,
    [`n:${file}:["project",0,"visualization","clearml"]`]: true, [`n:${file}:["secrets"]`]: true, [`n:${file}:["distributed"]`]: true };
  const r = render(file, [], [], { open }, {}, { master: { path: file, lock: null } });
  assert.match(r.html, /<details class="cfg master" data-key="cf:[^"]*master.yaml"[^>]* open><summary[^>]*>Master Control <small>master.yaml<\/small><\/summary>/);
  assert.match(r.html, /<div class="buttons"><button data-addproject="1"[^>]*>Add a Project<\/button><button data-open="[^"]*master.yaml"[^>]*>Open in editor<\/button>/);
  for (const [k, v] of [['root', 'A/'], ['params', 'config/params.yaml'], ['name', 'A'], ['db_env_file', 'secrets/db.env']]) {
    assert.match(r.html, new RegExp(`>${k}</label><span class="readonly" title="[^"]+">${v.replace('.', '\\.')}</span>`), k);
  }
  assert.match(r.html, /data-path="\[&quot;project&quot;,0,&quot;scripts&quot;,&quot;train&quot;\]"[^>]*>(<\/select>)?<small class="note hint" title="joins the model_hash[^"]*">run hash<\/small>/);
  assert.match(r.html, /data-path="\[&quot;distributed&quot;,&quot;enabled&quot;\]"[\s\S]{0,400}?>run hash<\/small>/);
  assert.match(r.html, /<input class="val" data-op="set" data-path="\[&quot;project&quot;,0,&quot;description&quot;\]" value="first"/);   // editable
  assert.match(r.html, /<input class="val" data-op="set" data-path="\[&quot;project&quot;,0,&quot;visualization&quot;,&quot;clearml&quot;,&quot;scope&quot;\]" value="global" list=/);
  assert.match(r.html, /<button class="x" data-removeproject="B" data-path="\[&quot;project&quot;,1\]"/);
  assert.doesNotMatch(r.html, /data-removeproject="A"/);                // default_project keeps its entry
  // default_project: a menu of the projects (choices refs; this harness has none, so a box with its value)
  assert.deepEqual(STATIC.rules['master.yaml'].find(x => x.p === 'default_project'), { p: 'default_project', ref: 'projects' });
  assert.match(r.html, /data-path="\[&quot;default_project&quot;\]" value="A"/);

  const why = 'a run is ongoing (make experiment PROJECT=A QUEUE=q): master.yaml is read by every run and by make, not frozen with the run\'s config, so it stays as it is until the run ends.';
  const locked = render(file, [], [], { open }, {}, { master: { path: file, lock: why } }).html;
  assert.match(locked, /<small>master.yaml · read only: a run is ongoing<\/small>/);
  assert.match(locked, /<p class="err locked">Read only: a run is ongoing \(make experiment PROJECT=A QUEUE=q\)/);
  assert.match(locked, /<button data-addproject="1" data-file="[^"]*master.yaml" disabled/);
  assert.doesNotMatch(locked, /<(input|select|button)(?! disabled)[^>]*data-(op|removeproject)=/);
});

test('RUN SINGLE EXPERIMENT: [Add a single Run][Open in editor]; each line ENV= PROJECT= DATASET= MODEL_ID= menus, Start and × at its end; no [+] row', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'single_runs.yaml');
  fs.writeFileSync(file, 'runs:\n  - make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=A\n  - make orchestrate ENV=dev PROJECT=P DATASET=ascad\n');
  const project = {
    datasets: ['ascad', 'ascad_d50'], modelIds: ['A', 'B'], singleRunsFile: file, configs: [{ file: 'single_runs.yaml', path: file }],
    singleRuns: [{ index: 0, line: 'make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=A', ENV: 'dev', PROJECT: 'P', DATASET: 'ascad', MODEL_ID: 'A', error: '' },
      { index: 1, line: 'make orchestrate ENV=dev PROJECT=P DATASET=ascad', ENV: 'dev', PROJECT: 'P', DATASET: 'ascad', error: 'ENV, DATASET and MODEL_ID are needed' }],
  };
  const r = render(file, [], [], {}, project);
  assert.match(r.html, /<details class="cfg single" data-key="cf:[^"]*single_runs.yaml"[^>]* open><summary[^>]*>Run Single Experiment <small>2 runs<\/small><\/summary>/);
  const line0 = 'make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=A';
  // [Add a single Run][Open in editor] side by side
  assert.match(r.html, new RegExp(`<div class="buttons"><button data-addsingle="${line0}" data-file="[^"]*single_runs.yaml" data-sha="s"[^>]*>Add a single Run</button><button data-open="[^"]*single_runs.yaml"[^>]*>Open in editor</button>`));
  // make orchestrate ENV=[▾] PROJECT=[▾] DATASET=[▾] MODEL_ID=[▾] … Start ×
  assert.match(r.html, /<div class="field qline single-line" data-path="\[&quot;runs&quot;,0\]"><span class="qfix">make orchestrate<\/span><span class="qkv">ENV=<select[^>]*data-qvar="ENV">[\s\S]*?<\/select><\/span><span class="qkv">PROJECT=<select[^>]*data-qvar="PROJECT"><option value="">\(none\)<\/option><option value="P" selected>P<\/option><\/select><\/span><span class="qkv">DATASET=<select/);
  assert.match(r.html, /data-qvar="MODEL_ID"><option value="">\(none\)<\/option><option value="A" selected>A<\/option><option value="B">B<\/option><\/select><\/span><span class="end"><button data-run="single" data-form-project="1" data-key="0" title="[^"]*\(tmux\)">Start<\/button><button class="x" data-op="remove"/);
  assert.match(r.html, /<button data-run="single" data-form-project="1" data-key="1" disabled title="ENV, DATASET and MODEL_ID are needed">Start/);
  // the same file under Experiment Settings: its lines without Start (Run Single Experiment starts them)
  const settings = (r.html.match(/<details class="cfg" data-key="cf:[^"]*single_runs.yaml"[\s\S]*?<\/details>/) || [''])[0];
  assert.match(settings, /data-qvar="MODEL_ID"/);                               // drawn there, with its menus
  assert.equal(settings.includes('data-run="single"'), false);
  assert.equal((r.html.match(/data-run="single" data-form-project="1" data-key="0"/g) || []).length, 1);
  // no [+] row under the lines (Add a single Run is the one way to add one); nothing staged as text
  assert.doesNotMatch(r.html, /qadd-row|data-singleplus|Orchestrate…|data-qadd=/);

  const form = { dataset: { file, sha: 's', project: 'P' } };
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: s => (s === 'button'
    ? { dataset: d, closest: x => (x === '.form' ? form : null) } : null) } });
  click({ addsingle: line0, file, sha: 's' });                                  // written at once, a line of menus after
  assert.equal(JSON.stringify(r.sent.filter(m => m.type === 'edit').map(m => [m.op, m.path, m.value])),
    JSON.stringify([['add', ['runs'], line0]]));

  const row = { dataset: { path: JSON.stringify(['runs', 0]) }, closest: s => (s === '.form' ? form : null),
    querySelectorAll: () => [{ dataset: { qvar: 'ENV' }, value: 'prod' }, { dataset: { qvar: 'PROJECT' }, value: 'P' },
      { dataset: { qvar: 'DATASET' }, value: 'ascad_d50' }, { dataset: { qvar: 'MODEL_ID' }, value: 'B' }] };
  r.on.change({ target: { dataset: { qvar: 'ENV' }, value: 'prod', blur: () => {}, closest: s => (s === '.qline' ? row : s === '.form' ? form : null) } });
  assert.match(r.now(), /1 unsaved change/);
  assert.match(r.now(), /data-key="0" disabled title="OK the changes first: Start runs the line as the file holds it">Start/);
  click({ addsingle: line0, file, sha: 's' });                                  // refused while a change is staged
  assert.equal(r.sent.filter(m => m.type === 'edit').length, 1);
  assert.match(r.now(), /refused: OK, Cancel or Reset the staged changes first, then add the run/);
});

test('master.yaml timezone: country and city menus; a country picked loads its cities at once, its principal city staged with it', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'master.yaml');
  fs.writeFileSync(file, 'default_project: "A"\ntimezone:\n  current_country: "Taiwan"\n  current_location: "Taipei"\n');
  const open = { 'p:P': true, [`cf:${file}`]: true, [`n:${file}:["timezone"]`]: true };
  const r = render(file, [], [], { open }, {}, { master: { path: file, lock: null } });
  const at = k => `data-path="\\[&quot;timezone&quot;,&quot;${k}&quot;\\]"`;
  assert.match(r.html, new RegExp(`<select class="val" data-op="set" ${at('current_country')}[^>]*>[\\s\\S]*?<option value="Taiwan" selected>Taiwan</option>`));
  assert.match(r.html, new RegExp(`<select class="val" data-op="set" ${at('current_location')}[^>]*><option value="Taipei" selected>Taipei</option></select>`));
  const form = { dataset: { file, sha: 's', project: '' } };
  const pick = (k, v) => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['timezone', k]) }, value: v, blur: () => {},
    closest: s => (s === '.form' ? form : null) } });
  pick('current_country', 'Germany');                                   // before OK: the cities of Germany, Berlin staged
  assert.match(r.now(), new RegExp(`<select class="val dirty" data-op="set" ${at('current_location')}[^>]*><option value="Berlin" selected>Berlin</option><option value="Busingen">Busingen</option></select>`));
  assert.match(r.now(), /2 unsaved changes/);
  pick('current_location', 'Busingen');                                 // a city of the country is kept when the country changes again
  pick('current_country', 'Germany');
  assert.match(r.now(), new RegExp(`${at('current_location')}[^>]*><option value="Berlin">Berlin</option><option value="Busingen" selected>Busingen</option>`));
  pick('current_country', 'Taiwan');                                    // back: Taipei, the file's own pair, nothing to save
  assert.doesNotMatch(r.now(), /class="pending"/);
});

test('Add a Project / Add a Model ask in the panel: the row under the button, the model\'s kind a menu; Add sends the name', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const master = path.join(dir, 'master.yaml');
  fs.writeFileSync(master, 'project:\n  - name: "A"\n    root: "A/"\n    params: "config/params.yaml"\ndefault_project: "A"\n');
  const m = render(master, [], [], { open: { 'p:P': true, [`cf:${master}`]: true } }, {}, { master: { path: master, lock: null } });
  const click = (r, d, form) => r.on.click({ preventDefault: () => {}, target: { closest: s => (s === 'button'
    ? { dataset: d, closest: x => (x === '.form' ? form : null) } : null) } });
  click(m, { addproject: '1', file: master }, null);
  assert.match(m.now(), /<div class="buttons"><button data-addproject="1"[^>]*>Add a Project<\/button><button data-open=[^>]*>Open in editor<\/button><\/div><div class="field ask" data-askrow="project"><label>project name<\/label>/);
  m.on.change({ target: { dataset: { askinput: 'project' }, value: 'A' } });
  click(m, { askok: 'project' }, null);                                 // taken: nothing sent
  assert.equal(m.sent.filter(x => x.type === 'add-project').length, 0);
  m.on.change({ target: { dataset: { askinput: 'project' }, value: 'NEW_P' } });
  click(m, { askok: 'project' }, null);
  assert.deepEqual(JSON.parse(JSON.stringify(m.sent.filter(x => x.type === 'add-project'))), [{ type: 'add-project', name: 'NEW_P' }]);

  const params = path.join(dir, 'params.yaml');
  fs.writeFileSync(params, 'model:\n  - model_id: A\n    architecture: MLP\n');
  const r = render(params, [['model']]);
  const form = { dataset: { file: params, sha: 's', project: 'P' } };
  click(r, { addmodel: '1' }, form);
  assert.match(r.now(), /data-askrow="model"><label>model_id<\/label><input class="val" data-askinput="model"[^>]*><select class="val" data-askkind="1"[^>]*><option value="CNN" selected>CNN<\/option>/);
  r.on.change({ target: { dataset: { askinput: 'model' }, value: 'MY_MLP' } });
  r.on.change({ target: { dataset: { askkind: '1' }, value: 'MLP' } });
  click(r, { askok: 'model' }, form);
  const sent = r.sent.filter(x => x.type === 'add-model');
  assert.deepEqual([sent.length, sent[0].name, sent[0].kind, JSON.stringify(sent[0].ids)], [1, 'MY_MLP', 'MLP', '["A"]']);
});


// ── Plan a Model in the params.yaml form ───────────────────────────────────────────────────────────────
const PLAN = 'model:\n  - model_id: A\n    architecture: MLP\n    architecture_args:\n      strict_graph_args: true\n      layers:\n' +
  '        - Flatten: {}\n        - Linear: {in_features: 700, out_features: "auto"}\n  - model_id: R\n    architecture: ResNet50_1D\n    architecture_args: {}\n';
function planForm(open = [['model']], messages = [], extra = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PLAN);
  return { file, r: render(file, open, messages, extra) };
}
const clickIn = (r, d, form) => r.on.click({ preventDefault: () => {}, target: { closest: s => (s === 'button'
  ? { dataset: d, closest: x => (x === '.form' ? form : null) } : null) } });

test('the model list shows Add a Model and Plan a Model at the left and Visualise beside each model\'s ×', () => {
  const { file, r } = planForm();
  assert.match(r.html, /<div class="buttons"><button data-addmodel="1"[^>]*>Add a Model<\/button><button data-planmodel="1"[^>]*>Plan a Model<\/button><button data-importmodel="1"[^>]*>Import a Model<\/button><\/div>/);
  assert.match(r.html, /#0 <small>model_id: A<\/small><span class="end"><button data-visualise="A"[^>]*>Visualise<\/button><button class="x" data-delentry="model"/);
  const form = { dataset: { file, sha: 's', project: 'P' } };
  clickIn(r, { planmodel: '1' }, form);
  clickIn(r, { visualise: 'A' }, form);
  assert.deepEqual(JSON.parse(JSON.stringify(r.sent.filter(m => ['plan-model', 'visualise'].includes(m.type)))),
    [{ type: 'plan-model', project: 'P', file }, { type: 'visualise', project: 'P', file, model_id: 'A' }]);
});

test('Visualise is disabled for a named network', () => {
  const { r } = planForm();
  assert.match(r.html, /<button data-visualise="R" disabled title="R is a named network[^"]*"[^>]*>Visualise<\/button>/);
});

test('limited kinds and layers are listed disabled with their reasons', () => {
  const { file, r } = planForm([['model'], ['model', 0], ['model', 0, 'architecture_args']],
    [], { adding: { [JSON.stringify(['model', 0, 'architecture_args', 'layers'])]: true } });
  clickIn(r, { addmodel: '1' }, { dataset: { file, sha: 's', project: 'P' } });
  const html = r.now();
  assert.match(html, /<option value="HYBRID" disabled title="none of its layers builds[^"]*">HYBRID \(not available\)<\/option>/);
  assert.match(html, /<option value="TRANSFORMER" disabled title="none of its layers builds/);
});

test('the + layer menu lists the limited layers disabled with their reasons', () => {
  const layersKey = JSON.stringify(['model', 0, 'architecture_args', 'layers']);
  const { r } = planForm([['model'], ['model', 0], ['model', 0, 'architecture_args'], ['model', 0, 'architecture_args', 'layers']], [], { adding: { [layersKey]: true } });
  assert.match(r.html, /<option value="LSTM" disabled title="returns \(output/);
  assert.match(r.html, /<option value="Conv1d">Conv1d<\/option>/);
});

test('a staged arrange from the planner shows on its model and goes with OK', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PLAN);
  const edit = { op: 'arrange', path: ['model', 0, 'architecture_args', 'layers'], value: '[{"from":1},{"value":{"Dropout":{"p":0.2}}},{"from":0}]' };
  const r = render(file, [['model'], ['model', 0], ['model', 0, 'architecture_args'], ['model', 0, 'architecture_args', 'layers']], [{ type: 'stage', file, model_id: 'A', edit }]);
  assert.match(r.html, /rearranged in the planner: 1 added, 2 moved, 0 changed, 0 removed/);
  assert.match(r.html, /1 unsaved change/);
  assert.deepEqual(JSON.parse(JSON.stringify(r.sent.filter(m => m.type === 'staged').at(-1))), { type: 'staged', file, count: 1, arranged: 1 });
  clickIn(r, { apply: '1' }, { dataset: { file, sha: 's', project: 'P' } });
  const applied = r.sent.filter(m => m.type === 'apply').at(-1);
  assert.deepEqual(JSON.parse(JSON.stringify(applied.edits)), [edit]);
});

test('an empty model\'s architecture menu lists the limited kinds disabled with their reasons', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'model:\n  - model_id: E\n    architecture: CNN\n    architecture_args:\n      strict_graph_args: true\n      layers: []\n');
  const r = render(file, [['model'], ['model', 0]]);
  assert.match(r.html, /data-op="set"[^>]*title="a layer model[^"]*"><option value="CNN" selected>CNN<\/option><option value="HYBRID" disabled title="none of its layers builds/);
});


test('a planner arrange is not staged over staged edits of that model\'s layers, nor layer edits over a staged arrange', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PLAN);
  const OPEN = [['model'], ['model', 0], ['model', 0, 'architecture_args'], ['model', 0, 'architecture_args', 'layers']];
  const form = { dataset: { file, sha: 's', project: 'P' } };
  const change = (r, p, value) => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(p) }, value, blur: () => {},
    closest: sel => (sel === '.form' ? form : null) } });
  const L = ['model', 0, 'architecture_args', 'layers'];
  const arrange = { op: 'arrange', path: L, value: '[{"from":1},{"from":0}]' };
  const r = render(file, OPEN);
  change(r, [...L, 1, 'Linear', 'in_features'], '9');
  r.post({ type: 'stage', file, model_id: 'A', edit: arrange });
  assert.doesNotMatch(r.now(), /rearranged in the planner/);
  assert.equal(r.sent.filter(m => m.type === 'stage-refused').at(-1).model_id, 'A');
  const q = render(file, OPEN, [{ type: 'stage', file, model_id: 'A', edit: arrange }]);
  change(q, [...L, 1, 'Linear', 'in_features'], '9');
  assert.match(q.now(), /refused: the planner(&#39;|')s changes of these layers are staged/);
  assert.match(q.now(), /1 unsaved change/);
});

test('an unchanged arrange from the planner drops its staged change', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PLAN);
  const L = ['model', 0, 'architecture_args', 'layers'];
  const r = render(file, [['model'], ['model', 0], ['model', 0, 'architecture_args'], L],
    [{ type: 'stage', file, model_id: 'A', edit: { op: 'arrange', path: L, value: '[{"from":1},{"from":0}]' } },
     { type: 'stage', file, model_id: 'A', edit: { op: 'arrange', path: L, value: '[{"from":0},{"from":1}]' } }]);
  assert.doesNotMatch(r.html, /rearranged in the planner|unsaved change/);
});

test('Visualise is disabled for a model without a layer list (a graph model)', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PLAN + '  - model_id: G\n    architecture: CNN\n    architecture_args:\n      nodes: []\n      outputs: []\n');
  const r = render(file, [['model']]);
  assert.match(r.html, /<button data-visualise="G" disabled title="G has no layer list \(architecture_args.layers\)[^"]*"/);
  assert.match(r.html, /<button data-visualise="A" title=/);
});

test('evaluation.test.load_model best_validation without checkpointing: true is marked red; fixing either key clears it at once', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'evaluation:\n  validation:\n    checkpointing: false\n  test:\n    load_model: best_validation\n');
  const r = render(file, [['evaluation'], ['evaluation', 'validation'], ['evaluation', 'test']]);
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  const change = (p, value) => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(p) }, value, blur: () => {},
    closest: sel => (sel === '.form' ? form : null) } });
  const warned = () => /<small class="note hint err"[^>]*>needs checkpointing: true<\/small>/.test(r.now());
  assert.ok(warned(), 'best_validation with checkpointing: false');
  change(['evaluation', 'validation', 'checkpointing'], 'true');
  assert.ok(!warned(), 'checkpointing turned on');
  change(['evaluation', 'validation', 'checkpointing'], 'false');
  assert.ok(warned());
  change(['evaluation', 'test', 'load_model'], 'final_epoch');
  assert.ok(!warned(), 'final_epoch needs no checkpoint');
});

test('evaluation.validation.judge_log is a true / false menu, held empty or set', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  const menu = sel => new RegExp(`<select class="val" data-op="set" ${at(['evaluation', 'validation', 'judge_log'])}[^>]*>${sel}</select>`);
  fs.writeFileSync(file, 'evaluation:\n  validation:\n    judge_log:\n');
  let r = render(file, [['evaluation'], ['evaluation', 'validation']]);
  assert.match(r.html, menu('<option value="" selected>\\(empty\\)</option><option value="true">true</option><option value="false">false</option>'));
  fs.writeFileSync(file, 'evaluation:\n  validation:\n    judge_log: true\n');
  r = render(file, [['evaluation'], ['evaluation', 'validation']]);
  assert.match(r.html, menu('<option value="true" selected>true</option><option value="false">false</option>'));
});

test('test_config.yaml: attack_distribution is a menu of full_population and disjoint, also in one model\'s own settings', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'test_config.yaml');
  fs.writeFileSync(file, 'test_request:\n  model: []\n  perform_attack:\n    test:\n      attack_times: 100\n      shuffle: true\n' +
    '      attack_distribution: full_population\n  models:\n    model_a:\n      perform_attack:\n        test:\n          attack_distribution: disjoint\n');
  const own = ['test_request', 'models', 'model_a', 'perform_attack', 'test'];
  const { html } = render(file, [['test_request'], ['test_request', 'perform_attack'], ['test_request', 'perform_attack', 'test'],
    ['test_request', 'models'], ['test_request', 'models', 'model_a'], own.slice(0, 4), own]);
  for (const [p, picked] of [[['test_request', 'perform_attack', 'test', 'attack_distribution'], 'full_population'], [[...own, 'attack_distribution'], 'disjoint']]) {
    const menu = html.match(new RegExp(`<select[^>]*${at(p)}[^>]*>([^]*?)</select>`));
    assert.ok(menu, `a menu at ${p.join('.')}`);
    assert.match(menu[1], /<option value="disjoint"/);
    assert.match(menu[1], /<option value="full_population"/);
    assert.match(menu[1], new RegExp(`<option value="${picked}" selected>`));
  }
});

test('seed: one value keeps its box and a [+]; the value added shows beside it, a list shows its values', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'seed: 42   # CV: the folds take the next seeds\n');
  const one = render(file, []).html;
  assert.match(one, new RegExp(`<input class="val" data-op="set" ${at(['seed'])} value="42"`));
  assert.match(one, /data-addopen="\[&quot;seed&quot;\]"[^>]*>\+<\/button>/);
  const r = render(file, [], [], { adding: { [JSON.stringify(['seed'])]: true } });
  assert.match(r.html, new RegExp(`<input class="add" data-op="add" ${at(['seed'])} placeholder="the new value">`));
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  r.on.change({ target: { dataset: { op: 'add', path: JSON.stringify(['seed']) }, value: '43', blur: () => {},
    closest: sel => (sel === '.form' ? form : null) } });
  assert.match(r.now(), /<span class="item added">43<button class="x" data-unstage=/);
  fs.writeFileSync(file, 'seed: [42, 43]\n');
  const list = render(file, []).html;
  for (const i of [0, 1]) assert.match(list, new RegExp(`data-op="remove" ${at(['seed', i])}`));
  assert.match(list, /data-addopen="\[&quot;seed&quot;\]"[^>]*>\+<\/button>/);
});

test('params.yaml device: a menu of cuda, cpu and cuda:N per GPU, with helper text under it for the chosen one', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'device: "cuda"   # the run fails without a GPU\n');
  const gpu = { gpus: [{ index: 0, name: 'NVIDIA GeForce RTX 5090' }] };
  const r = render(file, [], [], {}, {}, { gpu });
  const field = () => (r.now().match(/<div class="field help-field">([^]*?)<\/div>/) || [])[1] || '';   // the label's tooltip spans lines
  assert.match(field(), new RegExp(`<select class="val" data-op="set" ${at(['device'])}[^>]*><option value="cuda" selected>cuda</option><option value="cpu">cpu</option><option value="cuda:0">cuda:0</option></select>`));
  assert.match(field(), /<small class="note help">Trains on the GPU: cuda is PyTorch(?:'|&#39;)s name for an NVIDIA GPU \(it has no device called gpu\); a run fails \(GPU FAILED\) when it is not usable: never a CPU fallback\. Part of the run hash: another value makes new runs\.<\/small>/);
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  const change = value => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['device']) }, value, blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  change('cpu');
  assert.match(field(), /<option value="cpu" selected>[^]*Trains on the CPU on purpose: no GPU needed/);
  change('cuda:0');
  assert.match(field(), /Trains on GPU 0 \(NVIDIA GeForce RTX 5090\) only; a run fails when it is not usable\. This machine(?:'|&#39;)s only GPU: the same as cuda, but a different run identity\./);
});

test('the share icon (top right) opens the status page settings: Start / Stop, the address, each person\'s link and QR code, Add a person', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'seed: 42\n');
  const r = render(file, []);
  assert.doesNotMatch(r.html, /share-icon|sharebar/, 'the button is on the TraceFlow Monitor title line (package.json traceflow.share)');
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: d, closest: () => null } : null) } });
  r.post({ type: 'share-toggle' });                                    // the title line's button
  assert.deepEqual({ ...r.sent.at(-1) }, { type: 'status-open' });
  assert.match(r.now(), /Reading the status page…/);
  r.post({ type: 'status-page', running: false, enabled: true, why: '', host: '192.168.88.2', port: 8765, auto: true, session: 'tf-status', note: '', error: '',
    shares: [{ name: 'Professor', created: '2026-10-21T12:00:00+00:00', link: 'http://192.168.88.2:8765/?t=abc', qr: '<svg class="segno"></svg>' }] });
  const html = r.now();
  assert.match(html, /<span class="light grey"><\/span>not serving <button data-statusstart="1" title="serve the page on port 8765: make status_share in tmux session tf-status/);
  // the address (found by itself, no host in master.yaml) and the port, read-only (no menu), with helper text
  assert.doesNotMatch(html, /data-statushost/);
  assert.match(html, /<div class="field"><span class="lbl">Port<\/span><span class="readonly" title="master\.yaml status_share\.port: edit it there by hand">8765<\/span><\/div>/);
  assert.doesNotMatch(html.replace(/<details class="person".*?<\/details>/g, ''), /192\.168\.88\.2/, 'the address only inside each person\'s link');
  // each person a panel of their own: name, date, × at the right end; inside only their QR code and link (click: open here; ⧉: copy)
  const person = (html.match(/<details class="person" data-key="share:Professor" open>(.*?)<\/details>/) || [])[1] || '';
  assert.match(person, /^<summary[^>]*><b>Professor<\/b><small>2026-10-21<\/small><button class="x" data-statusremove="Professor" title="withdraw Professor(?:'|&#39;)s link[^"]*">×<\/button><\/summary>/);
  assert.match(person, /<div class="qr"[^>]*><svg class="segno"><\/svg><\/div><div class="linkrow"><button class="link" data-statusvisit="http:\/\/192\.168\.88\.2:8765\/\?t=abc" title="open Professor(?:'|&#39;)s link in your browser \(VS Code forwards the port over this session\); ⧉ copies it, to send it">Click to share or open in external browser<\/button><button class="copy" data-statuscopy="http:\/\/192\.168\.88\.2:8765\/\?t=abc" data-name="Professor" title="copy Professor(?:'|&#39;)s link, to send it">⧉<\/button><\/div>$/);
  click({ statusvisit: 'http://192.168.88.2:8765/?t=abc' });
  assert.deepEqual({ ...r.sent.at(-1) }, { type: 'status-visit', link: 'http://192.168.88.2:8765/?t=abc' });
  assert.doesNotMatch(person, /data-statusname|share-help/, 'Add a person and the instructions are outside the people panels');
  assert.match(html, /<div class="people">.*<\/div><div class="field addperson">.*<ul class="share-help">/);
  assert.match(html, /<li><b>Home screen:<\/b> iPhone: Safari or Chrome → Share → Add to Home Screen \(Chrome: iOS 16\.4 or later\)/);
  assert.match(html, /<li><b>Open:<\/b> <ol><li>needs: the same network as the TraceFlow repo host machine, or its VPN with port 8765 allowed; nothing else reaches it<\/li><li>without a VPN: the phone on the same Wi-Fi or network as the TraceFlow repo host machine; scan the QR code or open the link<\/li><li>with a VPN: turn the VPN on, then scan the QR code or open the link<\/li><\/ol><\/li>/);
  assert.match(html, /<li><b>Read-only:<\/b> only observation of the system status, no interaction\.<\/li>/);
  assert.match(html, /<li><b>Not opening\?:<\/b> <ol><li>the device is on the TraceFlow repo host machine(?:'|&#39;)s network: same Wi-Fi, or VPN on<\/li><li>the firewall lets TCP 8765 in \(Ubuntu: sudo ufw allow 8765\/tcp\)<\/li><li>a VPN that passes only SSH: request IT admin to allow 8765<\/li><li>guide: services\/doc\/11_phone_status_share\.md<\/li><\/ol><\/li>/);
  click({ statusstart: '1' });
  assert.equal(r.sent.at(-1).type, 'status-start');
  click({ statusremove: 'Professor' });
  assert.deepEqual({ ...r.sent.at(-1) }, { type: 'status-remove', name: 'Professor' });
  r.on.keydown({ target: { dataset: { statusname: '1' }, value: ' Student ' }, key: 'Enter', preventDefault: () => {} });
  assert.deepEqual({ ...r.sent.at(-1) }, { type: 'status-add', name: 'Student' });
  r.post({ type: 'status-page', running: true, enabled: true, why: '', host: '10.0.0.5', port: 8765, auto: false, session: 'tf-status', shares: [], note: '' });
  assert.doesNotMatch(r.now(), /10\.0\.0\.5/, 'not shown');
  r.post({ type: 'status-page', running: true, enabled: true, why: '', host: '192.168.88.2', port: 8765, auto: true, session: 'tf-status', shares: [], note: '' });
  assert.match(r.now(), /<span class="light green"><\/span>serving on port 8765 <button data-statusstop="1"/);
  // off in master.yaml (absent, or enabled: false): it says why; no Start
  r.post({ type: 'status-page', running: false, enabled: false, why: 'master.yaml has no status_share block: the status page is off (optional)', shares: [] });
  assert.match(r.now(), /<p class="note">Off: master\.yaml has no status_share block: the status page is off \(optional\)\.<\/p>/);
  assert.doesNotMatch(r.now(), /Turn it on|share-help/);
  assert.doesNotMatch(r.now(), /data-statusstart/);
  r.post({ type: 'share-toggle' });
  assert.doesNotMatch(r.now(), /class="share"/);
});

test('the status page settings start collapsed after every load (an update, a reload), whatever the last session left', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'seed: 42\n');
  const r = render(file, [], [], { share: true });                     // the state a reload brings back: open
  assert.doesNotMatch(r.html, /class="share"|Reading the status page/);
});

test('the status page button is on the TraceFlow Monitor title line: a command with a glowing icon for light and dark themes, packed', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'package.json'), 'utf8'));
  const cmd = pkg.contributes.commands.find(c => c.command === 'traceflow.share');
  assert.match(cmd.title, /^Status page: a read-only view of the runs in any browser, a desktop's or a phone's/);
  assert.deepEqual(cmd.icon, { light: 'media/share-light.png', dark: 'media/share-dark.png' });
  assert.deepEqual(pkg.contributes.menus['view/title'], [{ command: 'traceflow.share', when: 'view == traceflow.panel', group: 'navigation' }]);
  const pack = fs.readFileSync(path.join(__dirname, '..', 'pack.py'), 'utf8');
  for (const v of ['dark', 'light']) {
    assert.match(pack, new RegExp(`"media/share-${v}\\.png": BRAND / "traceflow-share-${v}\\.png"`));
    const png = fs.readFileSync(path.join(__dirname, '..', 'brand', `traceflow-share-${v}.png`));
    assert.equal(png.readUInt32BE(16), 32, 'drawn by make_brand.py share_icons: 32 px wide');
    assert.equal(png.readUInt32BE(20), 32);
  }
});

test('under keras the scheduler menu lists OneCycleLR disabled, available in pytorch; best_validation with OneCycleLR shows its hint', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-oc-'));
  const file = path.join(dir, 'params.yaml');
  const write = (framework, scheduler, load) => fs.writeFileSync(file, `deep_learning:\n  framework: ${framework}\n` +
    `evaluation:\n  validation:\n    checkpointing: true\n  test:\n    load_model: ${load}\ntrainer:\n  optimizer:\n` +
    `    - type: Adam\n      learning_rate: [1.0e-3]\n      scheduler:\n        type: ${scheduler}\n`);
  const open = [['evaluation'], ['evaluation', 'test'], ['trainer'], ['trainer', 'optimizer'], ['trainer', 'optimizer', 0],
    ['trainer', 'optimizer', 0, 'scheduler']];
  write('keras', 'none', 'final_epoch');
  const keras = render(file, open).html;
  const menu = keras.match(new RegExp(`<select[^>]*${at(['trainer', 'optimizer', 0, 'scheduler', 'type'])}[^>]*>([^]*?)</select>`))[1];
  assert.match(menu, /<option value="OneCycleLR" disabled title="available in pytorch, not in the framework this run uses \(deep_learning.framework\)">OneCycleLR \(available in pytorch\)<\/option>/);
  assert.match(menu, /<option value="CosineDecay">CosineDecay<\/option>/);                       // Keras's own, choosable
  const hint = html => /<small class="note hint" title="best_validation with OneCycleLR is allowed, your choice\.[^"]*selection-biased[^"]*">with OneCycleLR: see hint<\/small>/.test(html);
  write('pytorch', 'OneCycleLR', 'best_validation');
  assert.ok(hint(render(file, open).html), 'best_validation with OneCycleLR');
  write('pytorch', 'OneCycleLR', 'final_epoch');
  assert.ok(!hint(render(file, open).html), 'final_epoch: no hint');
  write('pytorch', 'StepLR', 'best_validation');
  assert.ok(!hint(render(file, open).html), 'another scheduler: no hint');
});

test('the blueprint: a key the file lacks shows faded, empty, with its menu and hint; an empty true/false key is a true / false menu; ' +
     'Load Blueprint stages every missing key (empty, a path with its value), OK writes them', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-bp-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'deep_learning:\n  framework: pytorch\ntrainer:\n  early_stopping:\n  epochs: [20]\n');
  const r = render(file, [['deep_learning'], ['trainer'], ['trainer', 'loss']], [], {}, {}, {}, { blueprint: STATIC.blueprint });
  const html = r.html;
  // a key the file lacks: faded, empty, its hint in the label's title
  assert.match(html, new RegExp('<label title="trainer › precision\\nThe training loop fp32 \\| tf32 \\| default">precision</label>' +
    `<select class="val absent" data-draft="${JSON.stringify(['trainer', 'precision']).replace(/"/g, '&quot;').replace(/[[\]]/g, '\\$&')}" data-t="enum"`));
  // a key of a menu the file lacks: the menu, (empty) first
  const prec = html.match(/data-draft="\[&quot;trainer&quot;,&quot;precision&quot;\]"[^>]*>([^]*?)<\/select>/);
  assert.ok(prec, 'precision: a menu');
  assert.match(prec[1], /^<option value="" selected>\(empty\)<\/option>/);
  assert.deepEqual([...prec[1].matchAll(/<option value="([^"]+)"/g)].map(m => m[1]).sort(), ['default', 'fp32', 'tf32']);
  // a block the file lacks: shown, its keys inside; the other framework's block is not
  assert.match(html, /<summary[^>]*>loss <small><span class="absentnote"[^>]*>not in the file<\/span><\/small><\/summary>/);
  assert.match(html, /<summary[^>]*>pytorch <small>/);
  assert.doesNotMatch(html, /<summary[^>]*>keras /);
  // an empty true / false key (choices.BOOLEAN_KEYS): a true / false menu, (empty) selected
  assert.match(html, new RegExp(`<select class="val" data-op="set" ${at(['trainer', 'early_stopping'])}[^>]*><option value="" selected>\\(empty\\)</option><option value="true">true</option><option value="false">false</option></select>`));
  // Load Blueprint: a warning first (extension.js), nothing staged or written until its OK; then written at once
  assert.match(html, new RegExp(`data-complete="${file.replace(/[/.]/g, '\\$&')}"[^>]*>Load Blueprint<`));
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button'
    ? { dataset: d, closest: s => (s === '.form' ? { dataset: { file, sha: 's', project: 'P' } } : null) } : null) } });
  click({ complete: file });
  assert.deepEqual(JSON.parse(JSON.stringify(r.sent.filter(m => m.type === 'complete-ask'))), [{ type: 'complete-ask', file, unsaved: 0 }]);
  assert.doesNotMatch(r.now(), /unsaved change/, 'nothing staged before the warning is answered');
  assert.ok(!r.sent.some(m => m.type === 'apply'), 'nothing written before the warning is answered');
  r.post({ type: 'complete-ok', file });
  const ok = r.sent.find(m => m.type === 'apply');
  assert.ok(ok, 'OK on the warning writes at once');
  assert.match(r.now(), /\d+ values from the blueprint written; those in [^<]*\btrainer\b[^<]* join the run hash/);
  assert.doesNotMatch(r.now().match(/those in ([^<]*) join the run hash/)[1], /\bpaths\b|\bnum_workers\b/, 'paths and workers do not join it');
  const edits = ok.edits.map(e => `${e.op} ${JSON.stringify(e.path)}${e.key ? ` ${e.key}` : ''}=${e.value}`);
  assert.ok(edits.includes('put ["trainer"] precision='), 'a missing key, empty');
  assert.ok(!ok.edits.some(e => JSON.stringify(e.path) === '["trainer","early_stopping"]'), 'an empty key the file has stays as it is');
  const paths = ok.edits.find(e => e.op === 'put' && JSON.stringify(e.path) === '[]' && e.key === 'paths');
  assert.equal(JSON.parse(paths.value).logs, 'artifacts/{{env}}/generated_models/{{model_name}}/logs', 'a whole block with its values');
  const sca = ok.edits.find(e => e.op === 'put' && JSON.stringify(e.path) === '[]' && e.key === 'sca');
  assert.equal(JSON.parse(sca.value).candidate_ranking.top_k, undefined, 'top_k: asked with mode top_k only');
  assert.ok(edits.includes('set ["sca","key_enumeration","enabled"]='), 'an empty value of a put block is then written empty');
  assert.ok(!ok.edits.some(e => JSON.stringify(e.path).includes('keras') || e.key === 'keras'), 'not the other framework\'s block');
  assert.ok(!ok.edits.some(e => e.key === 'data' || e.key === 'model'), 'data entries and models: Add a Dataset / Add a Model');
});

test('a new project\'s params.yaml: empty data and model lists show Add a Dataset, Add a Model and Plan a Model; an empty metrics_to_log its menu and [+]; paths is read only with Open in editor', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-new-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data: []\nmodel: []\nmetrics_to_log:\ntrainer:\n  optimizer: []\npaths:\n  logs: "artifacts/{{env}}/x"\n');
  const { html } = render(file, [['data'], ['model'], ['metrics_to_log'], ['trainer'], ['trainer', 'optimizer'], ['paths']]);
  assert.match(html, /<summary[^>]*>data <small>0<\/small><\/summary>[^]*?<button data-adddataset="1"[^>]*>Add a Dataset<\/button>[^]*?<p class="note">none yet<\/p>/);
  assert.match(html, /<summary[^>]*>model <small>0<\/small><\/summary>[^]*?<button data-addmodel="1"[^>]*>Add a Model<\/button><button data-planmodel="1"[^>]*>Plan a Model<\/button><button data-importmodel="1"[^>]*>Import a Model<\/button>[^]*?<p class="note">none yet<\/p>/);
  assert.match(html, /<summary[^>]*>optimizer <small>0<\/small><\/summary>/);
  assert.doesNotMatch(html, /data-op="add" data-path="\[&quot;data&quot;\]"|data-op="add" data-path="\[&quot;model&quot;\]"/, 'never a row of values');
  // an empty metrics_to_log (null in the file): the metric menu and [+] (yamledit add makes the list)
  const metrics = html.slice(html.search(/<summary[^>]*>metrics_to_log/), html.search(/<summary[^>]*>trainer/));
  assert.match(metrics, /<select class="add wide" data-metricpick="\[&quot;metrics_to_log&quot;\]"[^>]*><option value="">choose a metric…<\/option><option value="accuracy">accuracy<\/option>/);
  assert.match(metrics, /<button class="qadd" data-metricadd="\[&quot;metrics_to_log&quot;\]" data-path="\[&quot;metrics_to_log&quot;\]" disabled title="choose a metric first">\+<\/button>/);
  assert.doesNotMatch(metrics, /data-op="set" data-path="\[&quot;metrics_to_log&quot;\]"/);
  // paths: Open in editor under its line, every key read only
  const paths = html.slice(html.search(/<summary[^>]*>paths/));
  assert.match(paths, /<summary[^>]*>paths <small>1<\/small><\/summary><div class="buttons"><\/div><div class="buttons"><button data-open="[^"]*params.yaml" title="paths is edited in params.yaml, not here: the standard run layout">Open in editor<\/button><\/div>/);
  assert.match(paths, /<label[^>]*>logs<\/label><span class="readonly" title="paths is edited in params.yaml \(Open in editor above\), not here">artifacts\/\{\{env\}\}\/x<\/span>/);
  assert.doesNotMatch(paths, /data-path="\[&quot;paths&quot;,&quot;logs&quot;\]"/);
});

test('model_judge.yaml: process fields are menus of the judge\'s processes, metric fields menus of the columns a run records (typing stays possible)', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-mj-'));
  const file = path.join(dir, 'model_judge.yaml');
  fs.writeFileSync(file, 'model_judge:\n  ranking:\n    primary:\n      process: test\n      metric: accuracy\n    tie_breakers:\n      - process: validation\n        metric: key_rank\n  report_metrics:\n    - {process: test, metric: traces_to_ge_1}\n');
  const { html } = render(file, [['model_judge'], ['model_judge', 'ranking'], ['model_judge', 'ranking', 'primary'], ['model_judge', 'ranking', 'tie_breakers'],
    ['model_judge', 'ranking', 'tie_breakers', 0], ['model_judge', 'report_metrics'], ['model_judge', 'report_metrics', 0]]);
  const sel = html.match(new RegExp(`<select class="val" data-op="set" ${at(['model_judge', 'ranking', 'primary', 'process'])}[^>]*>([^]*?)</select>`));
  assert.ok(sel, 'process: a menu');
  assert.deepEqual([...sel[1].matchAll(/<option value="([^"]+)"/g)].map(m => m[1]).sort(), ['cross_validation', 'test', 'test_budget', 'train', 'validation']);
  for (const p of [['model_judge', 'ranking', 'primary', 'metric'], ['model_judge', 'ranking', 'tie_breakers', 0, 'metric'], ['model_judge', 'report_metrics', 0, 'metric']]) {
    const m = html.match(new RegExp(`<input class="val" data-op="set" ${at(p)} value="[^"]*" list="([^"]+)">`));
    assert.ok(m, `${p.join('.')}: a box with suggestions`);
    const list = html.match(new RegExp(`<datalist id="${m[1]}">([^]*?)</datalist>`))[1];
    for (const name of ['guessing_entropy', 'traces_to_ge_1', 'top_5_accuracy', 'key_rank', 'key_rank_reported', 'key_enumeration_position', 'selected_epoch']) assert.match(list, new RegExp(`<option value="${name}">`));
  }
});

test('a project folder master.yaml does not declare is listed: master.yaml integration missing, what is present or deleted, Add to master.yaml', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-orphan-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'seed: 1\n');
  const r = render(file, [], [], {}, {}, { orphans: [{ name: 'OLD_P', root: '/r/projects/OLD_P', present: true, config: true, artifacts: false }] });
  assert.match(r.html, /<span class="light red" title="not in master.yaml"><\/span><b>OLD_P<\/b> <small>master.yaml integration missing<\/small>/);
  assert.match(r.html, /projects\/OLD_P is not a project of master.yaml \(its entry was removed, or never written\)\. Project root: present · config: present · artifacts: deleted\./);
  assert.match(r.html, /<button data-declare="OLD_P"[^>]*>Add to master.yaml<\/button>/);
  r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: { declare: 'OLD_P' }, closest: () => null } : null) } });
  assert.equal(JSON.stringify(r.sent.filter(m => m.type === 'add-project')), JSON.stringify([{ type: 'add-project', name: 'OLD_P' }]));
});

test('experiment_queue.yaml: an empty queue is a line like any other (its key, 0 CMD, × at the end) with a row of menus to fill; a choice stages the line as a row; no "+ add" box', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-q-'));
  const file = path.join(dir, 'experiment_queue.yaml');
  fs.writeFileSync(file, 'fig1: []\nfig2:\n  - make orchestrate ENV=dev PROJECT=P DATASET=ascad MODEL_ID=M\n');
  const { html } = render(file, [['fig1'], ['fig2']]);
  assert.match(html, /<summary[^>]*>fig1 <small>0 CMD<\/small><span class="end"><button class="x" data-delentry="queue" data-name="fig1" data-path="\[&quot;fig1&quot;\]"[^>]*>×<\/button><\/span><\/summary>/);
  assert.match(html, /<summary[^>]*>fig2 <small>1 CMD<\/small><span class="end"><button class="x" data-delentry="queue" data-name="fig2"/);
  assert.match(html, /<button class="qadd" data-qadd="1" data-path="\[&quot;fig1&quot;\]" data-value="make orchestrate ENV=dev PROJECT=P"[^>]*>\+<\/button>/);
  assert.doesNotMatch(html, /placeholder="\+ add"|data-op="add" data-path="\[&quot;fig1&quot;\]"/, 'no "+ add" box');
  // the empty queue's row to fill: ENV and PROJECT chosen, DATASET and MODEL_ID (none)
  const draft = html.match(/<div class="field qline draft" data-draftline="1" data-path="\[&quot;fig1&quot;\]"[^>]*><span class="qfix">make orchestrate<\/span>([^]*?)<\/div>/);
  assert.ok(draft, 'a row to fill');
  assert.match(draft[1], /ENV=<select class="val" data-qvar="ENV">[^]*?<option value="dev" selected>dev<\/option>/);
  assert.match(draft[1], /PROJECT=<select class="val" data-qvar="PROJECT">[^]*?<option value="P" selected>P<\/option>/);
  assert.match(draft[1], /DATASET=<select class="val" data-qvar="DATASET"><option value="" selected>\(none\)<\/option>/);
  assert.match(draft[1], /<button class="x" data-nodraft="\[&quot;fig1&quot;\]" title="no line: the queue stays empty \(\+ adds one\)">×<\/button>$/, '× at its end, as on every line');
  // a choice in it stages the line, drawn as a row of menus (dirty) with × that drops it; the queue's line counts it
  const r = render(file, [['fig1'], ['fig2']], [], {}, { datasets: ['ascad'], modelIds: ['M'] });
  const form = { dataset: { file, sha: 's', project: 'P' } };
  const row = { dataset: { draftline: '1', path: JSON.stringify(['fig1']) }, closest: s => (s === '.form' ? form : null),
    querySelectorAll: () => [{ dataset: { qvar: 'ENV' }, value: 'dev' }, { dataset: { qvar: 'PROJECT' }, value: 'P' },
      { dataset: { qvar: 'DATASET' }, value: 'ascad' }, { dataset: { qvar: 'MODEL_ID' }, value: '' }] };
  r.on.change({ target: { dataset: { qvar: 'DATASET' }, value: 'ascad', blur: () => {}, closest: s => (s === '.qline' ? row : s === '.form' ? form : null) } });
  assert.match(r.now(), /<summary[^>]*>fig1 <small>1 CMD<\/small>/);
  const stagedRow = r.now().match(/<div class="field qline dirty" data-staged="(add\|[^"]*)" data-path="\[&quot;fig1&quot;\]"><span class="qfix">make orchestrate<\/span>([^]*?)<button class="x" data-unstage="\1"[^>]*>×<\/button><\/div>/);
  assert.ok(stagedRow, 'the staged line as a row');
  assert.match(stagedRow[2], /DATASET=<select class="val" data-qvar="DATASET">[^]*?<option value="ascad" selected>/);
  assert.doesNotMatch(r.now(), /qline draft|class="item added/);
  assert.match(r.now(), /1 unsaved change/);
  // the row to fill's × folds it away; + still adds a line
  const q2 = render(file, [['fig1']]);
  q2.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: { nodraft: '["fig1"]' }, closest: () => null } : null) } });
  assert.doesNotMatch(q2.now(), /qline draft/);
  assert.match(q2.now(), /<button class="qadd" data-qadd="1" data-path="\[&quot;fig1&quot;\]"/);
  // its menus change the addition; OK writes the line
  const srow = { dataset: { staged: stagedRow[1].replace(/&quot;/g, '"'), path: JSON.stringify(['fig1']) }, closest: s => (s === '.form' ? form : null),
    querySelectorAll: () => [{ dataset: { qvar: 'ENV' }, value: 'prod' }, { dataset: { qvar: 'PROJECT' }, value: 'P' },
      { dataset: { qvar: 'DATASET' }, value: 'ascad' }, { dataset: { qvar: 'MODEL_ID' }, value: 'M' }] };
  r.on.change({ target: { dataset: { qvar: 'ENV' }, value: 'prod', blur: () => {}, closest: s => (s === '.qline' ? srow : s === '.form' ? form : null) } });
  r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: { apply: '1' }, closest: s => (s === '.form' ? form : null) } : null) } });
  const ok = r.sent.filter(m => m.type === 'apply').at(-1);
  assert.equal(JSON.stringify(ok.edits.map(e => [e.op, e.path, e.value])), JSON.stringify([['add', ['fig1'], 'make orchestrate ENV=prod PROJECT=P DATASET=ascad MODEL_ID=M']]));
});

test('trainer.early_stopping is a block: enabled a true / false menu, monitor_metric the monitor menu with its hint; key_enumeration.modes shows the chosen mode only', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-es-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'sca:\n  key_enumeration:\n    enabled:\n    mode: neighbour_sweep\n    stop_on_success:\n    modes:\n      local_hamming:\n        top_k: []\n' +
    '      neighbour_sweep:\n        hamming_radius: []\n      support_weighted:\n        top_k: []\ntrainer:\n  early_stopping:\n    enabled:\n    monitor_metric:\n    patience:\n    min_delta:\nmetrics_to_log: [accuracy]\n');
  const opens = [['sca'], ['sca', 'key_enumeration'], ['sca', 'key_enumeration', 'modes'], ['sca', 'key_enumeration', 'modes', 'local_hamming'],
    ['sca', 'key_enumeration', 'modes', 'neighbour_sweep'], ['sca', 'key_enumeration', 'modes', 'support_weighted'], ['trainer'], ['trainer', 'early_stopping']];
  const r = render(file, opens);
  assert.match(r.html, new RegExp(`<select class="val" data-op="set" ${at(['trainer', 'early_stopping', 'enabled'])}[^>]*><option value="" selected>\\(empty\\)</option><option value="true">true</option>`));
  const mon = r.html.match(new RegExp(`<select[^>]*${at(['trainer', 'early_stopping', 'monitor_metric'])}[^>]*>([^]*?)</select><small class="note hint"[^>]*>early stopping and best-validation checkpoints</small>`));
  assert.ok(mon, 'the monitor menu under the block');
  assert.deepEqual([...mon[1].matchAll(/<option value="([^"]*)"/g)].map(m => m[1]), ['', 'none', 'accuracy', 'validation_objective']);
  // the enumeration's line: the owner's note; the modes: the chosen one's block only
  assert.match(r.html, /<summary[^>]*>key_enumeration <small>4 · system works; mode needs development<\/small><\/summary>/);
  assert.match(r.html, /<summary[^>]*>neighbour_sweep <small>1<\/small><\/summary>/);
  assert.doesNotMatch(r.html, /<summary[^>]*>(local_hamming|support_weighted) /);
  // the mode changed (staged): the shown block follows; comparison shows every one
  const form = { dataset: { file, sha: 's', project: 'P' } };
  const change = (p, value) => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(p) }, value, blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  change(['sca', 'key_enumeration', 'mode'], 'local_hamming');
  assert.match(r.now(), /<summary[^>]*>local_hamming <small>1<\/small><\/summary>/);
  assert.doesNotMatch(r.now(), /<summary[^>]*>(neighbour_sweep|support_weighted) /);
  change(['sca', 'key_enumeration', 'mode'], 'comparison');
  for (const m of ['local_hamming', 'neighbour_sweep', 'support_weighted']) assert.match(r.now(), new RegExp(`<summary[^>]*>${m} <small>1</small></summary>`));
});

test('tooltips: every key, block line and item line names the key path on hover, never a comment', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-tip-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, '# the run\'s seed\nseed: 42   # inline too\n# the training loop\ntrainer:\n  epochs: [1]   # how long\n  precision:\ndata:\n  - name: a   # the entry\n');
  const { html } = render(file, [['trainer'], ['data'], ['data', 0]], [], {}, {}, {}, { blueprint: STATIC.blueprint });
  assert.match(html, /<label title="seed">seed<\/label>/);
  assert.match(html, /<summary title="trainer">trainer <small>2<\/small><\/summary>/);
  assert.match(html, /<label title="trainer › epochs">epochs<\/label>/);
  assert.match(html, /<label title="trainer › precision">precision<\/label>/, 'the blueprint\'s comment is not a tooltip either');
  assert.match(html, /<summary title="data #0">#0 <small>name: a<\/small>/);
  assert.match(html, /<label title="data › 0 › name">name<\/label>/);
  assert.ok(!/the run&#39;s seed|how long|the entry|the training loop/.test(html), 'no comment text in the form');
});

test('data[].trace_normalization is a union on its mode: the chosen mode\'s keys only, drafted when the entry lacks them; eps with zscore and minmax, feature_range with a minmax step', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-norm-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data:\n  - name: a\n    type: ASCADf\n    trace_normalization:\n      mode: none\n      eps: 1.0e-8\n');
  const r = render(file, [['data'], ['data', 0], ['data', 0, 'trace_normalization']], [], {}, {}, {}, { blueprint: STATIC.blueprint });
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  const field = k => new RegExp(`<label title="data › 0 › trace_normalization › ${k}(\\n[^"]*)?">${k}</label>`);   // a drafted key's title carries its hint
  const set = (p, value) => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(p) }, value, blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  const draft = (p, t, value) => r.on.change({ target: { dataset: { draft: JSON.stringify(p), t, orig: '' }, value, blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  assert.match(r.html, field('mode'));
  assert.doesNotMatch(r.html, field('eps'), 'mode none reads no eps: the file\'s eps is not drawn');
  set(['data', 0, 'trace_normalization', 'mode'], 'zscore');
  for (const k of ['mean', 'std']) assert.match(r.now(), new RegExp(`${field(k).source}<input class="val absent" data-draft="\\[&quot;data&quot;,0,&quot;trace_normalization&quot;,&quot;${k}&quot;\\]"`), `${k} drafted`);
  assert.match(r.now(), new RegExp(`${field('eps').source}<input class="val" data-op="set"`), 'eps: the file\'s own value');
  set(['data', 0, 'trace_normalization', 'mode'], 'minmax');
  for (const k of ['min', 'max', 'eps']) assert.match(r.now(), field(k));
  assert.doesNotMatch(r.now(), field('mean'));
  set(['data', 0, 'trace_normalization', 'mode'], 'per_trace_standard');
  for (const k of ['eps', 'min', 'max', 'mean', 'std', 'steps', 'feature_range']) assert.doesNotMatch(r.now(), field(k), `${k} hidden`);
  set(['data', 0, 'trace_normalization', 'mode'], 'per_trace_minmax');
  assert.match(r.now(), field('feature_range'));
  set(['data', 0, 'trace_normalization', 'mode'], 'fitted');
  assert.match(r.now(), new RegExp(`${field('steps').source}<input class="val absent" data-draft="[^"]*steps[^"]*" data-t="list"`), 'steps drafted as a list');
  assert.doesNotMatch(r.now(), field('feature_range'), 'no minmax step yet');
  draft(['data', 0, 'trace_normalization', 'steps'], 'list', 'standard, minmax');
  assert.match(r.now(), field('feature_range'), 'a minmax step reads feature_range');
  // OK writes the mode and the staged keys of that mode, nothing of the others
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: d, closest: s => (s === '.form' ? form : null) } : null) } });
  click({ apply: '1' });
  const ok = r.sent.find(m => m.type === 'apply');
  const edits = ok.edits.map(e => `${e.op} ${JSON.stringify(e.path)}${e.key ? ` ${e.key}` : ''}=${JSON.stringify(e.value)}`);
  assert.ok(edits.includes('set ["data",0,"trace_normalization","mode"]="fitted"'), edits.join('\n'));
  assert.ok(edits.some(e => e.startsWith('put ["data",0,"trace_normalization"] steps=') && e.includes('standard') && e.includes('minmax')), edits.join('\n'));
  assert.ok(!edits.some(e => / (mean|std|min|max|eps)=/.test(e)), 'nothing of the other modes');
});

test('data[].profiling_split shows test_size, shuffle and random_state only while enabled is true; the template writes enabled: false', () => {
  assert.deepEqual(JSON.parse(JSON.stringify(STATIC.params.data_template.profiling_split)),
    { enabled: false, test_size: null, shuffle: null, random_state: null });
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-split-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data:\n  - name: a\n    type: ASCADf\n    profiling_split:\n      enabled: false\n      test_size: 0.1\n');
  const r = render(file, [['data'], ['data', 0], ['data', 0, 'profiling_split']], [], {}, {}, {}, { blueprint: STATIC.blueprint });
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  const field = k => new RegExp(`<label title="data › 0 › profiling_split › ${k}(\\n[^"]*)?">${k}</label>`);
  const set = (p, value) => r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(p) }, value, blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  assert.match(r.html, field('enabled'));
  for (const k of ['test_size', 'shuffle', 'random_state']) assert.doesNotMatch(r.html, field(k), `${k} hidden while enabled is false`);
  set(['data', 0, 'profiling_split', 'enabled'], 'true');
  assert.match(r.now(), new RegExp(`${field('test_size').source}<input class="val" data-op="set"`), 'test_size: the file\'s own value');
  for (const k of ['shuffle', 'random_state']) assert.match(r.now(), field(k), `${k} drawn once enabled`);
  // switched back off, the drafted arguments go; on again, they arrive again with their defaults
  set(['data', 0, 'profiling_split', 'enabled'], 'false');
  set(['data', 0, 'profiling_split', 'enabled'], 'true');
  const click = d => r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: d, closest: s => (s === '.form' ? form : null) } : null) } });
  click({ apply: '1' });
  const ok = r.sent.find(m => m.type === 'apply');
  const edits = ok.edits.map(e => `${e.op} ${JSON.stringify(e.path)}${e.key ? ` ${e.key}` : ''}=${JSON.stringify(e.value)}`);
  assert.ok(edits.includes('set ["data",0,"profiling_split","enabled"]="true"'), edits.join('\n'));
  assert.ok(edits.some(e => e.startsWith('put ["data",0,"profiling_split"] shuffle=') && e.includes('true')), edits.join('\n'));
  assert.ok(edits.some(e => e.startsWith('put ["data",0,"profiling_split"] random_state=') && e.includes('0')), edits.join('\n'));
  assert.ok(!edits.some(e => / test_size=/.test(e)), 'the file\'s test_size is kept, not replaced by the default');
});

test('data[].profiling_split switched on in an entry that lacks the block: the switch and the three defaults are written', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-split-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data:\n  - name: a\n    type: ASCADf\n    profiling_split:\n      enabled: false\n');
  const r = render(file, [['data'], ['data', 0], ['data', 0, 'profiling_split']], [], {}, {}, {}, { blueprint: STATIC.blueprint });
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['data', 0, 'profiling_split', 'enabled']) }, value: 'true', blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: { apply: '1' }, closest: s => (s === '.form' ? form : null) } : null) } });
  const edits = r.sent.find(m => m.type === 'apply').edits.map(e => `${e.op} ${JSON.stringify(e.path)}${e.key ? ` ${e.key}` : ''}=${JSON.stringify(e.value)}`);
  for (const [k, v] of Object.entries(STATIC.params.profiling_split.defaults)) {
    assert.ok(edits.some(e => e.startsWith(`put ["data",0,"profiling_split"] ${k}=`) && e.includes(String(v))), `${k}: ${edits.join('\n')}`);
  }
  assert.deepEqual(STATIC.params.profiling_split.defaults, { test_size: 0.1, shuffle: true, random_state: 0 });
});

test('Add a Dataset writes trace_normalization with the mode alone: no eps until zscore or minmax is chosen', () => {
  assert.deepEqual(JSON.parse(JSON.stringify(STATIC.params.data_template.trace_normalization)), { mode: null });
  assert.deepEqual(JSON.parse(JSON.stringify(STATIC.params.normalization.keys)),
    { none: [], auto: [], per_trace_standard: [], zscore: ['mean', 'std', 'eps'], minmax: ['min', 'max', 'eps'], per_trace_minmax: ['feature_range'], fitted: ['steps', 'feature_range'] });
});

test('seed: a list under cross-validation says whether the repeats share a training seed, as it is typed', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-seed-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'seed: [42, 53, 64]\ncross_validation:\n  enabled: true\n  n_folds: 10\n');
  const r = render(file, [], [], {}, {}, {}, { blueprint: STATIC.blueprint });
  assert.match(r.html, /<small class="note hint ok"[^>]*>3 repeats, at least 11 apart: no shared training seed<\/small>/);
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['seed', 1]) }, value: '50', blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  assert.match(r.now(), /<small class="note hint err"[^>]*>seeds 42 and 50 are 8 apart: repeats need at least 11 \(n_folds \+ 1\), e\.g\. 42, 53, 64<\/small>/);
  fs.writeFileSync(file, 'seed: [42, 50]\ncross_validation:\n  enabled: false\n  n_folds: 10\n');
  assert.doesNotMatch(render(file, [], [], {}, {}, {}, { blueprint: STATIC.blueprint }).html, /class="note hint (ok|err)"/, 'no folds: any seeds');
});

test('sca.attack_byte: the bytes no data entry defines are in the menu, never choosable, with why; the hint says which entries each byte serves', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-byte-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'sca:\n  attack_byte: [2]\ndata:\n  - name: a\n    type: DPAv4\n  - name: b\n    type: AES_HD\n');
  const refs = { P: { attack_bytes: ['0', '11'], attack_bytes_others: { 2: 'not defined for any data entry of this project: DPAv4 (1 entry), AES_HD (1 entry)' },
    attack_bytes_by_entry: { a: [0], b: [11] } } };
  const r = render(file, [['sca']], [], {}, {}, {}, { blueprint: STATIC.blueprint, refs });
  const menu = (r.html.match(new RegExp(`<select[^>]*${at(['sca', 'attack_byte', 0])}[^>]*>([^]*?)</select>`)) || [])[1] || '';
  // every entry defines one byte: a value in the file that differs from the entries' reads as text with its ×; no menu, no chooser
  assert.equal(menu, '', 'no menu');
  assert.match(r.html, /<span class="readonly[^"]*" title="as the file sets it; from data.type at the time of the run">2<\/span><\/span><\/span>/, 'the value as words, no ×');
  // the file's byte is the one every entry gives anyway: nothing drawn but the hint
  const same = { P: { ...refs.P, attack_bytes_by_entry: { a: [2], b: [2] } } };
  const sameHtml = render(file, [['sca']], [], {}, {}, {}, { blueprint: STATIC.blueprint, refs: same }).html;
  assert.doesNotMatch(sameHtml, /class="readonly[^"]*"[^>]*>2<\/span>/, 'the redundant 2 is not drawn');
  assert.match(sameHtml, />from data.type at the time of the run · byte 2: 2 entries<\/small>/);
  const choice = { P: { ...refs.P, attack_bytes_by_entry: { a: [0], c: Array.from({ length: 16 }, (_, i) => i) } } };
  const menu2 = (render(file, [['sca']], [], {}, {}, {}, { blueprint: STATIC.blueprint, refs: choice }).html.match(new RegExp(`<select[^>]*${at(['sca', 'attack_byte', 0])}[^>]*>([^]*?)</select>`)) || [])[1] || '';
  assert.match(menu2, /<option value="2" selected title="not defined for any data entry of this project: DPAv4 \(1 entry\), AES_HD \(1 entry\)">2 \(not defined for any data entry[^<]*<\/option>/, 'an entry allowing several bytes: the menu, with the unavailable ones marked');
  assert.match(menu2, /<option value="0">0<\/option><option value="11">11<\/option>/);
  assert.match(r.html, /<small class="note hint" title="from data.type at the time of the run\nbyte 0: a\nbyte 11: b">byte 0: 1 entry · byte 11: 1 entry<\/small>/);
  // empty: the hint says the byte follows the entry of the run
  fs.writeFileSync(file, 'sca:\n  attack_byte: []\ndata:\n  - name: a\n    type: DPAv4\n  - name: b\n    type: AES_HD\n');
  const empty = render(file, [['sca']], [], {}, {}, {}, { blueprint: STATIC.blueprint, refs }).html;
  assert.match(empty, />from data.type at the time of the run · byte 0: 1 entry · byte 11: 1 entry<\/small>/);
  // every entry defines one byte: the key stays, its chooser goes (a run takes its entry's byte)
  assert.doesNotMatch(empty, /data-addopen="\[&quot;sca&quot;,&quot;attack_byte&quot;\]"/, 'no chooser');
  const several = { P: { ...refs.P, attack_bytes_by_entry: { a: [0], c: Array.from({ length: 16 }, (_, i) => i) } } };
  assert.match(render(file, [['sca']], [], {}, {}, {}, { blueprint: STATIC.blueprint, refs: several }).html,
    /data-addopen="\[&quot;sca&quot;,&quot;attack_byte&quot;\]"/, 'an entry allowing several bytes: the chooser stays');
});

test('data[].type: beside it, type.attack_byte = {…}, the bytes the type defines (the entry\'s own as the file holds it; the type\'s for a type just chosen)', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-tb-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data:\n  - name: a\n    type: DPAv4\n  - name: b\n    type: AES_RD\n');
  const refs = { P: { attack_bytes: ['0'], attack_bytes_others: {}, attack_bytes_by_entry: { a: [0], b: [0] } } };
  const r = render(file, [['data'], ['data', 0], ['data', 1]], [], {}, {}, {}, { blueprint: STATIC.blueprint, refs });
  assert.match(r.html, new RegExp(`<select[^>]*${at(['data', 0, 'type'])}[^>]*>[^]*?</select><small class="note hint" title="the byte\\(s\\) this type defines a target for[^"]*">type.attack_byte = \\{0\\}</small>`));
  assert.equal(STATIC.params.type_bytes.AES_RD.join(), '0');
  assert.equal(STATIC.params.type_bytes.CHES_CTF_2018, 'any');
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['data', 1, 'type']) }, value: 'CHES_CTF_2018', blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  assert.match(r.now(), /type.attack_byte = \{0\.\.15\}<\/small>/, 'a type just chosen: its own bytes');
});

test('the judge metric menu offers every recorded metric (key_rank among them) and the run reader\'s own columns', () => {
  const list = (STATIC.rules['model_judge.yaml'] || []).filter(r => r.p === 'model_judge.ranking.primary.metric').flatMap(r => r.o || []);
  for (const name of ['key_rank', 'key_rank_reported', 'key_enumeration_position', 'selected_epoch', 'fit_time_s']) assert.ok(list.includes(name), name);
});

test('a written null is shown and kept: seed: null reads null (an unseeded run), an empty key reads empty; enabling profiling_split keeps random_state: null', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-null-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'seed: null\ndevice:\ndata:\n  - name: a\n    type: ASCADf\n    profiling_split:\n      enabled: false\n      random_state: null\n');
  const r = render(file, [['data'], ['data', 0], ['data', 0, 'profiling_split']], [], {}, {}, {}, { blueprint: STATIC.blueprint });
  assert.match(r.html, new RegExp(`<input class="val" data-op="set" ${at(['seed'])} value="null"`), 'seed: null shows null');
  const form = { dataset: { file, sha: 's', project: 'P' }, closest: () => null };
  r.on.change({ target: { dataset: { op: 'set', path: JSON.stringify(['data', 0, 'profiling_split', 'enabled']) }, value: 'true', blur: () => {}, closest: sel => (sel === '.form' ? form : null) } });
  r.on.click({ preventDefault: () => {}, target: { closest: sel => (sel === 'button' ? { dataset: { apply: '1' }, closest: s => (s === '.form' ? form : null) } : null) } });
  const edits = r.sent.find(m => m.type === 'apply').edits.map(e => `${e.op} ${JSON.stringify(e.path)}${e.key ? ` ${e.key}` : ''}=${JSON.stringify(e.value)}`);
  assert.ok(!edits.some(e => /random_state/.test(e)), `the written random_state: null is kept: ${edits.join('\n')}`);
  assert.ok(edits.some(e => / test_size=/.test(e)) && edits.some(e => / shuffle=/.test(e)), 'the absent ones arrive with their defaults');
});
