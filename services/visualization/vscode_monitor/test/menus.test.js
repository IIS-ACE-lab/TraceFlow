// services/visualization/vscode_monitor/test/menus.test.js
'use strict';
// The config forms' menus (webview/menus.js) with the rules choices.py prints.
const test = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('child_process');
const path = require('path');
const M = require('../webview/menus');
const E = require('./extension_stub');

const ROOT = path.resolve(__dirname, '../../../..');
const run = args => JSON.parse(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.choices', ...args], {
  cwd: ROOT, env: { ...process.env, CUDA_VISIBLE_DEVICES: '', PYTHONPATH: [ROOT, path.join(ROOT, 'services/side_channel_core'), path.join(ROOT, 'services/side_channel_core/src')].join(':') },
  encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim().split('\n').pop());
const menus = { rules: run(['static']).rules, refs: run(['refs', '--root', ROOT]) };
const G = '/x/projects/DLSCA_ASCAD/config/graph_config.yaml';
const P = '/x/projects/DLSCA_ASCAD/config/params.yaml';

test('choices.py reads every option list from the code, without errors', () => {
  assert.deepEqual(run(['static']).errors, []);
});

test('a graph job\'s type and plot_type are the validator\'s own sets', () => {
  assert.deepEqual(M.options(menus, G, 'DLSCA_ASCAD', null, ['jobs', 3, 'type']).opts.sort(),
    ['model_explanation', 'model_overlay', 'process_overlay', 'solo']);
  assert.ok(M.options(menus, G, 'DLSCA_ASCAD', null, ['jobs', 0, 'plot_type']).opts.includes('barh'));
  assert.equal(M.options(menus, G, 'DLSCA_ASCAD', null, ['jobs', 0, 'name']), null);          // free text: no menu
});

test('processes follow the job\'s source', () => {
  const tree = { k: 'map', entries: [{ key: 'jobs', node: { k: 'list', items: [
    { k: 'map', entries: [{ key: 'source', node: { k: 'scalar', text: '"cross_validation"' } }] }] } }] };
  assert.deepEqual(M.options(menus, G, 'DLSCA_ASCAD', tree, ['jobs', 0, 'processes', 0]).opts, ['train', 'validation']);
});

test('a dataset type, a sweep list item and a library-backed key', () => {
  assert.deepEqual(M.options(menus, P, 'DLSCA_ASCAD', null, ['data', 2, 'type']).opts,
    ['ASCADf', 'ASCADr', 'CHES_CTF_2018', 'AES_RD', 'DPAv4', 'AES_HD']);
  assert.deepEqual(M.options(menus, P, 'DLSCA_ASCAD', null, ['sca', 'leakage_model', 0]).opts, ['HW', 'ID', 'MSB']);
  // the optimizers of the run's framework (pytorch by default; keras: keras.optimizers, e.g. Lion)
  const opt = M.options(menus, P, 'DLSCA_ASCAD', null, ['trainer', 'optimizer', 'type']);
  assert.ok(!opt.free && opt.opts.includes('Adam') && opt.opts.includes('RMSprop') && opt.opts.includes('ASGD'));
  const keras = { k: 'map', entries: [{ key: 'deep_learning', node: { k: 'map', entries: [{ key: 'framework', node: { k: 'scalar', text: 'keras' } }] } }] };
  const kopt = M.options(menus, P, 'DLSCA_ASCAD', keras, ['trainer', 'optimizer', 0, 'type']).opts;
  assert.ok(kopt.includes('Lion') && !kopt.includes('ASGD'));
});

test('graph job models are the project\'s finished runs, typing still allowed', () => {
  const withRuns = { ...menus, refs: { DLSCA_ASCAD: { runs: ['kite_pierce', 'angle_ox'] } } };
  assert.deepEqual(M.options(withRuns, G, 'DLSCA_ASCAD', null, ['jobs', 1, 'models', 0]), { opts: ['kite_pierce', 'angle_ox'], free: true });
  assert.deepEqual(M.options(withRuns, '/x/services/side_channel_core/config/master.yaml', '', null, ['default_project']).opts, ['DLSCA_ASCAD']);
});

test('a facet table of boxes becomes its arrangement', () => {
  assert.deepEqual(M.facetLayout(1, 3, [['1', '2', '3']]), { mode: 'horizontal', order: [1, 2, 3] });
  assert.deepEqual(M.facetLayout(3, 1, [['2'], ['1'], ['']]), { mode: 'vertical', order: [2, 1] });
  assert.deepEqual(M.facetLayout(2, 2, [['1', '2'], ['3', '']]), { mode: 'grid', order: [[1, 2], [3]] });
  assert.deepEqual(M.facetLayout(2, 2, [['', ''], ['4', '1']]), { mode: 'grid', order: [[4, 1]] });
});

test('masterTier: a project\'s name, folders and files and the secrets are read only; scripts, extra_configs, distributed join the hash', () => {
  const tier = p => M.masterTier(p).tier;
  for (const p of [['project', 0, 'name'], ['project', 0, 'root'], ['project', 1, 'params'], ['project', 0, 'env_configs', 'dev'],
    ['project', 0, 'logs'], ['project', 0, 'exp_rerun', 'path'], ['project', 0, 'auto_search', 'config'],
    ['project', 0, 'visualization', 'graph_manager', 'config'], ['secrets', 'db_env_file'],
    ['status_share', 'host'], ['status_share', 'port'], ['status_share', 'enabled']]) assert.equal(tier(p), 'readonly', p.join('.'));
  assert.match(M.masterTier(['status_share', 'port']).why, /a TCP port the network lets through to this machine, and optionally host \(without it, this machine's own address\); edit it in master\.yaml by hand \(optional/);
  for (const p of [['distributed', 'enabled'], ['project', 0, 'scripts', 'train'], ['project', 0, 'extra_configs', 'graph'],
    ['project', 0, 'distributed_override', 'use']]) assert.equal(tier(p), 'hash', p.join('.'));
  for (const p of [['default_project'], ['timezone', 'current_location'], ['project', 0, 'description'], ['project', 0, 'exp_rerun', 'repeat'],
    ['project', 0, 'auto_search', 'enabled'], ['project', 0, 'visualization', 'clearml', 'scope'], ['project', 0, 'visualization', 'graph_manager', 'enabled'],
    ['project'], ['project', 1]]) assert.equal(tier(p), '', p.join('.'));
});

test('askError: the name an Add button asks for: typed, of its form, not taken', () => {
  assert.equal(M.askError('model', '', []), 'type the model_id');
  assert.equal(M.askError('model', 'my cnn', []), 'letters, digits, _ . - (no spaces)');
  assert.equal(M.askError('model', 'A', ['A']), 'A is a model already');
  assert.equal(M.askError('model', ' B ', ['A']), '');
  assert.equal(M.askError('queue', '_fig', []), 'letters, digits, _ . - (no spaces)');      // a queue key starts with a letter or digit
  assert.equal(M.askError('job', '{{metric}}_fig', []), '');
  assert.equal(M.askError('facet', '{{metric}}', []), 'letters, digits, _ . - (no spaces)');
  assert.equal(M.askError('project', 'P', ['P']), 'P is a project already');
  assert.equal(M.askError('dataset', 'ascad', ['ascad']), 'ascad is a data entry already');
});

test('a framework menu keeps the other framework\'s options, marked where they are; other keyed menus do not', () => {
  const fw = name => ({ k: 'map', entries: [{ key: 'deep_learning', node: { k: 'map', entries: [{ key: 'framework', node: { k: 'scalar', text: name } }] } }] });
  const keras = M.options(menus, P, 'DLSCA_ASCAD', fw('keras'), ['trainer', 'optimizer', 0, 'scheduler', 'type']);
  assert.ok(!keras.opts.includes('OneCycleLR') && keras.others.OneCycleLR === 'available in pytorch' && keras.others.CyclicLR === 'available in pytorch');
  assert.ok(Object.keys(keras.others).every(o => !keras.opts.includes(o)));          // never both choosable and marked
  const torch = M.options(menus, P, 'DLSCA_ASCAD', fw('pytorch'), ['trainer', 'optimizer', 0, 'scheduler', 'type']);
  assert.ok(torch.opts.includes('OneCycleLR') && torch.others.CosineDecay === 'available in keras' && !('none' in torch.others));
  const country = { k: 'map', entries: [{ key: 'timezone', node: { k: 'map', entries: [{ key: 'current_country', node: { k: 'scalar', text: 'Taiwan' } }] } }] };
  const rule = (menus.rules['master.yaml'] || []).find(r => r.by && r.by.key !== 'deep_learning.framework');
  if (rule) assert.equal(M.options(menus, '/x/services/side_channel_core/config/master.yaml', '', country, rule.p.split('.')).others, undefined);
});
