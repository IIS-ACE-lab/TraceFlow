// services/visualization/vscode_monitor/test/webview.test.js
'use strict';
// The webview (webview/panel.js) drawn in node with a minimal DOM: the Done list's metric row, the
// config forms' menus, a facet's Find Figure and its rows x columns layout table, the FIGURES rows.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const vm = require('vm');
const { execFileSync } = require('child_process');
const E = require('./extension_stub');

const ROOT = path.resolve(__dirname, '../../../..');
const PY = { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '', PYTHONPATH: [ROOT, path.join(ROOT, 'services/side_channel_core'), path.join(ROOT, 'services/side_channel_core/src')].join(':') } };
const lastJson = out => JSON.parse(out.trim().split('\n').pop());

const CONFIG = `jobs:
  - id: 0
    name: default
    type: model_overlay
    source: attack_metrics
    metric: mean_attack_rank_curve
    plot_type: line
    enabled: true
  - id: 3
    name: acc
    type: model_overlay
    source: cross_validation
    metric: accuracy
    processes: [validation]
    models: [kite_pierce]
faceting:
  - id: 1
    name: two
    panels:
      - file_id: 1
        address: jobs/job_001_a/a.png
      - file_id: 2
        address: jobs/job_002_b/b.png
    arrangement:
      mode: horizontal
      order: [1, 2]
`;

function render({ state, tree, file }) {
  const regions = {};
  const app = {
    set innerHTML(html) {
      for (const m of html.matchAll(/id="([^"]+)"/g)) regions[m[1]] = { innerHTML: '', contains: () => false };
    },
    querySelectorAll: sel => {
      if (sel === '[id]') return Object.entries(regions).map(([id, el]) => Object.assign(el, { id }));
      if (sel === 'details.done-run[data-want]') {               // the opened Done runs that still need their menu
        const html = Object.values(regions).map(r => r.innerHTML).join('\n');
        return [...html.matchAll(/<details class="done-run"[^>]* data-want="([^"]*)" data-name="([^"]*)"/g)]
          .map(m => ({ dataset: { want: m[1], name: m[2] } }));
      }
      return [];
    },
    addEventListener: (type, fn) => { appListeners[type] = fn; },
  };
  const appListeners = {};                                      // the panel's own input / click handlers, to drive them
  const listeners = {};
  const sent = [];
  const ctx = {
    acquireVsCodeApi: () => ({ postMessage: m => sent.push(m), getState: () => state, setState: () => {} }),
    document: { body: { dataset: { mode: 'main' } }, getElementById: () => app, activeElement: null },
    window: { addEventListener: (type, fn) => { listeners[type] = fn; }, TFMenus: require('../webview/menus') },
    console,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../webview/panel.js'), 'utf8'), ctx);
  listeners.message({ data: { type: 'tree', file, sha256: 's', tree, error: '', note: '' } });
  const post = msg => listeners.message({ data: msg });
  post.sent = sent;
  return { regions, post, appListeners };
}

test('the webview draws the metric row, the menus, Find Figure and the facet layout table', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const rules = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.choices', 'static'], PY)).rules;
  const k = p => `n:${file}:${JSON.stringify(p)}`;
  const state = {
    open: { 'dr:DLSCA_ASCAD:kite_pierce': true, 'dr:DLSCA_ASCAD:angle_ox': true, 'p:DLSCA_ASCAD': true, 'c:DLSCA_ASCAD': true, [`cf:${file}`]: true, [k(['jobs'])]: true, [k(['jobs', 0])]: true,
            [k(['faceting'])]: true, [k(['faceting', 0])]: true, [k(['faceting', 0, 'panels'])]: true, [k(['faceting', 0, 'panels', 0])]: true },
    group: { DLSCA_ASCAD: 'done' },
    figsel: { DLSCA_ASCAD: ['2'] },
    facsel: { DLSCA_ASCAD: ['1'] },
    gmetric: { DLSCA_ASCAD: { kite_pierce: 'cross_validation|validation|accuracy', angle_ox: 'cross_validation|validation|accuracy' } },
  };
  const { regions, post } = render({ state, tree, file });
  const done = (name, choices, ranks = null) => ({ hash: name, name, model: 'ASCAD_MLPbest', env: 'dev', status: 'DONE',
    batches: ['fig14'], spark: '', last: ranks && ranks.val ? ranks.val.rank : null, ranks, choices });
  const choices = [{ source: 'cross_validation', process: 'validation', metric: 'accuracy', kind: 'per_epoch' },
                   { source: 'cross_validation', process: 'validation', metric: 'mean_attack_rank_curve', kind: 'curve' }];
  post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules, refs: {} }, projects: [{
    name: 'DLSCA_ASCAD', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'idle' }, run: null, eta: null,
    tiles: { queued: [], running: [], done: [done('kite_pierce', choices, { val: { rank: 107, traces: 2500 }, same: { rank: 107.83, traces: 2500 }, test: { rank: 36, traces: 22500 }, budget: { rank: 112.4, traces: 1200 } }), done('angle_ox', choices)], failed: [] },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, queues: [], figures: [
      { id: 1, name: 'drawn', enabled: true, drawn: true, dir: 'job_001_a', dirPath: path.join(dir, 'job_001_a'), filePaths: [path.join(dir, 'job_001_a', 'a.pdf')] },
      { id: 2, name: 'not drawn', enabled: true, drawn: false, dir: 'job_002_b', dirPath: '', filePaths: [] }],
    configs: [{ file: 'graph_config.yaml', path: file }], errors: [],
    graphJobs: [{ id: 3, metric: 'accuracy', source: 'cross_validation', processes: ['validation'], models: ['kite_pierce'] }],
    facets: [{ id: 1, name: 'two', drawn: true, dir: 'facet_001_two', dirPath: path.join(dir, 'facet_001_two'), filePaths: [path.join(dir, 'facet_001_two', 'two.pdf')] },
      { id: 2, name: 'three', drawn: false, dir: null, dirPath: null, filePaths: [] }],
  }] });
  const html = Object.values(regions).map(r => r.innerHTML).join('\n');
  if (process.env.TF_DUMP) fs.writeFileSync(process.env.TF_DUMP, html);
  // Done: every run a drop-down; its line (Retest All, Retest Budget at the right end) is the summary, under it the metrics,
  // grouped, with the right button, then its Graph sub-panel: the ids and the jobs that list it, as forms
  // each Done line: validation, all N_val traces | test, first N_val traces | test, all N_test traces, each with its trace
  // count, coloured; – where a run has none; their key under the tiles
  assert.match(html, /kite_pierce[^]*?<small class="num ranks" title="[^"]*"><span class="rk val">107 <small>@2.5k<\/small><\/span> \| <span class="rk same">107.83 <small>@2.5k<\/small><\/span> \| <span class="rk budget">112.4 <small>@1.2k<\/small><\/span> \| <span class="rk test">36 <small>@22.5k<\/small><\/span><\/small>/);
  assert.match(html, /angle_ox[^]*?<span class="rk val">–<\/span> \| <span class="rk same">–<\/span> \| <span class="rk budget">–<\/span> \| <span class="rk test">–<\/span>/);
  assert.match(html, /<\/div><div class="done-bar"><span class="list-tools">[^]*?<\/span><div class="rank-legend"><span><span class="rk-key val"><\/span>validation, all N_val traces<\/span><span><span class="rk-key same"><\/span>test, first N_val traces<\/span><span><span class="rk-key budget"><\/span>test, budget N_test traces<\/span><span><span class="rk-key test"><\/span>test, all N_test traces<\/span><\/div><\/div><div class="budget-bar"[^>]*><\/div><ul class="runs box" data-scrollkey="DLSCA_ASCAD:done">/);
  // [Retest All] then the Budget box in the old [Retest Budget]'s size: [Test] and the entry (no params.yaml here: disabled)
  assert.match(html, /<li class="done-run"><details class="done-run" data-key="dr:DLSCA_ASCAD:kite_pierce" open><summary[^>]*>[^]*?kite_pierce[^]*?<span class="end">[^]*?data-kind="all"[^>]*>Retest All<\/button><span class="budget-form" data-size="Retest Budget" data-file=""><button [^>]*data-kind="budget" disabled[^>]*>Test<\/button><input class="val budget" placeholder="Budget" size="1" disabled[^>]*><\/span><\/span><\/summary><div class="metric-row"/);
  assert.match(html, /<details class="rungraph" data-key="rg:DLSCA_ASCAD:kite_pierce" data-file="[^"]*graph_config.yaml" open><summary[^>]*>Graph <small>id 3<\/small><\/summary><div class="form" data-file="[^"]*graph_config.yaml" data-project="DLSCA_ASCAD" data-sha="[^"]*"><details class="node" data-key="n:[^"]*:\[&quot;jobs&quot;,1\]"><summary[^>]*>#1 <small>name: acc<\/small><span class="end"><button class="x danger jobdel" data-jobdel="DLSCA_ASCAD" data-id="3"[^>]*>×<\/button><button class="jobdraw" data-run="draw" data-form-project="1" data-id="3"[^>]*>Draw<\/button><\/span><\/summary><\/details><\/div><\/details>/);
  // GRAPH_CONFIG.YAML: every job line and facet line ends in [Delete][Draw]; job 0 (the judge's) only Draw
  assert.match(html, /<summary[^>]*>#0 <small>[^<]*<\/small><span class="end"><button class="jobdraw" data-run="draw" data-form-project="1" data-id="0"[^>]*>Draw<\/button><\/span><\/summary>/);
  assert.match(html, /<summary[^>]*>#0 <small>[^<]*<\/small><span class="end"><button class="x danger facetdel" data-facetdel="DLSCA_ASCAD" data-id="1"[^>]*>×<\/button><button class="jobdraw" data-run="draw-facet" data-form-project="1" data-id="1"[^>]*>Draw<\/button><\/span><\/summary>/);
  // FIGURES: a Graph sub-panel (the jobs, as before) and a Faceting sub-panel, every facet with its light and buttons
  assert.match(html, /<details[^>]*data-key="fg:DLSCA_ASCAD"[^>]*><summary[^>]*>Graphs <small>1\/2 drawn<\/small>[^]*?Select all \(2\)/);
  const faceting = html.match(/data-key="ff:DLSCA_ASCAD"[^]*?<\/details>/)[0];
  assert.match(faceting, /<summary[^>]*>Faceting <small>1\/2 drawn<\/small>/);
  // above the facets: Select all, Draw all (n chosen), Delete all (the ticked that are drawn); facet 1 is ticked
  assert.match(faceting, /<div class="figall"><label class="delall"><input type="checkbox" data-facall="DLSCA_ASCAD"> Select all \(2\)<\/label><button data-facdrawall="DLSCA_ASCAD"[^>]*>Draw all \(1 chosen\)<\/button><button class="danger" data-figdelall="DLSCA_ASCAD" data-dirs="\[&quot;[^"]*facet_001_two&quot;\]"[^>]*>Delete all \(1 chosen\)<\/button><\/div>/);
  assert.match(faceting, /<input type="checkbox" data-facpick="DLSCA_ASCAD" data-id="1" checked> <span class="light green"><\/span><small>1<\/small>/);
  assert.match(faceting, /<span class="light green"><\/span><small>1<\/small><\/label><\/td><td class="num">1<\/td><td class="name"[^>]*>two<\/td><td class="acts"><button data-open="[^"]*two.pdf"[^>]*>Open<\/button><button data-reveal="[^"]*facet_001_two"[^>]*>Folder<\/button><button class="danger" data-figdel="DLSCA_ASCAD"[^>]*>Delete<\/button><button data-run="draw-facet"[^>]*data-id="1"[^>]*>Draw<\/button><\/td>/);
  // Graphs, the same order: drawn [Open][Folder][Delete][Draw], not drawn [Draw] alone
  assert.match(html, /<td class="name"[^>]*>drawn<\/td><td class="acts"><button data-open="[^"]*a.pdf"[^>]*>Open<\/button><button data-reveal="[^"]*job_001_a"[^>]*>Folder<\/button><button class="danger" data-figdel="DLSCA_ASCAD"[^>]*>Delete<\/button><button data-run="draw"[^>]*data-id="1"[^>]*>Draw<\/button><\/td>/);
  assert.match(html, /<td class="name"[^>]*>not drawn<\/td><td class="acts"><button data-run="draw"[^>]*data-id="2"[^>]*>Draw<\/button><\/td>/);
  assert.match(faceting, /<span class="light"><\/span><small><\/small><\/label><\/td><td class="num">2<\/td><td class="name"[^>]*>three<\/td><td class="acts"><button data-run="draw-facet"[^>]*data-id="2"[^>]*>Draw<\/button><\/td>/);
  // a job opened: inside it only OK, and only once one of its values is changed (Draw is on its line)
  assert.ok(!/<div class="buttons">[^<]*<button[^>]*data-run="draw"/.test(html));
  assert.match(html, /data-key="rg:DLSCA_ASCAD:angle_ox"[^>]*open><summary[^>]*>Graph <small>no graph job lists this run<\/small>/);

  // a run opened before its menu is read asks for it at once; a failed read shows why
  post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules, refs: {} }, projects: [{
    name: 'DLSCA_ASCAD', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'idle' }, run: null, eta: null,
    tiles: { queued: [], running: [], done: [done('kite_pierce', null), done('angle_ox', { error: 'boom' })], failed: [] },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, queues: [], figures: [],
    configs: [{ file: 'graph_config.yaml', path: file }], errors: [], graphJobs: [] }] });
  const again = Object.values(regions).map(r => r.innerHTML).join('\n');
  assert.match(again, /data-key="dr:DLSCA_ASCAD:kite_pierce" data-want="DLSCA_ASCAD" data-name="kite_pierce" open>[^]*?reading its metrics…/);
  assert.match(again, /its metrics could not be read: boom; close and open the run to read them again/);
  assert.ok(post.sent.some(m => m.type === 'want-choices' && m.name === 'kite_pierce'));
  assert.match(html, /<optgroup label="cross_validation · validation">/);
  assert.match(html, /data-gremove="DLSCA_ASCAD" data-name="kite_pierce"[^>]*>Delete from Graph Config/);       // in job 3
  assert.match(html, /data-gexport="DLSCA_ASCAD" data-name="angle_ox"[^>]*>Export to Graph Config/);           // in none
  // forms: a known set is a menu
  assert.match(html, /<select class="val" data-op="set" data-path="\[&quot;jobs&quot;,0,&quot;type&quot;\]"[\s\S]*?<option value="model_explanation">/);
  assert.match(html, /<select class="val" data-op="set" data-path="\[&quot;jobs&quot;,0,&quot;enabled&quot;\]"[\s\S]*?<option value="true" selected>/);
  // facets: Find Figure beside an address, the rows x columns table with a box per panel
  assert.match(html, /address[\s\S]*?Find Figure/);
  assert.match(html, /class="tp-grid"/);
  assert.match(html, /1 × 2: one row \(horizontal\)/);
  assert.match(html, /data-tablecell="0,1"[\s\S]*?<option value="2" selected/);
  // FIGURES: every row is [check box] (status light) then the rest, as in Housekeeping; any figure can be ticked
  assert.match(html, /<td class="pick"><label[^>]*><input type="checkbox" data-figpick="DLSCA_ASCAD" data-id="1"> <span class="light green"><\/span><small><\/small><\/label><\/td><td class="num">1<\/td>/);
  assert.match(html, /<td class="pick"><label[^>]*><input type="checkbox" data-figpick="DLSCA_ASCAD" data-id="2" checked> <span class="light"><\/span><small>1<\/small><\/label><\/td><td class="num">2<\/td>/);
  // above the rows: Select all, Draw all (n chosen); Export to Facet takes only ticked figures that are drawn
  assert.match(html, /<div class="figall"><label class="delall"><input type="checkbox" data-figall="DLSCA_ASCAD"> Select all \(2\)<\/label><button data-drawall="DLSCA_ASCAD" title="[^"]*">Draw all \(1 chosen\)<\/button><button class="danger" data-figdelall="DLSCA_ASCAD"[^>]*>Delete all \(0 chosen\)<\/button><\/div><table class="figures">/);
  assert.match(html, /data-facetexport="DLSCA_ASCAD" disabled/);
  // beside Draw all: Delete all of the ticked figures that are drawn (figure 2 is ticked, not drawn: none)
  assert.match(html, /Draw all \(1 chosen\)<\/button><button class="danger" data-figdelall="DLSCA_ASCAD" data-dirs="\[\]" disabled[^>]*>Delete all \(0 chosen\)<\/button><\/div>/);
});

test('FIGURES: Delete all counts the ticked figures that are drawn and carries their folders', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const state = { open: { 'p:DLSCA_ASCAD': true, 'f:DLSCA_ASCAD': true }, group: {}, figsel: { DLSCA_ASCAD: ['1', '2'] } };
  const { regions, post } = render({ state, tree, file });
  const drawnDir = path.join(dir, 'job_001_a');
  post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [{
    name: 'DLSCA_ASCAD', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'idle' }, run: null, eta: null,
    tiles: { queued: [], running: [], done: [], failed: [] },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, queues: [], figures: [
      { id: 1, name: 'drawn', enabled: true, drawn: true, dir: 'job_001_a', dirPath: drawnDir, filePaths: [] },
      { id: 2, name: 'not drawn', enabled: true, drawn: false, dir: 'job_002_b', dirPath: '', filePaths: [] }],
    configs: [{ file: 'graph_config.yaml', path: file }], errors: [], graphJobs: [] }] });
  const html = Object.values(regions).map(r => r.innerHTML).join('\n');
  const dirs = JSON.stringify([drawnDir]).replace(/"/g, '&quot;');
  assert.ok(html.includes(`Draw all (2 chosen)</button><button class="danger" data-figdelall="DLSCA_ASCAD" data-dirs="${dirs}" title=`));
  assert.match(html, />Delete all \(1 chosen\)<\/button>/);
});

test('Overlay Graph shows on a queue line only once every queue of its study is DONE, before Rerun and Log; its graph ids beside', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const { regions, post } = render({ state: { open: { 'p:DLSCA_ASCAD': true, 'q:DLSCA_ASCAD': true } }, tree, file });
  const q = (key, state, refs = []) => ({ key, state, done: 1, planned: 1, runs: 1, commands: 1, refs });
  post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [{
    name: 'DLSCA_ASCAD', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'idle' }, run: null, eta: null,
    tiles: { queued: [], running: [], done: [], failed: [] },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, figures: [],
    queues: [q('fig10', 'not started'), q('fig10_optimizers', 'done'), q('fig14_cnnbest', 'done'), q('fig14_vgg16', 'not started'),
      q('fig26', 'done', ['candle_flour', 'h26'])],
    configs: [{ file: 'graph_config.yaml', path: file }], errors: [],
    graphJobs: [{ id: 0, runs: ['candle_flour'] }, { id: 8, runs: ['candle_flour'], models: ['candle_flour'] }, { id: 9, runs: [], models: ['h26'] }] }] });
  const html = Object.values(regions).map(r => r.innerHTML).join('\n');
  const studied = [...html.matchAll(/data-run="study"[^>]*data-key="([^"]+)"/g)].map(m => m[1]);
  assert.deepEqual(studied, ['fig10_optimizers', 'fig26']);     // fig14 waits for fig14_vgg16; fig10 never ran
  const fig26 = html.match(/<tr><td>fig26<\/td>[^]*?<\/tr>/)[0];
  assert.match(fig26, /<td class="gids"[^>]*>graph #8, #9<\/td>/);                      // by name or hash; job 0 aside
  assert.deepEqual([...fig26.matchAll(/<button[^>]*>([^<]+)</g)].map(m => m[1]), ['Overlay Graph', 'Rerun', 'Log']);
  assert.match(html, /<tr><td>fig10<\/td>[^]*?<small>no graph<\/small>/);
  const labels = [...html.matchAll(/data-run="queue"[^>]*data-key="([^"]+)"[^>]*>([^<]+)</g)].map(m => [m[1], m[2]]);
  assert.deepEqual(labels, [['fig10', 'Start'], ['fig10_optimizers', 'Rerun'], ['fig14_cnnbest', 'Rerun'],
    ['fig14_vgg16', 'Start'], ['fig26', 'Rerun']]);                // a done queue reruns; the others start
});

test('experiment_queue.yaml: a line is the panel\'s words and four menus, with × and +; Add a Queue beside Open in editor', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-'));
  const file = path.join(dir, 'experiment_queue.yaml');
  fs.writeFileSync(file, 'fig26:\n  - make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest\n' +
    '  - make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=odd MODEL_ID=ASCAD_MLPbest\n  - make test ENV=dev PROJECT=DLSCA_ASCAD\n');
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const k = p => `n:${file}:${JSON.stringify(p)}`;
  const state = { open: { 'p:DLSCA_ASCAD': true, 'c:DLSCA_ASCAD': true, [`cf:${file}`]: true, [k(['fig26'])]: true } };
  const { regions, post } = render({ state, tree, file });
  post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [{
    name: 'DLSCA_ASCAD', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: ['ASCAD_MLPbest', 'ASCAD_CNNbase'], datasets: ['ascad', 'ascad_desync100'], light: { color: 'grey', label: 'idle' },
    run: null, eta: null, tiles: { queued: [], running: [], done: [], failed: [] },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, figures: [], facets: [], queues: [],
    configs: [{ file: 'experiment_queue.yaml', path: file }], errors: [], graphJobs: [] }] });
  const html = Object.values(regions).map(r => r.innerHTML).join('\n');
  assert.match(html, /Open in editor<\/button><button data-addqueue="DLSCA_ASCAD" data-file="[^"]*experiment_queue.yaml" data-sha="[^"]+"[^>]*>Add a Queue<\/button>/);
  assert.match(html, /<summary[^>]*>fig26 <small>3 CMD<\/small><span class="end"><button class="x" data-delentry="queue" data-name="fig26" data-path="\[&quot;fig26&quot;\]" title="delete queue fig26 and its 3 lines \(asks first\)">×<\/button><\/span><\/summary>/);   // the queue's line: key, its commands, × (asks first)
  const lines = [...html.matchAll(/<div class="field qline[^"]*" data-path="([^"]+)">([^]*?)<\/div>/g)];
  assert.equal(lines.length, 2);                                             // `make test …` is not a queue line: a text box
  assert.match(lines[0][2], /^<span class="qfix">make orchestrate<\/span><span class="qkv">ENV=<select class="val" data-qvar="ENV">[^]*?<option value="dev" selected>dev<\/option>/);
  assert.match(lines[0][2], /PROJECT=<select class="val" data-qvar="PROJECT">[^]*?<option value="DLSCA_ASCAD" selected>/);
  assert.match(lines[0][2], /MODEL_ID=<select class="val" data-qvar="MODEL_ID">[^]*?<option value="ASCAD_MLPbest" selected>ASCAD_MLPbest<\/option><option value="ASCAD_CNNbase">/);
  assert.match(lines[0][2], /data-op="remove"[^>]*>×<\/button>$/);
  assert.match(lines[1][2], /DATASET=<select class="val bad" data-qvar="DATASET"><option value="">\(none\)<\/option><option value="odd" selected>odd \(not an option\)<\/option>/);
  assert.match(html, /<input class="val" data-op="set" data-path="\[&quot;fig26&quot;,2\]" value="make test ENV=dev PROJECT=DLSCA_ASCAD"/);
  assert.match(html, /data-qadd="1" data-path="\[&quot;fig26&quot;\]" data-value="make test ENV=dev PROJECT=DLSCA_ASCAD"[^>]*>\+<\/button>/);
  // EXPERIMENT SETTINGS: the file's role on its summary line; QUEUES: the reminder on the button line
  assert.match(html, /<summary[^>]*>experiment_queue.yaml<small class="cfg-role">multiple one-off runs run collectively under a common ID<\/small><\/summary>/);
  assert.match(html, /<div class="buttons">[^]*?<small class="run-hint">Configure parameters correctly before starting the Run<\/small><\/div>/);
});

test('the run card: under its progress, its latest metrics, a line per metric with its value per process, better green and worse red', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const { regions, post } = render({ state: { open: { 'p:DLSCA_ASCAD': true }, group: { DLSCA_ASCAD: 'running' } }, tree, file });
  const value = (process, label, v, previous, direction, improved) => ({ process, label, value: v, previous, direction, improved });
  const live = { run: 'kite_pierce', source: 'fold 3', epoch: 47, metrics: [
    { name: 'mean_accuracy', values: [value('train', 'train', 0.0061, 0.0058, 'max', true),
      value('validation', 'validation (CV)', 0.0044, 0.0049, 'max', false)] },
    { name: 'mean_attack_rank_curve', values: [value('validation', 'validation (CV)', 12.3, 15, 'min', true)] },
    { name: 'CrossEntropyLoss', values: [value('train', 'train', 5.51, 5.53, 'min', true),
      value('validation', 'validation (CV)', 5.58, null, 'min', null)] }] };
  post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [{
    name: 'DLSCA_ASCAD', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'green', label: 'running' }, run: { name: 'kite_pierce', hash: 'h1', model: 'ASCAD_CNNbest', dataset: 'ascad', phase: 'fold 3/10 · epoch 47/100', percent: 40, batches: [], alive: { alive: true, label: '' }, live }, eta: null,
    tiles: { queued: [], running: [{ hash: 'h1', name: 'kite_pierce', model: 'ASCAD_CNNbest', env: 'dev', status: 'in_progress',
      batches: ['fig14'], spark: '', last: 12.3, epochs: 47, alive: true }], done: [], failed: [] },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, figures: [], facets: [], queues: [],
    configs: [], errors: [], graphJobs: [] }] });
  const html = Object.values(regions).map(r => r.innerHTML).join('\n');
  const block = html.match(/<div class="run">[^]*?(<div class="live">[^]*?<div class="live-legend">[^]*?<\/div><\/div>)<\/div>/);
  assert.ok(block, 'the live block sits in the run card, under its progress: visible without opening [Running]');
  assert.ok(!/<li>[^<]*<span class="dot[^]*?kite_pierce[^]*?<div class="live">/.test(html.split('<div class="run">')[0]),
    'the [Running] list no longer repeats it');
  // the head row: the fold and epoch over the names, each process over its values; the cells hold the values alone
  assert.match(block[1], /^<div class="live"><table class="live-rows"><tr class="live-cols"><th>fold 3 · epoch 47<\/th><th class="live-col">train<\/th><th class="live-col">validation \(CV\)<\/th><\/tr>/);
  const names = [...block[1].matchAll(/<tr><th>([^<]*)<\/th>/g)].map(m => m[1]);
  assert.deepEqual(names, ['mean_accuracy', 'mean_attack_rank_curve', 'CrossEntropyLoss']);   // one line per metric
  assert.match(block[1], /<tr><th>mean_accuracy<\/th><td title="was 0.005800 the epoch before; higher is better"><span class="num">0.006100<span class="live-dot better"><\/span><\/span><\/td><td title="was 0.004900 the epoch before; higher is better"><span class="num">0.004400<span class="live-dot worse"><\/span><\/span><\/td><\/tr>/);
  // a column per process: the rank has no train value, a blank cell keeps its validation under the others
  assert.match(block[1], /<tr><th>mean_attack_rank_curve<\/th><td><\/td><td title="was 15 the epoch before; lower is better"><span class="num">12.30<span class="live-dot better">/);
  assert.match(block[1], /title="first epoch"><span class="num">5.580<span class="live-dot same">/);
  assert.match(block[1], /<div class="live-legend"><span><span class="live-dot better"><\/span>better than the epoch before<\/span><span><span class="live-dot worse"><\/span>worse than the epoch before<\/span><span><span class="live-dot same"><\/span>unchanged, or the first epoch<\/span><\/div><\/div>$/);
});

test('the run card while the test attacks: the orange attack bar in the epoch\'s place, the test values so far, the key-recovery line', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const { regions, post } = render({ state: { open: { 'p:DLSCA_ASCAD': true }, group: { DLSCA_ASCAD: 'running' } }, tree, file });
  const value = (process, label, v, previous, direction, improved) => ({ process, label, value: v, previous, direction, improved });
  const now = (v, note) => ({ process: 'test', label: 'test', value: v, previous: null, direction: 'min', improved: null, live: true, note });
  const live = { run: 'kite_pierce', source: null, epoch: 20, attack: { attack: 37, attacks: 100, running: true },
    key_recovery: { source: 'validation, epoch 20', recovered: false, true_key_rank: 107, top_guess: '0xFB', ge: 107, sr1: 0 },
    metrics: [
      { name: 'mean_attack_rank_curve', values: [value('validation', 'validation', 107, 110, 'min', true), now(61.24, null)] },
      { name: 'traces_to_ge_1', values: [now(null, 'not reached yet')] },
      { name: 'per_attack_traces_to_ge_1', values: [now(342, '30 of 37 reached')] },
      { name: 'mean_accuracy', values: [value('train', 'train', 0.0412, 0.04, 'max', true), value('validation', 'validation', 0.0038, 0.004, 'max', false)] }] };
  post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [{
    name: 'DLSCA_ASCAD', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'green', label: 'running' }, run: { name: 'kite_pierce', hash: 'h1', model: 'ASCAD_CNNbest', dataset: 'ascad', phase: 'test', percent: 40, batches: [], alive: { alive: true, label: '' }, live, attack: { attack: 37, attacks: 100, running: true } }, eta: null,
    tiles: { queued: [], running: [{ hash: 'h1', name: 'kite_pierce', model: 'ASCAD_CNNbest', env: 'dev', status: 'in_progress',
      batches: ['fig14'], spark: '', last: 12.3, epochs: 47, alive: true }], done: [], failed: [] },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, figures: [], facets: [], queues: [],
    configs: [], errors: [], graphJobs: [] }] });
  const html = Object.values(regions).map(r => r.innerHTML).join('\n');
  // the phase line: the orange bar (glowing while it runs) and attack i of n, where epoch n/N stood
  assert.match(html, /<div class="phase attack-phase"[^>]*><span class="bar-label">test<\/span><span class="live-attack-bar running"><svg[^]*?<rect class="fill" width="37" height="4"\/><\/svg><\/span><span class="live-attack-count">attack 37 of 100<\/span><\/div>/);
  const block = html.match(/<div class="run">[^]*?(<div class="live">[^]*?<div class="live-legend">[^]*?<\/div><\/div>)<\/div>/);
  assert.ok(block, 'the live block sits in the run card');
  // no epoch in the head while the test attacks (the attack bar stands in the epoch's place); test over the test values
  assert.match(block[1], /^<div class="live"><table class="live-rows"><tr class="live-cols"><th><\/th><th class="live-col">train<\/th><th class="live-col">validation<\/th><th class="live-col live-test">test<\/th><\/tr>/);
  assert.match(block[1], /<tr><th>mean_attack_rank_curve<\/th><td><\/td><td title="was 110[^"]*"><span class="num">107<span class="live-dot better"><\/span><\/span><\/td><td class="live-test" title="the test so far: the attacks done until now"><span class="num">61.24<span class="live-dot same">/);
  assert.match(block[1], /<td class="live-test"[^>]*><span class="num">not reached yet<span class="live-dot same">/);
  assert.match(block[1], /<span class="num">342 <small>\(30 of 37 reached\)<\/small><span class="live-dot same">/);
  assert.match(block[1], /<div class="live-key">key recovery · validation, epoch 20: not recovered · true key rank 107 · top guess 0xFB · GE 107 · SR@1 0.00<\/div>/);
});

test('the run card is listed under [Running] in the run\'s place, and leaves when the run moves to [Done]', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const project = (run, tiles) => ({
    name: 'ZvW', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'green', label: 'running' }, run, eta: null, tiles,
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, figures: [], facets: [], queues: [],
    configs: [], errors: [], graphJobs: [] });
  const tile = { hash: 'h1', name: 'oatmeal_nanimo', model: 'Zaid_DPAv4', env: 'dev', status: 'DONE', batches: ['z_fig6_dpav4'],
                 spark: '', last: 1, epochs: 50, alive: true };
  const run = { name: 'oatmeal_nanimo', hash: 'h1', model: 'Zaid_DPAv4', dataset: 'dpav4_fs01', phase: 'test', percent: 98,
                batches: [], alive: { alive: true, label: '' }, attack: { attack: 37, attacks: 100, running: true } };
  const show = (group, p) => {
    const { regions, post } = render({ state: { open: { 'p:ZvW': true }, group: { ZvW: group } }, tree, file });
    post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [p] });
    return Object.values(regions).map(r => r.innerHTML).join('\n');
  };
  // running (its test under way): the whole card in the [Running] list, with the run's own buttons
  let html = show('running', project(run, { queued: [], running: [tile], done: [], failed: [] }));
  assert.match(html, /<ul class="runs"><li class="run-card"><div class="run"><div class="run-head">[^]*?oatmeal_nanimo[^]*?Terminate Run[^]*?Container log/);
  assert.match(html, /<span class="bar-label">test<\/span><span class="live-attack-bar running">/);
  assert.doesNotMatch(show('', project(run, { queued: [], running: [tile], done: [], failed: [] })), /class="run"/);   // only under [Running]
  // the run moved to [Done] (its beat ended): no card anywhere
  const after = project(null, { queued: [], running: [], done: [{ ...tile, ranks: null, choices: [] }], failed: [] });
  assert.doesNotMatch(show('running', after), /class="run"|run-card/);
  assert.doesNotMatch(show('done', after), /class="run"|run-card/);
});

test('the ETA row is drawn for every running project, with the reason it cannot say more yet', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-webview-eta-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const rules = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.choices', 'static'], PY)).rules;
  const { regions, post } = render({ state: { open: { 'p:ZvW': true }, group: { ZvW: 'running' } }, tree, file });   // the card: under [Running]
  const project = eta => ({
    name: 'ZvW', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'green', label: 'running' },
    run: { name: 'snooker_scrunchie', model: 'Zaid_ASCAD_N100', dataset: 'ascad_n100_none', hash: '533a1b85', percent: 1, phase: 'train',
           batches: ['s3_ascad_n100_zaid'], trial: 1, trials: 3, queue: 's3_ascad_n100_zaid', alive: true },
    eta, tiles: { queued: [], running: [], done: [], failed: [] }, tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} },
    queues: [{ key: 's3_ascad_n100_zaid', state: 'running', done: 0, planned: 6, runs: 6, commands: 2, env: 'dev' }], figures: [], facets: [], configs: [], errors: [], graphJobs: [],
  });
  const state = projects => ({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules, refs: {} }, projects });
  post(state([project({ runLeftS: null, runEndMs: null, batchLeftS: null, batchEndMs: null, upperBound: false, note: '' })]));
  let html = Object.values(regions).map(r => r.innerHTML).join('\n');
  // before the first progress: the row with its two cells and no note (nothing is wrong)
  assert.match(html, /<table class="eta"><tr><th>ETA<\/th><td>run: <small>estimating…<\/small><\/td><td>batch: <small>estimating…<\/small><\/td><\/tr><\/table>/);
  assert.doesNotMatch(html, /etanote/);
  post(state([project({ runLeftS: 180, runEndMs: Date.now() + 180000, batchLeftS: null, batchEndMs: null, upperBound: false,
                         note: 'batch ETA unavailable: the sweep of params.yaml did not load (sca.attack_byte sweep list must not be empty)' })]));
  html = Object.values(regions).map(r => r.innerHTML).join('\n');
  assert.match(html, /<th>ETA<\/th><td>run: \d\d:\d\d[^<]*<small>\(in [^)]*\)<\/small><\/td><td>batch: <small>estimating…<\/small><\/td>/);
  assert.match(html, /etanote">batch ETA unavailable: the sweep of params\.yaml did not load \(sca\.attack_byte sweep list must not be empty\)</);
  post(state([project(null)]));                                                  // no heartbeat: the row says so through the note of the extension; none here: no row
  html = Object.values(regions).map(r => r.innerHTML).join('\n');
  assert.doesNotMatch(html, /class="eta"/);
});

test('the image build bar: full and green once built, where it stopped otherwise; an SVG bar, no <progress>', () => {
  const { regions, post } = render({ state: {}, tree: null, file: '' });
  const drawn = build => {
    post({ type: 'state', busy: [], timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [],
           gpu: { gpus: [], error: 'no GPU here', docker: { build, building: false, containers: [] } } });
    return Object.values(regions).map(r => r.innerHTML).join('\n');
  };
  let html = drawn({ state: 'built', percent: 100, done: 14, total: 14, image: 'sidechannel:dev-536b38d', phase: '' });
  assert.match(html, /image built · sidechannel:dev-536b38d <span class="build-bar build-built"><svg[^>]*><rect class="track" width="100" height="4"\/><rect class="fill" width="100" height="4"\/><\/svg><\/span>/);
  assert.doesNotMatch(html, /<progress/);
  html = drawn({ state: 'built', percent: 0, done: 0, total: 0, image: 'sidechannel:dev-x', phase: '' });   // no steps counted: still full
  assert.match(html, /build-built"><svg[^>]*>.*<rect class="fill" width="100"/);
  html = drawn({ state: 'stopped', percent: 40, done: 4, total: 10, image: '', phase: '' });
  assert.match(html, /build stopped at 40% \(4\/10\) <span class="build-bar build-stopped"><svg[^>]*>.*<rect class="fill" width="40"/);
});

test('[Done] and [Failed] have a filter (a queue or a model) and a sort (ranks, alphabet, created; ascending or descending)', () => {
  const run = (hash, name, model, queue, started, ranks) => ({ hash, name, model, env: 'dev', status: 'DONE', queue, batches: [queue],
    trial: 1, trials: 10, spark: '', last: 1, ranks, choices: null, started });
  const rk = (val, same, test) => ({ val: { rank: val, traces: 1000 }, same: same == null ? null : { rank: same, traces: 1000 }, test: { rank: test, traces: 9000 } });
  const done = [run('h1', 'pecan_highlight', 'NoConv1_DPAv4', 'w_fig1_dpav4', '2026-10-25T18:00:00+08:00', rk(1, 1, 1)),
    run('h2', 'tusks_ointment', 'NoConv1_ASCAD_N50', 'w_fig2_ascad_n50', '2026-10-25T17:00:00+08:00', rk(30, null, 12)),
    run('h3', 'gumdrop_oboe', 'Zaid_ASCAD_N50', 'w_fig2_ascad_n50', '2026-10-25T19:00:00+08:00', rk(5, 7, 3))];
  const failed = [{ ...run('f1', 'zebra_one', 'Zaid_AES_HD', 'w_fig1_aes_hd', '2026-10-25T12:00:00+08:00', null), status: 'FAILED' },
    { ...run('f2', 'apple_two', 'Zaid_AES_RD', 'w_fig2_aes_rd', '2026-10-25T13:00:00+08:00', null), status: 'FAILED' }];
  const project = {
    name: 'ZvW', env: 'dev', envSource: 'test', path: '/x', configDir: '/x', graphJobsDir: '/x', session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'done' }, run: null, eta: null,
    tiles: { queued: [], running: [], failed, done },
    tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, figures: [], facets: [], queues: [], configs: [], errors: [], graphJobs: [] };
  const draw = (group, extra) => {
    const { regions, post } = render({ state: { open: { 'p:ZvW': true }, group: { ZvW: group }, ...extra }, tree: null, file: '' });
    post({ type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [project] });
    return Object.values(regions).map(r => r.innerHTML).join('\n');
  };
  const order = html => [...html.matchAll(/data-hash="(\w+)"/g)].map(m => m[1]).filter((h, k, a) => a.indexOf(h) === k);
  let html = draw('done');                                             // every run, oldest first (created, ascending)
  assert.match(html, /<div class="done-bar"><span class="list-tools"><label class="done-find"[^>]*><svg class="filter-icon"[^]*?<select data-listpick="done" data-project="ZvW"><option value="" selected>all runs \(3\)<\/option>/);
  assert.match(html, /<optgroup label="queue"><option value="queue:w_fig1_dpav4">w_fig1_dpav4 \(1\)<\/option><option value="queue:w_fig2_ascad_n50">w_fig2_ascad_n50 \(2\)<\/option><\/optgroup>/);
  assert.match(html, /<select data-listsort="done" data-project="ZvW"><option value="val">validation, all N_val traces<\/option><option value="same">test, first N_val traces<\/option><option value="budget">test, budget N_test traces<\/option><option value="test">test, all N_test traces<\/option><option value="name">alphabet<\/option><option value="created" selected>created<\/option><\/select><select data-listdir="done" data-project="ZvW"><option value="asc" selected>ascending<\/option><option value="desc">descending<\/option><\/select>/);
  assert.deepEqual(order(html), ['h2', 'h1', 'h3']);
  assert.deepEqual(order(draw('done', { listSort: { 'done:ZvW': { key: 'val', dir: 'asc' } } })), ['h1', 'h3', 'h2']);
  assert.deepEqual(order(draw('done', { listSort: { 'done:ZvW': { key: 'same', dir: 'desc' } } })), ['h3', 'h1', 'h2']);   // none last
  assert.deepEqual(order(draw('done', { listSort: { 'done:ZvW': { key: 'name', dir: 'desc' } } })), ['h2', 'h1', 'h3']);
  html = draw('done', { listPick: { 'done:ZvW': 'queue:w_fig2_ascad_n50' }, listSort: { 'done:ZvW': { key: 'test', dir: 'asc' } } });
  assert.match(html, /<small class="done-count">2 of 3<\/small>/);
  assert.deepEqual(order(html), ['h3', 'h2']);
  // [Failed]: the same tools, its sort by alphabet or created only (no test ranks); no ranks' key
  html = draw('failed', { listSort: { 'failed:ZvW': { key: 'name', dir: 'asc' } } });
  assert.match(html, /<select data-listsort="failed" data-project="ZvW"><option value="name" selected>alphabet<\/option><option value="created">created<\/option><\/select>/);
  assert.doesNotMatch(html, /rank-legend/);
  assert.deepEqual(order(html), ['f2', 'f1']);
  assert.equal(order(draw('failed', { listPick: { 'failed:ZvW': 'model:Zaid_AES_HD' } })).join(), 'f1');
});

test('a Done run and a TESTS model carry a Budget box: [Test] and their data entry\'s attack_trace_budget', () => {
  // owner 2026-10-28: [Retest Budget] / [Test Budget] become [Test] and an entry of params.yaml data[i].attack_trace_budget,
  // "Budget" its placeholder, a form field (set where the key is, put where it is not)
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-budget-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'data:\n  - name: ascad\n    attack_trace_budget: 250\n  - name: other\n    type: DPAv4\n');
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const state = { open: { 'p:ZvW': true, 't:ZvW': true }, group: { ZvW: 'done' } };
  const { regions, post, appListeners } = render({ state, tree, file });
  const run = { hash: 'h1', name: 'kite', model: 'M', env: 'dev', status: 'DONE', batches: [], spark: '', last: null, ranks: null,
                choices: null, dataset: 'ascad' };
  const stateMsg = { type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [], menus: { rules: {}, refs: {} }, projects: [{
    name: 'ZvW', env: 'dev', envSource: 'test', path: dir, configDir: dir, graphJobsDir: dir, session: null, runSession: null,
    runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'idle' }, run: null, eta: null,
    tiles: { queued: [], running: [], done: [run], failed: [] },
    tests: [{ hash: 'h2', name: 'pierce', model: 'M', env: 'dev', recorded: 'DONE', tests: [], ranks: null, dataset: 'other' }],
    housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} }, queues: [], figures: [], facets: [],
    configs: [{ file: 'params.yaml', path: file }], errors: [], graphJobs: [] }] };
  post(stateMsg);
  const html = Object.values(regions).map(r => r.innerHTML).join('\n');
  const esc = x => x.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  // the Done run: data entry ascad, its budget 250 in the box, a set of data[0].attack_trace_budget
  assert.match(html, new RegExp(`<span class="budget-form form" data-size="Retest Budget" data-file="${esc(file)}" data-project="ZvW" data-sha="s">` +
    '<button data-run="test-run" data-project="ZvW" data-key="h1" data-kind="budget" title="[^"]*250 traces[^"]*">Test</button>' +
    '<input class="val budget" data-budget="1" data-op="set" data-path="\\[&quot;data&quot;,0,&quot;attack_trace_budget&quot;\\]" value="250" placeholder="Budget" size="1"'));
  // the TESTS model: data entry other has no attack_trace_budget: an empty box (the placeholder shows), a put of the key
  assert.match(html, new RegExp('<span class="budget-form form" data-size="Test Budget"[^>]*><button data-run="external-test"[^>]*data-key="h2" data-kind="budget"[^>]*>Test</button>' +
    '<input class="val budget" data-budget="1" data-op="put" data-path="\\[&quot;data&quot;,1\\]" data-key="attack_trace_budget" value="" placeholder="Budget" size="1"'));
  assert.doesNotMatch(html, /unsaved change/);                                 // nothing staged: no OK / Cancel bar
  assert.doesNotMatch(html, />Retest Budget<|>Test Budget</);

  // typing 400 stages it at once (owner 2026-10-28: no click or Enter needed); the redraw shows the box changed, its
  // Test off until OK, and the bar: OK, Cancel, Reset
  const form = { dataset: { file, sha: 's', project: 'ZvW' }, querySelector: () => null };
  const box = { classList: { toggle: () => {} }, value: '400', closest: () => form,
                dataset: { budget: '1', op: 'set', path: JSON.stringify(['data', 0, 'attack_trace_budget']) } };
  appListeners.input({ target: box });
  post(stateMsg);
  const typed = Object.values(regions).map(r => r.innerHTML).join('\n');
  assert.match(typed, /<input class="val budget dirty" data-budget="1" data-op="set"[^>]*value="400"/);
  assert.match(typed, /<button data-run="test-run"[^>]*data-key="h1" data-kind="budget" disabled[^>]*>Test<\/button>/);
  assert.match(typed, /<div class="budget-bar"[^>]*><div class="form"[^>]*><div class="pending"><span>1 unsaved change in params.yaml<\/span>/);
  assert.match(typed, /data-apply="1"[^>]*>OK<\/button><button data-reset="1"[^>]*>Cancel<\/button><button data-budgetreset="1"[^>]*>Reset<\/button>/);
  // Reset: the Budget box's change undone, the box back to the file's 250
  const reset = { dataset: { budgetreset: '1' }, closest: sel => (sel === 'button' ? reset : sel === '.form' ? form : null) };
  appListeners.click({ target: reset, preventDefault: () => {} });
  post(stateMsg);
  const back = Object.values(regions).map(r => r.innerHTML).join('\n');
  assert.match(back, /<input class="val budget" data-budget="1" data-op="set"[^>]*value="250"/);
  assert.doesNotMatch(back, /unsaved change/);
});
