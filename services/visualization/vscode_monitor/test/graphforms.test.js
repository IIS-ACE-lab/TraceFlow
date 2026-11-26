// services/visualization/vscode_monitor/test/graphforms.test.js
'use strict';
// The graph_config.yaml forms drawn from graph_manager's keys (choices.py graph_forms): plot types by
// metric, array_epoch keys by mode, export formats with their settings, legend handle and mode,
// every style.axes key, a text rule panel per target, a facet's legend, panels, layout and text_rules;
// and the drafts (keys the file lacks) written as edits the file reads back.
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
const PY = { cwd: ROOT, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
  env: { ...process.env, CUDA_VISIBLE_DEVICES: '', PYTHONPATH: [ROOT, path.join(ROOT, 'services/side_channel_core'), path.join(ROOT, 'services/side_channel_core/src')].join(':') } };
const lastJson = out => JSON.parse(out.trim().split('\n').pop());
const STATIC = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.choices', 'static'], PY));

const CONFIG = `jobs:
  - id: 1
    name: rank
    type: model_overlay
    enabled: true
    source: cross_validation
    metric: mean_attack_rank_curve
    plot_type: line
    models: [a_run, b_run]
    array_epoch:
      mode: latest
      epoch: 100
    export:
      formats: [png, svg]
      dpi: 300
    legend:
      mode: auto
      handle: ["line", 1.0]
      labels:
        a_run: Adam
    style:
      axes:
        x_limits: [0, 950]
      text_rules:
        - target: title
          text: Rank
  - id: 2
    name: acc
    type: solo
    enabled: true
    source: cross_validation
    metric: mean_accuracy
    models: [a_run]
faceting:
  - id: 1
    name: two
    panels:
      - file_id: 1
        address: jobs/job_001_rank/a.png
      - file_id: 2
        address: jobs/job_002_acc/b.png
        label: (b) SGD
    arrangement:
      mode: horizontal
      order: [1, 2]
    legend:
      enabled: true
      mode: auto
    style:
      text_rules: []
`;

function render(file, open, extra = {}) {             // extra: {state, project} beside the defaults
  const tree = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).tree;
  const regions = {};
  const on = {};                                        // the webview's own listeners (change)
  const app = {
    set innerHTML(html) { for (const m of html.matchAll(/id="([^"]+)"/g)) regions[m[1]] = { innerHTML: '', contains: () => false }; },
    querySelectorAll: sel => (sel === '[id]' ? Object.entries(regions).map(([id, el]) => Object.assign(el, { id })) : []),
    addEventListener: (type, fn) => { on[type] = fn; },
  };
  const listeners = {};
  const hook = {};
  const k = p => `n:${file}:${JSON.stringify(p)}`;
  const state = { open: { 'p:P': true, 'c:P': true, [`cf:${file}`]: true, ...Object.fromEntries(open.map(p => [k(p), true])) }, ...(extra.state || {}) };
  const ctx = {
    acquireVsCodeApi: () => ({ postMessage: () => {}, getState: () => state, setState: () => {} }),
    document: { body: { dataset: { mode: 'main' } }, getElementById: () => app, activeElement: null },
    window: { addEventListener: (type, fn) => { listeners[type] = fn; }, TFMenus: M, TFPanelTest: hook },
    console,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../webview/panel.js'), 'utf8'), ctx);
  listeners.message({ data: { type: 'tree', file, sha256: 's', tree, error: '', note: '' } });
  listeners.message({ data: { type: 'state', busy: [], gpu: null, timezone: 'UTC', snapshots: {}, errors: [],
    menus: { rules: STATIC.rules, graph: STATIC.graph, refs: {} }, projects: [{
      name: 'P', env: 'dev', envSource: 'test', path: path.dirname(file), configDir: path.dirname(file), graphJobsDir: path.dirname(file),
      session: null, runSession: null, runAttached: 0, modelIds: [], datasets: [], light: { color: 'grey', label: 'idle' }, run: null, eta: null,
      tiles: { queued: [], running: [], done: [], failed: [] }, tests: [], housekeeping: { runs: [], total: 0, error: '', note: '', filters: {} },
      queues: [], figures: [], configs: [{ file: 'graph_config.yaml', path: file }], errors: [], graphJobs: [], ...(extra.project || {}) }] } });
  return { html: Object.values(regions).map(r => r.innerHTML).join('\n'), tree, hook, on,
    now: () => Object.values(regions).map(r => r.innerHTML).join('\n') };
}

test('plot types follow the metric: graph_manager and the panel agree', () => {
  const pt = STATIC.graph.plotTypes;
  const names = ['mean_accuracy', 'accuracy', 'loss', 'mean_attack_rank_curve', 'guessing_entropy_curve', 'sr_curve_3', 'full_ranks',
    'final_ranks', 'LDD', 'fold_rank_curves', 'key_candidate_scores', 'mean_fit_time_s', 'stable_rank1_trace'];
  names.push(...Object.keys(pt.kinds), 'mean_fit_time_{s}', 'fit_time_s', 'val_loss', 'sr_curve_10', 'unknown_metric');
  const cases = [];
  for (const type of ['solo', 'process_overlay', 'model_overlay', 'model_explanation']) {
    for (const mode of ['latest', 'selected', 'overlay', 'summary']) for (const n of names) cases.push([type, [n], mode]);
    for (const mode of ['latest', 'overlay']) {
      cases.push([type, ['mean_accuracy', 'mean_attack_rank_curve'], mode], [type, ['mean_accuracy', 'mean_fit_time_s'], mode],
        [type, ['full_ranks', 'mean_attack_rank_curve'], mode]);
    }
  }
  const py = lastJson(execFileSync(E.python(), ['-c', `import json,sys
from services.graph_manager.metric_router import plot_types_for
print(json.dumps([plot_types_for(t, m, mode) for t, m, mode in json.loads(sys.argv[1])]))`, JSON.stringify(cases)], PY));
  assert.deepEqual(cases.map(([t, m, mode]) => M.plotTypesFor(pt, t, m, mode)), py);
});

test('the graph forms show every key graph_manager reads, faded where it does not apply', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-gf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const j = (...p) => ['jobs', 0, ...p];
  const f = (...p) => ['faceting', 0, ...p];
  const { html } = render(file, [['jobs'], j(), j('array_epoch'), j('export'), j('legend'), j('style'), j('style', 'axes'), j('style', 'text_rules'),
    j('style', 'text_rules', '@title'), j('style', 'text_rules', '@x_label'), ['jobs', 1], ['faceting'], f(), f('legend'), f('export'), f('panels'),
    f('panels', 1), f('text_rules'), f('text_rules', '@panel_label')]);
  if (process.env.TF_DUMP) fs.writeFileSync(process.env.TF_DUMP, html);
  const sel = p => new RegExp(`<select[^>]*data-path="${JSON.stringify(p).replace(/"/g, '&quot;').replace(/[[\]]/g, '\\$&')}"[^>]*>([\\s\\S]*?)</select>`);
  // 1. plot_type: a curve in an overlay takes no bar; a solo scalar takes bar, not barh
  const opts = m => [...m[1].matchAll(/<option value="([^"]*)"/g)].map(x => x[1]);
  assert.deepEqual(opts(html.match(sel(j('plot_type')))), ['auto', 'line', 'line_scatter', 'scatter', 'step']);
  // 2. array_epoch: epoch is faded and unavailable under mode latest
  assert.match(html, /<div class="faded" title="used with array_epoch.mode: selected \(now latest\)">[\s\S]*?<input disabled/);
  // 3. export: svg has its settings, png its dpi (a draft: the file has none), the old dpi is flagged
  assert.match(html, /<div class="fmt-name">png<\/div><div class="field"><label[^>]*>dpi<\/label><input class="val absent" data-draft/);
  assert.match(html, /<div class="fmt-name">svg<\/div>[\s\S]*?sanitize[\s\S]*?validate_xml[\s\S]*?on_error/);
  assert.match(html, /<label>dpi<\/label><span class="note">300: moved to png.dpi<\/span>/);
  // the facet's default: svg only (owner 2026-10-27), no png / pdf
  const facetHtml = html.slice(html.indexOf('faceting'));
  assert.match(facetHtml, /<div class="fmt-name">svg<\/div>/);
  assert.doesNotMatch(facetHtml, /<div class="fmt-name">(png|pdf)<\/div>/);
  // 4. legend: the handle menu offers Character…; mode auto fades the labels
  assert.match(html, /data-handlepart="shape">[\s\S]*?<option value="line" selected>line<\/option>[\s\S]*?<option value="__char">Character…<\/option>/);
  assert.match(html, /<div class="faded" title="mode auto: the legend shows the model names[^"]*">[\s\S]*?a_run/);
  // 5. axes: every key; x_limits is Custom with its range, y_limits Auto
  for (const key of ['x_scale', 'y_scale', 'x_limits', 'y_limits', 'x_step', 'y_step', 'x_grid', 'x_grid_first', 'x_margin', 'y_margin']) assert.match(html, new RegExp(`>${key}</label>`));
  assert.match(html, /x_limits<\/label><select class="val" data-autopart="mode"><option value="auto">Auto<\/option><option value="custom" selected>Custom<\/option><\/select><input class="val" data-autopart="0" size="7" value="0"/);
  assert.match(html, /y_limits<\/label><select class="val" data-autopart="mode"><option value="auto" selected>Auto/);
  // 6. text rules: a panel per target, its fields under it (x_label not set yet: drafts)
  for (const t of ['title', 'x_label', 'y_label', 'legend_text']) assert.match(html, new RegExp(`<summary[^>]*>${t} <small>`));
  assert.match(html, /<summary[^>]*>x_label <small>not set: graph_manager's default<\/small><\/summary>[\s\S]*?italic[\s\S]*?font.family[\s\S]*?font.size/);
  assert.doesNotMatch(html, /underline/);
  // job 2 (solo scalar, no plot_type key): export, legend and style drawn from the defaults
  assert.match(html, /<summary[^>]*>export <small>svg<\/small><\/summary>/);                  // the default: svg only
  // 7. facet: legend handle with patch, entries faded under auto; a panel's label fields
  assert.match(html, /<option value="patch">patch<\/option>/);
  assert.match(html, /mode auto: the entries are the first panel&#39;s graph job&#39;s/);
  assert.match(html, />label_position<\/label><select class="val absent" data-draft/);
  // 8. layout: a 4 x 4 grid to pick from; Rows / Columns give its size
  assert.match(html, /<span class="tp-dim">Rows <input type="number" min="1" max="20" data-gridrows="1"[^>]*value="4"><\/span><span class="tp-dim">Columns <input[^>]*value="4">/);
  assert.doesNotMatch(html.slice(html.indexOf('tp-size'), html.indexOf('facet-cells')), /<label/);   // a .field label is a clipped 38% column
  assert.equal((html.match(/<div class="tp-row">/g) || []).length, 4);
  assert.match(html, /data-tablepick="4x4"/);
  // 9. facet: text_rules is its own (panel_label among its targets); the old style block is flagged
  assert.match(html, /<summary[^>]*>panel_label <small>/);
  assert.match(html, /<label>style<\/label><span class="note">\{…\}: moved to the facet&#39;s text_rules/);
});

test('drafts become edits the file reads back: new keys, new blocks, new text rules', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-gf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const { tree, hook } = render(file, []);
  const drafts = [
    { path: ['jobs', 0, 'export', 'png', 'dpi'], value: 150 },
    { path: ['jobs', 0, 'style', 'axes', 'y_limits'], value: [0, 128] },
    { path: ['jobs', 0, 'style', 'text_rules', { rule: 'title' }, 'italic'], value: true },
    { path: ['jobs', 0, 'style', 'text_rules', { rule: 'x_label' }, 'text'], value: 'Traces' },
    { path: ['jobs', 0, 'style', 'text_rules', { rule: 'x_label' }, 'font', 'size'], value: 9 },
    { path: ['jobs', 1, 'style', 'text_rules', { rule: 'title' }, 'text'], value: 'Accuracy' },
    { path: ['jobs', 1, 'legend', 'handle'], value: ['■', 4] },
    { path: ['faceting', 0, 'panels', 0, 'label_position'], value: 'top-left' },
    { path: ['faceting', 0, 'text_rules', { rule: 'panel_label' }, 'text'], value: '({letter})' },
    { path: ['faceting', 0, 'export', 'png', 'dpi'], value: 200 },
  ];
  const ops = hook.draftOps(tree, drafts.map(d => ({ op: 'draft', ...d })));
  const rank = e => ({ set: 0, put: 0, remove: 1, delete: 1, add: 2 })[e.op];
  for (const e of ops.map((e, i) => ({ ...e, i })).sort((a, b) => rank(a) - rank(b) || a.i - b.i)) {
    const sha = lastJson(execFileSync(E.python(), ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file], PY)).sha256;
    const args = ['-m', 'services.visualization.vscode_monitor.yamledit', 'edit', file, '--sha', sha, '--op', e.op, '--path', JSON.stringify(e.path), `--value=${e.value}`];
    if (e.op === 'put') args.push(`--key=${e.key}`);
    const out = lastJson(execFileSync(E.python(), args, PY));
    assert.ok(out.ok, `${JSON.stringify(e)}: ${JSON.stringify(out)}`);
  }
  const data = JSON.parse(execFileSync(E.python(), ['-c', 'import json,sys,yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))))', file], PY));
  const [j0, j1] = data.jobs;
  assert.deepEqual(j0.export.png, { dpi: 150 });
  assert.deepEqual(j0.style.axes, { x_limits: [0, 950], y_limits: [0, 128] });
  assert.deepEqual(j0.style.text_rules, [{ target: 'title', text: 'Rank', italic: true }, { target: 'x_label', text: 'Traces', font: { size: 9 } }]);
  assert.deepEqual(j1.style, { text_rules: [{ target: 'title', text: 'Accuracy' }] });
  assert.deepEqual(j1.legend, { handle: ['■', 4] });
  assert.equal(data.faceting[0].panels[0].label_position, 'top-left');
  assert.deepEqual(data.faceting[0].text_rules, [{ target: 'panel_label', text: '({letter})' }]);
  assert.deepEqual(data.faceting[0].export, { png: { dpi: 200 } });
  const text = fs.readFileSync(file, 'utf8');
  assert.match(text, /\n {4}style:\n {6}text_rules:\n {8}- target: title\n {10}text: Accuracy\n/);      // block style, not JSON
  assert.match(text, /\n {4}legend:\n {6}handle: \[■, 4\]\n/);
});

test('job forms: ids and an exported job\'s metric shown, formats as tick boxes, array_epoch latest, the style and text rule fixes', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-gf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG.replace('    metric: mean_accuracy\n', '    metric: mean_accuracy\n    processes:\n      - validation\n'));
  const j = (...p) => ['jobs', 0, ...p];
  const titleKey = JSON.stringify(j('style', 'text_rules', 0, 'color'));             // the title rule is the file's first
  const { html } = render(file, [['jobs'], j(), j('export'), j('legend'), j('style'), j('style', 'axes'), j('style', 'text_rules'), j('style', 'text_rules', '@title'),
    j('style', 'text_rules', '@legend_text'), ['jobs', 1], ['jobs', 1, 'array_epoch'], ['jobs', 1, 'processes'], ['faceting'], ['faceting', 0]],
  { project: { graphJobs: [{ id: 1, exported: true }, { id: 2, exported: false }] }, state: { cpick: { [titleKey]: true }, cmore: { [titleKey]: true } } });
  const job1 = html.slice(html.indexOf('data-key="n:' + file + ':[&quot;jobs&quot;,0]"'), html.indexOf('data-key="n:' + file + ':[&quot;jobs&quot;,1]"'));
  const job2 = html.slice(html.indexOf('data-key="n:' + file + ':[&quot;jobs&quot;,1]"'), html.indexOf('data-key="n:' + file + ':[&quot;faceting&quot;]"'));
  // [Add a Job] heads the jobs
  assert.match(html, /<div class="buttons"><button data-addjob="P"[^>]*>Add a Job<\/button><\/div>/);
  // 1. ids are typed (an integer; Add a Job / Add a Facet give the next one), jobs and facets alike
  assert.match(job1, /<label[^>]*>id<\/label><input class="val" data-op="set" data-path="\[&quot;jobs&quot;,0,&quot;id&quot;\]" value="1" title="an integer, unique in jobs: make graph GRAPH_ID=<id> draws it[^"]*">/);
  assert.match(html.slice(html.indexOf('faceting')), /<label[^>]*>id<\/label><input class="val" data-op="set" data-path="\[&quot;faceting&quot;,0,&quot;id&quot;\]" value="1" title="an integer, unique in faceting: make graph FACE_ID=<id>[^"]*">/);
  // × at the right end of a job's and a facet's line (by id: the figures move aside, later ids move down)
  assert.match(job1, /<summary[^>]*>#0 <small>name: rank<\/small><span class="end"><button class="x danger jobdel" data-jobdel="P" data-id="1"[^>]*>×<\/button><button class="jobdraw"[^>]*>Draw<\/button><\/span><\/summary>/);
  // 2. an exported job's metric is shown, a job made by hand keeps its menu
  assert.match(job1, /<label[^>]*>metric<\/label><span class="readonly"[^>]*>mean_attack_rank_curve<\/span>/);
  assert.match(job2, /data-path="\[&quot;jobs&quot;,1,&quot;metric&quot;\]"/);
  // 4. a job without array_epoch reads latest, and says so
  assert.match(job2, /<summary[^>]*>array_epoch <small>mode latest<\/small><\/summary>[^]*?<select class="val absent" data-draft="[^"]*array_epoch[^"]*mode[^"]*"[^>]*><option value="latest" selected>latest<\/option><option value="overlay">/);
  // 4b. a job without trace_budget or test offers both, empty (the run's curves and its latest All test)
  assert.match(job2, /<label[^>]*>trace_budget<\/label>[^]*?data-draft="\[&quot;jobs&quot;,1,&quot;trace_budget&quot;\]"/);
  assert.match(job2, /<label[^>]*>test<\/label>[^]*?data-draft="\[&quot;jobs&quot;,1,&quot;test&quot;\]"/);
  // 5. formats: a row of tick boxes; only the ticked formats show their settings
  assert.match(job1, /<span class="fmtboxes"><label class="fmtbox"><input type="checkbox" data-fmtcheck="eps"[^>]*> eps<\/label><label class="fmtbox"><input type="checkbox" data-fmtcheck="pdf"[^>]*> pdf<\/label><label class="fmtbox"><input type="checkbox" data-fmtcheck="png"[^>]* checked> png<\/label><label class="fmtbox"><input type="checkbox" data-fmtcheck="svg"[^>]* checked> svg<\/label><\/span>/);
  assert.match(job1, /<div class="fmt-name">png<\/div>/);
  assert.match(job1, /<div class="fmt-name">svg<\/div>/);
  assert.doesNotMatch(job1, /<div class="fmt-name">(pdf|eps)<\/div>/);
  // 6. axes: no ×; a Custom range reads lower limit / upper limit
  const axes = job1.slice(job1.search(/<summary[^>]*>axes/), job1.search(/<summary[^>]*>text_rules/));
  assert.doesNotMatch(axes, /class="x"/);
  assert.match(axes, /placeholder="lower limit">[^]*?placeholder="upper limit">/);
  // 7. text rules: the rule's × at its line's right end; text has Clear, no ×; bold, italic True / False; colour a palette; fonts a menu
  const title = job1.slice(job1.search(/<summary[^>]*>title/), job1.search(/<summary[^>]*>x_label/));
  assert.match(title, /^<summary[^>]*>title <small>Rank<\/small><\/summary>/);                    // no × on a target's line
  assert.match(title, /<label[^>]*>text<\/label><input class="val" data-op="set"[^>]*value="Rank"[^>]*><button data-clear="1"[^>]*>Clear<\/button><\/div>/);
  assert.doesNotMatch(title, /class="x"/);
  for (const key of ['bold', 'italic']) {
    assert.match(title, new RegExp(`<label[^>]*>${key}</label><select class="val absent" data-draft[^>]*><option value="true">True</option><option value="false" selected>False</option></select></div>`));
  }
  assert.match(title, /<label[^>]*>color<\/label><button class="cpick" data-cpick="[^"]*"[^>]*><svg class="swatch"[^>]*><rect[^>]*fill="#000000"\/><\/svg> black ▾<\/button><div class="cpalette">/);
  assert.equal((title.match(/<button class="cp[ "]/g) || []).length, 70);                      // 10 x 6 theme, 10 standard
  assert.match(title, /data-cset="#4472C4"[^>]*title="Blue, Accent 1"/);
  assert.match(title, /title="Blue, Accent 1, Lighter 80%"/);
  assert.match(title, /data-cset="#C00000"[^>]*title="Dark Red"/);
  assert.match(title, /<button class="cp-auto" data-cset="black"[^]*?Automatic<\/button>/);
  assert.match(title, /<button class="cp-morebtn on" data-cmorebtn="[^"]*">More Colors…<\/button><button class="honey" data-honey="1" data-cdraft="[^"]*"[^>]*><svg class="honeycomb"/);   // the file has no colour yet: a draft
  const honey = title.slice(title.indexOf('<svg class="honeycomb"'));
  assert.equal((honey.match(/<polygon /g) || []).length, 127 + 2 + 14);                    // the honeycomb, white and black, the greys
  assert.match(honey, /fill="#FFFFFF" data-hex="#FFFFFF"/);
  assert.match(honey, /fill="#000000" data-hex="#000000" class="on"/);                     // black, the colour now
  assert.doesNotMatch(title, /type="color"/);                                              // no system dialog
  // legend and export: no ×; legend_text's text is {model_name}, shown
  const legend = job1.slice(job1.search(/<summary[^>]*>legend/), job1.search(/<summary[^>]*>style/));
  const exp = job1.slice(job1.search(/<summary[^>]*>export/), job1.search(/<summary[^>]*>legend/));
  assert.doesNotMatch(legend.slice(0, legend.indexOf('labels')), /class="x"/);
  assert.doesNotMatch(exp.replace(/<div class="field stray[^]*?<\/div>/g, ''), /class="x"/);   // a stray key (dpi, moved to png.dpi) keeps its ×
  const lt = job1.slice(job1.search(/<summary[^>]*>legend_text/));
  assert.match(lt, /^<summary[^>]*>legend_text <small>\{model_name\}<\/small><\/summary><div class="field"><label>text<\/label><span class="readonly"[^>]*>\{model_name\}<\/span><\/div>/);
  // a job's processes: [+]
  assert.match(job2, /data-op="add" data-path="\[&quot;jobs&quot;,1,&quot;processes&quot;\]"[^>]*><option value="">\+<\/option>/);
  assert.match(title, /<label[^>]*>font.family<\/label><select class="val absent" data-draft[^>]*><option value="" selected>serif \(default\)<\/option><option value="serif">serif<\/option><option value="sans-serif">sans-serif<\/option><option value="monospace">monospace<\/option><option value="Latin Modern Mono">/);
  assert.match(title, /<label[^>]*>font.size<\/label><input class="val absent"[^>]*><\/div>/);
});

test('faceting: Add a Facet heads it (also when the file has none), + add a panel takes the next file_id', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-gf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG);
  const { html } = render(file, [['faceting'], ['faceting', 0], ['faceting', 0, 'panels']]);
  // under the section's own OK row (empty until a value changes): Add a Facet
  assert.match(html, /<summary[^>]*>faceting <small>1<\/small><\/summary><div class="buttons"><\/div><div class="buttons"><button data-addfacet="P"[^>]*>Add a Facet<\/button><\/div>/);
  assert.match(html, /<summary[^>]*>panels <small>2<\/small><\/summary>[^]*?<button data-addpanel="1" data-path="\[&quot;faceting&quot;,0,&quot;panels&quot;\]" data-next="3"[^>]*>\+ add a panel<\/button>/);
  const bare = path.join(dir, 'bare', 'graph_config.yaml');
  fs.mkdirSync(path.dirname(bare));
  fs.writeFileSync(bare, CONFIG.slice(0, CONFIG.indexOf('faceting:')));
  const none = render(bare, [['faceting']]).html;
  assert.match(none, /<summary[^>]*>faceting <small>0<\/small><\/summary><div class="buttons"><button data-addfacet="P"[^>]*>Add a Facet<\/button><\/div><p class="note">none yet<\/p>/);
});

test('a facet\'s custom legend: a row per colour, its colour and its model engraved, then the label; + adds a row', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-gf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, CONFIG.replace('    legend:\n      enabled: true\n      mode: auto\n',
    '    legend:\n      enabled: true\n      mode: custom\n      entries:\n        - label: Adam\n'));
  const f = (...p) => ['faceting', 0, ...p];
  const { html } = render(file, [['faceting'], f(), f('legend')]);
  const legend = html.slice(html.search(/<summary[^>]*>legend <small>mode custom/));
  // the first panel is job 1's (jobs/job_001_rank/…): its models a_run, b_run take C0, C1
  assert.match(legend, /<div class="field entry-row"><span class="readonly entry-tag" title="entry 1: colour C0"><svg class="swatch"[^]*?fill="#1f77b4"[^]*?<\/svg> C0 · a_run<\/span><input class="val" data-op="set" data-path="\[&quot;faceting&quot;,0,&quot;legend&quot;,&quot;entries&quot;,0,&quot;label&quot;\]" value="Adam"/);
  assert.match(legend, /<div class="field entry-row"><span class="readonly entry-tag" title="entry 2: colour C1">[^]*? C1 · b_run<\/span><input class="val" data-addmap="label" data-path="\[&quot;faceting&quot;,0,&quot;legend&quot;,&quot;entries&quot;\]" placeholder="its label, e.g. b_run">/);
  assert.equal((legend.match(/class="field entry-row"/g) || []).length, 2);
  assert.match(legend, /<button data-entryrow="[^"]*"[^>]*>\+<\/button>/);
  assert.doesNotMatch(legend.slice(0, legend.indexOf('entries')), /class="faded"/);          // custom: not faded
});

test('a key the file lacks: its draft goes when the value shown at first is chosen again', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, 'jobs:\n  - id: 0\n    title: t\n    enabled: true\n    metric: [mean_accuracy]\n    models: []\n');
  const p = ['jobs', 0, 'array_epoch', 'mode'];
  const r = render(file, [['jobs'], ['jobs', 0], ['jobs', 0, 'array_epoch']]);
  const dflt = STATIC.graph.arrayEpoch.default;
  assert.match(r.html, new RegExp(`data-draft="${JSON.stringify(p).replace(/"/g, '&quot;').replace(/[[\]]/g, '\\$&')}" data-t="enum" data-orig="${dflt}"`));
  const form = { dataset: { file, sha: 's', project: 'P' } };
  const pick = value => r.on.change({ target: { dataset: { draft: JSON.stringify(p), t: 'enum', orig: dflt }, value, blur: () => {},
    closest: sel => (sel === '.form' ? form : null) } });
  pick(STATIC.graph.arrayEpoch.modes.find(m => m !== dflt));
  assert.match(r.now(), /class="pending"/);
  pick(dflt);                                                            // back to what was shown
  assert.doesNotMatch(r.now(), /class="pending"/);
});

test('a job or facet without an id (a template\'s entry): × at the right end deletes the entry itself; no Draw', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-gf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, 'jobs:\n  - id:\n    name:\n    type:\nfaceting:\n  - id:\n    name: f\n');
  const { html } = render(file, [['jobs'], ['jobs', 0], ['faceting'], ['faceting', 0]]);
  assert.match(html, /<summary[^>]*>#0 <small>name: <\/small><span class="end"><button class="x" data-delentry="job" data-name="name: " data-path="\[&quot;jobs&quot;,0\]" title="delete this job from graph_config.yaml \(asks first\)">×<\/button><\/span><\/summary>/);
  assert.match(html, /<summary[^>]*>#0 <small>name: f<\/small><span class="end"><button class="x" data-delentry="facet" data-name="name: f" data-path="\[&quot;faceting&quot;,0\]"[^>]*>×<\/button><\/span><\/summary>/);
  assert.doesNotMatch(html, /data-run="draw"/);
  assert.match(html, /<label[^>]*>id<\/label><input class="val" data-op="set" data-path="\[&quot;jobs&quot;,0,&quot;id&quot;\]" value=""/);
});

test('a job of a new project: its empty true/false keys are true / false menus; processes is rows and a menu to add, in flow or block style, also while empty', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-gf-'));
  const file = path.join(dir, 'graph_config.yaml');
  fs.writeFileSync(file, 'jobs:\n  - id: 0\n    name: a\n    enabled:\n    processes: []\n    style:\n      grid:\n        enabled:\n' +
    '  - id: 1\n    name: b\n    processes: [train]\n');
  const { html } = render(file, [['jobs'], ['jobs', 0], ['jobs', 0, 'processes'], ['jobs', 0, 'style'], ['jobs', 0, 'style', 'grid'], ['jobs', 1], ['jobs', 1, 'processes']]);
  const at = p => `data-path="${JSON.stringify(p).replace(/"/g, '&quot;').replace(/[[\]]/g, '\\$&')}"`;
  for (const p of [['jobs', 0, 'enabled'], ['jobs', 0, 'style', 'grid', 'enabled']]) {
    assert.match(html, new RegExp(`<select class="val" data-op="set" ${at(p)}[^>]*><option value="" selected>\\(empty\\)</option><option value="true">true</option><option value="false">false</option></select>`), p.join('.'));
  }
  // processes: a block with its add menu while empty; a flow list: a row per process (menu, ×) and the add menu
  assert.match(html, /<summary[^>]*>processes <small>0<\/small><\/summary>[^]*?<select class="add wide" data-op="add" data-path="\[&quot;jobs&quot;,0,&quot;processes&quot;\]"[^>]*><option value="">\+<\/option><option value="test">test<\/option>/);
  assert.doesNotMatch(html, /<label[^>]*>processes<\/label><span class="values">/, 'never an inline row of values');
  assert.match(html, /<summary[^>]*>processes <small>1<\/small><\/summary>[^]*?<select class="val" data-op="set" data-path="\[&quot;jobs&quot;,1,&quot;processes&quot;,0\]"[^>]*>[^]*?<option value="train" selected>train<\/option>[^]*?<button class="x" data-op="remove" data-path="\[&quot;jobs&quot;,1,&quot;processes&quot;,0\]"[^>]*>×<\/button>/);
});
