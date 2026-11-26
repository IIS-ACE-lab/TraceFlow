// services/visualization/vscode_monitor/test/actions.test.js
'use strict';
// The buttons of extension.js with VS Code stubbed; nothing here starts a run.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const seen = { warning: [], info: [] };
// A dialog's whole text: its heading, and the detail a modal one wraps (extension.js dialog).
// Read as one line: dialogs are hard-wrapped (extension.js wrapText), so a phrase may span a break.
const text = a => String(a[1] && a[1].detail ? `${a[0]}\n${a[1].detail}` : a[0]).replace(/\s+/g, ' ');
const load = Module._load;
const vscodeStub = {                                  // one object: a test may replace its parts
  Uri: { file: f => ({ fsPath: f }) },
  workspace: { getConfiguration: () => ({ get: () => undefined }) },
  window: {
    showWarningMessage: async (...a) => { seen.warning.push(text(a)); return undefined; },
    // A choice (extension.js choose: the ENV, the reproduction's destination) answers with its first
    // button, as a pick list's first item did; every other dialog is cancelled.
    showInformationMessage: async (...a) => { seen.info.push(text(a)); return /^(ENV for|Where should)/.test(a[0]) && a.length > 2 ? a[2] : undefined; },
    showQuickPick: async items => items[0],
  },
};
Module._load = function (request, ...rest) {
  if (request === 'vscode') return vscodeStub;
  return load.call(this, request, ...rest);
};
const E = require('../extension');
const ROOT = path.resolve(__dirname, '../../../..');

function panel(project) {
  const p = new E.Panel({});
  p.root = ROOT;
  p.python = E.findPython();
  p.layout = { projects: [project] };
  return p;
}

const P = {
  name: 'DLSCA_ASCAD', model_ids: ['ASCAD_MLPbest'], datasets: ['ascad'],
  queues: [{ key: 'fig26', env: 'dev', commands: [{}] }, { key: 'fig2', env: 'dev', commands: [{}, {}, {}] }],
};

test('Start refuses while an experiment runs on the machine', async () => {
  const p = panel(P);
  p.pollProcesses = async () => { p.busy = ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26']; p.sessions = {}; };
  seen.warning.length = 0;
  await p.run({ action: 'queue', project: 'DLSCA_ASCAD', key: 'fig2' });
  assert.match(seen.warning[0], /not started. Let the current run finish: an experiment is running on this machine/);
  assert.match(seen.warning[0], /QUEUE=fig26/);
});

test('Start refuses while the project session still runs; asks with the planned count otherwise', { timeout: 60000 }, async () => {
  const p = panel(P);
  p.pollProcesses = async () => { p.busy = []; p.sessions = { 'tf-DLSCA_ASCAD': { command: 'make', busy: true } }; };
  seen.warning.length = 0;
  await p.run({ action: 'test', project: 'DLSCA_ASCAD' });
  assert.match(seen.warning[0], /tf-DLSCA_ASCAD is still running make/);
  p.pollProcesses = async () => { p.busy = []; p.sessions = {}; };
  seen.info.length = 0;
  await p.run({ action: 'queue', project: 'DLSCA_ASCAD', key: 'fig2' });          // Cancel: nothing starts
  assert.match(seen.info[0], /make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig2/);
  if (p.python !== 'python3') assert.match(seen.info[0], /plans \d+ experiment\(s\) \(\d+ per command × 3 commands\)/);
  await p.run({ action: 'orchestrate', project: 'DLSCA_ASCAD' });
  assert.match(seen.info[1], /^ENV for make orchestrate\? [\s\S]*picks the compose files/);              // a dialog: its detail wraps
  assert.match(seen.info[2], /make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest/);
});

test('Draw opens the job figure written after the click', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const p = panel({ ...P, graph_jobs_dir: dir, graph_jobs: [{ id: 64, name: 'fig', enabled: true, dir: 'job_064_fig', runs: [] }] });
  const lines = [];
  const opened = [];
  p.terminal = (name, line) => lines.push([name, line]);
  p.open = f => opened.push(f);
  try {
    await p.run({ action: 'draw', project: 'DLSCA_ASCAD', id: 64 });   // nothing ran here: the ENV is picked (dev)
    assert.match(lines[0][1], /conda activate sca && make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=64$/);
    fs.mkdirSync(path.join(dir, 'job_064_fig'));
    fs.writeFileSync(path.join(dir, 'job_064_fig', 'a.png'), '');
    fs.writeFileSync(path.join(dir, 'job_064_fig', 'a.pdf'), '');
    await new Promise(r => setTimeout(r, 1500));
    assert.deepEqual(opened, [path.join(dir, 'job_064_fig', 'a.pdf')]);
  } finally { p.stop(); }
});

test('Draw all draws the ticked jobs one after another in one terminal, a failure not stopping the rest', async () => {
  const p = panel(P);
  const lines = [];
  p.terminal = (name, line) => lines.push([name, line]);
  try {
    await p.run({ action: 'draw-all', project: 'DLSCA_ASCAD', ids: ['3', '1', 'x'] });
    assert.equal(lines.length, 1);
    assert.match(lines[0][1], /conda activate sca && \{ make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=3; make graph ENV=dev PROJECT=DLSCA_ASCAD GRAPH_ID=1; \}$/);
    await p.run({ action: 'draw-all', project: 'DLSCA_ASCAD', ids: [] });      // nothing ticked: nothing runs
    assert.equal(lines.length, 1);
  } finally { p.stop(); }
});

test('Faceting → Draw all draws the ticked facets one after another in one terminal', async () => {
  const p = panel(P);
  const lines = [];
  p.terminal = (name, line) => lines.push([name, line]);
  try {
    await p.run({ action: 'draw-all-facets', project: 'DLSCA_ASCAD', ids: ['2', '1'] });
    assert.equal(lines.length, 1);
    assert.match(lines[0][1], /conda activate sca && \{ make graph ENV=dev PROJECT=DLSCA_ASCAD FACE_ID=2; make graph ENV=dev PROJECT=DLSCA_ASCAD FACE_ID=1; \}$/);
  } finally { p.stop(); }
});

test('the real process poll sees running experiments', async () => {
  const p = panel(P);
  await p.pollProcesses();
  assert.ok(Array.isArray(p.busy) && typeof p.sessions === 'object');
  console.log('busy here:', p.busy, 'sessions:', p.sessions);
});

const { execFileSync, spawnSync } = require('child_process');
const C = require('../commands');
const tmuxOk = spawnSync('tmux', ['-V']).status === 0 && spawnSync('bash', ['-lc', 'conda info --base']).status === 0;

test('a panel-started run reads busy, ignores a typed Ctrl-C, and Stop leaves the session open and idle', { skip: !tmuxOk && 'tmux or conda missing', timeout: 60000 }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const name = `tf-busy-${process.pid}`;
  const p = panel(P);
  const wait = async (ok) => { for (let i = 0; i < 100 && !(await ok()); i++) await new Promise(r => setTimeout(r, 200)); };
  try {
    execFileSync('bash', ['-c', C.tmuxStart(name, dir, 'sleep 300', 'projects/P/artifacts/dev/tmux_logs/b.txt')]);
    await wait(async () => { await p.pollProcesses(); return p.sessions[name] && p.sessions[name].busy; });
    assert.equal(p.sessions[name].busy, true);                                 // pane_current_command says bash here
    // a Ctrl-C typed into the pane (as VS Code's Python extension injects into a revived terminal) and
    // the text after it must not stop the run
    execFileSync('tmux', ['send-keys', '-t', name, 'C-c']);
    execFileSync('tmux', ['send-keys', '-t', name, ' conda activate sca', 'Enter']);
    await new Promise(r => setTimeout(r, 1500));
    await p.pollProcesses();
    assert.equal(p.sessions[name].busy, true);
    execFileSync('bash', ['-c', C.tmuxStop(name)]);
    await wait(async () => { await p.pollProcesses(); return p.sessions[name] && !p.sessions[name].busy; });
    assert.equal(p.sessions[name].busy, false);
    const pane = execFileSync('tmux', ['capture-pane', '-p', '-S', '-', '-t', name]).toString();
    assert.match(pane, /finished, exit/);
  } finally {
    spawnSync('tmux', ['kill-session', '-t', name]);
  }
});

test('Attach and Kill session find the tmux session whose pane runs the make, whoever named it', { skip: !tmuxOk && 'tmux missing', timeout: 30000 }, async () => {
  const name = `tfhand-${process.pid}`;                      // a session typed by hand, not tf-<project>
  const p = panel({ ...P, name: 'TFTEST' });
  try {
    // a process whose command line is a running `make experiment … PROJECT=TFTEST` (argv[0] renamed)
    execFileSync('tmux', ['new-session', '-d', '-s', name,
      `bash -c 'exec -a make python3 -c "import time; time.sleep(60)" experiment ENV=dev PROJECT=TFTEST QUEUE=q'`]);
    for (let i = 0; i < 50 && !(p.runSessions || []).length; i++) {
      await new Promise(r => setTimeout(r, 100));
      await p.pollProcesses();
      p.runSessions = (p.runSessions || []).filter(s => s.session === name);
    }
    assert.equal(p.runSessionFor({ name: 'TFTEST' }).session, name);
  } finally {
    spawnSync('tmux', ['kill-session', '-t', name]);
  }
});

test('the tmux button follows the session running the run\'s own batch', () => {
  const p = panel(P);
  p.runSessions = [
    { session: 'q-other', cmds: ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig2'], make: true, attached: 1 },
    { session: 'q-mine', cmds: ['make orchestrate ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26 BATCH_ID=fig26-1'], make: true, attached: 0 },
  ];
  assert.equal(p.runSessionFor(P, ['fig26']).session, 'q-mine');            // BATCH=fig26, not QUEUE=fig2
  assert.equal(p.runSessionFor(P, ['fig2']).session, 'q-other');
  assert.equal(p.runSessionFor(P, []).session, 'q-other');                  // no batch: the project's first
  assert.equal(p.runSessionFor(P, ['fig26']).attached, 0);                  // detached: the button says Attach
  p.runSessions = [{ session: 'fig26', cmds: ['sudo nerdctl logs -f --tail 50 sidechannel'], viewer: true, attached: 1 }];
  assert.equal(p.runSessionFor(P, ['fig26']).session, 'fig26');            // its make ended; it follows the log
  p.runSessions.unshift({ session: 'tf-x', cmds: ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26'], make: true, attached: 0 });
  assert.equal(p.runSessionFor(P, ['fig26']).session, 'tf-x');             // a session running the make comes first
});

test('the panel\'s terminals are created hidden from auto-activation, then shown', () => {
  const created = [];
  vscodeStub.window.terminals = [];
  vscodeStub.window.createTerminal = o => { const t = { name: o.name, opts: o, shown: false, sent: [], show() { this.shown = true; },
                                                    sendText(x) { this.sent.push(x); }, dispose() {} }; created.push(t); return t; };
  const p = panel(P);
  p.terminal('TraceFlow · x', 'tmux attach -t fig26', true);
  assert.equal(created[0].opts.hideFromUser, true);          // vscode-python-envs skips it: no injected Ctrl-C
  assert.deepEqual([created[0].shown, created[0].sent], [true, ['tmux attach -t fig26']]);
});

test('Terminate Run asks once, then kills the run\'s tmux session and stops the stack', async () => {
  const p = panel(P);
  p.pollProcesses = async () => {};
  p.tick = async () => {};
  p.runSessionFor = () => ({ session: 'tf-DLSCA_ASCAD', cmds: ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26'], make: true });
  const lines = [];
  p.terminal = (name, line) => lines.push(line);
  seen.warning.length = 0;
  await p.run({ action: 'terminate-run', project: 'DLSCA_ASCAD' });            // Cancel: nothing runs
  assert.match(seen.warning[0], /^Terminate Run/);
  assert.match(seen.warning[0], /kill tmux session "tf-DLSCA_ASCAD"/);
  assert.match(seen.warning[0], /make stack-down ENV=dev/);
  assert.equal(lines.length, 0);
  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async (m, o, ...buttons) => { assert.deepEqual(buttons, ['Terminate Run']); return 'Terminate Run'; };
  try { await p.run({ action: 'terminate-run', project: 'DLSCA_ASCAD' }); } finally { vscodeStub.window.showWarningMessage = warn; }
  assert.match(lines[0], /tmux kill-session -t tf-DLSCA_ASCAD .*&& make stack-down ENV=dev$/);
});

const REG = {
  h1: { model_name: 'cake_bacon', model_id: 'ASCAD_MLPbest', env: 'dev', status: 'DONE', batches: ['fig26'], started_at: '2026-10-02' },
  h2: { model_name: 'meh_aqua', model_id: 'ASCAD_MLPbest', env: 'dev', status: 'in_progress', batches: ['fig26'], started_at: '2026-10-03' },
  h3: { model_name: 'lofty_owl', model_id: 'ASCAD_CNNbest', env: 'staging', status: 'DONE', batches: [], started_at: '2026-10-01' },
};
const planOf = (env, rows, perEnv = { dev: 2, staging: 1 }) => ({ project: 'DLSCA_ASCAD', env, total_selected: rows.length,
  runs_per_env: perEnv, wipes_project: false, selected: rows.map(([n, id, st, b]) => ({ hash: n, model_name: n, model_id: id,
    dataset: 'ascad', status: st, batches: b })) });

/** DELETE with the ticked hashes; each ENV's dry run is stubbed from `plans`. */
async function deleteWith(hashes, plans, choice = 'OK') {
  const p = panel(P);
  p.pollProcesses = async () => { p.busy = []; p.sessions = {}; };
  p.registry = () => REG;
  const warn = vscodeStub.window.showWarningMessage;
  const info = vscodeStub.window.showInformationMessage;
  const asked = { warn: [], info: [], buttons: [] };
  vscodeStub.window.showWarningMessage = async (m, o, ...b) => { asked.warn.push(text([m, o])); asked.buttons.push(b); return choice || undefined; };
  vscodeStub.window.showInformationMessage = async (m, o) => { asked.info.push(text([m, o])); return undefined; };
  const lines = [];
  p.terminal = (name, line) => lines.push(line);
  const bashed = [];
  p.capture = async line => { bashed.push(line); return `noise\n${JSON.stringify(plans[line.match(/ENV=(\w+)/)[1]])}\n`; };
  try {
    await p.run({ action: 'delete-database', project: 'DLSCA_ASCAD', hashes });
  } finally {
    vscodeStub.window.showWarningMessage = warn; vscodeStub.window.showInformationMessage = info;
  }
  return { asked, lines, bashed };
}

test('Delete database: refused while an experiment runs here', async () => {
  const p = panel(P);
  p.pollProcesses = async () => { p.busy = ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26']; p.sessions = {}; };
  seen.warning.length = 0;
  await p.run({ action: 'delete-database', project: 'DLSCA_ASCAD', hashes: ['h1'] });
  assert.match(seen.warning[0], /Deleting the database under a running experiment is refused/);
});

test('Delete database: nothing ticked deletes nothing', async () => {
  const { asked, lines, bashed } = await deleteWith([], {});
  assert.match(asked.info[0], /nothing selected, nothing deleted/);
  assert.equal(bashed.length, 0);
  assert.equal(lines.length, 0);
});

test('Delete database: DELETE dry-runs the ticked names per ENV; one dialog; OK deletes with CONFIRM=1, Cancel does not', async () => {
  const plans = { dev: planOf('dev', [['cake_bacon', 'ASCAD_MLPbest', 'DONE', ['fig26']]]) };
  let r = await deleteWith(['h1'], plans);
  assert.deepEqual(r.bashed.length, 1);
  assert.match(r.bashed[0], /make delete_database ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon JSON=1/);
  const m = r.asked.warn[0];
  assert.match(m, /Delete 1 experiment\(s\) of ENV=dev in DLSCA_ASCAD\?/);
  assert.match(m, /Total: 1 experiment\(s\)/);
  assert.match(m, /Runs per ENV: dev: 2 \(1 selected\) · staging: 1/);
  assert.match(m, /• cake_bacon · dev · ASCAD_MLPbest · ascad · DONE · fig26/);
  assert.match(m, /You sure to delete it\?$/);
  assert.deepEqual(r.asked.buttons[0], ['OK']);
  assert.match(r.lines[0], /make delete_database ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon CONFIRM=1$/);
  r = await deleteWith(['h1'], plans, null);                                           // Cancel
  assert.equal(r.lines.length, 0);
});

test('Delete database: Delete all ticks every run; one dry run per ENV, both confirmed in one line, no run left', async () => {
  const plans = {
    dev: planOf('dev', [['meh_aqua', 'ASCAD_MLPbest', 'in_progress', ['fig26']], ['cake_bacon', 'ASCAD_MLPbest', 'DONE', ['fig26']]]),
    staging: planOf('staging', [['lofty_owl', 'ASCAD_CNNbest', 'DONE', []]]),
  };
  const r = await deleteWith(['h1', 'h2', 'h3'], plans);
  assert.equal(r.bashed.length, 2);
  assert.match(r.bashed.join('\n'), /ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon,meh_aqua JSON=1/);
  assert.match(r.bashed.join('\n'), /ENV=staging PROJECT=DLSCA_ASCAD MODEL=lofty_owl JSON=1/);
  const m = r.asked.warn[0];
  assert.match(m, /Delete 3 experiment\(s\) of ENV=dev, staging in DLSCA_ASCAD\?/);
  assert.match(m, /Runs per ENV: dev: 2 \(2 selected\) · staging: 1 \(1 selected\)/);
  assert.match(m, /No run is left afterwards/);
  assert.match(r.lines[0], /MODEL=cake_bacon,meh_aqua CONFIRM=1 && make delete_database ENV=staging PROJECT=DLSCA_ASCAD MODEL=lofty_owl CONFIRM=1$/);
});

const snapRow = (name, status, extra = {}) => ({ env: 'dev', batches: ['fig26'], model_name: name, model_hash: `h_${name}`,
  model_id: 'ASCAD_MLPbest', dataset: 'ascad', status, registered: !!status, verified: true, problem: '', project: 'DLSCA_ASCAD',
  path: `/x/${name}_params.yaml`, ...extra });

test('SNAPSHOTS: Open opens a scanned file only; a refused snapshot is not reproduced', async () => {
  const p = panel(P);
  const opened = [];
  p.open = f => opened.push(f);
  p.startLong = async () => { throw new Error('must not start'); };
  p.snapshots = { 'p:DLSCA_ASCAD': { rows: [snapRow('a', ''), snapRow('b', '', { verified: false, problem: 'hashes to x here' })] } };
  await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'open', path: '/x/a_params.yaml' });
  assert.deepEqual(opened, ['/x/a_params.yaml']);
  seen.warning.length = 0;
  await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'open', path: '/etc/passwd' });                // not from the scan
  assert.match(seen.warning[0], /not in the last scan/);
  await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'reproduce', path: '/x/b_params.yaml' });
  assert.match(seen.warning[1], /b is refused: hashes to x here/);
});

test('SNAPSHOTS: Reproduce one snapshot: its ENV, project, name and batch; a DONE one asks and gets FORCE=1', async () => {
  const p = panel(P);
  const started = [];
  p.startLong = async (proj, target, vars, commands, note, opts) => { started.push({ proj: proj.name, target, vars, note, opts }); return true; };
  p.tick = () => {};
  p.snapshots = { 'p:DLSCA_ASCAD': { origin: 'DLSCA_ASCAD', rows: [snapRow('a', ''), snapRow('c', 'DONE', { env: 'staging' })] } };
  await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'reproduce', path: '/x/a_params.yaml' });
  assert.equal(started[0].target, 'reproduce');
  assert.deepEqual(started[0].vars, { ENV: 'dev', PROJECT: 'DLSCA_ASCAD', MODEL: 'a', BATCH: 'fig26', FORCE: '', REPRODUCE_ROOT: '' });
  assert.match(started[0].note, /^Reproducing a \(h_a\)\nfrom DLSCA_ASCAD · ENV dev \(from the snapshot\) · into project DLSCA_ASCAD · not registered$/);
  assert.equal(started[0].opts.before, null);
  assert.deepEqual(p.reproducing['p:DLSCA_ASCAD'].name, 'a');

  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async () => undefined;                                    // DONE: Cancel
  try { await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'reproduce', path: '/x/c_params.yaml' }); } finally { vscodeStub.window.showWarningMessage = warn; }
  assert.equal(started.length, 1);
  vscodeStub.window.showWarningMessage = async (m, o, ...b) => { assert.match(m, /is DONE in DLSCA_ASCAD \(ENV=staging\)/); return b[0]; };
  try { await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'reproduce', path: '/x/c_params.yaml' }); } finally { vscodeStub.window.showWarningMessage = warn; }
  assert.deepEqual(started[1].vars, { ENV: 'staging', PROJECT: 'DLSCA_ASCAD', MODEL: 'c', BATCH: 'fig26', FORCE: 1, REPRODUCE_ROOT: '' });
});

test('SNAPSHOTS: a CUSTOM snapshot is reproduced with the chosen folder as its root, placed after the confirmation', async () => {
  const p = panel(P);
  const started = [];
  p.startLong = async (proj, target, vars, commands, note, opts) => { started.push({ proj: proj.name, vars, note, opts }); return true; };
  p.tick = () => {};
  p.customFolder = () => '/abs/1/2/3/pikachu';
  p.snapshots = { custom: { origin: 'pikachu', rows: [snapRow('a', '', { target: '/abs/1/2/3/pikachu/artifacts/dev/generated_models/a/config/a_params.yaml', placed: false }),
                                                       snapRow('z', '', { project: 'NOPE' })] } };
  await p.snap({ scope: 'custom', action: 'reproduce', path: '/x/a_params.yaml' });
  assert.equal(started[0].proj, 'DLSCA_ASCAD');
  assert.deepEqual(started[0].vars, { ENV: 'dev', PROJECT: 'DLSCA_ASCAD', MODEL: 'a', BATCH: 'fig26', FORCE: '', REPRODUCE_ROOT: '/abs/1/2/3/pikachu' });
  assert.match(started[0].note, /from pikachu · ENV dev \(from the snapshot\) · outputs under \/abs\/1\/2\/3\/pikachu \(every paths: entry; configs and scripts of project DLSCA_ASCAD\)/);
  assert.match(started[0].note, /first copied to \/abs\/1\/2\/3\/pikachu\/artifacts\/dev\/generated_models\/a\/config\/a_params\.yaml/);
  assert.equal(typeof started[0].opts.before, 'function');                                         // placed by layout.py place --into, after OK
  assert.equal(started[0].opts.log, '/abs/1/2/3/pikachu/artifacts/dev/tmux_logs/fig26.txt');
  seen.warning.length = 0;
  await p.snap({ scope: 'custom', action: 'reproduce', path: '/x/z_params.yaml' });
  assert.match(seen.warning[0], /project NOPE is not in master.yaml/);
});

test('SNAPSHOTS: Reproduce asks the destination: the project itself, or a folder the snapshot is copied to first', async () => {
  const p = panel(P);
  const started = [];
  p.startLong = async (proj, target, vars, commands, note, opts) => { started.push({ vars, note, opts }); return true; };
  p.tick = () => {};
  const store = {};
  p.context = { workspaceState: { get: k => store[k], update: async (k, v) => { store[k] = v; } } };
  p.snapshots = { 'p:DLSCA_ASCAD': { rows: [snapRow('a', 'done')] } };
  const ask = vscodeStub.window.showInformationMessage;
  const asked = [];
  try {
    vscodeStub.window.showInformationMessage = async (m, o, ...b) => { asked.push([m, o.detail, b]); return b[0]; };   // the project itself
    seen.warning.length = 0;
    await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'reproduce', path: '/x/a_params.yaml' });         // DONE there: asks, Cancel
    assert.equal(asked[0][0], 'Where should the reproduction write its outputs?');
    assert.deepEqual(asked[0][2], ['In DLSCA_ASCAD', 'Choose a folder…']);
    assert.match(asked[0][1], /In DLSCA_ASCAD: [\s\S]*a DONE run's results there are replaced\.\n\nChoose a folder…: a folder outside the repository/);
    assert.match(seen.warning[0], /is DONE in DLSCA_ASCAD/);
    assert.equal(started.length, 0);
    const dest = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-dest-'));
    vscodeStub.window.showInformationMessage = async (m, o, ...b) => b[b.length - 1];               // Choose a folder…
    vscodeStub.window.showOpenDialog = async () => [{ fsPath: dest }];
    await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'reproduce', path: '/x/a_params.yaml' });         // not registered there: no DONE question
    assert.equal(started[0].vars.REPRODUCE_ROOT, dest);
    assert.equal(started[0].vars.FORCE, '');
    assert.match(started[0].note, new RegExp(`outputs under ${dest}[\\s\\S]*first copied to its params_out path under ${dest}`));
    assert.equal(typeof started[0].opts.before, 'function');
    assert.equal(started[0].opts.log, path.join(dest, 'artifacts', 'dev', 'tmux_logs', 'fig26.txt'));
    assert.equal(store['traceflow.reproduceDest'], dest);
    vscodeStub.window.showOpenDialog = async () => [{ fsPath: ROOT }];                               // the repository: refused
    await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'reproduce', path: '/x/a_params.yaml' });
    assert.match(seen.warning[seen.warning.length - 1], /holds the repository/);
    assert.equal(started.length, 1);
  } finally {
    vscodeStub.window.showInformationMessage = ask;
    delete vscodeStub.window.showOpenDialog;
  }
});

test('Figures: Delete removes a drawn job folder after one question; a folder outside graphs/jobs is refused', async () => {
  const jobs = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-jobs-'));
  const dir = path.join(jobs, 'job_048_fig20');
  fs.mkdirSync(path.join(dir, 'sub'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'sub', 'a.pdf'), '');
  const p = panel({ ...P, graph_jobs_dir: jobs });
  p.pollFigures = () => {};
  seen.warning.length = 0;
  await p.deleteFigure('DLSCA_ASCAD', dir);                                       // Cancel
  assert.match(seen.warning[0], /Delete the drawn figure job_048_fig20\?[\s\S]*2 file\(s\)[\s\S]*Draw makes it again/);
  assert.ok(fs.existsSync(dir));
  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async (...a) => { seen.warning.push(text(a)); return 'Delete'; };
  try {
    await p.deleteFigure('DLSCA_ASCAD', jobs);                                    // the jobs folder itself: refused
    assert.match(seen.warning[seen.warning.length - 1], /not a graph job folder/);
    await p.deleteFigure('DLSCA_ASCAD', dir);
  } finally { vscodeStub.window.showWarningMessage = warn; }
  assert.ok(!fs.existsSync(dir));
  assert.ok(fs.existsSync(jobs));
});

test('Done tile → Test: refused when the snapshot says perform_attack.test.attack_times: 0, else make test MODEL=<name>', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-test-'));
  const artifacts = path.join(tmp, 'artifacts');
  const snap = (name, times) => {
    const dir = path.join(artifacts, 'dev', 'generated_models', name, 'config');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, `${name}_params.yaml`), `perform_attack:\n  test:\n    attack_times: ${times}\n`);
  };
  snap('zero', 0);
  snap('three', 3);
  const p = panel({ ...P, artifacts, registry: path.join(artifacts, 'run_registry.json') });
  fs.writeFileSync(p.layout.projects[0].registry, JSON.stringify({
    h0: { model_name: 'zero', status: 'DONE', env: 'dev' }, h3: { model_name: 'three', status: 'DONE', env: 'dev' },
    hf: { model_name: 'broke', status: 'FAILED', env: 'dev' } }));
  const started = [];
  p.startLong = async (proj, target, vars, commands, note) => { started.push({ target, vars, note }); return true; };
  seen.warning.length = 0;
  await p.run({ action: 'test-run', project: 'DLSCA_ASCAD', key: 'h0' });
  assert.match(seen.warning[0], /zero is not tested[\s\S]*Refused: its snapshot says perform_attack: test: attack_times: 0/);
  assert.equal(started.length, 0);
  await p.run({ action: 'test-run', project: 'DLSCA_ASCAD', key: 'hf' });
  assert.match(seen.warning[1], /broke is FAILED: only a DONE run is tested/);
  await p.run({ action: 'test-run', project: 'DLSCA_ASCAD', key: 'h3' });
  assert.deepEqual(started[0].vars, { ENV: 'dev', PROJECT: 'DLSCA_ASCAD', MODEL: 'three' });
  assert.equal(started[0].target, 'test');
});

test('Failed tile → Delete: that run goes to the dry run whatever the Housekeeping filters show', async () => {
  const p = panel(P);
  const store = { 'traceflow.deleteFilters': { DLSCA_ASCAD: { STATUS: 'done' } } };      // would hide a failed run
  p.context = { workspaceState: { get: k => store[k], update: async (k, v) => { store[k] = v; } } };
  p.pollProcesses = async () => { p.busy = []; p.sessions = {}; };
  p.registry = () => ({ hf: { model_name: 'broke', status: 'FAILED', env: 'dev' }, hd: { model_name: 'fine', status: 'DONE', env: 'dev' } });
  const bashed = [];
  p.capture = async line => { bashed.push(line); return JSON.stringify(planOf('dev', [['broke', 'ASCAD_MLPbest', 'FAILED', []]])); };
  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async () => undefined;                     // Cancel after the dry run
  try { await p.run({ action: 'delete-run', project: 'DLSCA_ASCAD', key: 'hf' }); }
  finally { vscodeStub.window.showWarningMessage = warn; }
  assert.equal(bashed.length, 1);
  assert.match(bashed[0], /ENV=dev PROJECT=DLSCA_ASCAD MODEL=broke JSON=1/);
});

test('dialogs: a modal one shows its first line as the heading and the rest as the wrapped detail', async () => {
  const calls = [];
  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async (...a) => { calls.push(a); return 'OK'; };
  try {
    assert.equal(await E.dialog('showWarningMessage')('Heading?\nline one\nline two', { modal: true }, 'OK'), 'OK');
    assert.deepEqual(calls[0], ['Heading?', { modal: true, detail: 'line one\nline two' }, 'OK']);
    const long = 'TraceFlow: not torn down. Let the current run finish: an experiment is running on this machine right now.';
    await E.dialog('showWarningMessage')(long, { modal: true });
    assert.deepEqual(calls[1], ['TraceFlow: not torn down.', { modal: true, detail: 'Let the current run finish: an experiment is running on this machine\nright now.' }]);
    // every line of a modal dialog fits 72 characters; a long path breaks after a '/'
    const path_ = '/home/u/TraceFlow/projects/DLSCA_ASCAD/artifacts/dev/generated_models/eyelash_cheddar/config/eyelash_cheddar_params.yaml';
    await E.dialog('showWarningMessage')(`TraceFlow: eyelash_cheddar is not tested.\n\nRefused: its snapshot says perform_attack: test: attack_times: 0, so its test traces are never attacked.\n\n${path_}`, { modal: true });
    const shown = calls[2][1].detail.split('\n');
    assert.ok(shown.every(l => l.length <= 72), shown.join('|'));
    assert.equal(shown.join('').replace(/ /g, ''), `Refused:itssnapshotsaysperform_attack:test:attack_times:0,soitstesttracesareneverattacked.${path_}`);
    calls.length = 2;
    await E.dialog('showWarningMessage')('a notification\nkept whole');                     // not modal: as it is
    assert.deepEqual(calls[2], ['a notification\nkept whole']);
  } finally { vscodeStub.window.showWarningMessage = warn; }
});

test('TESTS → Test All / Test Budget: test_config.yaml\'s attack, make test MODEL=<run> EXTERNAL=1 [BUDGET=1], appended', async () => {
  const p = panel({ ...P, path: path.join(ROOT, 'projects', 'DLSCA_ASCAD'), registry: path.join(os.tmpdir(), 'none', 'run_registry.json') });
  p.registry = () => ({ h1: { model_name: 'eyelash_cheddar', status: 'DONE', env: 'dev', path: 'DLSCA_ASCAD/artifacts/dev/generated_models/eyelash_cheddar' },
                        h2: { model_name: 'broke', status: 'FAILED', env: 'dev' } });
  const started = [];
  p.startLong = async (proj, target, vars, commands, note, opts) => { started.push({ target, vars, note, opts }); return true; };
  seen.warning.length = 0;
  await p.run({ action: 'external-test', project: 'DLSCA_ASCAD', key: 'h2' });
  assert.match(seen.warning[0], /broke is FAILED: only a DONE run is tested/);
  await p.run({ action: 'external-test', project: 'DLSCA_ASCAD', key: 'h1' });
  assert.deepEqual(started[0].vars, { ENV: 'dev', PROJECT: 'DLSCA_ASCAD', MODEL: 'eyelash_cheddar', EXTERNAL: 1 });
  assert.match(started[0].note, /^Test All of eyelash_cheddar[^]*perform_attack\.test: attack_times 100, shuffle true, full_population; every test trace; attack_step 1; ge_thresholds 1/);
  assert.match(started[0].note, /appended[^]*The recorded test is not touched/);
  await p.run({ action: 'external-test', project: 'DLSCA_ASCAD', key: 'h1', kind: 'budget' });
  assert.deepEqual(started[1].vars, { ENV: 'dev', PROJECT: 'DLSCA_ASCAD', MODEL: 'eyelash_cheddar', EXTERNAL: 1, BUDGET: 1 });
  assert.match(started[1].note, /^Test Budget of eyelash_cheddar[^]*each attack its data entry's attack_trace_budget traces/);
  assert.equal(started[0].opts.log, path.join(ROOT, 'projects', 'DLSCA_ASCAD', 'artifacts', 'dev', 'generated_models', 'eyelash_cheddar', 'test', 'test_log.txt'));
});

test('TESTS lists every DONE run with its external tests in index order; Experiment Settings → test_config.yaml → Test is external', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-tests-'));
  const p = panel({ ...P, registry: path.join(tmp, 'run_registry.json') });
  fs.writeFileSync(p.layout.projects[0].registry, JSON.stringify({
    h1: { model_name: 'a', status: 'DONE', env: 'dev', model_id: 'M', started_at: '2026-10-01' },
    h2: { model_name: 'b', status: 'FAILED', env: 'dev' } }));
  fs.writeFileSync(path.join(tmp, 'test_registry.json'), JSON.stringify({ h1: { status: 'SKIPPED', external_tests: {
    ffff0000: { index: 2, status: 'FAILED', settings: { perform_attack: { test: { attack_times: 10 } } } },
    aaaa1111: { index: 1, status: 'DONE', recovered_key_rank: 1, runs: 2, settings: { perform_attack: { test: { attack_times: 100, shuffle: true } } } } } } }));
  const t = p.testsState(p.layout.projects[0], p.registry(p.layout.projects[0].registry));
  assert.deepEqual(t.map(m => [m.name, m.recorded, m.tests.map(x => [x.id, x.index, x.status, x.rank])]),
    [['a', 'SKIPPED', [['aaaa1111', 1, 'DONE', 1], ['ffff0000', 2, 'FAILED', undefined]]]]);
  const started = [];
  p.startLong = async (proj, target, vars) => { started.push({ target, vars }); return true; };
  p.envFor = async () => 'dev';
  await p.run({ action: 'test', project: 'DLSCA_ASCAD' });
  assert.deepEqual(started[0], { target: 'test', vars: { ENV: 'dev', PROJECT: 'DLSCA_ASCAD', EXTERNAL: 1 } });
});

test('Experiment Settings → import_config.yaml → Import runs make import for the project (every import not DONE)', async () => {
  const p = panel(P);
  const started = [];
  p.startLong = async (proj, target, vars, commands, note) => { started.push({ target, vars, commands, note }); return true; };
  p.envFor = async () => 'dev';
  await p.run({ action: 'import', project: 'DLSCA_ASCAD' });
  assert.deepEqual(started.map(s => [s.target, s.vars, s.commands]), [['import', { ENV: 'dev', PROJECT: 'DLSCA_ASCAD' }, null]]);
  assert.match(started[0].note, /not DONE yet \(new, failed or interrupted\)[\s\S]*DONE ones are skipped/);
});

/** A [n, 2] float64 .npy curve: [trace count, mean rank] rows. */
function npyCurve(file, rows) {
  const header = "{'descr': '<f8', 'fortran_order': False, 'shape': (" + rows.length + ', 2), }';
  const pad = 64 - ((10 + header.length + 1) % 64);
  const head = Buffer.concat([Buffer.from([0x93]), Buffer.from('NUMPY'), Buffer.from([1, 0]), Buffer.alloc(2),
                              Buffer.from(header + ' '.repeat(pad) + '\n')]);
  head.writeUInt16LE(head.length - 10, 8);
  const body = Buffer.alloc(rows.length * 16);
  rows.forEach(([x, y], i) => { body.writeDoubleLE(x, i * 16); body.writeDoubleLE(y, i * 16 + 8); });
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, Buffer.concat([head, body]));
}

test('[Done] its four ranks: the validation\'s, then the latest test of each kind, never an earlier one', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-ranks-'));
  const p = panel(P);
  const run = { hash: 'h1', name: 'm' };
  p.runDir = () => tmp;
  const curve = (dir, rows) => npyCurve(path.join(dir, 'arrays', 'test', 'mean_attack_rank_curve', 'ep0001.npy'), rows);
  npyCurve(path.join(tmp, 'metrics', 'attack_metrics', 'arrays', 'validation', 'mean_attack_rank_curve', 'ep0050.npy'), [[1, 128], [1000, 3]]);
  curve(path.join(tmp, 'metrics', 'attack_metrics'), [[1, 128], [1000, 5], [9000, 2]]);          // the training's: kind all
  const ext = id => path.join(tmp, 'test', id, 'metrics', 'attack_metrics');
  curve(ext('h1_et_1'), [[1, 128], [250, 1.2]]);                                                  // budget, older
  curve(ext('h1_et_2'), [[1, 128], [250, 40.5]]);                                                 // budget, latest (worse)
  curve(ext('h1_et_3'), [[1, 128], [1000, 7], [9000, 4]]);                                        // all, latest
  curve(ext('h1_et_4'), [[1, 128], [250, 1]]);                                                    // budget, FAILED: never read
  const rec = (index, kind, at, status = 'DONE') => ({ index, kind, status, finished_at: at, source: 'test', settings: {} });
  p.testRegistryFile = () => 'treg';
  p.registry = f => (f === 'treg' ? { h1: { finished_at: '2026-10-01T00:00:00Z', external_tests: {
    h1_et_1: rec(1, 'budget', '2026-10-02T00:00:00Z'), h1_et_2: rec(2, 'budget', '2026-10-04T00:00:00Z'),
    h1_et_3: rec(3, 'all', '2026-10-03T00:00:00Z'), h1_et_4: rec(4, 'budget', '2026-10-05T00:00:00Z', 'FAILED') } } } : {});
  const r = p.runRanks(P, run);
  assert.deepEqual(r.val, { traces: 1000, rank: 3 });
  assert.deepEqual(r.budget, { traces: 250, rank: 40.5 });       // the latest budget test, though et_1 was better
  assert.deepEqual(r.test, { traces: 9000, rank: 4 });           // the latest all test (et_3 after the training's)
  assert.deepEqual(r.same, { traces: 1000, rank: 7 });           // the latest test reaching N_val = 1000: et_3
  assert.deepEqual(p.testRanks(P, run, 'h1_et_2'), { val: { traces: 1000, rank: 3 }, same: null, budget: { traces: 250, rank: 40.5 }, test: null });
  assert.equal(p.safeRanks(P, { hash: 'h9' }).val.rank, 3);       // a run without tests: its validation only
});

test('SNAPSHOTS: Scan refuses an empty or wrong filter before scanning', async () => {
  const p = panel({ ...P, model_ids: ['ASCAD_MLPbest'], datasets: ['ascad'] });
  p.tick = () => {};
  seen.warning.length = 0;
  await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'scan', filters: {} });
  assert.match(p.snapshots['p:DLSCA_ASCAD'].error, /fill at least one filter/);
  await p.snap({ scope: 'p:DLSCA_ASCAD', action: 'scan', filters: { DATASET: 'ascad_desync9' } });
  assert.match(p.snapshots['p:DLSCA_ASCAD'].error, /wrong entry: DATASET=ascad_desync9 is not a dataset of params.yaml: ascad/);
  assert.equal(p.snapshots['p:DLSCA_ASCAD'].scanning, false);
  assert.equal(seen.warning.length, 2);
});

test('Docker tear down: refused while an experiment runs; asks ENV and once; OK runs make docker-teardown, Cancel nothing', async () => {
  const p = panel(P);
  const lines = [];
  p.terminal = (name, line) => lines.push(line);
  p.pollProcesses = async () => { p.busy = ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26']; p.sessions = {}; };
  seen.warning.length = 0;
  await p.dockerTeardown();                                                     // busy: the force offer, declined
  assert.match(seen.warning[0], /not torn down: an experiment is running on this machine[\s\S]*QUEUE=fig26[\s\S]*force the tear down/);
  assert.equal(lines.length, 0);
  p.pollProcesses = async () => { p.busy = []; p.sessions = {}; };
  p.docker.building = true;
  await p.dockerTeardown();                                                     // building: the force offer, declined
  assert.match(seen.warning[1], /not torn down: an image is building/);
  const warnForce = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async (...a) => { seen.warning.push(text(a)); return 'Force tear down'; };
  try { await p.dockerTeardown(); } finally { vscodeStub.window.showWarningMessage = warnForce; }
  assert.match(lines.pop(), /&& make docker-teardown ENV=dev FORCE=1$/);        // forced: kills first, then tears down
  assert.equal(p.docker.building, false);
  p.docker.building = false;
  await p.dockerTeardown();                                                     // Cancel
  assert.match(seen.warning[seen.warning.length - 1], /Tear down Docker \(ENV=dev\)\?[\s\S]*whole containerd namespace/);
  assert.equal(lines.length, 0);
  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async (...a) => { seen.warning.push(text(a)); return 'Tear down'; };
  try { await p.dockerTeardown(); } finally { vscodeStub.window.showWarningMessage = warn; }
  assert.match(lines[0], /&& make docker-teardown ENV=dev$/);
});

test('Kill build: nothing to kill without a build; asks once; OK runs the pkill line in a terminal', async () => {
  const p = panel(P);
  const lines = [];
  p.terminal = (name, line) => lines.push(line);
  p.builds = async () => [];
  const info = vscodeStub.window.showInformationMessage;
  const said = [];
  vscodeStub.window.showInformationMessage = async m => { said.push(m); };
  try { await p.dockerKillBuild(); } finally { vscodeStub.window.showInformationMessage = info; }
  assert.deepEqual(said, ['TraceFlow: no image build runs.']);
  p.builds = async () => ['4242 nerdctl --namespace default build -f deployment/docker/Dockerfile .'];
  seen.warning.length = 0;
  await p.dockerKillBuild();                                                    // Cancel
  assert.match(seen.warning[0], /Kill the image build\?[\s\S]*4242 nerdctl/);
  assert.equal(lines.length, 0);
  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async () => 'Kill build';
  try { await p.dockerKillBuild(); } finally { vscodeStub.window.showWarningMessage = warn; }
  assert.match(lines[0], /&& make kill-build ENV=dev$/);                         // the Makefile kills client, steps and daemon
  assert.equal(p.docker.building, false);
});

test('Delete database: only the ticked runs the filters show are deleted', async () => {
  const p = panel(P);
  const store = { 'traceflow.deleteFilters': { DLSCA_ASCAD: { ENV: 'dev' } } };
  p.context = { workspaceState: { get: k => store[k], update: async (k, v) => { store[k] = v; } } };
  p.pollProcesses = async () => { p.busy = []; p.sessions = {}; };
  p.registry = () => REG;
  const bashed = [];
  p.capture = async line => { bashed.push(line); return JSON.stringify(planOf('dev', [['cake_bacon', 'ASCAD_MLPbest', 'DONE', ['fig26']]])); };
  const warn = vscodeStub.window.showWarningMessage;
  vscodeStub.window.showWarningMessage = async () => undefined;                     // Cancel
  try { await p.run({ action: 'delete-database', project: 'DLSCA_ASCAD', hashes: ['h1', 'h3'] }); }   // h3 is staging: hidden
  finally { vscodeStub.window.showWarningMessage = warn; }
  assert.equal(bashed.length, 1);
  assert.match(bashed[0], /ENV=dev PROJECT=DLSCA_ASCAD MODEL=cake_bacon JSON=1/);
  p.tick = () => {};
  await p.setDelFilters('DLSCA_ASCAD', { ENV: ' ', NAME: 'owl' });
  assert.deepEqual(store['traceflow.deleteFilters'].DLSCA_ASCAD, { NAME: 'owl' });
});

test('Export to Graph Config: nothing picked, or picks of two metrics, is refused before graph_config.yaml is touched', async () => {
  const p = panel(P);
  p.followed = () => ({ env: 'dev' });
  seen.info.length = 0;
  seen.warning.length = 0;
  await p.run({ action: 'graph-export', project: 'DLSCA_ASCAD', picks: [] });
  assert.match(seen.info[0], /pick a metric beside each run/);
  await p.run({ action: 'graph-export', project: 'DLSCA_ASCAD', picks: [
    { name: 'a', key: 'cross_validation|validation|mean_attack_rank_curve' },
    { name: 'b', key: 'cross_validation|validation|mean_accuracy' }] });
  assert.match(seen.warning[0], /one figure draws one metric; the picked runs name 2/);
});

test('A graph job removed in the editor: the panel asks whether the later jobs move down, Leave the gap does nothing', async () => {
  const p = panel(P);
  const before = { DLSCA_ASCAD: [0, 1, 2, 3] };
  p.layout = { projects: [{ ...P, name: 'DLSCA_ASCAD', graph_jobs: [{ id: 0 }, { id: 1 }, { id: 3 }] }] };
  seen.info.length = 0;
  await p.offerGapClose(before);                                  // the stub cancels: the gap stays
  assert.match(seen.info[0], /graph job 2 removed from DLSCA_ASCAD's graph_config\.yaml\..*Shift ids moves every later job down/);
  seen.info.length = 0;
  await p.offerGapClose({ DLSCA_ASCAD: [0, 1, 3] });              // nothing removed: no question
  p.graphEditUntil = Date.now() + 1000;
  await p.offerGapClose(before);                                  // the panel's own edit: no question
  assert.equal(seen.info.length, 0);
});

test('Delete from Graph Config: a run in no such job is told so; Cancel removes nothing', async () => {
  const p = panel({ ...P, graph_jobs: [{ id: 0, models: ['a'] }, { id: 4, metric: 'accuracy', source: 'cross_validation',
    processes: ['validation'], models: ['a'] }] });
  seen.info.length = 0;
  seen.warning.length = 0;
  await p.run({ action: 'graph-remove', project: 'DLSCA_ASCAD', name: 'b', key: 'cross_validation|validation|accuracy' });
  assert.match(seen.info[0], /b is in no graph job drawing accuracy/);
  await p.run({ action: 'graph-remove', project: 'DLSCA_ASCAD', name: 'a', key: 'cross_validation|validation|accuracy' });
  assert.match(seen.warning[0], /remove a from graph job 4\?.*Graph job 4 then lists no run: it is deleted/);
});

test('Overlay Graph: the relative fit time asks for the reference model and sends it with the metrics', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const calls = path.join(dir, 'calls.jsonl');
  const plan = { ok: true, study: 'fig10_optimizers', queues: ['fig10_optimizers'], runs: ['a', 'b', 'c'],
    axes: [{ key: 'dataset', values: ['ascad', 'ascad_desync100'] }, { key: 'trainer.optimizer.type', values: ['Adam', 'RMSprop'] }],
    legend: ['trainer.optimizer.type'],
    values: { a: { dataset: 'ascad', 'trainer.optimizer.type': 'Adam' }, b: { dataset: 'ascad', 'trainer.optimizer.type': 'RMSprop' },
      c: { dataset: 'ascad_desync100', 'trainer.optimizer.type': 'RMSprop' } },
    metrics: [{ source: 'cross_validation', process: 'train', metric: 'mean_fit_time_s', kind: 'per_epoch', logged: true, relative: true }] };
  const python = path.join(dir, 'python');                    // a stand-in: answers study-plan, records study
  fs.writeFileSync(python, `#!/bin/sh\nprintf '%s\\n' "$*" >> ${JSON.stringify(calls)}\n` +
    `case "$*" in *study-plan*) echo '${JSON.stringify(plan)}';; *) echo '{"ok":true,"study":"fig10_optimizers","figures":2,"created":[{"id":9},{"id":10}],"kept":[]}';; esac\n`);
  fs.chmodSync(python, 0o755);
  const titles = [];
  const before = vscodeStub.window.showQuickPick;
  vscodeStub.window.showQuickPick = async (items, opts) => {
    titles.push(opts.title);
    if (/tick the axes/.test(opts.title)) return items.filter(i => i.picked);                    // the default legend
    if (/each ticked metric/.test(opts.title)) return items.filter(i => i.rel);                   // relative only
    return items.find(i => i.label === 'RMSprop');                                               // the reference
  };
  try {
    const p = panel({ ...P, configs: [] });
    p.python = python;
    seen.info.length = 0;
    await p.study(p.layout.projects[0], 'fig10_optimizers', 'dev');
    const sent = fs.readFileSync(calls, 'utf8').trim().split('\n');
    assert.equal(sent.length, 2);
    assert.match(sent[1], /study --project DLSCA_ASCAD --env dev --queue fig10_optimizers --legend trainer.optimizer.type --metrics {2}--relative cross_validation\|train\|mean_fit_time_s --reference \{"trainer.optimizer.type":"RMSprop"\}$/);
    assert.match(titles[2], /the reference model of the relative metric/);
    assert.match(seen.info[0], /created graph job\(s\) 9, 10/);
  } finally { vscodeStub.window.showQuickPick = before; }
});

test('Add a Queue: the key typed in the panel (not one there, no spaces), written with one line to fill at the file\'s end', async () => {
  const p = panel({ ...P, model_ids: ['ASCAD_MLPbest'], datasets: ['ascad'] });
  p.configFile = f => f;
  p.followed = () => ({ env: 'dev' });
  const applied = [];
  const notes = [];
  p.apply = async m => applied.push(m);
  p.sendTree = async (f, note) => notes.push(note);
  try {
    const msg = { project: 'DLSCA_ASCAD', file: '/q/experiment_queue.yaml', sha: 's', keys: ['fig26'] };
    await p.addQueue({ ...msg, name: 'fig26' });
    await p.addQueue({ ...msg, name: 'fig 27' });
    assert.deepEqual(notes, ['refused: no queue added: fig26 is a queue already', 'refused: no queue added: letters, digits, _ . - (no spaces)']);
    await p.addQueue({ ...msg, name: ' fig27 ' });
    assert.deepEqual(applied, [{ file: '/q/experiment_queue.yaml', sha: 's', edits: [
      { op: 'put', path: [], key: 'fig27', value: '' },
      { op: 'add', path: ['fig27'], value: 'make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=ASCAD_MLPbest' }] }]);
  } finally { p.stop(); }
});

test('Add a Job / Add a Facet: the name typed in the panel (its form, not taken), then graph_export.py new-job / new-facet', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const calls = path.join(dir, 'calls.txt');
  const python = path.join(dir, 'python');
  fs.writeFileSync(python, `#!/bin/sh\nprintf '%s\\n' "$*" >> ${JSON.stringify(calls)}\necho '{"ok":true,"created":9,"name":"x"}'\n`);
  fs.chmodSync(python, 0o755);
  const p = panel({ ...P, graph_jobs: [{ id: 0, name: 'fig2_rank' }], graph_facets: [{ id: 1, name: 'fig26_id_and_hw' }] });
  p.python = python;
  p.scheduleRelayout = () => {};
  seen.info.length = 0;
  seen.warning.length = 0;
  try {
    await p.addJob({ project: 'DLSCA_ASCAD', name: 'my figure' });
    await p.addJob({ project: 'DLSCA_ASCAD', name: 'fig2_rank' });
    await p.addFacet({ project: 'DLSCA_ASCAD', name: 'fig26_id_and_hw' });
    assert.deepEqual(seen.warning, ['TraceFlow: no job added: letters, digits, _ . - and {{metric}}-style placeholders, no spaces',
      'TraceFlow: no job added: fig2_rank is a job already', 'TraceFlow: no facet added: fig26_id_and_hw is a facet already']);
    assert.equal(fs.existsSync(calls), false);
    await p.addJob({ project: 'DLSCA_ASCAD', name: '{{metric}}_fig27' });
    await p.addFacet({ project: 'DLSCA_ASCAD', name: 'fig27_two' });
    assert.deepEqual(fs.readFileSync(calls, 'utf8').trim().split('\n'), [
      '-m services.graph_manager.graph_export new-job --project DLSCA_ASCAD --title {{metric}}_fig27',
      '-m services.graph_manager.graph_export new-facet --project DLSCA_ASCAD --title fig27_two']);
    assert.match(seen.info[0], /graph job 9 \(x\) added to graph_config.yaml: list its models, then set enabled: true/);
  } finally { p.stop(); }
});

test('Add a Model: the model_id and kind chosen in the panel (not one there), a layer model or a named one', async () => {
  const p = panel(P);
  p.configFile = f => f;
  const applied = [];
  const notes = [];
  p.apply = async m => applied.push(m);
  p.sendTree = async (f, note) => notes.push(note);
  p.modelIds = async () => ['ASCAD_CNNbase'];                 // read from the file at the moment of adding
  try {
    const msg = { project: 'DLSCA_ASCAD', file: '/p/params.yaml', sha: 's', ids: [], kinds: ['CNN', 'MLP'], named: ['ResNet50_1D'] };
    await p.addModel({ ...msg, name: 'ASCAD_CNNbase', kind: 'CNN' });
    await p.addModel({ ...msg, name: 'my cnn', kind: 'CNN' });
    await p.addModel({ ...msg, name: 'MY_CNN', kind: 'LSTM' });
    assert.deepEqual(notes, ['refused: no model added: ASCAD_CNNbase is a model already',
      'refused: no model added: letters, digits, _ . - (no spaces)', 'refused: no model added: LSTM is not a model kind here']);
    await p.addModel({ ...msg, name: 'MY_CNN', kind: 'CNN' });
    await p.addModel({ ...msg, name: 'MY_RES', kind: 'ResNet50_1D' });
    assert.deepEqual(applied.map(a => JSON.parse(a.edits[0].value)), [
      { model_id: 'MY_CNN', architecture: 'CNN', architecture_args: { strict_graph_args: true, layers: [] } },
      { model_id: 'MY_RES', architecture: 'ResNet50_1D', architecture_args: {} }]);
    assert.deepEqual(applied[0].edits[0].path, ['model']);
  } finally { p.stop(); }
});

test('a run of the project notes on the configs it reads that changes apply to the next run; graph_config and other projects no note', () => {
  const Q = { ...P, configs: [{ path: '/c/params.yaml' }, { path: '/c/experiment_queue.yaml' }, { path: '/c/graph_config.yaml' }, { path: '/c/dev.yaml' }] };
  const O = { name: 'OTHER', configs: [{ path: '/o/params.yaml' }] };
  const p = new E.Panel({});
  p.layout = { projects: [Q, O] };
  p.busy = [];
  assert.equal(p.runNote('/c/params.yaml'), null);
  p.busy = ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26'];
  for (const f of ['/c/params.yaml', '/c/experiment_queue.yaml', '/c/dev.yaml']) {
    assert.match(p.runNote(f), /^A run of DLSCA_ASCAD is ongoing \(make experiment .*QUEUE=fig26\): it reads the config frozen at its start, so changes apply to the next run\.$/);
  }
  assert.equal(p.runNote('/c/graph_config.yaml'), null);
  assert.equal(p.runNote('/o/params.yaml'), null);
  p.stop();
});

test('a queue\'s ×, and an id-less job\'s or facet\'s ×, ask first and delete the entry on Delete; other files and paths are refused', async () => {
  const p = new E.Panel({});
  p.layout = { projects: [{ ...P, configs: [{ path: '/c/experiment_queue.yaml' }, { path: '/c/graph_config.yaml' }, { path: '/c/params.yaml' }] }] };
  p.busy = [];
  const edits = [];
  p.edit = async (m, asked) => { edits.push([m, asked]); };
  const before = vscodeStub.window.showWarningMessage;
  let answer;
  vscodeStub.window.showWarningMessage = async (...a) => { seen.warning.push(text(a)); return answer; };
  try {
    seen.warning.length = 0;
    const q = { file: '/c/experiment_queue.yaml', sha: 's', kind: 'queue', name: 'fig26', path: ['fig26'] };
    await p.deleteEntry(q);
    assert.match(seen.warning[0], /Delete queue fig26 and its lines from experiment_queue.yaml\? Its runs \(if any were made\) keep their batch label/);
    assert.equal(edits.length, 0);
    answer = 'Delete';
    await p.deleteEntry(q);
    assert.deepEqual(edits.pop(), [{ file: '/c/experiment_queue.yaml', sha: 's', op: 'delete', path: ['fig26'] }, true]);
    await p.deleteEntry({ file: '/c/graph_config.yaml', sha: 's', kind: 'job', name: 'name: ', path: ['jobs', 0] });
    assert.match(seen.warning.at(-1), /Delete job \(name: \) #0 from graph_config.yaml\? An entry without an id/);
    assert.deepEqual(edits.pop()[0], { file: '/c/graph_config.yaml', sha: 's', op: 'delete', path: ['jobs', 0] });
    await p.deleteEntry({ file: '/c/graph_config.yaml', sha: 's', kind: 'facet', name: '', path: ['faceting', 2] });
    assert.deepEqual(edits.pop()[0], { file: '/c/graph_config.yaml', sha: 's', op: 'delete', path: ['faceting', 2] });
    // the wrong file or shape: nothing asked, nothing written
    const n = seen.warning.length;
    await p.deleteEntry({ file: '/c/params.yaml', sha: 's', kind: 'queue', name: 'x', path: ['x'] });
    await p.deleteEntry({ file: '/c/experiment_queue.yaml', sha: 's', kind: 'queue', name: 'x', path: ['x', 0] });
    await p.deleteEntry({ file: '/c/graph_config.yaml', sha: 's', kind: 'job', name: '', path: ['faceting', 0] });
    assert.equal(seen.warning.length, n);
    assert.equal(edits.length, 0);
  } finally { vscodeStub.window.showWarningMessage = before; p.stop(); }
});

test('a model\'s × asks first, naming the queue lines that use it, and deletes only on Delete; during a run it says the run keeps its copy', async () => {
  const Q = { ...P, configs: [{ path: '/c/params.yaml' }],
    queues: [{ key: 'fig26', commands: [{ MODEL_ID: 'ASCAD_CNNbase' }, { MODEL_ID: 'X' }] }, { key: 'fig2', commands: [{ MODEL_ID: 'X' }] }] };
  const p = new E.Panel({});
  p.layout = { projects: [Q] };
  p.busy = [];
  const edits = [];
  const asks = [];
  p.edit = async (m, asked) => { edits.push(m); asks.push(asked); };
  const before = vscodeStub.window.showWarningMessage;
  let answer;
  vscodeStub.window.showWarningMessage = async (...a) => { seen.warning.push(text(a)); return answer; };
  try {
    seen.warning.length = 0;
    const msg = { file: '/c/params.yaml', sha: 's', kind: 'model', name: 'ASCAD_CNNbase', path: ['model', 1] };
    await p.deleteEntry(msg);
    assert.match(seen.warning[0], /Delete model ASCAD_CNNbase from params.yaml\? MODEL_ID=ASCAD_CNNbase is named by queue fig26 \(1 line\)/);
    assert.equal(edits.length, 0);
    answer = 'Delete';
    await p.deleteEntry(msg);
    assert.deepEqual(edits, [{ file: '/c/params.yaml', sha: 's', op: 'delete', path: ['model', 1] }]);
    assert.equal(asks.pop(), true);
    await p.deleteEntry({ ...msg, kind: 'data', name: 'ascad', path: ['data', 0] });
    assert.match(seen.warning[2], /No queue line names DATASET=ascad/);
    p.busy = ['make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26'];
    await p.deleteEntry(msg);
    assert.match(seen.warning[3], /Delete model ASCAD_CNNbase .* A run of DLSCA_ASCAD is ongoing .* changes apply to the next run/);
    assert.equal(edits.length, 3);
  } finally { vscodeStub.window.showWarningMessage = before; p.stop(); }
});

test('MASTER CONTROL: Add a Project writes the blueprint entry; × removes one (never the default); read-only refused, hash asked; a run locks it', { timeout: 120000 }, async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(tmp, 'master.yaml');
  fs.writeFileSync(file, 'project:\n  - name: "A"\n    root: "A/"\n    params: "config/params.yaml"\n    scripts:\n      train: "src/sca/train.py"\ndefault_project: "A"\n');
  const p = new E.Panel({});
  Object.assign(p, { root: ROOT, python: E.findPython(), layout: { master: file, projects: [] }, busy: [] });
  const notes = [];
  p.masterChecked = async (f, note) => notes.push(note);
  p.sendTree = async (f, note) => notes.push(note);
  const before = [vscodeStub.window.showWarningMessage];
  let answer;
  vscodeStub.window.showWarningMessage = async (...a) => { seen.warning.push(text(a)); return answer; };
  const sha = () => JSON.parse(require('child_process').execFileSync(p.python, ['-m', 'services.visualization.vscode_monitor.yamledit', 'tree', file],
    { cwd: ROOT, env: { ...process.env, PYTHONPATH: ROOT } }).toString().trim().split('\n').pop()).sha256;
  try {
    seen.warning.length = 0;
    seen.info.length = 0;
    await p.addProject({ name: 'A' });                                                       // typed in the panel
    await p.addProject({ name: 'my project' });
    assert.deepEqual(notes.splice(0), ['refused: no project added: A is a project already',
      'refused: no project added: letters, digits, _ . - (no spaces)']);
    await p.addProject({ name: 'NEW_P' });
    let text1 = fs.readFileSync(file, 'utf8');
    assert.match(text1, /- name: NEW_P\n    description: ''\n    root: NEW_P\//);
    assert.match(text1, /    extra_configs:\n      graph: config\/graph_config.yaml/);       // the template's whole entry
    assert.match(text1, /^  - name: "A"\n    root: "A\/"/m);                                   // the first one as it was
    assert.match(notes.pop(), /^added project NEW_P/);
    assert.match(seen.info[0], /project NEW_P added to master.yaml; projects\/NEW_P\/config\/ gets the blueprint's files/);

    await p.removeProject({ name: 'A', path: ['project', 0] });                              // default_project
    assert.match(seen.warning.pop(), /A is default_project: choose another default_project first/);
    answer = 'Remove';
    await p.removeProject({ name: 'NEW_P', path: ['project', 1] });
    assert.match(seen.warning.pop(), /Remove project NEW_P from master.yaml\? Its folder projects\/NEW_P\/ and every run in it stay on disk/);
    assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /NEW_P/);

    await p.apply({ file, sha: sha(), edits: [{ op: 'set', path: ['project', 0, 'root'], value: 'B/' }] });
    assert.match(notes.pop(), /^refused: project › 0 › root: the project's folders and files/);
    answer = undefined;
    await p.apply({ file, sha: sha(), edits: [{ op: 'set', path: ['project', 0, 'scripts', 'train'], value: 'src/x.py' }] });
    assert.match(seen.warning.pop(), /Change a key of the run hash in master.yaml\? project › 0 › scripts › train/);
    assert.equal(notes.pop(), 'not written: the run hash keys were left as they are');
    answer = 'Write';
    await p.apply({ file, sha: sha(), edits: [{ op: 'set', path: ['project', 0, 'scripts', 'train'], value: 'src/x.py' }] });
    assert.match(notes.pop(), /^saved: 1 change\(s\) to master.yaml/);
    assert.match(fs.readFileSync(file, "utf8"), /train: "src\/x.py"/);                   // its quotes kept

    p.busy = ['make experiment ENV=dev PROJECT=A QUEUE=q'];
    text1 = fs.readFileSync(file, 'utf8');
    await p.addProject({ name: 'B' });
    assert.match(seen.warning.pop(), /no project added. a run is ongoing \(make experiment ENV=dev PROJECT=A QUEUE=q\): master.yaml is read by every run and by make/);
    await p.edit({ file, sha: sha(), op: 'set', path: ['default_project'], value: 'A' });
    assert.match(notes.pop(), /^refused: a run is ongoing/);
    assert.equal(fs.readFileSync(file, 'utf8'), text1);
  } finally { [vscodeStub.window.showWarningMessage] = before; p.stop(); }
});

test('RUN SINGLE EXPERIMENT → Start: the line as single_runs.yaml holds it, as make orchestrate; a line that cannot run is refused', async () => {
  const Q = { ...P, datasets: ['ascad'], model_ids: ['A'], single_runs: [
    { index: 0, line: 'make orchestrate ENV=prod PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=A', ENV: 'prod', DATASET: 'ascad', MODEL_ID: 'A', error: '' },
    { index: 1, line: 'make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad MODEL_ID=GONE', ENV: 'dev', DATASET: 'ascad', MODEL_ID: 'GONE', error: '' },
    { index: 2, line: 'make orchestrate ENV=dev PROJECT=X DATASET=ascad MODEL_ID=A', ENV: 'dev', PROJECT: 'X', DATASET: 'ascad', MODEL_ID: 'A', error: '' },
    { index: 3, line: 'make orchestrate ENV=dev PROJECT=OTHER DATASET=dpav4 MODEL_ID=Z', ENV: 'dev', PROJECT: 'OTHER', DATASET: 'dpav4', MODEL_ID: 'Z', error: '' },
    { index: 4, line: 'make orchestrate ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad', error: 'ENV, DATASET and MODEL_ID are needed' }] };
  const p = panel(Q);
  p.layout.projects.push({ name: 'OTHER', datasets: ['dpav4'], model_ids: ['Z'], queues: [] });
  p.followed = () => ({ env: 'dev', source: 'test' });
  const started = [];
  p.startLong = async (proj, kind, vars, n) => started.push([proj.name, kind, vars, n]);
  seen.warning.length = 0;
  try {
    await p.run({ action: 'single', project: 'DLSCA_ASCAD', key: '0' });
    assert.deepEqual(started, [['DLSCA_ASCAD', 'orchestrate', { ENV: 'prod', PROJECT: 'DLSCA_ASCAD', DATASET: 'ascad', MODEL_ID: 'A' }, 1]]);
    await p.run({ action: 'single', project: 'DLSCA_ASCAD', key: '1' });
    assert.match(seen.warning.pop(), /single run 2 not started: MODEL_ID=GONE is no model_id of DLSCA_ASCAD's params.yaml/);
    await p.run({ action: 'single', project: 'DLSCA_ASCAD', key: '2' });
    assert.match(seen.warning.pop(), /single run 3 not started: PROJECT=X is no project of master.yaml/);
    await p.run({ action: 'single', project: 'DLSCA_ASCAD', key: '4' });
    assert.match(seen.warning.pop(), /single run 5 not started: ENV, DATASET and MODEL_ID are needed/);
    await p.run({ action: 'single', project: 'DLSCA_ASCAD', key: '3' });    // another project: runs as that one
    assert.deepEqual(started.pop(), ['OTHER', 'orchestrate', { ENV: 'dev', PROJECT: 'OTHER', DATASET: 'dpav4', MODEL_ID: 'Z' }, 1]);
    await p.run({ action: 'single', project: 'DLSCA_ASCAD', key: '9' });
    assert.match(seen.warning.pop(), /no single run 9 in single_runs.yaml/);
    assert.equal(started.length, 1);
  } finally { p.stop(); }
});


// ── Plan a Model: the planner page's host ───────────────────────────────────────────────────────────────
const PARAMS = 'model:\n  - model_id: "A"   # the first\n    architecture: "MLP"\n    architecture_args:\n      layers:\n        - Flatten: {}\n        - Linear: {in_features: 700, out_features: "auto"}\n';
function planner() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-plan-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, PARAMS);
  const p = panel(P);
  const sent = [];
  const viewSent = [];
  p.sendTree = async () => {};
  p.view = { webview: { postMessage: async m => { viewSent.push(m); } } };
  p.openTrees.add(file);                                // Visualise is pressed in that params.yaml form: it is open
  const pl = { project: 'DLSCA_ASCAD', file, panel: { webview: { postMessage: async m => { sent.push(m); } } } };
  return { p, pl, file, sent, viewSent };
}
const LAYERS_A = [{ Flatten: {} }, { Linear: { in_features: 700, out_features: 'auto' } }];
const mlp = id => ({ model_id: id, architecture: 'MLP', architecture_args: { strict_graph_args: true, layers: [{ Flatten: {} }] } });

test('Plan a Model → Download: model_diagram.py gets the drawing and the env; the page is told what was saved, copied, not copied', async () => {
  const { p, pl, sent } = planner();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-dl-'));
  const fake = path.join(dir, 'python');                                  // stands in for model_diagram.py: records, answers
  const answer = path.join(dir, 'answer.json');
  fs.writeFileSync(fake, `#!/bin/sh\necho "$@" > ${dir}/args\ncat > ${dir}/in.json\ncat ${answer}\n`, { mode: 0o755 });
  p.python = fake;
  p.followed = () => ({ env: 'staging' });
  const root = p.root;
  try {
    fs.writeFileSync(answer, JSON.stringify({ written: [path.join(root, 'projects/DLSCA_ASCAD/artifacts/staging/model_preview/A.png')],
      copied: ['/x/r1/model/r1_diagram.png', '/x/r2/model/r2_diagram.png'], skipped: ['r3'], denied: [] }));
    await p.plannerDownload(pl, { type: 'download', title: 'A', model_id: 'A', env: '', layers: [], export: { formats: ['png'] } });
    assert.match(fs.readFileSync(path.join(dir, 'args'), 'utf8'), /-m services\.visualization\.vscode_monitor\.model_diagram --project DLSCA_ASCAD/);
    assert.equal(JSON.parse(fs.readFileSync(path.join(dir, 'in.json'), 'utf8')).env, 'staging', 'the env followed, when no run gives one');
    assert.deepEqual({ ...sent.at(-1) }, { type: 'saved',
      message: 'saved A.png in projects/DLSCA_ASCAD/artifacts/staging/model_preview; copied into the model folder of 2 run(s); not into r3 (other layers)' });
    fs.writeFileSync(answer, JSON.stringify({ written: [path.join(root, 'p/A.png')], copied: [], skipped: [], denied: ['r4'] }));
    await p.plannerDownload(pl, { type: 'download', title: 'A', model_id: 'A', env: 'dev', layers: [], export: { formats: ['png'] } });
    assert.equal(JSON.parse(fs.readFileSync(path.join(dir, 'in.json'), 'utf8')).env, 'dev', 'a trained run\'s own env');
    assert.equal(sent.at(-1).type, 'refused');
    assert.match(sent.at(-1).message, /refused: not into r4 \(owned by root: services\/doc\/03_file_permissions\.md §3\.1\)$/);
    fs.writeFileSync(answer, JSON.stringify({ error: 'ValueError: params.yaml paths.model_preview is not set' }));
    await p.plannerDownload(pl, { type: 'download', title: 'A', layers: [], export: {} });
    assert.equal(sent.at(-1).message, 'refused: not downloaded: ValueError: params.yaml paths.model_preview is not set');
  } finally { p.stop(); }
});

test('Plan a Model: Add the Model writes nothing for a taken model_id and appends a unique one', async () => {
  const { p, pl, file, sent } = planner();
  try {
    await p.plannerAdd(pl, mlp('A'));
    assert.equal(sent.at(-1).type, 'refused');
    assert.match(sent.at(-1).message, /A is a model already/);
    assert.equal(fs.readFileSync(file, 'utf8'), PARAMS);
    await p.plannerAdd(pl, mlp('B'));
    assert.equal(sent.at(-1).type, 'saved');
    assert.deepEqual(sent.at(-1).ids, ['A', 'B']);
    const text = fs.readFileSync(file, 'utf8');
    assert.ok(text.startsWith(PARAMS.split('\n').slice(0, 6).join('\n')) && /model_id: B/.test(text));
  } finally { p.stop(); }
});

test('Plan a Model: Add the Model is refused while the params form has staged edits', async () => {
  const { p, pl, file, sent } = planner();
  try {
    p.staged = { [file]: 2 };
    await p.plannerAdd(pl, mlp('B'));
    assert.equal(sent.at(-1).type, 'refused');
    assert.match(sent.at(-1).message, /2 staged change\(s\)/);
    assert.equal(fs.readFileSync(file, 'utf8'), PARAMS);
  } finally { p.stop(); }
});

test('Visualise → Add The Model: the panel\'s staged copy of the changes is not counted, dropped before the write, staged again when it fails', async () => {
  const { p, pl, file, sent, viewSent } = planner();
  const from = { model_id: 'A', base: LAYERS_A, arrange: [{ from: 1 }, { from: 0 }] };
  try {
    p.staged = { [file]: 2 };
    p.stagedArranged = { [file]: 1 };
    await p.plannerAdd(pl, mlp('B'), from);
    assert.match(sent.at(-1).message, /1 other staged change\(s\)/, 'the form\'s other staged change still refuses');
    assert.equal(viewSent.length, 0);
    p.staged = { [file]: 1 };
    const real = p.treeOf.bind(p);
    p.treeOf = async f => ({ ...(await real(f)), sha256: 'the text the tree was read from' });
    await p.plannerAdd(pl, mlp('B'), from);
    assert.match(sent.at(-1).message, /changed on disk/);
    assert.deepEqual(viewSent.map(m => m.edit.value), ['[{"from":0},{"from":1}]', '[{"from":1},{"from":0}]'], 'dropped, then staged again');
    assert.equal(fs.readFileSync(file, 'utf8'), PARAMS);
    p.treeOf = real;
    viewSent.length = 0;
    await p.plannerAdd(pl, mlp('B'), from);
    assert.equal(sent.find(m => m.type === 'saved').message, 'B added to params.yaml');
    assert.deepEqual(viewSent.map(m => m.edit.value), ['[{"from":0},{"from":1}]'], 'A stays as the file has it');
    assert.match(fs.readFileSync(file, 'utf8'), /model_id: B/);
  } finally { p.stop(); }
});

test('Visualise → Preview Code: the model\'s own entry under its model_id as OK would write it (comment and style kept), the edited line marked; nothing written', async () => {
  const { p, pl, file, sent } = planner();
  fs.writeFileSync(file, `# about B\n${PARAMS}  # about B\n  - model_id: "B"\n    architecture: "MLP"\n    architecture_args:\n      layers:\n        - Flatten: {}\n`);
  const text = fs.readFileSync(file, 'utf8');
  try {
    await p.plannerPreview(pl, { type: 'preview', model_id: 'A', arrange: [{ from: 0 }, { from: 1, value: { Linear: { in_features: 700, out_features: 9 } } }] });
    const m = sent.at(-1);
    assert.deepEqual(m.added, ['  - model_id: "A"   # the first', '    architecture: "MLP"', '    architecture_args:', '      layers:',
      '        - Flatten: {}', '        - Linear: {in_features: 700, out_features: 9}']);
    assert.deepEqual(m.marked, [5]);
    assert.equal(fs.readFileSync(file, 'utf8'), text);
    await p.plannerPreview(pl, { type: 'preview', model_id: 'B', arrange: [{ from: 0 }] });
    assert.deepEqual(sent.at(-1).added.slice(0, 2), ['  # about B', '  - model_id: "B"'], 'its comment above it');
    assert.deepEqual(sent.at(-1).marked, []);
    await p.plannerPreview(pl, { type: 'preview', model_id: 'GONE', arrange: [] });
    assert.deepEqual(sent.at(-1).added, ['GONE is no longer in params.yaml']);
  } finally { p.stop(); }
});

test('Plan a Model: a params.yaml changed on disk refuses the add and reloads the ids', async () => {
  const { p, pl, file, sent } = planner();
  try {
    const real = p.treeOf.bind(p);
    p.treeOf = async f => ({ ...(await real(f)), sha256: 'the text the tree was read from' });
    await p.plannerAdd(pl, mlp('B'));
    assert.equal(sent.at(-1).type, 'refused');
    assert.match(sent.at(-1).message, /changed on disk/);
    assert.deepEqual(sent.at(-1).ids, ['A']);
    assert.equal(fs.readFileSync(file, 'utf8'), PARAMS);
  } finally { p.stop(); }
});

test('Plan a Model: staging refuses when the visualised model is gone, else stages its arrange in the panel', async () => {
  const { p, pl, file, sent, viewSent } = planner();
  try {
    await p.plannerStage(pl, { model_id: 'GONE', base: LAYERS_A, arrange: [{ from: 0 }] });
    assert.equal(sent.at(-1).type, 'refused');
    assert.match(sent.at(-1).message, /GONE is no longer in params.yaml/);
    assert.equal(viewSent.length, 0);
    await p.plannerStage(pl, { model_id: 'A', base: LAYERS_A, arrange: [{ from: 1 }, { from: 0 }] });
    assert.deepEqual(viewSent.at(-1), { type: 'stage', file, model_id: 'A',
      edit: { op: 'arrange', path: ['model', 0, 'architecture_args', 'layers'], value: '[{"from":1},{"from":0}]' } });
  } finally { p.stop(); }
});

test('the panel\'s Add a Model refuses a model_id another window added meanwhile', async () => {
  const { p, file } = planner();
  const notes = [];
  p.configFile = f => f;
  p.sendTree = async (f, note) => notes.push(note);
  p.apply = async () => { throw new Error('nothing should be written'); };
  try {
    await p.addModel({ project: 'DLSCA_ASCAD', file, sha: 's', ids: [], kinds: ['CNN', 'MLP'], named: [], name: 'A', kind: 'CNN' });
    assert.deepEqual(notes, ['refused: no model added: A is a model already']);
  } finally { p.stop(); }
});

// ── final review fixes of the planner host ─────────────────────────────────────────────────────────────
test('Plan a Model: staging refuses when the model\'s layers changed since the page opened it, and sends the page the file\'s models', async () => {
  const { p, pl, sent, viewSent } = planner();
  try {
    await p.plannerStage(pl, { model_id: 'A', base: [{ Flatten: {} }], arrange: [{ from: 0 }] });
    assert.equal(viewSent.length, 0);
    assert.match(sent.find(m => m.type === 'refused').message, /A changed in params.yaml since the page opened it/);
    assert.deepEqual(sent.find(m => m.type === 'models').ids, ['A']);
    await p.plannerStage(pl, { model_id: 'A', base: LAYERS_A, arrange: [{ from: 1 }, { from: 0 }] });
    assert.equal(viewSent.at(-1).type, 'stage');
  } finally { p.stop(); }
});

test('Plan a Model: every write to the file sends its planner the models as they are now', async () => {
  const { p, pl, file, sent } = planner();
  p.planners = { DLSCA_ASCAD: pl };
  p.configFile = f => f;
  try {
    await p.plannerAdd(pl, mlp('B'));
    assert.deepEqual(sent.filter(m => m.type === 'models').at(-1).ids, ['A', 'B']);
    const t = await p.treeOf(file);
    await p.apply({ file, sha: t.sha256, edits: [{ op: 'arrange', path: ['model', 0, 'architecture_args', 'layers'], value: '[{"from":1},{"from":0}]' }] });
    const last = sent.filter(m => m.type === 'models').at(-1);
    assert.deepEqual(last.models[0].architecture_args.layers, [LAYERS_A[1], LAYERS_A[0]]);
  } finally { p.stop(); }
});

test('Plan a Model: the uniqueness check reads the ids from the very tree the write is checked against', async () => {
  const { p, pl, file, sent } = planner();
  p.modelIds = async () => [];                                // a second read that failed must not pass the check
  try {
    await p.plannerAdd(pl, mlp('A'));
    assert.equal(sent.at(-1).type, 'refused');
    assert.equal(fs.readFileSync(file, 'utf8'), PARAMS);
  } finally { p.stop(); }
});

test('Plan a Model: the planner tab keeps its page while hidden', async () => {
  const { p, file } = planner();
  let options = null;
  const fake = { webview: { html: '', onDidReceiveMessage: () => {}, postMessage: async () => {}, asWebviewUri: u => u, cspSource: 'c' },
    onDidDispose: () => {}, reveal: () => {}, title: '' };
  vscodeStub.window.createWebviewPanel = (id, title, col, opts) => { options = opts; return fake; };
  vscodeStub.ViewColumn = { Beside: 2 };
  p.context = { extensionPath: path.join(__dirname, '..') };
  p.layout = { projects: [{ ...P, name: 'DLSCA_ASCAD' }] };
  p.configFile = f => f;
  p.describeProject = async () => ({ framework: 'pytorch', input: [1, 700], models: [] });
  p.menus = () => ({ params: { models: null } });
  try {
    await p.openPlanner('DLSCA_ASCAD', file);
    assert.equal(options.retainContextWhenHidden, true);
  } finally { p.stop(); delete vscodeStub.window.createWebviewPanel; }
});

test('Plan a Model: a new check stops the one still running', async () => {
  const { p, pl } = planner();
  const old = { killed: false, kill() { this.killed = true; } };
  pl.child = old;
  try {
    await p.plannerCheck(pl, { layers: [{ Flatten: {} }], architecture: 'MLP' });
    assert.equal(old.killed, true);
  } finally { p.stop(); }
});

test('Plan a Model: a write at once (a layer moved in the panel) sends its planner the models too', async () => {
  const { p, pl, file, sent } = planner();
  p.planners = { DLSCA_ASCAD: pl };
  p.configFile = f => f;
  try {
    const t = await p.treeOf(file);
    await p.edit({ file, sha: t.sha256, op: 'move', path: ['model', 0, 'architecture_args', 'layers', 0], value: '1' });
    assert.deepEqual(sent.filter(m => m.type === 'models').at(-1).models[0].architecture_args.layers, [LAYERS_A[1], LAYERS_A[0]]);
  } finally { p.stop(); }
});

test('Plan a Model: a stage the panel refused is told to the planner page', () => {
  const { p, pl, file, sent } = planner();
  p.planners = { DLSCA_ASCAD: pl };
  p.configFile = f => f;
  try {
    p.onMessage({ type: 'stage-refused', file, model_id: 'A', reason: 'the panel has staged edits of A\'s layers' });
    assert.match(sent.at(-1).message, /refused: nothing staged: the panel has staged edits of A's layers/);
  } finally { p.stop(); }
});

test('Visualise → OK: the model\'s changes written as one arrange of its layers; refused over the panel\'s other staged changes or changed layers', async () => {
  const { p, pl, file, sent, viewSent } = planner();
  const yaml = () => fs.readFileSync(file, 'utf8');
  try {
    p.staged = { [file]: 2 };
    p.stagedArranged = { [file]: 1 };                                    // one is the planner's own, one another edit
    await p.plannerApply(pl, { model_id: 'A', base: LAYERS_A, arrange: [{ from: 1 }, { from: 0 }] });
    assert.match(sent.at(-1).message, /refused: nothing written: the panel's params\.yaml form has 1 other staged change\(s\)/);
    assert.equal(yaml(), PARAMS);
    p.staged = { [file]: 1 };                                            // only the planner's: OK writes it
    await p.plannerApply(pl, { model_id: 'A', base: [{ Flatten: {} }], arrange: [{ from: 0 }] });
    assert.match(sent.filter(m => m.type === 'refused').at(-1).message, /refused: nothing written: A changed in params\.yaml since the page opened it/);
    assert.equal(sent.at(-1).type, 'models', 'and the page is shown the model as the file holds it');
    assert.equal(yaml(), PARAMS);
    await p.plannerApply(pl, { model_id: 'A', base: LAYERS_A, arrange: [{ from: 1 }, { from: 0 }] });
    assert.equal(sent.find(m => m.type === 'saved').message, 'A\'s layers written to params.yaml');
    const now = JSON.parse(require('child_process').execFileSync(E.findPython(), ['-c',
      'import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))["model"]))', file]).toString());
    assert.deepEqual(now.find(m => m.model_id === 'A').architecture_args.layers, [LAYERS_A[1], LAYERS_A[0]]);
    const dropped = viewSent.find(m => m.type === 'stage');             // the panel's staged copy dropped first
    assert.deepEqual(JSON.parse(dropped.edit.value), [{ from: 0 }, { from: 1 }]);
    assert.ok(sent.some(m => m.type === 'models'), 'the page learns the file\'s models');
  } finally { p.stop(); }
});

test('the status page: links with their QR codes from status.py, a withdrawal asked first, Start in its own tmux session', async () => {
  const p = panel(P);
  const shares = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'tf-status-')), 'shares.json');
  process.env.TRACEFLOW_STATUS_SHARES = shares;
  const posted = [];
  p.view = { webview: { postMessage: async m => { posted.push(m); } } };
  p.statusRunning = async () => false;                              // not this machine's real tf-status session
  const info = vscodeStub.window.showInformationMessage;
  try {
    await p.statusShare('add', 'Professor');
    const st = posted.at(-1);
    assert.equal(st.type, 'status-page');
    assert.equal(st.running, false);
    assert.equal(st.enabled, true, 'master.yaml status_share');
    assert.match(st.host, /^\d{1,3}(?:\.\d{1,3}){3}$/, 'master.yaml status_share.host, or this machine\'s own address');
    assert.equal(st.port, 8765);
    assert.equal(st.shares.length, 1);
    assert.match(st.shares[0].link, new RegExp(`^http://${st.host.replace(/\./g, '\\.')}:8765/\\?t=[A-Za-z0-9_-]{24}$`));
    assert.match(st.shares[0].qr, /^<svg[^>]*class="segno"/);
    await p.statusShare('add', 'Professor');
    assert.match(posted.at(-1).note, /^refused: Professor has a link already/);
    await p.statusShare('remove', 'Professor');                         // Cancel: nothing withdrawn
    assert.equal(posted.at(-1).shares.length, 1);
    vscodeStub.window.showInformationMessage = async (m, o, ...b) => (/^Withdraw Professor's link\?/.test(m) ? b[0] : undefined);
    await p.statusShare('remove', 'Professor');
    assert.equal(posted.at(-1).shares.length, 0);
  } finally {
    vscodeStub.window.showInformationMessage = info;
    delete process.env.TRACEFLOW_STATUS_SHARES;
    p.stop();
  }
});

test('a person\'s link opens on this desktop: the loopback at its port, through VS Code\'s forwarding, with its token', async () => {
  const p = panel(P);
  const opened = [];
  const env = vscodeStub.env;
  vscodeStub.Uri = vscodeStub.Uri || {};
  const parse = vscodeStub.Uri.parse;
  vscodeStub.Uri.parse = s => ({ toString: () => s });
  vscodeStub.env = { ...(env || {}), asExternalUri: async u => ({ toString: () => u.toString().replace('localhost:8765', 'localhost:49152') }),
    openExternal: async u => { opened.push(u.toString()); return true; } };
  try {
    await p.statusVisit('http://192.168.88.2:8765/?t=Abc-_1');
    assert.deepEqual(opened, ['http://localhost:49152/?t=Abc-_1']);
  } finally { vscodeStub.env = env; vscodeStub.Uri.parse = parse; p.stop(); }
});

test('Judge asks the process and the metric among those the runs recorded, then runs make judge; a cancelled pick runs nothing', { timeout: 120000 }, async () => {
  const p = panel({ ...P, configs: [{ file: 'model_judge.yaml', path: path.join(ROOT, 'projects/DLSCA_ASCAD/config/model_judge.yaml') }] });
  const lines = [];
  p.runShort = (proj, key, line) => lines.push(line);        // no terminal, no result poll (awaitResult's timer)
  p.pollProcesses = async () => { p.busy = []; p.sessions = {}; };
  const picks = [];
  const quick = vscodeStub.window.showQuickPick;
  vscodeStub.window.showQuickPick = async (items, opts) => { picks.push([opts.title, items.map(i => i.label)]); return items[0]; };
  try {
    await p.run({ action: 'judge', project: 'DLSCA_ASCAD' });
  } finally { vscodeStub.window.showQuickPick = quick; }
  assert.equal(picks.length, 2);
  assert.match(picks[0][0], /rank the models on which process/);
  assert.ok(picks[0][1].includes('cross_validation'), 'the processes the DLSCA runs recorded');
  assert.equal(picks[0][1][0], 'cross_validation', "the file's own process first");
  assert.match(picks[1][0], /which cross_validation metric/);
  assert.equal(picks[1][1][0], 'mean_curve_final_ge', "the file's own metric first");
  assert.equal(lines.length, 1);
  assert.match(lines[0], /make judge ENV=dev PROJECT=DLSCA_ASCAD$/);
  // the first pick cancelled: no second question, nothing run, the file untouched
  const before = fs.readFileSync(path.join(ROOT, 'projects/DLSCA_ASCAD/config/model_judge.yaml'), 'utf8');
  vscodeStub.window.showQuickPick = async () => undefined;
  try { await p.run({ action: 'judge', project: 'DLSCA_ASCAD' }); } finally { vscodeStub.window.showQuickPick = quick; }
  assert.equal(lines.length, 1);
  assert.equal(fs.readFileSync(path.join(ROOT, 'projects/DLSCA_ASCAD/config/model_judge.yaml'), 'utf8'), before);
});
