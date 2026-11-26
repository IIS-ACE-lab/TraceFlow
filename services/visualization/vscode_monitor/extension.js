// services/visualization/vscode_monitor/extension.js
'use strict';
/**
 * extension.js
 * ============
 * The TraceFlow Monitor's VS Code glue. Paths come from
 * `python -m services.visualization.vscode_monitor.layout export`, re-run
 * whenever master.yaml or a project's config changes. Every poll it reads
 * each project's heartbeat and (when its mtime changed) run registry; every
 * 5 s nvidia-smi; every 10 s the graph job folders. One state message goes
 * to the side-bar view; a run's detail tab beside the editor gets that run's
 * curves. Nothing here starts or stops a run, so closing VS Code changes
 * nothing.
 *
 * Config forms: an open form is `yamledit.py tree`; the form stages its
 * changes and OK sends them together (apply), each one `yamledit.py edit`
 * carrying the hash of the text the form was drawn from.
 * When the file changed meanwhile (by hand, or another form) nothing is
 * written and the form is drawn again from the file: the file wins. A
 * change on disk redraws every open form of that file. On activation and on
 * every change of master.yaml, `layout.py scaffold` copies the blueprint
 * files a declared project lacks (never overwriting one).
 *
 * Buttons run one line from commands.js. Long ones (Orchestrate, Test, a
 * queue's Start) run detached in tmux session tf-<project> inside the sca
 * conda env, their output also appended to projects/<P>/artifacts/<ENV>/tmux_logs/, and a visible terminal
 * attaches to the session; closing it (or VS Code) leaves the run going.
 * Start refuses while any `make orchestrate|experiment|test` runs on this
 * machine (make orchestrate begins with make stack-down, which would stop
 * it) or while tf-<project> is still running something, and asks first,
 * showing the experiments params.yaml plans. Short ones (Draw, Find models,
 * Judge) run in the project's terminal and open their result beside the
 * panel once it is written; Test's test_registry.json opens likewise. Every
 * button's line first gives projects/ back to the user (sudo chown -R): the
 * container writes there as root.
 *
 * Author: Shraddha Mishra
 */

const vscode = require('vscode');
const crypto = require('crypto');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFile } = require('child_process');
const C = require('./commands');
const S = require('./state');
const G = require('./gpu');
const H = require('./history');
const M = require('./webview/menus');
const PC = require('./webview/planner_core');      // Plan a Model: the model_id rule the page and the write share           // masterTier: Master Control's read-only and hash keys

const GPU_MS = 5000;
const BUILD_LOG = 'logs/build_image.log';       // make build-image's output (Makefile BUILD_LOG)
const NERDCTL = ['-n', 'nerdctl', '--namespace', 'default'];   // sudo -n: fails instead of asking
const FIGURE_MS = 10000;
const PLANNER_VIEW = 'traceflow.planner';     // the planner tab's viewType (package.json onWebviewPanel; its serializer)
const CHOICES_KEY = 'traceflow.metricChoices';   // workspaceState: each Done run's metric menu, with the fingerprint of its files
const RELAYOUT_DEBOUNCE_MS = 500;
const SPARK = { w: 80, h: 14 };
const DETAIL_SPARK = { w: 160, h: 24 };
const MAX_SERIES = 40;
const RESULT_WAIT_MS = 20 * 60 * 1000;   // a short command's result is awaited this long
const TEST_WAIT_MS = 24 * 3600 * 1000;    // make test (a container run): its test_registry.json this long
const MASTER_GLOB = 'services/side_channel_core/config/master.yaml';
const cfg = key => vscode.workspace.getConfiguration('traceflow').get(key);

function findRoot() {
  if (cfg('root')) return path.resolve(cfg('root'));
  // Bootstrap only: the repository must be found before layout.py can run; every other path comes from layout.
  for (const f of vscode.workspace.workspaceFolders || []) {
    if (fs.existsSync(path.join(f.uri.fsPath, MASTER_GLOB))) return f.uri.fsPath;
  }
  return null;
}

/** The sca conda env's python (made by services/side_channel_core/infra/environment/setup_sca_env.sh), else python3. */
function findPython() {
  if (cfg('python')) return cfg('python');
  const bases = [process.env.CONDA_EXE ? path.dirname(path.dirname(process.env.CONDA_EXE)) : null,
    ...['anaconda3', 'miniconda3', 'miniforge3', 'mambaforge'].map(d => path.join(os.homedir(), d)), '/opt/conda'];
  for (const base of bases) {
    const p = base && path.join(base, 'envs', 'sca', 'bin', 'python');
    if (p && fs.existsSync(p)) return p;
  }
  return 'python3';
}

/**
 * The message dialogs. A modal one shows its first line as the dialog's heading and the rest as its
 * detail, which VS Code wraps: a long text is read in full, not cut. One line without a break is
 * split after its first sentence. A non-modal message (a notification) is passed on as it is.
 */
function dialog(kind) {
  return (text, ...rest) => {
    const show = vscode.window[kind].bind(vscode.window);
    const [opts] = rest;
    if (!opts || typeof opts !== 'object' || !opts.modal) return show(text, ...rest);
    if (opts.detail) return show(wrapText(text), { ...opts, detail: wrapText(opts.detail) }, ...rest.slice(1));
    const s = String(text);
    let cut = s.indexOf('\n');
    if (cut < 0 && s.length > 90) {
      const m = s.slice(20).match(/[.:?] /);
      cut = m ? 20 + m.index + 1 : -1;
    }
    if (cut <= 0) return show(wrapText(s), ...rest);
    return show(wrapText(s.slice(0, cut).trim()), { ...opts, detail: wrapText(s.slice(cut + 1).trim()) }, ...rest.slice(1));
  };
}

/**
 * A dialog's text broken into lines of at most `width` characters, whatever the dialog style wraps
 * (a native one may not; no dialog breaks a long path or command): at spaces, and a longer word (a path,
 * a make line's VAR=value) after a '/', ',' or '_', else at the width. The text's own line breaks stay.
 */
function wrapText(text, width = WRAP) {
  const breakWord = w => {
    const parts = [];
    while (w.length > width) {
      const at = Math.max(w.lastIndexOf('/', width), w.lastIndexOf(',', width), w.lastIndexOf('_', width));
      const cut = at > width / 3 ? at + 1 : width;
      parts.push(w.slice(0, cut));
      w = w.slice(cut);
    }
    return parts.concat(w);
  };
  return String(text).split('\n').map(line => {
    if (line.length <= width) return line;
    const indent = (line.match(/^\s*/) || [''])[0];
    const out = [];
    let cur = '';
    for (const word of line.trim().split(/ +/).flatMap(breakWord)) {
      if (cur && (cur.length + 1 + word.length) > width) { out.push(cur); cur = indent + word; }
      else cur = cur ? `${cur} ${word}` : indent + word;
    }
    if (cur) out.push(cur);
    return out.join('\n');
  }).join('\n');
}
const WRAP = 72;

const say = { warn: dialog('showWarningMessage'), info: dialog('showInformationMessage'), error: dialog('showErrorMessage') };
// Load Blueprint's warning (owner 2026-10-24, word for word)
const COMPLETE_WARNING = 'This action will override the current file and load the default values from the blueprint file.';

/**
 * A choice with words to read: a modal dialog whose detail (wrapped) explains it and whose buttons are
 * the answers. A pick list cuts a long line; it is kept for long lists of short names (MODEL_ID, DATASET).
 * Resolves to the chosen label, or undefined on Cancel.
 */
function choose(heading, detail, labels) {
  return say.info(heading, { modal: true, detail }, ...labels);
}

function readText(p) { try { return fs.readFileSync(p, 'utf8'); } catch (e) { return null; } }
function mtimeMs(p) { try { return fs.statSync(p).mtimeMs; } catch (e) { return null; } }
function inside(root, p) { const r = path.resolve(String(p || '')); return r === root || r.startsWith(root + path.sep); }

/** One helper call: `python -m services.visualization.vscode_monitor.<module> ...args`, stdout parsed as JSON. */
function runPy(root, python, module, args, extraEnv = {}) {
  const pythonPath = [root, path.join(root, 'services', 'side_channel_core'), path.join(root, 'services', 'side_channel_core', 'src')].join(path.delimiter);
  return new Promise((resolve, reject) => {
    execFile(python, ['-m', `services.visualization.vscode_monitor.${module}`, ...args],
      { cwd: root, env: { ...process.env, ...extraEnv, PYTHONPATH: pythonPath }, timeout: 60000, maxBuffer: 32 * 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) return reject(new Error(String(stderr || err.message).trim().split('\n').pop()));
        // stdout should be the JSON alone; a stray log line before it is skipped (the last JSON line counts)
        try { resolve(JSON.parse(stdout)); } catch (e) {
          const line = String(stdout).split('\n').reverse().find(l => l.trim().startsWith('{'));
          try { resolve(JSON.parse(line || '')); } catch (e2) { reject(new Error(`${module} printed no JSON`)); }
        }
      });
  });
}

/**
 * `python -m <module> ...args` for a module outside the monitor (graph_manager/graph_export.py): its
 * last JSON line, also on a non-zero exit, where it prints {"error": ...}.
 */
function runModule(root, python, module, args, input = null, onChild = null) {
  const pythonPath = [root, path.join(root, 'services', 'side_channel_core'), path.join(root, 'services', 'side_channel_core', 'src')].join(path.delimiter);
  return new Promise((resolve, reject) => {
    const child = execFile(python, ['-m', module, ...args],
      { cwd: root, env: { ...process.env, PYTHONPATH: pythonPath }, timeout: 120000, maxBuffer: 32 * 1024 * 1024 },
      (err, stdout, stderr) => {
        const line = String(stdout).split('\n').reverse().find(l => l.trim().startsWith('{'));
        try { return resolve(JSON.parse(line || '')); } catch (e) { /* no JSON: the failure below */ }
        reject(new Error(String(stderr || (err && err.message) || `${module} printed no JSON`).trim().split('\n').pop()));
      });
    if (input != null && child.stdin) child.stdin.end(input);           // e.g. the planner's layer list (model_check --layers-json -)
    if (onChild) onChild(child);
  });
}
const MODEL_CHECK = 'services.visualization.vscode_monitor.model_check';
const MODEL_JUDGE = 'services.model_judge.model_judge';                   // --list-json: what the runs recorded (Judge asks with it)
const MODEL_DIAGRAM = 'services.visualization.vscode_monitor.model_diagram';
const TRACE_VIEW = 'services.visualization.vscode_monitor.trace_view';
const STATUS = 'services.visualization.server.status';     // the read-only status page for phones and shared links
const STATUS_SESSION = 'tf-status';
/** The model_ids of a params.yaml form tree (yamledit tree). */
function idsOfTree(tree) {
  const list = tree && tree.k === 'map' ? (tree.entries.find(e => e.key === 'model') || {}).node : null;
  return list && list.k === 'list'
    ? list.items.map(it => (it.k === 'map' ? ((it.entries.find(e => e.key === 'model_id') || {}).node || {}).text : null)).filter(Boolean).map(String) : [];
}
const GRAPH_EXPORT = 'services.graph_manager.graph_export';
// the config files a run reads, from its frozen copy (utils/config_freeze.py): an edit during the run reaches the next one (Panel.runNote)
const RUN_FILES = /^(params[^/]*|experiment_queue|dev|staging|prod|test_config|auto_search|model_config)\.ya?ml$/;

/** stdout of a command, '' when it fails (a missing tmux or no match is not an error here). */
function output(cmd, args, cwd) {
  return new Promise(resolve => {
    execFile(cmd, args, { cwd, timeout: 10000 }, (err, stdout) => resolve(err ? '' : String(stdout)));
  });
}

/** Runs a line in bash; rejects with its stderr. */
function bash(line, cwd) {
  return new Promise((resolve, reject) => {
    execFile('bash', ['-c', line], { cwd, timeout: 30000 }, (err, stdout, stderr) => {
      if (err) reject(new Error(String(stderr || err.message).trim())); else resolve(String(stdout));
    });
  });
}

/** Relative paths of the files in a folder and its sub-folders, `depth` levels down. */
function listFiles(dir, depth = 1, rel = '') {
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) { return []; }
  const out = [];
  for (const e of entries) {
    const r = rel ? `${rel}/${e.name}` : e.name;
    if (e.isFile()) out.push(r);
    else if (e.isDirectory() && depth > 0) out.push(...listFiles(path.join(dir, e.name), depth - 1, r));
  }
  return out.sort();
}

class Panel {
  constructor(context) {
    this.context = context;
    this.view = null;
    this.detail = null;
    this.detailRun = null;          // {project, hash}
    this.layout = null;
    this.error = '';
    this.timers = [];
    this.watchers = [];
    this.registries = new Map();    // registry path -> {mtime, data}
    this.jobDirs = {};              // project name -> {job folder: [files]}
    this.gpus = [];
    this.gpuError = '';
    this.gpuHistory = new G.History(60);   // 5 minutes at one sample per 5 s
    this.docker = { containers: [], error: '', building: false };
    this.rankCache = new Map();     // epNNNN.npy path -> rank
    this.sparks = new Map();        // model hash -> {status, spark, last, epochs}
    this.liveOf = new Map();        // model hash -> {mtime, live, pending}: core/monitor.snapshot of the running run
    this.readers = new Map();       // history jsonl path -> {reader, records}
    this.openTrees = new Set();     // config files whose form is open
    this.trialsOf = {};             // project -> {mtime, trials, pending}: the sweep, for the batch ETA
    this.sessions = {};             // tmux session -> {command, busy}
    this.busy = [];                 // running make orchestrate|experiment|test, or a container's trainer
    this.waits = new Map();         // short command key -> timer awaiting its result
  }

  async start() {
    this.root = findRoot();
    this.python = findPython();
    this.envOverride = String(cfg('env') || '').trim();      // empty: follow the commands (state.followEnv)
    if (!this.root) {
      this.error = 'TraceFlow root not found: open the TraceFlow folder or set traceflow.root.';
      return this.push();
    }
    this.watch();
    await this.scaffold();
    await this.relayout();
    const pollMs = Math.max(1, Number(cfg('pollSeconds')) || 2) * 1000;
    for (const old of this.timers) clearInterval(old);    // a restart that overlapped this one
    this.timers = [
      setInterval(() => this.tick(), pollMs),
      setInterval(() => { this.pollGpu(); this.pollProcesses(); this.pollDocker(); }, GPU_MS),
      setInterval(() => this.pollFigures(), FIGURE_MS),
    ];
    this.pollGpu();
    this.pollProcesses();
    this.pollDocker();
    this.tick();
  }

  async restart() {
    this.stop();
    this.layout = null;
    this.error = '';
    await this.start();
  }

  stop() {
    for (const t of this.timers) clearInterval(t);
    for (const w of this.waits.values()) clearInterval(w);
    this.waits.clear();
    for (const w of this.watchers) w.dispose();
    this.timers = [];
    this.watchers = [];
    clearTimeout(this.relayoutTimer);
  }

  /** master.yaml (scaffold, then layout) and every project config (layout, and its open form). */
  watch() {
    for (const w of this.watchers) w.dispose();
    this.watchers = [];
    const base = vscode.Uri.file(this.root);
    const master = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(base, MASTER_GLOB));
    const declared = uri => {
      if (uri && this.openTrees.has(uri.fsPath)) this.sendTree(uri.fsPath);      // MASTER CONTROL's form, redrawn
      return this.scaffold().then(() => this.scheduleRelayout()).catch(e => this.report(e));
    };
    master.onDidChange(declared); master.onDidCreate(declared); master.onDidDelete(() => this.scheduleRelayout());
    const configs = vscode.workspace.createFileSystemWatcher(new vscode.RelativePattern(base, 'projects/*/config/*.yaml'));
    const changed = uri => { this.scheduleRelayout(); if (this.openTrees.has(uri.fsPath)) this.sendTree(uri.fsPath); };
    configs.onDidChange(changed); configs.onDidCreate(changed); configs.onDidDelete(changed);
    this.watchers.push(master, configs);
  }

  /** Copies the blueprint files every declared project lacks; says which, once. */
  async scaffold() {
    this.scaffoldError = '';
    try {
      const out = await runPy(this.root, this.python, 'layout', ['scaffold', '--root', this.root]);
      if (out.created && out.created.length) {
        say.info(`TraceFlow: created ${out.created.length} config file(s) from the blueprint: ${out.created.join(', ')}`);
      }
      // keys the blueprint has and a project's file lacked: written empty (the loader reads them as absent); the
      // ones with a value (a path, a constant) are left to Load Blueprint, which says what joins a hash
      const done = Object.entries(out.completed || {}).filter(([, r]) => (r.added || []).length || r.error);
      if (done.length) {
        say.info(`TraceFlow: ${done.map(([f, r]) => (r.error ? `${f}: ${r.error}` : `${f}: ${r.added.length} key(s) from the blueprint added (${r.added.slice(0, 6).join(', ')}${r.added.length > 6 ? ', …' : ''})`)).join('; ')}`);
      }
    } catch (e) {
      this.scaffoldError = `scaffold failed (${this.python}): ${e.message}`;
    }
  }

  /** The config file a form may edit: one of a declared project's config files. */
  /** The project a config file of the panel belongs to. */
  projectOfFile(file) {
    const f = path.resolve(String(file || ''));
    return ((this.layout && this.layout.projects) || []).find(p => (p.configs || []).some(c => path.resolve(c.path) === f)) || null;
  }

  /** The runs going on for a project (its make orchestrate / experiment / test / reproduce; a trainer naming no project counts for every one). */
  ongoing(p) {
    if (!p) return [];
    return (this.busy || []).filter(l => l.includes(`PROJECT=${p.name}`) || l.includes(`--project ${p.name}`) || !/PROJECT=|--project /.test(l));
  }

  /** The note a config file a run reads (params.yaml, the queue, the ENV overrides, test_config, auto_search,
   *  model_config) shows while a run of its project is ongoing: the run reads the copy frozen at its start
   *  (utils/config_freeze.py), so changes apply to the next run; null when no run is ongoing. */
  runNote(file) {
    if (!RUN_FILES.test(path.basename(String(file || '')))) return null;
    const p = this.projectOfFile(file);
    const runs = this.ongoing(p);
    return runs.length ? `A run of ${p.name} is ongoing (${runs[0].replace(/\s+/g, ' ').slice(0, 120)}): it reads the config ` +
      'frozen at its start, so changes apply to the next run.' : null;
  }

  configFile(file) {
    const f = path.resolve(String(file || ''));
    if (this.isMaster(f)) return f;
    for (const p of (this.layout && this.layout.projects) || []) {
      if (p.configs.some(c => path.resolve(c.path) === f)) return f;
    }
    return null;
  }

  // ── MASTER CONTROL: master.yaml, read by every run and by make (not frozen with a run's config) ──────

  isMaster(file) {
    return !!(this.layout && this.layout.master) && path.resolve(String(file || '')) === path.resolve(this.layout.master);
  }

  /** Why master.yaml may not be written now: any run on this machine reads it (every pass, and make). */
  masterLock() {
    const runs = this.busy || [];
    return runs.length ? `a run is ongoing (${runs[0].replace(/\s+/g, ' ').slice(0, 120)}): master.yaml is read by every run and by make, ` +
      'not frozen with the run\'s config, so it stays as it is until the run ends.' : null;
  }

  /** The panel's state of Master Control. */
  masterState() {
    if (!this.layout || !this.layout.master) return null;
    return { path: this.layout.master, lock: this.masterLock(), check: this.masterCheck || null };
  }

  /**
   * Before master.yaml is written: refused while a run is ongoing, refused for a read-only key
   * (menus.masterTier), asked for a key of the run hash. A refusal or a No: its note; null: write.
   */
  async masterGuard(edits) {
    const lock = this.masterLock();
    if (lock) return `refused: ${lock}`;
    const tiers = edits.map(e => ({ e, t: M.masterTier([...(e.path || []), ...(e.op === 'put' ? [e.key] : [])]) }));
    const ro = tiers.find(x => x.t.tier === 'readonly');      // a whole project entry (add, remove) is ['project'(, i)]: no tier
    if (ro) return `refused: ${(ro.e.path || []).join(' › ')}: ${ro.t.why}`;
    const hash = tiers.filter(x => x.t.tier === 'hash').map(x => (x.e.path || []).join(' › '));
    if (hash.length) {
      const ok = await say.warn('Change a key of the run hash in master.yaml?', { modal: true,
        detail: `${hash.join('\n')}\n\nThese keys join the model_hash of every new run: after the change no DONE run is skipped, every combination trains again as a new experiment.` }, 'Write');
      if (ok !== 'Write') return 'not written: the run hash keys were left as they are';
    }
    return null;
  }

  /** After master.yaml was written: every project loads as a run loads it, the timezone resolves (layout check-master). */
  async masterChecked(file, note) {
    let check;
    try { check = await runPy(this.root, this.python, 'layout', ['check-master', '--root', this.root]); } catch (e) { check = { errors: [e.message], projects: {} }; }
    const bad = Object.entries(check.projects || {}).filter(([, err]) => err).map(([n, err]) => `${n}: ${err}`)
      .concat(check.errors || [], check.timezone_error ? [`timezone: ${check.timezone_error}`] : []);
    this.masterCheck = check;
    const n = Object.keys(check.projects || {}).length;
    return this.sendTree(file, bad.length ? `${note}. Check failed: ${bad.join('; ')} (Reset or fix it before a run)`
      : `${note}. Checked: ${n} project${n === 1 ? '' : 's'} load as a run loads them; timezone ${check.timezone || 'of the system'}`);
  }

  /** MASTER CONTROL → Add a Project: its name, then blueprint/master_project.yaml under it; the master.yaml
   *  watcher then copies the blueprint's ten files to projects/<name>/config/ (scaffold). */
  async addProject(m = {}) {
    const file = this.layout && this.layout.master;
    if (!file) return undefined;
    const lock = this.masterLock();
    if (lock) return say.warn(`TraceFlow: no project added. ${lock}`, { modal: true });
    const tree = await runPy(this.root, this.python, 'yamledit', ['tree', file]);
    const list = ((tree.tree || {}).entries || []).find(e => e.key === 'project');
    const names = list && list.node.k === 'list' ? list.node.items.map(it => {
      const n = it.k === 'map' ? it.entries.find(e => e.key === 'name') : null;
      return n ? M.unquote(n.node.text) : '';
    }) : [];
    const name = String(m.name || '');                       // typed in the panel (its Add a Project row)
    const bad = M.askError('project', name, names);
    if (bad) return this.sendTree(file, `refused: no project added: ${bad}`);
    const entry = await runPy(this.root, this.python, 'layout', ['project-template', '--root', this.root, '--project', name.trim()]);
    const exists = fs.existsSync(path.join(this.root, 'projects', name.trim()));
    await this.edit({ file, sha: tree.sha256, op: list ? 'add' : 'put', path: list ? ['project'] : [], key: 'project',
      value: JSON.stringify(list ? entry : [entry]) }, false, `added project ${name.trim()}`);
    return say.info(`TraceFlow: project ${name.trim()} added to master.yaml; projects/${name.trim()}/config/ gets the blueprint's files` +
      `${exists ? ' it lacks (the folder was there: nothing in it is overwritten)' : ''}. Fill its params.yaml in Experiment Settings.`);
  }

  /** MASTER CONTROL → a project's ×: its entry leaves master.yaml (asked first); its folder and runs stay. */
  async removeProject(m) {
    const file = this.layout && this.layout.master;
    const name = String(m.name || '');
    if (!file || !Array.isArray(m.path) || m.path.length !== 2 || m.path[0] !== 'project' || !Number.isInteger(m.path[1])) return undefined;
    const lock = this.masterLock();
    if (lock) return say.warn(`TraceFlow: not removed. ${lock}`, { modal: true });
    const tree = await runPy(this.root, this.python, 'yamledit', ['tree', file]);
    const top = (tree.tree || {}).entries || [];
    const dflt = top.find(e => e.key === 'default_project');
    if (dflt && M.unquote(dflt.node.text) === name) {
      return say.warn(`TraceFlow: ${name} is default_project: choose another default_project first.`, { modal: true });
    }
    const ok = await say.warn(`Remove project ${name} from master.yaml?`, { modal: true,
      detail: `Its folder projects/${name}/ and every run in it stay on disk; the panel and make no longer see the project until it is declared again.` }, 'Remove');
    if (ok !== 'Remove') return undefined;
    return this.edit({ file, sha: tree.sha256, op: 'delete', path: m.path }, true, `removed project ${name} (its folder stays)`);
  }

  async sendTree(file, note = '') {
    try {
      const out = await runPy(this.root, this.python, 'yamledit', ['tree', file]);
      this.post(this.view, { type: 'tree', file, sha256: out.sha256, tree: out.tree, error: out.error || '', note });
      this.flushStages(file);                            // a planner's stage held for this form follows its tree
    } catch (e) {
      this.post(this.view, { type: 'tree', file, tree: null, error: e.message, note });
    }
  }

  /**
   * The changes a form staged, written when its OK is pressed (the panel never writes on input).
   * m: {file, sha (the text the form was drawn from), edits: [{op: set|put|add|remove, path, value, key}]}.
   * Order keeps every path valid: values first (set, put), then removals from the highest list index
   * down, then appends. Each write is checked against the hash the previous one returned; a file changed
   * on disk meanwhile refuses the rest (the file wins) and the form shows the file as it is.
   */
  async apply(m) {
    const file = this.configFile(m.file);
    if (!file) return;
    const edits = (Array.isArray(m.edits) ? m.edits : []).filter(e => APPLY_OPS.includes(e && e.op));
    if (!edits.length) return this.sendTree(file);
    const master = this.isMaster(file);
    if (master) {
      const no = await this.masterGuard(edits);
      if (no) return this.sendTree(file, no);
    }
    try { fs.accessSync(file, fs.constants.W_OK); } catch (e) {
      await bash(C.ownLine(true), this.root).catch(() => {});      // no terminal here: sudo -n, never a prompt
    }
    const ordered = orderEdits(edits);
    let sha = String(m.sha || '');
    let done = 0;
    for (const e of ordered) {
      const args = ['edit', file, '--sha', sha, '--op', e.op, '--path', JSON.stringify(e.path || [])];
      if (e.value != null) args.push(`--value=${String(e.value)}`);         // `--value=-1e-5`: never read as an option
      if (e.op === 'put') args.push(`--key=${String(e.key || '')}`);
      let out;
      try { out = await runPy(this.root, this.python, 'yamledit', args); } catch (err) { out = { error: err.message }; }
      const where = `${path.basename(file)} › ${(e.path || []).join(' › ')}`;
      if (out.stale) {
        return this.sendTree(file, `${path.basename(file)} changed on disk; reloaded from the file, ` +
          `${ordered.length - done} unsaved change(s) not written`);
      }
      if (out.error) return this.sendTree(file, `${where}: ${out.error} (${done} of ${ordered.length} change(s) written)`);
      sha = out.sha256;
      done += 1;
    }
    const saved = `saved: ${done} change(s) to ${path.basename(file)}`;
    await this.notifyPlanners(file);
    return master ? this.masterChecked(file, saved) : this.sendTree(file, saved);
  }

  /**
   * experiment_queue.yaml → Add a Queue: asks the new key (its runs' BATCH label; letters, digits, _ . -,
   * not a key there yet), then writes it with one line to fill in the form, at the end of the file:
   * make orchestrate ENV=<followed> PROJECT=<this> DATASET=<its first> MODEL_ID=<its first>.
   */
  async addQueue(m) {
    const file = this.configFile(m.file);
    const p = this.project(String(m.project || ''));
    if (!file || !p) return undefined;
    const tree = await runPy(this.root, this.python, 'yamledit', ['tree', file]).catch(() => ({}));
    const keys = tree.tree && tree.tree.k === 'map' ? tree.tree.entries.map(e => e.key)        // the queues there now
      : (Array.isArray(m.keys) ? m.keys : []).map(String);
    const name = String(m.name || '');                       // typed in the panel (its Add a Queue row)
    const bad = M.askError('queue', name, keys);
    if (bad) return this.sendTree(file, `refused: no queue added: ${bad}`);
    const line = ['make orchestrate', `ENV=${this.followed(p).env || 'dev'}`, `PROJECT=${p.name}`,
      ...((p.datasets || [])[0] ? [`DATASET=${p.datasets[0]}`] : []), ...((p.model_ids || [])[0] ? [`MODEL_ID=${p.model_ids[0]}`] : [])].join(' ');
    return this.apply({ file: m.file, sha: m.sha, edits: [{ op: 'put', path: [], key: name.trim(), value: '' }, { op: 'add', path: [name.trim()], value: line }] });
  }

  /**
   * GRAPH_CONFIG.YAML → jobs → Add a Job: asks its name, then graph_export.py new-job writes it at the next
   * id with no models yet (enabled: false, so make graph skips it until they are listed; array_epoch latest).
   */
  async addJob(m) {
    const p = this.project(String(m.project || ''));
    if (!p) return undefined;
    const name = String(m.name || '');                       // typed in the panel (its Add a Job row)
    const bad = M.askError('job', name, (p.graph_jobs || []).map(j => j.name));
    if (bad) return say.warn(`TraceFlow: no job added: ${bad}`);
    this.graphEditUntil = Date.now() + 15000;
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['new-job', '--project', p.name, '--title', name.trim()])
      .catch(e => ({ error: e.message }));
    this.graphEditUntil = Date.now() + 15000;
    if (out.error) return say.warn(`TraceFlow: no job added. ${out.error}`, { modal: true });
    say.info(`TraceFlow: graph job ${out.created} (${out.name}) added to graph_config.yaml: list its models, then set enabled: true.`);
    this.scheduleRelayout();
    return undefined;
  }

  /**
   * GRAPH_CONFIG.YAML → faceting → Add a Facet: asks its name, then graph_export.py new-facet writes it at
   * the next id with no panels yet (enabled: false until + add a panel and Find Figure have listed them).
   */
  async addFacet(m) {
    const p = this.project(String(m.project || ''));
    if (!p) return undefined;
    const name = String(m.name || '');                       // typed in the panel (its Add a Facet row)
    const bad = M.askError('facet', name, (p.graph_facets || []).map(f => f.name));
    if (bad) return say.warn(`TraceFlow: no facet added: ${bad}`);
    this.graphEditUntil = Date.now() + 15000;
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['new-facet', '--project', p.name, '--title', name.trim()])
      .catch(e => ({ error: e.message }));
    this.graphEditUntil = Date.now() + 15000;
    if (out.error) return say.warn(`TraceFlow: no facet added. ${out.error}`, { modal: true });
    say.info(`TraceFlow: facet ${out.created} (${out.name}) added to graph_config.yaml: + add a panel under its panels, Find Figure each file, then set enabled: true.`);
    this.scheduleRelayout();
    return undefined;
  }

  /**
   * params.yaml → model → Add a Model: asks its model_id (not one there yet) and its kind: a layer model
   * (CNN, MLP, TRANSFORMER, HYBRID: an empty layer list to fill with +) or a named model (a class of
   * core.model, PyTorch only); written as the last model of the list.
   */
  async addModel(m) {
    const p = this.project(String(m.project || ''));
    if (!p) return undefined;
    const ids = await this.modelIds(this.configFile(m.file) || m.file);   // the file now, not the form's copy (another window may have added one)
    const id = String(m.name || '').trim();                  // typed in the panel (its Add a Model row), with its kind
    const kinds = (Array.isArray(m.kinds) ? m.kinds : []).map(String);
    const named = (Array.isArray(m.named) ? m.named : []).map(String);
    const kind = String(m.kind || '');
    const bad = M.askError('model', id, ids) || (![...kinds, ...named].includes(kind) ? `${kind || 'no kind'} is not a model kind here` : '');
    if (bad) return this.sendTree(m.file, `refused: no model added: ${bad}`);
    const model = named.includes(kind) ? { model_id: id, architecture: kind, architecture_args: {} }
      : { model_id: id, architecture: kind, architecture_args: { strict_graph_args: true, layers: [] } };
    return this.apply({ file: m.file, sha: m.sha, edits: [{ op: 'add', path: ['model'], value: JSON.stringify(model) }] });
  }

  /**
   * params.yaml → model → Import a Model (utils/model_import.py): a model another study released, tested
   * through TraceFlow's own test without training. The file is read with h5py only (layout import-describe:
   * its format, Keras version, input, outputs and the leakage model they suggest, or why it is refused);
   * then its source, the data entry its traces come from (those that fit its input first), the leakage
   * model and, for a fitted scaler, its fit set. The import is added to the project's import_config.yaml
   * (created from the blueprint when missing; a file outside the project copied into its released/
   * folder: the run's container sees projects/ only), the file opens there for its other settings
   * (preprocessing, seed, attack), and Import now starts make import (tmux, after the busy checks).
   */
  async importModel(m) {
    const p = this.project(String(m.project || ''));
    if (!p) return undefined;
    const picked = await vscode.window.showOpenDialog({ canSelectMany: false, openLabel: 'Import',
      title: `Import a Model · ${p.name}: the released model file (Keras HDF5)`, defaultUri: vscode.Uri.file(p.path) });
    if (!picked || !picked.length) return undefined;
    const file = picked[0].fsPath;
    const info = await runPy(this.root, this.python, 'layout', ['import-describe', '--project', p.name, '--file', file])
      .catch(e => ({ error: e.message }));
    if (info.error) return say.warn(`TraceFlow: ${path.basename(file)} is not imported.\n\n${info.error}`, { modal: true });
    const about = `${path.basename(file)}: Keras ${info.keras_version || '?'} (${info.backend || '?'}), traces of ` +
      `${(info.input || [])[1]} samples, ${info.outputs} outputs (${info.output_activation}).`;
    const source = await vscode.window.showInputBox({ title: `Import a Model · ${p.name}`, prompt: `${about} Its source: the citation, or repository URL @ commit`,
      ignoreFocusOut: true, validateInput: v => (String(v || '').trim() ? '' : 'the source is required') });
    if (!source) return undefined;
    const fitting = (info.fitting || []).map(String);
    const others = (info.datasets || []).map(String).filter(d => !fitting.includes(d));
    const items = [...fitting.map(d => ({ label: d, description: `${(info.input || [])[1]} samples, as the model` })),
      ...others.map(d => ({ label: d, description: 'another trace length: refused by make import' }))];
    if (!items.length) return say.warn(`TraceFlow: params.yaml of ${p.name} has no data entry.`, { modal: true });
    const data = await vscode.window.showQuickPick(items, { title: `Import a Model · ${p.name}: the data entry its traces come from`, ignoreFocusOut: true });
    if (!data) return undefined;
    const models = ['ID', 'HW', 'MSB'];
    const suggested = info.leakage_suggestion && models.includes(info.leakage_suggestion) ? info.leakage_suggestion : null;
    const leakage = await vscode.window.showQuickPick(
      [...(suggested ? [suggested] : []), ...models.filter(x => x !== suggested)].map(x =>
        ({ label: x, description: x === suggested ? `${info.outputs} outputs suggest it` : '' })),
      { title: `Import a Model · ${p.name}: the labels it predicts (sca.leakage_model)`, ignoreFocusOut: true });
    if (!leakage) return undefined;
    const fit = await choose(`Fit set of a fitted scaler?`,
      'Only for a fitted trace_normalization (StandardScaler, MinMaxScaler): training fits it on the entry\'s ' +
      'training split, profiling on every profiling trace before any split (as Zaid\'s and Wouters\' scripts). ' +
      'Leave it empty for no fitted scaler; the preprocessing itself is set in import_config.yaml.',
      ['Empty', 'profiling', 'training']);
    if (!fit) return undefined;
    const out = await runPy(this.root, this.python, 'layout', ['import-add', '--project', p.name, '--file', file,
      '--source', String(source).trim(), '--data', data.label, '--leakage-model', leakage.label,
      ...(fit === 'Empty' ? [] : ['--fit-on', fit])]).catch(e => ({ error: e.message }));
    if (out.error) return say.warn(`TraceFlow: no import added. ${out.error}`, { modal: true });
    await this.openAt(out.config, `file: ${out.file}`);
    this.scheduleRelayout();
    const now = await say.info(`TraceFlow: ${out.file} added to ${path.basename(out.config)}` +
      `${out.copied ? ` (copied into the project: ${out.copied})` : ''}. Set its preprocessing (trace_normalization), ` +
      'seed and attack there if they are not the defaults, then import it: make import runs every model listed (DONE ones are skipped).',
      'Import now');
    if (now !== 'Import now') return undefined;
    const env = await this.envFor(this.followed(p), 'make import');
    if (!env) return undefined;
    return this.startLong(p, 'import', { ENV: env, PROJECT: p.name }, null,
      'Every model of import_config.yaml not DONE yet: its weights checked against the released model on every test trace, ' +
      'then the standard test. Nothing trains.');
  }

  /** params.yaml → a model → Check the model: model_check.py builds it as a run would; each layer's shape. */
  async checkModel(project, modelId) {
    const p = this.project(project);
    if (!p || !modelId) return undefined;
    this.post(this.view, { type: 'model-check', project: p.name, model_id: modelId, data: { checking: true } });
    // model_check finds the data entry the model fits (its trace length): nothing to choose, nothing read from the queues
    const data = await runModule(this.root, this.python, 'services.visualization.vscode_monitor.model_check', ['--project', p.name, '--model-id', modelId])
      .catch(e => ({ error: e.message }));
    return this.post(this.view, { type: 'model-check', project: p.name, model_id: modelId, data });
  }

  // ── Plan a Model: the planner page (webview/planner.js) in an editor tab, one per project ─────────────
  /** A config file's form tree as yamledit reads it now ({sha256, tree}). */
  treeOf(file) { return runPy(this.root, this.python, 'yamledit', ['tree', file]); }
  /** The model_ids of a params.yaml as the file holds them now. */
  async modelIds(file) {
    return idsOfTree((await this.treeOf(file).catch(() => ({}))).tree);
  }
  /** The project's framework, trace input and the models of its params.yaml (model_check describe). */
  describeProject(project, file, modelId = null) {
    return runModule(this.root, this.python, MODEL_CHECK, ['--project', project, '--describe', file, ...(modelId ? ['--model-id', modelId] : [])]);
  }
  /** After a write to a params.yaml: its planner pages get its models as they are now (they re-base a
   *  visualised model, or reload it when it changed otherwise). */
  async notifyPlanners(file, also = []) {
    const open = [...new Set([...Object.values(this.planners || {}).filter(pl => pl.file === file), ...also])];
    if (!open.length) return;
    const out = await runPy(this.root, this.python, 'yamledit', ['value', file, '--path', '["model"]']).catch(() => ({}));
    const models = Array.isArray(out.value) ? out.value.filter(m => m && typeof m === 'object') : [];
    for (const pl of open) this.post(pl.panel, { type: 'models', models, ids: models.map(m => String(m.model_id)) });
  }
  /** [Plan a Model] (no model: a clean skeleton) or [Visualise] (that model) of a project's params.yaml. */
  async openPlanner(project, file, modelId = null) {
    const p = this.project(String(project || ''));
    const params = this.configFile(file);
    if (!p || !params) return undefined;
    let pl = (this.planners || {})[p.name];
    if (!pl) {
      // retainContextWhenHidden: the design lives in the page; another tab in front must not wipe it
      pl = this.adoptPlanner(vscode.window.createWebviewPanel(PLANNER_VIEW, `Plan a Model · ${p.name}`, vscode.ViewColumn.Beside,
        { ...this.webviewOptions(), retainContextWhenHidden: true }), p.name);
    }
    pl.panel.reveal(vscode.ViewColumn.Beside);
    return this.initPlanner(pl, params, modelId, false);
  }
  /** A planner tab, opened now or brought back by VS Code after a reload: its page, its messages, its place. */
  adoptPlanner(panel, project) {
    this.planners = this.planners || {};
    const pl = { project, panel, seq: 0 };
    panel.webview.options = this.webviewOptions();
    panel.webview.html = this.html(panel.webview, 'planner', 'planner');
    panel.webview.onDidReceiveMessage(m => this.onPlannerMessage(pl, m));
    panel.onDidDispose(() => { if (this.planners[project] === pl) delete this.planners[project]; });
    this.planners[project] = pl;
    return pl;
  }
  /** The page's start: the project's models now, and (restore) "bring back the design you kept" after a reload.
   *  Sent once the page has said it is ready (a message to a page still loading is lost), with the layer forms
   *  (waited for: after a reload the tab comes back before choices.py has answered). */
  async initPlanner(pl, params, modelId, restore) {
    pl.file = params;
    pl.modelId = modelId;
    pl.panel.title = modelId ? `Visualise ${modelId} · ${pl.project}` : `Plan a Model · ${pl.project}`;
    if (!pl.ready) { pl.waiting = { modelId, restore }; return undefined; }
    // Visualise: describe names the model, so its start is the data entry the model fits (model_check)
    const [info, menus] = await Promise.all([this.describeProject(pl.project, params, modelId).catch(e => ({ error: e.message, models: [] })), this.menusLoaded()]);
    const model = modelId ? (info.models || []).find(m => m.model_id === modelId) : null;
    if (modelId && !model && !restore) return say.warn(`TraceFlow: ${modelId} is not in ${path.basename(params)}.`);
    const forms = (menus.params || {}).models || null;
    const g = menus.graph || {};                         // graph_manager's forms: the Download's export and text_rules
    return this.post(pl.panel, { type: 'init', project: pl.project, file: params, restore, framework: info.framework || 'pytorch', forms,
      graph: { export: (g.export || {}).job || null, textRule: g.textRule || null, diagram: g.diagram || null },
      models: info.models || [], ids: (info.models || []).map(m => String(m.model_id)), input: info.input || null, entries: info.entries || [],
      dataset: info.dataset || null, error: info.error || '', start: model ? { model } : {} });   // input null: no data entry (the page asks for a length)
  }
  /** VS Code's serializer: a planner tab open before the reload comes back with its design (the page keeps it
   *  in its tab state); a tab whose project or params.yaml is gone, or whose project has a tab already, closes. */
  async restorePlanner(panel, kept) {
    const st = kept || {};
    const p = this.project(String(st.project || ''));
    const params = this.configFile(st.file);
    if (!p || !params || (this.planners || {})[p.name]) { panel.dispose(); return undefined; }
    return this.initPlanner(this.adoptPlanner(panel, p.name), params, st.modelId ? String(st.modelId) : null, true);
  }
  /** A planner's stage for the panel's form of `file`: sent once the panel shows that form (its tree there),
   *  held until then (after a reload the panel comes back after the planner; the last stage wins). */
  postStage(file, msg) {
    this.waitingStages = this.waitingStages || {};
    if (this.view && this.openTrees.has(file)) { delete this.waitingStages[file]; return this.post(this.view, msg); }
    this.waitingStages[file] = msg;
    return undefined;
  }
  flushStages(file) {
    const msg = (this.waitingStages || {})[file];
    if (!msg || !this.view || !this.openTrees.has(file)) return;
    delete this.waitingStages[file];
    this.post(this.view, msg);
  }
  onPlannerMessage(pl, m) {
    if (!m || typeof m !== 'object') return undefined;
    const fail = e => this.post(pl.panel, { type: 'refused', message: `refused: ${e.message || e}` });
    if (m.type === 'ready') {                            // the page loaded (first, or again): its start now
      pl.ready = true;
      const w = pl.waiting || { modelId: pl.modelId, restore: true };
      pl.waiting = null;
      return this.initPlanner(pl, pl.file, w.modelId, w.restore).catch(e => this.report(e));
    }
    if (m.type === 'check') return this.plannerCheck(pl, m).catch(fail);
    if (m.type === 'sources') return this.plannerSources(pl, m).catch(fail);
    if (m.type === 'trace') return this.plannerTrace(pl, m).catch(fail);
    if (m.type === 'recovery') return this.plannerRecovery(pl, m).catch(fail);
    if (m.type === 'preview') return this.plannerPreview(pl, m).catch(fail);
    if (m.type === 'add-model') return this.plannerAdd(pl, m.model, m.from).catch(fail);
    if (m.type === 'stage') return this.plannerStage(pl, m).catch(fail);
    if (m.type === 'apply') return this.plannerApply(pl, m).catch(fail);
    if (m.type === 'download') return this.plannerDownload(pl, m).catch(e => this.report(e));
    if (m.type === 'close') pl.panel.dispose();
    return undefined;
  }
  /** Download: the page's model diagram drawn and saved by model_diagram.py (graph_manager's export) into
   *  paths.model_preview of the env (a trained run's own, else the one followed), then copied into model_out of
   *  each DONE run of the model with these layers; the page is told what was written. */
  async plannerDownload(pl, m) {
    const p = this.project(pl.project);
    const env = String(m.env || (p ? this.followed(p).env : '') || 'dev');
    const out = await runModule(this.root, this.python, MODEL_DIAGRAM, ['--project', pl.project], JSON.stringify({ ...m, env }))
      .catch(e => ({ error: e.message }));
    const rel = f => path.relative(this.root, f);
    const message = out.error ? `refused: not downloaded: ${out.error}` : [
      `saved ${(out.written || []).map(f => path.basename(f)).join(', ')} in ${rel(path.dirname((out.written || [''])[0]))}`,
      (out.copied || []).length ? `copied into the model folder of ${new Set(out.copied.map(f => path.dirname(f))).size} run(s)` : '',
      (out.skipped || []).length ? `not into ${out.skipped.join(', ')} (other layers)` : '',
      (out.denied || []).length ? `refused: not into ${out.denied.join(', ')} (owned by root: services/doc/03_file_permissions.md §3.1)` : '',
    ].filter(Boolean).join('; ');
    return this.post(pl.panel, { type: out.error || (out.denied || []).length ? 'refused' : 'saved', message });
  }
  /** TraceFlow's model check on the page's layers: a new request stops the one still running (each is a
   *  python process building the model); only the latest answer is shown. */
  async plannerCheck(pl, m) {
    const seq = ++pl.seq;
    if (pl.child) { try { pl.child.kill(); } catch (e) { /* ended */ } }
    const result = await runModule(this.root, this.python, MODEL_CHECK,
      ['--project', pl.project, '--layers-json', '-', '--architecture', String(m.architecture || 'CNN'), ...(m.dataset ? ['--dataset', String(m.dataset)] : []),
       ...(!m.dataset && m.input ? ['--input', String(parseInt(m.input, 10) || 0)] : [])],   // no data entry: the length typed on the page
      JSON.stringify(m.layers || []),
      child => { pl.child = child; })
      .catch(e => ({ error: e.message, layers: [] }));
    if (seq === pl.seq) { pl.child = null; this.post(pl.panel, { type: 'check', result }); }
  }
  /** The trace lines' sources (trace_view): the project's data entries, and (Visualise) the DONE runs of the model. */
  async plannerSources(pl, m) {
    const fail = e => ({ error: e.message });
    const [ent, runs] = await Promise.all([
      runModule(this.root, this.python, TRACE_VIEW, ['entries', '--project', pl.project]).catch(fail),
      m.model_id ? runModule(this.root, this.python, TRACE_VIEW, ['runs', '--project', pl.project, '--model-id', String(m.model_id)]).catch(fail) : { runs: [] }]);
    this.post(pl.panel, { type: 'sources', entries: ent.entries || [], runs: runs.runs || [], error: ent.error || runs.error || '' });
  }
  /** One trace through the design (untrained weights) or a trained run (trace_view): like the check, a new
   *  request stops the one still running and only the latest answer is shown. */
  async plannerTrace(pl, m) {
    const seq = pl.traceSeq = (pl.traceSeq || 0) + 1;
    if (pl.traceChild) { try { pl.traceChild.kill(); } catch (e) { /* ended */ } }
    const trace = ['--trace', String(Math.max(0, parseInt(m.trace, 10) || 0)), ...(m.to != null ? ['--to', String(parseInt(m.to, 10) || 0)] : [])];
    const args = m.run ? ['run', '--project', pl.project, '--run', String(m.run), ...trace]
      : ['design', '--project', pl.project, '--architecture', String(m.architecture || 'CNN'), ...(m.dataset ? ['--dataset', String(m.dataset)] : []), ...trace];
    const result = await runModule(this.root, this.python, TRACE_VIEW, args, m.run ? null : JSON.stringify(m.layers || []),
      child => { pl.traceChild = child; })
      .catch(e => ({ error: e.message, layers: [] }));
    if (seq === pl.traceSeq) { pl.traceChild = null; this.post(pl.panel, { type: 'trace', key: m.key, result }); }
  }
  /** Show Key Recovery: what a trained run recorded (trace_view recovery: the registries, nothing recomputed). */
  async plannerRecovery(pl, m) {
    const result = await runModule(this.root, this.python, TRACE_VIEW, ['recovery', '--project', pl.project, '--run', String(m.run || '')])
      .catch(e => ({ error: e.message }));
    this.post(pl.panel, { type: 'recovery', run: m.run, result });
  }
  // ── the status page (services/visualization/server/status.py): read-only, for phones and shared links ─
  /** Is the status page serving: its tmux pane's shell has a child process (as pollProcesses: a `bash -lc`
   *  pane reports bash while make runs; once the server ends, the pane's `exec bash` has none). */
  async statusRunning() {
    const pid = (await output('tmux', ['display-message', '-p', '-t', STATUS_SESSION, '#{pane_pid}'], this.root)).trim();
    return /^\d+$/.test(pid) && (await output('pgrep', ['-P', pid], this.root)).trim().length > 0;
  }
  /** Its settings (master.yaml status_share: enabled, port, host or this machine's own address; read-only here),
   *  links and QR codes. */
  async statusState() {
    const [conf, list, running] = await Promise.all([
      runModule(this.root, this.python, STATUS, ['settings']).catch(e => ({ enabled: false, why: e.message })),
      runModule(this.root, this.python, STATUS, ['share', 'list']).catch(e => ({ error: e.message, shares: [] })),
      this.statusRunning()]);
    const host = conf.host || '';
    const port = conf.port || null;
    this.qrOf = this.qrOf || new Map();
    const shares = await Promise.all((list.shares || []).map(async s => {
      const link = `http://${host}:${port}/?t=${s.token}`;
      if (!this.qrOf.has(link)) {
        this.qrOf.set(link, (await runModule(this.root, this.python, STATUS, ['qr', link]).catch(() => ({}))).svg || '');
      }
      return { name: s.name, created: s.created, link, qr: this.qrOf.get(link) };
    }));
    return { running, enabled: !!conf.enabled, why: conf.why || '', host, port, auto: !!conf.auto, shares, error: list.error || '',
      session: STATUS_SESSION };
  }
  /** The title line's status page button: the panel (brought into view first) opens or closes its settings. */
  async toggleShare() {
    if (!this.view) await vscode.commands.executeCommand('traceflow.panel.focus');
    this.post(this.view, { type: 'share-toggle' });
  }
  async statusSend(note = '') { this.post(this.view, { type: 'status-page', note, ...(await this.statusState()) }); }
  /** Start: make status_share in its own tmux session (it keeps serving when VS Code closes), on master.yaml's
   *  status_share port and host (or this machine's address); refused while it is off there. */
  async statusStart() {
    if (await this.statusRunning()) return this.statusSend();
    const st = await this.statusState();
    if (!st.enabled) return this.statusSend(`refused: ${st.why}`);
    const log = path.join(this.root, 'logs', 'status.txt');
    // the links the panel reads are the ones the page checks (tmux keeps its own environment)
    const shares = process.env.TRACEFLOW_STATUS_SHARES ? `TRACEFLOW_STATUS_SHARES=${C.quote(process.env.TRACEFLOW_STATUS_SHARES)} ` : '';
    const line = C.envLine(`mkdir -p ${C.word(path.dirname(log))} && ${shares}${C.makeLine('status_share', { ENV: 'dev' })} 2>&1 | tee -a ${C.word(log)}`);
    await bash(`tmux kill-session -t ${C.word(STATUS_SESSION)} 2>/dev/null; tmux new-session -d -s ${C.word(STATUS_SESSION)} -c ${C.quote(this.root)} ` +
      `${C.quote(`bash -lc ${C.quote(`${line}; echo "[traceflow] the status page stopped"; exec bash`)}`)}`, this.root);
    await new Promise(r => setTimeout(r, 1500));            // the server binds its address, or says why not (logs/status.txt)
    return this.statusSend(await this.statusRunning() ? '' : `it did not start: see logs/status.txt (tmux attach -t ${STATUS_SESSION})`);
  }
  /** A person's link opened here: on this machine's loopback (the page listens there too), which VS Code forwards to
   *  this desktop over the session (asExternalUri), with the link's token: no open port needed for the owner. */
  async statusVisit(link) {
    const u = new URL(link);
    const local = vscode.Uri.parse(`http://localhost:${u.port}/?t=${encodeURIComponent(u.searchParams.get('t') || '')}`);
    return vscode.env.openExternal(await vscode.env.asExternalUri(local));
  }
  async statusStop() {
    await bash(`tmux kill-session -t ${C.word(STATUS_SESSION)} 2>/dev/null || true`, this.root);
    return this.statusSend();
  }
  /** Add a person's link, or withdraw one (asks first: the link stops working at once). */
  async statusShare(action, name) {
    if (action === 'remove') {
      const ok = await choose(`Withdraw ${name}'s link?`, 'The link and its QR code stop working at once; a new link for them is a new token.', ['Withdraw']);
      if (ok !== 'Withdraw') return undefined;
    }
    const out = await runModule(this.root, this.python, STATUS, ['share', action, name]).catch(e => ({ error: e.message }));
    return this.statusSend(out.error ? `refused: ${out.error}` : '');
  }

  /** Preview Code: the lines Add the Model would write (yamledit preview, nothing written). */
  /** Preview Code. Plan a Model: the YAML Add the Model would write. Visualise (m.model_id, m.arrange): the
   *  model's own entry of params.yaml as OK would leave it (its comments and style kept), its changed lines marked. */
  async plannerPreview(pl, m) {
    if (!m.model_id) {
      const out = await runPy(this.root, this.python, 'yamledit', ['preview', pl.file, '--op', 'add', '--path', '["model"]', `--value=${JSON.stringify(m.model)}`]);
      return this.post(pl.panel, { type: 'preview', added: out.error ? [out.error] : out.added || [] });
    }
    const t = await this.treeOf(pl.file);
    const index = Panel.modelIndex(t.tree, m.model_id);
    if (index < 0) return this.post(pl.panel, { type: 'preview', added: [`${m.model_id} is no longer in ${path.basename(pl.file)}`] });
    const out = await runPy(this.root, this.python, 'yamledit', ['preview', pl.file, '--op', 'arrange',
      '--path', JSON.stringify(['model', index, 'architecture_args', 'layers']), `--value=${JSON.stringify(m.arrange || [])}`]);
    if (out.error) return this.post(pl.panel, { type: 'preview', added: [out.error] });
    const block = Panel.modelBlock(out.text || '', m.model_id);
    const fresh = {};
    (out.added || []).forEach(l => { fresh[l] = (fresh[l] || 0) + 1; });
    const marked = block.map((l, i) => (fresh[l] ? (fresh[l] -= 1, i) : -1)).filter(i => i >= 0);
    return this.post(pl.panel, { type: 'preview', added: block, marked });
  }
  /** A model's entry in params.yaml text: the comment lines right above it, its "- model_id:" line and every line
   *  under it, up to the next item or key at its level (trailing blank and comment lines left out). */
  static modelBlock(text, modelId) {
    const lines = String(text).split('\n');
    const id = String(modelId).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const at = lines.findIndex(l => new RegExp(`^\\s*- model_id:\\s*["']?${id}["']?\\s*(#.*)?$`).test(l));
    if (at < 0) return [];
    const indent = lines[at].indexOf('-');
    let end = at + 1;
    while (end < lines.length && (!lines[end].trim() || lines[end].trim().startsWith('#') || lines[end].search(/\S/) > indent)) end += 1;
    while (end > at + 1 && (!lines[end - 1].trim() || lines[end - 1].trim().startsWith('#'))) end -= 1;
    let start = at;
    while (start > 0 && lines[start - 1].trim().startsWith('#') && lines[start - 1].search(/\S/) === indent) start -= 1;
    return lines.slice(start, end);
  }
  /** Add the Model: a new model under model:, only with a model_id unique in the file as it is now, and never
   *  over the panel's staged edits of that file (the write would drop them). From Visualise's Configuration Change
   *  (from: the model the page opened, its base layers and arrange), the panel's staged copy of those changes is
   *  not counted: it is dropped before the write (the opened model stays as the file has it) and staged again
   *  when the write fails. */
  async plannerAdd(pl, model, from) {
    const refuse = (message, ids) => this.post(pl.panel, { type: 'refused', message: `refused: nothing written: ${message}`, ...(ids ? { ids } : {}) });
    const staged = ((this.staged || {})[pl.file] || 0) - (from ? (this.stagedArranged || {})[pl.file] || 0 : 0);
    if (staged > 0) return refuse(`the panel's ${path.basename(pl.file)} form has ${staged} ${from ? 'other ' : ''}staged change(s): OK or Cancel them first`);
    const t = await this.treeOf(pl.file);
    if (!t.tree) return refuse(`${path.basename(pl.file)} could not be read`);
    const ids = idsOfTree(t.tree);                          // the very text the write below is checked against
    const id = PC.idState(model && model.model_id, ids, M.askError);
    if (!id.ok) return refuse(id.error, ids);
    const index = from ? Panel.modelIndex(t.tree, from.model_id) : -1;
    const restage = arrange => index >= 0 && this.postStage(pl.file, { type: 'stage', file: pl.file, model_id: String(from.model_id),
      edit: { op: 'arrange', path: ['model', index, 'architecture_args', 'layers'], value: JSON.stringify(arrange) } });
    restage((from && from.base || []).map((_, i) => ({ from: i })));   // an arrange that keeps every layer: dropped there
    const out = await runPy(this.root, this.python, 'yamledit',
      ['edit', pl.file, '--sha', String(t.sha256 || ''), '--op', 'add', '--path', '["model"]', `--value=${JSON.stringify(model)}`]);
    if (out.stale || out.error) restage(from && from.arrange || []);
    if (out.stale) return refuse(`${path.basename(pl.file)} changed on disk; the model_ids are read again`, await this.modelIds(pl.file));
    if (out.error) return refuse(out.error, ids);
    this.post(pl.panel, { type: 'saved', message: `${model.model_id} added to ${path.basename(pl.file)}`, ids: [...ids, String(model.model_id)] });
    await this.notifyPlanners(pl.file);
    return this.sendTree(pl.file);
  }
  /** The index of a model under model: in a params.yaml form tree (yamledit tree), by its model_id; -1 when it is not there. */
  static modelIndex(tree, modelId) {
    const list = tree && tree.k === 'map' ? (tree.entries.find(e => e.key === 'model') || {}).node : null;
    return list && list.k === 'list'
      ? list.items.findIndex(it => it.k === 'map' && String(((it.entries.find(e => e.key === 'model_id') || {}).node || {}).text) === String(modelId)) : -1;
  }
  /** Visualise → OK: the page's changes of the model written to params.yaml as one arrange edit of its layers,
   *  only over the layers the page opened, and never while the panel's form of the file holds other staged changes
   *  (the write would drop them). The panel's staged copy of these changes is dropped first; the page learns the
   *  file's models after the write. */
  async plannerApply(pl, m) {
    const refuse = message => this.post(pl.panel, { type: 'refused', message: `refused: nothing written: ${message}` });
    const name = path.basename(pl.file);
    const others = ((this.staged || {})[pl.file] || 0) - ((this.stagedArranged || {})[pl.file] || 0);
    if (others > 0) return refuse(`the panel's ${name} form has ${others} other staged change(s): OK or Cancel them there first`);
    const t = await this.treeOf(pl.file);
    const index = Panel.modelIndex(t.tree, m.model_id);
    if (index < 0) return refuse(`${m.model_id} is no longer in ${name}`);
    const at = ['model', index, 'architecture_args', 'layers'];
    const now = await runPy(this.root, this.python, 'yamledit', ['value', pl.file, '--path', JSON.stringify(at)]).catch(() => ({}));
    if (!PC.same(now.value == null ? [] : now.value, m.base || [])) {
      refuse(`${m.model_id} changed in ${name} since the page opened it; it shows the model as it is now`);
      return this.notifyPlanners(pl.file, [pl]);
    }
    // the panel's staged copy goes first (an arrange that keeps every layer is dropped there)
    this.postStage(pl.file, { type: 'stage', file: pl.file, model_id: String(m.model_id),
      edit: { op: 'arrange', path: at, value: JSON.stringify((m.base || []).map((_, i) => ({ from: i }))) } });
    const out = await runPy(this.root, this.python, 'yamledit',
      ['edit', pl.file, '--sha', String(t.sha256 || ''), '--op', 'arrange', '--path', JSON.stringify(at), `--value=${JSON.stringify(m.arrange || [])}`]);
    if (out.stale) return refuse(`${name} changed on disk; try again`);
    if (out.error) return refuse(out.error);
    this.post(pl.panel, { type: 'saved', message: `${m.model_id}'s layers written to ${name}` });
    await this.notifyPlanners(pl.file, [pl]);
    return this.sendTree(pl.file);
  }
  /** Visualise: the page's changes of a model, staged in the panel's params.yaml form as one arrange edit of
   *  that model's layers (found by model_id in the file now; refused when it is gone). */
  async plannerStage(pl, m) {
    const t = await this.treeOf(pl.file);
    const index = Panel.modelIndex(t.tree, m.model_id);
    if (index < 0) return this.post(pl.panel, { type: 'refused', message: `refused: nothing staged: ${m.model_id} is no longer in params.yaml` });
    // the arrange indexes the layers the page opened: refused when the file's are others by now
    const now = await runPy(this.root, this.python, 'yamledit', ['value', pl.file, '--path', JSON.stringify(['model', index, 'architecture_args', 'layers'])]).catch(() => ({}));
    if (!PC.same(now.value == null ? [] : now.value, m.base || [])) {
      this.post(pl.panel, { type: 'refused', message: `refused: nothing staged: ${m.model_id} changed in params.yaml since the page opened it; it shows the model as it is now` });
      return this.notifyPlanners(pl.file, [pl]);
    }
    return this.postStage(pl.file, { type: 'stage', file: pl.file, model_id: String(m.model_id),
      edit: { op: 'arrange', path: ['model', index, 'architecture_args', 'layers'], value: JSON.stringify(m.arrange || []) } });
  }

  /**
   * params.yaml → a model's or a data entry's × : asks first, naming the queue lines that still use it
   * (MODEL_ID= / DATASET=: they would fail until changed); an ongoing run keeps its frozen copy.
   */
  async deleteEntry(m) {
    const file = this.configFile(m.file);
    const p = this.projectOfFile(file);
    if (!file || !p) return undefined;
    const model = m.kind === 'model';
    const name = String(m.name || '');
    const base = path.basename(file);
    // a whole queue of experiment_queue.yaml (its key), or a graph job / facet without an id (a template's entry)
    if (['queue', 'job', 'facet'].includes(m.kind)) {
      const queue = m.kind === 'queue';
      const okFile = queue ? /^experiment_queue\.ya?ml$/.test(base) : /^graph_config\.ya?ml$/.test(base);
      const okPath = Array.isArray(m.path) && (queue ? m.path.length === 1 && typeof m.path[0] === 'string'
        : m.path.length === 2 && m.path[0] === (m.kind === 'job' ? 'jobs' : 'faceting') && Number.isInteger(m.path[1]));
      if (!okFile || !okPath) return undefined;
      const what = queue ? `queue ${m.path[0]} and its lines` : `${m.kind} ${name ? `(${name}) ` : ''}#${m.path[1]}`;
      const run = this.runNote(file);
      const detail = queue ? 'Its runs (if any were made) keep their batch label and their records.' : 'An entry without an id: nothing drawn under it is moved.';
      const ok = await say.warn(`Delete ${what} from ${base}?`, { modal: true, detail: run ? `${detail}\n${run}` : detail }, 'Delete');
      if (ok !== 'Delete') return undefined;
      return this.edit({ file: m.file, sha: m.sha, op: 'delete', path: m.path }, true);
    }
    // only a whole model / data entry of params.yaml: the webview's message names nothing else
    if (!/^params[^/]*\.ya?ml$/.test(base) || !Array.isArray(m.path) || m.path.length !== 2 ||
        m.path[0] !== (model ? 'model' : 'data') || !Number.isInteger(m.path[1]) || !['model', 'data'].includes(m.kind)) return undefined;
    const key = model ? 'MODEL_ID' : 'DATASET';
    const users = (p.queues || []).map(q => [q.key, (q.commands || []).filter(c => c && c[key] === name).length]).filter(([, n]) => n);
    const what = model ? `model ${name}` : `data entry ${name}`;
    const detail = users.length
      ? `${key}=${name} is named by queue ${users.map(([k, n]) => `${k} (${n} line${n === 1 ? '' : 's'})`).join(', ')}: those commands fail until they name another.`
      : `No queue line names ${key}=${name}. Runs made with it keep their snapshots.`;
    const run = this.runNote(file);
    const ok = await say.warn(`Delete ${what} from params.yaml?`, { modal: true, detail: run ? `${detail}\n${run}` : detail }, 'Delete');
    if (ok !== 'Delete') return undefined;
    return this.edit({ file: m.file, sha: m.sha, op: 'delete', path: m.path }, true);
  }

  /** One change written at once; delete only after deleteEntry / removeProject asked (asked = true), never
   *  from a message. said: the note once written (default "saved: <file> › <path>"). */
  async edit(m, asked = false, said = '') {
    const file = this.configFile(m.file);
    if (!file) return;
    const master = this.isMaster(file);
    if (master) {
      const no = await this.masterGuard([m]);
      if (no) return this.sendTree(file, no);
    }
    try { fs.accessSync(file, fs.constants.W_OK); } catch (e) {
      await bash(C.ownLine(true), this.root).catch(() => {});      // no terminal here: sudo -n, never a prompt
    }
    const where = `${path.basename(file)} › ${(m.path || []).join(' › ')}`;
    let value = m.value == null ? undefined : String(m.value);
    if (!['set', 'put', 'add', 'remove', 'move'].concat(asked ? ['delete'] : []).includes(m.op)) {
      return this.sendTree(file, `refused: the panel does not ${m.op} entries; edit the file by hand`);
    }
    const args = ['edit', file, '--sha', String(m.sha || ''), '--op', String(m.op), '--path', JSON.stringify(m.path || [])];
    if (value !== undefined) args.push(`--value=${value}`);
    if (m.op === 'put') args.push(`--key=${String(m.key || '')}`);           // `--value=-1e-5`: never read as an option
    let out;
    try { out = await runPy(this.root, this.python, 'yamledit', args); } catch (e) { out = { error: e.message }; }
    if (out.stale) return this.sendTree(file, `${path.basename(file)} changed on disk; reloaded from the file, your change was not written`);
    if (out.error) return this.sendTree(file, out.error);
    await this.notifyPlanners(file);
    const saved = said || `saved: ${where}`;
    return master ? this.masterChecked(file, saved) : this.sendTree(file, saved);
  }

  scheduleRelayout() {
    clearTimeout(this.relayoutTimer);
    this.relayoutTimer = setTimeout(() => this.relayout().catch(e => this.report(e)), RELAYOUT_DEBOUNCE_MS);
  }

  async relayout() {
    const before = this.graphIds();
    try {
      this.layout = await runPy(this.root, this.python, 'layout', ['export', '--root', this.root]);
      this.error = '';
    } catch (e) {
      this.layout = null;
      this.error = `layout failed (${this.python}): ${e.message}`;
    }
    this.pollFigures();
    runPy(this.root, this.python, 'choices', ['refs', '--root', this.root])
      .then(out => { this.menuRefs = out; this.tick(); }).catch(() => {});
    this.offerGapClose(before).catch(e => this.report(e));
  }

  /**
   * The config forms' menus (choices.py): the option lists read from the code, once per session in
   * the background (they import torch), and the project names (runs, datasets, model ids, queues),
   * read again on every layout. {rules: {file: [rule]}, graph (the graph_config.yaml forms), refs: {project: {...}}, errors}.
   */
  menus() {
    if (!this.menuRules && !this.menuLoading && this.root && this.python) {
      this.menuLoading = true;
      this.menuPromise = runPy(this.root, this.python, 'choices', ['static'])
        .then(out => { this.menuRules = out; })
        .catch(e => { this.menuRules = { rules: {}, errors: [e.message] }; })
        .finally(() => { this.menuLoading = false; this.tick(); });
    }
    // blueprint too (choices.blueprint_forms): the keys a file lacks are drawn from it, and Load Blueprint
    // reads it; without it the panel shows a key only once the file holds it (2026-10-24: profiling_split's arguments
    // appeared only after OK)
    return { rules: (this.menuRules || {}).rules || {}, graph: (this.menuRules || {}).graph || null, params: (this.menuRules || {}).params || null,
      blueprint: (this.menuRules || {}).blueprint || null, refs: this.menuRefs || {}, errors: (this.menuRules || {}).errors || [] };
  }

  /** menus() once choices.py static has answered (a page that needs the forms waits for them). */
  async menusLoaded() {
    this.menus();
    if (this.menuPromise) await this.menuPromise.catch(() => {});
    return this.menus();
  }

  /** project -> the graph job ids of its graph_config.yaml, from the layout (facet ids under `facets:<project>`). */
  graphIds() {
    const out = {};
    for (const p of (this.layout && this.layout.projects) || []) {
      out[p.name] = (p.graph_jobs || []).map(j => j.id);
      out[`facets:${p.name}`] = (p.graph_facets || []).map(f => f.id);
    }
    return out;
  }

  /**
   * A job id gone from graph_config.yaml (deleted in the editor) while later ids remain: ask once
   * whether the later jobs move down (graph_export.py close-gap: ids, figure folders, facet
   * addresses, header lines; the deleted job's figures go to graphs/jobs/_deleted/). Not asked
   * after the panel's own export or shift, nor for a project seen for the first time.
   */
  async offerGapClose(before) {
    if (this.graphEditUntil && Date.now() < this.graphEditUntil) return;
    for (const p of (this.layout && this.layout.projects) || []) {
      const old = before[p.name];
      if (!old) continue;
      const now = new Set((p.graph_jobs || []).map(j => j.id));
      const gone = old.filter(id => id !== 0 && !now.has(id) && [...now].some(x => x > id)).sort((a, b) => b - a);
      if (!gone.length) continue;
      const pick = await say.info(`TraceFlow: graph job${gone.length > 1 ? 's' : ''} ${gone.slice().reverse().join(', ')} ` +
        `removed from ${p.name}'s graph_config.yaml.`, { modal: true,
        detail: 'Shift ids moves every later job down (its id, its figure folder, the faceting addresses into it and ' +
          'the JOB / GRAPH_ID lines of its header); the removed job\'s figures move to graphs/jobs/_deleted/. ' +
          'Leave the gap keeps every id.' }, 'Shift ids', 'Leave the gap');
      if (pick !== 'Shift ids') continue;
      const notes = [];
      for (const id of gone) {
        this.graphEditUntil = Date.now() + 15000;
        const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['close-gap', '--project', p.name, '--id', String(id)])
          .catch(e => ({ error: e.message }));
        if (out.error) { say.warn(`TraceFlow: ids not shifted after job ${id}: ${out.error}`); break; }
        notes.push(`job ${id}: ${Object.keys(out.moved || {}).length} later job(s) moved down, ` +
          `${(out.folders || []).length} figure folder(s) renamed, ${out.facet_addresses || 0} facet address(es)`);
        if ((out.old_references || []).length) notes.push(`still naming old ids (edit by hand): ${out.old_references.join(', ')}`);
      }
      if (notes.length) say.info(`TraceFlow: ${notes.join('\n')}`);
    }
    for (const p of (this.layout && this.layout.projects) || []) {   // the same for facets (FACE_ID)
      const old = before[`facets:${p.name}`];
      if (!old) continue;
      const now = new Set((p.graph_facets || []).map(f => f.id));
      const gone = old.filter(id => !now.has(id) && [...now].some(x => x > id)).sort((a, b) => b - a);
      if (!gone.length) continue;
      const pick = await say.info(`TraceFlow: facet${gone.length > 1 ? 's' : ''} ${gone.slice().reverse().join(', ')} removed ` +
        `from ${p.name}'s graph_config.yaml.`, { modal: true,
        detail: 'Shift ids moves every later facet down (its id, its output folder, the FACET / FACE_ID lines of its ' +
          'header); the removed facet\'s output moves to graphs/facets/_deleted/. Leave the gap keeps every id.' }, 'Shift ids', 'Leave the gap');
      if (pick !== 'Shift ids') continue;
      for (const id of gone) {
        this.graphEditUntil = Date.now() + 15000;
        const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['facet-close-gap', '--project', p.name, '--id', String(id)])
          .catch(e => ({ error: e.message }));
        if (out.error) { say.warn(`TraceFlow: facet ids not shifted after facet ${id}: ${out.error}`); break; }
      }
    }
  }

  /**
   * Done list → Delete from Graph Config: the run leaves the graph jobs drawing this metric
   * (graph_export.py remove); a job left with no run is deleted and the later ids move down.
   */
  async graphRemove(p, name, key) {
    const [source, processName, metric] = key.split('|');
    const jobs = (p.graph_jobs || []).filter(j => j.id !== 0 && j.metric === metric && j.source === source &&
      (j.processes || []).length === 1 && j.processes[0] === processName && (j.models || []).includes(name));
    if (!jobs.length) return say.info(`TraceFlow: ${name} is in no graph job drawing ${metric}.`);
    const emptied = jobs.filter(j => j.models.length === 1).map(j => j.id);
    const ok = await say.warn(`TraceFlow: remove ${name} from graph job ${jobs.map(j => j.id).join(', ')}?`, { modal: true,
      detail: `${metric} (${source}, ${processName}). Its models entry and legend label go.` +
        (emptied.length ? ` Graph job ${emptied.join(', ')} then lists no run: it is deleted and every later job moves down one id ` +
          '(figure folders, facet panels and header lines follow).' : '') }, 'Remove');
    if (ok !== 'Remove') return undefined;
    this.graphEditUntil = Date.now() + 15000;
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['remove', '--project', p.name, '--models', name,
      '--source', source, '--process', processName, '--metric', metric]).catch(e => ({ error: e.message }));
    if (out.error) return say.warn(`TraceFlow: not removed. ${out.error}`, { modal: true });
    return say.info(`TraceFlow: ${name} removed from graph job ${[...out.updated, ...out.deleted].join(', ')}` +
      `${out.deleted.length ? `; job ${out.deleted.join(', ')} deleted, later ids moved down` : ''}.`);
  }

  /** Find Figure (a facet panel's address): a drawn figure under graphs/, written relative to it. */
  /**
   * A config form's address → its folder button: a file (or folder) picked in the system, written relative
   * to its base (the TraceFlow root for a data entry's database, else the project's folder), as the file
   * writes it; the picker opens where the address points, else at its start (services/side_channel_core/data).
   */
  async pickPath(m) {
    const p = this.project(String(m.project || ''));
    if (!p || !this.root) return undefined;
    const base = m.base === 'root' ? this.root : p.path;
    const cur = String(m.cur || '').replace(/\{\{[^}]*\}\}.*$/, '');                   // a {{template}} part is no folder
    const at = cur ? (path.isAbsolute(cur) ? cur : path.join(base, cur)) : '';
    const open = [at, at && path.dirname(at), m.start ? path.join(this.root, String(m.start)) : '', base]
      .find(d => d && fs.existsSync(d) && fs.statSync(d).isDirectory());
    const folder = m.kind === 'folder' || (m.kind === 'any' && cur && !path.extname(cur));
    const picked = await vscode.window.showOpenDialog({
      defaultUri: vscode.Uri.file(open || base), canSelectFiles: !folder, canSelectFolders: folder, canSelectMany: false,
      openLabel: 'Use this', title: `${(m.path || []).join(' › ')}: pick the ${folder ? 'folder' : 'file'}`,
    });
    if (!picked || !picked[0]) return undefined;
    const chosen = picked[0].fsPath;
    const value = inside(base, chosen) ? path.relative(base, chosen).split(path.sep).join('/') : chosen;
    return this.post(this.view, { type: 'figure-picked', file: m.file, path: m.path, value });
  }

  /** FIND MODEL: every field the project's runs hold (model_filter.py --fields), for the model_filter.yaml form. */
  async filterFields(project, written = false) {
    const p = this.project(project);
    if (!p) return undefined;
    const data = await runModule(this.root, this.python, 'services.model_filter.model_filter', ['--project', p.name, '--fields'])
      .catch(e => ({ error: e.message }));
    return this.post(this.view, { type: 'filter-fields', project: p.name, data, written });
  }

  /** FIND MODEL → Find Models: the ticked conditions written to model_filter.where (--write-where), then make find_model. */
  async findModels(project, where) {
    const p = this.project(project);
    if (!p) return undefined;
    if (!Array.isArray(where) || !where.length) return say.warn('TraceFlow: tick one field at least: model_filter.where needs one condition.');
    const out = await runModule(this.root, this.python, 'services.model_filter.model_filter', ['--project', p.name, '--write-where', JSON.stringify(where)])
      .catch(e => ({ error: e.message }));
    if (out.error) return say.warn(`TraceFlow: model_filter.yaml not changed. ${out.error}`, { modal: true });
    this.filterFields(p.name, true).catch(e => this.report(e));
    return this.run({ action: 'find', project: p.name });
  }

  async findFigure(m) {
    const file = this.configFile(m.file);
    const p = this.project(String(m.project || ''));
    if (!file || !p) return undefined;
    const graphs = p.graphs_dir || path.dirname(p.graph_jobs_dir);
    const picked = await vscode.window.showOpenDialog({ defaultUri: vscode.Uri.file(fs.existsSync(p.graph_jobs_dir) ? p.graph_jobs_dir : graphs),
      canSelectMany: false, openLabel: 'Use this figure', filters: { Figures: ['pdf', 'png', 'svg'] } });
    if (!picked || !picked.length) return undefined;
    const chosen = picked[0].fsPath;
    // in a run's figure folder: one of its parent folders is a run's graphs/job_NNN_<name>/ (figures sit up to
    // four levels down: job_…/solo/<run>/<process>/)
    let inRun = false;
    for (let dir = path.dirname(chosen), n = 0; n < 6 && dir !== path.dirname(dir); dir = path.dirname(dir), n += 1) {
      if (this.isFigureDir(p, dir) && !inside(p.graph_jobs_dir, dir)) { inRun = true; break; }
    }
    if (!inside(path.resolve(graphs), chosen) && !inRun) {
      return say.warn(`TraceFlow: a facet's panel is a figure under ${graphs} or in a run's graphs folder; ${chosen} is not.`, { modal: true });
    }
    const base = inside(path.resolve(graphs), chosen) ? graphs : p.path;
    const value = path.relative(base, chosen).split(path.sep).join('/');
    return this.post(this.view, { type: 'figure-picked', file: m.file, path: JSON.parse(String(m.path || '[]')), value });
  }

  /**
   * FIGURES → Export to Facet: the ticked figure files, in the order ticked, as one new facet of
   * graph_config.yaml (graph_export.py export-facet). Draw offers make graph FACE_ID=<id>.
   */
  async facetExport(p, files) {
    const picked = (Array.isArray(files) ? files : []).map(String).filter(Boolean);
    if (picked.length < 2) return say.info('TraceFlow: tick at least two drawn figures to compose a facet.');
    this.graphEditUntil = Date.now() + 15000;
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['export-facet', '--project', p.name,
      '--files', picked.join(',')]).catch(e => ({ error: e.message }));
    if (out.error) return say.warn(`TraceFlow: no facet made. ${out.error}`, { modal: true });
    this.post(this.view, { type: 'facet-clear', project: p.name });
    const pick = await say.info(`TraceFlow: created facet ${out.created} (${out.name}, ${out.panels} panels).`, 'Draw', 'Open graph_config.yaml');
    if (pick === 'Draw') return this.run({ action: 'draw-facet', project: p.name, id: out.created });
    if (pick === 'Open graph_config.yaml') {
      const cfg = (p.configs || []).find(c => /graph_config\.ya?ml$/.test(c.path));
      if (cfg) vscode.window.showTextDocument(vscode.Uri.file(cfg.path));
    }
    return undefined;
  }

  /**
   * A Done run's three final true-key ranks, each with the traces it was read after: validation (the last
   * epoch's mean rank curve, as the tile's rank), the test at the same trace count (its curve read there:
   * the like-for-like comparison) and the test after all its traces. A rank falls as traces are added, so
   * only the first two compare. Null where the run has none (no test attacked, not written yet, or a test
   * shorter than the validation). Each curve file is written once and read once per session.
   */
  /**
   * A Done run's tests as rank sources: its training's recorded test (kind all) and every DONE test appended after it
   * (test_registry.json external_tests: Retest All / Budget, Test All / Budget), each with its kind, its finish time
   * and its mean rank curve (the GE over its attacks per trace count, metrics/.../mean_attack_rank_curve). The
   * validation's curve is the training's. A record of before the kinds is a budget test when its attack has a budget.
   */
  rankSources(project, run) {
    const metrics = path.join(this.runDir(project, run), 'metrics');
    const newest = dir => {
      let names = [];
      try { names = fs.readdirSync(dir).filter(n => /^ep\d+\.npy$/.test(n)).sort(); } catch { /* none */ }
      return names.length ? path.join(dir, names[names.length - 1]) : null;
    };
    const curve = dir => newest(path.join(dir, 'arrays', 'test', 'mean_attack_rank_curve'));
    const entry = (this.registry(this.testRegistryFile(project)) || {})[run.hash] || {};
    const recorded = curve(path.join(metrics, 'attack_metrics'));
    const sources = recorded ? [{ id: 'recorded', kind: 'all', source: 'training',
      time: Date.parse(entry.finished_at || '') || mtimeMs(recorded) || 0, file: recorded }] : [];
    for (const [id, t] of Object.entries(entry.external_tests || {})) {
      if (!t || typeof t !== 'object' || String(t.status || '').toUpperCase() !== 'DONE') continue;
      const budget = (((t.settings || {}).perform_attack || {}).test || {}).attack_trace_budget;
      const file = curve(path.join(this.runDir(project, run), 'test', id, 'metrics', 'attack_metrics'));
      if (file) sources.push({ id, kind: t.kind || (budget != null ? 'budget' : 'all'), source: t.source || 'test',
                               time: Date.parse(t.finished_at || '') || mtimeMs(file) || 0, file });
    }
    const validationDir = H.runCurveDir(metrics);
    return { sources, validation: validationDir ? newest(validationDir) : null };
  }

  /** [traces, rank] of a curve file, at `traces` or its last row (cached by the file's time). */
  curveAt(file, traces) {
    if (!file) return null;
    this.rowCache = this.rowCache || new Map();
    const key = `${file}|${mtimeMs(file)}|${traces}`;
    if (!this.rowCache.has(key)) {
      let row = null;
      try {
        const buf = fs.readFileSync(file);
        row = traces == null ? H.npyLastRow(buf) : H.npyRowAt(buf, traces);
      } catch { /* unreadable: none */ }
      this.rowCache.set(key, row && row.length >= 2 ? { traces: row[0], rank: row[1] } : null);
    }
    return this.rowCache.get(key);
  }

  /**
   * A Done run's four ranks (owner 2026-10-27), each the mean rank of the true key over the test's attacks (GE)
   * after the traces it gives: val, its validation's after all N_val traces (its training's, once); and from the
   * latest tests only, never an earlier one: same, the latest test whose curve reaches N_val, at N_val; budget, the
   * latest Budget test after its data entry's budget; test, the latest All test (the training's own, or a later
   * Retest All / Test All) after every test trace.
   */
  runRanks(project, run) {
    const { sources, validation } = this.rankSources(project, run);
    const val = this.curveAt(validation, null);
    const latest = kind => sources.filter(x => kind == null || x.kind === kind).sort((a, b) => b.time - a.time);
    const end = kind => { const x = latest(kind)[0]; return x ? this.curveAt(x.file, null) : null; };
    const reach = val ? latest(null).find(x => { const last = this.curveAt(x.file, null); return last && last.traces >= val.traces; }) : null;
    return { val, same: reach ? this.curveAt(reach.file, val.traces) : null, budget: end('budget'), test: end('all') };
  }

  /** runRanks, or null for a run whose folder cannot be read (its line shows no ranks). */
  safeRanks(project, run) {
    try { return this.runRanks(project, run); } catch { return null; }
  }

  /** One appended test's own ranks (its TESTS line): the run's validation, and this test's at N_val (when its curve
   *  reaches N_val) and at its end, under its kind (budget or all). */
  testRanks(project, run, id) {
    try {
      const { sources, validation } = this.rankSources(project, run);
      const x = sources.find(t => t.id === id);
      const val = this.curveAt(validation, null);
      if (!x) return { val, same: null, budget: null, test: null };
      const last = this.curveAt(x.file, null);
      const same = val && last && last.traces >= val.traces ? this.curveAt(x.file, val.traces) : null;
      return { val, same, budget: x.kind === 'budget' ? last : null, test: x.kind === 'all' ? last : null };
    } catch { return null; }
  }

  /**
   * The Done list's menus: each run's recorded (source, process, metric), from graph_export.py
   * choices. Reading them verifies every record's seal (about 1 s a run), so a menu is kept: in
   * this session, and in workspaceState with a fingerprint of the run's metric files, reused in
   * the next session while the files are unchanged (a Retest adds files: read again). Runs not
   * kept yet load in one background batch; a run opened in the Done list loads at once on its
   * own (loadChoices 'now'). A failed read shows its error and is tried again when reopened.
   */
  choicesFor(p, names) {
    this.graphChoices = this.graphChoices || {};
    this.choiceStamps = this.choiceStamps || {};
    const known = this.graphChoices[p.name] = this.graphChoices[p.name] || {};
    const env = this.followed(p).env || 'dev';
    // a menu kept in this session is read again when the run's records change: its test, written after the run
    // is listed done, or a Retest (the folders the records are written into change their time; a stat each)
    for (const n of names) {
      const key = `${p.name}|${env}|${n}`;
      const stamp = this.choicesStamp(p, n);
      if (Array.isArray(known[n]) && this.choiceStamps[key] !== undefined && this.choiceStamps[key] !== stamp) delete known[n];
      if (this.choiceStamps[key] === undefined || !(n in known)) this.choiceStamps[key] = stamp;
    }
    const stored = (this.context && this.context.workspaceState && this.context.workspaceState.get(CHOICES_KEY)) || {};
    for (const n of names) {
      if (n in known) continue;
      const hit = stored[`${p.name}|${env}|${n}`];
      if (hit && Array.isArray(hit.rows) && hit.sig === this.metricsSig(p, n)) known[n] = hit.rows;
    }
    this.loadChoices(p, names.filter(n => !(n in known)), 'batch');
    return known;
  }

  /** The times of the folders a run's records are written into (history rows, key-recovery reports, test arrays). */
  choicesStamp(p, name) {
    const metrics = path.join(p.path || '', 'artifacts', this.followed(p).env || 'dev', 'generated_models', name, 'metrics');
    return [['attack_metrics', 'scalars'], ['key_recovery'], ['attack_metrics', 'arrays', 'test']]
      .map(parts => mtimeMs(path.join(metrics, ...parts)) || 0).join(':');
  }

  /** A run's metric files as count:bytes:newest mtime (metrics/ and cross_validation/ of its folder). */
  metricsSig(p, name) {
    const run = path.join(p.path || '', 'artifacts', this.followed(p).env || 'dev', 'generated_models', name);
    let n = 0;
    let bytes = 0;
    let newest = 0;
    const walk = dir => {
      let entries;
      try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
      for (const e of entries) {
        const f = path.join(dir, e.name);
        if (e.isDirectory()) { walk(f); continue; }
        try { const st = fs.statSync(f); n += 1; bytes += st.size; newest = Math.max(newest, st.mtimeMs); } catch { /* gone meanwhile */ }
      }
    };
    walk(path.join(run, 'metrics'));
    walk(path.join(run, 'cross_validation'));
    return `${n}:${bytes}:${Math.round(newest)}`;
  }

  /** Reads the menus of `names` in one graph_export.py process; lane 'batch' (background) or 'now' (an opened run). */
  loadChoices(p, names, lane) {
    this.choicesLoading = this.choicesLoading || {};
    const flag = `${p.name}|${lane}`;
    if (!names.length || !this.root || !this.python || this.choicesLoading[flag]) return;
    const known = this.graphChoices[p.name];
    const env = this.followed(p).env || 'dev';
    this.choicesLoading[flag] = true;
    for (const n of names) if (!Array.isArray(known[n])) known[n] = null;
    const sigs = Object.fromEntries(names.map(n => [n, this.metricsSig(p, n)]));   // before the read: a file written meanwhile reads again
    const failed = msg => { for (const n of names) if (!Array.isArray(known[n])) known[n] = { error: msg }; };
    runModule(this.root, this.python, GRAPH_EXPORT, ['choices', '--project', p.name, '--env', env, '--models', names.join(',')])
      .then(out => {
        if (out.error) return failed(out.error);
        const read = names.filter(n => Array.isArray(out[n]));
        for (const n of read) known[n] = out[n];
        this.storeChoices(p.name, env, read.map(n => [n, sigs[n], out[n]]));
        return undefined;
      })
      .catch(e => failed(e.message || String(e)))
      .finally(() => { this.choicesLoading[flag] = false; this.tick(); });
  }

  /** An opened Done run whose menu is not read yet, or failed: read it now, on its own. */
  wantChoices(m) {
    const p = ((this.layout && this.layout.projects) || []).find(x => x.name === m.project);
    if (!p || !m.name) return;
    this.graphChoices = this.graphChoices || {};
    const known = this.graphChoices[p.name] = this.graphChoices[p.name] || {};
    if (!Array.isArray(known[m.name])) this.loadChoices(p, [m.name], 'now');
  }

  storeChoices(project, env, rows) {
    if (!rows.length || !this.context || !this.context.workspaceState) return;
    const all = { ...(this.context.workspaceState.get(CHOICES_KEY) || {}) };
    for (const [n, sig, menu] of rows) all[`${project}|${env}|${n}`] = { sig, rows: menu };
    this.context.workspaceState.update(CHOICES_KEY, all);
  }

  /**
   * Done list → Export to Graph Config: the picked runs and the one metric they were picked with
   * become a graph job (graph_export.py export): a new one, or the existing job of the same batch,
   * dataset, source, metric and process, which gets the new models. Draw offers make graph for it.
   */
  async graphExport(p, picks) {
    const rows = (Array.isArray(picks) ? picks : []).filter(x => x && x.name && x.key);
    if (!rows.length) return say.info('TraceFlow: pick a metric beside each run to export.');
    const keys = [...new Set(rows.map(x => x.key))];
    if (keys.length > 1) {
      return say.warn(`TraceFlow: one figure draws one metric; the picked runs name ${keys.length}:\n` +
        keys.map(k => k.split('|').join(' · ')).join('\n'), { modal: true });
    }
    const [source, processName, metric] = keys[0].split('|');
    const env = this.followed(p).env || 'dev';
    this.graphEditUntil = Date.now() + 15000;
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['export', '--project', p.name, '--env', env,
      '--models', rows.map(x => x.name).join(','), '--source', source, '--process', processName, '--metric', metric])
      .catch(e => ({ error: e.message }));
    if (out.error) return say.warn(`TraceFlow: not exported. ${out.error}`, { modal: true });
    this.post(this.view, { type: 'gexp-clear', project: p.name });
    const id = out.created != null ? out.created : out.updated;
    const what = out.created != null
      ? `created graph job ${id} (${out.name})`
      : (out.added || []).length ? `added ${out.added.join(', ')} to graph job ${id}` : `graph job ${id} already lists them`;
    const labels = Object.entries(out.labels || {}).map(([n, l]) => `${n}: ${l}`).join(', ');
    const pick = await say.info(`TraceFlow: ${what}. Legend: ${labels}`, 'Draw', 'Open graph_config.yaml');
    if (pick === 'Draw') return this.run({ action: 'draw', project: p.name, id });
    if (pick === 'Open graph_config.yaml') {
      const cfg = (p.configs || []).find(c => /graph_config\.ya?ml$/.test(c.path));
      if (cfg) vscode.window.showTextDocument(vscode.Uri.file(cfg.path));
    }
    return undefined;
  }

  /**
   * QUEUES → Overlay Graph: the finished runs of a queue's study as graph jobs (graph_export.py study-plan, study).
   * With two axes or more you tick the legend axes (each figure's lines; pre-ticked: the axis of most
   * values); every value of an unticked axis is a figure of its own. Then the metrics (metrics_to_log
   * pre-ticked; a fit time also as `relative`, which asks for the reference model, one of the lines):
   * one job per figure and metric. A study TraceFlow cannot make says so once, with
   * graph_config.yaml to do it by hand.
   */
  async study(p, key, env) {
    const args = ['--project', p.name, '--env', env, '--queue', key];
    const plan = await runModule(this.root, this.python, GRAPH_EXPORT, ['study-plan', ...args]).catch(e => ({ error: e.message }));
    if (plan.error) return this.studyFailed(p, plan);
    let legend = plan.legend;
    if (plan.axes.length >= 2) {
      const axes = await vscode.window.showQuickPick(plan.axes.map(a => ({
        label: a.key, description: `${a.values.length} values: ${a.values.join(', ')}`, picked: plan.legend.includes(a.key) })),
      { canPickMany: true, ignoreFocusOut: true,
        title: `Overlay graph of ${plan.study} (${plan.runs.length} runs): tick the axes drawn as lines; each value of an unticked axis is a figure` });
      if (!axes) return undefined;
      if (!axes.length) return say.warn('TraceFlow: tick one axis at least: the lines of each figure.');
      legend = axes.map(a => a.label);
    }
    const metrics = await vscode.window.showQuickPick(plan.metrics.flatMap(m => [
      { label: m.metric, description: `${m.source} · ${m.process}`, picked: m.logged, m },
      ...(m.relative ? [{ label: `relative ${m.metric}`, description: `${m.source} · ${m.process} · ÷ the reference line's (asked next), as bars`,
        picked: false, m, rel: true }] : [])]),
    { canPickMany: true, ignoreFocusOut: true, matchOnDescription: true,
      title: `Overlay graph of ${plan.study}: one graph job per figure for each ticked metric (metrics_to_log ticked)` });
    if (!metrics) return undefined;
    if (!metrics.length) return say.warn('TraceFlow: tick one metric at least.');
    let reference = null;
    if (metrics.some(x => x.rel)) {                  // relative: each figure's values over the reference line's
      const show = v => (v == null ? 'none' : String(v));
      const lines = new Map();
      for (const r of plan.runs) {
        const ref = Object.fromEntries(legend.map(a => [a, plan.values[r][a]]));
        const k = JSON.stringify(legend.map(a => ref[a]));
        if (!lines.has(k)) {
          lines.set(k, { ref, label: legend.length === 1 ? show(ref[legend[0]]) : legend.map(a => `${a.split('.').pop()}=${show(ref[a])}`).join(', '),
            description: legend.map(a => `${a} = ${show(ref[a])}`).join(', ') });
        }
      }
      const picked = await vscode.window.showQuickPick([...lines.values()], { ignoreFocusOut: true,
        title: `Overlay graph of ${plan.study}: the reference model of the relative metric(s); its value is 1.0 in every figure` });
      if (!picked) return undefined;
      reference = picked.ref;
    }
    // lines repeated on splits (profiling_split.random_state, a run each: Wouters' ten trainings) are pooled in a curve:
    // their band is asked, never assumed (graph_job_validator.GROUP_BANDS)
    let band = null;
    if ((plan.repeats || 1) > 1 && metrics.some(x => !x.rel && x.m.kind === 'curve')) {
      const pickedBand = await vscode.window.showQuickPick([
        { label: 'min_max', description: 'shade from the lowest to the highest run at each number of traces' },
        { label: 'std', description: 'shade the mean \u00b1 1 sample standard deviation of the runs' },
        { label: 'none', description: 'the runs thin and their mean bold, no band' }],
      { ignoreFocusOut: true, title: `Overlay graph of ${plan.study}: each line repeats on ${plan.repeats} splits; ` +
        'its runs are drawn thin, their mean bold: the band (Wouters et al. never say theirs)' });
      if (!pickedBand) return undefined;
      band = pickedBand.label;
    }
    this.graphEditUntil = Date.now() + 15000;
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['study', ...args, ...(band ? ['--band', band] : []), '--legend', legend.join(','),
      '--metrics', metrics.filter(x => !x.rel).map(x => `${x.m.source}|${x.m.process}|${x.m.metric}`).join(','),
      ...(reference ? ['--relative', metrics.filter(x => x.rel).map(x => `${x.m.source}|${x.m.process}|${x.m.metric}`).join(','),
        '--reference', JSON.stringify(reference)] : [])]).catch(e => ({ error: e.message }));
    this.graphEditUntil = Date.now() + 15000;
    if (out.error) return this.studyFailed(p, out);
    const ids = (out.created || []).map(c => c.id);
    const pick = await say.info(`TraceFlow: overlay graph of ${out.study}, ${out.figures} figure(s): ` +
      (ids.length ? `created graph job(s) ${ids.join(', ')}` : 'no new job') +
      ((out.kept || []).length ? `; already in graph_config.yaml: ${out.kept.join(', ')}` : '') + '.',
    ...(ids.length ? ['Draw all'] : []), 'Open graph_config.yaml');
    if (pick === 'Draw all') return this.run({ action: 'draw-all', project: p.name, ids });
    if (pick === 'Open graph_config.yaml') return this.openGraphConfig(p);
    return undefined;
  }

  async studyFailed(p, out) {
    if (!out.study_failed) return say.warn(`TraceFlow: ${out.error}`, { modal: true });
    const pick = await say.warn(`TraceFlow: ${out.error}`, { modal: true, detail: out.detail }, 'Open graph_config.yaml');
    return pick ? this.openGraphConfig(p) : undefined;
  }

  openGraphConfig(p) {
    const cfg = (p.configs || []).find(c => /graph_config\.ya?ml$/.test(c.path));
    return cfg ? vscode.window.showTextDocument(vscode.Uri.file(cfg.path)) : undefined;
  }

  tick() {
    try { this.push(); } catch (e) { this.report(e); }
  }

  // ── SNAPSHOTS: reproduce runs from their params_out snapshots (utils/reproduce.py) ─────────────────
  // Scopes: `p:<project>` (the project as master.yaml registers it) and `custom` (a folder you choose).
  // A scope is scanned only when its Scan is clicked; the webview filters the rows.

  snapState() {
    const rep = this.reproducing || {};
    const projects = ((this.layout && this.layout.projects) || []).map(p => ({ name: p.name, scope: `p:${p.name}`,
      live: rep[`p:${p.name}`] && rep[`p:${p.name}`].dest ? this.folderBeat(rep[`p:${p.name}`].dest) : null,
      ...((this.snapshots || {})[`p:${p.name}`] || {}) }));
    const folder = this.customFolder();
    const customDest = (rep.custom && rep.custom.dest) || folder;
    return { projects, custom: { scope: 'custom', folder, live: this.folderBeat(customDest), ...((this.snapshots || {}).custom || {}) },
             reproducing: this.reproducing || {} };
  }

  /** The newest heartbeat under a chosen folder (a run reproduced there writes <folder>/artifacts/<env>/heartbeat.json). */
  folderBeat(folder) {
    if (!folder) return null;
    let best = null;
    for (const env of ['dev', 'staging', 'prod']) {
      let hb = null;
      try { hb = S.parseHeartbeat(fs.readFileSync(path.join(folder, 'artifacts', env, 'heartbeat.json'), 'utf8')); } catch (e) { /* none */ }
      if (hb && (!best || Number(hb.written_ns) > Number(best.written_ns))) best = hb;
    }
    if (!best) return null;
    return { ...S.runSummary(best), hash: best.model_hash || '', env: best.env || '', alive: S.runAlive(best, best.model_hash, Date.now()) };
  }

  customFolder() {
    return (this.context && this.context.workspaceState && this.context.workspaceState.get('traceflow.snapshotsFolder')) || '';
  }

  async snap(m) {
    const scope = String(m.scope || '');
    if (m.action === 'choose') return this.chooseSnapshotFolder();
    if (m.action === 'scan') return this.scanSnapshots(scope, m.filters || {});
    if (m.action === 'scanall') return this.scanSnapshots(scope, {}, { all: true });
    const row = ((((this.snapshots || {})[scope]) || {}).rows || []).find(r => r.path === m.path);
    if (!row) return say.warn('TraceFlow: that snapshot is not in the last scan; Scan again.');
    if (m.action === 'open') return this.open(row.path);
    if (m.action === 'reproduce') return this.reproduceSnapshot(scope, row);
    return undefined;
  }

  async chooseSnapshotFolder() {
    const current = this.customFolder();
    const picked = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, canSelectMany: false,
      openLabel: 'Use this folder', title: 'Folder with snapshots (<model_name>_params.yaml, any depth)',
      defaultUri: current ? vscode.Uri.file(current) : (this.root ? vscode.Uri.file(this.root) : undefined) });
    if (!picked || !picked.length) return;
    await this.context.workspaceState.update('traceflow.snapshotsFolder', picked[0].fsPath);
    this.snapshots = { ...(this.snapshots || {}), custom: {} };
    this.tick();
  }

  /**
   * Scan one scope with its filters: at least one filled, every filled one valid (S.snapFilter); then the
   * snapshots that match all of them are listed, or an error says why none is.
   */
  async scanSnapshots(scope, filters = {}, { all = false } = {}) {
    this.snapshots = this.snapshots || {};
    const custom = scope === 'custom';
    const folder = custom ? this.customFolder() : '';
    if (custom && !folder) return this.chooseSnapshotFolder();
    const project = custom ? '' : scope.replace(/^p:/, '');
    const p = custom ? null : this.project(project);
    if (!custom && !p) return undefined;
    const known = p ? { modelIds: p.model_ids || [], datasets: p.datasets || [] } : {};
    const prev = this.snapshots[scope] || {};
    const pre = all ? {} : S.snapFilter([], filters, known);     // nothing filled or a wrong entry: no scan
    if (pre.error && !/^no snapshot matches/.test(pre.error)) {
      this.snapshots[scope] = { ...prev, rows: [], at: Date.now(), scanning: false, error: pre.error, filters };
      say.warn(`TraceFlow: ${pre.error}`);
      return this.tick();
    }
    this.snapshots[scope] = { ...prev, scanning: true, error: '', filters };
    this.tick();
    try {
      const args = custom ? ['snapshots', '--folder', folder, '--root', this.root] : ['snapshots', '--project', project, '--root', this.root];
      const out = await runPy(this.root, this.python, 'layout', args);
      const everyOne = out.snapshots || [];
      const found = all ? { rows: everyOne } : S.snapFilter(everyOne, filters, known);
      if (found.error) say.warn(`TraceFlow: ${found.error}`);
      this.snapshots[scope] = { rows: found.rows || [], total: everyOne.length, at: Date.now(), scanning: false, error: found.error || '',
                                filters, pattern: out.pattern || '', root: out.root || '', folder: out.folder || folder,
                                origin: custom ? (out.origin || path.basename(folder)) : project };
    } catch (e) {
      this.snapshots[scope] = { ...prev, scanning: false, error: e.message || String(e) };
    }
    this.tick();
    return undefined;
  }

  /**
   * Where a reproduction writes its outputs ({root}: '' = the project itself, else an absolute folder):
   * asked every time, since reproducing in the project (or in the scanned folder) can replace the outputs
   * there. A folder must lie outside the repository, or inside projects/ (utils/reproduce.py
   * check_output_root); the last one chosen is offered first.
   */
  async reproduceDestination(custom, projectName, p, folder) {
    const last = (this.context && this.context.workspaceState && this.context.workspaceState.get('traceflow.reproduceDest')) || '';
    const here = custom ? 'The snapshot folder' : `In ${projectName}`;
    const options = [
      [here, custom
        ? `${folder}: its runs, registries and figures under that folder; an earlier run of the same name there is replaced.`
        : `${p.path}: the project's own artifacts/ and registries; a DONE run's results there are replaced.`, { root: custom ? folder : '' }],
      ...(last && last !== folder ? [['Last chosen folder', `${last}: the reproduced runs' root, every paths: entry inside it.`, { root: last }]] : []),
      ['Choose a folder…', 'a folder outside the repository, or inside projects/: the project and the scanned folder stay untouched.', { choose: true }],
    ];
    const label = await choose('Where should the reproduction write its outputs?',
      options.map(([l, d]) => `${l}: ${d}`).join('\n\n'), options.map(([l]) => l));
    const pick = (options.find(([l]) => l === label) || [])[2];
    if (!pick) return null;
    if (!pick.choose) return { root: pick.root };
    const chosen = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, canSelectMany: false,
      openLabel: 'Reproduce into this folder', title: 'Destination of the reproduction (the runs\' root)',
      defaultUri: vscode.Uri.file(last || folder || this.root || os.homedir()) });
    if (!chosen || !chosen.length) return null;
    const root = chosen[0].fsPath;
    if (this.root && (root === this.root || inside(root, this.root))) {
      say.warn(`TraceFlow: ${root} holds the repository: choose a folder outside it, or inside projects/.`, { modal: true });
      return null;
    }
    if (this.context && this.context.workspaceState) await this.context.workspaceState.update('traceflow.reproduceDest', root);
    return { root };
  }

  /** A run's status in the registry of a folder (<folder>/artifacts/run_registry.json), '' when not there. */
  registryStatus(folder, hash) {
    const reg = this.registry(path.join(folder, 'artifacts', 'run_registry.json')) || {};
    return String(((reg[hash] || {}).status) || '');
  }

  /**
   * Reproduce one snapshot: make reproduce ENV=<the snapshot's ENV> PROJECT=<its project> MODEL=<its name>
   * (BATCH=<its first batch label>), in tmux after the usual checks. Registered or not, the run trains from
   * its snapshot under its own name and hash; an unfinished one resumes. A DONE one asks first and gets
   * FORCE=1 (trained again, its results replaced). A CUSTOM snapshot is first copied to its params_out path
   * in the project it names (layout.py place), after the confirmation.
   */
  async reproduceSnapshot(scope, row) {
    if (!row.verified) {
      return say.warn(`TraceFlow: ${row.model_name || row.folder} is refused:\n${row.problem}`, { modal: true });
    }
    const custom = scope === 'custom';
    const projectName = custom ? row.project : scope.replace(/^p:/, '');
    const p = this.project(projectName);
    if (!p) return say.warn(`TraceFlow: project ${projectName} is not in master.yaml.`, { modal: true });
    const folder = custom ? this.customFolder() : '';
    const dest = await this.reproduceDestination(custom, projectName, p, folder);
    if (!dest) return undefined;
    const into = dest.root;                                         // '' = the project itself
    const source = custom ? folder : p.path;
    const moved = into && into !== source;                          // a destination other than where it was scanned
    // DONE where it will run: the scanned place's registry, or the destination's own
    const status = moved ? this.registryStatus(into, row.model_hash) : String(row.status || '');
    let force = false;
    if (status.toLowerCase() === 'done') {
      const again = await say.warn(
        `${row.model_name} (${row.model_hash}) is DONE in ${into || projectName} (ENV=${row.env}).\n\n` +
        'Train it again from its snapshot? Its results are replaced (make reproduce … FORCE=1).', { modal: true }, 'Train again');
      if (again !== 'Train again') return undefined;
      force = true;
    }
    const origin = custom ? (((this.snapshots || {}).custom || {}).origin || path.basename(folder)) : projectName;
    const batch = (row.batches || [])[0] || '';
    const vars = { ENV: row.env, PROJECT: projectName, MODEL: row.model_name, BATCH: batch, FORCE: force ? 1 : '',
                   REPRODUCE_ROOT: into };
    const copy = moved || (custom && !row.placed);
    const note = `Reproducing ${row.model_name} (${row.model_hash})\nfrom ${origin} · ENV ${row.env} (from the snapshot) · ` +
      (into ? `outputs under ${into} (every paths: entry; configs and scripts of project ${projectName})`
        : `into project ${projectName}`) + ` · ${status || 'not registered'}` +
      (copy ? `\nThe snapshot is first copied to ${moved ? `its params_out path under ${into}` : row.target}` : '');
    const started = await this.startLong(p, 'reproduce', vars, null, note, {
      before: copy ? async () => {
        const out = await runPy(this.root, this.python, 'layout',
          ['place', '--file', row.path, '--into', into, '--from', source, '--root', this.root]);
        if (out.error) throw new Error(out.error);
      } : null,
      log: into ? path.join(into, 'artifacts', row.env, 'tmux_logs', `${batch || 'reproduce'}.txt`) : '',
    });
    if (started) {
      this.reproducing = { ...(this.reproducing || {}), [scope]: { name: row.model_name, hash: row.model_hash, env: row.env,
                                                                   origin, project: projectName, dest: into, at: Date.now() } };
      this.tick();
    }
    return started;
  }

  /** Shows a failure in the view instead of letting the poll loop die silently; repeats are shown once. */
  report(e) {
    const msg = `panel error: ${e && e.message ? e.message : String(e)}`;
    console.error('[traceflow]', e);
    if (msg === this.lastReported) return;
    this.lastReported = msg;
    this.post(this.view, { type: 'state', error: msg });
  }

  /** postMessage to a view or panel that may have been disposed. */
  post(target, msg) {
    if (!target) return;
    try { target.webview.postMessage(msg).then(undefined, () => {}); } catch (e) { /* disposed */ }
  }

  pollGpu() {
    execFile('nvidia-smi', G.QUERY, { timeout: 4000 }, (err, stdout, stderr) => {
      if (err) {
        this.gpus = [];
        this.gpuError = err.code === 'ENOENT' ? 'nvidia-smi not found: no GPU stats'
          : `nvidia-smi: ${String(stderr || stdout || err.message).trim().split('\n')[0]}`;
        return;
      }
      this.gpus = G.parse(stdout);
      this.gpuError = this.gpus.length ? '' : `nvidia-smi: ${String(stdout).trim().split('\n')[0] || 'no GPU reported'}`;
      this.gpuHistory.add(this.gpus);
    });
  }

  /**
   * The stack's containers (`sudo -n nerdctl ps -a`, read only while the view shows) and whether an image
   * build runs now. No password is asked: without passwordless sudo the row says so.
   */
  pollDocker() {
    if (!this.view || !this.view.visible) return;
    execFile('sudo', [...NERDCTL, 'ps', '-a', '--format', '{{.Names}}\t{{.Status}}\t{{.Image}}'], { timeout: 8000 }, (err, stdout, stderr) => {
      this.docker.containers = err ? [] : S.containerHealth(stdout);
      this.docker.error = err ? `nerdctl ps: ${String(stderr || err.message).trim().split('\n')[0]}` : '';
    });
    this.builds().then(b => { this.docker.building = b.length > 0; });
  }

  /** The running image builds: nerdctl processes (by name, not a command line merely mentioning it) with the word build. */
  async builds() {
    return (await output('pgrep', ['-a', 'nerdctl'])).split('\n').filter(l => l.split(/\s+/).includes('build'));
  }

  dockerState() {
    const file = this.root ? path.join(this.root, BUILD_LOG) : '';
    const t = file ? mtimeMs(file) : 0;
    const build = t ? S.buildProgress(readText(file), { ageS: (Date.now() - t) / 1000, building: this.docker.building, nowMs: Date.now() }) : null;
    return { build, building: !!this.docker.building, containers: this.docker.containers, error: this.docker.error };
  }

  /**
   * Figures → Delete: remove a drawn graph job's output folder (graphs/jobs/job_<id>_…/) after one
   * confirmation, so the job reads "not drawn" and Draw makes it afresh. A folder the container wrote
   * as root is removed with sudo in a terminal.
   */
  async deleteFigure(project, dir) {
    const p = this.project(project);
    if (!this.isFigureDir(p, dir)) {
      return say.warn('TraceFlow: not a graph job folder of this project; nothing deleted.');
    }
    let files = [];
    try { files = fs.readdirSync(dir, { recursive: true }).map(String); } catch (e) { return this.pollFigures(); }
    const ok = await say.warn(
      `Delete the drawn figure ${path.basename(dir)}?\n\n${dir}\n${files.length} file(s): ${files.slice(0, 8).join(', ')}${files.length > 8 ? ', …' : ''}\n\n` +
      `The ${path.basename(dir).startsWith('facet_') ? 'facet' : 'graph job'} stays in graph_config.yaml: Draw makes it again.`, { modal: true }, 'Delete');
    if (ok !== 'Delete') return undefined;
    try {
      fs.rmSync(dir, { recursive: true, force: true });
    } catch (e) {
      if (!/EACCES|EPERM/.test(String(e.code || e.message))) throw e;
      this.terminal('TraceFlow · delete figure', `sudo rm -rf -- ${C.quote(dir)} && echo 'deleted ${path.basename(dir)}'`, true);
    }
    this.pollFigures();
    return undefined;
  }

  /** FIGURES → Delete all (n chosen): the drawn figures ticked, after one confirmation listing them. */
  async deleteFigures(project, dirs) {
    const p = this.project(project);
    const ok = (dirs || []).map(String).filter(d => this.isFigureDir(p, d) && fs.existsSync(d));
    if (!ok.length) return say.warn('TraceFlow: no drawn figure folder of this project is ticked; nothing deleted.');
    const names = ok.map(d => path.basename(d));
    const answer = await say.warn(
      `Delete ${ok.length} drawn figure${ok.length === 1 ? '' : 's'}?

${names.slice(0, 12).join('\n')}${names.length > 12 ? '\n…' : ''}

` +
      `Their ${ok.every(d => path.basename(d).startsWith('facet_')) ? 'facets' : 'graph jobs'} stay in graph_config.yaml: Draw makes them again.`, { modal: true }, 'Delete all');
    if (answer !== 'Delete all') return undefined;
    const denied = [];
    for (const d of ok) {
      try { fs.rmSync(d, { recursive: true, force: true }); } catch (e) {
        if (!/EACCES|EPERM/.test(String(e.code || e.message))) throw e;
        denied.push(d);
      }
    }
    if (denied.length) {
      this.terminal('TraceFlow · delete figures', `sudo rm -rf -- ${denied.map(C.quote).join(' ')} && echo 'deleted ${denied.length} figure folder(s)'`, true);
    }
    this.pollFigures();
    return undefined;
  }

  /**
   * GRAPH_CONFIG.YAML → a facet's line → Delete: facet `id` out of graph_config.yaml (graph_export.py
   * delete-facet: its output moves to graphs/facets/_deleted/, every later facet moves down one id),
   * after one confirmation.
   */
  async deleteFacet(project, id) {
    const p = this.project(project);
    const facetId = Number(id);
    if (!p || !Number.isInteger(facetId)) return undefined;
    const answer = await say.warn(`Delete facet ${facetId} from graph_config.yaml?\n\n` +
      'Its output moves to graphs/facets/_deleted/ (nothing is removed), and every later facet moves down by one id.',
    { modal: true }, 'Delete');
    if (answer !== 'Delete') return undefined;
    this.graphEditUntil = Date.now() + 15000;           // the watcher must not offer to shift the ids again
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['delete-facet', '--project', p.name, '--id', String(facetId)]);
    this.graphEditUntil = Date.now() + 15000;
    if (out && out.error) return say.error(`TraceFlow: facet ${facetId} not deleted: ${out.error}`);
    say.info(`TraceFlow: facet ${facetId} deleted from graph_config.yaml; its output is under graphs/facets/_deleted/.`);
    this.scheduleRelayout();
    return undefined;
  }

  /**
   * A job's line (GRAPH_CONFIG.YAML, or a run's Graph sub-panel) → Delete: graph job `id` out of graph_config.yaml (graph_export.py
   * delete-job: its figures move to graphs/jobs/_deleted/, every later job moves down one id), after
   * one confirmation. Job 0 is the judge's and is never deleted.
   */
  async deleteJob(project, id) {
    const p = this.project(project);
    const jobId = Number(id);
    if (!p || !Number.isInteger(jobId)) return undefined;
    if (jobId === 0) return say.warn('TraceFlow: job 0 is the model judge\'s job; it is never deleted.');
    const job = (p.graph_jobs || []).find(j => j.id === jobId);
    const answer = await say.warn(
      `Delete graph job ${jobId}${job && job.name ? ` (${job.name})` : ''} from graph_config.yaml?

` +
      `Its figures move to graphs/jobs/_deleted/ (nothing is removed), and every later job moves down by one id.`,
      { modal: true }, 'Delete');
    if (answer !== 'Delete') return undefined;
    // delete-job writes graph_config.yaml twice (the job out, then the later ids down): the file watcher
    // must not take the first write for a job deleted by hand and offer to shift the ids again.
    this.graphEditUntil = Date.now() + 15000;
    const out = await runModule(this.root, this.python, GRAPH_EXPORT, ['delete-job', '--project', p.name, '--id', String(jobId)]);
    this.graphEditUntil = Date.now() + 15000;
    if (out && out.error) return say.error(`TraceFlow: graph job ${jobId} not deleted: ${out.error}`);
    say.info(`TraceFlow: graph job ${jobId} deleted from graph_config.yaml; its figures are under graphs/jobs/_deleted/.`);
    this.scheduleRelayout();
    this.pollFigures();
    return undefined;
  }

  /**
   * Kill build: stop the running `nerdctl … build` (and so the make that called it) in a terminal, where
   * sudo may ask, after one confirmation.
   */
  async dockerKillBuild() {
    const builds = await this.builds();
    this.docker.building = builds.length > 0;
    if (!builds.length) return say.info('TraceFlow: no image build runs.');
    const ok = await say.warn(
      'Kill the image build? The nerdctl build gets SIGINT (SIGTERM after 5 s); make build-image fails, and ' +
      `so does the make that started it (orchestrate, experiment). The layers built so far stay in the build cache.\n\n${builds.join('\n')}`,
      { modal: true }, 'Kill build');
    if (ok !== 'Kill build') return undefined;
    const env = ((this.layout && this.layout.projects) || []).map(p => this.followed(p).env).find(Boolean) || 'dev';
    this.terminal('TraceFlow · kill build', `cd ${C.quote(this.root)} && ${C.killBuildLine(env)}`, true);
    this.docker.building = false;                       // the poll confirms once the processes are gone
    return undefined;
  }

  /**
   * Docker tear down: make docker-teardown ENV=… in a terminal (sudo may ask there), after one
   * confirmation. Refused while an experiment runs here or an image builds.
   */
  async dockerTeardown() {
    await this.pollProcesses();
    const followed = ((this.layout && this.layout.projects) || []).map(p => this.followed(p).env).filter(Boolean);
    const envs = [...new Set([...followed, 'dev', 'staging', 'prod'])];
    if (this.busy.length || this.docker.building) {
      // Something would block the tear down: say what, and offer to kill it first (make docker-teardown FORCE=1:
      // the experiment's make and python, the image build and BuildKit's steps, the containers by force).
      const what = this.busy.length ? `an experiment is running on this machine:\n${this.busy.join('\n')}` : 'an image is building.';
      const force = await say.warn(`TraceFlow: not torn down: ${what}\n\nLet it finish, or force the tear down: the ` +
        'experiment and the build are killed first (the run is recorded FAILED when it beats again), then the stack is torn down.',
        { modal: true }, 'Force tear down');
      if (force !== 'Force tear down') return undefined;
      const env = await choose('ENV for make docker-teardown FORCE=1?', 'The ENV picks the compose files the stack was started with.', envs);
      if (!env) return undefined;
      this.terminal('TraceFlow · docker teardown', `cd ${C.quote(this.root)} && ${C.makeLine('docker-teardown', { ENV: env, FORCE: '1' })}`, true);
      this.docker.building = false;
      return undefined;
    }
    const env = await choose('ENV for make docker-teardown?',
      'The ENV picks the compose files the stack was started with' +
      (followed.length ? ` (the panel follows ENV=${followed.join(', ')}).` : '.') +
      ' The next question lists what the tear down removes.', envs);
    if (!env) return undefined;
    const ok = await say.warn(
      `Tear down Docker (ENV=${env})? It frees the GPU, memory and disk the stack holds:\n\n` +
      '1. make docker-clean: compose down -v and the orchestrator lock file\n' +
      '2. remove every sidechannel* container and sidechannel:* image\n' +
      '3. builder prune -af, image prune -f, volume prune -f\n' +
      '4. empty deployment/docker/tmp/ and remove the build log\n\n' +
      'The prunes act on the whole containerd namespace "default": other projects\' build cache, dangling ' +
      'images and unused volumes go too. The next make orchestrate rebuilds the image.',
      { modal: true }, 'Tear down');
    if (ok !== 'Tear down') return undefined;
    this.terminal('TraceFlow · docker teardown', `cd ${C.quote(this.root)} && ${C.makeLine('docker-teardown', { ENV: env })}`, true);
    return undefined;
  }

  /**
   * tmux sessions, busy while the pane's shell has a child process (its command: that child), and
   * the experiments running on this machine. pane_current_command alone is not enough: a pane
   * running `bash -lc 'make … | tee …'` reports `bash` while make runs.
   */
  async pollProcesses() {
    const list = await output('tmux', ['list-sessions', '-F', '#{session_name}\t#{pane_current_command}\t#{pane_pid}\t#{session_attached}']);
    const sessions = {};
    for (const line of list.split('\n').filter(Boolean)) {
      const [name, cmd, pid, attached] = line.split('\t');
      const kids = /^\d+$/.test(pid || '') ? (await output('pgrep', ['-l', '-P', pid])).split('\n').filter(Boolean) : [];
      sessions[name] = { command: kids.length ? kids[0].split(/\s+/)[1] : (cmd || ''), busy: kids.length > 0,
                         attached: Number(attached) || 0 };        // terminals attached to it now
    }
    this.sessions = sessions;
    const procs = (await output('pgrep', ['-af', C.BUSY_PATTERN])).split('\n')
      .map(l => l.match(/^(\d+)\s+(.*)$/)).filter(Boolean).map(m => ({ pid: m[1], cmd: m[2] }));
    this.busy = [...new Set(procs.map(p => p.cmd))];          // a trainer's DataLoader workers repeat its line
    this.polledBusy = true;
    // The tmux session holding each running make: walk its parents up to a pane's shell. Whoever
    // started it (the panel's tf-<project>, or a session typed by hand such as fig26). The container's
    // trainer has no tmux ancestor (its parent is containerd), so it never matches.
    const panes = {};
    for (const line of (await output('tmux', ['list-panes', '-a', '-F', '#{session_name}\t#{pane_pid}'])).split('\n').filter(Boolean)) {
      const [name, pid] = line.split('\t');
      panes[pid] = name;
    }
    const parent = {};
    for (const line of (await output('ps', ['-eo', 'pid=,ppid='])).split('\n')) {
      const m = line.trim().split(/\s+/);
      if (m.length === 2) parent[m[0]] = m[1];
    }
    const viewers = (await output('pgrep', ['-af', C.VIEWER_PATTERN])).split('\n')
      .map(l => l.match(/^(\d+)\s+(.*)$/)).filter(Boolean).map(m => ({ pid: m[1], cmd: m[2], viewer: true }));
    const held = {};
    for (const pr of procs.concat(viewers)) {
      for (let pid = pr.pid, hops = 0; pid && pid !== '1' && hops < 64; pid = parent[pid], hops++) {
        if (!panes[pid]) continue;
        const h = held[panes[pid]] = held[panes[pid]] || { cmds: [], viewer: false, make: false };
        h.cmds.push(pr.cmd);
        if (pr.viewer) h.viewer = true; else h.make = true;
        break;
      }
    }
    // make: the session runs the experiment's make; viewer: it only follows the container's log.
    this.runSessions = Object.entries(held).map(([session, h]) => ({
      session, cmds: h.cmds, make: h.make, viewer: h.viewer, attached: (sessions[session] || {}).attached || 0 }));
  }

  /**
   * The tmux session holding this run: the one whose make line names its batch (QUEUE=<b> or BATCH=<b>),
   * else this project's make (PROJECT=<name>), else any make, else a session following the container's
   * log (nerdctl logs: the queue's own session after its make stream ended).
   */
  runSessionFor(p, batches = (this.runBatches || {})[p.name] || []) {
    const all = this.runSessions || [];
    const names = (b, c) => new RegExp(`\\b(QUEUE|BATCH)=${b.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(\\s|$)`).test(c);
    const makes = all.filter(s => s.make);
    return makes.find(s => batches.some(b => s.cmds.some(c => names(b, c))))
      || makes.find(s => s.cmds.some(c => c.includes(`PROJECT=${p.name}`))) || makes[0]
      || all.find(s => s.viewer) || null;
  }

  /**
   * The folder holding a graph job's job_NNN_<name>/ folder (services/graph_manager/job_paths.py):
   * a job of one run, that run's paths.model_graphs folder for ENV `env`; else <project>/graphs/jobs/.
   */
  jobRoot(p, job, env) {
    const runs = (job && job.runs) || [];
    if (runs.length !== 1) return p.graph_jobs_dir;
    const rel = String(p.model_graphs || 'artifacts/{{env}}/generated_models/{{model_name}}/graphs')
      .replace(/\{\{\s*env\s*\}\}/g, env || 'dev').replace(/\{\{\s*model_name\s*\}\}/g, runs[0]);
    return path.isAbsolute(rel) ? rel : path.join(p.path, rel);
  }

  /** A figure folder TraceFlow draws into: under graphs/jobs/, or a run's graphs/job_NNN_<name>/. */
  isFigureDir(p, dir) {
    if (!p || !dir) return false;
    const d = path.resolve(dir);
    if (p.graph_jobs_dir && inside(p.graph_jobs_dir, d) && d !== path.resolve(p.graph_jobs_dir)) return true;
    // a facet's output folder: graphs/facets/facet_NNN_<name> (not facets/ itself, nor _deleted/)
    const facets = path.join(p.graphs_dir || path.dirname(p.graph_jobs_dir || ''), 'facets');
    if (path.dirname(d) === path.resolve(facets) && /^facet_\d{3}_/.test(path.basename(d))) return true;
    return !!p.path && inside(path.join(p.path, 'artifacts'), d) && /[\\/]graphs[\\/]job_\d{3}_[^\\/]+$/.test(d);
  }

  /** Every graph job's drawn folder, keyed by job id: {root, dir, files}; a job of one run looks in its run's folder. The same for every facet's output folder. */
  pollFigures() {
    const out = {};
    const facets = {};
    for (const p of (this.layout && this.layout.projects) || []) {
      const env = this.followed(p).env || 'dev';
      const listed = new Map();                     // a root read once per poll
      const entriesOf = root => {
        if (!listed.has(root)) {
          let entries = [];
          try { entries = fs.readdirSync(root, { withFileTypes: true }); } catch (e) { /* not drawn yet */ }
          listed.set(root, entries.filter(e => e.isDirectory() && e.name.startsWith('job_')).map(e => e.name));
        }
        return listed.get(root);
      };
      const dirs = {};
      for (const job of p.graph_jobs || []) {
        const root = this.jobRoot(p, job, env);
        const dir = entriesOf(root).includes(job.dir) ? job.dir : null;  // its exact folder (layout.py)
        if (dir) dirs[job.id] = { root, dir, files: listFiles(path.join(root, dir), 2) };
      }
      out[p.name] = dirs;
      // every facet's output folder: graphs/facets/facet_NNN_<name> (its exact name, layout.py)
      const froot = path.join(p.graphs_dir || path.dirname(p.graph_jobs_dir), 'facets');
      const fdirs = {};
      for (const f of p.graph_facets || []) {
        if (f.dir && fs.existsSync(path.join(froot, f.dir))) fdirs[f.id] = { root: froot, dir: f.dir, files: listFiles(path.join(froot, f.dir), 2) };
      }
      facets[p.name] = fdirs;
    }
    this.jobDirs = out;
    this.facetDirs = facets;
  }

  /** The run registry, parsed again only when its mtime changed; a half-written file keeps the last good copy. */
  registry(file) {
    const mtime = mtimeMs(file);
    const held = this.registries.get(file);
    if (mtime === null) { this.registries.delete(file); return null; }
    if (held && held.mtime === mtime) return held.data;
    try {
      const data = JSON.parse(fs.readFileSync(file, 'utf8'));
      if (data && typeof data === 'object') delete data._tfseal;       // the file's seal, not a run (utils/seal.py)
      this.registries.set(file, { mtime, data });
      return data;
    } catch (e) {
      return held ? held.data : null;
    }
  }

  runDir(project, run) {
    // registry `path` is relative to projects/ (e.g. DLSCA_ASCAD/artifacts/dev/generated_models/<name>)
    return run.path ? path.join(path.dirname(project.path), run.path)
      : path.join(project.artifacts, run.env || 'dev', 'generated_models', run.name);
  }

  /** [{env, hb}] for every artifacts/<env>/heartbeat.json of the project. */
  heartbeats(p, reg = this.registry(p.registry)) {
    let envs = [];
    try { envs = fs.readdirSync(p.artifacts, { withFileTypes: true }).filter(e => e.isDirectory()).map(e => e.name); } catch (e) { /* no runs yet */ }
    const hbs = envs.map(env => ({ env, hb: S.parseHeartbeat(readText(path.join(p.artifacts, env, 'heartbeat.json'))) }))
      .filter(h => h.hb);
    // a 'running' beat of a deleted run, or with no TraceFlow process left, is shown as stale
    return S.believedBeats(hbs, { registry: reg, busy: this.polledBusy ? this.busy : null });
  }

  /** The ENV this project follows now ({env, source}): the commands decide, the setting only when set. */
  followed(p, reg = this.registry(p.registry), hbs = this.heartbeats(p, reg)) {
    return S.followEnv({ override: this.envOverride, heartbeats: hbs, registry: reg, commands: this.busy || [], polled: !!this.polledBusy,
                         project: p.name, snapshot: this.latestSnapshot(p), nowMs: Date.now() });
  }

  /**
   * The newest params_out snapshot of the project ({env, name}): its `env:` key says the ENV the latest run
   * was made in. Looked for under artifacts/<env>/generated_models/<name>/config/; re-read every 15 s.
   */
  latestSnapshot(p) {
    if (!p || !p.path) return null;
    this.snapCache = this.snapCache || {};
    const held = this.snapCache[p.name];
    if (held && Date.now() - held.at < 15000) return held.value;
    let best = null;
    for (const env of ['dev', 'staging', 'prod']) {
      const models = path.join(p.path, 'artifacts', env, 'generated_models');
      let runs = [];
      try { runs = fs.readdirSync(models); } catch (e) { continue; }
      for (const name of runs) {
        const file = path.join(models, name, 'config', `${name}_params.yaml`);
        const t = mtimeMs(file);
        if (t && (!best || t > best.t)) best = { t, file, name };
      }
    }
    let value = null;
    if (best) {
      let text = '';
      try { text = fs.readFileSync(best.file, 'utf8'); } catch (e) { /* gone meanwhile */ }
      const m = text.match(/^env:\s*['"]?(dev|staging|prod)['"]?\s*$/m);
      if (m) value = { env: m[1], name: best.name };
    }
    this.snapCache[p.name] = { at: Date.now(), value };
    return value;
  }

  /** Housekeeping's filters of a project ({ENV, MODEL_ID, …}), kept in the workspace state. */
  delFilters(name) {
    const all = (this.context && this.context.workspaceState && this.context.workspaceState.get('traceflow.deleteFilters')) || {};
    return all[name] || {};
  }

  async setDelFilters(name, filters) {
    if (!this.context || !this.context.workspaceState) return;
    const all = { ...(this.context.workspaceState.get('traceflow.deleteFilters') || {}) };
    const kept = {};
    for (const k of S.RUN_FIELDS) if (String((filters || {})[k] || '').trim()) kept[k] = String(filters[k]).trim();
    all[name] = kept;
    await this.context.workspaceState.update('traceflow.deleteFilters', all);
    this.tick();
  }

  /**
   * The dataset a run was trained on: data.name of its params_out snapshot
   * (artifacts/<env>/generated_models/<name>/config/<name>_params.yaml). Kept once known; '' (no snapshot
   * yet) is looked for again after 30 s.
   */
  /** A registered run's params_out snapshot: artifacts/<env>/generated_models/<name>/config/<name>_params.yaml. */
  snapshotOf(p, r) {
    return path.join(p.artifacts || '', r.env || 'dev', 'generated_models', r.name, 'config', `${r.name}_params.yaml`);
  }

  /**
   * Housekeeping's PARAM filter (key.path=value[,…][; …]) over the rows the other filters left. The
   * snapshot values come from layout.py params, asked once per filter text and run (a snapshot is sealed);
   * meanwhile the rows already answered show and pending says more are being read.
   */
  paramFilter(p, rows, where) {
    this.paramCache = this.paramCache || new Map();
    const key = `${p.name}\n${where}`;
    const held = this.paramCache.get(key) || { match: {}, resolved: [], error: '', pending: false };
    this.paramCache.set(key, held);
    const missing = rows.filter(r => !(r.hash in held.match));
    if (missing.length && !held.pending && !held.error) {
      held.pending = true;
      const files = missing.map(r => this.snapshotOf(p, r));
      runPy(this.root, this.python, 'layout', ['params', '--where', where, ...files.flatMap(f => ['--snapshot', f])])
        .then(out => {
          held.error = out.error || '';
          held.resolved = out.resolved || [];
          missing.forEach((r, i) => { held.match[r.hash] = !!(out.match || {})[files[i]]; });
        })
        .catch(e => { held.error = `PARAM: ${e.message || e}`; })
        .finally(() => { held.pending = false; this.tick(); });
    }
    if (held.error) return { rows: [], error: held.error, note: '' };
    const note = held.pending ? 'reading PARAM from the run snapshots…'
      : held.resolved.length ? `PARAM: ${held.resolved.join('; ')}` : '';
    return { rows: rows.filter(r => held.match[r.hash] === true), error: '', note };
  }

  runDataset(p, r) {
    this.datasets = this.datasets || new Map();
    const held = this.datasets.get(r.hash);
    if (held && (held.value || Date.now() - held.at < 30000)) return held.value;
    const file = this.snapshotOf(p, r);
    const value = p.artifacts ? S.snapshotDataset(readText(file)) : '';
    this.datasets.set(r.hash, { value, at: Date.now() });
    return value;
  }

  /** A run tile's sparkline: rank at the last trace count, per epoch, of its last validation rank curve. */
  runSpark(project, run) {
    const held = this.sparks.get(run.hash);
    if (held && held.status === run.status && (run.group === 'done' || run.group === 'failed')) return held;
    const curve = H.runCurveDir(path.join(this.runDir(project, run), 'metrics'));
    const ranks = curve ? H.rankByEpoch(curve, this.rankCache).map(r => r.rank) : [];
    const out = { status: run.status, spark: H.sparkPath(ranks, SPARK.w, SPARK.h),
                  last: ranks.length ? ranks[ranks.length - 1] : null, epochs: ranks.length };
    this.sparks.set(run.hash, out);
    return out;
  }

  /**
   * The running run's latest metrics per process (train, validation, test when attacked) with their
   * change since the epoch before: core/monitor.snapshot, read again in the background only when one of
   * the run's history files changes. null until the first read is back.
   */
  liveMetrics(p, hash) {
    const r = S.registryEntries(this.registry(p.registry)).find(x => x.hash === hash);
    if (!r) return null;
    const dir = this.runDir(p, r);
    // its history rows, and while it tests the attack progress beside the heartbeat (utils/attack_progress.py)
    const progress = path.join(dir, '..', '..', 'attack_progress.json');
    const mtime = Math.max(0, ...H.historyFiles(path.join(dir, 'metrics')).map(f => mtimeMs(f) || 0), mtimeMs(progress) || 0);
    const held = this.liveOf.get(hash) || {};
    if (mtime && held.mtime !== mtime && !held.pending) {
      this.liveOf.set(hash, { ...held, pending: true });
      runModule(this.root, this.python, 'core.monitor', ['--run-dir', dir, '--json'])
        .then(out => { this.liveOf.set(hash, { mtime, live: out }); })
        .catch(() => { this.liveOf.set(hash, { mtime, live: held.live || null }); });
    }
    return held.live || null;
  }

  /** The project's sweep ([{position, epochs, passes}]), read again in the background when params.yaml changes. */
  trials(p) {
    const file = path.join(p.config_dir, 'params.yaml');
    const mtime = mtimeMs(file);
    const held = this.trialsOf[p.name] || {};
    if (mtime !== null && held.mtime !== mtime && !held.pending) {
      this.trialsOf[p.name] = { ...held, pending: true };
      runPy(this.root, this.python, 'layout', ['trials', '--project', p.name])
        .then(out => { this.trialsOf[p.name] = { mtime, trials: out.trials || [] }; })
        .catch(e => { this.trialsOf[p.name] = { mtime, trials: held.trials || [], error: e.message || String(e) }; });
    }
    return held.trials || [];
  }

  /**
   * The sweep size of one line of each running queue ({key: n}), counted from the params.yaml the queue froze at its
   * start (config/.frozen/<key>-<stamp>/, the copy every line of it reads): so the queue's total is right from the
   * start, before its first run registers. Counted once per frozen copy, in the background (layout plan with
   * TRACEFLOW_FROZEN_CONFIG naming the copy, as the run itself reads it).
   */
  frozenPlans(p) {
    const out = {};
    this.frozenPlanOf = this.frozenPlanOf || {};
    const root = path.join(p.config_dir, '.frozen');
    let dirs = [];
    try { dirs = fs.readdirSync(root, { withFileTypes: true }).filter(d => d.isDirectory()).map(d => d.name); } catch (e) { return out; }
    for (const key of S.activeQueues(this.busy || [])) {
      const dir = dirs.filter(d => d.startsWith(`${key}-`)).sort().pop();        // the newest: this execution's
      if (!dir) continue;
      const rel = path.relative(p.path, path.join(root, dir));
      const held = this.frozenPlanOf[rel];
      if (held && held.per != null) out[key] = held.per;
      else if (!held) {
        this.frozenPlanOf[rel] = { pending: true };
        // the queue's first line's DATASET: an entry's profiling_split.random_state list makes a run per split, and a
        // queue's lines share one such count (the panel sizes a queue as lines x one line's runs)
        const first = ((p.queues || []).find(q => q.key === key) || { commands: [] }).commands[0] || {};
        const args = ['plan', '--project', p.name, ...(first.DATASET ? ['--dataset', String(first.DATASET)] : [])];
        runPy(this.root, this.python, 'layout', args, { TRACEFLOW_FROZEN_CONFIG: rel, TRACEFLOW_FROZEN_PROJECT: p.name })
          .then(o => { this.frozenPlanOf[rel] = { per: o.per_command == null ? null : Number(o.per_command) }; })
          .catch(() => { this.frozenPlanOf[rel] = { per: null }; });
      }
    }
    return out;
  }

  /** Why the ETA row cannot be complete: shown under it only when something is wrong, never while it simply estimates. */
  etaNote(p, hb) {
    if (!hb) return `no heartbeat for ENV=${this.followed(p).env || 'dev'}: the run has not beaten yet, or it died`;
    const held = this.trialsOf[p.name] || {};
    if (held.error) return `batch ETA unavailable: the sweep of params.yaml did not load (${held.error})`;
    return '';
  }

  /** When the running run and its batch end (state.eta), from its heartbeat. */
  etaOf(p, hb) {
    if (!hb) return null;
    const batches = Array.isArray(hb.batches) ? hb.batches : [];
    const q = p.queues.find(x => batches.includes(x.key));
    let commandsLeft = 0;
    if (q) {                                        // the queue's commands after the one running now
      const i = q.commands.findIndex(c => c.MODEL_ID === hb.model_id && (!c.DATASET || c.DATASET === hb.dataset));
      commandsLeft = i >= 0 ? q.commands.length - i - 1 : 0;
    }
    const now = Date.now();
    const reg = this.registry(p.registry);
    const e = S.eta({ hb, nowMs: now, trials: this.trials(p), position: hb.trial, commandsLeft,
                      runsLeft: S.batchRunsLeft(p.queues, reg, hb, { perCommand: this.frozenPlans(p) }), lastRunS: S.lastRunSeconds(reg, hb),
                      batch: S.batchRuns(p.queues, reg, hb), runs: S.batchRunCount(p.queues, reg, hb, { perCommand: this.frozenPlans(p) }) });
    // a run's first seconds (no progress yet) tell nothing: the batch's last end, for the same batch, stands meanwhile
    this.lastBatchEta = this.lastBatchEta || {};
    const key = batches.join(',');
    const held = this.lastBatchEta[p.name];
    if (e.batchEndMs) this.lastBatchEta[p.name] = { key, endMs: e.batchEndMs };
    else if (held && held.key === key && key && held.endMs > now) Object.assign(e, { batchEndMs: held.endMs, batchLeftS: (held.endMs - now) / 1000 });
    return e;
  }

  /** The test's attack progress (utils/attack_progress.py: artifacts/<env>/attack_progress.json) of a run, read
   *  directly: no Python call, so a test of a second still shows; null when the file is another run's or absent. */
  attackOf(p, env, hash) {
    let body = null;
    try { body = JSON.parse(String(readText(path.join(p.artifacts, env || 'dev', 'attack_progress.json')) || '').split('\n')[0]); } catch { return null; }
    if (!body || body.model_hash !== hash || !Number(body.attacks)) return null;
    return { attack: Number(body.attack) || 0, attacks: Number(body.attacks), running: !!body.running,
             startedMs: Number(body.started_ns) / 1e6 || null, writtenMs: Number(body.written_ns) / 1e6 || null };   // the test's speed (eta)
  }

  projectState(p, nowMs) {
    const reg = this.registry(p.registry);
    const hbs = this.heartbeats(p, reg);
    const follow = this.followed(p, reg, hbs);
    const hb = (hbs.find(h => h.env === follow.env) || {}).hb || null;
    const perCommand = this.frozenPlans(p);       // running queues: their sweep size from their start
    // the beating run stays under Running while its test runs (its training is recorded DONE before the test)
    const beating = hb && hb.status === 'running' ? String(hb.model_hash || '') : '';
    const tiles = S.tiles(reg, p.queues, { active: S.activeQueues(this.busy), perCommand, beating });
    // Housekeeping's list: every run of the registry, newest first, ticked in the panel to be deleted
    const housekeeping = S.registryEntries(reg).sort((a, b) => String(b.started).localeCompare(String(a.started)))
      .map(r => ({ hash: r.hash, name: r.name, env: r.env, model: r.model, dataset: this.runDataset(p, r),
                   status: r.status, group: r.group, batches: r.batches }));
    const delFilters = this.delFilters(p.name);
    // the menus of the filters: every value the registry's runs hold (a model or dataset params.yaml
    // no longer lists included, so its runs can still be found and deleted), beside params.yaml's
    const uniq = xs => [...new Set(xs.filter(Boolean).map(String))].sort();
    const delOptions = {
      ENV: [...S.SNAP_ENVS],
      MODEL_ID: uniq([...(p.model_ids || []), ...housekeeping.map(r => r.model)]),
      DATASET: uniq([...(p.datasets || []), ...housekeeping.map(r => r.dataset)]),
      STATUS: uniq(housekeeping.flatMap(r => [String(r.status || '').toLowerCase(), r.group])),
      BATCH: uniq(housekeeping.flatMap(r => r.batches || [])),
    };
    let shown = S.runFilter(housekeeping, delFilters, { modelIds: delOptions.MODEL_ID, datasets: delOptions.DATASET });
    if (!shown.error && delFilters.PARAM) shown = this.paramFilter(p, shown.rows, delFilters.PARAM);
    // A run's trial number within its queue (k of the queue's planned n), by start order: right for
    // every row, also the older ones whose registry trial_index was numbered per pass.
    const pos = S.queuePositions(p.queues, reg, { perCommand });
    const trialOf = r => (pos[r.hash] ? { trial: pos[r.hash].k, trials: pos[r.hash].n, queue: pos[r.hash].queue }
      : { trial: r.trial, trials: r.trials, queue: '' });
    for (const g of Object.keys(tiles)) {
      tiles[g] = tiles[g].map(r => {
        if (r.pending) return { pending: true, hash: '', name: r.name, model: '', status: r.status, queue: r.queue };
        const s = this.runSpark(p, r);
        return { hash: r.hash, name: r.name, model: r.model, env: r.env, status: r.status, reason: r.reason, batches: r.batches,
                 started: r.started,                    // the list's "created" sort
                 ...trialOf(r), spark: s.spark, last: s.last, epochs: s.epochs,
                 ranks: g === 'done' ? this.runRanks(p, r) : null,
                 dataset: g === 'done' ? this.runDataset(p, r) : '',   // its data entry: the Budget box's attack_trace_budget
                 alive: g === 'running' ? S.runAlive(hb, r.hash, nowMs) : null };
      });
    }
    const menus = this.choicesFor(p, tiles.done.filter(r => !r.pending).map(r => r.name));
    tiles.done = tiles.done.map(r => (r.pending ? r : { ...r, choices: menus[r.name] || null }));
    const running = tiles.running[0] || null;
    let light = S.light(hb, nowMs);
    let run = hb && hb.status === 'running' ? { ...S.runSummary(hb), hash: hb.model_hash || '' } : null;
    if (run && pos[run.hash]) Object.assign(run, { trial: pos[run.hash].k, trials: pos[run.hash].n, queue: pos[run.hash].queue });
    if (!run && running) {
      // The registry says a run is in progress but no heartbeat is beating for this ENV.
      light = { color: 'grey', label: 'no heartbeat' };
      run = { name: running.name, model: running.model, dataset: '', hash: running.hash, percent: null,
              phase: hb && hb.stale ? `stale heartbeat: ${hb.stale}` : `no heartbeat for ENV=${follow.env}`, batches: running.batches,
              trial: running.trial, trials: running.trials, queue: running.queue };
    }
    if (run) run.alive = S.runAlive(hb, run.hash, nowMs);
    // the run card (listed under [Running] while the run is there): the test's attack bar in the phase's place,
    // the live block under it
    if (run && run.hash) {
      if (hb && hb.phase === 'test') run.attack = this.attackOf(p, follow.env, run.hash);
      run.live = this.liveMetrics(p, run.hash);
    }
    const graphsRoot = p.graphs_dir || path.dirname(p.graph_jobs_dir);
    const figures = S.figureStates(p.graph_jobs, this.jobDirs[p.name] || {}).map(f => {
      const filePaths = f.dir ? f.files.map(x => path.join(f.root, f.dir, x)) : [];
      const pdf = filePaths.find(x => /\.pdf$/i.test(x)) || filePaths[0];
      // a facet panel's address: under graphs/ relative to it, a run's figure relative to the project (artifacts/...)
      const base = pdf && inside(graphsRoot, pdf) ? graphsRoot : p.path;
      return { ...f, dirPath: f.dir ? path.join(f.root, f.dir) : null, filePaths,
        address: pdf ? path.relative(base, pdf).split(path.sep).join('/') : '' };
    });
    const facets = S.figureStates(p.graph_facets, (this.facetDirs || {})[p.name] || {}).map(f => {
      const filePaths = f.dir ? f.files.map(x => path.join(f.root, f.dir, x)) : [];
      return { ...f, dirPath: f.dir ? path.join(f.root, f.dir) : null, filePaths };
    });
    const sname = C.session(p.name);
    const s = this.sessions[sname];
    const session = s ? { name: sname, command: s.command, busy: s.busy } : null;
    this.runBatches = { ...(this.runBatches || {}), [p.name]: run ? run.batches || [] : [] };
    this.lastRunOf = { ...(this.lastRunOf || {}), [p.name]: run ? run.name : '' };
    const held = this.runSessionFor(p);
    return {
      name: p.name, env: follow.env, envSource: follow.source, path: p.path, configDir: p.config_dir, graphJobsDir: p.graph_jobs_dir, session,
      runSession: held ? held.session : null,
      runAttached: held ? held.attached : 0,
      modelIds: p.model_ids, datasets: p.datasets,
      singleRuns: p.single_runs || [], singleRunsFile: p.single_runs_file || '',
      graphJobs: (p.graph_jobs || []).map(j => ({ id: j.id, metric: j.metric, source: j.source, processes: j.processes, models: j.models, runs: j.runs,
        exported: !!j.exported })),
      light, run, tiles, tests: this.testsState(p, reg), housekeeping: { runs: shown.rows, total: housekeeping.length, error: shown.error || '', note: shown.note || '', filters: delFilters, options: delOptions },
      queues: S.queueStates(p.queues, reg, { perCommand }), figures, facets, eta: run ? { ...(this.etaOf(p, hb) || {}), note: this.etaNote(p, hb) } : null,
      configs: p.configs, runNotes: Object.fromEntries((p.configs || []).map(c => [c.path, this.runNote(c.path)]).filter(([, r]) => r)),
      errors: p.errors.concat(reg === null ? [`no run registry yet: ${p.registry}`] : []),
    };
  }

  gpuState() {
    return {
      docker: this.dockerState(),
      error: this.gpuError,
      gpus: this.gpus.map(g => {
        const h = this.gpuHistory.of(g.index);
        return { ...g, utilSpark: H.sparkPath(h.util, SPARK.w, SPARK.h), memSpark: H.sparkPath(h.mem, SPARK.w, SPARK.h) };
      }),
    };
  }

  push() {
    const now = Date.now();
    if (this.view && this.view.visible) {
      const msg = { type: 'state', error: this.error, gpu: this.gpuState(), busy: this.busy,
                    timezone: (this.layout && this.layout.timezone) || 'UTC', snapshots: this.snapState(), master: this.masterState() };
      if (this.layout) {
        msg.projects = this.layout.projects.map(p => this.projectState(p, now));
        msg.errors = this.layout.errors.concat(this.scaffoldError ? [this.scaffoldError] : []);
        msg.orphans = this.layout.orphans || [];            // project folders master.yaml does not declare
      }
      msg.menus = this.menus();
      msg.build = this.build();
      this.post(this.view, msg);
    }
    if (this.detail && this.detailRun) this.post(this.detail, this.detailState());
    this.lastReported = '';
  }

  /** Every curve of one run: each arrays/<process>/<metric>/ folder and each numeric field of its histories. */
  detailState() {
    const { project: name, hash } = this.detailRun;
    const p = this.layout && this.layout.projects.find(x => x.name === name);
    const run = p ? S.registryEntries(this.registry(p.registry)).find(r => r.hash === hash) : null;
    if (!p || !run) return { type: 'detail', missing: `${name} · ${hash}: not in the run registry` };
    const dir = this.runDir(p, run);
    const metrics = path.join(dir, 'metrics');
    const series = H.arrayDirs(metrics).slice(0, MAX_SERIES).map(d => {
      const values = H.rankByEpoch(d.dir, this.rankCache).map(r => r.rank);
      return { label: `${d.label} (last row)`, spark: H.sparkPath(values, DETAIL_SPARK.w, DETAIL_SPARK.h),
               last: values.length ? values[values.length - 1] : null, n: values.length, dir: d.dir };
    });
    for (const file of H.historyFiles(metrics)) {
      let held = this.readers.get(file);
      if (!held) { held = { reader: new H.JsonlReader(file), records: [] }; this.readers.set(file, held); }
      const fresh = held.reader.readNew();
      if (held.reader.rewound) held.records = [];                   // the file was rewritten (e.g. FORCE=1)
      held.records.push(...fresh);
      const rel = path.relative(metrics, path.dirname(path.dirname(file))).replace(/^cross_validation\/?/, '') || 'run';
      for (const [key, values] of Object.entries(H.scalarSeries(held.records))) {
        if (series.length >= MAX_SERIES) break;
        series.push({ label: `${rel} · ${path.basename(file, '.jsonl')} · ${key}`,
                      spark: H.sparkPath(values, DETAIL_SPARK.w, DETAIL_SPARK.h), last: values[values.length - 1], n: values.length, dir: path.dirname(file) });
      }
    }
    const [{ hb }] = S.believedBeats([{ hb: S.parseHeartbeat(readText(path.join(p.artifacts, run.env || 'dev', 'heartbeat.json'))) }],
      { busy: this.polledBusy ? this.busy : null });
    const live = hb && hb.model_hash === hash ? { ...S.runSummary(hb), light: S.light(hb, Date.now()) } : null;
    return { type: 'detail', project: name, run: { ...run, dir }, live, series };
  }

  /** A webview's page: panel (the side panel, the detail tab) or planner (Plan a Model: planner.html with
   *  planner_core.js), its scripts and stylesheet as webview URIs under the CSP's nonce. */
  html(webview, mode, page = 'panel') {
    const base = path.join(this.context.extensionPath, 'webview');
    // each file's URI carries a digest of its contents (?v=): the webview's resource cache keeps a file by its
    // URL across reloads, so a changed file (a reinstalled panel) must have a new one, or the old script runs
    const uri = f => `${webview.asWebviewUri(vscode.Uri.file(path.join(base, f))).toString()}?v=${this.digest(path.join(base, f))}`;
    const nonce = crypto.randomBytes(16).toString('hex');
    return fs.readFileSync(path.join(base, `${page}.html`), 'utf8')
      .replace(/\{\{css\}\}/g, uri(`${page}.css`)).replace(/\{\{js\}\}/g, uri(`${page}.js`)).replace(/\{\{menus\}\}/g, uri('menus.js'))
      .replace(/\{\{core\}\}/g, uri('planner_core.js'))
      .replace(/\{\{mode\}\}/g, mode).replace(/\{\{csp\}\}/g, webview.cspSource).replace(/\{\{nonce\}\}/g, nonce);
  }

  /** Load Blueprint, asked first (owner 2026-10-24): only OK lets the panel stage the blueprint's values
   *  and write them; Cancel (or closing the dialog) changes nothing. */
  async completeAsk(m) {
    const file = String(m.file || '');
    if (!this.root || !inside(this.root, file)) return;
    const unsaved = Number(m.unsaved) || 0;
    const ok = await say.warn(COMPLETE_WARNING, { modal: true,
      detail: `${path.basename(file)}: every key the blueprint has and the file lacks, and every empty value, gets the ` +
        'blueprint\'s value, and the file is written at once. Values the file already sets are kept.' +
        (unsaved ? ` The form's ${unsaved} unsaved change${unsaved === 1 ? '' : 's'} are written with them.` : '') }, 'OK');
    if (ok === 'OK') this.post(this.view, { type: 'complete-ok', file });
  }

  /** The first 10 hex digits of a file's SHA-1 (a missing file: 0): the ?v= of its webview URI, and the build. */
  digest(file) {
    try { return crypto.createHash('sha1').update(fs.readFileSync(file)).digest('hex').slice(0, 10); } catch (e) { return '0'; }
  }

  /** What is running: the extension's version and one digest over extension.js and webview/panel.js, shown in the
   *  panel so a reinstall can be told from a stale window (reinstall, then Developer: Reload Window; the stamp must change). */
  build() {
    if (!this.buildStamp) {
      let version = '';
      try { version = JSON.parse(fs.readFileSync(path.join(this.context.extensionPath, 'package.json'), 'utf8')).version || ''; } catch (e) { version = ''; }
      const files = ['extension.js', path.join('webview', 'panel.js')].map(f => path.join(this.context.extensionPath, f));
      let code = '';
      try { code = crypto.createHash('sha1').update(Buffer.concat(files.map(f => fs.readFileSync(f)))).digest('hex').slice(0, 10); } catch (e) { code = '0'; }
      this.buildStamp = `${version} ${code}`.trim();
    }
    return this.buildStamp;
  }

  webviewOptions() {
    return { enableScripts: true, localResourceRoots: [vscode.Uri.file(path.join(this.context.extensionPath, 'webview'))] };
  }

  resolveWebviewView(view) {
    this.view = view;
    view.webview.options = this.webviewOptions();
    view.webview.html = this.html(view.webview, 'main');
    view.webview.onDidReceiveMessage(m => this.onMessage(m));
    view.onDidChangeVisibility(() => this.tick());
    view.onDidDispose(() => { this.view = null; });
    this.tick();
  }

  onMessage(m) {
    if (!m || typeof m !== 'object') return;
    if (m.type === 'retry') this.restart().catch(e => this.report(e));
    else if (m.type === 'detail') this.openDetail(String(m.project), String(m.hash));
    else if (m.type === 'open' && this.root && inside(this.root, m.path)) this.open(m.path);
    else if (m.type === 'reveal' && this.root && inside(this.root, m.path)) this.reveal(m.path);
    else if (m.type === 'tree') {
      const file = this.configFile(m.file);
      if (!file) return;
      if (m.open === false) { this.openTrees.delete(file); return; }
      this.openTrees.add(file);
      this.sendTree(file);
    } else if (m.type === 'want-choices') this.wantChoices(m);
    else if (m.type === 'edit') this.edit(m).catch(e => this.report(e));
    else if (m.type === 'apply') this.apply(m).catch(e => this.report(e));
    else if (m.type === 'complete-ask') this.completeAsk(m).catch(e => this.report(e));
    else if (m.type === 'add-queue') this.addQueue(m).catch(e => this.report(e));
    else if (m.type === 'add-job') this.addJob(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'add-model') this.addModel(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'import-model') this.importModel(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'delete-entry') this.deleteEntry(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'add-project') this.addProject(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'remove-project') this.removeProject(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'check-model') this.checkModel(String(m.project || ''), String(m.model_id || '')).catch(e => this.report(e));
    else if (m.type === 'plan-model') this.openPlanner(m.project, m.file).catch(e => this.report(e));
    else if (m.type === 'status-open') this.statusSend().catch(e => this.report(e));
    else if (m.type === 'status-start') this.statusStart().catch(e => this.report(e));
    else if (m.type === 'status-stop') this.statusStop().catch(e => this.report(e));
    else if (m.type === 'status-add') this.statusShare('add', String(m.name || '')).catch(e => this.report(e));
    else if (m.type === 'status-remove') this.statusShare('remove', String(m.name || '')).catch(e => this.report(e));
    else if (m.type === 'status-visit') this.statusVisit(String(m.link || '')).catch(e => this.report(e));
    else if (m.type === 'status-copy') vscode.env.clipboard.writeText(String(m.link || '')).then(() => say.info(`TraceFlow: ${m.name}'s link copied.`));
    else if (m.type === 'visualise') this.openPlanner(m.project, m.file, String(m.model_id || '')).catch(e => this.report(e));
    else if (m.type === 'stage-refused') {
      const file = this.configFile(m.file) || m.file;
      for (const pl of Object.values(this.planners || {}).filter(x => x.file === file)) {
        this.post(pl.panel, { type: 'refused', message: `refused: nothing staged: ${String(m.reason || '')}` });
      }
    } else if (m.type === 'staged') {
      const file = this.configFile(m.file) || m.file;
      this.staged = { ...(this.staged || {}), [file]: Number(m.count) || 0 };
      this.stagedArranged = { ...(this.stagedArranged || {}), [file]: Number(m.arranged) || 0 };
    }
    else if (m.type === 'add-facet') this.addFacet(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'find-figure') this.findFigure(m).catch(e => this.report(e));
    else if (m.type === 'pick-path') this.pickPath(m).catch(e => this.report(e));
    else if (m.type === 'filter-fields') this.filterFields(String(m.project || '')).catch(e => this.report(e));
    else if (m.type === 'find-models') this.findModels(String(m.project || ''), m.where).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    else if (m.type === 'run') {
      this.run(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    } else if (m.type === 'snap') {
      this.snap(m).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    } else if (m.type === 'delfilter') {
      this.setDelFilters(String(m.project), m.filters).catch(e => this.report(e));
    } else if (m.type === 'figdel-all') {
      this.deleteFigures(String(m.project), m.dirs).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    } else if (m.type === 'jobdel') {
      this.deleteJob(String(m.project), m.id).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    } else if (m.type === 'facetdel') {
      this.deleteFacet(String(m.project), m.id).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    } else if (m.type === 'figdel') {
      this.deleteFigure(String(m.project), String(m.dir)).catch(e => say.error(`TraceFlow: ${e.message || e}`));
    } else if (m.type === 'docker-kill-build') {
      this.dockerKillBuild().catch(e => say.error(`TraceFlow: ${e.message || e}`));
    } else if (m.type === 'docker-teardown') {
      this.dockerTeardown().catch(e => say.error(`TraceFlow: ${e.message || e}`));
    }
  }

  // ── buttons ────────────────────────────────────────────────────────────────

  project(name) {
    return ((this.layout && this.layout.projects) || []).find(p => p.name === name) || null;
  }

  /**
   * The ENV a command runs in: the followed one (a run, the running command, a snapshot); before anything
   * has run there is none to follow, so the user picks it. '' when the pick is dismissed.
   */
  async envFor(follow, what) {
    if (follow.env) return follow.env;
    const env = await choose(`ENV for ${what}?`,
      'Nothing has run here yet, so there is no ENV to follow. Choose the one this command runs in: its compose ' +
      'files, and the artifacts/<ENV>/ folder its runs write to.', ['dev', 'staging', 'prod']);
    return env || '';
  }

  async run(m) {
    const p = this.project(String(m.project));
    if (!p) return;
    const follow = this.followed(p);
    const vars = { ENV: follow.env, PROJECT: p.name };          // the ENV the commands point at (see followEnv; '' before any run)
    const name = C.session(p.name);
    switch (m.action) {
      case 'orchestrate': {
        // No command line to follow here: ask, the followed ENV first.
        const envs = [follow.env, ...['dev', 'staging', 'prod'].filter(e => e !== follow.env)].filter(Boolean);
        const env = await choose('ENV for make orchestrate?',
          (follow.env ? `The panel follows ENV=${follow.env}, from ${follow.source}.` : 'Nothing has run here yet: there is no ENV to follow.') +
          ' The ENV picks the compose files and the artifacts/<ENV>/ folder the runs write to.', envs);
        if (!env) return;
        vars.ENV = env;
        const model = await this.pick(p.model_ids, 'MODEL_ID (params.yaml model:)');
        if (!model) return;
        const dataset = await this.pick(p.datasets, 'DATASET (params.yaml data:)');
        if (!dataset) return;
        return this.startLong(p, 'orchestrate', { ...vars, DATASET: dataset, MODEL_ID: model }, 1);
      }
      case 'test': {                                              // Experiment Settings → test_config.yaml → Test: external tests
        vars.ENV = await this.envFor(follow, 'make test EXTERNAL=1');
        if (!vars.ENV) return undefined;
        return this.startLong(p, 'test', { ...vars, EXTERNAL: 1 }, null,
          'External tests of the models listed in test_config.yaml, with its perform_attack settings: reports in each ' +
          'run\'s test/<model_hash>_et_<index>/, outcomes in test_registry.json under external_tests. The recorded tests are not touched.');
      }
      case 'import': {                                            // Experiment Settings → import_config.yaml → Import
        vars.ENV = await this.envFor(follow, 'make import');
        if (!vars.ENV) return undefined;
        return this.startLong(p, 'import', vars, null,
          'Every model of import_config.yaml not DONE yet (new, failed or interrupted): its weights checked against the released ' +
          'model on every test trace, then the standard test. DONE ones are skipped. A failed import stops the run, naming it.');
      }
      case 'external-test':
        return this.externalTest(p, String(m.key || ''), m.kind === 'budget');
      case 'test-log': {
        const r = S.registryEntries(this.registry(p.registry)).find(x => x.hash === m.key);
        const log = r ? path.join(this.runDir(p, r), 'test', 'test_log.txt') : '';
        if (!log || !fs.existsSync(log)) return say.info(`TraceFlow: no external test log yet${r ? ` for ${r.name}` : ''}.`);
        return this.terminal(`TraceFlow · ${r.name} · tests`, C.tailLog(log), true);
      }
      case 'test-registry':
        return this.openAt(this.testRegistryFile(p), `"${String(m.key || '')}": {`);
      case 'single': {
        // RUN SINGLE EXPERIMENT → Start: line m.key of single_runs.yaml as the file holds it (layout reads it;
        // not the webview's text); it runs as the project it names (its session and logs), with a DATASET and a
        // MODEL_ID of that project's params.yaml
        const r = (p.single_runs || [])[Number(m.key)];
        if (!r) return say.warn(`TraceFlow: no single run ${m.key} in single_runs.yaml`);
        const q = r.error ? null : this.project(String(r.PROJECT || p.name));
        const bad = r.error || (!q ? `PROJECT=${r.PROJECT} is no project of master.yaml`
          : !(q.datasets || []).includes(r.DATASET) ? `DATASET=${r.DATASET} is no data entry of ${q.name}'s params.yaml`
          : !(q.model_ids || []).includes(r.MODEL_ID) ? `MODEL_ID=${r.MODEL_ID} is no model_id of ${q.name}'s params.yaml`
          : !['dev', 'staging', 'prod'].includes(r.ENV) ? `ENV=${r.ENV} is not dev, staging or prod` : '');
        if (bad) return say.warn(`TraceFlow: single run ${Number(m.key) + 1} not started: ${bad}.`, { modal: true, detail: r.line });
        return this.startLong(q, 'orchestrate', { ENV: r.ENV, PROJECT: q.name, DATASET: r.DATASET, MODEL_ID: r.MODEL_ID }, 1);
      }
      case 'queue': {
        const q = p.queues.find(x => x.key === m.key);
        if (!q) return say.warn(`TraceFlow: no queue ${m.key} in experiment_queue.yaml`);
        // A queue runs in the ENV its lines name; make experiment refuses lines naming another.
        if (q.envs && q.envs.length > 1) {
          return say.warn(`TraceFlow: the lines of queue ${q.key} name different ENVs ` +
            `(${q.envs.join(', ')}); make experiment runs one. Make them agree in experiment_queue.yaml.`, { modal: true });
        }
        const env = q.env || await this.envFor(follow, `queue ${q.key} (its lines name no ENV)`);
        if (!env) return undefined;
        return this.startLong(p, 'experiment', { ...vars, ENV: env, QUEUE: q.key }, q.commands.length);
      }
      case 'stop': {
        if (!this.sessions[name]) return say.info(`TraceFlow: no session ${name}.`);
        const ok = await say.warn(
          `Send Ctrl-C to ${name} (${this.sessions[name].command})? The running make stops; DONE experiments stay done.`, { modal: true }, 'Stop');
        if (ok === 'Stop') await bash(C.tmuxStop(name), this.root);
        return this.pollProcesses();
      }
      case 'attach':
        if (!this.sessions[name]) return say.info(`TraceFlow: no session ${name}.`);
        return this.terminal(`TraceFlow · ${name}`, C.tmuxAttach(name), true);
      case 'attach-run': {
        await this.pollProcesses();
        const s = this.runSessionFor(p);
        if (s) return this.terminal(`TraceFlow · ${s.session}`, C.tmuxAttach(s.session), true);
        // No session holds the run (its make chain ended, e.g. interrupted): the container may still be
        // training, so Attach shows its live output instead (a stopped container ends the viewer at once).
        say.info(`TraceFlow: no tmux session holds ${p.name}'s experiment; ` +
          'showing the training container\'s live log instead.');
        return this.terminal(`TraceFlow · ${p.name} · container log`, C.containerLogLine(), true);
      }
      case 'container-log':
        // The training container's live output, whoever started it and whether or not a session holds it.
        return this.terminal(`TraceFlow · ${p.name} · container log`, C.containerLogLine(), true);
      case 'detach-run': {
        await this.pollProcesses();
        const s = this.runSessionFor(p);
        if (!s) {
          const viewer = vscode.window.terminals.find(t => t.name === `TraceFlow · ${p.name} · container log`);
          if (viewer) { viewer.dispose(); return say.info('TraceFlow: closed the container log; the training goes on.'); }
          return say.info(`TraceFlow: no tmux session is running ${p.name}'s experiment.`);
        }
        if (!s.attached) return say.info(`TraceFlow: ${s.session} is already detached.`);
        await bash(`tmux detach-client -s ${C.word(s.session)}`, this.root);
        say.info(`TraceFlow: detached ${s.session}; the run keeps going. Attach brings it back.`);
        await this.pollProcesses();
        return this.tick();
      }
      case 'kill-run': {
        await this.pollProcesses();
        const s = this.runSessionFor(p);
        if (!s) return say.info(`TraceFlow: no tmux session is running ${p.name}'s experiment.`);
        const ok = await say.warn(
          `Are you sure you want to kill tmux session "${s.session}"?\n\nIt runs: ${s.cmds[0]}\n\n` +
          (s.make
            ? 'Its make chain and log stream end. The container may keep training; ' +
              `make stack-down ENV=${follow.env} stops it. Finished runs stay DONE.`
            : 'It only shows the container\'s log: the viewer stops, the training goes on.'),
          { modal: true }, 'Yes, kill');
        if (ok !== 'Yes, kill') return;
        await bash(`tmux kill-session -t ${C.word(s.session)}`, this.root);
        await this.pollProcesses();
        return this.tick();
      }
      case 'terminate-run': {
        const env = await this.envFor(follow, 'Terminate Run');
        return env ? this.terminate(p, env) : undefined;
      }
      case 'delete-database':
        return this.deleteDatabase(p, m.hashes);
      case 'delete-run':                                          // Failed tile → Delete: this run, whatever Housekeeping filters
        return this.deleteDatabase(p, [m.key], { filtered: false });
      case 'test-run':
        return this.testRun(p, String(m.key || ''), m.kind === 'budget');
      case 'graph-export':
        return this.graphExport(p, m.picks);
      case 'graph-remove':
        return this.graphRemove(p, String(m.name || ''), String(m.key || ''));
      case 'facet-export':
        return this.facetExport(p, m.files);
      case 'draw-all-facets': {                                   // FIGURES → Faceting → Draw all: the ticked facets, one after another
        const ids = (m.ids || []).map(Number).filter(Number.isInteger);
        if (!ids.length) return undefined;
        vars.ENV = await this.envFor(follow, `make graph FACE_ID=${ids.join(',')}`);
        if (!vars.ENV) return undefined;
        const lines = ids.map(id => C.makeLine('graph', { ...vars, FACE_ID: id }));
        this.terminal(`TraceFlow · ${p.name}`, C.shortLine(`{ ${lines.join('; ')}; }`, this.root), false);
        return undefined;
      }
      case 'draw-facet': {
        const id = Number(m.id);
        if (!Number.isInteger(id)) return undefined;
        vars.ENV = await this.envFor(follow, `make graph FACE_ID=${id}`);
        if (!vars.ENV) return undefined;
        const facet = (p.graph_facets || []).find(f => f.id === id);
        return this.runShort(p, `facet ${id}`, C.makeLine('graph', { ...vars, FACE_ID: id }), since => {
          // The facet's newest .pdf (else .svg, the default since 2026-10-27, else .png) in its own folder
          // graphs/facets/facet_NNN_<name> (layout.py), written since the click.
          const dir = path.join(p.graphs_dir || path.dirname(p.graph_jobs_dir), 'facets');
          const hit = facet && facet.dir && fs.existsSync(path.join(dir, facet.dir)) ? facet.dir : null;
          if (!hit) return null;
          const rank = f => ({ pdf: 0, svg: 1, png: 2 }[(f.match(/\.(\w+)$/) || [])[1].toLowerCase()]);
          const files = fs.readdirSync(path.join(dir, hit)).map(f => path.join(dir, hit, f))
            .filter(f => /\.(pdf|svg|png)$/i.test(f) && mtimeMs(f) >= since)
            .sort((a, b) => (rank(a) - rank(b)) || mtimeMs(b) - mtimeMs(a));
          return files[0] || null;
        });
      }
      case 'log': {
        const lq = p.queues.find(x => x.key === m.key);
        const log = C.logFile('queue', { QUEUE: m.key, ENV: (lq && lq.env) || follow.env || 'dev' }, p.folder);
        if (!fs.existsSync(path.join(this.root, log))) return say.info(`TraceFlow: ${log} does not exist yet.`);
        return this.terminal(`TraceFlow · ${log}`, C.tailLog(log), true);
      }
      case 'draw': {
        const id = Number(m.id);
        if (!Number.isInteger(id)) return;
        vars.ENV = await this.envFor(follow, `make graph GRAPH_ID=${id}`);
        if (!vars.ENV) return undefined;
        return this.runShort(p, `draw ${id}`, C.makeLine('graph', { ...vars, GRAPH_ID: id }), since => this.newestFigure(p, id, since, vars.ENV));
      }
      case 'study':
        return this.study(p, String(m.key || ''), follow.env || 'dev');
      case 'draw-all': {                                          // FIGURES → Draw all: the ticked jobs, one after another
        const ids = (m.ids || []).map(Number).filter(Number.isInteger);
        if (!ids.length) return undefined;
        vars.ENV = await this.envFor(follow, `make graph GRAPH_ID=${ids.join(',')}`);
        if (!vars.ENV) return undefined;
        // `{ a; b; }` after the env is set up: a job that fails does not stop the rest; the lights turn green as figures are written.
        const lines = ids.map(id => C.makeLine('graph', { ...vars, GRAPH_ID: id }));
        this.terminal(`TraceFlow · ${p.name}`, C.shortLine(`{ ${lines.join('; ')}; }`, this.root), false);
        return undefined;
      }
      case 'find':
        vars.ENV = await this.envFor(follow, 'make find_model');
        if (!vars.ENV) return undefined;
        return this.runShort(p, 'find', C.makeLine('find_model', vars), since => (mtimeMs(p.filtered_models) >= since ? p.filtered_models : null));
      case 'judge': {
        vars.ENV = await this.envFor(follow, 'make judge');
        if (!vars.ENV) return;
        // The judge ranks on what the runs recorded: asked here (process, then metric), written into
        // model_judge.yaml, then make judge reads the file. A shipped file naming a process no run has
        // (test, validation under cross-validation) rejected every run (audit 2026-10-23).
        if (await this.judgeChoice(p, vars.ENV) === false) return;
        if (!vars.ENV) return undefined;
        return this.runShort(p, 'judge', C.makeLine('judge', vars), since => (mtimeMs(p.judge_output) >= since ? p.judge_output : null));
      }
      default:
    }
  }

  /**
   * Terminate: stop the experiment itself, not only its viewer. First the tmux session holding its make
   * chain (else `make experiment` would start the queue's next command), then `make stack-down ENV=…`,
   * which stops the training container. In a terminal: stack-down runs nerdctl through sudo.
   */
  async terminate(p, env) {
    await this.pollProcesses();
    const s = this.runSessionFor(p);
    const run = (this.lastRunOf && this.lastRunOf[p.name]) || '';
    const ok = await say.warn(
      `Terminate Run ${run ? `${run} (` : ''}${p.name}${run ? ')' : ''} in ENV=${env}?\n\n` +
      (s ? `1. kill tmux session "${s.session}" (its make chain: ${s.cmds[0]})\n` : '1. no tmux session holds it\n') +
      `2. make stack-down ENV=${env}: stops the training container.\n\n` +
      'The run stops now and stays in the registry as in progress; the next make orchestrate resumes or ' +
      'retrains it. Finished runs stay DONE.',
      { modal: true }, 'Terminate Run');
    if (ok !== 'Terminate Run') return;
    const kill = s ? `{ tmux kill-session -t ${C.word(s.session)} 2>/dev/null || true; } && ` : '';
    this.terminal(`TraceFlow · ${p.name} · terminate`, `cd ${C.quote(this.root)} && ${kill}${C.makeLine('stack-down', { ENV: env })}`, true);
    await this.pollProcesses();
    return this.tick();
  }

  /** artifacts/test_registry.json beside the project's run_registry.json. */
  testRegistryFile(p) {
    return path.join(path.dirname(p.registry || ''), 'test_registry.json');
  }

  /** Open a file beside the panel with the line holding `needle` selected (else at its top). */
  async openAt(file, needle) {
    if (!fs.existsSync(file)) return say.info(`TraceFlow: ${file} does not exist yet.`);
    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(file));
    const line = Math.max(0, doc.getText().split('\n').findIndex(l => l.includes(needle)));
    const at = new vscode.Position(line, 0);
    return vscode.window.showTextDocument(doc, { viewColumn: vscode.ViewColumn.Beside, preview: false, selection: new vscode.Range(at, at) });
  }

  /**
   * TESTS → a model → Test: an external test, make test ENV=<run's ENV> MODEL=<run> EXTERNAL=1, with the
   * model's settings from test_config.yaml (layout.py external; refused with the reason when they do not
   * attack). Its tmux log goes to <run>/test/test_log.txt.
   */
  async externalTest(p, hash, budget = false) {
    const r = S.registryEntries(this.registry(p.registry)).find(x => x.hash === hash);
    if (!r) return say.warn(`TraceFlow: run ${hash} is not in the registry.`);
    if (r.group !== 'done') return say.warn(`TraceFlow: ${r.name} is ${r.status}: only a DONE run is tested.`, { modal: true });
    const out = await runPy(this.root, this.python, 'layout', ['external', '--project', p.name, '--model', r.name]);
    if (out.error) return say.warn(`TraceFlow: ${r.name} is not tested.\n\n${out.error}`, { modal: true });
    const a = out.settings || {};
    const t = a.test || {};
    const note = `Test ${budget ? 'Budget' : 'All'} of ${r.name} (${r.hash}), with test_config.yaml's attack (${out.config}):\n` +
      `perform_attack.test: attack_times ${t.attack_times}, shuffle ${t.shuffle}, ${t.attack_distribution}; ` +
      `${budget ? 'each attack its data entry\'s attack_trace_budget traces' : 'every test trace'}; ` +
      `attack_step ${a.attack_step == null ? 'the snapshot\'s' : a.attack_step}; ge_thresholds ${a.ge_thresholds == null ? 'the snapshot\'s' : JSON.stringify(a.ge_thresholds)}.\n` +
      `Reports in ${path.join(this.runDir(p, r), 'test')}/${r.hash}_et_<index>/; the outcome in test_registry.json → ${r.hash} → ` +
      `external_tests → ${r.hash}_et_<index>, appended (every test is a record of its own; the latest of each kind is the one read). The recorded test is not touched.`;
    return this.startLong(p, 'test', { ENV: r.env || this.followed(p).env, PROJECT: p.name, MODEL: r.name, EXTERNAL: 1, ...(budget ? { BUDGET: 1 } : {}) }, null, note,
      { log: path.join(this.runDir(p, r), 'test', 'test_log.txt') });
  }

  /**
   * TESTS: every DONE run of the project, its recorded test's status and its external tests
   * (test_registry.json[hash].external_tests), in index order.
   */
  testsState(p, reg) {
    const treg = this.registry(this.testRegistryFile(p)) || {};
    return S.registryEntries(reg).filter(r => r.group === 'done')
      .sort((a, b) => String(b.started).localeCompare(String(a.started)))
      .map(r => {
        const entry = treg[r.hash] || {};
        const tests = Object.entries(entry.external_tests || {}).filter(([, t]) => t && typeof t === 'object')
          .map(([id, t]) => ({ id, index: Number(t.index) || 0, status: String(t.status || ''), reason: String(t.status_reason || ''), rank: t.recovered_key_rank,
                               recovered: t.key_recovered, finished: t.finished_at || '', runs: Number(t.runs) || 0,
                               attack: (t.settings || {}).perform_attack || {},
                               kind: t.kind || ((((t.settings || {}).perform_attack || {}).test || {}).attack_trace_budget != null ? 'budget' : 'all'),
                               source: t.source || 'test', ranks: this.testRanks(p, r, id) }))
          .sort((x, y) => x.index - y.index);
        return { hash: r.hash, name: r.name, model: r.model, env: r.env, recorded: String(entry.status || ''), tests,
                 dataset: this.runDataset(p, r),            // its data entry: the Budget box's attack_trace_budget
                 ranks: this.safeRanks(p, r) };           // its summary line's four ranks, as the Done list shows them
      });
  }

  /**
   * Done tile → Retest All / the Budget box's Test: make test ENV=<the run's ENV> MODEL=<its name> [BUDGET=1], which tests the
   * model from its own snapshot, appended. Refused when that snapshot says perform_attack.test.attack_times: 0 (nothing to attack:
   * the run was configured without a test), read with layout.py params.
   */
  async testRun(p, hash, budget = false) {
    const r = S.registryEntries(this.registry(p.registry)).find(x => x.hash === hash);
    if (!r) return say.warn(`TraceFlow: run ${hash} is not in the registry.`);
    if (r.group !== 'done') return say.warn(`TraceFlow: ${r.name} is ${r.status}: only a DONE run is tested.`, { modal: true });
    const snap = this.snapshotOf(p, r);
    if (!fs.existsSync(snap)) {
      return say.warn(`TraceFlow: not tested. ${r.name} has no snapshot at\n${snap}`, { modal: true });
    }
    const out = await runPy(this.root, this.python, 'layout', ['params', '--where', 'perform_attack.test.attack_times=0', '--snapshot', snap]);
    if (!out.error && (out.match || {})[snap]) {
      return say.warn(`TraceFlow: ${r.name} is not tested.\n\nRefused: its snapshot says perform_attack: test: attack_times: 0, ` +
        `so its test traces are never attacked (the run was configured without a test).\n\n${snap}`, { modal: true });
    }
    return this.startLong(p, 'test', { ENV: r.env || this.followed(p).env, PROJECT: p.name, MODEL: r.name, ...(budget ? { BUDGET: 1 } : {}) }, null,
      `Retest ${budget ? 'Budget' : 'All'} of ${r.name} (${r.hash}) with its own snapshot's attack, ` +
      `${budget ? 'each attack its data entry\'s attack_trace_budget traces' : 'on every test trace'}: appended as a record of its own ` +
      '(test/<hash>_et_<index>/); the recorded test is not touched, and the latest of each kind is the one read.');
  }

  /**
   * Housekeeping: delete the runs ticked in the panel (Delete all ticks every one). DELETE runs
   * make delete_database's own dry run (JSON=1) for the ticked names, once per ENV among them, and shows
   * the experiments it would delete, their total and the runs per ENV in one dialog: OK runs the same
   * lines with CONFIRM=1 in a terminal, Cancel closes. Refused while an experiment runs here.
   */
  async deleteDatabase(p, hashes, { filtered = true } = {}) {
    const why = await this.blocked(C.session(p.name));
    if (why) return say.warn(`${why}\n\nDeleting the database under a running experiment is refused.`, { modal: true });
    const wanted = new Set((hashes || []).map(String));
    const filters = filtered ? this.delFilters(p.name) : {};
    let shown = S.runFilter(S.registryEntries(this.registry(p.registry)).map(r => ({ ...r, dataset: this.runDataset(p, r) })),
      filters, { modelIds: p.model_ids, datasets: p.datasets });
    if (!shown.error && filters.PARAM) {
      shown = this.paramFilter(p, shown.rows, filters.PARAM);
      if (/^reading/.test(shown.note)) {
        return say.info('TraceFlow: the PARAM filter is still reading the snapshots; DELETE again in a moment.', { modal: true });
      }
    }
    const runs = shown.rows.filter(r => wanted.has(r.hash));             // ticked and shown by the filters
    if (!runs.length) {
      return say.info('TraceFlow: nothing selected, nothing deleted. Tick the runs to delete ' +
        '(or Delete all), then DELETE.', { modal: true });
    }
    const noEnv = runs.filter(r => !r.env);
    if (noEnv.length) {
      return say.warn(`TraceFlow: the registry names no ENV for ${noEnv.map(r => r.name).join(', ')}; ` +
        'make delete_database deletes per ENV. Nothing deleted.', { modal: true });
    }
    const byEnv = {};
    for (const r of runs) (byEnv[r.env] = byEnv[r.env] || []).push(r.name);
    const plans = [];
    for (const [env, names] of Object.entries(byEnv)) {
      const knobs = { ENV: env, PROJECT: p.name, MODEL: names.join(',') };
      try {
        const out = await this.capture(C.envLine(`${C.makeLine('delete_database', { ...knobs, JSON: 1 })} 2>&1`));
        const plan = JSON.parse(out.split('\n').reverse().find(l => l.trim().startsWith('{')) || 'null');
        if (plan && plan.total_selected) plans.push({ knobs, plan });
      } catch (e) {
        return say.warn(`TraceFlow: the dry run (make delete_database … JSON=1) failed:\n${e.message || e}`, { modal: true });
      }
    }
    if (!plans.length) return say.info('TraceFlow: no experiment matches; nothing deleted.', { modal: true });
    const total = plans.reduce((n, x) => n + x.plan.total_selected, 0);
    const perEnvRuns = plans[0].plan.runs_per_env || {};
    const picked = Object.fromEntries(plans.map(x => [x.plan.env, x.plan.total_selected]));
    const perEnv = Object.entries(perEnvRuns).map(([e, n]) => `${e}: ${n}${picked[e] ? ` (${picked[e]} selected)` : ''}`).join(' · ');
    const all = Object.values(perEnvRuns).reduce((n, v) => n + v, 0);
    const rows = plans.flatMap(x => x.plan.selected.map(r => ({ ...r, env: x.plan.env })));
    const MAX = 30;
    const list = rows.slice(0, MAX).map(r =>
      `• ${r.model_name || r.hash} · ${r.env} · ${r.model_id || '?'} · ${r.dataset || '?'} · ${r.status || '?'}${(r.batches || []).length ? ` · ${r.batches.join(', ')}` : ''}`).join('\n') +
      (rows.length > MAX ? `\n… and ${rows.length - MAX} more` : '');
    const confirms = plans.map(x => C.makeLine('delete_database', { ...x.knobs, CONFIRM: 1 }));
    const ok = await say.warn(
      `Delete ${total} experiment(s) of ENV=${plans.map(x => x.plan.env).join(', ')} in ${p.name}?\n\n` +
      `Total: ${total} experiment(s)\nRuns per ENV: ${perEnv}\n\n${list}\n\n` +
      (total >= all ? 'No run is left afterwards: artifacts/, model_selection/ and graphs/ go (the tmux logs stay).\n\n' : '') +
      `${confirms.join('\n')}\n\nYou sure to delete it?`,
      { modal: true }, 'OK');
    if (ok !== 'OK') return;
    this.terminal(`TraceFlow · ${p.name} · delete database`, C.shortLine(confirms.join(' && '), this.root), true);
  }

  /** The output of a short shell line run in the repository (a dry run). */
  capture(line) {
    return bash(line, this.root);
  }

  /**
   * Judge: the process and the metric to rank on, among those the project's finished runs recorded
   * (model_judge --list-json), the file's own first. A different choice is written into model_judge.yaml
   * (ranking.primary, filters.require_processes) before make judge. false: cancelled or nothing to judge.
   */
  async judgeChoice(p, env) {
    const file = (p.configs || []).map(c => c.path).find(f => /\/model_judge\.ya?ml$/.test(String(f)));
    let choices;
    try {
      choices = await runModule(this.root, this.python, MODEL_JUDGE, ['--project', p.name, '--env', env, '--list-json']);
    } catch (e) { say.warn(`TraceFlow: the judge cannot list what the runs recorded: ${e.message || e}`); return false; }
    const available = (choices && choices.available) || {};
    const processes = Object.keys(available).filter(k => (available[k] || []).length);
    if (!processes.length) { say.warn(`TraceFlow: no finished run of ENV=${env} records a metric to rank on; make judge has nothing to judge.`); return false; }
    const current = (choices && choices.current) || {};
    const first = (list, head) => [...list.filter(x => x === head), ...list.filter(x => x !== head)];
    const process = await vscode.window.showQuickPick(first(processes, current.process).map(k => ({
      label: k, description: `${available[k].length} metrics${k === current.process ? ' · model_judge.yaml' : ''}`,
    })), { title: 'Judge: rank the models on which process?', ignoreFocusOut: true });
    if (!process) return false;
    const metrics = first(available[process.label], current.metric);
    const direction = (choices && choices.direction) || {};
    const metric = await vscode.window.showQuickPick(metrics.map(m => ({
      label: m, description: `${direction[m] === 'min' ? 'lower is better' : 'higher is better'}${m === current.metric && process.label === current.process ? ' · model_judge.yaml' : ''}`,
    })), { title: `Judge: rank on which ${process.label} metric?`, ignoreFocusOut: true });
    if (!metric) return false;
    if (process.label === current.process && metric.label === current.metric) return true;
    if (!file) { say.warn('TraceFlow: no model_judge.yaml in the project: the choice cannot be written.'); return false; }
    const set = async (path, value) => {
      const tree = await this.treeOf(file);
      await this.edit({ file, sha: tree.sha256, op: 'set', path, value }, false, `judge ranks on ${process.label}.${metric.label}`);
    };
    await set(['model_judge', 'ranking', 'primary', 'process'], process.label);
    await set(['model_judge', 'ranking', 'primary', 'metric'], metric.label);
    // filters.require_processes: the chosen process alone (a process no run has rejects every run)
    const tree = await this.treeOf(file);
    const judge = tree.tree && tree.tree.k === 'map' ? (tree.tree.entries.find(e => e.key === 'model_judge') || {}).node : null;
    const filters = judge && judge.k === 'map' ? (judge.entries.find(e => e.key === 'filters') || {}).node : null;
    const req = filters && filters.k === 'map' ? (filters.entries.find(e => e.key === 'require_processes') || {}).node : null;
    const items = req && req.k === 'list' ? req.items.map(it => String(it.text == null ? '' : it.text)) : [];
    if (items.length !== 1 || items[0] !== process.label) {
      for (let i = items.length - 1; i >= 0; i--) {
        const t = await this.treeOf(file);
        await this.edit({ file, sha: t.sha256, op: 'remove', path: ['model_judge', 'filters', 'require_processes', i] }, false, `judge requires ${process.label}`);
      }
      const t = await this.treeOf(file);
      await this.edit({ file, sha: t.sha256, op: 'add', path: ['model_judge', 'filters', 'require_processes'], value: process.label }, false, `judge requires ${process.label}`);
    }
    return true;
  }

  async pick(items, title) {
    if (!items || !items.length) {
      say.warn(`TraceFlow: params.yaml lists no ${title.split(' ')[0]}.`);
      return undefined;
    }
    return items.length === 1 ? items[0] : vscode.window.showQuickPick(items, { title, ignoreFocusOut: true });
  }

  /** Why a long run may not start now (polled afresh), or '' when it may. */
  async blocked(name) {
    await this.pollProcesses();
    if (this.busy.length) {
      return `TraceFlow: not started. Let the current run finish: an experiment is running on this machine:\n${this.busy.join('\n')}\n` +
        'make orchestrate begins with make stack-down, which would stop it. Wait for it, or stop it first.';
    }
    const s = this.sessions[name];
    return s && s.busy ? `TraceFlow: ${name} is still running ${s.command}. Attach to it, or Stop it first.` : '';
  }

  /** A detached tmux run, after the busy checks and a confirmation; then a terminal attaches. */
  async startLong(p, target, vars, commands, note = '', opts = {}) {
    const name = C.session(p.name);
    let why = await this.blocked(name);
    if (why) return say.warn(why, { modal: true });
    let planned = '';
    if (commands) {
      let per = null;
      try { per = (await runPy(this.root, this.python, 'layout', ['plan', '--project', p.name])).per_command; } catch (e) { /* counted by the run itself */ }
      planned = per == null ? 'params.yaml plans an unknown number of experiments (the run will count them).'
        : `params.yaml plans ${per * commands} experiment(s)${commands > 1 ? ` (${per} per command × ${commands} commands)` : ''}; DONE ones are skipped.`;
    }
    const line = C.makeLine(target, vars);
    const choices = ['test', 'reproduce', 'import'].includes(target) ? ['Start'] : ['Start', 'Start with FORCE=1 (train DONE ones again)'];
    const choice = await say.info(
      `Start in tmux ${name}?\n${line}\n${planned}${note ? `\n${note}` : ''}`, { modal: true }, ...choices);
    if (!choice) return;
    why = await this.blocked(name);                 // the plan count and the dialog take time: check again
    if (why) return say.warn(why, { modal: true });
    const final = choice === 'Start' ? line : C.makeLine(target, { ...vars, FORCE: 1 });
    if (opts.before) {                               // a step that must follow the confirmation (e.g. placing a snapshot)
      try { await opts.before(); } catch (e) {
        return say.warn(`TraceFlow: not started: ${e.message || e}`, { modal: true });
      }
    }
    const kind = target === 'experiment' ? 'queue' : target;
    const log = opts.log || C.logFile(kind, vars, p.folder);
    if (this.sessions[name]) await bash(`tmux kill-session -t ${C.word(name)}`, this.root);   // a finished run's idle shell
    const since = Date.now() - 1000;
    await bash(C.tmuxStart(name, this.root, final, log), this.root);
    await this.pollProcesses();
    this.terminal(`TraceFlow · ${name}`, C.tmuxAttach(name), true);
    if (target === 'test') {                         // the test's result opens beside the panel once written
      this.awaitResult(`${p.name}:test`, s => (mtimeMs(p.test_registry) >= s ? p.test_registry : null), since, TEST_WAIT_MS);
    }
    this.tick();
    return true;
  }

  /** A short command in the project's terminal; its result opens beside the panel once written. */
  runShort(p, key, line, result) {
    const since = Date.now() - 1000;
    this.terminal(`TraceFlow · ${p.name}`, C.shortLine(line, this.root), false);
    this.awaitResult(`${p.name}:${key}`, result, since, RESULT_WAIT_MS);
  }

  /** Opens result(since) beside the panel as soon as it names a file; gives up after waitMs. */
  awaitResult(waitKey, result, since, waitMs) {
    clearInterval(this.waits.get(waitKey));
    const timer = setInterval(() => {
      const file = result(since);
      if (file) {
        clearInterval(timer);
        this.waits.delete(waitKey);
        this.pollFigures();
        this.open(file);
      } else if (Date.now() - since > waitMs) {
        clearInterval(timer);
        this.waits.delete(waitKey);
      }
    }, 1000);
    this.waits.set(waitKey, timer);
  }

  /** The newest .pdf (else .svg / .png) in job `id`'s figure folder, written since `since`. */
  newestFigure(p, id, since, env) {
    const job = (p.graph_jobs || []).find(j => j.id === id);
    if (!job || !job.dir) return null;
    const root = this.jobRoot(p, job, env || this.followed(p).env);
    const dirs = fs.existsSync(path.join(root, job.dir)) ? [job.dir] : [];
    let best = null;
    for (const d of dirs) {
      for (const f of listFiles(path.join(root, d), 2)) {
        const m = f.match(/\.(pdf|svg|png)$/i);
        if (!m) continue;
        const full = path.join(root, d, f);
        const t = mtimeMs(full);
        const rank = { pdf: 0, svg: 1, png: 2 }[m[1].toLowerCase()];
        if (t >= since && (!best || rank < best.rank || (rank === best.rank && t > best.t))) best = { full, t, rank };
      }
    }
    return best ? best.full : null;
  }

  /** A visible terminal by name; an attach replaces the old one (a tmux client only detaches). */
  terminal(name, line, replace) {
    let term = vscode.window.terminals.find(t => t.name === name);
    if (term && replace) { term.dispose(); term = null; }
    // hideFromUser: the Python extensions skip auto-activating such a terminal (vscode-python-envs:
    // shouldSkipTerminalActivation). Their activation types Ctrl-C and `conda activate …` into a new
    // terminal, which would stop the log viewer, or interrupt the make of an attached tmux session.
    // Our lines activate the sca env themselves where they need it. show() reveals it at once.
    // isTransient: VS Code does not revive the terminal after a window reload. A revived terminal is
    // re-activated by the Python extension (Ctrl-C, then `conda activate`), which reached the make of an
    // attached tmux session (11 Oct 09:16 and 16:25, 12 Oct 05:56 UTC: queue log stream lost).
    if (!term) term = vscode.window.createTerminal({ name, cwd: this.root, hideFromUser: true, isTransient: true });
    term.show();
    term.sendText(line);
  }

  /** A file beside the panel's editor; a folder is revealed in the explorer instead. */
  open(p) {
    let isDir = false;
    try { isDir = fs.statSync(p).isDirectory(); } catch (e) {
      return say.warn(`TraceFlow: ${p} does not exist yet.`);
    }
    if (isDir) return this.reveal(p);
    vscode.commands.executeCommand('vscode.open', vscode.Uri.file(p), { viewColumn: vscode.ViewColumn.Beside, preview: false })
      .then(undefined, e => say.error(`TraceFlow: ${e.message || e}`));
  }

  reveal(p) {
    if (!fs.existsSync(p)) return say.warn(`TraceFlow: ${p} does not exist yet.`);
    vscode.commands.executeCommand('revealInExplorer', vscode.Uri.file(p))
      .then(undefined, e => say.error(`TraceFlow: ${e.message || e}`));
  }

  openDetail(project, hash) {
    this.detailRun = { project, hash };
    const title = `TraceFlow · ${project} · ${hash}`;
    if (!this.detail) {
      this.detail = vscode.window.createWebviewPanel('traceflow.detail', title, vscode.ViewColumn.Beside, this.webviewOptions());
      this.detail.webview.html = this.html(this.detail.webview, 'detail');
      this.detail.webview.onDidReceiveMessage(m => this.onMessage(m));
      this.detail.onDidDispose(() => { this.detail = null; this.detailRun = null; });
    }
    this.detail.title = title;
    this.detail.reveal(vscode.ViewColumn.Beside, true);
    this.tick();
  }

  dispose() { this.stop(); }
}

function activate(context) {
  const panel = new Panel(context);
  const started = panel.start().catch(e => panel.report(e));
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('traceflow.panel', panel, { webviewOptions: { retainContextWhenHidden: true } }),
    // planner tabs stay open over a window reload, with their design, until closed by hand
    vscode.window.registerWebviewPanelSerializer(PLANNER_VIEW, {
      deserializeWebviewPanel: (tab, kept) => started.then(() => panel.restorePlanner(tab, kept)).catch(e => panel.report(e)),
    }),
    // the status page's button on the TraceFlow Monitor title line: its settings open or close in the panel
    vscode.commands.registerCommand('traceflow.share', () => panel.toggleShare()),
    vscode.workspace.onDidChangeConfiguration(e => {
      if (e.affectsConfiguration('traceflow')) panel.restart().catch(err => panel.report(err));
    }),
    { dispose: () => panel.dispose() },
  );
}

function deactivate() {}

/**
 * The order a form's staged edits are written in, so every path still names what it named when staged:
 * a staged arrange (the planner's new order of a list) drops the staged edits inside that list, which it carries;
 * set / put / arrange first (the file as it was), then add (an append moves no earlier index), then remove / delete
 * by their whole path, last first: a later item before an earlier one, inside an item before the item.
 */
const APPLY_OPS = ['set', 'put', 'add', 'remove', 'delete', 'arrange'];

function orderEdits(edits) {
  // a staged arrange (the planner's whole new order of a list) carries every staged edit inside that list
  const arranged = edits.filter(e => e.op === 'arrange').map(e => JSON.stringify(e.path || []).slice(0, -1));
  const inside = e => e.op !== 'arrange' && arranged.some(a => {
    const p = JSON.stringify(e.path || []);
    return p === `${a}]` || p.startsWith(`${a},`);
  });
  edits = edits.filter(e => !inside(e));
  const rank = e => ({ set: 0, put: 0, arrange: 0, add: 1, remove: 2, delete: 2 })[e.op];
  const later = (a, b) => {                     // > 0 when path a comes after path b in the file
    for (let i = 0; i < Math.min(a.length, b.length); i += 1) {
      if (a[i] === b[i]) continue;
      if (typeof a[i] === 'number' && typeof b[i] === 'number') return a[i] - b[i];
      return String(a[i]) < String(b[i]) ? -1 : 1;
    }
    return a.length - b.length;                 // a path inside another: the inner one first
  };
  return edits.map((e, i) => ({ ...e, i })).sort((a, b) =>
    rank(a) - rank(b) || (rank(a) === 2 ? later(b.path || [], a.path || []) : a.i - b.i));
}

module.exports = { activate, deactivate, Panel, findPython, listFiles, inside, dialog, wrapText, orderEdits, APPLY_OPS, COMPLETE_WARNING };     // Panel, findPython, listFiles, inside: for tests
