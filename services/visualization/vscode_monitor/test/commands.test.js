// services/visualization/vscode_monitor/test/commands.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync, spawnSync } = require('child_process');
const C = require('../commands');

test('make lines quote values and drop empty ones', () => {
  assert.equal(C.makeLine('experiment', { ENV: 'dev', PROJECT: 'DLSCA_ASCAD', QUEUE: 'fig26', FORCE: '' }),
    'make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26');
  assert.equal(C.makeLine('orchestrate', { MODEL_ID: "a b'c" }), `make orchestrate MODEL_ID='a b'\\''c'`);
  assert.equal(C.makeLine('graph', { GRAPH_ID: 64 }), 'make graph GRAPH_ID=64');
  assert.throws(() => C.makeLine('x; rm -rf /', {}), /bad make target/);
  assert.throws(() => C.makeLine('graph', { 'A;B': 1 }), /bad make variable/);
});

test('sessions, logs and the env', () => {
  assert.equal(C.session('DLSCA_ASCAD'), 'tf-DLSCA_ASCAD');
  assert.equal(C.session('a b/c'), 'tf-a_b_c');
  assert.equal(C.logFile('queue', { QUEUE: 'fig26', ENV: 'dev' }, 'DLSCA_ASCAD'), 'projects/DLSCA_ASCAD/artifacts/dev/tmux_logs/fig26.txt');
  assert.equal(C.logFile('orchestrate', { MODEL_ID: 'ASCAD_MLPbest', DATASET: 'ascad', ENV: 'prod' }, 'DLSCA_ASCAD'),
    'projects/DLSCA_ASCAD/artifacts/prod/tmux_logs/orchestrate_ASCAD_MLPbest_ascad.txt');
  assert.equal(C.logFile('queue', { QUEUE: '../x', ENV: 'dev' }, 'P'), 'projects/P/artifacts/dev/tmux_logs/.._x.txt');
  assert.throws(() => C.logFile('test', { ENV: 'dev' }, ''), /no project folder/);
  assert.throws(() => C.logFile('test', {}, 'P'), /no ENV/);
  assert.match(C.envLine('make judge'), /conda activate sca && make judge$/);
  assert.equal(C.quote("it's"), `'it'\\''s'`);
  assert.equal(C.ownLine(), '{ [ ! -d projects ] || [ -z "$(find projects -not -user "$(id -u)" -print -quit 2>/dev/null)" ] || ' +
    'sudo chown -R "$(id -u):$(id -g)" projects; }');
  assert.match(C.ownLine(true), /sudo -n chown/);
  assert.match(C.shortLine('make judge'), /^\{ \[ ! -d projects \] \|\| \[ -z .*find projects -not -user.* \] \|\| sudo chown -R .* projects; \} && source .*conda activate sca && make judge$/);
  assert.equal(C.logFile('orchestrate', { MODEL_ID: 'M', DATASET: 'd', ENV: 'dev', BATCH: 'fig26' }, 'P'), 'projects/P/artifacts/dev/tmux_logs/fig26.txt');
  assert.match(C.shortLine('make graph GRAPH_ID=83', '/repo root'), /^cd '\/repo root' && \{ \[ ! -d projects \]/);   // the Makefile is at the root
});

test('the busy pattern matches a running make orchestrate/experiment/test only', () => {
  const re = new RegExp(C.BUSY_PATTERN);
  assert.ok(re.test('make experiment ENV=dev PROJECT=DLSCA_ASCAD QUEUE=fig26'));
  assert.ok(re.test('make orchestrate ENV=dev PROJECT=P'));
  assert.ok(re.test('/usr/bin/make ENV=dev PROJECT=P test'));
  assert.ok(!re.test('make graph GRAPH_ID=1'));
  assert.ok(!re.test('bash -lc set -o pipefail; source x && make experiment ENV=dev'));   // the wrapper, not make
  assert.ok(!re.test('python -m pytest tests/'));
  assert.ok(!re.test("node -e assert.match(x, /make experiment ENV=dev/)"));                   // mentions it only
  assert.ok(re.test('make -C /repo experiment QUEUE=fig2'));
  assert.ok(re.test('python -m core.cli orchestrate-cmd --project DLSCA_ASCAD --dataset ascad --model-id M --env dev'));
  assert.ok(re.test('/usr/bin/python3.10 -m core.cli test-cmd --project P'));
  assert.ok(re.test('make reproduce ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26'));
  assert.ok(re.test('make import ENV=dev PROJECT=ZvW MODEL=zaid_ascad_n0'));                // make import starts the stack too
  assert.ok(re.test('/venv/bin/python -m core.cli train-cmd --project DLSCA_ASCAD --env dev -n a,b'));
  assert.ok(!re.test('python -m core.cli list-cmd'));
});

const tmux = spawnSync('tmux', ['-V']).status === 0;
const conda = spawnSync('bash', ['-lc', 'conda info --base']).status === 0;

test('tmuxStart runs the line in the env, logs it, keeps the session open', { skip: !(tmux && conda) && 'tmux or conda missing', timeout: 60000 }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const name = `tf-test-${process.pid}`;
  try {
    execFileSync('bash', ['-c', C.tmuxStart(name, dir, `python -c "import sys; print('hi from', sys.prefix.split('/')[-1]); sys.exit(3)"`, 'projects/P/artifacts/dev/tmux_logs/q.txt')]);
    let pane = '';
    for (let i = 0; i < 100 && !pane.includes('[traceflow]'); i++) {
      await new Promise(r => setTimeout(r, 200));
      pane = execFileSync('tmux', ['capture-pane', '-p', '-S', '-', '-t', name]).toString();
    }
    assert.match(pane, /finished, exit 3/);
    assert.match(fs.readFileSync(path.join(dir, 'projects', 'P', 'artifacts', 'dev', 'tmux_logs', 'q.txt'), 'utf8'), /hi from sca/);
    const cmd = execFileSync('tmux', C.tmuxBusy(name)).toString().trim();
    assert.equal(cmd, 'bash');                                    // idle: the line has ended
  } finally {
    spawnSync('tmux', ['kill-session', '-t', name]);
  }
});

test('the container log names the container on every line, in colour, as make orchestrate does', () => {
  assert.match(C.containerLogLine(), /nerdctl logs -f --tail 100 "\$c" 2>&1 \| sed -u "s\/\^\/\\x1b\[1;36m\$c \|\\x1b\[0m\/"$/);
});

test('the log file gets plain text: colour codes and redrawn progress bars removed, the terminal keeps them', () => {
  const { execFileSync } = require('child_process');
  const fs = require('fs'); const os = require('os'); const path = require('path');
  const log = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'tf-')), 'q.txt');
  const shown = execFileSync('bash', ['-c',
    `printf '\\033[1;36mside |\\033[0m [INFO] epoch 1   \\n 10%%\\r100%% done\\n' | ${C.teeToLog(log)}; sleep 0.3`]).toString();
  assert.match(shown, /\x1b\[1;36m/);                                  // the terminal still sees colours
  assert.equal(fs.readFileSync(log, 'utf8'), 'side | [INFO] epoch 1\n100% done\n');
});
