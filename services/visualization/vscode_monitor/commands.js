// services/visualization/vscode_monitor/commands.js
'use strict';
/**
 * commands.js
 * ===========
 * One shell line per button, as plain strings; extension.js runs them. Pure,
 * so `node --test` checks the quoting.
 *
 *   quote(s)                       a POSIX single-quoted word
 *   envLine(cmd)                   cmd inside the sca conda env (made by
 *                                  services/side_channel_core/infra/environment/setup_sca_env.sh)
 *   makeLine(target, vars)         `make TARGET K=V …` (empty values left out)
 *   session(project)               the tmux session of a project: tf-<project>
 *   logFile(kind, vars, folder)    projects/<folder>/artifacts/<ENV>/tmux_logs/<queue>.txt,
 *                                  …/orchestrate_<MODEL_ID>_<DATASET>.txt, …/test.txt: the
 *                                  project's own artifacts, beside its runs
 *   tmuxStart(session, cwd, line, log)  a detached tmux session running the line in the
 *                                  env, its output also appended to the log; the
 *                                  session stays open on a shell when the line ends
 *   tmuxAttach / tmuxStop / tmuxBusy / tailLog
 *   BUSY_PATTERN                   what counts as an experiment already running
 *   ownLine(nonInteractive)        give projects/ back to the user: the container writes
 *                                  as root, so host-side writers (find_model, judge,
 *                                  graph, test bookkeeping) hit `Permission denied`
 *
 * Author: Shraddha Mishra
 */

const ENV_NAME = 'sca';
const VAR = /^[A-Z][A-Z0-9_]*$/;
// `make orchestrate` begins with `make stack-down`: a second one would stop the running stack,
// whoever started it. Any of these processes on the machine makes Start refuse. Anchored at the
// start of the command line: the process itself is make (or /path/to/make), never a shell or an
// editor whose arguments merely mention it. The trainer inside the container (`python -m core.cli
// orchestrate-cmd|test-cmd`, visible from the host) counts too: the stack keeps training when the
// host make that started it is gone (e.g. its `nerdctl compose up` client exited).
const BUSY_PATTERN = '^((\\S*/)?make( \\S+)* (orchestrate|experiment|test|reproduce|import)( |$)|(\\S*/)?python[0-9.]* -m core\\.cli (orchestrate|test|train)-cmd( |$))';
// A viewer of the training container's log (`[sudo] nerdctl logs …`), e.g. in the tmux session that
// started the queue after its own log stream ended. Anchored like BUSY_PATTERN.
const VIEWER_PATTERN = '^(\\S*/)?(sudo( -\\S+)* )?(\\S*/)?nerdctl( \\S+)* logs( |$)';

function quote(s) {
  return `'${String(s).replace(/'/g, `'\\''`)}'`;
}

/** Leaves a word that needs no quoting as it is, so the lines stay readable. */
function word(s) {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(String(s)) ? String(s) : quote(s);
}

function envLine(cmd) {
  return `source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate ${ENV_NAME} && ${cmd}`;
}

function makeLine(target, vars = {}) {
  if (!/^[a-z][a-z0-9_-]*$/.test(target)) throw new Error(`bad make target: ${target}`);
  const parts = ['make', target];
  for (const [k, v] of Object.entries(vars)) {
    if (!VAR.test(k)) throw new Error(`bad make variable: ${k}`);
    if (v === undefined || v === null || v === '') continue;
    parts.push(`${k}=${word(v)}`);
  }
  return parts.join(' ');
}

function session(project) {
  const name = `tf-${String(project).replace(/[^A-Za-z0-9_-]/g, '_')}`;
  if (name === 'tf-') throw new Error('no project');
  return name;
}

function safeName(s) {
  return String(s || '').replace(/[^A-Za-z0-9_.-]/g, '_') || 'x';
}

/** projects/<folder>/artifacts/<ENV>/tmux_logs: beside that environment's runs. */
function logDir(folder, env) {
  if (!folder) throw new Error('no project folder');
  if (!env) throw new Error('no ENV');
  return `projects/${safeName(folder)}/artifacts/${safeName(env)}/tmux_logs`;
}

function logFile(kind, vars, folder) {
  const dir = logDir(folder, vars.ENV);
  if (kind === 'queue') return `${dir}/${safeName(vars.QUEUE)}.txt`;
  if (vars.BATCH) return `${dir}/${safeName(vars.BATCH)}.txt`;     // a batch's runs all append to <BATCH>.txt
  if (kind === 'orchestrate') return `${dir}/orchestrate_${safeName(vars.MODEL_ID)}_${safeName(vars.DATASET)}.txt`;
  return `${dir}/${safeName(kind)}.txt`;
}

/**
 * tmux new-session -d running `bash -lc` in the repository: the env, the line with its output
 * also appended to the log (pipefail keeps make's exit code), then a closing note and a shell
 * so the output stays readable after the end.
 */
/**
 * The output to the terminal as it is, and to the log file as plain text: colour and cursor codes
 * removed, of a line redrawn with carriage returns (a progress bar) only its last state kept, and
 * trailing spaces (a Rich block's padding) dropped.
 */
function teeToLog(log) {
  return `tee >(sed -u -e 's/\x1b\[[0-9;?]*[A-Za-z]//g' -e 's/.*\r//' -e 's/[[:space:]]*$//' >> ${word(log)})`;
}

function tmuxStart(name, cwd, line, log) {
  // `trap : INT`: Stop's SIGINT ends make, while this shell lives on to write the closing note (a trapped,
  // not ignored, signal: make and tee still get the default action).
  // `stty intr undef …` while the line runs: a Ctrl-C typed into the pane, by hand or injected by an
  // attached client (VS Code revives terminals on reload and the Python extension types Ctrl-C plus
  // `conda activate` into them), no longer reaches make; the training container keeps its log stream.
  // Stop signals the pane's process group instead (tmuxStop). `stty sane` gives the shell its Ctrl-C back.
  const inner = `set -o pipefail; trap : INT; ${ownLine()} && mkdir -p ${word(log.replace(/\/[^/]*$/, '') || '.')} && ` +
    `{ stty intr undef quit undef susp undef 2>/dev/null || true; } && ${envLine(`${line} 2>&1 | ${teeToLog(log)}`)}; ` +
    `rc=$?; stty sane 2>/dev/null; echo; echo "[traceflow] ${name} finished, exit $rc ($(date '+%F %T')); the session stays open, exit closes it"; exec bash`;
  return `tmux new-session -d -s ${word(name)} -c ${quote(cwd)} ${quote(`bash -lc ${quote(inner)}`)}`;
}

function tmuxAttach(name) {
  return `tmux attach -t ${word(name)}`;
}

/**
 * SIGINT to the session's command (the make line): the terminal's foreground process group, the one a
 * Ctrl-C would reach, signalled directly because tmuxStart switches the pane's Ctrl-C off while it runs.
 */
function tmuxStop(name) {
  const pane = `"$(tmux display-message -p -t ${word(name)} '#{pane_pid}')"`;
  return `kill -INT -- -"$(ps -o tpgid= -p ${pane} | tr -d ' ')"`;
}

/** The command in the session's pane: a shell when idle, make/python while running. */
function tmuxBusy(name) {
  return ['display-message', '-p', '-t', name, '#{pane_current_command}'];
}

function tailLog(log) {
  return `tail -n 200 -f ${word(log)}`;
}

/**
 * The live log of the running training container (its make/tmux stream may have ended), every line
 * prefixed with the container's name as `make orchestrate` prints it: `sidechannel |…`, in bold cyan.
 */
function containerLogLine() {
  return `c="$(sudo nerdctl ps --format '{{.Names}}' | grep -m1 '^sidechannel' || echo sidechannel)"; ` +
    `sudo nerdctl logs -f --tail 100 "$c" 2>&1 | sed -u "s/^/\\x1b[1;36m$c |\\x1b[0m/"`;
}

/**
 * A terminal line for a short command (graph, find_model, judge): to the repository root (the panel
 * reuses its terminal, which may have been cd'd elsewhere; the Makefile is only at the root), the env,
 * then the line.
 */
function shortLine(line, root) {
  return `${root ? `cd ${quote(root)} && ` : ''}${ownLine()} && ${envLine(line)}`;
}

/**
 * `sudo chown -R <you> projects/` (skipped when there is no projects/ here). The running container
 * writes as root and keeps doing so (root writes regardless of owner), so this never disturbs it.
 * nonInteractive: `sudo -n`, for calls without a terminal; it fails instead of asking for a password.
 */
/**
 * Stop a running image build: SIGINT to every `nerdctl … build` process (buildkit cancels the solve
 * when its client goes), SIGTERM to any left after 5 s. make build-image then fails, and so does the
 * make that called it. A line in the build log tells the panel the build was killed, not stopped.
 */
const BUILD_PATTERN = '^(\\S*/)?nerdctl( \\S+)* build( |$)';

/** make kill-build: the nerdctl client, BuildKit's running build steps and, if need be, the BuildKit daemon
 *  (the Makefile keeps the one implementation; the build log records the kill). */
function killBuildLine(env) {
  return makeLine('kill-build', { ENV: env || 'dev' });
}

function ownLine(nonInteractive = false) {
  // The container runs as the user (compose user: USER_ID:GROUP_ID), so everything under projects/ is the
  // user's; only runs made while it ran as root (before 2026-10-23) left root-owned files. chown only then
  // (make own-projects does the same before every make target that writes there), so no sudo prompt otherwise.
  const sudo = `sudo ${nonInteractive ? '-n ' : ''}`;
  return `{ [ ! -d projects ] || [ -z "$(find projects -not -user "$(id -u)" -print -quit 2>/dev/null)" ] || ` +
    `${sudo}chown -R "$(id -u):$(id -g)" projects; }`;
}

module.exports = {
  ENV_NAME, BUSY_PATTERN, VIEWER_PATTERN, quote, word, envLine, makeLine, teeToLog, session, logDir, logFile,
  tmuxStart, tmuxAttach, tmuxStop, tmuxBusy, tailLog, shortLine, ownLine, containerLogLine, BUILD_PATTERN, killBuildLine,
};
