// services/visualization/vscode_monitor/state.js
'use strict';
/**
 * state.js
 * ========
 * Turns the files a project writes into what the panel draws. Pure: no
 * vscode, no file system, so `node --test` runs it.
 *
 *   parseHeartbeat(text)       artifacts/<env>/heartbeat.json -> object | null
 *   light(hb, nowMs)           green < 15 s, amber 15-60 s, red running and
 *                              older, grey idle (no file, done, failed)
 *   runSummary(hb)             name, phase line, progress %
 *   runAlive(hb, hash, nowMs)  is this run alive: its own heartbeat (model_hash), status
 *                              running and a beat under 60 s old; otherwise dead (no beat,
 *                              stopped, ended, or another run is beating)
 *   tiles(registry, queues)    queued / running / done / failed runs; queued also
 *                              holds the trials a running sweep has not started yet:
 *                              total_remaining_trials of its latest running run
 *   queueStates(queues, reg)   per experiment_queue.yaml key, from the runs of its latest
 *                              execution (newest batch id <key>-<UTC start>): planned = the registry's
 *                              total_trials (the sweep size of one command) x the
 *                              key's commands; done, started, state
 *   queuePositions(queues, reg) model hash -> {queue, k, n}: the run is the k-th
 *                              of the queue's n planned trials (k by start order:
 *                              the same as trial_index for runs numbered by their
 *                              sweep position, and right for older rows too)
 *   followEnv({...})           the ENV the panel follows and why: only what runs or ran decides
 *   buildProgress(log, {...})  the image build's % from logs/build_image.log
 *   containerHealth(ps)        the sidechannel* containers and their health
 *                              (running run, running command's ENV=/--env, run in progress,
 *                              latest snapshot's env key, latest run); '' when nothing has;
 *                              the traceflow.env setting only when set
 *   eta({...})                 when the running run and its batch end: the run from its
 *                              heartbeat (elapsed x (1 - progress) / progress), the
 *                              batch adds every run still to come, line by line, each
 *                              at its model's measured whole run (batchRuns)
 *   figureStates(jobs, dirs)   per graph job: drawn when its folder holds a
 *                              .pdf, .svg or .png
 *   snapFilter(rows, f, known) the SNAPSHOTS Scan: at least one filter, every filled one valid,
 *                              the rows matching all of them (a field's comma-separated values are
 *                              alternatives); {error} when none is filled, one is wrong or none matches
 *
 * Author: Shraddha Mishra
 */

const GREEN_S = 15;
const AMBER_S = 60;

const STATUS_GROUP = {
  new: 'queued', queued: 'queued', pending: 'queued',
  in_progress: 'running', running: 'running',
  done: 'done',
  failed: 'failed', interrupted: 'failed', cancelled: 'failed', canceled: 'failed',
};

function statusGroup(status) {
  return STATUS_GROUP[String(status || '').trim().toLowerCase()] || 'queued';
}

function parseHeartbeat(text) {
  if (!text) return null;
  try {
    const hb = JSON.parse(text);
    return hb && typeof hb === 'object' && Number.isFinite(Number(hb.written_ns)) ? hb : null;
  } catch (e) {
    return null;                      // half-written or garbled: as if missing
  }
}

function ageS(hb, nowMs) {
  return (nowMs - Number(hb.written_ns) / 1e6) / 1000;
}

function light(hb, nowMs) {
  if (!hb) return { color: 'grey', label: 'idle' };
  const age = ageS(hb, nowMs);
  if (hb.stale) return { color: 'grey', label: `stale heartbeat: ${hb.stale}`, age };
  if (hb.status !== 'running') return { color: 'grey', label: String(hb.status || 'idle'), age };
  if (age < GREEN_S) return { color: 'green', label: 'running', age };
  if (age < AMBER_S) return { color: 'amber', label: `late ${Math.round(age)} s`, age };
  return { color: 'red', label: `no beat for ${Math.round(age)} s`, age };
}

function runAlive(hb, hash, nowMs) {
  if (!hash) return { alive: false, label: 'dead: no hash' };
  if (!hb || String(hb.model_hash || '') !== String(hash)) return { alive: false, label: 'dead: no heartbeat from this run' };
  const age = ageS(hb, nowMs);
  if (hb.status !== 'running') return { alive: false, label: `ended: ${hb.status || 'stopped'}`, age };
  if (age < AMBER_S) return { alive: true, label: `alive: last beat ${Math.max(0, Math.round(age))} s ago`, age };
  return { alive: false, label: `dead: no beat for ${Math.round(age)} s`, age };
}

/**
 * Why a heartbeat that says running is not believed, or '' when it is: its hash is not in the registry
 * (the run was deleted), or no TraceFlow process runs on the machine (the run was terminated, the stack
 * torn down: nothing ended the heartbeat). A run is registered before it starts beating, and its
 * trainer, also inside the container, is a host process. busy: the BUSY_PATTERN command lines, null
 * before the first poll.
 */
function staleBeat(hb, { registry, busy } = {}) {
  if (!hb || hb.status !== 'running') return '';
  const hash = String(hb.model_hash || '');
  const run = hb.model_name || hash;
  if (hash && registry && typeof registry === 'object' && !Object.prototype.hasOwnProperty.call(registry, hash)) {
    return `${run} is not in the run registry`;
  }
  if (Array.isArray(busy) && !busy.length) return `${run}: no TraceFlow process runs`;
  return '';
}

/** The heartbeats with every unbelieved 'running' one turned to status 'stale' (staleBeat says why). */
function believedBeats(heartbeats, opts) {
  return (heartbeats || []).map(h => {
    const why = staleBeat(h.hb, opts);
    return why ? { ...h, hb: { ...h.hb, status: 'stale', stale: why } } : h;
  });
}

function phaseLine(hb) {
  const parts = [];
  if (hb.phase === 'fold') parts.push(`fold ${hb.fold || '?'}/${hb.n_folds || '?'}`);
  else if (hb.phase) parts.push(String(hb.phase));
  const epochs = hb.phase === 'fold' ? hb.fold_epochs : hb.refit_epochs;
  if (hb.epoch && hb.phase !== 'test') parts.push(`epoch ${hb.epoch}/${epochs || '?'}`);
  return parts.join(' · ');
}

function runSummary(hb) {
  if (!hb) return null;
  const progress = Math.max(0, Math.min(1, Number(hb.progress) || 0));
  return {
    name: hb.model_name || hb.model_hash || '?',
    model: hb.model_id || '',
    dataset: hb.dataset || '',
    phase: phaseLine(hb),
    percent: Math.round(progress * 1000) / 10,
    batches: Array.isArray(hb.batches) ? hb.batches : [],
  };
}

function registryEntries(registry) {
  if (!registry || typeof registry !== 'object') return [];
  return Object.entries(registry)
    .filter(([, e]) => e && typeof e === 'object')
    .map(([hash, e]) => ({
      hash,
      name: e.model_name || hash,
      model: e.model_id || '',
      env: e.env || '',
      status: String(e.status || ''),
      group: statusGroup(e.status),
      reason: e.status_reason || '',
      batches: Array.isArray(e.batches) ? e.batches.map(String) : [],
      batchIds: Array.isArray(e.batch_ids) ? e.batch_ids.map(String) : [],
      started: e.started_at || '',
      finished: e.finished_at || '',
      path: e.path || '',
      session: e.run_session_id || '',
      trial: Number(e.trial_index) || null,
      trials: Number(e.total_trials) || null,
      remaining: e.total_remaining_trials == null || e.total_remaining_trials === '' ? NaN : Number(e.total_remaining_trials),
    }));
}

/** The queue keys of the experiment commands running now: `make experiment … QUEUE=k` or `experiment_queue … --queue k`. */
function activeQueues(commands) {
  return [...new Set((commands || []).map(c => (String(c).match(/(?:^|\s)(?:QUEUE=|--queue[ =])(\S+)/) || [])[1]).filter(Boolean))];
}

function tiles(registry, queues, { active = [], perCommand = {}, beating = '' } = {}) {
  const out = { queued: [], running: [], done: [], failed: [] };
  // beating: the hash of the run whose heartbeat says running. train_and_test records the training DONE before
  // the test starts, so a run under test reads done in the registry; it stays under Running (its live metrics,
  // the test's attack progress) until its heartbeat ends. A failure is never moved.
  const all = registryEntries(registry)
    .map(run => (beating && run.hash === beating && run.group === 'done' ? { ...run, group: 'running' } : run));
  for (const run of all) out[run.group].push(run);
  // Trials planned but not started are queued too. The registry holds a run only once the planner has
  // made its hash and name, when the run starts; these have neither yet ("not hashed yet").
  const pending = (name, queue) => ({ hash: '', name, pending: true, queue, model: '', status: 'not hashed yet',
                                      group: 'queued', batches: queue ? [queue] : [] });
  // 1. A queue that runs (a run of its latest batch in progress, or its experiment command alive between
  //    two commands): planned - started, the same plan the trial k/n of its runs counts against
  //    (queueStates, queuePositions). A queue of n commands of one trial each is n trials, though each
  //    command's registry rows say trial 1 of 1. Runs registered as queued (FORCE) are the execution's rows
  //    (executionRuns), listed above, so they are not counted twice.
  const covered = new Set();
  for (const q of queueStates(queues, registry, { perCommand })) {
    if (!q.planned || !(q.state === 'running' || active.includes(q.key))) continue;
    const mine = executionRuns(all, q.key);
    mine.forEach(r => covered.add(r.hash));
    for (let k = mine.length + 1; k <= q.planned; k++) out.queued.push(pending(`${q.key} trial ${k}/${q.planned}`, q.key));
  }
  // 2. A running run of no running queue (a plain make orchestrate): its sweep's total_trials (one ENV,
  //    model and session) after the latest run's trial_index. total_remaining_trials is not used when the
  //    total is known: written at a run's start it still counts that run (trial 14/14 records 1).
  const scopeOf = x => `${x.env}|${x.model}|${x.session}`;
  const seen = new Set();
  for (const r of out.running) {
    const scope = scopeOf(r);
    if (covered.has(r.hash) || seen.has(scope)) continue;
    seen.add(scope);
    const mine = all.filter(x => scopeOf(x) === scope && !covered.has(x.hash));
    const latest = mine.filter(x => x.group === 'running')
      .reduce((a, b) => ((Number(b.trial) || 0) > (Number(a.trial) || 0) ? b : a));
    const left = latest.remaining;
    const queuedRows = mine.filter(x => x.group === 'queued' && (Number(x.trial) || 0) > (Number(latest.trial) || 0)).length;
    const total = Number(latest.trials) || ((Number(latest.trial) || 0) + (Number.isFinite(left) ? left : 0));
    const base = Number(latest.trial) || (Number.isFinite(left) ? total - left : 0);   // trial number of the latest run
    if (!total || base >= total) continue;
    for (let k = base + queuedRows + 1; k <= total; k++) {
      out.queued.push(pending(`${latest.model || 'run'} trial ${k}/${total}`, ''));
    }
  }
  return out;
}

const byStart = (a, b) => String(a.started).localeCompare(String(b.started)) || a.hash.localeCompare(b.hash);

/**
 * The runs of a queue key's latest execution: batch ids are <key>-<UTC start> (utils/batch.py), so the
 * newest id is the greatest; a run the execution skipped as already DONE carries the id too. Rows written
 * before batch ids existed fall back to the key in `batches`.
 */
function latestBatchId(runs, key) {
  const esc = String(key).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const own = new RegExp(`^${esc}-\\d{8}T\\d{6}Z$`);
  const ids = runs.flatMap(r => r.batchIds.filter(id => own.test(id)));
  return ids.length ? ids.reduce((a, b) => (b > a ? b : a)) : null;
}
function latestBatchRuns(runs, key) {
  const latest = latestBatchId(runs, key);
  return latest ? runs.filter(r => r.batchIds.includes(latest)) : runs.filter(r => r.batches.includes(key));
}
/** The UTC start of a batch id (<key>-YYYYMMDDTHHMMSSZ), in ms; null when it has none. */
function batchStart(id) {
  const m = String(id || '').match(/-(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/);
  return m ? Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]) : null;
}

/**
 * A run a later execution of another queue trained again after `since` (the start of this queue's latest
 * batch): its row's start, sweep size (total_trials), trial number and status are now that execution's
 * (fig20's FORCE retraining fig24's haberdashery_flock: total_trials 5, in_progress).
 */
function takenSince(r, since) {
  return since != null && r.batchIds.some(id => {
    const at = batchStart(id);
    return at != null && at > since && Date.parse(r.started) >= at;
  });
}

/**
 * The runs of a queue key's latest execution (latestBatchRuns), and the runs FORCE queued for it: FORCE marks
 * every scoped run queued under the session it trains them in before it starts any, and they join the batch
 * only when trained. A session is this execution's when a run it trained (started after the batch began,
 * not taken by a later execution) carries it.
 */
function executionRuns(runs, key) {
  const mine = latestBatchRuns(runs, key);
  const since = batchStart(latestBatchId(runs, key));
  if (since == null) return mine;
  const sessions = new Set(mine.filter(r => r.session && Date.parse(r.started) >= since && !takenSince(r, since))
    .map(r => r.session));
  return [...mine, ...runs.filter(r => r.group === 'queued' && sessions.has(r.session) && !mine.includes(r))];
}

/**
 * Per queue key, the runs its latest execution makes: planned = its lines x the sweep size of its own runs, at
 * least one run per line; never what params.yaml plans now (it holds whichever queue was configured last:
 * fig20's five optimizers made fig5 8 x 5 = 40) and never a sweep another queue trained.
 */
function queueStates(queues, registry, { perCommand: known = {} } = {}) {
  const runs = registryEntries(registry);
  return (queues || []).map(q => {
    const mine = executionRuns(runs, q.key);
    const since = batchStart(latestBatchId(runs, q.key));
    const held = mine.filter(r => !takenSince(r, since));     // the rows that still describe this execution
    const count = (g, rows = mine) => rows.filter(r => r.group === g).length;
    const commands = (q.commands || []).length;
    // the sweep size of one line, from this queue's own runs only: those this execution trained (started after
    // the batch began), else those the queue trained first in any execution (a retry that has trained nothing
    // yet: its interrupted runs). A run it skipped as DONE that another queue trained carries that queue's
    // sweep (fig10's ketchup_computer: 5 trials, from fig10_optimizers), and a run another queue trained again
    // later carries that queue's (fig24's haberdashery_flock under fig20: 5): neither sizes it.
    const trained = since == null ? [] : held.filter(r => Date.parse(r.started) >= since);
    const origin = r => String(r.batchIds[0] || r.batches[0] || '');
    const own = held.filter(r => origin(r) === q.key || origin(r).startsWith(`${q.key}-`));
    const sizing = trained.length ? trained : own;
    // else, before the first run of a started queue registers, the sweep size its frozen params.yaml plans
    // (config/.frozen/<key>-<stamp>/, counted by the extension): the queue's own settings, known at its start
    const perCommand = Math.max(0, ...sizing.map(r => Number(r.trials) || 0)) || Math.max(0, Number(known[q.key]) || 0);
    const planned = perCommand ? Math.max(perCommand * Math.max(commands, 1), mine.length)
      : Math.max(commands, mine.length) || null;
    const done = count('done');
    let state = 'not started';
    if (mine.length) {
      // running only by a run this execution trains: when another queue retrains one of its runs, this
      // queue's Terminate Run would stop that queue
      if (count('running', held)) state = 'running';
      else if (count('failed')) state = 'failed';
      else if (done >= (planned || mine.length)) state = 'done';
      else if (done === mine.length) state = 'incomplete';     // stopped before every planned trial ran
      else state = 'queued';
    }
    return { key: q.key, commands, runs: mine.length, started: mine.length, planned, done, state,
      refs: [...new Set(mine.flatMap(r => [r.name, r.hash]))] };       // its runs, for the graph jobs listing them
  });
}

// a queue's runs in the order its execution makes them: the started ones by start, then those FORCE queued
// (their started_at is their earlier training) by the trial number FORCE gave them
const byPlace = (a, b) => (a.group === 'queued') - (b.group === 'queued')
  || (a.group === 'queued' ? (Number(a.trial) || 0) - (Number(b.trial) || 0) : 0) || byStart(a, b);

function queuePositions(queues, registry, { perCommand = {} } = {}) {
  const runs = registryEntries(registry);
  const out = {};
  for (const q of queueStates(queues, registry, { perCommand })) {
    const since = batchStart(latestBatchId(runs, q.key));
    executionRuns(runs, q.key).sort(byPlace).forEach((r, i) => {
      // a run another queue trained again later is that queue's trial k/n
      if (!out[r.hash] && !takenSince(r, since)) out[r.hash] = { queue: q.key, k: i + 1, n: q.planned || null };
    });
  }
  return out;
}

// an estimate from the first beat that shows training progress, once the run is a heartbeat old: the runs are short
// (a minute and a half), and waiting for 2 % and a minute hid the ETA for most of them
const ETA_MIN_PROGRESS = 0;          // any progress (the heartbeat counts the current epoch's finished batches)
const ETA_MIN_ELAPSED_S = 5;         // one heartbeat interval

/** Epoch-passes of one trial: epochs x passes (folds + refit) + 1 for the test, as heartbeat progress counts. */
const trialUnits = t => (Number(t.epochs) || 0) * (Number(t.passes) || 1) + 1;

/**
 * The runs of the running run's batch not started yet: its queue's plan (queueStates: the lines x their runs, the
 * panel's "done of planned") less the runs its execution has started, this one counted even before it registers;
 * without a planned queue (a single make orchestrate), the trials after this one that its own heartbeat reports
 * (trial k of n). A line's runs are the run's own count when params.yaml's sweep does not list them (a data entry's
 * splits: profiling_split.random_state). Null when neither tells.
 */
function batchRunsLeft(queues, registry, hb, { perCommand = {} } = {}) {
  if (!hb) return null;
  const keys = Array.isArray(hb.batches) ? hb.batches.map(String) : [];
  const sized = { ...Object.fromEntries(keys.map(k => [k, Number(hb.trials) || 0])), ...perCommand };
  const q = queueStates((queues || []).filter(x => keys.includes(x.key)), registry, { perCommand: sized })[0];
  if (q && q.planned) return Math.max(0, q.planned - q.started - (q.refs.includes(String(hb.model_hash || '')) ? 0 : 1));
  const n = Number(hb.trials) || 0;
  const k = Number(hb.trial) || 0;
  return n && k ? Math.max(0, n - k) : null;
}

/**
 * Seconds from the previous run's start to this run's, in this run's batch (its newest batch id; only runs that
 * batch started, not one it skipped as DONE): how long the last whole run took, start-up and test included. Null
 * for the batch's first run.
 */
function lastRunSeconds(registry, hb) {
  const runs = registryEntries(registry);
  const own = runs.find(r => r.hash === String((hb && hb.model_hash) || ''));
  if (!own || !own.batchIds.length) return null;
  const id = own.batchIds.reduce((a, b) => (b > a ? b : a));
  const since = batchStart(id);
  const start = Date.parse(own.started);
  const before = runs.filter(r => r !== own && r.batchIds.includes(id)).map(r => Date.parse(r.started))
    .filter(t => Number.isFinite(t) && (since == null || t >= since) && t < start);
  return Number.isFinite(start) && before.length ? (start - Math.max(...before)) / 1000 : null;
}

const medianOf = xs => {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : null;
};

/**
 * The runs the running run's batch still has to train, line by line, and how long each takes, from the queue and
 * the registry: {lines: [{model, runs}], runS: {model: s}, anyS, skips} or null when the run is not one of its
 * queue's lines (MODEL_ID and DATASET) or its sweep size is unknown.
 *   lines  the rest of the run's own line (its trials after this one) and every later line in full
 *          (experiment_queue.yaml's order; a line's runs = the run's trial count, trial k of n)
 *   runS   per model_id, the median of its last ten whole runs in this batch (one start to the next: start-up,
 *          training, test); anyS the batch's median over every model; a queue's lines can train models of
 *          different speeds, so each run to come is priced at its own model's
 *   skips  the queue's runs DONE in an earlier execution and not reached yet: this execution skips them in
 *          seconds, so they cost nothing
 */
function batchRuns(queues, registry, hb) {
  if (!hb) return null;
  const keys = Array.isArray(hb.batches) ? hb.batches.map(String) : [];
  const q = (queues || []).find(x => keys.includes(x.key));
  const per = Number(hb.trials) || 0;
  if (!q || !per || !Array.isArray(q.commands)) return null;
  const i = q.commands.findIndex(c => c && c.MODEL_ID === hb.model_id && (!c.DATASET || c.DATASET === hb.dataset));
  if (i < 0) return null;
  const lines = [{ model: String(hb.model_id), runs: Math.max(0, per - (Number(hb.trial) || per)) },
    ...q.commands.slice(i + 1).map(c => ({ model: String(c.MODEL_ID || ''), runs: per }))];
  const runs = registryEntries(registry);
  const own = runs.find(r => r.hash === String(hb.model_hash || ''));
  const id = own && own.batchIds.length ? own.batchIds.reduce((a, b) => (b > a ? b : a)) : latestBatchId(runs, q.key);
  const since = batchStart(id);
  const mine = runs.filter(r => id && r.batchIds.includes(id) && Number.isFinite(Date.parse(r.started))
    && (since == null || Date.parse(r.started) >= since)).sort((a, b) => Date.parse(a.started) - Date.parse(b.started));
  const cycles = {};
  const every = [];
  for (let k = 1; k < mine.length; k++) {
    const c = (Date.parse(mine[k].started) - Date.parse(mine[k - 1].started)) / 1000;    // the whole of run k - 1
    (cycles[mine[k - 1].model] = cycles[mine[k - 1].model] || []).push(c);
    every.push(c);
  }
  const runS = Object.fromEntries(Object.entries(cycles).map(([m, c]) => [m, medianOf(c.slice(-10))]));
  const skips = runs.filter(r => r.batches.includes(q.key) && r.group === 'done' && !(id && r.batchIds.includes(id))).length;
  return { lines, runS, anyS: medianOf(every.slice(-10)), skips };
}

/**
 * The running run's batch counted in runs (the batch is every run its queue plans, TOTAL RUNS QUEUED): {total, left},
 * left the runs not finished yet, this one included; null without a planned queue.
 */
function batchRunCount(queues, registry, hb, { perCommand = {} } = {}) {
  if (!hb) return null;
  const keys = Array.isArray(hb.batches) ? hb.batches.map(String) : [];
  const sized = { ...Object.fromEntries(keys.map(k => [k, Number(hb.trials) || 0])), ...perCommand };
  const q = queueStates((queues || []).filter(x => keys.includes(x.key)), registry, { perCommand: sized })[0];
  if (!q || !q.planned) return null;
  const notStarted = Math.max(0, q.planned - q.started - (q.refs.includes(String(hb.model_hash || '')) ? 0 : 1));
  return { total: q.planned, left: notStarted + 1 };
}

/**
 * When the running run ends, and its batch. hb: the heartbeat; trials: [{position, epochs, passes}] of the
 * sweep (layout.py trials); position: the run's trial number in it; commandsLeft: the queue's commands after
 * this one (each runs the whole sweep); runsLeft: the batch's runs not started yet (batchRunsLeft); lastRunS: the
 * batch's last whole run (lastRunSeconds); batch: the batch's runs line by line and their measured lengths (batchRuns).
 *
 * The run: elapsed x (1 - progress) / progress. The batch: every run its queue still has to train (TOTAL RUNS
 * QUEUED), this run's rest included. When the sweep lists the line's runs (as many trials as the run's trial count),
 * each run to come by its own epoch-passes (epochs x folds and refit, + the test) at this run's measured speed per
 * epoch-pass: the trials after this one and every later line's whole sweep, less the runs an earlier execution
 * finished (skipped in seconds). Otherwise, with the queue's lines (batch), each run to come at its own model's
 * measured whole run (else the batch's, else this run's length). Otherwise (no queue), runsLeft runs, each as long as
 * the batch's last whole run (lastRunS), else as this one. runs: {total, left} of the batch (batchRunCount), shown
 * beside its ETA. Null fields while it cannot tell yet. upperBound: early stopping may end runs sooner.
 */
function eta({ hb, nowMs, trials, position, commandsLeft = 0, runsLeft = null, lastRunS = null, batch = null, runs = null }) {
  const out = { runLeftS: null, runEndMs: null, batchLeftS: null, batchEndMs: null, upperBound: !!(hb && hb.early_stopping),
                runs: runs && Number(runs.total) > 0 ? { left: Number(runs.left), total: Number(runs.total) } : null };
  if (!hb || hb.status !== 'running' || !hb.started_ns) return out;
  const p = Number(hb.progress) || 0;
  const elapsed = (nowMs - Number(hb.started_ns) / 1e6) / 1000;
  if (p <= ETA_MIN_PROGRESS || elapsed < ETA_MIN_ELAPSED_S) return out;
  out.runLeftS = elapsed * (1 - p) / p;
  out.runEndMs = nowMs + out.runLeftS * 1000;
  // the run's speed: seconds per epoch-pass, from the units its own progress counts: its trial's epochs x passes
  // (cross-validation: the folds, and the refit unless cross_validation.refit.enabled is false) + the test, as the
  // sweep lists it; else from the heartbeat (folds x epochs + the refit's epochs + 1)
  const folds = Number(hb.n_folds) || 0;
  const epochs = Number(hb.fold_epochs || hb.epochs) || 0;
  const list = Array.isArray(trials) ? trials : [];
  const mine = list.find(t => Number(t.position) === Number(position));
  const total = mine ? trialUnits(mine) : folds * epochs + (Number(hb.refit_epochs) || epochs) + 1;
  const perLine = Number(hb.trials) || list.length;
  if (list.length && list.length === perLine && total && Number(position)) {
    // the sweep lists the line's runs (each its epochs and passes): every run still to come of the queue, this line's
    // trials after this one and every later line's whole sweep, each by its own epoch-passes at this run's speed;
    // the queue's runs an earlier execution finished are skipped in seconds (batch.skips, at a trial's mean length)
    const rate = elapsed / (p * total);
    const later = list.filter(t => Number(t.position) > Number(position)).reduce((s, t) => s + trialUnits(t), 0);
    const all = list.reduce((s, t) => s + trialUnits(t), 0);
    const skipped = Math.max(0, Number(batch && batch.skips) || 0) * (all / list.length);
    out.batchLeftS = out.runLeftS + Math.max(0, later + Math.max(0, commandsLeft) * all - skipped) * rate;
  } else if (batch && Array.isArray(batch.lines)) {
    // line by line, each run at its model's measured length; the runs finished before cost nothing
    const own = elapsed + out.runLeftS;
    const each = m => (Number(batch.runS && batch.runS[m]) > 0 ? Number(batch.runS[m]) : Number(batch.anyS) > 0 ? Number(batch.anyS) : own);
    const count = batch.lines.reduce((s, l) => s + Number(l.runs), 0);
    const cost = batch.lines.reduce((s, l) => s + Number(l.runs) * each(l.model), 0);
    const skipped = Math.min(count, Math.max(0, Number(batch.skips) || 0));
    out.batchLeftS = out.runLeftS + Math.max(0, cost - (count ? skipped * cost / count : 0));
  } else {
    // the runs still to come, each as long as the batch's last whole run, else as this one
    const left = runsLeft != null ? Number(runsLeft)
      : perLine && Number(position) ? Math.max(0, perLine - Number(position)) + Math.max(0, commandsLeft) * perLine : null;
    if (left == null) return out;
    out.batchLeftS = out.runLeftS + left * (Number(lastRunS) > 0 ? Number(lastRunS) : elapsed + out.runLeftS);
  }
  out.batchEndMs = nowMs + out.batchLeftS * 1000;
  return out;
}

const LIVE_BEAT_S = 600;            // a `running` heartbeat older than this is a crashed run's leftover

/** The ENV a command line names for a project: `make … PROJECT=<p> … ENV=<e>` or the trainer's `--project <p> --env <e>`. */
function commandEnv(cmd, project) {
  const line = String(cmd || '');
  const proj = line.match(/(?:^|\s)PROJECT=(\S+)/) || line.match(/(?:^|\s)--project[ =](\S+)/);
  if (project && proj && proj[1] !== project) return '';
  const env = line.match(/(?:^|\s)ENV=(dev|staging|prod)(?:\s|$)/) || line.match(/(?:^|\s)--env[ =](dev|staging|prod)(?:\s|$)/);
  return env ? env[1] : '';
}

/**
 * The ENV the panel follows, and why: only what has run or is running decides, never a guess.
 *   1. traceflow.env, when set
 *   2. the running run: the freshest heartbeat saying running
 *   3. the running command: the ENV= (make) or --env (trainer) of a command of this project running now
 *   4. a run in progress in the registry, while a TraceFlow process runs (polled: the machine was looked at)
 *   5. the latest snapshot: its env key (the latest run's params_out snapshot); else the latest registry run
 *   6. nothing: env '' and the panel shows no ENV until a run starts
 * heartbeats: [{env, hb}] from artifacts/<env>/heartbeat.json; commands: the command lines running on the
 * machine (BUSY_PATTERN); snapshot: {env, name} of the newest params_out snapshot, or null.
 */
function followEnv({ override, heartbeats, registry, commands, project, snapshot, nowMs, polled = false }) {
  if (override) return { env: override, source: 'the traceflow.env setting' };
  const now = nowMs || Date.now();
  const beating = (heartbeats || [])
    .filter(h => h.hb && h.hb.status === 'running' && (now - Number(h.hb.written_ns) / 1e6) / 1000 < LIVE_BEAT_S)
    .sort((a, b) => Number(b.hb.written_ns) - Number(a.hb.written_ns));
  if (beating.length) return { env: beating[0].env, source: `the running run ${beating[0].hb.model_name || ''}`.trim() };
  for (const cmd of commands || []) {
    const env = commandEnv(cmd, project);
    if (env) return { env, source: `the running command: ${String(cmd).slice(0, 120)}` };
  }
  const runs = registryEntries(registry).filter(r => r.env).sort(byStart).reverse();
  // in progress in the registry, but no process when the machine was looked at: a killed run
  const live = polled && !(commands || []).length ? null : runs.find(r => r.group === 'running');
  if (live) return { env: live.env, source: `the run in progress ${live.name}` };
  if (snapshot && snapshot.env) return { env: snapshot.env, source: `the env of the latest snapshot (${snapshot.name})` };
  if (runs.length) return { env: runs[0].env, source: `the latest run ${runs[0].name}` };
  return { env: '', source: '' };
}

const FIGURE_EXT = /\.(pdf|svg|png)$/i;

function figureStates(jobs, jobDirs) {
  // jobDirs: { <job id>: {root, dir: 'job_064_fig_26...', files: ['a.pdf', ...]} } (extension.js pollFigures:
  // the root is graphs/jobs/, or the run's graphs folder for a job of one run)
  return (jobs || []).map(job => {
    const hit = (jobDirs || {})[job.id];
    const files = hit ? (hit.files || []).filter(f => FIGURE_EXT.test(f)) : [];
    return { id: job.id, name: job.name, enabled: job.enabled !== false, dir: hit ? hit.dir : null,
      root: hit ? hit.root : null, files, drawn: files.length > 0 };
  });
}

const SNAP_FIELDS = ['BATCH', 'MODEL_ID', 'DATASET', 'ENV', 'LEAKAGE', 'STATUS', 'NAME'];
const SNAP_ENVS = ['dev', 'staging', 'prod'];
const SNAP_LEAKAGE = ['HW', 'ID', 'MSB'];
const SNAP_STATUS = ['not registered', 'new', 'queued', 'in_progress', 'interrupted', 'failed', 'done'];

/**
 * The SNAPSHOTS Scan filters. f: {BATCH, MODEL_ID, DATASET, ENV, LEAKAGE, STATUS, NAME} as typed;
 * known: {modelIds, datasets} of the project (empty for CUSTOM: any). Returns {rows} or {error}.
 */
function snapFilter(rows, f, known = {}) {
  const vals = k => String((f || {})[k] || '').split(',').map(v => v.trim()).filter(Boolean);
  const filled = SNAP_FIELDS.filter(k => vals(k).length);
  if (!filled.length) return { error: `fill at least one filter (${SNAP_FIELDS.join(', ')}) and Scan again` };
  const bad = [];
  const check = (k, allowed, label) => {
    const wrong = vals(k).filter(v => !allowed.map(a => a.toLowerCase()).includes(v.toLowerCase()));
    if (wrong.length) bad.push(`${k}=${wrong.join(',')} is not ${label}: ${allowed.join(', ')}`);
  };
  check('ENV', SNAP_ENVS, 'an ENV');
  check('LEAKAGE', SNAP_LEAKAGE, 'a leakage model');
  check('STATUS', SNAP_STATUS, 'a status');
  if ((known.modelIds || []).length) check('MODEL_ID', known.modelIds, 'a model_id of params.yaml');
  if ((known.datasets || []).length) check('DATASET', known.datasets, 'a dataset of params.yaml');
  const label = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
  if (vals('BATCH').some(v => !label.test(v))) bad.push(`BATCH=${vals('BATCH').join(',')}: a batch label is letters, digits, _ . -`);
  if (bad.length) return { error: `wrong entry: ${bad.join('; ')}` };
  const low = v => String(v == null ? '' : v).toLowerCase();
  const isIn = (val, k) => vals(k).some(v => low(val) === low(v));
  const keep = r => (!vals('BATCH').length || (r.batches || []).some(b => isIn(b, 'BATCH'))) &&
    (!vals('MODEL_ID').length || isIn(r.model_id, 'MODEL_ID')) && (!vals('DATASET').length || isIn(r.dataset, 'DATASET')) &&
    (!vals('ENV').length || isIn(r.env, 'ENV')) && (!vals('LEAKAGE').length || isIn(r.leakage_model, 'LEAKAGE')) &&
    (!vals('STATUS').length || isIn(r.status || 'not registered', 'STATUS')) &&
    (!vals('NAME').length || vals('NAME').some(v => low(r.model_name).includes(low(v)) || low(r.model_hash).startsWith(low(v))));
  const out = (rows || []).filter(keep);
  if (!out.length) {
    return { error: `no snapshot matches ${filled.map(k => `${k}=${vals(k).join(',')}`).join(' ')}` +
      ` (${(rows || []).length} scanned)` };
  }
  return { rows: out };
}

/**
 * The image build's progress from logs/build_image.log (make build-image, buildkit --progress=plain):
 * a Dockerfile step is a `#<id> [<stage> <k>/<m>]` line, done when its id says DONE or CACHED. The total
 * is the steps of the stages seen so far (a later stage adds its own when it begins).
 *   built     the log ends in `Build complete:`
 *   failed    an ERROR line
 *   building  the build process runs, or the log moved in the last 2 minutes
 *   stopped   neither (interrupted)
 * null without a log. ageS: seconds since the log was written; building: a `nerdctl build` runs now.
 */
function buildProgress(text, { ageS = 0, building = false, nowMs = Date.now() } = {}) {
  if (!text) return null;
  // BuildKit's plain progress: "#6 [base  2/10] RUN ..." (the step number is padded: \s+), "#6 DONE 1.2s" or
  // "#6 CACHED", then the export: "#15 exporting to image", "#15 exporting layers 113.6s done",
  // "#15 unpacking to <image> 38.0s done", "#15 DONE 151.6s". The export is two units of the bar (layers,
  // unpack): on a large image it is a third of the build.
  const stepOf = new Map();          // id -> 'stage k'
  const labelOf = new Map();         // 'stage k' -> what the step does (RUN pip install ..., COPY ...)
  const totals = new Map();          // stage -> m
  const done = new Set();
  let failed = false;
  let exportId = null;
  let exported = 0;                  // 0..2: layers exported, unpacked
  let current = '';                  // the step or export phase seen last and not done
  for (const line of String(text).split('\n')) {
    let m = line.match(/^#(\d+) \[(\S+)\s+(\d+)\/(\d+)\]\s*(.*)$/);
    if (m) {
      const key = `${m[2]} ${m[3]}`;
      stepOf.set(m[1], key); totals.set(m[2], Number(m[4]));
      if (!labelOf.has(key)) labelOf.set(key, m[5].replace(/\s+/g, ' ').slice(0, 60));
      if (!done.has(key)) current = `${key}/${m[4]} ${labelOf.get(key)}`.trim();
      continue;
    }
    m = line.match(/^#(\d+) exporting to image\b/);
    if (m) { exportId = m[1]; current = 'exporting the image'; continue; }
    if (exportId && line.startsWith(`#${exportId} `)) {
      if (/exporting layers .*\bdone\b/.test(line)) { exported = Math.max(exported, 1); current = 'unpacking the image'; }
      else if (/exporting layers/.test(line)) current = 'exporting layers';
      if (/unpacking to .*\bdone\b/.test(line) || /^#\d+ DONE\b/.test(line)) { exported = 2; current = ''; }
      continue;
    }
    m = line.match(/^#(\d+) (DONE|CACHED)\b/);
    if (m) {
      if (stepOf.has(m[1])) { done.add(stepOf.get(m[1])); if (current.startsWith(`${stepOf.get(m[1])}/`)) current = ''; }
      continue;
    }
    if (/^#\d+ ERROR\b|^ERROR:/.test(line)) failed = true;
  }
  const steps = [...totals.values()].reduce((a, b) => a + b, 0);
  const total = steps ? steps + 2 : 0;
  const complete = /^Build complete: /m.test(text);
  const killed = /^Build killed: /m.test(text);
  const image = (String(text).match(/^Fast build .*=> (\S+)/m) || [])[1] || '';
  const state = complete ? 'built' : killed && !building ? 'killed' : failed ? 'failed' : (building || ageS < 120) ? 'building' : 'stopped';
  const count = done.size + exported;
  const percent = complete ? 100 : total ? Math.min(99, Math.floor((100 * count) / total)) : 0;
  return { state, percent, done: complete ? total : count, total, image, phase: state === 'building' ? current : '' };
}

/**
 * The stack's containers (names starting sidechannel) from
 * `nerdctl ps -a --format '{{.Names}}\t{{.Status}}\t{{.Image}}'`: health running (Up), exited (code 0),
 * failed (another exit code), restarting, or created.
 */
function containerHealth(text) {
  return String(text || '').split('\n').map(l => l.split('\t')).filter(c => /^sidechannel/.test(c[0] || ''))
    .map(([name, status = '', image = '']) => {
      const exit = status.match(/^Exited \((-?\d+)\)/);
      const health = /^Up\b/.test(status) ? 'running' : exit ? (exit[1] === '0' ? 'exited' : 'failed')
        : /^Restarting/.test(status) ? 'restarting' : 'created';
      return { name, status, image, health };
    });
}

/**
 * Housekeeping's filters over the registry's runs. f: {ENV, MODEL_ID, DATASET, STATUS, BATCH, NAME}
 * as typed, comma-separated alternatives; an empty filter keeps every run. NAME matches a part of the
 * name, STATUS the status or its group (done, running, failed, queued), the rest the whole value.
 * known: {modelIds, datasets} of params.yaml. Returns {rows} or {error, rows: []} for a wrong entry.
 * PARAM (a value of the run's snapshot) is not filtered here: the extension asks layout.py params.
 */
const RUN_FIELDS = ['ENV', 'MODEL_ID', 'DATASET', 'STATUS', 'BATCH', 'NAME', 'PARAM'];

function runFilter(runs, f, known = {}) {
  const vals = k => String((f || {})[k] || '').split(',').map(v => v.trim()).filter(Boolean);
  const low = v => String(v == null ? '' : v).toLowerCase();
  const statuses = [...new Set((runs || []).flatMap(r => [low(r.status), low(r.group)]).filter(Boolean))].sort();
  const bad = [];
  const check = (k, allowed, label) => {
    const wrong = vals(k).filter(v => !allowed.map(low).includes(low(v)));
    if (wrong.length) bad.push(`${k}=${wrong.join(',')} is not ${label}: ${allowed.join(', ') || 'none'}`);
  };
  check('ENV', SNAP_ENVS, 'an ENV');
  if ((known.modelIds || []).length) check('MODEL_ID', known.modelIds, 'a model_id of params.yaml');
  if ((known.datasets || []).length) check('DATASET', known.datasets, 'a dataset of params.yaml');
  check('STATUS', statuses, 'a status of these runs');
  if (bad.length) return { error: `wrong entry: ${bad.join('; ')}`, rows: [] };
  const isIn = (val, k) => vals(k).some(v => low(val) === low(v));
  const keep = r => (!vals('ENV').length || isIn(r.env, 'ENV')) && (!vals('MODEL_ID').length || isIn(r.model, 'MODEL_ID')) &&
    (!vals('DATASET').length || isIn(r.dataset, 'DATASET')) &&
    (!vals('STATUS').length || isIn(r.status, 'STATUS') || isIn(r.group, 'STATUS')) &&
    (!vals('BATCH').length || (r.batches || []).some(b => isIn(b, 'BATCH'))) &&
    (!vals('NAME').length || vals('NAME').some(v => low(r.name).includes(low(v))));
  return { rows: (runs || []).filter(keep) };
}

/** data.name of a params snapshot's text (the first `name:` of the top-level data: block), '' if none. */
function snapshotDataset(text) {
  const block = (String(text || '').match(/^data:[^\n]*\n((?:[ \t-][^\n]*\n|[ \t]*\n)*)/m) || [])[1] || '';
  let best = null;
  for (const m of block.matchAll(/^([ \t-]*)name:[ \t]*["']?([^"'\s#]+)/gm)) if (!best || m[1].length < best[1].length) best = m;
  return best ? best[2] : '';
}

module.exports = {
  RUN_FIELDS, runFilter, snapshotDataset,
  buildProgress, containerHealth, batchRunsLeft, lastRunSeconds, batchRuns,
  SNAP_ENVS, SNAP_FIELDS, snapFilter,
  GREEN_S, AMBER_S, statusGroup, parseHeartbeat, light, runAlive, runSummary, registryEntries, tiles, queueStates, queuePositions,
  followEnv, commandEnv, eta, batchRunCount, figureStates, staleBeat, believedBeats, activeQueues, latestBatchRuns,
};
