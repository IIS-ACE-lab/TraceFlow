// services/visualization/vscode_monitor/test/state.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../state');

const NOW = 1_800_000_000_000;                    // ms
const beat = (ageS, extra = {}) => ({ status: 'running', written_ns: (NOW - ageS * 1000) * 1e6, ...extra });

test('light follows the age of the last beat', () => {
  assert.equal(S.light(null, NOW).color, 'grey');
  assert.equal(S.light(beat(2), NOW).color, 'green');
  assert.equal(S.light(beat(30), NOW).color, 'amber');
  assert.equal(S.light(beat(120), NOW).color, 'red');
  assert.equal(S.light(beat(120, { status: 'done' }), NOW).color, 'grey');
});

test('a garbled or half-written heartbeat counts as missing', () => {
  assert.equal(S.parseHeartbeat('{"status": "run'), null);
  assert.equal(S.parseHeartbeat(''), null);
  assert.equal(S.parseHeartbeat('{"status": "running"}'), null);          // no written_ns
  assert.equal(S.parseHeartbeat(JSON.stringify(beat(1))).status, 'running');
});

test('run summary names the phase and the percentage', () => {
  const r = S.runSummary(beat(1, { model_name: 'm', phase: 'fold', fold: 3, n_folds: 10, epoch: 7, fold_epochs: 100, progress: 0.2345 }));
  assert.equal(r.phase, 'fold 3/10 · epoch 7/100');
  assert.equal(r.percent, 23.5);
  assert.equal(S.runSummary(beat(1, { phase: 'test', progress: 0.99 })).phase, 'test');
});

test('tiles group the registry statuses', () => {
  const reg = { a: { status: 'DONE' }, b: { status: 'in_progress' }, c: { status: 'FAILED' }, d: { status: 'INTERRUPTED' },
                e: { status: 'queued' }, f: 'junk' };
  const t = S.tiles(reg);
  assert.deepEqual([t.queued.length, t.running.length, t.done.length, t.failed.length], [1, 1, 1, 2]);
  assert.deepEqual(S.tiles(null), { queued: [], running: [], done: [], failed: [] });
});

test('a run whose heartbeat still beats stays in Running though its training is recorded DONE (its test runs)', () => {
  // train_and_test records the training DONE before the test starts; the test's progress and live metrics are
  // shown under [Running], so the beating run is kept there until its heartbeat ends
  const reg = { h1: { status: 'DONE', model_name: 'pepper_basket' }, h2: { status: 'DONE', model_name: 'other' } };
  const t = S.tiles(reg, [], { beating: 'h1' });
  assert.deepEqual(t.running.map(r => r.hash), ['h1']);
  assert.deepEqual(t.done.map(r => r.hash), ['h2']);
  assert.equal(t.running[0].group, 'running');
  assert.deepEqual(S.tiles(reg, []).done.map(r => r.hash), ['h1', 'h2']);          // no beat: both done
  assert.deepEqual(S.tiles({ h3: { status: 'FAILED' } }, [], { beating: 'h3' }).failed.map(r => r.hash), ['h3']);   // a failure stays a failure
});

test('queue state comes from the runs labelled with the key', () => {
  const queues = [{ key: 'fig2', commands: [{}, {}] }, { key: 'fig3', commands: [{}] }, { key: 'fig4', commands: [{}] }];
  const reg = {
    a: { status: 'DONE', batches: ['fig2'] }, b: { status: 'DONE', batches: ['fig2', 'fig14'] },
    c: { status: 'in_progress', batches: ['fig3'] },
  };
  const q = S.queueStates(queues, reg);
  assert.deepEqual(q.map(x => x.state), ['done', 'running', 'not started']);
  assert.equal(q[0].done, 2);
});

test('a figure is drawn when its job folder holds a pdf, svg or png, wherever its folder is', () => {
  const jobs = [{ id: 64, name: 'fig_26_left', dir: 'job_064_fig_26_left' }, { id: 65, name: 'x', dir: 'job_065_x' },
    { id: 66, name: 'h', dir: 'job_066_h' }];
  const f = S.figureStates(jobs, {
    64: { root: '/p/graphs/jobs', dir: 'job_064_fig_26_left', files: ['a.pdf', 'raw.json'] },
    65: { root: '/p/artifacts/dev/generated_models/run/graphs', dir: 'job_065_x', files: ['notes.txt'] },   // one run: its folder
  });
  assert.deepEqual(f.map(x => [x.drawn, x.files, x.root]),
    [[true, ['a.pdf'], '/p/graphs/jobs'], [false, [], '/p/artifacts/dev/generated_models/run/graphs'], [false, [], null]]);
});

test('a queue is planned as total_trials x commands; positions follow start order', () => {
  const queues = [{ key: 'fig26', commands: [{}] }];
  const run = (status, started, trial) => ({ status, batches: ['fig26'], total_trials: 14, trial_index: trial, started_at: started });
  const reg = {                                    // trial_index numbered per pass (old rows): 1, 1, 2 — ignored
    a: run('DONE', '2026-10-11T03:07', 1), b: run('DONE', '2026-10-11T03:15', 1), c: run('in_progress', '2026-10-11T04:00', 2),
  };
  const [q] = S.queueStates(queues, reg);
  assert.deepEqual([q.planned, q.started, q.done, q.state], [14, 3, 2, 'running']);
  assert.deepEqual(S.queuePositions(queues, reg).c, { queue: 'fig26', k: 3, n: 14 });
  const t = S.tiles({ ...reg, c: { ...reg.c, trial_index: 3, total_remaining_trials: 11 } }, queues);
  assert.equal(t.queued.length, 11);              // total_remaining_trials of the running run: trials 4..14
  assert.deepEqual([t.queued[0].name, t.queued[10].name], ['fig26 trial 4/14', 'fig26 trial 14/14']);
});

test('queued counts total_remaining_trials of a fresh sweep, not earlier batches of the queue', () => {
  const queues = [{ key: 'fig26', commands: [{}] }];
  const old = i => ({ status: 'DONE', env: 'dev', model_id: 'M', run_session_id: 'S0', batches: ['fig26'],
                      batch_ids: ['fig26-20261012T165136Z'], trial_index: i, total_trials: 14, total_remaining_trials: 14 - i });
  const reg = Object.fromEntries([...Array(14).keys()].map(i => [`o${i}`, old(i + 1)]));
  reg.fresh = { status: 'in_progress', env: 'dev', model_id: 'M', run_session_id: 'S1', batches: ['fig26'],
                batch_ids: ['fig26-20261014T080000Z'], trial_index: 1, total_trials: 2, total_remaining_trials: 1 };
  assert.deepEqual(S.tiles(reg, queues).queued.map(r => r.name), ['fig26 trial 2/2']);
});

test('the last trial of a sweep leaves nothing pending, whatever total_remaining_trials says', () => {
  // fig26 on 2026-10-13: the running trial 14/14 recorded total_remaining_trials 1 (written at its start)
  const queues = [{ key: 'fig26', commands: [{}] }];
  const row = (status, t, left) => ({ status, env: 'dev', model_id: 'ASCAD_MLPbest', run_session_id: 'S', batches: ['fig26'],
                                      batch_ids: ['fig26-20261012T165136Z'], started_at: `2026-10-13T${String(t).padStart(2, '0')}`,
                                      trial_index: t, total_trials: 14, total_remaining_trials: left });
  const upTo = n => Object.fromEntries([...Array(n).keys()].map(i => [`h${i + 1}`,
    row(i + 1 < n ? 'DONE' : 'in_progress', i + 1, i + 1 === 14 ? 1 : 14 - i)]));
  assert.deepEqual(S.tiles(upTo(14), queues).queued, []);                              // no 'fig26 trial 15/14'
  assert.deepEqual(S.tiles(upTo(12), queues).queued.map(r => r.name), ['fig26 trial 13/14', 'fig26 trial 14/14']);
});

test('FORCE: registered queued runs are not counted twice', () => {
  const run = (status, t) => ({ status, env: 'dev', model_id: 'M', run_session_id: 'S', batches: [],
                                trial_index: t, total_trials: 4, total_remaining_trials: 4 - t });
  const reg = { a: run('in_progress', 1), b: run('queued', 2), c: run('queued', 3) };
  const q = S.tiles(reg, []).queued;
  assert.equal(q.length, 3);                              // b and c from the registry, trial 4 not started
  assert.deepEqual(q.filter(r => r.pending).map(r => r.name), ['M trial 4/4']);
});

test('a queue that stopped short of its planned trials is incomplete, not done', () => {
  const reg = { a: { status: 'DONE', batches: ['q'], total_trials: 3 } };
  assert.equal(S.queueStates([{ key: 'q', commands: [{}] }], reg)[0].state, 'incomplete');
  assert.equal(S.tiles(reg, [{ key: 'q', commands: [{}] }]).queued.length, 0);     // not running: nothing pending
});

test('a run outside a queue: not started = total - done - running in its sweep scope', () => {
  const run = (status, s, t) => ({ status, env: 'dev', model_id: 'M', run_session_id: s, total_trials: 5, batches: [],
                                   trial_index: t, total_remaining_trials: 5 - t });
  const reg = { a: run('DONE', 'S1', 1), b: run('DONE', 'S1', 2), c: run('in_progress', 'S1', 3), x: run('DONE', 'S0', 1) };
  const q = S.tiles(reg, []).queued;
  assert.deepEqual(q.map(r => r.name), ['M trial 4/5', 'M trial 5/5']);
  assert.ok(q.every(r => r.pending && r.status === 'not hashed yet'));
});

test('queued = the running queue\'s planned trials not started, the plan trial k/n counts against', () => {
  // fig24: 8 commands of one trial each; every registry row says trial 1 of 1
  const q = [{ key: 'fig24', commands: Array.from({ length: 8 }, () => ({})) }];
  const row = (status, started, session) => ({ status, batches: ['fig24'], env: 'dev', model_id: 'M', trial_index: 1,
    total_trials: 1, total_remaining_trials: 0, started_at: started, run_session_id: session });
  const one = { a: row('in_progress', '2026-10-14T06:16', 's1') };
  assert.deepEqual(S.queuePositions(q, one).a, { queue: 'fig24', k: 1, n: 8 });
  const t = S.tiles(one, q);
  assert.equal(t.queued.length, 7);
  assert.deepEqual([t.queued[0].name, t.queued[6].name], ['fig24 trial 2/8', 'fig24 trial 8/8']);
  // between two commands: no run in progress, the queue's command still alive
  const two = { a: row('DONE', '2026-10-14T06:16', 's1'), b: row('DONE', '2026-10-14T09:00', 's2') };
  assert.equal(S.tiles(two, q).queued.length, 0);
  assert.equal(S.tiles(two, q, { active: S.activeQueues(['make experiment ENV=dev PROJECT=P QUEUE=fig24']) }).queued.length, 6);
  assert.deepEqual(S.activeQueues(['python -m services.experiment_queue.experiment_queue --project P --env dev --queue fig24']), ['fig24']);
  // the last trial running: nothing after it (no trial 9/8)
  const all = Object.fromEntries(Array.from({ length: 8 }, (_, i) => [`h${i}`, row(i < 7 ? 'DONE' : 'in_progress', `2026-10-14T0${i}`, `s${i}`)]));
  assert.equal(S.tiles(all, q).queued.length, 0);
});

test('a running heartbeat of a deleted run, or with no TraceFlow process, is stale and not followed', () => {
  const reg = { aaa: { status: 'in_progress', env: 'dev', started_at: '2026-10-02', model_name: 'kept' } };
  const busy = ['make experiment ENV=dev PROJECT=P QUEUE=fig24'];
  assert.equal(S.staleBeat(beat(2, { model_hash: 'aaa' }), { registry: reg, busy }), '');
  assert.equal(S.staleBeat(beat(2, { model_hash: 'zzz', model_name: 'flock_fusion' }), { registry: reg, busy }),
    'flock_fusion is not in the run registry');
  assert.equal(S.staleBeat(beat(2, { model_hash: 'aaa', model_name: 'kept' }), { registry: reg, busy: [] }),
    'kept: no TraceFlow process runs');
  assert.equal(S.staleBeat(beat(2, { model_hash: 'aaa' }), { registry: reg, busy: null }), '');   // not polled yet
  assert.equal(S.staleBeat(beat(2, { status: 'done', model_hash: 'zzz' }), { registry: reg, busy: [] }), '');
  const [{ hb }] = S.believedBeats([{ env: 'dev', hb: beat(2, { model_hash: 'zzz', model_name: 'flock_fusion' }) }], { registry: reg, busy });
  assert.equal(hb.status, 'stale');
  assert.equal(S.light(hb, NOW).label, 'stale heartbeat: flock_fusion is not in the run registry');
  assert.equal(S.runAlive(hb, 'zzz', NOW).alive, false);
  // not followed: a stale beat, nor an in-progress registry row once the machine shows no process
  const f = o => S.followEnv({ nowMs: NOW, registry: { ...reg, b: { status: 'DONE', env: 'prod', started_at: '2026-10-01' } },
    heartbeats: [{ env: 'staging', hb }], project: 'P', ...o });
  assert.equal(f({}).env, 'dev');
  assert.equal(f({ polled: true, commands: [] }).source, 'the latest run kept');
});

test('the panel follows the ENV of what runs or ran, never a guess; the setting only when set', () => {
  const now = 1_800_000_000_000;
  const beat = (env, ageS, status = 'running') => ({ env, hb: { status, written_ns: (now - ageS * 1000) * 1e6, model_name: `m-${env}` } });
  const reg = { a: { status: 'DONE', env: 'staging', started_at: '2026-10-01', model_name: 'old' } };
  const f = o => S.followEnv({ nowMs: now, registry: reg, project: 'P', ...o });
  assert.deepEqual(f({ heartbeats: [beat('dev', 900), beat('prod', 3)] }), { env: 'prod', source: 'the running run m-prod' });
  assert.equal(f({ heartbeats: [beat('dev', 900)] }).env, 'staging');                 // a stale beat is a crashed run
  // the running command: make's ENV= or the trainer's --env, for this project only
  assert.deepEqual(f({ commands: ['make experiment ENV=prod PROJECT=P QUEUE=fig26'] }),
    { env: 'prod', source: 'the running command: make experiment ENV=prod PROJECT=P QUEUE=fig26' });
  assert.equal(f({ commands: ['/venv/bin/python -m core.cli train-cmd --project P --env dev -n a'] }).env, 'dev');
  assert.equal(f({ commands: ['make experiment ENV=prod PROJECT=OTHER QUEUE=q'] }).env, 'staging');
  assert.equal(f({ registry: { b: { status: 'in_progress', env: 'prod', started_at: '2026-10-02' }, ...reg } }).env, 'prod');
  // the latest snapshot's env key, before the registry's latest run
  assert.deepEqual(f({ snapshot: { env: 'prod', name: 'cake_bacon' } }), { env: 'prod', source: 'the env of the latest snapshot (cake_bacon)' });
  assert.equal(f({ override: 'prod', heartbeats: [beat('dev', 1)] }).env, 'prod');
  // nothing ran, nothing runs: no ENV, not a guess from experiment_queue.yaml
  assert.deepEqual(S.followEnv({}), { env: '', source: '' });
  assert.equal(S.commandEnv('make reproduce ENV=staging PROJECT=P MODEL=a', 'P'), 'staging');
  assert.equal(S.commandEnv('make graph ENV=dev PROJECT=P', 'Q'), '');
});

test('eta: the run from its progress, the batch adds the trials to come at the run\'s speed', () => {
  const now = 1_800_000_000_000;
  const hb = { status: 'running', started_ns: (now - 3600e3) * 1e6, progress: 0.25, n_folds: 10, fold_epochs: 100, refit_epochs: null };
  const trials = [1, 2, 3].map(position => ({ position, epochs: 100, passes: 11 }));
  const e = S.eta({ hb, nowMs: now, trials, position: 2 });
  assert.equal(e.runLeftS, 3 * 3600);                      // a quarter in an hour: three hours to go
  // speed: 3600 s for 0.25 x 1101 epoch-passes; trial 3 is 1101 more
  assert.ok(Math.abs(e.batchLeftS - (3 * 3600 + 1101 * 3600 / (0.25 * 1101))) < 1e-6);
  const two = S.eta({ hb, nowMs: now, trials, position: 2, commandsLeft: 1 });
  assert.ok(Math.abs(two.batchLeftS - e.batchLeftS - 3 * 1101 * 3600 / (0.25 * 1101)) < 1e-6);
  assert.equal(S.eta({ hb: { ...hb, progress: 0 }, nowMs: now, trials, position: 2 }).runLeftS, null);      // nothing trained yet
  assert.equal(S.eta({ hb: { ...hb, progress: 0.01 }, nowMs: now, trials, position: 2 }).runLeftS, 99 * 3600);  // from the first progress
  const young = { ...hb, started_ns: (now - 4e3) * 1e6 };
  assert.equal(S.eta({ hb: young, nowMs: now, trials, position: 2 }).runLeftS, null);                       // not a heartbeat old
  assert.equal(S.eta({ hb: { ...hb, early_stopping: true }, nowMs: now }).upperBound, true);
  assert.equal(S.eta({ hb: { ...hb, status: 'done' }, nowMs: now }).runLeftS, null);
});

test('a queue counts only the runs of its latest execution (newest batch id)', () => {
  const queues = [{ key: 'fig26', commands: [{}] }, { key: 'fig2', commands: [{}] }];
  const run = (status, id, t, total, started) => ({ status, batches: ['fig26'], batch_ids: [id], trial_index: t,
                                                    total_trials: total, started_at: started });
  const reg = {
    a: run('DONE', 'fig26-20261010T190703Z', 1, 14, '2026-10-11T03:07'),
    b: run('DONE', 'fig26-20261010T190703Z', 2, 14, '2026-10-11T03:15'),
    c: run('in_progress', 'fig26-20261012T053115Z', 1, 2, '2026-10-12T13:42'),
    d: { status: 'DONE', batches: ['fig2'], batch_ids: ['fig2-20261001T000000Z'], total_trials: 1 },
  };
  const [q26, q2] = S.queueStates(queues, reg);
  assert.deepEqual([q26.started, q26.planned, q26.done, q26.state], [1, 2, 0, 'running']);
  assert.deepEqual([q2.started, q2.done, q2.state], [1, 1, 'done']);          // fig2 ids never match fig26
  assert.deepEqual(S.queuePositions(queues, reg).c, { queue: 'fig26', k: 1, n: 2 });
  assert.equal(S.queuePositions(queues, reg).a, undefined);                   // an earlier execution
});

test('rows without batch ids fall back to the batches label', () => {
  const reg = { a: { status: 'DONE', batches: ['q'], total_trials: 1 } };
  assert.equal(S.queueStates([{ key: 'q', commands: [{}] }], reg)[0].state, 'done');
});

test('runAlive: green only for the run whose own heartbeat beats; red when it stops, ends or another run beats', () => {
  assert.equal(S.runAlive(beat(5, { model_hash: 'h1' }), 'h1', NOW).alive, true);
  assert.equal(S.runAlive(beat(30, { model_hash: 'h1' }), 'h1', NOW).alive, true);
  assert.match(S.runAlive(beat(90, { model_hash: 'h1' }), 'h1', NOW).label, /dead: no beat for 90 s/);
  assert.equal(S.runAlive(beat(5, { model_hash: 'h1', status: 'failed' }), 'h1', NOW).alive, false);
  assert.match(S.runAlive(beat(5, { model_hash: 'h2' }), 'h1', NOW).label, /no heartbeat from this run/);
  assert.equal(S.runAlive(null, 'h1', NOW).alive, false);
});

test('snapFilter: at least one filter, every filled one valid, the rows matching all of them', () => {
  const rows = [
    { env: 'dev', batches: ['fig26'], model_id: 'ASCAD_MLPbest', dataset: 'ascad', leakage_model: 'HW', status: 'DONE', model_name: 'cake_bacon', model_hash: '816fd431' },
    { env: 'dev', batches: ['fig26'], model_id: 'ASCAD_MLPbest', dataset: 'ascad', leakage_model: 'ID', status: '', model_name: 'celery_plush', model_hash: '7b8a271d' },
    { env: 'staging', batches: ['fig6'], model_id: 'ASCAD_CNNbase', dataset: 'ascad_desync100', leakage_model: 'ID', status: 'INTERRUPTED', model_name: 'x_y', model_hash: 'aaaa0000' },
  ];
  const known = { modelIds: ['ASCAD_MLPbest', 'ASCAD_CNNbase'], datasets: ['ascad', 'ascad_desync100'] };
  assert.match(S.snapFilter(rows, {}, known).error, /fill at least one filter/);
  assert.match(S.snapFilter(rows, { ENV: 'devv' }, known).error, /wrong entry: ENV=devv is not an ENV/);
  assert.match(S.snapFilter(rows, { MODEL_ID: 'ASCAD_MLP' }, known).error, /MODEL_ID=ASCAD_MLP is not a model_id of params.yaml/);
  assert.match(S.snapFilter(rows, { BATCH: 'fig 26' }, known).error, /a batch label is letters/);
  assert.match(S.snapFilter(rows, { BATCH: 'fig9' }, known).error, /no snapshot matches BATCH=fig9 \(3 scanned\)/);
  assert.deepEqual(S.snapFilter(rows, { BATCH: 'fig26' }, known).rows.map(r => r.model_name), ['cake_bacon', 'celery_plush']);
  assert.deepEqual(S.snapFilter(rows, { BATCH: 'fig26', LEAKAGE: 'hw' }, known).rows.map(r => r.model_name), ['cake_bacon']);   // AND
  assert.deepEqual(S.snapFilter(rows, { STATUS: 'not registered, interrupted' }, known).rows.map(r => r.model_name), ['celery_plush', 'x_y']);  // OR
  assert.deepEqual(S.snapFilter(rows, { NAME: '816f' }, known).rows.map(r => r.model_name), ['cake_bacon']);
  assert.deepEqual(S.snapFilter(rows, { MODEL_ID: 'anything' }, {}).error, 'no snapshot matches MODEL_ID=anything (3 scanned)');   // CUSTOM: any value
});

test('build progress: steps of the stages seen, done when DONE or CACHED; built, failed, building, stopped', () => {
  const head = 'Fast build (type=image -> containerd) => sidechannel:dev-ffc2908\n' +
    '#1 [internal] load build definition from Dockerfile\n#1 DONE 0.0s\n' +
    '#5 [base 1/4] FROM docker.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04@sha256:44e4\n#5 DONE 34.4s\n' +
    '#5 [base 1/4] FROM docker.io/nvidia/cuda:12.8.0-runtime-ubuntu24.04@sha256:44e4\n#5 DONE 34.5s\n' +
    '#6 [base 2/4] RUN set -eux; useradd\n#6 CACHED\n#7 [base 3/4] RUN apt-get update\n';
  // 4 steps + 2 export units (layers, unpack); the step running now is the phase
  assert.deepEqual(S.buildProgress(head, { ageS: 5 }), { state: 'building', percent: 33, done: 2, total: 6, image: 'sidechannel:dev-ffc2908',
    phase: 'base 3/4 RUN apt-get update' });
  assert.equal(S.buildProgress(head, { ageS: 600 }).state, 'stopped');
  assert.equal(S.buildProgress(head, { ageS: 600, building: true }).state, 'building');
  assert.equal(S.buildProgress(head + '#7 ERROR: process "apt-get" did not complete\n', { ageS: 5 }).state, 'failed');
  const built = S.buildProgress(head + '#7 DONE 82.2s\n#8 [base 4/4] WORKDIR /app\n#8 DONE 0.0s\nBuild complete: sidechannel:dev-ffc2908\n', { ageS: 900 });
  assert.deepEqual([built.state, built.percent, built.done], ['built', 100, 6]);
  // BuildKit pads the step number ([base  2/10]); the export counts until it is done, and the bar never shows
  // 100 before 'Build complete'
  const padded = 'Fast build (type=image -> containerd) => sidechannel:dev-x\n' +
    '#5 [base  1/2] FROM docker.io/nvidia/cuda\n#5 DONE 1.0s\n#6 [base  2/2] RUN pip install\n#6 DONE 180.0s\n' +
    '#15 exporting to image\n#15 exporting layers\n';
  assert.deepEqual(S.buildProgress(padded, { ageS: 5 }), { state: 'building', percent: 50, done: 2, total: 4, image: 'sidechannel:dev-x',
    phase: 'exporting layers' });
  const unpacking = S.buildProgress(padded + '#15 exporting layers 113.6s done\n#15 unpacking to sidechannel:dev-x\n', { ageS: 5 });
  assert.deepEqual([unpacking.percent, unpacking.phase], [75, 'unpacking the image']);
  const exported = S.buildProgress(padded + '#15 exporting layers 113.6s done\n#15 unpacking to sidechannel:dev-x 38.0s done\n#15 DONE 151.6s\n', { ageS: 5 });
  assert.deepEqual([exported.state, exported.percent, exported.done], ['building', 99, 4]);
  assert.equal(S.buildProgress(head + 'Build killed: 2026-10-13T21:00:00+00:00\n', { ageS: 5 }).state, 'killed');
  assert.equal(S.buildProgress(head + 'Build killed: x\n', { ageS: 5, building: true }).state, 'building');   // built again
  assert.equal(S.buildProgress('', {}), null);
});

test('container health of the sidechannel containers only', () => {
  const ps = 'sidechannel-dev-x\tUp 3 minutes\tsidechannel:dev-a\nharbor-core\tCreated\tgoharbor/harbor-core\n' +
    'sidechannel\tExited (137) 2 hours ago\tsidechannel:dev-b\nsidechannel-prod\tExited (0) 1 day ago\ti\n';
  assert.deepEqual(S.containerHealth(ps).map(c => [c.name, c.health]),
    [['sidechannel-dev-x', 'running'], ['sidechannel', 'failed'], ['sidechannel-prod', 'exited']]);
  assert.deepEqual(S.containerHealth(''), []);
});

test('Housekeeping filters: empty keeps every run; fields combine; NAME a part, STATUS the status or group; wrong entries refused', () => {
  const runs = [
    { hash: 'h1', name: 'cake_bacon', env: 'dev', model: 'ASCAD_MLPbest', dataset: 'ascad', status: 'DONE', group: 'done', batches: ['fig26'] },
    { hash: 'h2', name: 'meh_aqua', env: 'dev', model: 'ASCAD_CNNbest', dataset: 'ascad_desync50', status: 'failed', group: 'failed', batches: ['fig13'] },
    { hash: 'h3', name: 'lofty_owl', env: 'staging', model: 'ASCAD_MLPbest', dataset: 'ascad', status: 'in_progress', group: 'running', batches: [] },
  ];
  const known = { modelIds: ['ASCAD_MLPbest', 'ASCAD_CNNbest'], datasets: ['ascad', 'ascad_desync50'] };
  const names = f => S.runFilter(runs, f, known).rows.map(r => r.name);
  assert.deepEqual(names({}), ['cake_bacon', 'meh_aqua', 'lofty_owl']);
  assert.deepEqual(names({ ENV: 'dev' }), ['cake_bacon', 'meh_aqua']);
  assert.deepEqual(names({ ENV: 'dev', MODEL_ID: 'ascad_mlpbest' }), ['cake_bacon']);
  assert.deepEqual(names({ DATASET: 'ascad_desync50' }), ['meh_aqua']);
  assert.deepEqual(names({ STATUS: 'done,running' }), ['cake_bacon', 'lofty_owl']);
  assert.deepEqual(names({ BATCH: 'fig26, fig13' }), ['cake_bacon', 'meh_aqua']);
  assert.deepEqual(names({ NAME: 'OWL' }), ['lofty_owl']);
  assert.match(S.runFilter(runs, { ENV: 'qa' }, known).error, /wrong entry: ENV=qa is not an ENV/);
  assert.match(S.runFilter(runs, { DATASET: 'ascad_desync9' }, known).error, /DATASET=ascad_desync9 is not a dataset of params.yaml/);
  assert.match(S.runFilter(runs, { STATUS: 'queued' }, known).error, /STATUS=queued is not a status of these runs/);
});

test('a snapshot\'s dataset is the first name of its data block', () => {
  assert.equal(S.snapshotDataset('env: dev\ndata:\n- name: ascad\n  database: x.h5\n  attack:\n    name: inner\nmodel:\n  name: m\n'), 'ascad');
  assert.equal(S.snapshotDataset('data:\n  alias: a\n  name: "ascad_desync50"\n'), 'ascad_desync50');
  assert.equal(S.snapshotDataset('model:\n  name: m\n'), '');
});

test('a queue\'s planned runs come from the runs this execution trained, not a DONE run it skipped (fig10: 6, not 30)', () => {
  // ketchup_computer was trained under fig10_optimizers (5 optimizers: total_trials 5) and joined fig10's
  // execution as already DONE; foamy_crackle is fig10's own run of a 1-combination sweep
  const reg = {
    c8802d52: { model_name: 'ketchup_computer', status: 'DONE', env: 'dev', total_trials: 5, started_at: '2026-10-15T20:16:17+08:00',
      batches: ['fig10_optimizers', 'fig10'], batch_ids: ['fig10_optimizers-20261015T121500Z', 'fig10-20261019T231500Z'] },
    e26f1914: { model_name: 'foamy_crackle', status: 'INTERRUPTED', env: 'dev', total_trials: 1, started_at: '2026-10-20T07:16:18+08:00',
      batches: ['fig10'], batch_ids: ['fig10-20261019T231500Z'] },
  };
  const [q] = S.queueStates([{ key: 'fig10', commands: [{}, {}, {}, {}, {}, {}] }], reg);
  assert.deepEqual([q.planned, q.runs, q.done], [6, 2, 1]);
});

test('a retried queue that has trained nothing yet: its plan from the runs it trained before, not a run of another queue (fig10 retry: 6)', () => {
  // Retry Start opened fig10-20261019T232656Z; foamy_crackle (fig10's own, interrupted before it) and ketchup_computer
  // (trained by fig10_optimizers, total_trials 5) both carry the new id, and neither started after it
  const reg = {
    c8802d52: { model_name: 'ketchup_computer', status: 'DONE', env: 'dev', total_trials: 5, started_at: '2026-10-15T20:16:17+08:00',
      batches: ['fig10_optimizers', 'fig10'], batch_ids: ['fig10_optimizers-20261015T120942Z', 'fig10-20261019T230835Z', 'fig10-20261019T232656Z'] },
    e26f1914: { model_name: 'foamy_crackle', status: 'INTERRUPTED', env: 'dev', total_trials: 1, started_at: '2026-10-20T07:16:18+08:00',
      batches: ['fig10'], batch_ids: ['fig10-20261019T230835Z', 'fig10-20261019T232656Z'] },
  };
  const [q] = S.queueStates([{ key: 'fig10', commands: [{}, {}, {}, {}, {}, {}] }], reg);
  assert.equal(q.planned, 6);
});

test('a queue whose only run another queue trained: one run per line, never that run\'s sweep (fig10 after deleting foamy_crackle)', () => {
  const reg = {
    c8802d52: { model_name: 'ketchup_computer', status: 'DONE', env: 'dev', total_trials: 5, started_at: '2026-10-15T20:16:17+08:00',
      batches: ['fig10_optimizers', 'fig10'], batch_ids: ['fig10_optimizers-20261015T120942Z', 'fig10-20261019T230835Z', 'fig10-20261019T232656Z'] },
  };
  const [q] = S.queueStates([{ key: 'fig10', commands: [{}, {}, {}, {}, {}, {}] }], reg);
  assert.deepEqual([q.planned, q.done, q.state], [6, 1, 'incomplete']);
});

test('a queue with no run of its own is never sized by what params.yaml plans now: one run per line (fig5: 8, not 8 x fig20\'s 5)', () => {
  // params.yaml held fig20's five optimizers while fig5, fig22, fig23 waited: they showed 40, 35, 40
  const [q] = S.queueStates([{ key: 'fig5', commands: Array.from({ length: 8 }, () => ({})) }], {}, { perCommand: 5 });
  assert.deepEqual([q.planned, q.done, q.state], [8, 0, 'not started']);
});

// fig24 (8 lines, 1 trial each) trained haberdashery_flock (ASCAD_MLPbase); fig20 (1 line, 5 optimizers) retrains it
// with FORCE: its row now carries fig20's start, sweep (total_trials 5) and status. FORCE queued four older
// MLPbase runs under the same session, still holding only their old batch id.
const fig24 = 'fig24-20261019T151315Z';
const fig20 = 'fig20-20261020T085403Z';
const forced = () => {
  const reg = {
    hab: { model_name: 'haberdashery_flock', status: 'in_progress', env: 'dev', model_id: 'ASCAD_MLPbase', total_trials: 5, trial_index: 2,
      started_at: '2026-10-20T18:17:16+08:00', run_session_id: 'S20', batches: ['fig24', 'fig20'],
      batch_ids: ['fig24-20261019T081932Z', fig24, fig20], force_previous_status: 'DONE' },
    bir: { model_name: 'birthday_fusion', status: 'DONE', env: 'dev', model_id: 'ASCAD_MLPbase', total_trials: 5, trial_index: 1,
      started_at: '2026-10-20T16:55:14+08:00', run_session_id: 'S20', batches: ['fig20'],
      batch_ids: ['fig20-20261014T185623Z', fig20] },
  };
  for (let i = 0; i < 7; i++) {
    reg[`a${i}`] = { model_name: `act${i}`, status: 'DONE', env: 'dev', model_id: `ASCAD_MLP_${i}`, total_trials: 1, trial_index: 1,
      started_at: `2026-10-19T2${i}:00:00+08:00`, run_session_id: `S24-${i}`, batches: ['fig24'], batch_ids: [fig24] };
  }
  ['sc', 'ba', 'ob', 'wa'].forEach((h, i) => {
    reg[h] = { model_name: h, status: 'queued', env: 'dev', model_id: 'ASCAD_MLPbase', total_trials: 5, trial_index: 3 + i,
      started_at: `2026-10-15T0${4 + i}:00:00+08:00`, run_session_id: 'S20', batches: ['fig20'],
      batch_ids: ['fig20-20261014T185623Z'], force_previous_status: 'DONE' };
  });
  return reg;
};
const fig24Queues = [{ key: 'fig20', commands: [{}] }, { key: 'fig24', commands: Array.from({ length: 8 }, () => ({})) }];

test('a run another queue retrains neither sizes nor runs the queue it left (fig24: 8 planned, not 8 x fig20\'s 5)', () => {
  const reg = forced();
  const q24 = S.queueStates(fig24Queues, reg).find(q => q.key === 'fig24');
  assert.deepEqual([q24.planned, q24.done], [8, 7]);
  assert.notEqual(q24.state, 'running');                                    // its Terminate Run would stop fig20
  assert.equal(S.queuePositions([...fig24Queues].reverse(), reg).hab.queue, 'fig20');   // fig24 listed first: still fig20's
  assert.ok(!S.tiles(reg, fig24Queues).queued.some(r => r.queue === 'fig24'));
  // stopped there: the run fig20 broke fails fig24 too (Retry Start resumes it)
  const stopped = { ...reg, hab: { ...reg.hab, status: 'INTERRUPTED' } };
  const after = S.queueStates(fig24Queues, stopped).find(q => q.key === 'fig24');
  assert.deepEqual([after.planned, after.done, after.state], [8, 7, 'failed']);
});

test('the runs FORCE queued for the running execution count once: listed, numbered after the started ones, none pending on top', () => {
  const reg = forced();
  const q20 = S.queueStates(fig24Queues, reg).find(q => q.key === 'fig20');
  assert.deepEqual([q20.planned, q20.done, q20.state], [6, 1, 'running']);
  const pos = S.queuePositions(fig24Queues, reg);
  assert.deepEqual(['bir', 'hab', 'sc', 'ba', 'ob', 'wa'].map(h => pos[h].k), [1, 2, 3, 4, 5, 6]);
  assert.equal(pos.wa.n, 6);
  const queued = S.tiles(reg, fig24Queues).queued;
  assert.deepEqual(queued.map(r => r.hash).sort(), ['ba', 'ob', 'sc', 'wa']);         // the four FORCE rows, no 'fig20 trial k/6'
});

test('a started queue is sized from its start: its frozen params.yaml\'s sweep before the first run registers, its runs after', () => {
  const queues = [{ key: 's3_ascad_n100_zaid', commands: [{ MODEL_ID: 'A' }, { MODEL_ID: 'B' }] }];
  // started, nothing registered yet: 2 commands x the frozen sweep of 3 seeds
  const early = S.queueStates(queues, {}, { perCommand: { s3_ascad_n100_zaid: 3 } })[0];
  assert.deepEqual([early.planned, early.commands, early.done], [6, 2, 0]);
  assert.equal(S.queueStates(queues, {})[0].planned, 2);                                   // unknown: one per line, as before
  const pending = S.tiles({}, queues, { active: ['s3_ascad_n100_zaid'], perCommand: { s3_ascad_n100_zaid: 3 } }).queued;
  assert.equal(pending.length, 6);                                                         // every planned trial, from the start
  // once a run registers, the queue's own runs size it (its total_trials), whatever the frozen count says
  const reg = { h1: { model_name: 'r1', model_id: 'A', env: 'dev', status: 'IN_PROGRESS', batches: ['s3_ascad_n100_zaid'],
    batch_ids: ['s3_ascad_n100_zaid-20261024T010000Z'], started_at: '2026-10-24T09:00:00+08:00', trial_index: 1, total_trials: 3 } };
  assert.equal(S.queueStates(queues, reg, { perCommand: { s3_ascad_n100_zaid: 5 } })[0].planned, 6);
});

test('the batch ETA counts every run still to come: the queue\'s plan, else the run\'s own trial count; each as long as this run', () => {
  const now = 1_800_000_000_000;
  // ZvW: params.yaml's sweep is one trial, the line's ten splits come from the data entry; trial 4 of 10, 2 lines after it
  const hb = { status: 'running', started_ns: (now - 20e3) * 1e6, progress: 0.5, epochs: 50, n_folds: 0, trial: 4, trials: 10,
               batches: ['wq'], model_hash: 'h4' };
  const one = [{ position: 1, epochs: 50, passes: 1 }];
  const e = S.eta({ hb, nowMs: now, trials: one, position: 4, commandsLeft: 2 });
  assert.equal(e.runLeftS, 20);
  assert.equal(e.batchLeftS, 20 + (6 + 2 * 10) * 40);                 // 6 more on this line, 20 on the next two, 40 s each
  assert.equal(S.eta({ hb, nowMs: now, trials: one, position: 4, commandsLeft: 2, runsLeft: 23 }).batchLeftS, 20 + 23 * 40);
  // the queue's plan: 3 lines x 10, 4 started (this one among them): 26 to come
  const row = (h, t) => ({ model_name: h, env: 'dev', status: 'DONE', batches: ['wq'], batch_ids: ['wq-20261025T000000Z'],
    started_at: '2026-10-25T08:00:00+08:00', trial_index: t, total_trials: 10 });
  const reg = { h1: row('h1', 1), h2: row('h2', 2), h3: row('h3', 3), h4: { ...row('h4', 4), status: 'IN_PROGRESS' } };
  const queues = [{ key: 'wq', commands: [{}, {}, {}] }];
  assert.equal(S.batchRunsLeft(queues, reg, hb), 26);
  assert.equal(S.batchRunsLeft(queues, { h1: reg.h1 }, hb), 28);        // this run not registered yet: counted all the same
  assert.equal(S.batchRunsLeft([], {}, hb), 6);                         // no queue: the run's own line
  // DLSCA: the sweep lists the line's trials (5 of different epochs): each by its own epochs, as before
  const five = [1, 2, 3, 4, 5].map(position => ({ position, epochs: position * 10, passes: 1 }));
  const d = S.eta({ hb: { ...hb, trials: 5, trial: 2, epochs: 20 }, nowMs: now, trials: five, position: 2 });
  const rate = 20 / (0.5 * 21);                                          // s per epoch-pass
  assert.ok(Math.abs(d.batchLeftS - (20 + (31 + 41 + 51) * rate)) < 1e-9);
});

test('each run to come is as long as the batch\'s last whole run: the previous start to this one\'s', () => {
  const at = s => new Date(Date.UTC(2026, 9, 25, 18, 0, 0) + s * 1000).toISOString();
  const row = (start, t, id = 'wq-20261025T180000Z') => ({ model_name: `r${t}`, env: 'dev', status: 'DONE', batches: ['wq'],
    batch_ids: [id], started_at: at(start), trial_index: t, total_trials: 10 });
  const reg = { a: row(5, 1), b: row(30, 2), c: { ...row(56, 3), status: 'IN_PROGRESS' },
                old: { ...row(-600, 9), started_at: at(-600) } };            // skipped as DONE: started before the batch
  const hb = { status: 'running', model_hash: 'c', started_ns: Date.parse(at(56)) * 1e6, progress: 0.5, epochs: 50, n_folds: 0,
               trial: 3, trials: 10, batches: ['wq'] };
  assert.equal(S.lastRunSeconds(reg, hb), 26);
  assert.equal(S.lastRunSeconds({ a: reg.a }, { ...hb, model_hash: 'a' }), null);                // the batch's first run
  const nowMs = Date.parse(at(66));
  const e = S.eta({ hb, nowMs, trials: [{ position: 1, epochs: 50, passes: 1 }], position: 3, runsLeft: 7, lastRunS: 26 });
  assert.equal(e.runLeftS, 10);
  assert.equal(e.batchLeftS, 10 + 7 * 26);
});

test('the whole batch, line by line: each run to come at its own model\'s measured run; runs finished before cost nothing', () => {
  const at = s => new Date(Date.UTC(2026, 9, 25, 16, 0, 0) + s * 1000).toISOString();
  const id = 'wq-20261025T160000Z';
  const row = (model, start, trial, extra = {}) => ({ model_name: `${model}${trial}${start}`, model_id: model, env: 'dev', status: 'DONE',
    batches: ['wq'], batch_ids: [id], started_at: at(start), finished_at: at(start + 60), trial_index: trial, total_trials: 3, ...extra });
  // line 1: Z, 100 s a run (3 runs); line 2: N, 50 s a run (3 runs); line 3 (Z) running its trial 1
  const reg = { a: row('Z', 0, 1), b: row('Z', 100, 2), c: row('Z', 200, 3), d: row('N', 300, 1), e: row('N', 350, 2), f: row('N', 400, 3),
    g: row('Z', 450, 1, { status: 'IN_PROGRESS', finished_at: '' }),
    old: row('N', -900, 1, { batch_ids: ['wq-20261024T000000Z'] }) };     // line 4's trial 1, DONE in an earlier execution
  const hb = { status: 'running', model_hash: 'g', model_id: 'Z', dataset: 'z2', batches: ['wq'], trial: 1, trials: 3,
               started_ns: Date.parse(at(450)) * 1e6, progress: 0.5, epochs: 50, n_folds: 0 };
  const queues = [{ key: 'wq', commands: [{ MODEL_ID: 'Z', DATASET: 'z1' }, { MODEL_ID: 'N', DATASET: 'n1' },
                                          { MODEL_ID: 'Z', DATASET: 'z2' }, { MODEL_ID: 'N', DATASET: 'n2' }] }];
  const b = S.batchRuns(queues, reg, hb);
  assert.deepEqual(b.lines, [{ model: 'Z', runs: 2 }, { model: 'N', runs: 3 }]);
  assert.deepEqual(b.runS, { Z: 100, N: 50 });
  assert.equal(b.skips, 1);
  const nowMs = Date.parse(at(480));                                   // 30 s in, half done: 30 s left
  const e = S.eta({ hb, nowMs, trials: [{ position: 1, epochs: 50, passes: 1 }], position: 1, commandsLeft: 1, batch: b });
  assert.equal(e.runLeftS, 30);
  // 2 Z runs (200 s) + 3 N runs (150 s), less the one finished before at the average of the 5 (70 s)
  assert.equal(e.batchLeftS, 30 + 350 - 70);
  assert.equal(S.batchRuns(queues, reg, { ...hb, dataset: 'other' }), null);   // not one of the queue's lines: the fallback
});

test('batch ETA: a sweep of growing trials is priced by each trial\'s epoch-passes, not by the short runs done so far', () => {
  // fig26 (2026-10-27): 10-fold CV, refit off (passes 10), trials of 20 ... 800 epochs; trial 2 of 3 runs now
  const trials = [{ position: 1, epochs: 20, passes: 10 }, { position: 2, epochs: 200, passes: 10 }, { position: 3, epochs: 800, passes: 10 }];
  const started = 1_000_000;
  const hb = { status: 'running', started_ns: started * 1e6, progress: 0.5, trial: 2, trials: 3, n_folds: 10, fold_epochs: 200,
               batches: ['fig26'], model_id: 'M', model_hash: 'h2' };
  const nowMs = started + 1000 * 1000;                               // 1,000 s in, half done: 2,000 s per trial-2 run
  const shortRuns = { lines: [{ model: 'M', runs: 1 }], runS: { M: 200 }, anyS: 200, skips: 0 };   // trial 1 took 200 s
  const e = S.eta({ hb, nowMs, trials, position: 2, batch: shortRuns, runs: { total: 3, left: 2 } });
  assert.equal(Math.round(e.runLeftS), 1000);
  // trial 3: 800 x 10 + 1 units at trial 2's rate (2,000 s / 2,001 units): about 8,000 s, never 200 s
  assert.equal(Math.round(e.batchLeftS), Math.round(1000 + (800 * 10 + 1) * (1000 / (0.5 * (200 * 10 + 1)))));
  assert.deepEqual(e.runs, { left: 2, total: 3 });
});

test('batch run count: the runs the queue plans, and those not finished, this one included', () => {
  const queues = [{ key: 'q', commands: [{ MODEL_ID: 'M' }, { MODEL_ID: 'M' }] }];
  const hb = { batches: ['q'], trials: 3, model_hash: 'h2' };
  const registry = { h1: { model_name: 'a', status: 'DONE', batches: ['q'], model_id: 'M' }, h2: { model_name: 'b', status: 'RUNNING', batches: ['q'], model_id: 'M' } };
  assert.deepEqual(S.batchRunCount(queues, registry, hb), { total: 6, left: 5 });
  assert.equal(S.batchRunCount([], registry, hb), null);
});
