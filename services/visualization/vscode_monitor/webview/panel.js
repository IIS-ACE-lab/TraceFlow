// services/visualization/vscode_monitor/webview/panel.js
/**
 * panel.js
 * ========
 * Draws what extension.js sends and posts clicks back. Two modes, set by
 * the page: `main` (the side-bar view: GPU strip, then one collapsible per
 * project) and `detail` (one run, beside the editor). The main view is a
 * set of regions (GPU strip; per project its heading, current run, tiles,
 * queues, figures, configs, errors), each redrawn only when its own HTML
 * changes, so a progress bar ticking every two seconds never closes an
 * open section, and a region holding the focused input is left alone until
 * the input loses focus. Which sections are open, and which tile group is
 * listed, is kept in the webview state.
 *
 * Config forms: entries only (yamledit.py has dropped the comments). A
 * value is an input; a list of values is one input per item with × (remove)
 * and a `+ add` input. Changes are staged, not written: the form shows them
 * (highlighted, struck through, added) and writes them only on OK, the form's
 * (every change) or a block's (the changes inside it); Cancel forgets them.
 * The file is the source of truth: a change on disk redraws the form at once
 * and drops the staged changes drawn from the old text. The panel never
 * copies or deletes a block (do that in the file). Each write carries the
 * hash of the text the form was drawn from. Nested sections are drawn only
 * while open, so the 82 jobs of a graph_config.yaml cost nothing closed.
 *
 * Buttons with data-run post {type: 'run', action, project, key, id, env};
 * extension.js asks, checks and runs them.
 *
 * Author: Shraddha Mishra
 */
(function () {
  'use strict';
  const vscode = acquireVsCodeApi();
  const mode = document.body.dataset.mode;
  const app = document.getElementById('app');
  const GROUPS = ['queued', 'running', 'done', 'failed'];
  const GLYPH = { queued: '○', running: '◉', done: '✓', failed: '✕' };   // shape as well as colour: readable without colour
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const num = v => (v == null || !Number.isFinite(Number(v)) ? '-' : String(Math.round(Number(v) * 100) / 100));
  const send = m => vscode.postMessage(m);
  const ui = Object.assign({ open: {}, group: {} }, vscode.getState() || {});
  ui.share = false;                    // the status page's settings start collapsed after every load (an update, a reload)
  const save = () => vscode.setState(ui);
  let forceOpen = false;               // a key search draws its matches' parents open (configsHtml), without keeping that
  const isOpen = (key, dflt) => (forceOpen ? true : key in ui.open ? ui.open[key] : dflt);
  let drawn = '';
  let regions = null;              // main view: {id: {el, html}} once the skeleton exists
  let skeleton = '';
  let last = null;
  const trees = {};                // config file -> {sha256, tree, error, note}
  const pending = {};              // config file -> {sha, edits: {id: {op, path, value, key}}, seq}: staged, not written
  let curFile = '';                // the file whose form is being drawn (render helpers read its staged changes)
  let curProject = '';             // its project (the menus of project names read it)
  let MENUS = { rules: {}, refs: {} };   // choices.py: option lists from the code, project names
  const requested = new Set();     // config files whose form was asked for
  // a held mouse button: the element pressed. No region holding it is redrawn until its click has run (owner 2026-10-28:
  // Reset under a Budget box changed in place: the blur's redraw replaced the button and the click was lost)
  let pressed = null;
  let drawAfterPress = false;

  /** The whole page at once: errors and the detail view. */
  function draw(html) {
    regions = null;
    skeleton = '';
    if (html !== drawn) { app.innerHTML = html; drawn = html; }
  }

  function drawRegions(parts) {
    const focus = document.activeElement;
    // a list box keeps where it was scrolled to across the redraw
    const scrolled = new Map([...app.querySelectorAll('[data-scrollkey]')].map(el => [el.dataset.scrollkey, el.scrollTop]));
    for (const [id, html] of Object.entries(parts)) {
      const r = regions[id];
      if (!r || r.html === html) continue;
      if (focus && focus.tagName === 'INPUT' && r.el.contains(focus)) continue;   // typing: redraw after the change
      if (pressed && r.el.contains(pressed)) { drawAfterPress = true; continue; }  // clicking: redraw after the click
      r.el.innerHTML = html;
      r.html = html;
    }
    for (const el of app.querySelectorAll('[data-scrollkey]')) {
      if (scrolled.has(el.dataset.scrollkey)) el.scrollTop = scrolled.get(el.dataset.scrollkey);
    }
  }

  const spark = (d, w = 80, h = 14) => (d ? `<svg class="spark" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><path d="${esc(d)}"/></svg>` : '');
  const bar = pct => `<span class="bar"><svg viewBox="0 0 100 4" preserveAspectRatio="none"><rect class="track" width="100" height="4"/>` +
    `<rect class="fill" width="${Math.max(0, Math.min(100, Number(pct) || 0))}" height="4"/></svg></span><span class="pct">${num(pct)}%</span>`;
  const runBtn = (action, project, label, title, extra = '') =>
    `<button data-run="${action}" data-project="${esc(project)}"${extra}${title ? ` title="${esc(title)}"` : ''}>${label}</button>`;
  const btn = (attrs, label, title) => `<button ${attrs}${title ? ` title="${esc(title)}"` : ''}>${label}</button>`;
  // green: the run beats (its own heartbeat, under 60 s old); red: terminated or dead
  const aliveDot = a => (a ? `<span class="dot ${a.alive ? 'alive' : 'dead'}" role="img" aria-label="${esc(a.label)}" title="${esc(a.label)}"></span>` : '');
  const details = (key, dflt, summary, body) =>
    `<details data-key="${esc(key)}"${isOpen(key, dflt) ? ' open' : ''}><summary>${summary}</summary>${body}</details>`;

  /** Under the GPU rows: [Docker tear down], beside it the image build %, beneath that the containers' health. */
  function dockerHtml(docker) {
    if (!docker) return '';
    const b = docker.build;
    const steps = b && b.total ? ` (${num(b.done)}/${num(b.total)})` : '';
    // the bar stays after the build: full and named when built, where it stopped otherwise
    const build = !b ? 'no image build yet'
      : b.state === 'built' ? `image built · ${esc(b.image)}`
      : b.state === 'building' ? `building ${num(b.percent)}%${steps}${b.phase ? ` · <small>${esc(b.phase)}</small>` : ''}`
      : `build ${esc(b.state)} at ${num(b.percent)}%${steps}`;
    // drawn as the run card's bars are (an SVG track and fill), not a <progress>: the webview drew that one
    // empty after a build that had succeeded. Built is a full bar whatever the steps counted.
    const filled = b && b.state === 'built' ? 100 : Math.max(0, Math.min(100, Number(b && b.percent) || 0));
    const bar = b ? `<span class="build-bar build-${esc(b.state)}"><svg viewBox="0 0 100 4" preserveAspectRatio="none">` +
      `<rect class="track" width="100" height="4"/><rect class="fill" width="${filled}" height="4"/></svg></span>` : '';
    const kill = `<button data-docker="kill-build" class="danger small"${docker.building ? '' : ' disabled'} title="${docker.building
      ? 'Stop the running image build for good (make kill-build): the nerdctl client, every step BuildKit still runs, the BuildKit daemon if a step survives; the make that started it fails. Asks first.'
      : 'No image build runs.'}">Kill build</button>`;
    const color = { running: 'green', restarting: 'amber', created: 'amber', failed: 'red' };
    const health = docker.error ? `<span class="note">${esc(docker.error)}</span>`
      : !(docker.containers || []).length ? '<span class="note">no sidechannel container</span>'
      : docker.containers.map(c => `<div title="${esc(c.image)}"><span class="light ${color[c.health] || ''}"></span>` +
          `${esc(c.name)}: ${esc(c.health)} <small>(${esc(c.status)})</small></div>`).join('');
    return '<div class="docker"><button data-docker="teardown" class="danger" title="make docker-teardown: compose down -v, ' +
      'remove the sidechannel containers and images, prune the build cache, dangling images and unused volumes, ' +
      'empty deployment/docker/tmp/. Asks first; refused while an experiment runs.">Docker tear down</button>' +
      `<div class="docker-info"><div class="build">${build} ${bar} ${kill}</div><div class="health">${health}</div></div></div>`;
  }

  function gpuHtml(gpu) {
    if (!gpu) return '';
    if (!gpu.gpus || !gpu.gpus.length) return `<p class="note">${esc(gpu.error || 'reading nvidia-smi…')}</p>${dockerHtml(gpu.docker)}`;
    return gpu.gpus.map(g =>
      `<div class="gpu-row" title="${esc(g.name)}"><span class="gpu-name">GPU${esc(g.index)}</span>` +
      `<span class="num">${num(g.util)}%</span>${spark(g.utilSpark)}` +
      `<span class="num">${num(g.memUsed)}/${num(g.memTotal)} MiB</span>${spark(g.memSpark)}` +
      `<span class="num">${num(g.temp)}°C</span></div>`).join('') + dockerHtml(gpu.docker);
  }

  function summaryHtml(p) {
    const pct = p.run && p.run.percent != null ? ` · ${num(p.run.percent)}%` : '';
    return `<span class="light ${esc(p.light.color)}" title="${esc(p.light.label)}"></span><b>${esc(p.name)}</b>` +
      ` <small${p.env ? ` title="ENV followed from ${esc(p.envSource || '')}"` : ''}>${p.env ? `${esc(p.env)} · ` : ''}${esc(p.light.label)}${pct}</small>`;
  }

  // ETA in master.yaml's timezone (the zone of the registry's and the logs' timestamps), named by its city.
  const left = s => (s < 3600 ? `${Math.max(1, Math.round(s / 60))} min` : `${(s / 3600).toFixed(1)} h`);
  function endAt(ms, tz) {
    const d = new Date(ms);
    const time = d.toLocaleTimeString('en-GB', { timeZone: tz, hour: '2-digit', minute: '2-digit', hour12: false });
    const day = o => o.toLocaleDateString('en-CA', { timeZone: tz });
    const date = day(d) === day(new Date()) ? '' : `${d.toLocaleDateString('en-GB', { timeZone: tz, weekday: 'short', day: 'numeric', month: 'short' })} `;
    return `${date}${time} ${tz.split('/').pop().replace(/_/g, ' ')}`;
  }

  /**
   * Under the running run's progress: its latest metrics (core/monitor.snapshot), one line per metric with
   * its value per process (train | validation, "(CV)" in a fold | test when attacked), all from the fold or
   * refit being trained, under a head naming that source and the epoch. A dot shows how a value moved since
   * the epoch before, green when that is better and red when it is worse (utils/monitor_metric.metric_direction);
   * a fit time keeps the grey dot (it only grows: never better or worse). Only the metrics_to_log columns are lines (owner 2026-10-27).
   */
  function liveHtml(live) {
    const lines = live && Array.isArray(live.metrics) ? live.metrics.filter(m => m.values && m.values.length) : [];
    if (!lines.length) return '';
    const fmt = v => (Number.isInteger(v) || Math.abs(v) >= 100 ? String(Math.round(v * 100) / 100) : Number(v).toPrecision(4));
    // one dot per value: green better, red worse, grey unchanged or the first epoch
    const state = v => (v.improved === true ? 'better' : v.improved === false ? 'worse' : 'same');
    // a value as shown: its number, a live test value's note beside it ("not reached yet" alone)
    const shown = v => (v.value == null ? esc(v.note || '') : `${esc(fmt(v.value))}${v.note ? ` <small>(${esc(v.note)})</small>` : ''}`);
    const slot = v => {
      if (!v) return '<td></td>';
      const why = v.live ? 'the test so far: the attacks done until now'
        : v.previous == null ? 'first epoch'
        : `was ${fmt(v.previous)} the epoch before; ${v.direction === 'max' ? 'higher' : 'lower'} is better`;
      return `<td${v.live ? ' class="live-test"' : ''} title="${esc(why)}">` +
        `<span class="num">${shown(v)}<span class="live-dot ${state(v)}"></span></span></td>`;
    };
    // a column per process, in one order on every line: a process's values stand under each other, a blank cell
    // where a metric has none (a rank has no train value)
    const columns = ['train', 'validation', 'test'].filter(p => lines.some(m => m.values.some(v => v.process === p)));
    const a = live.attack;
    const head = [live.source, !a && live.epoch != null ? `epoch ${live.epoch}` : ''].filter(Boolean).join(' · ');
    // the head row: the source and epoch over the metric names, each process's name over its values
    // ("validation (CV)" inside a fold, as its values are labelled)
    const title = p => (lines.flatMap(m => m.values).find(v => v.process === p) || {}).label || p;
    const headRow = `<tr class="live-cols"><th>${esc(head)}</th>` +
      columns.map(p => `<th class="live-col${p === 'test' ? ' live-test' : ''}">${esc(title(p))}</th>`).join('') + '</tr>';
    const kr = live.key_recovery;
    const krLine = !kr ? '' : `<div class="live-key">key recovery · ${esc(kr.source)}: ${[kr.recovered ? 'recovered' : 'not recovered',
      kr.true_key_rank != null ? `true key rank ${kr.true_key_rank}` : '', kr.top_guess ? `top guess ${kr.top_guess}` : '',
      typeof kr.ge === 'number' ? `GE ${fmt(kr.ge)}` : '', typeof kr.sr1 === 'number' ? `SR@1 ${kr.sr1.toFixed(2)}` : '']
      .filter(Boolean).map(esc).join(' · ')}</div>`;
    const legend = [['better', 'better than the epoch before'], ['worse', 'worse than the epoch before'],
      ['same', 'unchanged, or the first epoch']]
      .map(([s, t]) => `<span><span class="live-dot ${s}"></span>${t}</span>`).join('');
    return `<div class="live"><table class="live-rows">${headRow}` +
      lines.map(m => `<tr><th>${esc(m.name)}</th>${columns.map(p => slot(m.values.find(v => v.process === p))).join('')}</tr>`).join('') +
      `</table>${krLine}<div class="live-legend">${legend}</div></div>`;
  }

  function runHtml(p) {
    // one make orchestrate: RUN SINGLE EXPERIMENT (single_runs.yaml), beside the queues
    const tools = p.env ? `<div class="note">ENV ${esc(p.env)}, followed from ${esc(p.envSource || '')}</div>` : '';
    return tools;                                   // the run's card is listed under [Running] (runCardHtml)
  }

  /** | ETA | run: <end> (in <left>) | batch: <end> (in <left>) |, in master.yaml's timezone. */
  function etaTable(p) {
    const e = p.eta;
    if (!e) return '';
    const tz = (last && last.timezone) || 'UTC';
    const cell = (end, s) => (end ? `${esc(endAt(end, tz))} <small>(in ${left(s)})</small>` : '<small>estimating…</small>');
    return `<table class="eta"${e.upperBound ? ' title="at the latest: early stopping may end runs sooner"' : ''}><tr>` +
      `<th>ETA${e.upperBound ? ' ≤' : ''}</th><td>run: ${cell(e.runEndMs, e.runLeftS)}</td>` +
      `<td>batch${e.runs ? ` <small title="the runs its queue plans (TOTAL RUNS QUEUED) not finished yet, this one included">(${esc(e.runs.left)} of ${esc(e.runs.total)} runs left)</small>` : ''}: ` +
      `${cell(e.batchEndMs, e.batchLeftS)}</td></tr></table>` +
      (e.note ? `<small class="note etanote">${esc(e.note)}</small>` : '');
  }

  /**
   * The running run's card, its item in the [Running] list: name, buttons, phase (the orange attack bar while
   * its test attacks), the train bar, ETA, queue, session, and its live metrics. It leaves with the run, when the
   * run moves to [Done].
   */
  function runCardHtml(p, line) {
    const r = p.run;
    const queues = (r.batches || []).map(k => {
      const q = p.queues.find(x => x.key === k);
      if (!q) return esc(k);
      const at = r.queue === k && r.trial ? ` · trial ${r.trial}/${r.trials || '?'}` : '';
      return `${esc(k)}: ${q.done} of ${q.planned || q.runs} done${at}`;
    }).join(' · ');
    const etaLine = etaTable(p);                    // one row under the progress bar
    const held = p.runSession;
    // Attach: the tmux session running it (else the container's live log); Kill session asks first.
    // Detach while a terminal is attached to that run's session, Attach when it is detached.
    const tmuxBtn = held && p.runAttached
      ? runBtn('detach-run', p.name, 'Detach', `tmux detach-client -s ${held}: the run keeps going`)
      : runBtn('attach-run', p.name, 'Attach', held ? `tmux attach -t ${held}` : 'no tmux session holds it');
    const logBtn = runBtn('container-log', p.name, 'Container log',
      'the training container\'s live output: sudo nerdctl logs -f --tail 100 sidechannel');
    const acts = `<span class="run-acts">${line ? rowActs(p, 'running', line) : ''}${tmuxBtn}${logBtn}` +
      `${runBtn('kill-run', p.name, 'Kill session', held ? `kill tmux session ${held} (asks first): the container may keep training` : 'no tmux session holds it')}</span>`;
    return `<div class="run"><div class="run-head">${aliveDot(r.alive)}${btn(`class="link" data-detail="1" data-project="${esc(p.name)}" data-hash="${esc(r.hash)}"`, esc(r.name))}` +
      ` <small>${esc([r.model, r.dataset].filter(Boolean).join(' · '))}</small>${acts}</div>` +
      `${phaseHtml(r)}${r.percent != null ? `<div class="progress"><span class="bar-label">train</span>${bar(r.percent)}</div>` : ''}${etaLine}` +
      `${queues ? `<div class="note">queue ${queues}</div>` : ''}` +
      `<div class="note">${held ? `tmux session: ${esc(held)}` : 'no tmux session holds this run (its log stream ended); Container log shows its live output'}</div>` +
      `${liveHtml(r.live)}</div>`;
  }

  /** The run's phase line; while its test attacks, the attack bar in the epoch's place: orange, glowing while it runs,
   *  labelled test as the main bar under it is labelled train, the two bars starting in one column. */
  function phaseHtml(r) {
    const a = r.attack;
    if (!a) return `<div class="phase">${esc(r.phase)}</div>`;
    const pct = a.attacks ? Math.max(0, Math.min(100, 100 * a.attack / a.attacks)) : 0;
    return `<div class="phase attack-phase" title="the test's attacks (perform_attack.test.attack_times)"><span class="bar-label">test</span>` +
      `<span class="live-attack-bar${a.running ? ' running' : ''}"><svg viewBox="0 0 100 4" preserveAspectRatio="none">` +
      `<rect class="track" width="100" height="4"/><rect class="fill" width="${pct}" height="4"/></svg></span>` +
      `<span class="live-attack-count">attack ${esc(a.attack)} of ${esc(a.attacks)}</span></div>`;
  }

  // ── the Budget box (owner 2026-10-28): in the place and size of the old [Retest Budget] / [Test Budget] button, a
  // [Test] and an entry of params.yaml data[i].attack_trace_budget, i the run's data entry; the entry is a form field
  // (staged, then OK / Cancel), "Budget" its placeholder when empty, emptied to clear it
  const paramsFileOf = p => ((p.configs || []).map(c => c.path).find(f => /\/params\.ya?ml$/.test(String(f)))) || '';
  /** params.yaml's data entry named `dataset`: {file, t, i, node} (node: its attack_trace_budget, null when absent). */
  function budgetOf(p, dataset) {
    const file = paramsFileOf(p);
    const t = file ? trees[file] : null;
    const data = t && t.tree ? entryOf(t.tree, 'data') : null;
    const i = data && data.k === 'list' && dataset
      ? data.items.findIndex(it => it.k === 'map' && unquote((entryOf(it, 'name') || {}).text || '') === dataset) : -1;
    return { file, t, i, node: i >= 0 ? entryOf(data.items[i], 'attack_trace_budget') : null };
  }
  /** [Test | Budget]: Test (action, BUDGET=1) attacks with the entry's traces per attack; size: the old button's label. */
  function budgetHtml(p, dataset, action, line, keyAttr, size) {
    const { file, t, i, node } = budgetOf(p, dataset);
    const where = `params.yaml data[${dataset || '?'}].attack_trace_budget`;
    const wrap = (inner, form) => `<span class="budget-form${form ? ' form' : ''}" data-size="${esc(size)}" data-file="${esc(file)}"` +
      `${form ? ` data-project="${esc(p.name)}" data-sha="${esc(t.sha256)}"` : ''}>${inner}</span>`;
    const test = (title, off) => runBtn(action, p.name, 'Test', title, `${keyAttr} data-kind="budget"${off ? ' disabled' : ''}`);
    const why = !file ? 'no params.yaml in this project' : !t ? 'reading params.yaml…' : t.error ? t.error
      : i < 0 ? `no params.yaml data entry named ${dataset || '(unknown)'}` : '';
    if (why) return wrap(`${test(`${line} BUDGET=1: ${why}`, true)}<input class="val budget" placeholder="Budget" size="1" disabled title="${esc(why)}">`, false);
    const path = node ? ['data', i, 'attack_trace_budget'] : ['data', i];
    const staged = stagedOf(file)[node ? editId('set', path) : editId('put', path, 'attack_trace_budget')];
    const value = staged ? staged.value : node ? unquote(node.text || '') : '';
    const op = node ? `data-op="set" ${pathAttr(path)}` : `data-op="put" ${pathAttr(path)} data-key="attack_trace_budget"`;
    const title = staged ? `${where}: unsaved; OK it first (the test reads the file)`
      : `${line} BUDGET=1: its attack, each attack ${value ? `${value} traces` : 'its data entry\'s attack_trace_budget traces (none set)'}` +
        ` (${where}), appended (asks first)`;
    return wrap(`${test(title, !!staged)}<input class="val budget${staged ? ' dirty' : ''}" data-budget="1" ${op} value="${esc(value)}" placeholder="Budget" size="1"` +
      ` title="${esc(`${where}: the traces of one test attack; empty: none (cleared)`)}">`, true);
  }
  /** A Budget box's staged edit: data[i].attack_trace_budget set, or the key put into data[i]. */
  const isBudgetEdit = e => (e.op === 'set' && e.path.length === 3 && e.path[0] === 'data' && e.path[2] === 'attack_trace_budget') ||
    (e.op === 'put' && e.path.length === 2 && e.path[0] === 'data' && e.key === 'attack_trace_budget');
  /** The unsaved changes of params.yaml: OK writes them all, Cancel undoes them all, Reset undoes the Budget boxes'
   *  (another change of params.yaml, from its form, stays). Shown as soon as a box changes (owner 2026-10-28). */
  function budgetBarInner(file, project) {
    const t = file ? trees[file] : null;
    const n = file ? Object.keys(stagedOf(file)).length : 0;
    if (!n || !t || !t.tree) return '';
    return `<div class="form" data-file="${esc(file)}" data-project="${esc(project)}" data-sha="${esc(t.sha256)}"><div class="pending">` +
      `<span>${n} unsaved change${n === 1 ? '' : 's'} in params.yaml</span>${btn('data-apply="1"', 'OK', 'write every change of params.yaml to the file')}` +
      `${btn('data-reset="1"', 'Cancel', 'undo every unsaved change of params.yaml')}` +
      `${btn('data-budgetreset="1"', 'Reset', 'undo the Budget boxes\' changes; another unsaved change of params.yaml stays')}</div></div>`;
  }
  const budgetBar = p => `<div class="budget-bar" data-file="${esc(paramsFileOf(p))}" data-project="${esc(p.name)}">` +
    `${budgetBarInner(paramsFileOf(p), p.name)}</div>`;

  /**
   * A listed run's buttons, by tile: done → Retest All, [Test | Budget] (its own snapshot's attack, every test trace or
   * the Budget box's traces per attack; each appended); running → Terminate Run, Log; failed → Delete.
   */
  function rowActs(p, group, r) {
    const hash = ` data-key="${esc(r.hash)}"`;
    if (group === 'done') {
      const line = `make test ENV=${r.env || p.env || ''} MODEL=${r.name}`;
      return runBtn('test-run', p.name, 'Retest All', `${line}: its own snapshot's attack on every test trace, appended (asks first)`, `${hash} data-kind="all"`) +
        budgetHtml(p, r.dataset, 'test-run', line, hash, 'Retest Budget');
    }
    if (group === 'running') {
      const key = r.queue || (r.batches || [])[0] || '';
      return runBtn('terminate-run', p.name, 'Terminate Run', `stop the run itself: kill its tmux session, then make stack-down ENV=${p.env} (asks first)`, ' class="danger"') +
        (key ? runBtn('log', p.name, 'Log', `tail -f projects/<P>/artifacts/${p.env}/tmux_logs/${key}.txt`, ` data-key="${esc(key)}"`)
          : runBtn('container-log', p.name, 'Log', 'the training container\'s live output: no batch log names this run'));
    }
    if (group === 'failed') {
      return runBtn('delete-run', p.name, 'Delete', `make delete_database MODEL=${r.name}: a dry run first, then asks once more`, `${hash} class="danger"`);
    }
    return '';
  }

  // [Done] and [Failed]: a filter (one queue or one model) and a sort, per list and project (webview state)
  const listKey = (group, name) => `${group}:${name}`;
  const listPick = (group, name) => ((ui.listPick || {})[listKey(group, name)] || '');
  const listSort = (group, name) => ({ key: 'created', dir: 'asc', ...((ui.listSort || {})[listKey(group, name)] || {}) });
  const SORTS = {
    done: [['val', 'validation, all N_val traces'], ['same', 'test, first N_val traces'], ['budget', 'test, budget N_test traces'],
           ['test', 'test, all N_test traces'], ['name', 'alphabet'], ['created', 'created']],
    failed: [['name', 'alphabet'], ['created', 'created']],          // a failed run has no test ranks
  };
  const queuesOf = r => [...new Set([r.queue, ...(r.batches || [])].filter(Boolean))];
  const picked = (r, pick) => {
    const [kind, value] = [pick.slice(0, pick.indexOf(':')), pick.slice(pick.indexOf(':') + 1)];
    return kind === 'queue' ? queuesOf(r).includes(value) : kind === 'model' ? r.model === value : true;
  };
  const sortValue = (r, key) => (key === 'name' ? String(r.name || '') : key === 'created' ? (Date.parse(r.started) || null)
    : r.ranks && r.ranks[key] && Number.isFinite(Number(r.ranks[key].rank)) ? Number(r.ranks[key].rank) : null);
  /** The rows in the sort's order; a row without the value (no test rank, no start time) last; ties keep their place. */
  function sortedRuns(rows, s) {
    const dir = s.dir === 'desc' ? -1 : 1;
    return rows.map((r, i) => [r, i]).sort(([a, i], [b, j]) => {
      const x = sortValue(a, s.key);
      const y = sortValue(b, s.key);
      if (x == null || y == null) return (x == null) - (y == null) || i - j;
      const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
      return c ? dir * c : i - j;
    }).map(([r]) => r);
  }
  const FILTER_ICON = '<svg class="filter-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M1.5 2.5h13l-5 6v5l-3 1.5v-6.5z"/></svg>';
  const SORT_ICON = '<svg class="filter-icon" viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11M1.5 11l2.5 2.5L6.5 11M11 13.5v-11M8.5 5L11 2.5 13.5 5"/></svg>';
  /** [Done] or [Failed]: the funnel and its menu (every queue and model of the list, with its count), then the sort
   *  and its direction. */
  function listToolsHtml(p, group, all, shown) {
    const name = esc(p.name);
    const pick = listPick(group, p.name);
    const s = listSort(group, p.name);
    const count = (key, rows) => [...rows.reduce((m, r) => { for (const k of key(r)) m.set(k, (m.get(k) || 0) + 1); return m; }, new Map())]
      .sort((a, b) => a[0].localeCompare(b[0]));
    const opts = (kind, pairs) => pairs.map(([v, n]) => `<option value="${esc(`${kind}:${v}`)}"${pick === `${kind}:${v}` ? ' selected' : ''}>${esc(v)} (${num(n)})</option>`).join('');
    const real = all.filter(r => !r.pending);
    const what = group === 'done' ? 'Done' : 'Failed';
    return `<label class="done-find" title="show the ${what} runs of one queue or one model">${FILTER_ICON}` +
      `<select data-listpick="${group}" data-project="${name}"><option value=""${pick ? '' : ' selected'}>all runs (${num(real.length)})</option>` +
      `<optgroup label="queue">${opts('queue', count(queuesOf, real))}</optgroup>` +
      `<optgroup label="model">${opts('model', count(r => (r.model ? [r.model] : []), real))}</optgroup></select>` +
      (pick ? `<small class="done-count">${num(shown)} of ${num(real.length)}</small>` : '') + '</label>' +
      `<label class="done-find" title="sort the ${what} runs; a run without the value (no test rank) comes last">${SORT_ICON}` +
      `<select data-listsort="${group}" data-project="${name}">${SORTS[group].map(([k, t]) => `<option value="${k}"${s.key === k ? ' selected' : ''}>${esc(t)}</option>`).join('')}</select>` +
      `<select data-listdir="${group}" data-project="${name}"><option value="asc"${s.dir === 'asc' ? ' selected' : ''}>ascending</option>` +
      `<option value="desc"${s.dir === 'desc' ? ' selected' : ''}>descending</option></select></label>`;
  }

  function tilesHtml(p) {
    const sel = ui.group[p.name] || '';
    const tiles = GROUPS.map(g => {
      const n = p.tiles[g].length;
      return btn(`class="tile ${g}${n ? ' some' : ' none'}${sel === g ? ' sel' : ''}" data-group="${g}" data-project="${esc(p.name)}"` +
        ` aria-pressed="${sel === g}"`,
        `<span class="count"><span class="glyph" aria-hidden="true">${GLYPH[g]}</span>${n}</span><span class="word">${g}</span>`,
        `${n} ${g} run${n === 1 ? '' : 's'}: click to list them`);
    }).join('');
    const all = sel ? p.tiles[sel] : [];
    const tools = sel === 'done' || sel === 'failed';
    const pick = tools ? listPick(sel, p.name) : '';
    const filtered = pick ? all.filter(r => !r.pending && picked(r, pick)) : all;
    const runs = tools ? sortedRuns(filtered, listSort(sel, p.name)) : filtered;
    // [Running]: the running run's whole card in its place (its line otherwise); a card no listed run carries
    // (a beating run the registry does not show yet) leads the list
    const card = r => (sel === 'running' && p.run && r.hash === p.run.hash ? `<li class="run-card">${runCardHtml(p, r)}</li>` : null);
    const lead = sel === 'running' && p.run && !runs.some(r => r.hash === p.run.hash) ? `<li class="run-card">${runCardHtml(p, null)}</li>` : '';
    const list = lead + runs.map(r => card(r) || (r.pending
      ? `<li class="pending"><span class="glyph" aria-hidden="true">○</span>${esc(r.name)} ` +
        '<small>not hashed yet: the planner makes its hash and name when it starts</small></li>'
      : sel === 'done' ? `<li class="done-run">${doneRunHtml(p, r)}</li>`
        : `<li>${runLine(p, sel, r)}</li>`)).join('');
    // [Done] and [Failed]: under the tiles, the filter and the sort at the left (under [Queued]); [Done]'s key of
    // its four ranks at the right
    const legend = tools ? `<div class="done-bar"><span class="list-tools">${listToolsHtml(p, sel, all, runs.length)}</span>` +
      (sel === 'done' ? `<div class="rank-legend">${RANK_KEYS.map(([k, t]) => `<span><span class="rk-key ${k}"></span>${t}</span>`).join('')}</div>` : '') + '</div>' : '';
    const none = pick && all.length ? `no ${sel === 'done' ? 'Done' : 'Failed'} run of ${esc(pick.replace(':', ' '))}` : `no ${sel} runs`;
    // [Queued], [Done], [Failed] with runs: a list box of limited height that scrolls; the tiles and the bar above stay out
    const box = list && ['queued', 'done', 'failed'].includes(sel) ? ` box" data-scrollkey="${esc(`${p.name}:${sel}`)}` : '';
    return `<div class="tiles">${tiles}</div>${legend}${sel === 'done' ? budgetBar(p) : ''}` +
      `${sel ? `<ul class="runs${box}">${list || `<li class="note">${none}</li>`}</ul>` : ''}`;
  }

  // The four ranks of a Done run, in this order and in increasing shades of fire (panel.css .rk): the validation's
  // (its training, once), then the latest test's: at N_val traces, at its data entry's budget, on every test trace
  const RANK_KEYS = [['val', 'validation, all N_val traces'], ['same', 'test, first N_val traces'],
                     ['budget', 'test, budget N_test traces'], ['test', 'test, all N_test traces']];

  /** A listed run's line: liveness, its name (the detail view), model and trial, spark, last rank, its buttons. */
  function runLine(p, sel, r) {
    return `${aliveDot(r.alive)}` +
      `${btn(`class="link" data-detail="1" data-project="${esc(p.name)}" data-hash="${esc(r.hash)}"`, esc(r.name), r.reason || r.status)}` +
      ` <small>${esc(r.model)}${r.trial ? ` · ${r.queue ? `${esc(r.queue)} ` : ''}trial ${r.trial}/${r.trials || '?'}` : ''}</small>` +
      `<span class="end">${spark(r.spark)}${rankHtml(sel, r)}${rowActs(p, sel, r)}</span>`;
  }

  /** The run's rank. A Done run: the mean rank of the true key over the attacks (GE) after the traces given, each
   *  with that count, coloured as the key under the tiles says (– where it has none): its validation's after all
   *  N_val traces (its training's, fixed), and of its latest tests (extension runRanks: never an earlier one) at the
   *  first N_val test traces, at its data entry's budget (the latest Budget test) and on every test trace (the latest
   *  All test). Only the first two compare; a rank falls as traces are added. Any other run: its validation rank. */
  function rankHtml(sel, r) {
    if (sel !== 'done') return `<small class="num">${r.last == null ? '' : `rank ${num(r.last)}`}</small>`;
    const k = n => (n >= 1000 ? `${Math.round(n / 100) / 10}k` : String(Math.round(n)));
    const part = (cls, v) => `<span class="rk ${cls}">${v ? `${num(v.rank)} <small>@${k(v.traces)}</small>` : '–'}</span>`;
    const x = r.ranks || {};
    return `<small class="num ranks" title="final true-key rank (the mean over the attacks) after the traces given: ` +
      `${RANK_KEYS.map(([, t]) => t).join(' | ')}; the latest test of each kind, never an earlier one">` +
      `${RANK_KEYS.map(([k]) => part(k, x[k])).join(' | ')}</small>`;
  }

  /**
   * A Done run as a drop-down: its line is the summary; opened, its metric export line and its Graph
   * sub-panel, which holds the graph_config.yaml jobs that list the run (models, or job 0's
   * top_models) as the GRAPH_CONFIG.YAML form draws them, so they are edited here directly.
   */
  function doneRunHtml(p, r) {
    const key = `dr:${p.name}:${r.hash}`;
    const open = isOpen(key, false);
    // opened before its menu is read (or after a failed read): the extension reads this run at once
    const want = open && (r.choices == null || r.choices.error) ? ` data-want="${esc(p.name)}" data-name="${esc(r.name)}"` : '';
    return `<details class="done-run" data-key="${esc(key)}"${want}${open ? ' open' : ''}><summary>${runLine(p, 'done', r)}</summary>` +
      (open ? metricRow(p, r) + runGraphsHtml(p, r) : '') + '</details>';
  }

  const graphFile = p => ((p.configs || []).find(c => c.file === 'graph_config.yaml') || {}).path || '';

  /** The jobs of a graph_config.yaml tree that list `name` (models, or top_models: job 0), with their index. */
  function ownedJobs(tree, name) {
    const jobs = entryOf(tree, 'jobs');
    if (!jobs || jobs.k !== 'list') return [];
    return jobs.items.map((item, i) => ({ item, i })).filter(({ item }) =>
      item.k === 'map' && ['models', 'top_models'].some(k => texts(entryOf(item, k)).map(unquote).includes(name)));
  }

  /** Under a Done run: its graph jobs (their ids in the title), each the GRAPH_CONFIG.YAML job form. */
  function runGraphsHtml(p, r) {
    const file = graphFile(p);
    const key = `rg:${p.name}:${r.hash}`;
    const open = isOpen(key, true);
    const t = file ? trees[file] : null;
    const owned = t && t.tree ? ownedJobs(t.tree, r.name) : null;
    const ids = owned ? owned.map(({ item }) => (entryOf(item, 'id') || {}).text).filter(v => v != null)
      : (p.graphJobs || []).filter(j => (j.models || []).includes(r.name)).map(j => j.id);
    const head = `Graph <small>${ids.length ? `id ${ids.map(esc).join(', ')}` : 'no graph job lists this run'}</small>`;
    let body;
    if (!file) body = '<p class="note">no graph_config.yaml in this project</p>';
    else if (!t) body = '<p class="note">reading graph_config.yaml…</p>';
    else if (t.error) body = `<p class="err">${esc(t.error)}</p>`;
    else if (!owned.length) body = '<p class="note">Export to Graph Config above puts this run into a graph job; it then shows here.</p>';
    else {
      curFile = file;
      curProject = p.name;
      const n = Object.keys(stagedOf(file)).length;
      const bar = n ? `<div class="pending"><span>${n} unsaved change${n === 1 ? '' : 's'} in graph_config.yaml</span>` +
        `${btn('data-apply="1"', 'OK', 'write every change of graph_config.yaml to the file')}${btn('data-discard="1"', 'Cancel', 'forget the changes')}</div>` : '';
      const note = t.note ? `<p class="${/refused|Error|changed on disk/.test(t.note) ? 'err' : 'note'}">${esc(t.note)}</p>` : '';
      body = `${note}<div class="form" data-file="${esc(file)}" data-project="${esc(p.name)}" data-sha="${esc(t.sha256)}">${bar}` +
        `${owned.map(({ item, i }) => itemHtml(item, i, file)).join('')}</div>`;
    }
    return `<details class="rungraph" data-key="${esc(key)}" data-file="${esc(file)}"${open ? ' open' : ''}><summary>${head}</summary>` +
      `${open ? body : ''}</details>`;
  }

  /** The graph jobs drawing this metric (source|process|metric) that list the run. */
  function jobsWith(p, name, key) {
    if (!key) return [];
    const [source, processName, metric] = key.split('|');
    return (p.graphJobs || []).filter(j => j.id !== 0 && j.metric === metric && j.source === source &&
      (j.processes || []).length === 1 && j.processes[0] === processName && (j.models || []).includes(name));
  }

  /**
   * Under a Done run: every metric it recorded (graph_export.py choices) and, for the one picked,
   * Export to Graph Config, or Delete from Graph Config when a graph job already draws it for this run.
   */
  function metricRow(p, r) {
    if (r.choices == null) return '<div class="metric-row"><small class="note">reading its metrics…</small></div>';
    if (r.choices.error) {
      return `<div class="metric-row"><small class="err">its metrics could not be read: ${esc(r.choices.error)}; close and open the run to read them again</small></div>`;
    }
    if (!r.choices.length) return '<div class="metric-row"><small class="note">no metrics recorded</small></div>';
    const picked = ((ui.gmetric || {})[p.name] || {})[r.name] || '';
    const groups = {};
    for (const c of r.choices) (groups[`${c.source} · ${c.process}`] = groups[`${c.source} · ${c.process}`] || []).push(c);
    const opts = Object.entries(groups).map(([g, cs]) => `<optgroup label="${esc(g)}">${cs.map(c => {
      const key = `${c.source}|${c.process}|${c.metric}`;
      return `<option value="${esc(key)}"${picked === key ? ' selected' : ''}>${esc(c.metric)}${c.kind === 'curve' ? ' (curve)' : ''}</option>`;
    }).join('')}</optgroup>`).join('');
    const listed = jobsWith(p, r.name, picked);
    const act = !picked ? btn('disabled', 'Export to Graph Config', 'pick a metric first')
      : listed.length
        ? btn(`class="danger" data-gremove="${esc(p.name)}" data-name="${esc(r.name)}" data-mkey="${esc(picked)}"`, 'Delete from Graph Config',
          `remove ${r.name} from graph job ${listed.map(j => j.id).join(', ')} (asks first)`)
        : btn(`data-gexport="${esc(p.name)}" data-name="${esc(r.name)}" data-mkey="${esc(picked)}"`, 'Export to Graph Config',
          'into the graph job of its batch, dataset and this metric (made when there is none): models, legend.labels, a header comment');
    return `<div class="metric-row"><select class="gexp" data-gmetric="${esc(p.name)}" data-name="${esc(r.name)}" title="every metric this run recorded">` +
      `<option value="">metric…</option>${opts}</select>${act}` +
      `${listed.length ? ` <small class="note">in graph job ${listed.map(j => esc(j.id)).join(', ')}</small>` : ''}</div>`;
  }

  /**
   * A queue row's action, by its state: running → Terminate Run; failed (a run of its latest batch
   * failed or was terminated) → Retry Start, the same make experiment, which resumes: DONE runs are
   * skipped, the failed and interrupted ones continue; done → Rerun (the same line); any other → Start.
   * Log follows in every row.
   */
  function queueAction(p, q, key) {
    if (!q.commands && q.state !== 'running') {            // no make line yet: nothing to start
      return '<button disabled title="no make line in this queue yet: Set Queue, choose the line\'s values, OK">Start</button>';
    }
    if (q.state === 'running') {
      return runBtn('terminate-run', p.name, 'Terminate Run',
        `stop ${q.key}'s run: kill its tmux session, then make stack-down ENV=${p.env} (asks first)`, ' class="danger"');
    }
    if (q.state === 'failed') {
      return runBtn('queue', p.name, 'Retry Start',
        `make experiment QUEUE=${q.key}: resumes, DONE runs skipped, failed and interrupted ones continued`, key);
    }
    if (q.state === 'done') {
      return runBtn('queue', p.name, 'Rerun',
        `make experiment QUEUE=${q.key}: every run of it is DONE, so each is skipped (FORCE=1 retrains)`, key);
    }
    return runBtn('queue', p.name, 'Start', `make experiment QUEUE=${q.key}`, key);
  }

  /**
   * A queue's study (graph_export.py study_name): the key before its first "_" (fig14_cnnbest, fig14_vgg16,
   * fig14_mlpbest are study fig14), the key itself when a queue is named by that stem (fig10_optimizers
   * beside fig10). Overlay Graph is offered only when every queue of the study is DONE.
   */
  function studyOf(p, k) {
    const stem = String(k).split('_')[0];
    return stem !== String(k) && p.queues.some(q => q.key === stem) ? String(k) : stem;
  }

  function studyBtn(p, q, key) {
    const name = studyOf(p, q.key);
    const group = p.queues.filter(o => studyOf(p, o.key) === name);
    if (!group.every(o => o.state === 'done')) return '';
    return runBtn('study', p.name, 'Overlay Graph', `graph jobs of study ${name} (${group.map(o => o.key).join(', ')}): ` +
      'you tick the axes drawn as lines and the metrics; every value of the other axes is a figure', key);
  }

  /** The graph jobs drawing a run of the queue's latest batch (job 0, the judge's, aside). */
  function queueJobs(p, q) {
    const refs = new Set(q.refs || []);
    return (p.graphJobs || []).filter(j => j.id !== 0 && [...(j.runs || []), ...(j.models || [])].some(r => refs.has(String(r))))
      .map(j => j.id);
  }

  // QUEUES and RUN SINGLE EXPERIMENT: a reminder on their button line, faded
  const RUN_HINT = '<small class="run-hint">Configure parameters correctly before starting the Run</small>';

  function queuesHtml(p) {
    const done = p.queues.filter(q => q.state === 'done').length;
    const key = q => ` data-key="${esc(q.key)}"`;
    const ids = q => queueJobs(p, q);
    const rows = p.queues.map(q =>
      `<tr><td>${esc(q.key)}</td><td><span class="state ${esc(q.state.replace(' ', '-'))}">${esc(q.state)}</span></td>` +
      `<td class="num" title="done of the trials planned (sweep size x commands)">${q.done}/${q.planned || q.runs}</td>` +
      `<td class="num" title="lines in experiment_queue.yaml">${q.commands} cmd</td>` +
      `<td class="gids" title="graph jobs drawing its runs (graph_config.yaml)">${ids(q).length ? `graph ${ids(q).map(i => `#${esc(i)}`).join(', ')}` : '<small>no graph</small>'}</td>` +
      `<td class="acts">${studyBtn(p, q, key(q))}${queueAction(p, q, key(q))}` +
      `${runBtn('log', p.name, 'Log', `tail -f projects/<P>/artifacts/${p.env}/tmux_logs/${q.key}.txt`, key(q))}</td></tr>`).join('');
    // Set Params / Set Queue: that file's form under Experiment Settings, opened
    const cfg = re => (p.configs || []).find(c => re.test(c.file));
    const setBtn = (c, label, what) => (c ? btn(`data-setconfig="${esc(c.path)}" data-project="${esc(p.name)}"`, label,
      `open Experiment Settings → ${c.file} (${what})`) : '');
    return details(`q:${p.name}`, false, `Queues <small>${done}/${p.queues.length} done</small>`,
      `<div class="buttons">${setBtn(cfg(/^params\.ya?ml$/), 'Set Params', 'the settings the queues sweep')}` +
      `${setBtn(cfg(/^experiment_queue\.ya?ml$/), 'Set Queue', 'the queues and their make lines')}${RUN_HINT}</div>` +
      (rows ? `<table>${rows}</table>` : '<p class="note">no experiment_queue.yaml entries</p>'));
  }

  // ── SNAPSHOTS ──────────────────────────────────────────────────────────────────────────────────
  // One sub-panel per master.yaml project and one CUSTOM (a folder you choose). Loaded only by Scan, with
  // the filters filled above it: at least one, every one valid; the snapshots matching all of them are
  // listed (state.js snapFilter), else an error says why. Every snapshot: Open, Reproduce.
  const SNAP_FILTERS = [['BATCH', 'batch'], ['MODEL_ID', 'model id'], ['DATASET', 'dataset'], ['ENV', 'env'],
    ['LEAKAGE', 'HW / ID'], ['STATUS', 'done, not registered…'], ['NAME', 'name or hash']];
  const lower = v => String(v == null ? '' : v).toLowerCase();

  function snapPanelHtml(st, title, head) {
    const scope = st.scope;
    const f = ((ui.snapf || {})[scope]) || {};
    const filters = SNAP_FILTERS.map(([k, hint]) =>
      `<label class="snapf">${k}<input data-snapfilter="${esc(scope)}|${k}" value="${esc(f[k] || '')}" placeholder="${esc(hint)}"></label>`).join('');
    const rows = st.rows || [];
    const shown = rows;
    const sbtn = (action, label, title, extra = '') =>
      `<button data-snap="${action}" data-scope="${esc(scope)}"${extra}${title ? ` title="${esc(title)}"` : ''}>${label}</button>`;
    const anyFilter = Object.values(f).some(v => String(v || '').trim());
    const live = st.live;
    const liveLine = live ? `<p class="reproducing"><span class="dot ${live.alive && live.alive.alive ? 'alive' : 'dead'}" title="${esc(live.alive ? live.alive.label : '')}"></span>` +
      `in this folder: <b>${esc(live.name)}</b> <small>${esc(live.hash)}</small> · ENV ${esc(live.env)} · ${esc(live.phase)}` +
      `${live.percent != null ? ` · ${esc(live.percent)}%` : ''}</p>` : '';
    const rep = ((last && last.snapshots && last.snapshots.reproducing) || {})[scope];
    const repLine = rep ? `<p class="reproducing">Reproducing <b>${esc(rep.name)}</b> <small>${esc(rep.hash)}</small> · ENV ${esc(rep.env)} ` +
      `(from the snapshot) · from <b>${esc(rep.origin)}</b> into ${esc(rep.project)} <small>started ${esc(new Date(rep.at).toLocaleTimeString('en-GB'))}</small></p>` : '';
    const list = shown.map(r => {
      const status = r.status || 'not registered';
      const facts = [r.model_id, r.dataset, r.leakage_model, r.epochs != null ? `${r.epochs} ep` : '', r.batch_size != null ? `batch ${r.batch_size}` : '',
        (r.batches || []).join(', ')].filter(Boolean).map(esc).join(' · ');
      const repTitle = !r.verified ? `refused: ${r.problem}`
        : `make reproduce ENV=${r.env} PROJECT=${r.project || ''} MODEL=${r.model_name}${lower(status) === 'done' ? ' FORCE=1 (DONE: asks first)' : ''}`;
      return `<li class="snap${r.verified ? '' : ' refused'}"><div><b>${esc(r.model_name || r.folder)}</b> <small>${esc(r.model_hash || '')}</small>` +
        `<span class="snap-acts">${sbtn('open', 'Open', `open ${r.path}`, ` data-path="${esc(r.path)}"`)}` +
        `${sbtn('reproduce', 'Reproduce', repTitle, ` data-path="${esc(r.path)}"${r.verified ? '' : ' disabled'}`)}</span></div>` +
        `<div class="note">from <b>${esc(st.origin || r.origin || '')}</b> · ENV ${esc(r.env || '?')} (from the snapshot)` +
        `${r.project && scope === 'custom' ? ` · project ${esc(r.project)}` : ''} · ${esc(status)}${facts ? ` · ${facts}` : ''}</div>` +
        `${r.verified ? '' : `<div class="err">${esc(r.problem)}</div>`}</li>`;
    }).join('');
    const state = st.scanning ? '<p class="note">scanning…</p>'
      : st.error ? `<p class="err">${esc(st.error)}</p>`
      : !st.at ? '<p class="note">fill at least one filter, then Scan</p>'
      : !st.total ? '<p class="note">no experiment snapshots found</p>' : '';
    const count = st.at && !st.error ? `<small>${shown.length} of ${st.total || 0} scanned</small>` : '';
    return details(`snap:${scope}`, false, `${esc(title)} ${count}`,
      `${head}<div class="snapfilters">${filters}</div>` +
      `<div class="buttons">${sbtn('scan', st.scanning ? 'Scanning…' : 'Scan', 'scan with the filters above: at least one; the snapshots matching all of them are listed')}` +
      `${anyFilter ? `<button data-snapclear="${esc(scope)}">Clear filters</button>` : ''}</div>` +
      `${repLine}${liveLine}${state}${list && !st.error ? `<ul class="snaps">${list}</ul>` : ''}`);
  }

  function snapshotsHtml(snap) {
    if (!snap) return '';
    const projects = (snap.projects || []).map(st => snapPanelHtml(st, st.name,
      `<p class="note">Scans <code>${esc(st.pattern || `projects/<root of ${st.name} in master.yaml>/artifacts/<ENV>/generated_models/<model_name>/config/<model_name>_params.yaml`)}</code>: ` +
      `${esc(st.name)} as master.yaml registers it (its root folder), joined with paths.params_out of its params.yaml.</p>`)).join('');
    const c = snap.custom || { scope: 'custom' };
    const custom = snapPanelHtml(c, 'CUSTOM',
      `<p class="note">Folder: <code>${esc(c.folder || 'none chosen')}</code> ` +
      `<button data-snap="choose" data-scope="custom" title="the folder holding the snapshots (<model_name>_params.yaml, at any depth)">Choose folder…</button>` +
      `${c.folder ? `<button data-snap="scanall" data-scope="custom" title="list every snapshot under the folder, no filter">Scan</button>` : ''}<br>` +
      'Every snapshot under it. The folder is the root of the reproduced runs: every paths: entry of the snapshot applies inside it ' +
      '(&lt;folder&gt;/artifacts/&lt;ENV&gt;/generated_models/&lt;model_name&gt;/…, the registries under &lt;folder&gt;/artifacts/); ' +
      'Reproduce first copies the snapshot to its params_out path there. Config files and scripts come from the project the snapshot names (master.yaml).</p>');
    return details('snapshots', false, 'SNAPSHOTS <small>reproduce runs from their config snapshots</small>',
      `${projects}${custom}<p class="note">Reproduce asks where the outputs go: the project (or the snapshot folder) itself, ` +
      'whose outputs a reproduction can replace, or a folder you choose (the runs\' root; the snapshot is copied there first). ' +
      'It runs make reproduce ENV=&lt;the snapshot\'s ENV&gt; PROJECT=… MODEL=&lt;its name&gt; [REPRODUCE_ROOT=&lt;folder&gt;] in tmux: ' +
      'registered or not, the run trains from its snapshot under its own name and hash; an unfinished one resumes; a DONE one asks, then trains again (FORCE=1). ' +
      'Refused while an experiment runs (let the current run finish).</p>');
  }

  const DEL_FIELDS = ['ENV', 'MODEL_ID', 'DATASET', 'STATUS', 'BATCH', 'NAME', 'PARAM'];

  /** The runs ticked for deletion in this project, among those the filters show. */
  function delSel(p) {
    const have = new Set(((p.housekeeping || {}).runs || []).map(r => r.hash));
    return new Set((((ui.del || {})[p.name]) || []).filter(h => have.has(h)));
  }

  function housekeepingHtml(p) {
    const hk = p.housekeeping || {};
    const runs = hk.runs || [];
    const f = hk.filters || {};
    const sel = delSel(p);
    const all = runs.length > 0 && sel.size === runs.length;
    const hints = { ENV: 'dev, staging, prod', MODEL_ID: (p.modelIds || []).join(', '), DATASET: (p.datasets || []).join(', '),
                    STATUS: 'done, failed, in_progress, …', BATCH: 'fig26, …', NAME: 'part of a name',
                    PARAM: 'trainer.epochs=800; deep_learning.optimizer.name=adam' };
    // ENV, MODEL_ID, DATASET, STATUS, BATCH: a menu of the values the registry's runs hold (any = every run);
    // NAME (a part of a name) and PARAM (key=value of the snapshot) are typed
    const opts = hk.options || {};
    const control = k => {
      const attrs = `data-delfilter="${esc(p.name)}|${k}"`;
      if (!opts[k]) return `<input ${attrs} value="${esc(f[k] || '')}" placeholder="${esc(hints[k])}">`;
      const cur = String(f[k] || '');
      const other = cur && !opts[k].includes(cur) ? `<option value="${esc(cur)}" selected>${esc(cur)}</option>` : '';
      return `<select ${attrs}><option value=""${cur ? '' : ' selected'}>any</option>${other}` +
        opts[k].map(o => `<option value="${esc(o)}"${o === cur ? ' selected' : ''}>${esc(o)}</option>`).join('') + '</select>';
    };
    const filters = '<div class="snapfilters">' + DEL_FIELDS.map(k =>
      `<label class="snapf${k === 'PARAM' ? ' wide' : ''}" title="${k === 'PARAM' ? 'a value of the run\'s snapshot (config/&lt;name&gt;_params.yaml): ' +
        'key.path=value, alternatives by comma, conditions by ; (all must hold); a list takes an index (data.0.name) or any item (data.name). ' +
        'The dotted form trainer.epochs.800 works too.' : ''}">${k}${control(k)}</label>`).join('') +
      (Object.keys(f).length ? `<button data-delclear="${esc(p.name)}" title="show every run">Clear</button>` : '') + '</div>';
    const filtered = Object.keys(f).length ? ` of ${num(hk.total)}` : '';
    const note = hk.note ? `<p class="note">${esc(hk.note)}</p>` : '';
    const list = hk.error ? `<p class="err">${esc(hk.error)}</p>`
      : !hk.total ? '<p class="note">no run in the registry: nothing to delete</p>'
      : !runs.length ? '<p class="note">no run matches the filters</p>'
      : `<label class="delall"><input type="checkbox" data-delall="${esc(p.name)}"${all ? ' checked' : ''}> Delete all (${runs.length}${filtered} shown)</label>` +
        '<ul class="dellist">' + runs.map(r =>
          `<li><label><input type="checkbox" data-del="${esc(p.name)}" data-hash="${esc(r.hash)}"${sel.has(r.hash) ? ' checked' : ''}> ` +
          `<span class="light ${esc(r.group === 'done' ? 'green' : r.group === 'failed' ? 'red' : r.group === 'running' ? 'amber' : '')}"></span>` +
          `<b>${esc(r.name)}</b> <small>${esc(r.env || '?')} · ${esc(r.model || '?')} · ${esc(r.dataset || '?')} · ${esc(r.status || '?')}` +
          `${(r.batches || []).length ? ` · ${esc(r.batches.join(', '))}` : ''}</small></label></li>`).join('') + '</ul>' +
        `<div class="buttons"><button data-deldb="${esc(p.name)}" class="danger"${sel.size ? '' : ' disabled'} ` +
        'title="make delete_database JSON=1 for the ticked runs (a dry run), then one last OK runs it with CONFIRM=1">' +
        `DELETE${sel.size ? ` (${sel.size})` : ''}</button></div>`;
    return details(`h:${p.name}`, false, 'Housekeeping', filters + note + list +
      '<p class="note">Filter the runs (empty = every run), tick the ones to delete (Delete all ticks every one shown), then DELETE: ' +
      'the dry run lists them and asks once more. Only ticked runs the filters show are deleted. ' +
      'Runs go from artifacts/, model_selection/ and graphs/ (tmux logs stay). Refused while an experiment runs here.</p>');
  }

  const figPdf = f => (f.filePaths || []).find(x => /\.pdf$/i.test(x)) || (f.filePaths || [])[0];
  /** A drawn figure's file under graphs/ (a facet panel's address); '' when it is not drawn. */
  // a drawn figure's facet address (extension.js: under graphs/, or artifacts/... for a run's own figure)
  const figAddr = f => (f && f.drawn && f.address ? f.address : '');
  /** The figures ticked in this project, as ids in the order ticked, among those listed. */
  function figSel(p) {
    const have = new Set(p.figures.map(f => String(f.id)));
    return (((ui.figsel || {})[p.name]) || []).filter(id => have.has(id));
  }
  /** The facets ticked in this project (FIGURES → Faceting), as ids in the order ticked. */
  function facSel(p) {
    const have = new Set((p.facets || []).map(f => String(f.id)));
    return (((ui.facsel || {})[p.name]) || []).filter(id => have.has(id));
  }
  /** The ticked figures that are drawn, in the order ticked: the panels of Export to Facet. */
  const facetFiles = p => figSel(p).map(id => figAddr(p.figures.find(f => String(f.id) === id))).filter(Boolean);

  /** A row's buttons before Draw, when its figure exists: [Open][Folder][Delete]; none before it is drawn. */
  function drawnActs(p, f, pdf) {
    if (!f.drawn || !f.dirPath) return '';
    return `${pdf ? btn(`data-open="${esc(pdf)}"`, 'Open', pdf) : ''}${btn(`data-reveal="${esc(f.dirPath)}"`, 'Folder', f.dirPath)}` +
      `<button class="danger" data-figdel="${esc(p.name)}" data-dir="${esc(f.dirPath)}" title="delete ${esc(f.dirPath)} (asks first); Draw makes it again">Delete</button>`;
  }

  function figuresHtml(p) {
    const drawnN = p.figures.filter(f => f.drawn).length;
    const sel = figSel(p);
    const files = facetFiles(p);
    const all = p.figures.length > 0 && sel.length === p.figures.length;
    const rows = p.figures.map(f => {
      const pdf = figPdf(f);
      const at = sel.indexOf(String(f.id));
      // As in Housekeeping: [box] (light) then the rest; the number is the tick order.
      return `<tr class="${f.enabled ? '' : 'off'}"><td class="pick"><label title="${f.drawn ? 'drawn' : 'not drawn yet'}; tick: Draw all, and (drawn) a panel of Export to Facet, in the order ticked">` +
        `<input type="checkbox" data-figpick="${esc(p.name)}" data-id="${esc(f.id)}"${at >= 0 ? ' checked' : ''}> ` +
        `<span class="light${f.drawn ? ' green' : ''}"></span><small>${at >= 0 ? at + 1 : ''}</small></label></td>` +
        `<td class="num">${esc(f.id)}</td><td class="name" title="${esc(f.name)}">${esc(f.name)}</td><td class="acts">` +
        `${drawnActs(p, f, pdf)}${runBtn('draw', p.name, 'Draw', `make graph GRAPH_ID=${f.id}; opens the figure`, ` data-id="${esc(f.id)}"`)}</td></tr>`;
    }).join('');
    const graph = details(`fg:${p.name}`, true, `Graphs <small>${drawnN}/${p.figures.length} drawn</small>`,
      `<div class="buttons">${btn(`data-reveal="${esc(p.graphJobsDir)}"`, 'Graph jobs folder')}` +
      `${btn(`data-facetexport="${esc(p.name)}"${files.length >= 2 ? '' : ' disabled'}`, 'Export to Facet',
        'the ticked figures that are drawn, in the order ticked, as one facet of graph_config.yaml (a row for up to three, else a grid)')}` +
      `${btn(`data-facetclear="${esc(p.name)}"${sel.length ? '' : ' disabled'}`, 'Clear', 'untick every figure')}</div>` +
      (rows ? `<div class="figall"><label class="delall"><input type="checkbox" data-figall="${esc(p.name)}"${all ? ' checked' : ''}> Select all (${p.figures.length})</label>` +
        `<button data-drawall="${esc(p.name)}"${sel.length ? '' : ' disabled'} title="make graph GRAPH_ID=… for each ticked figure, one after another, in one terminal">` +
        `Draw all (${sel.length} chosen)</button>${deleteAllHtml(p, sel)}</div><table class="figures">${rows}</table>` : '<p class="note">no graph_config.yaml jobs</p>'));
    const facets = p.facets || [];
    const fdrawn = facets.filter(f => f.drawn).length;
    return details(`f:${p.name}`, false, `Figures <small>${drawnN}/${p.figures.length} drawn · ${fdrawn}/${facets.length} facets</small>`,
      graph + details(`ff:${p.name}`, false, `Faceting <small>${fdrawn}/${facets.length} drawn</small>`, facetRowsHtml(p, facets)));
  }

  /**
   * FIGURES → Faceting: every facet of graph_config.yaml, [tick box] (light: drawn when its folder holds a
   * figure), Draw, Open, Folder, Delete; above the rows Select all, Draw all and Delete all of the ticked.
   */
  function facetRowsHtml(p, facets) {
    const sel = facSel(p);
    const all = facets.length > 0 && sel.length === facets.length;
    const rows = facets.map(f => {
      const pdf = figPdf(f);
      const at = sel.indexOf(String(f.id));
      return `<tr><td class="pick"><label title="${f.drawn ? 'drawn' : 'not drawn yet'}; tick: Draw all, Delete all, in the order ticked">` +
        `<input type="checkbox" data-facpick="${esc(p.name)}" data-id="${esc(f.id)}"${at >= 0 ? ' checked' : ''}> ` +
        `<span class="light${f.drawn ? ' green' : ''}"></span><small>${at >= 0 ? at + 1 : ''}</small></label></td>` +
        `<td class="num">${esc(f.id)}</td><td class="name" title="${esc(f.name)}">${esc(f.name)}</td><td class="acts">` +
        `${drawnActs(p, f, pdf)}${runBtn('draw-facet', p.name, 'Draw', `make graph FACE_ID=${f.id}; opens the facet`, ` data-id="${esc(f.id)}"`)}</td></tr>`;
    }).join('');
    if (!rows) return '<p class="note">no graph_config.yaml facets</p>';
    const dirs = sel.map(id => facets.find(f => String(f.id) === id)).filter(f => f && f.drawn && f.dirPath).map(f => f.dirPath);
    return `<div class="figall"><label class="delall"><input type="checkbox" data-facall="${esc(p.name)}"${all ? ' checked' : ''}> Select all (${facets.length})</label>` +
      `<button data-facdrawall="${esc(p.name)}"${sel.length ? '' : ' disabled'} title="make graph FACE_ID=… for each ticked facet, one after another, in one terminal">` +
      `Draw all (${sel.length} chosen)</button>` +
      `<button class="danger" data-figdelall="${esc(p.name)}" data-dirs="${esc(JSON.stringify(dirs))}"${dirs.length ? '' : ' disabled'}` +
      ` title="delete the drawn facets ticked (asks first); the facets stay in graph_config.yaml, Draw makes them again">Delete all (${dirs.length} chosen)</button></div>` +
      `<table class="figures">${rows}</table>`;
  }

  /** FIGURES → Delete all (n chosen): the ticked figures that are drawn; their folders, after one confirmation. */
  function deleteAllHtml(p, sel) {
    const dirs = sel.map(id => p.figures.find(f => String(f.id) === id)).filter(f => f && f.drawn && f.dirPath).map(f => f.dirPath);
    return `<button class="danger" data-figdelall="${esc(p.name)}" data-dirs="${esc(JSON.stringify(dirs))}"${dirs.length ? '' : ' disabled'}` +
      ` title="delete the drawn figures ticked (asks first); their graph jobs stay, Draw makes them again">Delete all (${dirs.length} chosen)</button>`;
  }

  /** One external test's attack, short: 100× shuffled, full_population, step 1. */
  function attackLine(a) {
    const t = (a && a.test) || {};
    return `${esc(t.attack_times)}× ${t.shuffle ? 'shuffled' : 'in order'}, ${esc(t.attack_distribution || '?')}` +
      `${t.attack_trace_budget != null ? `, ${esc(t.attack_trace_budget)} traces each` : ''}` +
      `${a && a.attack_step != null ? `, step ${esc(a.attack_step)}` : ''}`;
  }

  /**
   * TESTS: every DONE model, with Test (an external test: test_config.yaml's attack, beside the recorded
   * test), Log (its external tests' tmux log) and Registry (test_registry.json at its hash). Its external
   * tests are listed by index. A test is never deleted from here.
   */
  function testsHtml(p) {
    const models = p.tests || [];
    const count = models.reduce((n, m) => n + m.tests.length, 0);
    const key = m => ` data-key="${esc(m.hash)}"`;
    const rows = models.map(m => {
      const list = m.tests.map(t => `<li title="${esc(t.id)}: ${esc(t.reason)}"><small>#${esc(t.index)} ${esc(t.source || 'test')} ${esc(t.kind || 'all')} · ${attackLine(t.attack)} · ` +
        `<span class="state ${esc(String(t.status).toLowerCase().replace('_', '-'))}">${esc(t.status || '?')}</span>` +
        `${t.rank != null && t.status === 'DONE' ? ` · rank ${esc(t.rank)}` : ''}` +
        `${t.finished ? ` · ${esc(String(t.finished).replace('T', ' ').slice(0, 16))}` : ''}${t.runs > 1 ? ` · ${esc(t.runs)} runs` : ''}</small>` +
        `${t.ranks ? ` ${rankHtml('done', t)}` : ''}</li>`).join('');
      // one collapsible entry per model: its buttons in the summary line, its external tests inside
      const head = `<span class="test-head"><b>${esc(m.name)}</b> <small>${esc(m.model || '?')} · ${esc(m.env || '?')} · ` +
        `recorded test ${esc(m.recorded || '—')} · ${m.tests.length} external</small>` +
        `${m.ranks ? ` ${rankHtml('done', m)}` : ''}<span class="snap-acts">` +
        `${runBtn('external-test', p.name, 'Test All', `make test MODEL=${m.name} EXTERNAL=1: test_config.yaml's attack on every test trace, appended (asks first)`, `${key(m)} data-kind="all"`)}` +
        `${budgetHtml(p, m.dataset, 'external-test', `make test MODEL=${m.name} EXTERNAL=1`, key(m), 'Test Budget')}` +
        `${runBtn('test-log', p.name, 'Log', `tail -f <run>/test/test_log.txt: the tmux output of ${m.name}'s external tests`, key(m))}` +
        `${runBtn('test-registry', p.name, 'Registry', `test_registry.json at ${m.hash}`, key(m))}</span></span>`;
      return `<li class="test-model">${details(`tm:${p.name}:${m.hash}`, false, head,
        list ? `<ul class="ext-tests">${list}</ul>` : '<p class="note">no external test yet</p>')}</li>`;
    }).join('');
    return details(`t:${p.name}`, false, `Tests <small>${models.length} models · ${count} external</small>`,
      budgetBar(p) + (rows ? `<ul class="tests">${rows}</ul>` : '<p class="note">no DONE run to test</p>') +
      '<p class="note">Test attacks a trained model with the perform_attack settings of test_config.yaml ' +
      '(test_request.perform_attack, and test_request.models.&lt;name&gt; for its own), from its snapshot: ' +
      'the reports go to the run\'s test/&lt;model_hash&gt;_et_&lt;index&gt;/, the outcome to test_registry.json → its hash. ' +
      'Find the TESTS records under "external_tests", keyed &lt;model_hash&gt;_et_&lt;index&gt;, each with its status_reason. ' +
      'The same settings update their record; different settings append one. The recorded test (the Done list → Retest) is never touched.</p>');
  }

  // ── RUN SINGLE EXPERIMENT: single_runs.yaml, one make orchestrate per line, no BATCH label ──────────────
  // Add a single Run stages a line (the followed ENV, this project, its first dataset and model), each line
  // shows `make orchestrate`, its ENV / DATASET / MODEL_ID menus (PROJECT is this project), Start and ×; a
  // change goes through OK / Cancel / Reset. Start runs the line as the file holds it (written first).
  const isSingle = file => /\/single_runs\.ya?ml$/.test(String(file || ''));
  function singleRunsHtml(p) {
    if (!p.singleRunsFile) return '';
    const file = p.singleRunsFile;
    const key = `cf:${file}`;
    const open = isOpen(key, false);
    const n = (p.singleRuns || []).length;
    return `<details class="cfg single" data-key="${esc(key)}" data-file="${esc(file)}"${open ? ' open' : ''}>` +
      `<summary>Run Single Experiment <small>${n} run${n === 1 ? '' : 's'}</small></summary>${open ? formHtml(file, p.name, { start: true }) : ''}</details>`;
  }
  /** A new single run's line: the followed ENV, the project, its first dataset and model. */
  function singleLine(project) {
    const p = ((last && last.projects) || []).find(x => x.name === project) || {};
    return queueLine({ ENV: p.env || 'dev', PROJECT: project, DATASET: (p.datasets || [])[0] || '', MODEL_ID: (p.modelIds || [])[0] || '' });
  }
  /** single_runs.yaml's lines: `make orchestrate` and ENV= PROJECT= DATASET= MODEL_ID= menus, × at the right end
   *  (startBtn, Run Single Experiment's own form: Start beside it; the file under Experiment Settings has none);
   *  [+] under them writes a copy of the last line. A menu's change is staged (OK / Cancel / Reset). */
  function singleLinesHtml(node, file, startBtn = false) {
    const p = ((last && last.projects) || []).find(x => x.name === curProject) || {};
    const items = node && node.k === 'list' ? node.items : [];
    const staged = !!pending[file];
    const rows = items.map((it, i) => {
      const s = shown(it.path, it.text);
      const vars = parseQueueLine(unquote(s.text));
      const x = btn(`class="x" data-op="remove" ${pathAttr(it.path)}`, '×', removed(it.path) ? 'keep' : 'remove this run');
      const r = (p.singleRuns || [])[i];
      const start = btn(`data-run="single" data-form-project="1" data-key="${i}"${staged || !r || r.error ? ' disabled' : ''}`, 'Start',
        staged ? 'OK the changes first: Start runs the line as the file holds it' : r && r.error ? r.error : `${r ? r.line : ''} (tmux)`);
      const end = `<span class="end">${startBtn ? start : ''}${x}</span>`;
      if (!vars) return `<div class="field qline single-line${removed(it.path) ? ' removed' : ''}">${scalarHtml(it)}${end}</div>`;
      return `<div class="field qline single-line${removed(it.path) ? ' removed' : ''}${s.cls}" ${pathAttr(it.path)}>` +
        `<span class="qfix">make orchestrate</span>${QVARS.map(k => queueVarHtml(k, vars)).join('')}${end}</div>`;
    });
    const path = node ? node.path : ['runs'];
    const lastLine = items.length ? unquote(shown(items[items.length - 1].path, items[items.length - 1].text).text) : singleLine(curProject);
    return rows.join('') || '<p class="note">no single run yet: Add a single Run</p>';
  }

  // ── MASTER CONTROL: master.yaml, every project's entry and the settings of all of them ─────────────
  // One form (see *Editing*), read only while any run is ongoing (every run and make read it, and it is not
  // frozen with a run's config); a project's name, folders and files and the secrets are shown, not edited
  // (menus.masterTier); a key of the run hash is written after a confirmation; every write is checked:
  // each project loads as a run loads it. Add a Project writes blueprint/master_project.yaml under a name.
  const isMasterFile = file => /\/master\.ya?ml$/.test(String(file || ''));
  function masterHtml(m) {
    if (!m || !m.path) return '';
    const key = `cf:${m.path}`;
    const open = isOpen(key, false);
    return `<details class="cfg master" data-key="${esc(key)}" data-file="${esc(m.path)}"${open ? ' open' : ''}>` +
      `<summary>Master Control <small>master.yaml${m.lock ? ' · read only: a run is ongoing' : ''}</small></summary>` +
      (open ? formHtml(m.path, '') : '') + '</details>';
  }

  // EXPERIMENT SETTINGS: what each file does, at the right end of its line (a params*.yaml or an ENV
  // file not listed here by name falls under its pattern; any other file shows none)
  const CONFIG_ROLES = [
    [/^params[^/]*\.ya?ml$/, 'experiment parameters'],
    [/^experiment_queue\.ya?ml$/, 'multiple one-off runs run collectively under a common ID'],
    [/^single_runs\.ya?ml$/, 'one-off runs'],
    [/^(dev|staging|prod)\.ya?ml$/, 'ENV overrides of params.yaml'],
    [/^graph_config\.ya?ml$/, 'customise the figures and facets'],
    [/^model_judge\.ya?ml$/, 'ranks the finished models'],
    [/^model_filter\.ya?ml$/, 'finds models by filters'],
    [/^test_config\.ya?ml$/, 'run external tests of trained models separately'],
    [/^import_config\.ya?ml$/, 'test models other studies released (model import)'],
    [/^auto_search\.ya?ml$/, 'the AutoSearch study'],
    [/^model_config\.ya?ml$/, 'extra model settings'],
  ];
  const configRole = file => (CONFIG_ROLES.find(([re]) => re.test(file)) || [])[1] || '';

  // ── the key search of EXPERIMENT SETTINGS: a box beside Config folder; Enter searches every config file's keys
  // (exact or part of a key, any case) and shows each match under its parent panels, opened; the search stays
  // until its × is clicked, and the files show as before. ────────────────────────────────────────────────
  const SEARCH_ICON = '<svg class="searchicon" width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><circle cx="5" cy="5" r="3.6" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M7.7 7.7 11 11" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>';
  const searchOf = project => ((ui.cfgSearch || {})[project] || '');
  /** The tree cut to the keys that hold the query (and what is under them), with every parent between: a key that
   *  matches keeps its whole block; a list item stays when something in it matches. Paths are the file's own. */
  function pruneTree(node, q) {
    if (!node) return null;
    if (node.k === 'map') {
      const entries = node.entries.map(e => (String(e.key).toLowerCase().includes(q) ? e : { ...e, node: pruneTree(e.node, q) }))
        .filter(e => e.node && (String(e.key).toLowerCase().includes(q) || e.node.k === 'map' || e.node.k === 'list'));
      return entries.length ? { ...node, entries } : null;
    }
    if (node.k === 'list') {
      const items = node.items.map(it => pruneTree(it, q)).filter(Boolean);
      return items.length ? { ...node, items } : null;
    }
    return null;
  }
  /** How many keys of the tree hold the query. */
  const countMatches = (node, q) => (!node ? 0 : node.k === 'map'
    ? node.entries.reduce((n, e) => n + (String(e.key).toLowerCase().includes(q) ? 1 : 0) + countMatches(e.node, q), 0)
    : node.k === 'list' ? node.items.reduce((n, it) => n + countMatches(it, q), 0) : 0);
  function searchBoxHtml(project) {
    const q = searchOf(project);
    return `<span class="cfgsearch">${SEARCH_ICON}<input data-cfgsearch="${esc(project)}" value="${esc(q)}" placeholder="search keys"` +
      ` title="a key, or part of one: Enter shows every config file's matching keys with their parent panels">` +
      `${q ? btn(`class="x" data-cfgsearchclear="${esc(project)}"`, '×', 'end the search: Experiment Settings as before') : ''}</span>`;
  }

  function configsHtml(p) {
    const q = searchOf(p.name).toLowerCase();
    let found = 0;
    const files = p.configs.map(c => {
      const key = `cf:${c.path}`;
      const role = configRole(c.file);
      const head = `<summary>${esc(c.file)}${role ? `<small class="cfg-role">${esc(role)}</small>` : ''}`;
      if (q) {                                             // searching: the files with matches, opened to them
        const t = trees[c.path];
        if (!t) return `<details class="cfg" data-key="${esc(key)}" data-file="${esc(c.path)}">${head}<small> searching…</small></summary></details>`;
        if (!t.tree) return '';                            // read, and empty (or unreadable): nothing to find in it
        const cut = pruneTree(t.tree, q);
        if (!cut) return '';
        forceOpen = true;
        let form;
        try { form = formHtml(c.path, p.name, { tree: cut }); } finally { forceOpen = false; }
        // the matches shown (a key the form hides, e.g. the other framework's block, is not one); else the file's
        const shown = [...form.matchAll(/<label title="[^"]*">([^<]*)<\/label>/g)].filter(m => m[1].toLowerCase().includes(q)).length;
        const n = shown || countMatches(t.tree, q);
        found += n;
        return `<details class="cfg" data-key="${esc(key)}" data-file="${esc(c.path)}" open>${head}<small> ${n} match${n === 1 ? '' : 'es'}</small></summary>` +
          `${form}</details>`;
      }
      const open = isOpen(key, false);
      return `<details class="cfg" data-key="${esc(key)}" data-file="${esc(c.path)}"${open ? ' open' : ''}>${head}</summary>` +
        `${open ? formHtml(c.path, p.name) : ''}</details>`;
    }).join('');
    const body = q ? (files.trim() ? `<div class="cfgsearching">${files}</div>` : `<p class="note">no key holds “${esc(searchOf(p.name))}”</p>`)
      : (files || '<p class="note">no config files yet</p>');
    return details(`c:${p.name}`, false, `Experiment Settings <small>${q ? `${found} match${found === 1 ? '' : 'es'} of “${esc(searchOf(p.name))}”` : `${p.configs.length} files`}</small>`,
      `<div class="buttons">${btn(`data-reveal="${esc(p.configDir)}"`, 'Config folder')}${searchBoxHtml(p.name)}</div>` + body);
  }
  /** Enter in the search box: the query kept (until ×); the redraw asks for every config file's keys (requestTrees). */
  function startSearch(project, value) {
    ui.cfgSearch = { ...(ui.cfgSearch || {}), [project]: String(value || '').trim() };
    if (!ui.cfgSearch[project]) delete ui.cfgSearch[project];
    save();
    if (last) main(last);
  }
  /** ×: the search ended; the forms only it opened are let go (the extension stops watching them). */
  function endSearch(project) {
    delete (ui.cfgSearch || {})[project];
    save();
    const p = ((last && last.projects) || []).find(x => x.name === project);
    for (const c of (p && p.configs) || []) {
      if (!isOpen(`cf:${c.path}`, false) && trees[c.path]) { requested.delete(c.path); send({ type: 'tree', file: c.path, open: false }); }
    }
    if (last) main(last);
  }

  // ── config forms ───────────────────────────────────────────────────────────

  const pathAttr = path => `data-path="${esc(JSON.stringify(path))}"`;
  const nodeKey = (file, path) => `n:${file}:${JSON.stringify(path)}`;
  const isBlock = node => node && (node.k === 'map' || node.k === 'list') && !node.flow;
  const LABEL_KEYS = ['name', 'id', 'type', 'model_id', 'key', 'metric', 'target', 'label'];

  const FILE_ACTIONS = {
    'model_judge.yaml': ['judge', 'Judge', 'make judge; opens the selected model'],
    'test_config.yaml': ['test', 'Test', 'make test EXTERNAL=1: external tests of the models listed here, with the perform_attack settings here (tmux); the recorded tests are not touched'],
    'import_config.yaml': ['import', 'Import', 'make import: every model listed here that is not DONE (new, failed or interrupted ones) imported and tested through TraceFlow\'s own test (tmux); DONE ones are skipped'],
  };

  // ── An Add button asks its name in the panel (menus.ASKS): a row under the button, checked as one types
  // (its form, not taken), Add or Enter adds it; × closes it. The extension checks the name again. ──────
  const ASK_LISTS = { project: [['project'], 'name'], job: [['jobs'], 'name'], facet: [['faceting'], 'name'],
    model: [['model'], 'model_id'], dataset: [['data'], 'name'] };
  /** The names the file already has for an Add button's kind (a queue: the file's keys). */
  function takenNames(kind, file) {
    const tree = trees[file] && trees[file].tree;
    if (!tree) return [];
    if (kind === 'queue') return tree.k === 'map' ? tree.entries.map(e => e.key) : [];
    const [path, key] = ASK_LISTS[kind] || [];
    const list = path ? resolveIn(tree, path).node : null;
    return list && list.k === 'list' ? list.items.map(it => unquote(((it.k === 'map' && entryOf(it, key)) || {}).text || '')).filter(Boolean) : [];
  }
  const askErr = (kind, file, value) => window.TFMenus.askError(kind, value, takenNames(kind, file));
  /** The model kinds Add a Model offers: the layer models, and the named networks under PyTorch. */
  const modelKinds = () => (MF() ? [...MF().kinds, ...(kerasRun() ? [] : MF().named)] : []);
  function askHtml(kind, file) {
    const a = ui.ask;
    if (!a || a.kind !== kind || a.file !== file) return '';
    const spec = window.TFMenus.ASKS[kind];
    const err = askErr(kind, file, a.value || '');
    const kinds = kind === 'model' ? `<select class="val" data-askkind="1" title="what kind of model">${modelKinds()
      .map(k => kindOption(k, a.modelKind)).join('')}</select>` : '';
    return `<div class="field ask" data-askrow="${esc(kind)}"><label>${esc(spec.label)}</label>` +
      `<input class="val${a.value && err ? ' bad' : ''}" data-askinput="${esc(kind)}" value="${esc(a.value || '')}" placeholder="${esc(spec.hint)}" title="Enter adds it">` +
      `${kinds}${btn(`data-askok="${esc(kind)}"${err ? ' disabled' : ''}`, 'Add')}${btn('class="x" data-askclose="1"', '×', 'not now')}` +
      `<small class="note hint askerr">${esc(a.value ? err : spec.hint)}</small></div>`;
  }

  function formHtml(file, project, opts = {}) {
    const t = trees[file];
    const act = FILE_ACTIONS[file.split('/').pop()];
    const addQueue = isQueue(file) && t && t.tree ? btn(`data-addqueue="${esc(project)}" data-file="${esc(file)}" data-sha="${esc(t.sha256)}"`,
      'Add a Queue', 'a new key of experiment_queue.yaml (you name it: its runs\' BATCH label) with one make orchestrate line to fill') : '';
    // single_runs.yaml: Add a single Run first, beside Open in editor (written at once, then a line of menus)
    const addSingle = isSingle(file) && t && t.tree ? btn(`data-addsingle="${esc(singleLine(project))}" data-file="${esc(file)}" data-sha="${esc(t.sha256)}"`,
      'Add a single Run', 'a make orchestrate line written at once (no BATCH label): the followed ENV, this project, its first dataset and model, to change in its menus') : '';
    // master.yaml: Add a Project first, beside Open in editor (off while a run is ongoing)
    const lock = isMasterFile(file) && last && last.master ? last.master.lock : null;
    const addProject = isMasterFile(file) ? btn(`data-addproject="1" data-file="${esc(file)}"${lock ? ' disabled' : ''}`, 'Add a Project',
      'a new project entry in master.yaml (blueprint/master_project.yaml under the name you give); its config folder gets the blueprint\'s files') : '';
    const complete = bpFile(file) && t && t.tree ? btn(`data-complete="${esc(file)}"`, 'Load Blueprint',
      'Meant for starting, or starting over, a config setup. Overrides the whole config; applied only after OK in the confirmation.') : '';
    const tools = `<div class="buttons">${addProject}${addSingle}${btn(`data-open="${esc(file)}"`, 'Open in editor')}${addQueue}${complete}` +
      `${act ? runBtn(act[0], project, act[1], act[2]) : ''}${isSingle(file) && opts.start ? RUN_HINT : ''}</div>` +
      `${askHtml('project', file)}${askHtml('queue', file)}`;
    if (!t) return `${tools}<p class="note">reading…</p>`;
    const bad = /refused|Error|changed on disk/.test(t.note || '');
    const note = t.note ? `<p class="${bad ? 'err' : 'note'}">${esc(t.note)}</p>` : '';
    if (t.error) return `${tools}${note}<p class="err">${esc(t.error)}</p>`;
    if (!t.tree) return `${tools}${note}<p class="note">no entries</p>`;
    curFile = file;
    curProject = project;
    const n = Object.keys(stagedOf(file)).length;
    const bar = n ? `<div class="pending"><span>${n} unsaved change${n === 1 ? '' : 's'}</span>` +
      `${btn('data-apply="1"', 'OK', 'write every change of this form to the file')}` +
      `${btn('data-discard="1"', 'Cancel', 'undo the changes and close the form')}` +
      `${btn('data-reset="1"', 'Reset', 'undo the changes; the form stays open')}</div>` : '';
    // a run of the project is ongoing: it reads the config frozen at its start, so changes apply to the next run
    const p = ((last && last.projects) || []).find(x => x.name === project);
    const run = p && p.runNotes ? p.runNotes[file] : null;
    const form = `<div class="form" data-file="${esc(file)}" data-project="${esc(project)}" data-sha="${esc(t.sha256)}">${bar}${bodyHtml(opts.tree || t.tree, file, opts.start ? { start: true } : {})}</div>`;
    // master.yaml while any run is ongoing (lock, above): shown, not editable (Cancel still drops staged changes); the extension refuses too
    if (lock) return `${tools}${note}<p class="err locked">Read only: ${esc(lock)}</p>${disable(form).replace('<button disabled data-discard="1"', '<button data-discard="1"')}`;
    return `${tools}${note}${run ? `<p class="note running">${esc(run)}</p>` : ''}${form}`;
  }

  // graph_config.yaml jobs: top_models is the judge's (read only), legend.handle one fixed pair,
  // legend.labels one row per model of the job's models (top_models for job 0) in their order.
  const isGraph = file => /\/graph_config\.yaml$/.test(file);
  const at = (path, ...tail) => path.length === 2 + tail.length && path[0] === 'jobs' && typeof path[1] === 'number' &&
    tail.every((k, i) => path[2 + i] === k);
  const entryOf = (map, key) => (map && map.k === 'map' ? (map.entries.find(e => e.key === key) || {}).node : null);
  const texts = node => (node && node.k === 'list' ? node.items.filter(i => i.k === 'scalar').map(i => i.text) : []);

  /** The keys a job's legend labels: ep0001… for an epoch overlay, else its models (job 0: top_models). */
  function legendKeys(job) {
    const ae = entryOf(job, 'array_epoch');
    const mode = entryOf(ae, 'mode');
    if (mode && mode.k === 'scalar' && mode.text === 'overlay') {
      return texts(entryOf(ae, 'epochs')).map(e => `ep${String(Number(e)).padStart(4, '0')}`);
    }
    const top = entryOf(job, 'top_models');
    return texts(top && top.k === 'list' ? top : entryOf(job, 'models'));
  }

  /** legend.labels: one row per model; node null (no labels: key yet) stages drafts at path. */
  function labelsHtml(key, node, job, path = node.path) {
    const have = {};
    if (node && node.k === 'map') for (const e of node.entries) if (e.node.k === 'scalar') have[e.key] = e.node.text;
    const keys = legendKeys(job);
    const putOf = k => stagedOf(curFile)[editId('put', path, k)];
    const row = (k, extra) => {
      if (!node) {
        const d = stagedOf(curFile)[draftId([...path, k])];
        return `<div class="field"><label title="${esc(k)}">${esc(k)}</label><input class="val${d ? ' dirty' : ''}" data-draft="${esc(JSON.stringify([...path, k]))}"` +
          ` data-t="text" value="${esc(d ? d.value : '')}" placeholder="legend label"></div>`;
      }
      return `<div class="field${extra ? ' stray' : ''}"><label title="${esc(k)}">${esc(k)}</label>` +
        `<input class="val${putOf(k) ? ' dirty' : ''}" data-op="put" ${pathAttr(path)} data-key="${esc(k)}" value="${esc(putOf(k) ? putOf(k).value : (have[k] || ''))}"` +
        ` placeholder="${extra ? 'not in models' : 'legend label'}"></div>`;
    };
    const rows = keys.map(k => row(k, false)).join('') + Object.keys(have).filter(k => !keys.includes(k)).map(k => row(k, true)).join('');
    return details(`n:${path.join('/')}`, true, `${esc(key)} <small>${keys.length} from models</small>`,
      rows || '<p class="note">no models yet: list them under models: and a label row appears here for each</p>');
  }

  /** graph_config.yaml's jobs and faceting lists: their items, headed by Add a Job / Add a Facet. */
  const graphList = (file, node) => isGraph(file) && node.k === 'list' && node.path.length === 1 && ['jobs', 'faceting'].includes(node.path[0]);
  const addButton = section => `<div class="buttons">${section === 'jobs'
    ? btn(`data-addjob="${esc(curProject)}"`, 'Add a Job', 'a job at the next id: you name it, then list its models here (graph_export.py new-job)')
    : btn(`data-addfacet="${esc(curProject)}"`, 'Add a Facet', 'a facet at the next id: you name it, then + add a panel and Find Figure its files (graph_export.py new-facet)')}</div>` +
    askHtml(section === 'jobs' ? 'job' : 'facet', curFile);
  /** A facet's panels: each panel, then + add a panel (the next file_id; Find Figure fills its address). */
  function panelsHtml(node) {
    const items = node.k === 'list' ? node.items : [];
    const adds = Object.entries(stagedOf(curFile)).filter(([, e]) => e.op === 'add' && JSON.stringify(e.path) === JSON.stringify(node.path));
    const ids = items.map(it => Number(unquote((entryOf(it, 'file_id') || {}).text))).filter(Number.isFinite);
    const next = Math.max(0, ...ids, ...adds.map(([, e]) => Number((JSON.parse(e.value) || {}).file_id) || 0)) + 1;
    const staged = adds.map(([id, e]) => `<span class="item added">panel ${esc((JSON.parse(e.value) || {}).file_id)}` +
      `${btn(`class="x" data-unstage="${esc(id)}"`, '×', 'undo this addition')}</span>`).join('');
    return blockHtml(node.path, 'panels', String(items.length), () => blockButtons(node.path) +
      items.map((it, i) => itemHtml(it, i, curFile, {})).join('') +
      `<div class="field qadd-row">${staged}${btn(`data-addpanel="1" ${pathAttr(node.path)} data-next="${next}"`, '+ add a panel',
        `panel ${next}: its address is picked with Find Figure once OK has written it`)}</div>`);
  }

  // ── params.yaml models: Add a Model, the architecture its layers make, and the layer builder ─────────────
  // choices.py model_forms: every layer core.model builds (_LAYER_MAP) by group, each argument of its class
  // (torch.nn's signature), TraceFlow's activation; under keras the layers framework_switch maps.
  const MF = () => (MENUS && MENUS.params && MENUS.params.models) || null;
  const modelChecks = {};                     // project -> model_id -> model_check.py's answer
  const isLayers = (file, path) => isParams(file) && path.length === 4 && path[0] === 'model' && typeof path[1] === 'number' &&
    path[2] === 'architecture_args' && path[3] === 'layers';
  const kerasRun = () => frameworkBlock() === 'keras';
  /** Why a model kind cannot be chosen ('' when it can): core.model's layer-list limits, or no Keras layer. */
  function kindBlocked(kind) {
    const mf = MF();
    if (!mf || !mf.kinds.includes(kind)) return '';
    if ((mf.disabledKinds || {})[kind]) return mf.disabledKinds[kind];
    if (kerasRun() && mf.kerasKinds && !mf.kerasKinds.includes(kind)) return 'under deep_learning.framework: keras: Keras builds no attention or recurrent layer';
    return '';
  }
  /** Why a layer cannot go into a layer list ('' when it can): core.model.LAYER_LIST_LIMITS. */
  const layerBlocked = name => ((MF() || {}).limits || {})[name] || '';
  const kindOption = (k, cur) => {
    const why = kindBlocked(k);
    return `<option value="${esc(k)}"${k === cur ? ' selected' : ''}${why ? ` disabled title="${esc(why)}"` : ''}>${esc(k)}${why ? ' (not available)' : ''}</option>`;
  };
  function layerNames() {
    const mf = MF();
    return mf ? Object.keys(mf.layers).filter(n => !kerasRun() || mf.keras.layers.includes(n)) : [];
  }
  /** What a layer list makes: attention → TRANSFORMER (with convolutions or recurrence: HYBRID), recurrence →
   *  HYBRID, convolutions → CNN, else MLP (dense, normalisation, dropout, shape layers only). */
  function kindOfLayers(names) {
    const mf = MF();
    const groups = new Set(names.map(n => ((mf && mf.layers[n]) || {}).group));
    const kinds = new Set([...groups].map(g => (mf ? mf.kindOf[g] : null)).filter(Boolean));
    if (kinds.has('TRANSFORMER') && (groups.has('Convolution') || groups.has('Recurrent'))) return 'HYBRID';
    if (kinds.has('TRANSFORMER')) return 'TRANSFORMER';
    if (kinds.has('HYBRID')) return 'HYBRID';
    return kinds.has('CNN') ? 'CNN' : 'MLP';
  }
  const layerName = it => (it && it.k === 'map' && it.entries.length === 1 ? it.entries[0].key : null);
  function modelLayers(modelPath) {
    const n = resolve([...modelPath, 'architecture_args', 'layers']).node;
    return n && n.k === 'list' ? n.items.filter(it => !removed(it.path)).map(layerName).filter(Boolean) : [];
  }
  /** model.<i>.architecture: what its layers make, shown, once it has layers; a menu while it has none. */
  function architectureHtml(label, node) {
    const mf = MF();
    const s = shown(node.path, node.text);
    const cur = unquote(s.text);
    const names = modelLayers(node.path.slice(0, 2));
    if (mf && names.length && mf.kinds.includes(cur.toUpperCase())) {
      const made = kindOfLayers(names);
      const ok = made === cur.toUpperCase();
      return `<div class="field">${label}<span class="readonly" title="a layer model: core.model builds it from its layers">${esc(cur)}</span>` +
        (ok ? `<small class="note hint">its layers make a ${made}</small>`
          : `<small class="note hint err">its layers make a ${made}</small>${btn(`${pathAttr(node.path)} data-setto="${made}"`, `Set ${made}`, `write architecture: ${made}`)}`) + '</div>';
    }
    const opts = mf ? [...mf.kinds, ...(kerasRun() ? [] : mf.named)] : [cur];
    const known = opts.includes(cur);
    return `<div class="field">${label}<select class="val${s.cls}${known ? '' : ' bad'}" data-op="set" ${pathAttr(node.path)} title="a layer model (built from its layers), or a named network (PyTorch only)">` +
      (known ? '' : `<option value="${esc(cur)}" selected>${esc(cur)} (not an option)</option>`) +
      opts.map(o => (mf && mf.kinds.includes(o) ? kindOption(o, cur)
        : `<option value="${esc(o)}"${o === cur ? ' selected' : ''}>${esc(o)}${mf && mf.named.includes(o) ? ' (named)' : ''}</option>`)).join('') + '</select></div>';
  }
  /** A layer's argument as the form shows it: under keras, Pooling's types and the activations are the ones Keras
   *  builds; the others stay in the menu, never choosable, marked "available in pytorch" (owner 2026-10-23). */
  function layerArgSpec(name, a) {
    const mf = MF();
    if (!kerasRun() || !Array.isArray(a.o)) return a;
    const keep = name === 'Pooling' && a.k === 'type' ? mf.keras.pooling : a.k === 'activation' ? mf.keras.activations : null;
    if (!keep) return a;
    return { ...a, o: a.o.filter(o => keep.includes(o)),
      others: Object.fromEntries(a.o.filter(o => !keep.includes(o)).map(o => [o, 'available in pytorch'])) };
  }
  function layerRowHtml(it, j, count, check) {
    const name = layerName(it);
    const spec = name && MF() ? MF().layers[name] : null;
    if (!name) return itemHtml(it, j, curFile, {});
    const args = it.entries[0].node;
    const set = args && args.k === 'map' ? args.entries.filter(e => e.node.k === 'scalar' || e.node.flow)
      .map(e => `${e.key === 'activation' ? '' : `${e.key} `}${unquote(e.node.k === 'scalar' ? shown(e.node.path, e.node.text).text : '[…]')}`) : [];
    const shape = check && check.layers ? (check.layers.find(l => l.index === j) || {}).out : null;
    const bad = check && check.at === j;
    const gone = removed(it.path);
    const k = nodeKey(curFile, it.path);
    const open = isOpen(k, false);
    const moves = `${j > 0 ? btn(`data-movelayer="-1" ${pathAttr(it.path)}`, '▲', 'move it up') : ''}` +
      `${j < count - 1 ? btn(`data-movelayer="1" ${pathAttr(it.path)}`, '▼', 'move it down') : ''}` +
      btn(`class="x" data-op="delete" ${pathAttr(it.path)}`, gone ? 'keep' : '×', gone ? 'keep it' : 'remove this layer');
    const out = shape ? `<span class="shape" title="its output for one trace (channels, length)">→ (${esc(shape.join(', '))})</span>` : bad ? '<span class="shape err">✗</span>' : '';
    const body = () => {
      if (!spec) return `<p class="err">${esc(name)} is not a layer core.model builds${kerasRun() ? '' : ''}</p>` + bodyHtml(args || { k: 'map', path: [...it.path, name], entries: [] }, curFile, {});
      if (kerasRun() && !MF().keras.layers.includes(name)) return `<p class="err">${esc(name)} has no Keras layer (deep_learning.framework: keras builds ${esc(MF().keras.layers.join(', '))})</p>`;
      const known = spec.args.map(a => a.k);
      const strayArgs = args && args.k === 'map' ? args.entries.filter(e => !known.includes(e.key)).map(e => strayHtml(e.key, e.node, `not an argument of ${name}: the run refuses it`)).join('') : '';
      return spec.args.map(a => fieldHtml(a.k, [...it.path, name, a.k], layerArgSpec(name, a), null,
        { noX: true, binary: a.t === 'bool', required: a.required })).join('') + strayArgs;
    };
    return `<details class="node layer${gone ? ' removed' : ''}${bad ? ' bad' : ''}" data-key="${esc(k)}"${open ? ' open' : ''}><summary>#${j} ${esc(name)} ` +
      `<small>${esc(set.slice(0, 4).join(' · '))}</small>${out}<span class="end">${moves}</span></summary>${open ? body() : ''}</details>`;
  }
  /** A model's layers: Check the model, each layer, the ones staged, then [+]: the layers by group. */
  function layersHtml(node) {
    const modelPath = node.path.slice(0, 2);
    const id = valueAt([...modelPath, 'model_id']).text || '';
    const check = (modelChecks[curProject] || {})[id];
    const items = node.k === 'list' ? node.items : [];
    const key = JSON.stringify(node.path);
    const head = `<div class="buttons">${btn(`data-checkmodel="${esc(id)}"`, check && check.checking ? 'Checking…' : 'Check the model',
      'build it as a run would (core.model, one zero trace of the data entry the model fits: the entries\' trace lengths are tried in order): each layer\'s output shape')}` +
      (check && check.input && !check.checking ? `<small class="note">input (${esc(check.input.join(', '))})${check.dataset ? ` from ${esc(check.dataset)}` : ''}` +
        `${check.params != null ? ` · ${esc(Number(check.params).toLocaleString('en'))} parameters` : ''}` +
        `${(check.fits || []).length > 1 ? ` · also fits ${esc(check.fits.filter(f => f !== check.dataset).join(', '))} (the same flatten width)` : ''}</small>` : '') + '</div>' +
      (check && check.error && !check.checking ? `<p class="err">${check.at != null ? `layer #${esc(check.at)}: ` : ''}${esc(check.error)}</p>` : '');
    const arrangeId = editId('arrange', node.path);
    const arrange = stagedOf(curFile)[arrangeId];
    let planned = '';
    if (arrange) {                                          // the planner's changes of this model (Visualise), staged
      const entries = JSON.parse(arrange.value || '[]');
      const kept = entries.filter(e => e.from != null);
      planned = `<p class="note planned">rearranged in the planner: ${entries.length - kept.length} added, ` +
        `${entries.filter((e, i) => e.from != null && e.from !== i).length} moved, ${kept.filter(e => e.value).length} changed, ` +
        `${Math.max(0, items.length - kept.length)} removed${btn(`class="x" data-unstage="${esc(arrangeId)}"`, '×', 'undo the planner\'s changes')}</p>`;
    }
    const adds = Object.entries(stagedOf(curFile)).filter(([, e]) => e.op === 'add' && JSON.stringify(e.path) === key);
    const staged = adds.map(([sid, e]) => `<span class="item added">${esc(Object.keys(JSON.parse(e.value) || {})[0] || '')} (to add)` +
      `${btn(`class="x" data-unstage="${esc(sid)}"`, '×', 'undo this addition')}</span>`).join('');
    const mf = MF();
    const names = layerNames();
    const menu = mf ? mf.groups.map(g => {
      const these = names.filter(n => mf.layers[n].group === g);
      return these.length ? `<optgroup label="${esc(g)}">${these.map(n => (layerBlocked(n)
        ? `<option value="${esc(n)}" disabled title="${esc(layerBlocked(n))}">${esc(n)}</option>` : `<option value="${esc(n)}">${esc(n)}</option>`)).join('')}</optgroup>` : '';
    }).join('') : '';
    const control = (ui.adding || {})[key] && mf
      ? `<select class="add" data-addlayer="1" ${pathAttr(node.path)}><option value="">choose the layer…</option>${menu}</select>${closeAdd()}`
      : btn(`class="addopen" data-addopen="${esc(key)}"`, '+', 'add a layer at the end');
    return head + planned + items.map((it, j) => layerRowHtml(it, j, items.length, check)).join('') + `<div class="field qadd-row">${staged}${control}</div>`;
  }

  /** params.yaml trainer.optimizer: one block per optimizer (a sweep entry): its line (#i, its type and learning
   *  rates, × at the right end while another one is left) and, opened, its settings: type (the framework's
   *  optimizers), learning_rate, scheduler, weight_decay; its arguments are deep_learning.<framework>.optimizer_args.
   *  Add an Optimizer writes a new block at once, opened: a copy of the first one, to choose its optimizer and values in. */
  function optimizersHtml(node, file, ctx) {
    const items = node.items.filter(it => !removed(it.path));
    const rows = node.items.map((it, i) => {
      if (it.k !== 'map') return itemHtml(it, i, file, ctx);
      const k = nodeKey(file, it.path);
      const open = isOpen(k, false);
      const val = key => unquote(((entryOf(it, key) || {}).text) || '');
      const lr = entryOf(it, 'learning_rate');
      const rates = lr && lr.k === 'list' ? lr.items.map(x => unquote(shown(x.path, x.text).text)).join(', ') : val('learning_rate');
      const gone = removed(it.path);
      const x = items.length > 1 || gone ? btn(`class="x" data-op="delete" ${pathAttr(it.path)}`, gone ? 'keep' : '×', gone ? 'keep it' : 'remove this optimizer') : '';
      return `<details class="node${gone ? ' removed' : ''}" data-key="${esc(k)}"${open ? ' open' : ''}><summary>#${i} ` +
        `<small>type: ${esc(unquote(shown([...it.path, 'type'], val('type')).text))}${rates ? ` · learning_rate ${esc(rates)}` : ''}</small>` +
        `${x ? `<span class="end">${x}</span>` : ''}</summary>` + (open ? blockButtons(it.path) + bodyHtml(it, file, ctx) : '') + '</details>';
    }).join('');
    return rows + `<div class="field qadd-row">${btn(`data-addoptimizer="${node.items.length}" ${pathAttr(node.path)}`, 'Add an Optimizer',
      'a new optimizer block, written at once (a copy of the first one): choose its optimizer and values in it')}</div>`;
  }

  // sca.candidate_ranking: top_k is read with mode top_k only (key_recovery); the mode's options say what each ranks
  const TOP_K = ['sca', 'candidate_ranking', 'top_k'];
  function rankingMode() {
    const mode = resolve(['sca', 'candidate_ranking', 'mode']).node;
    const first = mode && mode.k === 'list' ? mode.items.find(it => !removed(it.path)) : mode;
    if (first && first.k === 'scalar') return unquote(shown(first.path, first.text).text);
    const added = Object.values(stagedOf(curFile)).find(e => e.op === 'add' && JSON.stringify(e.path) === '["sca","candidate_ranking","mode"]');
    return added ? unquote(added.value) : '';
  }
  function topKDraftHtml(node) {
    if (!isParams(curFile) || JSON.stringify(node.path) !== '["sca","candidate_ranking"]' || entryOf(node, 'top_k') || rankingMode() !== 'top_k') return '';
    const d = stagedOf(curFile)[draftId(TOP_K)];
    return `<div class="field"><label title="sca › candidate_ranking › top_k">top_k</label><input class="val${d ? ' dirty' : ' bad'}" ` +
      `data-draft="${esc(JSON.stringify(TOP_K))}" data-t="list" value="${esc(d ? JSON.stringify(d.value).replace(/^\[|\]$/g, '') : '')}" ` +
      `placeholder="required with mode top_k, e.g. 5"></div>`;
  }

  // params.yaml metrics_to_log: a metric that takes a value (sca.metrics.METRIC_OPTIONS: top_k_accuracy's k,
  // rank_k_success's k, traces_to_ge_k's k, ge_at_traces' traces, ...) shows it beside its name, the default
  // when the entry is a bare name; picked in [+], the value is asked before it is added ({name: {option: value}}).
  const MO = () => (MENUS && MENUS.params && MENUS.params.metric_options) || {};
  const isMetrics = (file, node) => isParams(file) && node.k === 'list' && JSON.stringify(node.path) === '["metrics_to_log"]';
  function metricOf(it) {                                   // {name, opts, path of the opts map | null}
    if (it.k === 'scalar') {
      const text = shown(it.path, it.text).text;
      if (/^\{/.test(String(text))) {                       // a bare name staged as {name: {option: value}}
        try { const o = JSON.parse(text); const name = Object.keys(o)[0]; return { name, opts: o[name] || {}, staged: true }; } catch (e) { /* text */ }
      }
      return { name: unquote(text), opts: {}, bare: true };
    }
    if (it.k === 'map' && it.entries.length === 1) {
      const e = it.entries[0];
      const opts = {};
      if (e.node.k === 'map') for (const x of e.node.entries) if (x.node.k === 'scalar') opts[x.key] = unquote(shown(x.node.path, x.node.text).text);
      return { name: e.key, opts, inner: e.node };
    }
    return null;
  }
  function metricsHtml(node) {
    const rows = node.items.filter(it => !hidden(it)).map(it => {
      const m = metricOf(it);
      const spec = m && MO()[m.name];
      let control;
      if (m && m.bare && !m.staged) control = scalarHtml(it);
      else control = `<span class="readonly mname" title="metrics_to_log entry">${esc(m ? m.name : '?')}</span>`;
      let opt = '';
      if (spec) {
        const o = spec.option;
        const inner = m.inner && m.inner.k === 'map' ? entryOf(m.inner, o) : null;
        const cur = m.opts[o] != null ? String(m.opts[o]) : '';
        const attrs = inner ? `data-op="set" ${pathAttr(inner.path)}`
          : m.inner && m.inner.k === 'map' ? `data-op="put" ${pathAttr(m.inner.path)} data-key="${esc(o)}"`
          : `data-metricopt="${esc(m.name)}" data-option="${esc(o)}" ${pathAttr(it.path)}`;
        opt = `<span class="mopt">${esc(o)} <input class="val${m.staged ? ' dirty' : ''}" type="number" min="${o === 'p' ? 0 : 1}" ${attrs} ` +
          `value="${esc(cur)}" placeholder="${esc(spec.default)} (default)" size="7" title="${esc(m.name)}: its ${esc(o)}"></span>`;
      }
      return `<div class="field${removed(it.path) ? ' removed' : ''}">${control}${opt}${itemX(node, it)}</div>`;
    }).join('');
    // the metrics staged for adding (OK writes them): rows like the written ones, each a menu (or its name and
    // value) and × that drops it again
    const adds = Object.entries(stagedOf(curFile)).filter(([, e]) => e.op === 'add' && JSON.stringify(e.path) === JSON.stringify(node.path));
    const staged = adds.map(([id, e]) => {
      const m = metricOf({ k: 'scalar', text: e.value, path: null }) || { name: e.value, opts: {}, bare: true };
      const spec = MO()[m.name];
      const menu = menuOf([...node.path, 0]);
      const taken = node.items.map(it => (metricOf(it) || {}).name).filter(Boolean);
      const control = m.bare && menu && menu.opts.length
        ? `<select class="val dirty" data-stagedadd="${esc(id)}" title="to be written by OK; another name replaces it">${menu.opts.filter(o => o === m.name || !taken.includes(o))
          .map(o => `<option value="${esc(o)}"${o === m.name ? ' selected' : ''}>${esc(o)}</option>`).join('')}</select>`
        : `<span class="readonly mname" title="to be written by OK">${esc(m.name)}</span>`;
      const opt = spec ? `<span class="mopt">${esc(spec.option)} <input class="val dirty" type="number" min="${spec.option === 'p' ? 0 : 1}" data-stagedopt="${esc(id)}" data-option="${esc(spec.option)}" ` +
        `value="${esc(m.opts[spec.option] != null ? m.opts[spec.option] : spec.default)}" size="7" title="${esc(m.name)}: its ${esc(spec.option)}"></span>` : '';
      return `<div class="field">${control}${opt}${btn(`class="x" data-unstage="${esc(id)}"`, '×', 'drop this addition')}</div>`;
    }).join('');
    const have = node.items.map(it => (metricOf(it) || {}).name).filter(Boolean)
      .concat(adds.map(([, e]) => (metricOf({ k: 'scalar', text: e.value, path: null }) || {}).name));
    // the standard of every project (owner 2026-10-22): one menu of every metric the system records, always
    // shown, and [+] adding the one chosen; a metric that takes a value (top_k_accuracy's k, ...) shows its
    // value box beside the menu, with its default, as a written one does
    const menu = menuOf([...node.path, 0]);
    const left = ((menu && menu.opts) || []).filter(o => !have.includes(o));
    const key = JSON.stringify(node.path);
    const picked = (ui.metricPick || {})[key] || '';
    const pspec = picked ? MO()[picked] : null;
    const pval = pspec ? ((ui.metricPickOpt || {})[key] != null ? ui.metricPickOpt[key] : pspec.default) : '';
    const pick = `<span class="mpick"><select class="add wide" data-metricpick="${esc(key)}" title="every metric the system records, not yet in the list">` +
      `<option value="">choose a metric…</option>${left.map(o => `<option value="${esc(o)}"${o === picked ? ' selected' : ''}>${esc(o)}</option>`).join('')}</select>` +
      (pspec ? `<span class="mopt">${esc(pspec.option)} <input class="val" type="number" min="${pspec.option === 'p' ? 0 : 1}" data-metricpickopt="${esc(key)}" ` +
        `value="${esc(pval)}" size="7" title="${esc(picked)}: its ${esc(pspec.option)} (default ${esc(pspec.default)})"></span>` : '') +
      `${btn(`class="qadd" data-metricadd="${esc(key)}" ${pathAttr(node.path)}${picked ? '' : ' disabled'}`, '+',
        picked ? `add ${picked}${pspec ? ` (${pspec.option} ${pval})` : ''} to metrics_to_log (OK writes it)` : 'choose a metric first')}</span>`;
    return rows + staged + `<div class="field">${pick}</div>`;
  }

  // ── the blueprint (choices.py blueprint_forms): every key a project's config file may hold, with the value a
  // new project starts with, its menu and its hint. A key the file lacks is drawn faded with that value; choosing
  // a value stages it (OK writes it). Load Blueprint stages every missing key and every empty value. ─
  const BP = () => (MENUS && MENUS.blueprint) || null;
  const bpFile = file => { const bp = BP(); return bp && !isMasterFile(file) ? bp.files[String(file || '').split('/').pop()] || null : null; };
  /** The blueprint's value at a path of the file: a list item is the blueprint's first item of that list, else
   *  templates.yaml's (data entries, models, optimizers); undefined where the blueprint has nothing. */
  function bpAt(file, path) {
    const f = bpFile(file);
    if (!f) return undefined;
    const bp = BP();
    let v = f.tree;
    const keys = [];
    for (const seg of path) {
      if (typeof seg === 'number') {
        const tpl = bp.lists[keys.join('.')];
        v = Array.isArray(v) && v.length && v[0] && typeof v[0] === 'object' ? v[0] : tpl ? bp.templates[tpl] : Array.isArray(v) ? v[seg] : undefined;
        keys.push('*');
      } else {
        v = v && typeof v === 'object' && !Array.isArray(v) ? v[seg] : undefined;
        keys.push(String(seg));
      }
      if (v === undefined) return undefined;
    }
    return v;
  }
  /** The tooltip of a label, a field row and a block or item line: the key's path (owner 2026-10-22: no comment
   *  text; the file's and the blueprint's comments were not accurate as tooltips). */
  const titled = (file, path, head) => esc(head);
  /** A key's hint: its comment in the blueprint (a list item's key: in templates.yaml). */
  function bpHint(file, path) {
    const f = bpFile(file);
    if (!f) return '';
    const bp = BP();
    const dotted = path.map(x => (typeof x === 'number' ? '*' : String(x)));
    for (const [list, tpl] of Object.entries(bp.lists)) {
      const lp = list.split('.');
      if (dotted.length > lp.length + 1 && lp.every((x, i) => dotted[i] === x) && dotted[lp.length] === '*') {
        const hint = bp.template_hints[[tpl, ...dotted.slice(lp.length + 1)].join('.')];
        if (hint) return hint;
      }
    }
    return f.hints[dotted.join('.')] || '';
  }
  /** A blueprint value as a field: its menu's options (a free menu suggests), true / false, a number, a list. */
  function bpSpec(path, d) {
    const list = Array.isArray(d);
    const menu = menuOf(list ? [...path, 0] : path);
    if (menu && menu.opts.length && !list && !menu.free) return { t: 'enum', o: menu.opts, d: d == null ? '' : d, empty: d == null };
    if (typeof d === 'boolean') return { t: 'bool', d };
    if (typeof d === 'number') return { t: Number.isInteger(d) ? 'int' : 'num', d };
    if (list) return { t: 'list', d: d.map(String).join(', '), ...(menu && menu.opts.length ? { o: menu.opts } : {}) };
    return { t: 'str', d: d == null ? '' : d, ...(menu && menu.opts.length ? { o: menu.opts } : {}) };
  }
  /** seed: [42, 53, 64] under cross-validation: a run trains with its seed and the next n_folds, so repeats must be
   *  n_folds + 1 apart (config_loader.check_seed_spacing refuses closer ones); said here as the list is typed. */
  function seedHint(file, node) {
    if (!isParams(file) || JSON.stringify(node.path) !== '["seed"]') return '';
    const vals = node.items.filter(it => !removed(it.path)).map(it => Number(unquote(shown(it.path, it.text).text)))
      .concat(Object.values(stagedOf(file)).filter(e => e.op === 'add' && JSON.stringify(e.path) === '["seed"]').map(e => Number(e.value)))
      .filter(Number.isFinite).sort((a, b) => a - b);
    if (vals.length < 2 || unquote(valueAt(['cross_validation', 'enabled']).text || '') !== 'true') return '';
    const folds = parseInt(String(valueAt(['cross_validation', 'n_folds']).text || '').replace(/[[\]]/g, ''), 10);
    if (!Number.isFinite(folds)) return '';
    const gap = folds + 1;
    const close = vals.slice(1).map((b, i) => [vals[i], b]).find(([a, b]) => b - a < gap);
    if (close) {
      return `<small class="note hint err" title="a run trains with its seed and the next ${folds} (the folds): two seeds closer than ${gap} share fold trainings; the config check refuses it">` +
        `seeds ${close[0]} and ${close[1]} are ${close[1] - close[0]} apart: repeats need at least ${gap} (n_folds + 1), e.g. ${vals.map((_, k) => vals[0] + k * gap).join(', ')}</small>`;
    }
    return `<small class="note hint ok" title="each run trains with its seed and the next ${folds} (the folds)">${vals.length} repeats, at least ${gap} apart: no shared training seed</small>`;
  }
  /** Beside data[].type: `type.attack_byte = {…}`, the byte(s) the type defines a target for within the entry's file
   *  (refs attack_bytes_by_entry for the entry as the file holds it; the type's own bytes for a type just chosen). */
  function typeByteHint(node) {
    const name = unquote((valueAt([...node.path.slice(0, 2), 'name']).text) || '');
    const chosen = unquote(shown(node.path, node.text).text || '');
    const byEntry = (((MENUS && MENUS.refs && MENUS.refs[curProject]) || {}).attack_bytes_by_entry || {})[name];
    const typeBytes = ((MENUS && MENUS.params && MENUS.params.type_bytes) || {})[chosen];
    const bytes = chosen === unquote(node.text) && byEntry ? byEntry : typeBytes;
    if (!bytes) return '';
    const text = bytes === 'any' || (Array.isArray(bytes) && bytes.length === 16) ? '{0..15}' : `{${bytes.join(', ')}}`;
    return `<small class="note hint" title="the byte(s) this type defines a target for, within the entry's file window; sca.attack_byte left empty takes it for a run of this entry (the source of the byte is the dataset chosen); a type with several bytes needs sca.attack_byte set">${esc(`type.attack_byte = ${text}`)}</small>`;
  }
  /** sca.attack_byte: empty, a run takes the byte of the data entry it is started on (DATASET=); the entries by the
   *  bytes they define (choices.py attack_bytes_by_entry: the type's target bytes within the file's window). */
  function attackByteHint(file, node) {
    if (!isParams(file) || JSON.stringify(node.path) !== '["sca","attack_byte"]') return '';
    const by = ((MENUS && MENUS.refs && MENUS.refs[curProject]) || {}).attack_bytes_by_entry;
    if (!by || !Object.keys(by).length) return '';
    const groups = {};                                   // the entries by the bytes they allow
    for (const [name, bytes] of Object.entries(by)) (groups[bytes.length === 16 ? 'any' : bytes.join(', ')] = groups[bytes.length === 16 ? 'any' : bytes.join(', ')] || []).push(name);
    const text = Object.entries(groups).map(([bytes, names]) => `byte ${bytes}: ${names.length} entr${names.length === 1 ? 'y' : 'ies'}`).join(' · ');
    const title = Object.entries(groups).map(([bytes, names]) => `byte ${bytes}: ${names.join(', ')}`).join('\n');
    const empty = !node.items.filter(it => !removed(it.path) && !redundantByte(node, it)).length && !Object.values(stagedOf(file)).some(e => e.op === 'add' && JSON.stringify(e.path) === '["sca","attack_byte"]');
    return `<small class="note hint" title="${esc(`from data.type at the time of the run\n${title}`)}">${esc(`${empty ? 'from data.type at the time of the run · ' : ''}${text}`)}</small>`;
  }
  const SKIP_ABSENT = ['["sca","candidate_ranking","top_k"]'];   // drawn by its own rule (topKDraftHtml)
  // ── data[].trace_normalization: a union on its mode (choices.py normalization_forms, from the loader's
  // NORMALIZATION_KEYS): the chosen mode's keys are shown, drafted when the entry lacks them, and written on OK;
  // the other modes' keys are not drawn (eps belongs to zscore and minmax; fitted's feature_range to a minmax step) ──
  const NORM = () => (MENUS && MENUS.params && MENUS.params.normalization) || null;
  const isNormPath = path => isParams(curFile) && path.length === 4 && path[0] === 'data' && path[2] === 'trace_normalization';
  // ── data[].profiling_split: test_size, shuffle and random_state are read only with enabled: true
  // (sca/dataset_types.profiling_split_problems); with enabled false (or unset) they are not drawn or completed ──
  const isSplitArg = path => path.length === 4 && path[0] === 'data' && path[2] === 'profiling_split' && path[3] !== 'enabled';
  const splitOn = block => String(valueAt([...block, 'enabled']).text || '') === 'true';
  const SPLIT = () => (MENUS && MENUS.params && MENUS.params.profiling_split) || null;
  /** profiling_split.enabled changed in the form: true stages each argument the entry lacks or leaves empty with its
   *  default (choices.py profiling_split_forms; written on OK, editable before), false drops what this staged. */
  function splitDefaults(form, enabledPath, on) {
    const s = SPLIT();
    if (!s || !form || !isParams(form.dataset.file)) return;
    curFile = form.dataset.file;                              // resolve reads the form being changed
    for (const [k, d] of Object.entries(s.defaults)) {
      const path = [...enabledPath.slice(0, 3), k];
      const r = resolve(path);
      if (!r.node) setDraft(form, path, on ? d : undefined);  // the entry lacks it: a draft, or none
      else if (r.node.k === 'scalar' && unquote(r.node.text) === '') {   // empty only: a written null (unseeded) is kept
        const p = pending[form.dataset.file];
        if (on) stageValue(form, r.path, 'set', String(d));    // written empty: given its default
        else if (p) { delete p.edits[editId('set', r.path)]; prune(form.dataset.file); }
      }
    }
  }
  const isSplitSwitch = path => Array.isArray(path) && path.length === 4 && path[0] === 'data' &&
    path[2] === 'profiling_split' && path[3] === 'enabled';
  /** The values of a flow list at path, as the file holds or the form stages them. */
  function listAt(path) {
    const r = resolve(path);
    if (r.node && r.node.k === 'list') return r.node.items.map(it => unquote(it.text));
    const d = stagedOf(curFile)[draftId(path)];
    return !d ? [] : Array.isArray(d.value) ? d.value.map(String) : String(d.value).split(',').map(x => x.trim()).filter(Boolean);
  }
  /** The keys the block's chosen mode reads: none until a mode is chosen. */
  function normKeys(block) {
    const n = NORM();
    if (!n) return [];
    const mode = unquote(valueAt([...block, 'mode']).text || '');
    const keys = n.keys[mode] || [];
    return mode === 'fitted' && !listAt([...block, 'steps']).includes('minmax') ? keys.filter(k => k !== 'feature_range') : keys;
  }
  function normDraftHtml(node, file) {
    if (node.k !== 'map' || !isParams(file) || node.path.length !== 3 || node.path[0] !== 'data' || node.path[2] !== 'trace_normalization') return '';
    const n = NORM();
    const has = k => node.entries.some(e => e.key === k);
    return normKeys(node.path).filter(k => !has(k)).map(k => {
      const d = (n.defaults || {})[k] == null ? null : n.defaults[k];
      return fieldHtml(k, [...node.path, k], bpSpec([...node.path, k], d), null, { hint: (n.hints || {})[k] || '' });
    }).join('');
  }
  /** The keys of the blueprint the map lacks, faded, with the blueprint's value; a block the file lacks
   *  holds its keys the same way. */
  function bpAbsentHtml(node, file) {
    const want = bpAt(file, node.path);
    if (!want || typeof want !== 'object' || Array.isArray(want)) return '';
    const has = k => node.k === 'map' && node.entries.some(e => e.key === k);
    return Object.keys(want).filter(k => !has(k)).map(k => bpAbsentKey(file, [...node.path, k], k, want[k])).join('');
  }
  function bpAbsentKey(file, path, key, d) {
    if (SKIP_ABSENT.includes(JSON.stringify(path))) return '';
    if (isParams(file) && isSplitArg(path) && !splitOn(path.slice(0, 3))) return '';
    if (isParams(file) && isSplitArg(path) && SPLIT() && d == null) d = SPLIT().defaults[key];   // shown as the default
    if (isParams(file) && path.length === 2 && path[0] === 'deep_learning' && MENUS && MENUS.params &&
        Object.values(MENUS.params.frameworks.blocks).includes(key) && key !== frameworkBlock()) return '';   // the other framework's block
    const hint = bpHint(file, path);
    if (d && typeof d === 'object' && !Array.isArray(d)) {
      if (!Object.keys(d).length) return '';           // an empty block ({}): nothing to draw
      const inner = Object.keys(d).map(k => bpAbsentKey(file, [...path, k], k, d[k])).join('');
      return inner ? blockHtml(path, key, `<span class="absentnote" title="${esc(hint || 'not in the file: the blueprint\'s keys, to fill')}">not in the file</span>`, () => inner) : '';
    }
    if (Array.isArray(d) && (d.some(x => x && typeof x === 'object') || BP().lists[path.join('.')])) return '';   // a list of blocks: its own Add button
    return fieldHtml(key, path, bpSpec(path, d), null, { hint });
  }
  /** Load Blueprint: every key the file lacks staged with the blueprint's value, and every empty
   *  value given the blueprint's (nothing written until OK; Cancel drops them). Returns how many. */
  function completeFromBlueprint(form, file) {
    const t = trees[file];
    if (!t || !t.tree || !bpFile(file)) return { n: 0, blocks: [] };
    let n = 0;
    const paths = [];
    const count = path => { n += 1; paths.push(path); };
    (function walk(node, want) {
      if (!want || typeof want !== 'object') return;
      if (node.k === 'list') {
        node.items.forEach((it, i) => { if (it.k === 'map') walk(it, bpAt(file, [...node.path, i])); });
        return;
      }
      if (node.k !== 'map' || Array.isArray(want)) return;
      for (const [k, w] of Object.entries(want)) {
        const path = [...node.path, k];
        const d = isParams(file) && isSplitArg(path) && SPLIT() && w == null ? SPLIT().defaults[k] : w;   // enabled: its default
        const child = entryOf(node, k);
        if (SKIP_ABSENT.includes(JSON.stringify(path))) continue;
        if (isParams(file) && isSplitArg(path) && !splitOn(path.slice(0, 3))) continue;   // enabled: false reads none
        if (isParams(file) && path.length === 2 && path[0] === 'deep_learning' && MENUS && MENUS.params &&
            Object.values(MENUS.params.frameworks.blocks).includes(k) && k !== frameworkBlock()) continue;
        if (!child) {
          if (Array.isArray(d) && (d.some(x => x && typeof x === 'object') || BP().lists[path.join('.')])) continue;   // a list of blocks: Add a ...
          if (d && typeof d === 'object' && !Array.isArray(d)) {                        // a block: its keys one by one
            if (Object.keys(d).length) walk({ k: 'map', path, entries: [] }, d); else { setDraft(form, path, {}); count(path); }
            continue;
          }
          setDraft(form, path, d);                                                      // empty too: the key, to fill
          count(path);
        } else if (child.k === 'scalar' && child.type === 'null' && unquote(child.text) === '' && d !== null && typeof d !== 'object') {   // empty, not a written null
          stageValue(form, child.path, 'set', String(d));     // as if typed: yamledit reads it as YAML
          count(child.path);
        } else if (child.k === 'list' && !child.items.length && Array.isArray(d) && d.length && d.every(x => x !== null && typeof x !== 'object')) {
          const p = pending[file] || (pending[file] = { sha: form.dataset.sha, edits: {}, seq: 0 });
          for (const x of d) { p.edits[`add|${JSON.stringify(child.path)}|${p.seq++}`] = { op: 'add', path: child.path, value: String(x) }; count(child.path); }
        } else if (child.k === 'map' || child.k === 'list') walk(child, d);
      }
    }(t.tree, bpFile(file).tree));
    // params.yaml: the staged keys that join the run hash (choices.HASHED), by their top block
    const hashed = isParams(file) ? (BP().hashed || []).map(h => h.split('.')) : [];
    const unhashed = isParams(file) ? (BP().unhashed || []).map(h => h.split('.')) : [];   // in a hashed block, left out
    const under = (p, keys) => keys.some(h => h.every((x, i) => x === '*' || String(p[i]) === x));   // *: any list item
    const blocks = [...new Set(paths.filter(p => under(p, hashed) && !under(p, unhashed)).map(p => String(p[0])))];
    return { n, blocks };
  }

  /** A list the panel draws as a block list with its own Add button (a project's lists: data entries, models,
   *  optimizers, metrics, a model's layers, graph jobs and facets): never as a row of values, even while empty. */
  /** A graph job's processes: one row per process (a menu, ×) and a menu to add one, as DLSCA_ASCAD's jobs show them,
   *  whether the file writes the list in block or flow style, also while empty. */
  const graphProcesses = (file, node) => isGraph(file) && node.k === 'list' && node.path.length === 3 && node.path[0] === 'jobs' && node.path[2] === 'processes';
  const blockList = (file, node) => node.k === 'list' && (isMetrics(file, node) || graphList(file, node) || isLayers(file, node.path) ||
    graphProcesses(file, node) ||
    (isQueue(file) && node.path.length === 1) ||          // a queue: its line (× at the end) and its make lines, also while empty
    (isParams(file) && ['["data"]', '["model"]', '["trainer","optimizer"]'].includes(JSON.stringify(node.path))));
  function bodyHtml(node, file, ctx = {}) {
    if (isSingle(file) && node.k === 'map' && node.path.length === 0) return singleLinesHtml(entryOf(node, 'runs'), file, !!ctx.start);
    if (isMetrics(file, node)) return metricsHtml(node);
    if (isParams(file) && node.k === 'map' && JSON.stringify(node.path) === '["paths"]') {     // read only here: edit it in the file
      return `<div class="buttons">${btn(`data-open="${esc(file)}"`, 'Open in editor', 'paths is edited in params.yaml, not here: the standard run layout')}</div>` +
        node.entries.map(e => entryHtml(e.key, e.node, file, node, ctx)).join('');
    }
    if (node.k === 'map') {
      const extra = isGraph(file) && GF() ? absentHtml(node, ctx) : topKDraftHtml(node) + normDraftHtml(node, file) + bpAbsentHtml(node, file);
      return (node.entries.map(e => entryHtml(e.key, e.node, file, node, ctx)).join('') + extra) || '<p class="note">{}</p>';
    }
    if (node.k === 'list' && node.scalars && node.flow && !blockList(file, node)) return `<div class="field">${valuesHtml(node)}</div>`;
    if (node.k === 'list' && node.scalars && isQueue(file) && node.path.length === 1) return queueLinesHtml(node);
    if (node.k === 'list' && node.scalars && (graphProcesses(file, node) || !blockList(file, node))) {   // a block list of values: one row per item
      return node.items.filter(it => !hidden(it)).map(it => `<div class="field${removed(it.path) ? ' removed' : ''}">${scalarHtml(it)}${itemX(node, it)}</div>`).join('') +
        `<div class="field">${addedHtml(node.path, true)}${addControl(node.path, node.items.map(it => unquote(it.text)), true)}</div>`;
    }
    if (isParams(file) && node.k === 'list' && JSON.stringify(node.path) === '["trainer","optimizer"]') return optimizersHtml(node, file, ctx);
    if (isParams(file) && node.k === 'list' && isLayers(file, node.path)) return layersHtml(node);
    if (isParams(file) && node.k === 'list' && JSON.stringify(node.path) === '["model"]' && MF()) {
      return `<div class="buttons">${btn('data-addmodel="1"', 'Add a Model', 'a new model: its model_id, then a layer model (CNN, MLP, ...) or a named network')}` +
        `${btn('data-planmodel="1"', 'Plan a Model', 'design a model on the planner page (an editor tab): a skeleton, patterns, every layer by hand, then Add the Model under a unique model_id')}` +
        `${btn('data-importmodel="1"', 'Import a Model', 'a model another study released (Keras HDF5): added to import_config.yaml, then make import tests it through TraceFlow\'s own test, without training (params.yaml is not changed)')}</div>` +
        askHtml('model', file) + (node.items.map((it, i) => itemHtml(it, i, file, ctx)).join('') || '<p class="note">none yet</p>');
    }
    if (isParams(file) && node.k === 'list' && JSON.stringify(node.path) === '["data"]' && MENUS && MENUS.params && MENUS.params.data_template) {
      return `<div class="buttons">${btn(`data-adddataset="1" ${pathAttr(node.path)}`, 'Add a Dataset',
        'a new data entry under the name you give, written at once with the blueprint\'s keys to fill: type, database, ranges, ' +
        'desync, normalisation (until its type and database are filled, a run of the project stops at the config check)')}</div>` +
        askHtml('dataset', file) + (node.items.map((it, i) => itemHtml(it, i, file, ctx)).join('') || '<p class="note">none yet</p>');
    }
    if (graphList(file, node)) return addButton(node.path[0]) + (node.items.map((it, i) => itemHtml(it, i, file, ctx)).join('') || '<p class="note">none yet</p>');
    if (node.k === 'list') return node.items.map((it, i) => itemHtml(it, i, file, ctx)).join('') || '<p class="note">[]</p>';
    return `<div class="field">${scalarHtml(node)}</div>`;
  }

  // Menus (choices.py): a key of a known set is a drop-down; a set the code extends (a torch class,
  // a font) suggests and still takes a typed name; any true/false value offers true / false. A value
  // the file holds that is not an option stays shown, marked, never changed by the menu.
  const { unquote } = window.TFMenus;
  const menuOf = path => plotTypeMenu(path) || monitorMenu(path) ||
    window.TFMenus.options(MENUS, curFile, curProject, trees[curFile] && trees[curFile].tree, path, n => shown(n.path, n.text).text);   // a menu following a key follows its staged value

  /** params.yaml trainer.monitor_metric: none, the monitors of the file's metrics_to_log, validation_objective
   *  (utils/monitor_metric.monitorable_metrics, read by early stopping and validation checkpointing). */
  const isMonitorPath = path => path[0] === 'trainer' && ((path.length === 2 && path[1] === 'monitor_metric') ||
    (path.length === 3 && path[1] === 'early_stopping' && path[2] === 'monitor_metric'));   // the block (2026-10-22), or the flat key of an older file
  function monitorMenu(path) {
    const pf = MENUS && MENUS.params;
    if (!pf || !isParams(curFile) || !isMonitorPath(path)) return null;
    const list = resolve(['metrics_to_log']).node;
    const items = list && list.k === 'list' ? list.items.filter(it => it.k === 'scalar' && !removed(it.path)).map(it => unquote(shown(it.path, it.text).text)) : [];
    items.push(...Object.values(stagedOf(curFile)).filter(e => e.op === 'add' && JSON.stringify(e.path) === '["metrics_to_log"]').map(e => e.value));
    const names = items.flatMap(n => pf.monitor.by_metric[n] || []);
    return { opts: [...new Set(['none', ...names, ...pf.monitor.always])], free: false };
  }

  /** The framework params.yaml trains with (deep_learning.framework, as staged), and its block's key. */
  function frameworkBlock() {
    const pf = MENUS && MENUS.params;
    if (!pf) return null;
    const v = valueAt(['deep_learning', 'framework']).text;
    return pf.frameworks.blocks[v || pf.frameworks.default] || null;
  }

  // An address (a file or a folder): its box, then a folder button that picks it in the system. The
  // database of a data entry starts at services/side_channel_core/data and is written from the TraceFlow
  // root, as the file writes it; any other address is written from the project's folder.
  const ADDRESS = /^(database|path|root|file|dir|folder)$|_(path|dir|file|root|folder)$/;
  function addressOf(file, path, key) {
    if (!isParams(file) && !/\/(test_config|model_judge|model_filter|auto_search)\.yaml$/.test(file)) return null;
    if (isParams(file) && path[0] === 'data' && key === 'database') return { base: 'root', start: 'services/side_channel_core/data', kind: 'file' };
    if ((isParams(file) && path[0] === 'paths' && path.length === 2) || ADDRESS.test(String(key))) return { base: 'project', start: '', kind: 'any' };
    return null;
  }
  const FOLDER_ICON = '<svg class="folder" width="14" height="12" viewBox="0 0 14 12"><path d="M1 2.5V10.5h12V3.5H6.5L5 2H1.5z" fill="none" stroke="currentColor" stroke-width="1.1"/></svg>';
  const listId = path => `dl-${JSON.stringify(path).replace(/[^A-Za-z0-9_-]/g, '_')}`;
  function menuControl(node, s, size) {
    const menu = menuOf(node.path) || (node.type === 'bool' || (node.type === 'null' && typeof bpAt(curFile, node.path) === 'boolean')
      ? { opts: ['true', 'false'], free: false } : null);
    if (!menu || !menu.opts.length) return null;
    const raw = unquote(s.text);
    const cur = node.type === 'bool' && /^(true|false)$/i.test(raw) ? raw.toLowerCase() : raw;   // YAML's True is true
    const width = size ? ` size="${size}"` : '';
    if (menu.free) {
      return `<input class="val${s.cls}" data-op="set" ${pathAttr(node.path)} value="${esc(s.text)}" list="${listId(node.path)}"${width}>` +
        `<datalist id="${listId(node.path)}">${menu.opts.map(o => `<option value="${esc(o)}">`).join('')}</datalist>`;
    }
    const known = cur === '' || menu.opts.includes(cur);
    // another framework's options (menus.js others): in the menu, never choosable, saying where they are
    const others = menu.others || {};
    // the phrase says why (another framework's: "available in pytorch"; a byte no entry defines: "not defined for ...")
    const elsewhere = why => (/^available in /.test(why) ? `${why}, not in the framework this run uses (deep_learning.framework)` : why);
    const options = (cur === '' ? ['<option value="" selected>(empty)</option>'] : [])
      .concat(known ? [] : [others[cur]
        ? `<option value="${esc(cur)}" selected title="${esc(elsewhere(others[cur]))}">${esc(cur)} (${esc(others[cur])})</option>`
        : `<option value="${esc(cur)}" selected>${esc(cur)} (not an option)</option>`])
      .concat(menu.opts.map(o => `<option value="${esc(o)}"${o === cur ? ' selected' : ''}>${esc((menu.labels && menu.labels[o]) || o)}</option>`))
      .concat(Object.keys(others).filter(o => o !== cur).map(o =>
        `<option value="${esc(o)}" disabled title="${esc(elsewhere(others[o]))}">${esc(o)} (${esc(others[o])})</option>`));
    return `<select class="val${s.cls}${known ? '' : ' bad'}" data-op="set" ${pathAttr(node.path)}` +
      ` title="${known ? 'one of the options the system defines' : 'not an option the system defines: pick one'}">${options.join('')}</select>`;
  }
  /** "+ add" of a value list: the options not in it yet, when its items take a menu. */
  // params.yaml: a list adds with a [+] button; it opens one control, a menu of the options left or a
  // box, whose value is the new entry. A one-choice list (sca.candidate_ranking.mode) takes no other.
  const ONE_CHOICE = [['sca', 'candidate_ranking', 'mode']];
  const oneChoice = listPath => isParams(curFile) && ONE_CHOICE.some(p => JSON.stringify(p) === JSON.stringify(listPath));
  const closeAdd = () => (isParams(curFile) ? btn('class="x" data-addclose="1"', '×', 'no new value') : '');
  /** sca.attack_byte needs choosing only when some data entry allows several bytes (CHES_CTF_2018, raw traces);
   *  otherwise a run takes its entry's byte (owner 2026-10-23: the key stays, its chooser goes). */
  function byteChoiceNeeded() {
    const by = ((MENUS && MENUS.refs && MENUS.refs[curProject]) || {}).attack_bytes_by_entry;
    if (!by || !Object.keys(by).length) return true;      // no entries read: the chooser stays
    return Object.values(by).some(bytes => Array.isArray(bytes) && bytes.length > 1);
  }
  function addControl(listPath, have, wide) {
    if (isParams(curFile) && JSON.stringify(listPath) === '["sca","attack_byte"]' && !byteChoiceNeeded()) return '';
    if (oneChoice(listPath) && (have.length || Object.values(stagedOf(curFile)).some(e => e.op === 'add' && JSON.stringify(e.path) === JSON.stringify(listPath)))) return '';
    const key = JSON.stringify(listPath);
    if (isParams(curFile) && !((ui.adding || {})[key])) return btn(`class="addopen" data-addopen="${esc(key)}"`, '+', 'add a value');
    // the opened [+] (owner 2026-10-27): its box and × in one roomy frame, not a 5em box squeezed among the values
    return isParams(curFile) ? `<span class="addbox">${addBox(listPath, have, wide)}${closeAdd()}</span>` : addBox(listPath, have, wide);
  }
  function addBox(listPath, have, wide) {
    const menu = menuOf([...listPath, 0]);
    const cls = `add${wide ? ' wide' : ''}`;
    const plus = isParams(curFile) ? 'choose…' : listPath.length === 3 && listPath[0] === 'jobs' && listPath[2] === 'processes' ? '+' : '+ add';
    if (menu && menu.opts.length && !menu.free) {
      const left = menu.opts.filter(o => !have.includes(o));
      return `<select class="${cls}" data-op="add" ${pathAttr(listPath)} title="add a process"><option value="">${plus}</option>` +
        `${left.map(o => `<option value="${esc(o)}">${esc((menu.labels && menu.labels[o]) || o)}</option>`).join('')}</select>`;
    }
    if (menu && menu.opts.length) {
      return `<input class="${cls}" data-op="add" ${pathAttr(listPath)} placeholder="${plus}" list="${listId([...listPath, 'add'])}">` +
        `<datalist id="${listId([...listPath, 'add'])}">${menu.opts.map(o => `<option value="${esc(o)}">`).join('')}</datalist>`;
    }
    return `<input class="${cls}" data-op="add" ${pathAttr(listPath)} placeholder="${isParams(curFile) ? 'the new value' : plus}">`;
  }

  // experiment_queue.yaml: a queue's lines are `make orchestrate ENV= PROJECT= DATASET= MODEL_ID=`; the
  // words are the panel's, the four values menus (ENV; master.yaml's projects; the project's datasets and
  // model ids). A line of any other shape stays one text box. Each menu stages the whole line.
  const isQueue = file => /\/experiment_queue\.ya?ml$/.test(file);
  const QVARS = ['ENV', 'PROJECT', 'DATASET', 'MODEL_ID'];
  function parseQueueLine(text) {
    const m = String(text).trim().match(/^make\s+orchestrate(?:\s+(.*))?$/);
    if (!m) return null;
    const vars = {};
    for (const tok of (m[1] || '').trim().split(/\s+/).filter(Boolean)) {
      const kv = tok.match(/^([A-Z_]+)=(\S*)$/);
      if (!kv || !QVARS.includes(kv[1]) || kv[1] in vars) return null;
      vars[kv[1]] = kv[2];
    }
    return vars;
  }
  const queueLine = vars => ['make orchestrate', ...QVARS.filter(k => vars[k]).map(k => `${k}=${vars[k]}`)].join(' ');
  function queueOptions(k, vars) {
    const projects = (last && last.projects) || [];
    const p = projects.find(x => x.name === (vars.PROJECT || curProject)) || projects.find(x => x.name === curProject) || {};
    return { ENV: ['dev', 'staging', 'prod'], PROJECT: projects.map(x => x.name), DATASET: p.datasets || [], MODEL_ID: p.modelIds || [] }[k];
  }
  function queueVarHtml(k, vars) {
    const cur = vars[k] || '';
    const opts = queueOptions(k, vars);
    const known = cur === '' || opts.includes(cur);
    const options = [`<option value=""${cur === '' ? ' selected' : ''}>(none)</option>`]
      .concat(known ? [] : [`<option value="${esc(cur)}" selected>${esc(cur)} (not an option)</option>`])
      .concat(opts.map(o => `<option value="${esc(o)}"${o === cur ? ' selected' : ''}>${esc(o)}</option>`));
    return `<span class="qkv">${k}=<select class="val${known ? '' : ' bad'}" data-qvar="${k}">${options.join('')}</select></span>`;
  }
  /** A queue's lines, every project alike (the DLSCA_ASCAD look, owner 2026-10-22): each a row of menus
   *  (ENV, PROJECT, DATASET, MODEL_ID) with × at its end; a line staged for adding is such a row too (its
   *  menus change the addition, × drops it); an empty queue shows one row to fill, which stages the line at
   *  its first choice; under the rows, + for one more (a copy of the last). */
  function queueLinesHtml(node) {
    const rows = node.items.map(it => {
      const s = shown(it.path, it.text);
      const vars = parseQueueLine(unquote(s.text));
      const x = btn(`class="x" data-op="remove" ${pathAttr(it.path)}`, '×', removed(it.path) ? 'keep' : 'remove this line');
      if (!vars) return `<div class="field${removed(it.path) ? ' removed' : ''}">${scalarHtml(it)}${x}</div>`;
      return `<div class="field qline${removed(it.path) ? ' removed' : ''}${s.cls}" ${pathAttr(it.path)}><span class="qfix">make orchestrate</span>` +
        `${QVARS.map(k => queueVarHtml(k, vars)).join('')}${x}</div>`;
    });
    const adds = Object.entries(stagedOf(curFile)).filter(([, e]) => e.op === 'add' && JSON.stringify(e.path) === JSON.stringify(node.path));
    const staged = adds.map(([id, e]) => {
      const vars = parseQueueLine(unquote(e.value));
      const x = btn(`class="x" data-unstage="${esc(id)}"`, '×', 'drop this line');
      if (!vars) return `<div class="field"><span class="item added wide">${esc(e.value)}</span>${x}</div>`;
      return `<div class="field qline dirty" data-staged="${esc(id)}" ${pathAttr(node.path)}><span class="qfix">make orchestrate</span>` +
        `${QVARS.map(k => queueVarHtml(k, vars)).join('')}${x}</div>`;
    });
    const projectEnv = (((last && last.projects) || []).find(x => x.name === curProject) || {}).env || 'dev';
    // an empty queue: a row to fill (nothing staged until a menu is chosen); × at its end, as on every line,
    // folds it away (the queue stays empty; + brings a line back)
    const qkey = JSON.stringify(node.path);
    const draft = !node.items.length && !adds.length && !(ui.noDraft || {})[qkey]
      ? `<div class="field qline draft" data-draftline="1" ${pathAttr(node.path)} title="the queue's first line: choose its values (OK writes it)"><span class="qfix">make orchestrate</span>` +
        `${QVARS.map(k => queueVarHtml(k, { ENV: projectEnv, PROJECT: curProject })).join('')}` +
        `${btn(`class="x" data-nodraft="${esc(qkey)}"`, '×', 'no line: the queue stays empty (+ adds one)')}</div>` : '';
    const lastLine = adds.length ? unquote(adds[adds.length - 1][1].value)
      : node.items.length ? unquote(shown(node.items[node.items.length - 1].path, node.items[node.items.length - 1].text).text)
      : queueLine({ ENV: projectEnv, PROJECT: curProject });
    return rows.join('') + staged.join('') + draft + `<div class="field qadd-row">` +
      `${btn(`class="qadd" data-qadd="1" ${pathAttr(node.path)} data-value="${esc(lastLine)}"`, '+', 'add a line: a copy of the last one, to change')}</div>`;
  }

  function scalarHtml(node) {
    if (node.multiline) return '<span class="note">multi-line text: edit it in the file</span>';
    const s = shown(node.path, node.text);
    const menu = menuControl(node, s);
    if (menu) return menu;
    const d = node.type === 'null' ? bpAt(curFile, node.path) : undefined;
    const suggest = d != null && typeof d !== 'object' ? `empty; the blueprint: ${d}` : 'empty';
    return `<input class="val${s.cls}" data-op="set" ${pathAttr(node.path)} value="${esc(s.text)}"` +
      ` placeholder="${node.type === 'null' ? esc(suggest) : ''}" title="${esc(node.type)}">`;
  }

  // params.yaml's lists are sweep values (every one a run): a value × removes leaves the form at once (OK
  // writes it, Cancel brings it back), and the last value has no ×: a list keeps one value at least.
  const sweepList = () => isParams(curFile);
  const kept = node => node.items.filter(it => !removed(it.path)).length +
    Object.values(stagedOf(curFile)).filter(e => e.op === 'add' && JSON.stringify(e.path) === JSON.stringify(node.path)).length;
  const fixedByteList = node => isParams(curFile) && JSON.stringify(node.path) === '["sca","attack_byte"]' && !byteChoiceNeeded();
  /** The one byte every data entry defines (as a string), or null when they differ or none is read. */
  function commonByte() {
    const by = ((MENUS && MENUS.refs && MENUS.refs[curProject]) || {}).attack_bytes_by_entry || {};
    const bytes = [...new Set(Object.values(by).map(b => String(b[0])))];
    return Object.keys(by).length && bytes.length === 1 ? bytes[0] : null;
  }
  /** A set attack byte that is the entries' own byte anyway: not drawn (the run takes it from data.type regardless). */
  const redundantByte = (node, it) => fixedByteList(node) && commonByte() != null && unquote(shown(it.path, it.text).text) === commonByte() && !removed(it.path);
  const itemX = (node, it) => ((sweepList() && kept(node) <= 1 && !removed(it.path)) || fixedByteList(node) ? ''   // a set byte: words only, no ×
    : btn(`class="x" data-op="remove" ${pathAttr(it.path)}`, '×', removed(it.path) ? 'keep' : 'remove'));
  const hidden = it => sweepList() && removed(it.path);
  function valuesHtml(node) {
    // sca.attack_byte where every entry defines one byte: a value the file holds reads as text (its × clears it: the
    // run then takes the entry's byte); no menu, since nothing is chosen here
    const fixedByte = fixedByteList(node);
    const items = node.items.filter(it => !hidden(it) && !redundantByte(node, it)).map(it => {
      const s = shown(it.path, it.text);
      const size = Math.max(2, String(s.text).length);
      const control = (fixedByte ? `<span class="readonly${s.cls}" title="as the file sets it; from data.type at the time of the run">${esc(unquote(s.text))}</span>` : null) ||
        menuControl(it, s, size) ||
        `<input class="val${s.cls}" data-op="set" ${pathAttr(it.path)} value="${esc(s.text)}" size="${size}">`;
      return `<span class="item${removed(it.path) ? ' removed' : ''}">${control}${itemX(node, it)}</span>`;
    }).join('');
    return `<span class="values">${items}${addedHtml(node.path, false)}${addControl(node.path, node.items.map(it => unquote(it.text)), false)}</span>`;
  }

  // Staged changes of the form being drawn: an input shows its staged value, × marks a removal, + adds.
  const editId = (op, path, key) => `${op}|${JSON.stringify(path)}${key != null ? `|${key}` : ''}`;
  const stagedOf = file => (pending[file] ? pending[file].edits : {});
  const within = (path, prefix) => path.length >= prefix.length && prefix.every((x, i) => path[i] === x);
  function shown(path, text) {
    const e = stagedOf(curFile)[editId('set', path)];
    return e ? { text: e.value, cls: ' dirty' } : { text, cls: '' };
  }
  const removed = path => !!(stagedOf(curFile)[editId('remove', path)] || stagedOf(curFile)[editId('delete', path)]);
  const addedHtml = (path, wide) => Object.entries(stagedOf(curFile))
    .filter(([, e]) => e.op === 'add' && JSON.stringify(e.path) === JSON.stringify(path))
    .map(([id, e]) => `<span class="item added${wide ? ' wide' : ''}">${esc(addedText(e.value))}${btn(`class="x" data-unstage="${esc(id)}"`, '×', 'undo this addition')}</span>`).join('');
  function addedText(value) {                             // {"top_k_accuracy": {"k": 7}} reads top_k_accuracy k=7
    try {
      const o = JSON.parse(value);
      const keys = o && typeof o === 'object' && !Array.isArray(o) ? Object.keys(o) : [];
      if (keys.length === 1 && o[keys[0]] && typeof o[keys[0]] === 'object') {
        return `${keys[0]} ${Object.entries(o[keys[0]]).map(([k, v]) => `${k}=${v}`).join(' ')}`;
      }
    } catch (e) { /* plain text */ }
    return value;
  }

  // The panel edits values only; it never copies or deletes an entry (do that by hand in the file).
  // OK writes the block's staged changes and appears only while the block has some.
  function blockButtons(path) {
    const dirty = Object.values(stagedOf(curFile)).some(e => within(e.path, path));
    return `<div class="buttons">${dirty ? btn(`data-apply="1" ${pathAttr(path)}`, 'OK', 'write the changes in this block to the file') : ''}</div>`;
  }

  /** Whether any trainer.optimizer entry (a sweep list or one block) schedules with OneCycleLR, as staged. */
  function oneCycleChosen() {
    const opt = resolve(['trainer', 'optimizer']).node;
    const paths = !opt ? [] : opt.k === 'list'
      ? opt.items.filter(it => !removed(it.path)).map(it => [...it.path, 'scheduler', 'type'])
      : [['trainer', 'optimizer', 'scheduler', 'type']];
    return paths.some(p => {
      const n = resolve(p).node;
      return !!n && n.k === 'scalar' && unquote(shown(n.path, n.text).text) === 'OneCycleLR';
    });
  }
  const ONE_CYCLE_BEST_HINT = 'best_validation with OneCycleLR is allowed, your choice. The best-validation epoch is usually '
    + 'mid-cycle, its learning rate not yet annealed to the floor the cycle ends at, and its validation values are '
    + 'selection-biased: the best over the epochs on the set that chose it (the judge labels them so). The test is the '
    + 'estimate of the tested model\'s performance; final_epoch tests the model of the completed cycle.';

  /** params.yaml device (utils/device.py): a menu of cuda, cpu and cuda:N per GPU nvidia-smi reports, with helper
   *  text under it saying what the chosen one does. It joins the run hash as written (utils/model_naming.py). */
  function deviceHelp(cur, gpus) {
    const gpu = (/^cuda:(\d+)$/.exec(cur) || [])[1];
    const g = gpu != null ? gpus.find(x => String(x.index) === gpu) : null;
    const what = cur === 'cuda' ? 'Trains on the GPU: cuda is PyTorch\'s name for an NVIDIA GPU (it has no device called gpu); a run fails (GPU FAILED) when it is not usable: never a CPU fallback.'
      : cur === 'cpu' ? 'Trains on the CPU on purpose: no GPU needed; slower, and CPU kernels give slightly different numbers.'
        : gpu != null ? `Trains on GPU ${gpu}${g ? ` (${g.name})` : ' (not reported by nvidia-smi here)'} only; a run fails when it is not usable.` +
          (gpus.length === 1 && g ? ' This machine\'s only GPU: the same as cuda, but a different run identity.' : '')
          : 'Not a device TraceFlow offers here.';
    return `${what} Part of the run hash: another value makes new runs.`;
  }
  function deviceHtml(label, node) {
    const s = shown(node.path, node.text);
    const cur = unquote(s.text);
    const gpus = ((last && last.gpu && last.gpu.gpus) || []);
    const opts = ['cuda', 'cpu', ...gpus.map(g => `cuda:${g.index}`)];
    if (cur && !opts.includes(cur)) opts.push(cur);
    return `<div class="field help-field">${label}<select class="val${s.cls}" data-op="set" ${pathAttr(node.path)} title="the device a run trains on (utils/device.py)">` +
      `${opts.map(o => `<option value="${esc(o)}"${o === cur ? ' selected' : ''}>${esc(o)}</option>`).join('')}</select>` +
      `<small class="note help">${esc(deviceHelp(cur, gpus))}</small></div>`;
  }

  // params.yaml: leakage_model takes only what the system defines; a data range is one [lower, upper] pair.
  const isParams = file => /\/params\.yaml$/.test(file);
  const LEAKAGE = ['ID', 'HW', 'MSB'];    // services/side_channel_core/src/sca/leakage_models.py (case-insensitive there too)
  const leakageOk = v => LEAKAGE.includes(String(v).trim().toUpperCase());
  const notDefined = v => `${v} not defined in system (${LEAKAGE.join(', ')})`;

  function leakageHtml(label, node) {
    const one = it => { const s = shown(it.path, it.text); return `<input class="val${s.text !== '' && !leakageOk(s.text) ? ' bad' : ''}${s.cls}" data-op="set" data-check="leakage"` +
      ` ${pathAttr(it.path)} value="${esc(s.text)}" size="${Math.max(3, String(s.text).length)}">`; };
    let inputs;
    if (node.k === 'scalar') inputs = one(node);
    else if (node.k === 'list' && node.scalars) {
      inputs = `<span class="values">${node.items.filter(it => !hidden(it)).map(it => `<span class="item${removed(it.path) ? ' removed' : ''}">${one(it)}` +
        `${itemX(node, it)}</span>`).join('')}` +
        `${addedHtml(node.path, false)}${(ui.adding || {})[JSON.stringify(node.path)]
          ? `<span class="addbox"><input class="add" data-op="add" data-check="leakage" ${pathAttr(node.path)} placeholder="the new value">${closeAdd()}</span>`
          : btn(`class="addopen" data-addopen="${esc(JSON.stringify(node.path))}"`, '+', 'add a value')}</span>`;
    } else return null;
    const vals = node.k === 'scalar' ? [node] : node.items;
    const errs = vals.filter(it => it.text !== '' && !leakageOk(it.text)).map(it => `<p class="err">${esc(notDefined(it.text))}</p>`).join('');
    return `<div class="field">${label}${inputs}</div>${errs}`;
  }

  function rangeHtml(label, node) {
    const items = node.k === 'list' ? node.items : [];
    const pair = items.length === 2 && items.every(it => it.k === 'scalar');
    if (!pair && !(node.k === 'list' && node.flow) && !(node.k === 'scalar' && node.type === 'null')) return null;
    const whole = stagedOf(curFile)[editId('set', node.path)];      // a staged [lower, upper] for a range not yet a pair
    const parts = whole ? String(whole.value).replace(/^\[|\]$/g, '').split(',').map(x => x.trim()) : null;
    const box = (i, ph) => {
      if (pair) {
        const s = shown(items[i].path, items[i].text);
        return `<input class="val${s.cls}" data-op="set" ${pathAttr(items[i].path)} value="${esc(s.text)}" placeholder="${ph}" size="7">`;
      }
      const v = parts ? parts[i] : ((items[i] || {}).text || '');
      return `<input class="val${parts ? ' dirty' : ''}" data-op="range" data-slot="${i}" ${pathAttr(node.path)} value="${esc(v)}" placeholder="${ph}" size="7">`;
    };
    return `<div class="field">${label}<span class="values">${box(0, 'lower')}${box(1, 'upper')}</span>` +
      `${pair ? '' : '<small class="note">fill both to save</small>'}</div>`;
  }

  /** A job graph_export.py made from finished runs (layout.py reads its header). */
  function exportedJob(job) {
    const id = Number(unquote((entryOf(job, 'id') || {}).text));
    const p = ((last && last.projects) || []).find(x => x.name === curProject);
    return !!(p && (p.graphJobs || []).some(j => j.id === id && j.exported));
  }

  // ── FIND MODEL (model_filter.yaml): every field the runs hold, ticked to make a condition ────────────
  // model_filter.py --fields lists them (where the filter finds each, its kind, its values); a ticked row
  // is a condition: an operator that fits the field, a value from the runs' (or typed). Find Models
  // writes the ticked conditions to model_filter.where (model_filter.py --write-where), then make find_model.
  const isFilter = file => /\/model_filter\.ya?ml$/.test(file);
  const filterData = {};                 // project -> model_filter.py --fields
  const filterDraft = {};                // file -> [{path, op, value, quantifier}]: the ticked conditions
  const askedFields = new Set();
  const MF_SOURCES = [['run', 'Run', 'the run_map entry: env, status, model_id, trial, session'],
    ['registry', 'Registry', 'run_registry.json: batches, batch ids, start and end, status reason'],
    ['sweep', 'Sweep values', 'the sweep point: trainer.*, sca.*, model.*'],
    ['snapshot', 'Snapshot', 'the training snapshot: data.*, cross_validation.*, perform_attack.*, …'],
    ['device', 'Device', 'device.yaml: the GPU and the software the run trained with']];
  const MF_OPS = { equals: 'equals: the value', not_equals: 'not_equals: any other value', in: 'in: one of the values ticked',
    not_in: 'not_in: none of the values ticked', gt: 'gt: greater than', gte: 'gte: at least', lt: 'lt: less than', lte: 'lte: at most',
    between: 'between: a number in [lower, upper]', matches: 'matches: a regular expression', contains: 'contains: a list member, or a substring',
    exists: 'exists: the run has it (True) or not (False)', count: 'count: the list has N entries' };
  const MF_BY_KIND = { list: ['contains', 'count', 'exists'], bool: ['equals', 'not_equals', 'exists'],
    number: ['equals', 'not_equals', 'in', 'not_in', 'gt', 'gte', 'lt', 'lte', 'between', 'exists'],
    text: ['equals', 'not_equals', 'in', 'not_in', 'matches', 'contains', 'exists'] };
  const mfShow = v => (typeof v === 'string' ? v : JSON.stringify(v));
  const mfTyped = text => { const v = String(text).trim(); return v !== '' && Number.isFinite(Number(v)) ? Number(v) : v; };
  function mfFromRaw(raw) {
    if (!raw || typeof raw !== 'object' || !raw.path) return null;
    const op = Object.keys(raw).find(k => k in MF_OPS);
    return op ? { path: String(raw.path), op, value: raw[op], quantifier: raw.quantifier } : null;
  }
  const mfToRaw = c => ({ path: c.path, [c.op]: c.value, ...(c.quantifier && c.quantifier !== 'any' ? { quantifier: c.quantifier } : {}) });
  function mfDefault(f, op) {
    const v = (f && f.values) || [];
    if (op === 'exists') return true;
    if (op === 'in' || op === 'not_in') return v.slice(0, 1);
    if (op === 'between') return v.length ? [v[0], v[v.length - 1]] : [0, 1];
    if (op === 'count') return 1;
    if (op === 'matches') return v.length ? String(v[0]) : '.*';
    return v.length ? v[0] : '';
  }
  function mfValueHtml(f, c, i) {
    const vals = (f && f.values) || [];
    const at = `data-mfi="${i}"`;
    if (c.op === 'exists') {
      return `<select class="val" data-mfbool="1" ${at}><option value="true"${c.value !== false ? ' selected' : ''}>True</option><option value="false"${c.value === false ? ' selected' : ''}>False</option></select>`;
    }
    if (c.op === 'in' || c.op === 'not_in') {
      const have = Array.isArray(c.value) ? c.value : [c.value];
      const all = [...vals, ...have.filter(v => !vals.some(w => mfShow(w) === mfShow(v)))];
      return `<span class="fmtboxes">${all.map((v, k) => `<label class="fmtbox"><input type="checkbox" data-mfin="${k}" ${at}` +
        `${have.some(w => mfShow(w) === mfShow(v)) ? ' checked' : ''}> ${esc(mfShow(v))}</label>`).join('')}</span>`;
    }
    if (c.op === 'between') {
      const [lo, hi] = Array.isArray(c.value) ? c.value : ['', ''];
      return `<input class="val" data-mftext="0" ${at} value="${esc(mfShow(lo))}" placeholder="lower" size="8">` +
        `<input class="val" data-mftext="1" ${at} value="${esc(mfShow(hi))}" placeholder="upper" size="8">`;
    }
    if (['equals', 'not_equals', 'contains'].includes(c.op) && vals.length) {
      const known = vals.some(v => mfShow(v) === mfShow(c.value));
      return `<select class="val" data-mfsel="1" ${at}>${known ? '' : `<option value="-1" selected>${esc(mfShow(c.value))}</option>`}` +
        vals.map((v, k) => `<option value="${k}"${mfShow(v) === mfShow(c.value) ? ' selected' : ''}>${esc(mfShow(v))}</option>`).join('') + '</select>';
    }
    const list = vals.length ? ` list="mf-${i}"` : '';
    return `<input class="val" data-mftext="v" ${at} value="${esc(mfShow(c.value))}"${list} placeholder="${c.op === 'count' ? 'N' : 'value'}">` +
      (vals.length ? `<datalist id="mf-${i}">${vals.map(v => `<option value="${esc(mfShow(v))}">`).join('')}</datalist>` : '');
  }
  function mfRowHtml(f, draft) {
    const i = draft.findIndex(c => c.path === f.path);
    const c = draft[i];
    const box = `<input type="checkbox" data-mfcheck="${esc(f.path)}"${c ? ' checked' : ''}>`;
    const name = `<span class="readonly mf-path" title="${esc(`${f.runs} run(s) have it${f.varies ? '; it differs between them' : ''}`)}">${esc(f.path)}</span>`;
    if (!c) {
      const vals = f.values.map(mfShow);
      const more = f.more ? `, … (${f.more} more)` : '';
      return `<div class="field mfrow${f.varies ? ' varies' : ''}">${box}${name}<small class="note mf-vals">${esc(vals.slice(0, 6).join(', ') + (vals.length > 6 ? ', …' : '') + more)}</small></div>`;
    }
    const ops = MF_BY_KIND[f.kind] || Object.keys(MF_OPS);
    const opSel = `<select class="val mf-op" data-mfop="${i}" title="${esc(MF_OPS[c.op] || '')}">${[...new Set([...ops, c.op])].map(o =>
      `<option value="${o}"${o === c.op ? ' selected' : ''} title="${esc(MF_OPS[o] || '')}">${o}</option>`).join('')}</select>`;
    const quant = f.path.includes('*') ? `<select class="val" data-mfquant="${i}" title="any: one value of the path satisfies it; all: every one">` +
      ['any', 'all'].map(q => `<option${(c.quantifier || 'any') === q ? ' selected' : ''}>${q}</option>`).join('') + '</select>' : '';
    return `<div class="field mfrow on">${box}${name}${opSel}${mfValueHtml(f, c, i)}${quant}</div>`;
  }
  function filterHtml(node) {
    const data = filterData[curProject];
    if (!askedFields.has(curProject)) { askedFields.add(curProject); send({ type: 'filter-fields', project: curProject }); }
    if (!data) return '<p class="note">reading the fields your runs hold…</p>';
    if (data.error) return `<p class="err">${esc(data.error)}</p>`;
    const draft = filterDraft[curFile] || (filterDraft[curFile] = (data.where || []).map(mfFromRaw).filter(Boolean));
    const changed = JSON.stringify(draft.map(mfToRaw)) !== JSON.stringify(data.where || []);
    const known = new Set(data.fields.map(f => f.path));
    const groups = MF_SOURCES.map(([src, title, what]) => {
      const fields = data.fields.filter(f => f.source === src).sort((a, b) => (b.varies - a.varies) || a.path.localeCompare(b.path));
      if (!fields.length) return '';
      const ticked = fields.filter(f => draft.some(c => c.path === f.path)).length;
      const k = `mf:${curFile}:${src}`;
      const open = isOpen(k, ['run', 'registry', 'sweep'].includes(src));
      return `<details class="node" data-key="${esc(k)}"${open ? ' open' : ''}><summary>${esc(title)} <small title="${esc(what)}">${fields.length} fields, ` +
        `${fields.filter(f => f.varies).length} differ between runs${ticked ? ` · ${ticked} ticked` : ''}</small></summary>` +
        (open ? fields.map(f => mfRowHtml(f, draft)).join('') : '') + '</details>';
    }).join('');
    const other = draft.filter(c => !known.has(c.path)).map(c => mfRowHtml({ path: c.path, kind: 'text', values: [], runs: 0, varies: false }, draft)).join('');
    return `<div class="field"><label>where</label><small class="note">tick what the models must have; ${data.runs} runs, ${data.fields.length} fields</small></div>` +
      `<div class="buttons">${btn(`data-findmodels="${esc(curProject)}" data-file="${esc(curFile)}"`, `Find Models (${draft.length} ticked)`,
        'write the ticked conditions to model_filter.where, then make find_model')}` +
      `${changed ? btn(`data-mfreset="${esc(curFile)}"`, 'Reset', 'the conditions as model_filter.yaml has them') : ''}</div>` +
      (other ? `<details class="node" open><summary>In the file, no run has it <small>kept while ticked</small></summary>${other}</details>` : '') + groups;
  }

  function entryHtml(key, node, file, parent, ctx = {}) {
    if (isParams(file) && key === 'metrics_to_log' && node.path.length === 1 && node.k === 'scalar' && node.type === 'null') {
      node = { k: 'list', path: node.path, flow: true, scalars: true, items: [] };   // empty: the metric menu and [+] (yamledit add makes the list)
    }
    const label = `<label title="${titled(file, node.path, node.path.join(' › '))}">${esc(key)}</label>`;
    if (isFilter(file) && node.path.length === 2 && node.path[0] === 'model_filter' && key === 'where') return filterHtml(node);
    if (isParams(file) && node.path.length === 2 && node.path[0] === 'deep_learning' && MENUS && MENUS.params &&
        Object.values(MENUS.params.frameworks.blocks).includes(key) && key !== frameworkBlock()) {
      return '';                                          // the other framework's settings: not read by this run
    }
    // keys that are no longer settings (2026-10-19): an older file still naming them shows them, read-only
    const retired = isParams(file) && node.k === 'scalar' && ((node.path.length === 2 && node.path[0] === 'sca' && key === 'rank_base')
      ? 'not a setting any more: TraceFlow ranks the keys from 1'
      : (node.path[0] === 'data' && key === 'multilabel') ? 'not a setting any more: single-label is the only mode' : null);
    if (retired) {
      return `<div class="field">${label}<span class="readonly" title="the system reads this value whether the key is there or not; it can be deleted">${esc(unquote(node.text))}</span>` +
        `<small class="note hint">${esc(retired)}</small></div>`;
    }
    if (isMasterFile(file) && node.k === 'scalar') {          // Master Control: read-only keys shown, hash keys marked
      const tier = window.TFMenus.masterTier(node.path);
      if (tier.tier === 'readonly') {
        return `<div class="field">${label}<span class="readonly" title="${esc(tier.why)}">${esc(unquote(node.text))}</span></div>`;
      }
      if (tier.tier === 'hash') {
        return `<div class="field">${label}${scalarHtml(node)}<small class="note hint" title="${esc(tier.why)}">run hash</small></div>`;
      }
    }
    if (isParams(file) && JSON.stringify(node.path) === JSON.stringify(TOP_K) && rankingMode() !== 'top_k') return '';   // read with mode top_k only
    if (isParams(file) && node.k === 'scalar' && node.path.length === 3 && node.path[0] === 'model' && key === 'architecture') return architectureHtml(label, node);
    if (isLayers(file, node.path) && node.k === 'list') {       // the layer builder (also for an empty `layers: []`)
      const n = node.items.length;
      return blockHtml(node.path, 'layers', `${n} layer${n === 1 ? '' : 's'}`, () => blockButtons(node.path) + layersHtml(node));
    }
    if (isParams(file) && JSON.stringify(node.path) === '["device"]' && node.k === 'scalar') return deviceHtml(label, node);
    if (isParams(file) && JSON.stringify(node.path) === '["seed"]' && node.k === 'scalar' && !node.multiline) {
      // a sweep key (sweep_engine: a seed list is one run per seed): [+] makes `seed: 42` a list [42, 43]
      return `<div class="field">${label}<span class="values"><span class="item">${scalarHtml(node)}</span>` +
        `${addedHtml(node.path, false)}${addControl(node.path, [unquote(node.text)], false)}</span></div>`;
    }
    if (isParams(file) && JSON.stringify(node.path) === '["evaluation","test","load_model"]' && node.k === 'scalar') {
      // best_validation needs evaluation.validation.checkpointing: true; without it the test loads final_epoch
      // (utils/checkpoint_policy fallback_reason). Read from the staged values: the warning goes once either is fixed
      const cp = resolve(['evaluation', 'validation', 'checkpointing']).node;
      const on = !!cp && cp.k === 'scalar' && unquote(shown(cp.path, cp.text).text).toLowerCase() === 'true';
      const best = unquote(shown(node.path, node.text).text) === 'best_validation';
      const warn = best && !on;
      // the owner's choice, with the hint (2026-10-22): OneCycleLR's best-validation epoch is usually mid-cycle
      const cycle = best && oneCycleChosen();
      return `<div class="field">${label}${scalarHtml(node)}` + (warn ? '<small class="note hint err" title="evaluation.validation.checkpointing is ' +
        'not true: no best-validation checkpoint is saved, so the test loads final_epoch">needs checkpointing: true</small>' : '') +
        (cycle ? `<small class="note hint" title="${esc(ONE_CYCLE_BEST_HINT)}">with OneCycleLR: see hint</small>` : '') + '</div>';
    }
    // data[].type: beside it, the byte(s) the type defines a target for (sca.attack_byte left empty takes it for a run of the entry)
    if (isParams(file) && node.k === 'scalar' && node.path.length === 3 && node.path[0] === 'data' && node.path[2] === 'type') {
      return `<div class="field">${label}${scalarHtml(node)}${typeByteHint(node)}</div>`;
    }
    if (isParams(file) && isMonitorPath(node.path) && node.k === 'scalar') {
      return `<div class="field">${label}${scalarHtml(node)}<small class="note hint" title="the metric early stopping watches, and the one best-validation checkpointing picks by (evaluation.validation.checkpointing); its direction follows the metric">early stopping and best-validation checkpoints</small></div>`;
    }
    // sca.key_enumeration.modes: only the chosen mode's block (sca.key_enumeration.mode, as staged); comparison: every one
    if (isParams(file) && node.path.length === 4 && node.path[0] === 'sca' && node.path[1] === 'key_enumeration' && node.path[2] === 'modes') {
      const chosen = unquote(valueAt(['sca', 'key_enumeration', 'mode']).text || '');
      if (chosen && chosen !== 'comparison' && chosen !== key) return '';
    }
    // data[].trace_normalization: the chosen mode's keys only (a key of another mode left in the file is not drawn)
    if (isNormPath(node.path) && key !== 'mode' && !normKeys(node.path.slice(0, 3)).includes(key)) return '';
    // data[].profiling_split: its arguments only while enabled is true (a value left in the file is not drawn)
    if (isParams(file) && isSplitArg(node.path) && !splitOn(node.path.slice(0, 3))) return '';
    if (isParams(file) && node.k === 'scalar' && node.path.length === 2 && node.path[0] === 'paths') {   // the run layout: edited in the file
      return `<div class="field">${label}<span class="readonly" title="paths is edited in params.yaml (Open in editor above), not here">${esc(unquote(node.text))}</span></div>`;
    }
    const where = node.k === 'scalar' ? addressOf(file, node.path, key) : null;
    if (where) {
      const s = shown(node.path, node.text);
      return `<div class="field">${label}<input class="val${s.cls}" data-op="set" ${pathAttr(node.path)} value="${esc(unquote(s.text))}">` +
        `<button class="pathpick" data-pickpath="${esc(JSON.stringify(node.path))}" data-base="${where.base}" data-start="${esc(where.start)}"` +
        ` data-kind="${where.kind}" data-cur="${esc(unquote(s.text))}" title="find it in the system">${FOLDER_ICON}</button></div>`;
    }
    if (isParams(file)) {
      const special = key === 'leakage_model' && !menuOf(node.k === 'list' ? [...node.path, 0] : node.path) ? leakageHtml(label, node)
        : /_range$/.test(key) && node.path[0] === 'data' ? rangeHtml(label, node) : null;
      if (special) return special;
    }
    if (isGraph(file) && node.path[0] === 'faceting' && node.path.length === 3 && key === 'arrangement' && node.k === 'map') {
      return arrangementHtml(label, node);
    }
    if (isGraph(file) && node.path[0] === 'faceting' && key === 'address' && node.k === 'scalar') {
      const s = shown(node.path, node.text);
      return `<div class="field">${label}<input class="val wide${s.cls}" data-op="set" ${pathAttr(node.path)} value="${esc(s.text)}"` +
        ` placeholder="jobs/job_NNN_<name>/…/figure.pdf (under graphs/)">` +
        `${btn(`data-findfig="${esc(JSON.stringify(node.path))}"`, 'Find Figure', 'pick a drawn figure under graphs/; its path is written here')}</div>`;
    }
    if (isGraph(file) && key === 'id' && node.path.length === 3 && ['jobs', 'faceting'].includes(node.path[0]) && node.k === 'scalar') {
      const s = shown(node.path, node.text);
      return `<div class="field">${label}<input class="val${s.cls}" data-op="set" ${pathAttr(node.path)} value="${esc(unquote(s.text))}" ` +
        `title="an integer, unique in ${node.path[0]}: make graph ${node.path[0] === 'jobs' ? 'GRAPH_ID' : 'FACE_ID'}=<id> draws it; Add a Job / Add a Facet give the next one"></div>`;
    }
    if (isGraph(file) && ctx.job && at(node.path, 'metric') && exportedJob(ctx.job)) {
      const names = node.k === 'list' ? texts(node).map(unquote) : [unquote(node.text)];
      return `<div class="field">${label}<span class="readonly" title="made by Export / Overlay Graph from the runs that recorded it">${esc(names.join(', '))}</span></div>`;
    }
    if (isGraph(file) && key === 'panels' && node.path.length === 3 && node.path[0] === 'faceting' && node.k === 'list') return panelsHtml(node);
    if (graphList(file, node) && node.flow) {                  // `faceting: []`: still a list to add to
      const k = nodeKey(file, node.path);
      const open = isOpen(k, false);
      return `<details class="node" data-key="${esc(k)}"${open ? ' open' : ''}><summary>${esc(key)} <small>0</small></summary>` +
        (open ? bodyHtml(node, file, ctx) : '') + '</details>';
    }
    const hooked = isGraph(file) && GF() && !ctx.plain ? graphEntryHtml(key, node, ctx) : null;
    if (hooked != null) return hooked;
    if (isGraph(file) && ctx.job) {
      if (at(node.path, 'top_models')) {                        // written by make judge: shown, not edited
        const names = texts(node);
        return `<div class="field">${label}<span class="readonly" title="rewritten by make judge">` +
          `${esc(names.join(', ') || 'none yet (make judge)')}</span></div>`;
      }
      if (at(node.path, 'legend', 'labels') && (node.k === 'map' || (node.k === 'scalar' && node.type === 'null'))) {
        return labelsHtml(key, node, ctx.job);
      }
      if (at(node.path, 'models') && node.k === 'scalar' && node.type === 'null') {   // empty: a list to add to
        return `<div class="field">${label}<input class="add wide" data-op="add" ${pathAttr(node.path)} placeholder="+ add a model name"></div>`;
      }
      if (at(node.path, 'models') && node.k === 'scalar') {     // one value where a list belongs
        return `<div class="field">${label}${scalarHtml(node)}</div>` +
          `<p class="err">models must be a list: write it as "- ${esc(node.text)}" in the file</p>`;
      }
    }
    if (node.k === 'scalar') return `<div class="field">${label}${scalarHtml(node)}</div>`;
    if (node.k === 'list' && node.scalars && node.flow && !blockList(file, node)) {
      const hint = seedHint(file, node);                             // under the value, aligned with it, wrapping
      return `<div class="field">${label}${valuesHtml(node)}${attackByteHint(file, node)}</div>${hint ? `<div class="field hintrow"><label></label>${hint}</div>` : ''}`;
    }
    const k = nodeKey(file, node.path);
    const open = isOpen(k, false);
    const count = node.k === 'map' ? node.entries.length : isParams(file) || isQueue(file) ? kept(node) : node.items.length;
    // a queue's line (experiment_queue.yaml): its key, then how many commands it runs
    let size = isQueue(file) && node.k === 'list' && node.path.length === 1 ? `${count} CMD` : count;
    // sca.key_enumeration (owner 2026-10-22): the enumeration works; its mode is still being developed
    if (isParams(file) && JSON.stringify(node.path) === '["sca","key_enumeration"]') size = `${count} · system works; mode needs development`;
    // A block's OK only for a block at a file's top level or right under a list item (a job's style,
    // legend, …), never deeper (style's font, line, …).
    const last = parent.path[parent.path.length - 1];
    const okLevel = parent.path.length === 0 || typeof last === 'number';
    const actions = isBlock(node) && !parent.flow && okLevel ? blockButtons(node.path) : '';
    const end = isQueue(file) && node.path.length === 1
      ? `<span class="end">${btn(`class="x" data-delentry="queue" data-name="${esc(key)}" ${pathAttr(node.path)}`, '×', `delete queue ${key} and its ${count} line${count === 1 ? '' : 's'} (asks first)`)}</span>` : '';
    return `<details class="node" data-key="${esc(k)}"${open ? ' open' : ''}><summary title="${titled(file, node.path, node.path.join(' › '))}">${esc(key)} <small>${size}</small>${end}</summary>` +
      (open ? `${actions}${bodyHtml(node, file, ctx)}` : '') + '</details>';
  }

  function itemHtml(item, i, file, ctx = {}) {
    const k = nodeKey(file, item.path);
    const open = isOpen(k, false);
    let label = '';
    if (item.k === 'map') {
      const e = LABEL_KEYS.map(n => item.entries.find(x => x.key === n && x.node.k === 'scalar')).find(Boolean);
      if (e) label = `${e.key}: ${e.node.text}`;
    } else if (item.k === 'scalar') label = item.text;
    // a job's or a facet's line (its summary): [Delete][Draw] at the right end; inside it only OK, once a value changed
    const section = isGraph(file) && item.k === 'map' && item.path.length === 2 ? item.path[0] : '';
    const idEntry = section ? item.entries.find(x => x.key === 'id' && x.node.k === 'scalar') : null;
    const id = idEntry ? esc(unquote(idEntry.node.text)) : '';
    let draw = '';
    let del = '';
    if (id && section === 'jobs') {
      draw = `<button class="jobdraw" data-run="draw" data-form-project="1" data-id="${id}" title="make graph GRAPH_ID=${id}">Draw</button>`;
      if (id !== '0') {
        del = `<button class="x danger jobdel" data-jobdel="${esc(curProject)}" data-id="${id}"` +
          ` title="delete graph job ${id} from graph_config.yaml (asks first): its figures move to graphs/jobs/_deleted/, later jobs move down one id">×</button>`;
      }
    } else if (id && section === 'faceting') {
      draw = `<button class="jobdraw" data-run="draw-facet" data-form-project="1" data-id="${id}" title="make graph FACE_ID=${id}">Draw</button>`;
      del = `<button class="x danger facetdel" data-facetdel="${esc(curProject)}" data-id="${id}"` +
        ` title="delete facet ${id} from graph_config.yaml (asks first): its output moves to graphs/facets/_deleted/, later facets move down one id">×</button>`;
    } else if (section) {                                 // no id yet (a template's entry): the entry itself goes
      const kind = section === 'jobs' ? 'job' : 'facet';
      del = btn(`class="x" data-delentry="${kind}" data-name="${esc(label)}" ${pathAttr(item.path)}`, '×', `delete this ${kind} from graph_config.yaml (asks first)`);
    }
    // params.yaml: a model's or a data entry's line: × at the right end while another one is left (asks first; refused while a run is ongoing)
    if (isParams(file) && item.k === 'map' && item.path.length === 2 && ['model', 'data'].includes(item.path[0])) {
      const list = resolve([item.path[0]]).node;
      const kind = item.path[0] === 'model' ? 'model' : 'data';
      const name = unquote((entryOf(item, kind === 'model' ? 'model_id' : 'name') || {}).text || '');
      if (list && list.k === 'list' && list.items.length > 1) {
        del = btn(`class="x" data-delentry="${kind}" data-name="${esc(name)}" ${pathAttr(item.path)}`, '×',
          `delete ${kind === 'model' ? `model ${name}` : `data entry ${name}`} from params.yaml (asks first, naming the queue lines that use it)`);
      }
      if (kind === 'model') {                               // [Visualise]: the model on the planner page (a layer list only)
        const arch = unquote((entryOf(item, 'architecture') || {}).text || '');
        const named = MF() && MF().named.includes(arch);
        const argsNode = entryOf(item, 'architecture_args');
        const layered = argsNode && argsNode.k === 'map' && entryOf(argsNode, 'layers');
        del = (named ? btn(`data-visualise="${esc(name)}" disabled title="${esc(name)} is a named network (${esc(arch)}): no layer list to draw"`, 'Visualise')
          : !layered ? btn(`data-visualise="${esc(name)}" disabled title="${esc(name)} has no layer list (architecture_args.layers): nothing to draw"`, 'Visualise')
          : btn(`data-visualise="${esc(name)}"`, 'Visualise', `open ${name} on the planner page: its changes are staged here, Add the Model adds a new one`)) + del;
      }
    }
    // master.yaml: a project's line: × removes its entry (asks first; its folder and runs stay); none on default_project
    if (isMasterFile(file) && item.k === 'map' && item.path.length === 2 && item.path[0] === 'project') {
      const name = unquote((entryOf(item, 'name') || {}).text || '');
      const dflt = unquote(((resolve(['default_project']).node) || {}).text || '');
      const locked = last && last.master && last.master.lock ? ' disabled' : '';
      if (name && name !== dflt) {
        del = btn(`class="x" data-removeproject="${esc(name)}" ${pathAttr(item.path)}${locked}`, '×',
          `remove project ${name} from master.yaml (asks first; projects/${name}/ and its runs stay)`);
      }
    }
    const buttons = blockButtons(item.path);
    const inner = section === 'jobs' ? { ...ctx, job: item } : ctx;
    const end = draw || del ? `<span class="end">${del}${draw}</span>` : '';
    return `<details class="node" data-key="${esc(k)}"${open ? ' open' : ''}><summary title="${titled(file, item.path.slice(0, -1), `${item.path.slice(0, -1).join(' › ')} #${i}`)}">#${i} <small>${esc(label)}</small>${end}</summary>` +
      (open ? `${buttons}${bodyHtml(item, file, inner)}` : '') + '</details>';
  }

  /** Asks for the form of every open config file not yet asked for, and of every file a key search (kept in the
   *  webview state, so also after a reload) still shows as searching…: without this a reloaded search waits forever. */
  function requestTrees() {
    const ask = file => { if (file && !requested.has(file)) { requested.add(file); send({ type: 'tree', file }); } };
    for (const el of app.querySelectorAll('details.cfg[open], details.done-run[open] details.rungraph[open], .budget-form[data-file]')) ask(el.dataset.file);
    for (const p of (last && last.projects) || []) if (searchOf(p.name)) for (const c of p.configs || []) if (!trees[c.path]) ask(c.path);
  }

  /** Stages one change of a form; nothing is written until OK. × on an item staged for removal keeps it. */
  /** A staged arrange (the planner's changes of a model's layers) whose list holds `path`, else null: the form's
   *  own edits of those layers wait until it is OK'd or undone (the two would index different lists). */
  function arrangeOver(file, path) {
    const p = pending[file];
    if (!p) return null;
    return Object.values(p.edits).find(e => e.op === 'arrange' && path.length > e.path.length &&
      JSON.stringify(path.slice(0, e.path.length)) === JSON.stringify(e.path)) || null;
  }
  const ARRANGED = 'refused: the planner\'s changes of these layers are staged: OK or undo them (× on "rearranged in the planner") first';
  function stage(el, op, value) {
    const form = el.closest('.form');
    if (!form) return;
    const file = form.dataset.file;
    const path = JSON.parse(el.dataset.path);
    if (arrangeOver(file, path)) { if (trees[file]) trees[file].note = ARRANGED; return; }
    const p = pending[file] || (pending[file] = { sha: form.dataset.sha, edits: {}, seq: 0 });
    if (op === 'add') p.edits[`add|${JSON.stringify(path)}|${p.seq++}`] = { op, path, value };
    else if (op === 'remove' || op === 'delete') {
      const id = editId(op, path);
      if (p.edits[id]) delete p.edits[id]; else p.edits[id] = { op, path };
    } else {
      const key = op === 'put' ? el.dataset.key : undefined;
      p.edits[editId(op, path, key)] = { op, path, value, key };
    }
    prune(file);
    if (trees[file]) trees[file].note = '';
  }

  /** Stage one edit from code (Find Figure, the facet layout table), as typing it would. */
  function stageValue(form, path, op, value, key) {
    if (!form) return;
    const file = form.dataset.file;
    if (arrangeOver(file, path)) { if (trees[file]) trees[file].note = ARRANGED; return; }
    const p = pending[file] || (pending[file] = { sha: form.dataset.sha, edits: {}, seq: 0 });
    p.edits[editId(op, path, key)] = { op, path, value, key };
    prune(file);
    if (trees[file]) trees[file].note = '';
  }

  /** A value as the file holds it, to compare a staged value with the file's: text unquoted, a number
   *  as a number, true / false in any case, a list item by item (a tree node or a typed value). */
  function canon(v) {
    if (Array.isArray(v)) return v.map(canon);
    if (v && typeof v === 'object') {
      if (v.k === 'scalar') return canon(unquote(v.text));
      if (v.k === 'list') return v.items.map(canon);
      return v.k ? { map: true } : v;                         // a mapping is never one typed value
    }
    if (typeof v !== 'string') return v;
    const s = v.trim();
    if (/^\[.*\]$/.test(s)) {
      try { return canon(JSON.parse(s)); } catch (e) { return s.slice(1, -1).split(',').map(x => x.trim()).filter(x => x !== '').map(canon); }
    }
    const u = unquote(/^".*"$/.test(s) ? (() => { try { return JSON.parse(s); } catch (e) { return s; } })() : s);
    if (u !== '' && Number.isFinite(Number(u))) return Number(u);
    if (/^(true|false)$/i.test(u)) return u.toLowerCase() === 'true';
    return u;
  }
  const sameValue = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));

  /** Drops the staged values the file already holds (a value changed, then changed back): when none is
   *  left, the [OK][Cancel][Reset] bar goes. */
  function prune(file) {
    const p = pending[file];
    const tree = trees[file] && trees[file].tree;
    if (!p) return;
    for (const [id, e] of Object.entries(tree ? p.edits : {})) {
      if (e.op !== 'set' && e.op !== 'put') continue;
      let node = resolveIn(tree, e.path).node;
      if (e.op === 'put') node = node && node.k === 'map' ? entryOf(node, e.key) : null;
      if (node && sameValue(e.value, node)) delete p.edits[id];
    }
    if (!Object.keys(p.edits).length) delete pending[file];
  }

  // ── facet layout: rows x columns, then which panel each box shows ──────────────────────────────
  const tableKey = path => `${curFile}|${JSON.stringify(path)}`;
  function panelIds(path) {
    const facet = trees[curFile] ? window.TFMenus.nodeAt(trees[curFile].tree, path.slice(0, 2)) : null;
    const panels = facet && facet.k === 'map' ? (facet.entries.find(e => e.key === 'panels') || {}).node : null;
    return (panels && panels.k === 'list' ? panels.items : []).map(it => {
      const get = k => (it.k === 'map' ? ((it.entries.find(e => e.key === k) || {}).node || {}).text : '');
      return { id: String(get('file_id')), address: window.TFMenus.unquote(get('address')) };
    }).filter(x => x.id);
  }
  /** The layout as the file has it: {r, c, cells: [[id|'']]}. */
  function fileLayout(node, ids) {
    const entry = k => (node.k === 'map' ? (node.entries.find(e => e.key === k) || {}).node : null);
    const mode = (entry('mode') || {}).text ? window.TFMenus.unquote(entry('mode').text) : 'horizontal';
    const order = entry('order');
    const items = order && order.k === 'list' ? order.items : [];
    let rows;
    if (items.some(it => it.k === 'list')) rows = items.map(it => (it.items || []).map(x => String(x.text)));
    else {
      const flat = items.length ? items.map(x => String(x.text)) : ids.map(x => x.id);
      const cols = Number((entry('columns') || {}).text) || Math.ceil(Math.sqrt(flat.length)) || 1;
      if (mode === 'vertical') rows = flat.map(x => [x]);
      else if (mode === 'grid') { rows = []; for (let i = 0; i < flat.length; i += cols) rows.push(flat.slice(i, i + cols)); }
      else rows = [flat];
    }
    const c = Math.max(1, ...rows.map(x => x.length));
    return { r: Math.max(1, rows.length), c, cells: rows.map(row => [...row, ...Array(c - row.length).fill('')]) };
  }
  function tableOf(path, node) {
    ui.facetTable = ui.facetTable || {};
    return ui.facetTable[tableKey(path)] || fileLayout(node, panelIds(path));
  }
  function stageLayout(form, path, t) {
    const { mode, order } = window.TFMenus.facetLayout(t.r, t.c, t.cells);
    stageValue(form, path, 'put', mode, 'mode');
    stageValue(form, path, 'put', JSON.stringify(order).replace(/,/g, ', '), 'order');
  }
  function setTable(form, path, r, c) {
    if (!form) return;
    curFile = form.dataset.file;                 // the clicked form's file, not the last one drawn
    const node = trees[curFile] ? window.TFMenus.nodeAt(trees[curFile].tree, path) : null;
    if (!node) return;
    const old = tableOf(path, node);
    const ids = panelIds(path).map(x => x.id);
    const cells = Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => ((old.cells[i] || [])[j]) || ''));
    const used = new Set(cells.flat().filter(Boolean));
    const free = ids.filter(x => !used.has(x));
    for (const row of cells) for (let j = 0; j < row.length; j++) if (!row[j] && free.length) row[j] = free.shift();
    const t = { r, c, cells };
    ui.facetTable[tableKey(path)] = t;
    save();
    stageLayout(form, path, t);
  }
  function setCell(form, path, at, value) {
    if (!form) return;
    curFile = form.dataset.file;
    const node = trees[curFile] ? window.TFMenus.nodeAt(trees[curFile].tree, path) : null;
    if (!node) return;
    const t = JSON.parse(JSON.stringify(tableOf(path, node)));
    const [i, j] = at.split(',').map(Number);
    for (const row of t.cells) for (let k = 0; k < row.length; k++) if (value && row[k] === value) row[k] = '';   // a panel once
    t.cells[i][j] = value;
    ui.facetTable[tableKey(path)] = t;
    save();
    stageLayout(form, path, t);
  }
  /** faceting.<i>.arrangement: hover a grid to pick rows x columns, then a box per cell. */
  function arrangementHtml(label, node) {
    const path = node.path;
    const ids = panelIds(path);
    const t = tableOf(path, node);
    // The picker grid: 4 x 4 unless Rows / Columns give its size (never smaller than the layout).
    const g = (ui.facetGrid || {})[tableKey(path)] || { r: 4, c: 4 };
    const G = { r: Math.max(g.r, t.r), c: Math.max(g.c, t.c) };
    const picker = Array.from({ length: G.r }, (_, i) => `<div class="tp-row">${Array.from({ length: G.c }, (_, j) =>
      `<button class="tp${i < t.r && j < t.c ? ' on' : ''}" data-tablepick="${i + 1}x${j + 1}" ${pathAttr(path)} title="${i + 1} × ${j + 1}"></button>`).join('')}</div>`).join('');
    const short = a => String(a).split('/').pop();
    const cell = (v, i, j) => `<select class="val" data-tablecell="${i},${j}" ${pathAttr(path)}><option value="">(empty)</option>` +
      `${ids.map(x => `<option value="${esc(x.id)}"${x.id === v ? ' selected' : ''} title="${esc(x.address)}">${esc(x.id)}: ${esc(short(x.address))}</option>`).join('')}</select>`;
    const grid = `<table class="facet-cells">${t.cells.map((row, i) => `<tr>${row.map((v, j) => `<td>${cell(v, i, j)}</td>`).join('')}</tr>`).join('')}</table>`;
    const mode = t.r === 1 ? 'one row (horizontal)' : t.c === 1 ? 'one column (vertical)' : 'grid';
    const placed = new Set(t.cells.flat().filter(Boolean));
    const left = ids.filter(x => !placed.has(x.id)).map(x => x.id);
    const size = `<div class="tp-size" title="the size of the grid to pick the layout from (1 to 20 each)"><span class="tp-size-name">Grid</span>` +
      `<span class="tp-dim">Rows <input type="number" min="1" max="20" data-gridrows="1" ${pathAttr(path)} value="${G.r}"></span>` +
      `<span class="tp-dim">Columns <input type="number" min="1" max="20" data-gridcols="1" ${pathAttr(path)} value="${G.c}"></span></div>`;
    return `<div class="field arrangement">${label}<div class="facet-table"><div><div class="tp-grid" title="pick rows × columns">${picker}</div>${size}</div>` +
      `<div><small class="note">${t.r} × ${t.c}: ${mode}${left.length ? `; not placed: panel ${left.join(', ')}` : ''}. OK writes mode and order.</small>${grid}</div></div></div>`;
  }

  // ── graph_config.yaml: blocks drawn from what graph_manager reads (choices.py graph_forms) ──────
  // Every key of export, legend, style.axes and text_rules is shown, the file's or graph_manager's
  // default. A key the file lacks is staged as a draft at its path ({rule: target} standing for a
  // text rule of that target); OK writes the drafts as puts (a new key or block) and adds (a rule).
  const GF = () => (MENUS && MENUS.graph) || null;
  const draftId = path => `draft|${JSON.stringify(path)}`;
  const isRule = seg => !!seg && typeof seg === 'object';
  const disable = html => html.replace(/<(input|select|button)\b/g, '<$1 disabled');
  const fade = (html, why) => `<div class="faded" title="${esc(why)}">${disable(html)}</div>`;
  const ruleTarget = it => (it && it.k === 'map' ? unquote(((it.entries.find(e => e.key === 'target') || {}).node || {}).text || '') : '');
  const childOf = (node, seg) => (!node ? null : node.k === 'map' ? (node.entries.find(e => e.key === String(seg)) || {}).node || null
    : node.k === 'list' ? node.items[seg] || null : null);

  /** The node at path in tree ({rule: t}: the first rule of target t), with its real path; nulls when absent. */
  function resolveIn(tree, path) {
    let node = tree;
    const real = [];
    for (const seg of path) {
      if (isRule(seg)) {
        const i = node && node.k === 'list' ? node.items.findIndex(it => ruleTarget(it) === seg.rule) : -1;
        if (i < 0) return { node: null, path: null };
        node = node.items[i];
        real.push(i);
      } else {
        node = childOf(node, seg);
        real.push(seg);
      }
      if (!node) return { node: null, path: null };
    }
    return { node, path: real };
  }
  const resolve = path => resolveIn(trees[curFile] && trees[curFile].tree, path);

  /** What a field shows: a staged value, the file's, a draft, or nothing ({text: null}). */
  function valueAt(path) {
    const r = resolve(path);
    if (r.node && r.node.k === 'scalar') {
      const s = shown(r.path, r.node.text);
      return { text: unquote(s.text), dirty: !!s.cls, path: r.path, node: r.node };
    }
    if (r.node) return { text: null, path: r.path, node: r.node };
    const d = stagedOf(curFile)[draftId(path)];
    return d ? { text: Array.isArray(d.value) ? JSON.stringify(d.value) : String(d.value), dirty: true, draft: true } : { text: null };
  }
  const drafted = base => Object.keys(stagedOf(curFile)).some(id => id.startsWith(`draft|${JSON.stringify(base).slice(0, -1)}`));

  /** A typed box value as YAML data: '' is no value (graph_manager's default). */
  function typed(text, t) {
    const v = String(text).trim();
    if (v === '') return undefined;
    if (t === 'bool') return v === 'true';
    if (t === 'int' && /^-?\d+$/.test(v)) return Number(v);
    if ((t === 'num' || t === 'int') && Number.isFinite(Number(v))) return Number(v);
    if (t === 'list') {                                   // 1, 50, 100 or ID, HW: numbers stay numbers, names text
      try { return JSON.parse(`[${v.replace(/^\[|\]$/g, '')}]`); } catch (e) {
        return v.replace(/^\[|\]$/g, '').split(',').map(x => x.trim()).filter(Boolean).map(x => (/^-?\d+(\.\d+)?(e-?\d+)?$/i.test(x) ? Number(x) : x.replace(/^(["'])(.*)\1$/, '$2')));
      }
    }
    if (t === 'any' && /^-?\d+(\.\d+)?(e-?\d+)?$/i.test(v)) return Number(v);      // a layer size, not text
    return v;
  }
  function setDraft(form, path, value) {
    const file = form.dataset.file;
    const p = pending[file] || (pending[file] = { sha: form.dataset.sha, edits: {}, seq: 0 });
    if (value === undefined) delete p.edits[draftId(path)];
    else p.edits[draftId(path)] = { op: 'draft', path, value };
    if (!Object.keys(p.edits).length) delete pending[file];
    if (trees[file]) trees[file].note = '';
  }

  /** One key: its menu or box. The file's key is set (× removes it: graph_manager's default again);
   *  one the file lacks shows the default and stages a draft. why: faded, unavailable (and why).
   *  opt (style and text_rules): noX, no ×; clear, a [Clear] that empties a text; binary, a true/false
   *  key as True / False alone (the default shown when the file lacks it). A colour is a palette. */
  function fieldHtml(label, path, spec, why, opt = {}) {
    const v = valueAt(path);
    const dflt = spec.d == null || spec.d === '' ? '' : Array.isArray(spec.d) ? JSON.stringify(spec.d) : String(spec.d);
    // a key the file lacks: its draft goes when the value shown at first (data-orig) is chosen again
    const orig = spec.t === 'bool' && opt.binary ? String(!!spec.d) : spec.t === 'enum' && opt.pick ? dflt : '';
    const attrs = v.path ? `data-op="set" ${pathAttr(v.path)}`
      : `data-draft="${esc(JSON.stringify(path))}" data-t="${esc(spec.t)}" data-orig="${esc(orig)}"`;
    const cur = v.text == null ? '' : v.text;
    const cls = `val${v.dirty ? ' dirty' : ''}${v.path || v.draft ? '' : ' absent'}`;
    let control;
    if (spec.t === 'bool' && opt.binary) {
      const on = (cur === '' ? String(!!spec.d) : cur) === 'true';
      control = `<select class="${cls}" ${attrs}><option value="true"${on ? ' selected' : ''}>True</option>` +
        `<option value="false"${on ? '' : ' selected'}>False</option></select>`;
    } else if (spec.t === 'color') {
      control = colorHtml(path, v, cur || dflt || 'black');
    } else if (spec.t === 'enum' && opt.pick && !v.path) {      // the default is the option selected, not an entry of its own
      const sel = cur || dflt;
      control = `<select class="${cls}" ${attrs}>${spec.o.map(o => `<option value="${esc(o)}"${o === sel ? ' selected' : ''}>${esc(o)}</option>`).join('')}</select>`;
    } else if (spec.t === 'enum' || spec.t === 'bool') {
      const opts = spec.t === 'bool' ? ['true', 'false'] : spec.o;
      const others = spec.others || {};                   // options another framework has: listed, never choosable
      const known = cur === '' || opts.includes(cur) || cur in others;
      control = `<select class="${cls}${known ? '' : ' bad'}" ${attrs}>` +
        (v.path ? '' : `<option value=""${cur === '' ? ' selected' : ''}>${spec.empty ? '(empty)' : `${esc(dflt || 'default')} (default)`}</option>`) +
        (known ? '' : `<option value="${esc(cur)}" selected>${esc(cur)} (not an option)</option>`) +
        opts.map(o => `<option value="${esc(o)}"${o === cur ? ' selected' : ''}>${esc(o)}</option>`).join('') +
        Object.keys(others).map(o => (o === cur
          ? `<option value="${esc(o)}" selected title="${esc(others[o])}, not in the framework this run uses (deep_learning.framework)">${esc(o)} (${esc(others[o])})</option>`
          : `<option value="${esc(o)}" disabled title="${esc(others[o])}, not in the framework this run uses (deep_learning.framework)">${esc(o)} (${esc(others[o])})</option>`)).join('') +
        '</select>';
    } else {
      const list = spec.o ? ` list="${listId(path)}"` : '';
      control = `<input class="${cls}${opt.required && cur === '' ? ' bad' : ''}" ${attrs} value="${esc(cur)}" placeholder="${esc(opt.required ? 'required' : dflt ? `${dflt} (default)` : '')}"${list}>` +
        (spec.o ? `<datalist id="${listId(path)}">${spec.o.map(o => `<option value="${esc(o)}">`).join('')}</datalist>` : '');
    }
    const gone = v.path && removed(v.path);
    let x = v.path && !opt.noX ? btn(`class="x" data-op="delete" ${pathAttr(v.path)}`, gone ? 'keep' : '×', gone ? 'keep' : 'remove the key: graph_manager uses its default') : '';
    if (opt.clear && cur !== '') {
      x = btn(`data-clear="1" ${v.path ? pathAttr(v.path) : `data-draftpath="${esc(JSON.stringify(path))}"`}`, 'Clear', 'empty this text');
    }
    const where = path.filter(x => !isRule(x)).join(' › ');
    const html = `<div class="field${gone ? ' removed' : ''}"><label title="${esc(opt.hint ? `${where}\n${opt.hint}` : where)}">${esc(label)}</label>${control}${x}</div>`;
    return why ? fade(html, why) : html;
  }

  // A colour as MS Word picks one: Automatic (black), the Office theme colours with their lighter and
  // darker shades, the standard colours, More Colors… (any colour). Written as #RRGGBB (matplotlib reads it).
  const THEME = [['#FFFFFF', 'White, Background 1'], ['#000000', 'Black, Text 1'], ['#E7E6E6', 'Gray-25%, Background 2'],
    ['#44546A', 'Blue-Gray, Text 2'], ['#4472C4', 'Blue, Accent 1'], ['#ED7D31', 'Orange, Accent 2'], ['#A5A5A5', 'Gray-50%, Accent 3'],
    ['#FFC000', 'Gold, Accent 4'], ['#5B9BD5', 'Blue, Accent 5'], ['#70AD47', 'Green, Accent 6']];
  const STANDARD = [['#C00000', 'Dark Red'], ['#FF0000', 'Red'], ['#FFC000', 'Orange'], ['#FFFF00', 'Yellow'], ['#92D050', 'Light Green'],
    ['#00B050', 'Green'], ['#00B0F0', 'Light Blue'], ['#0070C0', 'Blue'], ['#002060', 'Dark Blue'], ['#7030A0', 'Purple']];
  const TAB10 = { C0: '#1f77b4', C1: '#ff7f0e', C2: '#2ca02c', C3: '#d62728', C4: '#9467bd', C5: '#8c564b', C6: '#e377c2', C7: '#7f7f7f', C8: '#bcbd22', C9: '#17becf' };
  const NAMED = { black: '#000000', white: '#FFFFFF', gray: '#808080', grey: '#808080', red: '#FF0000', blue: '#0000FF', green: '#008000' };
  const hexOf = c => (/^#[0-9a-f]{6}$/i.test(String(c)) ? String(c).toUpperCase() : TAB10[c] || NAMED[String(c).toLowerCase()] || '#000000');
  /** A theme colour's shade as Office makes it: its HSL lightness moved toward white (+) or black (-). */
  function shade(hex, f) {
    const [r, g, b] = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255);
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    let h = 0;
    let s = 0;
    let l = (max + min) / 2;
    if (max !== min) {
      const d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      h = max === r ? (g - b) / d + (g < b ? 6 : 0) : max === g ? (b - r) / d + 2 : (r - g) / d + 4;
      h /= 6;
    }
    l = f > 0 ? l + (1 - l) * f : l * (1 + f);
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    const ch = x => {
      let u = x < 0 ? x + 1 : x > 1 ? x - 1 : x;
      u = u < 1 / 6 ? p + (q - p) * 6 * u : u < 1 / 2 ? q : u < 2 / 3 ? p + (q - p) * (2 / 3 - u) * 6 : p;
      return Math.round(u * 255).toString(16).padStart(2, '0');
    };
    return `#${(s === 0 ? [l, l, l].map(x => Math.round(x * 255).toString(16).padStart(2, '0')) : [ch(h + 1 / 3), ch(h), ch(h - 1 / 3)]).join('')}`.toUpperCase();
  }
  /** The five shades under a theme colour, as Word lists them. */
  function shades(hex, name) {
    const steps = hex === '#FFFFFF' ? [-0.05, -0.15, -0.25, -0.35, -0.5] : hex === '#000000' ? [0.5, 0.35, 0.25, 0.15, 0.05]
      : hex === '#E7E6E6' ? [-0.1, -0.25, -0.5, -0.75, -0.9] : [0.8, 0.6, 0.4, -0.25, -0.5];
    return steps.map(f => [shade(hex, f), `${name}, ${f > 0 ? 'Lighter' : 'Darker'} ${Math.round(Math.abs(f) * 100)}%`]);
  }
  const swatch = (hex, size = 12) => `<svg class="swatch" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}"><rect width="${size}" height="${size}" fill="${esc(hex)}"/></svg>`;
  function colorHtml(path, v, cur) {
    const key = JSON.stringify(path);
    const target = v.path ? pathAttr(v.path) : `data-cdraft="${esc(key)}"`;
    const head = `<button class="cpick${v.dirty ? ' dirty' : ''}" data-cpick="${esc(key)}" title="the colour: ${esc(cur)}">${swatch(hexOf(cur))} ${esc(cur)} ▾</button>`;
    if (!((ui.cpick || {})[key])) return head;
    const cell = ([hex, name]) => `<button class="cp${hexOf(cur) === hex ? ' on' : ''}" data-cset="${hex}" ${target} title="${esc(name)}">${swatch(hex)}</button>`;
    const columns = THEME.map(([hex, name]) => [[hex, name], ...shades(hex, name)]);
    const rows = [0, 1, 2, 3, 4, 5].map(r => `<div class="cp-row${r === 0 ? ' base' : ''}">${columns.map(c => cell(c[r])).join('')}</div>`).join('');
    return `${head}<div class="cpalette"><button class="cp-auto" data-cset="black" ${target}>${swatch('#000000')} Automatic</button>` +
      `<div class="cp-title">Theme Colors</div>${rows}<div class="cp-title">Standard Colors</div>` +
      `<div class="cp-row">${STANDARD.map(cell).join('')}</div>` +
      `<button class="cp-morebtn${(ui.cmore || {})[key] ? ' on' : ''}" data-cmorebtn="${esc(key)}">More Colors…</button>` +
      `${(ui.cmore || {})[key] ? `<button class="honey" data-honey="1" ${target} title="pick a colour">${honeycomb(hexOf(cur))}</button>` : ''}</div>`;
  }

  // More Colors…: Office's Colors honeycomb, drawn in the panel (one SVG: no system dialog that holds the
  // window). 127 cells in rings around white: the hue turns with the angle (blue at the top, magenta to the
  // right, red at the bottom, green to the left), the colour fills out to ring 4 and darkens to ring 6;
  // under it white, two rows of greys, black.
  const HEX_R = 7;
  const hexPoints = (cx, cy, r) => [0, 1, 2, 3, 4, 5].map(i => {
    const a = Math.PI / 180 * (60 * i - 90);
    return `${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`;
  }).join(' ');
  function hsl(h, s, l) {
    const k = n => (n + h / 30) % 12;
    const a = s * Math.min(l, 1 - l);
    const f = n => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
    return `#${[f(0), f(8), f(4)].map(x => Math.round(x * 255).toString(16).padStart(2, '0')).join('')}`.toUpperCase();
  }
  let HONEY = null;
  function honeyCells() {
    if (HONEY) return HONEY;
    const w = Math.sqrt(3) * HEX_R;
    const cells = [];
    for (let q = -6; q <= 6; q += 1) {
      for (let r = -6; r <= 6; r += 1) {
        const d = Math.max(Math.abs(q), Math.abs(r), Math.abs(q + r));
        if (d > 6) continue;
        const x = w * (q + r / 2);
        const y = 1.5 * HEX_R * r;
        const cw = (Math.atan2(x, -y) * 180 / Math.PI + 360) % 360;      // clockwise from the top
        const l = d <= 4 ? 1 - 0.5 * d / 4 : 0.5 - 0.2 * (d - 4) / 2;
        cells.push({ x, y, hex: d === 0 ? '#FFFFFF' : hsl((240 + cw) % 360, 1, l) });
      }
    }
    const greys = [...Array(14).keys()].map(i => { const v = Math.round(242 - i * (242 - 20) / 13).toString(16).padStart(2, '0'); return `#${v}${v}${v}`.toUpperCase(); });
    HONEY = { w, cells, greys };
    return HONEY;
  }
  function honeycomb(cur) {
    const { w, cells, greys } = honeyCells();
    const cx = 6.5 * w + 2;
    const cy = 9 * HEX_R + HEX_R + 2;                         // the top row's points inside the drawing
    const poly = (x, y, r, hex) => `<polygon points="${hexPoints(x, y, r)}" fill="${hex}" data-hex="${hex}"` +
      `${hex === cur ? ' class="on"' : hex === '#FFFFFF' ? ' class="edge"' : ''}/>`;   // white keeps a border, as in Office
    const gy = cy + 10.5 * HEX_R;
    const row = (k, y) => greys.slice(k * 7, k * 7 + 7).map((g, i) => poly(cx + (i - 3) * w + (k ? w / 2 : 0) - w / 4, y, HEX_R, g)).join('');
    const width = Math.round(cx * 2);
    const height = Math.round(gy + 2.5 * HEX_R + 2);
    return `<svg class="honeycomb" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
      cells.map(c => poly(cx + c.x, cy + c.y, HEX_R, c.hex)).join('') +
      poly(cx - 5.3 * w, gy + 0.75 * HEX_R, 1.9 * HEX_R, '#FFFFFF') + row(0, gy) + row(1, gy + 1.5 * HEX_R) +
      poly(cx + 5.3 * w, gy + 0.75 * HEX_R, 1.9 * HEX_R, '#000000') + '</svg>';
  }

  /** A key graph_manager does not read (moved or removed): shown, with × to delete it. */
  function strayHtml(key, node, why) {
    const what = node.k === 'scalar' ? node.text : node.k === 'list' ? '[…]' : '{…}';
    const gone = removed(node.path);
    return `<div class="field stray${gone ? ' removed' : ''}"><label>${esc(key)}</label><span class="note">${esc(what)}: ${esc(why)}</span>` +
      `${btn(`class="x" data-op="delete" ${pathAttr(node.path)}`, gone ? 'keep' : '×', gone ? 'keep' : 'delete it from the file')}</div>`;
  }
  const blockHtml = (path, label, summary, body) => {
    const k = nodeKey(curFile, path);
    const open = isOpen(k, false);
    return `<details class="node" data-key="${esc(k)}"${open ? ' open' : ''}><summary title="${titled(curFile, path.filter(x => !isRule(x)), path.filter(x => !isRule(x)).join(' › '))}">${esc(label)} <small>${summary}</small></summary>` +
      (open ? body() : '') + '</details>';
  };
  const strays = (node, known, moved = {}) => (node && node.k === 'map' ? node.entries.filter(e => !known.includes(e.key))
    .map(e => strayHtml(e.key, e.node, moved[e.key] || 'not read by graph_manager')).join('') : '');

  /** jobs.<i>.plot_type: only the types that change how the job's metric is drawn. */
  function plotTypeMenu(path) {
    if (!isGraph(curFile) || !GF() || path.length !== 3 || path[0] !== 'jobs' || path[2] !== 'plot_type') return null;
    const job = path.slice(0, 2);
    const metric = resolve([...job, 'metric']).node;
    const metrics = !metric ? [] : metric.k === 'list' ? metric.items.map(it => unquote(shown(it.path, it.text).text)) : [valueAt([...job, 'metric']).text];
    const type = valueAt([...job, 'type']).text || 'model_overlay';
    const mode = valueAt([...job, 'array_epoch', 'mode']).text || 'latest';
    return { opts: window.TFMenus.plotTypesFor(GF().plotTypes, type, metrics, mode), free: false };
  }

  /** export: formats as tick boxes in a row; only a ticked format shows its own settings (png: dpi; svg: …;
   *  pdf: none). A tick writes the whole list (the file's, or a draft when the file has none). */
  function exportHtml(path, node, kind) {
    const spec = GF().export[kind];
    const fmtNode = childOf(node, 'formats');
    let formats;
    let target;
    if (fmtNode && (fmtNode.k === 'list' || fmtNode.k === 'scalar')) {
      const whole = stagedOf(curFile)[editId('set', fmtNode.path)];
      formats = whole ? String(whole.value).replace(/^\[|\]$/g, '').split(',').map(s => unquote(s.trim())).filter(Boolean) : fmtNode.k === 'list' ? fmtNode.items.map(it => unquote(it.text)) : [unquote(fmtNode.text)];
      target = pathAttr(fmtNode.path);
    } else {
      const d = stagedOf(curFile)[draftId([...path, 'formats'])];
      formats = d ? d.value : spec.default;
      target = `data-fmtdraft="${esc(JSON.stringify([...path, 'formats']))}"`;
    }
    const boxes = spec.formats.map(f => `<label class="fmtbox"><input type="checkbox" data-fmtcheck="${esc(f)}" ${target}` +
      ` data-list="${esc(JSON.stringify(formats))}"${formats.includes(f) ? ' checked' : ''}> ${esc(f)}</label>`).join('');
    const fmtHtml = `<div class="field"><label>formats</label><span class="fmtboxes">${boxes}</span></div>`;
    const known = ['formats', ...spec.common.map(f => f.k), ...spec.formats];
    return blockHtml(path, 'export', esc(formats.join(', ')), () => blockButtons(path) + fmtHtml +
      spec.common.map(f => fieldHtml(f.k, [...path, f.k], f, null, { noX: true })).join('') +
      spec.formats.filter(f => formats.includes(f)).map(f => {
        const settings = spec.settings[f] || [];
        const body = settings.length ? settings.map(x => fieldHtml(x.k, [...path, f, x.k], x, null, { noX: true })).join('') : '<small class="note">vector: no settings</small>';
        return `<div class="fmt"><div class="fmt-name">${esc(f)}</div>${body}</div>`;
      }).join('') + strays(node, known, spec.moved));
  }

  /** legend.handle [shape, thickness]: a shape menu (Character… asks for the character) and the pt. */
  function handleHtml(path, kind, dflt) {
    const r = resolve(path);
    let cur = (dflt || []).map(String);
    let real = null;
    if (r.node && r.node.k === 'list') {
      real = r.path;
      const whole = stagedOf(curFile)[editId('set', r.path)];
      cur = whole ? JSON.parse(whole.value).map(String) : r.node.items.map(it => unquote(shown(it.path, it.text).text));
    } else {
      const d = stagedOf(curFile)[draftId(path)];
      if (d) cur = d.value.map(String);
    }
    const shapes = GF().handle[kind];
    const [shape = '', size = ''] = cur;
    const key = JSON.stringify(path);
    const char = (shape !== '' && !shapes.includes(shape)) || !!((ui.handleChar || {})[key]);
    const dirty = (real && stagedOf(curFile)[editId('set', real)]) || stagedOf(curFile)[draftId(path)] ? ' dirty' : '';
    const opts = (real ? '' : `<option value=""${shape === '' ? ' selected' : ''}>${esc(dflt ? dflt[0] : "Matplotlib's")} (default)</option>`) +
      shapes.map(o => `<option value="${esc(o)}"${!char && o === shape ? ' selected' : ''}>${esc(o)}</option>`).join('') +
      `<option value="__char"${char ? ' selected' : ''}>Character…</option>`;
    return `<div class="field handle" data-handle="${esc(key)}" data-real="${esc(JSON.stringify(real))}"><label title="[shape, thickness pt]">handle</label>` +
      `<select class="val${dirty}" data-handlepart="shape">${opts}</select>` +
      (char ? `<input class="val${dirty}" data-handlepart="char" maxlength="2" size="2" value="${esc(char && !shapes.includes(shape) ? shape : '')}" placeholder="■">` : '') +
      `<input class="val${dirty}" data-handlepart="size" type="number" min="0.1" step="0.1" value="${esc(size)}" placeholder="pt"></div>`;
  }

  /** The legend entries a facet takes in mode auto: its first panel's graph job's models, as that job's
   *  legend names them (faceting.py _job_entries), from the jobs of the form's file. */
  function facetAutoEntries(legendPath) {
    const tree = trees[curFile] && trees[curFile].tree;
    const facet = resolveIn(tree, legendPath.slice(0, 2)).node;
    const panels = childOf(facet, 'panels');
    const first = panels && panels.k === 'list' && panels.items[0] ? unquote(((entryOf(panels.items[0], 'address')) || {}).text || '') : '';
    const m = first.match(/(?:^|\/)job_(\d{3})_[^/]*\//);
    const jobs = childOf(tree, 'jobs');
    const job = m && jobs && jobs.k === 'list' ? jobs.items.find(j => Number(unquote((entryOf(j, 'id') || {}).text)) === Number(m[1])) : null;
    if (!job) return [];
    const models = texts(entryOf(job, Number(m[1]) === 0 ? 'top_models' : 'models')).map(unquote);
    const legend = entryOf(job, 'legend');
    const custom = unquote((entryOf(legend, 'mode') || {}).text || 'auto') === 'custom';
    const labels = custom ? entryOf(legend, 'labels') : null;
    return models.map(name => (labels && labels.k === 'map' ? unquote((entryOf(labels, name) || {}).text || '') : '') || name);
  }

  /** A facet's custom legend: a row per colour, the colour and what its first panel's job draws in it
   *  engraved, then the entry's label; + adds a row. Entry i takes colour C(i-1) (faceting.py). Rows the
   *  file lacks are filled in order (an entry's place is its colour). */
  function facetEntriesHtml(path, entries) {
    const auto = facetAutoEntries(path);
    const real = entries && entries.k === 'list';
    const d = stagedOf(curFile)[draftId([...path, 'entries'])];
    const fileRows = real ? entries.items : [];
    const adds = real ? Object.values(stagedOf(curFile)).filter(e => e.op === 'add' && JSON.stringify(e.path) === JSON.stringify(entries.path)) : [];
    const drafts = !real && d ? d.value : [];
    const have = fileRows.length + adds.length + drafts.length;
    const key = JSON.stringify(path);
    const rows = Math.max(have + 1, auto.length, 1) + ((ui.facetRows || {})[key] || 0);
    const out = [];
    for (let i = 0; i < rows; i += 1) {
      const colour = `C${i % 10}`;
      const tag = `<span class="readonly entry-tag" title="entry ${i + 1}: colour ${colour}">${swatch(TAB10[colour], 10)} ${colour}${auto[i] ? ` · ${esc(auto[i])}` : ''}</span>`;
      let box;
      if (i < fileRows.length) {
        const it = fileRows[i];
        const at = it.k === 'map' ? (entryOf(it, 'label') ? [...it.path, 'label'] : null) : it.path;
        const s = at ? shown(at, it.k === 'map' ? entryOf(it, 'label').text : it.text) : { text: '', cls: '' };
        box = at ? `<input class="val${s.cls}" data-op="set" ${pathAttr(at)} value="${esc(unquote(s.text))}" placeholder="its label">` : '<span class="note">no label</span>';
      } else if (i < have) {
        const v = i < fileRows.length + adds.length ? (JSON.parse(adds[i - fileRows.length].value) || {}).label : (drafts[i] || {}).label;
        box = `<span class="item added">${esc(v || '')}</span>`;
      } else if (i === have) {
        box = real ? `<input class="val" data-addmap="label" ${pathAttr(entries.path)} placeholder="its label${auto[i] ? `, e.g. ${esc(auto[i])}` : ''}">`
          : `<input class="val" data-draftmap="label" data-draftat="${esc(JSON.stringify([...path, 'entries']))}" data-list="${esc(JSON.stringify(drafts))}" placeholder="its label${auto[i] ? `, e.g. ${esc(auto[i])}` : ''}">`;
      } else {
        box = `<input class="val" disabled placeholder="after entry ${have + 1}">`;
      }
      out.push(`<div class="field entry-row">${tag}${box}</div>`);
    }
    return `<div class="field"><label>entries</label><small class="note">entry i takes colour C(i-1): the i-th model of a job</small></div>` +
      out.join('') + `<div class="field qadd-row">${btn(`data-entryrow="${esc(key)}"`, '+', 'one more entry row')}</div>`;
  }

  /** legend: every key graph_manager reads; labels (job) / entries (facet) fade while mode is auto. */
  function legendHtml(path, node, kind, ctx) {
    const spec = GF().legend[kind];
    const modeSpec = spec.find(f => f.k === 'mode');
    const mode = valueAt([...path, 'mode']).text || modeSpec.d;
    const auto = mode === 'auto';
    const fields = () => spec.map(f => (f.t === 'handle' ? handleHtml([...path, 'handle'], kind, f.d) : fieldHtml(f.k, [...path, f.k], f, null, { noX: true }))).join('');
    let list = () => '';
    if (kind === 'job') {
      const labels = childOf(node, 'labels');
      list = () => {
        const html = labels && (labels.k === 'map' || labels.type === 'null') ? labelsHtml('labels', labels, ctx.job) : labelsHtml('labels', null, ctx.job, [...path, 'labels']);
        return auto ? fade(html, 'mode auto: the legend shows the model names; pick custom to use these labels') : html;
      };
    } else {
      list = () => {
        const html = facetEntriesHtml(path, childOf(node, 'entries'));
        return auto ? fade(html, "mode auto: the entries are the first panel's graph job's") : html;
      };
    }
    const known = [...spec.map(f => f.k), kind === 'job' ? 'labels' : 'entries', ...(kind === 'facet' ? ['labels'] : [])];
    const moved = kind === 'facet' ? GF().moved.facetLegend : {};
    return blockHtml(path, 'legend', `mode ${esc(mode)}`, () => blockButtons(path) + fields() + list() + strays(node, known, moved));
  }

  /** style.axes: every key; limits, steps and x_grid are Auto (graph_manager's) or Custom (a range / a number). */
  function autoHtml(f, path) {
    const r = resolve(path);
    const key = JSON.stringify(path);
    const pair = f.t === 'pair';
    let vals = null;
    if (r.node && !removed(r.path)) {
      const whole = stagedOf(curFile)[editId('set', r.path)];
      if (whole) vals = [].concat(JSON.parse(whole.value)).map(String);
      else if (r.node.k === 'list') vals = r.node.items.map(it => unquote(shown(it.path, it.text).text));
      else if (r.node.type !== 'null') vals = [unquote(r.node.text)];
    } else if (!r.node) {
      const d = stagedOf(curFile)[draftId(path)];
      if (d) vals = [].concat(d.value).map(String);
    }
    const custom = !!vals || !!((ui.custom || {})[key]);
    const dirty = (r.path && (stagedOf(curFile)[editId('set', r.path)] || removed(r.path))) || stagedOf(curFile)[draftId(path)] ? ' dirty' : '';
    const boxes = custom ? (pair ? ['lower limit', 'upper limit'] : ['value']).map((ph, i) =>
      `<input class="val${dirty}" data-autopart="${i}" size="7" value="${esc((vals || [])[i] || '')}" placeholder="${ph}">`).join('') : '';
    return `<div class="field auto" data-auto="${esc(key)}" data-real="${esc(JSON.stringify(r.path))}" data-pair="${pair ? 1 : 0}">` +
      `<label>${esc(f.k)}</label><select class="val${dirty}" data-autopart="mode"><option value="auto"${custom ? '' : ' selected'}>Auto</option>` +
      `<option value="custom"${custom ? ' selected' : ''}>Custom</option></select>${boxes}</div>`;
  }
  function axesHtml(path, node) {
    const spec = GF().axes;
    return blockHtml(path, 'axes', '', () => spec.map(f => (f.t === 'pair' || f.t === 'auto' ? autoHtml(f, [...path, f.k]) : fieldHtml(f.k, [...path, f.k], f, null, { noX: true }))).join('') +
      strays(node, spec.map(f => f.k)));
  }

  /** text_rules: a panel per target, always there, with every formatting field under it. */
  function textRulesHtml(path, node, kind) {
    const T = GF().textRule;
    const items = node && node.k === 'list' ? node.items : [];
    const first = {};
    items.forEach((it, i) => { const t = ruleTarget(it); if (!(t in first)) first[t] = i; });
    const targets = T.targets[kind];
    const panels = () => targets.map(t => {
      const i = first[t];
      const base = i != null ? [...path, i] : [...path, { rule: t }];
      const gone = i != null && removed([...path, i]);
      const set = i != null || drafted(base);
      const text = valueAt([...base, 'text']).text;
      const k = nodeKey(curFile, [...path, `@${t}`]);
      const open = isOpen(k, false);
      const field = f => (t === 'legend_text' && f.k === 'text'
        ? `<div class="field"><label>text</label><span class="readonly" title="each legend entry: its model's label (legend.labels), else its name">${esc(text || '{model_name}')}</span></div>`
        : fieldHtml(f.k, [...base, ...f.k.split('.')], f, null, f.k === 'text' ? { noX: true, clear: true } : { noX: true, binary: true }));
      return `<details class="node${gone ? ' removed' : ''}" data-key="${esc(k)}"${open ? ' open' : ''}><summary>${esc(t)} ` +
        `<small>${set ? esc(text || '(no text)') : t === 'legend_text' ? '{model_name}' : "not set: graph_manager's default"}</small></summary>` +
        (open ? T.fields.map(field).join('') : '') + '</details>';
    }).join('');
    const others = () => items.map((it, i) => {
      const t = ruleTarget(it);
      if (first[t] === i && targets.includes(t)) return '';
      const why = !targets.includes(t) ? (T.removed[t] ? `removed: use ${T.removed[t]}` : 'not a target here') : `a second ${t} rule: only the first is used`;
      return strayHtml(`#${i} ${t}`, it, why);
    }).join('');
    const ok = path.length === 3 ? blockButtons(path) : '';       // a facet's own text_rules: OK here
    return blockHtml(path, 'text_rules', `${targets.filter(t => t in first).length}/${targets.length} set`, () => ok + panels() + others());
  }

  /** a job's style: its keys as before, then axes and text_rules drawn whole. */
  function styleHtml(path, node, ctx) {
    const rest = node && node.k === 'map' ? node.entries.filter(e => !['axes', 'text_rules'].includes(e.key))
      .map(e => entryHtml(e.key, e.node, curFile, node, { ...ctx, plain: false })).join('') : '';
    return blockHtml(path, 'style', '', () => blockButtons(path) + rest + axesHtml([...path, 'axes'], childOf(node, 'axes')) +
      textRulesHtml([...path, 'text_rules'], childOf(node, 'text_rules'), 'job'));
  }

  const NEEDS_MODE = { epoch: 'selected', epochs: 'overlay', summary: 'summary' };

  /** graph_config.yaml keys drawn from graph_manager's: null for a key drawn as any other. */
  function graphEntryHtml(key, node, ctx) {
    const p = node.path;
    const job = p[0] === 'jobs' && typeof p[1] === 'number';
    const facet = p[0] === 'faceting' && typeof p[1] === 'number';
    if (p.length === 3 && (job || facet)) {
      const kind = job ? 'job' : 'facet';
      if (key === 'export') return exportHtml(p, node, kind);
      if (key === 'legend') return legendHtml(p, node, kind, ctx);
      if (job && key === 'style') return styleHtml(p, node, ctx);
      if (facet && key === 'text_rules') return textRulesHtml(p, node, 'facet');
      if (facet && GF().moved.facet[key]) return strayHtml(key, node, `moved to ${GF().moved.facet[key]}`);
    }
    if (job && p.length === 4 && p[2] === 'array_epoch' && NEEDS_MODE[key]) {
      const mode = valueAt([...p.slice(0, 3), 'mode']).text || 'latest';
      const html = entryHtml(key, node, curFile, { path: p.slice(0, 3) }, { ...ctx, plain: true });
      return mode === NEEDS_MODE[key] ? html : fade(html, `used with array_epoch.mode: ${NEEDS_MODE[key]} (now ${mode})`);
    }
    return null;
  }

  /** The keys graph_manager reads that a job, a facet, a facet panel or array_epoch lacks. */
  function absentHtml(node, ctx) {
    const p = node.path;
    const has = k => node.entries.some(e => e.key === k);
    if (p.length === 0 && !has('faceting')) {                // no facet yet: faceting with Add a Facet
      return blockHtml(['faceting'], 'faceting', '0', () => addButton('faceting') + '<p class="note">none yet</p>');
    }
    if (p.length === 2 && p[0] === 'jobs' && typeof p[1] === 'number') {
      const ae = GF().arrayEpoch;
      return (has('array_epoch') ? '' : blockHtml([...p, 'array_epoch'], 'array_epoch', `mode ${esc(valueAt([...p, 'array_epoch', 'mode']).text || ae.default)}`,
        () => fieldHtml('mode', [...p, 'array_epoch', 'mode'], { t: 'enum', o: ae.modes, d: ae.default }, null, { pick: true }))) +
        (GF().jobKeys || []).filter(f => !has(f.k)).map(f => fieldHtml(f.k, [...p, f.k], f)).join('') +
        (has('export') ? '' : exportHtml([...p, 'export'], null, 'job')) +
        (has('legend') ? '' : legendHtml([...p, 'legend'], null, 'job', ctx)) +
        (has('style') ? '' : styleHtml([...p, 'style'], null, ctx));
    }
    if (p.length === 2 && p[0] === 'faceting' && typeof p[1] === 'number') {
      return (has('export') ? '' : exportHtml([...p, 'export'], null, 'facet')) +
        (has('legend') ? '' : legendHtml([...p, 'legend'], null, 'facet', ctx)) +
        (has('text_rules') ? '' : textRulesHtml([...p, 'text_rules'], null, 'facet'));
    }
    if (p.length === 4 && p[0] === 'faceting' && p[2] === 'panels' && typeof p[3] === 'number') {
      return GF().panel.filter(f => !has(f.k)).map(f => fieldHtml(f.k, [...p, f.k], f)).join('');
    }
    if (p.length === 3 && p[0] === 'jobs' && p[2] === 'array_epoch') {
      const ae = GF().arrayEpoch;
      const modeField = has('mode') ? '' : fieldHtml('mode', [...p, 'mode'], { t: 'enum', o: ae.modes, d: ae.default }, null, { pick: true });
      const mode = valueAt([...p, 'mode']).text || 'latest';
      const need = Object.keys(NEEDS_MODE).find(k => NEEDS_MODE[k] === mode && k !== 'summary');
      return modeField + (need && !has(need) ? fieldHtml(need, [...p, need], { t: need === 'epoch' ? 'int' : 'list', d: need === 'epochs' ? '1, 50, 100' : '' }) : '');
    }
    return '';
  }

  /** The drafts (keys the file lacks) as edits of the file: a set where the key is there by now, a put
   *  where a key or block is new (a list of rules: put [] then add each), an add for a new text rule. */
  /** The paths of every null leaf of a block (an empty value of the blueprint), to write it empty. */
  const nullLeaves = (v, path) => (v === null ? [path]
    : v && typeof v === 'object' && !Array.isArray(v) ? Object.entries(v).flatMap(([k, x]) => nullLeaves(x, [...path, k])) : []);
  function draftOps(tree, drafts) {
    const LEAF = '\u0001';
    const RULE = '\u0000';
    const root = {};
    for (const d of drafts) {
      let o = root;
      for (const seg of d.path) { const k = isRule(seg) ? RULE + seg.rule : String(seg); o = o[k] || (o[k] = {}); }
      o[LEAF] = d.value;
    }
    const plainMap = o => Object.fromEntries(Object.keys(o).filter(k => k !== LEAF).map(k => [k, plain(o[k])]));
    function plain(o) {
      if (LEAF in o) return o[LEAF];
      const keys = Object.keys(o);
      if (keys.length && keys.every(k => k[0] === RULE)) return keys.map(k => ({ target: k.slice(1), ...plainMap(o[k]) }));
      return plainMap(o);
    }
    const text = v => (v === null ? '' : JSON.stringify(v));   // JSON: a drafted text stays text (yamledit.render reads the panel's JSON as JSON); null: an empty value
    const ops = [];
    (function walk(node, path, o) {
      for (const k of Object.keys(o)) {
        if (k === LEAF) continue;
        if (k[0] === RULE) {
          const i = node && node.k === 'list' ? node.items.findIndex(it => ruleTarget(it) === k.slice(1)) : -1;
          if (i >= 0) walk(node.items[i], [...path, i], o[k]);
          else ops.push({ op: 'add', path, value: JSON.stringify({ target: k.slice(1), ...plainMap(o[k]) }) });
          continue;
        }
        const seg = node && node.k === 'list' ? Number(k) : k;
        const child = childOf(node, seg);
        const value = plain(o[k]);
        if (child && !(LEAF in o[k])) walk(child, [...path, seg], o[k]);
        else if (child) ops.push({ op: 'set', path: [...path, seg], value: text(value) });
        else if (!(LEAF in o[k]) && value && typeof value === 'object' && !Array.isArray(value)) {
          // drafted keys under a block the file lacks: the block (an empty value null), then each empty value written empty
          ops.push({ op: 'put', path, key: k, value: JSON.stringify(value) });
          for (const leaf of nullLeaves(value, [...path, k])) ops.push({ op: 'set', path: leaf, value: '' });
        }
        else if (Array.isArray(value) && value.length && value.every(x => x && typeof x === 'object' && !Array.isArray(x))) {
          ops.push({ op: 'put', path, key: k, value: '[]' });
          for (const x of value) ops.push({ op: 'add', path: [...path, k], value: JSON.stringify(x) });
        } else ops.push({ op: 'put', path, key: k, value: text(value) });
      }
    }(tree, [], root));
    return ops;
  }

  // ── the status page (the share icon, top right): a read-only view of the runs for phones, and links to share ─
  let statusPage = null;                                  // the host's answer: {running, enabled, why, host, port, auto, shares, note, error}
  /** The title line's button (package.json traceflow.share): the status page's settings, open or closed. */
  function toggleShare() {
    ui.share = !ui.share;
    save();
    if (ui.share) send({ type: 'status-open' });
    if (last) main(last);
  }
  function shareHtml() {
    if (!ui.share) return '';
    const s = statusPage;
    if (!s) return '<div class="share"><p class="note">Reading the status page…</p></div>';
    if (!s.enabled && !s.running) {                       // master.yaml status_share: absent or off (optional)
      return `<div class="share"><h3>Status page</h3><p class="note">Off: ${esc(s.why)}.</p></div>`;
    }
    // no address shown here (only inside each person's link): the port says enough
    const state = s.running
      ? `<p class="serving"><span class="light green"></span>serving on port ${esc(s.port)} ` +
        `${btn('data-statusstop="1"', 'Stop', `stop serving (tmux session ${s.session}); the links stop answering until Start`)}</p>`
      : `<p class="serving"><span class="light grey"></span>not serving ` +
        `${btn('data-statusstart="1"', 'Start', `serve the page on port ${s.port}: make status_share in tmux session ${s.session}, which keeps serving when VS Code closes`)}</p>`;
    // the port (master.yaml status_share.port), read-only here; the address is not shown
    const address = `<div class="field"><span class="lbl">Port</span><span class="readonly" title="master.yaml status_share.port: edit it there by hand">` +
      `${esc(s.port)}</span></div>`;
    // one panel per person: its heading (name, date, × at the right end) and, opened, only their QR code and link
    const people = (s.shares || []).length ? s.shares.map(p => `<details class="person" data-key="${esc(`share:${p.name}`)}"${isOpen(`share:${p.name}`, true) ? ' open' : ''}>` +
      `<summary><b>${esc(p.name)}</b><small>${esc(String(p.created || '').slice(0, 10))}</small>` +
      `${btn(`class="x" data-statusremove="${esc(p.name)}"`, '×', `withdraw ${p.name}'s link: it stops working at once (asks first)`)}</summary>` +
      `${p.qr ? `<div class="qr" title="${esc(`${p.name}'s link: scan it with the phone's camera`)}">${p.qr}</div>` : ''}` +
      // the link behind words, never shown (its address and token stay out of sight): click opens it, ⧉ copies it
      `<div class="linkrow">${btn(`class="link" data-statusvisit="${esc(p.link)}"`, 'Click to share or open in external browser',
        `open ${p.name}'s link in your browser (VS Code forwards the port over this session); ⧉ copies it, to send it`)}` +
      `${btn(`class="copy" data-statuscopy="${esc(p.link)}" data-name="${esc(p.name)}"`, '⧉', `copy ${p.name}'s link, to send it`)}</div></details>`).join('')
      : '<p class="note">No link yet: add a person below (yourself too).</p>';
    const add = `<div class="field addperson"><input class="add wide" data-statusname="1" placeholder="a name, e.g. Professor">` +
      `${btn('data-statusadd="1"', 'Add a person', 'a link and QR code of their own, withdrawn on its own')}</div>`;
    const help = [['Open', [`needs: the same network as the TraceFlow repo host machine, or its VPN with port ${s.port} allowed; nothing else reaches it`,
        'without a VPN: the phone on the same Wi-Fi or network as the TraceFlow repo host machine; scan the QR code or open the link',
        'with a VPN: turn the VPN on, then scan the QR code or open the link']],
      ['Home screen', 'iPhone: Safari or Chrome → Share → Add to Home Screen (Chrome: iOS 16.4 or later). Android: Chrome → ⋮ → Add to Home screen.'],
      ['Read-only', 'only observation of the system status, no interaction.'],
      ['Links', 'one per person (secrets/status_shares.json, not in git); × withdraws it at once.'],
      ['Not opening?', [        'the device is on the TraceFlow repo host machine\'s network: same Wi-Fi, or VPN on',
        `the firewall lets TCP ${s.port} in (Ubuntu: sudo ufw allow ${s.port}/tcp)`,
        `a VPN that passes only SSH: request IT admin to allow ${s.port}`,
        'guide: services/doc/11_phone_status_share.md']]];     // checked in this order
    return `<div class="share"><h3>Status page</h3>${state}` +
      `${s.note ? `<p class="${/^refused|did not/.test(s.note) ? 'err' : 'note'}">${esc(s.note)}</p>` : ''}${s.error ? `<p class="err">${esc(s.error)}</p>` : ''}` +
      `${address}<div class="people">${people}</div>${add}<ul class="share-help">${help.map(([k, t]) => `<li><b>${esc(k)}:</b> ` +
        `${Array.isArray(t) ? `<ol>${t.map(x => `<li>${esc(x)}</li>`).join('')}</ol>` : esc(t)}</li>`).join('')}</ul></div>`;
  }

  /** OK: writes the staged changes of the form (prefix null) or of the block at prefix. */
  function applyStaged(form, prefix) {
    const file = form.dataset.file;
    const p = pending[file];
    if (!p) return;
    const ids = Object.keys(p.edits).filter(id => !prefix || within(p.edits[id].path, prefix));
    if (!ids.length) return;
    const staged = ids.map(id => p.edits[id]);
    const edits = staged.filter(e => e.op !== 'draft').concat(draftOps(trees[file] && trees[file].tree, staged.filter(e => e.op === 'draft')));
    for (const id of ids) delete p.edits[id];
    const sha = p.sha;
    if (!Object.keys(p.edits).length) delete pending[file];
    if (trees[file]) trees[file].note = 'saving…';
    send({ type: 'apply', file, sha, edits });
  }

  function main(msg) {
    last = msg;
    if (!msg.projects) {
      return draw(msg.error ? `<p class="err">${esc(msg.error)}</p><button data-retry="1">Retry</button>`
        : '<p class="note">Reading the projects…</p>');
    }
    const orphans = msg.orphans || [];
    const shape = msg.projects.map(p => p.name).concat(orphans.map(o => `orphan:${o.name}`)).join('\n');
    if (!regions || skeleton !== shape) {
      drawn = '';
      skeleton = shape;
      app.innerHTML = '<div id="share"></div><div id="gpu" class="gpu"></div><div id="top"></div><div id="master"></div>' + msg.projects.map((p, i) =>
        `<details class="project" data-key="p:${esc(p.name)}"${isOpen(`p:${p.name}`, true) ? ' open' : ''}>` +
        `<summary id="p${i}-sum"></summary>` +
        ['run', 'tiles', 'queues', 'single', 'figures', 'tests', 'configs', 'housekeeping', 'errors'].map(s => `<div id="p${i}-${s}"></div>`).join('') +
        '</details>').join('') +
        orphans.map((o, i) => `<details class="project orphan" data-key="o:${esc(o.name)}"${isOpen(`o:${o.name}`, true) ? ' open' : ''}>` +
          `<summary id="o${i}-sum"></summary><div id="o${i}-body"></div></details>`).join('') +
        '<div id="snapshots"></div>' + (msg.projects.length || orphans.length ? '' : '<p class="note">No project in master.yaml.</p>');
      regions = {};
      for (const el of app.querySelectorAll('[id]')) regions[el.id] = { el, html: null };
    }
    const busy = (msg.busy || []).length
      ? `<p class="busy" title="${esc(msg.busy.join('\n'))}"><span class="light green"></span>running here: ${esc(msg.busy[0])}${msg.busy.length > 1 ? ` (+${msg.busy.length - 1})` : ''}</p>` : '';
    const parts = {
      share: shareHtml(),
      gpu: gpuHtml(msg.gpu),
      top: busy + [msg.error, ...(msg.errors || [])].filter(Boolean).map(e => `<p class="err">${esc(e)}</p>`).join('') +
        (msg.error ? '<button data-retry="1">Retry</button>' : '') +
        // the running build (extension.js build: version and panel.js digest): after a reinstall and Developer: Reload
        // Window this must change; the same stamp means the window still runs the old panel
        (msg.build ? `<small class="note build" title="the TraceFlow Monitor running in this window: its version and panel.js digest; a reinstall (make panel) followed by Developer: Reload Window changes it">build ${esc(msg.build)}</small>` : ''),
    };
    parts.snapshots = snapshotsHtml(msg.snapshots);
    parts.master = masterHtml(msg.master);
    orphans.forEach((o, i) => {                            // a project folder master.yaml does not declare
      parts[`o${i}-sum`] = `<span class="light red" title="not in master.yaml"></span><b>${esc(o.name)}</b> <small>master.yaml integration missing</small>`;
      const state = v => (v ? 'present' : 'deleted');
      parts[`o${i}-body`] = `<p class="note">projects/${esc(o.name)} is not a project of master.yaml (its entry was removed, or never written). ` +
        `Project root: ${state(o.present)} · config: ${state(o.config)} · artifacts: ${state(o.artifacts)}.</p>` +
        `<div class="buttons">${btn(`data-declare="${esc(o.name)}"`, 'Add to master.yaml',
          `a master.yaml entry named ${o.name} (blueprint/master_project.yaml); its config folder keeps what it has, the blueprint fills the rest`)}</div>`;
    });
    msg.projects.forEach((p, i) => {
      parts[`p${i}-sum`] = summaryHtml(p);
      parts[`p${i}-run`] = runHtml(p);
      parts[`p${i}-tiles`] = tilesHtml(p);
      parts[`p${i}-queues`] = queuesHtml(p);
      parts[`p${i}-single`] = singleRunsHtml(p);
      parts[`p${i}-figures`] = figuresHtml(p);
      parts[`p${i}-tests`] = testsHtml(p);
      parts[`p${i}-configs`] = configsHtml(p);
      parts[`p${i}-housekeeping`] = housekeepingHtml(p);
      parts[`p${i}-errors`] = p.errors.length ? `<ul class="errs">${p.errors.map(e => `<li>${esc(e)}</li>`).join('')}</ul>` : '';
    });
    drawRegions(parts);
    requestTrees();
    requestChoices();
    reportStaged();
  }

  /** Each config file's number of staged changes, sent when it changes (extension.js refuses the planner's
   *  Add the Model over them: the write would drop them). */
  const stagedSent = {};
  function reportStaged() {
    for (const file of new Set([...Object.keys(stagedSent), ...Object.keys(pending)])) {
      const edits = pending[file] ? Object.values(pending[file].edits) : [];
      const count = edits.length;
      const arranged = edits.filter(e => e.op === 'arrange').length;   // the planner's (Visualise), which its own OK writes
      if (stagedSent[file] !== `${count}/${arranged}`) { stagedSent[file] = `${count}/${arranged}`; send({ type: 'staged', file, count, arranged }); }
    }
  }

  /** Asks once for the metric menu of every opened Done run that has none yet (a failed one again when reopened). */
  const askedChoices = new Set();
  function requestChoices() {
    for (const el of app.querySelectorAll('details.done-run[data-want]')) {
      const id = `${el.dataset.want}|${el.dataset.name}`;
      if (!askedChoices.has(id)) { askedChoices.add(id); send({ type: 'want-choices', project: el.dataset.want, name: el.dataset.name }); }
    }
  }

  function detail(msg) {
    if (msg.missing) return draw(`<p class="note">${esc(msg.missing)}</p>`);
    const r = msg.run;
    const facts = [['status', r.status], ['model', r.model], ['env', r.env], ['queues', r.batches.join(', ') || '-'],
      ['run', r.trials ? `${r.trial}/${r.trials}` : '-'], ['started', r.started || '-'], ['finished', r.finished || '-'],
      ['reason', r.reason || '-']];
    const live = msg.live
      ? `<div class="run"><span class="light ${esc(msg.live.light.color)}"></span> ${esc(msg.live.phase)}<div class="progress">${bar(msg.live.percent)}</div></div>` : '';
    const rows = msg.series.map(s =>
      `<tr><td class="name" title="${esc(s.label)}">${esc(s.label)}</td><td>${spark(s.spark, 160, 24)}</td>` +
      `<td class="num">${num(s.last)}</td><td class="num">${s.n} ep</td><td>${btn(`data-reveal="${esc(s.dir)}"`, 'Folder', s.dir)}</td></tr>`).join('');
    draw(`<h2>${esc(r.name)} <small>${esc(r.hash)} · ${esc(msg.project)}</small></h2>` + live +
      `<div class="kv">${facts.map(([k, v]) => `<span>${k}</span><span>${esc(v)}</span>`).join('')}</div>` +
      `<div class="buttons">${btn(`data-reveal="${esc(r.dir)}"`, 'Run folder')}</div>` +
      `<h3>Curves <small>value at the last row, per epoch</small></h3>` +
      (rows ? `<table class="series">${rows}</table>` : '<p class="note">no metrics written yet</p>'));
  }

  /** An Add button: its row of the name to type opens under it, the cursor in its box. */
  function openAsk(kind, file, project) {
    ui.ask = { kind, file, project, value: '', modelKind: kind === 'model' ? modelKinds()[0] || '' : '' };
    if (last) main(last);
    const box = document.querySelector ? document.querySelector('[data-askinput]') : null;
    if (box && box.focus) box.focus();
  }
  /** Add (or Enter): the name checked again (its form, not taken), then written: the extension adds a project,
   *  a queue, a job, a facet or a model under it; a data entry is the blueprint's entry, written at once. */
  function submitAsk() {
    const a = ui.ask;
    if (!a) return;
    const name = String(a.value || '').trim();
    if (askErr(a.kind, a.file, name)) { if (last) main(last); return; }
    const t = trees[a.file] || {};
    if (a.kind === 'dataset' && pending[a.file]) {
      t.note = 'refused: OK, Cancel or Reset the staged changes first, then add the dataset';
      if (last) main(last);
      return;
    }
    if (a.kind === 'project') send({ type: 'add-project', name });
    else if (a.kind === 'queue') send({ type: 'add-queue', project: a.project, file: a.file, sha: t.sha256, keys: takenNames('queue', a.file), name });
    else if (a.kind === 'job' || a.kind === 'facet') send({ type: `add-${a.kind}`, project: a.project, name });
    else if (a.kind === 'model') {
      send({ type: 'add-model', project: a.project, file: a.file, sha: t.sha256, ids: takenNames('model', a.file), name,
        kind: a.modelKind || modelKinds()[0], kinds: (MF() || {}).kinds || [], named: kerasRun() ? [] : (MF() || {}).named || [] });
    } else if (a.kind === 'dataset') {
      const list = resolveIn(t.tree, ['data']).node;
      ui.open = { ...(ui.open || {}), [nodeKey(a.file, ['data', list && list.k === 'list' ? list.items.length : 0])]: true };
      save();
      send({ type: 'edit', file: a.file, sha: t.sha256, op: 'add', path: ['data'], value: JSON.stringify({ ...MENUS.params.data_template, name }) });
    }
    ui.ask = null;
    if (last) main(last);
  }
  // typing a name: its check under the box at once (no redraw: the cursor stays); Enter adds, Escape closes
  app.addEventListener('input', e => {
    const el = e.target;
    // a Budget box: staged at once, its marks and the bars of its params.yaml updated in place (the region holding the
    // caret is redrawn only after it, owner 2026-10-28)
    if (el.dataset && el.dataset.budget && el.dataset.op) {
      stage(el, el.dataset.op, el.value);
      const form = el.closest('.form');
      const file = form && form.dataset.file;
      const id = el.dataset.op === 'put' ? editId('put', JSON.parse(el.dataset.path), el.dataset.key) : editId('set', JSON.parse(el.dataset.path));
      const staged = !!stagedOf(file)[id];
      el.classList.toggle('dirty', staged);
      const test = form && form.querySelector('button[data-kind="budget"]');
      if (test) test.disabled = staged;
      const touched = [el];
      for (const slot of app.querySelectorAll('.budget-bar')) {
        if (slot.dataset.file === file) { slot.innerHTML = budgetBarInner(file, slot.dataset.project); touched.push(slot); }
      }
      // the regions changed in place no longer hold the html they last drew: the next redraw draws them whatever it is
      // (else Reset, whose html is the one drawn before the typing, was taken for no change)
      for (const r of Object.values(regions || {})) if (touched.some(n => r.el.contains(n))) r.html = null;
      return;
    }
    if (!el.dataset || !el.dataset.askinput || !ui.ask) return;
    ui.ask.value = el.value;
    const err = askErr(ui.ask.kind, ui.ask.file, el.value);
    const row = el.closest('[data-askrow]');
    const msg = row && row.querySelector('.askerr');
    const ok = row && row.querySelector('[data-askok]');
    if (msg) msg.textContent = el.value ? err : window.TFMenus.ASKS[ui.ask.kind].hint;
    if (ok) ok.disabled = !!err;
    if (el.classList) el.classList.toggle('bad', !!(el.value && err));
  });
  app.addEventListener('keydown', e => {
    const el = e.target;
    if (el.dataset && el.dataset.cfgsearch && e.key === 'Enter') {     // out of the box first: a region holding focus is not redrawn
      e.preventDefault();
      if (el.blur) el.blur();
      startSearch(el.dataset.cfgsearch, el.value);
      return;
    }
    if (el.dataset && el.dataset.statusname && e.key === 'Enter' && String(el.value || '').trim()) {   // Enter: Add a person
      e.preventDefault();
      send({ type: 'status-add', name: String(el.value).trim() });
      return;
    }
    if (!el.dataset || !el.dataset.askinput || !ui.ask) return;
    if (e.key === 'Enter') { e.preventDefault(); ui.ask.value = el.value; submitAsk(); }
    else if (e.key === 'Escape') { ui.ask = null; if (last) main(last); }
  });

  app.addEventListener('pointerdown', e => { pressed = e.target; }, true);
  // after the click (it runs on this release, before the timer): the redraw the press held back
  window.addEventListener('pointerup', () => {
    if (!pressed) return;
    setTimeout(() => {
      pressed = null;
      if (drawAfterPress) { drawAfterPress = false; if (mode === 'main' && last) main(last); }
    }, 0);
  }, true);
  app.addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.closest('summary')) e.preventDefault();          // a button in a summary line acts; it does not fold
    const d = b.dataset;
    if (d.cfgsearchclear) { endSearch(d.cfgsearchclear); return; }
    if (d.statusstart || d.statusstop) { send({ type: d.statusstart ? 'status-start' : 'status-stop' }); return; }
    if (d.statusremove) { send({ type: 'status-remove', name: d.statusremove }); return; }
    if (d.statuscopy) { send({ type: 'status-copy', name: d.name, link: d.statuscopy }); return; }
    if (d.statusvisit) { send({ type: 'status-visit', link: d.statusvisit }); return; }
    if (d.statusadd) {
      const box = app.querySelector('[data-statusname]');
      const name = box ? String(box.value || '').trim() : '';
      if (name) send({ type: 'status-add', name });
      return;
    }
    if (d.snap) {
      let filters;
      if (d.snap === 'scan') {                                   // the filters of this sub-panel, as typed now
        filters = {};
        for (const el of app.querySelectorAll('input[data-snapfilter]')) {
          const [scope, key] = el.dataset.snapfilter.split('|');
          if (scope === d.scope) filters[key] = el.value;
        }
        ui.snapf = ui.snapf || {};
        ui.snapf[d.scope] = filters;
        save();
      }
      send({ type: 'snap', action: d.snap, scope: d.scope, path: d.path, filters });
    } else if (d.snapclear) {
      if (ui.snapf) delete ui.snapf[d.snapclear];
      save();
      if (last) main(last);
    } else if (d.figdel) {
      send({ type: 'figdel', project: d.figdel, dir: d.dir });
    } else if (d.figdelall) {
      send({ type: 'figdel-all', project: d.figdelall, dirs: JSON.parse(d.dirs || '[]') });
    } else if (d.jobdel) {
      send({ type: 'jobdel', project: d.jobdel, id: d.id });
    } else if (d.facetdel) {
      send({ type: 'facetdel', project: d.facetdel, id: d.id });
    } else if (d.facetexport) {
      const p = ((last && last.projects) || []).find(x => x.name === d.facetexport);
      send({ type: 'run', action: 'facet-export', project: d.facetexport, files: p ? facetFiles(p) : [] });
    } else if (d.facdrawall) {
      const p = ((last && last.projects) || []).find(x => x.name === d.facdrawall);
      if (p && facSel(p).length) send({ type: 'run', action: 'draw-all-facets', project: d.facdrawall, ids: facSel(p) });
    } else if (d.drawall) {
      const p = ((last && last.projects) || []).find(x => x.name === d.drawall);
      if (p && figSel(p).length) send({ type: 'run', action: 'draw-all', project: d.drawall, ids: figSel(p) });
    } else if (d.facetclear) {
      ui.figsel = { ...(ui.figsel || {}), [d.facetclear]: [] };
      save();
      if (last) main(last);
    } else if (d.gexport) {
      send({ type: 'run', action: 'graph-export', project: d.gexport, picks: [{ name: d.name, key: d.mkey }] });
    } else if (d.gremove) {
      send({ type: 'run', action: 'graph-remove', project: d.gremove, name: d.name, key: d.mkey });
    } else if (d.findfig) {
      const form = b.closest('.form');
      if (form) send({ type: 'find-figure', file: form.dataset.file, project: form.dataset.project, path: d.findfig });
    } else if (d.draftdrop) {                                // a drafted list (export.formats not in the file yet): drop one
      const form = b.closest('.form');
      if (form) setDraft(form, JSON.parse(d.draftdrop), JSON.parse(d.list).filter(x => x !== d.item));
      if (last) main(last);
    } else if (d.tablepick) {                                // facet layout: rows x columns, as Word inserts a table
      const [r, c] = d.tablepick.split('x').map(Number);
      setTable(b.closest('.form'), JSON.parse(d.path), r, c);
      if (last) main(last);
    } else if (d.delclear) {
      send({ type: 'delfilter', project: d.delclear, filters: {} });
    } else if (d.deldb) {
      const p = ((last && last.projects) || []).find(x => x.name === d.deldb);
      if (p) send({ type: 'run', action: 'delete-database', project: p.name, hashes: [...delSel(p)] });
    } else if (d.run) {
      const project = d.formProject ? ((b.closest('.form') || {}).dataset || {}).project : d.project;
      send({ type: 'run', action: d.run, project, key: d.key, id: d.id, env: d.env, kind: d.kind });
    } else if (d.budgetreset) {                             // the Budget boxes' changes undone, the file's other ones kept
      const form = b.closest('.form');
      const p = form && pending[form.dataset.file];
      if (!p) return;
      for (const [id, e] of Object.entries(p.edits)) if (isBudgetEdit(e)) delete p.edits[id];
      if (!Object.keys(p.edits).length) delete pending[form.dataset.file];
      if (last) main(last);
    } else if (d.apply || d.discard || d.reset || d.unstage) {
      const form = b.closest('.form');
      if (!form) return;
      const file = form.dataset.file;
      if (d.apply) applyStaged(form, d.path ? JSON.parse(d.path) : null);
      else if (d.discard || d.reset) {                    // Reset: the changes undone; Cancel: undone, the form closed
        delete pending[file];
        const box = d.discard && form.closest('details[data-key]');
        if (box) {
          ui.open[box.dataset.key] = false;
          save();
          const shownElsewhere = [...app.querySelectorAll('details.rungraph[open], .budget-form[data-file]')].some(g => g !== box && g.dataset.file === file);
          if (box.dataset.key.startsWith('cf:') && !shownElsewhere) send({ type: 'tree', file, open: false });
        }
      }
      else if (pending[file]) {
        delete pending[file].edits[d.unstage];
        if (!Object.keys(pending[file].edits).length) delete pending[file];
      }
      if (last) main(last);
    } else if (d.qadd) { stage(b, 'add', d.value); if (last) main(last); }
    else if (d.nodraft) { ui.noDraft = { ...(ui.noDraft || {}), [d.nodraft]: true }; if (last) main(last); }
    else if (d.addclose) { ui.adding = {}; if (last) main(last); }
    else if (d.metricadd) {                                 // metrics_to_log [+]: the chosen metric, with its value when it takes one
      const name = (ui.metricPick || {})[d.metricadd] || '';
      if (!name) return;
      const spec = MO()[name];
      const v = spec ? Number((ui.metricPickOpt || {})[d.metricadd] != null ? ui.metricPickOpt[d.metricadd] : spec.default) : null;
      if (spec && !Number.isFinite(v)) return;
      stage(b, 'add', spec ? JSON.stringify({ [name]: { [spec.option]: v } }) : name);
      ui.metricPick = {};
      ui.metricPickOpt = {};
      if (last) main(last);
    }
    else if (d.addoptimizer != null && d.addoptimizer !== '') {   // Add an Optimizer: a block written at once, opened
      const form = b.closest('.form');
      if (!form) return;
      if (pending[form.dataset.file]) {
        if (trees[form.dataset.file]) trees[form.dataset.file].note = 'refused: OK or Cancel the staged changes first, then add the optimizer';
        if (last) main(last);
        return;
      }
      const first = resolve(['trainer', 'optimizer', 0]).node;
      const at = k => (first && first.k === 'map' ? entryOf(first, k) : null);
      const lr = at('learning_rate');
      const rates = lr && lr.k === 'list' ? lr.items.map(x => Number(unquote(x.text))).filter(Number.isFinite)
        : lr && Number.isFinite(Number(unquote(lr.text))) ? [Number(unquote(lr.text))] : [];
      const menu = menuOf(['trainer', 'optimizer', 0, 'type']);
      const type = unquote((at('type') || {}).text || '') || (menu && menu.opts[0]) || 'Adam';
      const block = { type, learning_rate: rates.length ? rates : [1e-5], scheduler: { type: 'none' }, weight_decay: 0.0 };
      const path = JSON.parse(d.path);
      ui.open = { ...(ui.open || {}), [nodeKey(form.dataset.file, [...path, Number(d.addoptimizer)])]: true };
      save();
      send({ type: 'edit', file: form.dataset.file, sha: form.dataset.sha, op: 'add', path, value: JSON.stringify(block) });
    }
    else if (d.adddataset || d.addmodel) {                  // Add a Dataset / a Model: their name asked in the panel
      const form = b.closest('.form');
      if (form) openAsk(d.adddataset ? 'dataset' : 'model', form.dataset.file, form.dataset.project);
    } else if (d.askok) submitAsk();                        // an asked name: Add
    else if (d.askclose) { ui.ask = null; if (last) main(last); }
    else if (d.addsingle) {                                // RUN SINGLE EXPERIMENT: Add a single Run: a line written at once
      const form = b.closest('.form');
      const file = d.file || (form && form.dataset.file);
      const sha = d.sha || (form && form.dataset.sha);
      if (!file) return;
      if (pending[file]) {
        if (trees[file]) trees[file].note = 'refused: OK, Cancel or Reset the staged changes first, then add the run';
        if (last) main(last);
      } else send({ type: 'edit', file, sha, op: 'add', path: ['runs'], value: d.addsingle });
    } else if (d.declare) {                                 // an undeclared project folder: its master.yaml entry under its own name
      send({ type: 'add-project', name: d.declare });
    } else if (d.addproject) {                              // MASTER CONTROL → Add a Project: its name asked in the panel
      openAsk('project', d.file, '');
    } else if (d.removeproject) {                           // MASTER CONTROL → a project's ×: the extension asks, then removes it
      const form = b.closest('.form');
      if (form && pending[form.dataset.file]) {
        if (trees[form.dataset.file]) trees[form.dataset.file].note = `refused: OK, Cancel or Reset the staged changes first, then remove ${d.removeproject}`;
        if (last) main(last);
      } else send({ type: 'remove-project', name: d.removeproject, path: JSON.parse(d.path) });
    } else if (d.delentry) {                                // a model's / data entry's ×: the extension asks, then writes it at once
      const form = b.closest('.form');
      if (form && pending[form.dataset.file]) {
        if (trees[form.dataset.file]) trees[form.dataset.file].note = `refused: OK or Cancel the staged changes first, then delete ${d.name}`;
        if (last) main(last);
      } else if (form) send({ type: 'delete-entry', file: form.dataset.file, sha: form.dataset.sha, kind: d.delentry, name: d.name, path: JSON.parse(d.path) });
    } else if (d.importmodel) {                             // Import a Model: the extension asks for the file and its settings (importModel)
      const form = b.closest('.form');
      if (form) send({ type: 'import-model', project: form.dataset.project, file: form.dataset.file });
    } else if (d.planmodel || d.visualise) {                // Plan a Model / Visualise: the planner page (extension.js openPlanner)
      const form = b.closest('.form');
      if (form) send(d.visualise ? { type: 'visualise', project: form.dataset.project, file: form.dataset.file, model_id: d.visualise }
        : { type: 'plan-model', project: form.dataset.project, file: form.dataset.file });
    } else if (d.checkmodel) {
      const form = b.closest('.form');
      if (form && d.checkmodel) send({ type: 'check-model', project: form.dataset.project, model_id: d.checkmodel });
    } else if (d.movelayer) {                               // ▲ ▼: written at once (the file must have no staged change)
      const form = b.closest('.form');
      if (form && pending[form.dataset.file]) {
        if (trees[form.dataset.file]) trees[form.dataset.file].note = 'refused: OK or Cancel the staged changes first, then move the layer';
        if (last) main(last);
      } else if (form) send({ type: 'edit', file: form.dataset.file, sha: form.dataset.sha, op: 'move', path: JSON.parse(d.path), value: d.movelayer });
    } else if (d.setto) { stage(b, 'set', d.setto); if (last) main(last); }
    else if (d.addopen) {                                   // [+]: one control for the new value, focused
      ui.adding = { [d.addopen]: true };
      if (last) main(last);
      const box = document.querySelectorAll ? [...document.querySelectorAll('[data-op="add"]')].find(x => x.dataset.path === d.addopen) : null;
      if (box) box.focus();
    }
    else if (d.addjob || d.addfacet) {                      // Add a Job / a Facet: its name asked in the panel
      const form = b.closest('.form');
      if (form) openAsk(d.addjob ? 'job' : 'facet', form.dataset.file, d.addjob || d.addfacet);
    }
    else if (d.findmodels) send({ type: 'find-models', project: d.findmodels, where: (filterDraft[d.file] || []).map(mfToRaw) });
    else if (d.mfreset) { delete filterDraft[d.mfreset]; if (last) main(last); }
    else if (d.pickpath) {                                  // an address: picked in the system, staged into its box
      const form = b.closest('.form');
      if (form) {
        send({ type: 'pick-path', file: form.dataset.file, project: form.dataset.project, path: JSON.parse(d.pickpath),
          base: d.base, start: d.start, kind: d.kind, cur: d.cur });
      }
    }
    else if (d.entryrow) { ui.facetRows = { ...(ui.facetRows || {}), [d.entryrow]: ((ui.facetRows || {})[d.entryrow] || 0) + 1 }; save(); if (last) main(last); }
    else if (d.addpanel) { stage(b, 'add', JSON.stringify({ file_id: Number(d.next), address: '' })); if (last) main(last); }
    else if (d.cpick) {                                       // a colour: open or close its palette
      ui.cpick = { [d.cpick]: !((ui.cpick || {})[d.cpick]) };
      if (last) main(last);
    } else if (d.cmorebtn) {                                 // More Colors…: the honeycomb under the palette
      ui.cmore = { [d.cmorebtn]: !((ui.cmore || {})[d.cmorebtn]) };
      if (last) main(last);
    } else if (d.honey) {                                   // a hexagon of the honeycomb
      const hex = e.target && e.target.getAttribute ? e.target.getAttribute('data-hex') : null;
      const form = b.closest('.form');
      if (hex && form && d.path) stage(b, 'set', hex);
      else if (hex && form && d.cdraft) setDraft(form, JSON.parse(d.cdraft), hex);
      if (hex) { ui.cpick = {}; ui.cmore = {}; }
      if (last) main(last);
    } else if (d.cset) {                                      // a colour picked from the palette
      const form = b.closest('.form');
      if (form && d.path) stage(b, 'set', d.cset);
      else if (form && d.cdraft) setDraft(form, JSON.parse(d.cdraft), d.cset);
      ui.cpick = {};
      ui.cmore = {};
      if (last) main(last);
    } else if (d.clear) {                                     // a text emptied
      const form = b.closest('.form');
      if (form && d.path) stage(b, 'set', '');
      else if (form && d.draftpath) setDraft(form, JSON.parse(d.draftpath), undefined);
      if (last) main(last);
    }
    else if (d.addqueue) openAsk('queue', d.file, d.addqueue);   // Add a Queue: its key asked in the panel
    else if (d.op) { stage(b, d.op); if (last) main(last); }
    else if (d.retry) send({ type: 'retry' });
    else if (d.docker) send({ type: d.docker === 'kill-build' ? 'docker-kill-build' : 'docker-teardown' });
    else if (d.detail) send({ type: 'detail', project: d.project, hash: d.hash });
    else if (d.open) send({ type: 'open', path: d.open });
    else if (d.complete) {                                  // Load Blueprint: a warning first (extension.js completeAsk)
      send({ type: 'complete-ask', file: d.complete, unsaved: Object.keys(stagedOf(d.complete)).length });
    }
    else if (d.setconfig) {                                 // QUEUES → Set Params / Set Queue: that form, opened
      ui.open = { ...(ui.open || {}), [`c:${d.project}`]: true, [`cf:${d.setconfig}`]: true };
      requested.delete(d.setconfig);
      save();
      if (last) main(last);
      const box = [...app.querySelectorAll('details.cfg')].find(x => x.dataset.file === d.setconfig);
      if (box && box.scrollIntoView) box.scrollIntoView({ block: 'start' });
    }
    else if (d.reveal) send({ type: 'reveal', path: d.reveal });
    else if (d.group) {
      ui.group[d.project] = ui.group[d.project] === d.group ? '' : d.group;
      save();
      if (last) main(last);
    }
  });
  // The layout picker lights the rows x columns under the pointer, as Word's Insert Table does.
  app.addEventListener('mouseover', e => {
    const b = e.target.closest && e.target.closest('[data-tablepick]');
    if (!b) return;
    const [r, c] = b.dataset.tablepick.split('x').map(Number);
    for (const x of b.closest('.tp-grid').querySelectorAll('[data-tablepick]')) {
      const [i, j] = x.dataset.tablepick.split('x').map(Number);
      x.classList.toggle('hover', i <= r && j <= c);
    }
  });
  app.addEventListener('mouseout', e => {
    const grid = e.target.closest && e.target.closest('.tp-grid');
    if (grid && !grid.contains(e.relatedTarget)) for (const x of grid.querySelectorAll('.hover')) x.classList.remove('hover');
  });
  app.addEventListener('change', e => {
    const el = e.target;
    if (el.dataset && (el.dataset.listpick || el.dataset.listsort || el.dataset.listdir)) {   // [Done] / [Failed]: filter, sort
      const key = listKey(el.dataset.listpick || el.dataset.listsort || el.dataset.listdir, el.dataset.project);
      if (el.dataset.listpick) (ui.listPick = ui.listPick || {})[key] = el.value;
      else {
        const s = { ...((ui.listSort = ui.listSort || {})[key] || {}) };
        s[el.dataset.listsort ? 'key' : 'dir'] = el.value;
        ui.listSort[key] = s;
      }
      save();
      if (last) main(last);
      return;
    }
    if (el.dataset && (el.dataset.askkind || el.dataset.askinput)) {   // an asked name / a model's kind: kept for Add
      if (ui.ask && el.dataset.askkind) ui.ask.modelKind = el.value;
      if (ui.ask && el.dataset.askinput) ui.ask.value = el.value;
      return;
    }
    if (el.dataset && (el.dataset.figpick || el.dataset.figall)) {   // FIGURES: tick a figure, or Select all
      const project = el.dataset.figpick || el.dataset.figall;
      const p = ((last && last.projects) || []).find(x => x.name === project);
      if (!p) return;
      let picks = figSel(p).filter(id => id !== el.dataset.id);
      if (el.dataset.figall) picks = el.checked ? p.figures.map(f => String(f.id)) : [];
      else if (el.checked) picks.push(el.dataset.id);
      ui.figsel = { ...(ui.figsel || {}), [project]: picks };
      save();
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset && (el.dataset.facpick || el.dataset.facall)) {   // FIGURES → Faceting: tick a facet, or Select all
      const project = el.dataset.facpick || el.dataset.facall;
      const p = ((last && last.projects) || []).find(x => x.name === project);
      if (!p) return;
      let picks = facSel(p).filter(id => id !== el.dataset.id);
      if (el.dataset.facall) picks = el.checked ? (p.facets || []).map(f => String(f.id)) : [];
      else if (el.checked) picks.push(el.dataset.id);
      ui.facsel = { ...(ui.facsel || {}), [project]: picks };
      save();
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset && el.dataset.gmetric) {                 // Done list: the metric picked under a run
      const project = el.dataset.gmetric;
      ui.gmetric = { ...(ui.gmetric || {}), [project]: { ...((ui.gmetric || {})[project] || {}), [el.dataset.name]: el.value } };
      save();
      el.blur();
      if (last) main(last);
      return;
    }
    const gform = el.closest && el.closest('.form');
    if (gform && el.dataset && (el.dataset.draft || el.dataset.draftlist || el.dataset.draftmap)) {   // a key the file lacks
      if (el.dataset.draft) {
        const back = el.dataset.orig != null && el.dataset.orig !== '' && sameValue(el.value, el.dataset.orig);
        setDraft(gform, JSON.parse(el.dataset.draft), back ? undefined : typed(el.value, el.dataset.t));
        if (isSplitSwitch(JSON.parse(el.dataset.draft))) splitDefaults(gform, JSON.parse(el.dataset.draft), el.value === 'true');
      }
      else if (el.dataset.draftlist && el.value) setDraft(gform, JSON.parse(el.dataset.draftlist), [...JSON.parse(el.dataset.list), el.value]);
      else if (el.dataset.draftmap && el.value.trim()) {
        setDraft(gform, JSON.parse(el.dataset.draftat), [...JSON.parse(el.dataset.list), { [el.dataset.draftmap]: el.value.trim() }]);
      }
      el.blur();
      if (last) main(last);
      return;
    }
    if (gform && el.dataset && el.dataset.fmtcheck) {       // export.formats: a tick writes the whole list
      const was = JSON.parse(el.dataset.list || '[]');
      const order = (GF() ? Object.values(GF().export).flatMap(s => s.formats) : []);
      const next = el.checked ? [...was, el.dataset.fmtcheck] : was.filter(f => f !== el.dataset.fmtcheck);
      if (!next.length) {                                    // graph_manager writes at least one format
        if (trees[gform.dataset.file]) trees[gform.dataset.file].note = 'refused: export.formats needs one format at least';
      } else {
        const list = [...new Set(next)].sort((a, b) => order.indexOf(a) - order.indexOf(b));
        if (el.dataset.fmtdraft) setDraft(gform, JSON.parse(el.dataset.fmtdraft), list);
        else stage(el, 'set', `[${list.join(', ')}]`);
      }
      el.blur();
      if (last) main(last);
      return;
    }
    if (gform && el.dataset && (el.dataset.mfcheck || el.dataset.mfop || el.dataset.mfsel || el.dataset.mfin || el.dataset.mftext || el.dataset.mfbool || el.dataset.mfquant)) {
      const file = gform.dataset.file;
      const data = filterData[gform.dataset.project] || { fields: [] };
      const draft = filterDraft[file] || (filterDraft[file] = []);
      const field = p => data.fields.find(f => f.path === p);
      const i = Number(el.dataset.mfi != null ? el.dataset.mfi : el.dataset.mfop != null ? el.dataset.mfop : el.dataset.mfquant);
      if (el.dataset.mfcheck) {                              // a field ticked: a condition; unticked: gone
        const p = el.dataset.mfcheck;
        const at = draft.findIndex(c => c.path === p);
        if (el.checked && at < 0) {
          const f = field(p);
          const op = ((MF_BY_KIND[(f || {}).kind] || ['equals'])[0]);
          draft.push({ path: p, op, value: mfDefault(f, op) });
        } else if (!el.checked && at >= 0) draft.splice(at, 1);
      } else if (el.dataset.mfop != null) {
        draft[i].op = el.value;
        draft[i].value = mfDefault(field(draft[i].path), el.value);
      } else if (el.dataset.mfquant != null) draft[i].quantifier = el.value;
      else if (el.dataset.mfbool) draft[i].value = el.value === 'true';
      else if (el.dataset.mfsel) {
        const vals = (field(draft[i].path) || {}).values || [];
        if (Number(el.value) >= 0) draft[i].value = vals[Number(el.value)];
      } else if (el.dataset.mfin != null) {
        const f = field(draft[i].path) || { values: [] };
        const have = Array.isArray(draft[i].value) ? draft[i].value : [draft[i].value];
        const all = [...f.values, ...have.filter(v => !f.values.some(w => mfShow(w) === mfShow(v)))];
        const v = all[Number(el.dataset.mfin)];
        const rest = have.filter(w => mfShow(w) !== mfShow(v));
        draft[i].value = el.checked ? all.filter(w => [...rest, v].some(x => mfShow(x) === mfShow(w))) : rest;
      } else if (el.dataset.mftext === 'v') draft[i].value = draft[i].op === 'matches' ? el.value : mfTyped(el.value);
      else if (el.dataset.mftext) {
        const pair = Array.isArray(draft[i].value) ? [...draft[i].value] : ['', ''];
        pair[Number(el.dataset.mftext)] = mfTyped(el.value);
        draft[i].value = pair;
      }
      el.blur();
      if (last) main(last);
      return;
    }
    if (gform && el.dataset && el.dataset.addlayer) {       // [+] of a model's layers: the layer picked
      const spec = el.value && MF() ? MF().layers[el.value] : null;
      if (spec) {
        const args = Object.fromEntries(spec.args.filter(a => a.required).map(a => [a.k, null]));     // to fill
        if (el.value === 'Pooling') Object.assign(args, { dim: 1, type: 'max', kernel_size: null });   // traces are 1-D
        stage(el, 'add', JSON.stringify({ [el.value]: args }));
      }
      ui.adding = {};
      el.blur();
      if (last) main(last);
      return;
    }
    if (gform && el.dataset && el.dataset.addmap) {         // a list of mappings (legend.entries): + add one by its first key
      if (el.value.trim()) { stage(el, 'add', JSON.stringify({ [el.dataset.addmap]: el.value.trim() })); }
      el.blur();
      if (last) main(last);
      return;
    }
    if (gform && el.dataset && el.dataset.handlepart) {     // legend.handle: [shape, thickness]
      const f = el.closest('[data-handle]');
      const path = JSON.parse(f.dataset.handle);
      const real = JSON.parse(f.dataset.real);
      const part = n => (f.querySelector(`[data-handlepart="${n}"]`) || {}).value || '';
      ui.handleChar = { ...(ui.handleChar || {}), [f.dataset.handle]: part('shape') === '__char' };
      save();
      const shape = part('shape') === '__char' ? part('char').trim() : part('shape');
      const size = part('size').trim();
      if (!real && !shape) setDraft(gform, path, undefined);
      else if (shape && size && Number.isFinite(Number(size))) {
        if (real) stageValue(gform, real, 'set', JSON.stringify([shape, Number(size)]));
        else setDraft(gform, path, [shape, Number(size)]);
      }
      el.blur();
      if (last) main(last);
      return;
    }
    if (gform && el.dataset && el.dataset.autopart) {       // style.axes: Auto, or Custom with its range / number
      const f = el.closest('[data-auto]');
      const path = JSON.parse(f.dataset.auto);
      const real = JSON.parse(f.dataset.real);
      const file = gform.dataset.file;
      const edits = () => (pending[file] || { edits: {} }).edits;
      if (el.dataset.autopart === 'mode') {
        ui.custom = { ...(ui.custom || {}), [f.dataset.auto]: el.value === 'custom' };
        save();
        if (real) {
          delete edits()[editId('set', real)];
          if (el.value === 'auto') stageValue(gform, real, 'delete');
          else delete edits()[editId('delete', real)];
          if (pending[file] && !Object.keys(pending[file].edits).length) delete pending[file];
        } else if (el.value === 'auto') setDraft(gform, path, undefined);
      } else {
        const parts = [...f.querySelectorAll('input[data-autopart]')].map(x => x.value.trim());
        if (parts.every(x => x !== '' && Number.isFinite(Number(x)))) {
          const value = f.dataset.pair === '1' ? parts.map(Number) : Number(parts[0]);
          if (real) { delete edits()[editId('delete', real)]; stageValue(gform, real, 'set', JSON.stringify(value)); }
          else setDraft(gform, path, value);
        }
      }
      el.blur();
      if (last) main(last);
      return;
    }
    if (gform && el.dataset && (el.dataset.gridrows || el.dataset.gridcols)) {     // facet layout: the picker grid's size
      const boxes = el.closest('.tp-size').querySelectorAll('input');
      const [rows, cols] = [...boxes].map(x => Math.min(20, Math.max(1, parseInt(x.value, 10) || 1)));
      const path = JSON.parse(el.dataset.path);
      curFile = gform.dataset.file;
      ui.facetGrid = { ...(ui.facetGrid || {}), [tableKey(path)]: { r: rows, c: cols } };
      save();
      const node = trees[curFile] ? window.TFMenus.nodeAt(trees[curFile].tree, path) : null;
      const t = node ? tableOf(path, node) : null;
      if (t && (t.r > rows || t.c > cols)) setTable(gform, path, Math.min(t.r, rows), Math.min(t.c, cols));   // a smaller grid trims the layout
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset && el.dataset.tablecell) {               // facet layout: the panel shown in one box
      setCell(el.closest('.form'), JSON.parse(el.dataset.path), el.dataset.tablecell, el.value);
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset && el.dataset.delfilter) {               // a Housekeeping filter: the extension filters the list
      const [name] = el.dataset.delfilter.split('|');
      const filters = {};
      for (const box of app.querySelectorAll('[data-delfilter]')) {
        const [n, key] = box.dataset.delfilter.split('|');
        if (n === name) filters[key] = box.value;
      }
      el.blur();
      send({ type: 'delfilter', project: name, filters });
      return;
    }
    if (el.dataset && (el.dataset.del || el.dataset.delall)) {   // Housekeeping: tick a run, or Delete all
      const name = el.dataset.del || el.dataset.delall;
      const p = ((last && last.projects) || []).find(x => x.name === name);
      if (!p) return;
      const sel = delSel(p);
      if (el.dataset.delall) { sel.clear(); if (el.checked) for (const r of (p.housekeeping || {}).runs || []) sel.add(r.hash); }
      else if (el.checked) sel.add(el.dataset.hash);
      else sel.delete(el.dataset.hash);
      ui.del = { ...(ui.del || {}), [name]: [...sel] };
      save();
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset && el.dataset.snapfilter) {              // a SNAPSHOTS filter: kept in the view state, applied at once
      const [scope, key] = el.dataset.snapfilter.split('|');
      ui.snapf = ui.snapf || {};
      ui.snapf[scope] = { ...(ui.snapf[scope] || {}), [key]: el.value };
      save();
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset && el.dataset.qvar) {                     // experiment_queue.yaml: one value of a line
      const row = el.closest('.qline');
      if (!row) return;
      const vars = Object.fromEntries([...row.querySelectorAll('select[data-qvar]')].map(s => [s.dataset.qvar, s.value]));
      if (row.dataset.staged) {                              // a line staged for adding: the addition changes
        const form = row.closest('.form');
        const p = form && pending[form.dataset.file];
        if (p && p.edits[row.dataset.staged]) p.edits[row.dataset.staged].value = queueLine(vars);
      } else if (row.dataset.draftline) stage(row, 'add', queueLine(vars));   // an empty queue's row: its line, staged
      else stage(row, 'set', queueLine(vars));
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset && el.dataset.metricopt) {              // metrics_to_log: a written metric's value
      const v = Number(el.value);
      if (el.value.trim() !== '' && Number.isFinite(v)) {
        stage(el, 'set', JSON.stringify({ [el.dataset.metricopt]: { [el.dataset.option]: v } }));
      }
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset.stagedadd || el.dataset.stagedopt) {     // metrics_to_log: a staged addition's name or value changed
      const form = el.closest('.form');
      const p = form && pending[form.dataset.file];
      const e = p && p.edits[el.dataset.stagedadd || el.dataset.stagedopt];
      if (e) {
        if (el.dataset.stagedadd) e.value = el.value;
        else {
          const m = metricOf({ k: 'scalar', text: e.value, path: null }) || { name: e.value };
          const v = Number(el.value);
          if (el.value.trim() !== '' && Number.isFinite(v)) e.value = JSON.stringify({ [m.name]: { [el.dataset.option]: v } });
        }
      }
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset.metricpick) {                            // metrics_to_log: the metric chosen in the menu, added by [+]
      ui.metricPick = { [el.dataset.metricpick]: el.value };
      ui.metricPickOpt = {};
      el.blur();
      if (last) main(last);
      return;
    }
    if (el.dataset.metricpickopt) {                         // its value, when the chosen metric takes one
      ui.metricPickOpt = { [el.dataset.metricpickopt]: el.value };
      el.blur();
      if (last) main(last);
      return;
    }
    if (!el.dataset || !el.dataset.op) return;
    const form = el.closest('.form');
    if (el.dataset.check === 'leakage' && el.value.trim() && !leakageOk(el.value)) {
      // refused before writing: the system would reject it at the first run
      el.blur();
      if (form && trees[form.dataset.file]) trees[form.dataset.file].note = `refused: ${notDefined(el.value.trim())}`;
      if (el.dataset.op === 'add') el.value = '';
      if (last) main(last);
      return;
    }
    if (el.dataset.op === 'range') {               // one [lower, upper] pair, written once both are there
      const boxes = [...el.closest('.field').querySelectorAll('input[data-op="range"]')];
      const [lo, hi] = boxes.map(b => b.value.trim());
      if (lo && hi) { stage(el, 'set', `[${lo}, ${hi}]`); el.blur(); if (last) main(last); }
      return;
    }
    if (el.dataset.op === 'add') {
      if (el.value.trim()) stage(el, 'add', el.value);
      el.value = '';
      ui.adding = {};                                       // the [+] control closes once its value is in
    } else {
      stage(el, el.dataset.op, el.value);
      // params.yaml: profiling_split.enabled switched, its arguments arrive with their defaults (or go)
      if (form && el.dataset.op === 'set' && isSplitSwitch(JSON.parse(el.dataset.path))) {
        splitDefaults(form, JSON.parse(el.dataset.path), el.value === 'true');
      }
      // master.yaml: a country chosen, its city follows at once (the current one kept when it is the country's,
      // else the country's principal city), staged with it
      if (form && isMasterFile(form.dataset.file) && el.dataset.path === '["timezone","current_country"]') {
        curFile = form.dataset.file;                        // resolve, menuOf and shown read the form being changed
        const loc = resolve(['timezone', 'current_location']).node;
        const menu = loc && loc.k === 'scalar' ? menuOf(loc.path) : null;
        if (menu && menu.opts.length && !menu.opts.includes(unquote(shown(loc.path, loc.text).text))) {
          stageValue(form, loc.path, 'set', menu.opts[0]);
        }
      }
    }
    el.blur();
    if (last) main(last);
  });
  // `toggle` does not bubble; capture it to remember which sections are open. Drawing a section
  // already open fires `toggle` too: only a change of state counts, so a redraw never loops.
  app.addEventListener('toggle', e => {
    const el = e.target;
    const key = el && el.dataset ? el.dataset.key : null;
    if (key && el.closest && el.closest('.cfgsearching')) return;    // opened by a key search: not kept as the panel's own
    if (!key || isOpen(key, el.classList.contains('project') || el.classList.contains('person')) === el.open) return;   // these start open
    ui.open[key] = el.open;
    save();
    if (key.startsWith('cf:')) {
      const file = el.dataset.file;
      const shownElsewhere = [...app.querySelectorAll('details.rungraph[open], .budget-form[data-file]')].some(g => g.dataset.file === file);
      if (el.open) requested.delete(file);
      else if (!shownElsewhere) send({ type: 'tree', file, open: false });
    }
    if (key.startsWith('dr:') && !el.open && el.dataset.want) askedChoices.delete(`${el.dataset.want}|${el.dataset.name}`);
    if ((key.startsWith('dr:') || key.startsWith('rg:')) && el.open) {
      for (const g of [el, ...el.querySelectorAll('details.rungraph')]) if (g.dataset.file) requested.delete(g.dataset.file);
    }
    if (mode === 'main' && last) main(last);              // draw what just opened (its children are lazy)
  }, true);
  window.addEventListener('message', ev => {
    const m = ev.data;
    if (mode === 'main' && m.type === 'tree') {
      // The file is the source of truth: a change on disk (a new hash not from this panel's own OK)
      // drops the changes staged from the old text, and the form shows the file at once.
      const p = pending[m.file];
      if (p && /^saved/.test(m.note || '')) p.sha = m.sha256;
      else if (p && m.sha256 && m.sha256 !== p.sha) {
        const n = Object.keys(p.edits).length;
        delete pending[m.file];
        m.note = `${m.note ? `${m.note}; ` : ''}the file changed on disk: ${n} unsaved change${n === 1 ? '' : 's'} discarded`;
      }
      trees[m.file] = { sha256: m.sha256, tree: m.tree, error: m.error, note: m.note };
      // After the panel's own save, a form left undrawn (its region holds the focused input) gets the
      // new hash, so the next field's edit is not refused as stale. A change from disk keeps the old
      // hash: an edit drawn from the old text must still be refused (the file wins).
      if (/^saved/.test(m.note || '')) {
        for (const form of app.querySelectorAll('.form')) if (form.dataset.file === m.file) form.dataset.sha = m.sha256;
      }
      if (last) main(last);
    }
    if (mode === 'main' && m.type === 'state') { if (m.menus) MENUS = m.menus; main(m); }
    if (mode === 'main' && m.type === 'facet-clear') {       // facet made: the ticks are done
      ui.figsel = { ...(ui.figsel || {}), [m.project]: [] };
      save();
      if (last) main(last);
    }
    if (mode === 'main' && m.type === 'stage' && m.edit && m.edit.op === 'arrange') {   // the planner's changes of a model (Visualise)
      const key = JSON.stringify(m.edit.path);
      const inside = e => e.op !== 'arrange' && JSON.stringify(e.path).startsWith(key.slice(0, -1) + ',');
      const p = pending[m.file];
      const id = editId('arrange', m.edit.path);
      const list = trees[m.file] && trees[m.file].tree ? resolveIn(trees[m.file].tree, m.edit.path).node : null;
      const entries = JSON.parse(m.edit.value || '[]');
      const unchanged = list && list.k === 'list' && entries.length === list.items.length && entries.every((e, i) => e.from === i && !('value' in e));
      if (p && Object.values(p.edits).some(inside)) {   // the form's own edits of these layers: theirs go first
        if (trees[m.file]) trees[m.file].note = `refused: the planner's changes of ${m.model_id} were not staged: OK or Cancel the staged edits of its layers first`;
        send({ type: 'stage-refused', file: m.file, model_id: m.model_id,
          reason: `the panel has staged edits of ${m.model_id}'s layers: OK or Cancel them there first` });
      } else if (unchanged) {
        if (p) { delete p.edits[id]; if (!Object.keys(p.edits).length) delete pending[m.file]; }
      } else {
        const q = p || (pending[m.file] = { sha: (trees[m.file] || {}).sha256, edits: {}, seq: 0 });
        q.edits[id] = m.edit;
      }
      if (last) main(last);
    }
    if (mode === 'main' && m.type === 'share-toggle') toggleShare();   // the title line's button
    if (mode === 'main' && m.type === 'status-page') {      // the status page's state (extension.js statusState)
      statusPage = m;
      if (last) main(last);
    }
    if (mode === 'main' && m.type === 'model-check') {      // model_check.py: each layer's shape, or where it failed
      modelChecks[m.project] = { ...(modelChecks[m.project] || {}), [m.model_id]: m.data };
      if (last) main(last);
    }
    if (mode === 'main' && m.type === 'filter-fields') {     // model_filter.py --fields; written: the draft is the file again
      filterData[m.project] = m.data;
      if (m.written) for (const f of Object.keys(filterDraft)) delete filterDraft[f];
      if (last) main(last);
    }
    if (mode === 'main' && m.type === 'complete-ok') {       // the warning answered OK: the blueprint's values staged and written at once
      const t = trees[m.file];
      const { n, blocks } = t ? completeFromBlueprint({ dataset: { file: m.file, sha: t.sha256 } }, m.file) : { n: 0, blocks: [] };
      if (!n) { if (t) t.note = 'nothing to complete: every key of the blueprint is in the file'; if (last) main(last); return; }
      applyStaged({ dataset: { file: m.file } }, null);     // as the form's OK (the form's other unsaved changes too, as the warning said)
      if (t) t.note = `${n} value${n === 1 ? '' : 's'} from the blueprint written` +
        (blocks.length ? `; those in ${blocks.join(', ')} join the run hash: a new run of a model trained before gets a new hash` : '');
      if (last) main(last);
      return;
    }
    if (mode === 'main' && m.type === 'figure-picked') {     // Find Figure: the file chosen, staged into the address
      const form = [...app.querySelectorAll('.form')].find(f => f.dataset.file === m.file);
      if (form) { stageValue(form, m.path, 'set', m.value); if (last) main(last); }
    }
    if (mode === 'detail' && m.type === 'detail') detail(m);
  });
  if (window.TFPanelTest) Object.assign(window.TFPanelTest, { draftOps });   // test/webview.test.js only
}());
