// services/visualization/mobile/status.js
/**
 * The status page: one card per project, as the TraceFlow Monitor's project heading shows it, refreshed every
 * 10 s from /status.json. The facts are turned into the light, phase, progress, ETA and counts by the panel's
 * own state.js (S): light, runSummary, eta, tiles, activeQueues. Read-only: nothing here starts or stops a run.
 *
 * view(data, S, nowMs) is pure (node --test runs it); the rest fetches and draws.
 *
 * Author: Shraddha Mishra
 */
(function (root) {
  'use strict';
  const REFRESH_MS = 10000;
  const esc = v => String(v == null ? '' : v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  /** The heartbeat the card follows: a running one, the newest first; else the newest. */
  function followed(heartbeats) {
    const all = Object.entries(heartbeats || {}).map(([env, hb]) => ({ env, hb }));
    const newest = list => list.sort((a, b) => Number(b.hb.written_ns || 0) - Number(a.hb.written_ns || 0))[0] || null;
    return newest(all.filter(x => x.hb.status === 'running')) || newest(all);
  }
  // The panel's ETA row (vscode_monitor/webview/panel.js runHtml): | ETA | run: <end> (in <left>) | batch: <end> (in <left>) |,
  // the ends in master.yaml's timezone, named by its city; ≤ while early stopping may end runs sooner.
  const left = s => (s < 3600 ? `${Math.max(1, Math.round(s / 60))} min` : `${(s / 3600).toFixed(1)} h`);
  function endAt(ms, tz, nowMs) {
    const d = new Date(ms);
    const time = d.toLocaleTimeString('en-GB', { timeZone: tz, hour: '2-digit', minute: '2-digit', hour12: false });
    const day = o => o.toLocaleDateString('en-CA', { timeZone: tz });
    const date = day(d) === day(new Date(nowMs)) ? '' : `${d.toLocaleDateString('en-GB', { timeZone: tz, weekday: 'short', day: 'numeric', month: 'short' })} `;
    return `${date}${time} ${tz.split('/').pop().replace(/_/g, ' ')}`;
  }
  /** The running run's and its batch's ends (state.eta, with the panel's inputs: the sweep's trials, the queue's commands left). */
  function etaHtml(p, hb, S, nowMs, tz) {
    const q = (p.queues || []).find(x => (Array.isArray(hb.batches) ? hb.batches : []).includes(x.key));
    let commandsLeft = 0;
    if (q) {                                        // the queue's commands after the one running now
      const i = q.commands.findIndex(c => c.MODEL_ID === hb.model_id && (!c.DATASET || c.DATASET === hb.dataset));
      commandsLeft = i >= 0 ? q.commands.length - i - 1 : 0;
    }
    const e = S.eta({ hb, nowMs, trials: p.trials || [], position: hb.trial, commandsLeft,
                      runsLeft: S.batchRunsLeft(p.queues, p.registry, hb), lastRunS: S.lastRunSeconds(p.registry, hb),
                      batch: S.batchRuns(p.queues, p.registry, hb) });
    const cell = (end, s) => (end ? `${esc(endAt(end, tz, nowMs))} <small>(in ${left(s)})</small>` : '<small>estimating…</small>');
    return `<table class="eta"${e.upperBound ? ' title="at the latest: early stopping may end runs sooner"' : ''}><tr>` +
      `<th>ETA${e.upperBound ? ' ≤' : ''}</th><td>run: ${cell(e.runEndMs, e.runLeftS)}</td><td>batch: ${cell(e.batchEndMs, e.batchLeftS)}</td></tr></table>`;
  }
  const fmt = v => (Number.isInteger(v) || Math.abs(v) >= 100 ? String(Math.round(v * 100) / 100) : Number(v).toPrecision(4));
  /** The running run's latest metrics, as the panel's Running tile shows them: one line per metric, its value
   * per process (train | validation | test), green better, red worse; while the test attacks, its bar (orange)
   * and attack i of n in the epoch's place, the test values so far in orange; the key-recovery report last. */
  function liveHtml(live) {
    const lines = live && Array.isArray(live.metrics) ? live.metrics.filter(m => m.values && m.values.length) : [];
    if (!lines.length) return '';
    const dot = v => (v.improved === true ? 'better' : v.improved === false ? 'worse' : 'same');
    const shown = v => (v.value == null ? esc(v.note || '') : `${esc(fmt(v.value))}${v.note ? ` <small>(${esc(v.note)})</small>` : ''}`);
    const a = live.attack;
    const head = [live.source, !a && live.epoch != null ? `epoch ${live.epoch}` : ''].filter(Boolean).join(' · ');
    const pct = a && a.attacks ? Math.round(100 * a.attack / a.attacks) : 0;
    const attack = !a ? '' : `<div class="attack"><span>attack</span><div class="abar${a.running ? ' running' : ''}"><i style-width="${pct}"></i></div>` +
      `<b>attack ${esc(a.attack)} of ${esc(a.attacks)}</b></div>`;
    const kr = live.key_recovery;
    const key = !kr ? '' : `<div class="key">key recovery · ${esc(kr.source)}: ${[kr.recovered ? 'recovered' : 'not recovered',
      kr.true_key_rank != null ? `true key rank ${kr.true_key_rank}` : '', kr.top_guess ? `top guess ${kr.top_guess}` : '',
      typeof kr.ge === 'number' ? `GE ${fmt(kr.ge)}` : '', typeof kr.sr1 === 'number' ? `SR@1 ${kr.sr1.toFixed(2)}` : '']
      .filter(Boolean).map(esc).join(' · ')}</div>`;
    // as the panel's run card: a head row (the source and epoch over the names, each process over its values), then
    // one row per metric, its values under their process, a blank cell where it has none
    const columns = ['train', 'validation', 'test'].filter(p => lines.some(m => m.values.some(v => v.process === p)));
    const title = p => (lines.flatMap(m => m.values).find(v => v.process === p) || {}).label || p;
    const cell = v => (!v ? '<td></td>' : `<td${v.live ? ' class="test"' : ''}>${shown(v)}<i class="dot ${dot(v)}"></i></td>`);
    const table = `<table class="rows"><tr class="cols"><th>${esc(head)}</th>` +
      columns.map(p => `<th class="col${p === 'test' ? ' test' : ''}">${esc(title(p))}</th>`).join('') + '</tr>' +
      lines.map(m => `<tr><th>${esc(m.name)}</th>${columns.map(p => cell(m.values.find(v => v.process === p))).join('')}</tr>`).join('') +
      '</table>';
    return `<div class="live">${attack}${table}${key}</div>`;
  }
  function gpuHtml(gpus) {
    return (gpus || []).map(g => `<div class="gpu"><b>GPU${esc(g.index)}</b> ${esc(g.name)} · ${esc(g.util)}% · ` +
      `${esc(((g.memUsed || 0) / 1024).toFixed(1))} / ${esc(((g.memTotal || 0) / 1024).toFixed(1))} GB · ${esc(g.temp)}°C</div>`).join('');
  }
  function cardHtml(p, S, nowMs, commands, tz) {
    const f = followed(p.heartbeats);
    const hb = f ? f.hb : null;
    const lamp = S.light(hb, nowMs);
    const run = S.runSummary(hb);
    const t = S.tiles(p.registry, p.queues, { active: S.activeQueues(commands),
      beating: hb && hb.status === 'running' ? String(hb.model_hash || '') : '' });   // a run under test stays running
    const counts = ['queued', 'running', 'done', 'failed'].map(k => `<span class="${k}"><b>${t[k].length}</b> ${k}</span>`).join('');
    const body = run && hb.status === 'running'
      ? `<div class="run"><b>${esc(run.name)}</b> <small>${esc([run.model, run.dataset].filter(Boolean).join(' · '))}</small></div>` +
        `<div class="phase">${esc(run.phase)}${Array.isArray(hb.batches) && hb.batches.length ? ` · ${esc(hb.batches.join(', '))}` : ''}` +
        `${hb.trial && hb.trials ? ` · trial ${esc(hb.trial)}/${esc(hb.trials)}` : ''}</div>` +
        `<div class="bar"><i style-width="${run.percent}"></i></div>` +
        `<div class="pct">${esc(run.percent)}%</div>${etaHtml(p, hb, S, nowMs, tz)}` +
        (p.live && p.live.hash === hb.model_hash ? liveHtml(p.live) : '')
      : `<div class="idle">${run ? `last: <b>${esc(run.name)}</b> · ${esc(hb.status)}` : 'no run yet'}</div>`;
    return `<section class="card"><h2><i class="lamp ${esc(lamp.color)}"></i>${esc(p.name)}${f ? ` <small class="env">${esc(f.env)}</small>` : ''}` +
      `<span class="state">${esc(lamp.label)}</span></h2>${body}<div class="counts">${counts}</div></section>`;
  }
  /** The whole page body for one /status.json answer. */
  function view(data, S, nowMs) {
    const projects = (data && data.projects) || [];
    return `<div class="gpus">${gpuHtml(data && data.gpus)}</div>` +
      (projects.length ? projects.map(p => cardHtml(p, S, nowMs, data.commands, (data && data.timezone) || 'UTC')).join('') : '<p class="note">No project in master.yaml.</p>');
  }

  // ── the browser ─────────────────────────────────────────────────────────────
  function boot() {
    const S = root.module.exports;
    const token = new URLSearchParams(root.location.search).get('t') || '';
    const app = root.document.getElementById('app');
    const foot = root.document.getElementById('foot');
    let last = null;
    let fetchedAt = 0;
    const draw = () => {
      if (!last) return;
      const nowMs = last.now_ms + (Date.now() - fetchedAt);       // the server's clock, not the phone's
      app.innerHTML = view(last, S, nowMs);
      for (const bar of app.querySelectorAll('[style-width]')) bar.style.width = `${bar.getAttribute('style-width')}%`;
    };
    const tick = async () => {
      try {
        const res = await fetch(`/status.json?t=${encodeURIComponent(token)}`, { cache: 'no-store' });
        if (!res.ok) throw new Error(res.status === 403 ? 'this link was withdrawn' : `HTTP ${res.status}`);
        last = await res.json();
        fetchedAt = Date.now();
        foot.textContent = `updated ${new Date().toLocaleTimeString()}${last.viewer ? ` · link: ${last.viewer}` : ''} · read-only`;
        draw();
      } catch (e) {
        foot.textContent = `offline: ${e.message || e}${last ? ` · showing ${new Date(fetchedAt).toLocaleTimeString()}` : ''} (same network or VPN?)`;
      }
    };
    tick();
    root.setInterval(tick, REFRESH_MS);
    root.document.addEventListener('visibilitychange', () => { if (!root.document.hidden) tick(); });   // back on screen: now
  }

  const api = { view, followed, endAt };
  if (typeof root.document === 'undefined') module.exports = api;
  else { root.TFStatus = api; root.addEventListener('DOMContentLoaded', boot); }
}(typeof window !== 'undefined' ? window : globalThis));
