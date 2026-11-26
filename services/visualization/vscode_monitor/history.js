// services/visualization/vscode_monitor/history.js
'use strict';
/**
 * history.js
 * ==========
 * What a run's sparkline is drawn from, and the sparkline itself.
 *
 *   JsonlReader(path).readNew()   complete new lines of a growing JSONL file
 *                                 since the last call (byte offset kept)
 *   npyLastRow(buffer)            the last row of a float64 2-D .npy array:
 *                                 a rank curve's endpoint [n traces, rank]
 *   rankByEpoch(dir, cache)       [{epoch, rank}] from a folder's epNNNN.npy
 *                                 files; cache (a Map) keeps each file's value,
 *                                 so a growing folder costs only its new files
 *   arrayDirs(metricsDir)         [{label, dir}] every arrays/<process>/<metric>/
 *                                 folder under a run's metrics/, label
 *                                 'fold_03 · validation/mean_attack_rank_curve'
 *   runCurveDir(metricsDir)       the folder the run tile's sparkline is drawn
 *                                 from: the last validation rank curve folder
 *   historyFiles(metricsDir)      every scalars/*_history.jsonl under metrics/
 *   scalarSeries(records)         {key: [values]} of a history's numeric fields
 *   sparkPath(values, w, h)       SVG path of a line through the values
 *
 * Author: Shraddha Mishra
 */

const fs = require('fs');
const path = require('path');

class JsonlReader {
  constructor(path) { this.path = path; this.offset = 0; this.rest = ''; }

  readNew() {
    let fd;
    this.rewound = false;
    try {
      const size = fs.statSync(this.path).size;
      if (size < this.offset) { this.offset = 0; this.rest = ''; this.rewound = true; }     // rewritten: read from the top
      if (size === this.offset) return [];
      fd = fs.openSync(this.path, 'r');
      const buf = Buffer.alloc(size - this.offset);
      fs.readSync(fd, buf, 0, buf.length, this.offset);
      this.offset = size;
      const text = this.rest + buf.toString('utf8');
      const lines = text.split('\n');
      this.rest = lines.pop();                                            // incomplete last line
      const out = [];
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const row = JSON.parse(line);
          if (row && typeof row === 'object') delete row._tfseal;          // the row's seal (utils/seal.py)
          out.push(row);
        } catch (e) { /* a broken line is skipped */ }
      }
      return out;
    } catch (e) {
      return [];
    } finally {
      if (fd !== undefined) fs.closeSync(fd);
    }
  }
}

/** Last row of a little-endian float64, C-order, 2-D .npy array; null otherwise. */
/** {start, rows, cols} of a little-endian float64, C-order 2-D .npy buffer; null for any other. */
function npyLayout(buffer) {
  if (!buffer || buffer.length < 10 || buffer.toString('latin1', 1, 6) !== 'NUMPY') return null;
  const major = buffer[6];
  const headerLen = major === 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
  const start = (major === 1 ? 10 : 12) + headerLen;
  const header = buffer.toString('latin1', major === 1 ? 10 : 12, start);
  if (!/'descr':\s*'<f8'/.test(header) || /'fortran_order':\s*True/.test(header)) return null;
  const shape = header.match(/'shape':\s*\((\d+),\s*(\d+)\)/);
  if (!shape) return null;
  const rows = Number(shape[1]);
  const cols = Number(shape[2]);
  if (!rows || !cols || buffer.length < start + rows * cols * 8) return null;
  return { start, rows, cols };
}

function npyRow(buffer, l, i) {
  const row = [];
  for (let c = 0; c < l.cols; c++) row.push(buffer.readDoubleLE(l.start + (i * l.cols + c) * 8));
  return row;
}

function npyLastRow(buffer) {
  const l = npyLayout(buffer);
  return l ? npyRow(buffer, l, l.rows - 1) : null;
}

/** The row of a curve [n traces, value] at `traces` traces: the last whose first column is at most it (the
 *  curve's trace axis rises); null before its first point or for another layout. */
function npyRowAt(buffer, traces) {
  const l = npyLayout(buffer);
  if (!l || l.cols < 2) return null;
  let lo = 0;
  let hi = l.rows - 1;
  if (buffer.readDoubleLE(l.start) > traces) return null;
  while (lo < hi) {                                     // the last row with traces <= `traces`
    const mid = (lo + hi + 1) >> 1;
    if (buffer.readDoubleLE(l.start + mid * l.cols * 8) <= traces) lo = mid; else hi = mid - 1;
  }
  return npyRow(buffer, l, lo);
}

/** [{epoch, rank}] from a folder's epNNNN.npy rank curves, in epoch order. */
function rankByEpoch(dir, cache) {
  let names;
  try { names = fs.readdirSync(dir); } catch (e) { return []; }
  const out = [];
  for (const name of names) {
    const m = name.match(/^ep(\d+)\.npy$/);
    if (!m) continue;
    const file = path.join(dir, name);
    let rank = cache ? cache.get(file) : undefined;
    if (rank === undefined) {
      try {
        const last = npyLastRow(fs.readFileSync(file));
        rank = last && last.length >= 2 ? last[1] : null;
      } catch (e) {
        rank = null;                                      // unreadable file skipped
      }
      if (cache && rank !== null) cache.set(file, rank);  // an epoch file is written once
    }
    if (rank !== null) out.push({ epoch: Number(m[1]), rank });
  }
  return out.sort((a, b) => a.epoch - b.epoch);
}

/** Every arrays/<process>/<metric>/ folder under a run's metrics/, sorted by label. */
function arrayDirs(metricsDir, depth = 6) {
  const out = [];
  const walk = (dir, rel, d) => {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) { return; }
    for (const e of entries) {
      if (!e.isDirectory()) continue;
      const sub = path.join(dir, e.name);
      if (e.name === 'arrays') {
        for (const proc of safeDirs(sub)) {
          for (const metric of safeDirs(path.join(sub, proc))) {
            out.push({ label: `${rel.join('/') || 'run'} · ${proc}/${metric}`, dir: path.join(sub, proc, metric) });
          }
        }
      } else if (d > 0) {
        walk(sub, rel.concat(e.name === 'cross_validation' ? [] : [e.name]), d - 1);
      }
    }
  };
  walk(metricsDir, [], depth);
  return out.sort((a, b) => a.label.localeCompare(b.label));
}

function safeDirs(dir) {
  try { return fs.readdirSync(dir, { withFileTypes: true }).filter(e => e.isDirectory()).map(e => e.name).sort(); } catch (e) { return []; }
}

/** The last validation rank curve folder (the current fold while a run trains), or null. */
function runCurveDir(metricsDir) {
  const dirs = arrayDirs(metricsDir).filter(d => /validation\/[^/]*rank/.test(d.label));
  return dirs.length ? dirs[dirs.length - 1].dir : null;
}

/** Every scalars/*_history.jsonl under a run's metrics/, sorted. */
function historyFiles(metricsDir, depth = 6) {
  const out = [];
  const walk = (dir, d) => {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch (e) { return; }
    for (const e of entries) {
      const sub = path.join(dir, e.name);
      if (e.isFile() && path.basename(dir) === 'scalars' && e.name.endsWith('_history.jsonl')) out.push(sub);
      else if (e.isDirectory() && d > 0 && e.name !== 'arrays') walk(sub, d - 1);
    }
  };
  walk(metricsDir, depth);
  return out.sort();
}

/** {key: [values]} of every numeric field but the epoch, in record order. */
function scalarSeries(records) {
  const out = {};
  for (const r of records || []) {
    for (const [k, v] of Object.entries(r || {})) {
      if (k === 'epoch' || typeof v !== 'number' || !Number.isFinite(v)) continue;
      (out[k] = out[k] || []).push(v);
    }
  }
  return out;
}

function sparkPath(values, width = 100, height = 20) {
  const v = (values || []).filter(Number.isFinite);
  if (!v.length) return '';
  const lo = Math.min(...v);
  const hi = Math.max(...v);
  const span = hi - lo || 1;
  const step = v.length > 1 ? width / (v.length - 1) : 0;
  return v.map((y, i) => `${i ? 'L' : 'M'}${(i * step).toFixed(1)},${(height - ((y - lo) / span) * height).toFixed(1)}`).join('');
}

module.exports = { JsonlReader, npyLastRow, npyRowAt, rankByEpoch, arrayDirs, runCurveDir, historyFiles, scalarSeries, sparkPath };
