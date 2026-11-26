// services/visualization/vscode_monitor/gpu.js
'use strict';
/**
 * gpu.js
 * ======
 * nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu
 *            --format=csv,noheader,nounits
 * parsed into one object per GPU, and a short rolling history for the
 * sparklines. Pure; extension.js runs the command.
 *
 * Author: Shraddha Mishra
 */

const QUERY = ['--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
  '--format=csv,noheader,nounits'];

function num(s) {
  const v = Number(String(s).trim());
  return Number.isFinite(v) ? v : null;
}

function parse(text) {
  return String(text || '').split('\n').map(l => l.trim()).filter(Boolean).map(line => {
    const [index, name, util, memUsed, memTotal, temp] = line.split(',').map(s => s.trim());
    return { index: num(index), name, util: num(util), memUsed: num(memUsed), memTotal: num(memTotal), temp: num(temp) };
  }).filter(g => g.index !== null);
}

/** Keeps the last `max` samples of each GPU's utilisation and memory share. */
class History {
  constructor(max = 60) { this.max = max; this.series = new Map(); }

  add(gpus) {
    for (const g of gpus) {
      const s = this.series.get(g.index) || { util: [], mem: [] };
      s.util.push(g.util ?? 0);
      s.mem.push(g.memTotal ? (100 * (g.memUsed ?? 0)) / g.memTotal : 0);
      if (s.util.length > this.max) { s.util.shift(); s.mem.shift(); }
      this.series.set(g.index, s);
    }
  }

  of(index) { return this.series.get(index) || { util: [], mem: [] }; }
}

module.exports = { QUERY, parse, History };
