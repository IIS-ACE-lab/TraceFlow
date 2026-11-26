// services/visualization/vscode_monitor/webview/planner_core.js
/**
 * Plan a Model, the logic (no DOM): the Type skeletons and their settings (numbers typed, steps and words chosen), the
 * in_channels / num_features / in_features each layer takes from the one above, keeping hand edits when a
 * skeleton dropdown changes, a params.yaml model in and out, the arrange edit a visualised model's changes
 * become (yamledit.py arrange), and the model_id check. The page (planner.js) draws it; TraceFlow's model
 * check (model_check.py) states the shapes and weights, these numbers only fill the layer arguments.
 *
 * A layer is {name, args, slot?, from?, handArgs?, moved?, added?, anchor?}: slot names its place in the
 * skeleton (b2.conv1, b2.pool, flatten, dense1, out), from its index in the visualised model, handArgs the
 * arguments set by hand (chain leaves them), moved a layer moved by hand, added a layer inserted by hand
 * (anchor: the slot it follows). A skeleton layer removed by hand is a slot in the page's removed list.
 *
 * Author: Shraddha Mishra
 */
(function (root) {
  'use strict';

  const copy = v => JSON.parse(JSON.stringify(v));
  const canon = v => (Array.isArray(v) ? v.map(canon)
    : v && typeof v === 'object' ? Object.fromEntries(Object.keys(v).sort().map(k => [k, canon(v[k])])) : v);
  const same = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));
  const asData = layers => layers.map(l => ({ [l.name]: l.args }));
  const layer = (name, args, slot) => ({ name, args, slot });

  // ── the Type skeletons: the settings and the layer list they build ──────────────────────────────────
  // Every Type has its own settings, none tied to a dataset or a paper: the layer menus list what core.model
  // builds (the page marks a layer core.model cannot build in a layer list disabled, with why), the numbers
  // are free, and the defaults are a small model that builds on any trace.
  const POOLS = ['none', 'max', 'avg'];
  const GROWTH = ['same', 'double', 'halve'];
  /** Filters of block b (1-based): the first block's, then the same, doubled or halved per block, stopped at
   *  the limit (a ceiling when doubling, a floor when halving; null: none). */
  const filtersOf = (o, b) => {
    const f = Number(o.filters);
    if (o.growth === 'double') return Math.min(f * 2 ** (b - 1), o.limit == null ? Infinity : o.limit);
    if (o.growth === 'halve') return Math.max(Math.floor(f / 2 ** (b - 1)), o.limit == null ? 1 : o.limit);
    return f;
  };
  /** A Conv1d's padding for the Padding setting (a design from before it: half). */
  const padOf = o => (o.padding === 'same' ? 'same' : o.padding === 'valid' ? 0 : Math.floor((o.kernel - 1) / 2));
  /** A per-block setting: one number for every block, or a list with one per block (the last for any after it). */
  const perBlock = (v, b) => (Array.isArray(v) ? (v[b - 1] != null ? v[b - 1] : v[v.length - 1]) : v);
  // Activations are per layer (Activation per layer): a design starts with relu where a block or a dense layer ends
  const ACT = 'relu';
  const POOL = { dim: 1, type: 'max', kernel_size: 2, stride: 2 };   // a block's pooling as it starts (Pooling per block)
  /** The Block Pattern: one block's layer kinds in order, conv / norm / pool, any of them, in any order, any number
   *  of times (none: an empty block). A design from before the setting: conv, its normalization, its pooling. */
  const PATTERN = ['conv', 'norm', 'pool'];
  function patternOf(o) {
    const raw = Array.isArray(o.pattern) ? o.pattern : ['conv', ...(o.norm && o.norm !== 'none' ? ['norm'] : []), ...(o.pool === 'none' ? [] : ['pool'])];
    return raw.filter(t => t !== 'norm' || o.norm !== 'none');
  }
  /** Flatten, the dense layers and the output Linear(out_features: auto) (a Dropout: per layer, Dropout per layer). */
  const head = (out, o) => {
    out.push(layer('Flatten', {}, 'flatten'));
    for (let d = 1; d <= o.dense; d += 1) {
      out.push(layer('Linear', { in_features: 0, out_features: o.units, activation: ACT }, `dense${d}`));
    }
    out.push(layer('Linear', { in_features: 0, out_features: 'auto' }, 'out'));
    return out;
  };
  const SKELETONS = {
    CNN: {
      knobs: [
        { k: 'blocks', label: 'Blocks', kind: 'int', min: 1, choices: 'blocks', d: 2 },
        // a pooling before block 1 (no convolution before it): the simplified networks that drop a first conv block
        { k: 'prepool', label: 'Pooling first', kind: 'enum', options: ['none', 'yes'], d: 'none' },
        // what one block is: its layer kinds in order (chips); the kinds before the pool repeat per block, then the pool
        { k: 'pattern', label: 'Block Pattern', kind: 'pattern', d: PATTERN },
        { k: 'convs', label: 'Repeats per block', kind: 'int', min: 0, d: 1, listable: true },   // one, or one per block: 2,2,3,3,3
        { k: 'filters', label: 'Filters (first block)', kind: 'int', min: 1, choices: 'powers', d: 32 },
        { k: 'growth', label: 'Filters per block', kind: 'enum', options: GROWTH, d: 'double' },
        { k: 'limit', label: 'Filter limit', kind: 'int', min: 1, optional: true, choices: 'limits', d: null },
        { k: 'kernel', label: 'Kernel size', kind: 'int', min: 1, d: 3 },
        // half: (kernel - 1) / 2 samples each side (an odd kernel keeps the length); same: torch's 'same'; valid: 0
        { k: 'padding', label: 'Padding', kind: 'enum', options: ['half', 'same', 'valid'], d: 'half' },
        { k: 'norm', label: 'Normalization', kind: 'enum', options: ['BatchNorm1d', 'InstanceNorm1d'], d: 'BatchNorm1d' },
        { k: 'dense', label: 'Dense layers', kind: 'int', min: 0, d: 1 },
        { k: 'units', label: 'Dense width', kind: 'int', min: 1, suggest: 'width', d: 128 },
      ],
          /** Per block: the Block Pattern, Repeats per block times (a Conv1d of the block's filters, a normalization of
       *  the Normalization's type, a pooling; relu on the last conv or normalization) except a pooling that ends the
       *  pattern, which ends the block once; then the head. Activations and poolings change per layer and per block. */
      build(o) {
        const out = o.prepool === 'yes' ? [layer('Pooling', { ...POOL }, 'prepool')] : [];
        const pattern = patternOf(o);
        const closing = pattern[pattern.length - 1] === 'pool';    // a pooling that ends the pattern ends the block, once
        const unit = closing ? pattern.slice(0, -1) : pattern;
        const actAt = unit.map(t => t !== 'pool').lastIndexOf(true);   // relu on the last conv or normalization
        const norm = o.norm && o.norm !== 'none' ? o.norm : 'BatchNorm1d';
        const many = t => unit.filter(x => x === t).length > 1;      // a kind twice in the pattern: its slots numbered
        for (let b = 1; b <= o.blocks; b += 1) {
          const f = filtersOf(o, b);
          const reps = perBlock(o.convs, b);
          for (let r = 1; r <= reps; r += 1) {
            unit.forEach((t, u) => {
              const act = u === actAt ? { activation: ACT } : {};
              const slot = `b${b}.${t}${r}${many(t) ? `_${u}` : ''}`;
              if (t === 'conv') out.push(layer('Conv1d', { in_channels: 0, out_channels: f, kernel_size: o.kernel, stride: 1, padding: padOf(o), ...act }, slot));
              else if (t === 'norm') out.push(layer(norm, { num_features: f, ...act }, slot));
              else out.push(layer('Pooling', { ...POOL }, slot));
            });
          }
          if (closing) out.push(layer('Pooling', { ...POOL }, `b${b}.pool`));
        }
        return head(out, o);
      },
    },
    MLP: {
      knobs: [
        { k: 'prepool', label: 'Pooling before Flatten', kind: 'enum', options: ['none', 'yes'], d: 'none' },
        { k: 'hidden', label: 'Hidden layers', kind: 'int', min: 1, d: 2 },
        { k: 'units', label: 'Units per layer', kind: 'int', min: 1, suggest: 'width', d: 128 },
        { k: 'norm', label: 'Normalization', kind: 'enum', options: ['none', 'BatchNorm1d', 'LayerNorm'], d: 'none' },
      ],
      /** Flatten, hidden x Linear(units) (relu on the normalization after it, if any, else on it),
       *  Linear(out_features: auto). */
      build(o) {
        const out = [...(o.prepool === 'yes' ? [layer('Pooling', { ...POOL }, 'prepool')] : []), layer('Flatten', {}, 'flatten')];
        const norm = o.norm && o.norm !== 'none' ? o.norm : null;
        for (let h = 1; h <= o.hidden; h += 1) {
          out.push(layer('Linear', { in_features: 0, out_features: o.units, ...(norm ? {} : { activation: ACT }) }, `dense${h}`));
          if (norm) out.push(layer(norm, { [norm === 'LayerNorm' ? 'normalized_shape' : 'num_features']: o.units, activation: ACT }, `dense${h}.norm`));
        }
        out.push(layer('Linear', { in_features: 0, out_features: 'auto' }, 'out'));
        return out;
      },
    },
    TRANSFORMER: {
      knobs: [
        { k: 'token', label: 'Token size (samples per token)', kind: 'int', min: 1, d: 16 },
        { k: 'width', label: 'Embedding width (d_model)', kind: 'int', min: 1, choices: 'powers', d: 64 },
        { k: 'heads', label: 'Attention heads', kind: 'int', min: 1, choices: 'divisors', d: 4 },
        { k: 'layers', label: 'Encoder layers', kind: 'int', min: 1, d: 2 },
        { k: 'ff', label: 'Feed-forward width', kind: 'int', min: 1, suggest: 'ff', d: 128 },
        { k: 'attnDropout', label: 'Encoder dropout', kind: 'float', min: 0, below: 1, d: 0.1 },
        { k: 'dense', label: 'Dense layers', kind: 'int', min: 0, d: 1 },
        { k: 'units', label: 'Dense width', kind: 'int', min: 1, suggest: 'width', d: 128 },
      ],
      /** A Conv1d token embedding (kernel = stride = the token size, out_channels = d_model), layers x
       *  TransformerEncoderLayer (batch_first), then the head. */
      build(o) {
        const out = [layer('Conv1d', { in_channels: 0, out_channels: o.width, kernel_size: o.token, stride: o.token, padding: 0 }, 'embed')];
        for (let e = 1; e <= o.layers; e += 1) {
          out.push(layer('TransformerEncoderLayer', { d_model: o.width, nhead: o.heads, dim_feedforward: o.ff,
            dropout: o.attnDropout, batch_first: true }, `enc${e}`));
        }
        return head(out, o);
      },
    },
    HYBRID: {
      knobs: [
        { k: 'blocks', label: 'Conv blocks', kind: 'int', min: 1, choices: 'blocks', d: 2 },
        { k: 'filters', label: 'Filters (first block)', kind: 'int', min: 1, choices: 'powers', d: 32 },
        { k: 'growth', label: 'Filters per block', kind: 'enum', options: GROWTH, d: 'double' },
        { k: 'kernel', label: 'Kernel size', kind: 'int', min: 1, d: 3 },
        { k: 'cell', label: 'Recurrent layer', kind: 'enum', options: ['LSTM', 'GRU', 'RNN'], d: 'LSTM' },
        { k: 'hiddenSize', label: 'Recurrent width (hidden_size)', kind: 'int', min: 1, d: 64 },
        { k: 'cellLayers', label: 'Recurrent layers (num_layers)', kind: 'int', min: 1, d: 1 },
        { k: 'dense', label: 'Dense layers', kind: 'int', min: 0, d: 1 },
        { k: 'units', label: 'Dense width', kind: 'int', min: 1, suggest: 'width', d: 128 },
      ],
      /** Per block one Conv1d (same padding, the activation) and a Pooling; one recurrent layer (batch_first,
       *  input_size from the channels above); then the head. */
      build(o) {
        const out = [];
        for (let b = 1; b <= o.blocks; b += 1) {
          out.push(layer('Conv1d', { in_channels: 0, out_channels: filtersOf({ ...o, limit: null }, b), kernel_size: o.kernel, stride: 1,
            padding: Math.floor((o.kernel - 1) / 2), activation: ACT }, `b${b}.conv1`));
          out.push(layer('Pooling', { ...POOL }, `b${b}.pool`));
        }
        out.push(layer(o.cell, { input_size: 0, hidden_size: o.hiddenSize, num_layers: o.cellLayers, batch_first: true }, 'recurrent'));
        return head(out, o);
      },
    },
  };
  const defaults = type => Object.fromEntries(SKELETONS[type].knobs.map(k => [k.k, k.d]));

  /** The menu of a numeric setting that comes in steps ({v, label}), or null for a typed one: blocks, the counts
   *  the trace length takes (each block's convs and pooling leave at least 1 sample; the label shows that
   *  length; at most 12); powers, the powers of two 4..1024; divisors, the head counts that divide the
   *  embedding width; limits, none ('') and the powers of two above the first block's filters when doubling
   *  (a ceiling), below them when halving (a floor), none alone when they stay the same (the label shows the
   *  filters per block). The current value is always offered. */
  const POWERS = [4, 8, 16, 32, 64, 128, 256, 512, 1024];
  const steps = (list, cur) => [...new Set([...list, cur].filter(Number.isFinite))].sort((a, b) => a - b).map(v => ({ v, label: String(v) }));
  function knobChoices(knob, knobs, input) {
    const cur = knobs[knob.k];
    if (knob.choices === 'powers') return steps(POWERS, cur);
    if (knob.choices === 'divisors') {
      const w = Number(knobs.width) || 1;
      return steps(Array.from({ length: Math.min(w, 32) }, (_, i) => i + 1).filter(h => w % h === 0), cur);
    }
    if (knob.choices === 'limits') {
      const f = Number(knobs.filters);
      const per = v => Array.from({ length: Number(knobs.blocks) || 1 }, (_, b) => filtersOf({ ...knobs, limit: v }, b + 1)).join(', ');
      const fit = knobs.growth === 'double' ? POWERS.filter(p => p > f) : knobs.growth === 'halve' ? POWERS.filter(p => p < f) : [];
      return [{ v: '', label: `none (${per(null)})` }, ...steps(fit, cur).map(c => ({ v: c.v, label: `${c.v} (${per(c.v)})` }))];
    }
    if (knob.choices !== 'blocks') return null;
    const out = [];
    let length = Number((input || [1, 700])[(input || [1, 700]).length - 1]);
    const k = Number(knobs.kernel) || 1;
    const pad = Math.floor((k - 1) / 2);
    const pattern = patternOf(knobs);
    const closing = pattern[pattern.length - 1] === 'pool';
    const unit = closing ? pattern.slice(0, -1) : pattern;
    const shrink = knobs.padding === 'same' ? 0 : knobs.padding === 'valid' ? k - 1 : k - 1 - 2 * pad;   // samples a conv takes off
    for (let b = 1; b <= 12; b += 1) {                   // each block's pooling as it starts: a window of 2
      const reps = perBlock(knobs.convs == null ? 1 : knobs.convs, b);
      for (let r = 0; r < reps; r += 1) {
        for (const t of unit) { if (t === 'conv') length -= shrink; else if (t === 'pool') length = Math.floor((length - 2) / 2) + 1; }
      }
      if (closing) length = Math.floor((length - 2) / 2) + 1;
      if (!(length >= 1)) break;
      out.push({ v: b, label: `${b} (length ${length})` });
    }
    if (Number.isFinite(cur) && !out.some(c => c.v === cur)) out.push({ v: cur, label: `${cur} (too many for this trace)` });
    return out;
  }
  /** Suggestions for a width setting, each {v, label} with its reason: first what the layers around it give
   *  (a dense width: the inputs it receives, the classes the output gives, and between them the geometric pyramid
   *  rule √(inputs × classes) (Masters 1993); a feed-forward width: 4 × and 2 × d_model, 4 × the original
   *  Transformer's), then the widths the project's models use, then powers of two. Any whole number is still a
   *  width: these are offered, never required. ctx: {knobs, inFeatures, classes, projectWidths}. */
  const WIDTH_POWERS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
  function widthSuggestions(knob, ctx) {
    if (!knob.suggest) return null;
    const out = [];
    const add = (v, label) => {
      const n = Math.round(Number(v));
      if (n >= 1 && !out.some(o => o.v === n)) out.push({ v: n, label });
    };
    if (knob.suggest === 'ff') {
      const d = Number((ctx.knobs || {}).width);
      if (d) { add(4 * d, `4 × d_model (${d}), the original Transformer's ratio`); add(2 * d, `2 × d_model (${d})`); }
    } else {
      const n = Number(ctx.inFeatures);
      const c = Number(ctx.classes);
      if (n && c) add(Math.sqrt(n * c), `geometric pyramid rule: √(${n} inputs × ${c} classes)`);
      if (n) add(n, `the inputs it receives (${n})`);
      if (c) add(c, `the classes the output gives (${c})`);
    }
    for (const v of ctx.projectWidths || []) add(v, 'a width this project\'s models use');
    for (const v of WIDTH_POWERS) add(v, 'a power of two');
    return out;
  }
  /** A skeleton setting as typed: a whole number, a decimal or one of its words; {value} or {error: why}. */
  function parseKnob(knob, text) {
    if (knob.kind === 'pattern') return parsePattern(Array.isArray(text) ? text.join(' ') : text);
    const t = String(text == null ? '' : text).trim();
    if (knob.optional && t === '') return { value: null };
    if (knob.listable && t.includes(',')) {                // one per block: 2,2,3,3,3 (all alike: one number)
      const parts = t.split(',').map(x => parseKnob({ ...knob, listable: false }, x));
      const bad = parts.find(x => x.error);
      if (bad) return { error: `${bad.error}, one per block, e.g. 2,2,3` };
      const values = parts.map(x => x.value);
      return { value: values.every(v => v === values[0]) ? values[0] : values };
    }
    if (knob.kind === 'enum') return t ? { value: t } : { error: 'choose one' };
    if (knob.kind === 'int' && !/^-?\d+$/.test(t)) return { error: 'a whole number' };
    const v = Number(t);
    if (t === '' || !Number.isFinite(v)) return { error: 'a number' };
    if (knob.min != null && v < knob.min) return { error: `at least ${knob.min}` };
    if (knob.max != null && v > knob.max) return { error: `at most ${knob.max}` };
    if (knob.below != null && v >= knob.below) return { error: `below ${knob.below}` };
    return { value: v };
  }

  /** A Block Pattern: its kinds (conv, norm, pool) in order, any of them, any number of times (none: an empty block). */
  function parsePattern(text) {
    const tokens = String(text == null ? '' : text).toLowerCase().split(/[\s,→>-]+/).filter(Boolean);
    const bad = tokens.find(t => !['conv', 'norm', 'pool'].includes(t));
    return bad ? { error: `${bad}: not a block layer (conv, norm, pool)` } : { value: tokens };
  }

  // ── the arguments a layer takes from the one above ───────────────────────────────────────────────────
  const num = v => (typeof v === 'number' && Number.isFinite(v) ? v : null);
  const conv1dLength = (len, a) => {
    const k = num(a.kernel_size) || 1;
    const s = num(a.stride) || 1;
    const d = num(a.dilation) || 1;
    if (a.padding === 'same') return len;
    const p = a.padding === 'valid' ? 0 : num(a.padding) || 0;
    return Math.floor((len + 2 * p - d * (k - 1) - 1) / s) + 1;
  };
  /** Fill in_channels / num_features / num_channels / in_features from the layer above (not on a hand-set
   *  layer); stops at a layer whose output it cannot work out (the model check then speaks). */
  function chain(layers, input = [1, 700]) {
    let [ch, len] = input;
    let flat = null;                                     // set after Flatten: the vector width
    const out = layers.map(l => ({ ...l, args: copy(l.args || {}) }));
    for (const l of out) {
      const a = l.args;
      const fill = (key, v) => { if (!(l.handArgs || []).includes(key) && v != null) a[key] = v; };
      if (l.name === 'Conv1d' || l.name === 'ConvTranspose1d') {
        fill('in_channels', ch);
        ch = num(a.out_channels);
        len = l.name === 'Conv1d' ? conv1dLength(len, a)
          : (len - 1) * (num(a.stride) || 1) - 2 * (num(a.padding) || 0) + (num(a.dilation) || 1) * ((num(a.kernel_size) || 1) - 1) + (num(a.output_padding) || 0) + 1;
      } else if (l.name === 'BatchNorm1d' || l.name === 'InstanceNorm1d') {
        fill('num_features', flat != null ? flat : ch);
      } else if (l.name === 'RNN' || l.name === 'LSTM' || l.name === 'GRU') {
        fill('input_size', ch);
        break;                                           // a tuple out: the shape below is the model check's
      } else if (l.name === 'GroupNorm') {
        fill('num_channels', ch);
      } else if (l.name === 'Pooling') {
        if (String(a.type || '').startsWith('global')) len = 1;
        else {
          const k = num(a.kernel_size) || 2;
          len = Math.floor((len + 2 * (num(a.padding) || 0) - k) / (num(a.stride) || k)) + 1;
        }
      } else if (l.name === 'ZeroPad1d') {
        const p = a.padding;
        len += Array.isArray(p) ? (num(p[0]) || 0) + (num(p[1]) || 0) : 2 * (num(p) || 0);
      } else if (l.name === 'Upsample' && num(a.scale_factor)) {
        len = Math.floor(len * a.scale_factor);
      } else if (l.name === 'Flatten') {
        flat = ch != null && len != null ? ch * len : null;
      } else if (l.name === 'Linear') {
        if (flat != null) { fill('in_features', flat); flat = num(a.out_features); }
        else { fill('in_features', len); len = num(a.out_features); }
      } else if (!/^(Dropout|Dropout1d|Dropout2d|Dropout3d|AlphaDropout|Identity|LayerNorm|LocalResponseNorm)$/.test(l.name)) {
        break;                                           // a layer whose output this does not work out
      }
      if (flat == null && (ch == null || !Number.isFinite(len))) break;   // the shape is no longer known
    }
    return out;
  }

  // ── hand edits and a rebuilt skeleton ───────────────────────────────────────────────────────────────
  /** Layers inserted by hand at `at`, each remembering the skeleton slot it follows. */
  function insert(layers, at, items) {
    const anchor = at > 0 ? (layers[at - 1].slot || layers[at - 1].anchor || null) : null;
    const added = items.map(i => ({ name: i.name, args: copy(i.args || {}), added: true, anchor }));
    return [...layers.slice(0, at), ...added, ...layers.slice(at)];
  }
  const hasHandEdits = (layers, removed = []) => removed.length > 0 || layers.some(l => (l.handArgs || []).length || l.moved || l.added);
  const where = slot => {
    const m = /^b(\d+)\./.exec(slot || '');
    return m ? `block ${m[1]}` : String(slot || '').replace(/^dense(\d+).*/, 'dense $1');
  };
  /** The skeleton of `type` for `knobs`, with the hand edits of `current` kept where they fit: a layer's
   *  hand-set arguments stay when its slot still exists, a removed slot stays removed, a layer added by hand
   *  follows its slot again; a moved layer goes back to the skeleton's order. */
  function rebuild(type, knobs, current, input, removed = []) {
    const fresh = SKELETONS[type].build({ ...defaults(type), ...knobs }).filter(l => !removed.includes(l.slot));
    const dropped = [];
    const result = fresh.map(l => {
      // a visualised model's layer keeps the line of params.yaml it came from (`from`), so a setting is a change of it
      const prev = current.find(c => c.slot === l.slot && c.name === l.name && c.from != null);
      const at = prev ? { ...l, from: prev.from } : l;
      const kept = current.find(c => (c.handArgs || []).length && c.slot === l.slot);
      if (!kept) return at;
      const hand = Object.fromEntries(kept.handArgs.filter(k => k in kept.args).map(k => [k, kept.args[k]]));
      return { ...at, args: { ...l.args, ...hand }, handArgs: [...kept.handArgs] };
    });
    for (const c of current.filter(x => (x.handArgs || []).length && x.slot && !result.some(f => f.slot === x.slot))) {
      dropped.push(`${c.name} (${where(c.slot)})`);
    }
    for (const c of current.filter(x => x.added)) {
      if (c.anchor && !result.some(r => r.slot === c.anchor || r.anchor === c.anchor)) { dropped.push(`${c.name} (after ${where(c.anchor)})`); continue; }
      let at = c.anchor ? result.findIndex(r => r.slot === c.anchor) + 1 : 0;
      while (at < result.length && result[at].added && result[at].anchor === c.anchor) at += 1;    // after earlier additions there
      result.splice(at, 0, { ...c, args: copy(c.args) });
    }
    return { layers: chain(result, input), dropped };
  }

  // ── the settings explained: one hint per setting (the window's hover text), one per choice of a menu ───────
  const HINTS = {
    blocks: 'conv blocks, each: its Conv1d layers (each with its normalization and activation), then its pooling; in brackets the trace length left after the last block',
    pattern: 'what one block is: its layers in order, any kinds, any number, any order (+ adds a kind, × takes one away, ◀ ▶ move it); it repeats Repeats per block times, a pooling at its very end once per block',
    convs: 'how many times the Block Pattern repeats in each block (a pooling at its very end: once per block; with one conv in the pattern: the block\'s Conv1d layers): one number for every block, or one per block, e.g. 2,2,3,3,3 (VGG-style)',
    filters: 'out_channels of the first block\'s Conv1d: how many features it learns at each sample',
    growth: 'how the filters change from one block to the next',
    limit: 'a cap on the filters: the most a block may have when doubling (the fewest when halving); in brackets each block\'s filters with this choice',
    kernel: 'kernel_size of every Conv1d: how many neighbouring samples one filter looks at',
    padding: 'samples added at each end of the trace before a Conv1d: whether a conv keeps the trace length',
    norm: 'the type of the Block Pattern\'s normalization (an MLP: the normalization after each hidden Linear)',
    dense: 'hidden Linear layers after Flatten, before the output layer (whose out_features are the classes)',
    units: 'out_features of each hidden dense layer; the suggestions give their reason',
    hidden: 'hidden Linear layers after Flatten, before the output layer (whose out_features are the classes)',
    token: 'samples per token: the trace is cut into tokens of this many samples',
    width: 'd_model: the width each token is embedded to',
    heads: 'attention heads: d_model is split among them (a divisor of it)',
    layers: 'TransformerEncoderLayer blocks',
    ff: 'the feed-forward width inside each encoder layer',
    attnDropout: 'dropout inside each encoder layer',
    cell: 'the recurrent layer after the conv blocks',
    hiddenSize: 'hidden_size: the recurrent layer\'s width',
    cellLayers: 'num_layers: recurrent layers stacked',
    prepool: 'a pooling layer first, with no convolution before it: the trace is pooled before the first block (CNN) or before Flatten (MLP); its type and window are the layer\'s own',
  };
  const TYPE_HINTS = { MLP: { units: 'out_features of each hidden Linear layer; the suggestions give their reason' } };
  const OPTION_HINTS = {
    growth: { double: 'twice the filters each block: 64, 128, 256, …', same: 'every block as many filters as the first', halve: 'half the filters each block' },
    limit: { '': 'no cap: the filters keep doubling (or halving)' },
    padding: { half: '(kernel − 1) / 2 at each end: an odd kernel keeps the trace length (3 → 1, 11 → 5)',
      same: 'PyTorch\'s \'same\': the output as long as the input, for any kernel', valid: 'none (0): each Conv1d shortens the trace by kernel − 1' },
    norm: { none: 'no normalization', BatchNorm1d: 'each channel normalized over the batch (running mean and variance)',
      InstanceNorm1d: 'each channel normalized over its own trace', LayerNorm: 'each trace\'s features normalized together' },
    pool: { none: 'no pooling: the block keeps the trace length', max: 'the largest value of each window', avg: 'the mean of each window' },
    cell: { LSTM: 'long short-term memory', GRU: 'gated recurrent unit', RNN: 'a plain (Elman) recurrent layer' },
    prepool: { none: 'the first layer is the first block\'s (or Flatten)', yes: 'a pooling layer comes first (max 2 as it starts; its type and window are its own)' },
  };
  /** A setting's meaning, for the window's hover text. */
  const knobHint = (type, k) => ((TYPE_HINTS[type] || {})[k] || HINTS[k] || '');
  /** A menu choice's meaning ('' when it needs none: an activation, a number). */
  const optionHint = (k, v) => ((OPTION_HINTS[k] || {})[v == null ? '' : String(v)] || '');

  // ── a params.yaml model in and out ────────────────────────────────────────────────────────────────────
  /** The skeleton settings a layer list was built with, read from the layers themselves (null when they
   *  cannot be): the caller checks that the skeleton builds exactly these layers. */
  function readKnobs(type, raw) {
    const names = raw.map(l => Object.keys(l)[0]);
    const named = n => raw.filter(l => Object.keys(l)[0] === n).map(l => l[n] || {});
    const linears = named('Linear');
    const after = n => { const i = names.indexOf(n); return i >= 0 ? names[i + 1] : null; };
    const d = defaults(type);
    const headOf = () => ({ dense: Math.max(0, linears.length - 1), units: linears.length > 1 ? linears[0].out_features : d.units });
    if (type === 'CNN') {
      const NORMS = ['BatchNorm1d', 'InstanceNorm1d'];
      const end = names.indexOf('Flatten') < 0 ? names.length : names.indexOf('Flatten');
      const kinds = names.slice(0, end).map(n => (n === 'Conv1d' ? 'conv' : NORMS.includes(n) ? 'norm' : n === 'Pooling' ? 'pool' : null));
      if (!kinds.length || kinds.some(k => !k)) return null;           // a layer the Block Pattern has no word for
      const conv = raw.slice(0, end).map((l, i) => (kinds[i] === 'conv' ? l.Conv1d : null));
      let blocks = [];                                     // each block's kinds (before its pool), and where it starts
      const pooled = kinds.includes('pool');
      if (pooled) {                                        // a block ends at its pool
        let cur = { kinds: [], start: 0 };
        kinds.forEach((k, i) => { if (k === 'pool') { blocks.push(cur); cur = { kinds: [], start: i + 1 }; } else cur.kinds.push(k); });
        if (cur.kinds.length) return null;                 // layers after the last pool: no block of the settings
      } else {                                             // without pooling a block is a run of one filter count
        let cur = null;
        kinds.forEach((k, i) => {
          if (k === 'conv' && (!cur || conv[i].out_channels !== cur.f)) { if (cur) blocks.push(cur); cur = { kinds: [], start: i, f: conv[i].out_channels }; }
          if (!cur) cur = { kinds: [], start: i, f: null };
          cur.kinds.push(k);
        });
        if (cur) blocks.push(cur);
      }
      if (!blocks.length) return null;
      if (blocks.every(b => !b.kinds.length)) {            // poolings only (ASCAD_CNNbase_conv0): the pattern is a pooling
        return { blocks: blocks.length, prepool: 'none', pattern: ['pool'], convs: 1, filters: d.filters, growth: d.growth, limit: null,
          kernel: d.kernel, padding: d.padding, norm: d.norm, ...headOf() };
      }
      let prepool = 'none';                                // a pooling alone before the first block: Pooling first
      if (pooled && blocks.length > 1 && !blocks[0].kinds.length && blocks.slice(1).every(b => b.kinds.length)) { prepool = 'yes'; blocks = blocks.slice(1); }
      // the pattern: the shortest run of kinds every block repeats a whole number of times
      const first = blocks[0].kinds;
      const L = Array.from({ length: first.length }, (_, n) => n + 1).find(n => blocks.every(b => b.kinds.length % n === 0 &&
        b.kinds.every((k, i) => k === first[i % n])));
      if (!L) return null;
      const unit = first.slice(0, L);
      const counts = blocks.map(b => b.kinds.length / L);
      const per = counts.every(c => c === counts[0]) ? counts[0] : counts;   // one number, or one per block
      if (blocks.some(b => !b.kinds.length)) return null;  // a block of poolings only among others
      const convs1 = unit.includes('conv');                // a pattern without a conv: no filters to read
      const firsts = convs1 ? blocks.map(b => conv[b.start + b.kinds.indexOf('conv')].out_channels) : [d.filters];
      const filters = firsts[0];
      const growth = blocks.length > 1 ? (firsts[1] > firsts[0] ? 'double' : firsts[1] < firsts[0] ? 'halve' : 'same') : d.growth;
      const free = firsts.map((_, b) => filtersOf({ filters, growth, limit: null }, b + 1));
      const limit = firsts.some((f, b) => f !== free[b]) ? (growth === 'double' ? Math.max(...firsts) : Math.min(...firsts)) : null;
      const c0 = conv.find(Boolean) || { kernel_size: d.kernel, padding: Math.floor((d.kernel - 1) / 2) };
      const normName = names.slice(0, end).find(n => NORMS.includes(n));
      const padding = c0.padding === 'same' ? 'same' : (c0.padding === 0 || c0.padding === 'valid') && c0.kernel_size > 1 ? 'valid' : 'half';
      return { blocks: blocks.length, prepool, pattern: [...unit, ...(pooled ? ['pool'] : [])], convs: per, filters, growth, limit,
        kernel: c0.kernel_size, padding, norm: normName || d.norm, ...headOf() };
    }
    if (type === 'MLP') {
      if (linears.length < 2) return null;
      const norm = SKELETONS.MLP.knobs.find(k => k.k === 'norm').options.includes(after('Linear')) ? after('Linear') : 'none';
      const prepool = names[0] === 'Pooling' && names[1] === 'Flatten' ? 'yes' : 'none';   // a pooling before Flatten
      return { prepool, hidden: linears.length - 1, units: linears[0].out_features, norm };
    }
    return null;                                         // TRANSFORMER, HYBRID: core.model builds none yet to read
  }
  /** A model of params.yaml on the page: its Type, its layers (each with its index), and the skeleton
   *  knobs when a skeleton builds exactly these layers (custom otherwise: the dropdowns then change nothing). */
  // a dropout is a layer of its own, after the layer it follows (Dropout per layer): not a setting
  const DROPOUTS = /^(Dropout|AlphaDropout)$/;
  function fromModel(model, forms, input) {
    const raw = ((model.architecture_args || {}).layers) || [];
    const kind = String(model.architecture || 'CNN').toUpperCase();
    const type = ['CNN', 'MLP', 'TRANSFORMER', 'HYBRID'].includes(kind) ? kind : 'CNN';
    const layers = raw.map((l, k) => { const [name] = Object.keys(l); return { name, args: copy(l[name] || {}), from: k }; });
    // the settings are read without the dropouts; each goes back after its layer, as a layer added there
    const core = layers.filter(l => !DROPOUTS.test(l.name));
    const coreRaw = core.map(l => raw[l.from]);
    let knobs = null;
    let mine = null;
    const read = SKELETONS[type] ? readKnobs(type, coreRaw) : null;
    if (read) {
      const built = SKELETONS[type].build(read);
      if (built.length === coreRaw.length && same(asData(chain(built, input)), coreRaw)) {
        knobs = read;
        mine = core.map((l, i) => ({ ...l, slot: built[i].slot }));
      } else if (built.length === coreRaw.length && built.every((b, i) => b.name === core[i].name)) {
        // the settings' layers, one for one, with some arguments of their own (a layer's own activation, padding,
        // kernel): the settings read, and each argument the file sets otherwise is that layer's hand edit
        // (an argument chain fills from the layer above, e.g. in_features after a padding of its own, is compared once
        // the others are applied: it follows them, and is the layer's own only when it still differs)
        const DERIVED = ['in_channels', 'num_features', 'in_features', 'input_size', 'num_channels'];
        const differ = (a, b, keys) => [...new Set([...Object.keys(a), ...Object.keys(b)])].filter(k => keys(k) && !same(a[k], b[k]));
        let own = built.map((b, i) => {
          const hand = differ(b.args, core[i].args, k => !DERIVED.includes(k));
          return { ...b, args: { ...b.args, ...Object.fromEntries(hand.map(k => [k, core[i].args[k]])) }, from: core[i].from, ...(hand.length ? { handArgs: hand } : {}) };
        });
        own = chain(own, input).map((b, i) => {
          const derived = differ(b.args, core[i].args, k => DERIVED.includes(k));
          const hand = [...(b.handArgs || []), ...derived];
          return { ...b, args: copy(core[i].args), ...(hand.length ? { handArgs: hand } : {}) };
        });
        if (same(asData(chain(own, input)), coreRaw)) { knobs = read; mine = own; }
      }
    }
    if (!mine) return { type, knobs: null, layers, custom: true };
    const out = [];
    let k = 0;
    for (const l of layers) {
      if (!DROPOUTS.test(l.name)) { out.push(mine[k]); k += 1; continue; }
      const before = out[out.length - 1];
      out.push({ ...l, added: true, anchor: before ? (before.slot || before.anchor || null) : null });
    }
    return { type, knobs, layers: out, custom: false };
  }
  const toModel = (id, type, layers) => ({ model_id: id, architecture: type,
    architecture_args: { strict_graph_args: true, layers: asData(layers) } });
  /** The page's layers as the arrange edit of the model they came from (yamledit.py arrange). */
  const arrangeOf = (original, layers) => layers.map(l => {
    if (l.from == null) return { value: { [l.name]: l.args } };
    return same({ [l.name]: l.args }, original[l.from]) ? { from: l.from } : { from: l.from, value: { [l.name]: l.args } };
  });

  /** A visualised model's unsaved changes against its layers in the file: layers added, removed, changed
   *  (arguments or kind) and moved (a kept layer now before one that came before it). */
  function changesOf(original, layers) {
    if (!Array.isArray(original)) return { added: 0, removed: 0, changed: 0, moved: 0, total: 0 };
    const plan = arrangeOf(original, layers);
    const kept = plan.filter(e => e.from != null);
    let moved = 0;
    let last = -1;
    for (const e of kept) { if (e.from < last) moved += 1; else last = e.from; }
    const out = { added: plan.length - kept.length, removed: original.length - new Set(kept.map(e => e.from)).size,
      changed: kept.filter(e => 'value' in e).length, moved };
    return { ...out, total: out.added + out.removed + out.changed + out.moved };
  }

  /** A model_id as Add the Model takes it: its form and unique among `ids`; a case-only twin is noted. */
  function idState(id, ids, askError) {
    const v = String(id == null ? '' : id).trim();
    const error = askError('model', v, ids);
    if (error) return { ok: false, error, note: '' };
    const twin = (ids || []).map(String).find(x => x.toLowerCase() === v.toLowerCase() && x !== v);
    return { ok: true, error: '', note: twin ? `differs only in case from ${twin}` : '' };
  }
  /** The model of params.yaml whose layers equal the design, else null. */
  function sameAs(layers, models) {
    const data = asData(layers);
    const hit = (models || []).find(m => same(((m.architecture_args || {}).layers) || null, data));
    return hit ? hit.model_id : null;
  }

  // ── the bar scale (model_diagram.py's scale_ticks is the same rule) ─────────────────────────────────
  const xOnLog = (v, max, width) => width * Math.log10(Math.max(1, v)) / Math.log10(max);
  /** Ticks at 1, 10, 100, ... and the axis end (max); powers of ten give way while their label would touch the
   *  end's (labels about `charPx` a character, the end's anchored at the axis end, the others centred). */
  function scaleTicks(max, width, charPx = 6) {
    const out = [];
    for (let t = 1; t < max; t *= 10) out.push(t);
    const touches = t => width - xOnLog(t, max, width) < charPx * (String(t).length / 2 + String(max).length) + 4;
    while (out.length > 1 && touches(out[out.length - 1])) out.pop();
    return [...out, max];
  }
  const api = { SKELETONS, filtersOf, knobChoices, widthSuggestions, parseKnob, defaults, chain, insert, hasHandEdits, rebuild, fromModel, toModel, arrangeOf, changesOf, idState, sameAs, same, asData,
    scaleTicks, knobHint, optionHint, parsePattern, patternOf };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TFPlanner = api;
}(typeof window !== 'undefined' ? window : globalThis));
