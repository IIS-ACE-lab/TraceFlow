// services/visualization/vscode_monitor/webview/menus.js
/**
 * The config forms' menus: which options a key takes, from the rules of choices.py.
 *
 *   ruleFor(rules, file, path)          the rule of a key path ("jobs.*.type"); a key's rule also
 *                                       covers the items of its sweep list
 *   plotTypesFor(pt, type, metrics, mode)
 *                                       the plot_type values that change the drawing of a job's
 *                                       metric(s): graph_manager's metric_router.plot_types_for,
 *                                       from its table (choices.py graph_forms.plotTypes)
 *   options(menus, file, project, tree, path)
 *                                       {opts, free} or null: the rule's options, the project's
 *                                       names for a "ref" rule, narrowed by a sibling key for a
 *                                       "by" rule (a graph job's processes by its source)
 *
 * Loaded before panel.js (window.TFMenus) and by the tests (module.exports).
 *
 * Author: Shraddha Mishra
 */
(function (root) {
  'use strict';
  const fileName = f => String(f || '').split('/').pop();
  const unquote = t => String(t == null ? '' : t).replace(/^(["'])(.*)\1$/, '$2');

  function nodeAt(tree, path) {
    let n = tree;
    for (const seg of path) {
      if (!n) return null;
      if (n.k === 'map') n = (n.entries.find(e => e.key === String(seg)) || {}).node;
      else if (n.k === 'list') n = n.items[seg];
      else return null;
    }
    return n || null;
  }

  function match(rules, file, path) {
    const segs = path.map(String);
    return ((rules || {})[fileName(file)] || []).find(x => {
      const p = x.p.split('.');
      return p.length === segs.length && p.every((part, i) => part === '*' || part === segs[i]);
    }) || null;
  }

  function ruleFor(rules, file, path) {
    return match(rules, file, path) || (typeof path[path.length - 1] === 'number' ? match(rules, file, path.slice(0, -1)) : null);
  }

  /** read(node): the text the form shows for a scalar node (its staged value when changed); default its text. */
  function options(menus, file, project, tree, path, read) {
    const rule = ruleFor((menus || {}).rules, file, path);
    if (!rule) return null;
    let opts = rule.o || [];
    if (rule.ref) {
      const refs = (menus || {}).refs || {};
      opts = rule.ref === 'projects' ? Object.keys(refs) : ((refs[project] || {})[rule.ref] || []);
    }
    let others = null;
    if (rule.ref && rule.others_ref) {                   // e.g. the attack bytes no data entry defines: listed, never choosable
      const o = ((((menus || {}).refs || {})[project]) || {})[rule.others_ref];
      if (o && typeof o === 'object' && Object.keys(o).length) others = { ...o };
    }
    if (rule.by && tree) {
      // by.key: a sibling (a job's source), or a dotted path from by.at (deep_learning.framework from the root;
      // timezone.current_country beside current_location), as staged when read is given
      const v = nodeAt(tree, [...path.slice(0, rule.by.at), ...String(rule.by.key).split('.')]);
      const text = v && v.k === 'scalar' ? (read ? read(v) : v.text) : '';
      const key = text ? unquote(text) : rule.by.default;
      opts = rule.by.map[key] || opts;
      if (rule.by.others && rule.by.map[key]) {
        // by.others (choices.py _by_framework): the options only another framework has, by name -> 'pytorch'
        others = {};
        for (const [name, list] of Object.entries(rule.by.map)) {
          if (name === key) continue;
          for (const o of list) if (!opts.includes(o)) others[o] = others[o] ? `${others[o]}, ${name}` : name;
        }
        for (const o of Object.keys(others)) others[o] = `available in ${others[o]}`;   // the phrase the menu shows
      }
    }
    const out = rule.l ? { opts, free: !!rule.free, labels: rule.l } : { opts, free: !!rule.free };   // l: an option's text in the menu
    if (others && Object.keys(others).length) out.others = others;   // shown disabled, with the phrase (why it is not choosable)
    return out;
  }

  /**
   * A facet's arrangement from its boxes (rows of panel file_ids, '' for an empty box): one row is
   * horizontal, one column vertical, else a grid with one list per row; empty boxes and rows drop.
   */
  function facetLayout(r, c, cells) {
    const rows = cells.map(row => row.filter(Boolean).map(Number)).filter(row => row.length);
    const mode = r === 1 ? 'horizontal' : c === 1 ? 'vertical' : 'grid';
    return { mode, order: mode === 'grid' ? rows : rows.flat() };
  }

  /** scalar | curve | fixed: metric_router.metric_kind from its table of the non-scalar names. */
  function metricKind(pt, metric) {
    const low = String(metric).trim().toLowerCase();
    return pt.kinds[low] || (pt.curvePrefixes.some(x => low.startsWith(x)) ? 'curve' : 'scalar');
  }

  function plotTypesFor(pt, type, metrics, mode) {
    if (type === 'model_explanation') return ['auto'];
    let allowed = null;
    for (const m of metrics) {
      const kind = metricKind(pt, m);
      const overlay = pt.overlay.includes(type) || (kind === 'curve' && mode === 'overlay');
      const these = new Set(pt.table[kind][overlay ? 'overlay' : 'solo']);
      allowed = allowed ? new Set([...allowed].filter(x => these.has(x))) : these;
    }
    return pt.order.filter(t => t === 'auto' || (allowed && allowed.has(t)));
  }

  /**
   * Master Control (master.yaml): {tier, why} of a key path. 'readonly': a project's name, folders and
   * files and the secrets' files (the panel shows them; edit them in the file by hand); 'hash': a key of
   * every new run's model_hash (model_naming.make_experiment_hash), written after a confirmation; ''.
   * The extension refuses a write to a read-only key and asks before a hash key (extension.js).
   */
  const MASTER_FILES = ['name', 'root', 'params', 'env_configs', 'code_dir', 'use_core_src', 'logs', 'models', 'metrics'];
  const MASTER_HASH = ['scripts', 'extra_configs', 'distributed_override'];
  function masterTier(path) {
    const p = (path || []).map(String);
    const readonly = why => ({ tier: 'readonly', why });
    const hash = { tier: 'hash', why: 'joins the model_hash of every new run: after a change no DONE run is skipped, every combination trains again' };
    if (p[0] === 'secrets') return readonly('a secrets file: edit it in master.yaml by hand');
    if (p[0] === 'status_share') return readonly('the status page: a TCP port the network lets through to this machine, and optionally host (without it, this machine\'s own address); edit it in master.yaml by hand (optional: enabled false where the server may not open a port)');
    if (p[0] === 'distributed') return hash;
    if (p[0] !== 'project' || p.length < 3) return { tier: '', why: '' };
    const k = p[2];
    if (k === 'name') return readonly('the project\'s name: its runs, registries and queues carry it; edit it in master.yaml by hand');
    if (MASTER_FILES.includes(k) || (k === 'exp_rerun' && p[3] === 'path') || (['auto_search', 'auto_mlops'].includes(k) && p[3] === 'config') ||
        (k === 'visualization' && p[3] === 'graph_manager' && p[4] === 'config')) {
      return readonly('the project\'s folders and files: edit it in master.yaml by hand');
    }
    if (MASTER_HASH.includes(k)) return hash;
    return { tier: '', why: '' };
  }

  /**
   * The names an Add button asks for, in the panel (the webview checks as one types, the extension again
   * before it writes): its label, a hint, its form, and what an existing one is called.
   */
  const ASKS = {
    project: { label: 'project name', hint: 'also its folder projects/<name>/', re: /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/, what: 'a project' },
    queue: { label: 'queue key', hint: 'its runs\' BATCH label, e.g. fig27 or fig27_vgg16', re: /^[A-Za-z0-9][A-Za-z0-9_.-]*$/, what: 'a queue' },
    job: { label: 'job name', hint: 'its folder is job_<id>_<name>, e.g. fig27_mean_accuracy', re: /^[A-Za-z0-9_.{}-]+$/, what: 'a job',
      form: 'letters, digits, _ . - and {{metric}}-style placeholders, no spaces' },
    facet: { label: 'facet name', hint: 'its folder is graphs/facets/facet_<id>_<name>, e.g. fig26_id_and_hw', re: /^[A-Za-z0-9_.-]+$/, what: 'a facet' },
    model: { label: 'model_id', hint: 'MODEL_ID= of make orchestrate, e.g. MY_CNN', re: /^[A-Za-z0-9_.-]+$/, what: 'a model' },
    dataset: { label: 'data name', hint: 'DATASET= of make orchestrate, e.g. ascad_d50', re: /^[A-Za-z0-9_.-]+$/, what: 'a data entry' },
  };
  /** Why the name typed for an Add button cannot be used ('' when it can): empty, its form, or taken. */
  function askError(kind, value, taken) {
    const a = ASKS[kind];
    const v = String(value == null ? '' : value).trim();
    if (!a) return `nothing asks for a ${kind}`;
    if (!v) return `type the ${a.label}`;
    if (!a.re.test(v)) return a.form || 'letters, digits, _ . - (no spaces)';
    if ((taken || []).map(String).includes(v)) return `${v} is ${a.what} already`;
    return '';
  }

  const api = { fileName, unquote, nodeAt, ruleFor, options, facetLayout, metricKind, plotTypesFor, masterTier, ASKS, askError };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.TFMenus = api;
}(typeof window !== 'undefined' ? window : globalThis));
