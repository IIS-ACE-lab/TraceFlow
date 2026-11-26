// services/visualization/vscode_monitor/webview/planner.js
/**
 * Plan a Model, the page (an editor tab beside the panel). Three columns read across: left the Type and its
 * skeleton settings (numbers typed, words chosen), the numbers, the model_id and the buttons; middle the layer cards, coloured by
 * group, each opening its arguments, a + in the gap left of any two; right one bar per layer on its row.
 * The settings make the whole model; a tick box on each card selects layers for Delete and Edit (one argument
 * form for all of them, set as hand edits). With Observe Trace, under each card a line: what that layer makes of one attack trace
 * (host: trace_view.py), drawn top to bottom when it arrives, or for a range of traces their mean with a band from the
 * lowest to the highest value at each point; the trace from a data entry of the project through
 * the design with untrained weights, or (Visualise) from a trained run's test.h5 through the run's own layers and
 * weights: then the cards are that run's layers, read-only, and its data entry is fixed.
 * The host (extension.js openPlanner) runs TraceFlow's model check, the YAML preview and the write; this page
 * never writes params.yaml. The design is kept in the tab's state, so a window reload brings the tab back with it.
 *
 * window.TFPlannerPage = {state, act(action), render()}: the DOM handlers only turn clicks into actions.
 *
 * Author: Shraddha Mishra
 */
(function () {
  'use strict';
  const P = window.TFPlanner;
  const M = window.TFMenus;
  const vscode = typeof acquireVsCodeApi === 'function' ? acquireVsCodeApi() : { postMessage: () => {} };
  const send = m => vscode.postMessage(m);
  const keep = v => { try { if (vscode.setState) vscode.setState(v); } catch (e) { /* no tab state: nothing kept */ } };
  const kept = () => { try { return vscode.getState ? vscode.getState() : null; } catch (e) { return null; } };
  const esc = v => String(v == null ? '' : v).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const KINDS = ['CNN', 'MLP', 'TRANSFORMER', 'HYBRID'];

  const state = { ready: false };

  // ── what the forms allow ───────────────────────────────────────────────────────────────────────────
  const forms = () => state.forms || { layers: {}, groups: [], limits: {}, disabledKinds: {}, kerasKinds: ['CNN', 'MLP'], keras: { layers: [] } };
  const keras = () => state.framework === 'keras';
  const activations = () => ((((forms().layers.Linear || {}).args) || []).find(a => a.k === 'activation') || {}).o || ['relu'];
  function kindBlocked(kind) {
    const f = forms();
    if (f.disabledKinds[kind]) return f.disabledKinds[kind];
    if (keras() && !f.kerasKinds.includes(kind)) return 'under deep_learning.framework: keras: Keras builds no attention or recurrent layer';
    if (!P.SKELETONS[kind]) return 'no skeleton yet';
    return '';
  }
  function layerBlocked(name) {
    const f = forms();
    if (f.limits[name]) return f.limits[name];
    if (keras() && !f.keras.layers.includes(name)) return `no Keras layer (deep_learning.framework: keras builds ${f.keras.layers.join(', ')})`;
    return '';
  }
  const groupOf = name => ((forms().layers[name] || {}).group || 'other').toLowerCase();
  // the input the layers chain from: [channels, length] of the data entry chosen (its target_range), or the length
  // typed when the project has no data entry yet; null until one is known (planner_core's geometry then takes its own
  // default for the menus, and no check is sent)
  const inputArg = () => state.input || undefined;
  const lengthOf = name => ((state.lengths || []).find(e => e.name === name) || {}).length || null;

  // ── state changes ──────────────────────────────────────────────────────────────────────────────────
  function startFrom(start, keepHistory) {
    if (!keepHistory) { state.undo = []; state.redo = []; }   // Undo / Redo: of this start's changes only
    state.original = null;
    state.modelId = null;
    if (start && start.model) {
      const m = P.fromModel(start.model, forms(), inputArg());
      state.type = m.type;
      state.knobs = m.knobs || P.defaults(P.SKELETONS[m.type] ? m.type : 'CNN');
      state.custom = m.custom;
      state.layers = m.layers;
      state.original = start.model.architecture_args.layers;
      state.modelId = start.model.model_id;
    } else {
      state.type = (start && start.type) || 'CNN';
      state.knobs = P.defaults(state.type);
      state.custom = false;
      state.layers = P.chain(P.SKELETONS[state.type].build(state.knobs), inputArg());
    }
    state.pending = null;
    state.dropped = [];
    state.removed = [];                                  // skeleton slots removed by hand (a hand edit)
    state.preview = null;
    state.open = null;
    state.message = '';
    state.addAsk = false;                                // the Configuration Change's model_id ask closes with it
    clearSelection();
    changed(false);
  }
  /** A tab VS Code brought back after a reload: the design it kept, checked against params.yaml now; a
   *  Visualise design is staged again (the panel's staged edits did not survive the reload). */
  function restore(d) {
    Object.assign(state, { type: d.type, knobs: d.knobs, custom: !!d.custom, layers: d.layers, removed: d.removed || [],
      original: d.original || null, modelId: d.modelId || null, start: d.start || {}, id: d.id || '', open: d.open == null ? null : d.open,
      previewOpen: !!d.previewOpen, pending: null, dropped: [], preview: null, message: '', knobErrors: {}, knobText: {},
      dataset: d.dataset == null ? null : d.dataset, traceIndex: Number(d.traceIndex) || 0, observe: !!d.observe,
      traceRange: !!d.traceRange, traceTo: d.traceTo == null ? null : Number(d.traceTo), dl: d.dl || null, leftWidth: d.leftWidth || null,
      input: state.input || (Array.isArray(d.input) ? d.input : null) });   // a typed length kept over the reload (the host knows none)
    clearSelection();
    changed(false);
    syncModels();
    if (state.modelId && !P.same(P.asData(state.layers), state.original)) stageNow();
  }
  /** Visualise: the visualised model as params.yaml holds it now (state.models) against the page's design. */
  function syncModels() {
    if (!state.modelId) return;
    const fresh = state.models.find(x => x.model_id === state.modelId);
    const layers = fresh && ((fresh.architecture_args || {}).layers);
    if (!fresh) {
      state.message = `${state.modelId} is no longer in params.yaml: Add the Model adds this design as a new model`;
      state.modelId = null;
      state.original = null;
      state.layers = state.layers.map(l => ({ ...l, from: undefined }));
    } else if (P.same(P.asData(state.layers), layers)) {
      state.original = layers;                           // the file is this design now (OK here or in the panel)
      state.start = { model: fresh };                    // Reset and Cancel go back to it, not to the layers before the write
      state.layers = state.layers.map((l, i) => ({ ...l, from: i }));
    } else if (!P.same(state.original, layers)) {
      const id = state.modelId;
      state.start = { model: fresh };
      startFrom(state.start);
      state.message = `${id} changed in params.yaml: the page shows it as it is now`;
    }
  }
  /** After any change of the layers: the arguments from the layer above, a new check, and (Visualise) the
   *  staged edits of the visualised model in the panel. */
  /** Preview Code: Visualise, the model's own entry as OK would write it; Plan a Model, what Add the Model would add. */
  const askPreview = () => send(visualising()
    ? { type: 'preview', model_id: state.modelId, arrange: P.arrangeOf(state.original, state.layers) }
    : { type: 'preview', model: P.toModel(String(state.id || '').trim() || 'NEW_MODEL', state.type, state.layers) });
  let checkTimer = null;
  /** The check's input: the data entry chosen, else the length typed (a project without a data entry). */
  const checkInput = () => (state.dataset ? { dataset: state.dataset } : state.input ? { input: state.input[1] } : {});
  const sendCheck = () => {
    checkTimer = null;
    send({ type: 'check', layers: P.asData(state.layers), architecture: state.type, ...checkInput() });
    if (!state.trained) askTrace();
  };
  function changed(stage = true) {
    state.layers = P.chain(state.layers, inputArg());
    state.check = null;
    state.preview = null;
    if (!state.trained) state.trace = null;              // the lines were of the layers before this change
    if (state.layers.length && state.input) {           // no input known yet: nothing to check against
      state.checking = true;
      // one check per pause (each is a python process): the last design, 400 ms after the last change
      if (typeof window.setTimeout === 'function') {
        if (checkTimer) window.clearTimeout(checkTimer);
        checkTimer = window.setTimeout(sendCheck, 400);
      } else sendCheck();
    } else state.checking = false;
    if (stage && state.modelId) stageNow();
    if (state.previewOpen) askPreview();                 // the open preview follows the design
  }
  /** Visualise: the page's design as the arrange of the visualised model, with the layers it was made from
   *  (the host refuses it when the file's model is no longer those layers). */
  const stageNow = () => send({ type: 'stage', model_id: state.modelId, base: state.original, arrange: P.arrangeOf(state.original, state.layers) });
  // ── the trace lines' requests ─────────────────────────────────────────────────────────────────────
  const maxTrace = () => (state.trained ? null : (entryOf(state.dataset) || {}).attack || null);
  const runOf = name => (state.runs || []).find(r => r.name === name) || null;
  const entryOf = name => (state.entries || []).find(e => e.name === name) || null;
  /** One trace through the trained run picked, else through the design (once a data entry is known). */
  const RANGE_MAX = 1000;                                // trace_view.MAX_RANGE
  function askTrace() {
    const index = Number(state.traceIndex) || 0;
    const span = state.traceRange ? { to: state.traceTo == null ? index : Number(state.traceTo) } : {};
    if (!state.observe) { state.tracing = false; return; }
    if (state.trained) send({ type: 'trace', run: state.trained, trace: index, ...span });
    else if (state.layers.length && state.dataset) {
      send({ type: 'trace', layers: P.asData(state.layers), architecture: state.type, dataset: state.dataset, trace: index, ...span });
    } else { state.tracing = false; return; }
    state.tracing = true;
  }
  /** The data entries came: the one kept (a reload), else the first that can be read; its trace length is the input. */
  function sourcesArrived(m) {
    state.entries = m.entries || [];
    state.runs = m.runs || [];
    state.sourcesError = m.error || '';
    const ok = e => e && !e.why;
    if (!ok(entryOf(state.dataset))) state.dataset = ((state.entries.find(ok)) || {}).name || null;
    if (state.trained && !runOf(state.trained)) state.trained = null;
    useEntry();
  }
  /** The chosen entry's trace length as the design's input (the layers chain from it), then a new check and trace. */
  function useEntry() {
    const e = entryOf(state.dataset);
    const length = (e && e.length) || lengthOf(state.dataset);
    if (length && (state.input || [])[1] !== length) state.input = [(state.input || [1])[0] || 1, length];
    changed(false);
  }
  // what a trained run's view refuses: it shows the run's training snapshot, never an edit of it
  const EDITS = new Set(['kind', 'knob', 'resolve', 'insert', 'plus', 'arg', 'dropout', 'pattern', 'pooling', 'actAll', 'move', 'remove', 'select', 'selectAll', 'deleteSelected',
    'editSelected', 'argSelected', 'reset', 'add', 'preview', 'open']);

  function applyKnob(k, v) {
    const knobs = { ...state.knobs, [k]: v };
    if (P.hasHandEdits(state.layers, state.removed) || state.custom) {
      state.pending = { knobs, k };
      if (visualising()) act0({ type: 'resolve', choice: 'keep' });   // Visualise: no question; Undo takes it back
      return;
    }
    state.knobs = knobs;
    state.layers = P.rebuild(state.type, knobs, state.layers, inputArg(), state.removed).layers;
    clearSelection();
    changed();
  }
  // The selected layers (their indices): kept on their layers when one is moved, added or removed by hand;
  // a skeleton rebuild clears them (the layers they named may be gone).
  function clearSelection() { state.selected = []; state.editSelected = false; }
  const selected = () => state.selected || [];
  function reindex(f) {
    state.selected = selected().map(f).filter(i => i != null).sort((a, b) => a - b);
    if (!state.selected.length) state.editSelected = false;
  }
  /** One argument typed for layer i (empty: its default), set as a hand edit (a skeleton rebuild keeps it). */
  function setArg(i, key, text) {
    const l = state.layers[i];
    const value = /^-?\d+(\.\d+)?([eE]-?\d+)?$/.test(text) ? Number(text) : text === '' ? undefined
      : text === 'true' ? true : text === 'false' ? false : text;
    const args = { ...l.args };
    if (value === undefined) delete args[key]; else args[key] = value;
    state.layers[i] = { ...l, args, handArgs: [...new Set([...(l.handArgs || []), key])] };
  }
  /** Remove layer i; a skeleton layer stays removed when the skeleton is rebuilt. */
  function removeAt(i) {
    const [gone] = state.layers.splice(i, 1);
    if (gone && gone.slot && !gone.added) state.removed.push(gone.slot);
    reindex(x => (x === i ? null : x > i ? x - 1 : x));
  }
  // ── Undo / Redo (Visualise): each change of the design one step; Reset one step too ─────────────────────
  const DESIGN_ACTS = new Set(['kind', 'knob', 'resolve', 'insert', 'plus', 'arg', 'dropout', 'pattern', 'pooling', 'actAll', 'move', 'remove',
    'deleteSelected', 'argSelected', 'reset']);
  const snap = () => JSON.parse(JSON.stringify({ type: state.type, knobs: state.knobs, custom: !!state.custom, layers: state.layers,
    removed: state.removed || [], dropped: state.dropped || [] }));
  const act = a => {
    const before = state.ready && !state.trained && visualising() && DESIGN_ACTS.has(a.type) ? snap() : null;
    const after = () => {
      if (before && JSON.stringify(before) !== JSON.stringify(snap())) { state.undo = [...(state.undo || []), before]; state.redo = []; }
    };
    if (!before) return act0(a);
    act0(a);
    after();
    return render();                                     // again: the bar's Undo / Redo follow the step just kept
  };
  /** Undo (back: -1) or Redo (forward: +1) one step of the design. */
  function step(dir) {
    const from = dir < 0 ? state.undo : state.redo;
    if (!from || !from.length) return;
    const to = from.pop();
    (dir < 0 ? (state.redo = state.redo || []) : (state.undo = state.undo || [])).push(snap());
    Object.assign(state, { type: to.type, knobs: to.knobs, custom: to.custom, layers: to.layers, removed: to.removed, dropped: to.dropped,
      pending: null, knobErrors: {}, knobText: {} });
    clearSelection();
    changed();
    if (state.previewOpen) askPreview();
  }
  const act0 = a => {
    if (!state.ready) return render();
    if (state.trained && EDITS.has(a.type)) {
      state.message = `refused: these are ${state.trained}'s layers as it trained (its snapshot): Back to the design to edit`;
      return render();
    }
    if (a.type === 'observe') {                         // Observe Trace / Hide Trace: the lines and their settings
      state.observe = !state.observe;
      state.trace = null;
      state.trained = null;
      state.showRecovery = false;
      state.recovery = null;
      state.traceError = '';
      if (!state.observe) state.tracing = false;
      else if (state.entries) askTrace();
      else send({ type: 'sources', model_id: state.modelId });   // read once, when first asked for
    } else if (a.type === 'dataset' && ((entryOf(a.value) && !entryOf(a.value).why) || (!state.entries && lengthOf(a.value)))) {
      state.dataset = a.value;                           // the Input menu (before the trace sources are read) or the trace's entry
      useEntry();
    } else if (a.type === 'inputLength') {               // no data entry in params.yaml: the length typed
      const n = parseInt(a.value, 10);
      state.inputError = Number.isInteger(n) && n > 0 ? '' : 'a whole number of samples, at least 1';
      if (!state.inputError) { state.input = [(state.input || [1])[0] || 1, n]; changed(false); }
    } else if (a.type === 'traceMode') {                 // one trace, or a range: its mean and band
      state.traceRange = a.value === 'range';
      if (state.traceRange && state.traceTo == null) state.traceTo = Math.min((Number(state.traceIndex) || 0) + 99, maxTrace() == null ? Infinity : maxTrace() - 1);
      state.traceError = '';
      state.trace = null;
      askTrace();
    } else if (a.type === 'traceIndex' || a.type === 'traceTo') {
      const v = String(a.value).trim();
      const max = maxTrace();
      const from = a.type === 'traceIndex' ? Number(v) : Number(state.traceIndex) || 0;
      const to = a.type === 'traceTo' ? Number(v) : state.traceTo;
      if (!/^\d+$/.test(v) || (max && Number(v) >= max)) {
        state.traceError = max ? `a whole number from 0 to ${max - 1}` : 'a whole number from 0';
      } else if (state.traceRange && to < from) {
        state.traceError = 'the range ends before it starts';
      } else if (state.traceRange && to - from + 1 > RANGE_MAX) {
        state.traceError = `a range takes at most ${RANGE_MAX} traces`;
      } else {
        state.traceError = '';
        if (a.type === 'traceIndex') state.traceIndex = Number(v); else state.traceTo = Number(v);
        state.trace = null;
        askTrace();
      }
    } else if (a.type === 'recovery' && state.trained) {   // Show / Hide Key Recovery: what the run recorded
      state.showRecovery = !state.showRecovery;
      state.recovery = null;
      if (state.showRecovery) send({ type: 'recovery', run: state.trained });
    } else if (a.type === 'weights') {                  // '' : the design, untrained; a run's name: its trained view
      const r = a.value ? runOf(a.value) : null;
      if (a.value && (!r || r.why)) return render();
      state.trained = r ? r.name : null;
      state.recovery = null;
      if (!r) state.showRecovery = false;
      else if (state.showRecovery) send({ type: 'recovery', run: r.name });
      state.message = '';
      state.open = null;
      state.adding = null;
      clearSelection();
      state.trace = null;
      if (r) askTrace(); else changed(false);
    } else if (a.type === 'kind' && !kindBlocked(a.value)) {
      state.type = a.value;
      state.knobs = P.defaults(a.value);
      state.custom = false;
      state.knobErrors = {};
      state.knobText = {};
      state.layers = P.chain(P.SKELETONS[a.value].build(state.knobs), inputArg());
      state.removed = [];
      clearSelection();
      changed();
    } else if (a.type === 'knob') {
      const knob = (P.SKELETONS[state.type] || { knobs: [] }).knobs.find(k => k.k === a.k);
      const parsed = knob ? P.parseKnob(knob, a.value) : { error: 'no such setting' };
      if (parsed.error) {                                // the bad text stays in its box with why; the design is untouched
        state.knobErrors = { ...(state.knobErrors || {}), [a.k]: parsed.error };
        state.knobText = { ...(state.knobText || {}), [a.k]: String(a.value) };
      } else {                                           // a good value: every box shows the value the design uses again
        state.knobErrors = {};
        state.knobText = {};
        applyKnob(a.k, parsed.value);
      }
    } else if (a.type === 'resolve' && state.pending) {
      if (a.choice === 'keep' || a.choice === 'rebuild') {
        const keep = a.choice === 'keep';
        const out = P.rebuild(state.type, state.pending.knobs, keep ? state.layers : [], inputArg(), keep ? state.removed : []);
        state.knobs = state.pending.knobs;
        state.custom = false;
        state.layers = out.layers;
        if (!keep) state.removed = [];
        state.dropped = a.choice === 'keep' ? out.dropped : [];
        clearSelection();
        changed();
      }
      state.pending = null;
    } else if (a.type === 'insert' && !layerBlocked(a.name)) {
      const spec = (forms().layers[a.name] || { args: [] }).args;
      const args = Object.fromEntries(spec.filter(x => x.required).map(x => [x.k, x.d == null ? 0 : x.d]));
      state.layers = P.insert(state.layers, a.at, [{ name: a.name, args }]);
      reindex(x => (x >= a.at ? x + 1 : x));
      state.open = a.at;
      state.adding = null;
      changed();
    } else if (a.type === 'plus') {                      // the + between two layers: the layer menu at that place
      state.adding = state.adding === a.at ? null : a.at;
    } else if (a.type === 'arg') {
      setArg(a.i, a.key, a.value);
      changed();
    } else if (a.type === 'pattern') {                   // a Block Pattern chip moved, taken out or added
      return editPattern(a.op, a.at, a.kind);
    } else if (a.type === 'pooling') {                   // Pooling per block
      setPooling(a.b, a.field, a.value);
      changed();
    } else if (a.type === 'actAll' && a.value) {         // every layer that has an activation now
      state.layers.forEach((l, i) => { if (l.args.activation) setArg(i, 'activation', a.value); });
      changed();
    } else if (a.type === 'dropout') {                   // Dropout per layer
      setDropout(a.i, a.value);
      changed();
    } else if (a.type === 'select') {                    // a card's tick box
      state.selected = a.on ? [...new Set([...selected(), a.i])].sort((x, y) => x - y) : selected().filter(x => x !== a.i);
      if (!state.selected.length) state.editSelected = false;
    } else if (a.type === 'selectAll') {
      state.selected = a.on ? state.layers.map((_, i) => i) : [];
      if (!a.on) state.editSelected = false;
    } else if (a.type === 'deleteSelected' && selected().length) {
      for (const i of [...selected()].sort((x, y) => y - x)) removeAt(i);   // from the last: the indices above stay
      clearSelection();
      changed();
    } else if (a.type === 'editSelected') {              // Edit N layers: the shared argument form, or folded away
      state.editSelected = !state.editSelected && selected().length > 0;
    } else if (a.type === 'argSelected' && a.value !== MIXED) {
      for (const i of selected()) setArg(i, a.key, a.value);
      changed();
    } else if (a.type === 'move') {
      const j = a.i + a.dir;
      if (j >= 0 && j < state.layers.length) {
        [state.layers[a.i], state.layers[j]] = [state.layers[j], state.layers[a.i]];
        state.layers[j] = { ...state.layers[j], moved: true };             // its arguments still follow the layer above
        reindex(x => (x === a.i ? j : x === j ? a.i : x));
        changed();
      }
    } else if (a.type === 'remove') {
      removeAt(a.i);
      changed();
    } else if (a.type === 'open') {
      state.open = state.open === a.i ? null : a.i;
    } else if (a.type === 'id') {
      state.id = a.value;
      return idChanged();
    } else if (a.type === 'reset') {
      state.knobErrors = {};
      state.knobText = {};
      startFrom(state.start, true);                      // Reset is one step Undo can take back
      if (state.previewOpen) askPreview();
      if (state.modelId) stageNow();                     // the model as in the file: the panel drops the planner's staged change
    } else if (a.type === 'applyChanges' && state.modelId) {   // OK: the visualised model's changes written to params.yaml
      const why = okBlocked();
      if (why) state.message = `refused: nothing written: ${why}`;
      else send({ type: 'apply', model_id: state.modelId, base: state.original, arrange: P.arrangeOf(state.original, state.layers) });
    } else if (a.type === 'discardChanges' && state.modelId) {  // Cancel: the changes undone, the window closed
      startFrom(state.start);
      stageNow();
      send({ type: 'close' });
    } else if (a.type === 'preview') {                   // Preview Code / Encapsulate: the YAML shown, or folded away
      state.previewOpen = !state.previewOpen;
      state.preview = null;
      if (state.previewOpen) askPreview();
    } else if (a.type === 'add') {
      const why = addBlocked();
      if (!why) {
        const model = P.toModel(String(state.id).trim(), state.type, state.layers);
        if (!visualising()) send({ type: 'add-model', model });
        else {                                           // the Configuration Change as a new model; the opened one stays
          state.addedAs = model.model_id;                // once written: the window shows the new model
          send({ type: 'add-model', model, from: { model_id: state.modelId, base: state.original, arrange: P.arrangeOf(state.original, state.layers) } });
        }
      }
    } else if (a.type === 'addAsk') {                    // Add The Model: its model_id asked in the bar
      state.addAsk = !!a.on;
      if (!a.on) state.id = '';
    } else if (a.type === 'undo' || a.type === 'redo') {
      step(a.type === 'undo' ? -1 : 1);
    } else if (a.type === 'cancel') {
      send({ type: 'close' });
    } else if (a.type === 'download') {
      if (!dlChanges().length && dlFormats(dlSaved()).length) { send(downloadRequest()); state.message = 'downloading…'; }
    } else if (a.type === 'dlOk') {
      dlState().saved = copyOf(dl());
      keepDesign();
    } else if (a.type === 'dlCancel') {                    // undo, and fold the text_rules away
      dlState().draft = copyOf(dlSaved());
      dlState().open = {};
      keepDesign();
    } else if (a.type === 'dlReset') {                     // undo: the settings as they were before the changes
      dlState().draft = copyOf(dlSaved());
      keepDesign();
    }
    return render();
  };
  /** Why OK cannot write the visualised model's changes yet ('' when it can): the model check must pass first. */
  function okBlocked() {
    if (state.checking) return 'TraceFlow\'s model check is still running';
    if (!state.check) return 'TraceFlow\'s model check has not answered yet';
    return state.check.error ? `the model check refuses it: ${state.check.error}` : '';
  }
  /** Visualise: any change of the model's layers (a setting, a chip, a per-layer menu, a card) recognised as a
   *  Configuration Change: OK writes it over the model, Cancel undoes it and closes the window, Reset undoes it (the
   *  model as params.yaml holds it), Add The Model asks a model_id and adds the design as a new model. Back at the
   *  saved layers, the bar goes. */
  const visualising = () => !!state.modelId && !state.trained;
  function pendingHtml() {
    if (!visualising()) return '';
    const c = P.changesOf(state.original, state.layers);
    if (!c.total) return '';
    const parts = ['changed', 'added', 'removed', 'moved'].filter(k => c[k]).map(k => `${c[k]} ${k}`).join(', ');
    const why = okBlocked();
    const id = P.idState(state.id, state.ids, M.askError);
    const ask = state.addAsk ? `<div class="addask"><label class="knob"><span>new model_id</span><input data-id="1" value="${esc(state.id || '')}"` +
      ` placeholder="${esc(M.ASKS.model.hint)}" title="the model_id the design is added under (Enter adds it)"></label>` +
      `<div class="idstate ${id.ok ? 'ok' : 'err'}">${esc(id.ok ? `unique ✓${id.note ? ` (${id.note})` : ''}` : id.error)}</div>` +
      `<div class="askbtns"><button class="primary" data-add="1"${addBlocked() ? ` disabled title="${esc(addBlocked())}"` : ` title="add the design to params.yaml as a new model; ${esc(state.modelId)} stays as it is"`}>Add</button>` +
      '<button data-addaskclose="1" title="not now">×</button></div></div>' : '';
    const u = (state.undo || []).length;
    const r = (state.redo || []).length;
    const steps = `<span class="steps"><button data-undo="1"${u ? ` title="undo the last change, one step back (${u} to undo; Ctrl+Z)"` : ' disabled title="nothing to undo"'}>↶ Undo</button>` +
      `<button data-redo="1"${r ? ` title="redo the change undone, one step forward (${r} to redo; Ctrl+Shift+Z or Ctrl+Y)"` : ' disabled title="nothing to redo"'}>↷ Redo</button></span>`;
    return `<div class="pending config"><span title="${esc(`${state.modelId}'s layers: ${parts}`)}"><b>Configuration Change</b> · ${esc(state.modelId)} · ${c.total} change${c.total === 1 ? '' : 's'}</span>` +
      `<button data-applychanges="1"${why ? ` disabled title="${esc(why)}"` : ` title="${esc(`override ${state.modelId} in params.yaml with the changes`)}"`}>OK</button>` +
      `<button data-discardchanges="1" title="${esc(`undo every change and close the window; ${state.modelId} stays as it is`)}">Cancel</button>` +
      `<button data-reset="1" title="${esc(`undo every change: ${state.modelId} as it was when the window opened it; the window stays open (Undo takes the Reset back)`)}">Reset</button>` +
      `<button data-addask="1" title="${esc(`asks a new model_id first, then adds the changed design under it; ${state.modelId} stays unchanged`)}">Add the Model</button>${steps}${ask}</div>`;
  }
  function addBlocked() {
    if (!state.layers.length) return 'the design has no layers';
    if (state.checking) return 'TraceFlow\'s model check is still running';
    if (!state.check) return 'TraceFlow\'s model check has not answered yet';
    if (state.check.error) return `the model check refuses it: ${state.check.error}`;
    const id = P.idState(state.id, state.ids, M.askError);
    return id.ok ? '' : id.error;
  }

  // ── drawing ────────────────────────────────────────────────────────────────────────────────────────
  const opt = (v, cur, label = v, blocked = '', hint = '') =>
    `<option value="${esc(v)}"${String(v) === String(cur) ? ' selected' : ''}${blocked ? ` disabled title="${esc(blocked)}"` : hint ? ` title="${esc(hint)}"` : ''}>` +
    `${esc(label)}${blocked ? ' (not available)' : ''}</option>`;
  function addMenu(at) {
    const f = forms();
    const groups = f.groups.map(g => {
      const names = Object.keys(f.layers).filter(n => f.layers[n].group === g);
      return names.length ? `<optgroup label="${esc(g)}">${names.map(n => opt(n, '', n, layerBlocked(n))).join('')}</optgroup>` : '';
    }).join('');
    return `<div class="plusrow open"><select class="addlayer" data-insert="${at}" title="choose the layer to add here (the first entry closes this menu)">` +
      `<option value="">choose a layer…</option>${groups}</select></div>`;
  }
  /** The + left of the gap above layer `at` (at = the count: after the last layer), or the layer menu it opened. */
  const plus = at => (state.adding === at ? addMenu(at)
    : `<div class="plusrow"><button class="plus" data-plus="${at}" title="${esc(at === 0 ? 'add a layer before the first one'
      : at === state.layers.length ? 'add a layer after the last one' : `add a layer between #${at - 1} and #${at}`)}">+</button></div>`);
  /** An option a layer argument takes under PyTorch only: in the menu under keras, never choosable, saying so. */
  function argBlocked(name, a, o) {
    if (!keras()) return '';
    const f = forms().keras || {};
    const keep = name === 'Pooling' && a.k === 'type' ? f.pooling : a.k === 'activation' ? f.activations : null;
    return keep && !keep.includes(o) ? 'available in pytorch, not under deep_learning.framework: keras' : '';
  }
  function argsHtml(l, i) {
    const spec = (forms().layers[l.name] || { args: [] }).args;
    return `<div class="args">${spec.map(a => `<label class="arg${a.required ? ' req' : ''}"><span>${esc(a.k)}</span>` +
      (a.o ? `<select data-arg="${esc(a.k)}" data-i="${i}" title="${esc(`${l.name}'s ${a.k}`)}">${opt('', l.args[a.k], '(default)')}${a.o.map(o => opt(o, l.args[a.k], o, argBlocked(l.name, a, o))).join('')}</select>`
        : `<input data-arg="${esc(a.k)}" data-i="${i}" value="${esc(l.args[a.k] == null ? '' : l.args[a.k])}" placeholder="${esc(a.d == null ? '' : a.d)}">`) +
      '</label>').join('')}</div>`;
  }
  /** Edit N layers: the arguments every selected layer takes (all of one type's; of mixed types, the ones they
   *  share, a menu offering only the options all take); a value that differs between them shows (mixed). */
  const MIXED = '__mixed__';                           // the (mixed) option: changes nothing
  function selectedArgsHtml() {
    const layers = selected().map(i => state.layers[i]).filter(Boolean);
    const specs = [...new Set(layers.map(l => l.name))].map(n => (forms().layers[n] || { args: [] }).args);
    const shared = specs[0].filter(a => specs.every(sp => sp.some(b => b.k === a.k))).map(a => {
      const menus = specs.map(sp => sp.find(b => b.k === a.k).o).filter(Boolean);
      return { ...a, o: menus.length === specs.length ? menus.reduce((x, y) => x.filter(o => y.includes(o))) : null };
    });
    if (!shared.length) return '<div class="args bulk"><small class="note">these layers share no argument</small></div>';
    const n = layers.length;
    return `<div class="args bulk">${shared.map(a => {
      const vals = layers.map(l => JSON.stringify(l.args[a.k] === undefined ? null : l.args[a.k]));
      const v = vals.every(x => x === vals[0]) ? layers[0].args[a.k] : MIXED;
      const mixed = v === MIXED;
      return `<label class="arg${a.required ? ' req' : ''}"><span>${esc(a.k)}</span>` + (a.o
        ? `<select data-argsel="${esc(a.k)}" title="${esc(`${a.k} of the ${n} selected layers`)}">${mixed ? `<option value="${MIXED}" selected>(mixed)</option>` : ''}` +
          `${opt('', mixed ? MIXED : v, '(default)')}${a.o.map(o => opt(o, v, o, layers.map(l => argBlocked(l.name, a, o)).find(Boolean) || '')).join('')}</select>`
        : `<input data-argsel="${esc(a.k)}" value="${esc(mixed || v == null ? '' : v)}" placeholder="${esc(mixed ? '(mixed)' : a.d == null ? '' : a.d)}"` +
          ` title="${esc(`${a.k} of the ${n} selected layers`)}">`) + '</label>';
    }).join('')}</div>`;
  }
  /** Over the cards: Select all, how many are selected, Delete N layers and Edit N layers. */
  function selectionHtml() {
    const n = selected().length;
    const all = n > 0 && n === state.layers.length;
    const off = n ? '' : ' disabled';
    const word = `${n} layer${n === 1 ? '' : 's'}`;
    return `<div class="selbar"><label class="selall" title="select every layer"><input type="checkbox" data-selall="1"${all ? ' checked' : ''}` +
      ` title="select every layer"> Select all</label><small>${n} selected</small>` +
      `<button class="danger" data-delsel="1"${off} title="${esc(n ? `remove the ${word} selected` : 'tick layers to remove them')}">Delete ${esc(word)}</button>` +
      `<button data-editsel="1"${off} title="${esc(n ? (state.editSelected ? 'fold the shared arguments away' : `set an argument on the ${word} selected at once`) : 'tick layers to edit them together')}">` +
      `${state.editSelected ? 'Done' : `Edit ${esc(word)}`}</button></div>` + (state.editSelected && n ? selectedArgsHtml() : '');
  }
  /** The cards drawn: the design's layers, or (trained view) the run's layers as its snapshot holds them. */
  function shownLayers() {
    const r = state.trained ? runOf(state.trained) : null;
    return r ? (r.layers || []).map(l => { const [name] = Object.keys(l); return { name, args: l[name] || {} }; }) : state.layers;
  }
  const traceLayer = i => (state.trace && state.trace.layers ? state.trace.layers.find(x => x.index === i) : null);
  const shapeOf = i => (state.trained ? (traceLayer(i) || {}).shape || null
    : state.check && state.check.layers ? (state.check.layers.find(x => x.index === i) || {}).out : null);
  // ── the scale: every bar measured on one log axis (a layer's length, or a vector's features) ─────────
  const SCALE_W = 170;
  const sizeOf = out => (out.length === 2 ? out[1] : out[0]);
  /** The axis runs from 1 to the largest size of the design (at least the trace length). */
  function scaleMax() {
    const sizes = shownLayers().map((_, i) => shapeOf(i)).filter(Boolean).map(sizeOf).filter(Number.isFinite);
    return Math.max(2, Number((state.input || [1, 700])[1]) || 2, ...sizes);
  }
  const xOn = (v, max, w) => w * Math.log10(Math.max(1, v)) / Math.log10(max);
  const xAt = (v, max) => xOn(v, max, SCALE_W);
  /** Ticks at 1, 10, 100, ... and the axis end; a power of ten whose label would touch the end's gives way to it. */
  const ticks = max => P.scaleTicks(max, SCALE_W);
  function scaleSvg() {
    const max = scaleMax();
    const ts = ticks(max);
    const marks = ts.map((t, k) => {
      const x = Math.min(SCALE_W - 0.5, Math.max(0.5, xAt(t, max))).toFixed(1);
      const anchor = k === 0 ? 'start' : k === ts.length - 1 ? 'end' : 'middle';
      return `<line x1="${x}" x2="${x}" y1="17" y2="22"/><text x="${x}" y="13" text-anchor="${anchor}">${t}</text>`;
    }).join('');
    return `<svg class="scale" width="${SCALE_W}" height="24"><line x1="0" x2="${SCALE_W}" y1="22" y2="22"/>${marks}</svg>`;
  }
  function barSvg(i) {
    const out = shapeOf(i);
    const max = scaleMax();
    const grid = ticks(max).map(t => { const x = Math.min(SCALE_W - 0.5, Math.max(0.5, xAt(t, max))).toFixed(1); return `<line class="grid" x1="${x}" x2="${x}" y1="0" y2="22"/>`; }).join('');
    if (!out) return `<svg class="bar" width="${SCALE_W}" height="22">${grid}</svg>`;
    const size = sizeOf(out);
    const ch = out.length === 2 ? out[0] : 1;
    const w = Math.max(2, Math.min(SCALE_W, xAt(size, max)));
    const h = Math.max(2, Math.min(20, 2 + 2 * Math.log2(ch + 1)));        // thickness: the channels
    const tip = out.length === 2 ? `length ${size} × ${ch} channels` : `${size} features`;
    return `<svg class="bar" width="${SCALE_W}" height="22"><title>${esc(tip)}</title>${grid}` +
      `<rect class="g-${esc(groupOf(shownLayers()[i].name))}" x="0" y="${((22 - h) / 2).toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="2"/></svg>`;
  }
  // ── the trace lines ──────────────────────────────────────────────────────────────────────────────────
  const LINE_W = 300;
  const fmt = v => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01) ? Number(v).toExponential(1) : String(Number(Number(v).toPrecision(3))));
  const still = () => typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  /** One line of values (scaled to its own range, a zero line when it crosses 0), drawn after `k` earlier lines
   *  when the trace has just arrived (SVG animate: the CSP allows no inline style), at once with reduced motion. */
  function lineSvg(values, group, k, band) {
    const v = (values || []).filter(Number.isFinite);
    if (!v.length) return '';
    const all = band ? [...v, ...band[0], ...band[1]].filter(Number.isFinite) : v;
    const lo = Math.min(...all);
    const hi = Math.max(...all);
    const y = x => (26 - (hi > lo ? (x - lo) / (hi - lo) : 0.5) * 24 + 1).toFixed(1);
    const xAt = (j, n) => (n > 1 ? j * LINE_W / (n - 1) : LINE_W / 2).toFixed(1);
    const pts = v.map((x, j) => `${xAt(j, v.length)},${y(x)}`).join(' ');
    const zero = lo < 0 && hi > 0 ? `<line class="zero" x1="0" x2="${LINE_W}" y1="${y(0)}" y2="${y(0)}"/>` : '';
    const draw = state.traceFresh && !still();
    // a range: the band from the lowest to the highest value at each point, under the mean
    const area = band && band[0].length === v.length ? `<polygon class="band" points="${[...band[1].map((x, j) => `${xAt(j, v.length)},${y(x)}`),
      ...band[0].map((x, j) => `${xAt(j, v.length)},${y(x)}`).reverse()].join(' ')}"${draw ? ' opacity="0"' : ''}>` +
      (draw ? `<animate attributeName="opacity" from="0" to="1" begin="${(k * 0.12).toFixed(2)}s" dur="0.45s" fill="freeze"/>` : '') + '</polygon>' : '';
    return `<svg class="trace g-${esc(group)}" viewBox="0 0 ${LINE_W} 28" preserveAspectRatio="none">${zero}${area}` +
      `<polyline points="${pts}"${draw ? ' pathLength="1" stroke-dasharray="1" stroke-dashoffset="1"' : ''}>` +
      (draw ? `<animate attributeName="stroke-dashoffset" from="1" to="0" begin="${(k * 0.12).toFixed(2)}s" dur="0.45s" fill="freeze"/>` : '') +
      '</polyline></svg>';
  }
  function traceRow(i, name) {
    const t = traceLayer(i);
    if (!t) return '';
    const what = t.kind === 'channels' ? `rms of ${t.shape[0]} channels × ${t.shape[1]}` : `${t.shape[0]} values`;
    const of = t.band ? 'the traces: the mean' : 'the trace';
    return `<div class="row tracerow">${lineSvg(t.values, groupOf(name), i + 1, t.band)}` +
      `<small class="tracecap" title="${esc(`what #${i} ${name} makes of ${of} of ${what}, from ${t.lo} to ${t.hi}`)}">${esc(`${fmt(t.lo)} … ${fmt(t.hi)}`)}</small></div>`;
  }
  /** Whose weights, which trace (the line over the trace lines, and the downloaded diagram's caption). */
  function traceWhere() {
    const t = state.trace || {};
    const src = t.source || {};
    const many = src.to != null && src.to !== src.trace;
    const which = many ? `traces ${src.trace}–${src.to} (mean and band)` : `trace ${src.trace}`;
    return state.trained
      ? `${state.trained}'s weights (${(t.checkpoint || {}).policy || '?'}, epoch ${(t.checkpoint || {}).epoch || '?'}) · ${which} of its test.h5 (${src.dataset}, ${src.type})`
      : `untrained weights · ${which} of ${src.dataset} (${src.type})${src.desync_offset ? ` · desync offset ${src.desync_offset}` : ''}`;
  }
  /** Over the cards: whose weights, which trace; the trace itself as the first line. */
  function traceHead() {
    const t = state.trace;
    if (state.tracing && !t) return '<p class="note tracenote">tracing…</p>';
    if (!t) return state.sourcesError ? `<p class="err tracenote">${esc(`trace lines: ${state.sourcesError}`)}</p>` : '';
    const src = t.source || {};
    const many = src.to != null && src.to !== src.trace;
    const where = traceWhere();
    const from = [src.file, many ? `${src.group || ''} rows ${src.row} to ${src.row_to}` : `${src.group || ''} row ${src.row}`,
      src.samples ? `samples ${src.samples[0]} to ${src.samples[1] - 1}` : '',
      src.desync_offset != null ? `desync offset ${src.desync_offset}` : src.desync ? `a desync offset of 0 to ${src.desync} per trace` : '',
      `normalization ${src.normalization || 'none'}`].filter(Boolean).join(' · ');
    const head = `<p class="note tracenote" title="${esc(from)}">${esc(where)}</p>`;
    const err = t.error ? `<p class="err tracenote">${esc(`trace${t.at != null ? ` at layer #${t.at}` : ''}: ${t.error}`)}</p>` : '';
    const input = t.input ? `<div class="row tracerow">${lineSvg(t.input.values, 'input', 0, t.input.band)}` +
      `<small class="tracecap" title="${esc(`the ${many ? 'traces' : 'trace'} the first layer takes: ${t.input.length} samples`)}">${esc(`${many ? `${t.count} traces` : 'trace'} · ${t.input.length} samples`)}</small></div>` : '';
    const desync = desyncNote(t);
    return head + notesHtml(desync) + err + input;
  }
  /** A faint note as short points: each a bold label, then its sentence. */
  const notesHtml = items => (items.length
    ? `<ul class="faint tracefaint">${items.map(([label, text]) => `<li><b>${esc(label)}:</b> ${esc(text)}</li>`).join('')}</ul>` : '');
  /** The note under the header: what the desync offset is, with this trace's numbers ([label, text] points). */
  function desyncNote(t) {
    const src = t.source || {};
    if (!src.desync) return [];
    const many = src.to != null && src.to !== src.trace;
    const out = [['Desync', `${src.dataset} sets attack_desync: ${src.desync}, so each trace's ${src.length || ''}-sample window starts a random ` +
      `0 to ${src.desync} samples late, imitating misaligned traces (a jitter countermeasure) the network has to cope with.`]];
    if (many) return [...out, ['Each trace', `has its own shift; the mean and band are over the ${t.count} shifted windows.`]];
    if (src.desync_offset == null) return out;
    if (state.trained) return [...out, [`Desync offset ${src.desync_offset}`, `the shift of trace ${src.trace} that ${state.trained} drew and stored in its test.h5.`]];
    const [a, b] = src.samples || [0, 0];
    return [...out,
      [`Desync offset ${src.desync_offset}`, `trace ${src.trace} starts ${src.desync_offset} samples late: samples ${a} to ${b - 1} of row ${src.row}, ` +
        `instead of ${a - src.desync_offset} to ${b - src.desync_offset - 1}.`],
      ['Drawn here', 'from the run\'s seed and the trace number; a run making its own test.h5 draws and stores its own.']];
  }
  /** The note under the top classes: what the classes, the true class and its rank are, with their values. */
  function classNote(t) {
    const g = t.target;
    if (!g) return [];
    const outputs = t.outputs == null ? g.classes : t.outputs;
    const sweep = k => ((g.swept || {})[k] ? ` (the first of the sweep ${g.swept[k].join(', ')})` : '');
    const out = [
      [`Outputs ${outputs}`, outputs === g.classes
        ? `the last layer's, one per class of the ${g.leakage_model} leakage model (${g.classes} classes); out_features: auto takes that count.`
        : `the last layer's, but the ${g.leakage_model} leakage model has ${g.classes} classes: the outputs do not match the labels ` +
          '(out_features: auto takes the leakage model\'s count).'],
      ['Settings', `sca.leakage_model ${g.leakage_model}${sweep('leakage_model')}, sca.attack_byte ${g.attack_byte}${sweep('attack_byte')}, from ${g.from}.`],
      ['Classes', `each class is one possible value of ${g.expression}, not a key byte.`]];
    const chance = (g.classes + 1) / 2;
    if (t.ranks) {
      return [...out,
        [`Median rank ${t.ranks.median} of ${g.classes}`, `the true class over ${t.ranks.count} traces, each ranked alone (best ${t.ranks.best}, ` +
          `worst ${t.ranks.worst}); a random guess averages ${chance}.`],
        ['Not guessing entropy', 'that combines the evidence of many traces (the run\'s rank curves).']];
    }
    if (!t.top) return out;
    const n = t.top.length;
    out.push([`Top classes ${t.top.map(c => c[0]).join(', ')}`, `the ${n} value${n === 1 ? '' : 's'} ` +
      `${state.trained ? `${state.trained} scores` : 'these weights score'} highest for trace ${(t.source || {}).trace}, best first` +
      `${state.trained ? '.' : '; the weights are untrained (random), so these guesses mean nothing yet.'}`]);
    if (t.label == null) return out;
    const bytes = t.bytes ? `, from its plaintext byte p[${g.attack_byte}] = ${t.bytes.plaintext} and the key byte k[${g.attack_byte}] = ${t.bytes.key}` : '';
    const side = t.label_rank === 1 ? 'the network guessed it right on this one trace' : t.label_rank < chance
      ? 'the network did better than chance on this one trace' : t.label_rank > chance ? 'the network did worse than chance on this one trace' : 'exactly chance';
    return [...out,
      [`True class ${t.label}`, `the real value for this trace${bytes}.`],
      [`Rank ${t.label_rank} of ${g.classes}`, `a random guess averages ${chance}, so ${side}.`],
      ['One trace', 'leaks little; key recovery combines many traces (the run\'s guessing entropy).']];
  }
  // ── Show Key Recovery: what the trained run recorded (the registries), as a table and short points ───
  const yes = v => (v === true ? '✓ yes' : v === false ? '✗ no' : v == null ? '—' : String(v));
  const krRow = (label, o) => `<tr><td>${esc(label)}</td><td class="${o.key_recovered === true ? 'ok' : o.key_recovered === false ? 'err' : ''}">` +
    `${esc(yes(o.key_recovered))}</td><td>${esc(o.recovered_key == null ? '—' : o.recovered_key)}</td><td>${esc(o.recovered_key_rank == null ? '—' : o.recovered_key_rank)}</td>` +
    `<td>${esc(o.stable_rank_1 || '—')}</td></tr>`;
  function recoveryHtml() {
    if (!state.showRecovery) return '';
    const k = state.recovery;
    if (!k) return '<p class="note tracenote">reading the key recovery…</p>';
    if (k.error) return `<p class="err tracenote">${esc(`key recovery: ${k.error}`)}</p>`;
    const rk = (k.rules || {}).recovery_rank;
    const head = `<tr><th>Attack</th><th>Recovered</th><th>Top key guess</th><th>True key rank</th><th>${rk == null || rk === 1 ? 'Rank 1' : `Rank ≤ ${rk}`} from</th></tr>`;
    const rows = [];
    if (k.cv) {
      for (const f of k.folds || []) rows.push(krRow(`fold ${f.fold}`, f));
      if (k.folds_note) rows.push(`<tr><td>folds</td><td colspan="4">${esc(k.folds_note)}</td></tr>`);
    } else if (k.validation) rows.push(krRow('validation', k.validation));
    // a test attacked (test_registry.json): its outcome; one that was not (SKIPPED, FAILED, running): its status and why
    const testRow = (label, o) => (o.status === 'DONE' || typeof o.key_recovered === 'boolean' ? krRow(label, o)
      : `<tr><td>${esc(label)}</td><td colspan="4">${esc(`${o.status || '?'}${o.status_reason ? ` · ${o.status_reason}` : ` · ${yes(o.key_recovered)}`}`)}</td></tr>`);
    rows.push(k.test ? testRow('recorded test', k.test) : '<tr><td>recorded test</td><td colspan="4">not tested yet</td></tr>');
    for (const e of k.external || []) {                 // each external test with its attack: 100× shuffled, full_population, step 1
      const a = ((e.settings || {}).perform_attack) || {};
      const t = a.test || {};
      const how = t.attack_times != null ? ` · ${t.attack_times}× ${t.shuffle ? 'shuffled' : 'in order'}, ${t.attack_distribution || '?'}` +
        `${t.attack_trace_budget != null ? `, ${t.attack_trace_budget} traces each` : ''}` +
        `${a.attack_step != null ? `, step ${a.attack_step}` : ''}` : '';
      rows.push(testRow(`external test ${e.index}${how}`, e));
    }
    const won = (k.folds || []).filter(f => f.key_recovered === true).length;
    const summary = k.cv && (k.folds || []).length ? ` · ${won} of ${k.folds.length} folds recovered the key` : '';
    // every number below is the run's: its snapshot (training, attacks) and its key-recovery reports (rules, traces)
    const tr = k.training || {};
    const rules = k.rules || {};
    const num = v => (v == null ? '?' : Number(v).toLocaleString('en'));
    const at = tr.attack || {};
    const attackText = a => (a && a.attack_times ? `${a.attack_times}× ${a.shuffle ? 'shuffled' : 'in order'}, ${a.attack_distribution}` +
      (a.attack_trace_budget != null ? `, ${a.attack_trace_budget} traces each` : '') : null);
    const r = rules.recovery_rank;
    const rankWord = r == null ? 'the recovery rank' : r === 1 ? 'rank 1' : `rank ${r} or better`;
    // the recorded test: what happened to it, never assumed
    const t = k.test || {};
    const tested = t.status === 'DONE' || typeof t.key_recovered === 'boolean';
    const testClause = tested ? `attacked on its ${num(t.traces || tr.test)} test traces (${attackText(at.test) || 'perform_attack.test'})`
      : t.status ? `its recorded test on the ${num(tr.test)} test traces was not run (${t.status}${t.status_reason ? `: ${t.status_reason}` : ''})`
        : `its ${num(tr.test)} test traces are not attacked yet`;
    const style = tr.cv ? `trained with ${tr.n_folds}-fold cross-validation (${tr.strategy})` : 'trained with a hold-out split (train / validation / test)';
    const sizes = [...new Set((k.folds || []).map(f => f.traces).filter(v => v != null))].sort((x, y) => x - y);
    const foldSize = !sizes.length ? 'traces as its report records' : sizes.length === 1 ? `${num(sizes[0])} traces` : `${num(sizes[0])} to ${num(sizes.at(-1))} traces`;
    const how = tr.cv
      ? [['Cross-validation', `the ${num(tr.profiling)} profiling traces in ${tr.n_folds} folds: each fold's model trains on the other ${tr.n_folds - 1} ` +
          `and is attacked on its held-out fold (${foldSize}); ` + (tr.refit === false
            ? 'no final model is trained (cross_validation.refit.enabled: false), so there is no model to test.'
            : `then the final model is refit on all ${num(tr.profiling)}; ${testClause}.`)]]
      : [['Hold-out split', `the model trains on the ${num(tr.profiling)} profiling traces (train.h5), is checked on ${num(tr.validation)} validation ` +
          `traces (val.h5); ${testClause.replace(/^attacked/, 'it is attacked')}.`]];
    const notes = [...how,
      ['Recovered', `the true key reached ${rankWord} (key_recovery_reporting.recovery_rank) after the attack combined the evidence of all its traces` +
        (rules.stable_required === true ? ', and stayed there to the last trace (require_stable_recovery: true).'
          : rules.stable_required === false ? '; reaching it once is enough (require_stable_recovery: false).' : '.')],
      ['Top key guess', 'the key byte the attack ranked first; the true key when recovered.'],
      ['True key rank', `where the true key ended among the ${num(rules.keyspace)} key guesses; ${rankWord} is recovered.`],
      [`${r === 1 || r == null ? 'Rank 1' : `Rank ≤ ${r}`} from`, `the trace count from which the true key stays at ${rankWord}, out of the attack's traces.`],
      ...(tr.cv ? [['Folds', `each fold's model, trained without that fold, attacked on its held-out traces (perform_attack.validation: ` +
          `${attackText(at.validation) || 'not attacked'}); the final model is not attacked on them.`]]
        : [['Validation', `the model attacked on its validation traces (perform_attack.validation: ${attackText(at.validation) || 'not attacked'}).`]]),
      ['Recorded test', `the final model on its test traces (perform_attack.test: ${attackText(at.test) || 'attack_times 0, not attacked'})` +
        `${(k.external || []).length ? '; external tests use the settings on their rows (test_config.yaml)' : ''}.`]];
    return `<div class="recovery"><p class="note tracenote"><b>${esc(`Key recovery · ${k.run}`)}</b>${esc(`${k.training ? ` · ${style}` : ''} · true key ${k.true_key || '?'}${summary}`)}</p>` +
      `<table class="kr">${head}${rows.join('')}</table>${notesHtml(notes)}</div>`;
  }
  /** Under the last card: the output's top classes; in the trained view the true class's rank. */
  function traceTail() {
    const t = state.trace;
    if (t && t.ranks) {
      return `<p class="note tracetop" title="each trace ranked alone (no evidence combined across traces: not guessing entropy)">` +
        `${esc(`true class: median rank ${t.ranks.median} over ${t.ranks.count} single traces (best ${t.ranks.best}, worst ${t.ranks.worst})`)}</p>` +
        notesHtml(classNote(t));
    }
    if (!t || !t.top) return '';
    const rank = t.label != null ? ` · true class ${t.label}: rank ${t.label_rank}` : '';
    return `<p class="note tracetop" title="the classes of the largest outputs (their order is the softmax's)">${esc(`top classes ${t.top.map(c => c[0]).join(', ')}${rank}`)}</p>` +
      notesHtml(classNote(t));
  }
  // ── the window follows the Download style (its draft: the menu changes it at once; OK keeps it for Download):
  // Full Version = the cards and their bars, Layer cards = the cards alone, Scaled Sparkline = the bars alone ──
  const pageStyle = () => (dl().style === 'cards' || dl().style === 'sparkline' ? dl().style : 'full');
  const SPARK_W = 220;                                  // the Scaled Sparkline's bar length on the page, px
  const thick = ch => Math.max(2, Math.min(12, 2 + 2 * Math.log2(ch + 1)));
  const argsText = args => Object.entries(args || {}).map(([k, v]) => `${k} ${Array.isArray(v) ? `[${v}]` : v}`).join(' · ');
  /** The layer legend of the Scaled Sparkline: each layer type's colour and how many (×n), at the corner chosen. */
  function sparkLegend(where) {
    const seen = new Map();
    for (const l of shownLayers()) seen.set(l.name, (seen.get(l.name) || 0) + 1);
    return `<div class="splegend ${where.endsWith('right') ? 'right' : 'left'}">` +
      [...seen].map(([name, n]) => `<span class="g-${esc(groupOf(name))}"><i class="swatch"></i>${esc(`${name} ×${n}`)}</span>`).join('') + '</div>';
  }
  /** Scaled Sparkline on the page: one bar per layer on the log scale (length: output length or features; thickness:
   *  channels), its index at the bar's start, the arguments outside the bars; portrait: layers down the page,
   *  landscape: across it, the bars growing up from one baseline. The layers are not edited here (the cards do). */
  function sparklineHtml() {
    const o = { arguments: true, orientation: 'portrait', ticks: true, legend: 'top-right', ...(dl().sparkline || {}) };
    const layers = shownLayers();
    const max = scaleMax();
    const ts = P.scaleTicks(max, SPARK_W);
    const at = v => Math.min(SPARK_W - 0.5, Math.max(0.5, xOn(v, max, SPARK_W)));
    const legend = where => (o.legend !== 'none' && o.legend.startsWith(where) ? sparkLegend(o.legend) : '');
    const tipOf = (l, i) => { const out = shapeOf(i); return !out ? `#${i} ${l.name}` : out.length === 2 ? `#${i} ${l.name}: length ${sizeOf(out)} × ${out[0]} channels` : `#${i} ${l.name}: ${sizeOf(out)} features`; };
    const note = `<p class="note sparknote" title="the Download's style: Full Version or Layer cards to edit the layers">${esc(`Scaled Sparkline · ${layers.length} layers${o.ticks ? ' · output length (log scale) · thickness: channels' : ''}`)}</p>`;
    if (o.orientation === 'portrait') {
      const marks = ts.map((t, k) => `<line x1="${at(t).toFixed(1)}" x2="${at(t).toFixed(1)}" y1="17" y2="22"/><text x="${at(t).toFixed(1)}" y="13" text-anchor="${k === 0 ? 'start' : k === ts.length - 1 ? 'end' : 'middle'}">${t}</text>`).join('');
      const axis = o.ticks ? `<div class="sprow spaxisrow"><span class="spidx"></span><svg class="scale spaxis" width="${SPARK_W}" height="24"><line x1="0" x2="${SPARK_W}" y1="22" y2="22"/>${marks}</svg></div>` : '';
      const grid = o.ticks ? ts.map(t => `<line class="grid" x1="${at(t).toFixed(1)}" x2="${at(t).toFixed(1)}" y1="0" y2="18"/>`).join('') : '';
      const rows = layers.map((l, i) => {
        const out = shapeOf(i);
        const bar = out ? `<rect class="g-${esc(groupOf(l.name))}" x="0" y="${((18 - thick(out.length === 2 ? out[0] : 1)) / 2).toFixed(1)}" width="${Math.max(2, Math.min(SPARK_W, xOn(sizeOf(out), max, SPARK_W))).toFixed(1)}" height="${thick(out.length === 2 ? out[0] : 1).toFixed(1)}" rx="2"/>` : '';
        return `<div class="sprow"><span class="spidx">#${i}</span><svg class="spbar" width="${SPARK_W}" height="18"><title>${esc(tipOf(l, i))}</title>${grid}${bar}</svg>` +
          `${o.arguments ? `<small class="spargs" title="${esc(`#${i} ${l.name}`)}">${esc(argsText(l.args))}</small>` : ''}</div>`;
      }).join('');
      return `${note}<div class="spark portrait">${legend('top')}${axis}${rows}${legend('bottom')}</div>`;
    }
    // landscape: a column per layer, the bars growing up from one baseline, the index under each, the arguments above
    const colW = 26, gap = 6, tickW = o.ticks ? 34 : 0, textH = o.arguments ? 150 : 0, top = 8;
    const base = top + textH + SPARK_W;
    const width = tickW + layers.length * (colW + gap);
    const axis = o.ticks ? `<line class="axis" x1="${tickW - 2}" x2="${tickW - 2}" y1="${top + textH}" y2="${base}"/>` +
      ts.map(t => `<line class="axis" x1="${tickW - 6}" x2="${tickW - 2}" y1="${(base - at(t)).toFixed(1)}" y2="${(base - at(t)).toFixed(1)}"/>` +
        `<text class="tick" x="${tickW - 8}" y="${(base - at(t) + 3).toFixed(1)}" text-anchor="end">${t}</text>` +
        `<line class="grid" x1="${tickW}" x2="${width}" y1="${(base - at(t)).toFixed(1)}" y2="${(base - at(t)).toFixed(1)}"/>`).join('') : '';
    const cols = layers.map((l, i) => {
      const x = tickW + i * (colW + gap);
      const cx = x + colW / 2;
      const out = shapeOf(i);
      const bw = thick(out && out.length === 2 ? out[0] : 1);
      const bh = out ? Math.max(2, Math.min(SPARK_W, xOn(sizeOf(out), max, SPARK_W))) : 0;
      const bar = out ? `<rect class="g-${esc(groupOf(l.name))}" x="${(cx - bw / 2).toFixed(1)}" y="${(base - bh).toFixed(1)}" width="${bw.toFixed(1)}" height="${bh.toFixed(1)}" rx="2"><title>${esc(tipOf(l, i))}</title></rect>` : '';
      const args = o.arguments ? `<text class="sparg" transform="rotate(-90 ${cx.toFixed(1)} ${(top + textH - 6).toFixed(1)})" x="${cx.toFixed(1)}" y="${(top + textH - 6).toFixed(1)}"><title>${esc(argsText(l.args))}</title>${esc(argsText(l.args).slice(0, 28))}</text>` : '';
      return `${bar}${args}<text class="spidx" x="${cx.toFixed(1)}" y="${base + 13}" text-anchor="middle">#${i}</text>`;
    }).join('');
    return `${note}<div class="spark landscape">${legend('top')}<svg class="sparkland" width="${width}" height="${base + 18}">${axis}${cols}</svg>${legend('bottom')}</div>`;
  }
  function cardsHtml() {
    const style = pageStyle();
    if (style === 'sparkline') return (state.trained ? recoveryHtml() : selectionHtml()) + sparklineHtml();
    const bars = style === 'full';
    const axis = bars ? `<div class="row axisrow"><small class="scalecap" title="each bar: the layer's output length (a vector's features) on a log scale; its thickness: the channels">` +
      `output length (log scale) · thickness: channels</small>${scaleSvg()}</div>` : '';
    const row = bars ? 'row' : 'row nobar';
    const bar = i => (bars ? barSvg(i) : '');
    if (state.trained) return recoveryHtml() + traceHead() + axis + trainedCards(row, bar) + traceTail();
    return selectionHtml() + traceHead() + axis + state.layers.map((l, i) => {
      const summary = Object.entries(l.args).filter(([k]) => k !== 'activation').slice(0, 4).map(([k, v]) => `${k} ${Array.isArray(v) ? `[${v}]` : v}`).join(' · ');
      const out = shapeOf(i);
      const bad = state.check && state.check.at === i;
      const mk = markOf(l, (l.handArgs || []).length);
      const tag = l.added && l.from == null ? '<span class="tag">added</span>' : mk.tag ? `<span class="tag${mk.cls}" title="${esc(mk.tip.trim().replace(/^\(|\)$/g, ''))}">${mk.tag}</span>` : '';
      const on = selected().includes(i);
      return `${plus(i)}<div class="${row}"><div class="card g-${esc(groupOf(l.name))}${bad ? ' bad' : ''}${on ? ' picked' : ''}">` +
        `<input type="checkbox" class="sel" data-sel="${i}"${on ? ' checked' : ''} title="${esc(`select #${i} for Delete or Edit`)}">` +
        `<button class="name" data-open="${i}" title="${esc(`${state.open === i ? 'hide' : 'show'} the arguments of #${i} ${l.name}`)}">#${i} ${esc(l.name)}</button>` +
        `<small>${esc(summary)}${l.args.activation ? ` · ${esc(l.args.activation)}` : ''}</small>${tag}` +
        `<span class="out">${out ? `(${esc(out.join(' × '))})` : bad ? '✗' : ''}</span>` +
        `<span class="acts"><button data-move="${i}" data-dir="-1" title="move this layer up">▲</button><button data-move="${i}" data-dir="1" title="move this layer down">▼</button>` +
        `<button class="x" data-remove="${i}" title="remove this layer">×</button></span></div>${bar(i)}</div>` +
        traceRow(i, l.name) + (state.open === i ? argsHtml(l, i) : '');
    }).join('') + plus(state.layers.length) + traceTail();
  }
  /** The trained view's cards: the run's layers, read-only (no +, ▲▼×, tick box or arguments). */
  function trainedCards(row = 'row', bar = barSvg) {
    return shownLayers().map((l, i) => {
      const summary = Object.entries(l.args).filter(([k]) => k !== 'activation').slice(0, 4).map(([k, v]) => `${k} ${Array.isArray(v) ? `[${v}]` : v}`).join(' · ');
      const out = shapeOf(i);
      return `<div class="${row}"><div class="card g-${esc(groupOf(l.name))} ro"><span class="name">#${i} ${esc(l.name)}</span>` +
        `<small>${esc(summary)}${l.args.activation ? ` · ${esc(l.args.activation)}` : ''}</small><span class="out">${out ? `(${esc(out.join(' × '))})` : ''}</span></div>` +
        `${bar(i)}</div>${traceRow(i, l.name)}`;
    }).join('');
  }
  /** What the width suggestions are computed from: the inputs the first dense layer receives (its in_features,
   *  filled from the layer above), the classes the model check's last layer gives, the project's dense widths. */
  function widthContext() {
    const first = state.layers.find(l => l.slot === 'dense1') || state.layers.find(l => l.slot === 'out');
    const last = state.check && state.check.layers && state.check.layers.length ? state.check.layers[state.check.layers.length - 1].out : null;
    const widths = (state.models || []).flatMap(m => (((m.architecture_args || {}).layers) || [])
      .map(l => (l && l.Linear ? l.Linear.out_features : null))).filter(v => Number.isInteger(v));
    return { knobs: state.knobs, inFeatures: first && Number.isFinite(first.args.in_features) ? first.args.in_features : null,
      classes: Array.isArray(last) && last.length === 1 ? last[0] : null, projectWidths: [...new Set(widths)].sort((a, b) => a - b) };
  }
  /** The Input: the data entry whose trace length the layers see (a menu of params.yaml's entries, the length beside
   *  each), or, in a project without a data entry yet, the length typed (channels 1). */
  function inputHtml() {
    const lengths = state.lengths || [];
    const shown = (state.input || []).join(' × ');
    if (lengths.length) {
      const tip = 'the data entry the model is built for: its target_range gives the trace length the first layer sees (channels × length)';
      return `<div class="stat"><span title="${esc(tip)}">Input</span><b><select data-inputentry="1" title="${esc(tip)}">` +
        lengths.map(e => opt(e.name, state.dataset || lengths[0].name, `${e.name} · ${e.length == null ? '?' : e.length}`)).join('') +
        `</select></b></div>`;
    }
    const tip = 'no data entry in params.yaml yet: the trace length the first layer sees, typed (Add a Dataset in the panel gives it); 1 channel';
    return `<div class="stat"><span title="${esc(tip)}">Input</span><b><input type="number" data-inputlength="1" min="1" step="1" value="${esc(state.input ? state.input[1] : '')}" ` +
      `placeholder="trace length" title="${esc(tip)}"></b></div>` +
      (state.inputError ? `<small class="knoberr">${esc(state.inputError)}</small>` : !state.input ? `<small class="note">${esc(tip)}${shown ? '' : ''}</small>` : '');
  }
  /** Data trace (a data entry, fixed in the trained view), the trace number, and the weights: untrained or a run. */
  function traceControls() {
    if (!state.observe) {
      return '<button class="observe" data-observe="1" title="show what each layer makes of one attack trace: its data entry, trace number and weights">Observe Trace</button>';
    }
    const r = state.trained ? runOf(state.trained) : null;
    const entries = state.entries || [];
    const data = r ? `<select data-dataset="1" disabled title="${esc(`fixed: the data entry ${r.name} trained on (its snapshot)`)}"><option>${esc(`${r.dataset} · ${r.type}`)}</option></select>`
      : `<select data-dataset="1" title="the data entry whose attack trace goes through the design">` +
        `${entries.length ? entries.map(e => opt(e.name, state.dataset, `${e.name} · ${e.type}`, e.why)).join('') : '<option value="">reading…</option>'}</select>`;
    const max = maxTrace();
    const runs = state.runs || [];
    const why = !state.modelId ? 'Visualise a saved model to see its trained runs' : !runs.length ? `no finished run of ${state.modelId} yet` : '';
    const weights = `<select data-weights="1"${why ? ` disabled title="${esc(why)}"` : ' title="untrained weights for the design, or a finished run of this model with its own layers and weights"'}>` +
      `${opt('', state.trained || '', 'untrained (this design)')}${runs.map(x => opt(x.name, state.trained || '', `${x.name} · ${x.framework || ''}`, x.why)).join('')}</select>`;
    const box = (key, value, what) => `<input type="number" data-${key}="1" min="0"${max ? ` max="${max - 1}"` : ''} step="1" value="${esc(value)}"` +
      ` title="${esc(`${what}: ${max ? `an attack trace, 0 to ${max - 1}` : 'an attack trace of its test.h5'}; Enter draws it`)}">`;
    const mode = `<select data-tracemode="1" title="one trace, or a range of traces (their mean and band)">` +
      `${opt('single', state.traceRange ? 'range' : 'single', 'single trace')}${opt('range', state.traceRange ? 'range' : 'single', 'range of traces')}</select>`;
    const numbers = state.traceRange
      ? `<label class="knob"><span>Traces</span><span class="span2">${box('traceindex', state.traceIndex || 0, 'the first trace')}<span>to</span>` +
        `${box('traceto', state.traceTo == null ? '' : state.traceTo, `the last trace (at most ${RANGE_MAX} in a range)`)}</span>`
      : `<label class="knob"><span>Trace number</span>${box('traceindex', state.traceIndex || 0, 'the trace')}`;
    return `<label class="knob"><span>Data trace</span>${data}</label>` +
      `<label class="knob"><span>Observe</span>${mode}${state.traceRange ? '<small class="faint">Each layer\'s line becomes the average over the range, ' +
        'with a shaded band from the lowest to the highest value at each point.</small>' : ''}</label>` +
      `${numbers}${state.traceError ? `<small class="knoberr">${esc(state.traceError)}</small>` : ''}</label>` +
      `<label class="knob"><span>Weights</span>${weights}</label>` +
      '<button class="observe" data-observe="1" title="hide the trace lines and their settings (nothing is traced while hidden)">Hide Trace</button>';
  }
  function leftHtml() {
    if (state.trained) {
      const r = runOf(state.trained) || {};
      return `<div class="trained"><b>${esc(`Trained: ${r.name}`)}</b><small>${esc(`${r.architecture || ''} · ${r.framework || ''} · ${r.env || ''}`)}</small>` +
        '<button data-design="1" title="back to the design: its layers, untrained weights">Back to the design</button>' +
        `<button data-recovery="1" title="${state.showRecovery ? 'hide the key recovery this run recorded' : 'show the key recovery this run recorded: its CV folds or validation, its test and external tests'}">` +
        `${state.showRecovery ? 'Hide Key Recovery' : 'Show Key Recovery'}</button></div>` + traceControls() + downloadHtml();
    }
    const sk = P.SKELETONS[state.type];
    const custom = ' disabled title="this model matches no skeleton: Rebuild from the skeleton first"';
    const knobs = sk ? sk.knobs.map(k => {
      const err = (state.knobErrors || {})[k.k];
      const shown = (state.knobText || {})[k.k] != null ? state.knobText[k.k] : state.knobs[k.k];
      const steps = P.knobChoices(k, state.knobs, inputArg());
      const hints = steps ? null : P.widthSuggestions(k, widthContext());
      const meaning = P.knobHint(state.type, k.k);          // the hover text: what the setting means
      const menu = items => `<select data-knob="${esc(k.k)}"${state.custom ? custom : ` title="${esc(`${k.label}: ${meaning}`)}"`}>` +
        `${state.custom ? '<option>custom</option>' : items}</select>`;
      const control = k.kind === 'pattern' ? patternHtml(k, meaning)
        : k.kind === 'enum'
        ? menu((k.options === 'activations' ? activations() : k.options === 'activationsOrSame' ? ['same', ...activations()] : k.options)
          .map(o => opt(o, state.knobs[k.k], o, forms().layers[o] ? layerBlocked(o) : '', P.optionHint(k.k, o))).join(''))   // a layer core.model cannot build: disabled, with why
        : steps ? menu(steps.map(c => opt(c.v, state.knobs[k.k] == null ? '' : state.knobs[k.k], c.label, '', P.optionHint(k.k, c.v) || meaning)).join(''))   // '': none
        : `<input type="${k.listable ? 'text' : 'number'}" data-knob="${esc(k.k)}" value="${esc(state.custom ? '' : shown)}"${k.min != null && !k.listable ? ` min="${k.min}"` : ''}` +
          `${k.max != null ? ` max="${k.max}"` : ''} step="${k.kind === 'int' ? 1 : 'any'}"${hints ? ` list="hint-${esc(k.k)}"` : ''}` +
          `${state.custom ? custom : ` title="${esc(`${k.label}: ${meaning}. Type a ${k.kind === 'int' ? 'whole number' : 'number'}${k.listable ? ', or one per block (2,2,3,3,3)' : ''}${hints ? ' or pick a suggestion' : ''}, Enter applies it`)}"`}>` +
          (hints ? `<datalist id="hint-${esc(k.k)}">${hints.map(h => `<option value="${h.v}" label="${esc(h.label)}">`).join('')}</datalist>` : '');
      return `<label class="knob"><span title="${esc(meaning)}">${esc(k.label)}</span>${control}${err ? `<small class="knoberr" title="${esc(k.label)}">${esc(err)}</small>` : ''}</label>`;
    }).join('') : '';
    const rebuild = state.custom ? '<button data-rebuildskeleton="1" title="replace the layers with the skeleton of these settings (asks first)">Rebuild from the skeleton</button>' : '';
    const perLayer = activationsHtml() + poolingsHtml() + dropoutsHtml();
    const flat = state.check && state.check.layers ? (state.check.layers.find(x => x.type === 'Flatten') || {}).out : null;
    const id = P.idState(state.id, state.ids, M.askError);
    const sameAs = P.sameAs(state.layers, state.models);
    const status = state.checking ? 'checking…' : !state.input ? '<span class="err">input length needed</span>' : !state.check ? '' : state.check.error
      ? `<span class="err">${state.check.at != null ? `layer #${esc(state.check.at)}: ` : ''}${esc(state.check.error)}</span>` : '<span class="ok">builds ✓</span>';
    return pendingHtml() +
      `<label class="knob"><span>Type</span><select data-kind="1" title="the kind of model: its skeleton settings follow">${KINDS.map(k => opt(k, state.type, k, kindBlocked(k))).join('')}</select></label>` +
      knobs + rebuild + perLayer +
      '<div class="stats">' +
      inputHtml() +
      `<div class="stat"><span>Flatten</span><b>${flat ? esc(Number(flat[0]).toLocaleString('en')) : '—'}</b></div>` +
      `<div class="stat"><span>Weights</span><b>${state.check && state.check.params != null ? esc(Number(state.check.params).toLocaleString('en')) : '—'}</b></div>` +
      `<div class="stat"><span>Same as</span><b>${esc(sameAs || 'a new model')}</b></div>` +
      `<div class="stat"><span>Model check</span><b>${status}</b></div></div>` + traceControls() +
      (visualising() ? '' : `<label class="knob"><span>model_id</span><input data-id="1" value="${esc(state.id || '')}" placeholder="${esc(M.ASKS.model.hint)}"></label>` +
        `<div class="idstate ${id.ok ? 'ok' : 'err'}">${esc(id.ok ? `unique ✓${id.note ? ` (${id.note})` : ''}` : id.error)}</div>`) +
      actionsHtml();
  }
  // ── Download: the model's diagram drawn and saved by model_diagram.py as graph_manager saves figures ──────
  // Its form shows a choice's settings only once it is chosen (a format's under its tick, the Scaled Sparkline's under
  // that style, a text rule's under its target). The formats, their settings and text_rules' fields are graph_manager's
  // (choices.py graph forms); the styles and targets are model_diagram.py's.
  const STYLE_LABEL = { full: 'Full Version', cards: 'Layer cards', sparkline: 'Scaled Sparkline' };
  const GF = () => state.graph || {};
  const FIRST = ['png', 'pdf', 'svg'];                    // shown first, in this order; graph_manager's others after them
  function exportForm() {
    const ex = GF().export || { formats: FIRST, common: [{ k: 'margin', t: 'num', d: 0.05 }], settings: { png: [{ k: 'dpi', t: 'int', d: 300 }] } };
    return { ...ex, formats: [...FIRST.filter(f => (ex.formats || []).includes(f)), ...(ex.formats || []).filter(f => !FIRST.includes(f))] };
  }
  const diagramForm = () => GF().diagram || { styles: ['full', 'cards', 'sparkline'], orientations: ['portrait', 'landscape'],
    legend: ['top-right', 'bottom-right', 'top-left', 'none'], themes: ['traceflow'], theme: 'traceflow' };
  const ruleForm = () => GF().textRule || { targets: { diagram: [] }, fields: [], textTargets: { diagram: [] } };
  const RULE_START = () => ({ enabled: false, font: { size: 8 } });   // a text rule as it starts: off, font size 8
  /** The Download's settings as they start: graph_manager's export defaults, the diagram's, every text rule off. */
  function dlDefaults() {
    const ex = exportForm();
    return { style: 'full', theme: diagramForm().theme || 'traceflow', arrows: false, traces: true,
      formats: Object.fromEntries((ex.formats || []).map(f => [f, FIRST.includes(f)])),
      settings: Object.fromEntries(Object.entries(ex.settings || {}).map(([f, list]) => [f, Object.fromEntries(list.map(x => [x.k, x.d]))])),
      margin: ((ex.common || []).find(c => c.k === 'margin') || { d: 0.05 }).d,
      sparkline: { arguments: true, orientation: 'portrait', ticks: true, legend: 'top-right' },
      rules: Object.fromEntries((((ruleForm().targets || {}).diagram) || []).map(t => [t, RULE_START()])) };
  }
  const copyOf = v => JSON.parse(JSON.stringify(v));
  /** The Download's settings, kept in the tab state: `saved` (what Download draws with; OK keeps the draft there) and
   *  `draft` (what the form shows and edits change), and which folds are open. */
  function dlState() {
    const d = state.dl;
    if (!d || !d.saved || !d.saved.settings) state.dl = { saved: dlDefaults(), draft: dlDefaults(), open: (d && d.open) || {} };
    return state.dl;
  }
  const dl = () => dlState().draft;
  const dlSaved = () => dlState().saved;
  /** The settings that differ between two of them, as paths (rules.title.font.size, formats.png, …). */
  function dlDiff(a, b, at = '') {
    const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
    return [...keys].flatMap(k => {
      const x = (a || {})[k];
      const y = (b || {})[k];
      const p = at ? `${at}.${k}` : k;
      return x && y && typeof x === 'object' && typeof y === 'object' ? dlDiff(x, y, p) : JSON.stringify(x) === JSON.stringify(y) ? [] : [p];
    });
  }
  const dlChanges = () => dlDiff(dlState().draft, dlState().saved);
  /** The changes recognised in the Download form: OK keeps them (Download draws with them); Cancel undoes them and
   *  folds the text_rules away; Reset undoes them, the form staying as it is (as Visualise's bar: Cancel closes). */
  function dlPendingHtml() {
    const n = dlChanges().length;
    if (!n) return '';
    return `<div class="pending dlnote"><span title="${esc(dlChanges().join(', '))}"><b>Download settings</b> · ${n} change${n === 1 ? '' : 's'} recognised</span></div>`;
  }
  /** Download, and beside it (while changes are recognised) OK, Cancel and Reset. */
  function dlButtonsHtml(why) {
    const download = `<button data-download="1"${why ? ` disabled title="${esc(why)}"` : ' title="save the diagram into paths.model_preview, and into model_out of each trained run with these layers"'}>Download</button>`;
    if (!dlChanges().length) return `<div class="dlbuttons">${download.replace('<button ', '<button class="wide" ')}</div>`;
    return `<div class="dlbuttons">${download}` +
      '<button data-dlok="1" title="keep these settings: Download draws with them">OK</button>' +
      '<button data-dlcancel="1" title="undo the changes and fold the text_rules away">Cancel</button>' +
      '<button data-dlreset="1" title="undo the changes: the settings as they were before; the form stays open">Reset</button></div>';
  }
  const dlFormats = (x = dl()) => exportForm().formats.filter(f => x.formats[f]);
  const tracesShown = () => !!(state.observe && state.trace && state.trace.layers && state.trace.layers.length);
  /** The picture's title: the trained run, the visualised model, the model_id typed, or a new model. */
  const diagramTitle = () => state.trained || state.modelId || String(state.id || '').trim() || 'new model';
  /** The Observe Trace settings drawn, as a file name part: trace, the data entry, untrained (a design's weights; a
   *  trained run's name is in the name already), t<n> or t<first>-<last> (a range). */
  function traceTag() {
    const src = (state.trace || {}).source || {};
    const many = src.to != null && src.to !== src.trace;
    return ['trace', src.dataset, state.trained ? '' : 'untrained', many ? `t${src.trace}-${src.to}` : `t${src.trace}`].filter(Boolean).join('_');
  }
  /** What model_diagram.py draws: the cards' layers, shapes, groups (and trace lines), the facts, the options. */
  function downloadRequest() {
    const r = state.trained ? runOf(state.trained) : null;
    const x = dlSaved();                                 // what OK kept
    const flat = !r && state.check && state.check.layers ? (state.check.layers.find(c => c.type === 'Flatten') || {}).out : null;
    const weights = !r && state.check && state.check.params != null ? Number(state.check.params).toLocaleString('en') : '';
    const traces = tracesShown() && x.traces && x.style !== 'sparkline';
    // the text rules enabled, as graph_manager's fields (enabled is the form's own switch)
    const rules = Object.fromEntries(Object.entries(x.rules || {}).filter(([, r]) => r && r.enabled)
      .map(([t, r]) => { const { enabled, ...rest } = r; void enabled; return [t, rest]; }));
    const line = t => (t ? { values: t.values, band: t.band || null, lo: t.lo, hi: t.hi } : null);
    const tag = traces ? traceTag() : '';
    // the file: the model, the trained run chosen, the trace settings drawn (model_diagram.py: <name>.<format>)
    const name = [state.modelId || String(state.id || '').trim() || 'new model', state.trained || '', tag].filter(Boolean).join('__');
    return { type: 'download', title: diagramTitle(), name, tag, run: state.trained || '', model_id: state.modelId || '', env: r ? r.env || '' : '',
      subtitle: [state.project, r ? `trained ${state.modelId}` : '', r ? r.architecture : state.type, r ? r.framework : state.framework].filter(Boolean).join(' · '),
      facts: [['Input', (state.input || []).join(' × ')], ['Flatten', flat ? Number(flat[0]).toLocaleString('en') : ''], ['Weights', weights]],
      max: scaleMax(),
      layers: shownLayers().map((l, i) => ({ name: l.name, group: groupOf(l.name), args: l.args, out: shapeOf(i), trace: traces ? line(traceLayer(i)) : null })),
      trace: traces ? { caption: traceWhere(), input: line(state.trace.input) } : null,
      style: x.style, theme: x.theme || diagramForm().theme, arrows: x.arrows, traces, sparkline: { ...x.sparkline }, text_rules: copyOf(rules),
      // a format's own settings only when it has any (graph_manager refuses an export.pdf block: a PDF has none)
      export: { formats: dlFormats(x), margin: x.margin,
        ...Object.fromEntries(dlFormats(x).filter(f => ((exportForm().settings || {})[f] || []).length).map(f => [f, { ...x.settings[f] }])) } };
  }
  /** One setting as its form type says: a tick box (bool), a number (int, num), a menu (enum, color). */
  function dlField(key, f, value, label = f.k) {
    const tip = `${label}${f.d != null ? ` (graph_manager's default ${f.d})` : ''}`;
    if (f.t === 'bool') return `<label title="${esc(tip)}"><input type="checkbox" data-dl="${esc(key)}"${value ? ' checked' : ''} title="${esc(tip)}"> ${esc(label)}</label>`;
    if (f.o) {
      return `<label title="${esc(tip)}">${esc(label)} <select data-dl="${esc(key)}" title="${esc(tip)}">` +
        `${f.t === 'rule' ? opt('', value == null ? '' : value, '(default)') : ''}${f.o.map(o => opt(o, value == null ? '' : value, o)).join('')}</select></label>`;
    }
    return `<label title="${esc(tip)}">${esc(label)} <input type="${f.t === 'text' ? 'text' : 'number'}" data-dl="${esc(key)}"` +
      `${f.t === 'int' ? ' step="1"' : f.t === 'num' ? ' step="0.01"' : ''} value="${esc(value == null ? '' : value)}"${f.t === 'text' ? ' class="wide"' : ''} title="${esc(tip)}"></label>`;
  }
  /** A fold of the form, open or closed by its summary (the page keeps which). */
  const fold = (key, head, body) => `<div class="dlfold"><button class="dlhead" data-dlfold="${esc(key)}" title="${dlState().open[key] ? 'fold it away' : 'show it'}">` +
    `${dlState().open[key] ? '▾' : '▸'} ${head}</button>${dlState().open[key] ? `<div class="dlbody">${body()}</div>` : ''}</div>`;
  /** text_rules: a fold per text of the diagram: enabled (true: the rule applies; false: the diagram's own look), then
   *  graph_manager's fields; every rule starts off, at font size 8. */
  function rulesHtml() {
    const R = ruleForm();
    const x = dl();
    const targets = (R.targets || {}).diagram || [];
    const on = targets.filter(t => (x.rules[t] || {}).enabled).length;
    return fold('rules', `text_rules <small>${on}/${targets.length} enabled</small>`, () => targets.map(t => fold(`rule:${t}`,
      `${esc(t)} <small>${(x.rules[t] || {}).enabled ? 'enabled' : "off: the diagram's own"}</small>`, () =>
      `<div class="dlrow">${dlField(`rules.${t}.enabled`, { k: 'enabled', t: 'enum', o: ['true', 'false'] }, String(!!(x.rules[t] || {}).enabled), 'enabled')}</div>` +
      (R.fields || [])
        .filter(f => f.k !== 'text' || ((R.textTargets || {}).diagram || []).includes(t))
        .map(f => {
          const v = f.k.split('.').reduce((o, k) => (o == null ? undefined : o[k]), x.rules[t]);
          return `<div class="dlrow">${dlField(`rules.${t}.${f.k}`, f.t === 'bool' ? { ...f, t: 'rule', o: ['true', 'false'] } : f.o ? { ...f, t: 'rule' } : f,
            f.t === 'bool' ? (v == null ? '' : String(v)) : v, f.k)}</div>`;
        }).join('') + `<button data-dlclear="${esc(t)}" title="${esc(`${t}: off, font size 8, nothing else set`)}">Clear</button>`)).join(''));
  }
  /** The Download form: the style, the theme, the style's own choices, arrows, trace lines, the formats with their settings,
   *  the margin, text_rules, then Download. */
  function downloadHtml() {
    const x = dl();
    const D = diagramForm();
    const ex = exportForm();
    const styleTip = 'the window and the diagram: Full Version: the layer cards and their bars; Layer cards: the cards alone; Scaled Sparkline: the bars alone (the layers are edited on the cards)';
    const style = `<div class="dlrow"><label title="${styleTip}">style ` +
      `<select data-dl="style" title="${styleTip}">${D.styles.map(v => opt(v, x.style, STYLE_LABEL[v] || v)).join('')}</select></label></div>`;
    const theme = `<div class="dlrow"><label title="the colours: a theme of services/visualization/utils/fig_theme.py (the paper's figures use the same)">theme ` +
      `<select data-dl="theme" title="the colours: a theme of services/visualization/utils/fig_theme.py (the paper's figures use the same)">` +
      `${(D.themes || []).map(v => opt(v, x.theme || D.theme, v)).join('')}</select></label></div>`;
    const spark = x.style === 'sparkline' ? '<div class="dlsub">' +
      `<div class="dlrow">${dlField('sparkline.arguments', { k: 'arguments', t: 'bool' }, x.sparkline.arguments, 'arguments')}` +
      `${dlField('sparkline.ticks', { k: 'tick labels', t: 'bool' }, x.sparkline.ticks, 'tick labels')}</div>` +
      `<div class="dlrow">${dlField('sparkline.orientation', { k: 'orientation', t: 'enum', o: D.orientations }, x.sparkline.orientation, 'orientation')}</div>` +
      `<div class="dlrow">${dlField('sparkline.legend', { k: 'layer legend', t: 'enum', o: D.legend }, x.sparkline.legend, 'layer legend')}</div></div>` : '';
    const flow = `<div class="dlrow">${dlField('arrows', { k: 'data-flow arrows', t: 'bool' }, x.arrows, 'data-flow arrows')}` +
      `${tracesShown() && x.style !== 'sparkline' ? dlField('traces', { k: 'trace lines', t: 'bool' }, x.traces, 'trace lines') : ''}</div>`;
    const formats = (ex.formats || []).map(f => `<div class="dlrow">${dlField(`formats.${f}`, { k: f, t: 'bool' }, x.formats[f], f.toUpperCase())}` +
      `${x.formats[f] ? ((ex.settings || {})[f] || []).map(s => dlField(`settings.${f}.${s.k}`, s, (x.settings[f] || {})[s.k], s.k)).join('') : ''}</div>`).join('');
    const margin = `<div class="dlrow">${dlField('margin', ((ex.common || []).find(c => c.k === 'margin')) || { k: 'margin', t: 'num', d: 0.05 }, x.margin, 'margin')}</div>`;
    const why = dlChanges().length ? 'OK or Cancel the changes to the Download settings first' : !dlFormats(dlSaved()).length ? 'tick a format' : '';
    return `<div class="download"><span class="lbl">Download</span>${style}${theme}${spark}${flow}${formats}${margin}${rulesHtml()}${dlPendingHtml()}` +
      `${dlButtonsHtml(why)}</div>`;
  }
  /** A Download setting changed in the draft (a path: style, formats.png, settings.png.dpi, rules.title.font.size, …);
   *  a text rule's field left empty is no longer set. The bar over Download counts it until OK, Cancel or Reset. */
  function setDl(key, t) {
    const x = dl();
    if (key === 'theme') dlState().themePreview = true;     // from now on the cards show the theme chosen
    const path = key.split('.');
    let value = t.type === 'checkbox' ? !!t.checked : t.value;
    if (t.type === 'number') value = t.value === '' ? '' : Number(t.value);
    if (path[0] === 'rules' && (value === 'true' || value === 'false')) value = value === 'true';
    let o = x;
    path.slice(0, -1).forEach(k => { o = o[k] = o[k] && typeof o[k] === 'object' ? o[k] : {}; });
    if (path[0] === 'rules' && value === '') delete o[path[path.length - 1]];
    else o[path[path.length - 1]] = value;
    if (path[0] === 'rules') {
      const rule = x.rules[path[1]];
      if (rule && rule.font && !Object.keys(rule.font).length) delete rule.font;
    }
    keepDesign();
    render();
  }
  // ── Activation per layer: every layer that can carry one, its own menu (an activation runs right after the
  // layer it is on: on a Conv1d before the normalization below it, on a BatchNorm1d after it) ───────────────
  const ACT_LAYERS = new Set(['Conv1d', 'ConvTranspose1d', 'BatchNorm1d', 'InstanceNorm1d', 'LayerNorm', 'GroupNorm', 'Linear']);
  /** A layer's mark on its row and card: edited. Visualise: only where it differs from params.yaml (a Configuration
   *  Change; with `key`, that argument only); what the file holds is not marked. Plan a Model: a hand edit. */
  function markOf(l, own, key) {
    const none = { cls: '', tag: '', tip: '' };
    if (!l) return none;
    if (visualising() && Array.isArray(state.original)) {
      const base = l.from == null ? null : state.original[l.from];
      const changed = !base || Object.keys(base)[0] !== l.name ||
        (key ? !P.same((base[l.name] || {})[key], l.args[key]) : !P.same({ [l.name]: l.args }, base));
      return changed ? { cls: ' chg', tag: 'edited', tip: ' (edited: not in params.yaml yet)' } : none;
    }
    return own ? { cls: ' own', tag: 'edited', tip: ' (its own, not the settings\')' } : none;
  }
  function activationsHtml() {
    const rows = state.layers.map((l, i) => ({ l, i })).filter(({ l }) => ACT_LAYERS.has(l.name) || l.args.activation);
    if (!rows.length) return '';
    const tally = {};
    rows.forEach(({ l }) => { const a = l.args.activation || 'none'; tally[a] = (tally[a] || 0) + 1; });
    const summary = Object.entries(tally).map(([a, n]) => `${a} ×${n}`).join(', ');
    const open = !!state.actOpen;
    const head = `<button class="acthead" data-actfold="1" title="${open ? 'fold it away' : 'each layer\'s own activation: which layer it runs after, and which one'}">` +
      `${open ? '▾' : '▸'} Activation per layer <small>${esc(summary)}</small></button>`;
    if (!open) return `<div class="actlayers">${head}</div>`;
    const carrying = rows.filter(({ l }) => l.args.activation);
    const shared = carrying.length && carrying.every(({ l }) => l.args.activation === carrying[0].l.args.activation) ? carrying[0].l.args.activation : '';
    const all = `<label class="actrow actall" title="every layer that has an activation now, set at once (each then its own)"><span>all that have one</span>` +
      `<select data-actall="1" title="every layer that has an activation now, set at once">${opt('', '', shared ? shared : '(mixed)')}` +
      `${activations().filter(a => a !== shared).map(a => opt(a, shared, a)).join('')}</select></label>`;
    const list = all + rows.map(({ l, i }) => {
      const cur = l.args.activation || '';
      const mk = markOf(l, (l.handArgs || []).includes('activation'), 'activation');
      const tip = `#${i} ${l.name}: the activation right after it${mk.tip}`;
      return `<label class="actrow${mk.cls}" title="${esc(tip)}"><span>#${i} ${esc(l.name)}${mk.tag ? ` <small>${mk.tag}</small>` : ''}</span>` +
        `<select data-layeract="${i}" title="${esc(tip)}">${opt('', cur, 'none')}${activations().map(a => opt(a, cur, a)).join('')}</select></label>`;
    }).join('');
    return `<div class="actlayers">${head}${list}</div>`;
  }
  // ── Block Pattern: one block's layer kinds as chips: ◀ ▶ move one, × takes it away, + adds a kind at the end (any
  // kind, any number, any order). Each change is the setting's new value (it rebuilds, asking after hand edits). ──
  const CHIP_LABEL = { conv: () => 'Conv1d', norm: () => (state.knobs.norm && state.knobs.norm !== 'none' ? state.knobs.norm : 'BatchNorm1d'), pool: () => 'Pooling' };
  function patternHtml(k, meaning) {
    const tokens = P.patternOf(state.knobs);
    if (state.custom) return '<div class="chips"><small class="faint">custom: Rebuild from the skeleton first</small></div>';
    const closing = i => tokens[i] === 'pool' && i === tokens.length - 1;
    const chips = tokens.map((t, i) => `<span class="chip chip-${t}" title="${esc(`${CHIP_LABEL[t]()}: ${closing(i) ? 'ends the block, once' : 'repeats Repeats per block times'}`)}">` +
      `${i > 0 ? `<button data-chip="left" data-at="${i}" title="move it before the one on its left">◀</button>` : ''}<span>${esc(CHIP_LABEL[t]())}</span>` +
      `${i < tokens.length - 1 ? `<button data-chip="right" data-at="${i}" title="move it after the one on its right">▶</button>` : ''}` +
      `<button class="x" data-chip="remove" data-at="${i}" title="take it out of the block">×</button></span>`).join('<span class="arrow">→</span>');
    const add = `<select data-chipadd="1" title="add a layer kind at the end of the block (◀ moves it)"><option value="">+</option>` +
      `<option value="conv">Conv1d</option><option value="norm">${esc(CHIP_LABEL.norm())}</option><option value="pool">Pooling</option></select>`;
    return `<div class="chips" title="${esc(`${k.label}: ${meaning}`)}">${chips || '<small class="faint">an empty block</small>'}${add}</div>`;
  }
  /** A chip moved, taken out or added: the Block Pattern's new value, through the setting (parsed, then applied). */
  function editPattern(op, at, kind) {
    const tokens = [...P.patternOf(state.knobs)];
    if (op === 'left' && at > 0) [tokens[at - 1], tokens[at]] = [tokens[at], tokens[at - 1]];
    else if (op === 'right' && at < tokens.length - 1) [tokens[at + 1], tokens[at]] = [tokens[at], tokens[at + 1]];
    else if (op === 'remove') tokens.splice(at, 1);
    else if (op === 'add' && kind) tokens.push(kind);
    act({ type: 'knob', k: 'pattern', value: tokens.join(' ') });
  }

  // ── Pooling per block: each block's pooling, its type and size (none takes it away, a type puts it back at the
  // block's end); a block's pooling is the layer of its slot, or one added after the block's layers ────────────
  const POOL_TYPES = ['none', 'max', 'avg'];
  function blocksOf() {
    const out = [];
    state.layers.forEach((l, i) => {
      const m = /^b(\d+)\./.exec(l.slot || l.anchor || '');
      if (!m) return;
      const b = Number(m[1]);
      let block = out.find(x => x.b === b);
      if (!block) { block = { b, last: i, pool: null }; out.push(block); }
      block.last = i;
      if (l.name === 'Pooling') block.pool = i;
    });
    return out;
  }
  function poolingsHtml() {
    if (state.type !== 'CNN' && state.type !== 'HYBRID') return '';
    const blocks = blocksOf();
    if (!blocks.length) return '';
    const tally = {};
    blocks.forEach(x => { const t = x.pool == null ? 'none' : String(state.layers[x.pool].args.type || 'max'); tally[t] = (tally[t] || 0) + 1; });
    const open = !!state.poolOpen;
    const head = `<button class="acthead" data-poolfold="1" title="${open ? 'fold it away' : 'each block\'s pooling: its type and size'}">` +
      `${open ? '▾' : '▸'} Pooling per block <small>${esc(Object.entries(tally).map(([t, n]) => `${t} ×${n}`).join(', '))}</small></button>`;
    if (!open) return `<div class="actlayers">${head}</div>`;
    const list = blocks.map(x => {
      const l = x.pool == null ? null : state.layers[x.pool];
      const type = l ? String(l.args.type || 'max') : 'none';
      const types = POOL_TYPES.includes(type) ? POOL_TYPES : [...POOL_TYPES, type];
      const mk = markOf(l, l && ((l.handArgs || []).length || (l.added && l.from == null)));
      const tip = `block ${x.b}'s pooling: ${l ? `#${x.pool} Pooling, ` : ''}its type (none: the block keeps its length) and its window (the stride too)${mk.tip}`;
      return `<div class="poolrow${mk.cls}" title="${esc(tip)}"><span>block ${x.b}${l ? ` <small>#${x.pool}${mk.tag ? ` ${mk.tag}` : ''}</small>` : ''}</span>` +
        `<select data-pooltype="${x.b}" title="${esc(tip)}">${types.map(t => opt(t, type, t, '', P.optionHint('pool', t))).join('')}</select>` +
        `<input type="number" min="1" step="1" data-poolsize="${x.b}" value="${esc(l ? (l.args.kernel_size == null ? 2 : l.args.kernel_size) : '')}"${l ? '' : ' disabled'} title="${esc(tip)}"></div>`;
    }).join('');
    return `<div class="actlayers">${head}${list}</div>`;
  }
  /** Pooling per block: block b's pooling set to a type (put back after the block when it had none), taken away
   *  (none), or given a window (its kernel_size and stride). */
  function setPooling(b, field, value) {
    const x = blocksOf().find(y => y.b === b);
    if (!x) return;
    if (field === 'type' && value === 'none') { if (x.pool != null) removeAt(x.pool); return; }
    if (x.pool == null) {
      if (field !== 'type') return;
      state.layers = P.insert(state.layers, x.last + 1, [{ name: 'Pooling', args: { dim: 1, type: value, kernel_size: 2, stride: 2 } }]);
      reindex(i => (i >= x.last + 1 ? i + 1 : i));
      return;
    }
    if (field === 'type') setArg(x.pool, 'type', value);
    else if (/^\d+$/.test(String(value)) && Number(value) >= 1) { setArg(x.pool, 'kernel_size', value); setArg(x.pool, 'stride', value); }
  }

  // ── Dropout per layer: every layer a Dropout can follow, its own menu; a value puts a Dropout layer right after it
  // (or sets its p), none takes it away. A dropout is a layer, not a setting: it stays when the settings change. ──
  const DROP_AFTER = new Set(['Conv1d', 'ConvTranspose1d', 'BatchNorm1d', 'InstanceNorm1d', 'LayerNorm', 'GroupNorm', 'Pooling', 'Flatten', 'Linear']);
  const DROP_PS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5];
  const isDropout = l => !!l && /^(Dropout|AlphaDropout)$/.test(l.name);
  function dropoutsHtml() {
    const last = state.layers.length - 1;                 // the output layer: nothing after it to drop
    const rows = state.layers.map((l, i) => ({ l, i })).filter(({ l, i }) => DROP_AFTER.has(l.name) && i < last);
    if (!rows.length) return '';
    const n = rows.filter(({ i }) => isDropout(state.layers[i + 1])).length;
    const open = !!state.dropOpen;
    const head = `<button class="acthead" data-dropfold="1" title="${open ? 'fold it away' : 'a Dropout after any layer: the share of its values zeroed while training (none: no Dropout)'}">` +
      `${open ? '▾' : '▸'} Dropout per layer <small>${n ? `${n} layer${n === 1 ? '' : 's'}` : 'none'}</small></button>`;
    if (!open) return `<div class="actlayers">${head}</div>`;
    const list = rows.map(({ l, i }) => {
      const next = state.layers[i + 1];
      const cur = isDropout(next) ? String(next.args.p == null ? 0.5 : next.args.p) : '';
      const ps = [...new Set([...DROP_PS.map(String), ...(cur ? [cur] : [])])].sort((a, b) => Number(a) - Number(b));
      const mk = markOf(cur ? next : null, true);
      const tip = `a Dropout right after #${i} ${l.name}: the share of its values zeroed while training (none: no Dropout)${mk.tip}`;
      return `<label class="actrow${mk.cls}" title="${esc(tip)}"><span>#${i} ${esc(l.name)}${visualising() && mk.tag ? ` <small>${mk.tag}</small>` : ''}</span>` +
        `<select data-layerdrop="${i}" title="${esc(tip)}">${opt('', cur, 'none')}${ps.map(p => opt(p, cur, p)).join('')}</select></label>`;
    }).join('');
    return `<div class="actlayers">${head}${list}</div>`;
  }
  /** Dropout per layer: after layer i, a Dropout of p (added, or its p set), or none (the Dropout taken away). */
  function setDropout(i, value) {
    const next = state.layers[i + 1];
    if (value === '') { if (isDropout(next)) removeAt(i + 1); return; }
    if (isDropout(next)) { setArg(i + 1, 'p', value); return; }
    state.layers = P.insert(state.layers, i + 1, [{ name: 'Dropout', args: { p: Number(value) } }]);
    reindex(x => (x >= i + 1 ? x + 1 : x));
  }
  /** Under the model_id: Add the Model (its tooltip says why it waits), Preview Code / Encapsulate, Reset, Cancel. */
  function actionsHtml() {
    const why = addBlocked();
    const preview = `<button data-preview="1" title="${esc(state.previewOpen ? 'fold the YAML away' : visualising()
      ? `show ${state.modelId}'s entry in params.yaml as OK would write it (edited lines marked)` : 'show the YAML Add the Model would write')}">${state.previewOpen ? 'Encapsulate' : 'Preview Code'}</button>`;
    if (visualising()) {                                 // Visualise: its changes go through the Configuration Change bar
      return `<div class="actions">${preview}${pendingHtml() ? '' : '<button data-cancel="1" title="close the window; nothing is written">Cancel</button>'}</div>` + downloadHtml();
    }
    return `<div class="actions"><button class="primary" data-add="1"${why ? ` disabled title="${esc(why)}"`
      : ' title="add this design to params.yaml as a new model under the model_id"'}>Add the Model</button>${preview}` +
      `<button data-reset="1" title="back to the clean skeleton">Reset</button>` +
      '<button data-cancel="1" title="close the planner; nothing is written">Cancel</button></div>' + downloadHtml();
  }
  /** The model_id typed: its state and Add the Model, without redrawing the page (the box keeps its focus). */
  /** The tab state VS Code keeps over a window reload: where the page belongs and its design. */
  const keepDesign = () => keep({ project: state.project, file: state.file, modelId: state.modelId, design: {
    type: state.type, knobs: state.knobs, custom: state.custom, layers: state.layers, removed: state.removed, original: state.original,
    modelId: state.modelId, start: state.start, id: state.id, open: state.open, previewOpen: state.previewOpen,
    dataset: state.dataset, traceIndex: state.traceIndex, observe: !!state.observe, traceRange: !!state.traceRange, traceTo: state.traceTo, dl: state.dl,
    input: state.input || null, leftWidth: state.leftWidth || null } });
  function idChanged() {
    keepDesign();
    if (typeof document.querySelector !== 'function') return render();
    const id = P.idState(state.id, state.ids, M.askError);
    const box = document.querySelector('.idstate');
    const add = document.querySelector('[data-add]');
    const why = addBlocked();
    if (box) { box.textContent = id.ok ? `unique ✓${id.note ? ` (${id.note})` : ''}` : id.error; box.className = `idstate ${id.ok ? 'ok' : 'err'}`; }
    if (add) { add.disabled = !!why; add.title = why || 'add this design to params.yaml as a new model under the model_id'; }
    return undefined;
  }
  let stale = false;                                     // a redraw waits while the model_id box has the focus
  function render() {
    const app = document.getElementById('app');
    if (!state.ready) { app.innerHTML = '<p class="note">Reading the project…</p>'; return; }
    keepDesign();
    const focus = document.activeElement;
    if (focus && focus.dataset && focus.dataset.id) { stale = true; return idChanged(); }
    stale = false;
    const ask = state.pending ? `<div class="ask"><p>${state.custom ? 'This model matches no skeleton.' : 'The layers have hand edits.'} ` +
      `Change ${esc(state.pending.k || 'the skeleton')}?</p><button data-resolve="keep" title="apply the setting and keep your hand edits where they still fit">Keep my changes where they fit</button>` +
      '<button data-resolve="rebuild" title="apply the setting and drop your hand edits">Rebuild from the settings</button>' +
      '<button data-resolve="cancel" title="leave the setting as it was">Cancel</button></div>' : '';
    const dropped = state.dropped && state.dropped.length ? `<p class="note">Not kept: ${esc(state.dropped.join(', '))}</p>` : '';
    const marked = new Set(state.previewMarked || []);
    const preview = state.previewOpen && state.preview ? `<pre class="preview">${state.preview.map((l, k) => (marked.has(k)
      ? `<span class="mk" title="edited: not in params.yaml yet">${esc(l)}</span>` : esc(l))).join('\n')}</pre>` : '';
    const message = state.message ? `<p class="${/refused|changed/.test(state.message) ? 'err' : 'note'}">${esc(state.message)}</p>` : '';
    app.innerHTML = `<h2>Plan a Model <small>${esc(state.project || '')}${state.modelId ? ` · ${esc(state.modelId)}` : ''}</small></h2>${ask}${dropped}${message}` +
      `<div class="planner"><div class="left">${leftHtml()}</div>` +
      '<div class="splitter" data-splitter="1" role="separator" aria-orientation="vertical" tabindex="0" title="drag to widen or narrow the settings; ← → when focused; double-click: back to 240 px"></div>' +
      `<div class="stack">${cardsHtml()}</div></div>${preview}`;
    themePreview();
    leftWidth();
  }
  // ── the splitter: the settings column's width, dragged (a CSS variable of .planner, set from script: the page's
  // CSP allows no inline style); kept in the tab state ────────────────────────────────────────────────────────
  const LEFT_W = 240;
  const clampLeft = w => Math.round(Math.max(180, Math.min(w, Math.max(260, ((typeof window.innerWidth === 'number' ? window.innerWidth : 1200) * 0.6)))));
  function leftWidth(w = state.leftWidth) {
    const el = typeof document.querySelector === 'function' ? document.querySelector('.planner') : null;
    if (el && el.style) el.style.setProperty('--leftw', `${w || LEFT_W}px`);
  }
  let drag = null;                                      // {x, w}: where the drag started
  function setLeft(w, keepIt = true) {
    state.leftWidth = clampLeft(w);
    leftWidth();
    if (keepIt) keepDesign();
  }
  /** The Download theme, once chosen there, on the cards: the theme's page, ink, muted and each layer group's colour as
   *  CSS variables of .stack (set from script: the page's CSP allows no inline style), as the diagram will draw them. */
  function themePreview() {
    const stack = typeof document.querySelector === 'function' ? document.querySelector('.stack') : null;
    if (!stack || !stack.style) return;
    const chosen = state.dl && state.dl.themePreview ? dl().theme : '';
    const pal = chosen ? ((diagramForm().palettes || {})[chosen] || null) : null;
    const vars = pal ? { '--vscode-editor-background': pal.page, '--vscode-foreground': pal.ink, '--vscode-descriptionForeground': pal.muted,
      '--vscode-widget-border': pal.muted, ...Object.fromEntries(Object.keys(pal).filter(k => !['page', 'ink', 'muted'].includes(k)).map(k => [`--tf-${k}`, pal[k]])) } : {};
    for (const k of [...stack.style].filter(k => k.startsWith('--'))) stack.style.removeProperty(k);
    Object.entries(vars).forEach(([k, v]) => stack.style.setProperty(k, v));
    if (stack.classList) stack.classList.toggle('themed', !!pal);
  }

  // ── the host and the DOM ───────────────────────────────────────────────────────────────────────────
  window.addEventListener('message', ev => {
    const m = ev.data || {};
    if (m.type === 'init') {
      Object.assign(state, { graph: m.graph || null, ready: true, project: m.project, file: m.file || '', framework: m.framework, forms: m.forms, models: m.models || [],
        ids: m.ids || [], input: m.input || null, lengths: m.entries || [], dataset: m.dataset || state.dataset || null, start: m.start || {}, id: state.id || '' });
      if (!state.input && state.dataset && lengthOf(state.dataset)) state.input = [1, lengthOf(state.dataset)];   // the host's probe failed, the entry's length still known
      const saved = m.restore ? kept() : null;
      state.trained = null;
      state.trace = null;
      if (saved && saved.design && Array.isArray(saved.design.layers)) restore(saved.design);
      else startFrom(state.start);
      state.entries = null;
      if (state.observe) send({ type: 'sources', model_id: state.modelId });   // the data entries and (Visualise) the model's runs
    } else if (m.type === 'sources') {
      sourcesArrived(m);
    } else if (m.type === 'recovery') {
      if (state.showRecovery && m.run === state.trained) state.recovery = m.result || {};
    } else if (m.type === 'trace') {
      state.tracing = false;
      state.trace = m.result;
      state.traceFresh = true;                           // drawn top to bottom once, on this render
    } else if (m.type === 'check') {
      state.checking = false;
      state.check = m.result;
    } else if (m.type === 'models') {                    // params.yaml was written: its models now
      state.ids = m.ids || [];
      state.models = m.models || state.models;
      const added = state.addedAs ? state.models.find(x => x.model_id === state.addedAs) : null;
      if (added) {                                       // Add The Model written: the original stays as the file has it
        state.start = { model: added };
        startFrom(state.start);
        state.message = `${added.model_id} added to params.yaml; the window now shows it`;
        state.addedAs = null;
        state.addAsk = false;
        state.id = '';
      }
      syncModels();
    } else if (m.type === 'preview') {
      if (state.previewOpen) { state.preview = m.added || []; state.previewMarked = m.marked || []; }   // a late answer after Encapsulate stays folded
    } else if (m.type === 'saved' || m.type === 'refused') {
      state.message = m.message || '';
      if (m.type === 'refused') state.addedAs = null;
      if (m.ids) state.ids = m.ids;
    }
    render();
    state.traceFresh = false;
  });
  const app = document.getElementById('app');
  const num = v => Number(v);
  const on = (type, fn) => app.addEventListener(type, ev => { const t = ev.target; if (t && t.dataset) fn(t.dataset, t, ev); });
  on('click', d => {
    if (d.open != null) act({ type: 'open', i: num(d.open) });
    else if (d.move != null) act({ type: 'move', i: num(d.move), dir: num(d.dir) });
    else if (d.remove != null) act({ type: 'remove', i: num(d.remove) });
    else if (d.plus != null) act({ type: 'plus', at: num(d.plus) });
    else if (d.resolve) act({ type: 'resolve', choice: d.resolve });
    else if (d.rebuildskeleton) { state.pending = { knobs: state.knobs, k: 'the skeleton' }; if (visualising()) act({ type: 'resolve', choice: 'rebuild' }); else render(); }
    else if (d.add) act({ type: 'add' });
    else if (d.preview) act({ type: 'preview' });
    else if (d.download) act({ type: 'download' });
    else if (d.actfold) { state.actOpen = !state.actOpen; render(); }
    else if (d.dropfold) { state.dropOpen = !state.dropOpen; render(); }
    else if (d.poolfold) { state.poolOpen = !state.poolOpen; render(); }
    else if (d.chip) act({ type: 'pattern', op: d.chip, at: num(d.at) });
    else if (d.addask) act({ type: 'addAsk', on: true });
    else if (d.undo) act({ type: 'undo' });
    else if (d.redo) act({ type: 'redo' });
    else if (d.addaskclose) act({ type: 'addAsk', on: false });
    else if (d.dlfold) { dlState().open[d.dlfold] = !dlState().open[d.dlfold]; keepDesign(); render(); }
    else if (d.dlclear) { dl().rules[d.dlclear] = RULE_START(); keepDesign(); render(); }
    else if (d.dlok) act({ type: 'dlOk' });
    else if (d.dlcancel) act({ type: 'dlCancel' });
    else if (d.dlreset) act({ type: 'dlReset' });
    else if (d.reset) act({ type: 'reset' });
    else if (d.cancel) act({ type: 'cancel' });
    else if (d.delsel) act({ type: 'deleteSelected' });
    else if (d.design) act({ type: 'weights', value: '' });
    else if (d.observe) act({ type: 'observe' });
    else if (d.recovery) act({ type: 'recovery' });
    else if (d.applychanges) act({ type: 'applyChanges' });
    else if (d.discardchanges) act({ type: 'discardChanges' });
    else if (d.editsel) act({ type: 'editSelected' });
  });
  on('change', (d, t) => {
    if (d.kind) act({ type: 'kind', value: t.value });
    else if (d.knob) act({ type: 'knob', k: d.knob, value: t.value });
    else if (d.insert != null) act(t.value ? { type: 'insert', at: num(d.insert), name: t.value } : { type: 'plus', at: num(d.insert) });
    else if (d.arg) act({ type: 'arg', i: num(d.i), key: d.arg, value: t.value });
    else if (d.sel != null) act({ type: 'select', i: num(d.sel), on: !!t.checked });
    else if (d.selall) act({ type: 'selectAll', on: !!t.checked });
    else if (d.argsel) act({ type: 'argSelected', key: d.argsel, value: t.value });
    else if (d.dataset || d.inputentry) act({ type: 'dataset', value: t.value });
    else if (d.inputlength) act({ type: 'inputLength', value: t.value });
    else if (d.traceindex) act({ type: 'traceIndex', value: t.value });
    else if (d.traceto) act({ type: 'traceTo', value: t.value });
    else if (d.tracemode) act({ type: 'traceMode', value: t.value });
    else if (d.weights != null) act({ type: 'weights', value: t.value });
    else if (d.layeract != null) act({ type: 'arg', i: num(d.layeract), key: 'activation', value: t.value });
    else if (d.layerdrop != null) act({ type: 'dropout', i: num(d.layerdrop), value: t.value });
    else if (d.chipadd) act({ type: 'pattern', op: 'add', kind: t.value });
    else if (d.pooltype != null) act({ type: 'pooling', b: num(d.pooltype), field: 'type', value: t.value });
    else if (d.poolsize != null) act({ type: 'pooling', b: num(d.poolsize), field: 'size', value: t.value });
    else if (d.actall) act({ type: 'actAll', value: t.value });
    else if (d.dl) setDl(d.dl, t);
  });
  on('input', (d, t) => { if (d.id) { state.id = t.value; idChanged(); } });
  // the splitter: drag (the pointer captured, so a fast drag stays with it), ← → by 16 px, double-click back to 240 px
  on('pointerdown', (d, t, ev) => {
    if (!d.splitter) return;
    if (ev.preventDefault) ev.preventDefault();
    if (t.setPointerCapture && ev.pointerId != null) t.setPointerCapture(ev.pointerId);
    if (t.classList) t.classList.add('dragging');
    drag = { x: ev.clientX, w: state.leftWidth || LEFT_W, el: t };
  });
  app.addEventListener('pointermove', ev => { if (drag) setLeft(drag.w + (ev.clientX - drag.x), false); });
  const endDrag = () => { if (!drag) return; if (drag.el && drag.el.classList) drag.el.classList.remove('dragging'); drag = null; keepDesign(); };
  app.addEventListener('pointerup', endDrag);
  app.addEventListener('pointercancel', endDrag);
  on('dblclick', d => { if (d.splitter) setLeft(LEFT_W); });
  on('keydown', (d, t, ev) => {
    if (d.id && ev.key === 'Enter' && state.addAsk) return act({ type: 'add' });   // the Configuration Change's model_id: Enter adds
    if (!d.splitter || (ev.key !== 'ArrowLeft' && ev.key !== 'ArrowRight')) return;
    if (ev.preventDefault) ev.preventDefault();
    setLeft((state.leftWidth || LEFT_W) + (ev.key === 'ArrowRight' ? 16 : -16));
  });
  on('focusout', d => { if (d.id && stale) render(); });

  if (typeof document.addEventListener === 'function') document.addEventListener('keydown', ev => {   // Visualise: Undo / Redo keys, outside text boxes
    const t = ev.target;
    if (!visualising() || !(ev.ctrlKey || ev.metaKey) || (t && /^(INPUT|TEXTAREA)$/.test(t.tagName))) return;
    const k = String(ev.key || '').toLowerCase();
    const dir = k === 'z' ? (ev.shiftKey ? 'redo' : 'undo') : k === 'y' ? 'redo' : '';
    if (!dir) return;
    if (ev.preventDefault) ev.preventDefault();
    act({ type: dir });
  });
  window.TFPlannerPage = { state, act, render };
  render();
  send({ type: 'ready' });                               // the host sends the start only now: earlier it would be lost
}());
