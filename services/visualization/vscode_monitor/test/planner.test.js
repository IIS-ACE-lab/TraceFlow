// services/visualization/vscode_monitor/test/planner.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('path');
const { execFileSync } = require('child_process');
const E = require('./extension_stub');
const M = require('../webview/menus');
const P = require('../webview/planner_core');

const REPO = path.resolve(__dirname, '..', '..', '..', '..');
// the models of a project's params.yaml, as data (PyYAML reads the file as a run does)
function models(project = 'DLSCA_ASCAD') {
  const file = path.join(REPO, 'projects', project, 'config', 'params.yaml');
  const out = execFileSync(E.python(), ['-c', 'import json, sys, yaml; print(json.dumps(yaml.safe_load(open(sys.argv[1]))["model"]))', file]);
  return JSON.parse(out.toString());
}
const layersOf = (all, id) => all.find(m => m.model_id === id).architecture_args.layers;
const asData = layers => layers.map(l => ({ [l.name]: l.args }));
const FORMS = { activations: ['relu', 'tanh', 'sigmoid', 'selu'] };
// a 4-block CNN in the universal CNN settings (they are not tied to it: it is one model of the project)
const CNN4 = { ...P.defaults('CNN'), blocks: 4, filters: 64, limit: 512, dense: 2, units: 4096 };

test('the settings and the Block Pattern express the project\'s layer models (ASCAD_CNNbase, _CNNbest, _MLPbest)', () => {
  const all = models();
  const cnn = P.SKELETONS.CNN;
  assert.deepEqual(asData(P.chain(cnn.build(CNN4))), layersOf(all, 'ASCAD_CNNbase'));
  assert.deepEqual(P.patternOf(CNN4), ['conv', 'norm', 'pool'], 'the pattern as it starts: Conv1d → BatchNorm1d → Pooling');
  const best = P.fromModel(all.find(m => m.model_id === 'ASCAD_CNNbest'), FORMS);   // avg pooling: each block's own
  assert.deepEqual([best.custom, best.knobs.blocks, best.knobs.kernel], [false, 5, 11]);
  assert.deepEqual([...new Set(best.layers.filter(l => l.handArgs).map(l => `${l.name}[${l.handArgs}]=${l.args.type}`))], ['Pooling[type]=avg']);
  const mlp = { ...P.defaults('MLP'), hidden: 5, units: 200 };
  assert.deepEqual(asData(P.chain(P.SKELETONS.MLP.build(mlp))), layersOf(all, 'ASCAD_MLPbest'));
});

test('no Activation, Activation after, Dense activation, Pooling or Pool size settings: activations per layer, poolings per block', () => {
  for (const type of Object.keys(P.SKELETONS)) {
    const keys = P.SKELETONS[type].knobs.map(k => k.k);
    for (const gone of ['activation', 'actAt', 'denseActivation', 'pool', 'poolSize', 'dropout']) assert.ok(!keys.includes(gone), `${type}.${gone}`);
  }
  const layers = P.SKELETONS.CNN.build(CNN4);
  assert.ok(layers.filter(l => l.name === 'BatchNorm1d').every(l => l.args.activation === 'relu'), 'relu on the last of the pattern before its pool');
  assert.ok(layers.filter(l => l.name === 'Conv1d').every(l => !('activation' in l.args)));
  assert.ok(layers.filter(l => l.name === 'Pooling').every(l => l.args.type === 'max' && l.args.kernel_size === 2));
});

test('every setting and every choice of its menu says what it means on hover (the Visualise window)', () => {
  for (const [type, sk] of Object.entries(P.SKELETONS)) {
    for (const k of sk.knobs) {
      assert.ok(P.knobHint(type, k.k).length > 10, `${type}.${k.k}`);
      if (Array.isArray(k.options)) for (const o of k.options) assert.ok(P.optionHint(k.k, o), `${type}.${k.k}=${o}`);
    }
  }
  const html = page().app.innerHTML;
  assert.match(html, /<span title="a cap on the filters: the most a block may have when doubling[^"]*">Filter limit<\/span>/);
  assert.match(html, /<option value="" selected title="no cap: the filters keep doubling \(or halving\)">none/);
  assert.match(html, /<option value="half" selected title="\(kernel − 1\) \/ 2 at each end: an odd kernel keeps the trace length[^"]*">half<\/option>/);
  assert.match(html, /<select data-knob="growth" title="Filters per block: how the filters change from one block to the next">/);
  assert.match(html, /<span title="what one block is: its layers in order[^"]*">Block Pattern<\/span>/);
});

test('Block Pattern: chips of the block\'s layer kinds; ◀ ▶ move one, × takes it out, + adds one: any kind, any number, any order', () => {
  const p = page();
  const chips = () => [...p.app.innerHTML.matchAll(/<span class="chip chip-(\w+)"/g)].map(m => m[1]).join(' ');
  assert.equal(chips(), 'conv norm pool');
  assert.match(p.app.innerHTML, /<span class="chip chip-conv"[^>]*><span>Conv1d<\/span><button data-chip="right" data-at="0"/);
  p.act({ type: 'pattern', op: 'left', at: 1 });                            // pre-norm: BatchNorm1d before the conv
  assert.equal(chips(), 'norm conv pool');
  assert.equal(p.state().layers[0].name, 'BatchNorm1d');
  assert.equal(p.state().layers[1].args.activation, 'relu', 'relu on the last before the pool');
  p.act({ type: 'pattern', op: 'remove', at: 0 });                          // VGG-style: conv → pool
  assert.equal(chips(), 'conv pool');
  assert.ok(!p.state().layers.some(l => l.name === 'BatchNorm1d'));
  p.act({ type: 'pattern', op: 'add', kind: 'conv' });                      // + adds at the end: conv pool conv
  assert.equal(chips(), 'conv pool conv');
  p.act({ type: 'pattern', op: 'add', kind: 'pool' });                      // a second pooling: conv pool conv pool
  assert.equal(chips(), 'conv pool conv pool');
  assert.equal(p.state().layers.filter(l => l.name === 'Pooling').length, 2 * p.state().knobs.blocks, 'two poolings in every block');
  p.act({ type: 'pattern', op: 'remove', at: 3 });
  p.act({ type: 'pattern', op: 'remove', at: 2 });
  p.act({ type: 'pattern', op: 'remove', at: 0 });                          // no conv at all: a block of a pooling
  assert.equal(chips(), 'pool');
  assert.doesNotMatch(p.app.innerHTML, /class="knoberr"/);
  assert.ok(!p.state().layers.some(l => l.name === 'Conv1d'));
  p.act({ type: 'pattern', op: 'remove', at: 0 });                          // nothing: an empty block, still allowed
  assert.match(p.app.innerHTML, /<div class="chips"[^>]*><small class="faint">an empty block<\/small>/);
  assert.deepEqual(P.parsePattern('pool norm conv pool'), { value: ['pool', 'norm', 'conv', 'pool'] });
  assert.match(P.parsePattern('conv lstm').error, /lstm: not a block layer/);
});

test('fromModel reads the Block Pattern: VGG16 is conv → pool, repeated 2,2,3,3,3', () => {
  const all = models();
  const base = P.fromModel(all.find(m => m.model_id === 'ASCAD_CNNbase'), FORMS);
  assert.deepEqual([base.type, base.custom, base.knobs.blocks, base.knobs.pattern], ['CNN', false, 4, ['conv', 'norm', 'pool']]);
  const vgg = P.fromModel(all.find(m => m.model_id === 'ASCAD_VGG16'), FORMS);
  assert.equal(vgg.custom, false);
  assert.deepEqual([vgg.knobs.blocks, vgg.knobs.pattern, vgg.knobs.convs, vgg.knobs.filters, vgg.knobs.limit], [5, ['conv', 'pool'], [2, 2, 3, 3, 3], 64, 512]);
  assert.deepEqual(asData(P.chain(P.SKELETONS.CNN.build(vgg.knobs))), layersOf(all, 'ASCAD_VGG16'));
  const conv0 = P.fromModel(all.find(m => m.model_id === 'ASCAD_CNNbase_conv0'), FORMS);
  assert.deepEqual([conv0.custom, conv0.knobs.pattern], [false, ['pool']], 'no conv: a block of a pooling');
  assert.deepEqual(asData(P.chain(P.SKELETONS.CNN.build(conv0.knobs))), layersOf(all, 'ASCAD_CNNbase_conv0'));
});

test('every layer-list model of DLSCA_ASCAD reads its settings: none custom; what is its own is an activation, or a pooling\'s type or window', () => {
  for (const m of models().filter(x => x.architecture_args && Array.isArray(x.architecture_args.layers))) {
    const r = P.fromModel(m, FORMS);
    assert.equal(r.custom, false, m.model_id);
    for (const l of r.layers.filter(x => x.handArgs)) {
      assert.ok(l.handArgs.every(k => k === 'activation' || (l.name === 'Pooling' && ['type', 'kernel_size', 'stride'].includes(k))), `${m.model_id} ${l.name} ${l.handArgs}`);
    }
  }
  const all = models();
  const pad = id => P.fromModel(all.find(m => m.model_id === id), FORMS).knobs.padding;
  assert.deepEqual([pad('ASCAD_CNNbase'), pad('ASCAD_CNNbase_kernel6'), pad('ASCAD_CNNbase_validpad')], ['half', 'same', 'valid']);
  const sig = P.fromModel(all.find(m => m.model_id === 'ASCAD_CNNbase_sigmoid'), FORMS);
  assert.deepEqual(sig.layers.filter(l => l.handArgs).map(l => `${l.name}=${l.args.activation}`), Array(4).fill('BatchNorm1d=sigmoid'));
});

test('a layer with arguments of its own (another activation, kernel) keeps the model\'s settings: those arguments are its hand edits', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const mixed = JSON.parse(JSON.stringify(base));
  mixed.architecture_args.layers[4].BatchNorm1d.activation = 'tanh';            // block 2's activation
  mixed.architecture_args.layers[3].Conv1d.kernel_size = 5;                    // block 2's kernel (its padding stays 1)
  mixed.architecture_args.layers[14].Linear.activation = 'selu';               // the second dense layer
  const r = P.fromModel(mixed, FORMS);
  assert.equal(r.custom, false);
  assert.deepEqual(r.layers.map((l, i) => (l.handArgs ? `#${i} ${l.handArgs.join('+')}` : '')).filter(Boolean),
    ['#3 kernel_size', '#4 activation', '#14 activation'], 'only what differs from the settings; in_features follows');
  assert.deepEqual(asData(P.chain(r.layers)), mixed.architecture_args.layers, 'the file\'s layers exactly');
  const rebuilt = P.rebuild('CNN', { ...r.knobs, units: 2048 }, r.layers);       // a setting changed: the layers' own kept
  assert.equal(rebuilt.layers[4].args.activation, 'tanh');
  assert.equal(rebuilt.layers[14].args.activation, 'selu');
  assert.equal(rebuilt.layers[13].args.out_features, 2048);
});

test('an activation on the conv (Conv1d → activation → norm) is each layer\'s own, as Activation per layer shows it', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const onConv = JSON.parse(JSON.stringify(base));
  onConv.architecture_args.layers.forEach(l => { if (l.Conv1d) l.Conv1d.activation = 'relu'; if (l.BatchNorm1d) delete l.BatchNorm1d.activation; });
  const r = P.fromModel(onConv, FORMS);
  assert.equal(r.custom, false);
  assert.deepEqual([...new Set(r.layers.filter(l => l.handArgs).map(l => `${l.name}=${l.args.activation || 'none'}`))], ['Conv1d=relu', 'BatchNorm1d=none']);
  assert.deepEqual(asData(P.chain(r.layers)), onConv.architecture_args.layers);
});

test('Activation per layer: a row per layer that can carry one, its own menu, and "all that have one"; a row set is the layer\'s own', () => {
  const p = page();
  const asked = p.sent.find(m => m.type === 'check');
  p.post({ type: 'check', result: checkOf(asked.layers) });
  assert.match(p.app.innerHTML, /<button class="acthead" data-actfold="1"[^>]*>▸ Activation per layer <small>(none ×\d+, relu ×\d+|relu ×\d+, none ×\d+)<\/small><\/button>/, 'a tally, in the order the layers have them');
  p.state().actOpen = true;
  p.act({ type: 'open', i: null });
  const rows = [...p.app.innerHTML.matchAll(/<label class="actrow[^"]*"[^>]*><span>#(\d+) (\w+)/g)].map(m => `${m[1]} ${m[2]}`);
  const kinds = asked.layers.map((l, i) => `${i} ${Object.keys(l)[0]}`).filter(x => /Conv1d|BatchNorm1d|Linear/.test(x));
  assert.deepEqual(rows, kinds, 'every Conv1d, normalization and Linear; no Flatten, Pooling, Dropout');
  assert.match(p.app.innerHTML, /<label class="actrow actall"[^>]*><span>all that have one<\/span><select data-actall="1"[^>]*><option value="" selected>relu<\/option>/);
  const dense = asked.layers.findIndex(l => l.Linear);
  p.act({ type: 'arg', i: dense, key: 'activation', value: 'tanh' });          // the first dense layer: tanh
  assert.ok(p.state().layers[dense].handArgs.includes('activation'));
  assert.match(p.app.innerHTML, new RegExp(`<label class="actrow own"[^>]*><span>#${dense} Linear <small>edited<\\/small>`));
  p.act({ type: 'knob', k: 'units', value: '256' });                        // a setting changes: with hand edits it asks first
  p.act({ type: 'resolve', choice: 'keep' });
  assert.equal(p.state().layers[dense].args.activation, 'tanh', 'kept');
  p.act({ type: 'actAll', value: 'selu' });                                 // every layer that has one: selu
  assert.ok(p.state().layers.filter(l => l.args.activation).every(l => l.args.activation === 'selu'));
  assert.ok(!p.state().layers.find(l => l.name === 'Conv1d').args.activation, 'a layer without one stays without');
});

test('Pooling per block: a row per block, its type and window; none takes the pooling away, a type puts it back after the block', () => {
  const p = page();
  const pools = () => p.state().layers.map((l, i) => (l.name === 'Pooling' ? `${i}:${l.args.type}/${l.args.kernel_size}/${l.args.stride}` : '')).filter(Boolean);
  const n = pools().length;
  p.state().poolOpen = true;
  p.act({ type: 'open', i: null });
  assert.equal((p.app.innerHTML.match(/<div class="poolrow/g) || []).length, n);
  assert.match(p.app.innerHTML, /<span>block 1 <small>#\d+<\/small><\/span><select data-pooltype="1"[^>]*><option value="none" title="no pooling[^"]*">none<\/option><option value="max" selected/);
  p.act({ type: 'pooling', b: 2, field: 'type', value: 'avg' });
  p.act({ type: 'pooling', b: 2, field: 'size', value: '4' });
  assert.match(pools()[1], /avg\/4\/4$/, 'block 2: avg, window and stride 4');
  p.act({ type: 'pooling', b: 1, field: 'type', value: 'none' });
  assert.equal(pools().length, n - 1);
  assert.match(p.app.innerHTML, /<span>block 1<\/span><select data-pooltype="1"[^>]*><option value="none" selected/);
  p.act({ type: 'pooling', b: 1, field: 'type', value: 'max' });           // back, at the block's end
  assert.equal(pools().length, n);
  const at = p.state().layers.findIndex(l => l.name === 'Pooling');
  assert.match(p.state().layers[at - 1].slot || '', /^b1\./, 'after block 1\'s layers');
});

test('dense layers with different activations: each its own', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const m = JSON.parse(JSON.stringify(base));
  const lin = m.architecture_args.layers.map((l, i) => (l.Linear && l.Linear.activation ? i : -1)).filter(i => i >= 0);
  m.architecture_args.layers[lin[0]].Linear.activation = 'tanh';
  m.architecture_args.layers[lin[1]].Linear.activation = 'sigmoid';
  const r = P.fromModel(m, FORMS);
  assert.equal(r.custom, false);
  assert.deepEqual(r.layers.map((l, i) => (l.handArgs ? `#${i} ${l.args.activation}` : '')).filter(Boolean), [`#${lin[0]} tanh`, `#${lin[1]} sigmoid`]);
});

test('Repeats per block takes one number, or one per block', () => {
  const k = P.SKELETONS.CNN.knobs.find(x => x.k === 'convs');
  assert.deepEqual(P.parseKnob(k, '2,2,3'), { value: [2, 2, 3] });
  assert.deepEqual(P.parseKnob(k, '2, 2'), { value: 2 }, 'all alike: one number');
  assert.match(P.parseKnob(k, '2,x').error, /whole number, one per block/);
  const n = P.SKELETONS.CNN.build({ ...CNN4, blocks: 3, convs: [1, 2] }).filter(l => l.name === 'Conv1d').length;
  assert.equal(n, 1 + 2 + 2, 'a block past the list takes its last number');
});

test('every layer-list model of DLSCA_ASCAD round-trips through fromModel and toModel unchanged', () => {
  for (const m of models().filter(x => x.architecture_args && Array.isArray(x.architecture_args.layers))) {
    const planned = P.fromModel(m, FORMS);
    const back = P.toModel(m.model_id, planned.type, planned.layers);
    assert.deepEqual(back.architecture_args.layers, m.architecture_args.layers, m.model_id);
    assert.deepEqual(P.arrangeOf(m.architecture_args.layers, planned.layers), m.architecture_args.layers.map((_, k) => ({ from: k })), m.model_id);
  }
});

test('chain fills in_features after Flatten (512 x 43 = 22016) and keeps a hand value', () => {
  const layers = P.SKELETONS.CNN.build(CNN4);
  const flat = layers.findIndex(l => l.name === 'Flatten');
  layers[flat + 1].args.in_features = 1;
  assert.equal(P.chain(layers)[flat + 1].args.in_features, 22016);
  layers[flat + 1].handArgs = ['in_features'];
  layers[flat + 1].args.in_features = 999;
  const chained = P.chain(layers);
  assert.equal(chained[flat + 1].args.in_features, 999);
  assert.equal(chained[flat + 2].args.in_features, 4096);                 // the layer after still follows
});

test('rebuild keeps a Dropout added after block 1 and reports a hand edit in a removed block', () => {
  const knobs = CNN4;
  let layers = P.SKELETONS.CNN.build(knobs);
  const pool1 = layers.findIndex(l => l.name === 'Pooling');
  layers = P.insert(layers, pool1 + 1, [{ name: 'Dropout', args: { p: 0.2 } }]);
  const conv4 = layers.map((l, i) => [l, i]).filter(([l]) => l.name === 'Conv1d')[3][1];
  layers[conv4] = { ...layers[conv4], args: { ...layers[conv4].args, kernel_size: 5, padding: 2 }, handArgs: ['kernel_size', 'padding'] };
  assert.equal(P.hasHandEdits(layers), true);
  const out = P.rebuild('CNN', { ...knobs, blocks: 3 }, layers);
  const names = out.layers.map(l => l.name);
  assert.equal(names[names.indexOf('Pooling') + 1], 'Dropout');
  assert.deepEqual(out.dropped, ['Conv1d (block 4)']);
  assert.equal(P.hasHandEdits(P.SKELETONS.CNN.build(knobs)), false);
});

test('a removed skeleton layer is a hand edit, and stays removed when the changes are kept', () => {
  const knobs = CNN4;
  const layers = P.SKELETONS.CNN.build(knobs);
  const dense2 = layers.find(l => l.slot === 'dense2');
  const without = layers.filter(l => l !== dense2);
  assert.equal(P.hasHandEdits(without, [dense2.slot]), true);
  const out = P.rebuild('CNN', { ...knobs, kernel: 11 }, without, undefined, [dense2.slot]);
  assert.equal(out.layers.some(l => l.slot === 'dense2'), false);
  assert.equal(out.layers.length, layers.length - 1);
});

test('an added layer keeps an argument set by hand; a moved layer still takes its arguments from the layer above', () => {
  let layers = P.insert(P.SKELETONS.MLP.build({ ...P.defaults('MLP'), units: 200 }), 1,
    [{ name: 'Linear', args: { in_features: 0, out_features: 50 } }]);
  layers[1] = { ...layers[1], args: { ...layers[1].args, in_features: 123 }, handArgs: ['in_features'] };
  assert.equal(P.chain(layers)[1].args.in_features, 123);
  const conv = P.chain(P.SKELETONS.CNN.build({ ...CNN4, blocks: 2, dense: 0 }));
  const moved = [conv[0], conv[1], conv[3], conv[2], ...conv.slice(4)].map((l, i) => (i === 2 ? { ...l, moved: true } : l));
  assert.equal(P.chain(moved)[2].args.in_channels, 64);
  assert.equal(P.hasHandEdits(moved), true);
});

test('arrangeOf a moved and an inserted layer', () => {
  const original = [{ Conv1d: { in_channels: 1, out_channels: 8, kernel_size: 3 } }, { Flatten: {} }, { Linear: { in_features: 5600, out_features: 'auto' } }];
  const planned = P.fromModel({ model_id: 'X', architecture: 'CNN', architecture_args: { layers: original } }, FORMS).layers;
  const moved = [planned[1], planned[0], ...P.insert([], 0, [{ name: 'Dropout', args: { p: 0.1 } }]), planned[2]];
  moved[3] = { ...moved[3], args: { in_features: 99, out_features: 'auto' } };
  assert.deepEqual(P.arrangeOf(original, moved), [{ from: 1 }, { from: 0 }, { value: { Dropout: { p: 0.1 } } },
    { from: 2, value: { Linear: { in_features: 99, out_features: 'auto' } } }]);
});

test('idState: unique ids pass, taken or malformed ones say why, a case-only variant passes with a note', () => {
  const ids = ['ASCAD_CNNbase', 'ASCAD_MLPbest'];
  assert.deepEqual(P.idState('MY_CNN', ids, M.askError), { ok: true, error: '', note: '' });
  assert.equal(P.idState('ASCAD_CNNbase', ids, M.askError).ok, false);
  assert.match(P.idState('ASCAD_CNNbase', ids, M.askError).error, /already/);
  assert.match(P.idState('my cnn', ids, M.askError).error, /no spaces/);
  const caseOnly = P.idState('ascad_cnnbase', ids, M.askError);
  assert.equal(caseOnly.ok, true);
  assert.match(caseOnly.note, /differs only in case from ASCAD_CNNbase/);
});

test('sameAs names the existing model a design equals, else null', () => {
  const all = models();
  const base = P.fromModel(all.find(m => m.model_id === 'ASCAD_CNNbase'), FORMS);
  assert.equal(P.sameAs(base.layers, all), 'ASCAD_CNNbase');
  assert.equal(P.sameAs(P.insert(base.layers, 0, [{ name: 'Dropout', args: { p: 0.1 } }]), all), null);
});

test('a knob is typed: whole numbers, decimals and words read as such; a bad value is refused with why', () => {
  const knob = k => P.SKELETONS.CNN.knobs.concat(P.SKELETONS.MLP.knobs, P.SKELETONS.TRANSFORMER.knobs).find(x => x.k === k);
  assert.deepEqual(P.parseKnob(knob('filters'), ' 48 '), { value: 48 });
  assert.deepEqual(P.parseKnob(knob('attnDropout'), '0.25'), { value: 0.25 });
  assert.deepEqual(P.parseKnob(knob('norm'), 'InstanceNorm1d'), { value: 'InstanceNorm1d' });
  assert.deepEqual(P.parseKnob(knob('pattern'), 'conv, norm → pool'), { value: ['conv', 'norm', 'pool'] });
  assert.match(P.parseKnob(knob('filters'), 'abc').error, /a whole number/);
  assert.match(P.parseKnob(knob('blocks'), '2.5').error, /a whole number/);
  assert.match(P.parseKnob(knob('attnDropout'), '1').error, /below 1/);
  assert.match(P.parseKnob(knob('kernel'), '0').error, /at least 1/);
});

test('a free kernel size keeps whole padding, and free values are read back from a model', () => {
  const knobs = { ...P.defaults('CNN'), blocks: 3, convs: 2, filters: 48, kernel: 4, dense: 1, units: 300 };
  const layers = P.chain(P.SKELETONS.CNN.build(knobs));
  assert.equal(layers[0].args.padding, 1);
  const back = P.fromModel(P.toModel('X', 'CNN', layers), FORMS);
  assert.deepEqual(back.knobs, knobs);
  assert.equal(back.custom, false);
  for (const more of [{ pattern: ['conv', 'pool'], growth: 'halve', filters: 256, limit: 128 }, { growth: 'same', norm: 'InstanceNorm1d' },
    { pattern: ['conv', 'norm'], blocks: 2 }, { pattern: ['norm', 'conv', 'pool'] }, { pattern: ['conv', 'conv', 'norm', 'pool'], convs: [1, 2, 2] }]) {
    const o = { ...knobs, ...more };
    assert.deepEqual(P.fromModel(P.toModel('X', 'CNN', P.chain(P.SKELETONS.CNN.build(o))), FORMS).knobs, o, JSON.stringify(more));
  }
  const mlp = { ...P.defaults('MLP'), hidden: 3, units: 64, norm: 'LayerNorm' };
  const mlpLayers = P.chain(P.SKELETONS.MLP.build(mlp));
  assert.deepEqual(mlpLayers[2].args, { normalized_shape: 64, activation: 'relu' });
  assert.deepEqual(P.fromModel(P.toModel('X', 'MLP', mlpLayers), FORMS).knobs, mlp);
  const bn = P.chain(P.SKELETONS.MLP.build({ ...mlp, norm: 'BatchNorm1d' }));
  assert.equal(bn[2].args.num_features, 64);                               // after Flatten: the vector width, not the channels
});

test('the filters of each block: the same, doubled or halved, stopped at the limit', () => {
  const of = o => [1, 2, 3, 4, 5].map(b => P.filtersOf({ filters: 64, limit: null, ...o }, b));
  assert.deepEqual(of({ growth: 'same' }), [64, 64, 64, 64, 64]);
  assert.deepEqual(of({ growth: 'double', limit: 512 }), [64, 128, 256, 512, 512]);
  assert.deepEqual(of({ growth: 'halve', limit: 16 }), [64, 32, 16, 16, 16]);
  assert.deepEqual(P.parseKnob(P.SKELETONS.CNN.knobs.find(k => k.k === 'limit'), ''), { value: null });
});

test('TRANSFORMER and HYBRID have their own settings and layer lists, from core.model\'s layer names', () => {
  const t = P.SKELETONS.TRANSFORMER.build(P.defaults('TRANSFORMER'));
  assert.deepEqual(t.map(l => l.name), ['Conv1d', 'TransformerEncoderLayer', 'TransformerEncoderLayer', 'Flatten', 'Linear', 'Linear']);
  assert.deepEqual(t[0].args, { in_channels: 0, out_channels: 64, kernel_size: 16, stride: 16, padding: 0 });
  assert.deepEqual(t[1].args, { d_model: 64, nhead: 4, dim_feedforward: 128, dropout: 0.1, batch_first: true });
  const h = P.chain(P.SKELETONS.HYBRID.build({ ...P.defaults('HYBRID'), cell: 'GRU' }));
  assert.deepEqual(h.map(l => l.name), ['Conv1d', 'Pooling', 'Conv1d', 'Pooling', 'GRU', 'Flatten', 'Linear', 'Linear']);
  assert.equal(h[4].args.input_size, 64);                                  // the channels of the conv above
  const heads = P.knobChoices(P.SKELETONS.TRANSFORMER.knobs.find(k => k.k === 'heads'), { width: 48, heads: 4 });
  assert.deepEqual(heads.map(c => c.v), [1, 2, 3, 4, 6, 8, 12, 16, 24]);
});

test('blocks offer the counts the trace can take, each with the length it leaves; filters offer powers of two', () => {
  const knob = k => P.SKELETONS.CNN.knobs.find(x => x.k === k);
  const base = CNN4;
  const blocks = P.knobChoices(knob('blocks'), base, [1, 700]);
  assert.deepEqual(blocks.map(c => c.v), [1, 2, 3, 4, 5, 6, 7, 8, 9]);                  // 700 halves 9 times to 1
  assert.equal(blocks[3].label, '4 (length 43)');
  const even = P.knobChoices(knob('blocks'), { ...base, kernel: 4, convs: 2 }, [1, 700]);  // each even conv drops one
  assert.deepEqual(even.map(c => c.v), [1, 2, 3, 4, 5, 6, 7]);
  assert.equal(even[6].label, '7 (length 3)');
  assert.equal(P.knobChoices(knob('blocks'), { ...base, pattern: ['conv', 'norm'] }, [1, 700]).length, 12);          // no pooling: nothing shortens it
  assert.equal(P.knobChoices(knob('blocks'), { ...base, padding: 'valid', kernel: 11 }, [1, 700])[0].label, '1 (length 345)'); // 700 − 10, halved
  assert.deepEqual(P.knobChoices(knob('filters'), base, [1, 700]).map(c => c.v), [4, 8, 16, 32, 64, 128, 256, 512, 1024]);
  assert.deepEqual(P.knobChoices(knob('filters'), { ...base, filters: 48 }, [1, 700]).map(c => c.v), [4, 8, 16, 32, 48, 64, 128, 256, 512, 1024]);
  assert.equal(P.knobChoices(knob('kernel'), base, [1, 700]), null);                    // typed
});

test('the filter limit is a menu: a ceiling above the first filters when doubling, a floor below when halving, each with the filters per block', () => {
  const limit = P.SKELETONS.CNN.knobs.find(x => x.k === 'limit');
  const up = P.knobChoices(limit, CNN4, [1, 700]);                                      // 64 doubled over 4 blocks, limit 512
  assert.deepEqual(up.map(c => c.v), ['', 128, 256, 512, 1024]);
  assert.equal(up[0].label, 'none (64, 128, 256, 512)');
  assert.equal(up[1].label, '128 (64, 128, 128, 128)');
  const down = P.knobChoices(limit, { ...CNN4, growth: 'halve', limit: null }, [1, 700]);
  assert.deepEqual(down.map(c => c.v), ['', 4, 8, 16, 32]);
  assert.equal(down[3].label, '16 (64, 32, 16, 16)');
  assert.deepEqual(P.knobChoices(limit, { ...CNN4, growth: 'same', limit: null }, [1, 700]).map(c => c.v), ['']);   // no effect: none alone
  assert.deepEqual(P.knobChoices(limit, { ...CNN4, limit: 300 }, [1, 700]).map(c => c.v), ['', 128, 256, 300, 512, 1024]);   // read from a model: offered
});

// ── the page (webview/planner.js) in node with a minimal DOM ───────────────────────────────────────────
const fs = require('fs');
const vm = require('vm');

function forms() {
  const out = execFileSync(E.python(), ['-c', 'import json; from services.visualization.vscode_monitor.choices import model_forms; print(json.dumps(model_forms()))'],
    { cwd: REPO, env: { ...process.env, CUDA_VISIBLE_DEVICES: '', PYTHONPATH: [REPO, path.join(REPO, 'services/side_channel_core'), path.join(REPO, 'services/side_channel_core/src')].join(':') } });
  return JSON.parse(out.toString().trim().split('\n').pop());
}
let FORMS_REAL = null;
// graph_manager's forms as the extension passes them to the page (choices.py graph_forms: export, text_rules, diagram)
function graphForms() {
  const out = execFileSync(E.python(), ['-c', 'import json; from services.visualization.vscode_monitor.choices import graph_forms; g = graph_forms(); ' +
    'print(json.dumps({"export": g["export"]["job"], "textRule": g["textRule"], "diagram": g["diagram"]}))'],
    { cwd: REPO, env: { ...process.env, CUDA_VISIBLE_DEVICES: '', PYTHONPATH: [REPO, path.join(REPO, 'services/side_channel_core'), path.join(REPO, 'services/side_channel_core/src')].join(':') } });
  return JSON.parse(out.toString().trim().split('\n').pop());
}
let GRAPH_REAL = null;
const listenerOf = (p, type) => p.appOn[type];
function page(init = {}, opts = {}) {
  FORMS_REAL = FORMS_REAL || forms();
  GRAPH_REAL = GRAPH_REAL || graphForms();
  let draws = 0;
  let html = '';
  const appOn = {};                                                   // the page's DOM handlers (click, change, ...)
  const docOn = {};                                                   // the document's (the Undo / Redo keys)
  const app = { get innerHTML() { return html; }, set innerHTML(v) { html = v; draws += 1; }, addEventListener: (t, fn) => { appOn[t] = fn; } };
  const found = {};
  const query = sel => found[sel] || (found[sel] = { textContent: '', className: '', disabled: false, title: '' });
  const listeners = {};
  const sent = [];
  const saved = { value: opts.saved == null ? null : opts.saved };    // the tab's vscode.setState, kept over a reload
  const ctx = {
    acquireVsCodeApi: () => ({ postMessage: m => sent.push(m), setState: v => { saved.value = JSON.parse(JSON.stringify(v)); }, getState: () => saved.value }),
    document: { getElementById: () => app, addEventListener: (t, fn) => { docOn[t] = fn; }, ...(opts.dom ? { querySelector: query, activeElement: null } : {}) },
    window: { addEventListener: (t, fn) => { listeners[t] = fn; }, TFMenus: M, TFPlanner: P, ...(opts.window || {}) },
    console,
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../webview/planner.js'), 'utf8'), ctx);
  const post = m => listeners.message({ data: m });
  post({ type: 'init', project: 'DLSCA_ASCAD', framework: 'pytorch', forms: FORMS_REAL, graph: GRAPH_REAL, models: models(), ids: models().map(m => m.model_id),
    input: [1, 700], start: {}, ...init });
  return { app, appOn, docOn, post, sent, act: a => ctx.window.TFPlannerPage.act(a), state: () => ctx.window.TFPlannerPage.state, draws: () => draws, found, ctx,
    saved: () => saved.value };
}
const checkOf = (layers, extra = {}) => ({ error: null, at: null, params: 1, layers: layers.map((l, i) => ({ index: i, type: Object.keys(l)[0], out: [8, 100] })), ...extra });

test('the scale\'s labels never touch: a power of ten next to the axis end gives way (1000 beside 10752)', () => {
  assert.deepEqual(P.scaleTicks(10752, 170), [1, 10, 100, 10752]);
  assert.deepEqual(P.scaleTicks(700, 170), [1, 10, 100, 700]);
  assert.deepEqual(P.scaleTicks(100000, 170), [1, 10, 100, 1000, 100000]);
});

test('Download: a choice\'s settings show only once it is chosen (formats, the Scaled Sparkline, text_rules); the page sends the model as drawn', () => {
  const p = page({}, { dom: true });
  const asked = p.sent.find(m => m.type === 'check');
  p.post({ type: 'check', result: checkOf(asked.layers) });
  const html = () => p.app.innerHTML;
  const change = (key, value, type = 'select-one') => listenerOf(p, 'change')({ target: { dataset: { dl: key }, type,
    value: String(value), checked: !!value } });
  // the style: Full Version, Layer cards, Scaled Sparkline
  assert.match(html(), /<select data-dl="style"[^>]*><option value="full" selected>Full Version<\/option><option value="cards">Layer cards<\/option><option value="sparkline">Scaled Sparkline<\/option>/);
  assert.doesNotMatch(html(), /data-dl="sparkline\./, 'the Scaled Sparkline\'s choices are out of sight');
  // the theme: every theme of services/visualization/utils/fig_theme.py, traceflow first chosen
  assert.match(html(), /<select data-dl="theme"[^>]*>(<option value="[a-z_]+"( selected)?>[a-z_]+<\/option>)+<\/select>/);
  for (const t of ['traceflow', 'cyberpunk', 'samurai', 'okabe_ito', 'grayscale', 'nord']) assert.match(html(), new RegExp(`<option value="${t}"`), t);
  assert.match(html(), /<option value="traceflow" selected>/);
  change('theme', 'cyberpunk');
  // formats ticked by default show their settings (graph_manager's); unticked, they are out of sight
  assert.match(html(), /data-dl="formats\.png" checked[^]*?data-dl="settings\.png\.dpi"[^>]*value="300"/);
  assert.match(html(), /data-dl="settings\.svg\.sanitize" checked/);
  assert.match(html(), /<select data-dl="settings\.svg\.on_error"/);
  assert.match(html(), /data-dl="formats\.eps" title/, 'eps listed, unticked');
  change('formats.png', false, 'checkbox');
  assert.doesNotMatch(html(), /settings\.png\.dpi/);
  change('style', 'sparkline');
  for (const k of ['sparkline.arguments', 'sparkline.ticks', 'sparkline.orientation', 'sparkline.legend']) assert.match(html(), new RegExp(`data-dl="${k.replace('.', '\\.')}"`), k);
  assert.match(html(), /<select data-dl="sparkline\.legend"[^>]*><option value="top-right" selected>top-right<\/option><option value="bottom-right">bottom-right<\/option><option value="top-left">top-left<\/option><option value="none">none<\/option>/);
  change('sparkline.orientation', 'landscape');
  change('arrows', true, 'checkbox');
  // text_rules: folded; a target's fields under it, graph_manager's; text only for title and subtitle
  assert.match(html(), /data-dlfold="rules"[^>]*>▸ text_rules <small>0\/9 enabled<\/small>/);
  listenerOf(p, 'click')({ target: { dataset: { dlfold: 'rules' } }, preventDefault: () => {} });
  listenerOf(p, 'click')({ target: { dataset: { dlfold: 'rule:title' } }, preventDefault: () => {} });
  listenerOf(p, 'click')({ target: { dataset: { dlfold: 'rule:arguments' } }, preventDefault: () => {} });
  assert.match(html(), /data-dl="rules\.title\.text"/);
  assert.doesNotMatch(html(), /data-dl="rules\.arguments\.text"/);
  assert.match(html(), /data-dl="rules\.arguments\.font\.size"/);
  // every rule starts off, at font size 8, with its enabled switch first
  assert.match(html(), /<select data-dl="rules\.title\.enabled"[^>]*><option value="true">true<\/option><option value="false" selected>false<\/option>/);
  assert.match(html(), /data-dl="rules\.title\.font\.size"[^>]*value="8"/);
  assert.match(html(), /text_rules <small>0\/9 enabled<\/small>/);
  change('rules.title.text', 'CNN base');
  change('rules.title.font.size', 16, 'number');
  change('rules.title.enabled', 'true');
  change('rules.arguments.italic', 'true');                                // set, but its rule stays off: not drawn
  // the changes are recognised: OK, Cancel, Reset; Download waits for OK
  assert.match(html(), /<div class="pending dlnote"><span title="[^"]*"><b>Download settings<\/b> · \d+ changes recognised<\/span><\/div>/);
  assert.match(html(), /<div class="dlbuttons"><button data-download="1" disabled title="OK or Cancel the changes to the Download settings first">Download<\/button><button data-dlok="1"[^>]*>OK<\/button><button data-dlcancel="1"[^>]*>Cancel<\/button><button data-dlreset="1"[^>]*>Reset<\/button><\/div>/, 'beside Download');
  const before = p.sent.length;
  p.act({ type: 'download' });
  assert.equal(p.sent.length, before, 'nothing sent before OK');
  listenerOf(p, 'click')({ target: { dataset: { dlok: '1' } }, preventDefault: () => {} });
  assert.doesNotMatch(html(), /changes recognised/);
  p.act({ type: 'download' });
  const m = p.sent.at(-1);
  assert.equal(m.type, 'download');
  assert.equal(m.title, 'new model');
  assert.equal(m.style, 'sparkline');
  assert.equal(m.theme, 'cyberpunk');
  assert.equal(m.arrows, true);
  assert.deepEqual({ ...m.sparkline }, { arguments: true, orientation: 'landscape', ticks: true, legend: 'top-right' });
  assert.deepEqual(JSON.parse(JSON.stringify(m.text_rules)), { title: { text: 'CNN base', font: { size: 16 } } });
  assert.deepEqual(JSON.parse(JSON.stringify(m.export)), { formats: ['pdf', 'svg'], margin: 0.05,
    svg: { sanitize: true, validate_xml: true, on_error: 'warn' } }, 'no pdf block: graph_manager refuses one');
  assert.match(html(), /data-dl="formats\.png"[^]*data-dl="formats\.pdf"[^]*data-dl="formats\.svg"[^]*data-dl="formats\.eps"/, 'PNG, PDF, SVG first');
  assert.equal(m.layers.length, asked.layers.length);
  assert.equal(m.traces, false);
  for (const f of ['svg', 'pdf']) change(`formats.${f}`, false, 'checkbox');   // nothing ticked: Download waits
  listenerOf(p, 'click')({ target: { dataset: { dlok: '1' } }, preventDefault: () => {} });
  p.act({ type: 'download' });
  assert.equal(p.sent.filter(x => x.type === 'download').length, 1);
  assert.match(html(), /<button class="wide" data-download="1" disabled title="tick a format">Download<\/button>/);
});

test('Download settings: Reset undoes the changes (the settings as they were before), Cancel undoes them and folds the text_rules away', () => {
  const p = page();
  const html = () => p.app.innerHTML;
  const change = (key, value, type = 'select-one') => listenerOf(p, 'change')({ target: { dataset: { dl: key }, type, value: String(value), checked: !!value } });
  const click = d => listenerOf(p, 'click')({ target: { dataset: d }, preventDefault: () => {} });
  change('theme', 'nord');
  click({ dlok: '1' });                                                       // kept: nord
  change('theme', 'dracula');
  change('style', 'cards');
  assert.match(html(), /2 changes recognised/);
  click({ dlfold: 'rules' });
  click({ dlreset: '1' });
  assert.doesNotMatch(html(), /changes recognised/);
  assert.match(html(), /<option value="nord" selected>/, 'back to the settings before the changes, not the defaults');
  assert.match(html(), /<option value="full" selected>Full Version/);
  assert.match(html(), /data-dlfold="rules"[^>]*>▾/, 'Reset leaves the form as it is');
  change('theme', 'monokai');
  click({ dlcancel: '1' });
  assert.match(html(), /<option value="nord" selected>/);
  assert.match(html(), /data-dlfold="rules"[^>]*>▸/, 'Cancel folds the text_rules away');
});

test('the Download theme, once chosen, recolours the cards in the window as the diagram will draw them', () => {
  const p = page({}, { dom: true });
  const props = new Map();
  const classes = new Set();
  const style = { setProperty: (k, v) => props.set(k, v), removeProperty: k => props.delete(k), [Symbol.iterator]: () => props.keys() };
  p.found['.stack'] = { style, classList: { toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)) } };
  p.act({ type: 'open', i: null });
  assert.equal(props.size, 0, 'before a theme is chosen: VS Code\'s own colours');
  listenerOf(p, 'change')({ target: { dataset: { dl: 'theme' }, type: 'select-one', value: 'cyberpunk' } });
  assert.equal(props.get('--vscode-editor-background'), '#0D1117');
  assert.equal(props.get('--tf-convolution'), '#00E5FF');
  assert.ok(classes.has('themed'));
  listenerOf(p, 'change')({ target: { dataset: { dl: 'theme' }, type: 'select-one', value: 'solarized' } });
  assert.equal(props.get('--vscode-editor-background'), '#FDF6E3');
  listenerOf(p, 'click')({ target: { dataset: { dlreset: '1' } }, preventDefault: () => {} });   // back to the theme kept
  assert.equal(props.get('--tf-convolution'), '#1A85FF', 'traceflow, the theme kept');
});

test('Download\'s file name: the model, the trained run chosen, and the Observe Trace settings drawn (data entry, weights, trace or range)', () => {
  const p = page();
  const asked = p.sent.find(m => m.type === 'check');
  p.post({ type: 'check', result: checkOf(asked.layers) });
  p.act({ type: 'download' });
  assert.deepEqual([p.sent.at(-1).name, p.sent.at(-1).tag, p.sent.at(-1).run], ['new model', '', '']);
  const lines = asked.layers.map((l, i) => ({ index: i, kind: 'vector', shape: [2], values: [1, 2], lo: 1, hi: 2 }));
  p.act({ type: 'observe' });
  p.post({ type: 'trace', result: { source: { dataset: 'ascad', type: 'ASCADf', trace: 0, to: 9 }, layers: lines, input: { values: [1, 2], length: 2 } } });
  p.act({ type: 'download' });
  let m = p.sent.at(-1);
  assert.deepEqual([m.name, m.tag, m.traces], ['new model__trace_ascad_untrained_t0-9', 'trace_ascad_untrained_t0-9', true]);
  assert.equal(m.trace.caption, 'untrained weights · traces 0–9 (mean and band) of ascad (ASCADf)');
  p.state().dl.saved.traces = false;                                   // the lines left out: the name says nothing of them
  p.state().dl.draft.traces = false;
  p.act({ type: 'download' });
  assert.equal(p.sent.at(-1).name, 'new model');
  p.state().dl.saved.traces = true;
  p.state().dl.draft.traces = true;
  Object.assign(p.state(), { modelId: 'A', trained: 'r1', runs: [{ name: 'r1', env: 'staging', architecture: 'MLP', framework: 'pytorch', layers: [{ Flatten: {} }] }] });
  p.post({ type: 'trace', result: { source: { dataset: 'ascad', type: 'ASCADf', trace: 5 }, checkpoint: { policy: 'final_epoch', epoch: 50 },
    layers: [{ index: 0, kind: 'vector', shape: [2], values: [1, 2], lo: 1, hi: 2 }] } });
  p.act({ type: 'download' });
  m = p.sent.at(-1);
  assert.deepEqual([m.name, m.tag, m.run, m.env, m.title], ['A__r1__trace_ascad_t5', 'trace_ascad_t5', 'r1', 'staging', 'r1']);
});

test('the splitter: drag it to widen or narrow the settings (180 px to 60 % of the window), ← → by 16, double-click back to 240; kept', () => {
  const p = page({}, { dom: true, window: { innerWidth: 1000 } });
  const width = new Map();
  p.found['.planner'] = { style: { setProperty: (k, v) => width.set(k, v) } };
  p.act({ type: 'open', i: null });
  assert.match(p.app.innerHTML, /<div class="left">[^]*<\/div><div class="splitter" data-splitter="1" role="separator" aria-orientation="vertical" tabindex="0" title="drag to widen/);
  assert.equal(width.get('--leftw'), '240px');
  const bar = { dataset: { splitter: '1' }, setPointerCapture: () => {}, classList: { add() {}, remove() {} } };
  p.appOn.pointerdown({ target: bar, clientX: 300, pointerId: 1, preventDefault: () => {} });
  p.appOn.pointermove({ clientX: 420 });
  assert.equal(width.get('--leftw'), '360px', '240 + 120');
  p.appOn.pointermove({ clientX: 2000 });
  assert.equal(width.get('--leftw'), '600px', 'at most 60 % of the window');
  p.appOn.pointermove({ clientX: -500 });
  assert.equal(width.get('--leftw'), '180px', 'at least 180 px');
  p.appOn.pointermove({ clientX: 360 });
  p.appOn.pointerup({});
  assert.equal(p.saved().design.leftWidth, 300, 'kept in the tab state');
  p.appOn.pointermove({ clientX: 900 });
  assert.equal(width.get('--leftw'), '300px', 'no drag after the pointer is up');
  p.appOn.keydown({ target: bar, key: 'ArrowRight', preventDefault: () => {} });
  assert.equal(width.get('--leftw'), '316px');
  p.appOn.dblclick({ target: bar });
  assert.equal(width.get('--leftw'), '240px');
});

test('the page says it is ready as soon as it loads (the host sends its start only then)', () => {
  assert.equal(page().sent[0].type, 'ready');
});

test('the page draws a card and a bar per layer, coloured by group', () => {
  const p = page();
  const asked = p.sent.find(m => m.type === 'check');
  p.post({ type: 'check', result: checkOf(asked.layers) });
  const html = p.app.innerHTML;
  assert.equal((html.match(/<div class="card g-/g) || []).length, asked.layers.length);
  assert.equal((html.match(/<rect class="g-/g) || []).length, asked.layers.length);
  assert.match(html, /card g-convolution/);
  assert.match(html, /card g-pooling/);
  assert.match(html, /card g-dense/);
});

test('the bars read against a scale: a log axis with ticks over the column, gridlines and a tooltip on every bar', () => {
  const p = page();
  const asked = p.sent.find(m => m.type === 'check');
  const outs = { Conv1d: [32, 700], BatchNorm1d: [32, 700], Pooling: [32, 350], Flatten: [5600], Linear: [128] };
  p.post({ type: 'check', result: { error: null, at: null, params: 1,
    layers: asked.layers.map((l, i) => ({ index: i, type: Object.keys(l)[0], out: outs[Object.keys(l)[0]] })) } });
  const html = p.app.innerHTML;
  const axis = (/<svg class="scale"[^]*?<\/svg>/.exec(html) || [''])[0];
  for (const t of ['1', '10', '100', '5600']) assert.match(axis, new RegExp(`<text[^>]*>${t}</text>`), t);
  assert.doesNotMatch(axis, /<text[^>]*>1000<\/text>/, '1000 would touch 5600: it gives way');
  assert.match(html, /class="scalecap"[^>]*>[^<]*log scale/);
  const bars = html.match(/<svg class="bar"[^]*?<\/svg>/g);
  assert.equal(bars.length, asked.layers.length);
  for (const b of bars) assert.match(b, /<line class="grid"/);
  assert.match(bars[0], /<title>length 700 × 32 channels<\/title>/);
  const width = b => Number(/<rect [^>]*width="([\d.]+)"/.exec(b)[1]);
  const pool = asked.layers.findIndex(l => Object.keys(l)[0] === 'Pooling');
  assert.ok(width(bars[0]) > width(bars[pool]), 'a pooled layer is shorter on the scale');
  assert.match(bars[asked.layers.findIndex(l => Object.keys(l)[0] === 'Flatten')], /<title>5600 features<\/title>/);
});

test('TRANSFORMER and HYBRID are listed disabled with their reasons', () => {
  const html = page().app.innerHTML;
  assert.match(html, /<option value="TRANSFORMER" disabled title="none of its layers builds/);
  assert.match(html, /<option value="HYBRID" disabled title="none of its layers builds[^"]*LSTM/);
  const p = page();
  p.act({ type: 'plus', at: 0 });
  assert.match(p.app.innerHTML, /<option value="LSTM" disabled title="returns \(output/);
});

test('under keras only Keras\'s layers are offered and the other Types are disabled', () => {
  const k = page({ framework: 'keras' });
  k.act({ type: 'plus', at: 1 });
  const html = k.app.innerHTML;
  assert.match(html, /<option value="LayerNorm" disabled title="no Keras layer/);
  assert.match(html, /<option value="Conv1d">/);
  assert.match(html, /<option value="TRANSFORMER" disabled/);
});

test('a knob change after a hand edit asks before rebuilding', () => {
  const p = page();
  const conv = p.state().layers.findIndex(l => l.name === 'Conv1d');
  p.act({ type: 'arg', i: conv, key: 'kernel_size', value: '5' });
  p.act({ type: 'knob', k: 'blocks', value: '3' });
  assert.match(p.app.innerHTML, /data-resolve="keep"/);
  assert.equal(p.state().knobs.blocks, 2);
  p.act({ type: 'resolve', choice: 'keep' });
  assert.equal(p.state().knobs.blocks, 3);
  assert.equal(p.state().layers.find(l => l.name === 'Conv1d').args.kernel_size, 5);
  assert.doesNotMatch(p.app.innerHTML, /data-resolve=/);
});

test('Preview Code shows the host\'s lines and turns into Encapsulate, which folds them away', () => {
  const p = page();
  assert.match(p.app.innerHTML, /data-preview="1"[^>]*>Preview Code<\/button>/);
  p.act({ type: 'preview' });
  assert.equal(p.sent.at(-1).type, 'preview');
  p.post({ type: 'preview', added: ['  - model_id: NEW_MODEL', '    architecture: CNN'] });
  assert.match(p.app.innerHTML, /<pre class="preview">  - model_id: NEW_MODEL\n    architecture: CNN<\/pre>/);
  assert.match(p.app.innerHTML, /data-preview="1"[^>]*>Encapsulate<\/button>/);
  const sent = p.sent.length;
  p.act({ type: 'preview' });
  assert.equal(p.sent.length, sent);
  assert.doesNotMatch(p.app.innerHTML, /class="preview"/);
  assert.match(p.app.innerHTML, /data-preview="1"[^>]*>Preview Code<\/button>/);
});

test('Add the Model is disabled with a reason while there are no layers or the check failed', () => {
  const p = page();
  p.act({ type: 'id', value: 'MY_CNN' });
  assert.match(p.app.innerHTML, /data-add="1" disabled title="TraceFlow&#?\w*;?s model check is still running|data-add="1" disabled title="TraceFlow's model check is still running/);
  const asked = p.sent.filter(m => m.type === 'check').at(-1);
  p.post({ type: 'check', result: checkOf(asked.layers, { error: 'mat1 and mat2 shapes cannot be multiplied', at: 13 }) });
  assert.match(p.app.innerHTML, /disabled title="the model check refuses it: mat1/);
  p.post({ type: 'check', result: checkOf(asked.layers) });
  assert.doesNotMatch(p.app.innerHTML, /data-add="1" disabled/);
  p.act({ type: 'add' });
  assert.equal(p.sent.at(-1).type, 'add-model');
  assert.equal(p.sent.at(-1).model.model_id, 'MY_CNN');
  while (p.state().layers.length) p.act({ type: 'remove', i: 0 });
  assert.match(p.app.innerHTML, /disabled title="the design has no layers"/);
});

test('Visualise starts from the model and stages its changes as an arrange of that model', () => {
  const all = models();
  const p = page({ start: { model: all.find(m => m.model_id === 'ASCAD_CNNbase') } });
  assert.equal(p.state().modelId, 'ASCAD_CNNbase');
  assert.equal(p.state().custom, false);
  p.act({ type: 'move', i: 1, dir: 1 });                      // BatchNorm below Pooling: no argument changes
  const staged = p.sent.filter(m => m.type === 'stage').at(-1);
  assert.equal(staged.model_id, 'ASCAD_CNNbase');
  assert.deepEqual(staged.arrange.slice(0, 3), [{ from: 0 }, { from: 2 }, { from: 1 }]);
  p.act({ type: 'move', i: 0, dir: 1 });                      // Conv1d below Pooling: still 64 channels after it
  assert.deepEqual(p.sent.filter(m => m.type === 'stage').at(-1).arrange.slice(0, 3), [{ from: 2 }, { from: 0 }, { from: 1 }]);
});


test('removing a skeleton layer counts as a hand edit: the next knob change asks', () => {
  const p = page();
  const n = p.state().layers.length;
  p.act({ type: 'remove', i: p.state().layers.findIndex(l => l.slot === 'dense2') });
  p.act({ type: 'knob', k: 'kernel', value: '11' });
  assert.match(p.app.innerHTML, /data-resolve="keep"/);
  p.act({ type: 'resolve', choice: 'keep' });
  assert.equal(p.state().layers.length, n - 1);
});

test('typing the model_id updates its state without redrawing the page', () => {
  const p = page({}, { dom: true });
  const before = p.draws();
  p.act({ type: 'id', value: 'ASCAD_CNNbase' });
  assert.equal(p.draws(), before);
  assert.match(p.found['.idstate'].textContent, /already/);
  assert.equal(p.found['[data-add]'].disabled, true);
});

test('after a write, a visualised model whose layers now equal the page is re-based; one changed otherwise is reloaded', () => {
  const all = models();
  const base = all.find(m => m.model_id === 'ASCAD_CNNbase');
  const p = page({ start: { model: base } });
  p.act({ type: 'move', i: 1, dir: 1 });
  const written = { ...base, architecture_args: { ...base.architecture_args, layers: P.asData(p.state().layers) } };
  p.post({ type: 'models', models: all.map(m => (m.model_id === 'ASCAD_CNNbase' ? written : m)), ids: all.map(m => m.model_id) });
  assert.deepEqual(p.state().layers.map(l => l.from), p.state().layers.map((_, i) => i));
  assert.deepEqual(p.state().original, written.architecture_args.layers);
  const other = { ...base, architecture_args: { ...base.architecture_args, layers: base.architecture_args.layers.slice(0, -1).concat([{ Linear: { in_features: 4096, out_features: 9 } }]) } };
  p.post({ type: 'models', models: all.map(m => (m.model_id === 'ASCAD_CNNbase' ? other : m)), ids: all.map(m => m.model_id) });
  assert.deepEqual(p.state().original, other.architecture_args.layers);
  assert.match(p.app.innerHTML, /ASCAD_CNNbase changed in params.yaml/);
  const stage = p.sent.filter(m => m.type === 'stage').at(-1);
  assert.ok(Array.isArray(stage.base));
});

test('Reset under Visualise stages the model unchanged, so the panel drops the planner\'s changes', () => {
  const p = page({ start: { model: models().find(m => m.model_id === 'ASCAD_CNNbase') } });
  p.act({ type: 'move', i: 1, dir: 1 });
  p.act({ type: 'reset' });
  const stage = p.sent.filter(m => m.type === 'stage').at(-1);
  assert.deepEqual(stage.arrange, stage.base.map((_, k) => ({ from: k })));
});

test('checks are debounced: quick changes send one check, for the last design', () => {
  const timers = [];
  const p = page({}, { window: { setTimeout: (fn) => { timers.push(fn); return timers.length; }, clearTimeout: id => { timers[id - 1] = null; } } });
  const sentBefore = p.sent.filter(m => m.type === 'check').length;
  p.act({ type: 'knob', k: 'blocks', value: '3' });
  p.act({ type: 'knob', k: 'blocks', value: '2' });
  assert.equal(p.sent.filter(m => m.type === 'check').length, sentBefore);
  timers.filter(Boolean).forEach(fn => fn());
  const checks = p.sent.filter(m => m.type === 'check');
  assert.equal(checks.length, sentBefore + 1);
  assert.equal(checks.at(-1).layers.filter(l => 'Pooling' in l).length, 2);
});


test('the page has no patterns and no Visualising note; a + between layers opens the layer menu there', () => {
  const p = page({ start: { model: models().find(m => m.model_id === 'ASCAD_CNNbest') } });
  const html = p.app.innerHTML;
  assert.doesNotMatch(html, /data-pattern|Patterns|Visualising/);
  assert.doesNotMatch(html, /<select class="addlayer"/);
  assert.equal((html.match(/<button class="plus" data-plus="\d+"/g) || []).length, p.state().layers.length + 1);
  p.act({ type: 'plus', at: 2 });
  assert.match(p.app.innerHTML, /<select class="addlayer" data-insert="2"/);
  p.act({ type: 'insert', at: 2, name: 'Dropout' });
  assert.equal(p.state().layers[2].name, 'Dropout');
  assert.doesNotMatch(p.app.innerHTML, /<select class="addlayer"/);
});

test('the four buttons sit under the model_id in the left column, with no reason text beside them', () => {
  const html = page().app.innerHTML;
  const left = html.slice(html.indexOf('<div class="left">'), html.indexOf('<div class="stack">'));
  assert.ok(left.indexOf('class="idstate') < left.indexOf('data-add="1"'));
  for (const b of ['data-add="1"', 'data-preview="1"', 'data-reset="1"', 'data-cancel="1"']) assert.ok(left.includes(b), b);
  assert.doesNotMatch(html, /<small class="note">type the model_id/);
  assert.doesNotMatch(html, /class="bottom"/);
});

test('every button and menu says what it does on hover', () => {
  const p = page();
  p.act({ type: 'open', i: 0 });
  const html = p.app.innerHTML;
  const untitled = [...html.matchAll(/<(button|select)\b[^>]*>/g)].map(m => m[0]).filter(t => !/ title="[^"]+"/.test(t));
  assert.deepEqual(untitled, []);
});

test('free numbers are typed in boxes, steps and words chosen from menus; a bad number is refused with why', () => {
  const p = page();
  assert.match(p.app.innerHTML, /<select data-knob="blocks"[^>]*>.*<option value="2" selected title="[^"]+">2 \(length 175\)<\/option>/);
  assert.match(p.app.innerHTML, /<select data-knob="filters"[^>]*>.*<option value="1024" title="[^"]+">1024<\/option>/);
  assert.match(p.app.innerHTML, /<input type="number" data-knob="kernel"[^>]*value="3"/);
  assert.match(p.app.innerHTML, /<select data-knob="limit"[^>]*><option value="" selected title="no cap: the filters keep doubling \(or halving\)">none \(32, 64\)<\/option><option value="64" title="[^"]+">64 \(32, 64\)<\/option>/);
  p.act({ type: 'knob', k: 'limit', value: '64' });
  assert.equal(p.state().knobs.limit, 64);
  p.act({ type: 'knob', k: 'limit', value: '' });
  assert.equal(p.state().knobs.limit, null);
  assert.match(p.app.innerHTML, /<div class="chips"[^>]*><span class="chip chip-conv"/, 'the Block Pattern\'s chips');
  assert.doesNotMatch(p.app.innerHTML, /data-knob="(pool|poolSize|activation|actAt|denseActivation)"/);
  assert.doesNotMatch(p.app.innerHTML.split('class="stack"')[0], /up to|per block, up|ASCAD_|halves the length/);
  p.act({ type: 'knob', k: 'filters', value: '16' });
  assert.equal(p.state().knobs.filters, 16);
  assert.equal(p.state().layers[0].args.out_channels, 16);
  p.act({ type: 'knob', k: 'kernel', value: 'abc' });
  assert.equal(p.state().knobs.kernel, 3);
  assert.match(p.app.innerHTML, /class="knoberr"[^>]*>a whole number/);
  p.act({ type: 'knob', k: 'kernel', value: '5' });
  assert.equal(p.state().knobs.kernel, 5);
  assert.equal(p.state().layers[0].args.padding, 2);
  assert.doesNotMatch(p.app.innerHTML, /class="knoberr"/);
  p.act({ type: 'kind', value: 'MLP' });
  p.act({ type: 'dropout', i: 1, value: '0.3' });                           // Dropout per layer: after #1
  assert.equal(p.state().layers[2].name, 'Dropout');
  assert.equal(p.state().layers[2].args.p, 0.3);
  assert.equal(p.state().knobs.dropout, undefined, 'not a setting any more');
});

test('Dropout per layer: a row per layer a Dropout can follow; a value adds a Dropout after it, another sets its p, none takes it away', () => {
  const p = page();
  const asked = p.sent.find(m => m.type === 'check');
  p.post({ type: 'check', result: checkOf(asked.layers) });
  assert.match(p.app.innerHTML, /<button class="acthead" data-dropfold="1"[^>]*>▸ Dropout per layer <small>none<\/small><\/button>/);
  assert.doesNotMatch(p.app.innerHTML, /data-knob="dropout"/, 'no Dropout setting');
  const n = p.state().layers.length;
  const pool = p.state().layers.findIndex(l => l.name === 'Pooling');
  p.act({ type: 'dropout', i: pool, value: '0.2' });
  assert.deepEqual([p.state().layers.length, p.state().layers[pool + 1].name, p.state().layers[pool + 1].args.p], [n + 1, 'Dropout', 0.2]);
  p.act({ type: 'dropout', i: pool, value: '0.5' });                       // its p, no second Dropout
  assert.deepEqual([p.state().layers.length, p.state().layers[pool + 1].args.p], [n + 1, 0.5]);
  p.state().dropOpen = true;
  p.act({ type: 'open', i: null });
  assert.match(p.app.innerHTML, new RegExp(`<label class="actrow own"[^>]*><span>#${pool} Pooling<\/span><select data-layerdrop="${pool}"[^>]*><option value="">none<\/option>.*<option value="0\.5" selected>0\.5<\/option>`));
  assert.doesNotMatch(p.app.innerHTML, new RegExp(`data-layerdrop="${n}"`), 'no row for the output layer');
  p.act({ type: 'knob', k: 'units', value: '256' });                        // a setting changes: the Dropout stays after its layer
  p.act({ type: 'resolve', choice: 'keep' });
  const at = p.state().layers.findIndex(l => l.name === 'Dropout');
  assert.equal(p.state().layers[at - 1].name, 'Pooling');
  p.act({ type: 'dropout', i: at - 1, value: '' });
  assert.ok(!p.state().layers.some(l => l.name === 'Dropout'), 'none: taken away');
});

test('a model with Dropout layers reads its settings; each Dropout stays after its layer (not custom)', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const m = JSON.parse(JSON.stringify(base));
  const L = m.architecture_args.layers;
  L.splice(3, 0, { Dropout: { p: 0.2 } });                                  // after block 1's pooling
  L.splice(L.findIndex(l => l.Flatten) + 2, 0, { Dropout: { p: 0.5 } });    // after the first dense layer
  const r = P.fromModel(m, FORMS);
  assert.equal(r.custom, false);
  assert.equal(r.knobs.blocks, 4);
  assert.deepEqual(asData(P.chain(r.layers)), L, 'the file\'s layers, in order');
  assert.deepEqual(r.layers.filter(l => l.name === 'Dropout').map(l => l.from), [3, L.findIndex(l => l.Flatten) + 2], 'each its place in the file');
  assert.deepEqual(P.arrangeOf(L, r.layers), L.map((_, k) => ({ from: k })), 'nothing changed');
});

test('the page keeps its design in the tab state, and a reloaded tab comes back with it', () => {
  const p = page({ file: '/w/params.yaml' });
  p.act({ type: 'plus', at: 1 });
  p.act({ type: 'insert', at: 1, name: 'Dropout' });
  p.act({ type: 'id', value: 'MINE' });
  const kept = p.saved();
  assert.equal(kept.project, 'DLSCA_ASCAD');
  assert.equal(kept.file, '/w/params.yaml');
  assert.equal(kept.design.layers[1].name, 'Dropout');
  const again = page({ file: '/w/params.yaml', restore: true }, { saved: kept });
  assert.equal(again.state().layers[1].name, 'Dropout');
  assert.equal(again.state().id, 'MINE');
  assert.equal(again.state().layers.length, p.state().layers.length);
  const fresh = page({ file: '/w/params.yaml' }, { saved: kept });            // a new Plan a Model: the clean skeleton
  assert.notEqual(fresh.state().layers[1].name, 'Dropout');
});

test('a reloaded Visualise tab stages its changes again, or shows the model as the file has it now', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const p = page({ start: { model: base }, file: '/w/params.yaml' });
  p.act({ type: 'remove', i: 1 });
  const kept = p.saved();
  assert.equal(kept.modelId, 'ASCAD_CNNbase');
  const again = page({ file: '/w/params.yaml', restore: true }, { saved: kept });
  assert.equal(again.state().modelId, 'ASCAD_CNNbase');
  assert.equal(again.state().layers.length, p.state().layers.length);
  const stage = again.sent.filter(m => m.type === 'stage').pop();
  assert.ok(stage, 'the panel lost its staged edits with the reload: the page stages them again');
  assert.deepEqual(stage.base, base.architecture_args.layers);
  const changedModels = models().map(m => (m.model_id === 'ASCAD_CNNbase'
    ? { ...m, architecture_args: { ...m.architecture_args, layers: m.architecture_args.layers.slice(0, -1) } } : m));
  const moved = page({ file: '/w/params.yaml', restore: true, models: changedModels }, { saved: kept });
  assert.equal(moved.state().layers.length, base.architecture_args.layers.length - 1);
  assert.match(moved.state().message, /changed in params\.yaml/);
});

test('layers ticked: Select all, Delete N layers, and Edit N layers sets one argument on each as a hand edit', () => {
  const p = page();
  const names = () => p.state().layers.map(l => l.name);
  assert.match(p.app.innerHTML, /<div class="selbar">.*<small>0 selected<\/small><button class="danger" data-delsel="1" disabled[^>]*>Delete 0 layers<\/button><button data-editsel="1" disabled/);
  const n = p.state().layers.length;
  assert.equal((p.app.innerHTML.match(/<input type="checkbox" class="sel" data-sel="\d+"/g) || []).length, n);
  // the two Conv1d (b1, b2): one argument form for both, kernel_size the same, set on both at once
  const convs = names().map((x, i) => (x === 'Conv1d' ? i : -1)).filter(i => i >= 0);
  for (const i of convs) p.act({ type: 'select', i, on: true });
  assert.match(p.app.innerHTML, /<small>2 selected<\/small><button class="danger" data-delsel="1" title="remove the 2 layers selected">Delete 2 layers/);
  p.act({ type: 'editSelected' });
  assert.match(p.app.innerHTML, /<div class="args bulk">.*<input data-argsel="kernel_size" value="3"/);
  p.act({ type: 'argSelected', key: 'kernel_size', value: '5' });
  for (const i of convs) {
    assert.equal(p.state().layers[i].args.kernel_size, 5);
    assert.ok(p.state().layers[i].handArgs.includes('kernel_size'));
  }
  // a mixed value shows (mixed); picking (mixed) changes nothing
  p.act({ type: 'arg', i: convs[0], key: 'stride', value: '2' });
  assert.match(p.app.innerHTML, /<input data-argsel="stride" value="" placeholder="\(mixed\)"/);
  // mixed types: only the arguments they share (Conv1d and Linear: activation, a menu of the options both take)
  p.act({ type: 'select', i: names().indexOf('Linear'), on: true });
  const bulk = p.app.innerHTML.match(/<div class="args bulk">(.*?)<\/div>/)[1];
  assert.doesNotMatch(bulk, /kernel_size|in_features/);
  assert.match(bulk, /<select data-argsel="activation"/);
  // a move keeps the selection on its layers; Delete removes them, from the last, and clears it
  p.act({ type: 'move', i: convs[0], dir: 1 });
  assert.deepEqual([...p.state().selected], [convs[0] + 1, convs[1], names().indexOf('Linear')].sort((a, b) => a - b));
  p.act({ type: 'deleteSelected' });
  assert.equal(p.state().layers.length, n - 3);
  assert.ok(!names().includes('Conv1d'));
  assert.deepEqual([...p.state().selected], []);
  // Select all ticks every card; a skeleton change clears the selection
  p.act({ type: 'selectAll', on: true });
  assert.equal(p.state().selected.length, n - 3);
  assert.match(p.app.innerHTML, /<input type="checkbox" data-selall="1" checked/);
  p.act({ type: 'reset' });
  assert.deepEqual([...p.state().selected], []);
});

// ── the trace lines ─────────────────────────────────────────────────────────────────────────────────────
const ENTRIES = [{ name: 'ascad', type: 'ASCADf', attack: 10000, length: 700, why: '' },
  { name: 'far', type: 'AES_RD', attack: 50, length: 3500, why: 'not downloaded on this machine yet' },
  { name: 'raw', type: 'ASCADf', attack: 10000, length: 700, why: '' }];
const traceOf = (n, extra = {}) => ({ weights: 'untrained', error: null, at: null, source: { dataset: 'ascad', type: 'ASCADf', trace: 0, desync_offset: 0 },
  input: { values: [0, 1, -1, 2], length: 700 }, top: [[3, 1.5], [7, 1.2]],
  layers: Array.from({ length: n }, (_, i) => ({ index: i, type: 'X', shape: [8, 100], kind: 'channels', values: [0, i + 1, 0], lo: 0, hi: i + 1 })), ...extra });

test('trace lines: hidden until Observe Trace; then the page asks for its sources, takes the first readable data entry, and asks for the trace with the check', () => {
  const p = page();
  assert.ok(!p.sent.some(m => m.type === 'sources' || m.type === 'trace'), 'nothing traced while hidden');
  assert.match(p.app.innerHTML, /<button class="observe" data-observe="1" title="[^"]+">Observe Trace<\/button>/);
  assert.doesNotMatch(p.app.innerHTML, /data-dataset=|data-traceindex=|data-weights=/);
  p.act({ type: 'observe' });
  assert.deepEqual({ ...p.sent.find(m => m.type === 'sources') }, { type: 'sources', model_id: null });
  p.sent.length = 0;
  p.post({ type: 'sources', entries: ENTRIES, runs: [], error: '' });
  assert.equal(p.state().dataset, 'ascad');
  const check = p.sent.find(m => m.type === 'check');
  const trace = p.sent.find(m => m.type === 'trace');
  assert.equal(check.dataset, 'ascad');
  assert.deepEqual([trace.dataset, trace.trace, trace.architecture], ['ascad', 0, 'CNN']);
  assert.deepEqual(trace.layers, check.layers);
  // the data menu: each entry with its type; one that cannot be read is disabled with why; the weights: untrained only
  assert.match(p.app.innerHTML, /<option value="far" disabled title="not downloaded on this machine yet">far · AES_RD \(not available\)<\/option>/);
  assert.match(p.app.innerHTML, /<select data-weights="1" disabled title="Visualise a saved model to see its trained runs">/);
  assert.match(p.app.innerHTML, /tracing…/);
  assert.match(p.app.innerHTML, />Hide Trace<\/button>/);
  p.post({ type: 'trace', result: traceOf(p.state().layers.length, { source: { dataset: 'ascad', type: 'ASCADf', trace: 0, row: 0, desync_offset: 0,
    file: 'data/ASCAD.h5', group: 'Attack_traces/traces', samples: [0, 700], normalization: 'none' } }) });
  assert.match(p.app.innerHTML, /title="data\/ASCAD\.h5 · Attack_traces\/traces row 0 · samples 0 to 699 · desync offset 0 · normalization none"/);
  p.sent.length = 0;
  p.act({ type: 'observe' });                                            // Hide Trace: the lines and settings go, nothing more is traced
  assert.doesNotMatch(p.app.innerHTML, /<svg class="trace|data-dataset=/);
  p.act({ type: 'knob', k: 'kernel', value: '5' });
  assert.ok(!p.sent.some(m => m.type === 'trace'));
  p.act({ type: 'observe' });                                            // again: the sources known, the trace at once
  assert.ok(!p.sent.some(m => m.type === 'sources'));
  assert.equal(p.sent.filter(m => m.type === 'trace').length, 1);
});

test('trace lines: the trace and one line per layer, drawn top to bottom once when they arrive (at once with reduced motion)', () => {
  const p = page();
  p.act({ type: 'observe' });
  p.post({ type: 'sources', entries: ENTRIES, runs: [], error: '' });
  const n = p.state().layers.length;
  p.post({ type: 'trace', result: traceOf(n) });
  const html = p.app.innerHTML;
  assert.equal((html.match(/<svg class="trace g-/g) || []).length, n + 1, 'the trace and every layer');
  assert.match(html, /<svg class="trace g-input"[^>]*>(?:(?!<\/svg>)[^])*begin="0\.00s"/);
  assert.match(html, new RegExp(`begin="${(n * 0.12).toFixed(2)}s"`), 'the last layer last');
  assert.match(html, /untrained weights · trace 0 of ascad \(ASCADf\)/);
  assert.match(html, /top classes 3, 7/);
  p.act({ type: 'open', i: 0 });                                         // a later redraw: the lines stand, not drawn again
  assert.doesNotMatch(p.app.innerHTML, /<animate /);
  assert.equal((p.app.innerHTML.match(/<svg class="trace g-/g) || []).length, n + 1);
  p.act({ type: 'knob', k: 'kernel', value: '5' });                      // a change of the design: its lines are gone until the new trace
  assert.equal((p.app.innerHTML.match(/<svg class="trace g-/g) || []).length, 0);
  const calm = page({}, { window: { matchMedia: q => ({ matches: /reduce/.test(q) }) } });
  calm.act({ type: 'observe' });
  calm.post({ type: 'sources', entries: ENTRIES, runs: [], error: '' });
  calm.post({ type: 'trace', result: traceOf(calm.state().layers.length) });
  assert.doesNotMatch(calm.app.innerHTML, /<animate /);
  assert.match(calm.app.innerHTML, /<svg class="trace g-input"/);
});

test('trace lines: the trace number takes a whole number inside the attack set; a data entry\'s length is the design\'s input', () => {
  const p = page();
  p.act({ type: 'observe' });
  p.post({ type: 'sources', entries: [{ ...ENTRIES[0], attack: 20 }, { name: 'long', type: 'ASCADf', attack: 5, length: 1400, why: '' }], runs: [], error: '' });
  p.sent.length = 0;
  p.act({ type: 'traceIndex', value: '20' });
  assert.match(p.app.innerHTML, /class="knoberr">a whole number from 0 to 19</);
  assert.ok(!p.sent.some(m => m.type === 'trace'));
  p.act({ type: 'traceIndex', value: '7' });
  assert.equal(p.sent.find(m => m.type === 'trace').trace, 7);
  p.act({ type: 'dataset', value: 'long' });
  assert.deepEqual([...p.state().input], [1, 1400]);
});

test('trace lines: a trained run of the model shows its own layers read-only, its data entry fixed; Back to the design', () => {
  const all = models();
  const base = all.find(m => m.model_id === 'ASCAD_CNNbase');
  const p = page({ start: { model: base } });
  p.act({ type: 'observe' });
  assert.deepEqual({ ...p.sent.find(m => m.type === 'sources') }, { type: 'sources', model_id: 'ASCAD_CNNbase' });
  const runLayers = [{ Flatten: {} }, { Linear: { in_features: 700, out_features: 256 } }];
  p.post({ type: 'sources', entries: ENTRIES, error: '', runs: [
    { name: 'tulip_hold', env: 'dev', framework: 'keras', dataset: 'ascad', type: 'ASCADf', architecture: 'MLP', layers: runLayers, why: '' },
    { name: 'broken', why: 'Training snapshot not found' }] });
  assert.match(p.app.innerHTML, /<option value="broken" disabled title="Training snapshot not found">/);
  const design = p.state().layers.length;
  p.sent.length = 0;
  p.act({ type: 'weights', value: 'tulip_hold' });
  assert.deepEqual({ ...p.sent.find(m => m.type === 'trace') }, { type: 'trace', run: 'tulip_hold', trace: 0 });
  assert.ok(!p.sent.some(m => m.type === 'stage'), 'nothing staged: the run is not an edit of the model');
  p.post({ type: 'trace', result: traceOf(2, { weights: 'trained', checkpoint: { policy: 'final_epoch', epoch: 200 }, label: 214, label_rank: 158,
    source: { dataset: 'ascad', type: 'ASCADf', trace: 0 } }) });
  const html = p.app.innerHTML;
  assert.equal((html.match(/<div class="card g-[^"]* ro">/g) || []).length, 2, 'the run\'s two layers');
  assert.doesNotMatch(html, /data-remove=|data-plus=|data-sel=|data-move=/);
  assert.match(html, /Trained: tulip_hold/);
  assert.match(html, /<select data-dataset="1" disabled title="fixed: the data entry tulip_hold trained on \(its snapshot\)"><option>ascad · ASCADf<\/option>/);
  assert.match(html, /tulip_hold's weights \(final_epoch, epoch 200\) · trace 0 of its test\.h5/);
  assert.match(html, /true class 214: rank 158/);
  const untitled = [...html.matchAll(/<(button|select)\b[^>]*>/g)].map(m => m[0]).filter(t => !/ title="[^"]+"/.test(t));
  assert.deepEqual(untitled, []);
  p.act({ type: 'knob', k: 'kernel', value: '5' });
  assert.match(p.app.innerHTML, /refused: these are tulip_hold's layers as it trained/);
  assert.equal(p.state().layers.length, design, 'the design is untouched');
  p.sent.length = 0;
  p.act({ type: 'weights', value: '' });
  assert.equal(p.state().trained, null);
  assert.doesNotMatch(p.app.innerHTML, / ro">/);
});

test('trace lines: a range of traces: From and To, the faint note, the mean with its band, and the true class\'s median rank', () => {
  const p = page();
  p.act({ type: 'observe' });
  p.post({ type: 'sources', entries: ENTRIES, runs: [], error: '' });
  assert.match(p.app.innerHTML, /<select data-tracemode="1" title="[^"]+"><option value="single" selected>single trace<\/option><option value="range">range of traces<\/option>/);
  assert.doesNotMatch(p.app.innerHTML, /class="faint"/);
  p.sent.length = 0;
  p.act({ type: 'traceMode', value: 'range' });
  assert.match(p.app.innerHTML, /<small class="faint">Each layer's line becomes the average over the range, with a shaded band from the lowest to the highest value at each point\.<\/small>/);
  assert.match(p.app.innerHTML, /data-traceindex="1"[^>]*value="0"[^]*<span>to<\/span><input type="number" data-traceto="1"[^>]*value="99"/);
  assert.deepEqual([p.sent.at(-1).trace, p.sent.at(-1).to], [0, 99]);
  p.act({ type: 'traceTo', value: '1000' });
  assert.match(p.app.innerHTML, /class="knoberr">a range takes at most 1000 traces</);
  p.act({ type: 'traceIndex', value: '200' });
  assert.match(p.app.innerHTML, /class="knoberr">the range ends before it starts</);
  p.act({ type: 'traceTo', value: '250' });
  p.act({ type: 'traceIndex', value: '200' });
  assert.deepEqual([p.sent.at(-1).trace, p.sent.at(-1).to], [200, 250]);
  const n = p.state().layers.length;
  const band = l => ({ ...l, band: [l.values.map(v => v - 1), l.values.map(v => v + 1)] });
  const result = traceOf(n, { count: 51, source: { dataset: 'ascad', type: 'ASCADf', trace: 200, to: 250, row: 200, row_to: 250, desync: 49,
    file: 'raw.h5', group: 'traces', samples: [45400, 46100], normalization: 'none' } });
  result.layers = result.layers.map(band);
  result.input = band(result.input);
  delete result.top;
  p.post({ type: 'trace', result });
  const html = p.app.innerHTML;
  assert.equal((html.match(/<polygon class="band"/g) || []).length, n + 1, 'a band under the trace and every layer');
  assert.match(html, /untrained weights · traces 200–250 \(mean and band\) of ascad/);
  assert.match(html, /title="raw\.h5 · traces rows 200 to 250 · samples 45400 to 46099 · a desync offset of 0 to 49 per trace · normalization none"/);
  assert.match(html, /51 traces · 700 samples/);
  p.post({ type: 'trace', result: { ...result, ranks: { median: 130, best: 1, worst: 256, count: 51 } } });
  assert.match(p.app.innerHTML, /true class: median rank 130 over 51 single traces \(best 1, worst 256\)/);
  p.act({ type: 'traceMode', value: 'single' });
  assert.equal(p.sent.at(-1).to, undefined);
});

test('trace lines: faint notes say what the desync offset and the classes are, as points with a bold label, with this trace\'s values', () => {
  const p = page();
  p.act({ type: 'observe' });
  p.post({ type: 'sources', entries: ENTRIES, runs: [], error: '' });
  const n = p.state().layers.length;
  const target = { leakage_model: 'ID', attack_byte: 2, target: 'sbox', classes: 256, expression: 'Sbox[p[2] ^ k[2]]', from: 'params.yaml', swept: {} };
  p.post({ type: 'trace', result: traceOf(n, { target, top: [[73, 1], [202, 0.9], [86, 0.8], [225, 0.7], [175, 0.6]],
    source: { dataset: 'ascad_desync50_raw', type: 'ASCADf', trace: 0, row: 50000, samples: [45421, 46121], desync_offset: 21, desync: 49, length: 700 } }) });
  // each note: a list of points, "label: text" (the label bold)
  const notes = () => [...p.app.innerHTML.matchAll(/<ul class="faint tracefaint">(.*?)<\/ul>/g)]
    .map(u => [...u[1].matchAll(/<li><b>([^<]*):<\/b> ([^<]*)<\/li>/g)].map(m => `${m[1]}: ${m[2]}`.replace(/&#39;/g, "'")));
  assert.deepEqual(notes(), [[
    'Desync: ascad_desync50_raw sets attack_desync: 49, so each trace\'s 700-sample window starts a random 0 to 49 samples late, imitating misaligned traces (a jitter countermeasure) the network has to cope with.',
    'Desync offset 21: trace 0 starts 21 samples late: samples 45421 to 46120 of row 50000, instead of 45400 to 46099.',
    'Drawn here: from the run\'s seed and the trace number; a run making its own test.h5 draws and stores its own.',
  ], [
    'Outputs 256: the last layer\'s, one per class of the ID leakage model (256 classes); out_features: auto takes that count.',
    'Settings: sca.leakage_model ID, sca.attack_byte 2, from params.yaml.',
    'Classes: each class is one possible value of Sbox[p[2] ^ k[2]], not a key byte.',
    'Top classes 73, 202, 86, 225, 175: the 5 values these weights score highest for trace 0, best first; the weights are untrained (random), so these guesses mean nothing yet.',
  ]]);
  // a sweep names its first value; a last layer that does not give the class count says so
  p.post({ type: 'trace', result: traceOf(n, { target: { ...target, leakage_model: 'HW', classes: 9, expression: 'HW(Sbox[p[2] ^ k[2]])',
    swept: { leakage_model: ['HW', 'ID'] } }, outputs: 256, top: [[3, 1]], source: { dataset: 'ascad', trace: 0, desync: 0 } }) });
  assert.deepEqual(notes()[0].slice(0, 2), ['Outputs 256: the last layer\'s, but the HW leakage model has 9 classes: the outputs do not match the labels (out_features: auto takes the leakage model\'s count).',
    'Settings: sca.leakage_model HW (the first of the sweep HW, ID), sca.attack_byte 2, from params.yaml.']);
  // the trained view: the true class, the bytes it is made of, its rank against chance
  p.post({ type: 'trace', result: traceOf(n, { target: { ...target, from: 'tulip_hold\'s training snapshot' }, top: [[194, 1]], label: 214, label_rank: 158,
    bytes: { plaintext: 170, key: 224 }, source: { dataset: 'ascad', type: 'ASCADf', trace: 0, row: 0, desync: 0 } }) });
  p.state().trained = 'tulip_hold';
  p.post({ type: 'check', result: null });
  assert.equal(notes().length, 1, 'no desync note without desync');
  assert.deepEqual(notes()[0].slice(1), ['Settings: sca.leakage_model ID, sca.attack_byte 2, from tulip_hold\'s training snapshot.',
    'Classes: each class is one possible value of Sbox[p[2] ^ k[2]], not a key byte.',
    'Top classes 194: the 1 value tulip_hold scores highest for trace 0, best first.',
    'True class 214: the real value for this trace, from its plaintext byte p[2] = 170 and the key byte k[2] = 224.',
    'Rank 158 of 256: a random guess averages 128.5, so the network did worse than chance on this one trace.',
    'One trace: leaks little; key recovery combines many traces (the run\'s guessing entropy).']);
  p.post({ type: 'trace', result: traceOf(n, { target, count: 20, ranks: { median: 130, best: 1, worst: 256, count: 20 }, top: undefined,
    source: { dataset: 'ascad_desync50_raw', trace: 0, to: 19, desync: 49, length: 700 } }) });
  assert.equal(notes()[0][1], 'Each trace: has its own shift; the mean and band are over the 20 shifted windows.');
  assert.deepEqual(notes()[1].slice(3), ['Median rank 130 of 256: the true class over 20 traces, each ranked alone (best 1, worst 256); a random guess averages 128.5.',
    'Not guessing entropy: that combines the evidence of many traces (the run\'s rank curves).']);
});

test('Show Key Recovery: a trained run\'s recorded attacks as a table: its folds, its test (or why not), its external tests', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbest');
  const p = page({ start: { model: base } });
  p.act({ type: 'observe' });
  p.post({ type: 'sources', entries: ENTRIES, error: '', runs: [{ name: 'angle_ox', env: 'dev', framework: 'pytorch', dataset: 'ascad', type: 'ASCADf',
    architecture: 'CNN', layers: [{ Flatten: {} }], why: '' }] });
  assert.doesNotMatch(p.app.innerHTML, /data-recovery=/, 'only in a trained view');
  p.act({ type: 'weights', value: 'angle_ox' });
  assert.match(p.app.innerHTML, /<button data-recovery="1" title="[^"]+">Show Key Recovery<\/button>/);
  p.sent.length = 0;
  p.act({ type: 'recovery' });
  assert.deepEqual({ ...p.sent.find(m => m.type === 'recovery') }, { type: 'recovery', run: 'angle_ox' });
  assert.match(p.app.innerHTML, /reading the key recovery…/);
  p.post({ type: 'recovery', run: 'angle_ox', result: { run: 'angle_ox', true_key: '224 (0xE0)', cv: true, error: null,
    training: { cv: true, n_folds: 10, strategy: 'k_fold', profiling: 50000, test: 10000,
      attack: { validation: { attack_times: 1, shuffle: false, attack_distribution: 'full_population' }, test: { attack_times: 0, shuffle: false, attack_distribution: 'full_population' } } },
    rules: { keyspace: 256, recovery_rank: 1, stable_required: true },
    folds: [{ fold: 1, key_recovered: true, recovered_key: '224 (0xE0)', recovered_key_rank: 1, stable_rank_1: '@2125 /5000', traces: 5000 },
      { fold: 5, key_recovered: false, recovered_key: '120 (0x78)', recovered_key_rank: 27, stable_rank_1: 'not reached /5000', traces: 5000 }],
    test: { status: 'SKIPPED', key_recovered: 'not available: perform_attack.test.attack_times is 0', status_reason: null },
    external: [{ index: 1, status: 'DONE', key_recovered: true, recovered_key: '224 (0xE0)', recovered_key_rank: 1,
      settings: { perform_attack: { test: { attack_times: 100, shuffle: true, attack_distribution: 'full_population' }, attack_step: 1 } } },
    { index: 2, status: 'FAILED', status_reason: 'checkpoint missing or corrupt: x.pt', settings: { perform_attack: { test: { attack_times: 10, shuffle: false,
      attack_distribution: 'disjoint' }, attack_step: 5 } } }] } });
  const html = p.app.innerHTML;
  assert.match(html, /<b>Key recovery · angle_ox<\/b> · trained with 10-fold cross-validation \(k_fold\) · true key 224 \(0xE0\) · 1 of 2 folds recovered the key/);
  const points = () => [...p.app.innerHTML.matchAll(/<li><b>([^<]*):<\/b> ([^<]*)<\/li>/g)].map(m => `${m[1]}: ${m[2]}`.replace(/&#39;/g, "'"));
  assert.deepEqual(points(), [
    'Cross-validation: the 50,000 profiling traces in 10 folds: each fold\'s model trains on the other 9 and is attacked on its held-out fold (5,000 traces); ' +
      'then the final model is refit on all 50,000; its recorded test on the 10,000 test traces was not run (SKIPPED).',
    'Recovered: the true key reached rank 1 (key_recovery_reporting.recovery_rank) after the attack combined the evidence of all its traces, and stayed there to the last trace (require_stable_recovery: true).',
    'Top key guess: the key byte the attack ranked first; the true key when recovered.',
    'True key rank: where the true key ended among the 256 key guesses; rank 1 is recovered.',
    'Rank 1 from: the trace count from which the true key stays at rank 1, out of the attack\'s traces.',
    'Folds: each fold\'s model, trained without that fold, attacked on its held-out traces (perform_attack.validation: 1× in order, full_population); the final model is not attacked on them.',
    'Recorded test: the final model on its test traces (perform_attack.test: attack_times 0, not attacked); external tests use the settings on their rows (test_config.yaml).']);
  assert.match(html, /<tr><td>fold 1<\/td><td class="ok">✓ yes<\/td><td>224 \(0xE0\)<\/td><td>1<\/td><td>@2125 \/5000<\/td><\/tr>/);
  assert.match(html, /<tr><td>fold 5<\/td><td class="err">✗ no<\/td><td>120 \(0x78\)<\/td><td>27<\/td><td>not reached \/5000<\/td><\/tr>/);
  assert.match(html, /<tr><td>recorded test<\/td><td colspan="4">SKIPPED · not available: perform_attack\.test\.attack_times is 0<\/td><\/tr>/);
  assert.match(html, /<tr><td>external test 1 · 100× shuffled, full_population, step 1<\/td><td class="ok">✓ yes<\/td><td>224 \(0xE0\)<\/td><td>1<\/td>/);
  assert.match(html, /<tr><td>external test 2 · 10× in order, disjoint, step 5<\/td><td colspan="4">FAILED · checkpoint missing or corrupt: x\.pt<\/td><\/tr>/);
  p.post({ type: 'recovery', run: 'angle_ox', result: { run: 'angle_ox', true_key: '224 (0xE0)', cv: true, error: null, folds: [], external: [],
    training: { cv: false, profiling: 50000, validation: 2000, test: 8000,
      attack: { validation: { attack_times: 1, shuffle: false, attack_distribution: 'full_population' }, test: { attack_times: 100, shuffle: true, attack_distribution: 'full_population' } } },
    rules: { keyspace: 256, recovery_rank: 2, stable_required: false },
    test: { status: 'DONE', key_recovered: true, recovered_key: '224 (0xE0)', recovered_key_rank: 1, traces: 8000 } } });          // the test traces attacked
  assert.match(p.app.innerHTML, /<tr><td>recorded test<\/td><td class="ok">✓ yes<\/td><td>224 \(0xE0\)<\/td><td>1<\/td>/);
  assert.match(p.app.innerHTML, /angle_ox<\/b> · trained with a hold-out split \(train \/ validation \/ test\) · true key/);
  assert.deepEqual(points().slice(0, 2), [
    'Hold-out split: the model trains on the 50,000 profiling traces (train.h5), is checked on 2,000 validation traces (val.h5); it is attacked on its 8,000 test traces (100× shuffled, full_population).',
    'Recovered: the true key reached rank 2 or better (key_recovery_reporting.recovery_rank) after the attack combined the evidence of all its traces; reaching it once is enough (require_stable_recovery: false).']);
  assert.match(p.app.innerHTML, /<th>Rank ≤ 2 from<\/th>/);
  assert.match(points()[3], /among the 256 key guesses; rank 2 or better is recovered\./);
  assert.match(html, /<li><b>Rank 1 from:<\/b> the trace count from which the true key stays at rank 1/);
  assert.match(html, />Hide Key Recovery<\/button>/);
  p.act({ type: 'recovery' });
  assert.doesNotMatch(p.app.innerHTML, /class="recovery"/);
  p.act({ type: 'recovery' });
  p.act({ type: 'weights', value: '' });                                 // back to the design: no run, no key recovery
  assert.equal(p.state().showRecovery, false);
});

test('changesOf: a visualised model\'s changes against its layers in the file: added, removed, changed, moved', () => {
  const file = [{ Flatten: {} }, { Linear: { in_features: 700, out_features: 10 } }, { Dropout: { p: 0.5 } }];
  const from = file.map((l, i) => { const [name] = Object.keys(l); return { name, args: { ...l[name] }, from: i }; });
  assert.deepEqual({ ...P.changesOf(file, from) }, { added: 0, removed: 0, changed: 0, moved: 0, total: 0 });
  const edited = [from[0], { ...from[1], args: { ...from[1].args, out_features: 20 } }, { name: 'Linear', args: { in_features: 20, out_features: 'auto' } }];
  assert.deepEqual({ ...P.changesOf(file, edited) }, { added: 1, removed: 1, changed: 1, moved: 0, total: 3 });
  assert.deepEqual({ ...P.changesOf(file, [from[0], from[2], from[1]]) }, { added: 0, removed: 0, changed: 0, moved: 1, total: 1 });
  assert.equal(P.changesOf(null, from).total, 0);
});

test('Visualise: any change is a Configuration Change with OK / Cancel / Reset / Add the Model / Undo / Redo; reverted, the bar goes', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const p = page({ start: { model: base } });
  const layers = () => p.state().layers;
  const bar = () => (p.app.innerHTML.match(/<div class="pending config">(.*?)<\/div>/) || [])[1] || '';
  assert.equal(bar(), '', 'no changes: no bar');
  assert.doesNotMatch(p.app.innerHTML, /data-id="1"|data-add="1"/, 'Visualise: no model_id box nor Add the Model of its own');
  assert.match(p.app.innerHTML, /data-cancel="1"/, 'the page\'s own Cancel while unchanged');
  const i = layers().findIndex(l => l.name === 'Conv1d');
  p.act({ type: 'arg', i, key: 'kernel_size', value: '5' });
  assert.match(bar(), /<b>Configuration Change<\/b> · ASCAD_CNNbase · 1 change<\/span>/);
  assert.ok(p.app.innerHTML.indexOf('class="pending config"') < p.app.innerHTML.indexOf('data-kind="1"'), 'the bar heads the settings');
  // reverted by hand to what the file holds: the bar goes
  p.act({ type: 'arg', i, key: 'kernel_size', value: String(base.architecture_args.layers[i].Conv1d.kernel_size) });
  assert.equal(bar(), '', 'back at the saved layers: no bar');
  p.act({ type: 'arg', i, key: 'kernel_size', value: '5' });
  assert.match(bar(), /title="ASCAD_CNNbase's layers: 1 changed"/);
  assert.match(bar(), /<button data-applychanges="1" disabled title="TraceFlow's model check (is still running|has not answered yet)">OK<\/button>/);
  assert.match(bar(), /<button data-discardchanges="1" title="undo every change and close the window; ASCAD_CNNbase stays as it is">Cancel<\/button>/);
  assert.match(bar(), /<button data-reset="1" title="undo every change: ASCAD_CNNbase as it was when the window opened it; the window stays open \(Undo takes the Reset back\)">Reset<\/button>/);
  assert.match(bar(), /<button data-addask="1" title="asks a new model_id first, then adds the changed design under it; ASCAD_CNNbase stays unchanged">Add the Model<\/button>/);
  assert.match(bar(), /OK<\/button><button data-discardchanges[^]*Cancel<\/button><button data-reset[^]*Reset<\/button><button data-addask[^]*Add the Model<\/button><span class="steps"><button data-undo="1" title="undo the last change, one step back \(\d+ to undo; Ctrl\+Z\)">↶ Undo<\/button><button data-redo="1" disabled title="nothing to redo">↷ Redo<\/button><\/span>$/, 'the order asked for');
  assert.doesNotMatch(p.app.innerHTML, /data-cancel="1"/, 'the bar\'s Cancel and Reset stand in for the page\'s');
  p.post({ type: 'check', result: checkOf(P.asData(layers())) });
  assert.match(bar(), /<button data-applychanges="1" title="override ASCAD_CNNbase in params\.yaml with the changes">OK<\/button>/);
  p.sent.length = 0;
  p.act({ type: 'applyChanges' });
  const ok = p.sent.find(m => m.type === 'apply');
  assert.equal(ok.model_id, 'ASCAD_CNNbase');
  assert.deepEqual(JSON.parse(JSON.stringify(ok.base)), base.architecture_args.layers);
  assert.equal(JSON.parse(JSON.stringify(ok.arrange))[i].value.Conv1d.kernel_size, 5);
  // the write: the file holds the design; no bar, and Reset goes back to what the file holds now
  const written = { ...base, architecture_args: { ...base.architecture_args, layers: P.asData(layers()) } };
  p.post({ type: 'models', models: [written], ids: ['ASCAD_CNNbase'] });
  assert.equal(bar(), '');
  p.act({ type: 'reset' });
  assert.equal(layers()[i].args.kernel_size, 5, 'Reset keeps what was written');
  // Reset undoes unsaved changes; Cancel undoes them and closes the window
  p.act({ type: 'arg', i, key: 'kernel_size', value: '7' });
  p.act({ type: 'reset' });
  assert.equal(layers()[i].args.kernel_size, 5);
  assert.equal(bar(), '');
  p.act({ type: 'arg', i, key: 'kernel_size', value: '7' });
  p.sent.length = 0;
  p.act({ type: 'discardChanges' });
  assert.equal(layers()[i].args.kernel_size, 5);
  assert.ok(p.sent.some(m => m.type === 'stage'), 'the panel drops its staged copy');
  assert.equal(p.sent.at(-1).type, 'close');
});

test('Visualise: what params.yaml holds is not marked; only a difference from it is edited, with the bar', () => {
  const sig = models().find(m => m.model_id === 'ASCAD_CNNbase_sigmoid');
  const p = page({ start: { model: sig } });
  p.state().actOpen = true;
  p.ctx.window.TFPlannerPage.render();
  const html = () => p.app.innerHTML;
  const i = p.state().layers.findIndex(l => l.name === 'BatchNorm1d');
  assert.match(html(), new RegExp(`<label class="actrow"[^>]*><span>#${i} BatchNorm1d</span>`));
  assert.doesNotMatch(html(), /<small>(edited|own|changed)<\/small>|class="tag|class="pending config"/, 'as saved: nothing marked');
  p.act({ type: 'arg', i, key: 'activation', value: 'relu' });
  assert.match(html(), new RegExp(`<label class="actrow chg"[^>]*><span>#${i} BatchNorm1d <small>edited</small>`));
  assert.match(html(), /class="pending config"/);
  p.act({ type: 'arg', i, key: 'activation', value: 'sigmoid' });
  assert.doesNotMatch(html(), /<small>edited<\/small>|class="pending config"/);
});

test('Visualise: Undo and Redo step through the changes one by one; Reset is a step; the keys too, outside text boxes', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const p = page({ start: { model: base } });
  const k = () => p.state().layers.find(l => l.name === 'Conv1d').args.kernel_size;
  const i = p.state().layers.findIndex(l => l.name === 'Conv1d');
  const k0 = k();
  p.act({ type: 'arg', i, key: 'kernel_size', value: '5' });
  p.act({ type: 'arg', i, key: 'kernel_size', value: '7' });
  p.act({ type: 'arg', i, key: 'kernel_size', value: '9' });
  assert.equal(p.state().undo.length, 3);
  p.act({ type: 'undo' });
  assert.equal(k(), 7);
  p.act({ type: 'undo' });
  assert.equal(k(), 5);
  assert.match(p.app.innerHTML, /<button data-redo="1" title="redo the change undone, one step forward \(2 to redo/);
  p.act({ type: 'redo' });
  assert.equal(k(), 7);
  p.act({ type: 'reset' });
  assert.equal(k(), k0, 'Reset: the model as the window opened it');
  assert.doesNotMatch(p.app.innerHTML, /class="pending config"/);
  p.act({ type: 'undo' });
  assert.equal(k(), 7, 'Undo takes the Reset back');
  const key = (o, tagName = 'BODY') => p.docOn.keydown({ ctrlKey: true, target: { tagName }, preventDefault() {}, ...o });
  key({ key: 'z' });
  assert.equal(k(), 5);
  key({ key: 'Z', shiftKey: true });
  assert.equal(k(), 7);
  key({ key: 'z' }, 'INPUT');
  assert.equal(k(), 7, 'a text box keeps its own Ctrl+Z');
  key({ key: 'y' });
  assert.equal(k(), k0, 'Redo does the Reset again');
  p.act({ type: 'arg', i, key: 'kernel_size', value: '3' });
  assert.equal(p.state().redo.length, 0, 'a new change ends the redo steps');
});

test('Visualise: a setting changed over a model\'s own values applies at once (keeping them where they fit), no question; Undo takes it back', () => {
  const sig = models().find(m => m.model_id === 'ASCAD_CNNbase_sigmoid');
  const p = page({ start: { model: sig } });
  p.act({ type: 'knob', k: 'kernel', value: '5' });
  assert.equal(p.state().pending, null);
  assert.doesNotMatch(p.app.innerHTML, /Keep my changes where they fit|Rebuild from the settings/);
  const convs = p.state().layers.filter(l => l.name === 'Conv1d');
  assert.ok(convs.every(l => l.args.kernel_size === 5));
  assert.ok(p.state().layers.filter(l => l.name === 'BatchNorm1d').every(l => l.args.activation === 'sigmoid'), 'its own activations kept');
  assert.match(p.app.innerHTML, /<b>Configuration Change<\/b> · ASCAD_CNNbase_sigmoid · 4 changes/, 'the four Conv1d lines changed, each layer still its own line');
  assert.deepEqual(JSON.parse(JSON.stringify(P.arrangeOf(sig.architecture_args.layers, p.state().layers))).map(e => e.from), [...Array(16).keys()]);
  p.act({ type: 'undo' });
  assert.ok(p.state().layers.filter(l => l.name === 'Conv1d').every(l => l.args.kernel_size === 3));
  assert.doesNotMatch(p.app.innerHTML, /class="pending config"/);
  // Plan a Model still asks
  const q = page({ start: {} });
  q.act({ type: 'arg', i: 0, key: 'kernel_size', value: '9' });
  q.act({ type: 'knob', k: 'kernel', value: '5' });
  assert.match(q.app.innerHTML, /Keep my changes where they fit/);
});

test('Visualise: Preview Code asks the model\'s own entry under its model_id (as OK would write it), edited lines marked', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const p = page({ start: { model: base } });
  const i = p.state().layers.findIndex(l => l.name === 'Conv1d');
  p.act({ type: 'arg', i, key: 'kernel_size', value: '5' });
  p.sent.length = 0;
  p.act({ type: 'preview' });
  const ask = p.sent.find(m => m.type === 'preview');
  assert.equal(ask.model_id, 'ASCAD_CNNbase');
  assert.equal(ask.model, undefined, 'not an add under NEW_MODEL');
  assert.equal(JSON.parse(JSON.stringify(ask.arrange))[i].value.Conv1d.kernel_size, 5);
  p.post({ type: 'preview', added: ['  - model_id: "ASCAD_CNNbase"', '        - Conv1d: {kernel_size: 5}'], marked: [1] });
  assert.match(p.app.innerHTML, /<pre class="preview">  - model_id: &quot;ASCAD_CNNbase&quot;\n<span class="mk" title="edited: not in params.yaml yet">        - Conv1d: \{kernel_size: 5\}<\/span><\/pre>/);
});

test('Visualise → Add The Model: asks a model_id in the bar, adds the design as a new model, then shows it', () => {
  const base = models().find(m => m.model_id === 'ASCAD_CNNbase');
  const p = page({ start: { model: base } });
  const layers = () => p.state().layers;
  const i = layers().findIndex(l => l.name === 'Conv1d');
  p.act({ type: 'arg', i, key: 'kernel_size', value: '5' });
  p.post({ type: 'check', result: checkOf(P.asData(layers())) });
  assert.doesNotMatch(p.app.innerHTML, /class="addask"/, 'asked only on Add The Model');
  p.act({ type: 'addAsk', on: true });
  assert.match(p.app.innerHTML, /<div class="addask"><label class="knob"><span>new model_id<\/span><input data-id="1"/);
  assert.match(p.app.innerHTML, /<button class="primary" data-add="1" disabled[^>]*>Add<\/button><button data-addaskclose="1"/, 'no model_id yet: Add waits');
  p.act({ type: 'addAsk', on: false });
  assert.doesNotMatch(p.app.innerHTML, /class="addask"/, '× closes the ask');
  p.act({ type: 'addAsk', on: true });
  p.state().id = 'ASCAD_CNNbase';
  p.ctx.window.TFPlannerPage.render();
  assert.match(p.app.innerHTML, /data-add="1" disabled/, 'a model_id the file has: refused');
  p.state().id = 'ASCAD_CNNbase_k5';
  p.ctx.window.TFPlannerPage.render();
  p.sent.length = 0;
  p.act({ type: 'add' });
  const add = p.sent.find(m => m.type === 'add-model');
  assert.equal(add.model.model_id, 'ASCAD_CNNbase_k5');
  assert.equal(add.from.model_id, 'ASCAD_CNNbase', 'the opened model: the host drops its staged copy');
  assert.deepEqual(JSON.parse(JSON.stringify(add.from.base)), base.architecture_args.layers);
  assert.equal(JSON.parse(JSON.stringify(add.from.arrange))[i].value.Conv1d.kernel_size, 5);
  // written: the window shows the new model, unchanged; the original is as the file holds it
  const added = { ...JSON.parse(JSON.stringify(add.model)) };
  p.post({ type: 'saved', message: 'ASCAD_CNNbase_k5 added to params.yaml', ids: ['ASCAD_CNNbase', 'ASCAD_CNNbase_k5'] });
  p.post({ type: 'models', models: [base, added], ids: ['ASCAD_CNNbase', 'ASCAD_CNNbase_k5'] });
  assert.equal(p.state().modelId, 'ASCAD_CNNbase_k5');
  assert.equal(layers()[i].args.kernel_size, 5);
  assert.doesNotMatch(p.app.innerHTML, /class="pending config"/, 'no changes left');
  assert.match(p.app.innerHTML, /ASCAD_CNNbase_k5 added to params\.yaml; the window now shows it/);
  // refused: the window stays on the design
  p.act({ type: 'arg', i, key: 'kernel_size', value: '7' });
  p.act({ type: 'addAsk', on: true });
  p.state().id = 'X';
  p.act({ type: 'add' });
  p.post({ type: 'refused', message: 'refused: nothing written: x' });
  p.post({ type: 'models', models: [base, added, { ...added, model_id: 'X' }], ids: ['ASCAD_CNNbase', 'ASCAD_CNNbase_k5', 'X'] });
  assert.equal(p.state().modelId, 'ASCAD_CNNbase_k5', 'a refused add does not switch');
});

test('width suggestions: from the layers around it (inputs, classes, the geometric pyramid rule; 4 × and 2 × d_model), then the project\'s widths, then powers of two', () => {
  const knob = (type, k) => P.SKELETONS[type].knobs.find(x => x.k === k);
  const dense = P.widthSuggestions(knob('CNN', 'units'), { inFeatures: 11200, classes: 256, projectWidths: [200, 4096] });
  assert.deepEqual(dense.slice(0, 5).map(s => [s.v, s.label]), [
    [1693, 'geometric pyramid rule: √(11200 inputs × 256 classes)'], [11200, 'the inputs it receives (11200)'],
    [256, 'the classes the output gives (256)'], [200, 'a width this project\'s models use'], [4096, 'a width this project\'s models use']]);
  assert.ok(dense.some(s => s.v === 1024 && s.label === 'a power of two'));
  assert.equal(dense.filter(s => s.v === 256).length, 1, 'each width once, with its first reason');
  const ff = P.widthSuggestions(knob('TRANSFORMER', 'ff'), { knobs: { width: 64 } });
  assert.deepEqual(ff.slice(0, 2).map(s => s.v), [256, 128]);
  assert.equal(P.widthSuggestions(knob('CNN', 'kernel'), {}), null, 'other numbers get none');
  assert.ok(knob('MLP', 'units').suggest && knob('HYBRID', 'units').suggest && knob('TRANSFORMER', 'units').suggest);
  // the page: Dense width is still a typed box, with the suggestions as its list, computed from the design
  const p = page();
  const n = p.state().layers.length;
  p.post({ type: 'check', result: { error: null, at: null, params: 1, layers: Array.from({ length: n }, (_, i) => ({ index: i, type: 'X', out: i === n - 1 ? [256] : [8, 100] })) } });
  const html = p.app.innerHTML;
  const inputs = p.state().layers.find(l => l.slot === 'dense1').args.in_features;
  assert.match(html, /<input type="number" data-knob="units"[^>]*list="hint-units"[^>]*title="Dense width: out_features of each hidden dense layer; the suggestions give their reason\. Type a whole number or pick a suggestion, Enter applies it">/);
  assert.match(html, new RegExp(`<datalist id="hint-units"><option value="${Math.round(Math.sqrt(inputs * 256))}" label="geometric pyramid rule: √\\(${inputs} inputs × 256 classes\\)">`));
  assert.doesNotMatch(html, /list="hint-kernel"/);
});

test('the window follows the Download style: Layer cards draws the cards alone, Scaled Sparkline the bars alone (index, bar, arguments, the axis, the layer legend); the cards come back with Full Version', () => {
  const p = page({}, { dom: true });
  const asked = p.sent.find(m => m.type === 'check');
  const outs = { Conv1d: [32, 700], BatchNorm1d: [32, 700], Pooling: [32, 350], Flatten: [5600], Linear: [128] };
  p.post({ type: 'check', result: { error: null, at: null, params: 1, layers: asked.layers.map((l, i) => ({ index: i, type: Object.keys(l)[0], out: outs[Object.keys(l)[0]] })) } });
  const html = () => p.app.innerHTML;
  const change = (key, value) => listenerOf(p, 'change')({ target: { dataset: { dl: key }, type: 'select-one', value: String(value), checked: !!value } });
  const n = asked.layers.length;
  assert.equal((html().match(/<svg class="bar"/g) || []).length, n, 'Full Version: a bar beside every card');
  change('style', 'cards');
  assert.equal((html().match(/<div class="card g-/g) || []).length, n, 'Layer cards: every card');
  assert.ok(!/<svg class="bar"/.test(html()) && !/class="scalecap"/.test(html()), 'no bars, no axis');
  assert.match(html(), /<div class="row nobar">/);
  change('style', 'sparkline');
  assert.ok(!/<div class="card g-/.test(html()), 'Scaled Sparkline: no cards');
  assert.match(html(), /<p class="note sparknote"[^>]*>Scaled Sparkline · \d+ layers · output length \(log scale\) · thickness: channels<\/p>/);
  assert.equal((html().match(/<div class="sprow"><span class="spidx">#\d+<\/span><svg class="spbar"/g) || []).length, n, 'a row per layer: its index at the bar\'s start');
  assert.match(html(), /<svg class="scale spaxis"[^>]*>[^]*?<text[^>]*>5600<\/text>/, 'the axis with tick labels');
  assert.match(html(), /<svg class="spbar"[^>]*><title>#0 Conv1d: length 700 × 32 channels<\/title>[^]*?<rect class="g-convolution"/);
  assert.match(html(), /<small class="spargs" title="#0 Conv1d">[^<]*kernel_size/, 'the arguments outside the bars');
  assert.match(html(), /<div class="splegend right"><span class="g-convolution"><i class="swatch"><\/i>Conv1d ×\d+<\/span>/, 'the layer legend, top-right');
  assert.ok(html().indexOf('splegend') < html().indexOf('spaxisrow'), 'top: above the bars');
  change('sparkline.legend', 'bottom-left');
  assert.match(html(), /<div class="splegend left">/);
  assert.ok(html().indexOf('splegend') > html().indexOf('spaxisrow'), 'bottom: under the bars');
  change('sparkline.arguments', '');
  assert.ok(!/class="spargs"/.test(html()), 'arguments off');
  change('sparkline.ticks', '');
  assert.ok(!/spaxisrow/.test(html()) && !/<line class="grid"/.test(html()), 'tick labels off: no axis, no gridlines');
  change('sparkline.orientation', 'landscape');
  assert.match(html(), /<div class="spark landscape"><svg class="sparkland"/);
  assert.equal((html().match(/<text class="spidx"/g) || []).length, n, 'landscape: the index under each column');
  assert.equal((html().match(/<rect class="g-/g) || []).length, n, 'a bar per layer, growing up from the baseline');
  change('style', 'full');
  assert.equal((html().match(/<div class="card g-/g) || []).length, n, 'the cards come back');
  assert.equal((html().match(/<svg class="bar"/g) || []).length, n);
});

test('a project without a data entry: the Input is typed; no check is sent until it is; then the check carries the length and the sidebar shows it', () => {
  const p = page({ project: 'P', models: [], ids: [], input: null, entries: [], dataset: null, error: 'no data entry in params.yaml yet' }, { dom: true });
  assert.ok(!p.sent.some(m => m.type === 'check'), 'nothing to check against yet');
  assert.match(p.app.innerHTML, /<div class="stat"><span title="no data entry in params.yaml yet[^"]*">Input<\/span><b><input type="number" data-inputlength="1" min="1" step="1" value="" placeholder="trace length"/);
  assert.match(p.app.innerHTML, /<b><span class="err">input length needed<\/span><\/b>/);
  listenerOf(p, 'change')({ target: { dataset: { inputlength: '1' }, type: 'number', value: 'abc' } });
  assert.match(p.app.innerHTML, /<small class="knoberr">a whole number of samples, at least 1<\/small>/);
  assert.ok(!p.sent.some(m => m.type === 'check'));
  listenerOf(p, 'change')({ target: { dataset: { inputlength: '1' }, type: 'number', value: '1200' } });
  const asked = p.sent.find(m => m.type === 'check');
  assert.ok(asked && asked.input === 1200 && !asked.dataset, 'the check carries the typed length, no data entry');
  assert.equal(JSON.stringify(p.state().input), '[1,1200]');
  assert.match(p.app.innerHTML, /data-inputlength="1" min="1" step="1" value="1200"/);
  assert.ok(p.state().layers.some(l => l.name === 'Linear' && l.args.in_features > 0), 'the layers chain from the typed length');
  assert.equal(JSON.stringify(p.saved().design.input), '[1,1200]', 'kept in the tab state');
});

test('a project with data entries: the Input is a menu of them with their lengths; choosing one re-chains the layers and asks a check with that entry', () => {
  const entries = [{ name: 'a', length: 700 }, { name: 'b', length: 1400 }];
  const p = page({ input: [1, 700], entries, dataset: 'a' }, { dom: true });
  assert.match(p.app.innerHTML, /<select data-inputentry="1"[^>]*><option value="a" selected>a · 700<\/option><option value="b">b · 1400<\/option><\/select>/);
  const before = p.sent.filter(m => m.type === 'check').length;
  listenerOf(p, 'change')({ target: { dataset: { inputentry: '1' }, type: 'select-one', value: 'b' } });
  const asked = p.sent.filter(m => m.type === 'check').slice(-1)[0];
  assert.ok(p.sent.filter(m => m.type === 'check').length > before && asked.dataset === 'b' && !asked.input);
  assert.equal(JSON.stringify(p.state().input), '[1,1400]');
  assert.match(p.app.innerHTML, /<option value="b" selected>b · 1400<\/option>/);
});

test('Pooling first: a pooling alone before block 1 (CNN) or before Flatten (MLP) reads as a setting, not custom; every ZvW model reads non-custom and round-trips exactly', () => {
  const all = models('ZvW');
  for (const m of all) {
    const r = P.fromModel(m, FORMS);
    assert.equal(r.custom, false, `${m.model_id} reads as custom`);
    assert.deepEqual(P.toModel(m.model_id, r.type, r.layers).architecture_args.layers, m.architecture_args.layers, `${m.model_id} round trip`);
  }
  const n50 = P.fromModel(all.find(m => m.model_id === 'NoConv1_ASCAD_N50'), FORMS);
  assert.deepEqual([n50.type, n50.knobs.prepool, n50.knobs.blocks, n50.knobs.pattern], ['CNN', 'yes', 2, ['conv', 'norm', 'pool']]);
  const n0 = P.fromModel(all.find(m => m.model_id === 'NoConv1_ASCAD_N0'), FORMS);
  assert.deepEqual([n0.type, n0.knobs.prepool, n0.knobs.hidden, n0.knobs.units], ['MLP', 'yes', 2, 10]);
  const zaid = P.fromModel(all.find(m => m.model_id === 'Zaid_ASCAD_N50'), FORMS);
  assert.equal(zaid.knobs.prepool, 'none');
  // the skeleton builds it back: a pooling first, then the blocks
  const built = P.SKELETONS.CNN.build({ ...P.defaults('CNN'), prepool: 'yes', blocks: 1 });
  assert.deepEqual(built.slice(0, 2).map(l => l.name), ['Pooling', 'Conv1d']);
  assert.equal(P.SKELETONS.MLP.build({ ...P.defaults('MLP'), prepool: 'yes' })[0].name, 'Pooling');
  for (const type of ['CNN', 'MLP']) assert.ok(P.knobHint(type, 'prepool').length > 10 && P.optionHint('prepool', 'yes') && P.optionHint('prepool', 'none'));
});

test('under keras a layer argument keeps PyTorch\'s own options in its menu, disabled, saying they are available in pytorch', () => {
  const p = page({ framework: 'keras' }, { dom: true });
  const i = p.state().layers.findIndex(l => l.name === 'Pooling');
  p.act({ type: 'open', i });
  const html = p.app.innerHTML;
  const menu = (html.match(new RegExp(`<select data-arg="type" data-i="${i}"[^>]*>([^]*?)</select>`)) || [])[1] || '';
  assert.match(menu, /<option value="avg"[^>]*>avg<\/option>/);
  assert.match(menu, /<option value="lp" disabled title="available in pytorch, not under deep_learning.framework: keras">lp \(not available\)<\/option>/);
  const j = p.state().layers.findIndex(l => l.name === 'Conv1d');
  p.act({ type: 'open', i: j });
  const act = (p.app.innerHTML.match(new RegExp(`<select data-arg="activation" data-i="${j}"[^>]*>([^]*?)</select>`)) || [])[1] || '';
  assert.match(act, /<option value="selu"[^>]*>selu<\/option>/);
  assert.match(act, /<option value="hardswish" disabled title="available in pytorch[^"]*">hardswish \(not available\)<\/option>/);
});
