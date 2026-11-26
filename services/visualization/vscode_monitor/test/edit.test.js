// services/visualization/vscode_monitor/test/edit.test.js
'use strict';
// The form edit flow of extension.js against the real yamledit.py (needs the sca env's python).
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

const answers = { warning: 'Delete', input: 'fig2b' };
const load = Module._load;
Module._load = function (request, ...rest) {
  if (request === 'vscode') {
    return {
      workspace: { getConfiguration: () => ({ get: () => undefined }) },
      window: {
        showWarningMessage: async (...a) => { answers.seen = a; return answers.warning; },
        showInputBox: async () => answers.input,
        showInformationMessage: () => {},
      },
    };
  }
  return load.call(this, request, ...rest);
};
const E = require('../extension');
const ROOT = path.resolve(__dirname, '../../../..');
const PYTHON = E.findPython();
const skip = PYTHON === 'python3' ? 'the sca conda env is not installed' : false;

function setup() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'experiment_queue.yaml');
  fs.writeFileSync(file, '# queues\nfig2:   # first\n  - make a\nfig3:\n  - make b\n');
  const panel = new E.Panel({});
  panel.root = ROOT;
  panel.python = PYTHON;
  panel.layout = { projects: [{ name: 'P', configs: [{ file: 'experiment_queue.yaml', path: file }] }] };
  const posted = [];
  panel.view = { webview: { postMessage: m => { posted.push(m); return Promise.resolve(); } } };
  return { panel, file, posted };
}

test('a form edit is written when the hash matches and refused when the file changed', { skip }, async () => {
  const { panel, file, posted } = setup();
  await panel.sendTree(file);
  const sha = posted.pop().sha256;
  await panel.edit({ file, sha, op: 'set', path: ['fig3', 0], value: 'make c' });
  assert.match(posted.pop().note, /^saved/);
  assert.equal(fs.readFileSync(file, 'utf8'), '# queues\nfig2:   # first\n  - make a\nfig3:\n  - make c\n');
  await panel.edit({ file, sha, op: 'set', path: ['fig3', 0], value: 'make d' });      // the old hash
  assert.match(posted.pop().note, /changed on disk/);
  assert.match(fs.readFileSync(file, 'utf8'), /make c/);
});

test('the panel never deletes or duplicates an entry', { skip }, async () => {
  const { panel, file, posted } = setup();
  await panel.sendTree(file);
  let sha = posted.pop().sha256;
  answers.warning = 'Delete';                                   // even a "yes" cannot delete
  await panel.edit({ file, sha, op: 'delete', path: ['fig3'] });
  assert.match(posted.pop().note, /refused: the panel does not delete/);
  assert.match(fs.readFileSync(file, 'utf8'), /fig3/);
  await panel.edit({ file, sha, op: 'duplicate', path: ['fig2'] });
  assert.match(posted.pop().note, /refused: the panel does not duplicate/);
  assert.doesNotMatch(fs.readFileSync(file, 'utf8'), /fig2b|fig2_copy/);
});

test('only a declared project config can be edited', { skip }, async () => {
  const { panel, posted } = setup();
  const other = path.join(os.tmpdir(), 'not-a-config.yaml');
  fs.writeFileSync(other, 'a: 1\n');
  await panel.edit({ file: other, sha: 'x', op: 'set', path: ['a'], value: '2' });
  assert.equal(posted.length, 0);
  assert.equal(fs.readFileSync(other, 'utf8'), 'a: 1\n');
});

test('OK writes the staged changes in one go, removals from the highest index, appends last', { skip }, async () => {
  const { panel, file, posted } = setup();
  fs.writeFileSync(file, '# queues\nfig2:   # first\n  - make a\n  - make x\n  - make y\nfig3:\n  - make b\n');
  await panel.sendTree(file);
  const sha = posted.pop().sha256;
  await panel.apply({ file, sha, edits: [
    { op: 'add', path: ['fig2'], value: 'make z' },             // staged first, written last
    { op: 'remove', path: ['fig2', 1] },
    { op: 'remove', path: ['fig2', 2] },                        // written before index 1
    { op: 'set', path: ['fig3', 0], value: 'make c' },
  ] });
  assert.match(posted.pop().note, /^saved: 4 change\(s\)/);
  assert.equal(fs.readFileSync(file, 'utf8'), '# queues\nfig2:   # first\n  - make a\n  - make z\nfig3:\n  - make c\n');
});

test('OK on a form drawn from an older text writes nothing (the file wins)', { skip }, async () => {
  const { panel, file, posted } = setup();
  await panel.sendTree(file);
  const sha = posted.pop().sha256;
  fs.writeFileSync(file, '# queues\nfig2:   # first\n  - make a\nfig3:\n  - make edited by hand\n');
  await panel.apply({ file, sha, edits: [{ op: 'set', path: ['fig3', 0], value: 'make c' }] });
  assert.match(posted.pop().note, /changed on disk; reloaded from the file, 1 unsaved change\(s\) not written/);
  assert.match(fs.readFileSync(file, 'utf8'), /make edited by hand/);
});

test('a model\'s × deletes that model from params.yaml once Delete is answered, the other one kept', { skip }, async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tf-'));
  const file = path.join(dir, 'params.yaml');
  fs.writeFileSync(file, 'model:\n  - model_id: A   # first\n    architecture: MLP\n  - model_id: B\n    architecture: CNN\n');
  const panel = new E.Panel({});
  panel.root = ROOT;
  panel.python = PYTHON;
  panel.layout = { projects: [{ name: 'P', configs: [{ file: 'params.yaml', path: file }], queues: [] }] };
  const posted = [];
  panel.view = { webview: { postMessage: m => { posted.push(m); return Promise.resolve(); } } };
  await panel.sendTree(file);
  const sha = posted.pop().sha256;
  answers.warning = 'Delete';
  await panel.deleteEntry({ file, sha, kind: 'model', name: 'B', path: ['model', 1] });
  assert.match(posted.pop().note, /^saved/);
  assert.equal(fs.readFileSync(file, 'utf8'), 'model:\n  - model_id: A   # first\n    architecture: MLP\n');
});

test('Load Blueprint is asked first: the exact warning, modal; only OK lets the panel write', async () => {
  const { panel, posted } = setup();
  const file = path.join(ROOT, 'projects', 'ZvW', 'config', 'params.yaml');   // inside the repository (only asked, never written)
  panel.post = (view, m) => posted.push(m);
  answers.warning = undefined;                             // Cancel, or the dialog closed
  await panel.completeAsk({ file, unsaved: 2 });
  const [text, opts, ...buttons] = answers.seen;
  const words = t => String(t).replace(/\s+/g, ' ').trim();   // dialog() wraps a long heading at 72 columns
  assert.equal(words(text), 'This action will override the current file and load the default values from the blueprint file.');
  assert.equal(words(text), E.COMPLETE_WARNING);
  assert.equal(opts.modal, true);
  assert.deepEqual(buttons, ['OK']);
  assert.match(words(opts.detail), /written at once/);
  assert.match(words(opts.detail), /2 unsaved changes are written with them/);
  assert.ok(!posted.some(m => m.type === 'complete-ok'), 'Cancel: nothing');
  answers.warning = 'OK';
  await panel.completeAsk({ file, unsaved: 0 });
  assert.deepEqual(posted.filter(m => m.type === 'complete-ok'), [{ type: 'complete-ok', file }]);
  await panel.completeAsk({ file: '/etc/passwd', unsaved: 0 });
  assert.equal(posted.filter(m => m.type === 'complete-ok').length, 1, 'a file outside the repository: refused');
  answers.warning = 'Delete';
});
