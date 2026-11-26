// services/visualization/vscode_monitor/test/loggrammar.test.js
// The .log grammar injection: an epoch block's value followed by ▲ is traceflow.better.log, one followed by ▼
// traceflow.worse.log (the workspace colours them green and red); the extension contributes and packs it.
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const grammar = JSON.parse(fs.readFileSync(path.join(ROOT, 'syntaxes', 'traceflow-log-arrows.tmLanguage.json'), 'utf8'));

/** Every [text, scope] the grammar's patterns mark on a line, first pattern first (as TextMate tries them). */
function marks(line) {
  const out = [];
  let rest = line;
  for (const p of grammar.patterns) {
    const re = new RegExp(p.match, 'gu');
    for (const m of rest.matchAll(re)) {
      if (p.captures) for (const [i, c] of Object.entries(p.captures)) out.push([m[Number(i)], c.name]);
      else out.push([m[0], p.name]);
    }
    rest = rest.replace(re, s => ' '.repeat(s.length));   // a span one pattern took is not matched again
  }
  return out;
}

test('a value before ▲ is better, before ▼ worse, the arrows and the key too', () => {
  assert.deepEqual(marks('[INFO] Training:     mean_accuracy             |    0.00925 ▼ |        0'),
    [['0.00925', 'traceflow.worse.log'], ['▼', 'traceflow.worse.log']]);
  assert.deepEqual(marks('[INFO] Training:     CrossEntropyLoss          |      5.485 ▲ |   5.9373 ▲'),
    [['5.485', 'traceflow.better.log'], ['▲', 'traceflow.better.log'], ['5.9373', 'traceflow.better.log'], ['▲', 'traceflow.better.log']]);
  assert.deepEqual(marks('final_ranks | | n=100, min=107.0000, mean=107.0000, max=107.0000, final true-key rank = 107 ▼'),
    [['107', 'traceflow.worse.log'], ['▼', 'traceflow.worse.log']]);
  assert.deepEqual(marks('    ▲ better, ▼ worse than the epoch before'),
    [['▲', 'traceflow.better.log'], ['▼', 'traceflow.worse.log']]);
  assert.deepEqual(marks('η_t                       | 0.00054972   |'), []);   // unchanged: no sign, no colour
});

test('the extension contributes the grammar into the Log language and packs its file', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
  assert.deepEqual(pkg.contributes.grammars, [{ scopeName: grammar.scopeName,
    path: './syntaxes/traceflow-log-arrows.tmLanguage.json', injectTo: ['text.log'] }]);
  assert.equal(grammar.injectionSelector, 'L:text.log');
  assert.match(fs.readFileSync(path.join(ROOT, 'pack.py'), 'utf8'), /"syntaxes\/traceflow-log-arrows\.tmLanguage\.json"/);
});
