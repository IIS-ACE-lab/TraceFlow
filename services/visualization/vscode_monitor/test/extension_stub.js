// services/visualization/vscode_monitor/test/extension_stub.js
'use strict';
// The python the panel uses, for tests that run choices.py (the sca env when it is there).
const fs = require('fs');
const path = require('path');
const os = require('os');
exports.python = () => {
  const env = path.join(os.homedir(), 'anaconda3', 'envs', 'sca', 'bin', 'python');
  return fs.existsSync(env) ? env : 'python3';
};
