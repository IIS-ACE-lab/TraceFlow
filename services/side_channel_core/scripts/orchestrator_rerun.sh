#!/usr/bin/env bash
# services/side_channel_core/scripts/orchestrator_rerun.sh
# Orchestrator Rerun Script
# - Reruns experiments in strict replay mode using config snapshots
# - Ensures reproducibility by using snapshot SHA256 checksums
# - Supports multiple models and repeat runs with unique trial directories

#Author: Shraddha Mishra
#==================================================================================
set -euo pipefail

log() { echo "[orchestrator_rerun] $*" >&2; }
die() { log "FATAL: $*"; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCC_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

: "${SCC_MASTER_YAML:=${SCC_ROOT}/config/master.yaml}"
: "${PROJECTS_ROOT:=/app/projects}"
: "${PROJECT:=${PROJECT:-}}"
: "${ENV:=${ENV:-dev}}"
: "${RERUN_EXPERIMENT:=${RERUN_EXPERIMENT:-}}"
: "${REPEAT:=${REPEAT:-}}"

[[ -n "${RERUN_EXPERIMENT}" ]] || die "RERUN_EXPERIMENT is required"
[[ -f "${SCC_MASTER_YAML}" ]] || die "master.yaml not found: ${SCC_MASTER_YAML}"

if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="/app"
fi
export PATH="/venv/bin:$PATH"

RESOLVED_JSON="$(
python - <<'PY'
import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import yaml
except Exception as e:
    print(f"yaml import failed: {e}", file=sys.stderr)
    raise SystemExit(2)

master_path = Path(os.environ["SCC_MASTER_YAML"]).resolve()
projects_root = Path(os.environ["PROJECTS_ROOT"]).resolve()
requested_project = (os.environ.get("PROJECT") or "").strip()
env_name = (os.environ.get("ENV") or "dev").strip()
rerun_names = (os.environ.get("RERUN_EXPERIMENT") or "").strip()
repeat_env = (os.environ.get("REPEAT") or "").strip()

if not rerun_names:
    print("RERUN_EXPERIMENT is required", file=sys.stderr)
    raise SystemExit(2)

data = yaml.safe_load(master_path.read_text()) or {}
projects = data.get("project") or []
if isinstance(projects, dict):
    projects = [projects]
if not isinstance(projects, list):
    projects = []

def find_project(name: str):
    for pr in projects:
        if isinstance(pr, dict) and pr.get("name") == name:
            return pr
    return None

def default_project():
    dp = data.get("default_project")
    if isinstance(dp, str) and dp.strip():
        return dp.strip()
    for pr in projects:
        if isinstance(pr, dict) and isinstance(pr.get("name"), str) and pr["name"].strip():
            return pr["name"].strip()
    return ""

project_name = requested_project or default_project()
if not project_name:
    print("Could not resolve project", file=sys.stderr)
    raise SystemExit(2)

project_block = find_project(project_name)
if not isinstance(project_block, dict):
    print(f"Project block not found for {project_name}", file=sys.stderr)
    raise SystemExit(2)

project_root_rel = (project_block.get("root") or project_name).strip().rstrip("/")
project_root = (projects_root / project_root_rel).resolve()

params_rel = (project_block.get("params") or "config/params.yaml").strip()
params_live = (project_root / params_rel).resolve()
if not params_live.exists():
    print(f"Live params.yaml not found: {params_live}", file=sys.stderr)
    raise SystemExit(2)

live_params = yaml.safe_load(params_live.read_text()) or {}
paths_cfg = live_params.get("paths") or {}

params_out_tmpl = paths_cfg.get("params_out")
if not isinstance(params_out_tmpl, str) or not params_out_tmpl.strip():
    print("paths.params_out missing in live params.yaml", file=sys.stderr)
    raise SystemExit(2)

exp_rerun = project_block.get("exp_rerun") or {}
rerun_path_tmpl = (
    exp_rerun.get("path")
    or "artifacts/{{env}}/generated_models/{{model_name}}/rerun/{{timestamp}}_trial_{{trial}}"
)

default_repeat = int(exp_rerun.get("repeat", 1))
repeat_final = int(repeat_env) if repeat_env else default_repeat
timestamp = datetime.utcnow().strftime("%Y_%m_%d_%H%M%S")

model_names = [x.strip() for x in rerun_names.split(",") if x.strip()]
if not model_names:
    print("No valid model names in RERUN_EXPERIMENT", file=sys.stderr)
    raise SystemExit(2)

out = {
    "project": project_name,
    "project_root": str(project_root),
    "params_out_tmpl": str(params_out_tmpl),
    "rerun_path_tmpl": str(rerun_path_tmpl),
    "repeat": repeat_final,
    "timestamp": timestamp,
    "model_names": model_names,
}
print(json.dumps(out))
PY
)" || die "failed to resolve rerun contract"

json_get() {
  local key="$1"
  python - "$RESOLVED_JSON" "$key" <<'PY'
import json
import sys
obj = json.loads(sys.argv[1])
val = obj.get(sys.argv[2], "")
if isinstance(val, list):
    import json as _json
    print(_json.dumps(val))
else:
    print(val)
PY
}

PROJECT="$(json_get project)"
PROJECT_ROOT="$(json_get project_root)"
PARAMS_OUT_TMPL="$(json_get params_out_tmpl)"
RERUN_PATH_TMPL="$(json_get rerun_path_tmpl)"
REPEAT_FINAL="$(json_get repeat)"
RUN_TIMESTAMP="$(json_get timestamp)"
MODEL_NAMES_JSON="$(json_get model_names)"

if ! [[ "${REPEAT_FINAL}" =~ ^[0-9]+$ ]] || [[ "${REPEAT_FINAL}" -lt 1 ]]; then
  die "REPEAT must be a positive integer"
fi

sha256_of_file() {
  local fp="$1"
  python - "$fp" <<'PY'
import hashlib
import sys
from pathlib import Path

p = Path(sys.argv[1])
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
}

render_snapshot_path() {
  local model_name="$1"
  python - "$PARAMS_OUT_TMPL" "$PROJECT_ROOT" "$ENV" "$model_name" <<'PY'
import sys
from pathlib import Path

tmpl = sys.argv[1]
project_root = Path(sys.argv[2])
env_name = sys.argv[3]
model_name = sys.argv[4]

out = (
    tmpl.replace("{{env}}", env_name)
        .replace("{{model_name}}", model_name)
        .replace("{{experiment_name}}", model_name)
)

p = Path(out)
if not p.is_absolute():
    p = (project_root / p).resolve()

print(str(p))
PY
}

render_model_root() {
  local model_name="$1"
  echo "${PROJECT_ROOT}/artifacts/${ENV}/generated_models/${model_name}"
}

next_trial_start() {
  local model_name="$1"
  python - "$PROJECT_ROOT" "$ENV" "$model_name" <<'PY'
import re
import sys
from pathlib import Path

project_root = Path(sys.argv[1])
env_name = sys.argv[2]
model_name = sys.argv[3]

rerun_root = project_root / "artifacts" / env_name / "generated_models" / model_name / "rerun"
max_trial = 0

if rerun_root.exists():
    for p in rerun_root.glob("*_trial_*"):
        m = re.search(r"_trial_(\d+)", p.name)
        if m:
            max_trial = max(max_trial, int(m.group(1)))

print(max_trial + 1)
PY
}

render_rerun_dir() {
  local model_name="$1"
  local trial="$2"
  python - "$RERUN_PATH_TMPL" "$PROJECT_ROOT" "$ENV" "$model_name" "$RUN_TIMESTAMP" "$trial" <<'PY'
import sys
from pathlib import Path

tmpl = sys.argv[1]
project_root = Path(sys.argv[2])
env_name = sys.argv[3]
model_name = sys.argv[4]
timestamp = sys.argv[5]
trial = sys.argv[6]

out = (
    tmpl.replace("{{env}}", env_name)
        .replace("{{model_name}}", model_name)
        .replace("{{experiment_name}}", model_name)
        .replace("{{timestamp}}", timestamp)
        .replace("{{trial}}", trial)
        .replace("{{trial_#}}", trial)
)

p = Path(out)
if not p.is_absolute():
    p = (project_root / p).resolve()

print(str(p))
PY
}

write_attempt_meta() {
  local model_name="$1"
  local trial="$2"
  local trial_dir="$3"
  local snapshot_path="$4"
  local snapshot_sha="$5"

  mkdir -p "${trial_dir}"

  cat > "${trial_dir}/rerun_dispatch.yaml" <<EOF
project: "${PROJECT}"
experiment_name: "${model_name}"
env: "${ENV}"
trial: ${trial}
repeat_total: ${REPEAT_FINAL}
pipeline: "strict_basic_train_test_replay"
timestamp: "${RUN_TIMESTAMP}"
rerun_artifacts_root: "${trial_dir}"
params_snapshot: "${snapshot_path}"
params_snapshot_sha256: "${snapshot_sha}"
strict_rerun_only: true
note: "experiment was rerun using config snapshot at ${snapshot_path}"
EOF
}

run_train_once() {
  local snapshot_path="$1"
  local model_name="$2"
  local snapshot_sha="$3"
  local rerun_dir="$4"

  log "running BASIC train() replay for model=${model_name}"
  log "rerun artifacts will be saved under: ${rerun_dir}"

  export ORCH_RERUN_SNAPSHOT_PATH="${snapshot_path}"
  export ORCH_RERUN_MODEL_NAME="${model_name}"
  export ORCH_RERUN_SNAPSHOT_SHA256="${snapshot_sha}"
  export ORCH_RERUN_ENV="${ENV}"
  export ORCH_RERUN_ARTIFACTS_ROOT="${rerun_dir}"

  python - <<'PY'
import os
import yaml
from pathlib import Path
from services.side_channel_core.src.sca.train import train

snapshot_path = Path(os.environ["ORCH_RERUN_SNAPSHOT_PATH"])
model_name = os.environ["ORCH_RERUN_MODEL_NAME"]
snapshot_sha = os.environ["ORCH_RERUN_SNAPSHOT_SHA256"]
env_name = os.environ["ORCH_RERUN_ENV"]
rerun_root = os.environ["ORCH_RERUN_ARTIFACTS_ROOT"]

cfg = yaml.safe_load(snapshot_path.read_text()) or {}

cfg["env"] = env_name
cfg["model_name"] = model_name
cfg["rerun"] = True
cfg["strict_rerun_only"] = True
cfg["bypass_registry"] = True
cfg["replay_snapshot_path"] = str(snapshot_path)
cfg["replay_snapshot_sha256"] = snapshot_sha

cfg.setdefault("paths", {})
cfg["paths"]["artifacts_root"] = rerun_root
cfg["paths"]["logs"] = f"{rerun_root}/logs"
cfg["paths"]["metrics"] = f"{rerun_root}/metrics"
cfg["paths"]["model"] = f"{rerun_root}/model"
cfg["paths"]["model_out"] = f"{rerun_root}/model"
cfg["paths"]["checkpoints"] = f"{rerun_root}/checkpoints"
cfg["paths"]["graphs"] = f"{rerun_root}/graphs"
cfg["paths"]["data"] = f"{rerun_root}/data"
cfg["paths"]["config"] = f"{rerun_root}/config"
cfg["paths"]["params_out"] = f"{rerun_root}/config/{model_name}_params.yaml"

data_name = None
if isinstance(cfg.get("data"), dict):
    data_name = cfg["data"].get("name")
cfg["paths"]["data_out"] = f"{rerun_root}/data/{data_name}.h5" if data_name else f"{rerun_root}/data/dataset.h5"

train(
    cfg=cfg,
    exec_process="train",
    override_model_name=model_name,
    model_id=None,
)
PY
}

run_test_once() {
  local snapshot_path="$1"
  local model_name="$2"
  local snapshot_sha="$3"
  local rerun_dir="$4"

  log "running BASIC test() replay for model=${model_name}"
  log "rerun artifacts will be saved under: ${rerun_dir}"

  export ORCH_RERUN_SNAPSHOT_PATH="${snapshot_path}"
  export ORCH_RERUN_MODEL_NAME="${model_name}"
  export ORCH_RERUN_SNAPSHOT_SHA256="${snapshot_sha}"
  export ORCH_RERUN_ENV="${ENV}"
  export ORCH_RERUN_ARTIFACTS_ROOT="${rerun_dir}"

  python - <<'PY'
import os
import yaml
from pathlib import Path
from services.side_channel_core.src.sca.test import test

snapshot_path = Path(os.environ["ORCH_RERUN_SNAPSHOT_PATH"])
model_name = os.environ["ORCH_RERUN_MODEL_NAME"]
snapshot_sha = os.environ["ORCH_RERUN_SNAPSHOT_SHA256"]
env_name = os.environ["ORCH_RERUN_ENV"]
rerun_root = os.environ["ORCH_RERUN_ARTIFACTS_ROOT"]

cfg = yaml.safe_load(snapshot_path.read_text()) or {}

cfg["env"] = env_name
cfg["model_name"] = model_name
cfg["strict_rerun_only"] = True
cfg["bypass_registry"] = True
cfg["replay_snapshot_path"] = str(snapshot_path)
cfg["replay_snapshot_sha256"] = snapshot_sha

cfg.setdefault("paths", {})
cfg["paths"]["artifacts_root"] = rerun_root
cfg["paths"]["logs"] = f"{rerun_root}/logs"
cfg["paths"]["metrics"] = f"{rerun_root}/metrics"
cfg["paths"]["model"] = f"{rerun_root}/model"
cfg["paths"]["model_out"] = f"{rerun_root}/model"
cfg["paths"]["checkpoints"] = f"{rerun_root}/checkpoints"
cfg["paths"]["graphs"] = f"{rerun_root}/graphs"
cfg["paths"]["data"] = f"{rerun_root}/data"
cfg["paths"]["config"] = f"{rerun_root}/config"
cfg["paths"]["params_out"] = f"{rerun_root}/config/{model_name}_params.yaml"

data_name = None
if isinstance(cfg.get("data"), dict):
    data_name = cfg["data"].get("name")
cfg["paths"]["data_out"] = f"{rerun_root}/data/{data_name}.h5" if data_name else f"{rerun_root}/data/dataset.h5"

cfg.setdefault("test", {})
cfg["test"]["model_name"] = model_name
cfg["test"]["model_checkpoint"] = f"{model_name}_trained.pt"

test(
    cfg=cfg,
    exec_process="test",
    override_model_name=model_name,
    model_id=None,
)
PY
}

export PROJECT
export PROJECT_ROOT
export FORCE=1
export STRICT_RERUN_ONLY=1

MODEL_NAMES="$(python - "$MODEL_NAMES_JSON" <<'PY'
import json, sys
print("\n".join(json.loads(sys.argv[1])))
PY
)"

log "resolved project=${PROJECT}"
log "resolved project_root=${PROJECT_ROOT}"
log "resolved repeat=${REPEAT_FINAL}"
log "resolved timestamp=${RUN_TIMESTAMP}"
log "requested model names: $(echo "$MODEL_NAMES" | paste -sd ',' -)"

while IFS= read -r MODEL_NAME_ITEM; do
  [[ -n "${MODEL_NAME_ITEM}" ]] || continue

  MODEL_ROOT="$(render_model_root "${MODEL_NAME_ITEM}")"
  SNAPSHOT_PATH="$(render_snapshot_path "${MODEL_NAME_ITEM}")"

  [[ -d "${MODEL_ROOT}" ]] || die "model directory not found for '${MODEL_NAME_ITEM}': ${MODEL_ROOT}"
  [[ -f "${SNAPSHOT_PATH}" ]] || die "snapshot cfg not found for '${MODEL_NAME_ITEM}': ${SNAPSHOT_PATH}"

  SNAPSHOT_SHA256="$(sha256_of_file "${SNAPSHOT_PATH}")"
  TRIAL_START="$(next_trial_start "${MODEL_NAME_ITEM}")"

  log "verified model exists: ${MODEL_ROOT}"
  log "experiment '${MODEL_NAME_ITEM}' will be rerun using config snapshot at ${SNAPSHOT_PATH}"
  log "reproducibility checksum guard: snapshot sha256=${SNAPSHOT_SHA256}"
  log "next unique trial number starts at: ${TRIAL_START}"

  for ((offset=0; offset<REPEAT_FINAL; offset++)); do
    TRIAL_NUM=$((TRIAL_START + offset))
    TRIAL_DIR="$(render_rerun_dir "${MODEL_NAME_ITEM}" "${TRIAL_NUM}")"
    write_attempt_meta "${MODEL_NAME_ITEM}" "${TRIAL_NUM}" "${TRIAL_DIR}" "${SNAPSHOT_PATH}" "${SNAPSHOT_SHA256}"

    log "trial $((offset + 1))/${REPEAT_FINAL} for model=${MODEL_NAME_ITEM} -> ${TRIAL_DIR}"
    export EXP_RERUN_OUTPUT_DIR="${TRIAL_DIR}"
    export RERUN_EXPERIMENT="${MODEL_NAME_ITEM}"

    run_train_once "${SNAPSHOT_PATH}" "${MODEL_NAME_ITEM}" "${SNAPSHOT_SHA256}" "${TRIAL_DIR}"
    run_test_once "${SNAPSHOT_PATH}" "${MODEL_NAME_ITEM}" "${SNAPSHOT_SHA256}" "${TRIAL_DIR}"
  done

done <<< "${MODEL_NAMES}"

log "all strict rerun trials completed successfully"
