# services/side_channel_core/src/core/monitor.py
"""
Live metrics of a TraceFlow run.

snapshot(run_dir) reads the run's newest history rows and returns one line per
metric the run records (its metrics_to_log, as the history rows hold them, and
nothing else: owner 2026-10-27), its
latest value per process, train | validation | test, with the value one epoch
earlier and whether it improved:

    mean_accuracy           train: 0.856844 ▲ | validation (CV): 0.004 ▼
    mean_attack_rank_curve                    | validation (CV):     6 ▼

- One source: train and validation come from the folder written last, the
  cross-validation fold being trained (metrics/cross_validation/fold_NN,
  "validation (CV)"), the refit, or a run without CV (metrics/attack_metrics),
  never a fold's validation beside the refit's train. The fold mean
  (cross_validation/mean) is not a live row and is skipped. Test, when
  perform_attack.test.attack_times > 0, comes from the run's test rows.
- A metric shows the processes that record it (a rank only validation and test).
- The learning objective is one line named after trainer.loss.type: the rows'
  training_objective and validation_objective, when metrics_to_log lists them
  (the heartbeat is not read: an unlisted loss is not shown).
- Improved: utils/monitor_metric.metric_direction, the one rule of early
  stopping, validation checkpointing and model_judge (mean_accuracy rising is
  better, a rank falling is better).
- A rank curve (an array metric, epNNNN.npy) shows its last point: the rank
  after all the traces of the evaluation.
- Fit times (mean_fit_time_s, fit_time_h, ...) are shown without a better or
  worse mark: a cumulative time only grows, epoch after epoch.

The TraceFlow Monitor (services/visualization/vscode_monitor) shows it under the
running run's summary line. From a terminal:

    python -m core.monitor --run-dir <run folder>            # every 5 s, until Ctrl-C
    python -m core.monitor --run-dir <run folder> --json     # one snapshot, as JSON

Author: Shraddha Mishra
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from utils import metric_lines
from utils.logger import _is_trace_threshold_metric
from utils.monitor_metric import metric_direction
from utils.seal import check as check_seal, load_json as load_seal_json, strip as strip_seal

PROCESSES = ("train", "validation", "test")
# The header of every history row (utils/metrics_logger) and its provenance: not metrics.
_HEADER = {"epoch", "timestamp", "model_name", "model_hash", "process", "env", "dependencies"}


def _rows(path: Path) -> list[dict[str, Any]]:
    """The complete JSON lines of a history file; a line still being written is skipped."""
    rows = []
    check_seal(path)  # utils/seal.py: a changed history raises, it is not shown as progress
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = strip_seal(json.loads(line))
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _traces_attacked(row: dict[str, Any], base: Path) -> Optional[int]:
    """How many traces a row's attacks had: the last trace count of its mean rank curve (else of its attack curves)."""
    for key, last in (("mean_attack_rank_curve", lambda a: a[-1, 0]), ("per_attack_curves", lambda a: a[0, -1, 0])):
        value = row.get(key)
        if isinstance(value, dict) and value.get("_type") == "ndarray" and value.get("path"):
            try:
                count = float(last(np.load(base / str(value["path"]), allow_pickle=False)))
            except (OSError, ValueError, IndexError):
                continue
            if math.isfinite(count) and count > 0:
                return int(count)
    return None


def _value(value: Any, base: Path) -> Optional[float]:
    """A number of a row, or an array metric's last point (a rank curve's [n traces, rank] -> rank)."""
    if isinstance(value, dict) and value.get("_type") == "ndarray" and value.get("path"):
        try:
            array = np.load(base / str(value["path"]), allow_pickle=False)
        except (OSError, ValueError):
            return None
        if array.ndim == 2 and array.shape[0] and array.shape[1] >= 2:
            return float(array[-1, 1])
        return float(array.reshape(-1)[-1]) if array.size else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _histories(metrics: Path, process: str) -> list[Path]:
    return [p for p in metrics.glob(f"**/scalars/*_{process}_history.jsonl")
            if "mean" not in p.relative_to(metrics).parts]


def _newest(files: list[Path]) -> Optional[Path]:
    return max(files, key=lambda p: p.stat().st_mtime, default=None)


def _test_attacked(run_dir: Path) -> bool:
    """perform_attack.test.attack_times > 0 in the run's params_out snapshot (config/<name>_params.yaml)."""
    try:
        snapshot = yaml.safe_load((run_dir / "config" / f"{run_dir.name}_params.yaml").read_text(encoding="utf-8")) or {}
        return int(((snapshot.get("perform_attack") or {}).get("test") or {}).get("attack_times") or 0) > 0
    except (OSError, ValueError, TypeError, yaml.YAMLError):
        return False


def snapshot(run_dir: str | Path) -> dict[str, Any]:
    """{"run": name, "source": "fold 3" | "refit" | None, "epoch": n, "metrics": [{name, values: [{process,
    label, value, previous, direction, improved}]}]}: one line per metric, its value per process. Train and
    validation come from one source, the folder written last (the fold being trained, the refit, or a run
    without CV); test, when attacked, from its own rows; the learning objective named after the loss."""
    run_dir = Path(run_dir)
    metrics = run_dir / "metrics"
    empty = {"run": run_dir.name, "source": None, "epoch": None, "metrics": [], "attack": None, "key_recovery": None}
    if not metrics.is_dir():
        return empty
    newest = _newest([p for process in ("train", "validation") for p in _histories(metrics, process)])
    folder = newest.parent.parent if newest else None           # metrics/attack_metrics or cross_validation/fold_NN
    fold = folder is not None and folder.parent.name == "cross_validation"
    source = f"fold {int(folder.name.split('_')[-1])}" if fold else (
        "refit" if folder is not None and (metrics / "cross_validation").is_dir() else None)
    loss = _loss_type(run_dir)
    lines: dict[str, list[dict[str, Any]]] = {}
    epoch = None
    for process in PROCESSES:
        if process == "test":
            path = _newest(_histories(metrics, "test")) if _test_attacked(run_dir) else None
        else:
            path = _newest(list(folder.glob(f"scalars/*_{process}_history.jsonl"))) if folder is not None else None
        rows = _rows(path) if path is not None else []
        if not rows:
            continue
        if process != "test":
            epoch = max(epoch or 0, int(rows[-1].get("epoch") or 0))
        base = path.parent.parent                       # arrays/... is relative to the fold's (or run's) folder
        last, before = rows[-1], (rows[-2] if len(rows) > 1 else {})
        label = "validation (CV)" if process == "validation" and fold else process
        for name, raw in last.items():
            if name in _HEADER or name.startswith("_"):
                continue
            value = _value(raw, base)
            if value is None:
                # a trace threshold the attacks never crossed (recorded null): "not reached by N traces", as the
                # run's log says it, N the traces this process's attacks had
                if raw is None and _is_trace_threshold_metric(name):
                    traces = _traces_attacked(last, base)
                    lines.setdefault(name, []).append({
                        **_entry(process, label, None, None, metric_direction(name)),
                        "note": f"not reached by {traces:,} traces" if traces else "not reached"})
                continue
            previous = _value(before.get(name), base) if before else None
            line = loss if name in ("training_objective", "validation_objective") else name
            entry = _entry(process, label, value, previous, metric_direction(name))
            if "fit_time" in name:
                entry["improved"] = None                 # a time only grows: neither better nor worse
            lines.setdefault(line, []).append(entry)
    attack = _test_attack(run_dir)
    if attack:
        for name, live in attack.pop("metrics").items():     # the attack metrics so far, in the test slot
            lines[name] = [v for v in lines.get(name, []) if v["process"] != "test"] + [_live_entry(name, live)]
    if lines:
        return {**empty, "source": source, "epoch": epoch, "attack": attack, "key_recovery": _key_recovery(metrics),
                "metrics": [{"name": name, "values": values} for name, values in lines.items()]}
    return empty


def _beat(run_dir: Path) -> Optional[dict[str, Any]]:
    """This run's heartbeat (artifacts/<env>/heartbeat.json), or None when it is another run's or unreadable."""
    try:
        beat_path = run_dir.parent.parent / "heartbeat.json"
        check_seal(beat_path)  # utils/seal.py
        beat = strip_seal(json.loads(beat_path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return None
    return beat if isinstance(beat, dict) and beat.get("model_name") == run_dir.name else None


def _test_attack(run_dir: Path) -> Optional[dict[str, Any]]:
    """{"attack": i, "attacks": n, "running": bool, "started_ns", "written_ns", "metrics": {name: live value}}
    while this run tests: its heartbeat in phase test and its artifacts/<env>/attack_progress.json
    (utils/attack_progress.py); the two times give the ETA the test's speed (None in a file without them)."""
    beat = _beat(run_dir)
    if not beat or beat.get("phase") != "test":
        return None
    try:
        progress = load_seal_json(run_dir.parent.parent / "attack_progress.json")
    except (OSError, ValueError):
        return None
    if not isinstance(progress, dict) or progress.get("model_name") != run_dir.name:
        return None
    return {"attack": int(progress.get("attack") or 0), "attacks": int(progress.get("attacks") or 0),
            "running": bool(progress.get("running")), "started_ns": progress.get("started_ns"),
            "written_ns": progress.get("written_ns"), "metrics": dict(progress.get("metrics") or {})}


def _live_entry(name: str, live: Any) -> dict[str, Any]:
    """A test value while the test attacks: a number, or none yet ("not reached yet"); N̄tGE with how many
    of the attacks so far reach rank 1."""
    note = None
    if isinstance(live, dict):
        note = f"{live.get('reached', 0)} of {live.get('of', 0)} reached"
        live = live.get("value")
    value = float(live) if isinstance(live, (int, float)) and not isinstance(live, bool) else None
    if value is None and note is None:
        note = "not reached yet"
    return {"process": "test", "label": "test", "value": value, "previous": None, "direction": metric_direction(name),
            "improved": None, "live": True, "note": note}


def _key_recovery(metrics: Path) -> Optional[dict[str, Any]]:
    """The newest key-recovery report on disk: the test's (key_recovery/test.json) once written, else the last
    validation report (key_recovery/validation_history.jsonl, the epoch's)."""
    tests = [p for p in metrics.glob("**/key_recovery/test.json")]
    if tests:
        try:
            report, source = load_seal_json(_newest(tests)), "test"
        except (OSError, ValueError):
            return None
    else:
        path = _newest([p for p in metrics.glob("**/key_recovery/validation_history.jsonl")
                        if "mean" not in p.relative_to(metrics).parts])
        rows = _rows(path) if path is not None else []
        if not rows:
            return None
        report = rows[-1]
        source = f"validation, epoch {report.get('epoch')}" if report.get("epoch") is not None else "validation"
    key, recovery = report.get("key") or {}, report.get("recovery") or {}
    return {"source": source, "recovered": key.get("recovered"), "true_key_rank": key.get("final_true_key_rank"),
            "top_guess": key.get("top_key_hypothesis_hex"), "ge": recovery.get("final_ge"),
            "sr1": recovery.get("final_sr_at_1")}


def _entry(process: str, label: str, value: float, previous: Optional[float], direction: str) -> dict[str, Any]:
    improved = None if previous is None or previous == value else (value > previous) == (direction == "max")
    return {"process": process, "label": label, "value": value, "previous": previous, "direction": direction,
            "improved": improved}


def _loss_type(run_dir: Path) -> str:
    """The column's name: the run's trainer.loss.type (the objective it minimises, e.g. CrossEntropyLoss),
    from its params_out snapshot; "learning objective" when the snapshot does not say."""
    try:
        snapshot = yaml.safe_load((run_dir / "config" / f"{run_dir.name}_params.yaml").read_text(encoding="utf-8")) or {}
        name = ((snapshot.get("trainer") or {}).get("loss") or {}).get("type")
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        name = None
    return str(name) if name else "learning objective"


def _shown(v: dict[str, Any]) -> str:
    """A value as printed: its number, a live value's note beside it ("not reached yet" stands alone)."""
    number = "" if v["value"] is None else format(v["value"], ".6g")
    return " ".join(x for x in (number, f"({v['note']})" if number and v.get("note") else v.get("note") or "") if x)


def _print(snap: dict[str, Any]) -> None:
    """One line per metric, a column per process (utils/metric_lines.py, as the run's logs draw each epoch).
    While the test attacks, an attack row takes the epoch's place: the bar under the first column (train),
    attack i of n under the second (validation); the key-recovery report closes the block."""
    green, red, orange, reset = "\033[32m", "\033[31m", "\033[38;5;208m", "\033[0m"
    attack = snap.get("attack")
    head = (snap["run"], snap["source"], None if attack else snap["epoch"] and f"epoch {snap['epoch']}")
    print(" · ".join(str(x) for x in head if x))
    if not snap["metrics"]:
        return
    rows = []
    for m in snap["metrics"]:
        cells = {}
        for v in m["values"]:
            mark = " " if v["improved"] is None else (f"{green}▲{reset}" if v["improved"] else f"{red}▼{reset}")
            cells[v["process"]] = (v["label"], _shown(v), mark)
        rows.append((m["name"], cells))
    name_width = len("attack") if attack else None
    if attack:
        cols, w = metric_lines.columns(rows), metric_lines.widths(rows)
        done, total = attack["attack"], max(attack["attacks"], 1)
        bar_cell = w[cols[0]][2] - 2
        filled = round(bar_cell * done / total)
        bar = f"{orange}{'█' * filled}{reset}{'░' * (bar_cell - filled)}  "
        text = f"attack {done} of {attack['attacks']}"
        rest = [text.ljust(w[cols[1]][2])] if len(cols) > 1 else [text]
        names = max(len(n) for n, _ in rows + [("attack", {})])
        print((f"  {'attack'.ljust(names)}  " + " | ".join([bar] + rest)).rstrip())
    for line in metric_lines.lines(rows, name_width=name_width):
        print(line)
    print(f"  {green}▲{reset} better, {red}▼{reset} worse than the epoch before")
    kr = snap.get("key_recovery")
    if kr:
        print(f"  key recovery · {_key_recovery_text(kr)}")


def _key_recovery_text(kr: dict[str, Any]) -> str:
    """'validation, epoch 20: not recovered · true key rank 107 · top guess 0xFB · GE 107 · SR@1 0.00'."""
    parts = ["recovered" if kr.get("recovered") else "not recovered"]
    if kr.get("true_key_rank") is not None:
        parts.append(f"true key rank {kr['true_key_rank']}")
    if kr.get("top_guess"):
        parts.append(f"top guess {kr['top_guess']}")
    if isinstance(kr.get("ge"), (int, float)):
        parts.append(f"GE {kr['ge']:.4g}")
    if isinstance(kr.get("sr1"), (int, float)):
        parts.append(f"SR@1 {kr['sr1']:.2f}")
    return f"{kr.get('source')}: " + " · ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live metrics of a TraceFlow run")
    parser.add_argument("--run-dir", required=True, help="the run's folder: artifacts/<env>/generated_models/<model_name>")
    parser.add_argument("--json", action="store_true", help="print one snapshot as JSON and stop")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between two looks (terminal mode)")
    args = parser.parse_args()
    if args.json:
        print(json.dumps(snapshot(args.run_dir)))
        return
    seen = None
    try:
        while True:
            snap = snapshot(args.run_dir)
            if snap != seen:
                print(f"\n{snap['run']}")
                _print(snap)
                seen = snap
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("monitor stopped")


if __name__ == "__main__":
    main()
