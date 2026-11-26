# services/visualization/vscode_monitor/trace_view.py
"""
One trace through a model, layer by layer: what each layer of the planner's design (untrained weights), or of a
trained run (its weights), makes of it. The planner draws one line per layer from it.

    python -m services.visualization.vscode_monitor.trace_view entries --project P
    python -m services.visualization.vscode_monitor.trace_view runs --project P --model-id M
    python -m services.visualization.vscode_monitor.trace_view design --project P --dataset D --trace K [--to L] [--architecture A] < layers.json
    python -m services.visualization.vscode_monitor.trace_view run --project P --run NAME --trace K [--to L]
    python -m services.visualization.vscode_monitor.trace_view recovery --project P --run NAME

entries: the project's data entries, each with its type and why it cannot be read here (its database file not
on this machine; nothing is downloaded). runs: the DONE runs of a model_id, each with the layers, data entry
and framework of its training snapshot. design: attack trace K of the data entry (attack_range start + K,
its trace window, a desync offset drawn with the run's seed as a run draws one, its trace_normalization; a
fitted scaler is fitted on the entry's profiling traces), through the design built by core.model with
freshly initialised weights (torch seeded with the run's seed). run: attack trace K of the run's own test.h5
with the run's scaler (as make test reads it), through the snapshot's model with the checkpoint make test
loads (evaluation.test.load_model, the manifest's SHA-256 and seals checked).

recovery: how the run was trained (its snapshot: cross-validation, k folds over the profiling traces then a
refit on all of them, or a hold-out train / validation / test split, utils/data_split.py) and the key recovery
it recorded, read as TraceFlow wrote it (seals checked), nothing recomputed: run_registry.json (without CV the validation attack; with CV each fold's attack on its held-out
traces, fold_key_recovery) and test_registry.json (the recorded test and every external test).

Each prints one JSON object. A layer's line: its output for that trace, at most POINTS values (bin means); a
layer with channels gives their root mean square per sample (where it responds), a vector its values. With
--to L, traces K to L (at most MAX_RANGE): each line is the mean of the traces' lines, with a band from their
lowest to their highest value at each point (bin minima and maxima); a trained run gives the true class's
median rank over those single traces (not guessing entropy: no evidence is combined across traces).

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from services.visualization.vscode_monitor.model_check import LayerFailed, _first, walk

POINTS = 120
TOP = 5
MAX_RANGE = 1000
CHUNK = 100                     # traces per forward pass of a range


def _attack_byte(cfg: dict, entry: dict) -> int:
    """The byte a run of this entry attacks: sca.attack_byte's first value, or, left empty, the one byte the
    entry's type defines (dataset_types.derived_attack_byte), else the first it allows."""
    from sca.dataset_types import attack_bytes_allowed, derived_attack_byte

    raw = _first((cfg.get("sca") or {}).get("attack_byte"))
    if raw not in (None, [], ""):
        return int(raw)
    byte = derived_attack_byte(entry)
    return int(byte if byte is not None else attack_bytes_allowed(entry)[0])


def _entries_of(cfg: dict) -> list[dict]:
    """The loaded config keys the data entries by name ({ascad: {...}}); params.yaml lists them."""
    data = cfg.get("data")
    if isinstance(data, dict) and "database" not in data:
        return [{"name": k, **v} for k, v in data.items() if isinstance(v, dict)]
    return [d for d in (data if isinstance(data, list) else [data]) if isinstance(d, dict)]


def _project_cfg(project: str) -> dict:
    from services.side_channel_core.utils.config_loader import get_project_config

    return copy.deepcopy(get_project_config(project, validate_data=False))


def _database(cfg: dict, entry: dict) -> tuple[Optional[Path], str]:
    """The entry's database file and why it cannot be read here ('' when it can). Never downloads."""
    from sca.dataset_types import registered
    from utils.path_utils import resolve_path

    raw = entry.get("database")
    if not raw:
        return None, "no database in this data entry"
    path = Path(str(resolve_path(str(raw), {**cfg, "data": entry}))).resolve()
    if path.is_file():
        return path, ""
    if registered(path):
        return path, f"not downloaded on this machine yet: the first run of this data entry fetches {path.name}"
    return path, f"not on this machine: {path}"


def entries(project: str) -> dict[str, Any]:
    """Each data entry: its type, its attack traces, its trace length (the window the attack byte reads), why not."""
    from sca.ascad_data_loader import _range_pair, _resolve_trace_selection, _target_points_for_attack_byte

    cfg = _project_cfg(project)
    out = []
    for entry in _entries_of(cfg):
        _, why = _database(cfg, entry)
        count = length = 0
        try:
            byte = _attack_byte(cfg, entry)
            lo, hi = _range_pair(entry, "attack_range")
            count = hi - lo
            length = _resolve_trace_selection(_target_points_for_attack_byte(entry, attack_byte=byte)).output_length
        except (KeyError, TypeError, ValueError) as exc:
            why = why or f"its attack_range or trace window cannot be read: {exc}"
        out.append({"name": str(entry.get("name")), "type": str(entry.get("type") or "ASCADf"), "attack": count, "length": length, "why": why})
    return {"entries": out}


def runs(project: str, model_id: str) -> dict[str, Any]:
    from sca.test_request import load_snapshot, resolve_models
    from utils.run_map import load_run_map

    cfg = _project_cfg(project)
    root = Path(cfg["project"]["root"])
    done = sorted({str(r.get("model_name")) for r in load_run_map(root).values()
                   if str(r.get("status") or "").upper() == "DONE" and str(r.get("model_id")) == str(model_id) and r.get("model_name")})
    out = []
    for name in done:
        row: dict[str, Any] = {"name": name, "why": ""}
        try:
            run = resolve_models(project, [name])[0]
            snap = load_snapshot(project, run)
            model = next((m for m in snap.get("model") or [] if isinstance(m, dict) and m.get("model_id") == snap.get("model_id")), None)
            entry = (_entries_of(snap) or [{}])[0]
            row.update(env=run.get("env"), framework=str((snap.get("deep_learning") or {}).get("framework") or "pytorch"),
                       dataset=entry.get("name"), type=str(entry.get("type") or "ASCADf"),
                       architecture=(model or {}).get("architecture"), layers=((model or {}).get("architecture_args") or {}).get("layers"))
            if not isinstance(row["layers"], list) or not row["layers"]:
                row["why"] = "its snapshot model has no layer list (a graph or a named model)"
        except Exception as exc:  # noqa: BLE001 - the run is listed, disabled with why
            row["why"] = f"{type(exc).__name__}: {exc}"
        out.append(row)
    return {"model_id": model_id, "runs": out}


# ── one trace ─────────────────────────────────────────────────────────────────

def _line(values: Any, reduce: Any = None) -> list[float]:
    """At most POINTS values: per bin (np.array_split) its mean, or `reduce` (np.min, np.max for a band), rounded."""
    import numpy as np

    a = np.asarray(values, dtype=np.float64).ravel()
    if a.size > POINTS:
        a = np.array([(reduce or np.mean)(b) for b in np.array_split(a, POINTS)])
    return [float(f"{v:.4g}") for v in a]


def _lines(y: Any) -> tuple[str, list, Any]:
    """Each trace's line of one layer's output (n x points): channels x samples give their root mean square per
    sample (where it responds), a vector its values; with the kind and one trace's shape."""
    import numpy as np

    a = y.detach().float().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)
    if a.ndim >= 3:
        return "channels", list(a.shape[1:]), np.sqrt((a.reshape(a.shape[0], a.shape[1], -1) ** 2).mean(axis=1))
    return "vector", list(a.shape[1:]), a.reshape(a.shape[0], -1)


class _Spread:
    """The lines of many traces, one chunk at a time: their sum, lowest and highest value at each point."""

    def __init__(self):
        self.n, self.sum, self.min, self.max = 0, None, None, None

    def add(self, lines: Any) -> None:
        import numpy as np

        lines = np.asarray(lines, dtype=np.float64)
        if self.sum is None:
            self.sum, self.min, self.max = lines.sum(axis=0), lines.min(axis=0), lines.max(axis=0)
        else:
            self.sum += lines.sum(axis=0)
            np.minimum(self.min, lines.min(axis=0), out=self.min)
            np.maximum(self.max, lines.max(axis=0), out=self.max)
        self.n += lines.shape[0]

    def row(self) -> dict[str, Any]:
        """One trace: its line; several: their mean and the band from the lowest to the highest value."""
        import numpy as np

        mean = self.sum / self.n
        if self.n == 1:
            return {"values": _line(mean), "lo": float(mean.min()) if mean.size else 0.0, "hi": float(mean.max()) if mean.size else 0.0}
        return {"values": _line(mean), "band": [_line(self.min, np.min), _line(self.max, np.max)],
                "lo": float(self.min.min()) if self.min.size else 0.0, "hi": float(self.max.max()) if self.max.size else 0.0}


def _through(net: Any, layer_defs: list, xs: Any, labels: Optional[list]) -> dict[str, Any]:
    """The traces xs (n x channels x samples) through the net, CHUNK at a time: each layer's line (one trace) or
    mean and band (several). One trace: the last output's top classes (raw values: their order is any softmax's)
    and, with its label, the true class's rank; several: the true class's ranks over the single traces."""
    import numpy as np
    import torch

    out: dict[str, Any] = {"layers": [], "error": None, "at": None, "count": int(len(xs))}
    spreads: dict[int, tuple[str, str, list, _Spread]] = {}
    ranks: list[int] = []
    first = None
    try:
        with torch.no_grad():
            for start in range(0, len(xs), CHUNK):
                last = None
                for index, name, y in walk(net, layer_defs, torch.from_numpy(np.ascontiguousarray(xs[start:start + CHUNK], dtype=np.float32))):
                    kind, shape, lines = _lines(y)
                    spreads.setdefault(index, (name, kind, shape, _Spread()))[3].add(lines)
                    last = y
                scores = np.asarray(last.detach().float().cpu().numpy() if hasattr(last, "detach") else last).reshape(len(xs[start:start + CHUNK]), -1)
                first = scores[0] if first is None else first
                for k, label in enumerate((labels or [])[start:start + CHUNK]):
                    order = np.argsort(-scores[k], kind="stable")
                    if 0 <= label < scores.shape[1]:
                        ranks.append(int(np.where(order == label)[0][0]) + 1)
    except LayerFailed as failed:
        out["error"], out["at"] = str(failed.cause), failed.index
        return out
    out["layers"] = [{"index": index, "type": name, "shape": shape, "kind": kind, **spread.row()}
                     for index, (name, kind, shape, spread) in sorted(spreads.items())]
    out["outputs"] = int(np.prod(out["layers"][-1]["shape"])) if out["layers"] else None   # what the built last layer gives
    if len(xs) == 1:
        order = np.argsort(-first, kind="stable")
        out["top"] = [[int(c), float(f"{first[c]:.4g}")] for c in order[:TOP]]
        if ranks:
            out["label"], out["label_rank"] = int(labels[0]), ranks[0]
    elif ranks:
        out["ranks"] = {"median": float(np.median(ranks)), "best": min(ranks), "worst": max(ranks), "count": len(ranks)}
    return out


def _span(trace: int, to: Optional[int], available: int, where: str) -> tuple[list[int], str]:
    """The trace numbers asked for (one, or trace to `to`), or why not."""
    last = trace if to is None else to
    if last < trace:
        return [], f"traces {trace} to {last}: the range ends before it starts"
    if last - trace + 1 > MAX_RANGE:
        return [], f"traces {trace} to {last}: a range takes at most {MAX_RANGE} traces"
    if trace < 0 or last >= available:
        return [], f"trace {last if trace >= 0 else trace}: {where} has traces 0 to {available - 1}"
    return list(range(trace, last + 1)), ""


def _shown_path(path: Path) -> str:
    """A file as the page names it: relative to the repository when under it."""
    from utils.path_utils import _get_workspace_root

    try:
        return str(Path(path).resolve().relative_to(_get_workspace_root()))
    except ValueError:
        return str(path)


def _target(cfg: dict, raw_sca: dict, where: str) -> dict[str, Any]:
    """What the output classes are: the leakage model of the label target of the data type (sca/dataset_types.py
    TARGET_TEXT, at the attack byte), how many classes it has (sca/leakage_models.get_num_classes), and where
    both were read (`where`; raw_sca: the sca block as written, a sweep list naming every value)."""
    from sca.dataset_types import TARGET_TEXT, label_target
    from sca.leakage_models import get_num_classes, normalize_leakage_model

    sca = cfg.get("sca") or {}
    model = normalize_leakage_model(_first(sca.get("leakage_model", "ID")))
    byte = _attack_byte(cfg, _entries_of(cfg)[0])          # cfg["data"] is the run's entry here
    target = label_target(cfg)
    expression = TARGET_TEXT.get(target, target).split(":")[0].split(", any byte")[0].replace("[b]", f"[{byte}]")
    swept = {k: [str(v) for v in raw_sca[k]] for k in ("leakage_model", "attack_byte")
             if isinstance(raw_sca.get(k), (list, tuple)) and len(raw_sca[k]) > 1}
    return {"leakage_model": model, "attack_byte": byte, "target": target, "classes": int(get_num_classes(model)),
            "expression": expression if model == "ID" else f"{model}({expression})", "from": where, "swept": swept}


def _seed(cfg: dict) -> int:
    return int(_first(cfg.get("seed", 42)))


def _fitted(cfg: dict, entry: dict, path: Path, selection: Any, policy: dict) -> Any:
    """The entry's fitted scaler, fitted on its profiling traces (the run's way, utils/trace_scaler.fit), kept
    in the temp folder by the entry's settings so the next trace does not fit it again."""
    import h5py

    from sca.ascad_data_loader import _get_raw_layout, _range_pair, _read_trace_chunk, profiling_indices
    from utils import trace_scaler

    key = hashlib.sha256(json.dumps([entry, str(path), path.stat().st_mtime], sort_keys=True, default=str).encode()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / f"traceflow-trace-view-scaler-{key}.json"
    if cache.is_file():
        return trace_scaler.from_record(json.loads(cache.read_text(encoding="utf-8")))
    lo, hi = _range_pair(entry, "profiling_range")
    with h5py.File(path, "r") as handle:
        traces = handle["Profiling_traces"]["traces"] if _get_raw_layout(handle) != "flat" else handle["traces"]
        rows, _ = _read_trace_chunk(traces, profiling_indices(entry), selection=selection, desync=0)
    pipeline = trace_scaler.fit(policy, rows)
    fitted_on = {"profiling_range": [lo, hi], "desync": 0}
    if (entry.get("profiling_split") or {}).get("enabled") is True:   # the rows the run trains on (ascad_data_loader)
        fitted_on["profiling_split"] = entry["profiling_split"]
    record = trace_scaler.to_record(pipeline, policy=policy, fitted_on=fitted_on)
    cache.write_text(json.dumps(record), encoding="utf-8")
    return pipeline


def design(project: str, layers: list, architecture: str, dataset: Optional[str], trace: int, to: Optional[int] = None) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import h5py
    import numpy as np
    import torch

    from core.framework_switch import keras_input_shape
    from core.model import _input_length, build_model
    from sca.ascad_data_loader import (_get_attack_byte, _get_raw_layout, _get_trace_normalization_cfg, _normalize_trace,
                                       _range_pair, _read_trace_chunk, _resolve_trace_selection, _target_points_for_attack_byte)
    from utils import trace_scaler

    from services.visualization.vscode_monitor.model_check import PLANNED

    cfg = _project_cfg(project)
    entry = next((e for e in _entries_of(cfg) if dataset is None or e.get("name") == dataset), None)
    out: dict[str, Any] = {"weights": "untrained", "error": None, "at": None, "layers": []}
    if entry is None:
        out["error"] = f"no data entry {dataset!r} in params.yaml"
        return out
    path, why = _database(cfg, entry)
    if why:
        out["error"] = why
        return out
    # one concrete run, as model_check builds it: this data entry, the sweep lists' first values
    raw_sca = copy.deepcopy(cfg.get("sca") or {})
    cfg["data"] = entry
    cfg["sca"] = {k: _first(v) for k, v in (cfg.get("sca") or {}).items()}
    cfg["sca"]["attack_byte"] = _attack_byte(cfg, entry)   # empty in the file: the entry's own byte
    cfg["model_id"] = PLANNED
    cfg["model"] = [{"model_id": PLANNED, "architecture": architecture, "architecture_args": {"layers": copy.deepcopy(layers)}}]
    cfg.setdefault("deep_learning", {})["framework"] = "pytorch"
    seed = _seed(cfg)
    lo, hi = _range_pair(entry, "attack_range")
    numbers, why = _span(trace, to, hi - lo, f"the attack set of {entry.get('name')}")
    if why:
        out["error"] = why
        return out
    selection = _resolve_trace_selection(_target_points_for_attack_byte(entry, attack_byte=_get_attack_byte(cfg)))
    desync = int(entry.get("attack_desync") or 0)
    norm = _get_trace_normalization_cfg(cfg)
    policy = trace_scaler.fitted_policy(cfg)
    scaler = _fitted(cfg, entry, path, selection, policy) if policy is not None else None
    rows, offsets = [], []
    with h5py.File(path, "r") as handle:
        flat = _get_raw_layout(handle) == "flat"
        traces = handle["traces"] if flat else handle["Attack_traces"]["traces"]
        for k in numbers:                                # each trace its own seeded desync offset
            row, offset = _read_trace_chunk(traces, np.array([lo + k]), selection=selection, desync=desync,
                                            rng=np.random.default_rng(seed + 1337 + k))
            rows.append(_normalize_trace(np.asarray(row[0], dtype=np.float32), row.dtype, norm))
            offsets.append(int(offset[0]))
    values = np.stack(rows)
    if scaler is not None:
        values = np.asarray(scaler.transform(values.astype(np.float64)), dtype=np.float32)
    channels, length = keras_input_shape(layers, _input_length(cfg))
    out["target"] = _target(cfg, raw_sca, "params.yaml")
    out["source"] = {"dataset": entry.get("name"), "type": str(entry.get("type") or "ASCADf"), "trace": numbers[0], "to": numbers[-1],
                     "row": int(lo + numbers[0]), "row_to": int(lo + numbers[-1]),
                     "file": _shown_path(path), "group": "traces" if flat else "Attack_traces/traces",
                     "samples": [int(selection.start + offsets[0]), int(selection.end_exclusive + offsets[0])] if len(numbers) == 1
                     else [int(selection.start), int(selection.end_exclusive)],
                     "desync_offset": offsets[0] if len(numbers) == 1 else None, "desync": desync, "length": int(selection.output_length),
                     "normalization": str(norm.get("mode", "none")) + (" (fitted on its profiling traces)" if policy else "")}
    spread = _Spread()
    spread.add(values)
    out["input"] = {"length": int(values.shape[1]), **spread.row()}
    if channels * length != values.shape[1]:
        out["error"] = f"the design takes {channels} x {length} samples; the trace window has {values.shape[1]}"
        return out
    torch.manual_seed(seed)                              # the same untrained weights on every edit
    try:
        net = build_model(cfg, PLANNED)
    except Exception as exc:  # noqa: BLE001 - the builder's own message is the answer
        out["error"] = str(exc)
        return out
    net.eval()
    out.update(_through(net, layers, values.reshape(len(numbers), channels, length), None))
    return out


def _checkpoint(cfg: dict, name: str) -> dict[str, Any]:
    """The checkpoint make test loads (evaluation.test.load_model, utils/checkpoint_policy): the manifest
    loaded and checked by checkpoint_policy, its file found under this machine's projects folder (the manifest
    names the container's), its SHA-256 the manifest's, its seal intact."""
    from utils.checkpoint_policy import (CheckpointPolicyError, checkpoint_paths, file_sha256, get_requested_test_policy,
                                         get_validation_checkpointing, load_checkpoint_manifest)
    from utils.seal import check as check_seal

    paths = checkpoint_paths(cfg, model_name=name)
    manifest = load_checkpoint_manifest(cfg, model_name=name)
    requested = get_requested_test_policy(cfg)
    best = requested == "best_validation" and get_validation_checkpointing(cfg)
    block = manifest.get("selection") if best else manifest["final"]
    if not isinstance(block, dict):
        raise CheckpointPolicyError("best_validation was requested, but the manifest has no selected checkpoint")
    raw, digest = (block.get("best_validation_checkpoint"), block.get("best_validation_checkpoint_sha256")) if best \
        else (block.get("checkpoint"), block.get("checkpoint_sha256"))
    expected = (paths.best_validation if best else paths.final_epoch).resolve()
    tail = expected.relative_to(Path(cfg["project"]["root"]).resolve().parent)    # <project>/artifacts/...
    if not str(raw or "").endswith("/" + str(tail)):
        raise CheckpointPolicyError(f"the manifest names {raw}, not this run's {tail}")
    if file_sha256(expected) != digest:
        raise CheckpointPolicyError(f"{expected.name}: SHA-256 differs from its manifest's")
    check_seal(expected)
    return {"path": str(expected), "policy": "best_validation" if best else "final_epoch", "requested": requested,
            "epoch": block.get("epoch"), "fallback": None if best or requested == "final_epoch"
            else "validation checkpointing was off: the final epoch's weights"}


def run(project: str, name: str, trace: int, to: Optional[int] = None) -> dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import numpy as np
    import torch

    from core.model import build_model
    from sca.ascad_data_loader import load_split_dataset
    from sca.test_request import load_snapshot, resolve_models
    from utils.path_utils import resolve_paths_dict
    from utils.trace_scaler import scale_test

    out: dict[str, Any] = {"weights": "trained", "run": name, "error": None, "at": None, "layers": []}
    run_ = resolve_models(project, [name])[0]
    cfg = load_snapshot(project, run_)
    cfg["paths_abs"] = resolve_paths_dict(cfg["paths"], cfg, inject=True, mkdir=False)
    model = next((m for m in cfg.get("model") or [] if isinstance(m, dict) and m.get("model_id") == cfg.get("model_id")), None)
    layers = ((model or {}).get("architecture_args") or {}).get("layers") or []
    entry = (_entries_of(cfg) or [{}])[0]
    out["checkpoint"] = _checkpoint(cfg, name)
    ds = load_split_dataset(cfg, "test.h5", split="attack", return_metadata=True)
    scale_test(cfg, ds)                                  # the run's own scaler (fitted mode), as make test reads it
    numbers, why = _span(trace, to, len(ds), f"{name}'s test.h5")
    if why:
        out["error"] = why
        return out
    items = [ds[k] for k in numbers]
    xs = np.stack([item[0].detach().float().numpy() for item in items])
    xs = xs.reshape(len(numbers), *(xs.shape[1:] if xs.ndim == 3 else (1, -1)))
    labels = [int(item[1]) for item in items]
    out["target"] = _target(cfg, cfg.get("sca") or {}, f"{name}'s training snapshot")
    meta = items[0][2] if len(items[0]) > 2 and isinstance(items[0][2], dict) else {}
    if len(numbers) == 1 and out["target"]["target"] == "sbox" and "plaintext" in meta and "key" in meta:
        b = out["target"]["attack_byte"]                 # the bytes the true class is made of
        out["bytes"] = {"plaintext": int(meta["plaintext"][b]), "key": int(meta["key"][b])}
    test_h5 = Path(cfg["paths_abs"]["data"]) / "test.h5"
    desync = int(entry.get("attack_desync") or 0)
    offsets = None
    if desync:                                           # the shifts this run drew and stored when it made its test.h5
        import h5py

        with h5py.File(test_h5, "r") as handle:
            stored = handle["Attack_traces"].get("desync_offsets")
            offsets = [int(v) for v in stored[numbers[0]:numbers[-1] + 1]] if stored is not None else None
    out["source"] = {"dataset": entry.get("name"), "type": str(entry.get("type") or "ASCADf"), "trace": numbers[0], "to": numbers[-1],
                     "row": numbers[0], "row_to": numbers[-1], "file": _shown_path(test_h5), "group": "Attack_traces/traces",
                     "desync": desync, "desync_offset": offsets[0] if offsets and len(numbers) == 1 else None,
                     "length": int(xs[0].size), "normalization": str((entry.get("trace_normalization") or {}).get("mode", "none"))}
    spread = _Spread()
    spread.add(xs.reshape(len(numbers), -1))
    out["input"] = {"length": int(xs[0].size), **spread.row()}
    torch.manual_seed(_seed(cfg))
    net = build_model(cfg, cfg.get("model_id"))
    state = torch.load(out["checkpoint"]["path"], map_location="cpu")
    net.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state, strict=True)
    net.eval()
    out.update(_through(net, layers, xs, labels))
    return out


_OUTCOME = ("key_recovered", "recovered_key", "recovered_key_rank")


def _training(cfg: dict) -> dict[str, Any]:
    """How a run was trained, from its snapshot: cross_validation (k folds over the profiling traces, then a refit on
    all of them) or the hold-out split (train.h5 = the profiling traces, val.h5 and test.h5 = attack-trace ranges)."""
    from sca.ascad_data_loader import _range_pair, profiling_indices

    cv = cfg.get("cross_validation") or {}
    entry = (_entries_of(cfg) or [{}])[0]

    def count(key: str) -> Optional[int]:
        try:
            lo, hi = _range_pair(entry, key)
            return hi - lo
        except (TypeError, ValueError):
            return None

    attack = cfg.get("perform_attack") or {}
    try:
        profiling = int(profiling_indices(entry).size)       # profiling_range, or the part its profiling_split trains on
    except (TypeError, ValueError):
        profiling = None
    out = {"cv": bool(cv.get("enabled")), "profiling": profiling, "test": count("test_range"),
           "attack": {p: {k: (attack.get(p) or {}).get(k) for k in ("attack_times", "shuffle", "attack_distribution", "attack_trace_budget")}
                      for p in ("validation", "test")}}
    if out["cv"]:
        out.update(n_folds=int(_first(cv.get("n_folds", 0)) or 0), strategy=str(_first(cv.get("strategy", "k_fold"))),
                   # cross_validation.refit.enabled (absent: true): false trains no final model
                   refit=not (isinstance(cv.get("refit"), dict) and cv["refit"].get("enabled") is False))
    else:
        out["validation"] = count("validation_range")
    return out


def _report(path: Path) -> Optional[dict[str, Any]]:
    """What a key-recovery report recorded about its attack (sealed; None when the run has none): its traces, the key
    space, the rank that counts as recovered and whether that rank had to hold to the last trace."""
    from utils.seal import load_json

    if not path.is_file():
        return None
    r = load_json(path) or {}
    rec, key = r.get("recovery") or {}, r.get("key") or {}
    return {"traces": (r.get("trace_population") or {}).get("used"), "keyspace": (r.get("target") or {}).get("keyspace_size"),
            "recovery_rank": rec.get("recovery_rank"), "stable_required": key.get("stable_recovery_required")}


def _reports(cfg: dict, name: str, training: dict) -> dict[str, Any]:
    """The run's key-recovery reports, found where it wrote them and never created: each fold's validation_best.json
    (CV), else its own validation_best.json, and its test.json."""
    from utils.cross_validation import get_fold_artifact_paths, resolve_cross_validation_paths
    from utils.path_utils import resolve_path

    paths = cfg.get("paths") or {}
    kr = Path(resolve_path(paths["key_recovery"], cfg, model_name=name)) if paths.get("key_recovery") \
        else Path(resolve_path(paths["metrics"], cfg, model_name=name)) / "key_recovery"
    out: dict[str, Any] = {"folds": {}, "validation": None, "test": _report(kr / "test.json")}
    if training["cv"]:
        cv_paths = resolve_cross_validation_paths(dict(cfg), create=False)
        n = training.get("n_folds") or 0
        for i in range(n):
            out["folds"][i + 1] = _report(get_fold_artifact_paths(cv_paths, fold_index=i, n_folds=n, create=False)["key_recovery"] / "validation_best.json")
    else:
        out["validation"] = _report(kr / "validation_best.json")
    return out


def recovery(project: str, name: str) -> dict[str, Any]:
    from sca.test_request import load_snapshot, resolve_models
    from utils.path_utils import resolve_paths_dict
    from utils.seal import load_json

    root = Path(_project_cfg(project)["project"]["root"])
    run_ = resolve_models(project, [name])[0]
    snap = load_snapshot(project, run_)
    snap["paths_abs"] = resolve_paths_dict(snap["paths"], snap, inject=True, mkdir=False)
    training = _training(snap)
    reports = _reports(snap, name, training)
    model_hash = str(run_["model_hash"])
    registry = load_json(root / "artifacts" / "run_registry.json") or {}
    entry = registry.get(model_hash) if isinstance(registry.get(model_hash), dict) else {}
    tests_path = root / "artifacts" / "test_registry.json"
    tests = load_json(tests_path) if tests_path.is_file() else {}
    test = tests.get(model_hash) if isinstance(tests.get(model_hash), dict) else {}
    folds = entry.get("fold_key_recovery")
    out: dict[str, Any] = {"run": name, "model_hash": model_hash, "true_key": entry.get("true_key"), "error": None, "training": training,
                           "cv": isinstance(folds, list),
                           "folds": [{k: f.get(k) for k in ("fold", *_OUTCOME, "stable_rank_1")} for f in folds or [] if isinstance(f, dict)],
                           "validation": None if isinstance(folds, list) else {k: entry.get(k) for k in _OUTCOME}}
    if isinstance(folds, str):                           # a CV run whose folds were not attacked says why
        out["cv"], out["validation"], out["folds_note"] = True, None, folds
    out["test"] = {k: test.get(k) for k in ("status", *_OUTCOME, "true_key", "status_reason", "finished_at")} if test else None
    external = test.get("external_tests") if isinstance(test.get("external_tests"), dict) else {}
    for f in out["folds"]:                             # each fold's traces, as its report recorded them
        f["traces"] = (reports["folds"].get(f.get("fold")) or {}).get("traces")
    if out["test"] is not None:
        out["test"]["traces"] = (reports["test"] or {}).get("traces")
    found = [r for r in [*reports["folds"].values(), reports["validation"], reports["test"]] if r]
    out["rules"] = {k: next((r[k] for r in found if r.get(k) is not None), None) for k in ("keyspace", "recovery_rank", "stable_required")}
    out["external"] = sorted(({"key": k, **{f: r.get(f) for f in ("index", "status", *_OUTCOME, "status_reason", "finished_at", "settings")}}
                              for k, r in external.items() if isinstance(r, dict)), key=lambda r: r.get("index") or 0)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One trace through a model, layer by layer (the planner's lines).")
    parser.add_argument("command", choices=["entries", "runs", "design", "run", "recovery"])
    parser.add_argument("--project", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--architecture", default="CNN")
    parser.add_argument("--run", default=None)
    parser.add_argument("--trace", type=int, default=0)
    parser.add_argument("--to", type=int, default=None, help="a range: traces --trace to --to (their mean and band)")
    args = parser.parse_args(argv)
    try:
        out = (entries(args.project) if args.command == "entries"
               else runs(args.project, str(args.model_id)) if args.command == "runs"
               else design(args.project, json.load(sys.stdin), args.architecture, args.dataset, args.trace, args.to) if args.command == "design"
               else recovery(args.project, str(args.run)) if args.command == "recovery"
               else run(args.project, str(args.run), args.trace, args.to))
    except Exception as exc:  # noqa: BLE001
        out = {"error": f"{type(exc).__name__}: {exc}", "layers": []}
    print(json.dumps(out))
    return 0 if not out.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
