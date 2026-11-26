# services/model_filter/model_filter.py
"""
Model filter: find the trained models of a project that have given parameters.

    make find_model ENV=dev PROJECT=DLSCA_ASCAD [FILTER=config/other.yaml]

Reads projects/<P>/config/model_filter.yaml, walks every run in
projects/<P>/artifacts/run_map.jsonl and keeps the runs whose parameters
satisfy the `where` conditions. The matching model names are written to
projects/<P>/artifacts/filtered_models.yaml (overwritten on every run), ready
to be copied into a graph job's `models` list. Metrics play no part: this
selects models by the traits they were trained with, not by how well they did.

Where a parameter is looked up (first hit wins):
  1. the run_map entry itself      (env, status, model_id, trial_index, run_session_id, ...)
  2. its run_registry.json entry   (batches, batch_ids, started_at, finished_at,
                                    status_reason, key_recovered, ...)
  3. its `sampled` sweep point     (model.*, trainer.*, sca.*)
  4. the run's training snapshot   (data.*, cross_validation.*, perform_attack.*, ...)
  5. the run's device record       (run_device.name, run_device.software.torch,
                                    run_device.probe.seconds_per_step, ...: device.yaml)
so any key of the resolved experiment config can be a condition. A truncated
or edited snapshot stops the filter (ModelFilterError); a missing one only
leaves its keys unmatched.

`equals: "a|b"` means any of a, b (an `in:` list in that order); `not_equals:
"a|b"` means none of them (`not_in:`). `contains: "a|b"` (or `[a, b]`) means the
value contains a or b, e.g. `path: batches, contains: "fig6|fig7"`. An
alternative that no run has is named in a warning.

Order of the output: the matches follow the `in:` lists of `where`, in the
order those lists name their values, the first condition first (so
`trainer.epochs in: [10, 25, 50]` lists the 10-epoch runs first). Runs that
the lists do not tell apart keep their training order (run_map.jsonl).

Author: Shraddha Mishra
"""

from __future__ import annotations

from services.side_channel_core.utils.config_errors import ConfigError
import json
import logging
import math
import os
import re
from numbers import Real
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import yaml

from services.side_channel_core.utils.data_reproducibility import read_params_snapshot
from services.side_channel_core.utils.logger import console
from services.side_channel_core.utils.run_map import load_run_map
from services.side_channel_core.utils.timestamps import now_iso
from services.side_channel_core.utils.type_coerce import coerce_scalar
from services.side_channel_core.utils.seal import seal_text
from services.side_channel_core.utils.seal import load_json

logger = logging.getLogger("ModelFilter")

CONFIG_KEY = "model_filter"
OUTPUT_FILE = "filtered_models.yaml"
MATCH_MODES = ("all", "any")
QUANTIFIERS = ("any", "all")
OPERATORS = (
    "equals", "not_equals", "in", "not_in",
    "gt", "gte", "lt", "lte", "between",
    "matches", "contains", "exists", "count",
)


class ModelFilterError(ConfigError):     # a configuration error (utils/config_errors.py), a ValueError as before
    """The filter config cannot be used as written."""


# =============================================================================
# Config
# =============================================================================
def load_filter_config(path: str | Path) -> dict[str, Any]:
    """Validate projects/<P>/config/model_filter.yaml and fill defaults."""
    path = Path(path)
    if not path.is_file():
        raise ModelFilterError(
            f"Model filter config not found: {path}. Copy "
            "services/side_channel_core/config/blueprint/model_filter.yaml"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping) or CONFIG_KEY not in raw:
        raise ModelFilterError(
            f"{path} must contain a top-level '{CONFIG_KEY}:' block (the file is "
            f"{'empty' if not raw else 'missing it'}); see "
            "services/side_channel_core/config/blueprint/model_filter.yaml"
        )
    block = raw[CONFIG_KEY] or {}
    if not isinstance(block, Mapping):
        raise ModelFilterError(f"'{CONFIG_KEY}' must be a mapping")
    unknown = sorted(set(block) - {"match", "status", "env", "where"})
    if unknown:
        raise ModelFilterError(f"Unknown key(s) in {CONFIG_KEY}: {unknown}; allowed: match, status, env, where")

    match = str(block.get("match") or "all").lower()
    if match not in MATCH_MODES:
        raise ModelFilterError(f"{CONFIG_KEY}.match must be one of {MATCH_MODES}, got {match!r}")
    status = block.get("status", "DONE")
    status = None if status in (None, "", "any") else str(status).upper()
    env = block.get("env")
    env = None if env in (None, "") else str(env)

    where = block.get("where")
    if not isinstance(where, list) or not where:
        raise ModelFilterError(f"{CONFIG_KEY}.where must be a non-empty list of conditions")
    conditions = [_validate_condition(c, index=i) for i, c in enumerate(where, start=1)]
    return {"match": match, "status": status, "env": env, "where": conditions, "source": str(path)}


def _validate_condition(cond: Any, *, index: int) -> dict[str, Any]:
    where = f"where[{index}]"
    if not isinstance(cond, Mapping):
        raise ModelFilterError(f"{where} must be a mapping with `path` and one operator")
    path = cond.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ModelFilterError(f"{where}.path is required (a dotted parameter path)")
    quantifier = str(cond.get("quantifier") or "any").lower()
    if quantifier not in QUANTIFIERS:
        raise ModelFilterError(f"{where}.quantifier must be one of {QUANTIFIERS}")
    ops = [k for k in cond if k in OPERATORS]
    extra = sorted(set(cond) - set(OPERATORS) - {"path", "quantifier"})
    if extra:
        raise ModelFilterError(f"{where}: unknown key(s) {extra}; operators: {', '.join(OPERATORS)}")
    if len(ops) != 1:
        raise ModelFilterError(f"{where} ({path}) needs exactly one operator, got {ops or 'none'}")
    op = ops[0]
    value = cond[op]
    if op in ("equals", "not_equals") and isinstance(value, str) and "|" in value:
        # "a|b" names alternatives: equals -> in, not_equals -> not_in, in the
        # order written (which also orders the output, as an in: list does).
        names = [part.strip() for part in value.split("|")]
        if not all(names):
            raise ModelFilterError(f"{where}.{op} {value!r} has an empty alternative between '|'")
        op, value = ("in" if op == "equals" else "not_in"), names
    if op == "contains" and isinstance(value, str) and "|" in value:
        # "a|b": the value at the path contains a or b (in that order).
        value = [part.strip() for part in value.split("|")]
        if not all(value):
            raise ModelFilterError(f"{where}.contains {cond[op]!r} has an empty alternative between '|'")
    if op == "contains" and isinstance(value, list) and not value:
        raise ModelFilterError(f"{where}.contains takes a value, \"a|b\" or a non-empty list")
    if op in ("in", "not_in") and not isinstance(value, list):
        raise ModelFilterError(f"{where}.{op} takes a list")
    if op in ("gt", "gte", "lt", "lte"):
        value = coerce_scalar(value)          # lt: 1e-4, a string to YAML 1.1 (see _same)
    if op == "between" and isinstance(value, list):
        value = [coerce_scalar(v) for v in value]
    if op == "between" and not (isinstance(value, list) and len(value) == 2 and all(_is_num(v) for v in value)):
        raise ModelFilterError(f"{where}.between takes [low, high]")
    if op in ("gt", "gte", "lt", "lte") and not _is_num(value):
        raise ModelFilterError(f"{where}.{op} takes a number")
    if op == "exists" and not isinstance(value, bool):
        raise ModelFilterError(f"{where}.exists takes true or false")
    if op == "matches":
        try:
            re.compile(str(value))
        except re.error as exc:
            raise ModelFilterError(f"{where}.matches is not a valid regular expression: {exc}") from exc
    if op == "count":
        if _is_num(value):
            value = {"min": int(value), "max": int(value)}
        elif not (isinstance(value, Mapping) and set(value) <= {"min", "max"} and value):
            raise ModelFilterError(f"{where}.count takes an integer or {{min, max}}")
    return {"path": path.strip(), "op": op, "value": value, "quantifier": quantifier}


# =============================================================================
# Parameter lookup
# =============================================================================
def _is_num(v: Any) -> bool:
    return isinstance(v, Real) and not isinstance(v, bool)


def _walk(node: Any, segments: list[str]) -> Iterator[Any]:
    """Yield every value at a dotted path; `*` expands a list, digits index it."""
    if not segments:
        yield node
        return
    head, rest = segments[0], segments[1:]
    if head in ("*", "[*]"):
        if isinstance(node, list):
            for item in node:
                yield from _walk(item, rest)
        return
    if isinstance(node, list):
        if head.isdigit() and int(head) < len(node):
            yield from _walk(node[int(head)], rest)
        return
    if isinstance(node, Mapping) and head in node:
        yield from _walk(node[head], rest)


class RunView:
    """One run of run_map.jsonl with lazy access to its training snapshot."""

    def __init__(
        self,
        model_hash: str,
        entry: Mapping[str, Any],
        project_root: Path,
        registry_entry: Optional[Mapping[str, Any]] = None,
    ):
        self.model_hash = model_hash
        self.entry = dict(entry)
        self.registry_entry = dict(registry_entry or {})
        self.model_name = str(entry.get("model_name") or "")
        self.project_root = project_root
        sampled = dict(entry.get("sampled") or {})
        model = sampled.get("model")
        if isinstance(model, list) and len(model) == 1 and isinstance(model[0], Mapping):
            sampled["model"] = model[0]  # one model block per run: allow `model.architecture`
        self.sampled = sampled
        self._snapshot: Optional[dict] = None
        self._device: Optional[dict] = None

    def _run_dir(self) -> Path:
        rel = self.entry.get("path") or f"artifacts/{self.entry.get('env', 'dev')}/generated_models/{self.model_name}"
        return self.project_root / str(rel)

    @property
    def device(self) -> dict:
        """{"run_device": ...} from the run's device.yaml; empty for runs made before it existed."""
        if self._device is None:
            from services.side_channel_core.utils.device_record import RECORD_KEY, read_device_record

            record = read_device_record(self._run_dir())
            self._device = {RECORD_KEY: record} if record else {}
        return self._device

    @property
    def snapshot(self) -> dict:
        if self._snapshot is None:
            path = self._run_dir() / "config" / f"{self.model_name}_params.yaml"
            if path.is_file():
                # A truncated or edited snapshot raises instead of being
                # filtered on partial data (read_params_snapshot logs the hint).
                try:
                    self._snapshot = read_params_snapshot(path)
                except ValueError as exc:
                    raise ModelFilterError(f"{self.model_name} ({self.model_hash}): {exc}") from exc
            else:
                self._snapshot = {}
                logger.debug("No training snapshot for %s at %s", self.model_name, path)
        return self._snapshot

    def values(self, path: str) -> list[Any]:
        """All values at `path`: run_map entry, registry entry, sweep point, snapshot, then device record."""
        segments = path.split(".")
        if segments[0] == "run_device":
            return list(_walk(self.device, segments))
        for source in (self.entry, self.registry_entry, self.sampled):
            found = list(_walk(source, segments))
            if found:
                return found
        return list(_walk(self.snapshot, segments))


# =============================================================================
# Conditions
# =============================================================================
def _test_one(op: str, value: Any, actual: Any) -> bool:
    if op == "equals":
        return _same(actual, value)
    if op == "not_equals":
        return not _same(actual, value)
    if op == "in":
        return any(_same(actual, v) for v in value)
    if op == "not_in":
        return not any(_same(actual, v) for v in value)
    if op in ("gt", "gte", "lt", "lte", "between"):
        if not _is_num(actual) or (isinstance(actual, float) and math.isnan(actual)):
            return False
        if op == "gt":
            return actual > value
        if op == "gte":
            return actual >= value
        if op == "lt":
            return actual < value
        if op == "lte":
            return actual <= value
        return value[0] <= actual <= value[1]
    if op == "matches":
        return re.fullmatch(str(value), str(actual)) is not None
    if op == "contains":
        if isinstance(value, list):                     # alternatives: any of them
            return any(_test_one("contains", alternative, actual) for alternative in value)
        if isinstance(actual, list):
            return any(_same(item, value) for item in actual)
        return isinstance(actual, str) and str(value) in actual
    if op == "count":
        if not isinstance(actual, (list, Mapping, str)):
            return False
        n = len(actual)
        return value.get("min", n) <= n <= value.get("max", n)
    raise ModelFilterError(f"Unknown operator {op!r}")


def _same(a: Any, b: Any) -> bool:
    # YAML 1.1 reads 1e-5 (no dot) as a string; training coerces it to the float its runs store
    # (utils/type_coerce.coerce_scalar), so a numeric string meets a number as that number.
    if _is_num(a) and isinstance(b, str):
        b = coerce_scalar(b)
    elif isinstance(a, str) and _is_num(b):
        a = coerce_scalar(a)
    if _is_num(a) and _is_num(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    return a == b and type(a) is type(b)


def evaluate(run: RunView, cond: Mapping[str, Any]) -> tuple[bool, list[Any]]:
    """(condition holds for this run, values found at the path)."""
    found = run.values(cond["path"])
    op, value = cond["op"], cond["value"]
    if op == "exists":
        return (bool(found) is value), found
    if not found:
        return False, found
    results = [_test_one(op, value, actual) for actual in found]
    return (all(results) if cond["quantifier"] == "all" else any(results)), found


# =============================================================================
# Run
# =============================================================================
def _warn_unmatched_alternatives(conditions: list, runs: list) -> None:
    """Name every value of an in: / contains condition that no run has (a typo, not labelled or not trained yet)."""
    for cond in conditions:
        if cond["op"] not in ("in", "contains"):
            continue
        asked = cond["value"] if isinstance(cond["value"], list) else [cond["value"]]
        found = [v for run in runs for v in run.values(cond["path"])]
        for wanted in asked:
            if cond["op"] == "in":
                seen = any(_same(actual, wanted) for actual in found)
            else:
                seen = any(_test_one("contains", wanted, actual) for actual in found)
            if not seen:
                logger.warning("model_filter: %s %s %r matched no run (typo, not labelled, or not trained yet?)",
                               cond["path"], cond["op"], wanted)


def _asked_rank(cond: Mapping[str, Any], found: list[Any]) -> int:
    """
    Position in an `in:` list of the first value this run has there; every
    other operator names no order (0). A run with none of the values (match:
    any) comes after those that have one.
    """
    asked = cond["value"]
    if cond["op"] == "in":
        hit = lambda wanted: any(_same(actual, wanted) for actual in found)
    elif cond["op"] == "contains" and isinstance(asked, list):
        hit = lambda wanted: any(_test_one("contains", wanted, actual) for actual in found)
    else:
        return 0
    for i, wanted in enumerate(asked):
        if hit(wanted):
            return i
    return len(asked)


def _read_run_registry(project_root: Path) -> dict[str, Any]:
    """artifacts/run_registry.json as written (read only; no healing write)."""
    path = Path(project_root) / "artifacts" / "run_registry.json"
    if not path.is_file():
        return {}
    try:
        data = load_json(path)  # utils/seal.py: verified, seal key dropped
    except json.JSONDecodeError as exc:
        raise ModelFilterError(f"{path} is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else {}


def find_runs(
    project_root: str | Path,
    where: list[Mapping[str, Any]],
    *,
    env: str | None = None,
    status: str | None = None,
    model_names: Optional[list[str]] = None,
) -> list[RunView]:
    """
    The runs of run_map.jsonl that satisfy every `where` condition (the
    conditions of model_filter.yaml), in training order. env, status and
    model_names narrow the candidates first. Used by graph jobs to pick the
    reference run of `relative:` by its configuration.
    """
    project_root = Path(project_root)
    conditions = [_validate_condition(cond, index=i) for i, cond in enumerate(where or [])]
    if not conditions:
        raise ModelFilterError("where needs at least one condition")
    registry = _read_run_registry(project_root)
    out: list[RunView] = []
    for model_hash, entry in load_run_map(project_root).items():
        if not isinstance(entry, Mapping):
            continue
        if env and str(entry.get("env") or "") != env:
            continue
        if status and str(entry.get("status") or "").upper() != status.upper():
            continue
        run = RunView(model_hash, entry, project_root,
                      registry.get(model_hash) if isinstance(registry.get(model_hash), Mapping) else None)
        if model_names is not None and run.model_name not in model_names and run.model_hash not in model_names:
            continue
        if all(evaluate(run, cond)[0] for cond in conditions):
            out.append(run)
    return out


def _display(values: list[Any]) -> Any:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def run_model_filter(
    config_path: str | Path,
    project: str,
    env: str | None = None,
    projects_root: str | Path = "projects",
) -> dict[str, Any]:
    cfg = load_filter_config(config_path)
    project_root = Path(projects_root) / project
    run_map = load_run_map(project_root)
    if not run_map:
        raise ModelFilterError(f"No runs in {project_root / 'artifacts' / 'run_map.jsonl'}; train a model first")

    wanted_env = cfg["env"] or env
    wanted_env = None if wanted_env in (None, "", "any") else str(wanted_env)
    registry = _read_run_registry(project_root)
    runs = [
        RunView(h, e, project_root, registry.get(h) if isinstance(registry.get(h), Mapping) else None)
        for h, e in run_map.items()
        if isinstance(e, Mapping)
    ]
    candidates = [
        r for r in runs
        if (cfg["status"] is None or str(r.entry.get("status") or "").upper() == cfg["status"])
        and (wanted_env is None or str(r.entry.get("env") or "") == wanted_env)
    ]

    matched, table_rows = [], []
    for run in candidates:
        verdicts, values, rank = [], {}, []
        for cond in cfg["where"]:
            ok, found = evaluate(run, cond)
            verdicts.append(ok)
            values[cond["path"]] = _display(found)
            rank.append(_asked_rank(cond, found))
        hit = all(verdicts) if cfg["match"] == "all" else any(verdicts)
        table_rows.append((run, hit, values, tuple(rank)))

    _warn_unmatched_alternatives(cfg["where"], candidates)

    # The order the filter asks for; sorted() is stable, so ties keep the
    # training order of run_map.jsonl.
    table_rows.sort(key=lambda row: (not row[1], row[3]))
    for run, hit, values, _rank in table_rows:
        if hit:
            matched.append({
                "model_name": run.model_name,
                "model_hash": run.model_hash,
                "model_id": run.entry.get("model_id"),
                "env": run.entry.get("env"),
                "values": values,
            })

    output = {
        "filter": str(cfg["source"]),
        "generated_at": now_iso(),
        "project": project,
        "env": wanted_env or "any",
        "status": cfg["status"] or "any",
        "match": cfg["match"],
        "where": [{"path": c["path"], c["op"]: c["value"]} for c in cfg["where"]],
        "candidates": len(candidates),
        "count": len(matched),
        "models": [m["model_name"] for m in matched],
        "details": matched,
    }
    out_path = write_output(project_root, output)
    _print_table(cfg, table_rows, out_path)
    return output


def write_output(project_root: Path, output: Mapping[str, Any]) -> Path:
    """Overwrite artifacts/filtered_models.yaml atomically."""
    out_dir = project_root / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / OUTPUT_FILE
    header = (
        f"# {Path('projects') / project_root.name / 'artifacts' / OUTPUT_FILE}\n"
        "# Written by `make find_model`; overwritten on every run. Copy `models`\n"
        "# into a graph job's `models` list to plot these models together.\n"
    )
    tmp = out_path.with_name(f".{out_path.name}.tmp")
    tmp.write_text(seal_text(out_path, header + yaml.safe_dump(dict(output), sort_keys=False, allow_unicode=True)), encoding="utf-8")
    os.replace(tmp, out_path)
    return out_path


def _print_table(cfg: Mapping[str, Any], rows: list, out_path: Path) -> None:
    from rich.table import Table

    paths = [c["path"] for c in cfg["where"]]
    hits = sum(1 for _, hit, _, _ in rows if hit)
    table = Table(
        title=f"Model filter: {hits} of {len(rows)} run(s) match ({cfg['match']} of {len(paths)} condition(s))",
        title_style="bold",
        header_style="bold bright_cyan",
    )
    table.add_column("model_name", no_wrap=True)
    table.add_column("model_hash", no_wrap=True)
    table.add_column("match", no_wrap=True)
    for p in paths:
        table.add_column(p)
    # Matches in the output order; the rest after them, by name.
    for run, hit, values, _rank in sorted(rows, key=lambda r: (not r[1], r[3] if r[1] else r[0].model_name)):
        style = "green" if hit else "dim"
        table.add_row(
            run.model_name, run.model_hash, "yes" if hit else "no",
            *[_cell(values[p]) for p in paths], style=style,
        )
    console.print(table)
    if hits == 0:
        console.print("[yellow]No model matches. The columns show what each run has at every "
                      "queried path; `null` means the path does not exist for that run.[/yellow]")
    console.print(f"[bold]Written:[/bold] {out_path}  ({hits} model name(s))")


def _cell(value: Any) -> str:
    if value is None:
        return "[red]null[/red]"
    if isinstance(value, (list, dict)):
        text = yaml.safe_dump(value, default_flow_style=True, width=10_000).strip()
    else:
        text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


# =============================================================================
# The fields a condition can name (the panel's FIND MODEL form)
# =============================================================================
FIELD_SOURCES = ("run", "registry", "sweep", "snapshot", "device")   # RunView.values' order: the first hit wins
_FIELD_VALUES = 40                                                   # distinct values listed per field
_SKIP_SNAPSHOT = {"sampled", "runtime", "paths"}                     # copies of the sweep point, per-run bookkeeping


def _leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """(path, value) of every leaf; a list of mappings is expanded as `*` (a condition's own syntax)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _leaves(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list) and value and all(isinstance(x, Mapping) for x in value):
        for item in value:
            yield from _leaves(item, f"{prefix}.*")
    else:
        yield prefix, value


def _kind(values: list[Any]) -> str:
    if values and all(isinstance(v, bool) for v in values):
        return "bool"
    if values and all(_is_num(v) for v in values):
        return "number"
    return "text"


def _sort_key(value: Any) -> tuple:
    return (0, value, "") if _is_num(value) else (1, 0, str(value))


def filter_fields(project_root: str | Path) -> dict[str, Any]:
    """Every parameter path a `where` condition can name, from every run of run_map.jsonl: where the
    filter finds it (run entry, registry, sweep point, snapshot, device record; the first source that
    has it wins, as RunView.values looks it up), its kind (number | text | bool, `list` when the runs
    hold lists there), how many runs have it, whether it differs between them, and its values (up
    to 40, sorted)."""
    project_root = Path(project_root)
    registry = _read_run_registry(project_root)
    seen: dict[str, dict[str, Any]] = {}
    runs = 0
    for model_hash, entry in load_run_map(project_root).items():
        if not isinstance(entry, Mapping):
            continue
        runs += 1
        run = RunView(model_hash, entry, project_root, registry.get(model_hash) if isinstance(registry.get(model_hash), Mapping) else None)
        try:
            snapshot = {k: v for k, v in run.snapshot.items() if k not in _SKIP_SNAPSHOT}
        except ModelFilterError:
            snapshot = {}
        sources = (("run", run.entry), ("registry", run.registry_entry), ("sweep", run.sampled), ("snapshot", snapshot),
                   ("device", run.device))
        here: dict[str, tuple[str, list[Any], bool]] = {}
        for source, doc in sources:
            for path, value in _leaves(doc):
                if not path or path in here or path.split(".")[0] in {"sampled", "_tfseal"}:
                    continue
                if any(path.startswith(q + ".") for q in here):                   # inside a value an earlier source has
                    continue
                listed = isinstance(value, list)
                here[path] = (source, value if listed else [value], listed)
        for path, (source, values, listed) in here.items():
            f = seen.setdefault(path, {"path": path, "source": source, "list": False, "runs": 0, "values": [], "_shapes": set()})
            f["runs"] += 1
            f["list"] = f["list"] or listed
            f["_shapes"].add(json.dumps(sorted(map(str, values))))
            for v in values:
                if v is not None and not isinstance(v, (dict, list)) and not any(_same(v, w) for w in f["values"]):
                    f["values"].append(v)
    fields = []
    for f in seen.values():
        values = sorted(f["values"], key=_sort_key)
        fields.append({"path": f["path"], "source": f["source"], "kind": "list" if f["list"] else _kind(values),
                       "runs": f["runs"], "varies": len(f["_shapes"]) > 1 or f["runs"] < runs,
                       "values": values[:_FIELD_VALUES], "more": max(0, len(values) - _FIELD_VALUES)})
    fields.sort(key=lambda f: (FIELD_SOURCES.index(f["source"]), f["path"]))
    return {"runs": runs, "fields": fields, "operators": list(OPERATORS)}


def write_where(config_path: str | Path, where: list[Mapping[str, Any]]) -> dict[str, Any]:
    """model_filter.where replaced by `where` (one mapping per condition, as written by hand), in block
    style; every other byte of the file, and the comments after the list, stay. Each condition is
    checked as the filter reads it; nothing is written unless the file reads back as exactly that."""
    from services.visualization.vscode_monitor import yamlspan

    path = Path(config_path)
    if not where:
        raise ModelFilterError("tick one condition at least: model_filter.where needs one")
    for i, cond in enumerate(where, start=1):
        _validate_condition(cond, index=i)
    text = path.read_text(encoding="utf-8")
    root = yamlspan.compose(text)
    block = next((v for k, v in getattr(root, "value", []) if k.value == CONFIG_KEY), None)
    node = next((v for k, v in getattr(block, "value", []) if k.value == "where"), None) if block is not None else None
    if node is None:
        raise ModelFilterError(f"{path} has no {CONFIG_KEY}.where to write")
    key = next(k for k, v in block.value if v is node)
    indent = " " * (key.start_mark.column + 2)
    def flow(value: Any) -> str:          # one line: a scalar as YAML writes it, a list or mapping in flow style
        out = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=1000).strip()
        return out[:-4].strip() if out.endswith("\n...") else out.removesuffix("...").strip()

    items = []
    for cond in where:
        lines = [f"{k}: {flow(v)}" for k, v in dict(cond).items()]       # path first, as written by hand
        items.append("\n".join([f"{indent}- {lines[0]}", *[f"{indent}  {line}" for line in lines[1:]]]))
    start = text.index(":", key.end_mark.index) + 1
    end = yamlspan.content_end(node)
    new = text[:start] + "\n" + "\n".join(items) + text[end:]
    expected = yaml.safe_load(text) or {}
    expected[CONFIG_KEY]["where"] = [dict(c) for c in where]
    if yaml.safe_load(new) != expected:
        raise ModelFilterError("the conditions do not read back as written; model_filter.yaml not changed")
    path.write_text(new, encoding="utf-8")
    return {"ok": True, "where": len(where)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Find trained models by their parameters.")
    parser.add_argument("--project", "-p", required=True)
    parser.add_argument("--env", "-e", default=None, help="run_map env to search (default: model_filter.env or any)")
    parser.add_argument("--config", default=None, help="default: projects/<project>/config/model_filter.yaml")
    parser.add_argument("--fields", action="store_true", help="print the fields a condition can name, as JSON (the panel)")
    parser.add_argument("--write-where", default=None, help="JSON list of conditions: model_filter.where, written in place")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = args.config or f"projects/{args.project}/config/model_filter.yaml"
    if args.fields or args.write_where is not None:
        try:
            if args.fields:
                out = filter_fields(Path("projects") / args.project)
                raw = (yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}).get(CONFIG_KEY) or {} if Path(config).is_file() else {}
                out["where"] = list(raw.get("where") or [])
            else:
                out = write_where(config, json.loads(args.write_where))
        except (ModelFilterError, ValueError) as exc:
            out = {"error": str(exc)}
        print(json.dumps(out, default=str))
        raise SystemExit(1 if "error" in out else 0)
    try:
        run_model_filter(config, project=args.project, env=args.env)
    except ModelFilterError as exc:
        raise SystemExit(f"model_filter: {exc}") from exc


if __name__ == "__main__":
    main()
