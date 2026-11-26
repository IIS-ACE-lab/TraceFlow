# services/graph_manager/metric_catalog.py
"""
What a model's recorded metrics offer to a graph job.

Walks the three metric sources of one model and reports, per process, which
metrics can be plotted. GraphManager prints this table when a job finds no
data, so a wrong metric/process/source is answered with the choices that do
exist instead of an empty figure.

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from services.side_channel_core.utils.logger import console
from services.side_channel_core.utils.seal import load_json, read_jsonl, strip as strip_seal

from .metric_loader import ARRAY_EXTENSIONS, EPOCH_FILE_RE, _key_recovery_row, fold_metric_dirs


# Counters, settings and provenance: recorded, but not model quality.
_BOOKKEEPING = re.compile(
    r".*_(correct|total|successes|trials|message)|epoch|timestamp|model_name|model_hash|process|env"
    r"|attack_byte|rank_base|n_folds|evaluated_traces|trace_order_repetitions"
    r"|recovery_rank|true_key_byte|top_key_hypothesis|keyspace_size|schema_version"
)

# Metrics that draw one figure per model.
_SOLO_ONLY = {"fold_rank_curves", "key_candidate_scores"}


def _scalar_fields(paths: Iterable[Path]) -> tuple[set[str], set[str]]:
    """(plottable, recorded but empty) numeric columns of history files."""
    seen: set[str] = set()
    valued: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = strip_seal(json.loads(line))
            # What a requested metric depends on (fit_time_s of mean_fit_time_s) plots too.
            nested = row.pop("dependencies", None)
            for key, value in {**(nested if isinstance(nested, dict) else {}), **row}.items():
                if _BOOKKEEPING.fullmatch(str(key)):
                    continue
                if value is None or isinstance(value, (bool, int, float)):
                    seen.add(key)
                    # null / NaN: recorded, but without a value to plot.
                    if value is not None and not (isinstance(value, float) and value != value):
                        valued.add(key)
    return valued, seen - valued


def _array_metrics(arrays_dir: Path, process: str) -> set[str]:
    root = arrays_dir / process
    if not root.is_dir():
        # metrics/attack_metrics/arrays/<process>/<metric>/epNNNN.npy, and
        # older layouts without the process level.
        root = arrays_dir
    found = set()
    for metric_dir in root.iterdir() if root.is_dir() else []:
        if metric_dir.is_dir() and any(
            f.suffix.lower() in ARRAY_EXTENSIONS and EPOCH_FILE_RE.match(f.stem)
            for f in metric_dir.iterdir()
        ):
            found.add(metric_dir.name)
    return found


def _history_dir_catalog(metrics_dir: Path) -> dict[str, tuple[set[str], set[str], set[str]]]:
    """process -> (scalars, arrays, empty scalars) of one scalars/arrays directory."""
    out: dict[str, tuple[set[str], set[str], set[str]]] = {}
    scalars = metrics_dir / "scalars"
    for path in sorted(scalars.glob("*_history.jsonl")) if scalars.is_dir() else []:
        match = re.search(r"_metrics_(\w+?)_history\.jsonl$", path.name) or re.search(r"^(\w+?)_history\.jsonl$", path.name)
        if not match:
            continue
        process = match.group(1)
        valued, empty = _scalar_fields([path])
        arrays = _array_metrics(metrics_dir / "arrays", process)
        out[process] = (valued - arrays, arrays, empty - arrays)
    return out


def _key_recovery_catalog(kr_dirs: Iterable[Path]) -> dict[str, tuple[set[str], set[str], set[str]]]:
    found: dict[str, tuple[set[str], set[str], set[str]]] = {}
    for kr_dir in kr_dirs:
        for path in sorted(kr_dir.glob("*.json")) + sorted(kr_dir.glob("*_history.jsonl")):
            process = path.stem.replace("_history", "").replace("_best", "")
            records = read_jsonl(path) if path.suffix == ".jsonl" else [load_json(path)]
            valued, seen, special = found.setdefault(process, (set(), set(), set()))
            for record in records:
                for key, value in _key_recovery_row(record).items():
                    if _BOOKKEEPING.fullmatch(str(key)):
                        continue
                    seen.add(key)
                    if value is not None:
                        valued.add(key)
                if (record.get("diagnostics") or {}).get("top_key_candidates"):
                    special.add("key_candidate_scores")
    # A field counts as empty only when no record anywhere carries a value.
    return {
        process: (valued, special, seen - valued)
        for process, (valued, seen, special) in found.items()
    }


def catalog(metrics_root: Path) -> dict[tuple[str, str], tuple[set[str], set[str], set[str]]]:
    """(source, process) -> (per-epoch metrics, per-trace metrics, empty metrics)."""
    metrics_root = Path(metrics_root)
    found: dict[tuple[str, str], tuple[set[str], set[str], set[str]]] = {}

    for process, entry in _history_dir_catalog(metrics_root / "attack_metrics").items():
        found[("attack_metrics", process)] = entry

    folds = fold_metric_dirs(metrics_root, strict=False)       # a menu: also while the run trains
    if folds:
        for process, (scalars, arrays, empty) in _history_dir_catalog(folds[0]).items():
            if "mean_attack_rank_curve" in arrays:
                arrays = arrays | {"fold_rank_curves"}
            found[("cross_validation", process)] = (scalars, arrays, empty)

    kr_dirs = [d for d in [metrics_root / "key_recovery"] + [f / "key_recovery" for f in folds] if d.is_dir()]
    for process, entry in _key_recovery_catalog(kr_dirs).items():
        found[("key_recovery", process)] = entry

    return found


# Shown under the table on every run, in the colour each class is drawn in.
_HINTS = [
    ("bright_green", "green", "plot against epochs (a bar per model when a process has one record, like test)"),
    ("bright_blue", "blue", "plot against traces or attacks; these use array_epoch"),
    ("bright_magenta", "magenta", "fold_rank_curves and key_candidate_scores, which need type: solo"),
    ("bright_black", "gray", "recorded but with no value yet, such as traces_to_ge_1 where GE never reached 1"),
]


def _catalog_table(title: str, entries: dict[tuple[str, str], tuple[set[str], set[str], set[str]]]) -> Any:
    from rich.table import Table
    from rich.text import Text

    table = Table(
        title=f"What you can choose, from your current data ({title})",
        title_style="bold bright_cyan",
        header_style="bold bright_yellow",
        show_lines=True,
    )
    table.add_column("source", style="bold bright_magenta", no_wrap=True)
    table.add_column("process", style="bold bright_cyan", no_wrap=True)
    table.add_column("metrics")

    for (source, process), (scalars, arrays, empty) in sorted(entries.items()):
        cell = Text()
        for name in sorted(scalars):
            cell.append(name, style="bright_green").append(", ")
        for name in sorted(arrays):
            cell.append(name, style="bright_magenta" if name in _SOLO_ONLY else "bright_blue").append(", ")
        for name in sorted(empty):
            cell.append(name, style="bright_black").append(", ")
        if cell.plain.endswith(", "):
            cell.right_crop(2)
        table.add_row(source, process, cell)

    return table


def print_metric_choices(models: Mapping[str, Path]) -> None:
    """
    Colored table of every (source, process, metric) the models recorded.

    Models whose recorded metrics match share one table.
    """
    groups: dict[Any, list[str]] = {}
    entries_by_key: dict[Any, dict[tuple[str, str], tuple[set[str], set[str], set[str]]]] = {}

    for model, metrics_root in models.items():
        entries = catalog(Path(metrics_root))
        if not entries:
            console.print(
                f"[bright_red]No recorded metrics under {metrics_root} for model {model!r}.[/bright_red]"
            )
            continue
        key = tuple(
            (source, process, tuple(sorted(a)), tuple(sorted(b)), tuple(sorted(c)))
            for (source, process), (a, b, c) in sorted(entries.items())
        )
        groups.setdefault(key, []).append(model)
        entries_by_key[key] = entries

    for key, names in groups.items():
        title = ", ".join(names[:3]) + (f" +{len(names) - 3} more" if len(names) > 3 else "")
        console.print(_catalog_table(title, entries_by_key[key]))

    if groups:
        for style, label, text in _HINTS:
            console.print(f"[{style}]Config hint  {label}[/{style}]: {text}", highlight=False)


def print_model_choices(
    *,
    missing: Iterable[str],
    available: Iterable[str],
    env: str,
    model_names_file: Path,
) -> None:
    """Name the models a job asked for that have no artifacts, and the ones that do."""
    from rich.table import Table
    from rich.text import Text

    missing = list(missing)
    available = list(available)

    table = Table(
        title=f"Model names in this job vs the models with artifacts (env={env})",
        title_style="bold bright_cyan",
        caption=f"generated models are listed in {model_names_file}",
        caption_style="bright_black",
        header_style="bold bright_yellow",
        show_lines=True,
    )
    table.add_column("state", style="bold", no_wrap=True)
    table.add_column("model_name")

    table.add_row(
        Text("no artifacts", style="bright_red"),
        Text(", ".join(missing), style="bright_red"),
    )
    table.add_row(
        Text("available", style="bright_green"),
        Text(", ".join(available), style="bright_green")
        if available
        else Text("none: train a model first", style="bright_black"),
    )

    console.print(table)


__all__ = ["catalog", "print_metric_choices", "print_model_choices"]
