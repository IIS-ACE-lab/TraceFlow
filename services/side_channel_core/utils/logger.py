# File: services/side_channel_core/utils/logger.py
'''
This module serves as an umbrella for all terminal printing logics, logging, log history generating logics used in SCA/services/.

Author: Shraddha Mishra
'''
from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
from collections.abc import Mapping, Sequence
from logging import Formatter
from logging.handlers import RotatingFileHandler as _RotatingFileHandler


def _load_sealed_json(path):
    """A registry, verified and without its seal key (utils/seal.py)."""
    try:
        from utils.seal import load_json
    except ImportError:  # imported as services.side_channel_core.utils
        from services.side_channel_core.utils.seal import load_json
    return load_json(path)


_ANSI = re.compile(r"\033\[[0-9;]*m")         # a colour code: the terminal shows it, a file does not


class RotatingFileHandler(_RotatingFileHandler):
    """A rotating log file that signs what it wrote when it closes or rotates
    (utils/seal.py: a "# tfseal1:..." line over every byte above it). It writes plain text: a record's colour
    codes (an epoch block's green ▲ and red ▼) are the terminal's, and an editor would show them as [31m."""

    def format(self, record):
        return _ANSI.sub("", super().format(record))

    def _seal(self) -> None:
        try:
            from utils.seal import seal_log_file
        except ImportError:  # imported as services.side_channel_core.utils
            from services.side_channel_core.utils.seal import seal_log_file
        try:
            seal_log_file(self.baseFilename)
        except Exception:  # a seal never stops the logging
            pass

    def doRollover(self):
        if self.stream:
            self.stream.flush()
        self._seal()
        super().doRollover()

    def close(self):
        try:
            if self.stream:
                self.stream.flush()
            self._seal()
        finally:
            super().close()
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.syntax import Syntax
from rich.text import Text

from utils.config_view import resolved_view
from utils.timestamps import apply_process_timezone

# Log times (asctime, Rich) follow master.yaml's timezone, like every
# timestamp written to registries and reports (utils/timestamps.py).
apply_process_timezone()

# Global console instance
_TERMINAL_WIDTH = max(
    80,
    int(os.getenv("COLUMNS", "160")) - 20,
)
console = Console(
    force_terminal=True,
    color_system="truecolor",
    width=_TERMINAL_WIDTH
)

# The log lines' console: the same stdout, coloured like `console`, but soft-wrapped,
# so a record is printed as one line whatever its length: the terminal wraps it, and
# the make / tmux log (the panel strips the colour codes) keeps it whole.
log_console = Console(
    force_terminal=True,
    color_system="truecolor",
    width=_TERMINAL_WIDTH,
    soft_wrap=True,
)


class LineRichHandler(RichHandler):
    """RichHandler's colours (its highlighter) and rich tracebacks, without its layout
    table: no columns, no padding to the console width, one record one line."""

    def render(self, *, record, traceback, message_renderable):
        return Group(message_renderable, traceback) if traceback else message_renderable

    def render_message(self, record, message):
        """A message carrying ANSI colour (an epoch block's arrows) is shown in those colours, not as codes."""
        if "\033[" in message:
            return Text.from_ansi(message)
        return super().render_message(record, message)


# ANSI color map → Rich styles
COLOR_MAP = {
    "gold": "bright_yellow",
    "yellow": "yellow",
    "green": "bright_green",
    "red": "bright_red",
    "blue": "bright_blue",
    "pink": "bright_magenta",
    "magenta": "magenta",
    "cyan": "bright_cyan",
    "white": "white",
    "gray": "bright_black",
}

_logger_contexts = {}  # cache: logs_dir → bool attached

REGISTRY_FILENAME = "run_registry.json"
def load_registry() -> dict:
    """
    Auto-discover the run registry by walking up from cwd until we find
    <project_root>/artifacts/run_registry.json.
    """
    current = Path.cwd()
    while current != current.parent:  # until root
        registry_path = current / "artifacts" / REGISTRY_FILENAME
        if registry_path.exists():
            return _load_sealed_json(registry_path)
        current = current.parent
    # fallback
    return {}


def pretty_print(
    message: str,
    color: str = "white",
    bold: bool = False,
    **kwargs,
) -> None:
    """
    Pretty print a message with color using Rich.

    Args:
        "gold": "bright_yellow",
        "yellow": "yellow",
        "green": "bright_green",
        "red": "bright_red",
        "blue": "bright_blue",
        "pink": "bright_magenta",
        "magenta": "magenta",
        "cyan": "bright_cyan",
        "white": "white",
        "gray": "bright_black"

        bold: whether to make the text bold
    """

    style = COLOR_MAP.get(color.lower(), "white")

    if bold:
        style = f"bold {style}"

    console.print(
        Text(str(message), style=style),
        #end="",
    )

    # Always flush for real-time orchestration logs
    try:
        console.file.flush()
    except Exception:
        pass
    
def get_logger(name: str, log_dir: str = None, model_hash: str = None) -> logging.Logger:
    """
    Exact logger:
    - If model_hash is given, locates that hash in run_registry.json and logs there.
    - Else falls back: log_dir > SCC_LOG_DIR > console only.
    - Safe: never crashes if no registry yet.
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, os.getenv("LOGLEVEL", "INFO").upper(), logging.INFO))
    logger.propagate = True
    # Always clear old handlers → avoids duplicate writes
    logger.handlers.clear()

    fmt = Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

    # Coloured as RichHandler colours them, one record one line: the plain
    # RichHandler wrapped every record at the console width and padded it with
    # spaces, so the make / tmux logs filled with padding and message fragments.
    ch = LineRichHandler(
        console=log_console,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )

    ch.setFormatter(fmt)
    logger.addHandler(ch)

    final_dir = None

    if model_hash:
        # Robustly find run_registry.json upwards
        reg_file = None
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "artifacts" / "run_registry.json"
            if candidate.exists():
                reg_file = candidate
                break

        if reg_file and reg_file.exists():
            reg = _load_sealed_json(reg_file)
            entry = reg.get(model_hash)
            if entry:
                final_dir = Path(entry["path"]) / "logs"

    # Fallback chain
    if not final_dir:
        if log_dir:
            final_dir = Path(log_dir)
        elif os.getenv("SCC_LOG_DIR"):
            final_dir = Path(os.getenv("SCC_LOG_DIR"))

    if final_dir:
        final_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            final_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # mark so same logger-name + model_hash is idempotent
    cache_key = f"{name}:{model_hash or log_dir or 'global'}"
    _logger_contexts[cache_key] = True

    return logger


def setup_logging(
    cfg: dict,
    project_name: str,
    model_name: str,
    base_dir: Path,
    process_name: str,
):
    """
    Local file + console logging only.

    ClearML task initialization is handled centrally elsewhere
    through the dedicated ClearML integration module.

    Args:
        cfg (dict): Full config.
        project_name (str): Unused here, kept for compatibility.
        model_name (str): Unused here, kept for compatibility.
        base_dir (Path): Root folder for this run's artifacts.
        process_name (str): Logical process name for logger naming.

    Returns:
        logger: standard logger instance
        task: always None (ClearML is no longer initialized here)
        safe_cfg: deepcopy of cfg with Path values stringified under safe_cfg["paths"]
    """
    _ = project_name
    _ = model_name

    final_dir = configure_global_logging(cfg=cfg)
    logger = get_logger(process_name, str(final_dir))

    safe_cfg = copy.deepcopy(cfg)
    for k, v in safe_cfg.get("paths", {}).items():
        if isinstance(v, Path):
            safe_cfg["paths"][k] = str(v)

    return logger, None, safe_cfg


def configure_global_logging(cfg: dict = None, log_dir: str = None, model_hash: str = None):
    """
        Configures the ROOT aggregate file handler:
        - Resolves log folder from cfg or registry or env.
        - Installs a RotatingFileHandler for all_logs.log.
        - Preserves unrelated handlers and existing console logging.
        - Captures propagated logging records; does not redirect stdout/stderr.
        """

    # 1) Figure out final_dir
    final_dir = None

    if cfg:
        from utils.path_utils import resolve_path  # lazy import breaks cycle
        final_dir = Path(resolve_path(cfg["paths"]["logs"], cfg))
    elif log_dir is not None:
        final_dir = Path(log_dir)
    elif model_hash:
        here = Path(__file__).resolve()
        for parent in here.parents:
            reg_file = parent / "artifacts" / "run_registry.json"
            if reg_file.exists():
                reg = _load_sealed_json(reg_file)
                if reg and model_hash in reg:
                    final_dir = Path(reg[model_hash]["path"]) / "logs"
                break
    elif os.getenv("SCC_LOG_DIR"):
        final_dir = Path(os.getenv("SCC_LOG_DIR"))
    else:
        raise ValueError("No valid logging folder found.")

    if final_dir is None:
        raise ValueError("No valid logging folder found for the requested context.")
    final_dir = final_dir.expanduser().resolve()
    final_dir.mkdir(parents=True, exist_ok=True)
    logfile = final_dir / "all_logs.log"

    # Check the live handler, not a historical context cache: A -> B -> A
    # must restore A's destination rather than keep writing to B.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    aggregate_handlers = [
        handler for handler in root_logger.handlers
        if getattr(handler, "_traceflow_aggregate", False)
        or (isinstance(handler, RotatingFileHandler)
            and Path(handler.baseFilename).name == "all_logs.log")
    ]
    if len(aggregate_handlers) == 1 and Path(aggregate_handlers[0].baseFilename) == logfile:
        return final_dir
    for handler in aggregate_handlers:
        root_logger.removeHandler(handler)
        handler.close()

    fmt = Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

    # File: all_logs.log
    fh = RotatingFileHandler(logfile, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh._traceflow_aggregate = True
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    return final_dir

# Metric fields used only to explain a public metric. These are intentionally
# omitted from the ordinary metric list so the terminal does not print duplicate
# or context-free provenance rows.
_AUXILIARY_METRIC_KEYS = frozenset(
    {
        "accuracy_correct",
        "accuracy_total",
        "sr_at_1_successes",
        "sr_at_1_trials",
        "successful_attacks",
        "failed_attack_count",
        "attack_repetitions",
        "traces_to_ge_1_message",
        "ge_threshold_status",
        "attack_protocol",
        "sampling",
    }
)

_METRIC_NAME_ALIASES = {
    "ge": "guessing_entropy",
    "final_ge": "guessing_entropy",
    "auc_guessing_entropy": "auc_ge",
    "success_rate_curve": "sr_curve",
    "auc_success_rate": "auc_sr",
    "rank_success": "rank_k_success",
}


def _strict_metric_int(value: Any) -> Optional[int]:
    """Return an exact integer or ``None`` without truncating floats."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (Integral, np.integer)):
        return int(value)
    if isinstance(value, (Real, np.floating)):
        numeric = float(value)
        if math.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
    return None


def _numeric_scalar(value: Any) -> Optional[float]:
    """Normalize a real scalar while keeping booleans out of numeric metrics."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (Real, np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        try:
            return float(value.item())
        except (TypeError, ValueError):
            return None
    return None


def _canonical_metric_name(name: Any) -> str:
    normalized = str(name).strip()
    # fit_time_{s} / fit_time_{h} / fit_time_{per_epoch} -> fit_time_s / ...
    plain = normalized.lower().replace("{", "").replace("}", "")
    if plain in ("fit_time", "mean_fit_time"):
        return f"{plain}_s"
    if plain.startswith(("fit_time_", "mean_fit_time_")):
        return plain
    return _METRIC_NAME_ALIASES.get(normalized, normalized)


def _expand_metric_spec(spec: Any) -> Optional[str]:
    """
    Expand one configured metric spec into the exact result key emitted by
    ``compute_metrics``. The defaults deliberately mirror metrics.py.
    """
    if isinstance(spec, str):
        name = _canonical_metric_name(spec)
        if name == "rank_k_success":
            return "rank_k_success@1"
        return name

    if not isinstance(spec, Mapping) or len(spec) != 1:
        return None

    raw_name, raw_opts = next(iter(spec.items()))
    name = _canonical_metric_name(raw_name)
    opts = raw_opts if isinstance(raw_opts, Mapping) else {}

    if name == "top_k_accuracy":
        return f"top_{int(opts.get('k', 5))}_accuracy"

    if name == "rank_k_success":
        return f"rank_k_success@{int(opts.get('k', 1))}"

    if name == "rank_percentile":
        return (
            f"rank_{int(opts['p'])}_percentile"
            if "p" in opts
            else "rank_percentile"
        )

    if name == "ge_reduction_rate":
        return (
            f"ge_reduction_rate@{int(opts['window'])}"
            if "window" in opts
            else "ge_reduction_rate"
        )

    if name in {"rank_at_traces", "ge_at_traces"}:
        budget_key = next((key for key in ("traces", "t", "n") if key in opts), None)
        if budget_key is None:
            return name
        prefix = "rank" if name == "rank_at_traces" else "ge"
        return f"{prefix}@{int(opts[budget_key])}"

    if name in {"time_to_recover", "auc_sr", "sr_curve"}:
        return f"{name}@{int(opts['k'])}" if "k" in opts else name

    if name == "traces_to_ge_k":
        return (
            f"traces_to_ge_{int(opts['k'])}"
            if "k" in opts
            else "traces_to_ge_k"
        )

    if name == "traces_to_sr_k":
        return (
            f"traces_to_sr_{int(opts['k'])}"
            if "k" in opts
            else "traces_to_sr_k"
        )

    return name


def _ordered_metric_names(cfg: Mapping[str, Any], results: Mapping[str, Any]) -> list[str]:
    """Resolve a stable, duplicate-free display order."""
    configured = cfg.get("metrics_to_log")
    if configured is None:
        configured = cfg.get("metrics")

    names: list[str] = []
    if configured:
        if not isinstance(configured, Sequence) or isinstance(configured, (str, bytes)):
            raise TypeError("metrics_to_log / metrics must be a list when present")
        for spec in configured:
            expanded = _expand_metric_spec(spec)
            if expanded:
                names.append(expanded)
    else:
        names.extend(
            str(key)
            for key in results
            if str(key) not in _AUXILIARY_METRIC_KEYS
            and not str(key).startswith("__")
        )

    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name in seen or name in _AUXILIARY_METRIC_KEYS or name.startswith("__"):
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _curve_trace_budget(value: Any) -> Optional[int]:
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] < 2:
        return None
    endpoint = _numeric_scalar(array[-1, 0])
    if endpoint is None or not math.isfinite(endpoint) or endpoint <= 0:
        return None
    return int(round(endpoint))


def _infer_trace_budget(results: Mapping[str, Any]) -> Optional[int]:
    """Infer the evaluated trace budget without adding another config source."""
    for key in ("mean_attack_rank_curve", "full_ranks", "std_attack_rank_curve"):
        if key in results:
            budget = _curve_trace_budget(results[key])
            if budget is not None:
                return budget

    for key, value in results.items():
        if str(key) == "sr_curve" or str(key).startswith("sr_curve@"):
            budget = _curve_trace_budget(value)
            if budget is not None:
                return budget

    for context_key in ("attack_protocol", "sampling", "trace_budget"):
        context = results.get(context_key)
        if not isinstance(context, Mapping):
            continue
        for field in (
            "trace_budget_effective",
            "traces_used",
            "evaluation_population_size",
            "used",
        ):
            parsed = _strict_metric_int(context.get(field))
            if parsed is not None and parsed > 0:
                return parsed

    return None


def _format_rate(
    value: float,
    *,
    numerator: Optional[int] = None,
    denominator: Optional[int] = None,
    unit: Optional[str] = None,
) -> str:
    if not math.isfinite(value):
        return "not available (non-finite result)"

    percent = 100.0 * value
    base = f"{percent:.2f}%"
    out_of_range = value < -1e-12 or value > 1.0 + 1e-12

    if numerator is not None and denominator is not None and denominator > 0:
        implied = numerator / float(denominator)
        detail_unit = f" {unit}" if unit else ""
        base += (
            f" ({numerator} / {denominator}{detail_unit}; "
            f"fraction={value:.6f})"
        )
        if not math.isclose(value, implied, rel_tol=1e-9, abs_tol=1e-12):
            base += f" [INCONSISTENT: counts imply {implied:.6f}]"
        if out_of_range:
            base += " [OUT OF RANGE: expected fraction in [0,1]]"
        return base

    base = f"{base} (fraction={value:.6f})"
    if out_of_range:
        base += " [OUT OF RANGE: expected fraction in [0,1]]"
    return base


def _format_rate_percent_only(value: float) -> str:
    """Format a public rate without implying independent trial counts."""
    if not math.isfinite(value):
        return "not available (non-finite result)"

    base = f"{100.0 * value:.2f}%"
    if value < -1e-12 or value > 1.0 + 1e-12:
        base += " [OUT OF RANGE: expected fraction in [0,1]]"
    return base


def _is_rate_metric(metric_name: str) -> bool:
    if metric_name == "accuracy":
        return True
    if metric_name.startswith("top_") and metric_name.endswith("_accuracy"):
        return True
    if metric_name in {"success_rate", "recovery_rate", "final_sr_at_1"}:
        return True
    if metric_name.startswith("rank_k_success@"):
        return True
    return False


def _is_trace_threshold_metric(metric_name: str) -> bool:
    return (
        metric_name in {
            "traces_to_ge_1",
            "traces_to_ge_k",
            "traces_to_sr_1",
            "traces_to_sr_k",
            "time_to_recover",
            "per_attack_traces_to_ge_1",
        }
        or metric_name.startswith("traces_to_ge_")
        or metric_name.startswith("traces_to_sr_")
        or metric_name.startswith("time_to_recover@")
    )


def _format_trace_threshold(value: float, *, trace_budget: Optional[int]) -> str:
    if not math.isfinite(value):
        if trace_budget is not None:
            return f"not reached within {trace_budget} traces"
        return "not reached within evaluated trace budget"

    if value < 0:
        return f"invalid trace count ({value:.4f})"

    rounded = int(round(value))
    if math.isclose(value, float(rounded), rel_tol=0.0, abs_tol=1e-9):
        return f"trace {rounded}"
    return f"trace {value:.4f}"


def _curve_summary(metric_name: str, value: Any) -> str:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] < 2:
        return f"array{array.shape}"
    if array.shape[0] == 0:
        return f"curve{array.shape} | empty"

    y = array[:, 1].astype(np.float64, copy=False)
    indices = (0, len(y) // 2, len(y) - 1)
    sampled = [float(y[index]) for index in indices]

    if metric_name == "sr_curve" or metric_name.startswith("sr_curve@"):
        formatted = [
            "NA" if not math.isfinite(item) else f"{100.0 * item:.2f}%"
            for item in sampled
        ]
        return (
            f"curve{array.shape} | SR@start={formatted[0]}, "
            f"SR@mid={formatted[1]}, SR@final={formatted[2]}"
        )

    if metric_name == "std_attack_rank_curve":
        label = "STD"
    elif metric_name == "full_ranks":
        label = "rank"
    else:
        label = "GE"

    formatted = ["NA" if not math.isfinite(item) else f"{item:.4f}" for item in sampled]
    return (
        f"curve{array.shape} | {label}@start={formatted[0]}, "
        f"{label}@mid={formatted[1]}, {label}@final={formatted[2]}"
    )


def _format_final_ranks(value: Any) -> str:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return "n=0 | empty"

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return f"n={array.size} | no finite ranks"

    details = (
        f"n={array.size}, min={np.min(finite):.4f}, "
        f"mean={np.mean(finite):.4f}, max={np.max(finite):.4f}"
    )
    if array.size <= 20:
        values = [
            int(item) if math.isfinite(item) and float(item).is_integer() else float(item)
            for item in array.tolist()
        ]
        details += f" | values={values}"
    if finite.size != array.size:
        details += f" | non_finite={array.size - finite.size}"
    return details


def _format_array_metric(metric_name: str, value: np.ndarray) -> str:
    if (
        metric_name in {"mean_attack_rank_curve", "std_attack_rank_curve", "full_ranks"}
        or metric_name == "sr_curve"
        or metric_name.startswith("sr_curve@")
    ):
        return _curve_summary(metric_name, value)

    array = np.asarray(value)
    if array.size == 0:
        return f"array{array.shape} | empty"

    if np.issubdtype(array.dtype, np.number):
        numeric = array.astype(np.float64, copy=False).reshape(-1)
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            return f"array{array.shape} | no finite values"
        return (
            f"array{array.shape} | min={np.min(finite):.4f}, "
            f"mean={np.mean(finite):.4f}, max={np.max(finite):.4f}"
        )

    return f"array{array.shape}"


def _format_metric_value(
    metric_name: str,
    value: Any,
    *,
    results: Mapping[str, Any],
    trace_budget: Optional[int],
) -> str:
    if metric_name == "final_ranks":
        return _format_final_ranks(value)

    if metric_name == "per_attack_curves" and isinstance(value, Sequence):
        shapes = [getattr(item, "shape", None) for item in value]
        unique_shapes = list(dict.fromkeys(str(shape) for shape in shapes))
        return (
            f"{len(value)} attack curves | "
            f"shape={unique_shapes[0] if len(unique_shapes) == 1 else unique_shapes}"
        )

    if isinstance(value, np.ndarray):
        return _format_array_metric(metric_name, value)

    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"

    if value is None and _is_trace_threshold_metric(metric_name):
        # Unreached threshold: compute_metrics records it as None.
        return _format_trace_threshold(float("nan"), trace_budget=trace_budget)

    scalar = _numeric_scalar(value)
    if scalar is not None:
        if _is_trace_threshold_metric(metric_name):
            return _format_trace_threshold(scalar, trace_budget=trace_budget)

        if metric_name == "accuracy":
            return _format_rate(
                scalar,
                numerator=_strict_metric_int(results.get("accuracy_correct")),
                denominator=_strict_metric_int(results.get("accuracy_total")),
                unit="traces",
            )

        if metric_name == "success_rate":
            return _format_rate_percent_only(scalar)

        if _is_rate_metric(metric_name):
            return _format_rate(scalar)

        if not math.isfinite(scalar):
            return "not available (non-finite result)"

        if isinstance(value, (Integral, np.integer)):
            return str(int(value))
        return f"{scalar:.4f}"

    if isinstance(value, Mapping):
        return f"mapping(keys={list(value.keys())})"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return f"list(len={len(value)})"

    return str(value)


_OBJECTIVES = ("training_objective", "validation_objective")
# An epoch block's arrows and their key, coloured in the terminal (green better, red worse); the log files get
# them plain (RotatingFileHandler drops the colour codes)
_BETTER = "\033[32m▲\033[0m"
_WORSE = "\033[31m▼\033[0m"
EPOCH_ARROWS_KEY = f"{_BETTER} better, {_WORSE} worse than the epoch before"


def _compact(value: float) -> str:
    """An integral value as an integer, any other with five significant digits (a learning rate of 1.23e-4
    stays 0.000123)."""
    return str(int(value)) if float(value).is_integer() else f"{value:.5g}"


def _attack_curves_text(value: Any) -> str:
    """per_attack_curves [attacks, traces, (traces, rank)]: its shape, the min / mean / max of the ranks (the
    rank column only: the trace counts beside them are no ranks) and the number of attacks."""
    curves = np.asarray(value, dtype=np.float64)
    if curves.ndim != 3 or curves.shape[-1] < 2 or not curves.size:
        return _format_metric_value("per_attack_curves", value, results={}, trace_budget=None)
    ranks = curves[..., 1]
    ranks = ranks[np.isfinite(ranks)]
    shape = f"array{tuple(int(n) for n in curves.shape)}"
    if not ranks.size:
        return f"{shape} | no finite rank, n={curves.shape[0]} attacks"
    return (f"{shape} | min={ranks.min():.4f}, mean={ranks.mean():.4f}, max={ranks.max():.4f}, "
            f"n={curves.shape[0]} attacks")


def _cell(metric_name: str, value: Any, *, results: Mapping[str, Any],
          trace_budget: Optional[int]) -> tuple[Optional[str], Optional[float], bool]:
    """(the value as its row shows it, its number for the arrow, whether it is a number). A scalar is a number
    (_compact; an accuracy with its counts); a curve, the attack curves, final_ranks and an unreached threshold
    are their whole summary in the cell: start / mid / final, min / mean / max, "not reached by N traces". A
    text column (loss_type) is no metric and is left out."""
    if metric_name == "per_attack_curves" and isinstance(value, (Sequence, np.ndarray)):
        return _attack_curves_text(value), None, False
    if metric_name == "per_attack_traces_to_ge_1" and value is not None:   # N̄tGE: a mean over attacks, no one trace
        number = _numeric_scalar(value)
        if number is not None and math.isfinite(number):
            reached = _strict_metric_int(results.get("per_attack_traces_to_ge_1_reached"))
            unreached = _strict_metric_int(results.get("per_attack_traces_to_ge_1_unreached")) or 0
            of = ("" if reached is None else f" (mean of {reached:,} attacks)" if not unreached
                  else f" (mean of the {reached:,} of {reached + unreached:,} attacks that reached it)")
            return f"{number:.5g} traces{of}", number, False
    if metric_name == "final_ranks":
        ranks = np.asarray(value, dtype=np.float64)
        mean = float(np.mean(ranks)) if ranks.size else None
        text = _format_final_ranks(value)
        return (f"{text}, final true-key rank = {_compact(mean)}" if mean is not None else text), mean, False
    if isinstance(value, np.ndarray):
        last = _numeric_scalar(value[-1, 1]) if value.ndim == 2 and value.shape[0] and value.shape[1] >= 2 else None
        return _format_metric_value(metric_name, value, results=results, trace_budget=trace_budget), last, False
    if isinstance(value, (str, bytes, Mapping)):
        return None, None, False
    if value is None:                      # an unreached trace threshold, after the traces the attack had
        if not _is_trace_threshold_metric(metric_name):
            return None, None, False
        return (f"not reached by {trace_budget:,} traces" if trace_budget else "not reached"), None, False
    number = _numeric_scalar(value)
    if metric_name == "accuracy" and number is not None and math.isfinite(number):    # the fraction and its counts
        correct, total = (_strict_metric_int(results.get(k)) for k in ("accuracy_correct", "accuracy_total"))
        counts = f" ({correct:,} of {total:,})" if correct is not None and total is not None else ""
        return _compact(number) + counts, number, True
    if number is not None and math.isfinite(number) and not _is_rate_metric(metric_name) \
            and not _is_trace_threshold_metric(metric_name):
        return _compact(number), number, True
    return _format_metric_value(metric_name, value, results=results, trace_budget=trace_budget), number, False


def _is_attack_metric(metric_name: str) -> bool:
    """An attack metric is one VALID_METRIC_PROCESSES records for no train row (ranks, guessing entropy, success
    rate, LDD): it is computed by attacking held-out traces, not from training. The objectives are the loss's
    (validation_objective is the training's loss on held-out traces); an unregistered metric is no attack
    metric."""
    from sca.metrics import valid_processes_for_metric

    return metric_name not in _OBJECTIVES and "train" not in valid_processes_for_metric(metric_name)


def _undirected(metric_name: str) -> bool:
    """A record with no better or worse and no mark: the parameter count and a fit time (it only grows). η_t
    has its arrows (metric_direction: its rise is ▲)."""
    return "fit_time" in metric_name or metric_name == "trainable_parameters"


def _visible(text: str) -> int:
    """The printed width of a cell: colour codes take none."""
    return len(_ANSI.sub("", text))


def format_epoch_table_lines(
    *,
    columns: Sequence[tuple[str, str, Mapping[str, Any], Optional[Mapping[str, Any]]]],
    elapsed: Optional[float],
    eta: Optional[float],
    epoch: int,
    epochs: int,
    cfg: Mapping[str, Any],
) -> list[str]:
    """One epoch's block, a table: a header row (metrics | train | validation), a rule, then one row per metric
    of metrics_to_log, its value per process with the arrow against the epoch before (green ▲ better, red ▼
    worse), then time and ETA; the arrows' key at the bottom.

        [Epoch 20/20] Results:
            metrics                   | train    | validation
            ---------------------------------------------------
            mean_accuracy             | 0.0062 ▼ |     0.0048 ▼
            CrossEntropyLoss          |   5.53 ▲ |     5.5581 ▲
            mean_attack_rank_curve    |          | curve(2500, 2) | GE@start=..., GE@mid=..., GE@final=...
            final_ranks               |          | n=100, min=..., mean=..., max=..., final true-key rank = 107 ▼
            time                      |     1.8s |
            ETA                       |     0.0m |

    The training's scalars come first, then the attack metrics (_is_attack_metric: those VALID_METRIC_PROCESSES
    records for no train row), each group in metrics_to_log order; a number is right-aligned in its column, a
    summary is whole in its cell (a curve's start / mid / final, the ranks' min / mean / max, "not reached by N
    traces"). The parameter count and a fit time get no arrow; η_t's rise is ▲ (metric_direction). columns: (process, label, results, the epoch before's results or None) in train,
    validation, test order. training_objective and validation_objective share one row named after
    trainer.loss.type. Fit time is the training's: a validation column beside a train column leaves it out."""
    from utils.monitor_metric import metric_direction

    merged: dict[str, Any] = {}
    for _process, _label, results, _before in columns:
        merged.update(results)
    loss = str(((cfg.get("trainer") or {}).get("loss") or {}).get("type") or "objective")
    has_train = any(process == "train" for process, *_ in columns)
    processes = [process for process, *_ in columns]
    # row name -> process -> (text, mark, is a number)
    rows: dict[str, dict[str, tuple[str, str, bool]]] = {}
    attack: set[str] = set()
    for name in _ordered_metric_names(cfg, merged):
        for process, _label, results, before in columns:
            if name not in results or (has_train and process != "train" and "fit_time" in name):
                continue
            budget = _infer_trace_budget(results)
            text, number, numeric = _cell(name, results[name], results=results, trace_budget=budget)
            if text is None:
                continue
            previous = (_cell(name, before[name], results=before, trace_budget=_infer_trace_budget(before))[1]
                        if before and name in before else None)
            mark = ""                      # no change, or a record with no mark (_undirected): no sign
            if not _undirected(name) and number is not None and previous is not None and number != previous \
                    and math.isfinite(number) and math.isfinite(previous):
                mark = _BETTER if (number > previous) == (metric_direction(name) == "max") else _WORSE
            row_name = loss if name in _OBJECTIVES else name
            rows.setdefault(row_name, {})[process] = (text, mark, numeric)
            if _is_attack_metric(name):
                attack.add(row_name)
    # the training's scalars first, then the attack metrics, each group in metrics_to_log order
    ordered = [n for n in rows if n not in attack] + [n for n in rows if n in attack]

    elapsed_value = _numeric_scalar(elapsed) if elapsed is not None else None
    eta_value = _numeric_scalar(eta) if eta is not None else None
    time_text = (f"{elapsed_value:.1f}s" if elapsed_value is not None and math.isfinite(elapsed_value)
                 and elapsed_value >= 0.0 else "NA")
    eta_text = (f"{eta_value / 60.0:.1f}m" if eta_value is not None and math.isfinite(eta_value)
                and eta_value >= 0.0 else "NA")
    table = [(n, rows[n]) for n in ordered] + [("time", {processes[0]: (time_text, "", True)}),
                                               ("ETA", {processes[0]: (eta_text, "", True)})]
    arrows = any(c[1] for _n, cells in table for c in cells.values())
    labels = {process: label for process, label, *_ in columns}

    # a column as wide as its header and its numbers (a number, its space and its arrow); a summary in the last
    # column runs past it, one elsewhere widens its column
    def width(i: int, process: str) -> int:
        cells = [cells[process] for _n, cells in table if process in cells]
        numbers = [_visible(t) + (2 if arrows else 0) for t, _m, numeric in cells if numeric]
        texts = [] if i == len(processes) - 1 else [_visible(t) + (2 if m else 0) for t, m, numeric in cells if not numeric]
        return max([len(labels[process])] + numbers + texts)

    widths = {p: width(i, p) for i, p in enumerate(processes)}
    name_width = max([len("metrics")] + [len(n) for n, _c in table])

    def cell(process: str, cells: Mapping[str, tuple[str, str, bool]], last: bool) -> str:
        if process not in cells:
            return "" if last else " " * widths[process]
        text, mark, numeric = cells[process]
        if numeric:
            shown = text.rjust(widths[process] - (2 if arrows else 0)) + ((" " + (mark or " ")) if arrows else "")
        else:
            shown = text + (" " + mark if mark else "")
        return shown if last else shown + " " * (widths[process] - _visible(shown))

    def row(name: str, cells: Mapping[str, Any]) -> str:
        parts = [cell(p, cells, i == len(processes) - 1) for i, p in enumerate(processes)]
        return (f"    {name.ljust(name_width)} | " + " | ".join(parts)).rstrip()

    header = (f"    {'metrics'.ljust(name_width)} | " +
              " | ".join(labels[p].ljust(widths[p]) if i < len(processes) - 1 else labels[p]
                         for i, p in enumerate(processes)))
    lines = [f"[Epoch {int(epoch)}/{int(epochs)}] Results:", header, "    " + "-" * (len(header) - 4)]
    lines += [row(n, cells) for n, cells in table]
    if arrows:
        lines.append(f"    {EPOCH_ARROWS_KEY}")             # the arrows' key, at the bottom of the block
    return lines


def print_blocked_results(
    logger,
    results: dict,
    elapsed: Optional[float],
    eta: Optional[float],
    epoch: int,
    epochs: int,
    cfg: dict,
    *,
    process: str = "validation",
    label: Optional[str] = None,
    previous: Optional[Mapping[str, Any]] = None,
    train: Optional[Mapping[str, Any]] = None,
    previous_train: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Log one epoch's block (format_epoch_table_lines): ``results`` are ``process``'s (validation each epoch,
    train for the refit, test for the test), ``train`` the epoch's training passes beside them, ``previous``
    and ``previous_train`` the epoch before's, for the arrows.

    Canonical metric values remain unchanged in ``results``. This function only
    applies human-readable units:

    - accuracy fractions -> percentages plus exact correct/total trace counts;
    - success-rate fractions -> percentages without presenting trace-order
      permutations as independent attacks;
    - unreached trace thresholds -> explicit text instead of ``nan``;
    - rank/SR curves -> their final point, and start/mid/final summaries under the table;
    - arrays -> finite-value summaries rather than raw dumps.
    """
    columns = []
    if train is not None:
        columns.append(("train", "train", train, previous_train))
    columns.append((process, label or process, results, previous))
    for line in format_epoch_table_lines(columns=columns, elapsed=elapsed, eta=eta, epoch=epoch,
                                         epochs=epochs, cfg=cfg):
        logger.info("%s", line)


def pretty_config(
    safe_cfg: dict,
    model_name: str,
    project_name: str,
    process_name: str,
    model_hash: str = None,
) -> str:
    """
    Pretty-print config to console,
    AND log plain JSON to:
      - this model's logs/{process_name}.log
      - plus ROOT (all_logs.log)
    without double console output.
    All terminal output uses the module-level Rich Console so the config block
    cannot interleave with subsequent standard logger output.   
    #---------Testing themes for rich.syntax.Syntax---------------
    # themes = [
    # "abap", "algol", "algol_nu", "arduino", "autumn", "bw", "borland", "coffee",
    # "colorful", "default", "dracula", "emacs", "friendly_grayscale", "friendly",
    # "fruity", "github-dark", "gruvbox-dark", "gruvbox-light", "igor", "inkpot",
    # "lightbulb", "lilypond", "lovelace", "manni", "material", "monokai", "murphy",
    # "native", "nord-darker", "nord", "one-dark", "paraiso-dark", "paraiso-light",
    # "pastie", "perldoc", "rainbow_dash", "rrt", "sas", "solarized-dark",
    # "solarized-light", "staroffice", "stata-dark", "stata-light", "tango",
    # "trac", "vim", "vs", "xcode", "zenburn"
    # ]
    
    """

    # The CONFIG dump shows what the run uses, as the snapshot records it (utils/config_view.py).
    config_json = json.dumps(
        resolved_view(safe_cfg, for_log=True),
        indent=2,
        default=str,
    )

    project_cfg = safe_cfg.get("project") or {}

    if not isinstance(project_cfg, dict):
        raise TypeError(
            "safe_cfg['project'] must be a mapping"
        )

    # Prefer the merged runtime configuration.
    if "print_config" in project_cfg:
        print_config_enabled = project_cfg["print_config"]

    else:
        # Some config-loader paths preserve only normalized project fields and
        # discard optional master.yaml keys. Resolve this presentation setting
        # directly from the authoritative selected project entry.
        master_path = Path(
            os.environ.get(
                "SCC_MASTER_YAML",
                "/app/services/side_channel_core/config/master.yaml",
            )
        ).expanduser().resolve()

        if not master_path.is_file():
            raise FileNotFoundError(
                f"Cannot resolve project.print_config: master.yaml not found at "
                f"{master_path}"
            )

        master_cfg = yaml.safe_load(
            master_path.read_text(encoding="utf-8")
        ) or {}

        projects = master_cfg.get("project") or []

        if isinstance(projects, dict):
            projects = [projects]

        selected_project = next(
            (
                item
                for item in projects
                if isinstance(item, dict)
                and str(item.get("name") or "").strip()
                == str(project_name).strip()
            ),
            None,
        )

        if selected_project is None:
            raise KeyError(
                f"Project {project_name!r} not found in {master_path}"
            )

        print_config_enabled = selected_project.get(
            "print_config",
            True,
        )

    if not isinstance(print_config_enabled, bool):
        raise TypeError(
            "project.print_config must be true or false, "
            f"got {print_config_enabled!r}"
        )
    if print_config_enabled:
        syntax = Syntax(
            config_json,
            "json",
            theme="fruity",
            line_numbers=False,
        )

        config_block = Group(
            Text(
                f"Configuring {process_name} for model: {model_name} "
                f"in project: {project_name}",
                style="bold yellow",
            ),
            Text(
                "Loaded config:",
                style="bold red",
            ),
            syntax,
        )

        console.print(config_block)

        try:
            console.file.flush()
        except Exception:
            pass



    # 2. Model-specific file logger
    file_logger = logging.getLogger(f"PrettyConfig_FileOnly_{model_name}_{process_name}")
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False  # do not bubble

    if not file_logger.handlers and model_hash:
        here = Path(__file__).resolve()
        for parent in here.parents:
            reg_file = parent / "artifacts" / "run_registry.json"
            if reg_file.exists():
                reg = _load_sealed_json(reg_file)
                if reg and model_hash in reg:
                    model_logs = Path(reg[model_hash]["path"]) / "logs"
                    model_logs.mkdir(parents=True, exist_ok=True)
                    fh = RotatingFileHandler(
                        model_logs / f"{process_name}.log",
                        maxBytes=10 * 1024 * 1024,
                        backupCount=5
                    )
                    fmt = logging.Formatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S"
                    )
                    fh.setFormatter(fmt)
                    file_logger.addHandler(fh)
                    break

    # 3. Write plain JSON to:
    #  - the model log
    #  - the ROOT logger (which has all_logs.log)
    file_logger.info("---------- CONFIG START ----------")
    for line in config_json.splitlines():
        file_logger.info(line)
    file_logger.info("----------- CONFIG END -----------")

    root_logger = logging.getLogger()
    root_logger.info("---------- CONFIG START ----------")
    for line in config_json.splitlines():
        root_logger.info(line)
    root_logger.info("----------- CONFIG END -----------")

    return config_json
