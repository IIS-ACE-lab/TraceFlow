# services/visualization/clearml/clearml_tracker.py

"""
Central ClearML integration for MLSCA.

Purpose
-------
Provide one reusable, production-oriented adapter for ClearML so that train,
validate, test, inference, sweeps, and reruns can all log through the same API.

Design goals
------------
- Safe to import even when ClearML is not installed.
- No-op behavior when ClearML is disabled in config.
- Stable handling of scalars, dicts, lists, numpy arrays, torch tensors, JSON,
  text, and files.
- Minimal assumptions about the caller.
- Works with your visualization config block:

    visualization:
      clearml:
        enabled: true
        host: "0.0.0.0"
        web_port: 8080
        api_port: 8008
        files_port: 8081
        web_url: "http://localhost:8080"
        api_url: "http://localhost:8008"
        files_url: "http://localhost:8081"

Typical usage
-------------
    tracker = ClearMLTracker.from_cfg(
        cfg=cfg,
        process_name="train",
        project_name=cfg["project"]["name"],
        task_name=model_name,
        tags=["sca", "train"],
    )

    tracker.connect_config(cfg)
    tracker.log_scalar("loss", train_loss, step=epoch, series="train")
    tracker.log_metrics({"accuracy": 0.92, "ge": 12.0}, step=epoch, series="val")
    tracker.upload_file("metrics_csv", csv_path)
    tracker.upload_object("epoch_metrics", metrics_dict)
    tracker.close()
Authors: Shraddha Mishra
"""
# services/visualization/clearml/clearml.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from services.side_channel_core.utils.secrets import SecretError, get_secret
from services.visualization.utils.helpers import _extract_trial_suffix, _probe_tcp, debug_resolve
from services.side_channel_core.utils.logger import pretty_print
from services.side_channel_core.utils.seal import strip as strip_seal
try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from clearml import Task, Logger
except Exception:
    Task = None  # type: ignore
    Logger = None  # type: ignore




JsonLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]



def workspace_root() -> Path:
    """The repository root: WORKSPACE_ROOT (the container's /app), else this checkout
    (services/visualization/clearml/ is three levels below it)."""
    env_root = os.environ.get("WORKSPACE_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    here = Path(__file__).resolve()
    return here.parents[3] if len(here.parents) > 3 else Path("/app")


@dataclass
class ClearMLSettings:
    enabled: bool = False
    scope: str = "local"

    web_url: Optional[str] = None
    api_url: Optional[str] = None
    files_url: Optional[str] = None
    verify_certificate: bool = False

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    access_key_name: str = "CLEARML_API_ACCESS_KEY"
    secret_key_name: str = "CLEARML_API_SECRET_KEY"

    clearml_env_file: Optional[str] = None

    healthcheck_enabled: bool = True
    healthcheck_path: str = "/debug.ping"
    healthcheck_timeout_sec: float = 3.0

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any]) -> "ClearMLSettings":
        project_block = (cfg.get("project") or {})
        visualization_block = cfg.get("visualization") or project_block.get("visualization") or {}
        clearml_block = visualization_block.get("clearml") or {}

        enabled = bool(clearml_block.get("enabled", False))
        scope = str(clearml_block.get("scope", "local")).strip().lower()

        profiles = clearml_block.get("profiles") or {}
        active = profiles.get(scope) or {}

        if not isinstance(active, Mapping):
            active = {}

        return cls(
            enabled=enabled,
            scope=scope,

            web_url=active.get("web_url"),
            api_url=active.get("api_url"),
            files_url=active.get("files_url"),
            verify_certificate=bool(active.get("verify_certificate", False)),

            access_key=active.get("access_key"),
            secret_key=active.get("secret_key"),
            access_key_name=str(active.get("access_key_name", "CLEARML_API_ACCESS_KEY")),
            secret_key_name=str(active.get("secret_key_name", "CLEARML_API_SECRET_KEY")),

            clearml_env_file=active.get("clearml_env_file"),

            healthcheck_enabled=bool(active.get("healthcheck_enabled", True)),
            healthcheck_path=str(active.get("healthcheck_path", "/debug.ping")),
            healthcheck_timeout_sec=float(active.get("healthcheck_timeout_sec", 3.0)),
        )


@dataclass
class ClearMLTracker:
    settings: ClearMLSettings
    process_name: str
    project_name: str
    task_name: str
    tags: List[str] = field(default_factory=list)
    task_type: str = "training"
    output_uri: Optional[str] = None
    task: Any = None
    logger: Any = None
    disabled_reason: Optional[str] = None
    _cfg: Optional[Mapping[str, Any]] = None
    @classmethod
    def from_cfg(
        cls,
        cfg: Mapping[str, Any],
        process_name: str,
        project_name: str,
        task_name: str,
        tags: Optional[Sequence[str]] = None,
        task_type: str = "training",
        output_uri: Optional[str] = None,
    ) -> "ClearMLTracker":
        settings = ClearMLSettings.from_cfg(cfg)
        tracker = cls(
            settings=settings,
            process_name=process_name,
            project_name=project_name,
            task_name=task_name,
            tags=list(tags or []),
            task_type=task_type,
            output_uri=output_uri,
        )
        tracker._cfg = cfg
        tracker._initialize()
        return tracker

    @property
    def enabled(self) -> bool:
        return self.task is not None and self.logger is not None

    def _resolve_secret(self, explicit_value: Optional[str], secret_name: str) -> Optional[str]:
        if explicit_value:
            return explicit_value

        secret_file = getattr(self.settings, "clearml_env_file", None)

        try:
            return get_secret(
                secret_name,
                secret_file=secret_file,
                required=True,
            )
        except SecretError as e:
            self.disabled_reason = (
                f"clearml secret lookup failed for {secret_name!r} "
                f"from {secret_file!r}: {e}"
            )
            return None
        except Exception as e:
            self.disabled_reason = (
                f"unexpected clearml secret lookup error for {secret_name!r} "
                f"from {secret_file!r}: {e}"
            )
            return None

    def _build_healthcheck_url(self) -> str:
        base = (self.settings.api_url or "").rstrip("/")
        path = self.settings.healthcheck_path.strip()
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    def _health_check(self) -> Tuple[bool, Optional[str]]:
        if not self.settings.healthcheck_enabled:
            return True, None

        url = self._build_healthcheck_url()
        try:
            req = Request(url, method="GET")
            with urlopen(req, timeout=self.settings.healthcheck_timeout_sec) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                body = resp.read().decode("utf-8", errors="replace").strip()

            if 200 <= status < 300:
                lowered = body.lower()
                if not body or any(token in lowered for token in ("pong", "ok", "alive")):
                    return True, None
                return True, None

            return False, f"clearml api healthcheck failed: status={status} url={url}"

        except HTTPError as e:
            return False, f"clearml api healthcheck HTTPError: code={e.code} url={url}"
        except URLError as e:
            return False, f"clearml api healthcheck URLError: {e.reason} url={url}"
        except Exception as e:
            return False, f"clearml api healthcheck failed: {e}"

    def _initialize(self) -> None:
        if not self.settings.enabled:
            self.disabled_reason = (
                f"clearml disabled under visualization.clearml.enabled "
                f"for project {self.project_name}"
            )
            return

        if not self.settings.web_url or not self.settings.api_url or not self.settings.files_url:
            self.disabled_reason = (
                f"clearml urls are incomplete for scope={self.settings.scope!r}: "
                f"web_url={self.settings.web_url!r}, "
                f"api_url={self.settings.api_url!r}, "
                f"files_url={self.settings.files_url!r}"
            )
            return

        # pretty_print(
        #     f"[ClearML CONFIGURED]\n"
        #     f"  scope   : {self.settings.scope}\n"
        #     f"  web APP : {self.settings.web_url}\n"
        #     f"  api     : {self.settings.api_url}\n"
        #     f"  files   : {self.settings.files_url}\n",
        #     color="cyan",
        # )

        secret_path_str = self.settings.clearml_env_file
        if not secret_path_str:
            self.disabled_reason = "clearml_env_file not defined in config"
            return

        secret_path = Path(secret_path_str)
        if not secret_path.is_absolute():
            # master.yaml names the file relative to the repository (secrets/clearml_global.env): /app in
            # the container (WORKSPACE_ROOT), the checkout on the host. The relative path is what the
            # run's snapshot records, so a snapshot holds no container path (audit 2026-10-23).
            secret_path = workspace_root() / secret_path

        if not secret_path.exists():
            self.disabled_reason = f"clearml secret file missing at {secret_path}"
            return

        self.settings.clearml_env_file = str(secret_path)

        if Task is None or Logger is None:
            self.disabled_reason = "clearml package not installed or failed to import in current Python environment"
            return

        access_key = self._resolve_secret(
            explicit_value=self.settings.access_key,
            secret_name=self.settings.access_key_name,
        )
        secret_key = self._resolve_secret(
            explicit_value=self.settings.secret_key,
            secret_name=self.settings.secret_key_name,
        )

        if not access_key or not secret_key:
            self.disabled_reason = (
                "missing clearml credentials from secrets provider "
                f"(scope={self.settings.scope!r}, secret_file={self.settings.clearml_env_file!r})"
            )
            return

        os.environ["CLEARML_API_HOST"] = self.settings.api_url
        os.environ["CLEARML_WEB_HOST"] = self.settings.web_url
        os.environ["CLEARML_FILES_HOST"] = self.settings.files_url
        os.environ["CLEARML_API_ACCESS_KEY"] = access_key
        os.environ["CLEARML_API_SECRET_KEY"] = secret_key
        os.environ["CLEARML_API_HOST_VERIFY_CERT"] = (
            "1" if self.settings.verify_certificate else "0"
        )
        ok_tcp, tcp_reason = _probe_tcp(self.settings.api_url, timeout=self.settings.healthcheck_timeout_sec)
        pretty_print(f"[ClearML TCP PROBE] {tcp_reason}", color= "red" if not ok_tcp else "yellow")
        # Optional fail-fast (config driven)
        clearml_block = (self._cfg.get("visualization") or {}).get("clearml") or {}
        pretty_print(f"[ClearML RESOLVE] {debug_resolve(self.settings.api_url)}", color="cyan")
        if not ok_tcp and clearml_block.get("fail_fast_on_unreachable_api", True):
            self.disabled_reason = tcp_reason
            return
        ok, reason = self._health_check()
        if not ok:
            pretty_print(
                f"[ClearML WARNING] Preflight healthcheck failed for scope={self.settings.scope!r}: {reason}. "
                f"Proceeding to Task.init() using configured URLs.",
                color="yellow",
            )


        task_type_enum = self._resolve_task_type(self.task_type)

        try:
            self.task = Task.init(
                project_name=self.project_name,
                task_name=self.task_name,
                task_type=task_type_enum,
                tags=self.tags + [f"clearml_scope:{self.settings.scope}"],
                output_uri=self.output_uri,
                reuse_last_task_id=False,
                continue_last_task=False,
            )
            self.logger = self.task.get_logger()
            pretty_print(
                    f"[ClearML ACTIVE]\n"
                    f"  scope   : {self.settings.scope}\n"
                    f"  web APP : {self.settings.web_url}\n",
                    color="cyan",
                )


        except Exception as e:
            self.task = None
            self.logger = None
            self.disabled_reason = f"clearml init failed: {e}"

    def _resolve_task_type(self, task_type: str) -> Any:
        if Task is None:
            return task_type
        task_type_str = (task_type or "training").strip().lower()
        mapping = {
            "training": Task.TaskTypes.training,
            "testing": Task.TaskTypes.testing,
            "inference": Task.TaskTypes.inference,
            "data_processing": Task.TaskTypes.data_processing,
            "optimizer": Task.TaskTypes.optimizer,
            "service": Task.TaskTypes.service,
            "monitor": Task.TaskTypes.monitor,
            "controller": Task.TaskTypes.controller,
            "qc": Task.TaskTypes.qc,
            "custom": Task.TaskTypes.custom,
        }
        return mapping.get(task_type_str, Task.TaskTypes.training)

    def connect_config(self, cfg: Mapping[str, Any], name: str = "config") -> None:
        if not self.enabled:
            return
        self.task.connect(self._to_serializable(dict(cfg)), name=name)

    def connect_parameters(self, params: Mapping[str, Any], name: str = "parameters") -> None:
        if not self.enabled:
            return
        self.task.connect(self._to_serializable(dict(params)), name=name)

    def set_comment(self, comment: str) -> None:
        if not self.enabled:
            return
        self.task.set_comment(comment)

    def log_scalar(self, title: str, value: Any, step: int, series: str = "default") -> None:
        if not self.enabled:
            return
        scalar = self._maybe_scalar(value)
        if scalar is None:
            return
        self.logger.report_scalar(title=title, series=series, value=scalar, iteration=int(step))

    def log_metrics(self, metrics: Mapping[str, Any], step: int, series: str = "default") -> None:
        if not self.enabled:
            return
        for key, value in metrics.items():
            scalar = self._maybe_scalar(value)
            if scalar is not None:
                self.logger.report_scalar(
                    title=str(key),
                    series=series,
                    value=scalar,
                    iteration=int(step),
                )

    def log_curve(
        self,
        title: str,
        x: Sequence[Any],
        y: Sequence[Any],
        step: int,
        series: str = "default",
        xaxis: str = "x",
        yaxis: str = "y",
    ) -> None:
        if not self.enabled:
            return
        x_vals = list(self._flatten_numeric(x))
        y_vals = list(self._flatten_numeric(y))
        if not x_vals or not y_vals or len(x_vals) != len(y_vals):
            return
        self.logger.report_scatter2d(
        title=title,
        series=series,
        scatter=list(zip(x_vals, y_vals)),
        iteration=int(step),
        xaxis=xaxis,
        yaxis=yaxis,
        mode="lines",
            )

    def log_curve_from_2d_array(
        self,
        title: str,
        curve: Any,
        step: int,
        series: str = "default",
        xaxis: str = "x",
        yaxis: str = "y",
    ) -> None:
        arr = self._to_numpy(curve)
        if arr is None or arr.ndim != 2 or arr.shape[1] < 2:
            return
        self.log_curve(
            title=title,
            x=arr[:, 0].tolist(),
            y=arr[:, 1].tolist(),
            step=step,
            series=series,
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def log_table(self, title: str, table: Sequence[Mapping[str, Any]], step: int, series: str = "default") -> None:
        if not self.enabled:
            return
        rows = [self._to_serializable(dict(row)) for row in table]
        self.logger.report_table(title=title, series=series, iteration=int(step), table_plot=rows)

    def upload_file(self, name: str, path: Union[str, Path]) -> None:
        if not self.enabled:
            return
        path = Path(path)
        if not path.exists():
            return
        self.task.upload_artifact(name=name, artifact_object=str(path))

    def upload_object(self, name: str, obj: Any, metadata: Optional[Mapping[str, Any]] = None) -> None:
        if not self.enabled:
            return
        safe_obj = self._to_serializable(obj)
        self.task.upload_artifact(
            name=name,
            artifact_object=safe_obj,
            metadata=self._to_serializable(dict(metadata or {})),
        )

    def upload_json_file(self, name: str, path: Union[str, Path]) -> None:
        if not self.enabled:
            return
        path = Path(path)
        if not path.exists():
            return
        try:
            payload = strip_seal(json.loads(path.read_text()))
        except Exception:
            self.upload_file(name=name, path=path)
            return
        self.upload_object(name=name, obj=payload)

    def log_mixed_metrics(
        self,
        payload: Mapping[str, Any],
        step: int,
        series: str = "default",
        curve_titles: Optional[Mapping[str, Tuple[str, str]]] = None,
    ) -> None:
        if not self.enabled:
            return
        curve_titles = dict(curve_titles or {})
        for key, value in payload.items():
            scalar = self._maybe_scalar(value)
            if scalar is not None:
                self.log_scalar(title=str(key), value=scalar, step=step, series=series)
                continue
            arr = self._to_numpy(value)
            if arr is not None and arr.ndim == 2 and arr.shape[1] >= 2:
                xaxis, yaxis = curve_titles.get(str(key), ("x", str(key)))
                self.log_curve_from_2d_array(
                    title=str(key),
                    curve=arr,
                    step=step,
                    series=series,
                    xaxis=xaxis,
                    yaxis=yaxis,
                )
                continue
            if isinstance(value, (dict, list, tuple)):
                self.upload_object(name=f"{series}_{key}_step_{step}", obj=value)

    def flush(self) -> None:
        if not self.enabled:
            return
        try:
            self.task.flush(wait_for_uploads=True)
        except Exception:
            pass

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.task.close()
        except Exception:
            pass

    def _maybe_scalar(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if np is not None and isinstance(value, np.generic):
            return float(value)
        if torch is not None and isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return None
        return None

    def _to_numpy(self, value: Any) -> Optional[Any]:
        if np is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, (list, tuple)):
            try:
                return np.asarray(value)
            except Exception:
                return None
        return None

    def _flatten_numeric(self, values: Sequence[Any]) -> Iterable[float]:
        for v in values:
            scalar = self._maybe_scalar(v)
            if scalar is not None:
                yield scalar

    def _to_serializable(self, obj: Any) -> JsonLike:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if np is not None:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
        if torch is not None and isinstance(obj, torch.Tensor):
            if obj.numel() == 1:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Mapping):
            return {str(k): self._to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_serializable(v) for v in obj]
        return str(obj)


class NullTracker(ClearMLTracker):
    def __init__(self) -> None:
        super().__init__(
            settings=ClearMLSettings(enabled=False),
            process_name="noop",
            project_name="noop",
            task_name="noop",
        )
        self.disabled_reason = "null tracker"


def build_tracker_from_run_context(
    cfg: Mapping[str, Any],
    process_name: str,
    model_name: str,
    model_hash: Optional[str] = None,
    extra_tags: Optional[Sequence[str]] = None,
    task_type: str = "training",
) -> ClearMLTracker:
    project_block = cfg.get("project") or {}
    project_name = str(project_block.get("name", "MLSCA"))
    env_name = str(cfg.get("env", "dev"))

    tags: List[str] = [
        "MLSCA",
        f"group:{model_name}",
        f"process:{process_name}",
        f"env:{env_name}",
    ]

    if model_hash:
        tags.append(f"model_hash:{model_hash}")

    if extra_tags:
        tags.extend([str(t) for t in extra_tags])

    trial_suffix = _extract_trial_suffix(cfg)
    task_name = f"{model_name}_{process_name}{trial_suffix}"

    return ClearMLTracker.from_cfg(
        cfg=cfg,
        process_name=process_name,
        project_name=project_name,
        task_name=task_name,
        tags=tags,
        task_type=task_type,
    )
