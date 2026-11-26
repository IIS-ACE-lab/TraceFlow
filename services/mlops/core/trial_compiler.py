# services/mlops/core/trial_compiler.py
"""
Trial Compiler (MLOps → SCC handoff)
===================================

Goal
----
Convert AutoMLOps hypothesis + sampled search parameters into a single SCC-runnable
`run_cfg` dict *without requiring fundamental changes in SCC*.

This file is the contract enforcer:
- MLOps may contain rich “lego” configs (hypotheses, search spaces, pruning rules, etc.)
- SCC should only receive:
    (A) a normal SCC config dict with SCC-known keys updated
    (B) optional namespaced metadata that SCC can safely ignore, but that:
        - affects model hashing/identity (so trials are unique)
        - is written to artifacts for audit/debug

Design rules (high standard)
----------------------------
1) SCC remains the executor. We do not override SCC internals.
2) Only update SCC-owned namespaces when we are confident SCC understands them:
   - cfg["trainer"], cfg["metrics_to_log"]/cfg["metrics"], cfg["seed"], cfg["data"] (optional)
3) Everything else goes under a safe namespace: cfg["_auto_mlops"]
4) Trial uniqueness MUST affect SCC run hashing:
   We inject a small non-breaking field into cfg["model"]["_auto_mlops"].
5) auto_params artifact:
   We always write a trial-level YAML snapshot to a project-scoped path:
     artifacts/{env}/benchmark/auto_mlops/projects/{project_name}/hypotheses/{hypothesis_id}/trials/{trial_id}/auto_params.yaml

Author
------
Shraddha Mishra + production-grade refactor scaffold
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from services.mlops.utils.auto_config import MLOpsConfig
from services.side_channel_core.utils.config_loader import get_project_config
from services.side_channel_core.utils.seal import seal_text


Json = Dict[str, Any]


# ----------------------------
# Small utilities
# ----------------------------

def _deep_merge(a: Any, b: Any) -> Any:
    """
    Deep merge b into a (returns new object).

    - dict + dict: recurse
    - list + list: b overrides (we do NOT concat lists)
    - otherwise: b overrides
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            if k in out:
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = v
        return out
    return b


def _as_int(v: Any, *, ctx: str, strict: bool) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        if strict:
            raise ValueError(f"[trial_compiler] {ctx} must be int")
        return None
    return int(v)


def _as_float(v: Any, *, ctx: str, strict: bool) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        if strict:
            raise ValueError(f"[trial_compiler] {ctx} must be float")
        return None
    return float(v)


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Public data models
# ----------------------------

@dataclass(frozen=True)
class TrialMeta:
    project_name: str
    hypothesis_id: str
    trial_id: str
    env: str
    sota_trial: Optional[str] = None
    auto_params_path: Optional[str] = None


@dataclass(frozen=True)
class CompileResult:
    run_cfg: Json
    meta: TrialMeta
    hypothesis: Json


class TrialCompiler:
    """
    Production-grade trial compiler.

    Typical usage:
        compiler = TrialCompiler()
        hyp = compiler.load_hypothesis_block(project_name="DLSCA_ASCAD", hypothesis_id="H1")

        res = compiler.compile(
            project_name="DLSCA_ASCAD",
            hypothesis_id="H1",
            hypothesis=hyp,
            trial_id="12",
            sampled=optuna_params,
            env="dev",
            sota_trial=None,
        )
        run_cfg = res.run_cfg
        meta = res.meta
        # hand run_cfg to SCC via scc_bridge.run_trial_via_scc(...)
    """

    def __init__(self, *, mlops_cfg: Optional[MLOpsConfig] = None) -> None:
        self.cfg = mlops_cfg or MLOpsConfig()

    # ----------------------------
    # Hypothesis loader (MLOps-owned YAML)
    # ----------------------------

    def load_hypothesis_block(self, *, project_name: str, hypothesis_id: str, env: Optional[str] = None) -> Json:
        """
        Load and return the hypothesis dict from auto_mlops.yaml.

        Expected YAML layout (you already use this):
          project:
            - name: DLSCA_ASCAD
              hypotheses:
                - hypothesis_id: H1
                  ...
        """
        env = str(env or os.environ.get("ENV", "dev")).strip() or "dev"

        auto_cfg = self.cfg.get_map(
            "auto_mlops",
            project_name=project_name,
            resolve_as_path=False,
            mkdir=True,
            strict=True,
            extra={"env": env},
        )
        if not isinstance(auto_cfg, dict):
            raise TypeError("[trial_compiler] auto_mlops config must be a dict")

        projects = auto_cfg.get("project") or []
        if not isinstance(projects, list):
            raise TypeError("[trial_compiler] auto_mlops.yaml 'project' must be a list")

        proj_block: Optional[dict] = None
        for p in projects:
            if isinstance(p, dict) and str(p.get("name", "")).strip() == project_name:
                proj_block = p
                break
        if proj_block is None:
            raise KeyError(f"[trial_compiler] project '{project_name}' not found in auto_mlops.yaml")

        hyps = proj_block.get("hypotheses") or []
        if not isinstance(hyps, list):
            raise TypeError("[trial_compiler] project.hypotheses must be a list")

        for h in hyps:
            if isinstance(h, dict) and str(h.get("hypothesis_id", "")).strip() == hypothesis_id:
                return h

        raise KeyError(f"[trial_compiler] hypothesis '{hypothesis_id}' not found under project '{project_name}'")

    # ----------------------------
    # Compiler entry point
    # ----------------------------

    def compile(
        self,
        *,
        project_name: str,
        hypothesis_id: str,
        hypothesis: Json,
        trial_id: str,
        sampled: Json,
        env: Optional[str] = None,
        sota_trial: Optional[str] = None,
        strict: bool = True,
    ) -> CompileResult:
        """
        Build SCC run_cfg by:
        1) load SCC base config via get_project_config(project_name)
        2) apply safe SCC-known overrides (trainer etc.)
        3) inject auto_mlops metadata + model hash influence
        4) write auto_params trial snapshot for audit (project-scoped path)
        """
        env = str(env or os.environ.get("ENV", "dev")).strip() or "dev"

        if not isinstance(hypothesis, dict):
            raise ValueError("[trial_compiler] hypothesis must be dict")
        if not isinstance(sampled, dict):
            raise ValueError("[trial_compiler] sampled must be dict")

        base_cfg: Json = get_project_config(project_name)
        if not isinstance(base_cfg, dict) or not base_cfg:
            raise ValueError(f"[trial_compiler] SCC base config invalid for project: {project_name}")

        run_cfg: Json = dict(base_cfg)

        # Ensure SCC cfg knows env (SCC already uses this pattern)
        run_cfg["env"] = env

        # -------------
        # (A) SCC-safe merges
        # -------------

        # Trainer overrides (safe subset)
        trainer_overrides = self._compile_trainer_overrides(hypothesis=hypothesis, sampled=sampled, strict=strict)
        if trainer_overrides:
            run_cfg["trainer"] = _deep_merge(run_cfg.get("trainer") or {}, trainer_overrides)

        # Seed override (optional)
        if "seed" in sampled and sampled["seed"] is not None:
            seed_v = sampled["seed"]
            if isinstance(seed_v, bool) or not isinstance(seed_v, int):
                if strict:
                    raise ValueError("[trial_compiler] sampled.seed must be int")
            else:
                run_cfg["seed"] = int(seed_v)

        # Optional dataset override (only if sampled provides SCC-shaped dict)
        if "data" in sampled and isinstance(sampled["data"], dict):
            run_cfg["data"] = dict(sampled["data"])

        # -------------
        # (B) AutoMLOps metadata injection
        # -------------

        auto_block: Json = {
            "trial_id": str(trial_id),
            "hypothesis_id": str(hypothesis_id),
            "project_name": str(project_name),
            "env": env,
            "sota_trial": str(sota_trial) if sota_trial is not None else None,
            "hypothesis": hypothesis,
            "sampled_params": sampled,
        }

        run_cfg["_auto_mlops"] = _deep_merge(run_cfg.get("_auto_mlops") or {}, auto_block)

        # Critical: ensure trial uniqueness influences SCC experiment hash.
        # We inject into cfg["model"] but in a field SCC should ignore.
        model_cfg = dict(run_cfg.get("model") or {})
        model_cfg["_auto_mlops"] = _deep_merge(
            model_cfg.get("_auto_mlops") or {},
            {
                "trial_id": str(trial_id),
                "hypothesis_id": str(hypothesis_id),
                "project_name": str(project_name),
                # keep this small but identity-changing
                "model_spec": self._compile_model_identity_spec(hypothesis=hypothesis, sampled=sampled),
                # also keep a small trainer spec fragment if you want uniqueness from compiler-side knobs
                "trainer_spec": self._compile_trainer_identity_spec(hypothesis=hypothesis, sampled=sampled),
            },
        )
        run_cfg["model"] = model_cfg

        # -------------
        # (C) Persist auto_params snapshot (trial-scoped, project-scoped, no SCC model_name dependency)
        # -------------

        auto_params_path = self._write_trial_auto_params(
            run_cfg=run_cfg,
            project_name=project_name,
            env=env,
            trial_id=str(trial_id),
            hypothesis_id=str(hypothesis_id),
        )

        meta = TrialMeta(
            project_name=str(project_name),
            hypothesis_id=str(hypothesis_id),
            trial_id=str(trial_id),
            env=env,
            sota_trial=str(sota_trial) if sota_trial is not None else None,
            auto_params_path=str(auto_params_path),
        )

        return CompileResult(run_cfg=run_cfg, meta=meta, hypothesis=hypothesis)

    # ----------------------------
    # Internal compilers
    # ----------------------------

    def _compile_trainer_overrides(self, *, hypothesis: Json, sampled: Json, strict: bool) -> Json:
        """
        Produce trainer overrides that SCC likely understands.

        We accept both:
          - hypothesis["trainer"] fixed knobs
          - sampled["trainer"] suggested knobs

        But we map only safe SCC keys:
          epochs, batch_size, loss_reduction, early_stopping, patience, min_delta,
          monitor_metric, optimizer/learning_rate/weight_decay, scheduler.type/scheduler.args,
          gradient.clip_norm, gradient.accumulation_steps, gradient.mixed_precision

        Unknown keys stay in _auto_mlops.
        """
        hyp_tr = hypothesis.get("trainer") or {}
        sp_tr = sampled.get("trainer") or {}

        if hyp_tr and not isinstance(hyp_tr, dict):
            raise ValueError("[trial_compiler] hypothesis.trainer must be dict")
        if sp_tr and not isinstance(sp_tr, dict):
            raise ValueError("[trial_compiler] sampled.trainer must be dict")

        merged = _deep_merge(hyp_tr if isinstance(hyp_tr, dict) else {}, sp_tr if isinstance(sp_tr, dict) else {})
        out: Json = {}

        # scalar safe keys
        for key, ctx in [
            ("epochs", "trainer.epochs"),
            ("batch_size", "trainer.batch_size"),
        ]:
            if key in merged:
                vv = _as_int(merged.get(key), ctx=ctx, strict=strict)
                if vv is not None:
                    out[key] = vv

        for key in ["loss_reduction", "monitor_metric"]:
            if key in merged and merged[key] is not None:
                out[key] = merged[key]

        if "early_stopping" in merged and merged["early_stopping"] is not None:
            if isinstance(merged["early_stopping"], bool):
                out["early_stopping"] = bool(merged["early_stopping"])
            elif strict:
                raise ValueError("[trial_compiler] trainer.early_stopping must be bool")

        for key, ctx, caster in [
            ("patience", "trainer.patience", _as_int),
            ("min_delta", "trainer.min_delta", _as_float),
        ]:
            if key in merged:
                vv = caster(merged.get(key), ctx=ctx, strict=strict)  # type: ignore[arg-type]
                if vv is not None:
                    out[key] = vv

        # optimizer mapping
        opt = merged.get("optimizer")
        if isinstance(opt, str):
            out["optimizer"] = opt
        elif isinstance(opt, dict):
            if "type" in opt and opt["type"] is not None:
                out["optimizer"] = str(opt["type"])
            if "learning_rate" in opt and opt["learning_rate"] is not None:
                out["learning_rate"] = float(opt["learning_rate"])
            if "weight_decay" in opt and opt["weight_decay"] is not None:
                out["weight_decay"] = float(opt["weight_decay"])

            extra_args: Json = {}
            for k in ["betas", "eps"]:
                if k in opt and opt[k] is not None:
                    extra_args[k] = opt[k]
            if extra_args:
                out["optimizer_args"] = extra_args
        elif opt is not None and strict:
            raise ValueError("[trial_compiler] trainer.optimizer must be str|dict")

        # scheduler mapping
        sch = merged.get("scheduler")
        if isinstance(sch, dict):
            stype = sch.get("type")
            if stype is not None:
                out.setdefault("scheduler", {})
                out["scheduler"]["type"] = str(stype)

            sargs = sch.get("args")
            if isinstance(sargs, dict):
                out.setdefault("scheduler", {})
                # Unwrap args by scheduler name if present
                if isinstance(stype, str) and stype in sargs and isinstance(sargs[stype], dict):
                    out["scheduler"]["args"] = sargs[stype]
                else:
                    out["scheduler"]["args"] = sargs
        elif sch is not None and strict:
            raise ValueError("[trial_compiler] trainer.scheduler must be dict if provided")

        # gradient mapping
        grad = merged.get("gradient")
        if isinstance(grad, dict):
            gout: Json = {}
            if "clip_norm" in grad and grad["clip_norm"] is not None:
                gout["clip_norm"] = float(grad["clip_norm"])
            if "accumulation_steps" in grad and grad["accumulation_steps"] is not None:
                gout["accumulation_steps"] = int(grad["accumulation_steps"])
            if "mixed_precision" in grad and grad["mixed_precision"] is not None:
                if isinstance(grad["mixed_precision"], bool):
                    gout["mixed_precision"] = bool(grad["mixed_precision"])
                elif strict:
                    raise ValueError("[trial_compiler] trainer.gradient.mixed_precision must be bool")
            if gout:
                out["gradient"] = gout

        return out

    def _compile_model_identity_spec(self, *, hypothesis: Json, sampled: Json) -> Json:
        """
        Produce a *small* identity spec that:
        - is stable
        - differentiates trials
        - does not break SCC model builder

        This does NOT attempt to generate SCC layers.
        """
        hm = hypothesis.get("model") or {}
        sm = sampled.get("model") or {}

        spec: Json = {}

        if isinstance(hm, dict):
            for k in ["family", "generator"]:
                if k in hm and hm[k] is not None:
                    spec[k] = hm[k]
            if "search_space" in hm and hm["search_space"] is not None:
                spec["search_space"] = hm["search_space"]

        if isinstance(sm, dict) and sm:
            spec["sampled"] = sm

        # If sampled uses dotted keys (like "model.architecture"), preserve too
        dotted_model = {k: v for k, v in sampled.items() if isinstance(k, str) and k.startswith("model.")}
        if dotted_model:
            spec["sampled_dotted"] = dotted_model

        return spec

    def _compile_trainer_identity_spec(self, *, hypothesis: Json, sampled: Json) -> Json:
        """
        Tiny trainer spec fragment for identity (optional).
        Keeps only a few keys so hashing stays stable + small.
        """
        tr = {}
        if isinstance(hypothesis.get("trainer"), dict):
            tr = _deep_merge(tr, hypothesis["trainer"])
        if isinstance(sampled.get("trainer"), dict):
            tr = _deep_merge(tr, sampled["trainer"])

        if not isinstance(tr, dict):
            return {}

        keep = {}
        for k in ("epochs", "batch_size", "learning_rate", "weight_decay", "optimizer", "scheduler"):
            if k in tr:
                keep[k] = tr[k]
        return keep

    def _write_trial_auto_params(
        self,
        *,
        run_cfg: Json,
        project_name: str,
        env: str,
        trial_id: str,
        hypothesis_id: str,
    ) -> Path:
        """
        Always write a trial-scoped snapshot that does NOT depend on SCC model_name.

        Path:
          {project_root}/artifacts/{env}/benchmark/auto_mlops/projects/{project_name}/hypotheses/{hypothesis_id}/trials/{trial_id}/auto_params.yaml
        """
        proj_root = Path(run_cfg.get("project", {}).get("root") or "")
        if not proj_root:
            raise ValueError("[trial_compiler] run_cfg.project.root missing (SCC base cfg should include it)")

        p = (
            proj_root
            / f"artifacts/{env}/benchmark/auto_mlops/projects/{project_name}/hypotheses/{hypothesis_id}/trials/{trial_id}/auto_params.yaml"
        )
        _ensure_parent(p)

        payload = {
            "project_name": project_name,
            "env": env,
            "trial_id": trial_id,
            "hypothesis_id": hypothesis_id,
            "compiled_at": None,  # runner may fill wall-clock; keep compiler pure
            "trainer_overrides": run_cfg.get("trainer"),
            "model_identity": (run_cfg.get("model") or {}).get("_auto_mlops"),
            "_auto_mlops": run_cfg.get("_auto_mlops"),
        }

        p.write_text(seal_text(p, yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)))  # utils/seal.py
        return p
