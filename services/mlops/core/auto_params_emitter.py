# services/mlops/core/auto_params_emitter.py
# ============================================================
# Auto-params emitter (PURE LOGIC)
# - Turns hypothesis search spaces into proposed params (dict)
# - Deterministic: same inputs -> same outputs
# - No Optuna / no training calls
# ============================================================

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

Json = Dict[str, Any]


# ============================================================
# Data model
# ============================================================

@dataclass(frozen=True)
class ProposedParams:
    """
    Proposed params payload for a single trial.

    `params` is a nested dict suitable for merging into a params/config YAML.
    `flat` is a dotted-key map for logging / registries.
    `choices` records which options were selected for each search-space key.
    """
    params: Json
    flat: Dict[str, Any]
    choices: Dict[str, Any] = field(default_factory=dict)

    hypothesis_id: str = ""
    trial_id: int = 0
    seed: str = ""


# ============================================================
# Public API
# ============================================================

def emit_auto_params(
    compiled: Json,
    *,
    hypothesis_id: str,
    trial_id: int,
    seed_salt: str = "MLSCA_AUTO_PARAMS_V1",
    include_blocks: Optional[Sequence[str]] = None,
    exclude_blocks: Optional[Sequence[str]] = None,
) -> ProposedParams:
    """
    Build proposed parameters for a given hypothesis + trial.

    Inputs:
      - compiled: deterministic compiled view (core/context.py)
      - hypothesis_id: e.g. "H1"
      - trial_id: monotonic integer (used to choose from spaces deterministically)
      - include_blocks/exclude_blocks: which hypothesis blocks to emit (default emits all known)

    Output:
      ProposedParams:
        params: nested dict to write as auto_params.yaml
        flat: dotted dict for quick lookup/logging
        choices: dotted dict recording selected indices/values for list spaces
    """
    if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
        raise ValueError("[core/auto_params_emitter] trial_id must be non-negative int")

    hyp = _get_hypothesis_block(compiled, hypothesis_id=hypothesis_id)

    # Default blocks to include: (only those present will be included)
    default_blocks = ("learning", "model", "trainer", "pruning", "benchmarking")
    blocks = list(include_blocks) if include_blocks is not None else list(default_blocks)
    if exclude_blocks:
        ex = {str(x).strip() for x in exclude_blocks if isinstance(x, str) and x.strip()}
        blocks = [b for b in blocks if b not in ex]

    # Deterministic seed string
    seed = _stable_seed(
        project_name=str((compiled.get("project") or {}).get("name") or ""),
        hypothesis_id=hypothesis_id,
        trial_id=trial_id,
        salt=seed_salt,
    )

    out_params: Json = {
        "meta": {
            "project": str((compiled.get("project") or {}).get("name") or ""),
            "env": str(compiled.get("env") or ""),
            "model_name": str(compiled.get("model_name") or ""),
            "hypothesis_id": hypothesis_id,
            "trial_id": int(trial_id),
            "seed": seed,
            "seed_salt": seed_salt,
        }
    }

    flat: Dict[str, Any] = {}
    choices: Dict[str, Any] = {}

    for block in blocks:
        if block not in hyp:
            continue
        raw = hyp.get(block)
        selected = _select_from_space(
            raw,
            trial_id=trial_id,
            seed=seed,
            prefix=f"{block}",
            flat_out=flat,
            choices_out=choices,
        )
        out_params[block] = selected

    # Also emit naming helpers if available
    _inject_model_tag_helpers(compiled, hypothesis=hyp, params=out_params, flat=flat)

    return ProposedParams(
        params=out_params,
        flat=flat,
        choices=choices,
        hypothesis_id=hypothesis_id,
        trial_id=trial_id,
        seed=seed,
    )


def emit_auto_params_flat_only(
    compiled: Json,
    *,
    hypothesis_id: str,
    trial_id: int,
    seed_salt: str = "MLSCA_AUTO_PARAMS_V1",
) -> Dict[str, Any]:
    """
    Convenience: returns dotted flat map only.
    """
    return emit_auto_params(compiled, hypothesis_id=hypothesis_id, trial_id=trial_id, seed_salt=seed_salt).flat


# ============================================================
# Internals
# ============================================================

def _get_hypothesis_block(compiled: Json, *, hypothesis_id: str) -> Json:
    p = compiled.get("project") or {}
    hyps = p.get("hypotheses") or []
    if not isinstance(hyps, list):
        raise ValueError("[core/auto_params_emitter] compiled.project.hypotheses must be a list")

    hid = str(hypothesis_id).strip()
    for h in hyps:
        if isinstance(h, dict):
            if str(h.get("hypothesis_id") or h.get("id") or "").strip() == hid:
                return h
    raise KeyError(f"[core/auto_params_emitter] hypothesis_id not found in compiled: {hid}")


def _stable_seed(*, project_name: str, hypothesis_id: str, trial_id: int, salt: str) -> str:
    msg = f"{salt}::proj={project_name}::hyp={hypothesis_id}::trial={trial_id}"
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()[:32]


def _pick_index(*, seed: str, key: str, trial_id: int, n: int) -> int:
    """
    Deterministic pick in [0, n-1] based on (seed, key, trial_id).
    """
    if n <= 0:
        raise ValueError("[core/auto_params_emitter] cannot pick index from empty list")
    h = hashlib.sha256(f"{seed}::{key}::{trial_id}".encode("utf-8")).digest()
    x = int.from_bytes(h[:8], "big", signed=False)
    return x % n


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool))


def _select_from_space(
    obj: Any,
    *,
    trial_id: int,
    seed: str,
    prefix: str,
    flat_out: Dict[str, Any],
    choices_out: Dict[str, Any],
) -> Any:
    """
    Selects deterministically from a "search space" object.

    Supported patterns:
      - scalar -> scalar
      - list of scalars -> choose one
      - list of lists (e.g. betas) -> choose one (keeps list)
      - dict -> recurse values
      - list of dicts -> choose one dict then recurse

    NOTE: This is intentionally generic and does NOT encode trainer/model semantics.
    """
    # None stays None
    if obj is None:
        flat_out[prefix] = None
        return None

    # Scalar: fixed
    if _is_scalar(obj):
        flat_out[prefix] = obj
        return obj

    # List: choose element (or recurse on chosen)
    if isinstance(obj, list):
        if not obj:
            raise ValueError(f"[core/auto_params_emitter] empty list space at '{prefix}'")

        idx = _pick_index(seed=seed, key=prefix, trial_id=trial_id, n=len(obj))
        chosen = obj[idx]

        choices_out[prefix] = {"kind": "list", "index": idx, "n": len(obj), "value": _safe_preview(chosen)}

        # If list-of-dicts, recurse into chosen dict (so nested spaces inside work)
        if isinstance(chosen, dict):
            selected_dict: Json = {}
            for k, v in chosen.items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError(f"[core/auto_params_emitter] invalid dict key in list space at '{prefix}'")
                selected_dict[k] = _select_from_space(
                    v,
                    trial_id=trial_id,
                    seed=seed,
                    prefix=f"{prefix}.{k}",
                    flat_out=flat_out,
                    choices_out=choices_out,
                )
            # record the chosen dict itself too (compact)
            flat_out[prefix] = _safe_preview(selected_dict)
            return selected_dict

        # If list-of-lists (e.g. betas pairs), keep list
        if isinstance(chosen, list):
            flat_out[prefix] = chosen
            return chosen

        # scalar chosen
        if _is_scalar(chosen) or chosen is None:
            flat_out[prefix] = chosen
            return chosen

        # Unknown chosen type
        flat_out[prefix] = _safe_preview(chosen)
        return chosen

    # Dict: recurse each field
    if isinstance(obj, dict):
        out: Json = {}
        for k, v in obj.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError(f"[core/auto_params_emitter] invalid dict key at '{prefix}'")
            out[k] = _select_from_space(
                v,
                trial_id=trial_id,
                seed=seed,
                prefix=f"{prefix}.{k}",
                flat_out=flat_out,
                choices_out=choices_out,
            )
        flat_out[prefix] = _safe_preview(out)
        return out

    # Fallback: keep as-is (but log preview)
    flat_out[prefix] = _safe_preview(obj)
    return obj


def _safe_preview(v: Any) -> Any:
    """
    Keeps flat logs readable; avoids exploding large nested structures.
    """
    try:
        if _is_scalar(v) or v is None:
            return v
        if isinstance(v, list):
            if len(v) <= 6:
                return v
            return {"type": "list", "len": len(v), "head": v[:3], "tail": v[-2:]}
        if isinstance(v, dict):
            if len(v) <= 8:
                return v
            keys = list(v.keys())
            return {"type": "dict", "len": len(v), "keys_head": keys[:6]}
        return str(v)
    except Exception:
        return "<unpreviewable>"


def _inject_model_tag_helpers(compiled: Json, *, hypothesis: Json, params: Json, flat: Dict[str, Any]) -> None:
    """
    Optional helpers:
      - model_tag: derived from auto_mlops.yaml trial_naming.model_tag_from
    This is still pure and deterministic.
    """
    search = compiled.get("search") or {}
    tn = (search.get("trial_naming") or {})
    tag_from = tn.get("model_tag_from") or []
    if not isinstance(tag_from, list) or not tag_from:
        return

    # Read values from params using dotted paths in model_tag_from.
    parts: List[str] = []
    for path in tag_from:
        if not isinstance(path, str) or not path.strip():
            continue
        val = _get_dotted(params, path.strip())
        if val is None:
            # try compiled as fallback (sometimes they reference global)
            val = _get_dotted(compiled, path.strip())
        if isinstance(val, (str, int, float, bool)) and str(val).strip():
            parts.append(str(val).strip())
    if not parts:
        return

    tag = "_".join(parts)[:128]
    params.setdefault("meta", {})
    if isinstance(params["meta"], dict):
        params["meta"]["model_tag"] = tag
    flat["meta.model_tag"] = tag


def _get_dotted(d: Any, path: str) -> Any:
    cur = d
    for part in path.split("."):
        if not part:
            return None
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur
