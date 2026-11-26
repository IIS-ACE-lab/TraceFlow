# services/side_channel_core/utils/model_naming.py
"""
model_naming.py
===============

Stable model naming and experiment hashing utilities.

Purpose
-------
- Generate deterministic model_hash from relevant experiment configuration.
- Map model_hash -> human-readable model_name.
- Persist that name mapping separately from AutoSearch run_map.jsonl.

Important
---------
This module must NOT write:

    artifacts/run_map.jsonl
    artifacts/run_map.csv

Those are reserved/removed for AutoSearch run mapping.

This module writes only:

    artifacts/model_name_map.json

Author: Shraddha Mishra
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from services.side_channel_core.utils.atomic_io import atomic_write_text
from services.side_channel_core.utils.logger import get_logger
from services.side_channel_core.utils.path_utils import normalize_paths_for_dump
from services.side_channel_core.utils.seal import load_json
from services.side_channel_core.src.sca.dataset_types import OPERATOR_KEYS, TEST_TIME_KEYS, hash_entry, hash_variant

logger = get_logger("ModelNaming")

MAP_JSON = "model_name_map.json"


def _hashed_evaluation(evaluation: Any) -> Any:
    """The evaluation section without evaluation.validation.judge_log, a log line per epoch
    (utils/validation_judge.py) that changes no number: a run keeps its hash whether it logs or not."""
    validation = evaluation.get("validation") if isinstance(evaluation, dict) else None
    if not isinstance(validation, dict) or "judge_log" not in validation:
        return evaluation                                       # untouched: every hash made without it stays
    validation = {k: v for k, v in validation.items() if k != "judge_log"}
    out = {k: v for k, v in evaluation.items() if k != "validation"}
    if validation:
        out["validation"] = validation
    return out or None                                          # judge_log alone: as if no evaluation section


def _without_empty(node: Any) -> Any:
    """``node`` without its None mapping values, at every depth (lists kept as they are)."""
    if isinstance(node, dict):
        return {key: _without_empty(value) for key, value in node.items() if value is not None}
    return node


def make_experiment_hash(cfg: Dict[str, Any]) -> str:
    """
    Build deterministic 8-char SHA1 hash from fields that define experiment identity.

    This hash must remain stable across reruns for the same effective experiment.

    SCA training identity
    ---------------------
    SCA fields that change the supervised learning problem MUST participate
    in model identity.

    In particular:

        variant
        attack_byte
        leakage_model

    must produce distinct hashes when their resolved runtime values differ.

    The complete ``perform_attack`` and ``evaluation`` sections also join the
    identity (project owner's request): two runs that attack or evaluate
    differently are distinct experiments even when the trained weights match.
    So do the other settings that change a published number: ``metrics_cfg``
    (rank floor, probability contract), ``validation.rank_step`` /
    ``rank_eval_interval`` (single-run rank curve, rank-evaluated epochs),
    ``key_recovery_reporting.recovery_rank`` / ``require_stable_recovery``
    (the recovered flag) and ``device`` (CUDA kernels differ numerically from
    CPU). Display-only reporting and enumeration configuration stays excluded.
    The code version is not hashed (TODO.md).

    Layout independence (2026-09-29, SYS-05): no absolute path is hashed.
    The project joins by name (master.yaml maps it to its root); script,
    distributed-script and data.database paths are hashed relative to the
    project root or the detected workspace root (path_utils.to_portable_path),
    so a host checkout anywhere, the container's /app and a reloaded
    params_out snapshot all give the same hash. A path no relative form maps
    back to (outside both roots, e.g. a dataset the user placed elsewhere, or
    <workspace>/datasets/...) is hashed as written; config_loader warns when
    data.database is one.
    """
    sca = cfg.get("sca") or {}

    if not isinstance(sca, dict):
        raise TypeError(
            "cfg['sca'] must be a mapping when present"
        )

    data = cfg["data"]
    if isinstance(data, dict):
        # Artifact-production switches (generate_data_out, force_regenerate_*, split_chunk_size,
        # contract_verification_samples): none changes the experiment.
        # attack_trace_budget: read only when a trained run is tested (owner 2026-10-27).
        data = {k: v for k, v in data.items() if k not in OPERATOR_KEYS and k not in TEST_TIME_KEYS}
        # The database type (src/sca/dataset_types.py): `type: ASCADf` is left out, so an
        # ASCAD fixed-key run keeps the hash it had before types existed.
        data = hash_entry(data)

    relevant = {
        "data": data,
        "model": cfg["model"],
        "trainer": cfg["trainer"],

        "sca_training_contract": {
            # The configured sca.variant (runs and snapshots made before database types),
            # else the key mode of the run's database type: the same value either way.
            "variant": hash_variant(cfg),
            "attack_byte": sca.get(
                "attack_byte"
            ),
            "leakage_model": sca.get(
                "leakage_model"
            ),
        },

        # Every requested metric, the fit_time ones included (owner ruling
        # 2026-10-14): a run is identified by everything it records, so a run
        # without its training time is not the same experiment as one with it.
        "metrics": sorted(
            cfg.get(
                "metrics_to_log",
                cfg.get(
                    "metrics",
                    [],
                ),
            ),
            key=lambda x: str(x),
        ),

        "seed": cfg["seed"],

        "project": {
            "name": cfg["project"]["name"],
            "scripts": cfg["project"]["scripts"],

            **(
                {
                    "distributed_override": (
                        cfg["project"][
                            "distributed_override"
                        ]
                    )
                }
                if "distributed_override"
                in cfg["project"]
                else {}
            ),

            # extra_configs.import (the model-import list, utils/model_import.py) never joins:
            # adding it to a project must not change the hash of a run it already has.
            **(
                {
                    "extra_configs": {
                        key: value
                        for key, value in cfg["project"]["extra_configs"].items()
                        if key != "import"
                    }
                    if isinstance(cfg["project"]["extra_configs"], dict)
                    else cfg["project"]["extra_configs"]
                }
                if "extra_configs"
                in cfg["project"]
                else {}
            ),
        },

        "distributed": cfg.get(
            "distributed",
            {},
        ),

        # Complete sections, as configured (owner request). Adding them
        # changed every pre-existing hash once; that is accepted.
        # Empty keys dropped (a snapshot shows perform_attack.<process>.attack_trace_budget empty
        # when unset): the hash of every run made before the key, whose configs never held one.
        "perform_attack": _without_empty(cfg.get("perform_attack")),
        "evaluation": _hashed_evaluation(cfg.get("evaluation")),
        "metrics_cfg": cfg.get("metrics_cfg"),
        "validation": {
            key: (cfg.get("validation") or {}).get(key)
            for key in ("rank_step", "rank_eval_interval")
        },
        "key_recovery": {
            key: (cfg.get("key_recovery_reporting") or {}).get(key)
            for key in ("recovery_rank", "require_stable_recovery")
        },
        "device": cfg.get("device"),
    }

    # deep_learning.framework and its initialisers decide which library's
    # functions train the model (core/framework_switch.py). Only when the
    # block is present, keeping every hash made before it stable (without it
    # a run is PyTorch with PyTorch's default initialisation, as before).
    from services.side_channel_core.src.core.framework_switch import resolve_framework

    # A model import (utils/model_import.py): the released file's SHA-256 and format are the
    # weights the run tests, so every file is its own run. Only when present, keeping every
    # trained run's hash.
    imported = cfg.get("model_import")
    if isinstance(imported, dict):
        relevant["model_import"] = {key: imported.get(key) for key in ("sha256", "format")}

    framework_payload = resolve_framework(cfg, validate=False).hash_payload()
    if framework_payload is not None:
        relevant["deep_learning"] = framework_payload

    # CV changes the trained model (full-profiling refit, no validation
    # selection), so it joins the identity. Only when enabled, keeping every
    # existing non-CV hash stable. report_aggregate is reporting-only, and
    # refit.enabled never joins (owner ruling 2026-10-22): the folds and their CV
    # results are the same with or without the final training.
    cv = cfg.get("cross_validation") or {}
    if isinstance(cv, dict) and cv.get("enabled") is True:
        relevant["cross_validation"] = {
            key: cv.get(key)
            for key in ("enabled", "n_folds", "strategy", "shuffle")
        }
        # partition_seed changes which traces each fold trains on: it joins only when set,
        # so every run made without it (the partition from seed) keeps its hash.
        if cv.get("partition_seed") is not None:     # an empty key is no key
            relevant["cross_validation"]["partition_seed"] = cv["partition_seed"]

    # dataloader.drop_last_train decides whether each epoch trains on its last incomplete batch,
    # so it changes the traces trained on and the step count a OneCycleLR is sized from. It joins
    # only when false: true is the default (absent too), and every run made so far trained with it.
    from services.side_channel_core.utils.dataloader_utils import _get_dataloader_cfg

    if _get_dataloader_cfg(cfg)[0]["drop_last_train"] is False:
        relevant["dataloader"] = {"drop_last_train": False}

    # Machine-independent paths only (see docstring).
    relevant = normalize_paths_for_dump(
        relevant,
        project_root=cfg["project"].get("root"),
    )

    buf = json.dumps(
        relevant,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha1(
        buf.encode("utf-8")
    ).hexdigest()[:8]


def _mapping_path(project_root: Path) -> Path:
    return Path(project_root).expanduser().resolve() / "artifacts" / MAP_JSON


def _load_mapping(project_root: Path) -> Dict[str, str]:
    """
    Load model_hash -> model_name mapping.

    Backward compatibility:
    If the old artifacts/run_map.json exists and is a simple hash->name dict,
    it may be migrated manually later. This function does not read it anymore
    because artifacts/run_map.json now belongs to AutoSearch.

    A missing file is an empty map. A damaged one raises: returning {} would
    give every established hash a new name on the next save, detaching it
    from its run directory and params_out snapshot. The file is not
    rewritten while it is damaged.
    """
    path = _mapping_path(project_root)

    if not path.exists():
        return {}

    hint = (
        f"Restore {path} from a backup or fix it by hand (one \"model_hash\": "
        "\"model_name\" pair per run, as in artifacts/run_registry.json); to discard "
        "the runs instead, remove it and run `make delete_database ENV=<env> "
        "PROJECT=<project> CONFIRM=1`. It is not rewritten while it is damaged."
    )

    try:
        data = load_json(path)  # utils/seal.py: verified, seal key dropped
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("%s is unreadable or not valid JSON (%s). %s", path, exc, hint)
        raise ValueError(f"{path} is unreadable or not valid JSON ({exc}). {hint}") from exc

    if not isinstance(data, dict):
        logger.error("%s must hold a JSON object model_hash -> model_name. %s", path, hint)
        raise ValueError(f"{path} must hold a JSON object model_hash -> model_name. {hint}")

    out: Dict[str, str] = {}
    hash_by_name: Dict[str, str] = {}

    for key, value in data.items():
        h = str(key).strip()
        name = value.strip() if isinstance(value, str) else ""

        if not h or not name:
            logger.error("%s has an invalid entry %r: %r. %s", path, key, value, hint)
            raise ValueError(
                f"{path} has an invalid entry {key!r}: {value!r} (expected "
                f"model_hash: model_name, both non-empty strings). {hint}"
            )

        # One name per hash: the name is the run directory and the graph/test selector.
        if name in hash_by_name:
            logger.error("%s maps %s and %s to the same name %r. %s", path, hash_by_name[name], h, name, hint)
            raise ValueError(
                f"{path} maps {hash_by_name[name]} and {h} to the same model_name "
                f"{name!r}. {hint}"
            )

        hash_by_name[name] = h
        out[h] = name

    return out


def _save_mapping(project_root: Path, mapping: Dict[str, str]) -> None:
    """
    Save model_hash -> model_name mapping.

    Writes only:
        artifacts/model_name_map.json (atomically)

    Does not write CSV.
    """
    art = Path(project_root).expanduser().resolve() / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    path = art / MAP_JSON

    clean = {
        str(h): str(name)
        for h, name in mapping.items()
        if str(h).strip() and str(name).strip()
    }

    atomic_write_text(path, json.dumps(clean, indent=2, ensure_ascii=False))


FUNNY_WORDS_CMD = ["funny-words", "-n", "1", "-w", "2", "-d", "_"]


def _generate_funny_name(fallback_hash: str | None = None) -> str:
    """One two-word model_name from the funny-words CLI (pip package funny-words).

    The model_name is the run directory, so there is no substitute name: a
    missing or failing funny-words raises instead of inventing run_<hash>.
    fallback_hash is accepted for older callers and not used.
    """
    try:
        out = subprocess.check_output(FUNNY_WORDS_CMD, stderr=subprocess.DEVNULL)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "funny-words is not installed: it generates every model_name. Install it "
            "(services/side_channel_core/infra/environment/requirements.txt: funny-words==1.7) "
            "in the sca environment and the training image."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"funny-words failed (exit {exc.returncode}); no model_name generated") from exc
    name = out.decode().strip()
    if not name or "None" in name:
        raise RuntimeError(f"funny-words returned no usable name: {name!r}")
    return name


# A generated name is a folder and a shell word (make test MODEL=..., make delete_database MODEL=...): funny-words
# can return punctuation (sickish_o'clock, 2026-10-29), so a name with anything but these characters is drawn again
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def generate_unique_model_name(used_names: set[str], attempts: int = 50) -> str:
    """A funny-words model_name that no other hash uses (the name is the run directory) and that is a plain shell word."""
    for _ in range(attempts):
        name = _generate_funny_name()
        if name not in used_names and SAFE_NAME.fullmatch(name):
            return name
    raise RuntimeError(f"Could not generate a model_name unused by the other hashes in {attempts} attempts")


def choose_run_name(cfg: Dict[str, Any], alias: Optional[str] = None) -> str:
    """
    Mutates cfg:
        - cfg["dataset_alias"]
        - cfg["model_hash"]
        - cfg["model_name"]

    Returns:
        model_name
    """
    h = make_experiment_hash(cfg)

    project_root = Path(cfg["project"]["root"])
    mapping = _load_mapping(project_root)
    # A name belongs to one hash: it is the run directory.
    used_names = {str(v) for k, v in mapping.items() if k != h}

    if alias:
        name = str(alias)
        if name in used_names:
            raise ValueError(
                f"Alias {name!r} is already the model_name of another hash in "
                f"{_mapping_path(project_root)}; choose a different --alias."
            )
        mapping[h] = name

    elif h in mapping:
        name = str(mapping[h])

    else:
        name = generate_unique_model_name(used_names)
        mapping[h] = name

    _save_mapping(project_root, mapping)

    data = cfg.get("data")
    if isinstance(data, dict):
        cfg["dataset_alias"] = alias or data.get("name") or cfg.get("dataset_alias")

    cfg["model_name"] = name
    cfg["model_hash"] = h

    return name
