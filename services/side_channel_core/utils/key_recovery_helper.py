#services/side_channel_core/utils/key_recovery_helper.py
"""
Final completed-cycle key-recovery reporting helper.

This module is called after one model has completed its full lifecycle:

    training
    -> validation
    -> testing
    -> key enumeration
    -> final combined summary

Responsibilities
----------------
- validate key-enumeration configuration;
- run completed-cycle enumeration integration;
- reload the persisted validation/test reports through strict validation;
- require validation/test enumeration modes to match configuration;
- require validation/test policy fingerprints to match;
- print one combined key-recovery summary.
Work flow
---------
The final completed-cycle reporting helper is invoked after one model has
completed its full lifecycle:
    training
    ↓
    validation
    ↓
    testing
    ↓
    completed-cycle enumeration integration
    ↓
    reload persisted reports
    ↓
    strict lifecycle validation
    ↓
    strict enumeration validation
    ↓
    one final combined summary

This module does not implement enumeration algorithms, resolve policies, or
modify candidate ordering.

Enumeration/reporting failures remain non-fatal to an already-successful
training/testing lifecycle. They are logged clearly instead of causing the
model to be retrained.

Author: Shraddha Mishra
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from src.sca.key_enumeration import (
    KeyEnumerationError,
    validate_key_enumeration_contract,
)
from src.sca.perform_attack import attack_plan
from src.sca.key_recovery import (
    KeyRecoveryError,
    key_recovery_outcome,
    key_recovery_paths,
    load_key_recovery_report,
    log_combined_key_recovery_summary,
    validate_key_recovery_contract,
)
from utils.lifecycle_outcome import update_lifecycle_registry_outcome
from utils.path_utils import resolve_path
from utils.key_enumeration_helper import (
    _run_completed_cycle_key_enumeration,
)
from utils.seal import check as check_seal


class KeyRecoveryHelperError(RuntimeError):
    """Raised when final completed-cycle reporting is internally inconsistent."""


def _resolve_model_name(
    cfg: Mapping[str, Any],
) -> str:
    """Resolve the concrete model name used by persisted recovery artifacts."""
    runtime = cfg.get("runtime") or {}

    if not isinstance(runtime, Mapping):
        raise KeyRecoveryHelperError(
            "cfg.runtime must be a mapping"
        )

    model_name = (
        cfg.get("model_name")
        or runtime.get("model_name")
    )

    if (
        model_name is None
        or not str(model_name).strip()
    ):
        raise KeyRecoveryHelperError(
            "model_name is required to resolve key-recovery reports"
        )

    return str(model_name).strip()


def _load_optional_report(
    *,
    path: Path,
    expected_process: str,
    logger: Any,
) -> Optional[Dict[str, Any]]:
    """
    Load one persisted key-recovery report through strict validation.

    Missing reports are represented as ``None`` so the final summary can still
    render an unavailable section. Existing but invalid reports are not hidden;
    validation errors are propagated to the caller.
    """
    if not path.is_file():
        logger.warning(
            "%s key-recovery report is missing: %s",
            expected_process.capitalize(),
            path,
        )
        return None

    report = load_key_recovery_report(
        path
    )

    process = str(
        report.get(
            "process",
            "",
        )
    ).strip().lower()

    if process != expected_process:
        raise KeyRecoveryHelperError(
            f"Expected process={expected_process!r} "
            f"in {path}, got {process!r}"
        )

    return report

def _enumeration_section(
    report: Mapping[str, Any],
    *,
    label: str,
) -> Mapping[str, Any]:
    """Return one required validated enumeration section."""
    enumeration = report.get(
        "enumeration"
    )

    if not isinstance(
        enumeration,
        Mapping,
    ):
        raise KeyRecoveryHelperError(
            f"{label} report is missing a valid enumeration section"
        )

    return enumeration


def _require_matching_lifecycle(
    *,
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> None:
    """Require validation/test reports to belong to one model lifecycle."""
    for field in (
        "schema_version",
        "rank_contract_version",
        "rank_base",
    ):
        validation_value = validation_report.get(field)
        test_value = test_report.get(field)

        if validation_value != test_value:
            raise KeyRecoveryHelperError(
                "Validation and test key-recovery reports use different "
                f"contracts: {field} differs "
                f"({validation_value!r} != {test_value!r})"
            )

    validation_model = (
        validation_report.get(
            "model"
        )
        or {}
    )

    test_model = (
        test_report.get(
            "model"
        )
        or {}
    )

    if not isinstance(
        validation_model,
        Mapping,
    ):
        raise KeyRecoveryHelperError(
            "validation report.model must be a mapping"
        )

    if not isinstance(
        test_model,
        Mapping,
    ):
        raise KeyRecoveryHelperError(
            "test report.model must be a mapping"
        )

    for field in (
        "name",
        "hash",
    ):
        validation_value = (
            validation_model.get(
                field
            )
        )

        test_value = (
            test_model.get(
                field
            )
        )

        if (
            validation_value is None
            or not str(validation_value).strip()
            or test_value is None
            or not str(test_value).strip()
        ):
            raise KeyRecoveryHelperError(
                "Validation and test key-recovery reports must both "
                f"declare a non-empty model.{field}"
            )

        if validation_value != test_value:
            raise KeyRecoveryHelperError(
                "Validation and test key-recovery reports "
                "belong to different model lifecycles: "
                f"model.{field} differs "
                f"({validation_value!r} != {test_value!r})"
            )

    validation_target = validation_report.get("target")
    test_target = test_report.get("target")

    if not isinstance(validation_target, Mapping):
        raise KeyRecoveryHelperError(
            "validation report.target must be a mapping"
        )

    if not isinstance(test_target, Mapping):
        raise KeyRecoveryHelperError(
            "test report.target must be a mapping"
        )

    for field in (
        "cipher",
        "attack_byte",
        "key_mode",
        "leakage_model",
        "keyspace_size",
    ):
        validation_value = validation_target.get(field)
        test_value = test_target.get(field)

        if validation_value != test_value:
            raise KeyRecoveryHelperError(
                "Validation and test key-recovery reports target "
                f"different attacks: target.{field} differs "
                f"({validation_value!r} != {test_value!r})"
            )

    validation_epoch = validation_report.get("epoch")
    if (
        isinstance(validation_epoch, bool)
        or not isinstance(validation_epoch, int)
        or validation_epoch < 1
    ):
        raise KeyRecoveryHelperError(
            "Selected validation report must declare epoch >= 1"
        )

    checkpoint_path = test_model.get("source_checkpoint")
    checkpoint_epoch = test_model.get("source_checkpoint_epoch")
    checkpoint_sha256 = test_model.get("source_checkpoint_sha256")

    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise KeyRecoveryHelperError(
            "Test report must declare a non-empty model.source_checkpoint"
        )

    if (
        isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, int)
        or checkpoint_epoch < 1
    ):
        raise KeyRecoveryHelperError(
            "Test report must declare model.source_checkpoint_epoch >= 1"
        )

    if checkpoint_epoch != validation_epoch:
        raise KeyRecoveryHelperError(
            "Test checkpoint epoch does not match the selected validation "
            f"epoch ({checkpoint_epoch} != {validation_epoch})"
        )

    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_sha256)
    ):
        raise KeyRecoveryHelperError(
            "Test report must declare a lowercase 64-character "
            "model.source_checkpoint_sha256"
        )

def _load_cv_headline(cfg: Mapping[str, Any], logger: Any) -> Dict[str, Any]:
    """Read the CV aggregate headline and each fold's final-epoch report.

    A fold report that cannot be loaded, or that belongs to another model or
    epoch than the headline, is logged as an error and shown with its
    problem, never as this run's result.
    """
    from utils.cross_validation import (
        fold_seed,
        get_fold_artifact_paths,
        resolve_cross_validation_paths,
    )

    cv_paths = resolve_cross_validation_paths(dict(cfg), create=False)
    summary_path = cv_paths.summary
    if not summary_path.is_file():
        raise KeyRecoveryHelperError(
            f"Final CV reporting requires the CV aggregate summary: {summary_path}"
        )
    check_seal(summary_path)  # utils/seal.py
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle) or {}
    headline = summary.get("headline") if isinstance(summary, Mapping) else None
    if not isinstance(headline, Mapping) or not headline:
        raise KeyRecoveryHelperError(f"{summary_path} has no CV headline")
    headline = dict(headline)

    n_folds = int(headline["n_folds"])
    # Folds are assessed at the fixed trainer.epochs; the headline carries it
    # only when the fold-mean curve was computed.
    expected = {
        "model.name": headline.get("model_name"),
        "model.hash": headline.get("model_hash"),
        "epoch": headline.get("epoch") or (summary.get("refit") or {}).get("resolved_epochs"),
    }
    fold_reports = []
    for fold_index in range(n_folds):
        path = get_fold_artifact_paths(
            cv_paths, fold_index=fold_index, n_folds=n_folds, create=False
        )["key_recovery"] / "validation_best.json"
        report = problem = None
        if not path.is_file():
            logger.warning("CV fold %d key-recovery report is missing: %s", fold_index + 1, path)
        else:
            try:
                report = load_key_recovery_report(path)
            except Exception as exc:
                problem = f"report damaged ({path}): {type(exc).__name__}: {exc}"
            else:
                found = {
                    "model.name": (report.get("model") or {}).get("name"),
                    "model.hash": (report.get("model") or {}).get("hash"),
                    "epoch": report.get("epoch"),
                }
                differ = [
                    f"{field} {found[field]!r} != {expected[field]!r}"
                    for field in expected
                    if found[field] != expected[field]
                ]
                if differ:
                    report = None
                    problem = f"report of another run ({path}): {'; '.join(differ)}"
            if problem:
                logger.error(
                    "CV fold %d key-recovery report rejected, %s. Hint: the fold "
                    "reports are rewritten by retraining this model; check that "
                    "no other run writes to %s.",
                    fold_index + 1,
                    problem,
                    path.parent,
                )
        fold_reports.append({
            "fold_number": fold_index + 1,
            "seed": fold_seed(summary["seed"], fold_index) if "seed" in summary else "n/a",
            "report": report,
            "problem": problem,
        })
    # Display-only; never persisted to summary.yaml or the registry.
    headline["fold_reports"] = fold_reports
    return headline


def _require_matching_cv_lifecycle(
    *,
    cv_headline: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> None:
    """Require the CV aggregate and the refit's test report to share one run."""
    test_model = test_report.get("model") or {}
    for field in ("name", "hash"):
        if cv_headline.get(f"model_{field}") != test_model.get(field):
            raise KeyRecoveryHelperError(
                f"CV aggregate and test report differ in model {field}: "
                f"{cv_headline.get(f'model_{field}')!r} != {test_model.get(field)!r}"
            )
    # The refit trains for the same fixed epoch count at which folds were assessed.
    if test_model.get("source_checkpoint_epoch") != cv_headline.get("epoch"):
        raise KeyRecoveryHelperError(
            "Refit checkpoint epoch does not match the CV assessment epoch "
            f"({test_model.get('source_checkpoint_epoch')} != {cv_headline.get('epoch')})"
        )


def _require_matching_enumeration_contract(
    *,
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
    expected_mode: str,
) -> None:
    """
    Require validation/test enumeration evidence to match configuration.

    Ordinary modes
    --------------
    For local_hamming, neighbour_sweep, and support_weighted:

        validation.mode == expected_mode
        test.mode       == expected_mode

    and both reports must use the same frozen policy fingerprint.

    Comparison mode
    ---------------
    Validation owns strategy selection by comparing the raw AI baseline with
    all configured enumeration algorithms:

        validation.mode = comparison

    Validation publishes:

        selected_mode

    When an enumeration algorithm wins, validation additionally publishes:

        selected_policy
        selected_policy_fingerprint

    Test behavior is determined exclusively by the validation winner:

        selected_mode = local_hamming / neighbour_sweep / support_weighted
            -> execute only that frozen algorithm on test

        selected_mode = ai
            -> execute no enumeration on test and preserve the raw
               AI ranking as the selected result

    Test must never re-run comparison or replace the validation winner.
    """
    validation_enumeration = (
        _enumeration_section(
            validation_report,
            label="validation",
        )
    )

    test_enumeration = (
        _enumeration_section(
            test_report,
            label="test",
        )
    )

    validation_mode = str(
        validation_enumeration.get(
            "mode",
            "",
        )
    ).strip().lower()

    test_mode = str(
        test_enumeration.get(
            "mode",
            "",
        )
    ).strip().lower()

    # =============================================================
    # COMPARISON META-MODE
    # =============================================================
    if expected_mode == "comparison":
        if validation_mode != "comparison":
            raise KeyRecoveryHelperError(
                "Comparison configuration requires "
                "validation enumeration.mode='comparison'; "
                f"observed={validation_mode!r}"
            )

        selected_mode = str(
            validation_enumeration.get(
                "selected_mode",
                "",
            )
        ).strip().lower()

        if selected_mode not in {
            "ai",
            "local_hamming",
            "neighbour_sweep",
            "support_weighted",
        }:
            raise KeyRecoveryHelperError(
                "Validation comparison selected_mode must be "
                "ai, local_hamming, neighbour_sweep, or support_weighted; "
                f"observed={selected_mode!r}"
            )

        # ---------------------------------------------------------
        # AI BASELINE WON VALIDATION
        # ---------------------------------------------------------
        if selected_mode == "ai":
            if test_mode != "comparison":
                raise KeyRecoveryHelperError(
                    "AI-selected comparison requires "
                    "test enumeration.mode='comparison'; "
                    f"observed={test_mode!r}"
                )

            if (
                test_enumeration.get(
                    "selected_by_comparison"
                )
                is not True
            ):
                raise KeyRecoveryHelperError(
                    "AI-selected test evidence must declare "
                    "selected_by_comparison=true"
                )

            if (
                str(
                    test_enumeration.get(
                        "selection_split",
                        "",
                    )
                ).strip().lower()
                != "validation"
            ):
                raise KeyRecoveryHelperError(
                    "AI-selected test evidence must declare "
                    "selection_split='validation'"
                )

            test_selected_mode = str(
                test_enumeration.get(
                    "selected_mode",
                    "",
                )
            ).strip().lower()

            if test_selected_mode != "ai":
                raise KeyRecoveryHelperError(
                    "AI-selected test evidence must preserve "
                    "selected_mode='ai'"
                )

            if (
                test_enumeration.get(
                    "executed"
                )
                is not False
            ):
                raise KeyRecoveryHelperError(
                    "AI-selected comparison must not execute "
                    "key enumeration on test"
                )

            return

        # ---------------------------------------------------------
        # ENUMERATION ALGORITHM WON VALIDATION
        # ---------------------------------------------------------
        if test_mode != selected_mode:
            raise KeyRecoveryHelperError(
                "Test enumeration mode does not match the "
                "validation-selected comparison winner: "
                f"selected={selected_mode!r}, "
                f"test={test_mode!r}"
            )

        validation_fingerprint = (
            validation_enumeration.get(
                "selected_policy_fingerprint"
            )
        )

        test_fingerprint = (
            test_enumeration.get(
                "policy_fingerprint"
            )
        )

        if (
            not isinstance(
                validation_fingerprint,
                str,
            )
            or not validation_fingerprint
        ):
            raise KeyRecoveryHelperError(
                "Validation comparison selected policy "
                "fingerprint is missing"
            )

        if (
            not isinstance(
                test_fingerprint,
                str,
            )
            or not test_fingerprint
        ):
            raise KeyRecoveryHelperError(
                "Test enumeration policy fingerprint is missing"
            )

        if (
            validation_fingerprint
            != test_fingerprint
        ):
            raise KeyRecoveryHelperError(
                "Test enumeration did not use the "
                "validation-selected frozen policy"
            )

        selected_policy = (
            validation_enumeration.get(
                "selected_policy"
            )
        )

        if not isinstance(
            selected_policy,
            Mapping,
        ):
            raise KeyRecoveryHelperError(
                "Validation comparison selected_policy "
                "must be a mapping"
            )

        selected_policy_mode = str(
            selected_policy.get(
                "mode",
                "",
            )
        ).strip().lower()

        if selected_policy_mode != selected_mode:
            raise KeyRecoveryHelperError(
                "Validation comparison selected_policy.mode "
                "does not match selected_mode"
            )

        if (
            selected_policy.get(
                "policy_fingerprint"
            )
            != validation_fingerprint
        ):
            raise KeyRecoveryHelperError(
                "Validation comparison selected_policy "
                "fingerprint is inconsistent"
            )

        if (
            test_enumeration.get(
                "policy_frozen"
            )
            is not True
        ):
            raise KeyRecoveryHelperError(
                "Test enumeration policy_frozen must be true"
            )

        if (
            test_enumeration.get(
                "selected_by_comparison"
            )
            is not True
        ):
            raise KeyRecoveryHelperError(
                "Test enumeration must declare "
                "selected_by_comparison=true"
            )

        if (
            str(
                test_enumeration.get(
                    "selection_split",
                    "",
                )
            ).strip().lower()
            != "validation"
        ):
            raise KeyRecoveryHelperError(
                "Comparison-selected test enumeration must "
                "declare selection_split='validation'"
            )

        test_selected_mode = str(
            test_enumeration.get(
                "selected_mode",
                "",
            )
        ).strip().lower()

        if test_selected_mode != selected_mode:
            raise KeyRecoveryHelperError(
                "Test selected_mode metadata does not match "
                "the validation-selected comparison winner"
            )

        return

    # =============================================================
    # ORDINARY SINGLE-MODE CONTRACT
    # =============================================================
    if expected_mode not in {
        "local_hamming",
        "neighbour_sweep",
        "support_weighted",
    }:
        raise KeyRecoveryHelperError(
            "Unsupported expected enumeration mode: "
            f"{expected_mode!r}"
        )

    if validation_mode != expected_mode:
        raise KeyRecoveryHelperError(
            "Validation enumeration mode does not match "
            f"configuration: expected={expected_mode!r}, "
            f"observed={validation_mode!r}"
        )

    if test_mode != expected_mode:
        raise KeyRecoveryHelperError(
            "Test enumeration mode does not match "
            f"configuration: expected={expected_mode!r}, "
            f"observed={test_mode!r}"
        )

    if validation_mode != test_mode:
        raise KeyRecoveryHelperError(
            "Validation and test enumeration modes differ"
        )

    validation_frozen = (
        validation_enumeration.get(
            "policy_frozen"
        )
    )

    test_frozen = (
        test_enumeration.get(
            "policy_frozen"
        )
    )

    if validation_frozen is not True:
        raise KeyRecoveryHelperError(
            "Validation enumeration policy_frozen must be true"
        )

    if test_frozen is not True:
        raise KeyRecoveryHelperError(
            "Test enumeration policy_frozen must be true"
        )

    validation_fingerprint = (
        validation_enumeration.get(
            "policy_fingerprint"
        )
    )

    test_fingerprint = (
        test_enumeration.get(
            "policy_fingerprint"
        )
    )

    if (
        not isinstance(
            validation_fingerprint,
            str,
        )
        or not validation_fingerprint
    ):
        raise KeyRecoveryHelperError(
            "Validation enumeration policy fingerprint is missing"
        )

    if (
        not isinstance(
            test_fingerprint,
            str,
        )
        or not test_fingerprint
    ):
        raise KeyRecoveryHelperError(
            "Test enumeration policy fingerprint is missing"
        )

    if (
        validation_fingerprint
        != test_fingerprint
    ):
        raise KeyRecoveryHelperError(
            "Validation and test enumeration policy fingerprints differ"
        )
def _log_final_summary(
    *,
    cfg: Mapping[str, Any],
    logger: Any,
    validation_report: Optional[
        Mapping[str, Any]
    ],
    test_report: Optional[
        Mapping[str, Any]
    ],
    expected_mode: Optional[str],
    cv_headline: Optional[Mapping[str, Any]] = None,
) -> None:
    """Render exactly one final combined key-recovery summary."""
    log_combined_key_recovery_summary(
        logger,
        cfg=cfg,
        cv_headline=cv_headline,
        validation_report=(
            validation_report
        ),
        test_report=(
            test_report
        ),
        expected_enumeration_mode=(
            expected_mode
        ),
    )


def _enumeration_requested(cfg: Any) -> bool:
    """True when the run asked for key enumeration."""
    enumeration = ((cfg or {}).get("sca") or {}).get("key_enumeration") or {}
    return isinstance(enumeration, Mapping) and enumeration.get("enabled") is True


CV_KEY_RECOVERED = "not applicable in CV as no validation training in final fit."
REGISTRY_OUTCOME_FIELDS = ("key_recovered", "true_key", "recovered_key", "recovered_key_rank")


def _not_attacked(process: str) -> str:
    return f"not available: perform_attack.{process}.attack_times is 0"


def _split_file(cfg: Mapping[str, Any], name: str) -> Path:
    """The run's own data split (paths.data), as data_split writes it."""
    return Path(
        resolve_path(cfg["paths"]["data"], cfg, model_name=_resolve_model_name(cfg))
    ) / name


def _true_key_from_split(cfg: Mapping[str, Any], name: str, group: str) -> Any:
    """
    The key byte at sca.attack_byte of the split's traces: an int for a fixed
    key, else a description; never None, so the registries always say what
    the key is or why it is unknown.
    """
    if _split_file(cfg, name).is_file():
        check_seal(_split_file(cfg, name))  # a changed split stops the run, it is not "unreadable"
    try:
        import h5py
        import numpy as np

        attack_byte = int(cfg["sca"]["attack_byte"])
        with h5py.File(_split_file(cfg, name), "r") as handle:
            keys = np.unique(handle[group]["metadata"]["key"][:, attack_byte])
    except Exception as exc:
        return f"not available: {name} unreadable ({type(exc).__name__}: {exc})"
    if keys.size == 1:
        return int(keys[0])
    return f"variable key: {keys.size} distinct values of key byte {attack_byte} in {name}"


def _format_key(byte: Any) -> str:
    return f"{int(byte)} (0x{int(byte):02X})"


def _split_of(process: str, *, cv_enabled: bool) -> tuple:
    """
    The data split a process attacks: perform_attack.test attacks test.h5,
    perform_attack.validation the CV folds' held-out profiling traces
    (train.h5) or, without CV, val.h5.
    """
    if process == "test":
        return ("test.h5", "Attack_traces")
    if cv_enabled:
        return ("train.h5", "Profiling_traces")
    return ("val.h5", "Attack_traces")


def _true_key(
    cfg: Mapping[str, Any],
    process: str,
    *,
    cv_enabled: bool,
    logger: Any,
    reports: Any = (),
) -> Any:
    """
    true_key for the registries, as "224 (0xE0)": read from the data split the
    data loader wrote and the process attacked; a report's true_key_byte is
    the fallback when the split cannot be read, and a report that disagrees
    with the data is logged.
    """
    data_key = _true_key_from_split(cfg, *_split_of(process, cv_enabled=cv_enabled))
    reported = next(
        (
            (report.get("key") or {}).get("true_key_byte")
            for report in reports
            if isinstance(report, Mapping)
            and (report.get("key") or {}).get("true_key_byte") is not None
        ),
        None,
    )
    if isinstance(data_key, int):
        if reported is not None and int(reported) != data_key:
            logger.warning(
                "%s true_key differs: the data split holds %s, the report says %s; "
                "the registry records the data split's key.",
                process, _format_key(data_key), _format_key(reported),
            )
        return _format_key(data_key)
    if reported is not None:
        return _format_key(reported)
    return data_key


def _registry_outcome(outcome: Mapping[str, Any], true_key: Any) -> Dict[str, Any]:
    """A report's outcome for the lifecycle registries: key bytes as "224 (0xE0)"."""
    fields = dict(outcome, true_key=true_key)
    if isinstance(fields.get("recovered_key"), int):
        fields["recovered_key"] = _format_key(fields["recovered_key"])
    return fields


def _not_attacked_outcome(
    cfg: Mapping[str, Any], process: str, *, cv_enabled: bool, logger: Any
) -> Dict[str, Any]:
    """Registry fields of a process that perform_attack.<process>.attack_times: 0 leaves unattacked."""
    outcome = dict.fromkeys(REGISTRY_OUTCOME_FIELDS, _not_attacked(process))
    outcome["true_key"] = _true_key(cfg, process, cv_enabled=cv_enabled, logger=logger)
    if process == "validation" and cv_enabled:
        outcome["key_recovered"] = CV_KEY_RECOVERED
        # The folds are what perform_attack.validation attacks under CV.
        outcome["fold_key_recovery"] = _not_attacked(process)
    return outcome


def _fold_key_recovery(cv_headline: Optional[Mapping[str, Any]]) -> list:
    """
    Key-recovery outcome of every CV fold: its held-out part, final epoch
    (the fold's validation_best.json), with stable_rank_1 as the summary
    prints it ("@150 /5000", "not reached /5000"). A fold whose report is
    missing or was rejected says so instead of showing another run's result.
    """
    rows = []
    for fold in (cv_headline or {}).get("fold_reports") or []:
        if not isinstance(fold, Mapping):
            continue
        row: Dict[str, Any] = {"fold": fold.get("fold_number")}
        report = fold.get("report")
        if isinstance(report, Mapping):
            outcome = _registry_outcome(key_recovery_outcome(report), None)
            # As in the summary's fold line: the trace count from which the
            # true key stays at rank 1, out of the fold's held-out traces.
            stable = (report.get("recovery") or {}).get("stable_rank1_trace")
            used = (report.get("trace_population") or {}).get("used")
            row.update({
                "key_recovered": outcome["key_recovered"],
                "recovered_key": outcome["recovered_key"],
                "recovered_key_rank": outcome["recovered_key_rank"],
                "stable_rank_1": (
                    (f"@{stable}" if stable is not None else "not reached")
                    + (f" /{used}" if used is not None else "")
                ),
            })
        else:
            row["key_recovered"] = f"not available: {fold.get('problem') or 'fold report missing'}"
        rows.append(row)
    return rows


def _cv_run_outcome(
    cfg: Mapping[str, Any], cv_headline: Optional[Mapping[str, Any]], *, logger: Any
) -> Dict[str, Any]:
    """
    run_registry.json fields of a CV lifecycle whose folds were attacked: the
    final fit is never attacked on validation data, so it has no outcome of
    its own; the true key is the folds' (held-out profiling traces). The
    validation that did run is the folds': fold_key_recovery lists the outcome
    of each.
    """
    folds = [
        fold.get("report")
        for fold in (cv_headline or {}).get("fold_reports") or []
        if isinstance(fold, Mapping)
    ]
    return {
        "key_recovered": CV_KEY_RECOVERED,
        "true_key": _true_key(cfg, "validation", cv_enabled=True, logger=logger, reports=folds),
        "recovered_key": CV_KEY_RECOVERED,
        "recovered_key_rank": CV_KEY_RECOVERED,
        "fold_key_recovery": _fold_key_recovery(cv_headline),
    }


def _record_key_recovery_registries(
    *,
    cfg: Mapping[str, Any],
    logger: Any,
    validation_report: Optional[Mapping[str, Any]],
    test_report: Optional[Mapping[str, Any]],
    cv_headline: Optional[Mapping[str, Any]] = None,
    test_attacked: bool = True,
    validation_attacked: bool = True,
) -> None:
    """
    Record the key-recovery outcome of a completed cycle in the registries.

    test_registry.json carries the test outcome
    (perform_attack.test); run_registry.json carries the validation outcome
    (perform_attack.validation) of the training run, or, for a CV lifecycle,
    the CV reason and the true key. A process that
    perform_attack.<process>.attack_times: 0 leaves unattacked has no report;
    its fields say so, and true_key is read from its data split. Both
    registries are written before key enumeration exists, so they are patched
    here, once the reports are final.

    run_registry.json is touched only when validation_report or cv_headline
    is given or validation_attacked is False (``make test`` passes none).
    """
    artifacts_dir = Path(cfg["project"]["root"]) / "artifacts"
    model_hash = cfg.get("model_hash")
    cv_cfg = cfg.get("cross_validation") or {}
    cv_enabled = isinstance(cv_cfg, Mapping) and cv_cfg.get("enabled") is True

    try:
        if test_report is not None:
            test_fields = key_recovery_outcome(test_report)

            update_lifecycle_registry_outcome(
                registry_path=artifacts_dir / "test_registry.json",
                model_hash=model_hash,
                fields=_registry_outcome(
                    test_fields,
                    _true_key(cfg, "test", cv_enabled=cv_enabled, logger=logger, reports=[test_report]),
                ),
            )
        elif not test_attacked:
            update_lifecycle_registry_outcome(
                registry_path=artifacts_dir / "test_registry.json",
                model_hash=model_hash,
                fields=_not_attacked_outcome(cfg, "test", cv_enabled=cv_enabled, logger=logger),
            )

        if validation_report is not None:
            update_lifecycle_registry_outcome(
                registry_path=artifacts_dir / "run_registry.json",
                model_hash=model_hash,
                fields=_registry_outcome(
                    key_recovery_outcome(validation_report),
                    _true_key(
                        cfg, "validation", cv_enabled=cv_enabled, logger=logger,
                        reports=[validation_report],
                    ),
                ),
            )
        elif not validation_attacked:
            update_lifecycle_registry_outcome(
                registry_path=artifacts_dir / "run_registry.json",
                model_hash=model_hash,
                fields=_not_attacked_outcome(cfg, "validation", cv_enabled=cv_enabled, logger=logger),
            )
        elif cv_enabled and cv_headline is not None:
            update_lifecycle_registry_outcome(
                registry_path=artifacts_dir / "run_registry.json",
                model_hash=model_hash,
                fields=_cv_run_outcome(cfg, cv_headline, logger=logger),
            )

    except Exception as exc:
        # Reporting must not undo a finished lifecycle.
        logger.warning(
            "Key-recovery outcome was not recorded in the registries: %s",
            exc,
        )


def run_final_key_recovery_summary(
    *,
    cfg: Mapping[str, Any],
    logger: Any,
) -> bool:
    """
    Run enumeration and print the final completed-cycle summary.

    Returns
    -------
    bool
        ``True`` when final reporting completed successfully.

        ``False`` when enumeration/report validation failed. Such failures do
        not invalidate an already-successful model training/testing lifecycle.
    """
    try:
        reporting_contract = validate_key_recovery_contract(cfg)

        if not reporting_contract["enabled"]:
            logger.info(
                "Final key-recovery reporting skipped: "
                "key_recovery_reporting.enabled=false"
            )
            return True

        contract = (
            validate_key_enumeration_contract(
                cfg
            )
        )

        expected_mode: Optional[
            str
        ] = None

        if contract[
            "enabled"
        ]:
            mode = contract.get(
                "mode"
            )

            if mode not in {
                "local_hamming",
                "neighbour_sweep",
                "support_weighted",
                "comparison",
            }:
                raise KeyRecoveryHelperError(
                    "Enabled key enumeration has unsupported mode: "
                    f"{mode!r}"
                )

            expected_mode = mode

        model_name = (
            _resolve_model_name(
                cfg
            )
        )

        paths = key_recovery_paths(
            cfg,
            model_name=model_name,
        )

        if contract[
            "enabled"
        ]:
            enumeration_ok = (
                _run_completed_cycle_key_enumeration(
                    cfg=cfg,
                    logger=logger,
                )
            )

            if not enumeration_ok:
                logger.warning(
                    "Final key-enumeration integration failed for "
                    "model=%s mode=%s",
                    model_name,
                    expected_mode,
                )

                cv_on = (cfg.get("cross_validation") or {}).get("enabled") is True
                validation_report = (
                    None
                    if cv_on
                    else _load_optional_report(
                        path=paths[
                            "validation_best"
                        ],
                        expected_process="validation",
                        logger=logger,
                    )
                )

                test_report = (
                    _load_optional_report(
                        path=paths[
                            "test"
                        ],
                        expected_process="test",
                        logger=logger,
                    )
                )

                _log_final_summary(
                    cfg=cfg,
                    logger=logger,
                    validation_report=(
                        validation_report
                    ),
                    test_report=(
                        test_report
                    ),
                    expected_mode=(
                        expected_mode
                    ),
                )

                return False

        cv_cfg = cfg.get("cross_validation") or {}
        cv_enabled = isinstance(cv_cfg, Mapping) and cv_cfg.get("enabled") is True

        validation_report = (
            None
            if cv_enabled
            else _load_optional_report(
                path=paths[
                    "validation_best"
                ],
                expected_process="validation",
                logger=logger,
            )
        )

        # perform_attack.test.attack_times: 0 writes no test.json.
        test_attacked = attack_plan(cfg, "test")["attack_times"] > 0
        test_report = (
            _load_optional_report(
                path=paths[
                    "test"
                ],
                expected_process="test",
                logger=logger,
            )
            if test_attacked
            else None
        )

        # CV folds are weightless and the refit has no validation data, so
        # a CV lifecycle never produces validation_best.json; fold evidence
        # lives under paths.cross_validation; its aggregate headline is shown.

        cv_headline = None
        if cv_enabled:
            cv_headline = _load_cv_headline(cfg, logger)
        elif validation_report is None and attack_plan(cfg, "validation")["attack_times"]:
            raise KeyRecoveryHelperError(
                "Final reporting requires validation_best.json"
            )

        if test_report is None and test_attacked:
            raise KeyRecoveryHelperError(
                "Final reporting requires test.json"
            )

        if test_report is not None and cv_headline is not None:
            _require_matching_cv_lifecycle(
                cv_headline=cv_headline,
                test_report=test_report,
            )
        elif test_report is not None and validation_report is not None:
            _require_matching_lifecycle(
                validation_report=validation_report,
                test_report=test_report,
            )

        if contract[
            "enabled"
        ] and validation_report is not None and test_report is not None:
            # CV has no single validation-selected enumeration strategy.
            if expected_mode is None:
                raise KeyRecoveryHelperError(
                    "Enumeration is enabled but expected_mode "
                    "could not be resolved"
                )

            _require_matching_enumeration_contract(
                validation_report=(
                    validation_report
                ),
                test_report=(
                    test_report
                ),
                expected_mode=(
                    expected_mode
                ),
            )

        _record_key_recovery_registries(
            cfg=cfg,
            logger=logger,
            validation_report=validation_report,
            test_report=test_report,
            cv_headline=cv_headline,
            test_attacked=test_attacked,
            validation_attacked=attack_plan(cfg, "validation")["attack_times"] > 0,
        )

        _log_final_summary(
            cfg=cfg,
            logger=logger,
            validation_report=(
                validation_report
            ),
            test_report=(
                test_report
            ),
            cv_headline=cv_headline,
            expected_mode=(
                expected_mode
            ),
        )

        logger.info(
            "Final key-recovery reporting completed: "
            "model=%s enumeration=%s",
            model_name,
            (
                expected_mode
                if expected_mode is not None
                else "disabled"
            ),
        )

        return True

    except (
        KeyEnumerationError,
        KeyRecoveryError,
        KeyRecoveryHelperError,
    ) as exc:
        # Enumeration is an explicit request: its failure is loud. Reporting
        # faults without enumeration stay a warning, as before.
        if _enumeration_requested(cfg):
            logger.error(
                "Final key-recovery reporting failed: %s",
                exc,
            )
            raise

        logger.warning(
            "Final key-recovery reporting was not completed: %s",
            exc,
            exc_info=True,
        )
        return False

    except Exception as exc:
        if _enumeration_requested(cfg):
            logger.error(
                "Unexpected final key-recovery reporting failure: %s",
                exc,
            )
            raise

        logger.warning(
            "Unexpected final key-recovery reporting failure: %s",
            exc,
            exc_info=True,
        )
        return False


def run_test_key_recovery_summary(
    *,
    cfg: Mapping[str, Any],
    logger: Any,
) -> bool:
    """
    Test-only key-recovery reporting for a fresh re-test (``make test``).

    The test report was just rewritten by the test run. When the model uses
    key enumeration, the strategy is selected on the stored validation
    evidence exactly as at training, and only test.json is enriched. Only
    the test outcome is recorded (test_registry.json) and printed; validation_best.json and
    run_registry.json are not touched.
    """
    reporting_contract = validate_key_recovery_contract(cfg)
    if not reporting_contract["enabled"]:
        logger.info("Test key-recovery reporting skipped: key_recovery_reporting.enabled=false")
        return True
    # perform_attack.test.attack_times: 0 attacks nothing and writes no
    # test.json; a test.json left on disk is never read as this test's outcome.
    if attack_plan(cfg, "test")["attack_times"] == 0:
        logger.info(
            "Test key-recovery reporting skipped: perform_attack.test.attack_times=0 "
            "(the test process is not attacked)"
        )
        _record_key_recovery_registries(
            cfg=cfg,
            logger=logger,
            validation_report=None,
            test_report=None,
            test_attacked=False,
        )
        return True

    model_name = _resolve_model_name(cfg)
    paths = key_recovery_paths(cfg, model_name=model_name)
    cv_cfg = cfg.get("cross_validation") or {}
    cv_enabled = isinstance(cv_cfg, Mapping) and cv_cfg.get("enabled") is True

    contract = validate_key_enumeration_contract(cfg)
    expected_mode = contract.get("mode") if contract["enabled"] else None
    if contract["enabled"] and not cv_enabled:
        if not _run_completed_cycle_key_enumeration(cfg=cfg, logger=logger, test_only=True):
            raise KeyRecoveryHelperError(
                f"Test key enumeration failed for model={model_name} mode={expected_mode}"
            )

    test_report = _load_optional_report(path=paths["test"], expected_process="test", logger=logger)
    if test_report is None:
        raise KeyRecoveryHelperError("Test key-recovery reporting requires test.json")
    # Read only to know which enumeration mode the test ran; never printed.
    validation_report = (
        None
        if cv_enabled or expected_mode is None
        else _load_optional_report(path=paths["validation_best"], expected_process="validation", logger=logger)
    )

    _record_key_recovery_registries(
        cfg=cfg,
        logger=logger,
        validation_report=None,
        test_report=test_report,
    )
    log_combined_key_recovery_summary(
        logger,
        cfg=cfg,
        validation_report=validation_report,
        test_report=test_report,
        expected_enumeration_mode=expected_mode,
        include_validation=False,
    )
    return True


# Preserve the established orchestrator-facing helper name when existing code
# imports it directly. This is not an obsolete algorithm compatibility alias;
# both names invoke the same single completed-cycle reporting path.
def _log_completed_cycle_key_recovery_summary(
    *,
    cfg: Mapping[str, Any],
    logger: Any,
) -> bool:
    """
    Backward-compatible private entry point.

    Historical SCC/MLOps callers may still import this name directly.
    All execution is delegated to the single canonical completed-cycle
    reporting implementation so lifecycle logic is not duplicated.
    """
    return run_final_key_recovery_summary(
        cfg=cfg,
        logger=logger,
    )
def log_final_key_recovery_summary(
    *,
    cfg: Mapping[str, Any],
    logger: Any,
) -> bool:
    """Run the canonical final completed-cycle reporting path."""
    return run_final_key_recovery_summary(
        cfg=cfg,
        logger=logger,
    )


__all__ = [
    "KeyRecoveryHelperError",
    "_log_completed_cycle_key_recovery_summary",
    "log_final_key_recovery_summary",
    "run_final_key_recovery_summary",
    "run_test_key_recovery_summary",

]
