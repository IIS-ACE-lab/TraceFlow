# services/side_channel_core/utils/delete_database.py
"""
Delete the run database of one environment, or of named models, from a
project's ``artifacts/``, ``model_selection/`` and ``graphs/`` directories.

What "the database" is for one project::

    artifacts/run_registry.json            hash -> training lifecycle (has "env")
    artifacts/test_registry.json           hash -> test lifecycle (has "env")
    artifacts/run_map.jsonl                JSONL, one record per run (has "env")
    artifacts/model_name_map.json          hash -> model_name
    artifacts/done_model_names.yaml        model names offered to graph jobs
    config/graph_config.yaml               the deleted names leave its jobs (graph_export.prune)
    artifacts/<env>/generated_models/<model_name>/   every file the run wrote
    artifacts/<env>/tmux_logs/<batch>.txt  never: a batch's make/tmux output (every run of it, appended run
                                           after run) is kept by every deletion, a whole env's or project's too
    artifacts/filtered_models.yaml         output of make find_model (has "env")
    model_selection/*                      output of make judge (has "env")

Scope. Only one project is touched, the one given by ``--project-root``;
inside it only the runs of ``--env``. ``--dataset`` narrows that to the runs
trained on the named dataset(s), ``--model`` to the named runs (model name or
hash), ``--batch`` to the runs labelled with the named batch(es) (the
run_registry ``batches`` of ``make experiment QUEUE=`` / ``BATCH=``); they may
be combined, and a run must match each one given. A run's dataset is read from the config snapshot
in its directory (``config/<model_name>_params.yaml``, ``data.name``); a run
without a snapshot has no known dataset and is kept by a ``--dataset``
deletion.

Without ``--model``, ``--dataset`` and ``--batch`` every run of the environment goes,
including stale registry rows whose hash no longer maps to a sweep point, and
the whole ``artifacts/<env>`` tree. Otherwise only the selected runs go. The two derived documents are whole-document outputs, so they are
deleted (not edited) when they belong to the environment and, for a
``--model`` or ``--dataset`` deletion, mention one of the deleted runs; re-run
``make find_model`` / ``make judge`` afterwards.

Files and directories shared by several runs are removed only when every
run they hold is deleted. When no run of any environment or dataset is left
in the project, those directories are removed completely, lock files and empty registries included; the next
``make orchestrate`` creates them again. While runs remain, a registry that
has become empty is deleted instead of being left as an empty file, and
``graphs/`` is kept, because a figure cannot be attributed to the runs it
shows. Nothing happens without ``--yes``: the default is a dry run that
prints the plan.

Usage (repo root)::

    make delete_database ENV=dev PROJECT=DLSCA_ASCAD                # dry run
    make delete_database ENV=dev PROJECT=DLSCA_ASCAD CONFIRM=1      # delete
    make delete_database ENV=dev PROJECT=DLSCA_ASCAD MODEL=mesh_chicken,21f73d17 CONFIRM=1
    make delete_database ENV=dev PROJECT=DLSCA_ASCAD DATASET=ascad_desync100 CONFIRM=1
    make delete_database ENV=dev PROJECT=DLSCA_ASCAD BATCH=fig26 CONFIRM=1

Author: Shraddha Mishra
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from services.side_channel_core.utils.model_naming import _load_mapping
from services.side_channel_core.utils.seal import seal_text
from services.side_channel_core.utils.seal import load_json, read_jsonl


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    data = load_json(path) if path.read_text(encoding="utf-8").strip() else {}  # utils/seal.py
    if not isinstance(data, dict):
        # Read as {}, the registry would be rewritten without its entries.
        raise SystemExit(
            f"[delete_database] {path} is not a JSON object keyed by model_hash. Restore it "
            "from a backup or fix it by hand; it is not rewritten while it is damaged."
        )
    return data


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for row in read_jsonl(path):  # utils/seal.py: verified, seal keys dropped
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(seal_text(path, text), encoding="utf-8")  # utils/seal.py
    os.replace(tmp, path)


def _write_json(path: Path, data: dict) -> None:
    _write_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    _write_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _norm(value: Any) -> str:
    return str(value or "").strip()


class DeletionPlan:
    def __init__(
        self,
        project_root: Path,
        env: str,
        models: list[str],
        datasets: Iterable[str] = (),
        batches: Iterable[str] = (),
    ) -> None:
        self.project_root = project_root
        self.env = env
        self.selectors = {m.strip() for m in models if m.strip()}
        self.datasets = {d.strip() for d in datasets if d.strip()}
        self.batches = {b.strip() for b in batches if b.strip()}
        # Runs of the env whose dataset is not recorded; a --dataset deletion keeps them.
        self.unknown_dataset: set[str] = set()
        self.artifacts = project_root / "artifacts"
        self.model_selection = project_root / "model_selection"
        # graphs/ is the configured directory; graph/ is removed too if present.
        self.graph_dirs = [project_root / "graphs", project_root / "graph"]
        self.run_registry = _load_json(self.artifacts / "run_registry.json")
        self.test_registry = _load_json(self.artifacts / "test_registry.json")
        # A damaged map raises (with a hint); a missing one is empty.
        self.name_map = _load_mapping(project_root)
        self.run_map = _load_jsonl(self.artifacts / "run_map.jsonl")
        self.hashes, self.names = self._select()

    # ------------------------------------------------------------------ select
    def _in_env(self, record: dict) -> bool:
        return _norm(record.get("env")) == self.env

    @property
    def scoped(self) -> bool:
        """True when only some runs of the environment are selected."""
        return bool(self.selectors or self.datasets or self.batches)

    def dataset_of(self, model_name: str) -> str:
        """Dataset a run was trained on, from its config snapshot; '' if unknown."""
        config_dir = self.artifacts / self.env / "generated_models" / model_name / "config"
        for snapshot in sorted(config_dir.glob("*_params.yaml")):
            try:
                data = (yaml.safe_load(snapshot.read_text(encoding="utf-8")) or {}).get("data")
            except Exception:
                continue
            if isinstance(data, list) and data:
                data = data[0]
            if isinstance(data, dict) and _norm(data.get("name")):
                return _norm(data.get("name"))
        return ""

    def _select(self) -> tuple[set[str], set[str]]:
        """Hashes and model names of the runs to delete."""
        hashes: set[str] = set()
        names: set[str] = set()
        sources = [
            (h, i) for h, i in self.run_registry.items() if self._in_env(i)
        ] + [
            (h, i) for h, i in self.test_registry.items() if self._in_env(i)
        ] + [
            (_norm(r.get("model_hash")), r) for r in self.run_map if self._in_env(r)
        ]
        # A run's batches: its run_registry row (test and run_map rows of the same hash follow it).
        batches_of: dict[str, set[str]] = {}
        for reg in (self.run_registry, self.test_registry):
            for h, i in reg.items():
                if isinstance(i, dict):
                    batches_of.setdefault(_norm(h), set()).update(_norm(b) for b in (i.get("batches") or []) if _norm(b))
        for h, info in sources:
            h = _norm(h)
            name = _norm(info.get("model_name")) or _norm(self.name_map.get(h))
            if self.selectors and h not in self.selectors and name not in self.selectors:
                continue
            if self.batches and not (batches_of.get(h, set()) & self.batches):
                continue
            if self.datasets:
                dataset = self.dataset_of(name) if name else ""
                if not dataset:
                    self.unknown_dataset.add(name or h)
                    continue
                if dataset not in self.datasets:
                    continue
            if h:
                hashes.add(h)
            if name:
                names.add(name)
        if self.selectors and not self.datasets and not self.batches:
            # A name or hash that appears only in model_name_map.
            for h, name in self.name_map.items():
                if h in self.selectors or _norm(name) in self.selectors:
                    hashes.add(_norm(h))
                    names.add(_norm(name))
        self.unknown_dataset -= names | hashes
        return hashes, names

    # ------------------------------------------------------------------- plan
    def run_dirs(self) -> list[Path]:
        env_root = self.artifacts / self.env / "generated_models"
        if not self.scoped:
            return [self.artifacts / self.env] if (self.artifacts / self.env).exists() else []
        return [env_root / n for n in sorted(self.names) if (env_root / n).exists()]

    def derived_files(self) -> list[Path]:
        """find_model / judge outputs of this env that would be stale afterwards."""
        candidates = [self.artifacts / "filtered_models.yaml"] + sorted(
            p for p in (self.project_root / "model_selection").glob("*") if p.is_file()
        )
        stale: list[Path] = []
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            env_line = next((l for l in text.splitlines() if l.strip().strip('",').startswith(("env:", '"env"'))), "")
            if env_line and self.env not in env_line:
                continue  # another environment's output
            # Selectors are matched too, so a run already removed from the
            # registries can still be purged from these documents by name.
            mentions = any(token in text for token in self.hashes | self.names | self.selectors)
            if not self.scoped or mentions:
                stale.append(path)
        return stale

    def remaining_runs(self) -> int:
        """Runs of any environment that stay in the project after the deletion."""
        kept = {
            _norm(h) for h in list(self.run_registry) + list(self.test_registry) + list(self.name_map)
        } | {
            _norm(r.get("model_hash")) for r in self.run_map
        }
        kept = {h for h in kept if h and h not in self.hashes}
        removed = {d.resolve() for d in self.run_dirs()}
        on_disk = {
            d.name for d in self.artifacts.glob("*/generated_models/*")
            if d.is_dir() and d.resolve() not in removed and d.parent.parent.resolve() not in removed
        }
        # A run is counted once, whether it is known by its hash, its directory or both.
        names = {h: _norm(self.name_map.get(h)) for h in kept}
        for registry in (self.run_registry, self.test_registry):
            for h, info in registry.items():
                if h in kept and not names[h]:
                    names[h] = _norm(info.get("model_name"))
        for row in self.run_map:
            h = _norm(row.get("model_hash"))
            if h in kept and not names[h]:
                names[h] = _norm(row.get("model_name"))
        return len({name or h for h, name in names.items()} | on_disk)

    def wipe_dirs(self) -> list[Path]:
        """The project's output directories, when the project holds no run afterwards."""
        if self.remaining_runs():
            return []
        return [
            d for d in [self.artifacts, self.model_selection, *self.graph_dirs]
            if d.exists()
        ]

    def summary(self) -> dict:
        """The selection as data (``--json``): every selected run and the run counts per environment."""
        rows = []
        for h in sorted(self.hashes):
            info = self.run_registry.get(h) or self.test_registry.get(h) or {}
            name = _norm(info.get("model_name")) or _norm(self.name_map.get(h))
            rows.append({
                "hash": h, "model_name": name, "model_id": _norm(info.get("model_id")),
                "dataset": self.dataset_of(name) if name else "", "status": _norm(info.get("status")),
                "batches": [_norm(b) for b in (info.get("batches") or []) if _norm(b)],
            })
        per_env: dict[str, int] = {}
        for h, info in self.run_registry.items():
            if isinstance(info, dict):
                per_env[_norm(info.get("env")) or "?"] = per_env.get(_norm(info.get("env")) or "?", 0) + 1
        return {
            "project": self.project_root.name, "env": self.env,
            "scope": {"datasets": sorted(self.datasets), "models": sorted(self.selectors), "batches": sorted(self.batches)},
            "selected": rows, "total_selected": len(rows),
            "runs_per_env": per_env,
            "kept_unknown_dataset": sorted(self.unknown_dataset),
            "wipes_project": bool(self.wipe_dirs()),
        }

    def describe(self) -> list[str]:
        lines = [
            f"project      : {self.project_root}",
            f"env          : {self.env}",
            "scope        : " + (
                "; ".join(
                    part for part in (
                        "datasets " + ", ".join(sorted(self.datasets)) if self.datasets else "",
                        "models " + ", ".join(sorted(self.selectors)) if self.selectors else "",
                        "batches " + ", ".join(sorted(self.batches)) if self.batches else "",
                    ) if part
                ) or "whole environment"
            ),
            f"runs         : {len(self.hashes)} hash(es) -> {', '.join(sorted(self.names)) or '-'}",
            f"run_registry : {sum(1 for h in self.run_registry if h in self.hashes)} entr(y/ies)",
            f"test_registry: {sum(1 for h in self.test_registry if h in self.hashes)} entr(y/ies)",
            f"run_map      : {sum(1 for r in self.run_map if _norm(r.get('model_hash')) in self.hashes)} record(s)",
            f"model names  : {sum(1 for h in self.name_map if h in self.hashes)} map entr(y/ies), "
            f"{len(self.names)} done_model_names.yaml entr(y/ies)",
        ]
        if self.unknown_dataset:
            lines.append(
                "kept         : " + ", ".join(sorted(self.unknown_dataset))
                + "  (no config snapshot, dataset unknown)"
            )
        wipe = self.wipe_dirs()
        if wipe:
            lines.append("remaining    : no run of any environment or dataset; the directories are removed completely")
            for d in wipe:
                files = sum(1 for f in d.rglob("*") if f.is_file() and "tmux_logs" not in f.parts)
                lines.append(f"wipe         : {d}  ({files} file(s))")
                for logs in sorted(d.glob("*/tmux_logs")) if d == self.artifacts else []:
                    lines.append(f"kept         : {logs}  (the batches' make/tmux output, appended run after run)")
            return lines
        lines.append(f"remaining    : {self.remaining_runs()} run(s) stay in the project")
        for d in self.run_dirs():
            lines.append(f"directory    : {d}")
            if not self.scoped and (d / "tmux_logs").is_dir():
                lines.append(f"kept         : {d / 'tmux_logs'}  (the batches' make/tmux output, appended run after run)")
        for f in self.derived_files():
            lines.append(f"derived      : {f}  (regenerate with make find_model / make judge)")
        if self.stale_heartbeat():
            lines.append(f"heartbeat    : {self.stale_heartbeat()}  (names a deleted run)")
        for path, label, others in self.tmux_logs():
            stay = f"; {others} other run(s) of it stay" if others else "; no run of it stays"
            lines.append(f"kept         : {path}  (tmux log of batch {label}, the record of how it ran{stay})")
        for d in self.graph_dirs:
            if d.exists():
                lines.append(f"kept         : {d}  (figures cannot be attributed to single runs)")
        return lines

    def tmux_logs(self) -> list[tuple[Path, str, int]]:
        """The tmux logs of the deleted runs' batches: (artifacts/<env>/tmux_logs/<batch>.txt, batch, the runs of
        this env that keep the batch label). Every one is kept (the record of how the batch ran); the dry run
        names them."""
        labels: set[str] = set()
        for h in self.hashes:
            info = self.run_registry.get(h) or self.test_registry.get(h) or {}
            labels |= {_norm(b) for b in (info.get("batches") or []) if _norm(b)}
        out = []
        for label in sorted(labels):
            path = self.artifacts / self.env / "tmux_logs" / f"{label}.txt"
            if not path.is_file():
                continue
            others = sum(
                1 for h, info in self.run_registry.items()
                if isinstance(info, dict) and _norm(h) not in self.hashes and self._in_env(info)
                and label in {_norm(b) for b in (info.get("batches") or [])}
            )
            out.append((path, label, others))
        return out

    def stale_heartbeat(self) -> Path | None:
        """artifacts/<env>/heartbeat.json when it names a run to delete: left behind, it would show the
        deleted run as running in the TraceFlow Monitor."""
        path = self.artifacts / self.env / "heartbeat.json"
        try:
            beat = load_json(path)
        except (OSError, ValueError):
            return None
        if not isinstance(beat, dict):
            return None
        named = {_norm(beat.get("model_hash")), _norm(beat.get("model_name"))} - {""}
        return path if named & (self.hashes | self.names) else None

    # ------------------------------------------------------------------ apply
    def apply(self) -> None:
        wipe = self.wipe_dirs()
        if wipe:
            for d in wipe:
                if d == self.artifacts:
                    _wipe_keeping_tmux_logs(d)
                else:
                    shutil.rmtree(d)
            return

        keep = lambda h: _norm(h) not in self.hashes  # noqa: E731

        def rewrite(path: Path, entries: Any, writer: Any) -> None:
            """Write the kept entries; a registry left without entries is deleted."""
            if not path.is_file():
                return
            if entries:
                writer(path, entries)
            else:
                path.unlink()

        rewrite(self.artifacts / "run_registry.json", {h: i for h, i in self.run_registry.items() if keep(h)}, _write_json)
        rewrite(self.artifacts / "test_registry.json", {h: i for h, i in self.test_registry.items() if keep(h)}, _write_json)
        rewrite(self.artifacts / "model_name_map.json", {h: n for h, n in self.name_map.items() if keep(h)}, _write_json)
        rewrite(self.artifacts / "run_map.jsonl", [r for r in self.run_map if keep(r.get("model_hash"))], _write_jsonl)
        names_yaml = self.artifacts / "done_model_names.yaml"
        if names_yaml.is_file():
            kept = [
                line for line in names_yaml.read_text(encoding="utf-8").splitlines()
                if not (line.startswith("- ") and line[2:].strip().strip("'\"") in self.names)
            ]
            if any(line.startswith("- ") for line in kept):
                _write_atomic(names_yaml, "\n".join(kept) + "\n")
            else:
                names_yaml.unlink()
                lock = names_yaml.with_name(f".{names_yaml.name}.lock")
                if lock.exists():
                    lock.unlink()
        for f in self.derived_files():
            f.unlink()
        beat = self.stale_heartbeat()
        if beat:
            beat.unlink()
        for d in self.run_dirs():
            if self.scoped:
                shutil.rmtree(d)
            else:                               # the whole environment, its batches' tmux logs kept
                _wipe_env_keeping_tmux_logs(d)
        if self.model_selection.is_dir() and not any(self.model_selection.iterdir()):
            self.model_selection.rmdir()


def _wipe_env_keeping_tmux_logs(env_dir: Path) -> None:
    """Delete artifacts/<env>/ except its tmux_logs/ (the batches' make/tmux output); the directory goes when it has none."""
    for child in env_dir.iterdir():
        if child.name == "tmux_logs" and child.is_dir():
            continue
        shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
    if not any(env_dir.iterdir()):
        env_dir.rmdir()


def _wipe_keeping_tmux_logs(artifacts: Path) -> None:
    """Delete artifacts/ except every artifacts/<env>/tmux_logs/: the make/tmux output of the
    batches (one <BATCH>.txt each, appended run after run) is the record of how a figure was made."""
    for env_dir in artifacts.iterdir():
        if env_dir.is_dir() and (env_dir / "tmux_logs").is_dir():
            for child in env_dir.iterdir():
                if child.name == "tmux_logs":
                    continue
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
        elif env_dir.is_dir() and not env_dir.is_symlink():
            shutil.rmtree(env_dir)
        else:
            env_dir.unlink()
    if not any(artifacts.iterdir()):          # no batch log to keep: the directory goes too
        artifacts.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", required=True, help="projects/<PROJECT>")
    parser.add_argument("--env", required=True, help="dev | staging | prod")
    parser.add_argument("--model", default="", help="comma-separated model names or hashes; default: every run of the env")
    parser.add_argument("--dataset", default="", help="comma-separated dataset names (data.name); default: every dataset")
    parser.add_argument("--batch", default="", help="comma-separated batch names (run_registry batches); default: every batch")
    parser.add_argument("--yes", action="store_true", help="apply the deletion (default: dry run)")
    parser.add_argument("--json", action="store_true", help="dry run printed as one JSON object (the selected runs, run counts per env)")
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    if not (project_root / "config").is_dir():
        raise SystemExit(f"[delete_database] not a project root: {project_root}")
    env = _norm(args.env)
    if env not in {"dev", "staging", "prod"}:
        raise SystemExit(f"[delete_database] env must be dev, staging or prod, got {env!r}")

    plan = DeletionPlan(project_root, env, args.model.split(","), args.dataset.split(","), args.batch.split(","))
    if args.json:
        if args.yes:
            raise SystemExit("[delete_database] --json is a dry run; it does not take --yes (CONFIRM=1)")
        print(json.dumps(plan.summary()))
        return 0
    header = "DELETE DATABASE" if args.yes else "DELETE DATABASE (dry run; add CONFIRM=1 to apply)"
    print(header)
    print("-" * len(header))
    for line in plan.describe():
        print(line)
    if not plan.hashes and not plan.run_dirs() and not plan.derived_files() and not plan.wipe_dirs():
        if plan.scoped:
            raise SystemExit(
                f"[delete_database] nothing recorded for env={env} matches "
                f"{sorted(plan.selectors | plan.datasets | plan.batches)}"
            )
        print("nothing to delete")
        return 0
    if not args.yes:
        return 0
    try:
        plan.apply()
    except PermissionError as exc:
        raise SystemExit(
            f"[delete_database] permission denied: {exc.filename}\n"
            "   Files written by the training container belong to root. Run:\n"
            f'   sudo env "PATH=$PATH" make delete_database ENV={env} PROJECT={project_root.name} CONFIRM=1'
        ) from exc
    print("deleted")
    # The deleted runs leave every graph job of the project (models, legend.labels, top_models):
    # services/graph_manager/graph_export.py prune. The deletion stands whatever this reports.
    if plan.names:
        try:
            from services.graph_manager.graph_export import Project, prune

            pruned = prune(Project(project_root.name, env), sorted(plan.names))
            jobs = pruned.get("jobs") or []
            print(f"graph jobs   : removed from {', '.join(map(str, jobs))}" if jobs else "graph jobs   : none listed them")
        except Exception as exc:   # noqa: BLE001 - the deletion is done; say what is left to edit
            print(f"[delete_database][WARN] graph_config.yaml not updated ({exc}); remove "
                  f"{', '.join(sorted(plan.names))} from its jobs by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
