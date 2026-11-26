# services/side_channel_core/src/cli.py

"""
CLI Kickoff for Side-Channel Analysis Core
===========================================

Fully wired to your config_loader and using get_project_config
as the only source of truth for master + params. It still does
per-dataset filtering, and dynamically loads whatever scripts
you’ve registered in master.yaml. Fully powered with sweep yaml config function.

All core modules must export exactly one top-level function with signature:

    train(cfg: dict)
    infer(cfg: dict)
    monitor(cfg: dict)
    generate(cfg: dict)
    validate(cfg: dict)

Usage:

cd ~/mila/SCA/services/side_channel_core
export PYTHONPATH=$PYTHONPATH:$(pwd)


0. python -m core.cli --help
1. python -m core.cli gen-data   --project DLSCA_ASCAD --dataset ascad
2. python -m core.cli train-cmd  --project DLSCA_ASCAD --dataset ascad
3. python -m core.cli test-cmd  --project DLSCA_ASCAD --dataset ascad
4. python -m core.cli orchestrate-cmd --project DLSCA_ASCAD --dataset ascad
|| live metrics of a run (also the TraceFlow Monitor's [Running] list):
python -m core.monitor --run-dir projects/DLSCA_ASCAD/artifacts/dev/generated_models/<model_name>

Graphs: make graph GRAPH_ID=<id> (services/graph_manager).

Author: Shraddha Mishra
"""
import os
import sys
import yaml
import typer
import copy
from typing      import List, Optional
from pygments import highlight
from services.side_channel_core.utils.model_naming import choose_run_name
from pygments.lexers import YamlLexer
from pygments.formatters import TerminalFormatter
from pathlib              import Path
from importlib            import util
from utils.logger         import get_logger
from utils.error_handler  import catch_exceptions
from utils.config_loader  import get_project_config
from services.side_channel_core.src.sca.dataset_types import check_run_settings
from utils.path_utils import resolve_path
from services.side_channel_core.utils.model_naming import make_experiment_hash
from utils.sweep_engine import  load_registry
from utils.sweep_runner import run_sweep
from utils.sweep_test_runner     import run_test_sweep
from utils.cli_helpers import comma_split, extract_hash_from_name
from utils.container_guard import assert_in_container
assert_in_container()
# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


logger = get_logger("CLI")
app = typer.Typer(help="Side-Channel Analysis Core CLI")
sys.path.append(str(Path(__file__).resolve().parents[3]))
# Resolve default config path relative to this file
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = str(BASE_DIR / "config" / "graph_config.yaml")
VALID_HYPOTHESES = {"H1", "H2", "H3", "H4"}
#--------------------------------------CORE COMMANDS--------------------------------------
def _validate_hypothesis(h: Optional[str]) -> Optional[str]:
    if h is None:
        return None
    h = h.strip().upper()
    if h not in VALID_HYPOTHESES:
        raise typer.BadParameter(f"Invalid hypothesis '{h}'. Allowed: {sorted(VALID_HYPOTHESES)}")
    return h

def _inject_hypothesis_into_cfg(cfg: dict, hypothesis: Optional[str]) -> None:
    hypothesis = _validate_hypothesis(hypothesis)
    if hypothesis is None:
        return
    cfg.setdefault("auto_mlops", {})
    cfg["auto_mlops"]["hypothesis_id"] = hypothesis
    # Optional convenience:
    cfg["hypothesis_id"] = hypothesis
    
    
def _load_module(script_path: str):
    """
    Dynamically load a .py file as a module.
    Inserts the script's directory at front of sys.path so that
    `import model` or `import utils.logger` works correctly.
    """
    p = Path(script_path)
    sys.path.insert(0, str(p.parent))
    try:
        spec = util.spec_from_file_location(p.stem, p)
        mod  = util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def _run_command(
    project: str,
    dataset: str,
    script_key: str,
    entry_point: str,
    help_text: str,
):
    """
    Helper to:
        1) load full config
        2) select dataset
        3) load the requested script
        4) call the entry point
    """
    cfg = get_project_config(project_name=project)

    # select only the requested dataset
    if dataset not in cfg["data"]:
        raise typer.Exit(f"dataset '{dataset}' not defined in params.yaml")
    cfg["data"] = cfg["data"][dataset]
    check_run_settings(cfg)   # attack byte, CV and sca.variant against the entry's database type

    script_path = cfg["project"]["scripts"].get(script_key)
    if not script_path:
        raise typer.Exit(f"no script configured for '{script_key}' in master.yaml")
    module = _load_module(script_path)

    if not hasattr(module, entry_point):
        raise typer.Exit(f"{Path(script_path).name} must define a `{entry_point}(cfg)` function")
    getattr(module, entry_point)(cfg)




from utils.cli_helpers import select_data_entry as _select_data_entry  # noqa: E402  (no default entry)

@app.command("train-cmd")
@catch_exceptions
def train_cmd(
    project: str = typer.Option(...,                "--project",            "-p", help="Project to run tests for"),
    dataset: Optional[str] = typer.Option(None,     "--dataset",            "-d", help="The data entry of params.yaml to run (DATASET); -n takes each registered run's own"),
    hypothesis: Optional[str] = typer.Option(None,  "--hypothesis", "--hy", help="AutoML hypothesis to run: H1 | H2 | H3 | H4"),
    env: str = typer.Option(os.environ.get("ENV"),                   "--env", "-e", help="Environment: dev | staging | prod"),
    include_models: List[str] = typer.Option([],    "--model-name",         "-n",help="Re-run these registered models (name or hash) from their snapshot: train, then test", callback=comma_split,),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Model ID from params.yaml"),
    retry_failed: bool = typer.Option(False,        "--retry-failed",       "-r", help="Re-run tests for models marked as failed."),
    resume_interrupted: bool = typer.Option(False,  "--resume-interrupted", "-c", help="Re-run tests for models marked as in_progress/INTERRUPTED."),
    force: bool = typer.Option(False,               "--force",              "-f", help="Re-run tests for models already marked as DONE. in addition, re-run all models marked as failed, in_progress, or interrupted."),
    alias: Optional[str] = typer.Option(None,       "--alias",              "-a", help="Override the first experiment's model_name."),
    ):
    """
    Kick off a training run for the selected project + dataset.
    The real work (random model_name, dir creation, path-templating, etc.)
    lives in src/core/train.py.
    """

    # 1) load + merge master.yaml + params.yaml
    cfg = get_project_config(project_name=project)
    _inject_hypothesis_into_cfg(cfg, hypothesis)
    effective_model_id = (
        model_id
        or os.environ.get("MODEL_ID")
        or cfg.get("model_id")
    )

    if effective_model_id:
        cfg["model_id"] = str(effective_model_id)
    project_root = Path(cfg["project"]["root"])
    # if include_models:
    #     cfg["model_name"] = include_models[0]
    # 2) pick only the requested dataset. -n (registered runs, make reproduce) trains each run from its
    #    own snapshot, whose data entry is the run's: DATASET is not needed and is not assumed.
    if not include_models:
        _select_data_entry(cfg, dataset, "train-cmd")
        check_run_settings(cfg)   # attack byte, CV and sca.variant against the entry's database type
    elif dataset:
        _select_data_entry(cfg, dataset, "train-cmd")

    # 3) distributed? delegate if needed
    # dist_cfg = cfg.get("distributed", {})
    # use_dist = backend or dist_cfg.get("enabled", False)
    # if use_dist:
    #     which      = backend or "ddp"
    #     script_key = {"ddp": "ddp_script"}[which]
    #     script_path= dist_cfg.get(script_key)
    #     if not script_path:
    #         raise typer.Exit(f"no distributed script for '{which}' in master.yaml")
    #     mod = _load_module(script_path)
    #     return mod.main(cfg)

    # 4) standard train
    script = cfg["project"]["scripts"]["train"]
    mod    = _load_module(script)

    # 5) resolve all path‐templates now that model_name & dataset_alias are set
    # DO NOT mutate cfg["paths"] templates in-place.
    # Keep templates in cfg["paths"] and store resolved values separately.
    # from utils.path_utils import resolve_path
    # cfg["paths_abs"] = {}
    # for key, tmpl in (cfg.get("paths") or {}).items():
    #     cfg["paths_abs"][key] = resolve_path(tmpl, cfg)

    # Rule: do NOT use params_out existence to decide "fresh" vs "existing".
    # Source of truth for "existing" is run_registry.json.
    if include_models:
        # -n re-runs registered runs only, the orchestrator's way
        # (core.orchestrator.main, registered_only=True): params_out snapshot
        # (resolve_registered_run_cfg), retry budget (-f overrides), then
        # train_and_test (train, then the paired test). Status updates are
        # keyed by model_hash; a failure exits non-zero. -r/-c do not apply:
        # the named hashes run whatever their status.
        registry = load_registry(project_root)

        # Build quick lookup sets
        known_names = {info.get("model_name") for info in registry.values() if isinstance(info, dict)}
        known_hashes = set(registry.keys())

        selectors, unknown = [], []
        for raw_input in include_models:
            raw_input = str(raw_input).strip()
            if not raw_input:
                continue
            model_hash = extract_hash_from_name(raw_input)
            if model_hash in known_hashes:
                selectors.append(model_hash)
            elif raw_input in known_names:
                selectors.append(raw_input)
            else:
                unknown.append(raw_input)

        if unknown:
            message = (
                f"train-cmd -n: {unknown} not found in {project_root / 'artifacts' / 'run_registry.json'}. "
                "-n re-runs registered models only (name -> hash: artifacts/model_name_map.json). "
                "Start new runs with make orchestrate ENV=<env> PROJECT=<project> "
                "(ALIAS=<name> names the first one)."
            )
            logger.error(message)
            raise ValueError(message)
        if alias:
            message = (
                "train-cmd -n: --alias names a new run; a registered run keeps its model_name "
                "(identity is frozen). Drop --alias, or start a new run with make orchestrate ALIAS=<name>."
            )
            logger.error(message)
            raise ValueError(message)
        if retry_failed or resume_interrupted:
            logger.warning(
                "train-cmd -n: -r/-c have no effect here; %s re-run whatever their status.", selectors
            )

        effective_model_id = model_id or os.environ.get("MODEL_ID") or cfg.get("model_id")
        if not effective_model_id:
            # The named runs' own architecture, when they agree on one.
            registered_ids = {
                str(info.get("model_id") or "").strip()
                for h, info in registry.items()
                if isinstance(info, dict) and (h in selectors or info.get("model_name") in selectors)
            }
            if len(registered_ids) == 1:
                effective_model_id = registered_ids.pop() or None

        from core import orchestrator

        orchestrator.main(
            project=project,
            dataset=dataset,
            env=env,
            include_models=selectors,
            model_id=effective_model_id,
            force=force,
            registered_only=True,
        )
        return


    # Normal sweep mode
    run_sweep(
        cmd_fn=mod.train,
        exec_process="train",
        cfg=cfg,
        project_root=project_root,
        include_model_name=[],
        retry_failed=retry_failed,
        resume_interrupted=resume_interrupted,
        alias=alias,
        force=force,
    )


@app.command("test-cmd")
@catch_exceptions
def test_cmd(
    project: str = typer.Option(..., "--project", "-p", help="Project to run tests for"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="The data entry of params.yaml to test (DATASET); -n takes each registered run's own"),
    env: str = typer.Option(os.environ.get("ENV"), "--env", "-e", help="Environment: dev | staging | prod"),
    include_models: List[str] = typer.Option([], "--model-name", "-n", help="Only re-test these models (by name or hash)", callback=comma_split),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Model ID from params.yaml"),
    retry_failed: bool = typer.Option(False, "--retry-failed", "-r", help="Re-run tests for models marked as failed."),
    resume_interrupted: bool = typer.Option(False, "--resume-interrupted", "-c", help="Re-run tests for models marked as in_progress/INTERRUPTED."),
    force: bool = typer.Option(False, "--force", "-f", help="Re-run tests for models already marked as DONE. In addition, re-run all models marked as failed, in_progress, or interrupted."),
    test_request: bool = typer.Option(False, "--test-request", help="make test: re-test the models given by -n (or listed in config/test_config.yaml) from their own training snapshots; params.yaml is not read. The results replace the models' recorded tests."),
    test_config: Optional[str] = typer.Option(None, "--test-config", help="With --test-request: another request file, relative to the project root."),
    external: bool = typer.Option(False, "--external", help="With --test-request: attack with test_config.yaml's perform_attack settings instead of the snapshot's; the reports go to <run>/test/<test id>/ and the outcome to test_registry.json's external_tests, beside the recorded test."),
    budget: bool = typer.Option(False, "--budget", help="With --test-request: a Budget test, each attack the data entry's data[].attack_trace_budget traces (else an All test: every test trace)."),
):
    if test_request:
        from services.side_channel_core.src.sca.test_request import run_test_request

        run_test_request(
            project=project,
            models=include_models or None,
            env=env,
            load_module=_load_module,
            test_config=test_config,
            external=external,
            kind="budget" if budget else "all",
        )
        return

    cfg = get_project_config(project_name=project)
    effective_model_id = (
        model_id
        or os.environ.get("MODEL_ID")
        or cfg.get("model_id")
    )

    if effective_model_id:
        cfg["model_id"] = str(effective_model_id)
    _select_data_entry(cfg, dataset, "test-cmd")
    # -n tests registered runs from their own snapshots, as train-cmd -n trains them: their settings were
    # checked when they were planned (or imported, utils/model_import.py), and params.yaml's are not theirs.
    # Checking params.yaml here refused an imported AES_HD run for params.yaml's unshuffled validation
    # attack, which the import does not run (2026-10-27).
    if not include_models:
        check_run_settings(cfg)   # attack byte, CV and sca.variant against the entry's database type

    if model_id:
        cfg["model_id"] = model_id

    script = cfg["project"]["scripts"]["test"]
    mod = _load_module(script)

    run_test_sweep(
        cmd_fn=mod.test,
        exec_process="test",
        cfg=cfg,
        project_root=Path(cfg["project"]["root"]),
        include_models=include_models,
        force=force,
        retry_failed=retry_failed,
        resume_interrupted=resume_interrupted,
    )


@app.command("gen-data")
@catch_exceptions
def gen_data_cmd(
    project: str = typer.Option(None, help="Which project to run"),
    dataset: Optional[str] = typer.Option(None, help="The data entry of params.yaml to generate (DATASET)"),
    alias:   str = typer.Option(None, help="Optional new name for the generated dataset"),
):
    """
    Generate & label traces for the selected project + dataset.
    """
    # 1) load + merge master.yaml + params.yaml
    cfg = get_project_config(project_name=project)

    # 2) pick only the requested dataset
    _select_data_entry(cfg, dataset, "gen-data")
    check_run_settings(cfg)   # attack byte, CV and sca.variant against the entry's database type

    # 3) inject alias into cfg["data"] if provided
    if alias:
        cfg["data"]["output_name"] = alias

    # 4) pick a two-word funny name (or timestamp) + compute stable hash
    choose_run_name(cfg, alias)
    # 4.5) compute & stash the hash
    cfg["model_hash"] = make_experiment_hash(cfg)
    # 5) now render every single path‐template in cfg["paths"]
    cfg["paths_abs"] = {}
    for name, tmpl in (cfg.get("paths") or {}).items():
        cfg["paths_abs"][name] = resolve_path(tmpl, cfg)

    # 6) load & sanity‐check your data_loader entrypoint
    script_path = cfg["project"]["scripts"].get("data_loader")
    if not script_path:
        raise typer.Exit("no script configured for 'data_loader' in master.yaml")
    module = _load_module(script_path)
    if not hasattr(module, "generate"):
        raise typer.Exit("data_loader.py must define a `generate(cfg)` function")

    # 7) for printing only, convert all Paths → str so YAML.dump won’t choke
    dump_cfg = copy.deepcopy(cfg)
    for k, v in dump_cfg["paths"].items():
        dump_cfg["paths"][k] = str(v)

    cfg_yaml = yaml.safe_dump(dump_cfg, sort_keys=False, default_flow_style=False)
    colored  = highlight(cfg_yaml, YamlLexer(), TerminalFormatter())
    RED, RESET = "\x1b[31m", "\x1b[0m"

    typer.secho("\n[INFO] CLI: ▶ Resolved configuration for gen-data:", fg="green")
    typer.echo(f"{RED}{colored}{RESET}")

    # 8) finally invoke your data‐generation
    module.generate(cfg)


@app.command("orchestrate-cmd")
@catch_exceptions
def orchestrate_cmd(
    project: str = typer.Option(..., "--project", "-p"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="The data entry of params.yaml to run (DATASET)"),
    env: str = typer.Option(os.environ.get("ENV"), "--env", "-e", help="Environment: dev | staging | prod"),
    hypothesis: Optional[str] = typer.Option(None, "--hypothesis", "--hy", help="AutoML hypothesis to run: H1 | H2 | H3 | H4"),
    include_models: List[str] = typer.Option([], "--model-name", "-n", help="Run train→test for these models", callback=comma_split),
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Model ID from params.yaml"),
    retry_failed: bool = typer.Option(False, "--retry-failed", "-r"),
    resume_interrupted: bool = typer.Option(False, "--resume-interrupted", "-c"),
    force: bool = typer.Option(False, "--force", "-f"),
    alias: Optional[str] = typer.Option(None, "--alias", "-a"),
):
    """
    Orchestrate train → test in sweep style.
    """
    from core import orchestrator

    get_project_config(project_name=project)

    orchestrator.main(
        project=project,
        dataset=dataset,
        env=env,
        hypothesis=hypothesis,
        include_models=include_models,
        model_id=model_id,
        retry_failed=retry_failed,
        resume_interrupted=resume_interrupted,
        force=force,
        alias=alias,
    )


if __name__ == "__main__":
    app()
