"""delete_database: only the selected runs go; shared files go with their last run."""

import json

import pytest
import yaml

from services.side_channel_core.utils.delete_database import DeletionPlan, main
from utils.seal import strip as strip_seal  # rows and registries carry their "_tfseal" (utils/seal.py)


RUNS = [  # hash, name, env, dataset
    ("h1", "alpha", "dev", "ascad"),
    ("h2", "beta", "dev", "ascad_desync100"),
    ("h3", "gamma", "staging", "ascad"),
]


def make_project(root, runs=RUNS):
    (root / "config").mkdir(parents=True)
    artifacts = root / "artifacts"
    artifacts.mkdir(), (root / "graphs" / "jobs").mkdir(parents=True)
    (root / "model_selection").mkdir()
    (root / "graphs" / "jobs" / "figure.png").write_text("png")
    registry = {h: {"env": env, "model_name": name, "model_hash": h} for h, name, env, _ in runs}
    for file in ("run_registry.json", "test_registry.json"):
        (artifacts / file).write_text(json.dumps(registry))
    (artifacts / "model_name_map.json").write_text(json.dumps({h: name for h, name, _, _ in runs}))
    rows = "".join(json.dumps({"env": env, "model_name": name, "model_hash": h}) + "\n" for h, name, env, _ in runs)
    (artifacts / "run_map.jsonl").write_text(rows)
    (artifacts / "done_model_names.yaml").write_text("# names\n" + "".join(f"- {name}\n" for _, name, _, _ in runs))
    (artifacts / ".done_model_names.yaml.lock").write_text("")
    for _, name, env, dataset in runs:
        config = artifacts / env / "generated_models" / name / "config"
        config.mkdir(parents=True)
        (config / f"{name}_params.yaml").write_text(yaml.safe_dump({"data": [{"name": dataset}]}))
    return root


def hashes(root, file="run_registry.json"):
    return sorted(strip_seal(json.loads((root / "artifacts" / file).read_text())))


def test_dataset_deletion_removes_only_its_runs(tmp_path):
    root = make_project(tmp_path / "P")
    assert main(["--project-root", str(root), "--env", "dev", "--dataset", "ascad", "--yes"]) == 0

    assert hashes(root) == ["h2", "h3"]
    assert not (root / "artifacts/dev/generated_models/alpha").exists()
    assert (root / "artifacts/dev/generated_models/beta").is_dir()
    # gamma is trained on ascad too, but in another environment.
    assert (root / "artifacts/staging/generated_models/gamma").is_dir()
    assert (root / "graphs/jobs/figure.png").is_file()
    rows = [strip_seal(json.loads(line)) for line in (root / "artifacts/run_map.jsonl").read_text().splitlines()]
    assert sorted(r["model_hash"] for r in rows) == ["h2", "h3"]
    assert "- alpha" not in (root / "artifacts/done_model_names.yaml").read_text()


def test_environment_deletion_keeps_other_environments(tmp_path):
    root = make_project(tmp_path / "P")
    assert main(["--project-root", str(root), "--env", "dev", "--yes"]) == 0

    assert hashes(root) == ["h3"]
    assert not (root / "artifacts/dev").exists()
    assert (root / "artifacts/staging/generated_models/gamma").is_dir()
    assert (root / "artifacts").is_dir() and (root / "graphs").is_dir()


def test_directories_go_with_the_last_run_only(tmp_path):
    root = make_project(tmp_path / "P", runs=RUNS[:2])
    main(["--project-root", str(root), "--env", "dev", "--dataset", "ascad", "--yes"])
    assert (root / "artifacts").is_dir() and (root / "graphs").is_dir()

    main(["--project-root", str(root), "--env", "dev", "--dataset", "ascad_desync100", "--yes"])
    assert sorted(p.name for p in root.iterdir()) == ["config"]


def test_model_and_dataset_combine_and_unknown_dataset_is_kept(tmp_path):
    root = make_project(tmp_path / "P")
    plan = DeletionPlan(root, "dev", ["beta"], ["ascad"])
    assert plan.hashes == set()  # beta is not trained on ascad

    (root / "artifacts/dev/generated_models/alpha/config/alpha_params.yaml").unlink()
    plan = DeletionPlan(root, "dev", [], ["ascad"])
    assert plan.hashes == set() and plan.unknown_dataset == {"alpha"}

    with pytest.raises(SystemExit):
        main(["--project-root", str(root), "--env", "dev", "--dataset", "nowhere", "--yes"])
    assert hashes(root) == ["h1", "h2", "h3"]


def test_dry_run_deletes_nothing(tmp_path):
    root = make_project(tmp_path / "P")
    assert main(["--project-root", str(root), "--env", "dev"]) == 0
    assert hashes(root) == ["h1", "h2", "h3"]
    assert (root / "artifacts/dev/generated_models/alpha").is_dir()


def test_batch_deletion_removes_only_the_runs_of_that_batch(tmp_path):
    root = make_project(tmp_path / "P")
    registry = strip_seal(json.loads((root / "artifacts/run_registry.json").read_text()))
    registry["h1"]["batches"] = ["fig26"]
    registry["h2"]["batches"] = ["fig6"]
    (root / "artifacts/run_registry.json").write_text(json.dumps(registry))

    assert main(["--project-root", str(root), "--env", "dev", "--batch", "fig26", "--yes"]) == 0
    assert hashes(root) == ["h2", "h3"]
    assert hashes(root, "test_registry.json") == ["h2", "h3"]      # the test row follows its run
    assert not (root / "artifacts/dev/generated_models/alpha").exists()
    assert (root / "artifacts/dev/generated_models/beta").is_dir()

    # combined with --dataset, a run must match both
    plan = DeletionPlan(root, "dev", [], ["ascad"], ["fig6"])
    assert plan.hashes == set()
    with pytest.raises(SystemExit, match="nothing recorded"):
        main(["--project-root", str(root), "--env", "dev", "--batch", "nope"])


def test_json_dry_run_lists_the_selection_and_deletes_nothing(tmp_path, capsys):
    root = make_project(tmp_path / "P")
    registry = strip_seal(json.loads((root / "artifacts/run_registry.json").read_text()))
    registry["h1"].update(batches=["fig26"], model_id="ASCAD_MLPbest", status="done")
    (root / "artifacts/run_registry.json").write_text(json.dumps(registry))

    assert main(["--project-root", str(root), "--env", "dev", "--batch", "fig26", "--json"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["total_selected"] == 1 and out["env"] == "dev"
    assert out["selected"][0] == {"hash": "h1", "model_name": "alpha", "model_id": "ASCAD_MLPbest",
                                  "dataset": "ascad", "status": "done", "batches": ["fig26"]}
    assert out["runs_per_env"] == {"dev": 2, "staging": 1}
    assert hashes(root) == ["h1", "h2", "h3"]                      # a dry run
    with pytest.raises(SystemExit, match="does not take --yes"):
        main(["--project-root", str(root), "--env", "dev", "--json", "--yes"])


def test_heartbeat_naming_a_deleted_run_goes_with_it(tmp_path):
    root = make_project(tmp_path / "P")
    beat = root / "artifacts/dev/heartbeat.json"
    beat.write_text(json.dumps({"status": "running", "model_hash": "h1", "model_name": "alpha"}))
    main(["--project-root", str(root), "--env", "dev", "--model", "beta", "--yes"])
    assert beat.is_file()                     # names a kept run: the monitor still follows it
    main(["--project-root", str(root), "--env", "dev", "--model", "alpha", "--yes"])
    assert not beat.exists()                  # left behind, the monitor would show alpha running
