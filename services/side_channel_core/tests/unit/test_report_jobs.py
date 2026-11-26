# services/side_channel_core/tests/unit/test_report_jobs.py
"""
The per-configuration graph jobs of services/graph_manager/report_jobs.py (TODO.md §29): each passes the
graph_config.yaml schema and the job validator, and a W configuration pools its trainings.

Run from the repository root:
    PYTHONPATH="$PWD:$PWD/services/side_channel_core:$PWD/services/side_channel_core/src" \
        conda run -n sca python -m pytest services/side_channel_core/tests/unit/test_report_jobs.py -q
"""
from services.graph_manager import report_jobs
from services.graph_manager.graph_job_validator import validate_graph_job
from services.side_channel_core.utils import config_schema


def _configuration(runs):
    return {"dataset": "ascad_n100", "letter": "h", "data": "ascad_n100_hstd_w", "model": "NoConv1_ASCAD_N100",
            "queue": "w_fig2_ascad_n100_noconv1", "who": "W", "network": "NoConv1", "preprocessing": "hstd", "runs": runs}


def test_jobs_pass_the_schema_and_the_validator():
    for runs in (["run_a"], [f"run_{i}" for i in range(10)]):
        jobs = report_jobs.jobs_for(_configuration(runs), 1)
        config_schema.check_keys({"jobs": jobs}, config_schema.GRAPH, source="graph_config.yaml")
        for job in jobs:
            validate_graph_job(job)
        assert [j["id"] for j in jobs] == [1, 2, 3]
        assert [j["name"] for j in jobs] == ["p2_VIh_loss", "p2_VIh_ge", "p2_VIh_confidence"]


def test_a_w_configuration_pools_its_trainings_and_draws_the_overlays():
    runs = [f"run_{i}" for i in range(10)]
    loss, ge, confidence = report_jobs.jobs_for(_configuration(runs), 7)
    assert confidence["groups"] == [{"label": "VI.h W · NoConv1 · hstd", "models": runs, "band": "std"}]
    assert ge["groups"] == [{**confidence["groups"][0], "members": False, "color": "#4A148C", "band_color": "#7E57C2"}]
    assert ge["overlays"]["colors"] == {"geometric": "#00C853", "median": "#FF00FF", "ntge": "#222222"}   # Fig. 8's
    assert {k: v for k, v in ge["overlays"].items() if k != "colors"} == {
        "ge_averages": ["geometric", "median"], "ntge": True, "trace_lines": ["n_val", "budget"]}
    assert confidence["plot_type"] == "strip" and confidence["source"] == "key_recovery"
    assert loss["metric_process"] == {"training_objective": "train"}
    assert loss["groups"] == [{**confidence["groups"][0], "members": False}]   # the loss: mean and band, no curve each
    assert loss["style"]["std_band"] == {"alpha": .22} and ge["style"]["std_band"] == {"alpha": .32}   # Fig. 8's bands
    assert [e["label"] for e in loss["legend"]["entries"]] == ["training, mean of 10 ± 1 SD", "validation, mean of 10 ± 1 SD"]
    assert all(j.get("groups") is None for j in report_jobs.jobs_for(_configuration(["run_a"]), 1))   # one training


def test_code_readings_are_not_p2_results():
    assert report_jobs.code_reading("z_fig7_aes_hd_code") and report_jobs.code_reading("w_fig2_ascad_n100_linear")
    assert not report_jobs.code_reading("w_fig2_ascad_n100") and not report_jobs.code_reading("s3_ascad_n100_wouters")


def test_every_job_draws_into_the_projects_graphs_even_of_one_training(tmp_path):
    """folder: project (2026-10-30): a configuration of one training (Zaid's) draws into graphs/jobs/, which git
    keeps, so the paper's link to each of its graphs opens that graph alone, not its facet."""
    import pytest

    from services.graph_manager import job_paths
    from services.graph_manager.graph_job_validator import GraphJobValidationError

    where = dict(project_root=tmp_path, graphs_root=tmp_path / "graphs", env="dev")
    for job in report_jobs.jobs_for(_configuration(["run_a"]), 1):
        assert validate_graph_job(job)["folder"] == "project"
        assert job_paths.job_dir(validate_graph_job(job), **where).parent == tmp_path / "graphs" / "jobs"
    one = {**report_jobs.jobs_for(_configuration(["run_a"]), 1)[0]}
    one.pop("folder")
    assert "folder" not in validate_graph_job(one)                                     # the default: its run's folder
    assert job_paths.job_dir(one, **where).parts[-4:-1] == ("generated_models", "run_a", "graphs")
    with pytest.raises(GraphJobValidationError, match="one of"):
        validate_graph_job({**one, "folder": "elsewhere"})
