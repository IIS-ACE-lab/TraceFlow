# services/side_channel_core/tests/unit/test_test_cmd_named_runs.py
"""test-cmd -n tests registered runs from their own snapshots, as train-cmd -n trains them: params.yaml's
run settings are not checked against them (2026-10-27: an imported AES_HD run was refused for params.yaml's
unshuffled validation attack, which the import does not run). Without -n, params.yaml's settings are the
run's and are checked."""

import pytest


@pytest.mark.parametrize("named, checked", [(["e085dfb4"], False), ([], True)])
def test_params_yaml_is_checked_only_for_the_runs_it_plans(monkeypatch, named, checked):
    monkeypatch.setenv("ALLOW_HOST_RUN", "1")             # utils/container_guard.py checks on import: the test runs on the host
    import core.cli as cli

    calls = []
    cfg = {"project": {"root": "/tmp/P", "scripts": {"test": "src/sca/test.py"}}}
    monkeypatch.setattr(cli, "get_project_config", lambda project_name: cfg)
    monkeypatch.setattr(cli, "_select_data_entry", lambda *a, **k: None)
    monkeypatch.setattr(cli, "check_run_settings", lambda c: calls.append("checked"))
    monkeypatch.setattr(cli, "_load_module", lambda script: type("M", (), {"test": staticmethod(lambda *a, **k: None)}))
    monkeypatch.setattr(cli, "run_test_sweep", lambda **kwargs: calls.append(("sweep", tuple(kwargs["include_models"]))))
    cli.test_cmd.__wrapped__(project="P", dataset=None, env="dev", include_models=named, model_id="imported",
                             retry_failed=False, resume_interrupted=False, force=False, test_request=False,
                             test_config=None, external=False)
    assert ("checked" in calls) is checked
    assert ("sweep", tuple(named)) in calls
