# services/side_channel_core/tests/unit/test_checkpoint_dir.py
"""paths.checkpoints exists only for a run that writes checkpoints (evaluation.validation.checkpointing)."""

import pytest

from utils.checkpoint_policy import clear_checkpoint_artifacts


def _cfg(tmp_path, checkpointing):
    run = tmp_path / "artifacts/dev/generated_models/m"
    return {
        "env": "dev", "model_name": "m",
        "evaluation": {"validation": {"checkpointing": checkpointing}, "test": {"load_model": "final_epoch"}},
        "paths": {"checkpoints": str(run / "checkpoints"), "model_out": str(run / "model")},
        "project": {"root": str(tmp_path)},
    }, run


@pytest.mark.parametrize("checkpointing", [False, True])
def test_the_checkpoint_folder_only_with_checkpointing(tmp_path, checkpointing):
    cfg, run = _cfg(tmp_path, checkpointing)
    clear_checkpoint_artifacts(cfg, model_name="m")
    assert (run / "model").is_dir()
    assert (run / "checkpoints").is_dir() is checkpointing
