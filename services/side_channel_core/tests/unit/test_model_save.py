# services/side_channel_core/tests/unit/test_model_save.py
"""TODO §20: a model save that fails, or writes a file that does not read back, publishes nothing."""

import pytest
import torch

import utils.epoch_utils as eu
from utils.save_utils import save_model


def test_a_saved_checkpoint_reads_back_as_the_models_weights(tmp_path):
    model = torch.nn.Linear(4, 3)
    path = eu._save_model_atomically(model=model, path=tmp_path / "model" / "m_trained.pt")
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert list(loaded) == list(model.state_dict()) and torch.equal(loaded["weight"], model.weight.detach())


def test_a_truncated_save_raises_and_publishes_nothing(tmp_path, monkeypatch):
    def truncated(*, model, path, verbose):
        torch.save(model.state_dict(), path)
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])                 # a disk that filled up half way

    monkeypatch.setattr(eu, "save_model", truncated)
    destination = tmp_path / "model" / "m_trained.pt"
    with pytest.raises(Exception):
        eu._save_model_atomically(model=torch.nn.Linear(4, 3), path=destination)
    assert not destination.exists() and not list(destination.parent.glob(".*.tmp"))


def test_save_model_raises_instead_of_warning(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    with pytest.raises(Exception):
        save_model(torch.nn.Linear(2, 2), blocker / "m.pt", verbose=False)
