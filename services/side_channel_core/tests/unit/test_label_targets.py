# services/side_channel_core/tests/unit/test_label_targets.py
"""Label targets (src/sca/leakage_models.py): sbox unchanged, sbox_masked (DPAv4), inv_sbox_hd (AES_HD)."""

import numpy as np
import pytest

from sca import leakage_models as lm
from utils.aes import AES_INV_SBOX, AES_SBOX

rng = np.random.default_rng(1)
N = 64


def _rows(**extra):
    rows = {"plaintext": rng.integers(0, 256, (N, 16), dtype=np.uint8),
            "key": np.tile(rng.integers(0, 256, 16, dtype=np.uint8), (N, 1))}
    rows.update(extra)
    return rows


def test_the_inverse_sbox_undoes_the_sbox():
    assert np.array_equal(AES_INV_SBOX[AES_SBOX], np.arange(256))


@pytest.mark.parametrize("model", ["ID", "HW", "MSB"])
def test_the_sbox_target_is_the_existing_table_path_exactly(model):
    rows = _rows()
    p, k = rows["plaintext"][:, 2], rows["key"][:, 2]
    assert np.array_equal(lm.target_hypotheses(rows, target="sbox", byte=2, leakage_model=model),
                          lm.build_label_table(model)[p])
    assert np.array_equal(lm.target_labels(rows, target="sbox", byte=2, leakage_model=model),
                          lm.compute_leakage_label(p, k, model))


@pytest.mark.parametrize("model", ["ID", "HW"])
def test_the_masked_target_xors_each_traces_mask(model):
    rows = _rows(mask=rng.integers(0, 256, N, dtype=np.uint8))
    p, k, m = rows["plaintext"][:, 0], rows["key"][:, 0], rows["mask"]
    labels = lm.target_labels(rows, target="sbox_masked", byte=0, leakage_model=model)
    assert np.array_equal(labels, lm.apply_leakage_model(AES_SBOX[p ^ k] ^ m, model))
    hypotheses = lm.target_hypotheses(rows, target="sbox_masked", byte=0, leakage_model=model)
    assert np.array_equal(hypotheses[np.arange(N), k], labels)        # the true key's column is the label


def test_the_aes_hd_target_is_zaids_last_round_formula():
    ciphertext = np.zeros((N, 16), np.uint8)
    ciphertext[:, 11] = rng.integers(0, 256, N)
    ciphertext[:, 7] = rng.integers(0, 256, N)
    rows = {"ciphertext": ciphertext, "key": np.zeros((N, 16), np.uint8)}   # key byte 0x00 by convention
    labels = lm.target_labels(rows, target="inv_sbox_hd", byte=11, leakage_model="ID")
    assert np.array_equal(labels, AES_INV_SBOX[ciphertext[:, 11]] ^ ciphertext[:, 7])
    hypotheses = lm.target_hypotheses(rows, target="inv_sbox_hd", byte=11, leakage_model="ID")
    assert np.array_equal(hypotheses[:, 0], labels)
    assert np.array_equal(hypotheses[:, 5], AES_INV_SBOX[ciphertext[:, 11] ^ 5] ^ ciphertext[:, 7])
    with pytest.raises(ValueError, match="defined for attack byte"):
        lm.target_labels(rows, target="inv_sbox_hd", byte=3, leakage_model="ID")


def test_a_structured_metadata_array_works_like_a_mapping():
    dtype = np.dtype([("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,)), ("mask", np.uint8)])
    rows = np.zeros(N, dtype)
    rows["plaintext"] = rng.integers(0, 256, (N, 16))
    rows["mask"] = rng.integers(0, 256, N)
    as_dict = {name: rows[name] for name in dtype.names}
    assert np.array_equal(lm.target_labels(rows, target="sbox_masked", byte=0, leakage_model="ID"),
                          lm.target_labels(as_dict, target="sbox_masked", byte=0, leakage_model="ID"))


def _meta_rows(rows):
    """The attack path's metadata: one mapping per trace (sca/ascad_data_loader._meta_void_to_dict)."""
    return [{name: values[i] for name, values in rows.items()} for i in range(N)]


@pytest.mark.parametrize("target,byte", [("sbox_masked", 0), ("inv_sbox_hd", 11)])
def test_the_attack_ranks_key_guesses_with_the_types_target(target, byte):
    import torch
    from sca.rank_curve import fixed_key_attack_evidence

    rows = _rows(mask=rng.integers(0, 256, N, dtype=np.uint8), ciphertext=rng.integers(0, 256, (N, 16), dtype=np.uint8))
    if target == "inv_sbox_hd":
        rows["key"] = np.zeros((N, 16), np.uint8)
    labels = lm.target_labels(rows, target=target, byte=byte, leakage_model="ID")
    preds = torch.full((N, 256), 0.1 / 255, dtype=torch.float64)
    preds[torch.arange(N), torch.as_tensor(labels, dtype=torch.long)] = 0.9         # a model that learnt the target
    run = dict(preds=preds, metadata=_meta_rows(rows), target_byte=byte, step=8, tie_break_seed=0)
    assert fixed_key_attack_evidence(**run, target=target)["final_true_key_rank"] == 1
    assert fixed_key_attack_evidence(**run)["final_true_key_rank"] > 1               # read as the S-box output it is not


def test_the_label_gate_checks_the_types_target_and_refuses_another(tmp_path):
    import h5py
    from utils.label_checks import check_label_integrity

    rows = _rows(mask=rng.integers(0, 256, N, dtype=np.uint8))
    dtype = np.dtype([("plaintext", np.uint8, (16,)), ("key", np.uint8, (16,)), ("mask", np.uint8)])
    meta = np.zeros(N, dtype)
    for name in dtype.names:
        meta[name] = rows[name]
    path = tmp_path / "split.h5"
    with h5py.File(path, "w") as f:
        f.attrs["sca_label_target"] = "sbox_masked"
        f["Attack_traces/metadata"] = meta
        f["Attack_traces/labels"] = lm.target_labels(rows, target="sbox_masked", byte=0, leakage_model="ID")
    check_label_integrity(str(path), attack_byte=0, leakage_model="ID", target="sbox_masked", verbose=False)
    with pytest.raises(ValueError, match="labelled for target 'sbox_masked'"):
        check_label_integrity(str(path), attack_byte=0, leakage_model="ID", verbose=False)   # an S-box run
    with h5py.File(path, "r+") as f:
        f["Attack_traces/labels"][0] ^= 1
    with pytest.raises(ValueError, match="Label integrity FAILED"):
        check_label_integrity(str(path), attack_byte=0, leakage_model="ID", target="sbox_masked", n=N, verbose=False)
