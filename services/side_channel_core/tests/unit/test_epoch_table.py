# services/side_channel_core/tests/unit/test_epoch_table.py
"""utils/logger.format_epoch_table_lines: each epoch's block in the run's logs, a table: a header row
(metrics | train | validation) and a rule, the scalar metrics first, the attack metrics after them with their whole
summary in their cell, time and ETA as rows at the bottom, then the arrows' key."""

import re

import numpy as np

from utils.logger import EPOCH_ARROWS_KEY, format_epoch_table_lines

UP, DOWN = "\033[32m▲\033[0m", "\033[31m▼\033[0m"          # green better, red worse

CFG = {"metrics_to_log": ["mean_attack_rank_curve", "per_attack_curves", "final_ranks", "traces_to_ge_1",
                          "mean_fit_time_{s}", "η_t", "mean_accuracy", "training_objective",
                          "validation_objective"],
       "trainer": {"loss": {"type": "CrossEntropyLoss"}}}
X = np.arange(1, 2501, dtype=float)


def _plain(line):
    return re.sub(r"\033\[[0-9;]*m", "", line)


def _cells(line):
    """A row's cells, padding stripped: [metric, train, validation, ...]."""
    return [part.strip() for part in line.split(" | ")]


def _validation(ge, accuracy, objective):
    curves = np.stack([np.column_stack([X, np.linspace(255, ge, 2500)])] * 100)          # [attacks, traces, 2]
    return {"mean_attack_rank_curve": np.column_stack([X, np.linspace(115, ge, 2500)]),
            "per_attack_curves": curves, "final_ranks": [ge] * 100,
            "traces_to_ge_1": None, "mean_accuracy": accuracy, "validation_objective": objective,
            "loss_type": "CrossEntropyLoss", "mean_fit_time_s": 0.5}


def _train(accuracy, objective, seconds):
    return {"mean_accuracy": accuracy, "training_objective": objective, "η_t": 0.000123,
            "mean_fit_time_s": seconds, "loss_type": "CrossEntropyLoss", "train_loss_reduction": "mean"}


def _block():
    return format_epoch_table_lines(
        columns=[("train", "train", _train(0.0062, 5.53, 3.6508), _train(0.0063, 5.5301, 3.4888)),
                 ("validation", "validation", _validation(107.0, 0.0048, 5.5581), _validation(102.0, 0.0048, 5.5582))],
        elapsed=1.8, eta=0.0, epoch=20, epochs=20, cfg=CFG)


def _rows(lines):
    """metric name -> its cells after the name, from the table rows (after the rule)."""
    start = next(i for i, line in enumerate(lines) if set(line.strip()) == {"-"}) + 1
    out = {}
    for line in lines[start:]:
        if " | " in line:
            name, _sep, rest = _plain(line).removesuffix("|").rstrip().partition(" | ")   # a train-only row ends in |
            out[name.strip()] = [part.strip() for part in rest.split(" | ", 1)] if rest.strip() else []
    return out


def test_a_header_row_a_rule_scalars_first_attack_metrics_after_time_and_eta_last_then_the_key():
    lines = _block()
    assert lines[0] == "[Epoch 20/20] Results:"
    assert _cells(lines[1]) == ["metrics", "train", "validation"]
    assert set(lines[2].strip()) == {"-"} and len(lines[2]) == len(lines[1])
    assert list(_rows(lines)) == ["mean_fit_time_s", "η_t", "mean_accuracy", "CrossEntropyLoss",   # scalars
                                  "mean_attack_rank_curve", "per_attack_curves", "final_ranks", "traces_to_ge_1",
                                  "time", "ETA"]
    assert lines[-1] == f"    {EPOCH_ARROWS_KEY}"
    assert EPOCH_ARROWS_KEY == f"{UP} better, {DOWN} worse than the epoch before"


def test_the_separators_stand_in_one_column_and_numbers_are_right_aligned():
    lines = [_plain(line) for line in _block()]
    table = [line for line in lines[1:] if " | " in line]
    assert len({line.index(" | ") for line in table}) == 1        # every row's first | in one column
    numbers = [line for line in table if line.split(" | ")[0].strip() in ("mean_accuracy", "CrossEntropyLoss", "time")]
    assert len({len(line.split(" | ")[1].rstrip(" ▲▼|")) for line in numbers}) == 1  # train numbers end together
    second = {line.index(" | ", line.index(" | ") + 1) for line in table if line.count(" | ") >= 2 and
              line.split(" | ")[0].strip() in ("metrics", "mean_accuracy", "CrossEntropyLoss", "final_ranks")}
    assert len(second) == 1                                       # the train | validation separator in one column


def test_each_row_holds_its_values_arrows_and_whole_summary():
    lines = _block()
    rows = _rows(lines)
    assert rows["mean_accuracy"] == ["0.0062 ▼", "0.0048"]
    assert rows["CrossEntropyLoss"] == ["5.53 ▲", "5.5581 ▲"]                       # one objective row
    assert rows["η_t"] == ["0.000123"]                                  # not 0.0001
    assert rows["mean_fit_time_s"] == ["3.6508"]                                  # the training's; no arrow, it grows
    assert rows["mean_attack_rank_curve"] == [
        "", "curve(2500, 2) | GE@start=115.0000, GE@mid=110.9984, GE@final=107.0000 ▼"]
    # the ranks only: min 107, max 255 (no trace count of 2,500 among them)
    assert rows["per_attack_curves"] == [
        "", "array(100, 2500, 2) | min=107.0000, mean=181.0000, max=255.0000, n=100 attacks"]
    assert rows["final_ranks"] == ["", "n=100, min=107.0000, mean=107.0000, max=107.0000, final true-key rank = 107 ▼"]
    assert rows["traces_to_ge_1"] == ["", "not reached by 2,500 traces"]
    assert rows["time"] == ["1.8s"] and rows["ETA"] == ["0.0m"]
    assert any(DOWN in line and "final true-key rank = 107" in line for line in lines)     # coloured in the line
    assert not any("loss_type" in line or "train_loss_reduction" in line for line in lines)


def test_the_test_block_has_one_column_and_no_key():
    lines = format_epoch_table_lines(columns=[("test", "test", _validation(36.0, 0.004, 5.6), None)],
                                     elapsed=None, eta=None, epoch=1, epochs=1, cfg=CFG)
    assert _cells(lines[1]) == ["metrics", "test"]
    rows = _rows(lines)
    assert rows["final_ranks"] == ["n=100, min=36.0000, mean=36.0000, max=36.0000, final true-key rank = 36"]
    assert rows["time"] == ["NA"]
    assert EPOCH_ARROWS_KEY not in lines[-1]                     # no arrows: no key


def test_the_log_files_get_the_arrows_plain_the_terminal_keeps_their_colour(tmp_path):
    """A log file is plain text: its handler drops the colour codes (an editor showed them as [31m▼[0m)."""
    import logging

    from utils.logger import RotatingFileHandler

    handler = RotatingFileHandler(tmp_path / "Training.log")
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("Training", logging.INFO, __file__, 1, f"mean_accuracy | 0.0062 {DOWN} | {EPOCH_ARROWS_KEY}",
                               None, None)
    handler.emit(record)
    handler.flush()
    written = (tmp_path / "Training.log").read_text(encoding="utf-8").splitlines()[0]
    handler.close()
    assert written == "mean_accuracy | 0.0062 ▼ | ▲ better, ▼ worse than the epoch before"
    assert "\033" not in written


def test_a_numeric_attack_metric_stands_with_the_attack_metrics_settings_get_no_arrow_ntge_is_a_mean():
    """The groups come from VALID_METRIC_PROCESSES (an attack metric has no train row), not from whether a value
    is a number: guessing_entropy is one number and still an attack metric. η_t's rise is ▲ green, its fall ▼ red, no
    change no sign. N̄tGE is a mean of the attacks' crossings, so it reads as traces over attacks, not as one trace."""
    cfg = {**CFG, "metrics_to_log": ["guessing_entropy", "per_attack_traces_to_ge_1", "η_t",
                                     "mean_accuracy"]}

    def validation(ge, ntge):
        return {"guessing_entropy": ge, "per_attack_traces_to_ge_1": ntge, "per_attack_traces_to_ge_1_reached": 30,
                "per_attack_traces_to_ge_1_unreached": 7, "mean_accuracy": 0.004}

    def train(lr):
        return {"η_t": lr, "mean_accuracy": 0.006}

    lines = format_epoch_table_lines(columns=[("train", "train", train(0.0082), train(0.0071)),
                                              ("validation", "validation", validation(3.5, 5.64), validation(4.0, 6.1))],
                                     elapsed=1.0, eta=1.0, epoch=2, epochs=3, cfg=cfg)
    rows = _rows(lines)
    assert list(rows) == ["η_t", "mean_accuracy", "guessing_entropy", "per_attack_traces_to_ge_1",
                          "time", "ETA"]
    assert rows["η_t"] == ["0.0082 ▲"]                                  # it rose: ▲, green as every ▲
    assert lines[-1] == f"    {EPOCH_ARROWS_KEY}"
    fell = format_epoch_table_lines(columns=[("train", "train", train(0.0071), train(0.0082))],
                                    elapsed=1.0, eta=1.0, epoch=3, epochs=3, cfg=cfg)
    assert _rows(fell)["η_t"] == ["0.0071 ▼"]                            # it fell: ▼, red
    same = format_epoch_table_lines(columns=[("train", "train", train(0.0071), train(0.0071))],
                                    elapsed=1.0, eta=1.0, epoch=3, epochs=3, cfg=cfg)
    assert _rows(same)["η_t"] == ["0.0071"]                              # no change: no sign
    assert rows["guessing_entropy"] == ["", "3.5 ▲"]
    assert rows["per_attack_traces_to_ge_1"] == ["", "5.64 traces (mean of the 30 of 37 attacks that reached it) ▲"]
    every = {**validation(3.5, 5.64), "per_attack_traces_to_ge_1_unreached": 0}
    single = format_epoch_table_lines(columns=[("test", "test", every, None)], elapsed=None, eta=None,
                                      epoch=1, epochs=1, cfg=cfg)
    assert _rows(single)["per_attack_traces_to_ge_1"] == ["5.64 traces (mean of 30 attacks)"]
