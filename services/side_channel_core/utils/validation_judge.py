# services/side_channel_core/utils/validation_judge.py
"""
The validation judge: one line per epoch that reads the epoch's train and validation accuracy against chance.

evaluation.validation.judge_log: true logs it after every epoch that has a validation pass (a run without
cross-validation, every fold; the refit has no held-out traces), for example

    Validation judge, epoch 2/75: memorising its training traces: train accuracy 0.1744 above chance 0.0039,
    climbing; validation 0.0034 at chance 0.0039 (17 of 5,000, p = 0.81)

false, empty or absent: no line. A log only: it reads what the epoch already computed, writes nothing, and stays
out of the run hash (utils/model_naming._hashed_evaluation).

Chance, from the run's own traces, for any database type, leakage model and split
- chance is the accuracy the epoch's predictions would have if they ignored the traces: the predictions shuffled
  against the labels, sum over classes of (share of predictions in c) x (share of labels in c). It follows the
  labels the split actually holds (a fixed key over few traces is not uniform; HW labels are 70/256 class 4) and
  the classes the model predicts (always predicting HW 4 is chance, not learning), so no dataset or ratio is
  assumed: train and validation each against their own traces.
- above chance: an exact one-sided binomial test of the correct count on the accuracy's own n traces
  (scipy.stats.binom.sf), p < ALPHA (three standard errors of a normal); a validation set of 50 traces is
  judged as exactly as one of 5,000.
      generalising                   validation above chance
      memorising its training traces train above chance, validation at chance
      not learning yet               both at chance
- train accuracy is the running accuracy of the epoch's training passes (sca/metrics.py), so it climbs when it
  is higher than the epoch before's.
- overfitting: the validation objective rose over the last OVERFIT_EPOCHS epochs while the training objective
  fell, added as a clause.
- accuracy is not the attack: a model at chance accuracy can still rank the key first (Picek et al., TCHES
  2019(1)); when the epoch's validation was attacked, the guessing entropy after all its traces is added beside
  random's 128.5.

Author: Shraddha Mishra
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
from scipy.stats import binom

from services.side_channel_core.utils.config_errors import ConfigError

ALPHA = 0.00135         # one-sided p of three standard errors of a normal
RANDOM_GE = 128.5       # the true key's mean rank among 256 hypotheses ranked at random (ranks from 1)
OVERFIT_EPOCHS = 3      # epochs of rising validation objective under a falling training objective


def judge_log_enabled(cfg: Mapping[str, Any]) -> bool:
    """evaluation.validation.judge_log: true or false; absent or empty is false."""
    validation = ((cfg.get("evaluation") or {}).get("validation") or {})
    value = validation.get("judge_log")
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ConfigError(f"evaluation.validation.judge_log must be true or false, got {value!r}")
    return value


def _array(values: Any) -> np.ndarray:
    """A torch tensor (any device) or array-like as a numpy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


class Reading:
    """One process's accuracy against chance: correct of n, chance and the binomial test's p."""

    def __init__(self, logits: Any, labels: Any):
        predicted = _array(logits).argmax(axis=1).astype(np.int64)
        labels = _array(labels).astype(np.int64).reshape(-1)
        self.n = int(labels.size)
        if self.n == 0:
            raise ValueError("the validation judge needs at least one trace")
        self.correct = int((predicted == labels).sum())
        self.accuracy = self.correct / self.n
        width = int(max(predicted.max(), labels.max())) + 1
        self.chance = float(np.dot(np.bincount(predicted, minlength=width), np.bincount(labels, minlength=width))
                            / self.n ** 2)
        # P(at least `correct` hits | n, chance); chance 0 or 1 fixes the accuracy (no class shared, or one class
        # predicted and labelled throughout), so it is never above chance
        self.p = float(binom.sf(self.correct - 1, self.n, self.chance)) if 0 < self.chance < 1 else 1.0
        self.above = self.p < ALPHA

    def evidence(self) -> str:
        p = "p < 0.001" if self.p < 0.001 else f"p = {self.p:.2g}"
        return f"({self.correct:,} of {self.n:,}, {p})"


def guessing_entropy(results: Mapping[str, Any]) -> Optional[float]:
    """The epoch's validation guessing entropy after all its traces, when the epoch attacked: the
    guessing_entropy value, else the mean of final_ranks, else the last point of mean_attack_rank_curve."""
    if isinstance(results.get("guessing_entropy"), (int, float)):
        return float(results["guessing_entropy"])
    ranks = results.get("final_ranks")
    if ranks is not None and np.asarray(ranks).size:
        return float(np.mean(np.asarray(ranks, dtype=np.float64)))
    curve = results.get("mean_attack_rank_curve")
    if curve is not None:
        curve = np.asarray(curve, dtype=np.float64)
        if curve.ndim == 2 and curve.shape[0] and curve.shape[1] >= 2:
            return float(curve[-1, 1])
        if curve.ndim == 1 and curve.size:
            return float(curve[-1])
    return None


class ValidationJudge:
    """The verdict of each epoch of one training loop; it keeps the epochs before for trends."""

    def __init__(self) -> None:
        self._train_accuracy: Optional[float] = None
        self._objectives: list[tuple[float, float]] = []

    def line(self, *, epoch: int, epochs: int, train: tuple[Any, Any], validation: tuple[Any, Any],
             train_objective: Optional[float] = None, validation_objective: Optional[float] = None,
             validation_results: Optional[Mapping[str, Any]] = None) -> str:
        """One line for the epoch: train and validation as (logits, labels) of the epoch's traces."""
        tr, va = Reading(*train), Reading(*validation)
        before, self._train_accuracy = self._train_accuracy, tr.accuracy
        trend = "" if before is None or before == tr.accuracy else (", climbing" if tr.accuracy > before else ", falling")
        train_text = f"train accuracy {tr.accuracy:.4f} {'above' if tr.above else 'at'} chance {tr.chance:.4f}{trend}"
        val_text = f"validation {va.accuracy:.4f} {'above' if va.above else 'at'} chance {va.chance:.4f} {va.evidence()}"

        if va.above:
            clauses = [f"generalising: {val_text}; {train_text}"]
        elif tr.above:
            clauses = [f"memorising its training traces: {train_text}; {val_text}"]
        else:
            clauses = [f"not learning yet: {train_text}; {val_text}"]

        if train_objective is not None and validation_objective is not None:
            self._objectives.append((float(train_objective), float(validation_objective)))
            recent = self._objectives[-(OVERFIT_EPOCHS + 1):]
            if len(recent) == OVERFIT_EPOCHS + 1 and all(
                    b[1] > a[1] and b[0] < a[0] for a, b in zip(recent, recent[1:])):
                clauses.append(f"validation objective rising {OVERFIT_EPOCHS} epochs while the training "
                               "objective falls: overfitting")
        ge = guessing_entropy(validation_results or {})
        if ge is not None:
            clauses.append(f"validation guessing entropy {ge:.4g} after all traces (random {RANDOM_GE})")
        return f"Validation judge, epoch {epoch}/{epochs}: " + "; ".join(clauses)


__all__ = ["ALPHA", "RANDOM_GE", "OVERFIT_EPOCHS", "judge_log_enabled", "Reading", "guessing_entropy",
           "ValidationJudge"]
