# services/side_channel_core/utils/seed.py
""" 
Seed Utilities for Reproducibility
This module provides functions to set random seeds across various libraries
to ensure reproducible results in experiments. It includes setting seeds for Python's
random module, NumPy, and PyTorch (both CPU and CUDA). Additionally, it provides
a function to create a worker initialization function for DataLoader workers to
ensure they are seeded correctly.

Author: Shraddha Mishra
"""
# services/side_channel_core/utils/seed.py
from __future__ import annotations

import os
import random
from typing import Callable

import numpy as np
import torch


# cfg["seed_drawn"]: the seed an unseeded run (seed: null) drew for itself; written into its params_out snapshot, so
# its test, its attacks and a reproduce use the same one. Never part of the run hash (seed: null is).
SEED_DRAWN_KEY = "seed_drawn"


def run_seed(cfg: dict) -> int:
    """
    The seed a run uses. seed an integer: that one. seed: null (an unseeded run, as Zaid et al.'s and Wouters et
    al.'s code, which seed nothing): one seed drawn from the operating system the first time the run asks, kept
    in cfg["seed_drawn"] and reused from then on (a fresh run draws a new one). No seed key at all: 42, as before.
    """
    cfg = cfg if cfg is not None else {}
    if "seed" in cfg and cfg["seed"] is None:
        drawn = cfg.get(SEED_DRAWN_KEY)
        if drawn is None:
            import secrets

            drawn = secrets.randbelow(2**31 - 1)
            if isinstance(cfg, dict):
                cfg[SEED_DRAWN_KEY] = drawn
        if isinstance(drawn, bool) or not isinstance(drawn, int):
            raise TypeError(f"{SEED_DRAWN_KEY} must be an integer, got {drawn!r}")
        return int(drawn)
    return int(cfg.get("seed", 42))


def set_seed(cfg: dict) -> int:
    """
    Set seeds for reproducibility. Returns the seed used (run_seed: with seed: null, the one this run drew).
    """
    seed = run_seed(cfg)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Deterministic kernels everywhere they exist. cuBLAS needs this variable
    # before its first call; set_seed runs before any model is built. An op
    # with no deterministic kernel warns instead of stopping the run.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    """
    Returns a DataLoader worker_init_fn that deterministically seeds each worker.
    """
    seed = int(seed)

    def _init(worker_id: int) -> None:
        s = seed + int(worker_id)
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    return _init


def make_torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    """
    Create a deterministic torch.Generator for DataLoader shuffling.
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g
