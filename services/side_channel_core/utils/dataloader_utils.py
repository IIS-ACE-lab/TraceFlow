# services/side_channel_core/utils/dataloader_utils.py

from __future__ import annotations

from typing import Any, Dict, Tuple


def _get_dataloader_cfg(cfg: dict) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Backwards-compatible DataLoader config normalization.

    Supports:
      - cfg["dataloader"] : dict (optional)
      - cfg["num_workers"]: dict {train,val} OR int (applies to both)

    Returns:
      (dl_cfg, nw_cfg)
        dl_cfg:
          pin_memory: bool
          persistent_workers: bool            # global preference; loader builder must still gate by nw>0
          prefetch_factor: int | None         # None when illegal (nw==0)
          drop_last_train: bool
        nw_cfg:
          train: int
          val  : int

    SCA-fit relevance:
      - determinism and correctness are handled elsewhere (seed + worker_init_fn + generator).
      - this helper ensures performance knobs are legal and predictable.
    """
    if not isinstance(cfg, dict):
        cfg = {}

    dl = cfg.get("dataloader") or {}
    if not isinstance(dl, dict):
        dl = {}

    nw_raw = cfg.get("num_workers", {})  # may be dict or int

    # Normalize num_workers
    # defaults (owner 2026-10-23): the blueprint's values, 4 training workers and 1 validation worker
    if isinstance(nw_raw, int):
        nw_train = nw_val = int(nw_raw)
    elif isinstance(nw_raw, dict):
        nw_train = int(nw_raw.get("train", 4))
        nw_val = int(nw_raw.get("val", 1))
    else:
        nw_train = 4
        nw_val = 1

    # Clamp to sane bounds
    if nw_train < 0:
        nw_train = 0
    if nw_val < 0:
        nw_val = 0

    # Defaults (safe, production-friendly)
    pin_memory = bool(dl.get("pin_memory", True))
    persistent_workers_pref = bool(dl.get("persistent_workers", True))
    drop_last_train = bool(dl.get("drop_last_train", True))     # default true (owner 2026-10-23): full batches only

    # prefetch_factor is only valid when num_workers > 0
    pf_raw = dl.get("prefetch_factor", 4)
    try:
        pf_int = int(pf_raw)
    except Exception:
        pf_int = 4

    if pf_int < 2:
        pf_int = 2

    # Provide a legal, normalized config:
    # - keep prefetch_factor=None when it would be illegal for a given loader
    #   (loader builder will decide per-loader)
    out_dl: Dict[str, Any] = {
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers_pref,
        "prefetch_factor": pf_int,  # builder must still omit when nw==0
        "drop_last_train": drop_last_train,
    }

    out_nw = {"train": nw_train, "val": nw_val}

    # Optional: tighten legality hints (does not override builder’s final guardrails)
    # If BOTH train and val are single-process loaders, persistent_workers is meaningless.
    if nw_train == 0 and nw_val == 0:
        out_dl["persistent_workers"] = False

    return out_dl, out_nw
