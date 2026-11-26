# services/model_judge/report_writer.py

"""
Report Writer for Model Judge.

Writes into projects/<PROJECT>/<output_dir>:
- latest_selection.json     full result (ranking, leaderboard, rejected)
- selected_model.yaml       the winner, its evidence and artifact paths
- leaderboard.json          ranked candidates
- selected_candidates.yaml  compact ranked list (consumed by rerun tooling)

Files are written atomically.

Author: Shraddha Mishra
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from services.side_channel_core.utils.seal import seal_text


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(seal_text(path, text), encoding="utf-8")  # utils/seal.py
    os.replace(tmp, path)


def write_outputs(result: dict[str, Any], output_dir: Path, outputs: dict[str, str]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = result["selected"]

    _atomic_write(
        output_dir / outputs.get("latest_selection", "latest_selection.json"),
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(
        output_dir / outputs.get("leaderboard", "leaderboard.json"),
        json.dumps(result["leaderboard"], indent=2, allow_nan=False) + "\n",
    )
    _atomic_write(
        output_dir / outputs.get("selected_model", "selected_model.yaml"),
        yaml.safe_dump(
            {
                "schema_version": "selected_model.v2",
                "project": result["project"],
                "env": result["env"],
                "generated_at": result["generated_at"],
                "model_name": selected["model_name"],
                "model_hash": selected["model_hash"],
                "model_id": selected["model_id"],
                "decision": selected["decision"],
                "selection_uses_test_data": result["selection_uses_test_data"],
                "primary": selected["primary"],
                "selection_metrics": selected["selection_metrics"],
                "reported_metrics": selected["reported_metrics"],
                "cv_verdict": selected["cv_verdict"],
                "validation_selection_biased": selected["validation_selection_biased"],
                "artifacts": selected["artifacts"],
            },
            sort_keys=False,
        ),
    )
    _atomic_write(
        output_dir / outputs.get("selected_candidates", "selected_candidates.yaml"),
        yaml.safe_dump(
            {
                "schema_version": "selected_candidates.v2",
                "project": result["project"],
                "env": result["env"],
                "candidate_list": [item["model_name"] for item in result["leaderboard"]],
                "candidates": [
                    {
                        "rank": item["rank"],
                        "model_name": item["model_name"],
                        "model_hash": item["model_hash"],
                        "selection_metrics": item["selection_metrics"],
                    }
                    for item in result["leaderboard"]
                ],
            },
            sort_keys=False,
        ),
    )
