# services/visualization/clearml/clearml_smoke.py

from __future__ import annotations

from pathlib import Path
import yaml

from services.visualization.clearml.clearml_tracker import ClearMLTracker


def load_cfg() -> dict:
    root = Path(__file__).resolve().parents[3]
    master_path = root / "services" / "side_channel_core" / "config" / "master.yaml"
    master = yaml.safe_load(master_path.read_text()) or {}

    default_project = master.get("default_project", "DLSCA_ASCAD")
    projects = master.get("project", []) or []

    selected_project = None
    for p in projects:
        if p.get("name") == default_project:
            selected_project = p
            break

    if selected_project is None:
        raise ValueError(f"Could not find default_project={default_project!r} in master.yaml")

    cfg = {
        "project": {
            "name": selected_project["name"],
            "root": selected_project.get("root"),
        },
        "visualization": selected_project.get("visualization", {}),
        "secrets": master.get("secrets", {}),
    }
    return cfg


def main() -> None:
    cfg = load_cfg()

    clearml_cfg = cfg.get("visualization", {}).get("clearml", {})
    scope = clearml_cfg.get("scope", "local")

    tracker = ClearMLTracker.from_cfg(
        cfg=cfg,
        process_name="smoke",
        project_name=cfg["project"]["name"],
        task_name=f"smoke_test/{scope}",
        tags=["smoke", str(scope)],
        task_type="testing",
    )

    print("scope   =", scope)
    print("enabled =", tracker.enabled)
    print("failure_reason  =", tracker.disabled_reason)

    if not tracker.enabled:
        return

    tracker.connect_parameters(
        {
            "scope": scope,
            "project": cfg["project"]["name"],
            "purpose": "clearml smoke test",
        },
        name="smoke_context",
    )

    tracker.log_scalar("smoke_metric", 1.0, step=0, series="sanity")
    tracker.upload_object(
        "smoke_payload",
        {
            "ok": True,
            "scope": scope,
            "message": "ClearML smoke test passed",
        },
    )

    tracker.close()
    print("Smoke test completed successfully.")
    

if __name__ == "__main__":
    main()
