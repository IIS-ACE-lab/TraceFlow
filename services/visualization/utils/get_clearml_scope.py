#!/usr/bin/env python3
#services/visualization/utils/get_clearml_scope.py
"""

Author: Shraddha Mishra
"""
import yaml
from pathlib import Path
import os

master = Path("services/side_channel_core/config/master.yaml")
data = yaml.safe_load(master.read_text()) or {}

requested_project = os.getenv("PROJECT", "").strip()
default_project = (data.get("default_project") or "").strip()

projects = data.get("project") or []
if isinstance(projects, dict):
    projects = [projects]

project_name = requested_project or default_project
scope = "local"

for pr in projects:
    if isinstance(pr, dict) and pr.get("name") == project_name:
        viz = pr.get("visualization") or {}
        clearml = viz.get("clearml") or {}
        scope = str(clearml.get("scope", "local")).strip().lower()
        break

print(scope)
