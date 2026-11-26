# core/utils/container_guard.py
'''
This module provides utilities to ensure that certain code paths are executed
within a containerized environment. It checks for specific files and cgroup
information to determine if the code is running inside a container.

Author: Shraddha Mishra

'''
import os, re, sys, pathlib

def _in_container_files() -> bool:
    return any(pathlib.Path(p).exists() for p in ("/.dockerenv", "/run/.containerenv"))

def _in_container_cgroup() -> bool:
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as f:
            cg = f.read()
        return bool(re.search(r"(docker|containerd|podman|kubepods)", cg, re.I))
    except Exception:
        return False

def assert_in_container():
    # Allow an explicit bypass for rare local testing
    if os.environ.get("ALLOW_HOST_RUN") == "1":
        return

    # Treat RUN_CONTEXT=container as authoritative (compose sets this)
    if os.environ.get("RUN_CONTEXT", "").lower() == "container":
        return

    # Otherwise, accept common runtime markers as a fallback
    if _in_container_files() or _in_container_cgroup():
        return

    # Bare metal (or unknown env): refuse to run
    sys.stderr.write(
        "Refusing to run on bare metal.\n"
        "   This command must run inside the sidechannel container.\n"
        "   Start it with: make orchestrate ENV=dev|staging|prod\n"
    )
    sys.exit(23)
