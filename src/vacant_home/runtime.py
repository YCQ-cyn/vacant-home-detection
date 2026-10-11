"""Resolve private runtime workspaces separately from published source code."""
from __future__ import annotations

import os
from pathlib import Path


def workspace(component: str = "recognition") -> Path:
    if component not in {"recognition", "high_power", "peer_audit"}:
        raise ValueError(f"Unknown workspace component: {component}")
    configured = os.environ.get("VACANT_HOME_WORKSPACE")
    base = Path(configured).expanduser() if configured else Path.home() / ".vacant-home-workspace"
    return base.resolve() / component


def ensure_workspace() -> Path:
    root = workspace()
    for path in (root, root / "inputs", root / "models", root / "case_plots", workspace("high_power")):
        path.mkdir(parents=True, exist_ok=True)
    return root

