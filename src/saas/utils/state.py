"""Shared state helpers for pipeline and dashboard."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict


_STATE_LOCK = threading.Lock()


def update_metrics_state(path: Path, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _STATE_LOCK:
        data: Dict[str, Any] = {}
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                data = {}
        data.update(payload)
        path.write_text(json.dumps(data, indent=2))


def read_metrics_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


__all__ = ["update_metrics_state", "read_metrics_state"]
