"""Utilities for resolving IASENIOR project paths."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "BASE_DIR",
    "DATASETS_DIR",
    "LOGS_DIR",
    "MODELS_DIR",
    "ensure_structure",
    "get_path",
]

_DEFAULT_BASE = Path(os.environ.get("IASENIOR_BASE_DIR", "/mnt/data/SAAS"))


def _detect_repo_root() -> Path:
    """Detect the repository root when running from a cloned repo."""

    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "src").exists():
            return candidate
    return here.parent


if not _DEFAULT_BASE.exists():
    _DEFAULT_BASE = _detect_repo_root()

BASE_DIR = _DEFAULT_BASE
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"


def ensure_structure() -> None:
    """Create the expected IASENIOR directory structure if missing."""

    for path in (LOGS_DIR, MODELS_DIR, DATASETS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_path(*parts: str) -> Path:
    """Return a path inside ``BASE_DIR`` joined with ``parts``."""

    return BASE_DIR.joinpath(*parts)
