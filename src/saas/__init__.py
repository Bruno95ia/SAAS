"""SAAS proof-of-concept package."""
from __future__ import annotations

try:  # pragma: no cover - registration only happens if dependencies are present
    import torch.serialization
    from ultralytics.nn.tasks import DetectionModel
except Exception:  # pragma: no cover - optional dependency during installation
    pass
else:  # pragma: no cover
    torch.serialization.add_safe_globals([DetectionModel])

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings"]
