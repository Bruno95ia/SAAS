"""Video helper utilities."""
from __future__ import annotations

import cv2
from pathlib import Path
from typing import Optional


class FrameWriter:
    """Persist the most recent annotated frame for a camera."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, frame) -> str:
        cv2.imwrite(str(self.output_path), frame)
        return str(self.output_path)


def read_frame_bytes(path: Path) -> Optional[bytes]:
    if not path.exists():
        return None
    return path.read_bytes()


__all__ = ["FrameWriter", "read_frame_bytes"]
