"""System and inference metrics utilities."""
from __future__ import annotations

import collections
import psutil

from dataclasses import dataclass
from typing import Deque, Dict


@dataclass
class PerformanceSnapshot:
    fps: float
    frame_time: float


class PerformanceTracker:
    """Track moving average of frames per second for inference."""

    def __init__(self, window: int = 60) -> None:
        self.window = window
        self._history: Deque[PerformanceSnapshot] = collections.deque(maxlen=window)

    def observe(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0:
            return 0.0
        fps = 1.0 / elapsed_seconds
        self._history.append(PerformanceSnapshot(fps=fps, frame_time=elapsed_seconds))
        return fps

    def average_fps(self) -> float:
        if not self._history:
            return 0.0
        return sum(snapshot.fps for snapshot in self._history) / len(self._history)


class SystemMetricsCollector:
    """Collect CPU, memory, disk usage metrics."""

    @staticmethod
    def collect() -> Dict[str, float]:
        cpu = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage("/").percent
        return {"cpu": cpu, "memory": memory, "disk": disk}


__all__ = ["PerformanceSnapshot", "PerformanceTracker", "SystemMetricsCollector"]
