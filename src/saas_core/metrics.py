from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import psutil

from .camera import camera_manager
from .config import get_settings
from .db import get_session
from .models import Event

logger = logging.getLogger(__name__)
settings = get_settings()


def compute_metrics() -> Dict:
    manager_metrics = camera_manager.metrics()
    now = datetime.utcnow()
    one_day_ago = now - timedelta(hours=24)
    events_per_hour = [0] * 24
    last_fall_ts = None
    previous_fall_ts = None

    with get_session() as session:
        events = (
            session.query(Event)
            .filter(Event.start_ts >= one_day_ago)
            .order_by(Event.start_ts.asc())
            .all()
        )
        recent_events = (
            session.query(Event)
            .order_by(Event.start_ts.desc())
            .limit(10)
            .all()
        )

    for event in events:
        hour_index = int((event.start_ts - one_day_ago).total_seconds() // 3600)
        if 0 <= hour_index < 24:
            events_per_hour[hour_index] += 1
        if event.label == "fall":
            previous_fall_ts = last_fall_ts
            last_fall_ts = event.start_ts

    time_between_falls = None
    if last_fall_ts and previous_fall_ts:
        time_between_falls = (last_fall_ts - previous_fall_ts).total_seconds() / 60.0

    disk_usage = psutil.disk_usage(str(settings.storage.data_dir))

    metrics = {
        "total_cameras": manager_metrics["total_cameras"],
        "active_cameras": manager_metrics["active_cameras"],
        "falls_detected": manager_metrics["falls_detected"],
        "fps_average": manager_metrics["fps_average"],
        "events_per_hour": events_per_hour,
        "time_between_falls": time_between_falls,
        "recent_events": [
            {
                "id": event.id,
                "camera_id": event.camera_id,
                "start_ts": event.start_ts,
                "end_ts": event.end_ts,
                "label": event.label,
                "score": float(event.score) if event.score is not None else None,
                "clip_path": event.clip_path,
            }
            for event in recent_events
        ],
        "storage_free_gb": round(disk_usage.free / (1024 ** 3), 2),
        "gpu_available": manager_metrics["gpu_available"],
        "cpu_usage": manager_metrics["cpu_usage"],
    }
    return metrics
