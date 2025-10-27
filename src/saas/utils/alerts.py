"""Utilities for storing and dispatching alerts."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence

import requests

from saas.config import get_settings

LOGGER = logging.getLogger(__name__)


@dataclass
class Alert:
    camera: str
    label: str
    confidence: float
    timestamp: float
    frame_path: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["timestamp"] = self.timestamp
        return data


class AlertStore:
    """Simple SQLite-backed persistence for detection alerts."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path else settings.alerts_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    frame_path TEXT
                )
                """
            )

    def add(self, alert: Alert) -> None:
        LOGGER.info(
            "Saving alert %s from camera %s (%.2f)", alert.label, alert.camera, alert.confidence
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO alerts (camera, label, confidence, timestamp, frame_path) VALUES (?, ?, ?, ?, ?)",
                (alert.camera, alert.label, alert.confidence, alert.timestamp, alert.frame_path),
            )

    def list(self, limit: int = 100) -> List[Alert]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT camera, label, confidence, timestamp, frame_path FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cursor.fetchall()
        return [Alert(*row) for row in rows]

    def clear(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM alerts")


def build_alert_from_detection(camera: str, label: str, confidence: float, frame_path: str | None) -> Alert:
    timestamp = time.time()
    return Alert(camera=camera, label=label, confidence=confidence, timestamp=timestamp, frame_path=frame_path)


def send_alert(alert: Alert, session: requests.Session | None = None) -> None:
    """Send the alert payload to the configured API endpoint."""

    settings = get_settings()
    url = f"{settings.api_url.rstrip('/')}/alerts"
    payload = alert.to_dict()
    headers = {"X-API-Key": settings.api_key}

    try:
        http = session or requests.Session()
        response = http.post(url, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        LOGGER.info("Alert dispatched to API (%s)", url)
    except Exception as exc:  # pragma: no cover - network failures are expected sometimes
        LOGGER.warning("Failed to dispatch alert to API: %s", exc)


def serialize_alerts(alerts: Sequence[Alert]) -> str:
    return json.dumps([alert.to_dict() for alert in alerts], indent=2)


def alerts_to_dataframe(alerts: Sequence[Alert]) -> List[dict]:
    return [alert.to_dict() for alert in alerts]


__all__ = ["Alert", "AlertStore", "alerts_to_dataframe", "build_alert_from_detection", "send_alert", "serialize_alerts"]
