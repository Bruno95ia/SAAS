"""Persistência simples em SQLite para eventos de alerta."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from saas import config
from saas.utils.logger import get_logger

LOGGER = get_logger("saas.store")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    type TEXT NOT NULL,
    score REAL,
    clip_path TEXT,
    extra TEXT
);
"""


@dataclass(slots=True)
class Alert:
    camera_id: str
    type: str
    score: float = 0.0
    clip_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _connect() -> sqlite3.Connection:
    config.ensure_runtime_directories()
    path = config.DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(SCHEMA)
    return conn


def insert_alert(alert: Alert) -> int:
    LOGGER.debug("Persistindo alerta camera=%s tipo=%s", alert.camera_id, alert.type)
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO alerts (ts, camera_id, type, score, clip_path, extra)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                alert.ts,
                alert.camera_id,
                alert.type,
                float(alert.score),
                alert.clip_path,
                json.dumps(alert.extra) if alert.extra is not None else None,
            ),
        )
        conn.commit()
        alert_id = int(cur.lastrowid)
    LOGGER.info("Alerta registrado id=%s camera=%s tipo=%s", alert_id, alert.camera_id, alert.type)
    return alert_id


def recent(limit: int = 50) -> List[Dict[str, Any]]:
    with _connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts, camera_id, type, score, clip_path, extra
            FROM alerts
            ORDER BY datetime(ts) DESC
            LIMIT ?
            """,
            (int(limit),),
        )
        rows = cur.fetchall()

    payload: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("extra"):
            try:
                item["extra"] = json.loads(item["extra"])
            except json.JSONDecodeError:
                LOGGER.warning("Falha ao decodificar campo extra do alerta id=%s", item.get("id"))
        payload.append(item)
    return payload


def health_check() -> Dict[str, Any]:
    """Retorna informações básicas sobre o banco de dados."""

    path = config.DATABASE_PATH
    status = {"path": str(path), "exists": Path(path).exists(), "count": 0}
    try:
        with _connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) FROM alerts")
            (count,) = cur.fetchone()
            status["count"] = int(count)
            status["status"] = "ok"
    except Exception as exc:  # pragma: no cover - diagnóstico
        status["status"] = "error"
        status["error"] = str(exc)
        LOGGER.exception("Healthcheck do banco falhou")
    return status
