from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sqlite3
from typing import List, Optional, Dict, Any
import json

DB = Path("events.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  camera_id TEXT NOT NULL,
  type TEXT NOT NULL,
  score REAL,
  clip_path TEXT,
  extra TEXT
);
"""

def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute(SCHEMA)
    return con

@dataclass
class Alert:
    camera_id: str
    type: str
    score: float = 0.0
    clip_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    ts: str = datetime.utcnow().isoformat()

def insert_alert(a: Alert) -> int:
    con = _conn()
    cur = con.cursor()
    cur.execute(
        "INSERT INTO alerts(ts,camera_id,type,score,clip_path,extra) VALUES(?,?,?,?,?,?)",
        (a.ts, a.camera_id, a.type, a.score, a.clip_path,
         None if a.extra is None else json.dumps(a.extra)),
    )
    con.commit()
    return cur.lastrowid

def recent(n: int = 50) -> List[Dict[str, Any]]:
    con = _conn()
    cur = con.cursor()
    cur.execute("SELECT id,ts,camera_id,type,score,clip_path,extra FROM alerts ORDER BY ts DESC LIMIT ?", (n,))
    cols = [c[0] for c in cur.description]
    out = []
    for row in cur.fetchall():
        item = dict(zip(cols, row))
        if item.get("extra"):
            try:
                item["extra"] = json.loads(item["extra"])
            except Exception:
                pass
        out.append(item)
    return out
