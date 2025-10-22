from fastapi import FastAPI, Header, HTTPException, Request
from datetime import datetime
import sqlite3
import os
from saas import config

app = FastAPI()

# Carrega a chave de API da variável de ambiente
API_KEY = os.getenv("SAAS_API_KEY", "minha-chave-forte")

DB_PATH = "events.db"


# ---------- Health Check ----------
@app.get("/health")
def health():
    return {"ok": True}


# ---------- Endpoint para registrar alertas ----------
@app.post("/post-alert")
async def post_alert(request: Request, x_api_key: str = Header(None, alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    data = await request.json()
    camera = data.get("camera")
    label = data.get("label")
    confidence = data.get("confidence")
    ts = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS events (camera TEXT, label TEXT, confidence REAL, ts TEXT)"
    )
    cursor.execute(
        "INSERT INTO events (camera, label, confidence, ts) VALUES (?, ?, ?, ?)",
        (camera, label, confidence, ts),
    )
    conn.commit()
    conn.close()

    return {"message": "Evento salvo", "data": data, "timestamp": ts}


# ---------- Endpoint para listar alertas ----------
@app.get("/alerts")
def get_alerts(limit: int = 100, x_api_key: str = Header(None, alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS events (camera TEXT, label TEXT, confidence REAL, ts TEXT)"
    )
    cursor.execute(
        "SELECT camera, label, confidence, ts FROM events ORDER BY ts DESC LIMIT ?",
        (limit,),
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {"camera": r[0], "label": r[1], "confidence": r[2], "ts": r[3]} for r in rows
    ]
