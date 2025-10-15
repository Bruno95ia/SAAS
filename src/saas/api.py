from typing import Optional, Dict, Any, List
import os
import requests
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .clipper import save_clip_from_file
from .store import Alert, insert_alert, recent

API_KEY = os.getenv("SAAS_API_KEY", "dev-key")  # defina em prod!

app = FastAPI(title="SAAS Fall Detection API")
Path("runs/clips").mkdir(parents=True, exist_ok=True)
app.mount("/clips", StaticFiles(directory="runs/clips"), name="clips")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def require_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

class AlertIn(BaseModel):
    camera_id: str
    type: str                # "fall", "no_fall", "warning"
    score: float = 0.0
    clip_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

subscribers: List[WebSocket] = []

@app.get("/health")
def health(): return {"ok": True}

@app.get("/alerts", dependencies=[Depends(require_api_key)])
def get_alerts(limit: int = 50):
    return recent(limit)

@app.post("/alerts", dependencies=[Depends(require_api_key)])
async def post_alert(a: AlertIn):
    aid = insert_alert(Alert(**a.model_dump()))
    dead = []
    for ws in subscribers:
        try:
            await ws.send_json({"event": "alert", "id": aid, "data": a.model_dump()})
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in subscribers:
            subscribers.remove(ws)
    return {"id": aid}

@app.websocket("/ws")
async def ws_alerts(ws: WebSocket):
    await ws.accept()
    subscribers.append(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        if ws in subscribers:
            subscribers.remove(ws)

            
API = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000")
KEY = os.getenv("SAAS_API_KEY", "minha-chave-forte")

def on_fall_detected(src_video_path: str, event_ts_sec: float, score: float, camera_id: str):
    file_path, clip_url = save_clip_from_file(
        src_video=src_video_path,
        event_ts_sec=event_ts_sec,
        pre_sec=5.0, post_sec=5.0,
        camera_id=camera_id,
        api_base_url=API,
    )
    # envia o alerta
    r = requests.post(f"{API}/alerts",
                      headers={"X-API-Key": KEY},
                      json={"camera_id": camera_id, "type": "fall", "score": score,
                            "clip_path": clip_url, "extra": {"local_path": file_path}})
    r.raise_for_status()