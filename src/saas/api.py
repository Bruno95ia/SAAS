"""API FastAPI responsável por servir clipes e registrar alertas."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from saas import config
from saas.clipper import save_clip_from_file
from saas.store import Alert, health_check, insert_alert, recent
from saas.utils.logger import get_logger

LOGGER = get_logger("saas.api")


class AlertIn(BaseModel):
    camera_id: str = Field(..., description="Identificador lógico da câmera")
    type: str = Field(..., description="Tipo do evento (ex.: fall)")
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    clip_path: Optional[str] = Field(default=None, description="URL do clipe gravado")
    extra: Optional[Dict[str, Any]] = Field(default=None, description="Metadados adicionais")


class AlertOut(AlertIn):
    id: int
    ts: str


class HealthComponent(BaseModel):
    status: str
    details: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    components: Dict[str, HealthComponent]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config.ensure_runtime_directories()
    settings = config.load_api_settings()
    app.state.settings = settings
    app.state.subscribers: List[WebSocket] = []
    LOGGER.info("API inicializada url=%s", settings.url)
    try:
        yield
    finally:
        LOGGER.info("Encerrando API")


def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    request: Request = Depends(),
) -> str:
    app = request.app if request is not None else app_instance
    key = getattr(app.state, "settings", config.load_api_settings()).key
    if not key:
        return ""
    if x_api_key != key:
        LOGGER.warning("Chave de API inválida")
        raise HTTPException(status_code=401, detail="invalid api key")
    return key


app_instance = FastAPI(title="SAAS Fall Detection API", lifespan=lifespan)
app = app_instance

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/clips", StaticFiles(directory=config.CLIPS_DIR, check_dir=False), name="clips")


def _directory_status() -> Dict[str, Any]:
    paths = {
        "runs": config.RUNS_DIR,
        "buffer": config.BUFFER_DIR,
        "clips": config.CLIPS_DIR,
        "logs": config.LOG_DIR,
    }
    details = {name: {"path": str(path), "exists": Path(path).exists()} for name, path in paths.items()}
    status = "ok" if all(item["exists"] for item in details.values()) else "warn"
    return {"status": status, "details": details}


def _weights_status() -> Dict[str, Any]:
    candidate = config.resolve_weights_path(str(config.DEFAULT_WEIGHTS))
    exists = Path(candidate).exists()
    return {
        "status": "ok" if exists else "warn",
        "details": {"path": str(candidate), "exists": exists},
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    db_status = health_check()
    dirs = _directory_status()
    weights = _weights_status()
    status = "ok"
    components = {
        "database": HealthComponent(status=db_status.get("status", "error"), details=db_status),
        "directories": HealthComponent(status=dirs["status"], details=dirs["details"]),
        "weights": HealthComponent(status=weights["status"], details=weights["details"]),
    }
    if any(component.status != "ok" for component in components.values()):
        status = "warn"
    return HealthResponse(status=status, components=components)


@app.get("/alerts", response_model=List[AlertOut], dependencies=[Depends(require_api_key)])
async def get_alerts(limit: int = 50) -> List[AlertOut]:
    rows = recent(limit)
    return [AlertOut(**row) for row in rows]


@app.post("/alerts", dependencies=[Depends(require_api_key)])
async def post_alert(payload: AlertIn) -> JSONResponse:
    alert = Alert(**payload.model_dump())
    alert_id = insert_alert(alert)
    message = {"event": "alert", "id": alert_id, "data": payload.model_dump()}
    dead: List[WebSocket] = []
    for ws in list(app.state.subscribers):
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            app.state.subscribers.remove(ws)
        except ValueError:
            pass
    LOGGER.info("POST /alerts id=%s camera=%s", alert_id, payload.camera_id)
    return JSONResponse({"id": alert_id})


@app.websocket("/ws")
async def websocket_alerts(ws: WebSocket) -> None:
    await ws.accept()
    app.state.subscribers.append(ws)
    LOGGER.info("WebSocket conectado (%s)", len(app.state.subscribers))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        LOGGER.info("WebSocket desconectado")
    finally:
        if ws in app.state.subscribers:
            app.state.subscribers.remove(ws)


def on_fall_detected(src_video_path: str, event_ts_sec: float, score: float, camera_id: str) -> None:
    """Helper utilizado pelo pipeline para registrar alertas diretamente."""

    settings = config.load_api_settings()
    local_path, clip_url = save_clip_from_file(
        src_video=src_video_path,
        event_ts_sec=event_ts_sec,
        pre_sec=5.0,
        post_sec=5.0,
        camera_id=camera_id,
        api_base_url=settings.url,
    )

    payload = {
        "camera_id": camera_id,
        "type": "fall",
        "score": float(score),
        "clip_path": clip_url,
        "extra": {"local_path": local_path},
    }
    response = requests.post(
        f"{settings.url}/alerts",
        headers={"X-API-Key": settings.key, "Content-Type": "application/json"},
        json=payload,
        timeout=10,
    )
    response.raise_for_status()
