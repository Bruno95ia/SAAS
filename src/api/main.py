from __future__ import annotations

import logging
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from saas_core.camera import camera_manager
from saas_core.config import get_settings
from saas_core.db import get_session, init_database
from saas_core.metrics import compute_metrics
from saas_core.models import Camera
from saas_core.schemas import CameraCreate, CameraRead, CameraStatus, Metrics

logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title="SAAS Fall Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    logger.info("API startup")
    init_database()
    camera_manager.load_from_db()
    with get_session() as session:
        cameras = session.query(Camera).filter(Camera.enabled.is_(True)).all()
        for camera in cameras:
            camera_manager.start_camera(camera)


@app.post("/cameras", response_model=CameraRead)
def create_camera(payload: CameraCreate):
    with get_session() as session:
        camera = Camera(name=payload.name, rtsp=payload.rtsp, enabled=payload.enabled)
        session.add(camera)
        session.flush()
        camera_id = camera.id
        camera_data = CameraRead.from_orm(camera)
    if payload.enabled and camera_id is not None:
        with get_session() as session:
            camera_db = session.get(Camera, camera_id)
            if camera_db:
                camera_manager.start_camera(camera_db)
    return camera_data


@app.get("/cameras", response_model=List[CameraRead])
def list_cameras():
    with get_session() as session:
        cameras = session.query(Camera).order_by(Camera.id.asc()).all()
        return cameras


@app.get("/cameras/status", response_model=List[CameraStatus])
def camera_status():
    statuses = camera_manager.list_status()
    statuses.sort(key=lambda item: item["id"])  # type: ignore[index]
    return [CameraStatus(**status) for status in statuses]


@app.post("/cameras/{camera_id}/start", response_model=CameraStatus)
def start_camera(camera_id: int):
    with get_session() as session:
        camera = session.get(Camera, camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        camera.enabled = True
    camera_manager.start_camera(camera)
    status = next((s for s in camera_manager.list_status() if s["id"] == camera.id), None)
    if not status:
        raise HTTPException(status_code=500, detail="Unable to start camera")
    return CameraStatus(**status)


@app.post("/cameras/{camera_id}/stop", response_model=CameraStatus)
def stop_camera(camera_id: int):
    with get_session() as session:
        camera = session.get(Camera, camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail="Camera not found")
        camera.enabled = False
    camera_manager.stop_camera(camera_id)
    status = next((s for s in camera_manager.list_status() if s["id"] == camera_id), None)
    if not status:
        status = {
            "id": camera_id,
            "name": camera.name if camera else "unknown",
            "enabled": False,
            "status": "disabled",
            "fps": 0.0,
            "last_event": None,
            "last_error": None,
        }
    return CameraStatus(**status)


@app.get("/stream/{camera_id}")
def stream_camera(camera_id: int):
    try:
        generator = camera_manager.get_stream(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(generator, media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/metrics", response_model=Metrics)
def metrics():
    data = compute_metrics()
    return Metrics(**data)
