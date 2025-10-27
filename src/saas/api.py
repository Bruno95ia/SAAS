"""FastAPI application powering the SAAS proof of concept."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from saas.config import get_settings
from saas.utils import Alert, AlertStore, configure_logging

configure_logging()
LOGGER = logging.getLogger(__name__)
settings = get_settings()
alert_store = AlertStore(settings.alerts_db_path)
uploads_dir = Path("/mnt/data/SAAS/uploads")
uploads_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="SAAS API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid api key")
    return settings.api_key


class AlertPayload(BaseModel):
    camera: str = Field(..., description="Camera identifier")
    label: str = Field(..., description="Detected label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: Optional[float] = Field(None, description="Unix timestamp of the event")
    frame_path: Optional[str] = Field(None, description="Path to the annotated frame")

    def to_alert(self) -> Alert:
        ts = self.timestamp or datetime.utcnow().timestamp()
        return Alert(
            camera=self.camera,
            label=self.label,
            confidence=self.confidence,
            timestamp=ts,
            frame_path=self.frame_path,
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/alerts")
def list_alerts(limit: int = 100, _: str = Depends(get_api_key)) -> List[dict]:
    return [alert.to_dict() for alert in alert_store.list(limit=limit)]


@app.post("/alerts", status_code=201)
def create_alert(payload: AlertPayload, _: str = Depends(get_api_key)) -> dict:
    alert = payload.to_alert()
    alert_store.add(alert)
    LOGGER.info("Stored alert from %s (%s)", alert.camera, alert.label)
    return alert.to_dict()


@app.post("/upload")
def upload_video(file: UploadFile = File(...), _: str = Depends(get_api_key)) -> dict:
    target_path = uploads_dir / file.filename
    with target_path.open("wb") as buffer:
        buffer.write(file.file.read())
    LOGGER.info("Uploaded file saved to %s", target_path)
    return {"filename": file.filename, "path": str(target_path)}


__all__ = ["app"]
