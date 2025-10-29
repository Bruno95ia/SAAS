"""FastAPI backend that orchestrates IASENIOR training via Ultralytics HUB."""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .utils.iasenior_paths import LOGS_DIR, MODELS_DIR, ensure_structure
from .utils.hub_integration import HubError, HubRun, UltralyticsHubManager

LOGGER = logging.getLogger("iasenior.train_api")
logging.basicConfig(level=logging.INFO)

ensure_structure()

PROGRESS_FILE = LOGS_DIR / "train_progress.json"
HISTORY_FILE = LOGS_DIR / "treinos.csv"


class TrainRequest(BaseModel):
    dataset: str = Field(..., description="Dataset identifier or slug registered on HUB")
    epochs: int = Field(50, ge=1, le=500)
    model: str = Field("yolov8n.pt", description="Ultralytics model to use as the base")
    imgsz: int = Field(640, description="Image size used during training")
    batch: int = Field(-1, description="Batch size, -1 lets YOLO decide")
    device: str = Field("auto", description="Training device string understood by YOLO")
    notes: Optional[str] = Field(None, description="Optional annotations that will be logged")


class TrainStatusResponse(BaseModel):
    run_id: Optional[str]
    status: str
    progress: Dict[str, Any]
    metrics: list[dict[str, Any]]
    weights: Dict[str, Optional[str]]
    project_url: Optional[str]
    notes: Optional[str]
    updated_at: datetime


class LatestModelResponse(BaseModel):
    path: Optional[str]
    updated_at: Optional[datetime]


class TrainingService:
    """Service layer for handling HUB training orchestration."""

    def __init__(self) -> None:
        api_key = os.environ.get("ULTRALYTICS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ULTRALYTICS_API_KEY environment variable is required for IASENIOR training API"
            )
        self.hub = UltralyticsHubManager(api_key)
        self.state: Dict[str, Any] = self._load_state()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load_state(self) -> Dict[str, Any]:
        if PROGRESS_FILE.exists():
            try:
                return json.loads(PROGRESS_FILE.read_text())
            except json.JSONDecodeError:
                LOGGER.warning("Unable to parse train_progress.json, starting fresh")
        return {
            "run_id": None,
            "status": "idle",
            "progress": {},
            "metrics": [],
            "weights": {},
            "project_url": None,
            "notes": None,
            "updated_at": datetime.utcnow().isoformat(),
        }

    def _save_state(self) -> None:
        PROGRESS_FILE.write_text(json.dumps(self.state, indent=2, default=str))

    def _append_history(self, row: Dict[str, Any]) -> None:
        headers = [
            "timestamp",
            "dataset",
            "run_id",
            "model",
            "epochs",
            "status",
            "map50",
            "map5095",
            "precision",
            "recall",
            "loss",
            "notes",
        ]
        file_exists = HISTORY_FILE.exists()
        with HISTORY_FILE.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Business operations
    # ------------------------------------------------------------------
    def start_training(self, request: TrainRequest) -> HubRun:
        run_name = f"{request.dataset}-{datetime.utcnow():%Y%m%d-%H%M%S}"
        try:
            hub_run = self.hub.start_remote_training(
                dataset=request.dataset,
                model=request.model,
                epochs=request.epochs,
                run_name=run_name,
                imgsz=request.imgsz,
                batch=request.batch,
                device=request.device,
            )
        except HubError as exc:
            LOGGER.exception("Failed to trigger HUB training")
            raise HTTPException(status_code=502, detail=str(exc))

        self.state.update(
            {
                "run_id": hub_run.run_id,
                "status": "starting",
                "dataset": request.dataset,
                "model": request.model,
                "epochs": request.epochs,
                "progress": {
                    "current_epoch": 0,
                    "epochs": request.epochs,
                },
                "metrics": [],
                "weights": {},
                "project_url": hub_run.project_url,
                "notes": request.notes,
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
        self._save_state()
        self._append_history(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "dataset": request.dataset,
                "run_id": hub_run.run_id,
                "model": request.model,
                "epochs": request.epochs,
                "status": "starting",
                "map50": "",
                "map5095": "",
                "precision": "",
                "recall": "",
                "loss": "",
                "notes": request.notes or "",
            }
        )
        return hub_run

    def refresh_status(self) -> TrainStatusResponse:
        run_id = self.state.get("run_id")
        if not run_id:
            return TrainStatusResponse(
                run_id=None,
                status="idle",
                progress=self.state.get("progress", {}),
                metrics=[],
                weights={},
                project_url=None,
                notes=self.state.get("notes"),
                updated_at=datetime.utcnow(),
            )

        try:
            status_payload = self.hub.get_run_status(run_id)
        except HubError as exc:
            LOGGER.error("Failed to query HUB status: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))

        metrics = status_payload.get("metrics", [])
        latest_metric = metrics[-1] if metrics else {}
        progress = {
            "epochs": self.state.get("epochs"),
            "current_epoch": latest_metric.get("epoch"),
        }
        if latest_metric:
            progress.update({
                "loss": latest_metric.get("train/loss"),
                "map50": latest_metric.get("metrics/mAP50"),
                "map50_95": latest_metric.get("metrics/mAP50-95"),
                "precision": latest_metric.get("metrics/precision"),
                "recall": latest_metric.get("metrics/recall"),
            })

        self.state.update(
            {
                "status": status_payload.get("status", "unknown"),
                "progress": progress,
                "metrics": metrics,
                "weights": status_payload.get("weights", {}),
                "updated_at": datetime.utcnow().isoformat(),
            }
        )
        self._save_state()

        return TrainStatusResponse(
            run_id=run_id,
            status=self.state.get("status", "unknown"),
            progress=progress,
            metrics=metrics,
            weights=status_payload.get("weights", {}),
            project_url=self.state.get("project_url"),
            notes=self.state.get("notes"),
            updated_at=datetime.utcnow(),
        )

    def latest_model(self) -> LatestModelResponse:
        if not MODELS_DIR.exists():
            return LatestModelResponse(path=None, updated_at=None)
        weights = sorted(MODELS_DIR.glob("*.pt"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not weights:
            return LatestModelResponse(path=None, updated_at=None)
        latest = weights[0]
        return LatestModelResponse(path=str(latest), updated_at=datetime.fromtimestamp(latest.stat().st_mtime))


service = TrainingService()
app = FastAPI(title="IASENIOR Training API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/train/start")
def start_training(request: TrainRequest) -> Dict[str, Any]:
    run = service.start_training(request)
    return {
        "message": "Treinamento iniciado",
        "run_id": run.run_id,
        "project_url": run.project_url,
    }


@app.get("/api/train/status", response_model=TrainStatusResponse)
def get_status() -> TrainStatusResponse:
    return service.refresh_status()


@app.get("/api/models/latest", response_model=LatestModelResponse)
def get_latest_model() -> LatestModelResponse:
    return service.latest_model()
