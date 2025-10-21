from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    rtsp: str = Field(..., min_length=5, max_length=512)
    enabled: bool = True


class CameraRead(BaseModel):
    id: int
    name: str
    rtsp: str
    enabled: bool
    created_at: datetime

    class Config:
        orm_mode = True


class CameraStatus(BaseModel):
    id: int
    name: str
    enabled: bool
    status: str
    fps: float
    last_event: Optional[datetime]
    last_error: Optional[str] = None


class EventRead(BaseModel):
    id: int
    camera_id: int
    start_ts: datetime
    end_ts: Optional[datetime]
    label: str
    score: Optional[float]
    clip_path: Optional[str]

    class Config:
        orm_mode = True


class FrameLabelRead(BaseModel):
    id: int
    event_id: int
    ts: datetime
    x1: int
    y1: int
    x2: int
    y2: int
    cls: str
    score: float

    class Config:
        orm_mode = True


class Metrics(BaseModel):
    total_cameras: int
    active_cameras: int
    falls_detected: int
    fps_average: float
    events_per_hour: List[int]
    time_between_falls: Optional[float]
    recent_events: List[EventRead]
    storage_free_gb: float
    gpu_available: bool
    cpu_usage: float
