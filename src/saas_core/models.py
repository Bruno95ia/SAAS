from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    rtsp: Mapped[str] = mapped_column(String(512), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    events: Mapped[List["Event"]] = relationship(back_populates="camera", cascade="all, delete")


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[int] = mapped_column(ForeignKey("cameras.id"), nullable=False)
    start_ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    end_ts: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[Optional[float]] = mapped_column(Numeric(5, 2))
    clip_path: Mapped[Optional[str]] = mapped_column(String(512))

    camera: Mapped[Camera] = relationship(back_populates="events")
    frames: Mapped[List["FrameLabel"]] = relationship(
        back_populates="event", cascade="all, delete-orphan"
    )


class FrameLabel(Base):
    __tablename__ = "frame_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    x1: Mapped[int] = mapped_column(Integer)
    y1: Mapped[int] = mapped_column(Integer)
    x2: Mapped[int] = mapped_column(Integer)
    y2: Mapped[int] = mapped_column(Integer)
    cls: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Numeric(5, 2))

    event: Mapped[Event] = relationship(back_populates="frames")
