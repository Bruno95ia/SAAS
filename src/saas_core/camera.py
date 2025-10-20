from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Generator, List, Optional

import cv2
import numpy as np
import psutil
import torch

from .config import get_settings
from .detector import Detection, fall_detector
from .db import get_session
from .models import Camera, Event, FrameLabel

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class StreamFrame:
    timestamp: datetime
    image: bytes


@dataclass
class CameraRuntime:
    camera: Camera
    thread: threading.Thread | None = None
    running: bool = False
    frame_queue: Deque[StreamFrame] = field(default_factory=lambda: deque(maxlen=settings.max_stream_backlog))
    fps_history: Deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_event_ts: Optional[datetime] = None


class CameraWorker:
    def __init__(self, runtime: CameraRuntime) -> None:
        self.runtime = runtime
        self.capture: Optional[cv2.VideoCapture] = None
        self.clip_dir = settings.storage.data_dir / "clips"
        self.clip_dir.mkdir(parents=True, exist_ok=True)

    def _open_capture(self) -> bool:
        logger.info("Opening camera %s", self.runtime.camera.name)
        self.capture = cv2.VideoCapture(self.runtime.camera.rtsp)
        return self.capture.isOpened()

    def _write_clip(self, frames: List[np.ndarray], event: Event) -> Optional[str]:
        if not frames:
            return None
        path = self.clip_dir / f"event_{event.id}_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = max(1, int(self.runtime.fps_history[-1] if self.runtime.fps_history else settings.stream_fps))
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        for frame in frames:
            writer.write(frame)
        writer.release()
        logger.info("Clip saved to %s", path)
        return str(path)

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0) if det.label.lower() != "fall" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{det.label}:{det.score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def _push_frame(self, frame: np.ndarray) -> None:
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            return
        self.runtime.frame_queue.append(
            StreamFrame(timestamp=datetime.utcnow(), image=buffer.tobytes())
        )

    def run(self) -> None:
        if not self._open_capture():
            logger.error("Failed to open camera %s", self.runtime.camera.name)
            self.runtime.running = False
            return

        fall_frames: List[np.ndarray] = []
        fall_event: Optional[Event] = None
        clip_window = timedelta(seconds=settings.detection.clip_length)
        last_detection_time: Optional[datetime] = None

        while self.runtime.running:
            start = time.time()
            ret, frame = self.capture.read() if self.capture else (False, None)
            if not ret or frame is None:
                logger.warning("Camera %s frame grab failed", self.runtime.camera.name)
                time.sleep(1)
                continue

            detections = fall_detector.infer(frame)
            fall_detection = fall_detector.detect_fall(detections)
            frame_drawn = self._draw_detections(frame.copy(), detections)
            self._push_frame(frame_drawn)

            if fall_detection:
                fall_frames.append(frame)
                if fall_event is None:
                    with get_session() as session:
                        event = Event(
                            camera_id=self.runtime.camera.id,
                            label="fall",
                            score=fall_detection.score,
                        )
                        session.add(event)
                        session.flush()
                        fall_event = event
                        logger.info("Fall event %s created", event.id)
                else:
                    logger.debug("Continuing fall event %s", fall_event.id)

                with get_session() as session:
                    db_event = session.get(Event, fall_event.id)
                    if db_event is None:
                        continue
                    db_event.end_ts = datetime.utcnow()
                    label = FrameLabel(
                        event_id=db_event.id,
                        ts=datetime.utcnow(),
                        x1=fall_detection.bbox[0],
                        y1=fall_detection.bbox[1],
                        x2=fall_detection.bbox[2],
                        y2=fall_detection.bbox[3],
                        cls="fall",
                        score=fall_detection.score,
                        )
                    session.add(label)
                    last_detection_time = datetime.utcnow()
                    self.runtime.last_event_ts = last_detection_time

            if fall_event and fall_frames and last_detection_time:
                # check if event window elapsed
                if datetime.utcnow() - last_detection_time > clip_window:
                    with get_session() as session:
                        db_event = session.get(Event, fall_event.id)
                        if db_event:
                            clip_path = self._write_clip(fall_frames, db_event)
                            db_event.clip_path = clip_path
                    fall_frames.clear()
                    fall_event = None
                    last_detection_time = None

            elapsed = time.time() - start
            if elapsed > 0:
                fps = 1.0 / elapsed
                self.runtime.fps_history.append(fps)
            time.sleep(max(0.001, (1 / settings.stream_fps) - elapsed))

        if self.capture:
            self.capture.release()
        self.runtime.running = False


class CameraManager:
    def __init__(self) -> None:
        self._runtimes: Dict[int, CameraRuntime] = {}
        self._lock = threading.Lock()

    def load_from_db(self) -> None:
        with get_session() as session:
            cameras = session.query(Camera).filter(Camera.enabled.is_(True)).all()
            for camera in cameras:
                self._runtimes[camera.id] = CameraRuntime(camera=camera)

    def start_camera(self, camera: Camera) -> None:
        with self._lock:
            runtime = self._runtimes.get(camera.id)
            if runtime is None:
                runtime = CameraRuntime(camera=camera)
                self._runtimes[camera.id] = runtime
            if runtime.running:
                return
            runtime.running = True
            worker = CameraWorker(runtime)
            thread = threading.Thread(target=worker.run, daemon=True)
            runtime.thread = thread
            thread.start()
            logger.info("Camera %s started", camera.name)

    def stop_camera(self, camera_id: int) -> None:
        with self._lock:
            runtime = self._runtimes.get(camera_id)
            if runtime and runtime.running:
                runtime.running = False
                if runtime.thread:
                    runtime.thread.join(timeout=5)
                logger.info("Camera %s stopped", runtime.camera.name)

    def enqueue_frame(self, camera_id: int, frame: StreamFrame) -> None:
        runtime = self._runtimes.get(camera_id)
        if runtime:
            runtime.frame_queue.append(frame)

    def get_stream(self, camera_id: int) -> Generator[bytes, None, None]:
        runtime = self._runtimes.get(camera_id)
        if not runtime:
            raise ValueError("Camera not found")

        while True:
            if not runtime.running:
                time.sleep(0.1)
                continue
            if not runtime.frame_queue:
                time.sleep(0.05)
                continue
            frame = runtime.frame_queue[-1]
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame.image + b"\r\n"
            )

    def metrics(self) -> Dict[str, float]:
        fps_values = [np.mean(runtime.fps_history) for runtime in self._runtimes.values() if runtime.fps_history]
        active = sum(1 for runtime in self._runtimes.values() if runtime.running)
        total = len(self._runtimes)
        falls = 0
        with get_session() as session:
            falls = session.query(Event).filter(Event.label == "fall").count()
        cpu_usage = psutil.cpu_percent()
        free = psutil.disk_usage(str(settings.storage.data_dir)).free / (1024 ** 3)
        gpu_available = torch.cuda.is_available()
        return {
            "fps_average": float(np.mean(fps_values)) if fps_values else 0.0,
            "active_cameras": active,
            "total_cameras": total,
            "falls_detected": falls,
            "cpu_usage": cpu_usage,
            "storage_free": free,
            "gpu_available": gpu_available,
        }

    def list_status(self) -> List[Dict[str, Optional[float]]]:
        statuses = []
        for runtime in self._runtimes.values():
            statuses.append(
                {
                    "id": runtime.camera.id,
                    "name": runtime.camera.name,
                    "enabled": runtime.camera.enabled,
                    "status": "running" if runtime.running else "stopped",
                    "fps": float(np.mean(runtime.fps_history)) if runtime.fps_history else 0.0,
                    "last_event": runtime.last_event_ts,
                }
            )
        return statuses


camera_manager = CameraManager()
