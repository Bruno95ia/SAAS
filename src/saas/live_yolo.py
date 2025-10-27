"""Live YOLOv8 inference service for the SAAS stack."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, List

import cv2
from ultralytics import YOLO

from saas.config import get_settings
from saas.utils import (
    Alert,
    AlertStore,
    FrameWriter,
    PerformanceTracker,
    build_alert_from_detection,
    configure_logging,
    send_alert,
    update_metrics_state,
)

configure_logging()
LOGGER = logging.getLogger(__name__)


@dataclass
class CameraRuntime:
    name: str
    source: str
    writer: FrameWriter
    tracker: PerformanceTracker


class LiveYOLOService:
    """Process configured cameras and dispatch alerts in real time."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.model = YOLO(str(self.settings.weights_path))
        self.alert_store = AlertStore(self.settings.alerts_db_path)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._runtimes: Dict[str, CameraRuntime] = {}

        LOGGER.info("LiveYOLOService initialized with settings:\n%s", self.settings.to_json())

        for index, camera_source in enumerate(self.settings.cameras, start=1):
            camera_name = f"Camera {index:02d}"
            frame_path = self.settings.stream_dir / f"camera_{index:02d}.jpg"
            runtime = CameraRuntime(
                name=camera_name,
                source=camera_source,
                writer=FrameWriter(frame_path),
                tracker=PerformanceTracker(),
            )
            self._runtimes[camera_name] = runtime

    def start(self) -> None:
        LOGGER.info("Starting YOLO inference threads for %d cameras", len(self._runtimes))
        for runtime in self._runtimes.values():
            thread = threading.Thread(target=self._run_camera, args=(runtime,), daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        LOGGER.info("Stopping LiveYOLOService")
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2)

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                time.sleep(1)
        except KeyboardInterrupt:  # pragma: no cover
            LOGGER.info("Keyboard interrupt received, stopping service")
            self.stop()

    def _run_camera(self, runtime: CameraRuntime) -> None:
        LOGGER.info("Starting camera loop: %s (%s)", runtime.name, runtime.source)
        capture = cv2.VideoCapture(runtime.source)
        if not capture.isOpened():
            LOGGER.error("Unable to open video source %s", runtime.source)
            return

        while not self._stop.is_set():
            success, frame = capture.read()
            if not success:
                LOGGER.warning("Stream ended for %s, restarting", runtime.name)
                capture.release()
                time.sleep(1)
                capture = cv2.VideoCapture(runtime.source)
                if not capture.isOpened():
                    LOGGER.error("Failed to reopen stream %s", runtime.source)
                    break
                continue

            start_time = time.time()
            results = self.model(frame)
            annotated_frame = results[0].plot()
            runtime.writer.save(annotated_frame)

            runtime.tracker.observe(time.time() - start_time)
            self._handle_detections(runtime, results)

        capture.release()
        LOGGER.info("Camera loop finished: %s", runtime.name)

    def _handle_detections(self, runtime: CameraRuntime, results) -> None:
        suspicious_labels = {label.lower() for label in self.settings.suspicious_labels}
        alias_map = {key.lower(): value for key, value in self.settings.label_aliases.items()}
        for prediction in results:
            boxes = getattr(prediction, "boxes", None)
            if boxes is None:
                continue

            names = prediction.names
            for box in boxes:
                cls_index = int(box.cls)
                label = names.get(cls_index, str(cls_index)) if isinstance(names, dict) else names[cls_index]
                label_norm = label.lower()
                confidence = float(box.conf)

                mapped_label = alias_map.get(label_norm, label)
                if mapped_label.lower() in suspicious_labels:
                    alert = build_alert_from_detection(
                        camera=runtime.name,
                        label=mapped_label,
                        confidence=confidence,
                        frame_path=str(runtime.writer.output_path),
                    )
                    self.alert_store.add(alert)
                    send_alert(alert)
                    self._log_detection(alert)

        average_fps = runtime.tracker.average_fps()
        update_metrics_state(self.settings.metrics_path, fps=average_fps)

    def _log_detection(self, alert: Alert) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp))
        label_display = alert.label.title() if isinstance(alert.label, str) else str(alert.label)
        LOGGER.warning(
            "🚨 %s detectada — %.0f%% — %s — %s",
            label_display,
            alert.confidence * 100,
            alert.camera,
            timestamp,
        )


def main() -> None:
    service = LiveYOLOService()
    service.run_forever()


if __name__ == "__main__":
    main()
