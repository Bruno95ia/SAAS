from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from .config import get_settings

logger = logging.getLogger(__name__)


settings = get_settings()


@dataclass(slots=True)
class Detection:
    label: str
    score: float
    bbox: Tuple[int, int, int, int]


class FallDetector:
    def __init__(self) -> None:
        model_path = settings.detection.model_path
        logger.info("Loading YOLO model from %s", model_path)
        self.model = YOLO(model_path)
        if settings.detection.device:
            self.model.to(settings.detection.device)
        self.confidence = settings.detection.confidence
        self.iou = settings.detection.iou

    def infer(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())
                if score < self.confidence:
                    continue
                label = self.model.model.names.get(cls_id, str(cls_id))  # type: ignore[attr-defined]
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                detections.append(
                    Detection(
                        label=label,
                        score=score,
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    )
                )
        return detections

    def detect_fall(self, detections: Iterable[Detection]) -> Optional[Detection]:
        person_boxes = [d for d in detections if d.label.lower() in {"person", "fall"}]
        if not person_boxes:
            return None
        # simple heuristic: detect if bounding box is wider than tall
        for det in person_boxes:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            if width > height * 1.2 or det.label.lower() == "fall":
                logger.debug("Fall detected with bbox %s and score %.2f", det.bbox, det.score)
                return Detection(label="fall", score=det.score, bbox=det.bbox)
        return None


fall_detector = FallDetector()
