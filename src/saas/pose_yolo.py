# src/saas/pose_yolo.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2
from ultralytics import YOLO

# COCO 17 keypoints: x,y,conf por ponto
# Vamos salvar: keypoints (T,17,3), bbox (T,4) em pixels, conf (T,)
# Se múltiplas pessoas, pegamos a "melhor" por frame (maior conf * área)

def _best_person(result) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    # result: ultralytics result (single image)
    if result.boxes is None or result.keypoints is None: 
        return None
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else None
    bconf = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else None
    kpts = result.keypoints.data.cpu().numpy() if result.keypoints is not None else None  # (N,17,3)
    cls  = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else None
    if boxes is None or kpts is None or cls is None: 
        return None
    idxs = np.where(cls == 0)[0]  # classe 0 = person
    if len(idxs) == 0: 
        return None
    # score = conf * área (preferir pessoa grande e confiante)
    scores = []
    for i in idxs:
        x1,y1,x2,y2 = boxes[i]
        area = max(1.0, (x2-x1)*(y2-y1))
        c = float(bconf[i]) if bconf is not None else 0.5
        scores.append((i, c*area))
    best_i = sorted(scores, key=lambda t: t[1], reverse=True)[0][0]
    return kpts[best_i], boxes[best_i], float(bconf[best_i]) if bconf is not None else 0.5

def extract_poses_yolo(
    video_path: str, 
    weights: str = "yolov8n-pose.pt", 
    imgsz: int = 640
) -> Dict[str, Any]:
    model = YOLO(weights)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu vídeo: {video_path}")
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    k_all, b_all, c_all = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(frame, imgsz=imgsz, conf=0.25, verbose=False)[0]
        got = _best_person(res)
        if got is None:
            k = np.zeros((17,3), dtype=np.float32)
            b = np.zeros((4,), dtype=np.float32)
            c = 0.0
        else:
            k, b, c = got
            k = k.astype(np.float32)  # (17,3) com x,y em px e conf por ponto
            b = b.astype(np.float32)  # (x1,y1,x2,y2) px
        k_all.append(k); b_all.append(b); c_all.append(c)
    cap.release()
    keypoints = np.stack(k_all, axis=0) if k_all else np.zeros((1,17,3), dtype=np.float32)
    bboxes    = np.stack(b_all, axis=0) if b_all else np.zeros((1,4), dtype=np.float32)
    conf      = np.array(c_all, dtype=np.float32) if c_all else np.zeros((1,), dtype=np.float32)
    return {"keypoints": keypoints, "bboxes": bboxes, "conf": conf}

def save_npz(video_path: str, out_dir: str = "runs/yolo_feats", weights="yolov8n-pose.pt", imgsz=640):
    outd = Path(out_dir); outd.mkdir(parents=True, exist_ok=True)
    data = extract_poses_yolo(video_path, weights=weights, imgsz=imgsz)
    out = outd / (Path(video_path).stem + ".npz")
    np.savez_compressed(out, **data)
    print("saved:", out)
    