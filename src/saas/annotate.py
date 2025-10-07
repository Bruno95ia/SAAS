from __future__ import annotations
import json, math
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

def _color(label: str) -> tuple[int,int,int]:
    h = (hash(label) % 180)
    c = np.uint8([[[h, 200, 255]]])  # HSV
    return tuple(int(x) for x in cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0,0].tolist())

def annotate_clip(
    src_path: str,
    dst_path: Optional[str] = None,
    events: Optional[List[Dict[str, Any]]] = None,
    events_json: Optional[str] = None,
    thickness: int = 2,
    font_scale: float = 0.6,
    fps_override: Optional[float] = None,
) -> str:
    """
    Desenha caixas no vídeo com base nos eventos:
      - t0, t1 em segundos relativos ao CLIP
      - bbox = [x, y, w, h] relativos (0..1)
      - label (opcional), score (opcional)
    Retorna o caminho do MP4 anotado.
    """
    if dst_path is None:
        p = Path(src_path)
        dst_path = str(p.with_name(p.stem + "_annot").with_suffix(".mp4"))

    if events is None and events_json:
        with open(events_json, "r") as f:
            events = json.load(f)
    events = events or []

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {src_path}")

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Não foi possível criar: {dst_path}")

    # indexa eventos por frame
    frame_events: Dict[int, List[Dict[str, Any]]] = {}
    for ev in events:
        t0 = float(ev.get("t0", 0.0))
        t1 = float(ev.get("t1", t0))
        label = str(ev.get("label", "event"))
        score = float(ev.get("score", 0.0))
        x,y,wr,hr = ev["bbox"]  # relativos
        f0 = max(0, int(math.floor(t0 * fps)))
        f1 = max(f0, int(math.ceil (t1 * fps)))
        for fi in range(f0, f1 + 1):
            frame_events.setdefault(fi, []).append({"bbox_rel": (x,y,wr,hr), "label": label, "score": score})

    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for ev in frame_events.get(fi, []):
            x,y,wr,hr = ev["bbox_rel"]
            x1 = int(x * w); y1 = int(y * h)
            x2 = int((x+wr) * w); y2 = int((y+hr) * h)
            color = _color(ev["label"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
            text = ev["label"] + (f" {ev['score']:.2f}" if ev["score"] > 0 else "")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness-1))
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, text, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), max(1, thickness-1), cv2.LINE_AA)

        out.write(frame); fi += 1

    cap.release(); out.release()
    return dst_path
