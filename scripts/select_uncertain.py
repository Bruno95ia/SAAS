#!/usr/bin/env python3
"""
select_uncertain.py — Active Learning helper

Vasculha vídeos/imagens e salva snapshots de frames com
probabilidades "incertas" para a classe alvo (ex.: Queda),
facilitando priorização de anotação no CVAT.

Uso:
  python scripts/select_uncertain.py \
    --model runs/detect/train/weights/best.pt \
    --source ./data \
    --out active_learning/uncertain \
    --class-name Queda \
    --low 0.40 --high 0.60 \
    --stride 3 --max-per-video 50

Requisitos: ultralytics, opencv-python, numpy
"""
import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics não encontrado. Instale com: pip install ultralytics")

IMG_EXTS = (".jpg", ".jpeg", ".png")
VID_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def is_media(p: Path) -> bool:
    s = p.suffix.lower()
    return s in IMG_EXTS + VID_EXTS


def list_sources(src: Path) -> List[Path]:
    if src.is_file():
        return [src]
    items = []
    for f in sorted(src.iterdir()):
        if f.is_file() and is_media(f):
            items.append(f)
    return items


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def overlay_text(img, text, org=(10, 30)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="caminho do .pt treinado (YOLO)")
    ap.add_argument("--source", required=True, help="arquivo único ou diretório com mídias")
    ap.add_argument("--out", default="active_learning/uncertain", help="pasta de saída")
    ap.add_argument("--class-name", default="Queda", help="nome da classe alvo no modelo YOLO")
    ap.add_argument("--low", type=float, default=0.4, help="limite inferior da zona de incerteza")
    ap.add_argument("--high", type=float, default=0.6, help="limite superior da zona de incerteza")
    ap.add_argument("--stride", type=int, default=3, help="pular N frames entre inferências de vídeo")
    ap.add_argument("--max-per-video", type=int, default=50, help="máximo de snapshots por vídeo")
    ap.add_argument("--save-json", action="store_true", help="salvar metadados JSON por vídeo")
    args = ap.parse_args()

    src = Path(args.source)
    out_root = ensure_dir(Path(args.out))

    model = YOLO(args.model)
    names = model.model.names if hasattr(model, "model") else model.names

    # mapear class index
    try:
        class_id = [k for k, v in names.items() if str(v).lower() == args.class_name.lower()][0]
    except Exception:
        raise SystemExit(f"Classe '{args.class_name}' não encontrada nas names do modelo: {names}")

    items = list_sources(src)
    if not items:
        raise SystemExit(f"Nenhuma mídia encontrada em {src}")

    print(f"Processando {len(items)} arquivos de {src}")

    for media_path in items:
        stem = media_path.stem
        out_dir = ensure_dir(out_root / stem)
        meta = {"file": str(media_path), "snapshots": []}
        saved = 0

        if media_path.suffix.lower() in IMG_EXTS:
            img = cv2.imread(str(media_path))
            if img is None:
                print(f"[AVISO] Falha ao abrir imagem: {media_path}")
                continue
            r = model.predict(img, verbose=False)[0]
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    if cls == class_id and args.low <= conf <= args.high:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        out_img = out_dir / f"{ts}_uncertain_{conf:.2f}.jpg"
                        overlay_text(img, f"{args.class_name} conf={conf:.2f}")
                        cv2.imwrite(str(out_img), img)
                        meta["snapshots"].append({"frame": None, "conf": conf, "path": str(out_img)})
                        saved += 1
            if args.save_json:
                with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            continue

        # Vídeo
        cap = cv2.VideoCapture(str(media_path))
        if not cap.isOpened():
            print(f"[AVISO] Falha ao abrir vídeo: {media_path}")
            continue
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % max(1, args.stride) != 0:
                continue

            r = model.predict(frame, verbose=False)[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                if cls == class_id and args.low <= conf <= args.high:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_img = out_dir / f"{ts}_f{frame_idx:06d}_uncertain_{conf:.2f}.jpg"
                    frame_copy = frame.copy()
                    overlay_text(frame_copy, f"{args.class_name} conf={conf:.2f} f={frame_idx}")
                    cv2.imwrite(str(out_img), frame_copy)
                    meta["snapshots"].append({"frame": int(frame_idx), "conf": conf, "path": str(out_img)})
                    saved += 1
                    if saved >= args.max_per_video:
                        break
            if saved >= args.max_per_video:
                break

        cap.release()
        if args.save_json:
            with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"{stem}: {saved} snapshots incertos salvos em {out_dir}")


if __name__ == "__main__":
    main()
