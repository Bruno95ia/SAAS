# src/saas/batch_yolo_extract.py
from __future__ import annotations
from pathlib import Path
import argparse
from saas.pose_yolo import save_npz as save_yolo_npz
from saas.pose_features_yolo import process_npz as feats_from_npz

def main():
    ap = argparse.ArgumentParser(description="Extração em lote: YOLOv8-pose -> keypoints -> features")
    ap.add_argument("-i","--input", type=Path, default=Path("runs/clips"), help="pasta base dos vídeos")
    ap.add_argument("--pattern", default="*.mp4", help="glob de busca (ex: *.mp4)")
    ap.add_argument("--weights", default="yolov8n-pose.pt", help="peso YOLOv8-pose")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out-yolo", type=Path, default=Path("runs/yolo_feats"))
    ap.add_argument("--out-feats", type=Path, default=Path("runs/feats"))
    args = ap.parse_args()

    vids = sorted(args.input.rglob(args.pattern))
    if not vids:
        print("[info] nenhum vídeo encontrado.")
        return

    args.out_yolo.mkdir(parents=True, exist_ok=True)
    args.out_feats.mkdir(parents=True, exist_ok=True)

    # 1) keypoints + bboxes (YOLOv8-pose)
    for v in vids:
        save_yolo_npz(str(v), out_dir=str(args.out_yolo), weights=args.weights, imgsz=args.imgsz)

    # 2) features T×5
    for p in sorted(args.out_yolo.glob("*.npz")):
        feats_from_npz(p, args.out_feats)

if __name__ == "__main__":
    main()