mkdir -p scripts # pyright: ignore[reportUndefinedVariable]
cat > scripts/gen_dataset_from_video.py <<'PY'
import argparse
import csv
import math
import os
import random
import shutil
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml

def natural_key(p: Path):
    # Ordenação humana por nome de arquivo
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]

def extract_frames_from_video(video_path: Path, out_dir: Path, fps_target: float, imgsz: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Falha ao abrir {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(src_fps / fps_target)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    saved = 0
    idx = 0
    basename = video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with tqdm(total=total, desc=f"{basename}", unit="f") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_interval == 0:
                if imgsz:
                    h, w = frame.shape[:2]
                    scale = imgsz / max(h, w)
                    if scale < 1.0:
                        frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                fname = f"{basename}_frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_dir / fname), frame)
                saved += 1
            idx += 1
            pbar.update(1)
    cap.release()
    return saved

def split_dataset(image_files, splits=(0.7, 0.2, 0.1), seed=42):
    assert math.isclose(sum(splits), 1.0, abs_tol=1e-6), "Splits devem somar 1.0"
    random.Random(seed).shuffle(image_files)
    n = len(image_files)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])
    train = image_files[:n_train]
    val = image_files[n_train:n_train+n_val]
    test = image_files[n_train+n_val:]
    return train, val, test

def ensure_yolo_tree(root: Path):
    for split in ["train", "val", "test"]:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

def move_with_label_stub(files, split_dir: Path):
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    for f in files:
        dst = images_dir / f.name
        shutil.move(str(f), str(dst))
        # cria .txt vazio correspondente (para ser anotado depois)
        (labels_dir / (dst.stem + ".txt")).touch()

def write_data_yaml(root: Path, class_names):
    data = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

def main():
    ap = argparse.ArgumentParser(description="Gera dataset YOLO a partir de vídeos.")
    ap.add_argument("--videos_dir", type=Path, required=True, help="Pasta com vídeos (mp4/avi/mov...).")
    ap.add_argument("--out_dir", type=Path, default=Path("dataset"), help="Saída do dataset.")
    ap.add_argument("--fps", type=float, default=5.0, help="FPS alvo para extração de frames.")
    ap.add_argument("--imgsz", type=int, default=640, help="Maior lado das imagens (resize downscale). 0=sem resize")
    ap.add_argument("--splits", type=float, nargs=3, default=(0.7, 0.2, 0.1), help="Proporções train/val/test.")
    ap.add_argument("--seed", type=int, default=42, help="Seed do embaralhamento.")
    ap.add_argument("--ext", type=str, nargs="+", default=[".mp4", ".avi", ".mov", ".mkv"], help="Extensões suportadas.")
    ap.add_argument("--classes", type=str, nargs="+", default=["normal", "fall"], help="Nomes de classes YOLO.")
    args = ap.parse_args()

    frames_tmp = args.out_dir / "_frames_tmp"
    frames_tmp.mkdir(parents=True, exist_ok=True)

    # 1) extrair frames
    video_paths = sorted(
        [p for p in args.videos_dir.iterdir() if p.suffix.lower() in [e.lower() for e in args.ext]],
        key=natural_key
    )
    if not video_paths:
        raise SystemExit(f"Nenhum vídeo com extensões {args.ext} em {args.videos_dir}")

    index_csv = args.out_dir / "index.csv"
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(index_csv, "w", newline="", encoding="utf-8") as idxf:
        writer = csv.writer(idxf)
        writer.writerow(["image_path", "video_name", "frame_index", "approx_sec"])

        for vp in video_paths:
            saved = extract_frames_from_video(
                video_path=vp,
                out_dir=frames_tmp,
                fps_target=args.fps,
                imgsz=(args.imgsz or None),
            )
            # aproximar segundo pelo índice/frame_interval ~ fps_target
            # vamos recuperar frame_index do nome salvo
            for img in sorted(frames_tmp.glob(f"{vp.stem}_frame_*.jpg"), key=natural_key):
                try:
                    idx = int(img.stem.split("_")[-1])
                except Exception:
                    idx = -1
                approx_sec = round((idx / args.fps), 2) if idx >= 0 else -1
                writer.writerow([str(img), vp.name, idx, approx_sec])               

     all_imgs = sorted(frames_tmp.glob("*.jpg"), key=natural_key)
    train, val, test = split_dataset(all_imgs, splits=tuple(args.splits), seed=args.seed)
    ensure_yolo_tree(args.out_dir)
    move_with_label_stub(train, args.out_dir / "train")
    move_with_label_stub(val, args.out_dir / "val")
    move_with_label_stub(test, args.out_dir / "test")
    # remover tmp
    shutil.rmtree(frames_tmp, ignore_errors=True)            
    write_data_yaml(args.out_dir, class_names=args.classes)

    print("\n✅ Dataset pronto em:", args.out_dir.resolve())
    print("   - data.yaml")
    print("   - train/val/test com images/ e labels/ (labels vazios para você anotar)")
    print("   - index.csv (mapa image→vídeo→timestamp)\n")
    print("👉 Próximos passos: anotar as caixas e classes (YOLO) nas pastas train/labels, val/labels, test/labels.")

if __name__ == "__main__":
    main()
