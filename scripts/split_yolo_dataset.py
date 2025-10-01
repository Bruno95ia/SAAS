import argparse, random, shutil
from pathlib import Path
import yaml

def main(src, train=0.8, val=0.1, test=0.1, seed=42):
    random.seed(seed)
    src = Path(src)
    assert (src / "images").exists() and (src / "labels").exists(), "Esperava pastas images/ e labels/"

    imgs = sorted([p for p in (src / "images").glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}])
    base = [p.stem for p in imgs]
    random.shuffle(base)

    n = len(base)
    n_train = int(n*train)
    n_val   = int(n*val)
    n_test  = n - n_train - n_val
    splits = {
        "train": base[:n_train],
        "val":   base[n_train:n_train+n_val],
        "test":  base[n_train+n_val:]
    }

    for split, names in splits.items():
        (src / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (src / f"labels/{split}").mkdir(parents=True, exist_ok=True)
        for stem in names:
            si = src / "images" / f"{stem}.jpg"
            if not si.exists():
                for ext in [".png",".jpeg",".bmp",".tif",".tiff"]:
                    alt = src / "images" / f"{stem}{ext}"
                    if alt.exists():
                        si = alt; break
            sl = src / "labels" / f"{stem}.txt"
            if si.exists(): shutil.move(str(si), src / f"images/{split}/{si.name}")
            if sl.exists(): shutil.move(str(sl), src / f"labels/{split}/{sl.name}")

    # atualizar data.yaml (mantém nc/names se já existirem)
    data_yaml = src / "data.yaml"
    if data_yaml.exists():
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    data["path"]  = str(src)
    data["train"] = "images/train"
    data["val"]   = "images/val"
    data["test"]  = "images/test"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print(f"✅ Split feito: {n_train} train, {n_val} val, {n_test} test")
    print(f"   Atualizado: {data_yaml}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Pasta do dataset YOLO (com images/ e labels/)")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.1)
    ap.add_argument("--seed",  type=int,   default=42)
    args = ap.parse_args()
    main(args.src, args.train, args.val, args.test, args.seed)
