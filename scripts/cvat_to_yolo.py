import zipfile, shutil
from pathlib import Path
import yaml, argparse

def convert_cvat_zip(zip_path, out_dir):
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Extrai o zip exportado do CVAT
    raw = out_dir / "raw"
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(raw)

    # Pasta padrão do export YOLO 1.1 do CVAT (às vezes vem com capitalização diferente)
    candidates = [raw / "obj_train_data", raw / "obj_Train_data", raw]
    src = next((p for p in candidates if p.exists()), raw)

    images = out_dir / "images"
    labels = out_dir / "labels"
    images.mkdir(exist_ok=True)
    labels.mkdir(exist_ok=True)

    for ext in ("*.jpg","*.jpeg","*.png"):
        for p in src.glob(ext):
            shutil.move(str(p), images / p.name)
    for p in src.glob("*.txt"):
        shutil.move(str(p), labels / p.name)

    # data.yaml (ajuste de classes conforme seu projeto)
    data = {
        "path": str(out_dir.resolve()),
        "train": "images",
        "val": "images",   # por ora usamos o mesmo; depois podemos fazer split
        "test": "images",
        "nc": 2,
        "names": ["Pessoa1", "Queda"],
    }
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    print("✅ Dataset pronto em:", out_dir)
    print("   -", out_dir / "images")
    print("   -", out_dir / "labels")
    print("   -", out_dir / "data.yaml")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Arquivo .zip exportado do CVAT (YOLO 1.1)")
    ap.add_argument("--out", default="dataset", help="Pasta de saída do dataset")
    args = ap.parse_args()
    convert_cvat_zip(args.zip, args.out)
