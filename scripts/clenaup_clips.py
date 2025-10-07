import argparse
from pathlib import Path
import time, os

def main():
    ap = argparse.ArgumentParser(description="Remove clipes antigos")
    ap.add_argument("--dir", default="runs/clips")
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    cutoff = time.time() - args.days*86400
    removed=0
    for f in Path(args.dir).rglob("*.mp4"):
        if f.stat().st_mtime < cutoff:
            f.unlink(); removed+=1
    print(f"removidos {removed} clipes antigos (> {args.days} dias)")

if __name__=="__main__":
    main()