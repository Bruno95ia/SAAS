import argparse, csv, datetime

def main():
    ap = argparse.ArgumentParser(description="Marca eventos manualmente (quedas / não-quedas)")
    ap.add_argument("--out", default="groundtruth.csv")
    ap.add_argument("--camera", default="cam01")
    args = ap.parse_args()

    print("Digite 'fall' ou 'no_fall' e pressione Enter (Ctrl+C para sair).")
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        while True:
            try:
                ev = input("> ").strip()
                if ev not in ("fall","no_fall"): continue
                ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
                w.writerow([args.camera, ts, ev])
                print(f"[salvo] {ev} @ {ts}")
            except KeyboardInterrupt:
                break

if __name__=="__main__":
    main()