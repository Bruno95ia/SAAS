import sqlite3, csv, argparse
from datetime import datetime, timedelta
from collections import defaultdict

def parse_time(ts):
    return datetime.fromisoformat(ts.replace("Z",""))

def load_groundtruth(path):
    gt = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            gt[row["camera_id"]].append({
                "timestamp": parse_time(row["timestamp"]),
                "event": row["event"]
            })
    return gt

def main():
    ap = argparse.ArgumentParser(description="Avaliação de calibração")
    ap.add_argument("--db", default="events.db")
    ap.add_argument("--gt", required=True, help="CSV ground truth")
    ap.add_argument("--window", type=int, default=10,
                    help="janela (s) para casar evento real com alerta")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("SELECT camera_id, created_at FROM alerts WHERE type='fall'")
    alerts = defaultdict(list)
    for cam, ts in cur.fetchall():
        alerts[cam].append(parse_time(ts))

    gt = load_groundtruth(args.gt)

    results=[]
    for cam in gt.keys():
        n_falls = sum(1 for e in gt[cam] if e["event"]=="fall")
        n_nf    = sum(1 for e in gt[cam] if e["event"]=="no_fall")
        cam_alerts = alerts.get(cam,[])
        tp=fp=fn=0; tta=[]
        for e in gt[cam]:
            if e["event"]=="fall":
                # procura alerta próximo
                matched = None
                for a in cam_alerts:
                    if abs((a-e["timestamp"]).total_seconds()) <= args.window:
                        matched = a; break
                if matched: 
                    tp+=1; tta.append((matched-e["timestamp"]).total_seconds())
                else:
                    fn+=1
            elif e["event"]=="no_fall":
                # se houver alerta próximo = falso positivo
                for a in cam_alerts:
                    if abs((a-e["timestamp"]).total_seconds()) <= args.window:
                        fp+=1; break
        sens = tp/max(1,n_falls)
        prec = tp/max(1,tp+fp)
        fprh = fp / (len(gt[cam])/3600) if len(gt[cam]) else 0
        avg_tta = sum(tta)/len(tta) if tta else None
        results.append((cam, n_falls, tp, fn, fp, sens, prec, fprh, avg_tta))

    print("camera | falls | TP | FN | FP | sens | prec | FP/h | TTA(s)")
    for r in results:
        cam, nf,tp,fn,fp,sens,prec,fprh,tta = r
        print(f"{cam:6} | {nf:5} | {tp:2} | {fn:2} | {fp:2} | {sens:.2f} | {prec:.2f} | {fprh:.2f} | {tta:.2f if tta else -1}")

if __name__=="__main__":
    main()