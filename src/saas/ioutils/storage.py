import os, time, uuid, cv2, glob
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_snapshot(frame, outdir="alerts", prob=0.0):
    ensure_dir(outdir)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fid = uuid.uuid4().hex[:6]
    path = os.path.join(outdir, f"{ts}_{fid}_p{int(prob*100)}.jpg")
    cv2.imwrite(path, frame)
    return path

def purge_old(outdir="alerts", retain_hours=48):
    if retain_hours is None: return
    ensure_dir(outdir)
    cutoff = time.time() - retain_hours*3600
    for p in glob.glob(os.path.join(outdir, "*.jpg")):
        try:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
        except Exception:
            pass
