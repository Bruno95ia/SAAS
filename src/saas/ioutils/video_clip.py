import os, time, uuid, cv2

def write_mp4(frames, fps: float, outdir="alerts", basename=None):
    if not frames:
        return None
    os.makedirs(outdir, exist_ok=True)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # compatível com macOS
    ts = time.strftime("%Y%m%d-%H%M%S")
    fid = uuid.uuid4().hex[:6]
    name = basename or f"{ts}_{fid}"
    path = os.path.join(outdir, f"{name}.mp4")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    try:
        for f in frames:
            vw.write(f)
    finally:
        vw.release()
    return path
