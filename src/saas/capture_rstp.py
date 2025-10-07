# src/saas/capture_rtsp.py
import argparse, subprocess, os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Captura RTSP → buffer de 2s (para clipes)")
    ap.add_argument("--camera", required=True)
    ap.add_argument("--rtsp", required=True)
    ap.add_argument("--out", default="runs/buffer")
    ap.add_argument("--segment", type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.out)/args.camera
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", "tcp", "-stimeout", "5000000",
        "-i", args.rtsp, "-reset_timestamps", "1",
        "-c", "copy", "-f", "segment",
        "-segment_time", str(args.segment), "-strftime", "1",
        f"{outdir}/%Y%m%d/%H%M%S.m4s"
    ]
    print(" ".join(cmd))
    subprocess.run(cmd)

if __name__=="__main__":
    main()