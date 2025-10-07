# src/saas/live_yolo.py
from __future__ import annotations
import os, math, time, argparse
import csv, datetime as dt
from pathlib import Path
import numpy as np
import cv2
import requests
from ultralytics import YOLO
from saas.infer_tcn import TCNInfer   # para rodar o ONNX do TCN
from saas.pose_features_yolo import features_from_kpts
from saas.clipper_rtsp import collect_clip
from saas.annotate import annotate_clip
from saas.pose_yolo import _best_person  # reaproveita seletor de "melhor pessoa"

def trunk_angle(k: np.ndarray) -> float:
    # COCO order em Ultralytics: L_SHO=5, R_SHO=6, L_HIP=11, R_HIP=12
    L_SHO, R_SHO, L_HIP, R_HIP = 5,6,11,12
    if k[L_SHO,2]<0.2 or k[R_SHO,2]<0.2 or k[L_HIP,2]<0.2 or k[R_HIP,2]<0.2:
        return 0.0
    sho = (k[L_SHO,:2] + k[R_SHO,:2]) / 2.0
    hip = (k[L_HIP,:2] + k[R_HIP,:2]) / 2.0
    v = sho - hip
    # 0 rad = em pé; ~pi/2 = deitado
    return math.atan2(abs(v[0]), max(1.0, abs(v[1])))

def vy_norm(prev_hip_y: float|None, hip_y: float, bbox_h: float) -> float:
    if prev_hip_y is None or bbox_h <= 1: return 0.0
    return (hip_y - prev_hip_y) / bbox_h

def run(args):
    api_url = args.api_url.rstrip("/")
    api_key = args.api_key
    cam_id  = args.camera
    buffer_dir = args.buffer
    tcn = None
window = []
if args.use_tcn:
    tcn = TCNInfer(args.tcn_path)
    tcn.warmup(T=args.tcn_window, F=5)  # 5 features: ratio, trunk_angle, vy, knee_angle, step_len
    print(f"[live] TCN carregado de {args.tcn_path}, janela {args.tcn_window}")

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(0 if args.rtsp.lower() == "webcam" else args.rtsp)
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu fonte: {args.rtsp}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    prev_hip = None
    flat_since = None
    last_alert = 0.0

    theta_rot = math.radians(args.theta_deg)  # conversão p/ rad
    CONF_MIN  = args.conf

    print(f"[live] camera={cam_id} src={args.rtsp} theta={args.theta_deg} vy_min={args.vy_min} flat_sec={args.flat_sec}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue

        res = model.predict(frame, imgsz=args.imgsz, conf=CONF_MIN, verbose=False)[0]
        best = _best_person(res)
        H,W = frame.shape[:2]

        if best is None:
            flat_since = None
            prev_hip = None
            cv2.imshow(cam_id, frame)
            if cv2.waitKey(1) & 0xFF == 27: break
            continue

        kpts, box, bconf = best
        x1,y1,x2,y2 = box.astype(int)
        # debug visual
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

        # features
        bbox_h = max(1.0, y2-y1)
        L_HIP, R_HIP = 11,12
        hip = (kpts[L_HIP,1] + kpts[R_HIP,1]) / 2.0 if kpts[L_HIP,2]>0.2 and kpts[R_HIP,2]>0.2 else (y1+y2)/2.0
        ang = trunk_angle(kpts)
        vy  = vy_norm(prev_hip, hip, bbox_h)
        prev_hip = hip

        # regras
        probable = (ang > theta_rot and vy > args.vy_min)
        flat = (bbox_h / max(1.0, x2-x1)) < args.flat_ratio

        if probable and flat_since is None:
            flat_since = time.time()
        if not flat:
            flat_since = None

        confirmed = flat_since is not None and (time.time() - flat_since) >= args.flat_sec

        # hud
        color = (0,0,255) if confirmed else (0,255,0)
        cv2.putText(frame, f"ang={ang*180/math.pi:.0f} vy={vy:.2f} flat={flat}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if confirmed and (time.time() - last_alert) > args.debounce:
            last_alert = time.time()
            try:
                # 1) gera clipe [-pre,+post] do ring buffer
                import datetime as dt
                t_utc = dt.datetime.now(dt.timezone.utc)
                local_path, rel_url = collect_clip(buffer_dir, cam_id, t_utc, pre=args.pre, post=args.post)

                # 2) anota (bbox representativa ao longo do clipe)
                ev_bbox = [x1/W, y1/H, (x2-x1)/W, (y2-y1)/H]
                events = [{"t0": 0.0, "t1": args.pre+args.post, "bbox": ev_bbox, "label":"fall", "score": float(bconf)}]
                annotated = annotate_clip(local_path, events=events)
                clip_url = f"{api_url}/clips/{Path(annotated).name}"

                # 3) POST alerta
                r = requests.post(f"{api_url}/alerts",
                    headers={"X-API-Key": api_key, "Content-Type":"application/json"},
                    json={"camera_id": cam_id, "type":"fall", "score": float(bconf),
                          "clip_path": clip_url,
                          "extra":{"source":"yolov8", "angle_deg": ang*180/math.pi, "vy_norm": vy}},
                    timeout=5)
                r.raise_for_status()
                print("[ALERTA] publicado:", clip_url)
            except Exception as e:
                print("[ERRO] gerar/publicar clipe:", e)

        cv2.imshow(cam_id, frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()

def build_argparser():
    ap = argparse.ArgumentParser(description="YOLOv8-pose live fall-detection (baseline sem treino)")
    ap.add_argument("--camera", default="cam01", help="ID lógico da câmera")
    ap.add_argument("--rtsp",   required=True, help='"webcam" ou URL RTSP')
    ap.add_argument("--buffer", required=True, help="pasta do ring buffer: runs/buffer/<camera>")
    ap.add_argument("--weights", default="yolov8n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--log-detections", default="runs/logs/detections.csv",
                help="CSV onde salvar timestamp, camera_id, tcn_score e features")
    # regras
    ap.add_argument("--theta-deg", type=float, default=55.0, help="limiar rotação do tronco (graus)")
    ap.add_argument("--vy-min",    type=float, default=0.25, help="velocidade vertical normalizada mínima")
    ap.add_argument("--flat-ratio",type=float, default=0.60, help="H/W < flat_ratio => deitado")
    ap.add_argument("--flat-sec",  type=float, default=2.0,  help="persistência deitado para confirmar (s)")
    ap.add_argument("--pre",  type=float, default=5.0, help="segundos antes do evento no clipe")
    ap.add_argument("--post", type=float, default=5.0, help="segundos após o evento no clipe")
    ap.add_argument("--debounce", type=float, default=10.0, help="mínimo entre alertas (s)")
    # API
    ap.add_argument("--api-url", default=os.getenv("SAAS_API_URL","http://127.0.0.1:8000"))
    ap.add_argument("--api-key", default=os.getenv("SAAS_API_KEY","minha-chave-forte"))
    ap.add_argument("--use-tcn", action="store_true", help="Usa o TCN treinado (ONNX) em tempo real")
    ap.add_argument("--tcn-path", default="runs/models/tcn.onnx")
    ap.add_argument("--tcn-window", type=int, default=32, help="tamanho da janela temporal")
    return ap

if __name__ == "__main__":
    run(build_argparser().parse_args())