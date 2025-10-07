import os
import cv2
import yaml
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

from inference.pose import PoseEstimator  # garante None quando não houver detecção

# ================================
# Logger
# ================================
logging.basicConfig(
    filename="falls.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("app")

# ================================
# Utilidades
# ================================
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def alert_fall(prob, snapshot=None, video=None, frame_idx=None):
    payload = {
        "event": "fall_alert",
        "prob": float(prob),
        "snapshot": snapshot,
        "video": video,
        "frame_idx": int(frame_idx) if frame_idx is not None else None,
    }
    print(json.dumps(payload, ensure_ascii=False))
    logger.info(f"ALERTA DE QUEDA: prob={prob:.3f} video={video} frame={frame_idx} snap={snapshot}")

def save_snapshot(frame, video_path, frame_idx):
    stem = Path(video_path).stem
    snap_dir = ensure_dir(f"runs/detect/snapshots/{stem}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"{snap_dir}/{ts}_frame{frame_idx:06d}.jpg"
    cv2.imwrite(out_path, frame)
    return out_path

# pares de conexões para desenhar o esqueleto (COCO-like 17kps — ajuste conforme seu modelo)
SKELETON = [
    (5, 7), (7, 9),       # braço esquerdo
    (6, 8), (8,10),       # braço direito
    (11,13), (13,15),     # perna esquerda
    (12,14), (14,16),     # perna direita
    (5,6), (5,11), (6,12),# tronco
    (11,12), (5,12), (6,11)
]

def draw_pose_and_box(frame, kps):
    """
    kps: array [1, num_kps, 3] ou [num_kps, 3] com (x,y,score)
    Desenha círculos, esqueleto e uma caixa envolvendo os pontos válidos.
    Retorna (x1,y1,x2,y2) ou None se não der para calcular.
    """
    if kps is None:
        return None
    pts = kps
    if pts.ndim == 3:
        pts = pts[0]  # [num_kps, 3]
    xs, ys, ss = pts[:,0], pts[:,1], pts[:,2]
    valid = ss > 0
    if not np.any(valid):
        return None

    # desenha pontos
    for (x, y, s) in pts:
        if s > 0:
            cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)

    # desenha esqueleto
    nk = pts.shape[0]
    for a,b in SKELETON:
        if a < nk and b < nk and ss[a] > 0 and ss[b] > 0:
            ax,ay = int(pts[a,0]), int(pts[a,1])
            bx,by = int(pts[b,0]), int(pts[b,1])
            cv2.line(frame, (ax,ay), (bx,by), (0,255,255), 2)

    # caixa
    x1, y1 = int(np.min(xs[valid])), int(np.min(ys[valid]))
    x2, y2 = int(np.max(xs[valid])), int(np.max(ys[valid]))
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    return (x1,y1,x2,y2)

# ================================
# Processamento de 1 vídeo (gera anotado)
# ================================
def process_video(cfg, video_path, pose):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"[ERRO] Não foi possível abrir {video_path}"
        print(msg)
        logger.error(msg)
        return

    in_fps = cap.get(cv2.CAP_PROP_FPS)
    if not in_fps or np.isnan(in_fps) or in_fps <= 1:
        in_fps = float(cfg["video"].get("target_fps", 12))
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 640)
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    if W <= 0 or H <= 0:
        W, H = 640, 480

    # writer para salvar anotado
    out_dir = ensure_dir("runs/detect/annotated")
    stem = Path(video_path).stem
    out_path = f"{out_dir}/{stem}_annot.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # compatível com macOS
    writer = cv2.VideoWriter(out_path, fourcc, in_fps, (W, H))
    if not writer.isOpened():
        fourcc2 = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc2, in_fps, (W, H))

    # histerese simples (config.yaml)
    min_frames   = int(cfg["logic"]["min_consecutive_frames"])
    clear_frames = int(cfg["logic"]["min_clear_frames"])
    cooldown     = float(cfg["logic"]["alert_cooldown_s"])

    fall_counter = 0
    clear_counter = 0
    alerted = False
    last_alert_time = 0.0
    frame_idx = 0

    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1) keypoints do YOLO-pose
        kps = pose.keypoints(frame)
        if kps is None:
            cv2.putText(frame, "sem pessoa", (10, H-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
            writer.write(frame)
            continue

        # 2) desenha esqueleto + caixa
        _ = draw_pose_and_box(frame, kps)

        # 3) heurística simples de “queda” (exemplo)
        try:
            hip_y  = kps[0, 2, 1]
            knee_y = kps[0, 3, 1]
            fallen = hip_y > knee_y
        except Exception:
            fallen = False

        # 4) histerese N/M
        if fallen:
            fall_counter += 1
            clear_counter = 0
        else:
            clear_counter += 1
            fall_counter = 0

        # 5) confirmar/limpar estado
        if (fall_counter >= min_frames) and (not alerted):
            now = time.time()
            if now - last_alert_time >= cooldown:
                snap_path = save_snapshot(frame, video_path, frame_idx)
                alert_fall(prob=0.90, snapshot=snap_path, video=str(video_path), frame_idx=frame_idx)
                alerted = True
                last_alert_time = now

        if alerted and clear_counter >= clear_frames:
            alerted = False

        # 6) overlay de status
        if alerted:
            cv2.putText(frame, "QUEDA CONFIRMADA", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        # 7) grava frame anotado
        writer.write(frame)

    cap.release()
    writer.release()
    dt = time.time() - t0
    fps_eff = frame_idx / dt if dt > 0 else 0.0
    print(f"[INFO] Vídeo anotado salvo em: {out_path} | frames={frame_idx} | fps~{fps_eff:.1f}")

# ================================
# Main: processa arquivo ou todos os .mp4 do diretório
# ================================
def main(cfg):
    src = cfg["video"]["source"]

    # PoseEstimator
    device = cfg["pose"].get("device", "cpu")
    pose = PoseEstimator(
        model_path=cfg["pose"]["model"],
        conf=cfg["pose"]["conf"],
        device=device,
    )

    videos = []
    if os.path.isdir(src):
        videos = [os.path.join(src, f) for f in sorted(os.listdir(src))
                  if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        print(f"Carregando {len(videos)} vídeos de {src}")
    else:
        videos = [src]
        print(f"Processando arquivo único: {src}")

    if not videos:
        raise RuntimeError(f"Nenhum vídeo encontrado em '{src}'.")

    for vid in videos:
        print(f"Processando {vid}")
        try:
            process_video(cfg, vid, pose)
        except Exception as e:
            logger.exception(f"Falha ao processar {vid}: {e}")
            print(f"[ERRO] Falha ao processar {vid}: {e}")

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    main(cfg)
    