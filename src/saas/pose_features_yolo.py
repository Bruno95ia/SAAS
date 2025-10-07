# src/saas/pose_features_yolo.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

# COCO-17 indices (Ultralytics): https://docs.ultralytics.com
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SHO=5; R_SHO=6; L_ELB=7; R_ELB=8; L_WR=9; R_WR=10
L_HIP=11; R_HIP=12; L_KNEE=13; R_KNEE=14; L_ANK=15; R_ANK=16

def _normalize_xy(kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    kpts: (T,17,3) com coords em pixels (x,y,conf)
    retorna: xy_norm (T,17,2) normalizado por ombro-ombro e centrado no quadril médio,
             vis (T,17) booleano
    """
    vis = kpts[...,2] > 0.35
    # centros
    hip = 0.5*(kpts[:,L_HIP,:2] + kpts[:,R_HIP,:2])
    # escala: distância ombro-ombro
    shoulder_dist = np.linalg.norm(kpts[:,L_SHO,:2] - kpts[:,R_SHO,:2], axis=1) + 1e-6
    xy = kpts[:,:,:2] - hip[:,None,:]
    xy /= shoulder_dist[:,None,None]
    return xy.astype(np.float32), vis.astype(np.float32)

def _angle(a, b, c):  # ângulo ABC
    ba = a-b; bc = c-b
    den = (np.linalg.norm(ba,axis=-1)*np.linalg.norm(bc,axis=-1) + 1e-6)
    cos = (ba*bc).sum(axis=-1) / den
    return np.arccos(np.clip(cos, -1.0, 1.0))

def features_from_kpts(kpts: np.ndarray, bboxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    kpts: (T,17,3), bboxes: (T,4) em pixels (x1,y1,x2,y2)
    retorna:
      X: (T,F) com features [ratio, trunk_angle, hip_vy, knee_angle, step_len]
      mask: (T,) frames válidos
    """
    T = kpts.shape[0]
    xy, vis = _normalize_xy(kpts)     # (T,17,2), (T,17)
    # torso/width ratio
    torso = np.linalg.norm(xy[:,L_SHO,:] - xy[:,L_HIP,:], axis=1)           # (T,)
    width = np.linalg.norm(xy[:,L_SHO,:] - xy[:,R_SHO,:], axis=1) + 1e-6
    ratio = (torso/width)[:,None]
    # ângulo do tronco vs. vertical
    trunk = xy[:,L_SHO,:] - xy[:,L_HIP,:]
    trunk_angle = np.arctan2(np.abs(trunk[:,0]), np.maximum(1e-6, np.abs(trunk[:,1])) )[:,None]  # 0=em pé, ~pi/2=deitado
    # velocidade vertical do quadril em pixels relativos
    hip_y = xy[:,L_HIP,1]
    vy = np.concatenate([[0.0], np.diff(hip_y)])[:,None]
    # joelho (curvatura perna)
    knee_ang = _angle(xy[:,L_HIP,:], xy[:,L_KNEE,:], xy[:,L_ANK,:])[:,None]
    # comprimento de passo (distância entre tornozelos)
    step_len = np.linalg.norm(xy[:,L_ANK,:] - xy[:,R_ANK,:], axis=1)[:,None]

    X = np.concatenate([ratio, trunk_angle, vy, knee_ang, step_len], axis=1).astype(np.float32)  # (T,5)
    # máscara de visibilidade: média de visibilidade acima de 0.2
    mask = (vis.mean(axis=1) > 0.2).astype(np.float32)
    return X, mask

def process_npz(in_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(in_path)
    kpts = d["keypoints"].astype(np.float32)  # (T,17,3)
    bboxes = d["bboxes"].astype(np.float32)   # (T,4)
    X, mask = features_from_kpts(kpts, bboxes)
    out = out_dir / (in_path.stem + ".npz")
    np.savez_compressed(out, X=X, mask=mask)
    print("saved:", out)

def main():
    ap = argparse.ArgumentParser(description="YOLOv8-pose → features temporais")
    ap.add_argument("-i","--input", type=Path, default=Path("runs/yolo_feats"))
    ap.add_argument("-o","--out",   type=Path, default=Path("runs/feats"))
    args = ap.parse_args()
    for p in sorted(args.input.glob("*.npz")):
        process_npz(p, args.out)

if __name__ == "__main__":
    main()