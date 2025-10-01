import torch
from ultralytics import YOLO

class PoseEstimator:
    def __init__(self, model_path, conf=0.5, device="cpu"):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        self.device = device

    def keypoints(self, frame):
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)

        if not results:
            return None

        r = results[0]

        # --- PROTEÇÃO: nenhum bounding box encontrado ---
        if r.boxes is None or r.boxes.conf is None or r.boxes.conf.numel() == 0:
            return None

        idx = int(r.boxes.conf.argmax().item())

        # --- PROTEÇÃO: se não houver keypoints válidos ---
        if r.keypoints is None or r.keypoints.data.numel() == 0:
            return None

        kps = r.keypoints[idx].data.cpu().numpy()
        return kps
    