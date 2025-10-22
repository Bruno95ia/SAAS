from collections import deque
import numpy as np, time

class FallModel:
    def predict_prob(self, window_np: np.ndarray) -> float:
        nose_y = window_np[:, -1]
        dy = np.diff(nose_y[-5:]).sum() if len(nose_y) >= 5 else 0.0
        p = 1.0 / (1.0 + np.exp(-4.0 * dy))
        return float(np.clip(p, 0.0, 1.0))

class FallDecision:
    def __init__(self, prob_th=0.85, min_frames=6, cooldown_s=30):
        self.prob_th, self.min_frames, self.cooldown_s = prob_th, min_frames, cooldown_s
        self.buf = deque(maxlen=24); self.last = 0
    def update(self, p):
        self.buf.append(p)
        ok = sum(1 for q in self.buf if q >= self.prob_th) >= self.min_frames
        now = time.time()
        if ok and (now - self.last) > self.cooldown_s:
            self.buf.clear(); self.last = now; return True
        return False
