import numpy as np
REF_POINTS = ("left_shoulder","right_shoulder","left_hip","right_hip","nose")

def normalize_pose(kps):
    cx = (kps["left_hip"][0] + kps["right_hip"][0]) / 2.0
    cy = (kps["left_hip"][1] + kps["right_hip"][1]) / 2.0
    scale = np.hypot(kps["left_shoulder"][0]-kps["right_shoulder"][0],
                     kps["left_shoulder"][1]-kps["right_shoulder"][1]) + 1e-6
    feats = []
    for name in REF_POINTS:
        x,y,_ = kps[name]
        feats.extend([(x - cx)/scale, (y - cy)/scale])
    return np.array(feats, dtype=np.float32)

class TemporalWindow:
    def __init__(self, T=24):
        self.T = T; self.buf = []
    def push(self, feat):
        self.buf.append(feat)
        if len(self.buf) > self.T: self.buf.pop(0)
    def ready(self): return len(self.buf) == self.T
    def get(self):   return np.stack(self.buf, axis=0)
