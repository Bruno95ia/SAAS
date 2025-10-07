# src/saas/infer_tcn.py
from __future__ import annotations
import onnxruntime as ort
import numpy as np
from pathlib import Path

class TCNInfer:
    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name= self.sess.get_outputs()[0].name
        self.window = None

    def warmup(self, T:int, F:int):
        self.window = np.zeros((1,T,F), dtype=np.float32)

    def push_and_score(self, feat_t: np.ndarray) -> np.ndarray:
        """
        feat_t: (F,) -> desliza numa janela (1,T,F) e retorna logits (1,C)
        """
        assert self.window is not None
        self.window = np.roll(self.window, -1, axis=1)
        self.window[0,-1,:] = feat_t
        logits = self.sess.run([self.output_name], {self.input_name: self.window})[0]  # (1,C)
        return logits[0]