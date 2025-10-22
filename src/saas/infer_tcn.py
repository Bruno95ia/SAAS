"""Inferência leve com modelos TCN exportados em ONNX."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

from saas.utils.logger import get_logger

LOGGER = get_logger("saas.tcn")


class TCNInfer:
    """Mantém uma janela deslizante de features para inferência online."""

    def __init__(self, onnx_path: str, providers: Optional[list[str]] = None) -> None:
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"Modelo TCN não encontrado: {onnx_path}")
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(str(path), providers=providers)
        except Exception:  # pragma: no cover - onnxruntime detalha internamente
            LOGGER.exception("Falha ao carregar modelo TCN em %s", path)
            raise
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.window: Optional[np.ndarray] = None

    def warmup(self, T: int, F: int) -> None:
        self.window = np.zeros((1, T, F), dtype=np.float32)

    def push_and_score(self, feat_t: np.ndarray) -> np.ndarray:
        if self.window is None:
            raise RuntimeError("TCNInfer.warmup precisa ser chamado antes da inferência")
        if feat_t.shape[-1] != self.window.shape[-1]:
            raise ValueError("Dimensão de feature incompatível com a janela configurada")

        self.window = np.roll(self.window, -1, axis=1)
        self.window[0, -1, :] = feat_t
        logits = self.session.run([self.output_name], {self.input_name: self.window})[0]
        return logits[0]
