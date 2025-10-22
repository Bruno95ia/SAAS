from collections import deque
import time

class FrameRing:
    """
    Guarda frames recentes com timestamps. Permite dump dos últimos 't' segundos.
    """
    def __init__(self, max_seconds: float, fps: float):
        self.fps = float(fps)
        self.max_seconds = float(max_seconds)
        self.maxlen = max(1, int(self.fps * self.max_seconds))
        self.buf = deque(maxlen=self.maxlen)  # cada item: (ts, frame)

    def push(self, frame, ts=None):
        self.buf.append((ts if ts is not None else time.time(), frame))

    def dump_last_seconds(self, seconds: float):
        if not self.buf:
            return []
        cutoff = (self.buf[-1][0]) - float(seconds)
        return [f for (t, f) in self.buf if t >= cutoff]
