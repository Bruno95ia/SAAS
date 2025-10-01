import cv2, time

class VideoSource:
    def __init__(self, cfg):
        self.uri = cfg["source"]
        if str(self.uri).isdigit():
            self.uri = int(self.uri)
        self.target_dt = 1.0 / cfg.get("target_fps", 12)
        self.reconnect_delay = cfg.get("reconnect_delay_s", 3)
        self.cap = None
        self._open()

    def _open(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"Falha ao abrir fonte: {self.uri}")

    def frames(self):
        while True:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(self.reconnect_delay); self._open(); continue
            yield frame, t0
            dt = time.time() - t0
            if dt < self.target_dt:
                time.sleep(self.target_dt - dt)
