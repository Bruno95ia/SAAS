import time, requests
from ioutils.storage import save_snapshot, purge_old
from privacy.blur import maybe_blur

class AlertDispatcher:
    def __init__(self, cfg_alerts, cfg_privacy):
        self.enabled = bool(cfg_alerts.get("enabled", True))
        self.url = cfg_alerts.get("webhook_url")
        self.min_interval = int(cfg_alerts.get("min_interval_s", 30))
        self.last_ts = 0
        self.priv = cfg_privacy

    def maybe_send(self, frame_bgr, prob: float, clip_path: str | None = None):
        if not self.enabled or not self.url:
            return None
        now = time.time()
        if (now - self.last_ts) < self.min_interval:
            return None  # debounce
        # snapshot com blur
        out = maybe_blur(frame_bgr, enabled=bool(self.priv.get("face_blur", True)))
        path = save_snapshot(out, outdir="alerts", prob=prob)
        # retenção
        purge_old(outdir="alerts", retain_hours=self.priv.get("retain_hours", 48))
        # webhook
        payload = {"type":"fall", "ts": int(now), "prob": prob, "snapshot": path, "clip": clip_path}
        try:
            requests.post(self.url, json=payload, timeout=4)
        except Exception:
            pass
        self.last_ts = now
        return path
