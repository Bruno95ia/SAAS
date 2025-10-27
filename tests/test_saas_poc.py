from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient
from ultralytics import YOLO

from saas.api import app
from saas.config import get_settings
from saas.utils import AlertStore


settings = get_settings()
client = TestClient(app)
alert_store = AlertStore(settings.alerts_db_path)


def append_log(message: str) -> None:
    timestamp = datetime.utcnow().isoformat()
    with settings.tests_log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def extract_labels(results) -> list[str]:
    labels: list[str] = []
    aliases = {key.lower(): value.lower() for key, value in settings.label_aliases.items()}
    for prediction in results:
        boxes = getattr(prediction, "boxes", None)
        if boxes is None:
            continue
        names = prediction.names
        for box in boxes:
            cls_idx = int(box.cls)
            label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else names[cls_idx]
            label_norm = label.lower()
            labels.append(aliases.get(label_norm, label_norm))
    return labels


def test_yolo_inference_contains_fall_label():
    model = YOLO(str(settings.weights_path))
    results = model(str(settings.cameras[0]))
    labels = extract_labels(results)
    append_log(f"Inferência executada com labels: {labels}")

    suspicious = {label.lower() for label in settings.suspicious_labels}
    assert any(label in suspicious for label in labels), "Nenhuma detecção de queda encontrada"


def test_api_endpoints():
    alert_store.clear()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    headers = {"X-API-Key": settings.api_key}
    response = client.get("/alerts", headers=headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

    append_log("API endpoints verificados com sucesso")
