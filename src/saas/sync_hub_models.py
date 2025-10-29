"""Synchronize IASENIOR models from Ultralytics HUB."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import schedule

from .utils.iasenior_paths import LOGS_DIR, MODELS_DIR, ensure_structure
from .utils.hub_integration import HubError, UltralyticsHubManager

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("iasenior.sync")

SYNC_STATE_FILE = LOGS_DIR / "model_sync.json"
PROGRESS_FILE = LOGS_DIR / "train_progress.json"
HISTORY_FILE = LOGS_DIR / "treinos.csv"


def load_json(path: Path) -> Dict[str, Optional[str]]:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            LOGGER.warning("Arquivo %s corrompido, ignorando", path)
    return {}


def save_json(path: Path, payload: Dict[str, Optional[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def append_history(run_id: str, status: str, note: str) -> None:
    headers = [
        "timestamp",
        "dataset",
        "run_id",
        "model",
        "epochs",
        "status",
        "map50",
        "map5095",
        "precision",
        "recall",
        "loss",
        "notes",
    ]
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = HISTORY_FILE.exists()
    with HISTORY_FILE.open("a", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "dataset": "",
                "run_id": run_id,
                "model": "",
                "epochs": "",
                "status": status,
                "map50": "",
                "map5095": "",
                "precision": "",
                "recall": "",
                "loss": "",
                "notes": note,
            }
        )


def cleanup_old_models(latest: Path) -> None:
    for model_file in MODELS_DIR.glob("*.pt"):
        if model_file == latest:
            continue
        backup_dir = MODELS_DIR / "archive"
        backup_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        model_file.rename(backup_dir / f"{model_file.stem}-{timestamp}{model_file.suffix}")


def sync_once(manager: UltralyticsHubManager) -> None:
    progress = load_json(PROGRESS_FILE)
    run_id = progress.get("run_id")
    if not run_id:
        LOGGER.info("Nenhum treino ativo para sincronizar.")
        return

    try:
        status = manager.get_run_status(run_id)
    except HubError as exc:
        LOGGER.error("Erro ao consultar status do HUB: %s", exc)
        return

    best_weights = status.get("weights", {}).get("best")
    if not best_weights:
        LOGGER.info("Modelo %s ainda não possui pesos 'best' disponíveis.", run_id)
        return

    sync_state = load_json(SYNC_STATE_FILE)
    if sync_state.get("best_url") == best_weights and sync_state.get("run_id") == run_id:
        LOGGER.info("Modelo %s já está sincronizado.", run_id)
        return

    try:
        destination = manager.download_best_weights(run_id, MODELS_DIR / "iasenior_best.pt")
    except HubError as exc:
        LOGGER.error("Falha ao baixar pesos: %s", exc)
        return

    cleanup_old_models(destination)
    save_json(
        SYNC_STATE_FILE,
        {
            "run_id": run_id,
            "best_url": best_weights,
            "updated_at": datetime.utcnow().isoformat(),
            "local_path": str(destination),
        },
    )
    append_history(run_id, "synced", f"Modelo sincronizado: {destination.name}")
    LOGGER.info("Modelo %s sincronizado em %s", run_id, destination)


def main() -> int:
    ensure_structure()
    api_key = os.environ.get("ULTRALYTICS_API_KEY")
    if not api_key:
        LOGGER.error("ULTRALYTICS_API_KEY não configurado. Abortando sincronização.")
        return 1
    try:
        manager = UltralyticsHubManager(api_key)
    except HubError as exc:
        LOGGER.error("Falha na autenticação HUB: %s", exc)
        return 2

    sync_once(manager)
    schedule.every(6).hours.do(sync_once, manager=manager)
    LOGGER.info("Sincronização agendada a cada 6 horas.")
    while True:  # pragma: no cover - scheduler loop
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
