"""Command line helper to trigger IASENIOR trainings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

from saas.utils.iasenior_paths import DATASETS_DIR, LOGS_DIR, ensure_structure
from saas.utils.hub_integration import HubError, UltralyticsHubManager

PROGRESS_FILE = LOGS_DIR / "train_progress.json"
HISTORY_FILE = LOGS_DIR / "treinos.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IASENIOR training orchestrator")
    parser.add_argument("dataset", help="Dataset identifier on HUB or local dataset folder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--api-key", dest="api_key", default=os.environ.get("ULTRALYTICS_API_KEY"))
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--fallback-local",
        action="store_true",
        help="Fallback to local YOLO training if HUB orchestration fails",
    )
    return parser.parse_args()


def append_history(row: Dict[str, Any]) -> None:
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
        writer.writerow(row)


def save_progress(payload: Dict[str, Any]) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(payload, indent=2, default=str))


def run_local_training(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "Ultralytics package is required for local fallback training but is not available"
        ) from exc

    data_yaml = DATASETS_DIR / args.dataset / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML não encontrado: {data_yaml}")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="IASENIOR",
        name=f"local-{datetime.utcnow():%Y%m%d-%H%M%S}",
    )
    run_id = f"local-{datetime.utcnow():%Y%m%d%H%M%S}"
    metrics = getattr(results, "results_dict", {}) if results else {}
    save_progress(
        {
            "run_id": run_id,
            "status": "completed",
            "dataset": args.dataset,
            "model": args.model,
            "epochs": args.epochs,
            "progress": {"current_epoch": args.epochs, "epochs": args.epochs},
            "metrics": metrics,
            "weights": {},
            "mode": "local",
            "updated_at": datetime.utcnow().isoformat(),
        }
    )
    append_history(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": args.dataset,
            "run_id": run_id,
            "model": args.model,
            "epochs": args.epochs,
            "status": "completed",
            "map50": metrics.get("metrics/mAP50"),
            "map5095": metrics.get("metrics/mAP50-95"),
            "precision": metrics.get("metrics/precision"),
            "recall": metrics.get("metrics/recall"),
            "loss": metrics.get("train/loss"),
            "notes": args.notes,
        }
    )
    return {"run_id": run_id, "mode": "local", "metrics": metrics}


def main() -> int:
    ensure_structure()
    args = parse_args()
    if not args.api_key:
        print("ULTRALYTICS_API_KEY não definido. Configure a variável de ambiente ou use --api-key.", file=sys.stderr)
        if not args.fallback_local:
            return 1
    try:
        manager = UltralyticsHubManager(args.api_key) if args.api_key else None
    except HubError as exc:
        print(f"Falha ao autenticar no HUB: {exc}", file=sys.stderr)
        manager = None
        if not args.fallback_local:
            return 2

    if manager is None:
        result = run_local_training(args)
        print(json.dumps(result, indent=2, default=str))
        return 0

    run_name = f"cli-{args.dataset}-{datetime.utcnow():%Y%m%d-%H%M%S}"
    try:
        hub_run = manager.start_remote_training(
            dataset=args.dataset,
            model=args.model,
            epochs=args.epochs,
            run_name=run_name,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )
    except HubError as exc:
        print(f"Erro ao iniciar treino no HUB: {exc}", file=sys.stderr)
        if args.fallback_local:
            result = run_local_training(args)
            print(json.dumps(result, indent=2, default=str))
            return 0
        return 3

    payload = {
        "run_id": hub_run.run_id,
        "status": "starting",
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.epochs,
        "progress": {"current_epoch": 0, "epochs": args.epochs},
        "metrics": [],
        "weights": {},
        "mode": "hub",
        "project_url": hub_run.project_url,
        "notes": args.notes,
        "updated_at": datetime.utcnow().isoformat(),
    }
    save_progress(payload)
    append_history(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "dataset": args.dataset,
            "run_id": hub_run.run_id,
            "model": args.model,
            "epochs": args.epochs,
            "status": "starting",
            "map50": "",
            "map5095": "",
            "precision": "",
            "recall": "",
            "loss": "",
            "notes": args.notes,
        }
    )
    print(json.dumps({"run_id": hub_run.run_id, "project_url": hub_run.project_url}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
