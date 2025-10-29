"""Helpers to interact with Ultralytics HUB."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from .iasenior_paths import MODELS_DIR, ensure_structure

try:  # Optional dependency for richer HUB interaction
    from hub_sdk import HUBClient  # type: ignore
except Exception:  # pragma: no cover - dependency missing at import time
    HUBClient = None  # type: ignore

HUB_API_ROOT = os.environ.get("ULTRALYTICS_HUB_API", "https://api.ultralytics.com")
HUB_WEB_ROOT = os.environ.get("ULTRALYTICS_HUB_WEB", "https://hub.ultralytics.com")


@dataclass(slots=True)
class HubRun:
    """Simple container for HUB run metadata."""

    run_id: str
    name: str
    dataset: str
    model: str
    project_url: Optional[str] = None
    raw_output: Optional[str] = None


class HubError(RuntimeError):
    """Domain specific error raised for HUB related issues."""


class UltralyticsHubManager:
    """High level helper to orchestrate HUB operations used by IASENIOR."""

    _run_pattern = re.compile(r"https?://hub\.ultralytics\.com/(?:models|training)/([a-f0-9-]{10,})", re.IGNORECASE)

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("ULTRALYTICS_API_KEY")
        if not self.api_key:
            raise HubError("Ultralytics HUB API key not configured. Set ULTRALYTICS_API_KEY.")
        os.environ.setdefault("HUB_API_KEY", self.api_key)
        os.environ.setdefault("ULTRALYTICS_API_KEY", self.api_key)
        ensure_structure()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------
    def _client(self) -> Optional["HUBClient"]:
        if HUBClient is None:
            return None
        try:
            return HUBClient({"api_key": self.api_key})
        except Exception as exc:  # pragma: no cover - depends on external SDK
            raise HubError(f"Failed to instantiate HUBClient: {exc}")

    # ------------------------------------------------------------------
    # Dataset helpers
    # ------------------------------------------------------------------
    def upload_dataset(self, dataset_zip: Path, name: Optional[str] = None, task: str = "detect") -> Dict[str, Any]:
        """Upload a dataset zip file using the HUB SDK when available."""

        client = self._client()
        dataset_zip = dataset_zip.expanduser().resolve()
        if not dataset_zip.exists():
            raise HubError(f"Dataset file not found: {dataset_zip}")

        if client:
            dataset = client.dataset()
            payload: Dict[str, Any] = {
                "meta": {"name": name or dataset_zip.stem},
                "task": task,
            }
            dataset.create_dataset(payload)
            dataset.upload_dataset(str(dataset_zip))
            return {"id": dataset.id, "name": dataset.data.get("meta", {}).get("name")}

        # CLI fallback
        cmd = [
            sys.executable,
            "-m",
            "ultralytics",
            "datasets",
            "upload",
            f"source={dataset_zip}",
        ]
        if name:
            cmd.append(f"name={name}")
        result = self._run_command(cmd)
        return result

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def start_remote_training(
        self,
        *,
        dataset: str,
        model: str,
        epochs: int,
        run_name: str,
        project: str = "IASENIOR",
        imgsz: int = 640,
        batch: int = -1,
        device: str = "auto",
    ) -> HubRun:
        """Start a HUB training session and return metadata."""

        cmd = [
            sys.executable,
            "-m",
            "ultralytics",
            "train",
            f"model={model}",
            f"data=hub://{dataset}",
            f"epochs={epochs}",
            f"imgsz={imgsz}",
            f"batch={batch}",
            f"device={device}",
            f"project={project}",
            f"name={run_name}",
            "hub=True",
        ]
        result = self._run_command(cmd)
        run_id = result.get("run_id")
        if not run_id:
            match = self._run_pattern.search(result.get("stdout", ""))
            if match:
                run_id = match.group(1)
        if not run_id:
            raise HubError("Could not determine HUB run ID from Ultralytics output.")
        project_url = result.get("run_url") or self._build_run_url(run_id)
        return HubRun(
            run_id=run_id,
            name=run_name,
            dataset=dataset,
            model=model,
            project_url=project_url,
            raw_output=result.get("stdout"),
        )

    def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Return the HUB run status, metrics and weight URLs."""

        client = self._client()
        if client:
            model = client.model(run_id)
            model.get_data()
            metrics = model.get_metrics() or []
            return {
                "status": model.data.get("status"),
                "config": model.data.get("config", {}),
                "metrics": metrics,
                "weights": {
                    "best": model.get_weights_url("best"),
                    "last": model.get_weights_url("last"),
                },
            }

        # REST fallback if hub_sdk is unavailable
        headers = {"x-api-key": self.api_key}
        response = requests.get(f"{HUB_API_ROOT}/v1/models/{run_id}", headers=headers, timeout=30)
        if response.status_code >= 300:
            raise HubError(f"Failed to query HUB status: {response.status_code} {response.text}")
        data = response.json().get("data", {})
        metrics = self._fetch_metrics(run_id, headers)
        weights = {"best": data.get("weights"), "last": data.get("resume")}
        return {"status": data.get("status"), "config": data.get("config", {}), "metrics": metrics, "weights": weights}

    def download_best_weights(self, run_id: str, destination: Optional[Path] = None) -> Path:
        """Download the best weights for a HUB run and return the path."""

        client = self._client()
        weights_url: Optional[str] = None
        if client:
            model = client.model(run_id)
            model.get_data()
            weights_url = model.get_weights_url("best")
        if not weights_url:
            headers = {"x-api-key": self.api_key}
            response = requests.get(f"{HUB_API_ROOT}/v1/models/{run_id}", headers=headers, timeout=30)
            response.raise_for_status()
            weights_url = response.json().get("data", {}).get("weights")
        if not weights_url:
            raise HubError("Run does not expose downloadable weights yet.")

        ensure_structure()
        destination = destination or MODELS_DIR / f"{run_id}_best.pt"
        with requests.get(weights_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with destination.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        return destination

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_run_url(self, run_id: str) -> str:
        return f"{HUB_WEB_ROOT}/models/{run_id}"

    def _run_command(self, cmd: list[str]) -> Dict[str, Any]:
        """Execute a CLI command and capture stdout/stderr."""

        with self._lock:
            process = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "ULTRALYTICS_API_KEY": self.api_key},
            )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        payload: Dict[str, Any] = {"stdout": stdout, "stderr": stderr, "returncode": process.returncode}
        if process.returncode != 0:
            raise HubError(f"Command failed: {' '.join(cmd)}\n{stderr or stdout}")
        try:
            json_payload = json.loads(stdout)
            if isinstance(json_payload, dict):
                payload.update(json_payload)
        except json.JSONDecodeError:
            pass
        return payload

    def _fetch_metrics(self, run_id: str, headers: Dict[str, str]) -> list[dict[str, Any]]:
        response = requests.get(f"{HUB_API_ROOT}/v1/models/{run_id}/metrics", headers=headers, timeout=30)
        if response.status_code >= 300:
            return []
        return response.json().get("data", [])
