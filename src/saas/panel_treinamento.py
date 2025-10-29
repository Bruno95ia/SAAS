"""Streamlit panel to control IASENIOR HUB trainings."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

from .utils.iasenior_paths import LOGS_DIR, MODELS_DIR, ensure_structure

ensure_structure()

API_ROOT = os.environ.get("IASENIOR_TRAIN_API", "http://localhost:8000")
REFRESH_SECONDS = int(os.environ.get("IASENIOR_PANEL_REFRESH", "15"))

st.set_page_config(page_title="IASENIOR Treinamentos", layout="wide")
st.title("Painel de Treinamento IASENIOR")


@st.cache_data(ttl=REFRESH_SECONDS)
def load_history() -> pd.DataFrame:
    history_file = LOGS_DIR / "treinos.csv"
    if not history_file.exists():
        return pd.DataFrame(columns=[
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
        ])
    return pd.read_csv(history_file)


def fetch_json(path: str) -> Dict[str, Any]:
    response = requests.get(f"{API_ROOT}{path}", timeout=20)
    response.raise_for_status()
    return response.json()


def start_training_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{API_ROOT}/api/train/start", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Backend")
    try:
        status = fetch_json("/api/train/status")
        st.success("Backend online")
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Falha ao conectar com API: {exc}")
        status = {
            "status": "offline",
            "progress": {},
            "metrics": [],
            "weights": {},
            "run_id": None,
            "project_url": None,
            "notes": None,
            "updated_at": datetime.utcnow().isoformat(),
        }
    st.json(status)

    st.divider()
    st.header("Modelo mais recente")
    try:
        latest = fetch_json("/api/models/latest")
        if latest.get("path"):
            st.success(f"{latest['path']}")
            path_obj = Path(latest["path"])
            if path_obj.exists():
                st.download_button(
                    "Baixar modelo",
                    data=path_obj.read_bytes(),
                    file_name=path_obj.name,
                )
        else:
            st.info("Nenhum modelo sincronizado ainda.")
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Erro ao consultar modelo mais recente: {exc}")


st.subheader("Iniciar novo treino")
with st.form("start-train"):
    dataset = st.text_input("Dataset (ID HUB)", help="Use o identificador do dataset no Ultralytics HUB")
    epochs = st.number_input("Épocas", min_value=1, max_value=500, value=50)
    model = st.text_input("Modelo base", value="yolov8n.pt")
    imgsz = st.number_input("Image size", min_value=320, max_value=2048, value=640, step=32)
    batch = st.number_input("Batch size", min_value=-1, max_value=256, value=-1)
    device = st.text_input("Device", value="auto")
    notes = st.text_area("Anotações", height=80)
    submitted = st.form_submit_button("Iniciar treino")

    if submitted:
        if not dataset:
            st.warning("Informe o dataset que está registrado no HUB.")
        else:
            try:
                response = start_training_request(
                    {
                        "dataset": dataset,
                        "epochs": epochs,
                        "model": model,
                        "imgsz": imgsz,
                        "batch": batch,
                        "device": device,
                        "notes": notes or None,
                    }
                )
                st.success(f"Treino iniciado! Run ID: {response['run_id']}")
                if response.get("project_url"):
                    st.markdown(f"[Abrir no HUB]({response['project_url']})")
                st.cache_data.clear()
            except Exception as exc:  # pragma: no cover - UI only
                st.error(f"Erro ao iniciar treino: {exc}")


st.subheader("Progresso do treino atual")
status = fetch_json("/api/train/status")
cols = st.columns(4)
cols[0].metric("Status", status.get("status", "desconhecido"))
progress = status.get("progress", {})
cols[1].metric("Época", f"{progress.get('current_epoch')}/{progress.get('epochs')}")
cols[2].metric("mAP50", progress.get("map50"))
cols[3].metric("Perda", progress.get("loss"))

progress_bar = st.progress(0.0)
current_epoch = progress.get("current_epoch") or 0
total_epochs = progress.get("epochs") or 0
if current_epoch and total_epochs:
    completion = current_epoch / max(total_epochs, 1)
    progress_bar.progress(min(1.0, completion))
else:
    progress_bar.progress(0.0)

if status.get("project_url"):
    st.markdown(f"[Abrir corrida no HUB]({status['project_url']})")
if status.get("notes"):
    st.info(f"Notas: {status['notes']}")

st.divider()
st.subheader("Histórico de treinos")
history_df = load_history()
if history_df.empty:
    st.info("Nenhum treino registrado ainda.")
else:
    history_df["timestamp"] = pd.to_datetime(history_df["timestamp"]).dt.tz_localize(None)
    st.dataframe(history_df.sort_values("timestamp", ascending=False), use_container_width=True)
