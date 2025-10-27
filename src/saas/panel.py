"""Streamlit dashboard for the SAAS proof of concept."""
from __future__ import annotations

import time
from datetime import datetime
from typing import List

import altair as alt
import pandas as pd
import requests
import streamlit as st

from saas.config import get_settings
from saas.utils import AlertStore, SystemMetricsCollector, read_metrics_state

settings = get_settings()
alert_store = AlertStore(settings.alerts_db_path)
stream_dir = settings.stream_dir
metrics_path = settings.metrics_path

st.set_page_config(page_title="SAAS Monitor", layout="wide")
st.title("🧠 SAAS — Monitoramento Inteligente")
st.caption("Detecção automática de quedas e movimentos suspeitos com YOLOv8")

refresh_interval = st.sidebar.slider("Intervalo de atualização (s)", min_value=2, max_value=15, value=5)
st.sidebar.write("Configurações lidas de src/saas/config.py:")
st.sidebar.code(settings.to_json(), language="json")


def fetch_alerts() -> List[dict]:
    try:
        response = requests.get(
            f"{settings.api_url.rstrip('/')}/alerts",
            headers={"X-API-Key": settings.api_key},
            timeout=3,
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return [alert.to_dict() for alert in alert_store.list(limit=200)]


def format_alert(alert: dict) -> str:
    timestamp = datetime.fromtimestamp(alert.get("timestamp", time.time())).strftime("%H:%M:%S")
    confidence = float(alert.get("confidence", 0.0)) * 100
    camera = alert.get("camera", "Câmera")
    label = alert.get("label", "Alerta")
    label_display = label.title() if isinstance(label, str) else str(label)
    return f"🚨 {label_display} detectada — {confidence:.0f}% de confiança — {camera} — {timestamp}"


alerts = fetch_alerts()
metrics = SystemMetricsCollector.collect()
state = read_metrics_state(metrics_path)
fps_value = float(state.get("fps", 0.0))
alert_count = len(alerts)

col_cpu, col_memory, col_disk, col_fps, col_alerts = st.columns(5)
col_cpu.metric("CPU %", f"{metrics['cpu']:.1f}")
col_memory.metric("Memória %", f"{metrics['memory']:.1f}")
col_disk.metric("Disco %", f"{metrics['disk']:.1f}")
col_fps.metric("FPS médio", f"{fps_value:.1f}")
col_alerts.metric("Alertas", f"{alert_count}")

if alerts:
    st.success(format_alert(alerts[0]))
else:
    st.info("Nenhum alerta registrado até o momento.")

monitoring_tab, history_tab, charts_tab = st.tabs([
    "Monitoramento ao vivo",
    "Histórico de alertas",
    "Gráficos e métricas",
])

with monitoring_tab:
    st.subheader("📺 Vídeo das câmeras")
    for index, camera_path in enumerate(settings.cameras, start=1):
        frame_path = stream_dir / f"camera_{index:02d}.jpg"
        placeholder = st.empty()
        frame_bytes = None
        if frame_path.exists():
            frame_bytes = frame_path.read_bytes()
        if frame_bytes:
            placeholder.image(frame_bytes, caption=f"Câmera {index:02d}", use_column_width=True)
        else:
            placeholder.warning(f"Aguardando frames para a câmera {index:02d} ({camera_path})")

with history_tab:
    st.subheader("📜 Últimos alertas")
    df = pd.DataFrame(alerts)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.sort_values("timestamp", ascending=False)
        st.dataframe(df, use_container_width=True)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV", data=csv_bytes, file_name="saas_alerts.csv", mime="text/csv")
    else:
        st.info("Nenhum alerta disponível.")

with charts_tab:
    st.subheader("📈 Distribuição de alertas")
    df = pd.DataFrame(alerts)
    if df.empty:
        st.info("Sem dados suficientes para gráficos.")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        if df.empty:
            st.info("Sem dados válidos para exibir.")
        else:
            hourly = (
                df.assign(hour=lambda d: d["timestamp"].dt.floor("H"))
                .groupby("hour")
                .size()
                .reset_index(name="count")
            )
            chart_hour = (
                alt.Chart(hourly)
                .mark_bar()
                .encode(x="hour:T", y="count:Q")
                .properties(height=300)
            )
            st.altair_chart(chart_hour, use_container_width=True)

            by_camera = df.groupby("camera").size().reset_index(name="count")
            chart_camera = (
                alt.Chart(by_camera)
                .mark_bar()
                .encode(x="camera:N", y="count:Q", color="camera:N")
                .properties(height=300)
            )
            st.altair_chart(chart_camera, use_container_width=True)

st.caption(f"Atualização automática a cada {refresh_interval} segundos.")
time.sleep(refresh_interval)
st.experimental_rerun()
