from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API_URL = os.getenv("API_URL", "http://api:8000")
STREAM_URL_TEMPLATE = os.getenv("STREAM_URL_TEMPLATE", f"{API_URL}/stream/{{camera_id}}")

st.set_page_config(page_title="SAAS Fall Detection", layout="wide", page_icon="🤖")
st.title("📹 SAAS - Monitoramento de Quedas")

refresh_rate = st.sidebar.slider("Atualização (segundos)", 5, 60, 10)


def fetch_metrics() -> Dict[str, Any]:
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Falha ao obter métricas: {exc}")
        return {
            "total_cameras": 0,
            "active_cameras": 0,
            "falls_detected": 0,
            "fps_average": 0.0,
            "events_per_hour": [0] * 24,
            "time_between_falls": None,
            "recent_events": [],
            "storage_free_gb": 0.0,
            "gpu_available": False,
            "cpu_usage": 0.0,
        }
    data = response.json()
    data["recent_events"] = [
        {
            **event,
            "start_ts": datetime.fromisoformat(event["start_ts"]),
            "end_ts": datetime.fromisoformat(event["end_ts"]) if event["end_ts"] else None,
        }
        for event in data.get("recent_events", [])
    ]
    return data


def fetch_cameras() -> List[Dict[str, Any]]:
    try:
        response = requests.get(f"{API_URL}/cameras", timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        st.sidebar.error(f"Falha ao carregar câmeras: {exc}")
        return []
    return response.json()


def control_cameras():
    st.sidebar.header("Controle de Câmeras")
    with st.sidebar.form("add-camera"):
        name = st.text_input("Nome", key="camera-name")
        rtsp = st.text_input("RTSP", key="camera-rtsp")
        enabled = st.checkbox("Ativa", value=True)
        submitted = st.form_submit_button("Adicionar câmera")
        if submitted:
            if not name or not rtsp:
                st.sidebar.error("Nome e RTSP são obrigatórios")
            else:
                payload = {"name": name, "rtsp": rtsp, "enabled": enabled}
                response = requests.post(f"{API_URL}/cameras", json=payload, timeout=10)
                if response.ok:
                    st.success("Câmera adicionada com sucesso!")
                    st.experimental_rerun()
                else:
                    st.error(f"Erro ao adicionar câmera: {response.text}")

    cameras = fetch_cameras()
    for camera in cameras:
        col1, col2 = st.sidebar.columns([2, 1])
        col1.markdown(f"**{camera['name']}**")
        start_clicked = col2.button("▶", key=f"start-{camera['id']}")
        stop_clicked = col2.button("⏸", key=f"stop-{camera['id']}")
        if start_clicked:
            try:
                response = requests.post(f"{API_URL}/cameras/{camera['id']}/start", timeout=10)
                if response.ok:
                    st.sidebar.success(f"Câmera {camera['name']} iniciada")
                else:
                    st.sidebar.error(f"Falha ao iniciar: {response.text}")
            except requests.RequestException as exc:
                st.sidebar.error(f"Erro de conexão: {exc}")
        if stop_clicked:
            try:
                response = requests.post(f"{API_URL}/cameras/{camera['id']}/stop", timeout=10)
                if response.ok:
                    st.sidebar.warning(f"Câmera {camera['name']} parada")
                else:
                    st.sidebar.error(f"Falha ao parar: {response.text}")
            except requests.RequestException as exc:
                st.sidebar.error(f"Erro de conexão: {exc}")

    return cameras


def render_metrics(metrics: Dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Câmeras totais", metrics["total_cameras"])
    col2.metric("Câmeras ativas", metrics["active_cameras"])
    col3.metric("Quedas detectadas", metrics["falls_detected"])
    col4.metric("FPS médio", f"{metrics['fps_average']:.1f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Tempo médio entre quedas (min)", metrics.get("time_between_falls") or "-" )
    col6.metric("CPU %", f"{metrics['cpu_usage']:.1f}")
    col7.metric("GPU disponível", "Sim" if metrics["gpu_available"] else "Não")
    col8.metric("Espaço livre (GB)", metrics["storage_free_gb"])

    events_df = pd.DataFrame({
        "Hora": list(range(24)),
        "Quedas/hora": metrics["events_per_hour"],
    }).set_index("Hora")
    st.subheader("Eventos por hora (últimas 24h)")
    st.bar_chart(events_df)

    st.subheader("Histórico de quedas")
    events = pd.DataFrame(metrics["recent_events"])
    if not events.empty:
        events["start_ts"] = events["start_ts"].astype(str)
        events["end_ts"] = events["end_ts"].astype(str)
        st.dataframe(events)
        csv = events.to_csv(index=False).encode("utf-8")
        st.download_button("Exportar CSV", csv, "eventos.csv", "text/csv")
    else:
        st.info("Nenhum evento registrado")


def render_live_view(cameras: List[Dict[str, Any]]) -> None:
    st.subheader("Visualização ao vivo")
    if not cameras:
        st.info("Cadastre uma câmera para visualizar o vídeo ao vivo.")
        return

    options = {camera["name"]: camera["id"] for camera in cameras}
    selected_name = st.selectbox("Selecione a câmera", list(options.keys()))
    selected_id = options[selected_name]
    stream_url = STREAM_URL_TEMPLATE.format(camera_id=selected_id)
    st.markdown(
        f"<img src='{stream_url}' style='width: 100%; border-radius: 8px; border: 1px solid #444;' />",
        unsafe_allow_html=True,
    )


cameras = control_cameras()
metrics_data = fetch_metrics()
render_metrics(metrics_data)
render_live_view(cameras)

st.sidebar.markdown("---")
st.sidebar.markdown("### Logs")
st.sidebar.markdown("Verifique /data/logs/saas-api.log no servidor para detalhes.")

st_autorefresh(interval=refresh_rate * 1000, key="panel-refresh")
