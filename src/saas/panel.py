import os
import time
import streamlit as st
import requests
import psutil
import simpleaudio as sa

API_URL = "http://127.0.0.1:8000"
ALERT_SOUND = "/mnt/data/SAAS_core/SAAS/assets/alert.wav"

st.set_page_config(page_title="Painel IASenior", layout="wide")

st.title("🎥 Painel de Monitoramento IASenior")

# Status de câmeras
st.subheader("📡 Status de Câmeras Conectadas")
try:
    resp = requests.get(f"{API_URL}/cameras")
    cameras = resp.json().get("cameras", [])
    st.metric("Câmeras Conectadas", len(cameras))
except Exception:
    st.error("Falha ao obter câmeras da API.")

# Acurácia média da IA
st.subheader("📊 Desempenho da IA")
try:
    acc = requests.get(f"{API_URL}/accuracy").json().get("accuracy", 0)
    st.metric("Acurácia Média (%)", f"{acc:.2f}")
except Exception:
    st.warning("Acurácia ainda não disponível.")

# Alertas
st.subheader("🚨 Alertas Recentes")
try:
    alerts = requests.get(f"{API_URL}/alerts").json().get("alerts", [])
    if alerts:
        for a in alerts[-5:]:
            st.warning(f"{a['timestamp']} → {a['message']}")
            if os.path.exists(ALERT_SOUND):
                sa.WaveObject.from_wave_file(ALERT_SOUND).play()
    else:
        st.info("Sem alertas recentes.")
except Exception:
    st.error("Falha ao carregar alertas.")

# Recursos do sistema
st.subheader("💻 Recursos do Servidor")
cpu = psutil.cpu_percent(interval=1)
mem = psutil.virtual_memory().percent
st.progress(cpu / 100)
st.write(f"CPU: {cpu:.1f}% | Memória: {mem:.1f}%")

# Atualização automática
st.sidebar.markdown("## ⚙️ Atualização Automática")
refresh = st.sidebar.slider("Intervalo (segundos)", 5, 60, 10)
if st.sidebar.button("🔄 Atualizar Agora"):
    st.experimental_rerun()
else:
    time.sleep(refresh)
    st.experimental_rerun()
