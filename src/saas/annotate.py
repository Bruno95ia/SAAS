import streamlit as st
import requests
import os
import cv2
import torch
import time
from datetime import datetime
from ultralytics import YOLO

# Configurações API
API_URL = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("SAAS_API_KEY", "minha-chave-forte")
HEADERS = {"X-API-Key": API_KEY}

# Configuração do modelo
MODEL_PATH = "/mnt/data/SAAS_core/SAAS/src/saas/weights/best.pt"
os.makedirs("/mnt/data/SAAS_core/SAAS/runs", exist_ok=True)
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="Painel SAAS", layout="wide")
st.title("📹 Monitoramento SAAS com IA – YOLO + Alertas Automáticos")

# -------- Função: POST de alerta --------
def post_alert(label, confidence):
    ts = datetime.now().isoformat()
    try:
        data = {"label": label, "confidence": confidence, "timestamp": ts}
        requests.post(f"{API_URL}/post-alert", headers=HEADERS, json=data)
    except Exception as e:
        st.error(f"Falha ao enviar alerta: {e}")

# -------- Função: Buscar alertas --------
def get_alerts():
    try:
        r = requests.get(f"{API_URL}/alerts", headers=HEADERS)
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"Erro {r.status_code} na API")
    except Exception as e:
        st.error(f"Falha ao conectar: {e}")
    return []

# -------- Exibir alertas --------
st.subheader("📋 Últimos alertas detectados")
alerts = get_alerts()

if alerts:
    for a in alerts[-10:]:
        ts = a.get("ts") or a.get("timestamp") or "sem data"
        st.write(f"🟢 {a.get('label', 'desconhecido')} | Confiança: {a.get('confidence', 0):.2f} | {ts}")
else:
    st.info("Nenhum alerta recente encontrado.")

# -------- Inferência em tempo real --------
st.subheader("🎥 Câmera com IA em tempo real")
rtsp = st.text_input("URL RTSP rtsp://127.0.01:8554/cam01")

if st.button("Iniciar"):
    cap = cv2.VideoCapture(0 if rtsp == "0" else rtsp)
    frame_area = st.empty()
    st.info("Processando stream com YOLO...")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Stream encerrado.")
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Verifica detecções
        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf > 0.7:
                # Salvar frame e enviar alerta
                filename = f"/mnt/data/SAAS_core/SAAS/runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{cls}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                post_alert(cls, conf)

        frame_area.image(annotated)
        time.sleep(0.03)

    cap.release()
