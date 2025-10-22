import streamlit as st
import requests
import os
import cv2
import numpy as np
from PIL import Image

# Configuração
API_URL = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("SAAS_API_KEY", "minha-chave-forte")

st.set_page_config(page_title="Painel SAAS", layout="wide")
st.title("📹 Monitoramento SAAS com IA")

# ========== Função para buscar alertas ==========
def get_alerts():
    try:
        r = requests.get(f"{API_URL}/alerts", headers={"X-API-Key": API_KEY})
        if r.status_code == 200:
            return r.json()
        else:
            st.warning(f"Erro {r.status_code} na API")
    except Exception as e:
        st.error(f"Falha ao conectar: {e}")
    return []

# ========== Exibição de alertas ==========
st.subheader("📋 Últimos alertas detectados")
alerts = get_alerts()

if alerts:
    for a in alerts[-10:]:
        ts = a.get("timestamp") or a.get("ts") or "sem data"
        st.write(f"🟢 {a['label']} | Confiança: {a['confidence']:.2f} | {ts}")
else:
    st.info("Nenhum alerta recente encontrado.")

# ========== Exibição de stream com inferência ==========
st.subheader("🎥 Câmera com inferência em tempo real")

rtsp = st.text_input("URL RTSP (ou 0 para webcam local)", "0")

if st.button("Iniciar"):
    cap = cv2.VideoCapture(0 if rtsp == "0" else rtsp)
    frame_area = st.empty()

    st.info("A IA está processando o vídeo...")
    # Aqui simula inferência (YOLO pode ser integrado com saída real)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Stream encerrado.")
            break

        # Simulação de detecção: desenha caixa central
        h, w, _ = frame.shape
        cv2.rectangle(frame, (w//3, h//3), (w*2//3, h*2//3), (0,255,0), 2)
        cv2.putText(frame, "IA: pessoa detectada", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_area.image(frame)

    cap.release()
