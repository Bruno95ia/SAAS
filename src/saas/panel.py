import streamlit as st
import requests
import os
import subprocess
import cv2
import psutil
import sqlite3
import pandas as pd
import time
import tempfile

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
API_URL = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("SAAS_API_KEY", "minha-chave-forte")
DB_PATH = os.getenv("SAAS_DB_PATH", "src/saas/events.db")

st.set_page_config(page_title="IA Senior - Painel Operacional", layout="wide")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controle da Stack")

if st.sidebar.button("🚀 Iniciar Stack"):
    subprocess.Popen(["bash", "start_saas.sh"])
    st.sidebar.success("Stack reiniciada com sucesso.")

if st.sidebar.button("🧹 Parar Stack"):
    subprocess.Popen(["bash", "stop_saas.sh"])
    st.sidebar.warning("Stack interrompida.")

if st.sidebar.button("🔍 Testar API"):
    try:
        r = requests.get(f"{API_URL}/health")
        st.sidebar.success(f"API OK: {r.json()}")
    except Exception as e:
        st.sidebar.error(f"Erro: {e}")

# Status do sistema
cpu = psutil.cpu_percent(interval=0.5)
mem = psutil.virtual_memory().percent
disk = psutil.disk_usage('/mnt/data').percent

st.sidebar.markdown("### 💻 Status do Sistema")
st.sidebar.write(f"CPU: {cpu}% | Memória: {mem}% | Disco: {disk}%")

# =========================
# CABEÇALHO PRINCIPAL
# =========================
st.title("🎥 Painel de Monitoramento - IA Senior")

st.markdown("""
O painel permite visualizar o vídeo, acompanhar inferências em tempo real e consultar eventos registrados.
""")

# =========================
# ABA 1 - INFERÊNCIA
# =========================
tab1, tab2, tab3 = st.tabs(["🧠 Inferência", "📊 Histórico", "📈 Monitoramento"])

with tab1:
    st.subheader("Execução de Inferência em Vídeo ou RTSP")

    url_input = st.text_input("Digite o caminho do vídeo local ou URL RTSP:", "/mnt/data/sample.mp4")
    start_button = st.button("Iniciar IA")

    if start_button:
        st.info("Executando inferência...")
        cap = cv2.VideoCapture(url_input)
        frame_container = st.empty()
        label_area = st.empty()
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            cv2.imwrite(temp_file.name, frame)
            try:
                with open(temp_file.name, "rb") as f:
                    files = {"file": f}
                    headers = {"Authorization": f"Bearer {API_KEY}"}
                    res = requests.post(f"{API_URL}/infer", files=files, headers=headers, timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        label_area.write(f"Frame {count}: {data.get('labels', [])}")
            except Exception as e:
                label_area.error(f"Erro: {e}")
            finally:
                os.unlink(temp_file.name)
            frame_container.image(frame, channels="BGR", use_column_width=True)
        cap.release()
        st.success("✅ Inferência concluída.")

# =========================
# ABA 2 - HISTÓRICO
# =========================
with tab2:
    st.subheader("📄 Eventos registrados")
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM events ORDER BY id DESC LIMIT 50", conn)
        st.dataframe(df)
        conn.close()
    except Exception as e:
        st.warning(f"Banco ainda vazio ou inacessível: {e}")

# =========================
# ABA 3 - MONITORAMENTO
# =========================
with tab3:
    st.subheader("📈 Diagnóstico em Tempo Real")

    col1, col2, col3 = st.columns(3)
    col1.metric("CPU", f"{cpu} %")
    col2.metric("Memória", f"{mem} %")
    col3.metric("Disco (/mnt/data)", f"{disk} %")

    st.markdown("### 🔎 Logs recentes")
    try:
        log_output = subprocess.check_output(["tail", "-n", "20", "/var/log/mediamtx.log"]).decode("utf-8")
        st.text_area("MediaMTX", log_output, height=200)
    except Exception:
        st.info("Nenhum log encontrado ou arquivo ainda não criado.")
