import streamlit as st
import subprocess
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Monitoramento SAAS com IA", layout="wide")

st.title("📸 Monitoramento SAAS com IA")

tab1, tab2, tab3 = st.tabs(["🎥 Câmera / Vídeo", "⚠️ Alertas recentes", "📈 Diagnóstico"])

# =========================
# ABA 1 - CÂMERA / VÍDEO
# =========================
with tab1:
    st.subheader("Transmissão e Análise de Vídeo")

    video_path = st.text_input("Caminho do vídeo local ou URL RTSP:")
    if st.button("Iniciar"):
        st.info(f"Transmitindo de: {video_path}")
        st.image("/mnt/data/SAAS_core/SAAS/src/saas/static/sample_frame.jpg", use_container_width=True)

# =========================
# ABA 2 - ALERTAS
# =========================
with tab2:
    st.subheader("⚠️ Últimos alertas detectados")
    try:
        import sqlite3
        conn = sqlite3.connect("/mnt/data/SAAS_core/SAAS/src/saas/events.db")
        df = None
        try:
            df = st.dataframe(conn.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10").fetchall())
        except Exception:
            st.info("Nenhum alerta registrado ainda.")
        conn.close()
    except Exception as e:
        st.warning(f"Banco ainda vazio ou inacessível: {e}")

# =========================
# ABA 3 - MONITORAMENTO
# =========================
with tab3:
    st.subheader("📈 Diagnóstico em Tempo Real")

    col1, col2, col3 = st.columns(3)
    col1.metric("CPU", "carregando...")
    col2.metric("Memória", "carregando...")
    col3.metric("Disco", "carregando...")

    st.markdown("### 🔎 Logs recentes")
    try:
        log_output = subprocess.check_output(["tail", "-n", "20", "/mnt/data/logs/mediamtx.log"], text=True)
        st.text_area("MediaMTX", log_output, height=200)
    except Exception:
        st.info("Nenhum log encontrado em /mnt/data/logs/.")

    try:
        api_log = subprocess.check_output(["tail", "-n", "20", "/mnt/data/logs/saas_api.log"], text=True)
        st.text_area("API FastAPI", api_log, height=200)
    except Exception:
        st.info("Logs da API ainda não disponíveis.")
