import streamlit as st
import subprocess
import warnings
import psutil
import sqlite3
import os
from datetime import datetime, timedelta
import pandas as pd
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Monitoramento SAAS com IA", layout="wide")

st.title("📸 Monitoramento SAAS com IA")

tab1, tab2, tab3 = st.tabs(["🎥 Câmera / Vídeo", "⚠️ Alertas recentes", "📈 Diagnóstico e Acuracidade"])

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
        conn = sqlite3.connect("/mnt/data/SAAS_core/SAAS/src/saas/events.db")
        df = pd.read_sql_query("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 10", conn)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Nenhum alerta registrado ainda.")
        conn.close()
    except Exception as e:
        st.warning(f"Banco ainda vazio ou inacessível: {e}")

# =========================
# ABA 3 - MONITORAMENTO E ACURACIDADE
# =========================
with tab3:
    st.subheader("📈 Diagnóstico e Status Operacional")

    # --- Métricas de sistema ---
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/mnt/data").percent
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU (%)", f"{cpu:.1f}")
    col2.metric("Memória (%)", f"{mem:.1f}")
    col3.metric("Disco (/mnt/data)", f"{disk:.1f}")

    # --- Câmeras conectadas ---
    try:
        result = subprocess.check_output(
            ["grep", "-c", "RTSP connection", "/mnt/data/logs/mediamtx.log"], text=True
        ).strip()
        st.metric("Câmeras conectadas (ativas)", result)
    except Exception:
        st.metric("Câmeras conectadas (ativas)", "0")

    # --- Serviços ativos ---
    try:
        api_status = subprocess.check_output(
            ["curl", "-s", "http://127.0.0.1:8000/health"], text=True
        )
        st.success("✅ API online") if '"ok":true' in api_status else st.warning("⚠️ API inativa")
    except Exception:
        st.warning("⚠️ API inativa")

    # --- Inferências e acuracidade ---
    acc = 0
    infs = 0
    try:
        conn = sqlite3.connect("/mnt/data/SAAS_core/SAAS/src/saas/events.db")
        cur = conn.cursor()
        cur.execute("""
            SELECT confidence FROM alerts
            WHERE timestamp >= datetime('now','-5 minute')
        """)
        confs = [row[0] for row in cur.fetchall()]
        infs = len(confs)
        acc = sum(confs)/len(confs) if confs else 0
        conn.close()
        st.metric("Inferências (últ. 5 min)", infs)
        st.metric("Acuracidade média (%)", f"{acc*100:.2f}")
    except Exception:
        st.metric("Inferências (últ. 5 min)", "0")
        st.metric("Acuracidade média (%)", "0")

    # --- Último alerta e som ---
    try:
        conn = sqlite3.connect("/mnt/data/SAAS_core/SAAS/src/saas/events.db")
        cur = conn.cursor()
        cur.execute("SELECT label, confidence, timestamp FROM alerts ORDER BY timestamp DESC LIMIT 1")
        last = cur.fetchone()
        if last:
            label, conf, ts = last
            st.info(f"🕒 Último alerta: **{label}** ({conf*100:.1f}%) - {ts}")
            if label.lower() in ["queda", "intrusão", "colisão", "pessoa caída"]:
                components.html(
                    """
                    <audio autoplay>
                      <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
                    </audio>
                    """,
                    height=0,
                )
        conn.close()
    except Exception:
        st.info("Nenhum alerta recente encontrado.")

    # --- Gráfico de acuracidade (últimas 24h) ---
    try:
        conn = sqlite3.connect("/mnt/data/SAAS_core/SAAS/src/saas/events.db")
        df_acc = pd.read_sql_query("""
            SELECT timestamp, confidence FROM alerts
            WHERE timestamp >= datetime('now','-24 hour')
        """, conn)
        conn.close()
        if not df_acc.empty:
            df_acc["timestamp"] = pd.to_datetime(df_acc["timestamp"])
            df_acc["hora"] = df_acc["timestamp"].dt.strftime("%H:%M")
            df_acc["acuracia"] = df_acc["confidence"] * 100
            df_acc["média móvel"] = df_acc["acuracia"].rolling(10, min_periods=1).mean()
            st.markdown("### 📊 Histórico de Acuracidade (últimas 24h)")
            st.line_chart(df_acc.set_index("hora")[["acuracia", "média móvel"]])
        else:
            st.info("Sem dados recentes de acuracidade.")
    except Exception as e:
        st.warning(f"Erro ao gerar gráfico: {e}")

    # --- Modelo e versão ---
    model_path = "/mnt/data/SAAS_core/SAAS/src/saas/weights/best.pt"
    if os.path.exists(model_path):
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M")
        st.success(f"Modelo ativo: **best.pt** (atualizado em {mod_time})")
    else:
        st.warning("Modelo best.pt não encontrado.")

    # --- Último commit Git ---
    try:
        git_info = subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%h - %s (%ci)"],
            text=True,
            cwd="/mnt/data/SAAS_core/SAAS"
        )
        st.text(f"📦 Git: {git_info}")
    except Exception:
        st.info("Status Git não disponível.")
