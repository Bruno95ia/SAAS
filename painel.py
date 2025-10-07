import os, json, time, threading, datetime as dt
import streamlit as st
import pandas as pd
import requests

# --- Config (pode sobrescrever por env) ---
API     = os.getenv("SAAS_API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("SAAS_API_KEY", "dev-key")   # use a mesma da API

HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Quedas – Painel", layout="wide")
st.title("Painel de Detecção de Quedas")
st.caption(f"API: {API}")

# --- Helpers ----------------------------------------------------------------
def safe_get(path: str, params=None):
    """GET com tratamento de erro → retorna [] quando falha."""
    try:
        r = requests.get(f"{API}{path}", headers=HEADERS, params=params, timeout=5)
        if r.status_code != 200:
            # mostra o payload de erro da API
            try:
                msg = r.json()
            except Exception:
                msg = r.text
            st.warning(f"API {r.status_code}: {msg}")
            return []
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        st.warning(f"Falha ao buscar {path}: {e}")
        return []

def to_df(data):
    """Converte lista de dicts em DataFrame (ou vazio)."""
    return pd.DataFrame(data) if isinstance(data, list) and len(data) else pd.DataFrame([])

# --- Estado compartilhado (para realtime) -----------------------------------
if "alerts_buffer" not in st.session_state:
    st.session_state.alerts_buffer = []  # prepend conforme chegam

# Realtime via WebSocket (opcional). Mantido simples (sem auth no WS).
USE_WS = os.getenv("SAAS_WS", "0") == "1"
def start_ws_once():
    try:
        from websocket import WebSocketApp
    except Exception:
        st.info("WebSocket desabilitado (instale: `pip install websocket-client` ou set SAAS_WS=0).")
        return
    def _on_message(_ws, message):
        try:
            msg = json.loads(message)
            if msg.get("event") == "alert":
                st.session_state.alerts_buffer.insert(0, {
                    "id":       msg.get("id"),
                    "ts":       dt.datetime.utcnow().isoformat(),
                    "camera_id":msg["data"].get("camera_id"),
                    "type":     msg["data"].get("type"),
                    "score":    msg["data"].get("score"),
                    "clip_path":msg["data"].get("clip_path"),
                    "extra":    msg["data"].get("extra"),
                })
                st.session_state["_tick"] = time.time()  # força rerender
        except Exception:
            pass
    def _runner(url):
        from websocket import WebSocketApp
        ws = WebSocketApp(url, on_message=_on_message)
        while True:
            try:
                ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception:
                time.sleep(2)

    if "ws_started" not in st.session_state:
        ws_url = API.replace("http", "ws") + "/ws"
        threading.Thread(target=_runner, args=(ws_url,), daemon=True).start()
        st.session_state.ws_started = True
        st.info("WebSocket conectado (tempo real).")

if USE_WS:
    start_ws_once()

# --- Layout ------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Métricas (hoje)")
    base = to_df(safe_get("/alerts", params={"limit": 200}))
    if len(st.session_state.alerts_buffer):
        base = pd.concat([to_df(st.session_state.alerts_buffer), base], ignore_index=True)
    if not base.empty:
        base["ts_dt"] = pd.to_datetime(base["ts"], errors="coerce")
        today = base[base["ts_dt"].dt.date == dt.date.today()]
        st.metric("Quedas detectadas", int((today["type"] == "fall").sum()))
        st.metric("Alertas totais", int(len(today)))
    else:
        st.info("Sem dados (ainda).")

with col1:
    st.subheader("Alertas recentes")
    data = to_df(safe_get("/alerts", params={"limit": 50}))
    if len(st.session_state.alerts_buffer):
        data = pd.concat([to_df(st.session_state.alerts_buffer), data], ignore_index=True)
    data = data.drop_duplicates(subset=["id"], keep="first") if not data.empty else data

    if not data.empty:
        st.dataframe(
            data[["id", "ts", "camera_id", "type", "score", "clip_path"]],
            use_container_width=True, height=420
        )
        # player de clipe (se houver)
        clips = [""] + list(data["clip_path"].dropna().unique())
        sel = st.selectbox("Ver clipe", options=clips, index=0)
        if sel:
            st.video(sel)
    else:
        st.info("Sem alertas recentes.")

# Rodapé
st.caption("Use SAAS_API_URL / SAAS_API_KEY (env) para apontar a API e autenticar.")