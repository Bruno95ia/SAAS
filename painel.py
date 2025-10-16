import os, json, time, threading, datetime as dt
import streamlit as st
import pandas as pd
import requests
import altair as alt

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
    base = to_df(safe_get("/alerts", params={"limit": 500}))
    if len(st.session_state.alerts_buffer):
        base = pd.concat([to_df(st.session_state.alerts_buffer), base], ignore_index=True)

    if not base.empty:
        base = base.drop_duplicates(subset=["id"], keep="first")
        base["ts_dt"] = pd.to_datetime(base["ts"], errors="coerce")
        base["type_norm"] = base["type"].fillna("").str.lower()
        today = base[base["ts_dt"].dt.date == dt.date.today()]

        total_alerts = int(len(today))
        total_falls = int((today["type_norm"] == "fall").sum())
        last_fall_ts = (
            today[today["type_norm"] == "fall"]["ts_dt"].max()
            if total_falls
            else None
        )
        avg_score = today["score"].dropna().mean() if total_alerts else None
        cameras_today = today["camera_id"].dropna().unique()

        st.metric("Alertas totais", total_alerts)
        st.metric("Quedas detectadas", total_falls)
        st.metric(
            "Última queda",
            last_fall_ts.strftime("%H:%M:%S") if last_fall_ts else "–",
        )
        st.metric(
            "Câmeras ativas",
            len(cameras_today),
            help="Número de câmeras que geraram alertas hoje",
        )
        if avg_score is not None:
            st.caption(f"Score médio dos alertas de hoje: {avg_score:.2f}")
    else:
        st.info("Sem dados (ainda).")

with col1:
    st.subheader("Alertas recentes")
    data = to_df(safe_get("/alerts", params={"limit": 200}))
    if len(st.session_state.alerts_buffer):
        data = pd.concat([to_df(st.session_state.alerts_buffer), data], ignore_index=True)
    data = data.drop_duplicates(subset=["id"], keep="first") if not data.empty else data

    if not data.empty:
        data["ts_dt"] = pd.to_datetime(data["ts"], errors="coerce")
        data["type_norm"] = data["type"].fillna("").str.lower()

        tipos = sorted(data["type_norm"].unique())
        cameras = sorted(data["camera_id"].dropna().unique())

        with st.expander("Filtros", expanded=True):
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                tipo_sel = st.multiselect(
                    "Tipo de alerta",
                    options=tipos,
                    default=tipos,
                    format_func=lambda v: v or "Sem tipo",
                )
            with col_f2:
                cam_sel = st.multiselect(
                    "Câmeras",
                    options=cameras,
                    default=cameras,
                )
            with col_f3:
                score_min = st.slider(
                    "Score mínimo",
                    min_value=float(data["score"].fillna(0).min()),
                    max_value=float(data["score"].fillna(0).max() or 1.0),
                    value=float(data["score"].fillna(0).min()),
                    step=0.01,
                )

        tipo_sel = tipo_sel or tipos  # mantém todos quando nada selecionado
        cam_sel = cam_sel or cameras

        filtered = data[
            data["type_norm"].isin(tipo_sel)
            & data["camera_id"].isin(cam_sel)
            & (data["score"].fillna(0) >= score_min)
        ]

        st.write(f"{len(filtered)} alertas dentro dos filtros")

        st.dataframe(
            filtered[["id", "ts", "camera_id", "type", "score", "clip_path"]],
            use_container_width=True,
            height=360,
        )

        st.download_button(
            "Baixar CSV",
            data=filtered[["id", "ts", "camera_id", "type", "score", "clip_path"]].to_csv(index=False).encode("utf-8"),
            file_name="alertas_filtrados.csv",
            mime="text/csv",
        )

        # player de clipe (se houver)
        clips = [""] + list(filtered["clip_path"].dropna().unique())
        sel = st.selectbox("Ver clipe", options=clips, index=0)
        if sel:
            st.video(sel)

        tab_timeline, tab_breakdown = st.tabs(["Linha do tempo", "Distribuições"])
        with tab_timeline:
            timeline = (
                filtered.dropna(subset=["ts_dt"])
                .assign(hour=lambda df: df["ts_dt"].dt.floor("H"))
                .groupby("hour")
                .size()
                .reset_index(name="alertas")
            )
            if not timeline.empty:
                chart = (
                    alt.Chart(timeline)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("hour:T", title="Hora"),
                        y=alt.Y("alertas:Q", title="Alertas"),
                        tooltip=["hour:T", "alertas:Q"],
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Sem dados suficientes para a linha do tempo.")

        with tab_breakdown:
            by_cam = (
                filtered.groupby("camera_id")
                .size()
                .reset_index(name="alertas")
                .sort_values("alertas", ascending=False)
            )
            if not by_cam.empty:
                chart = (
                    alt.Chart(by_cam)
                    .mark_bar()
                    .encode(
                        x=alt.X("alertas:Q", title="Alertas"),
                        y=alt.Y("camera_id:N", title="Câmera", sort="-x"),
                        tooltip=["camera_id:N", "alertas:Q"],
                    )
                    .properties(height=250)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Sem alertas para exibir distribuição por câmera.")
    else:
        st.info("Sem alertas recentes.")

# Rodapé
st.caption("Use SAAS_API_URL / SAAS_API_KEY (env) para apontar a API e autenticar.")