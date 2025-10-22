except Exception as e:
    st.warning(f"Banco ainda vazio ou inacessível: {e}")

# =========================
# ABA 3 - MONITORAMENTO
# =========================
with tab3:
    st.subheader("📈 Diagnóstico em Tempo Real")

    # Métricas principais do sistema
    col1, col2, col3 = st.columns(3)
    col1.metric("CPU", f"{cpu} %")
    col2.metric("Memória", f"{mem} %")
    col3.metric("Disco (/mnt/data)", f"{disk} %")

    # Logs recentes do MediaMTX e API
    st.markdown("### 🔎 Logs recentes")

    try:
        log_output = subprocess.check_output(
            ["tail", "-n", "20", "/mnt/data/logs/mediamtx.log"],
            text=True
        )
        st.text_area("MediaMTX", log_output, height=200)
    except Exception:
        st.info("Nenhum log encontrado ou arquivo ainda não criado em /mnt/data/logs/.")

    try:
        api_log = subprocess.check_output(
            ["tail", "-n", "20", "/mnt/data/logs/saas_api.log"],
            text=True
        )
        st.text_area("API FastAPI", api_log, height=200)
    except Exception:
        st.info("Logs da API ainda não disponíveis.")
