#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON:-python3}
API_PORT=${API_PORT:-8000}
PANEL_PORT=${PANEL_PORT:-8501}
LOG_DIR="$ROOT_DIR/runs/logs"
mkdir -p "$LOG_DIR"

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export SAAS_API_URL=${SAAS_API_URL:-http://127.0.0.1:$API_PORT}

start_service() {
  local cmd="$1"
  local pid_file="$2"
  local log_file="$3"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
    echo "Processo já em execução (PID $(cat "$pid_file")). Pare antes de iniciar novamente." >&2
    return 0
  fi
  echo "Iniciando: $cmd"
  nohup bash -c "$cmd" >>"$log_file" 2>&1 &
  echo $! >"$pid_file"
}

start_service \
  "$PYTHON_BIN -m uvicorn saas.api:app --host 0.0.0.0 --port $API_PORT" \
  "$ROOT_DIR/.saas_api.pid" \
  "$LOG_DIR/api.out"

start_service \
  "$PYTHON_BIN -m streamlit run painel.py --server.port $PANEL_PORT --server.address 0.0.0.0" \
  "$ROOT_DIR/.saas_panel.pid" \
  "$LOG_DIR/panel.out"

cat <<EOF
SAAS iniciado.
  API.......: http://0.0.0.0:$API_PORT (log: $LOG_DIR/api.out)
  Painel....: http://0.0.0.0:$PANEL_PORT (log: $LOG_DIR/panel.out)
EOF
