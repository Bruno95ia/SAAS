#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
LOG_DIR="/mnt/data/SAAS/runs/logs"
VENV_DIR="$ROOT_DIR/.venv"
APP_DIR="$ROOT_DIR/src"

mkdir -p "$LOG_DIR"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip >/dev/null
pip install -q -r "$ROOT_DIR/requirements.txt" >/dev/null

export SAAS_API_URL="http://0.0.0.0:8000"
export SAAS_API_KEY="saas-poc-key"

pkill -f "uvicorn" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
pkill -f "saas.run_pipeline" 2>/dev/null || true
pkill -f "mediamtx" 2>/dev/null || true

if command -v mediamtx >/dev/null; then
  nohup mediamtx >"$LOG_DIR/mediamtx.log" 2>&1 &
  echo "MediaMTX iniciado (logs em $LOG_DIR/mediamtx.log)"
else
  echo "MediaMTX não encontrado, seguindo sem ele."
fi

nohup uvicorn saas.api:app --app-dir "$APP_DIR" --host 0.0.0.0 --port 8000 >"$LOG_DIR/api.log" 2>&1 &
nohup streamlit run "$APP_DIR/saas/panel.py" --server.address 0.0.0.0 --server.port 8501 >"$LOG_DIR/panel.log" 2>&1 &
nohup python -m saas.run_pipeline >"$LOG_DIR/pipeline.log" 2>&1 &

sleep 5

echo "✅ Stack SAAS totalmente inicializada!"
echo "📍 API:    http://0.0.0.0:8000"
echo "📍 Painel: http://0.0.0.0:8501"
echo "📁 Logs:   $LOG_DIR"
