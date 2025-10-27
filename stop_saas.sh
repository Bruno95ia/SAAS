#!/bin/bash
set -euo pipefail

pkill -f "uvicorn saas.api" 2>/dev/null || true
pkill -f "streamlit run" 2>/dev/null || true
pkill -f "python -m saas.run_pipeline" 2>/dev/null || true
pkill -f "mediamtx" 2>/dev/null || true

sleep 2

echo "Processos SAAS ativos:" 
ps aux | egrep "uvicorn|streamlit|saas.run_pipeline|mediamtx" | grep -v grep || echo "Nenhum processo SAAS em execução."

echo "✅ Stack SAAS totalmente encerrada."
