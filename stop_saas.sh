#!/bin/bash
set -e

echo "🧹 Encerrando stack SAAS..."

# ==========================
# Parar processos ativos
# ==========================
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
pkill -f "mediamtx" 2>/dev/null || true
pkill -f "capture_rstp" 2>/dev/null || true
pkill -f "live_yolo" 2>/dev/null || true

# ==========================
# Confirmar encerramento
# ==========================
sleep 3
echo "🧾 Processos restantes:"
ps aux | egrep "uvicorn|streamlit|mediamtx|capture|yolo" | grep -v grep || true

# ==========================
# Limpeza opcional de logs
# ==========================
if [ "$1" == "--clean" ]; then
    echo "🧽 Limpando logs antigos..."
    rm -f /var/log/saas_api.log /var/log/saas_panel.log /var/log/mediamtx.log 2>/dev/null || true
    echo "🗑️ Logs removidos."
fi

echo "✅ Stack SAAS totalmente encerrada."

