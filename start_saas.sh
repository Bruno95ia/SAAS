#!/bin/bash
set -e

echo "🔧 Iniciando stack SAAS..."

# ==========================
# Diretório de logs
# ==========================
LOG_DIR="/mnt/data/logs"
if [ ! -d "$LOG_DIR" ]; then
  echo "📁 Criando diretório de logs..."
  sudo mkdir -p "$LOG_DIR"
  sudo chmod -R 777 "$LOG_DIR"
fi

# ==========================
# Ambiente e dependências
# ==========================
if [ -d "/mnt/data/SAAS/saas_venv" ]; then
  source /mnt/data/SAAS/saas_venv/bin/activate
else
  echo "⚙️ Criando ambiente virtual..."
  python3 -m venv /mnt/data/SAAS/saas_venv
  source /mnt/data/SAAS/saas_venv/bin/activate
fi

echo "📦 Instalando dependências..."
pip install -q --upgrade pip >/dev/null
pip install -q opencv-python-headless==4.10.0.84 >/dev/null
pip install -q -r /mnt/data/SAAS_core/SAAS/requirements.txt >/dev/null

# ==========================
# Variáveis de ambiente
# ==========================
export SAAS_API_URL="http://127.0.0.1:8000"
export SAAS_API_KEY="minha-chave-forte"
export SAAS_DB_PATH="/mnt/data/SAAS_core/SAAS/src/saas/events.db"

# ==========================
# Parar instâncias antigas
# ==========================
echo "🧹 Encerrando processos antigos..."
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "streamlit" 2>/dev/null || true
pkill -f "mediamtx" 2>/dev/null || true

# ==========================
# Subir MediaMTX
# ==========================
echo "📡 Iniciando MediaMTX..."
nohup /usr/local/bin/mediamtx -config /usr/local/bin/mediamtx.yml >"$LOG_DIR/mediamtx.log" 2>&1 &

# ==========================
# Subir API (corrigido com caminho absoluto)
# ==========================
echo "⚙️ Iniciando API..."
nohup uvicorn saas.api:app --app-dir /mnt/data/SAAS_core/SAAS/src --host 0.0.0.0 --port 8000 >"$LOG_DIR/saas_api.log" 2>&1 &

# ==========================
# Subir painel Streamlit (corrigido com caminho absoluto)
# ==========================
echo "🧠 Iniciando painel Streamlit..."
nohup streamlit run /mnt/data/SAAS_core/SAAS/src/saas/panel.py --server.port 8501 --server.address 0.0.0.0 >"$LOG_DIR/saas_panel.log" 2>&1 &

# ==========================
# Confirmação
# ==========================
sleep 5
echo "✅ Stack SAAS totalmente inicializada!"
echo "📍 API:     http://127.0.0.1:8000"
echo "📍 Painel:  http://127.0.0.1:8501"
echo "📁 Logs:    $LOG_DIR"
# --- AUTO PUSH GIT ---
echo "[Git] Sincronizando alterações com o repositório remoto..."
cd /mnt/data/SAAS_core/SAAS
git add .
git commit -m "Auto push: atualização automática do servidor em $(date '+%Y-%m-%d %H:%M:%S')" || true
git pull origin main --allow-unrelated-histories --strategy-option ours || true
git push origin main --force
echo "[Git] Sincronização concluída com sucesso."
