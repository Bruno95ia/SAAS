# ==========================================
# Makefile - Gerenciamento da Stack SAAS
# ==========================================

SHELL := /bin/bash
PROJECT_DIR := /mnt/data/SAAS_core/SAAS
LOG_DIR := /mnt/data/logs
PYTHON := python3

# ------------------------------------------
# Targets principais
# ------------------------------------------

start:
	@echo "🚀 Iniciando stack SAAS..."
	bash $(PROJECT_DIR)/start_saas.sh

stop:
	@echo "🧹 Encerrando stack SAAS..."
	bash $(PROJECT_DIR)/stop_saas.sh

restart:
	@echo "🔁 Reiniciando stack SAAS..."
	make stop
	sleep 3
	make start

status:
	@echo "📊 Status dos serviços ativos:"
	ps aux | egrep "uvicorn|streamlit|mediamtx" | grep -v grep || echo "Nenhum processo ativo."

logs:
	@echo "🪵 Logs da stack SAAS:"
	@echo "----------------------"
	@echo "🔸 API:"
	@tail -n 10 $(LOG_DIR)/saas_api.log 2>/dev/null || echo "sem logs"
	@echo ""
	@echo "🔸 Painel:"
	@tail -n 10 $(LOG_DIR)/saas_panel.log 2>/dev/null || echo "sem logs"
	@echo ""
	@echo "🔸 MediaMTX:"
	@tail -n 10 $(LOG_DIR)/mediamtx.log 2>/dev/null || echo "sem logs"

health:
	@echo "🧠 Testando saúde da API..."
	curl -s http://127.0.0.1:8000/health || echo "API não respondeu"

clean:
	@echo "🧽 Limpando logs antigos..."
	rm -f $(LOG_DIR)/*.log 2>/dev/null || true
	@echo "✅ Logs removidos."

# ------------------------------------------
# Diagnóstico e manutenção
# ------------------------------------------

deps:
	@echo "📦 Reinstalando dependências..."
	pip install -r $(PROJECT_DIR)/requirements.txt

check:
	@echo "🔍 Verificando diretórios essenciais..."
	@test -d $(LOG_DIR) || (echo "❌ Diretório de logs não encontrado."; exit 1)
	@test -d $(PROJECT_DIR)/src/saas || (echo "❌ Estrutura src/saas ausente."; exit 1)
	@echo "✅ Estrutura verificada com sucesso."
