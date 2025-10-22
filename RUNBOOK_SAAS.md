# 🧠 RUNBOOK — SAAS Infra & IA Pipeline

## 📁 Estrutura do Projeto
/mnt/data/SAAS_core/SAAS
│
├── start_saas.sh         # inicialização completa da stack
├── stop_saas.sh          # encerramento seguro
├── Makefile              # automação via comandos make
├── requirements.txt       # dependências Python
├── src/saas/
│   ├── api.py             # API FastAPI (porta 8000)
│   ├── panel.py           # painel Streamlit (porta 8501)
│   ├── capture_rstp.py    # captura RTSP / vídeo local
│   ├── weights/best.pt    # modelo YOLO
│   ├── runs/              # frames e inferências
│   └── events.db          # banco SQLite
└── /mnt/data/logs         # logs operacionais

## ⚙️ Procedimentos Principais

### ▶️ Iniciar stack
make start

### ⏹️ Parar stack
make stop

### 🔁 Reiniciar
make restart

### 📊 Status
make status

### 🪵 Logs
make logs

## 🧩 Diagnóstico e Saúde

### Verificar API
make health
# ou
curl http://127.0.0.1:8000/health

### Verificar painel
http://52.14.83.142:8501

### Diagnóstico completo
make check

## 🧠 Testes de Inferência

### Vídeo local
1. Suba um vídeo em /mnt/data/sample.mp4.
2. No painel, informe: /mnt/data/sample.mp4
3. Clique Iniciar → os frames e labels aparecem em tempo real.

### Stream RTSP
rtsp://admin:SENHA@192.168.0.101:554/Streaming/Channels/101

## 🧹 Manutenção

### Limpar logs
make clean

### Reinstalar dependências
make deps

### Backup de runs e banco
tar -czf /mnt/data/backups/saas_$(date +%F).tar.gz /mnt/data/saas/runs /mnt/data/SAAS_core/SAAS/src/saas/events.db

## ⚠️ Problemas Comuns
| Sintoma | Causa provável | Solução |
|----------|----------------|----------|
| ModuleNotFoundError: src | Execução fora da raiz | Use --app-dir /mnt/data/SAAS_core/SAAS/src |
| Invalid value: panel.py | Caminho relativo errado | Use caminho absoluto |
| Permission denied /var/log | Logs em diretório sem permissão | Use /mnt/data/logs |
| Painel não carrega | cv2 ausente | pip install opencv-python-headless |
| / 99% cheio | Loops em runs/ | find /mnt/data -type d -name runs -mindepth 2 -exec rm -rf {} + |

## 🧾 Logs e Monitoramento
| Componente | Log | Local |
|-------------|-----|--------|
| MediaMTX | /mnt/data/logs/mediamtx.log | Atividade RTSP |
| API FastAPI | /mnt/data/logs/saas_api.log | Requisições e erros |
| Painel Streamlit | /mnt/data/logs/saas_panel.log | Eventos e status UI |

## 📦 Atualização de versão
make stop && cd /mnt/data/SAAS_core/SAAS && git pull && make start

## 🧰 Recuperação rápida
make stop && make clean && make start
