# SAAS – Runbook Operacional

## Pré-requisitos

| Componente | Versão sugerida | Observações |
|------------|-----------------|-------------|
| Python     | 3.10 – 3.12     | Utilize `python3 -m venv venv` para isolar dependências. |
| FFmpeg     | 6.x             | Necessário para captura RTSP/screen e cortes de clipes. |
| CUDA (opcional) | 12.x + drivers | Habilita aceleração no YOLO/ONNX. A aplicação cai para CPU automaticamente. |
| jq         | 1.6             | Usado por `make health` e `make alerts`. |

Certifique-se também de ter acesso de escrita ao diretório `runs/` e porta 8000 (API) e 8501 (painel) liberadas no firewall.

## Estrutura de diretórios

```
SAAS/
├── runs/
│   ├── buffer/        # segmentos gravados por câmera (gerados pelo CaptureManager)
│   ├── clips/         # clipes finais servidos pela API
│   ├── logs/          # arquivos *.log, api.out, panel.out
│   └── results/       # artefatos diversos (modelos, métricas)
├── src/saas/          # código-fonte (API, pipelines, utilitários)
├── start_saas.sh      # inicia API + painel em background
├── stop_saas.sh       # encerra os serviços
├── requirements.txt   # dependências de runtime
└── RUNBOOK_SAAS.md    # este documento
```

## Setup local

1. **Criar ambiente virtual**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Configurar variáveis** (opcional)
   ```bash
   export SAAS_API_KEY="minha-chave-forte"
   export SAAS_API_URL="http://127.0.0.1:8000"
   export SAAS_YOLO_WEIGHTS="/caminho/para/yolov8n.pt"  # se desejar sobrescrever
   ```
3. **Criar diretórios** (automaticamente gerados, mas pode ser feito manualmente)
   ```bash
   mkdir -p runs/buffer runs/clips runs/logs runs/results weights
   ```
4. **Subir serviços**
   ```bash
   ./start_saas.sh
   ```
   - API acessível em http://127.0.0.1:8000/health
   - Painel Streamlit em http://127.0.0.1:8501

5. **Parar serviços**
   ```bash
   ./stop_saas.sh
   ```

## Setup em servidor Ubuntu 22.04

1. **Atualizar código**
   ```bash
   cd /opt/SAAS
   git pull origin main
   ```
2. **Ambiente virtual**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Variáveis de ambiente** – configure em `/etc/environment` ou systemd (API key, RTSPs, etc.).
4. **Iniciar serviços** (idealmente via systemd; abaixo comando direto):
   ```bash
   ./stop_saas.sh || true
   ./start_saas.sh
   ```
5. **Monitorar logs**
   ```bash
   tail -f runs/logs/api.out runs/logs/panel.out runs/logs/saas.log
   ```

## Comandos essenciais

| Ação | Comando |
|------|---------|
| Rodar testes | `make test` |
| Checar saúde da API | `make health` |
| Listar alertas | `make alerts` |
| Executar inferência live | `make live-yolo RTSP=rtsp://... CAMERA=cam01` |
| Pipeline offline | `make run-pipeline CAMERA=cam01` |

### Healthcheck (`/health`)
Resposta exemplo:
```json
{
  "status": "ok",
  "components": {
    "database": {"status": "ok", "details": {"path": ".../events.db", "count": 42}},
    "directories": {"status": "ok", "details": {"runs": {"exists": true}, ...}},
    "weights": {"status": "warn", "details": {"exists": false}}
  }
}
```
Use `make health` (requer `jq`) para visualizar.

## Captura e inferência

### CaptureManager (RTSP / webcam / tela)

```bash
PYTHONPATH=src python -m saas.capture_rstp \
  --camera cam01 \
  --rtsp rtsp://usuario:senha@host/stream \
  --segment 2 --reconnect 5
```

- **Webcam USB:** `--rtsp "local:/dev/video0,fps=30,size=1280x720"`
- **Tela (Linux/X11):** `--rtsp "screen:display=:0.0,size=1920x1080,fps=25"`
- **Tela (macOS):** `--rtsp screen` (usa AVFoundation padrão)

Os segmentos `.m4s` serão salvos em `runs/buffer/<camera>/AAAAmmdd/HHMMSS.m4s`.

### Live YOLO

```bash
PYTHONPATH=src python -m saas.live_yolo \
  --camera cam01 \
  --rtsp "rtsp://..." \
  --buffer runs/buffer/cam01 \
  --weights weights/yolov8n.pt \
  --display
```

- `--display` abre janela com bounding box, rótulo e confiança.
- `--use-tcn` habilita o modelo temporal (`runs/models/tcn.onnx`).
- Alertas confirmados geram clipes anotados e POST na API com `angle_deg`, `vy_norm` e `tcn_prob` em `extra`.

## Logs

- Logs estruturados (`loguru`) em `runs/logs/saas.log` (rotacionado 10 MB, retenção 14 dias).
- Saída da API: `runs/logs/api.out`
- Saída do painel: `runs/logs/panel.out`
- Use `tail -f` ou `less +F` para acompanhar.

## Troubleshooting

### Captura (RTSP / webcam / tela)
- **Falha de conexão**: verifique URL, credenciais e firewall. FFmpeg tenta reconectar a cada `--reconnect` segundos.
- **Webcam não abre**: confirme dispositivo (`/dev/video*`) e permissões (`sudo usermod -aG video <user>`).
- **Tela Linux**: exporte `DISPLAY=:0.0` e garanta que o usuário possua acesso (`xhost +local:`).

### YOLO / inferência
- Certifique-se de que `weights/yolov8n.pt` existe ou aponte `SAAS_YOLO_WEIGHTS`.
- Em GPUs Nvidia, valide `nvidia-smi`. Caso contrário, o Ultralytics usa CPU automaticamente.
- Ajuste `--conf`, `--theta-deg`, `--vy-min` conforme necessário.

### Banco de dados (SQLite)
- Verifique permissões do arquivo `events.db`.
- Utilize `sqlite3 events.db "SELECT COUNT(*) FROM alerts;"` para inspeção manual.
- Healthcheck retornará `status=error` se não conseguir abrir o banco.

### Painel / rede
- Caso `make health` falhe, verifique se a API está em execução (`ps aux | grep uvicorn`).
- Se o painel não conectar ao WebSocket, defina `SAAS_WS=0` para desabilitar tempo real ou instale `websocket-client`.
- Ajuste `SAAS_API_URL` para apontar ambientes remotos.

## Checklist pós-deploy

- [ ] `make health` retorna `status: ok` ou `warn` apenas em componentes opcionais.
- [ ] Painel acessível e exibindo alertas com colunas de métricas (score, ângulo, velocidade, TCN).
- [ ] Clipes gerados acessíveis em `http://<host>:8000/clips/<arquivo>.mp4`.
- [ ] Logs (`runs/logs/saas.log`, `api.out`, `panel.out`) sem erros críticos.
- [ ] Banco `events.db` registrando novos alertas (`sqlite3 events.db 'select count(*) from alerts;'`).
- [ ] Serviços iniciados via `./start_saas.sh` com PIDs registrados em `.saas_api.pid` e `.saas_panel.pid`.

## Fluxo de validação ponta-a-ponta

1. `./start_saas.sh`
2. Capturar stream (ex.: `python -m saas.capture_rstp --camera cam01 --rtsp rtsp://...`).
3. Executar live YOLO com `--display` e forçar um evento (queda simulada).
4. Confirmar alerta no painel e em `make alerts`.
5. Verificar clipe anotado em `runs/clips/` e via endpoint `/clips/<arquivo>`.

---
**Comando rápido (ponta a ponta em ambiente local):**
```bash
./start_saas.sh && \ 
PYTHONPATH=src python -m saas.capture_rstp --camera cam01 --rtsp "local:/dev/video0" --segment 2 & \ 
PYTHONPATH=src python -m saas.live_yolo --camera cam01 --rtsp "local:/dev/video0" --display
```
> Ajuste `--rtsp` conforme a sua fonte de vídeo.
