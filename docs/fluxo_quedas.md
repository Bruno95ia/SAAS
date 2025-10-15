# Fluxo de Detecção de Quedas e Entrega de Alertas

Este documento resume como os componentes atuais trabalham em conjunto para detectar quedas, gerar clipes de evidência e entregar os alertas ao painel de monitoramento.

## 1. Captura contínua da câmera

* Script: `src/saas/capture_rstp.py`
* Uso: `python -m saas.capture_rstp --camera CAM01 --rtsp rtsp://usuario:senha@ip:porta/stream --out runs/buffer`
* Função: grava segmentos (`.m4s`) de duração curta (padrão 2s) em pastas por data. Isso mantém um "ring buffer" organizado para posterior montagem de clipes.

## 2. Pipeline online com YOLOv8

* Script: `src/saas/live_yolo.py`
* Uso: `python -m saas.live_yolo --camera CAM01 --rtsp rtsp://... --buffer runs/buffer --api-url http://127.0.0.1:8000 --api-key dev-key`
* Função: lê o stream, roda o YOLOv8 Pose, calcula ângulo do tronco, velocidade vertical e achatamento do bounding box. Opcionalmente roda o classificador temporal (TCN).
* Critérios:
  - Queda provável quando tronco gira acima de `--theta-deg` e a velocidade vertical normalizada (`vy`) é maior que `--vy-min`.
  - Confirmação ocorre quando a pessoa permanece deitada (`flat_ratio`) por `--flat-sec` segundos.

## 3. Montagem do clipe e anotação

* Função: `saas.clipper.collect_clip`
* Quando uma queda é confirmada, o pipeline busca os segmentos correspondentes ao intervalo `[pre, post]` em torno do evento.
* Os segmentos são concatenados e um clipe definitivo é cortado com FFmpeg (ou OpenCV como fallback).
* Função: `saas.annotate.annotate_clip` desenha a caixa da pessoa no clipe final para facilitar a revisão.

## 4. Publicação do alerta

* O `live_yolo` chama `requests.post` para `POST /alerts` da API (`src/saas/api.py`).
* Payload inclui:
  - `camera_id`
  - `type="fall"`
  - `score` (confiança do detector)
  - `clip_path` apontando para o arquivo anotado em `runs/clips` servido via FastAPI
  - `extra` com métricas da detecção (`angle_deg`, `vy_norm`, `tcn_prob`)

## 5. API e armazenamento

* A API (FastAPI) valida a chave `X-API-Key`, salva o alerta no banco SQLite (`events.db`) e serve os clipes em `/clips`.
* WebSocket `/ws` transmite alertas em tempo real para o painel.

## 6. Painel de monitoramento

* App Streamlit (`painel.py`).
* Consulta periódica a `/alerts` e, opcionalmente, assina o WebSocket.
* Exibe tabela de alertas, métricas do dia e player de vídeo usando o `clip_path` recebido.

## 7. Checklist rápido

1. Rodar o capturador RTSP para cada câmera.
2. Executar o pipeline YOLO ao vivo apontando para o mesmo `--buffer`.
3. Subir a API (`uvicorn saas.api:app --reload`).
4. Ajustar `SAAS_API_URL` e `SAAS_API_KEY` no painel e abrir `streamlit run painel.py`.

Com este fluxo ativo, quedas detectadas são registradas com os respectivos clipes anotados e apresentados no painel de monitoramento.
