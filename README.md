# 🧠 SAAS – Plataforma de Detecção de Quedas

Sistema completo para monitoramento de quedas com IA, composto por API FastAPI, painel Streamlit e banco PostgreSQL. O deploy é feito com Docker Compose, garantindo persistência no disco `/mnt/data` e suporte ao modelo YOLOv8.

## 📦 Arquitetura

| Serviço       | Tecnologias                      | Porta | Descrição |
| ------------- | -------------------------------- | ----- | --------- |
| `saas-db`     | PostgreSQL 15                    | 5432  | Banco relacional com SQLAlchemy/Alembic |
| `saas-api`    | FastAPI + Ultralytics YOLOv8     | 8000  | API REST, captura RTSP, processamento e geração de eventos |
| `saas-painel` | Streamlit                        | 8501  | Dashboard em tempo real com métricas, controle e visualização |
| `saas-data`   | Python utility (opcional)        | -     | Contêiner auxiliar para sincronização de datasets |

Os serviços compartilham `/mnt/data/saas_data` para logs, clips e datasets, e o PostgreSQL persiste em `/mnt/data/pgdata`.

## 🚀 Pré-requisitos

* Docker e Docker Compose instalados (Ubuntu 24.04).
* Diretórios locais criados:
  ```bash
  sudo mkdir -p /mnt/data/saas_data /mnt/data/pgdata
  sudo chown -R $USER:$USER /mnt/data/saas_data /mnt/data/pgdata
  ```
* GPU opcional com drivers/NVIDIA Container Toolkit (para aceleração YOLOv8).

## ⚙️ Configuração

1. Clone o repositório no servidor (ex.: `/mnt/data/SAAS`).
2. Configure variáveis no arquivo `.env` (já incluído):
   ```env
   POSTGRES_USER=saas
   POSTGRES_PASSWORD=saas
   POSTGRES_DB=saas
   POSTGRES_HOST=db
   POSTGRES_PORT=5432
   DATA_DIR=/data
   ROBOFLOW_API_KEY=   # opcional para sincronizar datasets
   HF_TOKEN=           # opcional para datasets no Hugging Face
   ```
3. Ajuste permissões do diretório `/mnt/data/saas_data` caso necessário (logs e clips são escritos em `/mnt/data/saas_data`).

## ▶️ Execução

```bash
# Dentro de /mnt/data/SAAS
docker compose up -d --build
```

Isso provisiona automaticamente os serviços `saas-db`, `saas-api` e `saas-painel`. O contêiner `saas-data` é opcional e pode ser iniciado com `docker compose --profile data up -d` caso deseje executar scripts de sincronização manualmente.

### Logs e Persistência
* API grava logs em `/mnt/data/saas_data/logs/saas-api.log`.
* Clips de eventos são armazenados em `/mnt/data/saas_data/clips`.
* Base PostgreSQL persiste em `/mnt/data/pgdata`.

## 🌐 Endpoints principais (FastAPI)

| Método | Endpoint              | Descrição |
| ------ | --------------------- | --------- |
| POST   | `/cameras`            | Cadastra uma câmera RTSP |
| GET    | `/cameras`            | Lista câmeras cadastradas |
| POST   | `/cameras/{id}/start` | Inicia captura e detecção |
| POST   | `/cameras/{id}/stop`  | Pausa captura |
| GET    | `/stream/{id}`        | Stream MJPEG com boxes/labels |
| GET    | `/metrics`            | Métricas agregadas, status das câmeras e últimas quedas |

Para testar:
```bash
curl -X POST http://localhost:8000/cameras \
  -H 'Content-Type: application/json' \
  -d '{"name": "Entrada", "rtsp": "rtsp://usuario:senha@ip/stream", "enabled": true}'
```

## 📊 Painel Streamlit

* Acesse: [http://localhost:8501](http://localhost:8501)
* Recursos:
  * Contagem de câmeras ativas e quedas detectadas.
  * FPS médio, tempo médio entre quedas, uso de CPU/GPU e espaço livre.
  * Gráfico de eventos por hora (últimas 24h).
  * Histórico das 10 últimas quedas com exportação CSV.
  * Controle para adicionar/iniciar/parar câmeras.
  * Visualização ao vivo utilizando o endpoint `/stream/{id}`.

## 🧠 IA de Detecção

* Modelo padrão: `yolov8n-pose.pt` (Ultralytics 8.3.40).
* A detecção identifica pessoas e aplica heurística para quedas (bounding box com orientação horizontal ou label `fall`).
* Clips de 10s são gerados para cada evento.
* O código tenta usar GPU automaticamente quando disponível (`torch.cuda.is_available()`), com fallback para CPU.

## 🗄️ Banco de Dados & ORM

* PostgreSQL 15 com SQLAlchemy 2.0 e migração automática (criação de schema na inicialização).
* Tabelas principais:
  * `cameras`: cadastro de câmeras RTSP.
  * `events`: registros de eventos de queda, timestamp e score.
  * `frame_labels`: bounding boxes por frame de evento.

## 🔌 Conectores de Dados

O módulo `src/data_connector.py` oferece funções para sincronizar datasets externos para `/mnt/data/saas_data/datasets`:

```python
from data_connector import sync_roboflow, sync_huggingface
sync_roboflow("org/projeto", "version")
sync_huggingface("autor/dataset", "path/arquivo.zip")
```
Configure `ROBOFLOW_API_KEY` ou `HF_TOKEN` no `.env` para habilitar.

## ✅ Checklist de Validação

1. `docker compose up -d` sobe todos os serviços.
2. Acesse o painel em `http://<servidor>:8501`.
3. Adicione uma câmera RTSP pelo painel ou API.
4. Visualize a transmissão com boxes/labels e confirme registro de eventos.
5. Verifique logs em `/mnt/data/saas_data/logs/saas-api.log` e clips em `/mnt/data/saas_data/clips`.

## 🤝 Contribuição

Pull requests são bem-vindos! Certifique-se de executar `docker compose build` após alterações em dependências e atualizar a documentação quando necessário.
