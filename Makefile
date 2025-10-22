.PHONY: help install api panel health alerts kill-api run-pipeline live-yolo extract-yolo-feats train-tcn test lint format

API_URL ?= http://127.0.0.1:8000
API_KEY ?= minha-chave-forte
STREAMLIT_PORT ?= 8501
PYTHON ?= python3
CAMERA ?= cam01

help:
	@echo "Targets disponíveis:"
	@echo "  make install          # instala dependências"
	@echo "  make api              # inicia a API FastAPI (porta 8000)"
	@echo "  make panel            # inicia o painel Streamlit (porta 8501)"
	@echo "  make health           # consulta /health"
	@echo "  make alerts           # lista alertas recentes"
	@echo "  make run-pipeline     # anota vídeos e (opcional) publica alertas"
	@echo "  make live-yolo        # executa inferência ao vivo"
	@echo "  make test             # roda a suíte de testes"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

api:
	SAAS_API_KEY=$(API_KEY) SAAS_API_URL=$(API_URL) PYTHONPATH=src \
		uvicorn saas.api:app --host 0.0.0.0 --port 8000 --reload

panel:
	SAAS_API_URL=$(API_URL) SAAS_API_KEY=$(API_KEY) \
		streamlit run painel.py --server.port $(STREAMLIT_PORT) --server.address 0.0.0.0

health:
	curl -fsS $(API_URL)/health | jq .

alerts:
	curl -fsS -H "X-API-Key: $(API_KEY)" $(API_URL)/alerts | jq .

kill-api:
	- pkill -f "uvicorn saas.api:app" || true
	@echo "uvicorn finalizado (se existia)."

run-pipeline:
	SAAS_API_URL=$(API_URL) SAAS_API_KEY=$(API_KEY) PYTHONPATH=src \
		python -m saas.run_pipeline -i runs/clips --post --camera $(CAMERA)

live-yolo:
	SAAS_API_URL=$(API_URL) SAAS_API_KEY=$(API_KEY) PYTHONPATH=src \
		python -m saas.live_yolo --camera $(CAMERA) --rtsp "$(RTSP)" --buffer runs/buffer/$(CAMERA)

extract-yolo-feats:
	PYTHONPATH=src python -m saas.batch_yolo_extract -i runs/clips --pattern "*.mp4" --weights yolov8n-pose.pt

train-tcn:
	PYTHONPATH=src python -m saas.train_tcn_yolo --labels labels.csv --feats runs/feats --out runs/models --epochs 25

test:
	PYTHONPATH=src pytest -q

lint:
	ruff check src tests

format:
	ruff format src tests
