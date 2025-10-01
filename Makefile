# ===============================
# VARIÁVEIS GLOBAIS
# ===============================
PORT := $(shell grep -E 'metrics_port:' config.yaml | head -n1 | sed -E 's/.*metrics_port:\s*([0-9]+).*/\1/')

SHELL := /bin/bash
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# ===============================
# VENV / DEPENDÊNCIAS
# ===============================
.PHONY: venv install dev app run stop restart status logs logs-follow clean clean-snapshots fmt lint fix check

venv:
	python3 -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate ; \
	$(PY) -m pip install --upgrade pip wheel setuptools ; \
	$(PIP) install -r requirements.txt

# ===============================
# EXECUÇÃO DO APP
# ===============================
dev:
	@echo "== Rodando em modo DEV (Ctrl+C para parar) =="
	. $(VENV)/bin/activate ; \
	$(PY) app.py

app:
	@echo "== Rodando aplicação principal com config.yaml =="
	. $(VENV)/bin/activate ; \
	$(PY) app.py

run: app

stop:
	- lsof -t -i :$(PORT) | xargs -r kill -9

restart:
	$(MAKE) stop
	$(MAKE) run

status:
	@echo "== Status da porta $(PORT) =="
	- lsof -i :$(PORT) || true

logs:
	@echo "== Últimos 20 eventos do log =="
	tail -n 20 falls.log || echo "Sem logs ainda. Rode 'make run' primeiro."

logs-follow:
	@echo "== Acompanhando log em tempo real (Ctrl+C para parar) =="
	tail -f falls.log

# ===============================
# LIMPEZA / FORMATAÇÃO
# ===============================
clean:
	@echo "== Limpando __pycache__, logs e snapshots =="
	-find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	-rm -f falls.log
	-rm -rf runs alerts

clean-snapshots:
	@echo "== Limpando snapshots antigos =="
	. $(VENV)/bin/activate ; \
	$(PY) -c "from yaml import safe_load; from ioutils.storage import purge_old; cfg=safe_load(open('config.yaml','r',encoding='utf-8')); retain=cfg.get('privacy',{}).get('retain_hours',48); purge_old(outdir='alerts', retain_hours=retain); print(f'OK: snapshots com mais de {retain}h removidos.')"

fmt:
	@echo "== Formatando com Black e organizando imports com Ruff =="
	$(PIP) install black ruff
	$(VENV)/bin/black .
	$(VENV)/bin/ruff check . --select I --fix

lint:
	@echo "== Analisando código com Ruff (sem alterar arquivos) =="
	$(PIP) install ruff
	$(VENV)/bin/ruff check .

fix:
	@echo "== Corrigindo problemas auto-fixáveis com Ruff =="
	$(PIP) install ruff
	$(VENV)/bin/ruff check . --fix

check:
	@echo "== Verificando formatação (Black) e lint (Ruff) =="
	$(PIP) install black ruff
	$(VENV)/bin/black --check .
	$(VENV)/bin/ruff check .

# ===============================
# YOLO / DATASET WORKFLOW
# ===============================
.PHONY: dataset dataset-clean dataset-cvat dataset-split yolo-train yolo-resume yolo-predict yolo-export-onnx postprocess clean-runs

# Parâmetros padrão
DATASET_DIR ?= datasets/queda_caso1_yolo
DATA_YAML   ?= $(DATASET_DIR)/data.yaml
EPOCHS      ?= 50
IMG         ?= 640
CONF        ?= 0.5
SRC         ?= $(DATASET_DIR)/images
RUN_DIR     ?= runs/detect/train
BEST_PT     ?= $(RUN_DIR)/weights/best.pt
LAST_PT     ?= $(RUN_DIR)/weights/last.pt
OUT_ONNX    ?= models/falls_yolov8.onnx

# Gerar dataset YOLO a partir de vídeos
dataset:
	@[ -n "$(VIDEOS_DIR)" ] || (echo "Defina VIDEOS_DIR, ex: make dataset VIDEOS_DIR=./meus_videos FPS=5"; exit 1)
	. $(VENV)/bin/activate ; \
	$(PIP) install tqdm ; \
	$(PY) scripts/gen_dataset_from_video.py --videos_dir "$(VIDEOS_DIR)" --out_dir dataset --fps $${FPS:-5} --imgsz $${IMGSZ:-640} --splits 0.7 0.2 0.1 --seed 42

dataset-clean:
	rm -rf dataset

# Converter export do CVAT
dataset-cvat:
	@[ -n "$(ZIP)" ] || (echo "Use: make dataset-cvat ZIP=./export.zip [OUT=./datasets/saida]"; exit 1)
	. $(VENV)/bin/activate ; \
	$(PY) scripts/cvat_to_yolo.py --zip "$(ZIP)" --out $${OUT:-$(DATASET_DIR)}

# Split dataset (80/10/10)
dataset-split:
	. $(VENV)/bin/activate ; \
	$(PIP) install -q ultralytics ; \
	yolo data split source=$(DATASET_DIR) seed=42 ratios=0.8 0.1 0.1

# Treino YOLO
yolo-train:
	. $(VENV)/bin/activate ; \
	$(PIP) install -q ultralytics ; \
	yolo detect train data=$(DATA_YAML) model=yolov8n.pt epochs=$(EPOCHS) imgsz=$(IMG)

# Retomar treino
yolo-resume:
	@[ -f "$(LAST_PT)" ] || (echo "Checkpoint não encontrado em $(LAST_PT). Rode 'make yolo-train' antes."; exit 1)
	. $(VENV)/bin/activate ; \
	$(PIP) install -q ultralytics ; \
	yolo detect train resume model=$(LAST_PT) epochs=$(EPOCHS)

# Predição
yolo-predict:
	@[ -n "$(SRC)" ] || (echo "Use: make yolo-predict SRC=./caminho"; exit 1)
	@[ -f "$(BEST_PT)" ] || (echo "Modelo não encontrado em $(BEST_PT). Rode 'make yolo-train' antes."; exit 1)
	. $(VENV)/bin/activate ; \
	$(PIP) install -q ultralytics ; \
	yolo detect predict model=$(BEST_PT) source="$(SRC)" conf=$(CONF) project=runs/detect name=pred_manual exist_ok=True

# Exportar para ONNX
yolo-export-onnx:
	@[ -f "$(BEST_PT)" ] || (echo "Modelo não encontrado em $(BEST_PT). Rode 'make yolo-train' antes."; exit 1)
	. $(VENV)/bin/activate ; \
	$(PIP) install -q ultralytics onnx onnxruntime ; \
	yolo export model=$(BEST_PT) format=onnx opset=12 ; \
	mkdir -p models ; \
	mv $$(dirname $(BEST_PT))/best.onnx $(OUT_ONNX) ; \
	echo "ONNX salvo em $(OUT_ONNX)"

# Pós-processamento temporal
postprocess:
	. $(VENV)/bin/activate ; \
	$(PY) scripts/postprocess_falls.py

# Limpar runs
clean-runs:
	-rm -rf runs/detect/train runs/detect/predict*
	# ===============================
# ACTIVE LEARNING (Mineração de incertezas)
# ===============================
ACTIVE_OUT    ?= active_learning/uncertain
ACTIVE_LOW    ?= 0.4
ACTIVE_HIGH   ?= 0.6
ACTIVE_STRIDE ?= 3
ACTIVE_MAX    ?= 50

active-learning:
	. $(VENV)/bin/activate ; \
	$(PIP) install -q ultralytics ; \
	$(PY) scripts/select_uncertain.py \
	  --model $(BEST_PT) \
	  --source ./data \
	  --out $(ACTIVE_OUT) \
	  --class-name Queda \
	  --low $(ACTIVE_LOW) --high $(ACTIVE_HIGH) \
	  --stride $(ACTIVE_STRIDE) --max-per-video $(ACTIVE_MAX) \
	  --save-json
# ===============================
# RELATÓRIOS / VALIDAÇÃO
# ===============================
REPORT_XLSX ?= reports/validacao_quedas.xlsx

validation-sheet:
	. $(VENV)/bin/activate ; \
	$(PIP) install -q pandas openpyxl ; \
	$(PY) scripts/make_validation_sheet.py --out $(REPORT_XLSX) ; \
	echo ">> Planilha gerada em $(REPORT_XLSX)"
	# ===============================
# MODO DE EXECUÇÃO POR PERFIL DE CONFIG
# ===============================
.PHONY: local rtsp

CFG_BACKUP := config.yaml.bak

local:
	@echo "== Modo LOCAL (./data) =="
	@[ -f config.local.yaml ] || (echo "Crie config.local.yaml primeiro"; exit 1)
	@cp config.yaml $(CFG_BACKUP) 2>/dev/null || true
	cp config.local.yaml config.yaml
	$(MAKE) run
	@mv $(CFG_BACKUP) config.yaml 2>/dev/null || true

rtsp:
	@echo "== Modo RTSP (câmera IP) =="
	@[ -f config.rtsp.yaml ] || (echo "Crie config.rtsp.yaml primeiro"; exit 1)
	@cp config.yaml $(CFG_BACKUP) 2>/dev/null || true
	cp config.rtsp.yaml config.yaml
	$(MAKE) rtsp-run
	@mv $(CFG_BACKUP) config.yaml 2>/dev/null || true

# Alias dedicado para RTSP (usa app.py com config atual)
rtsp-run:
	. $(VENV)/bin/activate ; \
	$(PY) app.py