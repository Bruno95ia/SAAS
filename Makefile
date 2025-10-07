# =========[ CONFIG ]=========
API_URL ?= http://127.0.0.1:8000
API_KEY ?= minha-chave-forte

# parâmetros “por padrão” (customize ao chamar: CLIP=..., CAMERA=..., SCORE=...)
CLIP   ?= runs/clips/TESTE01.mp4
CAMERA ?= cam01
SCORE  ?= 0.92

PYTHON ?= python

# =========[ HELP ]=========
.PHONY: help
help:
	@echo "Targets:"
	@echo "  make install            # instala libs essenciais"
	@echo "  make api                # sobe a API FastAPI (porta 8000)"
	@echo "  make panel              # sobe o painel Streamlit (porta 8501)"
	@echo "  make health             # testa /health"
	@echo "  make alerts             # lista /alerts"
	@echo "  make convert-clips      # converte todos .avi em runs/clips para .mp4"
	@echo "  make post-alert CLIP=...# publica um alerta apontando para CLIP"
	@echo "  make open-clip CLIP=... # abre a URL do clipe no navegador"
	@echo "  make kill-api           # encerra uvicorn rodando localmente"

# =========[ SETUP ]=========
.PHONY: install
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install fastapi uvicorn streamlit pandas requests websocket-client moviepy

# =========[ RUN ]=========
.PHONY: api
api:
	SAAS_API_KEY=$(API_KEY) uvicorn saas.api:app --reload --port 8000 --app-dir src

.PHONY: panel
panel:
	SAAS_API_URL=$(API_URL) SAAS_API_KEY=$(API_KEY) streamlit run painel.py

# =========[ CHECKS ]=========
.PHONY: health alerts
health:
	curl -s $(API_URL)/health | jq .

alerts:
	curl -s -H "X-API-Key: $(API_KEY)" $(API_URL)/alerts | jq .

# =========[ VIDEO ]=========
# Converte todos .avi/.AVI dentro de runs/clips para .mp4
.PHONY: convert-clips
convert-clips:
	@echo "Convertendo .avi -> .mp4 em runs/clips ..."
	@mkdir -p runs/clips
	@bash -lc 'while IFS= read -r -d "" f; do \
		ffmpeg -y -i "$$f" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k "$${f%.*}.mp4"; \
	done < <(find runs/clips -type f \( -iname "*.avi" \) -print0)'
	@echo "OK."

# Publica alerta apontando para um arquivo local servido pela API
.PHONY: post-alert
post-alert:
	@fname=$$(basename "$(CLIP)"); \
	url="$(API_URL)/clips/$$fname"; \
	echo "Postando alerta: camera=$(CAMERA) score=$(SCORE) clip=$$url"; \
	curl -s -X POST "$(API_URL)/alerts" \
		-H "Content-Type: application/json" \
		-H "X-API-Key: $(API_KEY)" \
		-d "$$(jq -nc --arg cam '$(CAMERA)' --arg url "$$url" --argjson score $(SCORE) \
			'{camera_id:$$cam, type:"fall", score:$$score, clip_path:$$url}')" | jq .

# Abre o clipe no navegador (macOS: open, Linux: xdg-open)
.PHONY: open-clip
open-clip:
	@fname=$$(basename "$(CLIP)"); \
	url="$(API_URL)/clips/$$fname"; \
	echo "Abrindo $$url"; \
	if command -v open >/dev/null 2>&1; then open "$$url"; \
	elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$$url"; \
	else echo "Abra manualmente: $$url"; fi

# =========[ UTIL ]=========
.PHONY: kill-api
kill-api:
	- pkill -f "uvicorn saas.api:app" || true
	@echo "uvicorn finalizado (se existia)."
	.PHONY: run-pipeline
run-pipeline:
	SAAS_API_URL=$(API_URL) SAAS_API_KEY=$(API_KEY) \
	PYTHONPATH=src python -m saas.run_pipeline -i runs/clips --post --camera $(CAMERA) --overwrite
	
	.PHONY: live-yolo
live-yolo:
	SAAS_API_URL=$(API_URL) SAAS_API_KEY=$(API_KEY) \
	PYTHONPATH=src python -m saas.live_yolo \
	  --camera $(CAMERA) --rtsp "$(RTSP)" --buffer runs/buffer/$(CAMERA) \
	  --weights yolov8n-pose.pt --imgsz 640 --theta-deg 55 --vy-min 0.25 --flat-ratio 0.6 --flat-sec 2.0
	  .PHONY: extract-yolo-feats
extract-yolo-feats:
	PYTHONPATH=src python -m saas.batch_yolo_extract -i runs/clips --pattern "*.mp4" --weights yolov8n-pose.pt

.PHONY: train-tcn
train-tcn:
	PYTHONPATH=src python -m saas.train_tcn_yolo --labels labels.csv --feats runs/feats --out runs/models --epochs 25