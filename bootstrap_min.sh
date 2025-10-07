#!/usr/bin/env bash
set -e

# Estrutura de pastas
mkdir -p inference signals privacy io utils tests scripts "data/samples" deploy

# requirements.txt
cat > requirements.txt <<'REQ'
pyyaml>=6
prometheus-client>=0.20
REQ

# config.yaml
cat > config.yaml <<'YAML'
runtime:
  metrics_port: 9108
YAML

# app.py
cat > app.py <<'PY'
import time, signal, sys, yaml
from utils.metrics import bootstrap_metrics
from utils.log import get_logger

log = get_logger("app")

def main(cfg):
    bootstrap_metrics(cfg["runtime"]["metrics_port"])
    log.info({"event":"service_start", "port":cfg["runtime"]["metrics_port"]})
    while True:
        time.sleep(1)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    cfg = yaml.safe_load(open("config.yaml","r",encoding="utf-8"))
    main(cfg)
PY

# utils
cat > utils/__init__.py <<'PY'
PY

cat > utils/metrics.py <<'PY'
from prometheus_client import Counter, Histogram, start_http_server
alive_beats = Counter("alive_beats_total", "Sinal de vida do serviço")
def bootstrap_metrics(port=9108):
    start_http_server(port)
PY

cat > utils/log.py <<'PY'
import logging, json, sys
def get_logger(name):
    l = logging.getLogger(name); l.setLevel(logging.INFO)
    if not l.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter('%(message)s'))
        l.addHandler(h)
    old_info = l.info
    def info(obj): old_info(json.dumps(obj) if not isinstance(obj, str) else obj)
    l.info = info
    return l
PY

echo "✅ Bootstrap criado com sucesso em $(pwd)"
echo "Próximos passos:"
echo "  python -m venv .venv && source .venv/bin/activate"
echo "  python -m pip install --upgrade pip wheel setuptools"
echo "  pip install -r requirements.txt"
echo "  python app.py"