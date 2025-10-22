#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

stop_service() {
  local pid_file="$1"
  local name="$2"
  if [[ ! -f "$pid_file" ]]; then
    echo "$name não está em execução (PID ausente)."
    return
  fi
  local pid
  pid=$(cat "$pid_file")
  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "Encerrando $name (PID $pid)"
    kill "$pid" || true
    wait "$pid" 2>/dev/null || true
  else
    echo "$name já finalizado."
  fi
  rm -f "$pid_file"
}

stop_service "$ROOT_DIR/.saas_panel.pid" "Painel"
stop_service "$ROOT_DIR/.saas_api.pid" "API"

echo "Serviços SAAS finalizados."
