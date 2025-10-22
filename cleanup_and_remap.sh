#!/bin/bash
set -e

echo "=== [1/5] Verificando uso de disco ==="
df -h /

echo "=== [2/5] Limpando caches e logs antigos ==="
sudo apt clean
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf /var/cache/apt/archives/*
sudo journalctl --vacuum-time=2d
sudo rm -rf /var/log/*.gz /var/log/*.[0-9] || true

echo "=== [3/5] Criando diretórios no disco maior ==="
sudo mkdir -p /mnt/data/apt-cache
sudo mkdir -p /mnt/data/tmp
sudo mkdir -p /mnt/data/logs

echo "=== [4/5] Remapeando caches do sistema para /mnt/data ==="
sudo rm -rf /var/cache/apt/archives
sudo ln -s /mnt/data/apt-cache /var/cache/apt/archives
sudo rm -rf /tmp
sudo ln -s /mnt/data/tmp /tmp
sudo rm -rf /var/log
sudo ln -s /mnt/data/logs /var/log

echo "=== [5/5] Atualizando pacotes e confirmando ==="
sudo apt update -y || true

echo "✅ Limpeza e remapeamento concluídos."
df -h
