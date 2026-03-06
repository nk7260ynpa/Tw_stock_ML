#!/bin/bash
# 啟動 ML Dashboard Web 服務（docker compose）

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 確保資料夾存在
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/model"
mkdir -p "${SCRIPT_DIR}/data"

# 確保 db_network 存在
docker network create db_network 2>/dev/null || true

# 建立 Docker image 並啟動服務
echo "啟動 ML Dashboard Web 服務..."
docker compose -f "${SCRIPT_DIR}/docker/docker-compose.yaml" up -d --build

echo "ML Dashboard 已啟動：http://localhost:5002"
