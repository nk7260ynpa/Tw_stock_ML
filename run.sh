#!/bin/bash
# 啟動主程式：建立並啟動 Docker container 執行主程式

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE_NAME="tw-stock-ml"
readonly CONTAINER_NAME="tw-stock-ml"

# 確保資料夾存在
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/model"
mkdir -p "${SCRIPT_DIR}/data"

# 建立 Docker image
bash "${SCRIPT_DIR}/docker/build.sh"

# 確保 db_network 存在
docker network create db_network 2>/dev/null || true

# 停止並移除已存在的 container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "停止並移除舊的 container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}"
fi

# 啟動 Docker container 並掛載 logs 資料夾
echo "啟動 Docker container: ${CONTAINER_NAME}"
docker run \
    --name "${CONTAINER_NAME}" \
    --network db_network \
    -e DB_HOST=tw_stock_database \
    -e DB_USER=root \
    -e DB_PASSWORD=stock \
    -e DB_NAME=TWSE \
    -v "${SCRIPT_DIR}/logs:/app/logs" \
    -v "${SCRIPT_DIR}/data:/app/data" \
    -v "${SCRIPT_DIR}/model:/app/model" \
    "${IMAGE_NAME}:latest"
