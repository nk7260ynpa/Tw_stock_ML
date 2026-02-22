#!/bin/bash
# 啟動主程式：建立並啟動 Docker container 執行主程式

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly IMAGE_NAME="tw-stock-ml"
readonly CONTAINER_NAME="tw-stock-ml"

# 確保 logs 資料夾存在
mkdir -p "${SCRIPT_DIR}/logs"

# 建立 Docker image
bash "${SCRIPT_DIR}/docker/build.sh"

# 停止並移除已存在的 container
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "停止並移除舊的 container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}"
fi

# 啟動 Docker container 並掛載 logs 資料夾
echo "啟動 Docker container: ${CONTAINER_NAME}"
docker run \
    --name "${CONTAINER_NAME}" \
    -v "${SCRIPT_DIR}/logs:/app/logs" \
    -v "${SCRIPT_DIR}/data:/app/data" \
    "${IMAGE_NAME}:latest"
