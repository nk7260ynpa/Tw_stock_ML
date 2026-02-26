FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 設定 PYTHONPATH 使 import 路徑從 src 開始
ENV PYTHONPATH=/app

# 複製專案程式碼
COPY . .

# 建立 logs 資料夾
RUN mkdir -p /app/logs

CMD ["python", "src/main.py"]
