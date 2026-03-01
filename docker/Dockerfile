FROM python:3.11-slim

WORKDIR /app

# 安裝系統依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# 複製並安裝 Python 釘版依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 以 editable 模式安裝專案套件，使 import 路徑正確解析
RUN pip install --no-cache-dir -e .

# 建立 logs 資料夾
RUN mkdir -p /app/logs

CMD ["python", "src/main.py"]
