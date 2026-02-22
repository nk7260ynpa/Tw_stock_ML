# 台股機器學習分析系統

使用機器學習技術分析台灣股票市場的專案。

## 專案架構

```
Tw_stock_ML/
├── docker/                 # Docker 相關設定
│   ├── Dockerfile          # Docker image 定義
│   ├── build.sh            # 建立 Docker image 腳本
│   └── docker-compose.yaml # Docker Compose 設定
├── src/                    # 主程式碼
│   ├── main.py             # 主程式進入點
│   ├── metrics/            # 模型評估指標
│   │   ├── price_metrics.py      # 價格距離指標（MAE、RMSE、MAPE）
│   │   └── direction_metrics.py  # 方向正確率指標
│   └── utils/              # 工具模組
│       └── logger.py       # 日誌工具
├── tests/                  # 單元測試
├── data/                   # 資料存放
├── logs/                   # 日誌存放
├── requirements.txt        # Python 依賴
├── run.sh                  # 主程式啟動腳本
└── README.md
```

## 環境需求

- Docker
- Docker Compose

## 使用方法

### 建立 Docker Image

```bash
bash docker/build.sh
```

### 啟動主程式

```bash
bash run.sh
```

此腳本會自動建立 Docker image、啟動 container，並掛載 `logs/` 資料夾。

### 執行測試

```bash
docker compose -f docker/docker-compose.yaml run --rm tw-stock-ml pytest tests/ -v
```

## 授權條款

MIT License
