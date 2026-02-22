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
│   ├── database/           # 資料庫存取模組
│   │   ├── connection.py         # 連線管理
│   │   └── stock_repository.py   # 股票資料查詢
│   ├── metrics/            # 模型評估指標
│   │   ├── price_metrics.py      # 價格距離指標（MAE、RMSE、MAPE）
│   │   └── direction_metrics.py  # 方向正確率指標
│   ├── model/              # 模型訓練模組
│   │   └── xgboost_model.py      # XGBoost 訓練/預測/評估/儲存/GPU 偵測
│   ├── preprocessing/      # 資料前處理模組
│   │   ├── technical_indicators.py # 技術指標計算（SMA、EMA、RSI、MACD、布林帶、ATR）
│   │   ├── feature_engineer.py   # 特徵選取與目標建構
│   │   ├── split.py              # 時間序列切分
│   │   ├── scaler.py             # StandardScaler 標準化
│   │   └── pipeline.py           # 一站式前處理管線
│   └── utils/              # 工具模組
│       └── logger.py       # 日誌工具
├── tests/                  # 單元測試
├── model/                  # 訓練完成的模型存放
├── data/                   # 訓練資料存放
├── logs/                   # 日誌存放
├── requirements.txt        # Python 依賴
├── run.sh                  # 主程式啟動腳本
└── README.md
```

## 環境需求

- Docker
- Docker Compose
- 台股資料庫（`Tw_stock_DB`）需先啟動，提供 `db_network` Docker 網路

## 使用方法

### 建立 Docker Image

```bash
bash docker/build.sh
```

### 啟動主程式

```bash
bash run.sh
```

此腳本會自動建立 Docker image、建立 `db_network` 網路、啟動 container，並掛載 `logs/`、`data/`、`model/` 資料夾。

### 執行測試

```bash
docker compose -f docker/docker-compose.yaml run --rm tw-stock-ml pytest tests/ -v
```

## 授權條款

MIT License
