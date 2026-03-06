# 台股機器學習分析系統

> **版本**：0.2.0
> **Docker Image**：`nk7260ynpa/tw-stock-ml`
> **Port**：5002

使用機器學習技術分析台灣股票市場的專案，提供 Web Dashboard 與批次訓練兩種模式。

## 功能

- **Web Dashboard**：深色主題 UI，支援股票搜尋、K 線圖表、技術指標疊加、ML 預測分析
- **K 線圖表**：TradingView Lightweight Charts v4，台股慣例紅漲綠跌
- **技術指標**：SMA、EMA、RSI、MACD、布林帶、ATR
- **ML 預測**：XGBoost 模型訓練與預測，顯示評估指標與特徵重要度
- **批次訓練**：命令列模式，查詢股價資料、前處理、訓練模型並評估

## 專案架構

```
Tw_stock_ML/
├── docker/                 # Docker 相關設定
│   ├── Dockerfile          # Docker image 定義
│   ├── build.sh            # 建立 Docker image 腳本
│   └── docker-compose.yaml # Docker Compose 設定（port 5002）
├── src/                    # 主程式碼
│   ├── main.py             # 主程式進入點（--web 啟動 Web Dashboard）
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
│   ├── utils/              # 工具模組
│   │   └── logger.py       # 日誌工具
│   └── web/                # Flask Web 模組
│       ├── __init__.py     # App Factory
│       ├── routes/         # Blueprint 路由
│       │   ├── dashboard.py      # 頁面路由（GET /）
│       │   └── api.py            # API 端點（搜尋、日線、預測）
│       ├── templates/      # Jinja2 模板
│       │   ├── base.html         # 基礎模板
│       │   └── dashboard.html    # 主頁面
│       └── static/         # 靜態資源
│           ├── css/main.css      # 深色主題樣式
│           └── js/
│               ├── chart.js      # K 線圖渲染
│               └── app.js        # 頁面互動邏輯
├── tests/                  # 單元測試
├── model/                  # 訓練完成的模型存放
├── data/                   # 訓練資料存放
├── logs/                   # 日誌存放
├── pyproject.toml          # PEP 621 專案定義（套件安裝用）
├── requirements.txt        # Python 釘版依賴（Docker 環境用）
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

### 啟動 Web Dashboard

```bash
bash run.sh
```

服務啟動後可在 http://localhost:5002 存取 Web Dashboard。

此腳本會自動建立 Docker image、建立 `db_network` 網路、啟動 container 並映射 port 5002。

### 批次訓練模式

```bash
docker compose -f docker/docker-compose.yaml run --rm tw-stock-ml python src/main.py
```

### 執行測試

```bash
docker compose -f docker/docker-compose.yaml run --rm tw-stock-ml pytest tests/ -v
```

## API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/api/stocks/search?q=xxx` | 搜尋股票（代碼或名稱） |
| GET | `/api/stocks/<code>/daily?start=&end=` | 取得日線資料 |
| GET | `/api/stocks/<code>/info` | 取得股票基本資訊 |
| GET | `/api/stocks/<code>/indicators?start=&end=` | 取得技術指標序列 |
| POST | `/api/predict` | 執行 ML 預測 |

### ML 預測 API 範例

```json
// POST /api/predict
// Request Body
{"stock_code": "2330"}

// Response
{
    "stock_code": "2330",
    "train_samples": 180,
    "test_samples": 45,
    "n_features": 20,
    "predictions": [{"date": "2024-01-02", "actual": 590.0, "predicted": 588.5}],
    "metrics": {
        "price_MAE": 5.23,
        "price_RMSE": 7.12,
        "price_MAPE": 0.89,
        "directional_accuracy": 55.56
    },
    "feature_importance": [{"name": "ClosingPrice", "importance": 0.234}]
}
```

## CI/CD

本專案使用 GitHub Actions 自動建置並發布 Docker image 至 DockerHub。

### 觸發條件

推送符合 `v*.*.*` 格式的 tag 時自動觸發：

```bash
git tag v0.2.0
git push origin v0.2.0
```

### 發布流程

1. GitHub Actions 自動建置 Docker image
2. 同時推送版本號 tag（如 `0.2.0`）和 `latest` tag 至 DockerHub
3. 使用 GitHub Actions cache 加速後續建置

### DockerHub

- Image：`nk7260ynpa/tw-stock-ml`
- 需設定 GitHub Secrets：`DOCKER_USERNAME`、`DOCKER_PASSWORD`

## 授權條款

MIT License
