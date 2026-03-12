"""台股機器學習分析主程式。

此模組為專案進入點，支援兩種模式：
1. 批次訓練模式（預設）：查詢股價資料、前處理、訓練模型並評估。
2. Web 模式（--web）：啟動 Flask Web Dashboard。
"""

import argparse
from pathlib import Path

import numpy as np

from src.database.connection import get_engine
from src.database.stock_repository import get_daily_prices
from src.model.xgboost_model import (
    detect_device,
    evaluate_return_model,
    get_high_dim_params,
    save_model,
    train_xgboost,
)
from src.preprocessing.pipeline import preprocess_forward_pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

DATA_DIR = Path("data")
MODEL_DIR = Path("model")


def save_training_data(data, stock_code: str) -> None:
    """將訓練與測試資料儲存至 data/ 資料夾。

    Args:
        data: 前處理結果（SlidingWindowData 或 IndicatorData）。
        stock_code: 股票代碼。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{stock_code}_train_test.npz"
    np.savez(
        path,
        X_train=data.X_train,
        X_test=data.X_test,
        y_train=data.y_train,
        y_test=data.y_test,
    )
    logger.info("訓練資料已儲存至 %s", path)


def run_training() -> None:
    """執行批次訓練模式。"""
    logger.info("台股機器學習分析系統啟動（批次訓練模式）")

    stock_code = "2330"

    # 取得資料
    engine = get_engine()
    df = get_daily_prices(engine, stock_code)
    logger.info("取得 %d 筆 %s 股價資料", len(df), stock_code)

    # 前處理（前瞻指標滑動視窗管線）
    data = preprocess_forward_pipeline(df)
    logger.info(
        "前處理完成：訓練 %d 筆（%d 維特徵），測試 %d 筆，window=%d, horizon=%d",
        data.X_train.shape[0],
        data.X_train.shape[1],
        data.X_test.shape[0],
        data.window_size,
        data.horizon,
    )

    # 儲存訓練資料
    save_training_data(data, stock_code)

    # 高維特徵參數 + 訓練模型
    device = detect_device()
    params = get_high_dim_params(device)
    model = train_xgboost(
        data.X_train, data.y_train,
        data.X_test, data.y_test,
        params=params,
    )

    # 儲存模型
    model_path = MODEL_DIR / f"xgboost_{stock_code}.json"
    save_model(model, model_path)

    # 報酬率 + 價格雙重評估
    results = evaluate_return_model(
        model, data.X_test, data.y_test, data.base_prices_test.values,
    )
    logger.info("評估結果：%s", results)


def run_web(host: str = "0.0.0.0", port: int = 5002) -> None:
    """啟動 Flask Web Dashboard。

    Args:
        host: 監聽位址，預設 0.0.0.0。
        port: 監聽連接埠，預設 5002。
    """
    from src.web import create_app

    logger.info("台股 ML 預測儀表板啟動：%s:%d", host, port)
    app = create_app()
    app.run(host=host, port=port, debug=False)


def main() -> None:
    """主程式進入點，解析命令列參數並執行對應模式。"""
    parser = argparse.ArgumentParser(description="台股機器學習分析系統")
    parser.add_argument(
        "--web",
        action="store_true",
        help="啟動 Flask Web Dashboard 模式",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Web Dashboard 監聽連接埠（預設 5002）",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Web Dashboard 監聽位址（預設 0.0.0.0）",
    )

    args = parser.parse_args()

    if args.web:
        run_web(host=args.host, port=args.port)
    else:
        run_training()


if __name__ == "__main__":
    main()
