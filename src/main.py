"""台股機器學習分析主程式。

此模組為專案進入點，負責查詢股價資料、前處理、訓練模型並評估。
"""

from pathlib import Path

import numpy as np

from src.database.connection import get_engine
from src.database.stock_repository import get_daily_prices
from src.model.xgboost_model import (
    detect_device,
    evaluate_return_model,
    get_small_data_params,
    save_model,
    train_xgboost,
)
from src.preprocessing.pipeline import preprocess_indicator_pipeline
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


def main() -> None:
    """主程式進入點。"""
    logger.info("台股機器學習分析系統啟動")

    stock_code = "2330"

    # 取得資料
    engine = get_engine()
    df = get_daily_prices(engine, stock_code)
    logger.info("取得 %d 筆 %s 股價資料", len(df), stock_code)

    # 前處理（技術指標 + 報酬率目標）
    data = preprocess_indicator_pipeline(df, target_type="return")
    logger.info(
        "前處理完成：訓練 %d 筆（%d 維特徵），測試 %d 筆",
        data.X_train.shape[0],
        data.X_train.shape[1],
        data.X_test.shape[0],
    )

    # 儲存訓練資料
    save_training_data(data, stock_code)

    # 小資料參數 + 訓練模型
    device = detect_device()
    params = get_small_data_params(device)
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
        model, data.X_test, data.y_test, data.base_prices.values,
    )
    logger.info("評估結果：%s", results)


if __name__ == "__main__":
    main()
