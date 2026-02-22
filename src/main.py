"""台股機器學習分析主程式。

此模組為專案進入點，負責查詢股價資料、前處理、訓練模型並評估。
"""

from src.database.connection import get_engine
from src.database.stock_repository import get_daily_prices
from src.model.xgboost_model import evaluate_model, train_xgboost
from src.preprocessing.pipeline import preprocess_sliding_pipeline
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """主程式進入點。"""
    logger.info("台股機器學習分析系統啟動")

    # 取得資料
    engine = get_engine()
    df = get_daily_prices(engine, "2330")
    logger.info("取得 %d 筆 2330 股價資料", len(df))

    # 前處理（滑動視窗）
    data = preprocess_sliding_pipeline(df, window_size=40)
    logger.info(
        "前處理完成：訓練 %d 筆，測試 %d 筆",
        data.X_train.shape[0],
        data.X_test.shape[0],
    )

    # 訓練模型
    model = train_xgboost(
        data.X_train, data.y_train,
        data.X_test, data.y_test,
    )

    # 評估模型
    results = evaluate_model(model, data.X_test, data.y_test)
    logger.info("評估結果：%s", results)


if __name__ == "__main__":
    main()
