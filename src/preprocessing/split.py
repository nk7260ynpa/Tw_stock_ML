"""時間序列資料切分模組。

提供按時間順序切分訓練集與測試集的功能。
"""

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_ratio: float = 0.2,
    dates: pd.Series | None = None,
) -> dict:
    """按時間順序切分訓練集與測試集。

    不進行 shuffle，確保訓練集在前、測試集在後，
    避免時間序列資料洩漏。

    Args:
        X: 特徵 DataFrame。
        y: 目標 Series。
        test_ratio: 測試集比例，預設 0.2。
        dates: 日期 Series，若提供則一併切分。

    Returns:
        包含以下鍵值的字典：
        - X_train: 訓練特徵 DataFrame。
        - X_test: 測試特徵 DataFrame。
        - y_train: 訓練目標 Series。
        - y_test: 測試目標 Series。
        - train_dates: 訓練日期 Series（若提供 dates）。
        - test_dates: 測試日期 Series（若提供 dates）。

    Raises:
        ValueError: 當 test_ratio 不在 (0, 1) 範圍內時。
    """
    if not 0 < test_ratio < 1:
        raise ValueError(f"test_ratio 必須在 (0, 1) 範圍內，收到 {test_ratio}")

    n = len(X)
    split_idx = n - int(n * test_ratio)

    result = {
        "X_train": X.iloc[:split_idx].reset_index(drop=True),
        "X_test": X.iloc[split_idx:].reset_index(drop=True),
        "y_train": y.iloc[:split_idx].reset_index(drop=True),
        "y_test": y.iloc[split_idx:].reset_index(drop=True),
        "train_dates": None,
        "test_dates": None,
    }

    if dates is not None:
        result["train_dates"] = dates.iloc[:split_idx].reset_index(drop=True)
        result["test_dates"] = dates.iloc[split_idx:].reset_index(drop=True)

    logger.info(
        "資料切分完成：訓練 %d 筆，測試 %d 筆（比例 %.1f%%）",
        split_idx,
        n - split_idx,
        test_ratio * 100,
    )
    return result
