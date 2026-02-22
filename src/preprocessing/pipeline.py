"""前處理管線模組。

提供一站式前處理函式，將每日行情 DataFrame 轉換為可供模型使用的資料集。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing.feature_engineer import (
    build_feature_target,
    build_feature_target_with_indicators,
    create_sliding_windows,
)
from src.preprocessing.scaler import fit_transform_train_test
from src.preprocessing.split import time_series_split
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class PreprocessedData:
    """前處理完成的資料集。

    Attributes:
        X_train: 標準化後的訓練特徵陣列。
        X_test: 標準化後的測試特徵陣列。
        y_train: 訓練目標陣列（原始值，未標準化）。
        y_test: 測試目標陣列（原始值，未標準化）。
        feature_scaler: 已 fit 的 StandardScaler 物件。
        feature_names: 特徵欄位名稱清單。
        train_dates: 訓練集日期 Series。
        test_dates: 測試集日期 Series。
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    feature_names: list[str]
    train_dates: pd.Series
    test_dates: pd.Series


def preprocess_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "ClosingPrice",
    test_ratio: float = 0.2,
) -> PreprocessedData:
    """執行完整的前處理管線。

    流程：特徵工程 → 時間序列切分 → 標準化。
    y 保持原始值不標準化，以便直接與 metrics 模組搭配使用。

    Args:
        df: 每日行情 DataFrame。
        feature_columns: 特徵欄位名稱清單，預設為 5 個核心價量欄位。
        target_column: 目標欄位名稱，預設為 ClosingPrice。
        test_ratio: 測試集比例，預設 0.2。

    Returns:
        PreprocessedData 資料類別實例。
    """
    # 特徵工程
    features, target, dates = build_feature_target(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    # 時間序列切分
    split_result = time_series_split(
        features, target, test_ratio=test_ratio, dates=dates,
    )

    # 標準化（僅 fit 訓練集）
    X_train_scaled, X_test_scaled, scaler = fit_transform_train_test(
        split_result["X_train"].values,
        split_result["X_test"].values,
    )

    result = PreprocessedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=split_result["y_train"].values,
        y_test=split_result["y_test"].values,
        feature_scaler=scaler,
        feature_names=list(features.columns),
        train_dates=split_result["train_dates"],
        test_dates=split_result["test_dates"],
    )

    logger.info("前處理管線完成")
    return result


@dataclass
class SlidingWindowData:
    """滑動視窗前處理完成的資料集。

    Attributes:
        X_train: 標準化後的訓練特徵陣列。
        X_test: 標準化後的測試特徵陣列。
        y_train: 訓練目標陣列（原始值，未標準化）。
        y_test: 測試目標陣列（原始值，未標準化）。
        feature_scaler: 已 fit 的 StandardScaler 物件。
        feature_names: 原始特徵欄位名稱清單。
        window_size: 滑動視窗大小。
        train_dates: 訓練集日期 Series。
        test_dates: 測試集日期 Series。
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    feature_names: list[str]
    window_size: int
    train_dates: pd.Series
    test_dates: pd.Series


def preprocess_sliding_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "ClosingPrice",
    window_size: int = 40,
    test_ratio: float = 0.2,
) -> SlidingWindowData:
    """執行滑動視窗前處理管線。

    流程：特徵工程 → 滑動視窗 → 時間序列切分 → 標準化。
    y 保持原始值不標準化，以便直接與 metrics 模組搭配使用。
    標準化在展平後的高維空間上操作。

    Args:
        df: 每日行情 DataFrame。
        feature_columns: 特徵欄位名稱清單，預設為 5 個核心價量欄位。
        target_column: 目標欄位名稱，預設為 ClosingPrice。
        window_size: 滑動視窗大小（交易日數），預設 40。
        test_ratio: 測試集比例，預設 0.2。

    Returns:
        SlidingWindowData 資料類別實例。
    """
    # 特徵工程
    features, target, dates = build_feature_target(
        df,
        feature_columns=feature_columns,
        target_column=target_column,
    )

    # 滑動視窗
    X_windows, y_windows, window_dates = create_sliding_windows(
        features, target, dates, window_size=window_size,
    )

    # 時間序列切分（轉為 DataFrame/Series 以複用 time_series_split）
    X_df = pd.DataFrame(X_windows)
    y_series = pd.Series(y_windows)
    split_result = time_series_split(
        X_df, y_series, test_ratio=test_ratio, dates=window_dates,
    )

    # 標準化（僅 fit 訓練集，在展平後的高維空間上操作）
    X_train_scaled, X_test_scaled, scaler = fit_transform_train_test(
        split_result["X_train"].values,
        split_result["X_test"].values,
    )

    result = SlidingWindowData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=split_result["y_train"].values,
        y_test=split_result["y_test"].values,
        feature_scaler=scaler,
        feature_names=list(features.columns),
        window_size=window_size,
        train_dates=split_result["train_dates"],
        test_dates=split_result["test_dates"],
    )

    logger.info("滑動視窗前處理管線完成")
    return result


@dataclass
class IndicatorData:
    """技術指標前處理完成的資料集。

    Attributes:
        X_train: 標準化後的訓練特徵陣列。
        X_test: 標準化後的測試特徵陣列。
        y_train: 訓練目標陣列（報酬率或價格）。
        y_test: 測試目標陣列（報酬率或價格）。
        feature_scaler: 已 fit 的 StandardScaler 物件。
        feature_names: 技術指標特徵名稱清單。
        target_type: 目標類型，"return" 或 "price"。
        train_dates: 訓練集日期 Series。
        test_dates: 測試集日期 Series。
        base_prices: 測試集基準價格（用於報酬率逆推絕對價格），
            當 target_type="return" 時不為 None。
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_scaler: StandardScaler
    feature_names: list[str]
    target_type: str
    train_dates: pd.Series
    test_dates: pd.Series
    base_prices: pd.Series | None


def preprocess_indicator_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_type: str = "return",
    target_column: str = "ClosingPrice",
    test_ratio: float = 0.2,
) -> IndicatorData:
    """執行技術指標前處理管線。

    流程：compute_all_indicators → build_feature_target_with_indicators
    → time_series_split → fit_transform_train_test。

    Args:
        df: 每日行情 DataFrame。
        feature_columns: 特徵欄位名稱清單，None 時自動使用所有技術指標欄位。
        target_type: 目標類型，"return" 為報酬率，"price" 為絕對價格。
        target_column: 目標欄位名稱，預設為 ClosingPrice。
        test_ratio: 測試集比例，預設 0.2。

    Returns:
        IndicatorData 資料類別實例。
    """
    # 特徵工程（含技術指標計算）
    features, target, dates = build_feature_target_with_indicators(
        df,
        feature_columns=feature_columns,
        target_type=target_type,
        target_column=target_column,
    )

    # 時間序列切分
    split_result = time_series_split(
        features, target, test_ratio=test_ratio, dates=dates,
    )

    # 標準化（僅 fit 訓練集）
    X_train_scaled, X_test_scaled, scaler = fit_transform_train_test(
        split_result["X_train"].values,
        split_result["X_test"].values,
    )

    # 計算基準價格（測試集前一天的收盤價，用於報酬率逆推）
    base_prices = None
    if target_type == "return" and target_column in features.columns:
        n_train = len(split_result["y_train"])
        # 基準價格：測試集每筆對應的當天收盤價
        # 因為 target = (close_{t+1} - close_t) / close_t
        # 所以 close_{t+1} = close_t * (1 + target)
        test_base = features[target_column].iloc[n_train:].reset_index(drop=True)
        base_prices = test_base

    result = IndicatorData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=split_result["y_train"].values,
        y_test=split_result["y_test"].values,
        feature_scaler=scaler,
        feature_names=list(features.columns),
        target_type=target_type,
        train_dates=split_result["train_dates"],
        test_dates=split_result["test_dates"],
        base_prices=base_prices,
    )

    logger.info("技術指標前處理管線完成")
    return result
