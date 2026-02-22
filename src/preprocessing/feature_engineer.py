"""特徵工程模組。

提供特徵選取與預測目標建構功能。
"""

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_FEATURE_COLUMNS = [
    "OpeningPrice",
    "HighestPrice",
    "LowestPrice",
    "ClosingPrice",
    "TradeVolume",
]


def select_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """從每日行情 DataFrame 選取特徵欄位。

    Args:
        df: 每日行情 DataFrame，需包含指定的特徵欄位。
        feature_columns: 要選取的欄位名稱清單，預設為 5 個核心價量欄位。

    Returns:
        僅包含特徵欄位的 DataFrame。

    Raises:
        KeyError: 當 DataFrame 缺少指定欄位時。
    """
    columns = feature_columns or _DEFAULT_FEATURE_COLUMNS
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"DataFrame 缺少以下欄位：{missing}")
    return df[columns].copy()


def create_target(
    df: pd.DataFrame,
    target_column: str = "ClosingPrice",
) -> pd.Series:
    """建構隔天收盤價作為預測目標。

    使用 shift(-1) 將目標欄位向前移動一列，最後一列為 NaN。

    Args:
        df: 每日行情 DataFrame，需包含目標欄位。
        target_column: 目標欄位名稱，預設為 ClosingPrice。

    Returns:
        隔天收盤價 Series，最後一列為 NaN。

    Raises:
        KeyError: 當 DataFrame 缺少目標欄位時。
    """
    if target_column not in df.columns:
        raise KeyError(f"DataFrame 缺少目標欄位：{target_column}")
    return df[target_column].shift(-1)


def build_feature_target(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = "ClosingPrice",
    date_column: str = "Date",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """組合特徵選取與目標建構，並移除最後一列 NaN。

    Args:
        df: 每日行情 DataFrame。
        feature_columns: 特徵欄位名稱清單，預設為 5 個核心價量欄位。
        target_column: 目標欄位名稱，預設為 ClosingPrice。
        date_column: 日期欄位名稱，預設為 Date。

    Returns:
        (features, target, dates) 三元組：
        - features: 特徵 DataFrame（已移除最後一列）。
        - target: 隔天收盤價 Series（已移除最後一列 NaN）。
        - dates: 日期 Series（已移除最後一列）。

    Raises:
        KeyError: 當 DataFrame 缺少必要欄位時。
    """
    features = select_features(df, feature_columns)
    target = create_target(df, target_column)

    # 移除最後一列（target 為 NaN）
    valid_mask = target.notna()
    features = features.loc[valid_mask].reset_index(drop=True)
    target = target.loc[valid_mask].reset_index(drop=True)

    if date_column in df.columns:
        dates = df.loc[valid_mask, date_column].reset_index(drop=True)
    else:
        dates = pd.Series(range(valid_mask.sum()), name="index")

    logger.info(
        "特徵工程完成：%d 筆資料，%d 個特徵",
        len(features),
        len(features.columns),
    )
    return features, target, dates


def create_sliding_windows(
    features: pd.DataFrame,
    target: pd.Series,
    dates: pd.Series,
    window_size: int = 40,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """將時間序列轉換為滑動視窗樣本。

    每個樣本由連續 window_size 天的特徵展平而成，
    目標為視窗結束後隔天的收盤價。

    Args:
        features: 特徵 DataFrame，shape (n_days, n_features)。
        target: 目標 Series，長度為 n_days（已由 build_feature_target 建立）。
        dates: 日期 Series，長度為 n_days。
        window_size: 滑動視窗大小（交易日數），預設 40。

    Returns:
        (X_windows, y_windows, window_dates) 三元組：
        - X_windows: shape (n_samples, window_size * n_features) 展平特徵陣列。
        - y_windows: shape (n_samples,) 目標陣列。
        - window_dates: 每個樣本對應的預測日期 Series。

    Raises:
        ValueError: 當資料筆數不足以建立至少一個視窗時。
    """
    n_days = len(features)
    n_features = features.shape[1]
    n_samples = n_days - window_size

    if n_samples <= 0:
        raise ValueError(
            f"資料筆數 ({n_days}) 不足以建立視窗 "
            f"(至少需要 {window_size + 1} 筆)"
        )

    feature_values = features.values
    target_values = target.values

    X_windows = np.empty((n_samples, window_size * n_features))
    y_windows = np.empty(n_samples)

    for i in range(n_samples):
        X_windows[i] = feature_values[i:i + window_size].flatten()
        y_windows[i] = target_values[i + window_size - 1]

    window_dates = dates.iloc[window_size:].reset_index(drop=True)

    logger.info(
        "滑動視窗建立完成：%d 個樣本，視窗大小 %d，展平維度 %d",
        n_samples,
        window_size,
        window_size * n_features,
    )
    return X_windows, y_windows, window_dates
