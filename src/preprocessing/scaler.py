"""特徵標準化模組。

提供 StandardScaler 封裝，僅在訓練集上 fit，避免資料洩漏。
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """在訓練集上 fit StandardScaler。

    Args:
        X_train: 訓練集特徵陣列。

    Returns:
        已 fit 的 StandardScaler 物件。
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    logger.info("StandardScaler fit 完成：%d 個特徵", X_train.shape[1])
    return scaler


def transform_features(
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """使用已 fit 的 scaler 轉換特徵。

    Args:
        scaler: 已 fit 的 StandardScaler 物件。
        X: 待轉換的特徵陣列。

    Returns:
        標準化後的特徵陣列。
    """
    return scaler.transform(X)


def fit_transform_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """在訓練集上 fit 並同時轉換訓練集與測試集。

    Args:
        X_train: 訓練集特徵陣列。
        X_test: 測試集特徵陣列。

    Returns:
        (X_train_scaled, X_test_scaled, scaler) 三元組：
        - X_train_scaled: 標準化後的訓練集特徵。
        - X_test_scaled: 標準化後的測試集特徵。
        - scaler: 已 fit 的 StandardScaler 物件。
    """
    scaler = fit_scaler(X_train)
    X_train_scaled = transform_features(scaler, X_train)
    X_test_scaled = transform_features(scaler, X_test)
    return X_train_scaled, X_test_scaled, scaler
