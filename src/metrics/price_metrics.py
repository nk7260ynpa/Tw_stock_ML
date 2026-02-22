"""價格距離指標模組。

提供預測股價與實際股價之間的距離衡量函式。
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """計算平均絕對誤差（Mean Absolute Error）。

    Args:
        y_true: 實際值陣列。
        y_pred: 預測值陣列。

    Returns:
        平均絕對誤差。
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """計算均方根誤差（Root Mean Squared Error）。

    Args:
        y_true: 實際值陣列。
        y_pred: 預測值陣列。

    Returns:
        均方根誤差。
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """計算平均絕對百分比誤差（Mean Absolute Percentage Error）。

    回傳百分比形式（例如 5.0 代表 5%）。
    當 y_true 包含 0 時，該筆資料會被排除計算。

    Args:
        y_true: 實際值陣列。
        y_pred: 預測值陣列。

    Returns:
        平均絕對百分比誤差（百分比形式）。
    """
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
