"""方向正確率指標模組。

提供預測漲跌方向是否與實際一致的衡量函式。
"""

import numpy as np


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """計算方向正確率（Directional Accuracy）。

    模擬真實預測情境，模型只能看到昨天的實際收盤價。
    比較以下兩者的正負號是否一致：
    - 實際方向：y_true[i] - y_true[i-1]
    - 預測方向：y_pred[i] - y_true[i-1]

    第一筆資料無前一天可比較，從第二筆開始計算。

    Args:
        y_true: 實際值陣列。
        y_pred: 預測值陣列。

    Returns:
        方向正確率（0.0 ~ 1.0）。若長度不足以比較則回傳 0.0。
    """
    if len(y_true) < 2:
        return 0.0

    actual_direction = y_true[1:] - y_true[:-1]
    predicted_direction = y_pred[1:] - y_true[:-1]

    # 正負號一致即為正確（包含雙方皆為 0 的情況）
    correct = np.sign(actual_direction) == np.sign(predicted_direction)
    return float(np.mean(correct))
