"""模型訓練模組。

提供 XGBoost 模型訓練、預測與評估功能。
"""

from src.model.xgboost_model import (
    detect_device,
    evaluate_model,
    get_default_params,
    predict,
    train_xgboost,
)

__all__ = [
    "detect_device",
    "get_default_params",
    "train_xgboost",
    "predict",
    "evaluate_model",
]
