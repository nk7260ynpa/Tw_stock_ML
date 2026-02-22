"""XGBoost 模型訓練模組。

提供 XGBoost 迴歸模型的訓練、預測、評估與裝置偵測功能。
"""

import numpy as np
import xgboost as xgb

from src.metrics.direction_metrics import directional_accuracy
from src.metrics.price_metrics import mae, mape, rmse
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def detect_device() -> str:
    """偵測可用裝置，回傳 "cuda" 或 "cpu"。

    XGBoost 僅支援 NVIDIA CUDA GPU，不支援 Apple Metal。
    當 CUDA 不可用時自動 fallback 至 CPU。

    Returns:
        裝置字串："cuda" 或 "cpu"。
    """
    try:
        # 嘗試建立小型模型測試 CUDA 是否可用
        test_model = xgb.XGBRegressor(
            n_estimators=1, device="cuda", verbosity=0,
        )
        test_X = np.array([[1.0, 2.0]])
        test_y = np.array([1.0])
        test_model.fit(test_X, test_y)
        logger.info("偵測到 CUDA GPU，使用 GPU 加速")
        return "cuda"
    except xgb.core.XGBoostError:
        logger.info("CUDA 不可用，使用 CPU 訓練")
        return "cpu"


def get_default_params(device: str) -> dict:
    """取得 XGBoost 預設參數。

    Args:
        device: 裝置字串，"cuda" 或 "cpu"。

    Returns:
        XGBoost 參數字典。
    """
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "early_stopping_rounds": 50,
        "device": device,
    }


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
) -> xgb.XGBRegressor:
    """訓練 XGBoost 迴歸模型。

    Args:
        X_train: 訓練特徵陣列。
        y_train: 訓練目標陣列。
        X_val: 驗證特徵陣列，用於 early stopping。
        y_val: 驗證目標陣列，用於 early stopping。
        params: XGBoost 參數字典，None 時使用預設參數。

    Returns:
        訓練完成的 XGBRegressor 模型。
    """
    if params is None:
        device = detect_device()
        params = get_default_params(device)

    model = xgb.XGBRegressor(**params, verbosity=1)

    fit_kwargs: dict = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = True

    logger.info(
        "開始訓練 XGBoost：%d 筆訓練資料，%d 個特徵",
        X_train.shape[0],
        X_train.shape[1],
    )
    model.fit(X_train, y_train, **fit_kwargs)

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        logger.info("訓練完成，最佳迭代次數：%d", best_iteration)
    else:
        logger.info("訓練完成")

    return model


def predict(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """使用模型進行預測。

    Args:
        model: 訓練完成的 XGBRegressor 模型。
        X: 待預測的特徵陣列。

    Returns:
        預測值陣列。
    """
    return model.predict(X)


def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """評估模型，回傳各項指標。

    Args:
        model: 訓練完成的 XGBRegressor 模型。
        X_test: 測試特徵陣列。
        y_test: 測試目標陣列（原始值）。

    Returns:
        包含 MAE、RMSE、MAPE、方向正確率的字典。
    """
    y_pred = predict(model, X_test)

    results = {
        "MAE": mae(y_test, y_pred),
        "RMSE": rmse(y_test, y_pred),
        "MAPE": mape(y_test, y_pred),
        "directional_accuracy": directional_accuracy(y_test, y_pred),
    }

    logger.info(
        "模型評估完成 — MAE: %.2f, RMSE: %.2f, MAPE: %.2f%%, 方向正確率: %.2f%%",
        results["MAE"],
        results["RMSE"],
        results["MAPE"],
        results["directional_accuracy"] * 100,
    )
    return results
