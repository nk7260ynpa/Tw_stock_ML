"""XGBoost 模型訓練模組。

提供 XGBoost 迴歸模型的訓練、預測、評估、儲存與裝置偵測功能。
"""

from pathlib import Path

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


def save_model(
    model: xgb.XGBRegressor,
    path: str | Path,
) -> Path:
    """將訓練完成的模型儲存至指定路徑。

    Args:
        model: 訓練完成的 XGBRegressor 模型。
        path: 儲存路徑（含檔名），例如 "model/xgboost_2330.json"。

    Returns:
        儲存的完整路徑。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info("模型已儲存至 %s", path)
    return path


def get_small_data_params(device: str) -> dict:
    """取得小資料量專用 XGBoost 參數。

    針對樣本數少（< 200 筆）的情境，使用較保守的超參數，
    搭配 L1/L2 正則化與 subsample 降低過擬合風險。

    Args:
        device: 裝置字串，"cuda" 或 "cpu"。

    Returns:
        XGBoost 參數字典。
    """
    return {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 3,
        "early_stopping_rounds": 30,
        "device": device,
    }


def evaluate_return_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_prices: np.ndarray,
) -> dict:
    """評估報酬率模型，同時計算報酬率指標與價格指標。

    預測報酬率後逆推絕對價格：predicted_price = base_price * (1 + predicted_return)。

    Args:
        model: 訓練完成的 XGBRegressor 模型。
        X_test: 測試特徵陣列。
        y_test: 測試目標陣列（報酬率）。
        base_prices: 基準價格陣列，長度與 y_test 相同。

    Returns:
        包含報酬率指標與價格指標的字典：
        - return_MAE: 報酬率 MAE。
        - return_RMSE: 報酬率 RMSE。
        - price_MAE: 逆推價格 MAE。
        - price_RMSE: 逆推價格 RMSE。
        - price_MAPE: 逆推價格 MAPE (%)。
        - directional_accuracy: 方向正確率。
    """
    y_pred_return = predict(model, X_test)

    # 報酬率指標
    return_mae_val = mae(y_test, y_pred_return)
    return_rmse_val = rmse(y_test, y_pred_return)

    # 逆推絕對價格
    actual_prices = base_prices * (1 + y_test)
    predicted_prices = base_prices * (1 + y_pred_return)

    # 價格指標
    price_mae_val = mae(actual_prices, predicted_prices)
    price_rmse_val = rmse(actual_prices, predicted_prices)
    price_mape_val = mape(actual_prices, predicted_prices)

    # 方向正確率（用逆推價格計算）
    da = directional_accuracy(actual_prices, predicted_prices)

    results = {
        "return_MAE": return_mae_val,
        "return_RMSE": return_rmse_val,
        "price_MAE": price_mae_val,
        "price_RMSE": price_rmse_val,
        "price_MAPE": price_mape_val,
        "directional_accuracy": da,
    }

    logger.info(
        "報酬率模型評估完成 — 報酬率 MAE: %.6f, 價格 MAE: %.2f, "
        "價格 MAPE: %.2f%%, 方向正確率: %.2f%%",
        results["return_MAE"],
        results["price_MAE"],
        results["price_MAPE"],
        results["directional_accuracy"] * 100,
    )
    return results
