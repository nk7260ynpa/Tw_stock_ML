"""XGBoost 模型模組單元測試。"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_regression

from src.model.xgboost_model import (
    detect_device,
    evaluate_model,
    get_default_params,
    predict,
    save_model,
    train_xgboost,
)


@pytest.fixture
def regression_data():
    """建立合成迴歸資料。"""
    X, y = make_regression(
        n_samples=200, n_features=25, noise=10.0, random_state=42,
    )
    # 切分前 160 筆為訓練，後 40 筆為測試
    return {
        "X_train": X[:160],
        "y_train": y[:160],
        "X_test": X[160:],
        "y_test": y[160:],
    }


class TestDetectDevice:
    """detect_device 函式測試。"""

    def test_returns_valid_device(self):
        """應回傳 'cpu' 或 'cuda'。"""
        device = detect_device()
        assert device in ("cpu", "cuda")


class TestGetDefaultParams:
    """get_default_params 函式測試。"""

    def test_cpu_params(self):
        """CPU 參數應包含必要鍵值。"""
        params = get_default_params("cpu")
        assert params["device"] == "cpu"
        assert params["n_estimators"] == 500
        assert params["max_depth"] == 6
        assert params["learning_rate"] == 0.05
        assert params["early_stopping_rounds"] == 50

    def test_cuda_params(self):
        """CUDA 參數應設定 device 為 'cuda'。"""
        params = get_default_params("cuda")
        assert params["device"] == "cuda"


class TestTrainXgboost:
    """train_xgboost 函式測試。"""

    def test_train_without_validation(self, regression_data):
        """不提供驗證集時應能正常訓練。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        assert model is not None

    def test_train_with_validation(self, regression_data):
        """提供驗證集時應使用 early stopping。"""
        params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "early_stopping_rounds": 5,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            regression_data["X_test"],
            regression_data["y_test"],
            params=params,
        )
        assert model is not None
        assert model.best_iteration is not None


class TestPredict:
    """predict 函式測試。"""

    def test_predict_shape(self, regression_data):
        """預測結果 shape 應與測試集一致。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        y_pred = predict(model, regression_data["X_test"])
        assert y_pred.shape == regression_data["y_test"].shape


class TestEvaluateModel:
    """evaluate_model 函式測試。"""

    def test_returns_all_metrics(self, regression_data):
        """應回傳包含 MAE、RMSE、MAPE、directional_accuracy 的字典。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        results = evaluate_model(
            model,
            regression_data["X_test"],
            regression_data["y_test"],
        )
        assert "MAE" in results
        assert "RMSE" in results
        assert "MAPE" in results
        assert "directional_accuracy" in results

    def test_metric_types(self, regression_data):
        """各項指標應為浮點數。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        results = evaluate_model(
            model,
            regression_data["X_test"],
            regression_data["y_test"],
        )
        for key, value in results.items():
            assert isinstance(value, float), f"{key} 應為 float，實際為 {type(value)}"

    def test_mae_rmse_non_negative(self, regression_data):
        """MAE 與 RMSE 應為非負值。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        results = evaluate_model(
            model,
            regression_data["X_test"],
            regression_data["y_test"],
        )
        assert results["MAE"] >= 0
        assert results["RMSE"] >= 0

    def test_directional_accuracy_range(self, regression_data):
        """方向正確率應在 0.0 ~ 1.0 之間。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        results = evaluate_model(
            model,
            regression_data["X_test"],
            regression_data["y_test"],
        )
        assert 0.0 <= results["directional_accuracy"] <= 1.0


class TestSaveModel:
    """save_model 函式測試。"""

    def test_save_creates_file(self, regression_data, tmp_path):
        """應成功儲存模型檔案。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        model_path = tmp_path / "test_model.json"
        result = save_model(model, model_path)
        assert result.exists()
        assert result.stat().st_size > 0

    def test_save_creates_parent_dirs(self, regression_data, tmp_path):
        """應自動建立不存在的父資料夾。"""
        params = {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "device": "cpu",
        }
        model = train_xgboost(
            regression_data["X_train"],
            regression_data["y_train"],
            params=params,
        )
        model_path = tmp_path / "subdir" / "model.json"
        result = save_model(model, model_path)
        assert result.exists()
