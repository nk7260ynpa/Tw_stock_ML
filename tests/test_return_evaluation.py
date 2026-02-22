"""報酬率評估測試。"""

import numpy as np
import pytest
import xgboost as xgb

from src.model.xgboost_model import evaluate_return_model, get_small_data_params


class TestGetSmallDataParams:
    """小資料參數測試。"""

    def test_returns_dict(self):
        """回傳字典。"""
        params = get_small_data_params("cpu")
        assert isinstance(params, dict)

    def test_contains_regularization(self):
        """包含正則化參數。"""
        params = get_small_data_params("cpu")
        assert "reg_alpha" in params
        assert "reg_lambda" in params
        assert params["reg_alpha"] > 0
        assert params["reg_lambda"] > 0

    def test_contains_subsample(self):
        """包含 subsample 參數。"""
        params = get_small_data_params("cpu")
        assert "subsample" in params
        assert 0 < params["subsample"] <= 1

    def test_max_depth_conservative(self):
        """max_depth 較保守（<= 4）。"""
        params = get_small_data_params("cpu")
        assert params["max_depth"] <= 4

    def test_device_setting(self):
        """裝置設定正確。"""
        params = get_small_data_params("cuda")
        assert params["device"] == "cuda"
        params_cpu = get_small_data_params("cpu")
        assert params_cpu["device"] == "cpu"

    def test_early_stopping(self):
        """包含 early_stopping_rounds。"""
        params = get_small_data_params("cpu")
        assert "early_stopping_rounds" in params
        assert params["early_stopping_rounds"] > 0


class TestEvaluateReturnModel:
    """報酬率模型評估測試。"""

    @pytest.fixture()
    def trained_model(self):
        """用合成資料訓練簡易模型。"""
        np.random.seed(42)
        n_train = 50
        n_features = 10
        X_train = np.random.randn(n_train, n_features)
        y_train = np.random.randn(n_train) * 0.01

        model = xgb.XGBRegressor(
            n_estimators=10, max_depth=2, verbosity=0,
        )
        model.fit(X_train, y_train)
        return model

    @pytest.fixture()
    def test_data(self):
        """合成測試資料。"""
        np.random.seed(123)
        n_test = 20
        n_features = 10
        X_test = np.random.randn(n_test, n_features)
        y_test = np.random.randn(n_test) * 0.01
        base_prices = np.full(n_test, 1000.0)
        return X_test, y_test, base_prices

    def test_returns_all_keys(self, trained_model, test_data):
        """回傳所有預期指標鍵值。"""
        X_test, y_test, base_prices = test_data
        results = evaluate_return_model(trained_model, X_test, y_test, base_prices)
        expected_keys = {
            "return_MAE", "return_RMSE",
            "price_MAE", "price_RMSE", "price_MAPE",
            "directional_accuracy",
        }
        assert set(results.keys()) == expected_keys

    def test_all_values_are_float(self, trained_model, test_data):
        """所有指標值為浮點數。"""
        X_test, y_test, base_prices = test_data
        results = evaluate_return_model(trained_model, X_test, y_test, base_prices)
        for key, val in results.items():
            assert isinstance(val, float), f"{key} 不是 float"

    def test_non_negative_metrics(self, trained_model, test_data):
        """MAE、RMSE、MAPE 為非負數。"""
        X_test, y_test, base_prices = test_data
        results = evaluate_return_model(trained_model, X_test, y_test, base_prices)
        assert results["return_MAE"] >= 0
        assert results["return_RMSE"] >= 0
        assert results["price_MAE"] >= 0
        assert results["price_RMSE"] >= 0
        assert results["price_MAPE"] >= 0

    def test_directional_accuracy_range(self, trained_model, test_data):
        """方向正確率在 0~1 之間。"""
        X_test, y_test, base_prices = test_data
        results = evaluate_return_model(trained_model, X_test, y_test, base_prices)
        assert 0 <= results["directional_accuracy"] <= 1

    def test_price_reconstruction_reasonable(self, trained_model, test_data):
        """價格逆推結果合理（base=1000，日報酬率 ~1%，價格 MAE 應遠小於 100）。"""
        X_test, y_test, base_prices = test_data
        results = evaluate_return_model(trained_model, X_test, y_test, base_prices)
        assert results["price_MAE"] < 100
