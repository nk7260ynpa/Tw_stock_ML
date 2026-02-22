"""特徵標準化模組單元測試。"""

import numpy as np
import pytest

from src.preprocessing.scaler import (
    fit_scaler,
    fit_transform_train_test,
    transform_features,
)


@pytest.fixture
def train_test_data():
    """建立訓練與測試特徵陣列。"""
    X_train = np.array([
        [100.0, 1000.0],
        [200.0, 2000.0],
        [300.0, 3000.0],
        [400.0, 4000.0],
        [500.0, 5000.0],
    ])
    X_test = np.array([
        [600.0, 6000.0],
        [700.0, 7000.0],
    ])
    return X_train, X_test


class TestFitScaler:
    """fit_scaler 函式測試。"""

    def test_returns_scaler(self, train_test_data):
        """應回傳 StandardScaler 物件。"""
        X_train, _ = train_test_data
        scaler = fit_scaler(X_train)
        assert hasattr(scaler, "mean_")
        assert hasattr(scaler, "scale_")

    def test_scaler_mean(self, train_test_data):
        """scaler 的 mean 應等於訓練集的平均值。"""
        X_train, _ = train_test_data
        scaler = fit_scaler(X_train)
        expected_mean = np.mean(X_train, axis=0)
        np.testing.assert_array_almost_equal(scaler.mean_, expected_mean)

    def test_scaler_std(self, train_test_data):
        """scaler 的 scale 應等於訓練集的標準差。"""
        X_train, _ = train_test_data
        scaler = fit_scaler(X_train)
        expected_std = np.std(X_train, axis=0)
        np.testing.assert_array_almost_equal(scaler.scale_, expected_std)


class TestTransformFeatures:
    """transform_features 函式測試。"""

    def test_train_mean_near_zero(self, train_test_data):
        """訓練集標準化後，各特徵平均值應接近 0。"""
        X_train, _ = train_test_data
        scaler = fit_scaler(X_train)
        X_scaled = transform_features(scaler, X_train)
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)

    def test_train_std_near_one(self, train_test_data):
        """訓練集標準化後，各特徵標準差應接近 1。"""
        X_train, _ = train_test_data
        scaler = fit_scaler(X_train)
        X_scaled = transform_features(scaler, X_train)
        # ddof=0 與 StandardScaler 一致
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)

    def test_shape_preserved(self, train_test_data):
        """標準化後形狀應不變。"""
        X_train, _ = train_test_data
        scaler = fit_scaler(X_train)
        X_scaled = transform_features(scaler, X_train)
        assert X_scaled.shape == X_train.shape


class TestFitTransformTrainTest:
    """fit_transform_train_test 函式測試。"""

    def test_train_standardized(self, train_test_data):
        """訓練集標準化後 mean≈0、std≈1。"""
        X_train, X_test = train_test_data
        X_train_scaled, _, _ = fit_transform_train_test(X_train, X_test)
        assert np.allclose(np.mean(X_train_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_train_scaled, axis=0), 1, atol=1e-10)

    def test_test_uses_train_stats(self, train_test_data):
        """測試集應使用訓練集的統計量標準化。"""
        X_train, X_test = train_test_data
        _, X_test_scaled, scaler = fit_transform_train_test(X_train, X_test)
        # 手算：train_mean=[300, 3000], train_std=[141.42, 1414.21]
        # X_test[0] = (600-300)/141.42 ≈ 2.121
        expected_first = (600.0 - 300.0) / np.std(X_train[:, 0])
        assert X_test_scaled[0, 0] == pytest.approx(expected_first, rel=1e-3)

    def test_inverse_transform(self, train_test_data):
        """inverse_transform 應能正確還原原始值。"""
        X_train, X_test = train_test_data
        X_train_scaled, X_test_scaled, scaler = fit_transform_train_test(
            X_train, X_test,
        )
        X_train_restored = scaler.inverse_transform(X_train_scaled)
        X_test_restored = scaler.inverse_transform(X_test_scaled)
        np.testing.assert_array_almost_equal(X_train_restored, X_train)
        np.testing.assert_array_almost_equal(X_test_restored, X_test)

    def test_returns_scaler(self, train_test_data):
        """應回傳 scaler 物件。"""
        X_train, X_test = train_test_data
        _, _, scaler = fit_transform_train_test(X_train, X_test)
        assert hasattr(scaler, "mean_")
