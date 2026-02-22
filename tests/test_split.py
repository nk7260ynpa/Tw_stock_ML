"""時間序列切分模組單元測試。"""

import pandas as pd
import pytest

from src.preprocessing.split import time_series_split


@pytest.fixture
def sample_data():
    """建立 10 筆模擬特徵與目標資料。"""
    X = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })
    y = pd.Series([100.0, 200.0, 300.0, 400.0, 500.0,
                    600.0, 700.0, 800.0, 900.0, 1000.0])
    dates = pd.Series(pd.date_range("2024-01-02", periods=10, freq="B"))
    return X, y, dates


class TestTimeSeriesSplit:
    """time_series_split 函式測試。"""

    def test_default_ratio(self, sample_data):
        """預設 0.2 比例，10 筆中 8 筆訓練、2 筆測試。"""
        X, y, dates = sample_data
        result = time_series_split(X, y, dates=dates)
        assert len(result["X_train"]) == 8
        assert len(result["X_test"]) == 2
        assert len(result["y_train"]) == 8
        assert len(result["y_test"]) == 2

    def test_custom_ratio(self, sample_data):
        """自訂比例 0.3，10 筆中 7 筆訓練、3 筆測試。"""
        X, y, _ = sample_data
        result = time_series_split(X, y, test_ratio=0.3)
        assert len(result["X_train"]) == 7
        assert len(result["X_test"]) == 3

    def test_time_order_preserved(self, sample_data):
        """訓練集在前、測試集在後，無 shuffle。"""
        X, y, _ = sample_data
        result = time_series_split(X, y)
        # 訓練集最後一筆 < 測試集第一筆
        assert result["X_train"]["feature_a"].iloc[-1] < \
            result["X_test"]["feature_a"].iloc[0]

    def test_no_data_leakage(self, sample_data):
        """訓練集最後日期應早於測試集第一日期。"""
        X, y, dates = sample_data
        result = time_series_split(X, y, dates=dates)
        assert result["train_dates"].iloc[-1] < result["test_dates"].iloc[0]

    def test_dates_split(self, sample_data):
        """日期應正確切分。"""
        X, y, dates = sample_data
        result = time_series_split(X, y, dates=dates)
        assert len(result["train_dates"]) == 8
        assert len(result["test_dates"]) == 2

    def test_dates_none_when_not_provided(self, sample_data):
        """未提供日期時，train_dates 與 test_dates 應為 None。"""
        X, y, _ = sample_data
        result = time_series_split(X, y)
        assert result["train_dates"] is None
        assert result["test_dates"] is None

    def test_invalid_ratio_raises(self, sample_data):
        """test_ratio 超出範圍應拋出 ValueError。"""
        X, y, _ = sample_data
        with pytest.raises(ValueError, match="test_ratio"):
            time_series_split(X, y, test_ratio=0.0)
        with pytest.raises(ValueError, match="test_ratio"):
            time_series_split(X, y, test_ratio=1.0)
        with pytest.raises(ValueError, match="test_ratio"):
            time_series_split(X, y, test_ratio=-0.1)

    def test_y_values_match(self, sample_data):
        """切分後的 y 值應與原始資料一致。"""
        X, y, _ = sample_data
        result = time_series_split(X, y)
        # 訓練集 y 應為前 8 筆
        assert list(result["y_train"]) == [100.0, 200.0, 300.0, 400.0,
                                            500.0, 600.0, 700.0, 800.0]
        # 測試集 y 應為後 2 筆
        assert list(result["y_test"]) == [900.0, 1000.0]
