"""前瞻指標滑動視窗管線單元測試。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import ForwardIndicatorData, preprocess_forward_pipeline


@pytest.fixture
def large_daily_df():
    """建立 300 筆模擬每日行情 DataFrame，足以通過前處理管線。"""
    n = 300
    np.random.seed(42)
    base_price = 500.0
    prices = base_price + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-02", periods=n, freq="B"),
        "SecurityCode": ["2330"] * n,
        "OpeningPrice": prices + np.random.randn(n),
        "HighestPrice": prices + abs(np.random.randn(n)) * 3,
        "LowestPrice": prices - abs(np.random.randn(n)) * 3,
        "ClosingPrice": prices,
        "TradeVolume": np.random.randint(20000, 50000, n).astype(float),
    })


@pytest.fixture
def small_daily_df():
    """建立 30 筆模擬每日行情 DataFrame，資料量不足。"""
    n = 30
    np.random.seed(42)
    base_price = 500.0
    prices = base_price + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "SecurityCode": ["2330"] * n,
        "OpeningPrice": prices + np.random.randn(n),
        "HighestPrice": prices + abs(np.random.randn(n)) * 3,
        "LowestPrice": prices - abs(np.random.randn(n)) * 3,
        "ClosingPrice": prices,
        "TradeVolume": np.random.randint(20000, 50000, n).astype(float),
    })


class TestPreprocessForwardPipeline:
    """preprocess_forward_pipeline 函式測試。"""

    def test_returns_forward_indicator_data_type(self, large_daily_df):
        """應回傳 ForwardIndicatorData 實例。"""
        result = preprocess_forward_pipeline(
            large_daily_df, window_size=20, horizon=5,
        )
        assert isinstance(result, ForwardIndicatorData)

    def test_feature_dimensions(self, large_daily_df):
        """X 維度應等於 window_size * n_features。"""
        window_size = 20
        result = preprocess_forward_pipeline(
            large_daily_df, window_size=window_size, horizon=5,
        )
        n_features = len(result.feature_names)
        expected_dim = window_size * n_features
        assert result.X_train.shape[1] == expected_dim
        assert result.X_test.shape[1] == expected_dim

    def test_base_prices_test_length(self, large_daily_df):
        """base_prices_test 長度應與測試集一致。"""
        result = preprocess_forward_pipeline(
            large_daily_df, window_size=20, horizon=5,
        )
        assert len(result.base_prices_test) == len(result.y_test)

    def test_target_dates_test_length(self, large_daily_df):
        """target_dates_test 長度應與測試集一致。"""
        result = preprocess_forward_pipeline(
            large_daily_df, window_size=20, horizon=5,
        )
        assert len(result.target_dates_test) == len(result.y_test)

    def test_insufficient_data_raises_error(self, small_daily_df):
        """資料量不足時應拋出 ValueError。"""
        with pytest.raises(ValueError):
            preprocess_forward_pipeline(
                small_daily_df, window_size=60, horizon=20,
            )

    def test_window_size_and_horizon_stored(self, large_daily_df):
        """window_size 與 horizon 應正確儲存在結果中。"""
        result = preprocess_forward_pipeline(
            large_daily_df, window_size=30, horizon=10,
        )
        assert result.window_size == 30
        assert result.horizon == 10

    def test_train_test_split_is_temporal(self, large_daily_df):
        """訓練集日期應全部早於測試集日期。"""
        result = preprocess_forward_pipeline(
            large_daily_df, window_size=20, horizon=5,
        )
        if result.train_dates is not None and result.test_dates is not None:
            max_train = pd.Timestamp(result.train_dates.max())
            min_test = pd.Timestamp(result.test_dates.min())
            assert max_train < min_test
