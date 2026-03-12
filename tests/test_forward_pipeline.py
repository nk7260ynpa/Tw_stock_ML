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


@pytest.fixture
def daily_df_with_zero():
    """建立含 ClosingPrice=0 的 300 筆模擬行情，模擬零值問題。"""
    n = 300
    np.random.seed(42)
    base_price = 500.0
    prices = base_price + np.cumsum(np.random.randn(n) * 2)
    prices = np.maximum(prices, 1.0)
    # 在中間插入零值記錄
    prices[150] = 0.0
    opening = prices + np.random.randn(n)
    opening[150] = 0.0
    highest = prices + abs(np.random.randn(n)) * 3
    highest[150] = 0.0
    lowest = prices - abs(np.random.randn(n)) * 3
    lowest[150] = 0.0
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-02", periods=n, freq="B"),
        "SecurityCode": ["2317"] * n,
        "OpeningPrice": opening,
        "HighestPrice": highest,
        "LowestPrice": lowest,
        "ClosingPrice": prices,
        "TradeVolume": np.random.randint(20000, 50000, n).astype(float),
    })


class TestForwardPipelineWithZeroValues:
    """含零值資料的前瞻管線測試。"""

    def test_pipeline_completes_with_zero_price(self, daily_df_with_zero):
        """含零值的資料應能正常完成管線處理，不會因 inf 而失敗。"""
        result = preprocess_forward_pipeline(
            daily_df_with_zero, window_size=20, horizon=5,
        )
        assert isinstance(result, ForwardIndicatorData)
        assert not np.isinf(result.X_train).any(), "X_train 含有 inf 值"
        assert not np.isinf(result.X_test).any(), "X_test 含有 inf 值"
        assert not np.isinf(result.y_train).any(), "y_train 含有 inf 值"
        assert not np.isinf(result.y_test).any(), "y_test 含有 inf 值"
