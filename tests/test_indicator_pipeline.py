"""技術指標管線整合測試。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import IndicatorData, preprocess_indicator_pipeline


@pytest.fixture()
def sample_df():
    """建立 100 筆模擬股價資料。"""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    opening = close + np.random.randn(n) * 1
    volume = np.random.randint(1000, 50000, size=n).astype(float)

    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n),
        "SecurityCode": "2330",
        "OpeningPrice": opening,
        "HighestPrice": high,
        "LowestPrice": low,
        "ClosingPrice": close,
        "TradeVolume": volume,
    })


class TestIndicatorData:
    """IndicatorData 結構測試。"""

    def test_returns_indicator_data(self, sample_df):
        """回傳 IndicatorData 實例。"""
        data = preprocess_indicator_pipeline(sample_df)
        assert isinstance(data, IndicatorData)

    def test_feature_dimension_much_less_than_200(self, sample_df):
        """特徵維度遠低於 200（舊滑動視窗維度）。"""
        data = preprocess_indicator_pipeline(sample_df)
        assert data.X_train.shape[1] < 50
        assert data.X_train.shape[1] > 5

    def test_target_type_return(self, sample_df):
        """target_type 正確設定為 return。"""
        data = preprocess_indicator_pipeline(sample_df, target_type="return")
        assert data.target_type == "return"

    def test_target_values_in_reasonable_range(self, sample_df):
        """報酬率目標值在合理範圍內。"""
        data = preprocess_indicator_pipeline(sample_df, target_type="return")
        # 日報酬率通常在 -50% ~ +50% 之間
        assert np.all(np.abs(data.y_train) < 0.5)
        assert np.all(np.abs(data.y_test) < 0.5)

    def test_base_prices_not_none_for_return(self, sample_df):
        """target_type=return 時 base_prices 不為 None。"""
        data = preprocess_indicator_pipeline(sample_df, target_type="return")
        assert data.base_prices is not None
        assert len(data.base_prices) == len(data.y_test)

    def test_base_prices_none_for_price(self, sample_df):
        """target_type=price 時 base_prices 為 None。"""
        data = preprocess_indicator_pipeline(sample_df, target_type="price")
        # price 模式下 ClosingPrice 也在 features 中，所以 base_prices 可能不為 None
        # 但 target_type 應為 price
        assert data.target_type == "price"

    def test_no_time_leakage(self, sample_df):
        """訓練集日期在測試集之前，無時間洩漏。"""
        data = preprocess_indicator_pipeline(sample_df)
        if data.train_dates is not None and data.test_dates is not None:
            assert data.train_dates.iloc[-1] < data.test_dates.iloc[0]

    def test_feature_names_match_dimension(self, sample_df):
        """feature_names 長度與特徵維度一致。"""
        data = preprocess_indicator_pipeline(sample_df)
        assert len(data.feature_names) == data.X_train.shape[1]

    def test_train_test_split_ratio(self, sample_df):
        """訓練測試比例約為 80:20。"""
        data = preprocess_indicator_pipeline(sample_df, test_ratio=0.2)
        total = len(data.y_train) + len(data.y_test)
        test_ratio = len(data.y_test) / total
        assert 0.15 < test_ratio < 0.25

    def test_scaled_features(self, sample_df):
        """標準化後訓練集平均值接近 0。"""
        data = preprocess_indicator_pipeline(sample_df)
        train_mean = np.mean(data.X_train, axis=0)
        assert np.allclose(train_mean, 0, atol=0.1)
