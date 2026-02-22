"""滑動視窗模組單元測試。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineer import create_sliding_windows
from src.preprocessing.pipeline import SlidingWindowData, preprocess_sliding_pipeline


@pytest.fixture
def daily_price_df_50():
    """建立 50 筆模擬每日行情 DataFrame。"""
    np.random.seed(42)
    n = 50
    base_price = 580.0
    prices = base_price + np.cumsum(np.random.randn(n) * 2)

    return pd.DataFrame({
        "Date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "SecurityCode": ["2330"] * n,
        "OpeningPrice": prices,
        "HighestPrice": prices + np.random.uniform(1, 5, n),
        "LowestPrice": prices - np.random.uniform(1, 5, n),
        "ClosingPrice": prices + np.random.randn(n) * 0.5,
        "TradeVolume": np.random.randint(20000, 40000, n).astype(float),
    })


@pytest.fixture
def features_target_dates(daily_price_df_50):
    """從 50 筆資料建立特徵、目標、日期（49 筆，去 NaN）。"""
    from src.preprocessing.feature_engineer import build_feature_target
    return build_feature_target(daily_price_df_50)


class TestCreateSlidingWindows:
    """create_sliding_windows 函式測試。"""

    def test_output_shape(self, features_target_dates):
        """驗證輸出 shape：49 筆資料、window_size=5 → (44, 25)。"""
        features, target, dates = features_target_dates
        X, y, d = create_sliding_windows(features, target, dates, window_size=5)

        assert X.shape == (44, 25)  # 49 - 5 = 44 樣本，5 * 5 = 25 維
        assert y.shape == (44,)
        assert len(d) == 44

    def test_target_values_correct(self, features_target_dates):
        """驗證目標值為視窗最後一天的隔天收盤價。"""
        features, target, dates = features_target_dates
        X, y, d = create_sliding_windows(features, target, dates, window_size=5)

        # 樣本 0 的目標 = target[4]（第 5 天的隔天收盤價）
        assert y[0] == pytest.approx(target.iloc[4])
        # 樣本 1 的目標 = target[5]
        assert y[1] == pytest.approx(target.iloc[5])
        # 最後一個樣本 i=43 的目標 = target[43+5-1] = target[47]
        assert y[-1] == pytest.approx(target.iloc[47])

    def test_feature_window_content(self, features_target_dates):
        """驗證展平的特徵內容正確。"""
        features, target, dates = features_target_dates
        X, y, d = create_sliding_windows(features, target, dates, window_size=5)

        # 樣本 0 應包含 features[0:5] 展平
        expected = features.iloc[0:5].values.flatten()
        np.testing.assert_array_equal(X[0], expected)

        # 樣本 2 應包含 features[2:7] 展平
        expected = features.iloc[2:7].values.flatten()
        np.testing.assert_array_equal(X[2], expected)

    def test_no_time_leakage(self, features_target_dates):
        """驗證無時間洩漏：視窗不包含目標日期的資料。"""
        features, target, dates = features_target_dates
        X, y, d = create_sliding_windows(features, target, dates, window_size=5)

        # 樣本 i 的視窗為 features[i:i+5]，目標日期為 dates[i+5]
        # 視窗最後一天為 dates[i+4]，應早於目標日期 dates[i+5]
        for i in range(min(5, len(d))):
            window_last_date = dates.iloc[i + 4]
            target_date = d.iloc[i]
            assert window_last_date < target_date

    def test_dates_alignment(self, features_target_dates):
        """驗證日期對齊：window_dates 從 dates[window_size] 開始。"""
        features, target, dates = features_target_dates
        _, _, d = create_sliding_windows(features, target, dates, window_size=5)

        assert d.iloc[0] == dates.iloc[5]
        assert d.iloc[-1] == dates.iloc[48]

    def test_insufficient_data_raises(self, features_target_dates):
        """資料不足時應拋出 ValueError。"""
        features, target, dates = features_target_dates
        with pytest.raises(ValueError, match="不足"):
            create_sliding_windows(features, target, dates, window_size=100)

    def test_window_size_1(self, features_target_dates):
        """window_size=1 時每個樣本等於單天特徵。"""
        features, target, dates = features_target_dates
        X, y, d = create_sliding_windows(features, target, dates, window_size=1)

        assert X.shape == (48, 5)  # 49 - 1 = 48
        np.testing.assert_array_equal(X[0], features.iloc[0].values)


class TestPreprocessSlidingPipeline:
    """preprocess_sliding_pipeline 端到端測試。"""

    def test_returns_sliding_window_data(self, daily_price_df_50):
        """應回傳 SlidingWindowData 實例。"""
        result = preprocess_sliding_pipeline(
            daily_price_df_50, window_size=5,
        )
        assert isinstance(result, SlidingWindowData)

    def test_data_shapes(self, daily_price_df_50):
        """驗證資料形狀。"""
        result = preprocess_sliding_pipeline(
            daily_price_df_50, window_size=5, test_ratio=0.2,
        )
        # 50 筆 → 49 筆（去 NaN）→ 44 個視窗樣本
        total = result.X_train.shape[0] + result.X_test.shape[0]
        assert total == 44
        assert result.X_train.shape[1] == 25  # 5 * 5
        assert result.X_test.shape[1] == 25

    def test_y_not_scaled(self, daily_price_df_50):
        """y 應保持原始值。"""
        result = preprocess_sliding_pipeline(
            daily_price_df_50, window_size=5,
        )
        all_y = np.concatenate([result.y_train, result.y_test])
        # 模擬資料的股價範圍在 560~610 左右
        assert np.all(all_y > 500)
        assert np.all(all_y < 700)

    def test_train_features_standardized(self, daily_price_df_50):
        """訓練集特徵標準化後 mean 接近 0。"""
        result = preprocess_sliding_pipeline(
            daily_price_df_50, window_size=5,
        )
        train_mean = np.mean(result.X_train, axis=0)
        assert np.allclose(train_mean, 0, atol=1e-10)

    def test_feature_names_and_window_size(self, daily_price_df_50):
        """feature_names 與 window_size 應正確。"""
        result = preprocess_sliding_pipeline(
            daily_price_df_50, window_size=5,
        )
        expected_names = [
            "OpeningPrice", "HighestPrice", "LowestPrice",
            "ClosingPrice", "TradeVolume",
        ]
        assert result.feature_names == expected_names
        assert result.window_size == 5

    def test_dates_no_leakage(self, daily_price_df_50):
        """訓練集最後日期應早於測試集第一日期。"""
        result = preprocess_sliding_pipeline(
            daily_price_df_50, window_size=5,
        )
        assert result.train_dates.iloc[-1] < result.test_dates.iloc[0]
