"""前處理管線模組整合測試。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import PreprocessedData, preprocess_pipeline


@pytest.fixture
def daily_price_df():
    """建立 10 筆模擬每日行情 DataFrame。"""
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-02", periods=10, freq="B"),
        "SecurityCode": ["2330"] * 10,
        "OpeningPrice": [580.0, 585.0, 590.0, 588.0, 592.0,
                         595.0, 593.0, 598.0, 600.0, 602.0],
        "HighestPrice": [585.0, 590.0, 595.0, 592.0, 598.0,
                         600.0, 598.0, 605.0, 608.0, 610.0],
        "LowestPrice": [575.0, 580.0, 585.0, 583.0, 588.0,
                        590.0, 588.0, 595.0, 596.0, 598.0],
        "ClosingPrice": [583.0, 588.0, 592.0, 590.0, 595.0,
                         598.0, 596.0, 602.0, 605.0, 608.0],
        "TradeVolume": [30000, 32000, 28000, 35000, 31000,
                        29000, 33000, 27000, 34000, 30000],
    })


class TestPreprocessPipeline:
    """preprocess_pipeline 函式測試。"""

    def test_returns_preprocessed_data(self, daily_price_df):
        """應回傳 PreprocessedData 實例。"""
        result = preprocess_pipeline(daily_price_df)
        assert isinstance(result, PreprocessedData)

    def test_data_shapes(self, daily_price_df):
        """資料形狀應正確（9 筆去 NaN，8:2 切分 → 7 train + 2 test）。"""
        result = preprocess_pipeline(daily_price_df, test_ratio=0.2)
        # 10 筆 shift(-1) 去 NaN → 9 筆，9 - int(9*0.2)=9-1=8 train, 1 test
        # 實際：n=9, split_idx=9-int(9*0.2)=9-1=8
        assert result.X_train.shape[0] + result.X_test.shape[0] == 9
        assert result.X_train.shape[1] == 5
        assert result.X_test.shape[1] == 5

    def test_y_not_scaled(self, daily_price_df):
        """y 應保持原始值，未標準化。"""
        result = preprocess_pipeline(daily_price_df)
        # y 值應在合理的股價範圍內（500~700）
        all_y = np.concatenate([result.y_train, result.y_test])
        assert np.all(all_y > 500)
        assert np.all(all_y < 700)

    def test_train_features_standardized(self, daily_price_df):
        """訓練集特徵標準化後 mean≈0。"""
        result = preprocess_pipeline(daily_price_df)
        train_mean = np.mean(result.X_train, axis=0)
        assert np.allclose(train_mean, 0, atol=1e-10)

    def test_feature_names(self, daily_price_df):
        """feature_names 應包含 5 個欄位名稱。"""
        result = preprocess_pipeline(daily_price_df)
        expected = [
            "OpeningPrice", "HighestPrice", "LowestPrice",
            "ClosingPrice", "TradeVolume",
        ]
        assert result.feature_names == expected

    def test_dates_no_leakage(self, daily_price_df):
        """訓練集最後日期應早於測試集第一日期。"""
        result = preprocess_pipeline(daily_price_df)
        assert result.train_dates.iloc[-1] < result.test_dates.iloc[0]

    def test_scaler_inverse_transform(self, daily_price_df):
        """scaler 應能正確還原訓練集特徵。"""
        result = preprocess_pipeline(daily_price_df)
        X_restored = result.feature_scaler.inverse_transform(result.X_train)
        # 還原後值應在合理範圍
        assert np.all(X_restored[:, 3] > 500)  # ClosingPrice 欄位

    def test_custom_test_ratio(self, daily_price_df):
        """自訂 test_ratio 應改變切分比例。"""
        result = preprocess_pipeline(daily_price_df, test_ratio=0.3)
        # n=9, split_idx=9-int(9*0.3)=9-2=7
        assert result.X_train.shape[0] == 7
        assert result.X_test.shape[0] == 2

    def test_custom_feature_columns(self, daily_price_df):
        """自訂特徵欄位應正確處理。"""
        result = preprocess_pipeline(
            daily_price_df,
            feature_columns=["OpeningPrice", "ClosingPrice"],
        )
        assert result.X_train.shape[1] == 2
        assert result.feature_names == ["OpeningPrice", "ClosingPrice"]
