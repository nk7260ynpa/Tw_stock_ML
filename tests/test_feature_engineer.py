"""特徵工程模組單元測試。"""

import pandas as pd
import pytest

from src.preprocessing.feature_engineer import (
    build_feature_target,
    create_target,
    select_features,
)


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


class TestSelectFeatures:
    """select_features 函式測試。"""

    def test_default_columns(self, daily_price_df):
        """預設選取 5 個核心價量欄位。"""
        result = select_features(daily_price_df)
        expected_columns = [
            "OpeningPrice", "HighestPrice", "LowestPrice",
            "ClosingPrice", "TradeVolume",
        ]
        assert list(result.columns) == expected_columns
        assert len(result) == 10

    def test_custom_columns(self, daily_price_df):
        """自訂特徵欄位。"""
        result = select_features(daily_price_df, ["OpeningPrice", "ClosingPrice"])
        assert list(result.columns) == ["OpeningPrice", "ClosingPrice"]

    def test_missing_column_raises(self, daily_price_df):
        """缺少欄位時應拋出 KeyError。"""
        with pytest.raises(KeyError, match="不存在的欄位"):
            select_features(daily_price_df, ["不存在的欄位"])

    def test_returns_copy(self, daily_price_df):
        """回傳的 DataFrame 應為副本，不影響原始資料。"""
        result = select_features(daily_price_df)
        result.iloc[0, 0] = -999.0
        assert daily_price_df["OpeningPrice"].iloc[0] == 580.0


class TestCreateTarget:
    """create_target 函式測試。"""

    def test_shift_minus_one(self, daily_price_df):
        """隔天收盤價應為 shift(-1) 結果。"""
        target = create_target(daily_price_df)
        # 第 0 筆的目標 = 第 1 筆的收盤價 = 588.0
        assert target.iloc[0] == 588.0
        # 第 8 筆的目標 = 第 9 筆的收盤價 = 608.0
        assert target.iloc[8] == 608.0

    def test_last_row_is_nan(self, daily_price_df):
        """最後一列應為 NaN。"""
        target = create_target(daily_price_df)
        assert pd.isna(target.iloc[-1])

    def test_length_matches(self, daily_price_df):
        """長度應與原始 DataFrame 相同。"""
        target = create_target(daily_price_df)
        assert len(target) == len(daily_price_df)

    def test_missing_column_raises(self, daily_price_df):
        """缺少目標欄位時應拋出 KeyError。"""
        with pytest.raises(KeyError, match="不存在的欄位"):
            create_target(daily_price_df, target_column="不存在的欄位")


class TestBuildFeatureTarget:
    """build_feature_target 函式測試。"""

    def test_removes_last_row(self, daily_price_df):
        """應移除最後一列 NaN，回傳 9 筆資料。"""
        features, target, dates = build_feature_target(daily_price_df)
        assert len(features) == 9
        assert len(target) == 9
        assert len(dates) == 9

    def test_target_values(self, daily_price_df):
        """目標值應為隔天收盤價。"""
        _, target, _ = build_feature_target(daily_price_df)
        # 第 0 筆目標 = 第 1 筆收盤價 = 588.0
        assert target.iloc[0] == 588.0
        # 最後一筆目標 = 第 9 筆收盤價 = 608.0
        assert target.iloc[-1] == 608.0

    def test_no_nan_in_target(self, daily_price_df):
        """目標值不應包含 NaN。"""
        _, target, _ = build_feature_target(daily_price_df)
        assert not target.isna().any()

    def test_dates_alignment(self, daily_price_df):
        """日期應與特徵對齊。"""
        features, _, dates = build_feature_target(daily_price_df)
        assert len(dates) == len(features)
        # 第一筆日期應為 2024-01-02
        assert dates.iloc[0] == pd.Timestamp("2024-01-02")

    def test_custom_columns(self, daily_price_df):
        """自訂特徵欄位應正確回傳。"""
        features, _, _ = build_feature_target(
            daily_price_df, feature_columns=["OpeningPrice", "ClosingPrice"],
        )
        assert list(features.columns) == ["OpeningPrice", "ClosingPrice"]
