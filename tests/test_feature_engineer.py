"""特徵工程模組單元測試。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineer import (
    build_feature_target,
    build_feature_target_with_indicators,
    build_feature_target_with_indicators_forward,
    create_target,
    create_target_return_forward,
    select_features,
)
from src.preprocessing.technical_indicators import compute_all_indicators


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


class TestCreateTargetReturnForward:
    """create_target_return_forward 函式測試。"""

    def test_returns_correct_horizon_return(self, daily_price_df):
        """應正確計算前瞻 horizon 天的報酬率。"""
        horizon = 3
        result = create_target_return_forward(daily_price_df, horizon=horizon)
        # 第 0 筆：(Close[3] - Close[0]) / Close[0] = (590 - 583) / 583
        expected = (590.0 - 583.0) / 583.0
        assert abs(result.iloc[0] - expected) < 1e-10

    def test_last_horizon_rows_are_nan(self, daily_price_df):
        """末尾 horizon 筆應為 NaN。"""
        horizon = 3
        result = create_target_return_forward(daily_price_df, horizon=horizon)
        # 最後 3 筆應為 NaN
        assert result.iloc[-3:].isna().all()
        # 前 7 筆不應為 NaN
        assert result.iloc[:-3].notna().all()

    def test_raises_on_missing_column(self, daily_price_df):
        """缺少目標欄位時應拋出 KeyError。"""
        with pytest.raises(KeyError, match="不存在的欄位"):
            create_target_return_forward(
                daily_price_df, target_column="不存在的欄位",
            )


@pytest.fixture
def large_daily_price_df():
    """建立 100 筆模擬每日行情 DataFrame，用於技術指標測試。"""
    n = 100
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


class TestBuildFeatureTargetWithIndicatorsForward:
    """build_feature_target_with_indicators_forward 函式測試。"""

    def test_returns_correct_tuple_types(self, large_daily_price_df):
        """應回傳 (DataFrame, Series, Series) 三元組。"""
        features, target, dates = build_feature_target_with_indicators_forward(
            large_daily_price_df, horizon=5,
        )
        assert isinstance(features, pd.DataFrame)
        assert isinstance(target, pd.Series)
        assert isinstance(dates, pd.Series)

    def test_removes_nan_rows(self, large_daily_price_df):
        """目標值不應包含 NaN。"""
        _, target, _ = build_feature_target_with_indicators_forward(
            large_daily_price_df, horizon=5,
        )
        assert not target.isna().any()

    def test_horizon_parameter(self, large_daily_price_df):
        """不同 horizon 應產生不同長度的資料。"""
        _, target_5, _ = build_feature_target_with_indicators_forward(
            large_daily_price_df, horizon=5,
        )
        _, target_10, _ = build_feature_target_with_indicators_forward(
            large_daily_price_df, horizon=10,
        )
        # horizon 越大，有效資料越少（末尾被截斷更多）
        assert len(target_5) > len(target_10)


@pytest.fixture
def daily_price_df_with_zero():
    """建立含 ClosingPrice=0 的 100 筆模擬行情，模擬 2317 零值問題。"""
    n = 100
    np.random.seed(42)
    base_price = 500.0
    prices = base_price + np.cumsum(np.random.randn(n) * 2)
    prices = np.maximum(prices, 1.0)  # 確保正值
    # 在中間插入一筆全零記錄
    prices[50] = 0.0
    opening = prices + np.random.randn(n)
    opening[50] = 0.0
    highest = prices + abs(np.random.randn(n)) * 3
    highest[50] = 0.0
    lowest = prices - abs(np.random.randn(n)) * 3
    lowest[50] = 0.0
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-02", periods=n, freq="B"),
        "SecurityCode": ["2317"] * n,
        "OpeningPrice": opening,
        "HighestPrice": highest,
        "LowestPrice": lowest,
        "ClosingPrice": prices,
        "TradeVolume": np.random.randint(20000, 50000, n).astype(float),
    })


class TestInfHandling:
    """inf 值處理測試 — 驗證含零值資料不會產生 inf。"""

    def test_compute_all_indicators_no_inf(self, daily_price_df_with_zero):
        """compute_all_indicators 輸出不應包含 inf。"""
        result = compute_all_indicators(
            daily_price_df_with_zero, drop_warmup_rows=True,
        )
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result[numeric_cols].values).any(), (
            "compute_all_indicators 輸出含有 inf 值"
        )

    def test_build_forward_no_inf(self, daily_price_df_with_zero):
        """build_feature_target_with_indicators_forward 不應產生 inf。"""
        features, target, _ = build_feature_target_with_indicators_forward(
            daily_price_df_with_zero, horizon=5,
        )
        assert not np.isinf(target.values).any(), "target 含有 inf 值"
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        assert not np.isinf(features[numeric_cols].values).any(), (
            "features 含有 inf 值"
        )

    def test_build_indicators_return_no_inf(self, daily_price_df_with_zero):
        """build_feature_target_with_indicators (return) 不應產生 inf。"""
        features, target, _ = build_feature_target_with_indicators(
            daily_price_df_with_zero, target_type="return",
        )
        assert not np.isinf(target.values).any(), "target 含有 inf 值"
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        assert not np.isinf(features[numeric_cols].values).any(), (
            "features 含有 inf 值"
        )
