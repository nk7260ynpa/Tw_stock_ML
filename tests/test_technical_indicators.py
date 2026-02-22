"""技術指標計算模組測試。"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.technical_indicators import (
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_return_rate,
    add_rsi,
    add_sma,
    add_volume_ma,
    compute_all_indicators,
)


@pytest.fixture()
def sample_df():
    """建立 100 筆模擬股價資料。"""
    np.random.seed(42)
    n = 100
    # 模擬隨機漫步價格
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


class TestAddReturnRate:
    """日報酬率測試。"""

    def test_return_column_exists(self, sample_df):
        """驗證新增 Return 欄位。"""
        result = add_return_rate(sample_df)
        assert "Return" in result.columns

    def test_first_row_is_nan(self, sample_df):
        """第一筆報酬率為 NaN。"""
        result = add_return_rate(sample_df)
        assert pd.isna(result["Return"].iloc[0])

    def test_return_values_reasonable(self, sample_df):
        """報酬率數值在合理範圍（日報酬率通常 < 20%）。"""
        result = add_return_rate(sample_df)
        valid = result["Return"].dropna()
        assert (valid.abs() < 0.5).all()


class TestAddSma:
    """簡單移動平均測試。"""

    def test_default_sma_columns(self, sample_df):
        """驗證預設 SMA 欄位存在。"""
        result = add_sma(sample_df)
        for w in [5, 10, 20]:
            assert f"SMA_{w}" in result.columns

    def test_custom_windows(self, sample_df):
        """驗證自訂視窗。"""
        result = add_sma(sample_df, windows=[3, 7])
        assert "SMA_3" in result.columns
        assert "SMA_7" in result.columns

    def test_sma_values(self, sample_df):
        """驗證 SMA_5 數值正確。"""
        result = add_sma(sample_df, windows=[5])
        expected = sample_df["ClosingPrice"].rolling(5).mean()
        pd.testing.assert_series_equal(result["SMA_5"], expected, check_names=False)


class TestAddEma:
    """指數移動平均測試。"""

    def test_default_ema_columns(self, sample_df):
        """驗證預設 EMA 欄位存在。"""
        result = add_ema(sample_df)
        assert "EMA_12" in result.columns
        assert "EMA_26" in result.columns


class TestAddRsi:
    """RSI 指標測試。"""

    def test_rsi_column_exists(self, sample_df):
        """驗證 RSI 欄位存在。"""
        result = add_rsi(sample_df)
        assert "RSI_14" in result.columns

    def test_rsi_range(self, sample_df):
        """RSI 數值在 0~100 之間。"""
        result = add_rsi(sample_df)
        valid = result["RSI_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestAddMacd:
    """MACD 指標測試。"""

    def test_macd_columns_exist(self, sample_df):
        """驗證 MACD 欄位存在。"""
        result = add_macd(sample_df)
        assert "MACD" in result.columns
        assert "MACD_Signal" in result.columns
        assert "MACD_Hist" in result.columns

    def test_macd_hist_equals_diff(self, sample_df):
        """驗證 MACD_Hist = MACD - MACD_Signal。"""
        result = add_macd(sample_df)
        expected = result["MACD"] - result["MACD_Signal"]
        pd.testing.assert_series_equal(
            result["MACD_Hist"], expected, check_names=False,
        )


class TestAddBollingerBands:
    """布林帶測試。"""

    def test_bb_columns_exist(self, sample_df):
        """驗證布林帶欄位存在。"""
        result = add_bollinger_bands(sample_df)
        assert "BB_Upper" in result.columns
        assert "BB_Lower" in result.columns
        assert "BB_PctB" in result.columns

    def test_upper_above_lower(self, sample_df):
        """上軌大於下軌。"""
        result = add_bollinger_bands(sample_df)
        valid = result.dropna()
        assert (valid["BB_Upper"] > valid["BB_Lower"]).all()


class TestAddAtr:
    """ATR 指標測試。"""

    def test_atr_column_exists(self, sample_df):
        """驗證 ATR 欄位存在。"""
        result = add_atr(sample_df)
        assert "ATR_14" in result.columns

    def test_atr_positive(self, sample_df):
        """ATR 值為正數。"""
        result = add_atr(sample_df)
        valid = result["ATR_14"].dropna()
        assert (valid > 0).all()


class TestAddVolumeMa:
    """成交量移動平均測試。"""

    def test_volume_ma_column_exists(self, sample_df):
        """驗證 Volume_MA 欄位存在。"""
        result = add_volume_ma(sample_df)
        assert "Volume_MA_5" in result.columns


class TestComputeAllIndicators:
    """一站式計算測試。"""

    def test_no_nan_after_drop(self, sample_df):
        """移除暖身期後無 NaN。"""
        result = compute_all_indicators(sample_df, drop_warmup_rows=True)
        assert result.isna().sum().sum() == 0

    def test_all_indicator_columns_present(self, sample_df):
        """所有技術指標欄位都存在。"""
        result = compute_all_indicators(sample_df)
        expected_cols = [
            "Return", "SMA_5", "SMA_10", "SMA_20",
            "EMA_12", "EMA_26", "RSI_14",
            "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Lower", "BB_PctB",
            "ATR_14", "Volume_MA_5",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少欄位：{col}"

    def test_row_count_after_warmup(self, sample_df):
        """暖身期移除後剩餘資料量合理（100 筆應剩餘 70+ 筆）。"""
        result = compute_all_indicators(sample_df)
        assert len(result) >= 70

    def test_keep_warmup_rows(self, sample_df):
        """不移除暖身期時保留全部資料。"""
        result = compute_all_indicators(sample_df, drop_warmup_rows=False)
        assert len(result) == len(sample_df)

    def test_original_df_not_modified(self, sample_df):
        """原始 DataFrame 不被修改。"""
        original_cols = list(sample_df.columns)
        compute_all_indicators(sample_df)
        assert list(sample_df.columns) == original_cols
