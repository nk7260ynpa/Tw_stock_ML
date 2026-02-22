"""技術指標計算模組。

提供常用技術指標計算功能，包含移動平均、RSI、MACD、布林帶、ATR 等。
"""

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def add_return_rate(df: pd.DataFrame, column: str = "ClosingPrice") -> pd.DataFrame:
    """新增日報酬率欄位。

    Args:
        df: 包含價格欄位的 DataFrame。
        column: 計算報酬率的欄位名稱。

    Returns:
        新增 Return 欄位的 DataFrame。
    """
    df = df.copy()
    df["Return"] = df[column].pct_change()
    return df


def add_sma(
    df: pd.DataFrame,
    column: str = "ClosingPrice",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """新增簡單移動平均欄位。

    Args:
        df: 包含價格欄位的 DataFrame。
        column: 計算 SMA 的欄位名稱。
        windows: 移動平均視窗大小清單，預設 [5, 10, 20]。

    Returns:
        新增 SMA 欄位的 DataFrame。
    """
    df = df.copy()
    windows = windows or [5, 10, 20]
    for w in windows:
        df[f"SMA_{w}"] = df[column].rolling(window=w).mean()
    return df


def add_ema(
    df: pd.DataFrame,
    column: str = "ClosingPrice",
    spans: list[int] | None = None,
) -> pd.DataFrame:
    """新增指數移動平均欄位。

    Args:
        df: 包含價格欄位的 DataFrame。
        column: 計算 EMA 的欄位名稱。
        spans: EMA 週期清單，預設 [12, 26]。

    Returns:
        新增 EMA 欄位的 DataFrame。
    """
    df = df.copy()
    spans = spans or [12, 26]
    for s in spans:
        df[f"EMA_{s}"] = df[column].ewm(span=s, adjust=False).mean()
    return df


def add_rsi(
    df: pd.DataFrame,
    column: str = "ClosingPrice",
    period: int = 14,
) -> pd.DataFrame:
    """新增相對強弱指標（RSI）欄位。

    Args:
        df: 包含價格欄位的 DataFrame。
        column: 計算 RSI 的欄位名稱。
        period: RSI 週期，預設 14。

    Returns:
        新增 RSI 欄位的 DataFrame。
    """
    df = df.copy()
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(
    df: pd.DataFrame,
    column: str = "ClosingPrice",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """新增 MACD 指標欄位。

    Args:
        df: 包含價格欄位的 DataFrame。
        column: 計算 MACD 的欄位名稱。
        fast: 快線 EMA 週期，預設 12。
        slow: 慢線 EMA 週期，預設 26。
        signal: 訊號線 EMA 週期，預設 9。

    Returns:
        新增 MACD、MACD_Signal、MACD_Hist 欄位的 DataFrame。
    """
    df = df.copy()
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    column: str = "ClosingPrice",
    window: int = 20,
    num_std: int = 2,
) -> pd.DataFrame:
    """新增布林帶欄位。

    Args:
        df: 包含價格欄位的 DataFrame。
        column: 計算布林帶的欄位名稱。
        window: 移動平均視窗大小，預設 20。
        num_std: 標準差倍數，預設 2。

    Returns:
        新增 BB_Upper、BB_Lower、BB_PctB 欄位的 DataFrame。
    """
    df = df.copy()
    sma = df[column].rolling(window=window).mean()
    std = df[column].rolling(window=window).std()

    df["BB_Upper"] = sma + num_std * std
    df["BB_Lower"] = sma - num_std * std

    band_width = df["BB_Upper"] - df["BB_Lower"]
    df["BB_PctB"] = (df[column] - df["BB_Lower"]) / band_width
    return df


def add_atr(
    df: pd.DataFrame,
    high_column: str = "HighestPrice",
    low_column: str = "LowestPrice",
    close_column: str = "ClosingPrice",
    period: int = 14,
) -> pd.DataFrame:
    """新增平均真實範圍（ATR）欄位。

    Args:
        df: 包含高低收價欄位的 DataFrame。
        high_column: 最高價欄位名稱。
        low_column: 最低價欄位名稱。
        close_column: 收盤價欄位名稱。
        period: ATR 週期，預設 14。

    Returns:
        新增 ATR 欄位的 DataFrame。
    """
    df = df.copy()
    high = df[high_column]
    low = df[low_column]
    prev_close = df[close_column].shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"ATR_{period}"] = true_range.rolling(window=period).mean()
    return df


def add_volume_ma(
    df: pd.DataFrame,
    volume_column: str = "TradeVolume",
    window: int = 5,
) -> pd.DataFrame:
    """新增成交量移動平均欄位。

    Args:
        df: 包含成交量欄位的 DataFrame。
        volume_column: 成交量欄位名稱。
        window: 移動平均視窗大小，預設 5。

    Returns:
        新增 Volume_MA 欄位的 DataFrame。
    """
    df = df.copy()
    df[f"Volume_MA_{window}"] = df[volume_column].rolling(window=window).mean()
    return df


def compute_all_indicators(
    df: pd.DataFrame,
    drop_warmup_rows: bool = True,
) -> pd.DataFrame:
    """一站式計算所有技術指標。

    依序計算日報酬率、SMA、EMA、RSI、MACD、布林帶、ATR、成交量 MA，
    並可選擇移除暖身期的 NaN 列。

    Args:
        df: 每日行情 DataFrame，需包含 ClosingPrice、HighestPrice、
            LowestPrice、TradeVolume 等欄位。
        drop_warmup_rows: 是否移除暖身期 NaN 列，預設 True。

    Returns:
        包含所有技術指標的 DataFrame。
    """
    result = df.copy()
    result = add_return_rate(result)
    result = add_sma(result)
    result = add_ema(result)
    result = add_rsi(result)
    result = add_macd(result)
    result = add_bollinger_bands(result)
    result = add_atr(result)
    result = add_volume_ma(result)

    if drop_warmup_rows:
        before = len(result)
        result = result.dropna().reset_index(drop=True)
        dropped = before - len(result)
        logger.info("技術指標計算完成，移除 %d 筆暖身期資料，剩餘 %d 筆", dropped, len(result))
    else:
        logger.info("技術指標計算完成（保留暖身期），共 %d 筆", len(result))

    return result
