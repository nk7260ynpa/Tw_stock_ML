"""股票資料查詢模組。

提供從資料庫查詢台股資料的函式，所有函式回傳 pd.DataFrame。
"""

from datetime import date

import pandas as pd
from sqlalchemy import Engine, text

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_daily_prices(
    engine: Engine,
    security_code: str,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """取得單一股票每日股價。

    Args:
        engine: SQLAlchemy Engine 實例。
        security_code: 證券代碼。
        start_date: 起始日期（含），None 表示不限。
        end_date: 結束日期（含），None 表示不限。

    Returns:
        每日股價 DataFrame，按 Date 升冪排序。
    """
    query = "SELECT * FROM DailyPrice WHERE SecurityCode = :code"
    params: dict = {"code": security_code}

    if start_date is not None:
        query += " AND Date >= :start"
        params["start"] = start_date.isoformat()
    if end_date is not None:
        query += " AND Date <= :end"
        params["end"] = end_date.isoformat()

    query += " ORDER BY Date ASC"

    logger.info("查詢股價：%s（%s ~ %s）", security_code, start_date, end_date)

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    return df


def get_daily_prices_multi(
    engine: Engine,
    security_codes: list[str],
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """取得多檔股票每日股價。

    Args:
        engine: SQLAlchemy Engine 實例。
        security_codes: 證券代碼清單。
        start_date: 起始日期（含），None 表示不限。
        end_date: 結束日期（含），None 表示不限。

    Returns:
        每日股價 DataFrame，按 SecurityCode、Date 升冪排序。
    """
    if not security_codes:
        return pd.DataFrame()

    placeholders = ", ".join(
        f":code_{i}" for i in range(len(security_codes))
    )
    query = f"SELECT * FROM DailyPrice WHERE SecurityCode IN ({placeholders})"
    params: dict = {
        f"code_{i}": code for i, code in enumerate(security_codes)
    }

    if start_date is not None:
        query += " AND Date >= :start"
        params["start"] = start_date.isoformat()
    if end_date is not None:
        query += " AND Date <= :end"
        params["end"] = end_date.isoformat()

    query += " ORDER BY SecurityCode ASC, Date ASC"

    logger.info(
        "查詢多檔股價：%s（%s ~ %s）",
        security_codes, start_date, end_date,
    )

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    return df


def get_stock_info(engine: Engine, security_code: str) -> pd.DataFrame:
    """取得股票基本資訊。

    LEFT JOIN StockName、CompanyInfo、IndustryMap 三張表。

    Args:
        engine: SQLAlchemy Engine 實例。
        security_code: 證券代碼。

    Returns:
        股票基本資訊 DataFrame。
    """
    query = """
        SELECT
            sn.SecurityCode,
            sn.StockName,
            ci.IndustryCode,
            im.Industry,
            ci.CompanyName,
            ci.SpecialShares,
            ci.NormalShares,
            ci.PrivateShares
        FROM StockName sn
        LEFT JOIN CompanyInfo ci ON sn.SecurityCode = ci.SecurityCode
        LEFT JOIN IndustryMap im ON ci.IndustryCode = im.IndustryCode
        WHERE sn.SecurityCode = :code
    """
    params: dict = {"code": security_code}

    logger.info("查詢股票資訊：%s", security_code)

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    return df


def get_all_security_codes(engine: Engine) -> pd.DataFrame:
    """取得所有證券代碼與名稱。

    Args:
        engine: SQLAlchemy Engine 實例。

    Returns:
        包含 SecurityCode 與 StockName 的 DataFrame。
    """
    query = "SELECT SecurityCode, StockName FROM StockName ORDER BY SecurityCode ASC"

    logger.info("查詢所有證券代碼")

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    return df
