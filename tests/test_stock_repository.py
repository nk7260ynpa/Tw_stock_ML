"""股票資料查詢模組測試。

使用 SQLite in-memory 資料庫進行測試，不依賴實際 MySQL。
"""

from datetime import date

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.database.stock_repository import (
    get_all_security_codes,
    get_daily_prices,
    get_daily_prices_multi,
    get_stock_info,
)


@pytest.fixture
def engine():
    """建立 SQLite in-memory 測試引擎並預載資料。"""
    eng = create_engine("sqlite:///:memory:")

    daily_price_df = pd.DataFrame([
        {
            "Date": "2024-01-02",
            "SecurityCode": "2330",
            "TradeVolume": 1000,
            "Transaction": 500,
            "TradeValue": 500000,
            "OpeningPrice": 500.0,
            "HighestPrice": 510.0,
            "LowestPrice": 495.0,
            "ClosingPrice": 505.0,
            "Change": 5.0,
            "LastBestBidPrice": 504.0,
            "LastBestBidVolume": 100,
            "LastBestAskPrice": 506.0,
            "LastBestAskVolume": 100,
            "PriceEarningratio": 20.0,
        },
        {
            "Date": "2024-01-03",
            "SecurityCode": "2330",
            "TradeVolume": 1200,
            "Transaction": 600,
            "TradeValue": 612000,
            "OpeningPrice": 505.0,
            "HighestPrice": 515.0,
            "LowestPrice": 500.0,
            "ClosingPrice": 510.0,
            "Change": 5.0,
            "LastBestBidPrice": 509.0,
            "LastBestBidVolume": 120,
            "LastBestAskPrice": 511.0,
            "LastBestAskVolume": 80,
            "PriceEarningratio": 20.5,
        },
        {
            "Date": "2024-01-04",
            "SecurityCode": "2330",
            "TradeVolume": 900,
            "Transaction": 450,
            "TradeValue": 459000,
            "OpeningPrice": 510.0,
            "HighestPrice": 512.0,
            "LowestPrice": 505.0,
            "ClosingPrice": 508.0,
            "Change": -2.0,
            "LastBestBidPrice": 507.0,
            "LastBestBidVolume": 90,
            "LastBestAskPrice": 509.0,
            "LastBestAskVolume": 110,
            "PriceEarningratio": 20.3,
        },
        {
            "Date": "2024-01-02",
            "SecurityCode": "2317",
            "TradeVolume": 2000,
            "Transaction": 800,
            "TradeValue": 200000,
            "OpeningPrice": 100.0,
            "HighestPrice": 102.0,
            "LowestPrice": 99.0,
            "ClosingPrice": 101.0,
            "Change": 1.0,
            "LastBestBidPrice": 100.5,
            "LastBestBidVolume": 200,
            "LastBestAskPrice": 101.5,
            "LastBestAskVolume": 150,
            "PriceEarningratio": 15.0,
        },
    ])
    daily_price_df.to_sql("DailyPrice", eng, index=False)

    stock_name_df = pd.DataFrame([
        {"SecurityCode": "2330", "StockName": "台積電"},
        {"SecurityCode": "2317", "StockName": "鴻海"},
        {"SecurityCode": "9999", "StockName": "測試股"},
    ])
    stock_name_df.to_sql("StockName", eng, index=False)

    company_info_df = pd.DataFrame([
        {
            "SecurityCode": "2330",
            "IndustryCode": "24",
            "CompanyName": "台灣積體電路製造股份有限公司",
            "SpecialShares": 0,
            "NormalShares": 25930380458,
            "PrivateShares": 0,
        },
        {
            "SecurityCode": "2317",
            "IndustryCode": "24",
            "CompanyName": "鴻海精密工業股份有限公司",
            "SpecialShares": 0,
            "NormalShares": 13860064000,
            "PrivateShares": 0,
        },
    ])
    company_info_df.to_sql("CompanyInfo", eng, index=False)

    industry_map_df = pd.DataFrame([
        {"IndustryCode": "24", "Industry": "半導體業"},
    ])
    industry_map_df.to_sql("IndustryMap", eng, index=False)

    return eng


class TestGetDailyPrices:
    """測試 get_daily_prices 函式。"""

    def test_basic_query(self, engine):
        """測試基本查詢。"""
        df = get_daily_prices(engine, "2330")
        assert len(df) == 3
        assert df.iloc[0]["SecurityCode"] == "2330"

    def test_date_filter_start(self, engine):
        """測試起始日期篩選。"""
        df = get_daily_prices(
            engine, "2330", start_date=date(2024, 1, 3),
        )
        assert len(df) == 2
        assert df.iloc[0]["Date"] == "2024-01-03"

    def test_date_filter_end(self, engine):
        """測試結束日期篩選。"""
        df = get_daily_prices(
            engine, "2330", end_date=date(2024, 1, 3),
        )
        assert len(df) == 2
        assert df.iloc[-1]["Date"] == "2024-01-03"

    def test_date_filter_range(self, engine):
        """測試日期範圍篩選。"""
        df = get_daily_prices(
            engine, "2330",
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
        )
        assert len(df) == 2

    def test_no_data(self, engine):
        """測試查無資料。"""
        df = get_daily_prices(engine, "0000")
        assert len(df) == 0

    def test_order_by_date_asc(self, engine):
        """測試結果按日期升冪排序。"""
        df = get_daily_prices(engine, "2330")
        dates = df["Date"].tolist()
        assert dates == sorted(dates)


class TestGetDailyPricesMulti:
    """測試 get_daily_prices_multi 函式。"""

    def test_multi_codes(self, engine):
        """測試多檔股票查詢。"""
        df = get_daily_prices_multi(engine, ["2330", "2317"])
        assert len(df) == 4

    def test_empty_codes(self, engine):
        """測試空清單。"""
        df = get_daily_prices_multi(engine, [])
        assert len(df) == 0

    def test_multi_with_date_filter(self, engine):
        """測試多檔股票搭配日期篩選。"""
        df = get_daily_prices_multi(
            engine, ["2330", "2317"],
            start_date=date(2024, 1, 3),
        )
        assert len(df) == 2
        assert all(d >= "2024-01-03" for d in df["Date"].tolist())

    def test_no_data(self, engine):
        """測試查無資料。"""
        df = get_daily_prices_multi(engine, ["0000"])
        assert len(df) == 0


class TestGetStockInfo:
    """測試 get_stock_info 函式。"""

    def test_basic_query(self, engine):
        """測試基本查詢。"""
        df = get_stock_info(engine, "2330")
        assert len(df) == 1
        assert df.iloc[0]["StockName"] == "台積電"
        assert df.iloc[0]["Industry"] == "半導體業"
        assert df.iloc[0]["CompanyName"] == "台灣積體電路製造股份有限公司"

    def test_no_company_info(self, engine):
        """測試有 StockName 但無 CompanyInfo 的情況。"""
        df = get_stock_info(engine, "9999")
        assert len(df) == 1
        assert df.iloc[0]["StockName"] == "測試股"
        assert df.iloc[0]["CompanyName"] is None

    def test_no_data(self, engine):
        """測試查無資料。"""
        df = get_stock_info(engine, "0000")
        assert len(df) == 0


class TestGetAllSecurityCodes:
    """測試 get_all_security_codes 函式。"""

    def test_basic_query(self, engine):
        """測試基本查詢。"""
        df = get_all_security_codes(engine)
        assert len(df) == 3
        assert "SecurityCode" in df.columns
        assert "StockName" in df.columns

    def test_order_by_code(self, engine):
        """測試結果按代碼升冪排序。"""
        df = get_all_security_codes(engine)
        codes = df["SecurityCode"].tolist()
        assert codes == sorted(codes)
