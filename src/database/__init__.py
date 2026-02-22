"""資料庫存取模組。

提供資料庫連線管理與股票資料查詢功能。
"""

from src.database.connection import get_engine
from src.database.stock_repository import (
    get_all_security_codes,
    get_daily_prices,
    get_daily_prices_multi,
    get_stock_info,
)

__all__ = [
    "get_engine",
    "get_daily_prices",
    "get_daily_prices_multi",
    "get_stock_info",
    "get_all_security_codes",
]
