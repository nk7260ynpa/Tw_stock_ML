"""資料庫連線管理模組。

提供資料庫連線引擎的建立與管理功能。
"""

import os

from sqlalchemy import create_engine, Engine

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_HOST = "tw_stock_database"
_DEFAULT_USER = "root"
_DEFAULT_PASSWORD = "stock"
_DEFAULT_DB_NAME = "TWSE"


def get_engine(
    host: str | None = None,
    user: str | None = None,
    password: str | None = None,
    db_name: str | None = None,
) -> Engine:
    """建立並回傳 SQLAlchemy 資料庫引擎。

    參數優先順序：函式參數 > 環境變數 > 預設值。

    Args:
        host: 資料庫主機位址。
        user: 資料庫使用者名稱。
        password: 資料庫密碼。
        db_name: 資料庫名稱。

    Returns:
        SQLAlchemy Engine 實例。
    """
    host = host or os.environ.get("DB_HOST", _DEFAULT_HOST)
    user = user or os.environ.get("DB_USER", _DEFAULT_USER)
    password = password or os.environ.get("DB_PASSWORD", _DEFAULT_PASSWORD)
    db_name = db_name or os.environ.get("DB_NAME", _DEFAULT_DB_NAME)

    url = f"mysql+pymysql://{user}:{password}@{host}/{db_name}"
    engine = create_engine(url, pool_pre_ping=True)

    logger.info("建立資料庫連線引擎：%s@%s/%s", user, host, db_name)

    return engine
