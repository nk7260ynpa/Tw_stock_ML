"""資料庫連線管理模組測試。"""

import os
from unittest import mock

from src.database.connection import get_engine


class TestGetEngine:
    """測試 get_engine 函式。"""

    def test_default_values(self):
        """測試使用預設值建立引擎。"""
        with mock.patch.dict(os.environ, {}, clear=True):
            engine = get_engine()

        url_str = str(engine.url)
        assert "tw_stock_database" in url_str
        assert "root" in url_str
        assert "TWSE" in url_str

    def test_env_override(self):
        """測試環境變數覆蓋預設值。"""
        env = {
            "DB_HOST": "env_host",
            "DB_USER": "env_user",
            "DB_PASSWORD": "env_pass",
            "DB_NAME": "env_db",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            engine = get_engine()

        url_str = str(engine.url)
        assert "env_host" in url_str
        assert "env_user" in url_str
        assert "env_db" in url_str

    def test_param_override(self):
        """測試函式參數覆蓋環境變數與預設值。"""
        env = {
            "DB_HOST": "env_host",
            "DB_USER": "env_user",
            "DB_PASSWORD": "env_pass",
            "DB_NAME": "env_db",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            engine = get_engine(
                host="param_host",
                user="param_user",
                password="param_pass",
                db_name="param_db",
            )

        url_str = str(engine.url)
        assert "param_host" in url_str
        assert "param_user" in url_str
        assert "param_db" in url_str

    def test_engine_has_pool_pre_ping(self):
        """測試引擎啟用 pool_pre_ping。"""
        engine = get_engine()
        assert engine.pool._pre_ping is True
