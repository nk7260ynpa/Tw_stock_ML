"""Logger 工具單元測試。"""

import logging

from src.utils.logger import setup_logger


def test_setup_logger_returns_logger() -> None:
    """測試 setup_logger 回傳 Logger 實例。"""
    logger = setup_logger("test_logger")
    assert isinstance(logger, logging.Logger)


def test_setup_logger_has_handlers() -> None:
    """測試 logger 包含 console 與 file handler。"""
    logger = setup_logger("test_handler_logger")
    handler_types = [type(h) for h in logger.handlers]
    assert logging.StreamHandler in handler_types
    assert logging.FileHandler in handler_types


def test_setup_logger_no_duplicate_handlers() -> None:
    """測試重複呼叫不會新增重複 handler。"""
    logger = setup_logger("test_dup_logger")
    handler_count = len(logger.handlers)
    setup_logger("test_dup_logger")
    assert len(logger.handlers) == handler_count
