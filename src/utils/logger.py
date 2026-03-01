"""日誌工具模組。

提供統一的 logging 設定，支援同時輸出至終端與檔案。
"""

import logging
import os
from datetime import datetime


def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """設定並回傳 logger 實例。

    Args:
        name: Logger 名稱，通常使用 __name__。
        log_level: 日誌等級，預設為 INFO。

    Returns:
        設定完成的 Logger 實例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # 避免重複新增 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 終端輸出 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 檔案輸出 handler
    # 優先使用環境變數 LOG_DIR，其次使用工作目錄下的 logs/
    log_dir = os.environ.get("LOG_DIR", os.path.join(os.getcwd(), "logs"))
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("%Y%m%d") + ".log"
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_filename),
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
