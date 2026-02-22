"""台股機器學習分析主程式。

此模組為專案進入點，負責初始化 logging 並啟動主要流程。
"""

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main() -> None:
    """主程式進入點。"""
    logger.info("台股機器學習分析系統啟動")
    logger.info("系統初始化完成")


if __name__ == "__main__":
    main()
