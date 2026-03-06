"""Flask 應用程式工廠模組。

提供 ML Dashboard 的 Flask app 建立與設定。
"""

from flask import Flask

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_app() -> Flask:
    """建立並設定 Flask 應用程式。

    Returns:
        已註冊藍圖的 Flask app 實例。
    """
    app = Flask(__name__)

    from src.web.routes.api import api_bp
    from src.web.routes.dashboard import dashboard_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)

    logger.info("Flask 應用程式建立完成")
    return app
