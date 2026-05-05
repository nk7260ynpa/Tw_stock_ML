"""Flask 應用程式工廠模組。

提供 ML Dashboard 的 Flask app 建立與設定。
"""

import os

from flask import Flask

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ScriptNameMiddleware:
    """WSGI 中介層：設定 SCRIPT_NAME 讓 Flask 產生帶 base path 前綴的 URL。

    Dashboard 反向代理將 `/app/ml/*` 請求 strip 前綴後轉發至 Flask，
    Flask 只會收到根路徑的請求（如 `/`、`/api/...`）。但為了讓
    `url_for()` 產生帶前綴的連結，讓瀏覽器的後續請求仍走回代理，
    需要透過 WSGI 的 `SCRIPT_NAME` 環境變數告知 Flask 外部應用程式根路徑。

    Attributes:
        app: 被包裝的 WSGI 應用程式。
        script_name: 要設定的 SCRIPT_NAME（外部應用程式根路徑）。
    """

    def __init__(self, app, script_name: str) -> None:
        """初始化 ScriptNameMiddleware。

        Args:
            app: 要包裝的 WSGI 應用程式。
            script_name: 外部存取的根路徑（如 `/app/ml`），不可以 `/` 結尾。
        """
        self.app = app
        self.script_name = script_name

    def __call__(self, environ, start_response):
        """WSGI 入口：設定 SCRIPT_NAME 後交由下游 app 處理。

        Dashboard 代理已將前綴 strip，故 PATH_INFO 保持不變。

        Args:
            environ: WSGI 環境變數字典。
            start_response: WSGI start_response callable。
        """
        environ["SCRIPT_NAME"] = self.script_name
        return self.app(environ, start_response)


def create_app() -> Flask:
    """建立並設定 Flask 應用程式。

    若環境變數 `SCRIPT_NAME` 有設定，會自動掛載 `ScriptNameMiddleware`
    讓 Flask 產生帶前綴的 URL，支援透過 Dashboard 反向代理存取。

    Returns:
        已註冊藍圖的 Flask app 實例。
    """
    app = Flask(__name__)

    from src.web.routes.api import api_bp
    from src.web.routes.dashboard import dashboard_bp

    app.register_blueprint(dashboard_bp)
    app.register_blueprint(api_bp)

    script_name = os.environ.get("SCRIPT_NAME", "").rstrip("/")
    if script_name:
        app.wsgi_app = ScriptNameMiddleware(app.wsgi_app, script_name)
        logger.info("已啟用 SCRIPT_NAME 中介層：%s", script_name)

    logger.info("Flask 應用程式建立完成")
    return app
