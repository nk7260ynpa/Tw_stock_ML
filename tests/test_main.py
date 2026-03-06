"""主程式單元測試。"""

from unittest.mock import patch

from src.main import main, run_web


def test_main_default_mode_calls_run_training() -> None:
    """預設模式應呼叫 run_training。"""
    with patch("src.main.run_training") as mock_train, \
         patch("sys.argv", ["main.py"]):
        main()
        mock_train.assert_called_once()


def test_main_web_mode_calls_run_web() -> None:
    """--web 參數應呼叫 run_web。"""
    with patch("src.main.run_web") as mock_web, \
         patch("sys.argv", ["main.py", "--web"]):
        main()
        mock_web.assert_called_once_with(host="0.0.0.0", port=5002)


def test_main_web_mode_with_custom_port() -> None:
    """--web --port 應傳入自訂 port。"""
    with patch("src.main.run_web") as mock_web, \
         patch("sys.argv", ["main.py", "--web", "--port", "8080"]):
        main()
        mock_web.assert_called_once_with(host="0.0.0.0", port=8080)


def test_run_web_creates_flask_app() -> None:
    """run_web 應建立 Flask app 並啟動。"""
    with patch("src.web.create_app") as mock_create:
        mock_app = mock_create.return_value
        run_web(host="127.0.0.1", port=5002)
        mock_create.assert_called_once()
        mock_app.run.assert_called_once_with(
            host="127.0.0.1", port=5002, debug=False,
        )
