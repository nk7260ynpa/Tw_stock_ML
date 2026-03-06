"""Web 模組單元測試。

測試 Flask app 工廠、頁面路由與 API 端點。
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.web import create_app


@pytest.fixture
def app():
    """建立測試用 Flask app。"""
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """建立測試用 Flask client。"""
    return app.test_client()


class TestAppFactory:
    """Flask App Factory 測試。"""

    def test_create_app_returns_flask_instance(self, app):
        """應回傳 Flask 實例。"""
        from flask import Flask
        assert isinstance(app, Flask)

    def test_app_has_blueprints(self, app):
        """應已註冊 dashboard 和 api 藍圖。"""
        assert "dashboard" in app.blueprints
        assert "api" in app.blueprints


class TestDashboardRoute:
    """儀表板頁面路由測試。"""

    def test_index_returns_200(self, client):
        """首頁應回傳 200。"""
        response = client.get("/")
        assert response.status_code == 200

    def test_index_contains_title(self, client):
        """首頁應包含頁面標題。"""
        response = client.get("/")
        assert "台股 ML 預測分析" in response.data.decode("utf-8")

    def test_index_contains_search_input(self, client):
        """首頁應包含股票搜尋輸入框。"""
        response = client.get("/")
        assert "stock-search" in response.data.decode("utf-8")

    def test_index_contains_chart_section(self, client):
        """首頁應包含圖表區塊。"""
        response = client.get("/")
        assert "main-chart" in response.data.decode("utf-8")

    def test_index_contains_predict_button(self, client):
        """首頁應包含預測按鈕。"""
        response = client.get("/")
        assert "btn-predict" in response.data.decode("utf-8")


class TestSearchAPI:
    """股票搜尋 API 測試。"""

    def test_search_empty_keyword_returns_empty_list(self, client):
        """空關鍵字應回傳空陣列。"""
        response = client.get("/api/stocks/search")
        assert response.status_code == 200
        assert response.get_json() == []

    def test_search_empty_q_returns_empty_list(self, client):
        """空 q 參數應回傳空陣列。"""
        response = client.get("/api/stocks/search?q=")
        assert response.status_code == 200
        assert response.get_json() == []

    @patch("src.web.routes.api.get_engine")
    @patch("src.web.routes.api.get_all_security_codes")
    def test_search_returns_results(self, mock_get_codes, mock_engine, client):
        """搜尋應回傳匹配的股票。"""
        mock_engine.return_value = MagicMock()
        mock_get_codes.return_value = pd.DataFrame({
            "SecurityCode": ["2330", "2331", "1234"],
            "StockName": ["台積電", "精英", "大明"],
        })

        response = client.get("/api/stocks/search?q=233")
        assert response.status_code == 200

        results = response.get_json()
        assert len(results) == 2
        assert results[0]["code"] == "2330"
        assert results[1]["code"] == "2331"

    @patch("src.web.routes.api.get_engine")
    @patch("src.web.routes.api.get_all_security_codes")
    def test_search_by_name(self, mock_get_codes, mock_engine, client):
        """應支援用名稱搜尋。"""
        mock_engine.return_value = MagicMock()
        mock_get_codes.return_value = pd.DataFrame({
            "SecurityCode": ["2330", "2317"],
            "StockName": ["台積電", "鴻海"],
        })

        response = client.get("/api/stocks/search?q=台積")
        assert response.status_code == 200

        results = response.get_json()
        assert len(results) == 1
        assert results[0]["code"] == "2330"
        assert results[0]["name"] == "台積電"


class TestDailyAPI:
    """日線資料 API 測試。"""

    @patch("src.web.routes.api.get_engine")
    @patch("src.web.routes.api.get_daily_prices")
    def test_daily_returns_empty_for_no_data(
        self, mock_get_prices, mock_engine, client,
    ):
        """查無資料時應回傳空陣列。"""
        mock_engine.return_value = MagicMock()
        mock_get_prices.return_value = pd.DataFrame()

        response = client.get("/api/stocks/2330/daily")
        assert response.status_code == 200
        assert response.get_json() == []

    @patch("src.web.routes.api.get_engine")
    @patch("src.web.routes.api.get_daily_prices")
    def test_daily_returns_data(self, mock_get_prices, mock_engine, client):
        """應回傳日線資料。"""
        mock_engine.return_value = MagicMock()
        mock_get_prices.return_value = pd.DataFrame({
            "Date": ["2024-01-02"],
            "OpeningPrice": [100.0],
            "HighestPrice": [105.0],
            "LowestPrice": [99.0],
            "ClosingPrice": [103.0],
            "TradeVolume": [50000.0],
        })

        response = client.get("/api/stocks/2330/daily?start=2024-01-01&end=2024-01-31")
        assert response.status_code == 200

        data = response.get_json()
        assert len(data) == 1
        assert data[0]["date"] == "2024-01-02"
        assert data[0]["open"] == 100.0
        assert data[0]["close"] == 103.0

    def test_daily_invalid_date_returns_400(self, client):
        """無效日期格式應回傳 400。"""
        response = client.get("/api/stocks/2330/daily?start=invalid")
        assert response.status_code == 400


class TestPredictAPI:
    """ML 預測 API 測試。"""

    def test_predict_missing_body_returns_400(self, client):
        """缺少請求內容應回傳 400。"""
        response = client.post(
            "/api/predict",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_predict_missing_stock_code_returns_400(self, client):
        """缺少 stock_code 應回傳 400。"""
        response = client.post(
            "/api/predict",
            json={},
        )
        assert response.status_code == 400

    @patch("src.web.routes.api.get_engine")
    @patch("src.web.routes.api.get_daily_prices")
    def test_predict_empty_data_returns_404(
        self, mock_get_prices, mock_engine, client,
    ):
        """查無資料時應回傳 404。"""
        mock_engine.return_value = MagicMock()
        mock_get_prices.return_value = pd.DataFrame()

        response = client.post(
            "/api/predict",
            json={"stock_code": "9999"},
        )
        assert response.status_code == 404

    @patch("src.web.routes.api.get_engine")
    @patch("src.web.routes.api.get_daily_prices")
    def test_predict_insufficient_data_returns_400(
        self, mock_get_prices, mock_engine, client,
    ):
        """資料不足時應回傳 400。"""
        mock_engine.return_value = MagicMock()
        mock_get_prices.return_value = pd.DataFrame({
            "Date": ["2024-01-02"],
            "OpeningPrice": [100.0],
            "HighestPrice": [105.0],
            "LowestPrice": [99.0],
            "ClosingPrice": [103.0],
            "TradeVolume": [50000.0],
        })

        response = client.post(
            "/api/predict",
            json={"stock_code": "2330"},
        )
        assert response.status_code == 400


class TestHelperFunctions:
    """輔助函式測試。"""

    def test_safe_float_with_none(self):
        """None 應回傳 None。"""
        from src.web.routes.api import _safe_float
        assert _safe_float(None) is None

    def test_safe_float_with_nan(self):
        """NaN 應回傳 None。"""
        import numpy as np
        from src.web.routes.api import _safe_float
        assert _safe_float(float("nan")) is None
        assert _safe_float(np.nan) is None

    def test_safe_float_with_number(self):
        """正常數值應回傳 float。"""
        from src.web.routes.api import _safe_float
        assert _safe_float(42) == 42.0
        assert _safe_float(3.14) == 3.14

    def test_safe_float_with_string_number(self):
        """數值字串應回傳 float。"""
        from src.web.routes.api import _safe_float
        assert _safe_float("123.45") == 123.45

    def test_safe_float_with_invalid_string(self):
        """非數值字串應回傳 None。"""
        from src.web.routes.api import _safe_float
        assert _safe_float("abc") is None

    def test_get_feature_importance(self):
        """應回傳排序後的特徵重要度清單。"""
        from src.web.routes.api import _get_feature_importance

        mock_model = MagicMock()
        mock_model.feature_importances_ = [0.1, 0.5, 0.3, 0.1]
        names = ["feat_a", "feat_b", "feat_c", "feat_d"]

        result = _get_feature_importance(mock_model, names)
        assert len(result) == 4
        assert result[0]["name"] == "feat_b"
        assert result[0]["importance"] == 0.5
        assert result[1]["name"] == "feat_c"
