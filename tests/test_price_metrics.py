"""價格距離指標單元測試。"""

import numpy as np
import pytest

from src.metrics.price_metrics import mae, mape, rmse


class TestMAE:
    """MAE 函式測試。"""

    def test_basic(self):
        """基本正確性驗證。"""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 280.0])
        # (10 + 10 + 20) / 3 = 13.333...
        assert mae(y_true, y_pred) == pytest.approx(13.3333, rel=1e-3)

    def test_perfect_prediction(self):
        """完美預測時誤差應為 0。"""
        y_true = np.array([50.0, 100.0, 150.0])
        assert mae(y_true, y_true) == pytest.approx(0.0)

    def test_single_element(self):
        """長度為 1 的陣列。"""
        y_true = np.array([100.0])
        y_pred = np.array([105.0])
        assert mae(y_true, y_pred) == pytest.approx(5.0)


class TestRMSE:
    """RMSE 函式測試。"""

    def test_basic(self):
        """基本正確性驗證。"""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 280.0])
        # sqrt((100 + 100 + 400) / 3) = sqrt(200) ≈ 14.142
        assert rmse(y_true, y_pred) == pytest.approx(14.1421, rel=1e-3)

    def test_perfect_prediction(self):
        """完美預測時誤差應為 0。"""
        y_true = np.array([50.0, 100.0, 150.0])
        assert rmse(y_true, y_true) == pytest.approx(0.0)

    def test_single_element(self):
        """長度為 1 的陣列。"""
        y_true = np.array([100.0])
        y_pred = np.array([105.0])
        assert rmse(y_true, y_pred) == pytest.approx(5.0)


class TestMAPE:
    """MAPE 函式測試。"""

    def test_basic(self):
        """基本正確性驗證。"""
        y_true = np.array([100.0, 200.0, 50.0])
        y_pred = np.array([110.0, 190.0, 45.0])
        # (10/100 + 10/200 + 5/50) / 3 * 100 = (0.1 + 0.05 + 0.1) / 3 * 100 ≈ 8.333
        assert mape(y_true, y_pred) == pytest.approx(8.3333, rel=1e-3)

    def test_perfect_prediction(self):
        """完美預測時誤差應為 0。"""
        y_true = np.array([50.0, 100.0, 150.0])
        assert mape(y_true, y_true) == pytest.approx(0.0)

    def test_zero_in_y_true(self):
        """y_true 包含 0 時應排除該筆資料。"""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 190.0])
        # 排除第一筆，(10/100 + 10/200) / 2 * 100 = 7.5
        assert mape(y_true, y_pred) == pytest.approx(7.5, rel=1e-3)

    def test_all_zero_in_y_true(self):
        """y_true 全為 0 時應回傳 0.0。"""
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([10.0, 20.0])
        assert mape(y_true, y_pred) == pytest.approx(0.0)

    def test_single_element(self):
        """長度為 1 的陣列。"""
        y_true = np.array([200.0])
        y_pred = np.array([210.0])
        # 10/200 * 100 = 5.0
        assert mape(y_true, y_pred) == pytest.approx(5.0)
