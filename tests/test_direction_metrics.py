"""方向正確率指標單元測試。"""

import numpy as np
import pytest

from src.metrics.direction_metrics import directional_accuracy


class TestDirectionalAccuracy:
    """directional_accuracy 函式測試。"""

    def test_basic(self):
        """基本正確性驗證。"""
        # 日期:      D0    D1    D2    D3
        y_true = np.array([100.0, 110.0, 105.0, 115.0])
        y_pred = np.array([100.0, 108.0, 102.0, 112.0])
        # D1: 實際方向 110-100=+10, 預測方向 108-100=+8 → 同正 ✓
        # D2: 實際方向 105-110=-5, 預測方向 102-110=-8 → 同負 ✓
        # D3: 實際方向 115-105=+10, 預測方向 112-105=+7 → 同正 ✓
        # 正確率 = 3/3 = 1.0
        assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_all_wrong(self):
        """全部預測方向錯誤。"""
        y_true = np.array([100.0, 110.0, 105.0])
        y_pred = np.array([100.0, 95.0, 115.0])
        # D1: 實際 +10, 預測 95-100=-5 → 方向不同 ✗
        # D2: 實際 -5, 預測 115-110=+5 → 方向不同 ✗
        # 正確率 = 0/2 = 0.0
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_perfect_prediction(self):
        """完美預測（y_pred == y_true）時正確率應為 1.0。"""
        y_true = np.array([100.0, 110.0, 105.0, 115.0])
        assert directional_accuracy(y_true, y_true) == pytest.approx(1.0)

    def test_single_element(self):
        """長度為 1 的陣列，無法比較應回傳 0.0。"""
        y_true = np.array([100.0])
        y_pred = np.array([105.0])
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_flat_market(self):
        """全平盤（所有值相同）。"""
        y_true = np.array([100.0, 100.0, 100.0])
        y_pred = np.array([100.0, 100.0, 100.0])
        # 實際方向全為 0，預測方向全為 0，sign(0)==sign(0) → 正確
        assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0)

    def test_partial_correct(self):
        """部分正確。"""
        y_true = np.array([100.0, 110.0, 105.0, 115.0])
        y_pred = np.array([100.0, 108.0, 112.0, 112.0])
        # D1: 實際 +10, 預測 108-100=+8 → ✓
        # D2: 實際 -5, 預測 112-110=+2 → ✗
        # D3: 實際 +10, 預測 112-105=+7 → ✓
        # 正確率 = 2/3 ≈ 0.6667
        assert directional_accuracy(y_true, y_pred) == pytest.approx(0.6667, rel=1e-3)
