"""儀表板頁面路由。

提供 ML Dashboard 首頁的頁面路由。
"""

from flask import Blueprint, render_template

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def index():
    """渲染 ML Dashboard 首頁。"""
    return render_template("dashboard.html")
