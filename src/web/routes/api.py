"""API 路由模組。

提供股票搜尋、日線資料查詢與 ML 預測的 RESTful API 端點。
"""

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request

from src.database.connection import get_engine
from src.database.stock_repository import (
    get_all_security_codes,
    get_daily_prices,
    get_stock_info,
)
from src.model.xgboost_model import (
    detect_device,
    evaluate_return_model,
    get_small_data_params,
    predict,
    save_model,
    train_xgboost,
)
from src.preprocessing.pipeline import preprocess_indicator_pipeline
from src.preprocessing.technical_indicators import compute_all_indicators
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

MODEL_DIR = Path("model")


@api_bp.route("/stocks/search")
def search_stocks():
    """搜尋股票代碼或名稱。

    Query Params:
        q: 搜尋關鍵字（股票代碼或名稱的部分匹配）。

    Returns:
        符合條件的股票清單 JSON 陣列。
    """
    keyword = request.args.get("q", "").strip()
    if not keyword:
        return jsonify([])

    try:
        engine = get_engine()
        df = get_all_security_codes(engine)

        mask = (
            df["SecurityCode"].str.contains(keyword, case=False, na=False)
            | df["StockName"].str.contains(keyword, case=False, na=False)
        )
        matched = df[mask].head(20)

        results = [
            {
                "code": row["SecurityCode"],
                "name": row["StockName"],
            }
            for _, row in matched.iterrows()
        ]
    except Exception:
        logger.exception("股票搜尋失敗")
        return jsonify({"error": "資料庫查詢失敗"}), 500

    return jsonify(results)


@api_bp.route("/stocks/<code>/daily")
def get_stock_daily(code: str):
    """取得指定股票的日線資料。

    Path Params:
        code: 股票代碼。

    Query Params:
        start: 起始日期（YYYY-MM-DD），選填。
        end: 結束日期（YYYY-MM-DD），選填。

    Returns:
        日線資料 JSON 陣列，每筆含 date、open、high、low、close、volume。
    """
    start_str = request.args.get("start", "")
    end_str = request.args.get("end", "")

    start_date = None
    end_date = None

    try:
        if start_str:
            start_date = date.fromisoformat(start_str)
        if end_str:
            end_date = date.fromisoformat(end_str)
    except ValueError:
        return jsonify({"error": "日期格式錯誤，請使用 YYYY-MM-DD"}), 400

    try:
        engine = get_engine()
        df = get_daily_prices(engine, code, start_date, end_date)

        if df.empty:
            return jsonify([])

        data = []
        for _, row in df.iterrows():
            date_val = row.get("Date", "")
            if isinstance(date_val, (datetime, date)):
                date_val = date_val.strftime("%Y-%m-%d")
            else:
                date_val = str(date_val)

            data.append({
                "date": date_val,
                "open": _safe_float(row.get("OpeningPrice")),
                "high": _safe_float(row.get("HighestPrice")),
                "low": _safe_float(row.get("LowestPrice")),
                "close": _safe_float(row.get("ClosingPrice")),
                "volume": _safe_float(row.get("TradeVolume")),
            })
    except Exception:
        logger.exception("股價查詢失敗：%s", code)
        return jsonify({"error": "資料庫查詢失敗"}), 500

    return jsonify(data)


@api_bp.route("/stocks/<code>/info")
def get_stock_detail(code: str):
    """取得股票基本資訊。

    Path Params:
        code: 股票代碼。

    Returns:
        股票基本資訊 JSON。
    """
    try:
        engine = get_engine()
        df = get_stock_info(engine, code)

        if df.empty:
            return jsonify({"error": "找不到該股票資訊"}), 404

        row = df.iloc[0]
        result = {
            "code": str(row.get("SecurityCode", "")),
            "name": str(row.get("StockName", "")),
            "industry": str(row.get("Industry", "")),
            "company_name": str(row.get("CompanyName", "")),
        }
    except Exception:
        logger.exception("股票資訊查詢失敗：%s", code)
        return jsonify({"error": "資料庫查詢失敗"}), 500

    return jsonify(result)


@api_bp.route("/stocks/<code>/indicators")
def get_stock_indicators(code: str):
    """取得股票技術指標資料（用於圖表疊加）。

    Path Params:
        code: 股票代碼。

    Query Params:
        start: 起始日期（YYYY-MM-DD），選填。
        end: 結束日期（YYYY-MM-DD），選填。

    Returns:
        含技術指標的日線資料 JSON。
    """
    start_str = request.args.get("start", "")
    end_str = request.args.get("end", "")

    start_date = None
    end_date = None

    try:
        if start_str:
            start_date = date.fromisoformat(start_str)
        if end_str:
            end_date = date.fromisoformat(end_str)
    except ValueError:
        return jsonify({"error": "日期格式錯誤，請使用 YYYY-MM-DD"}), 400

    try:
        engine = get_engine()
        df = get_daily_prices(engine, code, start_date, end_date)

        if df.empty:
            return jsonify({"error": "查無資料"}), 404

        df_ind = compute_all_indicators(df, drop_warmup_rows=False)

        indicator_series = _build_indicator_series(df_ind)
    except Exception:
        logger.exception("技術指標計算失敗：%s", code)
        return jsonify({"error": "技術指標計算失敗"}), 500

    return jsonify(indicator_series)


@api_bp.route("/predict", methods=["POST"])
def run_predict():
    """執行 ML 預測。

    Request Body (JSON):
        stock_code: 股票代碼（必填）。

    Returns:
        JSON 含：
        - predictions: 預測結果（日期、實際值、預測值）。
        - metrics: 評估指標（MAE、RMSE、MAPE、方向正確率）。
        - feature_importance: 特徵重要度排名。
    """
    data = request.get_json()
    if not data or "stock_code" not in data:
        return jsonify({"error": "需要 stock_code 欄位"}), 400

    stock_code = data["stock_code"]
    logger.info("開始 ML 預測：%s", stock_code)

    try:
        # 取得資料
        engine = get_engine()
        df = get_daily_prices(engine, stock_code)

        if df.empty:
            return jsonify({"error": f"查無 {stock_code} 的股價資料"}), 404

        if len(df) < 50:
            return jsonify({
                "error": f"{stock_code} 資料不足（{len(df)} 筆），至少需要 50 筆",
            }), 400

        # 前處理
        pipeline_data = preprocess_indicator_pipeline(
            df, target_type="return",
        )

        # 訓練模型
        device = detect_device()
        params = get_small_data_params(device)
        model = train_xgboost(
            pipeline_data.X_train,
            pipeline_data.y_train,
            pipeline_data.X_test,
            pipeline_data.y_test,
            params=params,
        )

        # 儲存模型
        model_path = MODEL_DIR / f"xgboost_{stock_code}.json"
        save_model(model, model_path)

        # 評估
        eval_results = evaluate_return_model(
            model,
            pipeline_data.X_test,
            pipeline_data.y_test,
            pipeline_data.base_prices.values,
        )

        # 預測值
        y_pred_return = predict(model, pipeline_data.X_test)
        base_prices = pipeline_data.base_prices.values
        actual_prices = base_prices * (1 + pipeline_data.y_test)
        predicted_prices = base_prices * (1 + y_pred_return)

        # 組裝預測結果
        test_dates = pipeline_data.test_dates
        predictions = []
        for i in range(len(test_dates)):
            date_val = test_dates.iloc[i]
            if isinstance(date_val, (datetime, date)):
                date_str = date_val.strftime("%Y-%m-%d")
            else:
                date_str = str(date_val)

            predictions.append({
                "date": date_str,
                "actual": round(float(actual_prices[i]), 2),
                "predicted": round(float(predicted_prices[i]), 2),
            })

        # 特徵重要度
        feature_importance = _get_feature_importance(
            model, pipeline_data.feature_names,
        )

        # 評估指標
        metrics = {
            "price_MAE": round(eval_results["price_MAE"], 2),
            "price_RMSE": round(eval_results["price_RMSE"], 2),
            "price_MAPE": round(eval_results["price_MAPE"], 2),
            "directional_accuracy": round(
                eval_results["directional_accuracy"] * 100, 2,
            ),
            "return_MAE": round(eval_results["return_MAE"], 6),
            "return_RMSE": round(eval_results["return_RMSE"], 6),
        }

        result = {
            "stock_code": stock_code,
            "train_samples": int(pipeline_data.X_train.shape[0]),
            "test_samples": int(pipeline_data.X_test.shape[0]),
            "n_features": int(pipeline_data.X_train.shape[1]),
            "predictions": predictions,
            "metrics": metrics,
            "feature_importance": feature_importance,
        }

        logger.info("ML 預測完成：%s", stock_code)
        return jsonify(result)

    except Exception:
        logger.exception("ML 預測失敗：%s", stock_code)
        return jsonify({"error": f"ML 預測失敗：{stock_code}"}), 500


def _safe_float(value) -> float | None:
    """安全轉換為 float，處理 NaN 與 None。

    Args:
        value: 要轉換的值。

    Returns:
        float 值，若無法轉換則回傳 None。
    """
    if value is None:
        return None
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _build_indicator_series(df: pd.DataFrame) -> dict:
    """從含技術指標的 DataFrame 建構前端需要的指標序列。

    Args:
        df: 含技術指標欄位的 DataFrame。

    Returns:
        指標序列字典，key 為指標名稱，value 為陣列。
    """
    indicator_map = {
        "SMA_5": "MA5",
        "SMA_10": "MA10",
        "SMA_20": "MA20",
        "EMA_12": "EMA12",
        "EMA_26": "EMA26",
        "RSI_14": "RSI14",
        "MACD": "DIF",
        "MACD_Signal": "MACD",
        "MACD_Hist": "OSC",
        "BB_Upper": "BB_Upper",
        "BB_Lower": "BB_Lower",
    }

    series = {}
    for col, label in indicator_map.items():
        if col in df.columns:
            values = df[col].tolist()
            series[label] = [
                None if (v is None or (isinstance(v, float) and np.isnan(v)))
                else round(v, 4)
                for v in values
            ]

    return series


def _get_feature_importance(model, feature_names: list[str]) -> list[dict]:
    """取得 XGBoost 模型的特徵重要度排名。

    Args:
        model: 訓練完成的 XGBRegressor 模型。
        feature_names: 特徵名稱清單。

    Returns:
        特徵重要度清單（由高到低排序），每項含 name 與 importance。
    """
    importance = model.feature_importances_
    paired = list(zip(feature_names, importance))
    paired.sort(key=lambda x: x[1], reverse=True)

    result = []
    for name, imp in paired[:20]:
        result.append({
            "name": name,
            "importance": round(float(imp), 6),
        })

    return result
