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
    get_high_dim_params,
    predict,
    save_model,
    train_xgboost,
)
from src.preprocessing.pipeline import (
    ForwardIndicatorData,
    preprocess_forward_pipeline,
)
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


@api_bp.route("/predict", methods=["POST"])
def run_predict():
    """執行 ML 預測。

    Request Body (JSON):
        stock_code: 股票代碼（必填）。

    Returns:
        JSON 含：
        - predictions: 預測結果（日期、實際值、預測值）。
        - metrics: 評估指標（MAE、RMSE、MAPE、高於/低於實際價格統計）。
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

        if len(df) < 200:
            return jsonify({
                "error": f"{stock_code} 資料不足（{len(df)} 筆），至少需要 200 筆",
            }), 400

        # 前處理（前瞻指標滑動視窗管線）
        window_size = 60
        horizon = 20
        pipeline_data = preprocess_forward_pipeline(
            df, window_size=window_size, horizon=horizon,
        )

        # 訓練模型（高維特徵參數）
        device = detect_device()
        params = get_high_dim_params(device)
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
            pipeline_data.base_prices_test.values,
        )

        # 預測值與逆推價格
        y_pred_return = predict(model, pipeline_data.X_test)
        base_prices = pipeline_data.base_prices_test.values
        actual_prices = base_prices * (1 + pipeline_data.y_test)
        predicted_prices = base_prices * (1 + y_pred_return)

        # 組裝預測結果（使用 target_date 與 predict_date）
        test_dates = pipeline_data.test_dates
        target_dates = pipeline_data.target_dates_test
        predictions = []
        for i in range(len(test_dates)):
            date_val = test_dates.iloc[i]
            if isinstance(date_val, (datetime, date)):
                predict_date_str = date_val.strftime("%Y-%m-%d")
            else:
                predict_date_str = str(date_val)

            predictions.append({
                "predict_date": predict_date_str,
                "target_date": target_dates[i],
                "actual": round(float(actual_prices[i]), 2),
                "predicted": round(float(predicted_prices[i]), 2),
            })

        # 未來預測：使用最近 window_size 天資料預測未來第 horizon 天
        future_prediction = _build_future_prediction(
            pipeline_data, model, df, window_size, horizon,
        )

        # 特徵重要度
        feature_importance = _get_feature_importance(
            model, pipeline_data.feature_names,
        )

        # 評估指標
        metrics = {
            "price_MAE": round(eval_results["price_MAE"], 2),
            "price_RMSE": round(eval_results["price_RMSE"], 2),
            "price_MAPE": round(eval_results["price_MAPE"], 2),
            "return_MAE": round(eval_results["return_MAE"], 6),
            "return_RMSE": round(eval_results["return_RMSE"], 6),
            "above_actual_count": eval_results["above_actual_count"],
            "below_actual_count": eval_results["below_actual_count"],
            "above_actual_ratio": round(
                eval_results["above_actual_ratio"] * 100, 2,
            ),
            "below_actual_ratio": round(
                eval_results["below_actual_ratio"] * 100, 2,
            ),
        }

        result = {
            "stock_code": stock_code,
            "train_samples": int(pipeline_data.X_train.shape[0]),
            "test_samples": int(pipeline_data.X_test.shape[0]),
            "n_features": int(pipeline_data.X_train.shape[1]),
            "window_size": window_size,
            "horizon": horizon,
            "predictions": predictions,
            "metrics": metrics,
            "feature_importance": feature_importance,
            "future_prediction": future_prediction,
        }

        logger.info("ML 預測完成：%s", stock_code)
        return jsonify(result)

    except Exception:
        logger.exception("ML 預測失敗：%s", stock_code)
        return jsonify({"error": f"ML 預測失敗：{stock_code}"}), 500


def _build_future_prediction(
    pipeline_data: ForwardIndicatorData,
    model,
    df: pd.DataFrame,
    window_size: int,
    horizon: int,
) -> dict | None:
    """建立未來預測結果。

    使用最近 window_size 天的標準化特徵預測未來第 horizon 天的報酬率，
    再逆推為預測價格。

    Args:
        pipeline_data: 前處理結果。
        model: 訓練完成的 XGBoost 模型。
        df: 原始每日行情 DataFrame。
        window_size: 滑動視窗大小。
        horizon: 前瞻天數。

    Returns:
        含 predict_date、target_date、predicted_price 的字典，失敗回傳 None。
    """
    try:
        from src.preprocessing.feature_engineer import (
            build_feature_target_with_indicators_forward,
        )

        # 取得包含技術指標的完整特徵（不濾除末尾 NaN target）
        from src.preprocessing.technical_indicators import compute_all_indicators
        df_ind = compute_all_indicators(df, drop_warmup_rows=True)

        exclude = {"Date", "SecurityCode"}
        feature_cols = [c for c in df_ind.columns if c not in exclude]
        features_full = df_ind[feature_cols]

        if len(features_full) < window_size:
            return None

        # 取最後 window_size 天的特徵
        last_window = features_full.iloc[-window_size:].values.flatten().reshape(1, -1)

        # 基準價格：最後一天的 ClosingPrice
        base_price = float(df_ind["ClosingPrice"].iloc[-1])

        # 標準化
        last_window_scaled = pipeline_data.feature_scaler.transform(last_window)

        # 預測報酬率
        predicted_return = float(predict(model, last_window_scaled)[0])

        # 逆推價格
        predicted_price = round(base_price * (1 + predicted_return), 2)

        # 預測日期（最後一天）
        last_date = df_ind["Date"].iloc[-1]
        if isinstance(last_date, (datetime, date)):
            predict_date_str = last_date.strftime("%Y-%m-%d")
        else:
            predict_date_str = str(last_date)

        # 目標日期：從原始 df 推算 + horizon 個交易日
        all_dates_sorted = sorted(df["Date"].unique())
        last_date_ts = pd.Timestamp(last_date)
        matching = [
            i for i, d in enumerate(all_dates_sorted)
            if pd.Timestamp(d) == last_date_ts
        ]
        if matching and matching[0] + horizon < len(all_dates_sorted):
            target_date = pd.Timestamp(
                all_dates_sorted[matching[0] + horizon],
            ).strftime("%Y-%m-%d")
        else:
            # 若超出已知資料，估算加 28 個自然日
            from datetime import timedelta
            estimated = last_date_ts + timedelta(days=28)
            target_date = estimated.strftime("%Y-%m-%d")

        return {
            "predict_date": predict_date_str,
            "target_date": target_date,
            "predicted_price": predicted_price,
            "predicted_return": round(predicted_return, 6),
            "base_price": round(base_price, 2),
        }
    except Exception:
        logger.exception("未來預測建立失敗")
        return None


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


_INDICATOR_NAME_ZH = {
    "Return": "日報酬率",
    "SMA_5": "5日均線(SMA5)",
    "SMA_10": "10日均線(SMA10)",
    "SMA_20": "20日均線(SMA20)",
    "EMA_12": "12日指數均線(EMA12)",
    "EMA_26": "26日指數均線(EMA26)",
    "RSI_14": "相對強弱指標(RSI14)",
    "MACD": "MACD(DIF)",
    "MACD_Signal": "MACD訊號線",
    "MACD_Hist": "MACD柱狀體(OSC)",
    "BB_Upper": "布林上軌",
    "BB_Lower": "布林下軌",
    "BB_PctB": "布林%B",
    "ATR_14": "平均真實範圍(ATR14)",
    "Volume_MA_5": "5日成交量均線",
}

_translate_cache: dict[str, str] | None = None


def _load_translate_table() -> dict[str, str]:
    """從 TWSE.Translate 載入英中對照表（帶快取）。"""
    global _translate_cache
    if _translate_cache is not None:
        return _translate_cache

    try:
        engine = get_engine()
        df = pd.read_sql("SELECT English, Chinese FROM TWSE.Translate", engine)
        _translate_cache = dict(zip(df["English"], df["Chinese"]))
    except Exception:
        logger.warning("無法載入 TWSE.Translate 對照表，使用本地映射")
        _translate_cache = {}

    return _translate_cache


def _translate_feature_name(name: str) -> str:
    """將特徵英文名稱翻譯為中文。

    優先查詢 TWSE.Translate 對照表，不存在則查本地技術指標映射。

    Args:
        name: 特徵英文名稱。

    Returns:
        中文名稱，若無對照則回傳原始名稱。
    """
    translate = _load_translate_table()
    if name in translate:
        return translate[name]
    return _INDICATOR_NAME_ZH.get(name, name)


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
            "name": _translate_feature_name(name),
            "importance": round(float(imp), 6),
        })

    return result
