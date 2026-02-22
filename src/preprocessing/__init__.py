"""資料前處理模組。

提供特徵工程、時間序列切分、標準化與一站式前處理管線。
"""

from src.preprocessing.feature_engineer import (
    build_feature_target,
    create_sliding_windows,
    create_target,
    select_features,
)
from src.preprocessing.pipeline import (
    PreprocessedData,
    SlidingWindowData,
    preprocess_pipeline,
    preprocess_sliding_pipeline,
)
from src.preprocessing.scaler import (
    fit_scaler,
    fit_transform_train_test,
    transform_features,
)
from src.preprocessing.split import time_series_split

__all__ = [
    "select_features",
    "create_target",
    "build_feature_target",
    "create_sliding_windows",
    "time_series_split",
    "fit_scaler",
    "transform_features",
    "fit_transform_train_test",
    "PreprocessedData",
    "preprocess_pipeline",
    "SlidingWindowData",
    "preprocess_sliding_pipeline",
]
