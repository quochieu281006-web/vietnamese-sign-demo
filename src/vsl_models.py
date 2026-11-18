"""Đặc trưng cho dấu chữ TĨNH tiếng Việt (mu, moc, trang): 21 điểm mốc chuẩn hóa như ASL (63 chiều).

(Nhánh dấu thanh bằng clip đã tạm dừng, xem archive_tone/.)
"""
from __future__ import annotations

import numpy as np

from data_processing import normalize_landmarks

MODELS_STATIC = "VSL_static_svm.joblib"


def static_features(X: np.ndarray) -> np.ndarray:
    """(n,21,3) -> (n,63)."""
    return np.stack([normalize_landmarks(x) for x in X])
