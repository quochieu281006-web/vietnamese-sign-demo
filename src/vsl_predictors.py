"""Nạp bộ phân loại tĩnh hợp nhất (chữ ASL + dấu chữ tiếng Việt) thành hàm dự đoán cho VietnameseSession."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from data_processing import normalize_landmarks

ROOT = Path(__file__).resolve().parents[1]
STATIC_MODEL = "VSL_unified_static.joblib"


def load_static_predictor():
    """static_fn(lm)->(nhãn, conf). lm là 21 điểm mốc từ khung webcam đã lật gương."""
    model = joblib.load(ROOT / "models" / STATIC_MODEL)
    classes = list(model.classes_)

    def static_fn(lm: np.ndarray):
        p = model.predict_proba(normalize_landmarks(lm)[None])[0]
        k = int(p.argmax())
        return classes[k], float(p[k])

    return static_fn


def load_tone_predictor():
    """tone_fn(clip (30,21,3))->(nhãn, conf), hoặc None nếu chưa có mô hình (chạy src/train_vsl_tone.py)."""
    f = ROOT / "models" / "VSL_tone_rf.joblib"
    if not f.exists():
        return None
    from vsl_tone_features import combined_features
    bundle = joblib.load(f)
    model, labels = bundle["model"], bundle["labels"]
    model.n_jobs = 1

    def tone_fn(clip: np.ndarray):
        p = model.predict_proba(combined_features(clip[None]))[0]
        k = int(p.argmax())
        return labels[k], float(p[k])

    return tone_fn
