"""Lưu và đọc kết quả dự đoán trên tập test để so sánh công bằng giữa các mô hình."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RESULT_DIR = Path(__file__).resolve().parents[1] / "models" / "results"


def save_result(name: str, test_idx: np.ndarray, y_pred: np.ndarray, train_seconds: float,
                infer_ms_per_sample: float, note: str = "") -> None:
    """test_idx: chỉ số mẫu test trong mảng dữ liệu toàn cục. y_pred: nhãn dự đoán (cùng bảng lớp)."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RESULT_DIR / f"{name}.npz", test_idx=test_idx, y_pred=y_pred)
    (RESULT_DIR / f"{name}.json").write_text(json.dumps({
        "name": name, "train_seconds": round(train_seconds, 1),
        "infer_ms_per_sample": round(infer_ms_per_sample, 3), "note": note},
        ensure_ascii=False), encoding="utf-8")


def load_results() -> dict:
    out = {}
    for js in sorted(RESULT_DIR.glob("*.json")):
        meta = json.loads(js.read_text(encoding="utf-8"))
        arr = np.load(js.with_suffix(".npz"))
        out[meta["name"]] = {**meta, "test_idx": arr["test_idx"], "y_pred": arr["y_pred"]}
    return out


def metric_row(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
