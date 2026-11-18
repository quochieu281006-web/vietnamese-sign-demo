"""Huấn luyện bộ phân loại TĨNH hợp nhất: chữ ASL (A-Z, del, space) + dấu tiếng Việt bằng hình tay tĩnh\n(dấu chữ mu, moc, trang và dấu thanh sac, huyen, hoi, nga, nang - nhãn nào có dữ liệu thì được học).

Gộp chung để mô hình tự phân biệt "dấu chữ" với "chữ cái thường" (tránh nhận nhầm hình tay chữ cái thành dấu).
Chạy:  .venv\\Scripts\\python.exe src\\train_vsl_unified_static.py
Lưu:   models/VSL_unified_static.joblib, reports/vsl_unified_static.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import vsl_dataset as vs
from vsl_models import static_features

ROOT = Path(__file__).resolve().parents[1]
ASL_NPZ = [ROOT / "data" / "processed" / "asl_features.npz"]
PER_CLASS = 300


def make_model():
    return make_pipeline(StandardScaler(), MLPClassifier((128, 64), max_iter=300, random_state=0))


def load_asl(rng: np.random.Generator):
    """Điểm mốc ASL (tập train), bỏ lớp `nothing` và các khung mất tay (toàn 0)."""
    f = next(p for p in ASL_NPZ if p.exists())
    d = np.load(f, allow_pickle=True)
    classes = [str(c) for c in d["classes"]]
    lm, lab, split = d["landmarks"], d["labels"], d["splits"]
    keep = (split == "train") & (np.abs(lm).sum((1, 2)) > 0) & (np.array(classes)[lab] != "nothing")
    idx = []
    for k in np.unique(lab[keep]):
        ii = np.where(keep & (lab == k))[0]
        idx.extend(rng.choice(ii, min(PER_CLASS, len(ii)), replace=False))
    idx = np.array(idx)
    lm = lm[idx].copy()
    # Ảnh ASL không lật gương, còn webcam (và dữ liệu tự thu) lấy từ khung đã lật -> lật x để cùng hệ tọa độ.
    lm[..., 0] = 1.0 - lm[..., 0]
    return lm, np.array(classes)[lab[idx]]


def mirror(lm: np.ndarray) -> np.ndarray:
    """Tay trái là ảnh đối xứng của tay phải: lật trục x của điểm mốc (n,21,3)."""
    out = lm.copy()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def load_training_landmarks(with_marks: bool = False):
    """Điểm mốc thô (n,21,3) trong hệ tọa độ khung webcam đã lật gương, nhãn, nhóm chống rò rỉ."""
    rng = np.random.default_rng(0)
    Xa, ya = load_asl(rng)
    d = vs.load_all()
    if not with_marks:                                   # kiểu Telex: chỉ cần chữ cái, bỏ mẫu dấu thu bằng hình tay
        keep = ~np.isin(d["y"], vs.DEFAULT_LABELS)
        d = {k: v[keep] for k, v in d.items()}
    # Chữ cái tự thu (a, e, o, u...) gộp vào lớp ASL tương ứng (viết hoa); dấu (nếu bật) giữ nguyên.
    yd = np.array([v if v in vs.DEFAULT_LABELS else v.upper() for v in d["y"]])
    X = np.concatenate([Xa, d["X"]])
    y = np.concatenate([ya, yd])
    # nhóm: ASL mỗi mẫu một nhóm riêng; mẫu tự thu theo lần quay (tránh rò rỉ giữa các khung liền kề)
    gd = d["take"] + 1000 * np.unique(d["person"], return_inverse=True)[1]
    g = np.concatenate([np.arange(len(ya)) + 10_000, gd])
    return X, y, g


def main():
    import sys
    Xraw, y0, g0 = load_training_landmarks(with_marks="--with-marks" in sys.argv)
    # Học cả hai tay: mỗi mẫu thêm một bản lật gương (tay trái). Bản lật cùng nhóm với bản gốc.
    X = static_features(np.concatenate([Xraw, mirror(Xraw)]))
    y, groups = np.concatenate([y0, y0]), np.concatenate([g0, g0])
    pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in GroupKFold(5).split(X, y, groups):
        pred[te] = make_model().fit(X[tr], y[tr]).predict(X[te])
    rep = classification_report(y, pred, digits=3, output_dict=True)
    marks = {k: rep[k] for k in vs.DEFAULT_LABELS if k in rep}   # dấu (chỉ có khi --with-marks)
    if marks:
        print({k: (round(v["precision"], 3), round(v["recall"], 3)) for k, v in marks.items()})
    # Telex dùng s f r x j d w a e o u: in recall các chữ này để biết chữ nào dễ nhầm
    print("recall chu Telex:", {c: round(rep[c]["recall"], 3) for c in "SFRXJDWAEOU" if c in rep})
    print("accuracy tong:", round(rep["accuracy"], 3))
    model = make_model().fit(X, y)
    joblib.dump(model, ROOT / "models" / "VSL_unified_static.joblib")
    (ROOT / "reports" / "vsl_unified_static.json").write_text(
        json.dumps({"accuracy": rep["accuracy"], "marks": marks, "n": int(len(y))}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
