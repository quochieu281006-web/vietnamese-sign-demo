"""Huấn luyện nhận dạng dấu thanh bằng CLIP chuyển động (RandomForest) để dùng trong vsl_app.

Dữ liệu: data/vsl_clips/<người>.npz (X (n,30,21,3), y, hand). Cải tiến so với bản cũ:
  - tăng cường lệch thời gian (vẽ sớm/muộn, nhanh/chậm) cho clip dấu thanh;
  - thêm clip `none` KHÓ tạo tự động (cử chỉ vẽ dở, giữ một chữ cái, đổi chữ cái) - xem vsl_tone_negatives.py;
  - đánh giá học người này thử người kia (không chỉ CV ngẫu nhiên).
Chạy:  .venv\\Scripts\\python.exe src\\train_vsl_tone.py  ->  models/VSL_tone_rf.joblib, reports/vsl_tone.json
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from train_vsl_unified_static import load_asl
from vsl_tone_features import combined_features, timing_augment
from vsl_tone_negatives import make_negatives

ROOT = Path(__file__).resolve().parents[1]
CLIP_DIR = ROOT / "data" / "vsl_clips"
TONES = ["hoi", "huyen", "nang", "nga", "sac"]


def load_clips():
    X, y, p = [], [], []
    for f in sorted(CLIP_DIR.glob("*.npz")):
        d = np.load(f)
        X.append(d["X"]); y.extend(d["y"].tolist()); p.extend([f.stem] * len(d["y"]))
    return np.concatenate(X), np.array(y), np.array(p)


def load_poses(n: int = 600, seed: int = 0) -> np.ndarray:
    """Hình tay chữ cái ASL (điểm mốc thô, đã lật cho khớp webcam) làm nền cho clip `none` khó."""
    lm, _ = load_asl(np.random.default_rng(seed))
    return lm[np.random.default_rng(seed).choice(len(lm), min(n, len(lm)), replace=False)]


def build_set(X, y_txt, labels, rng, poses, n_aug: int = 4, hard_negatives: bool = True):
    """Gốc + n_aug bản lệch thời gian cho clip dấu thanh + clip `none` khó. Trả về (X, y chỉ số)."""
    Xs, ys = [X], [np.array([labels.index(v) for v in y_txt])]
    is_tone = np.isin(y_txt, TONES)
    for _ in range(n_aug):
        Xs.append(np.stack([timing_augment(c, rng, max_shift=8) for c in X[is_tone]]))
        ys.append(ys[0][is_tone])
    if hard_negatives:
        neg = make_negatives(X[is_tone], poses, rng)
        Xs.append(neg); ys.append(np.full(len(neg), labels.index("none")))
    return np.concatenate(Xs), np.concatenate(ys)


def make_rf():
    return RandomForestClassifier(300, random_state=0, n_jobs=-1)


def fit_tone_model(X, y_txt, labels, rng, poses, **kw):
    Xa, ya = build_set(X, y_txt, labels, rng, poses, **kw)
    return make_rf().fit(combined_features(Xa), ya)


def as_tone_fn(model, labels):
    model.n_jobs = 1                       # dự đoán từng clip: một luồng nhanh hơn

    def tone_fn(clip: np.ndarray):
        p = model.predict_proba(combined_features(clip[None]))[0]
        k = int(p.argmax())
        return labels[k], float(p[k])
    return tone_fn


def main():
    X, y_txt, person = load_clips()
    labels = sorted(set(y_txt) | {"none"})
    y = np.array([labels.index(v) for v in y_txt])
    poses = load_poses()
    rng = np.random.default_rng(0)
    acc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        m = fit_tone_model(X[tr], y_txt[tr], labels, rng, poses)
        acc.append((m.predict(combined_features(X[te])) == y[te]).mean())
    cross = {}
    for a, b in (("ten_ban", "huong"), ("huong", "ten_ban")):
        tr, te = person == a, (person == b) & np.isin(y_txt, ["nang", "sac"])
        m = fit_tone_model(X[tr], y_txt[tr], labels, rng, poses)
        cross[f"{a}->{b}"] = round(float((m.predict(combined_features(X[te])) == y[te]).mean()), 3)
    print(f"CV 5 fold: {np.mean(acc):.3f} +- {np.std(acc):.3f} | học người này thử người kia (nang/sac): {cross}")
    model = fit_tone_model(X, y_txt, labels, rng, poses)
    joblib.dump({"model": model, "labels": labels}, ROOT / "models" / "VSL_tone_rf.joblib")
    (ROOT / "reports" / "vsl_tone.json").write_text(json.dumps(
        {"cv_mean": float(np.mean(acc)), "cv_std": float(np.std(acc)), "cross_person": cross,
         "n_clips": int(len(y)), "labels": labels}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
