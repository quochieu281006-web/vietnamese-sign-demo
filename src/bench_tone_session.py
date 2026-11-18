"""Đo độ ổn định dấu thanh trong điều kiện GẦN DÙNG THẬT: chạy cả phiên (VietnameseSession) trên clip chưa từng huấn luyện.

Mỗi clip kiểm tra được biến đổi ngẫu nhiên: vẽ nhanh/chậm (1.4-3.0 giây), đứng yên trước 0.2-1.2 giây và sau 0.3-1.0 giây,
nhiễu nhẹ. Mô hình dấu thanh chỉ được huấn luyện trên các fold khác (không rò rỉ).
Kết quả mỗi nhãn: đúng / sai nhãn / không chốt; với `none`: bao nhiêu lần chốt nhầm dấu thanh.
Chạy:  .venv\\Scripts\\python.exe src\\bench_tone_session.py [--aug N]
"""
from __future__ import annotations

import argparse
import collections

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from train_vsl_tone import as_tone_fn, fit_tone_model, load_clips, load_poses
from vsl_predictors import load_static_predictor
from vsl_session import VietnameseSession

FPS = 20
TONE_SET = {"sac", "huyen", "hoi", "nga", "nang"}


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def stretch(clip: np.ndarray, seconds: float) -> list:
    """Clip 30 khung -> chuỗi khung ở FPS thật kéo dài `seconds` giây."""
    n = max(int(seconds * FPS), 2)
    t = np.linspace(0, len(clip) - 1, n)
    flat = clip.reshape(len(clip), -1)
    return list(np.stack([np.interp(t, np.arange(len(clip)), flat[:, d]) for d in range(flat.shape[1])], 1)
                .reshape(n, 21, 3).astype(np.float32))


POSES = None


def train_fn(X, y_txt, labels, n_aug: int, rng, hard: bool):
    """Mô hình dấu thanh chỉ huấn luyện trên các clip trong fold huấn luyện (không rò rỉ)."""
    global POSES
    POSES = load_poses() if POSES is None else POSES
    return as_tone_fn(fit_tone_model(X, y_txt, labels, rng, POSES, n_aug=n_aug, hard_negatives=hard), labels)


def run_session(static_fn, tone_fn, frames, **kw):
    c = Clock()
    s = VietnameseSession(static_fn, tone_fn, clock=c, **kw)
    s.tokens = ["a"]
    for lm in frames:
        c.t += 1 / FPS
        s.update(lm)
    return [t for t in s.tokens[1:] if t != "space"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", type=int, default=0, help="số bản tăng cường thời gian cho mỗi clip huấn luyện")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--tone-conf", type=float, default=0.5)
    ap.add_argument("--test-person", default="", help="chỉ thử trên clip của người này (vd huong)")
    ap.add_argument("--no-hard", action="store_true", help="không dùng clip `none` khó (để so sánh bản cũ)")
    ap.add_argument("--letter-streams", type=int, default=40, help="số chuỗi chỉ-chữ-cái (không có dấu thanh) để đếm dấu giả")
    args = ap.parse_args()
    X, y_txt, person = load_clips()
    labels = sorted(set(y_txt) | {"none"}); y = np.array([labels.index(v) for v in y_txt])
    static_fn = load_static_predictor()
    rng = np.random.default_rng(0)
    res = collections.defaultdict(collections.Counter)
    if args.test_person:      # chỉ kiểm tra trên clip của một người; luôn huấn luyện kèm toàn bộ người khác
        idx = np.where(person == args.test_person)[0]
        splits = [(np.setdiff1d(np.arange(len(y)), idx[te]), idx[te])
                  for _, te in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y[idx])]
    else:
        splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    false_tones = streams = 0
    poses = load_poses()
    for tr, te in splits:
        tone_fn = train_fn(X[tr], y_txt[tr], labels, args.aug, rng, not args.no_hard)
        for _ in range(args.letter_streams // len(splits)):     # chuỗi chữ cái thật: giữ 3 chữ, đổi chữ giữa chừng
            frames = []
            for i in rng.choice(len(poses), 3, replace=False):
                frames += [poses[i] + rng.normal(0, 0.003, poses[i].shape).astype(np.float32)
                           for _ in range(int(rng.uniform(1.2, 2.0) * FPS))] + [None] * 8
            false_tones += len([t for t in run_session(static_fn, tone_fn, frames, tone_conf=args.tone_conf) if t in TONE_SET])
            streams += 1
        for i in te:
            for _ in range(args.variants):
                f = stretch(X[i], rng.uniform(1.4, 3.0))
                frames = [f[0]] * int(rng.uniform(0.2, 1.2) * FPS) + f + [f[-1]] * int(rng.uniform(0.3, 1.0) * FPS)
                frames = [(x + rng.normal(0, 0.003, x.shape)).astype(np.float32) for x in frames]
                got = run_session(static_fn, tone_fn, frames, tone_conf=args.tone_conf)
                lb = y_txt[i]
                tones = [t for t in got if t in TONE_SET]
                extra = len(got) - len(tones)                      # chữ cái/dấu chữ bị gõ thừa trong lúc vẽ dấu
                if lb == "none":                                   # giữ một chữ là hợp lệ; chỉ tính dấu thanh giả
                    res[lb]["dung (khong chot dau)" if not tones else "CHOT NHAM " + tones[0]] += 1
                else:
                    res[lb]["dung" if tones == [lb] else ("khong chot" if not tones else f"sai:{'+'.join(tones)}")] += 1
                    if tones == [lb] and extra:
                        res[lb]["+chu thua"] += 1                  # dấu đúng nhưng kèm chữ thừa (đếm riêng, không tính lỗi dấu)
    tot_ok = tot = 0
    for lb in labels:
        c = res[lb]; n = sum(v for k, v in c.items() if k != "+chu thua"); ok = c["dung"] + c["dung (khong chot dau)"]
        tot_ok += ok; tot += n
        print(f"{lb:6s} n={n:3d} dung {ok / n:5.1%} | " + ", ".join(f"{k}:{v}" for k, v in c.most_common() if not k.startswith("dung")))
    print(f"Chuoi chi-co-chu-cai: {false_tones} dau thanh gia / {streams} chuoi")
    print(f"TONG dung {tot_ok / tot:.1%}  (aug={args.aug}, hard_negatives={not args.no_hard}, tone_conf={args.tone_conf})")


if __name__ == "__main__":
    main()
