"""Tạo clip "KHÔNG PHẢI DẤU THANH" khó (hard negatives) cho nhận dạng dấu thanh bằng clip.

Lúc dùng thật, cửa sổ 2 giây trượt liên tục nên thường chứa những thứ mà clip quay có đếm ngược không có:
  1. cử chỉ mới vẽ DỞ (chưa xong) -> không được chốt sớm;
  2. bàn tay đang giữ một chữ cái (hình tay bất kỳ, rung nhẹ, trôi chậm);
  3. bàn tay đổi từ chữ này sang chữ khác (hình tay đổi nhanh nhưng cổ tay gần đứng yên).
Cả ba đều gán nhãn `none`. Clip có dạng (30, 21, 3) như dữ liệu thật (toạ độ gốc MediaPipe, khung webcam đã lật).
"""
from __future__ import annotations

import numpy as np

N = 30


def partial_gesture(clip: np.ndarray, rng: np.random.Generator, lo: int = 6, hi: int = 14) -> np.ndarray:
    """Cửa sổ kết thúc khi cử chỉ mới được `k` khung đầu (k <= 14/30: chưa tới nửa). Phía trước là tay đứng yên."""
    k = int(rng.integers(lo, hi + 1))
    return np.concatenate([np.repeat(clip[:1], N - k, axis=0), clip[:k]]).astype(np.float32)


def held_letter(pose: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Một hình tay giữ yên N khung: rung nhẹ + trôi chậm vị trí/kích thước (người dùng chỉnh tay)."""
    t = np.linspace(0, 1, N)[:, None, None]
    drift = rng.normal(0, 0.04, (1, 1, 2)) * t                         # trôi tối đa vài % khung hình
    grow = 1 + rng.normal(0, 0.06) * t                                  # lại gần / ra xa camera một chút
    c = pose[:, :2].mean(0)
    xy = (pose[None, :, :2] - c) * grow + c + drift
    z = np.repeat(pose[None, :, 2:], N, axis=0)
    out = np.concatenate([xy, z], axis=2) + rng.normal(0, 0.003, (N, 21, 3))
    return out.astype(np.float32)


def letter_switch(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hình tay đổi từ `a` sang `b` ở một thời điểm ngẫu nhiên trong 3-9 khung, cổ tay gần đứng yên."""
    start, length = int(rng.integers(6, 18)), int(rng.integers(3, 10))
    w = np.clip((np.arange(N) - start) / length, 0, 1)[:, None, None]
    out = (1 - w) * a[None] + w * b[None]
    return (out + rng.normal(0, 0.003, out.shape)).astype(np.float32)


def make_negatives(tone_clips: np.ndarray, poses: np.ndarray, rng: np.random.Generator,
                   n_partial: int = 1, n_held: int = 150, n_switch: int = 150) -> np.ndarray:
    """tone_clips (n,30,21,3) clip dấu thanh thật; poses (m,21,3) hình tay chữ cái. Trả về (k,30,21,3)."""
    out = []
    for c in tone_clips:
        out += [partial_gesture(c, rng) for _ in range(n_partial)]
    for i in rng.choice(len(poses), n_held, replace=len(poses) < n_held):
        out.append(held_letter(poses[i], rng))
    for _ in range(n_switch):
        i, j = rng.choice(len(poses), 2, replace=False)
        a, b = poses[i], poses[j]
        shift = rng.normal(0, 0.02, (1, 2)) + (a[:, :2].mean(0) - b[:, :2].mean(0))    # đặt b cùng vị trí a: chỉ đổi hình
        b = b.copy(); b[:, :2] += shift
        out.append(letter_switch(a, b, rng))
    return np.stack(out)
