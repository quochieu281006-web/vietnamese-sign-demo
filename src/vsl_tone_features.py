"""Đặc trưng cho nhận dạng dấu thanh bằng CLIP chuyển động (bản chịu nhiễu, xem archive_tone/src/tone_robust.py).

Clip (30 khung, 21 điểm, 3 toạ độ gốc MediaPipe) -> làm sạch (lọc trung vị + Savitzky-Golay)
-> hình tay chuẩn hóa + quỹ đạo -> vector phẳng cho RandomForest.
Các hàm phải GIỐNG HỆT lúc huấn luyện (tone_robust.py), nếu không mô hình sẽ lệch.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from data_processing import normalize_landmarks

N_FRAMES = 30
MIN_SCALE = 0.05
TIPS = (8, 20)


def resample_clip(frames: list, n: int = N_FRAMES, min_valid: float = 0.6):
    """List khung (21,3) hoặc None (mất tay) -> (n,21,3), nội suy khung mất; None nếu thấy tay < min_valid."""
    if not frames:
        return None
    valid = np.array([f is not None for f in frames])
    if valid.mean() < min_valid or valid.sum() < 2:
        return None
    t_all, t_ok = np.arange(len(frames)), np.where(valid)[0]
    arr = np.stack([frames[i] for i in t_ok]).reshape(len(t_ok), -1)
    full = np.stack([np.interp(t_all, t_ok, arr[:, d]) for d in range(arr.shape[1])], axis=1)
    t_new = np.linspace(0, len(frames) - 1, n)
    out = np.stack([np.interp(t_new, t_all, full[:, d]) for d in range(full.shape[1])], axis=1)
    return out.reshape(n, 21, 3).astype(np.float32)


def timing_augment(clip: np.ndarray, rng: np.random.Generator, max_shift: int = 10,
                   speed: tuple[float, float] = (0.75, 1.3)) -> np.ndarray:
    """Biến thể thời gian của clip (30,21,3): dịch lệch ±max_shift khung và co/giãn tốc độ, giữ đúng 30 khung.

    Mô phỏng người dùng bắt đầu cú vẽ sớm/muộn so với cửa sổ 2 giây và vẽ nhanh/chậm hơn lúc thu. Khung ngoài
    biên lặp khung đầu/cuối (tay đứng yên)."""
    c = (N_FRAMES - 1) / 2
    t = c + (np.arange(N_FRAMES) - c) * rng.uniform(*speed) + rng.uniform(-max_shift, max_shift)
    flat = clip.reshape(N_FRAMES, -1)
    out = np.stack([np.interp(t, np.arange(N_FRAMES), flat[:, d]) for d in range(flat.shape[1])], axis=1)
    return out.reshape(N_FRAMES, 21, 3).astype(np.float32)


def clean_clip(clip: np.ndarray) -> np.ndarray:
    c = median_filter(clip, size=(5, 1, 1), mode="nearest")
    return savgol_filter(c, 7, 2, axis=0, mode="interp").astype(np.float32)


def _shape_feats(clip: np.ndarray) -> np.ndarray:
    """(30,21,3) -> (30,65): hình tay chuẩn hóa từng khung + quỹ đạo cổ tay chia cỡ tay."""
    shape = np.stack([normalize_landmarks(f) for f in clip])
    scale = float(np.mean([np.linalg.norm((f - f[0])[9, :2]) for f in clip])) or 1.0
    traj = (clip[:, 0, :2] - clip[0, 0, :2]) / scale
    return np.concatenate([shape, traj], axis=1).astype(np.float32)


def _traj_feats(clip: np.ndarray) -> np.ndarray:
    """(30,21,3) đã làm sạch -> (30,6): cổ tay so với khung đầu + 2 đầu ngón so với cổ tay, chia cỡ tay (có sàn)."""
    scale = max(float(np.median([np.linalg.norm(f[9, :2] - f[0, :2]) for f in clip])), MIN_SCALE)
    wrist = (clip[:, 0, :2] - clip[0, 0, :2]) / scale
    tips = [(clip[:, t, :2] - clip[:, 0, :2]) / scale for t in TIPS]
    return np.concatenate([wrist, *tips], axis=1).astype(np.float32)


def _flat_shape(F: np.ndarray, n_seg: int = 6) -> np.ndarray:
    segs = np.array_split(np.arange(F.shape[1]), n_seg)
    return np.concatenate([np.concatenate([F[:, s].mean(1) for s in segs], axis=1), F.std(1)], axis=1)


def _flat_traj(F: np.ndarray, n_seg: int = 6) -> np.ndarray:
    seg = np.array_split(np.arange(F.shape[1]), n_seg)
    path = np.linalg.norm(np.diff(F[:, :, :2], axis=1), axis=2).sum(1, keepdims=True)
    return np.concatenate([np.concatenate([F[:, s].mean(1) for s in seg], axis=1), F.std(1), path,
                           _wiggle(F)], axis=1)


def _wiggle(F: np.ndarray, dead: float = 0.04) -> np.ndarray:
    """Số lần ĐỔI CHIỀU của quỹ đạo cổ tay và đầu ngón trỏ theo từng trục (bỏ qua rung nhỏ dưới `dead`).

    Phân biệt dấu hỏi (một khúc cong) với dấu ngã (lượn sóng, nhiều lần đổi chiều). (n,30,6) -> (n,4)."""
    out = []
    for cols in ((0, 1), (2, 3)):                      # cổ tay xy; đầu ngón trỏ so với cổ tay xy
        v = np.diff(F[:, :, cols], axis=1)             # (n,29,2)
        counts = np.zeros((len(F), 2))
        for i in range(len(F)):
            for ax in range(2):
                s = v[i, :, ax][np.abs(v[i, :, ax]) > dead]
                counts[i, ax] = int((np.diff(np.sign(s)) != 0).sum()) if len(s) > 1 else 0
        out.append(counts)
    return np.concatenate(out, axis=1)


def combined_features(X: np.ndarray) -> np.ndarray:
    """(n,30,21,3) -> (n,d): hình tay của clip đã làm sạch + quỹ đạo (đặc trưng khuyến nghị)."""
    Xc = np.stack([clean_clip(x) for x in X])
    shape = _flat_shape(np.stack([_shape_feats(c) for c in Xc]))
    traj = _flat_traj(np.stack([_traj_feats(clean_clip(x)) for x in X]))
    return np.concatenate([shape, traj], axis=1)
