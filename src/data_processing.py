"""Tiền xử lý dữ liệu ASL Alphabet: chọn mẫu, chia tập theo khối, trích điểm mốc MediaPipe.

Dùng chung cho các notebook. Mỗi ảnh cho ra 3 dạng dữ liệu:
  - image    : toàn khung 64x64 RGB (cho hướng A)
  - landmark : 21 điểm mốc bàn tay (21x3) hoặc NaN nếu MediaPipe không thấy tay (hướng B)
  - crop     : vùng bàn tay cắt theo điểm mốc 64x64 RGB (hướng C)
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
PROC_DIR = ROOT / "data" / "processed"
IMG_SIZE = 64
BLOCK = 150  # số khung hình liên tiếp gộp thành 1 khối khi chia tập (chống rò rỉ dữ liệu)

CLASSES = sorted(p.name for p in RAW_DIR.iterdir() if p.is_dir()) if RAW_DIR.exists() else []


def _frame_index(path: Path) -> int:
    """A123.jpg -> 123 (số thứ tự khung hình liên tiếp trong video gốc)."""
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def list_samples(n_per_class: int = 1000, seed: int = 42) -> list[dict]:
    """Chọn n_per_class ảnh/lớp, rải đều theo thứ tự khung hình, gán nhóm khối và tập chia.

    Chia train/val/test = 70/15/15 theo KHỐI khung liên tiếp: các ảnh gần giống nhau
    luôn nằm cùng một tập, nên điểm test không bị cao giả.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ci, cls in enumerate(CLASSES):
        files = sorted((RAW_DIR / cls).glob("*.jpg"), key=_frame_index)
        if n_per_class < len(files):
            keep = np.linspace(0, len(files) - 1, n_per_class).round().astype(int)
            files = [files[i] for i in keep]
        groups = sorted({_frame_index(f) // BLOCK for f in files})
        rng.shuffle(groups)
        n_tr, n_va = int(len(groups) * 0.7), int(len(groups) * 0.15)
        split_of = {g: ("train" if i < n_tr else "val" if i < n_tr + n_va else "test")
                    for i, g in enumerate(groups)}
        for f in files:
            g = _frame_index(f) // BLOCK
            rows.append({"path": str(f), "label": ci, "cls": cls, "group": f"{cls}_{g}",
                         "split": split_of[g]})
    return rows


def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """(21,3) -> vector 63 chiều bất biến theo vị trí và kích thước bàn tay.

    Tịnh tiến về cổ tay (điểm 0), chia cho khoảng cách cổ tay -> gốc ngón giữa (điểm 9).
    """
    pts = np.asarray(lm, dtype=np.float32).copy()
    pts -= pts[0]
    scale = float(np.linalg.norm(pts[9, :2])) or 1.0
    return (pts / scale).reshape(-1)


def crop_hand(img_rgb: np.ndarray, lm: np.ndarray, pad: float = 0.25) -> np.ndarray:
    """Cắt hình vuông quanh bàn tay theo hộp bao các điểm mốc, rồi resize IMG_SIZE."""
    h, w = img_rgb.shape[:2]
    xs, ys = lm[:, 0] * w, lm[:, 1] * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + pad)
    x0, x1 = int(max(cx - half, 0)), int(min(cx + half, w))
    y0, y1 = int(max(cy - half, 0)), int(min(cy + half, h))
    crop = img_rgb[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img_rgb
    return cv2.resize(crop, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


_hands = None


def _init_worker():
    global _hands
    import mediapipe as mp
    _hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                      min_detection_confidence=0.5)


def _process_one(path: str):
    img = cv2.imread(path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    res = _hands.process(rgb)
    if res.multi_hand_landmarks:
        lm = np.array([[p.x, p.y, p.z] for p in res.multi_hand_landmarks[0].landmark],
                      dtype=np.float32)
        return small, lm, crop_hand(rgb, lm)
    return small, np.full((21, 3), np.nan, np.float32), np.zeros_like(small)


def build_dataset(n_per_class: int = 1000, seed: int = 42, workers: int | None = None,
                  out_name: str = "asl_features.npz") -> Path:
    """Chạy MediaPipe trên các ảnh đã chọn, lưu cache .npz để các notebook sau dùng lại."""
    from tqdm.auto import tqdm
    rows = list_samples(n_per_class, seed)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    paths = [r["path"] for r in rows]
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        results = list(tqdm(ex.map(_process_one, paths, chunksize=32), total=len(paths)))
    ok = [i for i, r in enumerate(results) if r is not None]
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out = PROC_DIR / out_name
    np.savez(out,
             images=np.stack([results[i][0] for i in ok]),
             landmarks=np.stack([results[i][1] for i in ok]),
             crops=np.stack([results[i][2] for i in ok]),
             labels=np.array([rows[i]["label"] for i in ok]),
             splits=np.array([rows[i]["split"] for i in ok]),
             groups=np.array([rows[i]["group"] for i in ok]),
             classes=np.array(CLASSES))
    return out


def load_dataset(name: str = "asl_features.npz") -> dict:
    d = np.load(PROC_DIR / name, allow_pickle=False)
    return {k: d[k] for k in d.files}
