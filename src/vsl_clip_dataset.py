"""Lưu / đọc clip điểm mốc cho ký hiệu CÓ CHUYỂN ĐỘNG (dấu thanh) — data/vsl_clips/<người>.npz.

Mỗi clip = chuỗi N_FRAMES khung x 21 điểm x 3 toạ độ (toạ độ gốc của MediaPipe, chưa chuẩn hóa,
vì quỹ đạo bàn tay trong khung hình chính là thông tin cần học). Mỗi clip là một lần quay (take).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from vsl_dataset import safe_name

CLIP_DIR = Path(__file__).resolve().parents[1] / "data" / "vsl_clips"
N_FRAMES = 30
# `none` = clip KHÔNG phải dấu thanh (giữ yên một chữ, hoặc đưa tay vào/ra) để mô hình biết "không có dấu".
DEFAULT_LABELS = ["sac", "huyen", "hoi", "nga", "nang", "none"]


def resample_clip(frames: list, n: int = N_FRAMES, min_valid: float = 0.6):
    """Chuẩn hóa clip về đúng n khung. `frames`: list (21,3) hoặc None khi mất tay.

    Khung mất tay được nội suy tuyến tính. Trả về None nếu số khung thấy tay < min_valid.
    """
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


MIN_HAND_SCALE = 0.09   # cỡ tay tối thiểu (cổ tay -> gốc ngón giữa, theo cỡ khung); nhỏ hơn thì landmark rung mạnh


def hand_scale(clip: np.ndarray) -> float:
    """Cỡ bàn tay trung vị của clip (cổ tay -> gốc ngón giữa, đơn vị cỡ khung hình)."""
    return float(np.median([np.linalg.norm(f[9, :2] - f[0, :2]) for f in clip]))


def person_file(person: str) -> Path:
    return CLIP_DIR / f"{safe_name(person)}.npz"


def load_person(person: str) -> dict:
    f = person_file(person)
    if not f.exists():
        return {"X": np.zeros((0, N_FRAMES, 21, 3), np.float32), "y": np.array([], dtype="<U8"),
                "hand": np.array([], dtype="<U5")}
    d = np.load(f)
    return {k: d[k] for k in ("X", "y", "hand")}


def _save(person: str, d: dict) -> None:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(person_file(person), **d)


def append_clip(person: str, label: str, hand: str, clip: np.ndarray) -> int:
    """Thêm một clip (N_FRAMES,21,3). Trả về số clip của nhãn đó."""
    assert clip.shape == (N_FRAMES, 21, 3), clip.shape
    d = load_person(person)
    d = {"X": np.concatenate([d["X"], clip[None].astype(np.float32)]),
         "y": np.concatenate([d["y"], np.array([label], dtype="<U8")]),
         "hand": np.concatenate([d["hand"], np.array([hand], dtype="<U5")])}
    _save(person, d)
    return int((d["y"] == label).sum())


def count_clips(person: str, label: str) -> int:
    return int((load_person(person)["y"] == label).sum())


def delete_last_clip(person: str, label: str) -> int:
    d = load_person(person)
    idx = np.where(d["y"] == label)[0]
    if not len(idx):
        return 0
    keep = np.ones(len(d["y"]), bool); keep[idx[-1]] = False
    _save(person, {k: v[keep] for k, v in d.items()})
    return 1


def load_all() -> dict:
    parts = []
    for f in sorted(CLIP_DIR.glob("*.npz")):
        d = load_person(f.stem)
        d["person"] = np.array([f.stem] * len(d["y"]), dtype="<U32")
        parts.append(d)
    if not parts:
        return {"X": np.zeros((0, N_FRAMES, 21, 3), np.float32), "y": np.array([]), "person": np.array([])}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summary() -> str:
    d = load_all()
    if not len(d["y"]):
        return "Chưa có clip."
    people, labels = sorted(set(d["person"])), sorted(set(d["y"]))
    lines = ["nhan    " + "  ".join(f"{p[:10]:>10}" for p in people) + "     tong"]
    for lb in labels:
        cnt = [int(((d["y"] == lb) & (d["person"] == p)).sum()) for p in people]
        lines.append(f"{lb:<7} " + "  ".join(f"{c:>10}" for c in cnt) + f"  {sum(cnt):>7}")
    lines.append(f"\nTong {len(d['y'])} clip, {len(people)} nguoi, {len(labels)} nhan")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
