"""Lưu / đọc bộ dữ liệu điểm mốc ngôn ngữ ký hiệu Việt Nam tự thu (data/vsl/<người>.npz).

Mỗi file là một người ra dấu. Chỉ lưu 21 điểm mốc (x, y, z) của MediaPipe, không lưu ảnh.
Mỗi mẫu có thêm mã `take` (lần quay): các mẫu cùng một lần quay gần như trùng nhau nên khi
đánh giá phải chia theo take hoặc theo người, không chia ngẫu nhiên từng mẫu.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

VSL_DIR = Path(__file__).resolve().parents[1] / "data" / "vsl"

# Chữ gốc a, e, o, u... lấy từ bộ ASL; ở đây thu các DẤU bằng HÌNH TAY TĨNH để ghép với chữ gốc:
#   dấu chữ: mu (^: â ê ô), moc (ơ ư), trang (ă)
#   dấu thanh: sac, huyen, hoi, nga, nang (mỗi dấu một hình tay tĩnh, giữ yên ~1 giây)
# Mọi nhãn trong DEFAULT_LABELS là "dấu" (chữ thường); nhãn khác (a, e, o, u...) được hiểu là chữ cái ASL.
# Đổi nhãn khi thu bằng tham số --labels.
MARK_LABELS = ["mu", "moc", "trang"]
TONE_LABELS = ["sac", "huyen", "hoi", "nga", "nang"]
DEFAULT_LABELS = MARK_LABELS + TONE_LABELS


def safe_name(person: str) -> str:
    """Chuẩn hóa tên người thành tên file an toàn."""
    return "".join(c for c in person.lower() if c.isalnum() or c in "-_") or "unknown"


def person_file(person: str) -> Path:
    return VSL_DIR / f"{safe_name(person)}.npz"


def _empty() -> dict:
    return {"X": np.zeros((0, 21, 3), np.float32), "y": np.array([], dtype="<U8"),
            "hand": np.array([], dtype="<U5"), "take": np.array([], dtype=np.int32)}


def load_person(person: str) -> dict:
    """Đọc dữ liệu đã thu của một người; trả về dict rỗng nếu chưa có."""
    f = person_file(person)
    if not f.exists():
        return _empty()
    d = np.load(f)
    out = {k: d[k] for k in ("X", "y", "hand")}
    out["take"] = d["take"] if "take" in d.files else np.zeros(len(out["y"]), np.int32)
    return out


def _save(person: str, d: dict) -> None:
    VSL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(person_file(person), X=d["X"], y=d["y"], hand=d["hand"], take=d["take"])


def append_samples(person: str, label: str, hand: str, samples: list[np.ndarray]) -> int:
    """Thêm MỘT lần quay (nhiều mẫu 21x3) cho nhãn; trả về tổng số mẫu của nhãn đó."""
    d = load_person(person)
    if samples:
        take_id = int(d["take"].max()) + 1 if len(d["take"]) else 1
        n = len(samples)
        d = {"X": np.concatenate([d["X"], np.stack(samples).astype(np.float32)]),
             "y": np.concatenate([d["y"], np.array([label] * n, dtype="<U8")]),
             "hand": np.concatenate([d["hand"], np.array([hand] * n, dtype="<U5")]),
             "take": np.concatenate([d["take"], np.full(n, take_id, np.int32)])}
        _save(person, d)
    return int((d["y"] == label).sum())


def count_takes(person: str, label: str) -> int:
    d = load_person(person)
    return len(set(d["take"][d["y"] == label].tolist()))


def delete_last_take(person: str, label: str) -> int:
    """Xóa lần quay gần nhất của nhãn (khi quay hỏng). Trả về số mẫu đã xóa."""
    d = load_person(person)
    idx = np.where(d["y"] == label)[0]
    if not len(idx):
        return 0
    last = d["take"][idx].max()
    drop = idx[d["take"][idx] == last]
    keep = np.ones(len(d["y"]), bool); keep[drop] = False
    _save(person, {k: v[keep] for k, v in d.items()})
    return int(len(drop))


def load_all() -> dict:
    """Gộp mọi người: X (n,21,3), y nhãn, person tên người, hand tay, take mã lần quay."""
    parts = []
    for f in sorted(VSL_DIR.glob("*.npz")):
        d = load_person(f.stem)
        d["person"] = np.array([f.stem] * len(d["y"]), dtype="<U32")
        parts.append(d)
    if not parts:
        return {**_empty(), "person": np.array([], dtype="<U32")}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def summary() -> str:
    """Bảng 'số mẫu (số lần quay)' theo người x nhãn, để nhóm biết còn thiếu gì."""
    d = load_all()
    if not len(d["y"]):
        return "Chưa có dữ liệu."
    people, labels = sorted(set(d["person"])), sorted(set(d["y"]))
    lines = ["nhan   " + "  ".join(f"{p[:10]:>11}" for p in people) + "      tong"]
    for lb in labels:
        cells, tot = [], 0
        for p in people:
            m = (d["y"] == lb) & (d["person"] == p)
            cells.append(f"{int(m.sum())}({len(set(d['take'][m].tolist()))})"); tot += int(m.sum())
        lines.append(f"{lb:<6} " + "  ".join(f"{c:>11}" for c in cells) + f"  {tot:>8}")
    lines.append(f"\nTong {len(d['y'])} mau, {len(people)} nguoi, {len(labels)} nhan "
                 "(moi o = so mau (so lan quay))")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
