"""LÆ°u clip dáº¥u thanh Láº¤Y Tá»ª LĂC DĂ™NG á»¨NG Dá»¤NG THáº¬T (khi báº¡n báº¥m phĂ­m 1-5 Ä‘á»ƒ sá»­a/bá»• sung dáº¥u).

ÄĂ¢y lĂ  dá»¯ liá»‡u Ä‘Ăºng Ä‘iá»u kiá»‡n sá»­ dá»¥ng (tá»‘c Ä‘á»™, vá»‹ trĂ­ tay, khoáº£ng cĂ¡ch tháº­t) nĂªn giĂºp mĂ´ hĂ¬nh sĂ¡t thá»±c táº¿ hÆ¡n dá»¯ liá»‡u quay
cĂ³ Ä‘áº¿m ngÆ°á»£c. LÆ°u cĂ¹ng Ä‘á»‹nh dáº¡ng vá»›i dá»¯ liá»‡u clip: archive_tone/data/vsl_clips/live.npz (X, y, hand).
train_vsl_tone.py Ä‘á»c má»i file .npz trong thÆ° má»¥c Ä‘Ă³ nĂªn tá»± dĂ¹ng dá»¯ liá»‡u nĂ y khi huáº¥n luyá»‡n láº¡i.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

LIVE_FILE = Path(__file__).resolve().parents[1] / "data" / "vsl_clips" / "live.npz"


def append_live_clip(label: str, clip: np.ndarray, hand: str = "right") -> int:
    """ThĂªm má»™t clip (30,21,3) Ä‘Ă£ gĂ¡n nhĂ£n. Tráº£ vá» tá»•ng sá»‘ clip live cá»§a nhĂ£n Ä‘Ă³."""
    assert clip.shape == (30, 21, 3), clip.shape
    if LIVE_FILE.exists():
        d = np.load(LIVE_FILE)
        X, y, h = d["X"], d["y"], d["hand"]
    else:
        LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        X, y, h = np.zeros((0, 30, 21, 3), np.float32), np.array([], dtype="<U8"), np.array([], dtype="<U5")
    X = np.concatenate([X, clip[None].astype(np.float32)])
    y = np.concatenate([y, np.array([label], dtype="<U8")])
    h = np.concatenate([h, np.array([hand], dtype="<U5")])
    np.savez(LIVE_FILE, X=X, y=y, hand=h)
    return int((y == label).sum())

