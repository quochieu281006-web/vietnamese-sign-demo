"""Ghép chuỗi ký hiệu (chữ gốc + dấu + dấu thanh) thành chữ tiếng Việt có dấu.

Token đầu vào (đúng thứ tự người ra dấu, dấu thanh đánh SAU các chữ của âm tiết):
  - chữ gốc: 'a'..'z'
  - dấu chữ: 'mu' (^: a->â e->ê o->ô, và d->đ theo quy ước nhóm), 'moc' (o->ơ u->ư), 'trang' (a->ă)
  - dấu thanh: 'sac' 'huyen' 'hoi' 'nga' 'nang'
  - 'space' ngắt từ
Ví dụ "Hà Nội": h a huyen space n o mu i nang -> "hà nội".

Kiểu đặt dấu thanh: hiện đại (hòa, thủy), dấu thanh đặt lên nguyên âm có dấu chữ (ê, ơ...) nếu có.
"""
from __future__ import annotations

import unicodedata

MODIFIERS = {"mu": {"a": "â", "e": "ê", "o": "ô", "d": "đ"}, "moc": {"o": "ơ", "u": "ư"}, "trang": {"a": "ă"}}
TONES = {"sac": "́", "huyen": "̀", "hoi": "̉", "nga": "̃", "nang": "̣"}
VOWELS = set("aăâeêioôơuưy")
MARKED = set("ăâêôơư")


def _nucleus_index(s: list[str]) -> int | None:
    """Vị trí nguyên âm mang dấu thanh trong một âm tiết (đã áp dấu chữ)."""
    idx = []
    for i, ch in enumerate(s):
        if ch not in VOWELS:
            continue
        if ch == "u" and i > 0 and s[i - 1] == "q":                      # qu: u thuộc phụ âm
            continue
        if ch == "i" and i > 0 and s[i - 1] == "g" and i + 1 < len(s) and s[i + 1] in VOWELS:
            continue                                                       # gi + nguyên âm: i thuộc phụ âm
        idx.append(i)
    if not idx:
        return None
    marked = [i for i in idx if s[i] in MARKED]
    if marked:
        return marked[-1]                                                  # ê, ơ, ư...: ưu tiên, lấy cái cuối
    if len(idx) == 1:
        return idx[0]
    if len(idx) == 2:
        # có âm cuối: dấu ở nguyên âm 2 (toán, hoàn); không âm cuối: nguyên âm 1 (mái, hòa, thủy)
        return idx[1] if idx[-1] < len(s) - 1 else idx[0]
    return idx[1]                                                          # ba nguyên âm: ngoài, khuỷu


def compose_syllable(tokens: list[str]) -> tuple[str, list[str]]:
    """Ghép một âm tiết. Trả về (chữ, danh sách cảnh báo)."""
    chars: list[str] = []
    tone: str | None = None
    warnings: list[str] = []
    for t in tokens:
        if t in TONES:
            if tone is not None:
                warnings.append(f"bỏ dấu thanh thừa '{t}'")
            else:
                tone = t
        elif t in MODIFIERS:
            if chars and chars[-1] in MODIFIERS[t]:
                chars[-1] = MODIFIERS[t][chars[-1]]
            else:
                prev = chars[-1] if chars else "(đầu từ)"
                warnings.append(f"dấu '{t}' không đi với '{prev}', bỏ qua")
        elif len(t) == 1 and t.isalpha():
            chars.append(t.lower())
        else:
            warnings.append(f"token lạ '{t}'")
    if tone is not None:
        i = _nucleus_index(chars)
        if i is None:
            warnings.append(f"không có nguyên âm để đặt dấu '{tone}'")
        else:
            chars[i] = chars[i] + TONES[tone]
    return unicodedata.normalize("NFC", "".join(chars)), warnings


def compose_text(tokens: list[str]) -> tuple[str, list[str]]:
    """Ghép cả chuỗi token (có 'space' ngắt từ). Trả về (văn bản, cảnh báo)."""
    out, warnings, cur = [], [], []
    for t in tokens:
        if t == "space":
            word, w = compose_syllable(cur); out.append(word); warnings += w; cur = []
        else:
            cur.append(t)
    word, w = compose_syllable(cur); out.append(word); warnings += w
    return " ".join(out) if len(out) > 1 else out[0], warnings
