"""Lõi nhận dạng tiếng Việt theo thời gian thực (không phụ thuộc OpenCV/webcam, kiểm thử được).

Mỗi khung: điểm mốc tay (hoặc None) đi qua hai nhánh:
  - TĨNH: bộ phân loại (chữ ASL, và dấu chữ mu/moc/trang nếu mô hình có học) -> LetterStabilizer chốt khi giữ yên ~1 giây.
  - DẤU THANH BẰNG CLIP (tùy chọn, cần tone_fn): lưu lịch sử 2 giây; khi tay chuyển động đủ lớn, cứ ~0.2 giây phân loại
    cửa sổ vừa qua; chốt khi cùng một dấu thanh (khác `none`) đủ tin cậy 2 lần liên tiếp.
Tay đang vung nhanh thì nhánh tĩnh tạm dừng. Sau khi chốt dấu thanh nghỉ `cooldown`, chỉ hoạt động lại khi tay đã dừng.
Token (chữ gốc, mu/moc/trang, sac/huyen/hoi/nga/nang, space) ghép thành chữ Việt bằng vietnamese_composer.compose_text.
Phím 1-5 hoặc dấu thanh bằng clip thêm token dấu thanh. Một âm tiết chỉ nhận một dấu thanh.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np

from letter_stabilizer import LetterStabilizer
from vietnamese_composer import TONES, compose_text
from vsl_tone_features import resample_clip


def motion_stats(frames: list[np.ndarray]) -> tuple[float, float]:
    """(độ dài quỹ đạo cổ tay, biên độ lớn nhất), chia cho cỡ bàn tay. Clip `none` ~0.5, dấu thanh ~2-3."""
    if len(frames) < 2:
        return 0.0, 0.0
    w = np.array([f[0, :2] for f in frames])
    scale = float(np.mean([np.linalg.norm(f[9, :2] - f[0, :2]) for f in frames])) or 1.0
    return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum() / scale), float(np.ptp(w, axis=0).max() / scale)


class VietnameseSession:
    def __init__(self, static_fn: Callable, tone_fn: Callable | None = None, clock=time.monotonic,
                 hold_seconds: float = 1.0, min_conf: float = 0.8, tone_conf: float = 0.5,
                 window_s: float = 2.0, eval_every: float = 0.2, min_path: float = 0.5, min_range: float = 0.25,
                 fast_path: float = 0.4, cooldown: float = 1.2, rearm_path: float = 0.12,
                 composer: Callable = compose_text):
        """static_fn(lm)->(nhãn, conf); tone_fn(clip (30,21,3))->(nhãn, conf) hoặc None để chỉ dùng phím.
        composer(tokens)->(văn bản, cảnh báo): mặc định compose_text; nhánh Telex truyền telex/src/vietnamese_telex.telex_text."""
        self.static_fn, self.tone_fn, self.clock, self.composer = static_fn, tone_fn, clock, composer
        self.stab = LetterStabilizer(hold_seconds=hold_seconds, min_conf=min_conf, clock=clock)
        self.tone_conf, self.window_s, self.eval_every = tone_conf, window_s, eval_every
        self.min_path, self.min_range, self.fast_path = min_path, min_range, fast_path
        self.cooldown, self.rearm_path = cooldown, rearm_path
        self.tokens: list[str] = []
        self.hist: deque = deque()                 # (t, lm hoặc None)
        self.last_eval, self.block_until, self.disarmed = 0.0, 0.0, False
        self.pending_tone: str | None = None
        self.status = ""                           # nhãn đang thấy, để hiển thị
        self.tone_info = ""                        # lý do nhánh dấu thanh chưa chốt, để hiển thị/chẩn đoán

    # --- hiển thị / chỉnh sửa ---
    @property
    def text(self) -> str:
        return self.composer(self.tokens)[0]

    def progress(self) -> float:
        return self.stab.progress()

    def clear(self):
        self.tokens.clear(); self.hist.clear(); self.stab.clear(); self.pending_tone = None

    def backspace(self):
        if self.tokens:
            self.tokens.pop()

    def last_clip(self):
        """Clip (30,21,3) của ~2 giây gần nhất, hoặc None nếu mất tay quá nhiều (để lưu dữ liệu dùng thật)."""
        return resample_clip([lm for _, lm in self.hist])

    def add_tone(self, tone: str) -> bool:
        """Thêm dấu thanh (phím / clip). Bỏ qua nếu chưa có âm tiết hoặc token cuối đã là dấu thanh / khoảng cách."""
        if tone not in TONES or not self.tokens or self.tokens[-1] in TONES or self.tokens[-1] == "space":
            return False
        self.tokens.append(tone)
        return True

    # --- xử lý từng khung ---
    def update(self, lm: np.ndarray | None) -> str | None:
        """Trả về token vừa chốt (nếu có)."""
        now = self.clock()
        if self.tone_fn is not None:
            self.hist.append((now, lm))
            while self.hist and now - self.hist[0][0] > self.window_s:
                self.hist.popleft()
            if self.disarmed and now >= self.block_until and self._recent_path(0.4) < self.rearm_path:
                self.disarmed = False
            if now < self.block_until or (self.disarmed and lm is not None and self._recent_path(0.4) >= self.rearm_path):
                self.stab.update(None)
                self.status = ""
                return None
            tone = self._maybe_tone(now) if lm is not None else None
            if tone:
                return self._commit_tone(tone, now)
            fast = self._recent_path(0.5) >= self.fast_path
        else:
            fast = False

        label, conf = (None, 0.0) if lm is None or fast or self.disarmed else self.static_fn(lm)
        self.status = "" if label is None else f"{label} {conf:.0%}"
        done = self.stab.update(label, conf)
        return self._commit_static(done) if done else None

    def _recent_path(self, seconds: float) -> float:
        now = self.clock()
        fr = [lm for t, lm in self.hist if lm is not None and now - t <= seconds]
        return motion_stats(fr)[0] if len(fr) >= 3 else 0.0

    def _maybe_tone(self, now: float) -> str | None:
        if now - self.last_eval < self.eval_every:
            return None
        valid = [lm for _, lm in self.hist if lm is not None]
        if len(valid) < 8 or now - self.hist[0][0] < 0.8 * self.window_s:
            self.tone_info = "dau thanh: dang gom khung hinh"
            return None
        path, rng = motion_stats(valid)
        if path < self.min_path or rng < self.min_range:
            self.pending_tone = None
            self.tone_info = f"dau thanh: tay it chuyen dong (path {path:.1f}/{self.min_path}, bien {rng:.1f}/{self.min_range})"
            return None
        self.last_eval = now
        clip = resample_clip([lm for _, lm in self.hist])
        if clip is None:
            self.tone_info = "dau thanh: mat tay qua nhieu"
            return None
        label, conf = self.tone_fn(clip)
        self.status = f"~{label} {conf:.0%}"
        self.tone_info = f"dau thanh: doan {label} {conf:.0%}" + (" (thap)" if conf < self.tone_conf and label != "none" else "")
        if label == "none" or conf < self.tone_conf:
            self.pending_tone = None
            return None
        if self.pending_tone == label:
            return label
        self.pending_tone = label
        return None

    def _commit_tone(self, label: str, now: float) -> str | None:
        ok = self.add_tone(label)
        self.block_until, self.disarmed = now + self.cooldown, True
        self.hist.clear(); self.pending_tone = None; self.stab.reset_tracking()
        return label if ok else None

    def _commit_static(self, label: str) -> str:
        if label == "del":
            self.backspace()
        elif label == "space":
            self.tokens.append("space")
        else:
            self.tokens.append(label.lower())
        return label
