"""Làm mượt dự đoán từng khung hình và ghép chữ cái thành từ.

Luật chốt chữ:
  1. Bỏ phiếu số đông trong cửa sổ `window` khung gần nhất.
  2. Chữ ứng viên phải giữ ổn định `hold_seconds` và độ tin cậy TB >= `min_conf` mới được chốt.
  3. Chốt xong phải "nhả" (mất tay, hoặc đổi sang chữ khác >= `release_seconds`) mới nhận chữ lặp.
  4. Lớp `space` thêm khoảng trắng, `del` xóa chữ cuối, `nothing` coi như không có tay.
"""
from __future__ import annotations

import time
from collections import Counter, deque


class LetterStabilizer:
    def __init__(self, window: int = 8, hold_seconds: float = 1.0, min_conf: float = 0.8,
                 release_seconds: float = 0.4, lost_grace: float = 0.25, clock=time.monotonic):
        self.lost_grace = lost_grace
        self.lost_since: float | None = None
        self.window = window
        self.hold_seconds = hold_seconds
        self.min_conf = min_conf
        self.release_seconds = release_seconds
        self.clock = clock
        self.text = ""
        self.reset_tracking()

    def reset_tracking(self):
        self.votes: deque = deque(maxlen=self.window)
        self.candidate: str | None = None
        self.candidate_since = 0.0
        self.last_committed: str | None = None
        self.released_since: float | None = None
        self.lost_since = None

    def clear(self):
        self.text = ""
        self.reset_tracking()

    def progress(self) -> float:
        """Tiến độ giữ chữ hiện tại (0..1), dùng để vẽ thanh tiến trình."""
        if self.candidate is None:
            return 0.0
        return min(1.0, (self.clock() - self.candidate_since) / self.hold_seconds)

    def update(self, label: str | None, conf: float = 0.0) -> str | None:
        """Nhận dự đoán của một khung (label=None nếu không thấy tay). Trả về chữ vừa chốt, nếu có."""
        now = self.clock()
        missing = label is None or label == "nothing"
        if not missing:
            self.lost_since = None
        elif self.lost_since is None:
            self.lost_since = now
        # MediaPipe thỉnh thoảng mất tay 1-2 khung: chỉ coi là rút tay khi mất quá `lost_grace`.
        if missing and now - self.lost_since < self.lost_grace:
            return None
        if missing:
            self.votes.clear()
            self.candidate = None
            if self.last_committed is not None and self.released_since is None:
                self.released_since = self.lost_since
            if self.released_since is not None and now - self.released_since >= self.release_seconds:
                self.last_committed = None
            return None

        self.votes.append((label, conf))
        top, _ = Counter(l for l, _ in self.votes).most_common(1)[0]
        confs = [c for l, c in self.votes if l == top]
        mean_conf = sum(confs) / len(confs)

        if top != self.candidate:
            self.candidate, self.candidate_since = top, now
            if top != self.last_committed:
                self.last_committed = None  # đổi sang chữ khác = đã nhả chữ trước
            return None

        if top == self.last_committed:
            if self.released_since is not None and now - self.released_since >= self.release_seconds:
                self.last_committed = None
            else:
                return None
        if now - self.candidate_since >= self.hold_seconds and mean_conf >= self.min_conf \
                and len(self.votes) >= self.window // 2:
            return self._commit(top, now)
        return None

    def _commit(self, label: str, now: float) -> str:
        if label == "space":
            self.text += " "
        elif label == "del":
            self.text = self.text[:-1]
        else:
            self.text += label
        self.last_committed = label
        self.released_since = None
        self.candidate_since = now
        return label
