"""Đọc to chữ tiếng Việt: edge-tts (giọng vi-VN, cần mạng) -> dự phòng pyttsx3 khi lỗi/mất mạng.

Chạy riêng để thử:  .venv\\Scripts\\python.exe src\\tts_vietnamese.py "xin chào"
- Đọc trong luồng riêng nên không làm đứng video; các lần đọc xếp hàng, không chồng tiếng.
- Tổng hợp trước (prefetch) khi chữ đã ổn định -> bấm S gần như đọc ngay; chữ đã đọc được lưu cache.
- edge-tts thỉnh thoảng trả "No audio" ở lần gọi đầu: tự thử lại trước khi rơi về giọng offline (giọng Anh, đọc sai).
Văn bản được gửi tới dịch vụ của Microsoft khi dùng edge-tts.
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import sys
import tempfile
import threading

VOICE = "vi-VN-HoaiMyNeural"        # nữ; giọng nam: vi-VN-NamMinhNeural
RATE = "-20%"                       # chậm lại cho dễ nghe (0% = bình thường, -30% = rất chậm)
RETRIES = 3

_cache: dict[str, str] = {}         # văn bản -> đường dẫn mp3
_cache_lock = threading.Lock()      # bảo vệ cache + tránh tổng hợp trùng một câu
_play_lock = threading.Lock()       # chỉ phát một câu mỗi lúc (MCI dùng chung một alias)


def _play_mp3(path: str) -> None:
    """Phát mp3 bằng MCI của Windows (không cần thư viện thêm), chờ phát xong."""
    mci = ctypes.windll.winmm.mciSendStringW
    mci('close vsl_tts', None, 0, 0)
    if mci(f'open "{path}" type mpegvideo alias vsl_tts', None, 0, 0):
        raise RuntimeError("không mở được file âm thanh")
    try:
        mci('play vsl_tts wait', None, 0, 0)
    finally:
        mci('close vsl_tts', None, 0, 0)


def _synthesize(text: str) -> str:
    """Tạo mp3 bằng edge-tts (thử lại nếu dịch vụ trả rỗng). Có cache theo văn bản."""
    with _cache_lock:
        path = _cache.get(text)
        if path and os.path.exists(path):
            return path
        import edge_tts
        last: Exception | None = None
        for _ in range(RETRIES):
            fd, path = tempfile.mkstemp(suffix=".mp3"); os.close(fd)
            try:
                asyncio.run(edge_tts.Communicate(text, VOICE, rate=RATE).save(path))
                if os.path.getsize(path) > 0:
                    _cache[text] = path
                    return path
            except Exception as exc:  # noqa: BLE001 - thử lại
                last = exc
            os.remove(path)
        raise RuntimeError(f"edge-tts thất bại sau {RETRIES} lần: {last}")


def _speak_offline(text: str) -> None:
    import pyttsx3
    eng = pyttsx3.init(); eng.say(text); eng.runAndWait()


def speak_now(text: str) -> str:
    """Đọc đồng bộ; trả về 'edge', 'offline' hoặc 'loi'."""
    try:
        path = _synthesize(text)
        with _play_lock:
            _play_mp3(path)
        return "edge"
    except Exception as exc:  # noqa: BLE001 - mất mạng/lỗi dịch vụ: chuyển sang giọng offline
        print("edge-tts lỗi, dùng giọng offline (có thể không đọc đúng tiếng Việt):", exc)
    try:
        with _play_lock:
            _speak_offline(text)
        return "offline"
    except Exception as exc:  # noqa: BLE001 - tính năng phụ, không được làm sập ứng dụng
        print("Không đọc được thành tiếng:", exc); return "loi"


def speak(text: str) -> None:
    """Đọc trong luồng riêng, không chặn video."""
    if text.strip():
        threading.Thread(target=speak_now, args=(text,), daemon=True).start()


def prefetch(text: str) -> None:
    """Tổng hợp sẵn âm thanh của `text` ở luồng nền (lỗi thì bỏ qua) để lần speak() sau đọc ngay."""
    if not text.strip() or text in _cache:
        return

    def _run():
        try:
            _synthesize(text)
        except Exception:  # noqa: BLE001 - chỉ là tối ưu tốc độ
            pass
    threading.Thread(target=_run, daemon=True).start()


if __name__ == "__main__":
    print(speak_now(" ".join(sys.argv[1:]) or "xin chào, đây là bản thử giọng đọc tiếng Việt"))
