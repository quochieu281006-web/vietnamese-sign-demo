r"""Web demo ký tự tiếng Việt. Chạy: .venv\Scripts\python.exe src\web_app.py."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import mimetypes
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from vietnamese_composer import MODIFIERS, TONES
from vsl_predictors import load_static_predictor, load_tone_predictor
from vsl_session import VietnameseSession

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
LOG = logging.getLogger("vsl.web")
MAX_BODY = 1024 * 1024
MAX_TOKENS = 2000
CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(0,17),(17,18),(18,19),(19,20))


class ClientError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def decode_frame(body):
    # Browser sends a JPEG at 640x480; reject huge image dimensions before OpenCV.
    from PIL import Image
    try:
        with Image.open(io.BytesIO(body)) as image:
            if image.format != "JPEG" or max(image.size) > 1280 or min(image.size) < 16:
                raise ValueError("size/format")
        frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("decode")
        return frame
    except Exception as exc:
        raise ClientError("Khung hình JPEG không hợp lệ (tối đa 1280 px).") from exc


class ClientSession:
    def __init__(self, static_fn, tone_fn):
        self.capture_time = time.monotonic()
        self.core = VietnameseSession(static_fn, tone_fn, clock=lambda: self.capture_time)
        self.lock = threading.Lock()
        self.last_seen = time.monotonic()
        self.last_frame = 0.0
        self.hands = None
        self.closed = False
        self.source = None
        self.last_sample = None

    def reset_tracking(self):
        self.core.stab.reset_tracking()
        self.core.hist.clear()
        self.core.pending_tone = None
        self.core.disarmed = False
        self.core.block_until = 0.0
        self.core.status = ""
        self.core.tone_info = ""
        self.last_frame = 0.0
        self.last_sample = None
        self.source = None

    def snapshot(self, landmarks=None, committed=None):
        text, warnings = self.core.composer(self.core.tokens)
        return {"text": text, "tokens": self.core.tokens[-40:],
                "status": self.core.status, "progress": self.core.progress(),
                "tone_enabled": self.core.tone_fn is not None,
                "warnings": warnings[-3:], "landmarks": landmarks,
                "committed": committed, "tone_info": self.core.tone_info}

    def update_sample(self, lm, timestamp, source):
        if len(self.core.tokens) >= MAX_TOKENS:
            raise ClientError("Văn bản đã đầy. Hãy sao chép rồi xóa để tiếp tục.", 409)
        if self.source != source or (self.last_sample is not None and timestamp - self.last_sample > 0.75):
            self.reset_tracking()
        self.source, self.last_sample, self.capture_time = source, timestamp, timestamp
        return self.core.update(lm)

    def landmarks(self, payload):
        """Process acquisition timestamps, never HTTP arrival timing.

        A single request can contain multiple captured frames, so a slow network
        doesn't discard their trajectory or turn a delayed batch into a long hold.
        """
        if not isinstance(payload, dict) or not isinstance(payload.get("frames"), list):
            raise ClientError("Thiếu chuỗi điểm mốc.")
        frames = payload["frames"]
        if not 1 <= len(frames) <= 60:
            raise ClientError("Mỗi gói cần từ 1 đến 60 khung điểm mốc.")
        samples = []
        previous = self.last_sample if self.source == "browser" else None
        for frame in frames:
            if not isinstance(frame, dict):
                raise ClientError("Khung điểm mốc không hợp lệ.")
            timestamp, lm = frame.get("t"), frame.get("landmarks")
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp) or not 0 <= timestamp <= 1e9):
                raise ClientError("Thời điểm khung hình không hợp lệ.")
            if previous is not None and timestamp <= previous:
                raise ClientError("Khung hình phải tăng theo thời gian.", 409)
            if lm is not None:
                try:
                    lm = np.asarray(lm, dtype=np.float32)
                except (TypeError, ValueError) as exc:
                    raise ClientError("Điểm mốc không hợp lệ.") from exc
                if lm.shape != (21, 3) or not np.isfinite(lm).all() or np.abs(lm).max() > 5:
                    raise ClientError("Cần 21 điểm mốc hữu hạn, mỗi điểm có x/y/z.")
            samples.append((float(timestamp), lm))
            previous = timestamp
        if samples[-1][0] - samples[0][0] > 5:
            raise ClientError("Một gói không được dài quá 5 giây.")
        committed = []
        for timestamp, lm in samples:
            token = self.update_sample(lm, timestamp, "browser")
            if token:
                committed.append(token)
        result = self.snapshot(committed=committed[-1] if committed else None)
        result["committed_tokens"] = committed
        result["processed_frames"] = len(samples)
        return result

    def frame(self, frame):
        if len(self.core.tokens) >= MAX_TOKENS:
            raise ClientError("Văn bản đã đầy. Hãy sao chép rồi xóa để tiếp tục.", 409)
        if self.hands is None:
            from mediapipe_runtime import prepare_mediapipe
            mp = prepare_mediapipe()
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.6, min_tracking_confidence=0.5)
        # Same orientation as vsl_app.py: mirrored pixels, not just mirrored CSS.
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        lm = None
        if result.multi_hand_landmarks:
            lm = np.array([[p.x, p.y, p.z] for p in result.multi_hand_landmarks[0].landmark], np.float32)
        committed = self.update_sample(lm, time.monotonic(), "server")
        return self.snapshot(None if lm is None else lm.tolist(), committed)

    def action(self, action):
        if action == "clear":
            self.core.clear()
            self.reset_tracking()
        elif action == "backspace":
            self.core.backspace()
        elif action == "pause":
            self.reset_tracking()
            if self.hands:
                self.hands.close()
                self.hands = None
        elif len(self.core.tokens) >= MAX_TOKENS:
            raise ClientError("Văn bản đã đầy. Hãy xóa để tiếp tục.", 409)
        elif action == "space":
            self.core.tokens.append("space")
        elif action in TONES:
            if not self.core.add_tone(action):
                raise ClientError("Hãy nhận diện chữ trước khi thêm dấu. Muốn đổi dấu, xóa token dấu cũ trước.")
        elif action in MODIFIERS:
            if not self.core.tokens or self.core.tokens[-1] not in MODIFIERS[action]:
                raise ClientError("Dấu chữ chưa phù hợp: mũ cho a/e/o/d, móc cho o/u, trăng cho a.")
            self.core.tokens.append(action)
        else:
            raise ClientError("Thao tác không hợp lệ.")
        return self.snapshot()

    def close(self):
        self.closed = True
        if self.hands:
            self.hands.close()
            self.hands = None


class Registry:
    def __init__(self, static_fn, tone_fn, capacity=8, ttl=600):
        self.static_fn, self.tone_fn = static_fn, tone_fn
        self.capacity, self.ttl = capacity, ttl
        self.sessions = {}
        self.lock = threading.Lock()

    def create(self):
        with self.lock:
            now = time.monotonic()
            for key, session in list(self.sessions.items()):
                if now - session.last_seen > self.ttl and session.lock.acquire(blocking=False):
                    try:
                        session.close()
                        del self.sessions[key]
                    finally:
                        session.lock.release()
            if len(self.sessions) >= self.capacity:
                raise ClientError("Máy chủ đang đủ phiên dùng thử. Hãy thử lại sau ít phút.", 503)
            key = secrets.token_urlsafe(32)
            session = ClientSession(self.static_fn, self.tone_fn)
            self.sessions[key] = session
            return key, session

    def get(self, key):
        with self.lock:
            session = self.sessions.get(key)
            if session is None or session.closed:
                raise ClientError("Phiên đã kết thúc. Tải lại trang để bắt đầu phiên mới.", 401)
            session.last_seen = time.monotonic()
            return session

    def remove(self, key):
        with self.lock:
            session = self.sessions.pop(key, None)
        if session:
            with session.lock:
                session.close()

    def close(self):
        for key in list(self.sessions):
            self.remove(key)


async def synthesize(text):
    import edge_tts
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, "vi-VN-HoaiMyNeural", rate="-20%").stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    if not audio:
        raise RuntimeError("No audio")
    return bytes(audio)


def handler_for(registry):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def reply(self, status, body, content_type="application/json; charset=utf-8", cache="no-store"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("Permissions-Policy", "camera=(self), microphone=()")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass  # Do not log webcam frames, session tokens, or text.

        def do_GET(self):
            path = urlsplit(self.path).path
            assets = {"/": ("index.html", "text/html; charset=utf-8"),
                      "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                      "/hand-worker.js": ("hand-worker.js", "text/javascript; charset=utf-8"),
                      "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path == "/api/health":
                return self.reply(200, {"ready": True, "tone_enabled": registry.tone_fn is not None})
            if path.startswith("/vendor/mediapipe/0.10.21/"):
                asset = (WEB / path.lstrip("/")).resolve()
                vendor = (WEB / "vendor" / "mediapipe" / "0.10.21").resolve()
                if not asset.is_relative_to(vendor) or not asset.is_file():
                    return self.reply(404, {"error": "Không tìm thấy tài nguyên."})
                mime = {".mjs": "text/javascript", ".wasm": "application/wasm", ".task": "application/octet-stream"}.get(asset.suffix)
                return self.reply(200, asset.read_bytes(), mime or mimetypes.guess_type(asset.name)[0] or "application/octet-stream", "public, max-age=31536000, immutable")
            if path not in assets:
                return self.reply(404, {"error": "Không tìm thấy trang."})
            name, mime = assets[path]
            self.reply(200, (WEB / name).read_bytes(), mime)

        def do_POST(self):
            try:
                # No cross-origin APIs. Works both on localhost and behind HTTPS tunnels.
                origin = self.headers.get("Origin")
                if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                    raise ClientError("Yêu cầu khác nguồn bị từ chối.", 403)
                if self.headers.get("Transfer-Encoding"):
                    self.close_connection = True
                    raise ClientError("Không hỗ trợ kiểu truyền này.", 400)
                size = int(self.headers.get("Content-Length", "0"))
                if size < 0 or size > MAX_BODY:
                    self.close_connection = True
                    raise ClientError("Dữ liệu gửi lên quá lớn.", 413)
                body = self.rfile.read(size)
                path = urlsplit(self.path).path
                if path == "/api/session":
                    key, session = registry.create()
                    return self.reply(201, {"session": key, **session.snapshot()})
                key = self.headers.get("X-Session", "")
                session = registry.get(key)
                if path == "/api/close":
                    registry.remove(key)
                    return self.reply(200, {"closed": True})
                with session.lock:
                    if session.closed:
                        raise ClientError("Phiên đã kết thúc. Hãy tải lại trang.", 401)
                    if path == "/api/frame":
                        return self.reply(200, session.frame(decode_frame(body)))
                    if path == "/api/landmarks":
                        return self.reply(200, session.landmarks(json.loads(body)))
                    if path == "/api/action":
                        payload = json.loads(body)
                        if not isinstance(payload, dict) or not isinstance(payload.get("action"), str):
                            raise ClientError("Thao tác không hợp lệ.")
                        return self.reply(200, session.action(payload["action"]))
                    if path == "/api/speech":
                        text = session.core.text.strip()
                        if not text:
                            raise ClientError("Chưa có văn bản để đọc.")
                        if len(text) > 500:
                            raise ClientError("Mỗi lần đọc tối đa 500 ký tự.")
                    else:
                        raise ClientError("Không tìm thấy API.", 404)
                try:
                    audio = asyncio.run(asyncio.wait_for(synthesize(text), timeout=20))
                except Exception:
                    return self.reply(503, {"error": "Không kết nối được giọng Việt trực tuyến. Có thể dùng giọng trình duyệt nếu thiết bị hỗ trợ."})
                self.reply(200, audio, "audio/mpeg")
            except ClientError as exc:
                self.reply(exc.status, {"error": str(exc)})
            except (ValueError, TypeError, UnicodeError):
                self.reply(400, {"error": "Dữ liệu yêu cầu không hợp lệ."})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True
            except Exception:
                LOG.exception("API failed")
                self.reply(500, {"error": "Lỗi xử lý. Dừng camera và thử lại; xem cửa sổ máy chủ để kiểm tra."})
    return Handler


def main():
    parser = argparse.ArgumentParser(description="Web local nhận diện ký tự tiếng Việt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-tone", action="store_true", help="Chỉ dùng nút thêm dấu thanh")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Browser extracts landmarks by default. Load Python MediaPipe lazily only
    # for compatibility mode, reducing server startup time and idle memory.
    static, tone = load_static_predictor(), None if args.no_tone else load_tone_predictor()
    # Serialize access to shared estimators; MediaPipe state is per visitor.
    model_lock = threading.Lock()
    def synchronized(fn):
        def predict(lm):
            with model_lock:
                return fn(lm)
        return predict
    registry = Registry(synchronized(static), synchronized(tone) if tone else None)
    server = ThreadingHTTPServer((args.host, args.port), handler_for(registry))
    server.daemon_threads = True
    print(f"\nVSL web: http://localhost:{args.port}\nGiu cua so nay mo. Ctrl+C de dung.\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        registry.close()


if __name__ == "__main__":
    main()
