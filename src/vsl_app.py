"""Ứng dụng webcam đánh vần tiếng Việt: chữ ASL + dấu chữ bằng HÌNH TAY (mu / moc / trang)
+ dấu thanh bằng CỬ CHỈ (vẽ bằng tay, cần models/VSL_tone_rf.joblib) hoặc phím 1-5.

Chạy:  .venv\\Scripts\\python.exe src\\vsl_app.py [--camera 0] [--hold 1.0] [--no-tone] [--tone-conf 0.5]
Ví dụ "hà nội": giữ H, A, vẽ dấu huyền, SPACE, giữ N, O, hình `mu`, I, vẽ dấu nặng.
Dấu thanh bằng cử chỉ: vẽ dấu (~2 giây) sau âm tiết; bấm phím 1-5 ngay sau đó để xác nhận/sửa và lưu clip thật.
Phím: 1 sắc | 2 huyền | 3 hỏi | 4 ngã | 5 nặng | C xóa | BACKSPACE xóa 1 token | SPACE cách | S đọc | Q thoát
Cần: models/VSL_unified_static.joblib (src/train_vsl_unified_static.py --with-marks); tùy chọn models/VSL_tone_rf.joblib.
Nhánh Telex (gõ bằng chữ cái) nằm riêng ở telex/ và gọi lại main() này với composer và hint khác.
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vsl_live_clips import append_live_clip
from vsl_predictors import load_static_predictor, load_tone_predictor
from vietnamese_composer import compose_text
from vsl_session import VietnameseSession
from tts_vietnamese import prefetch, speak

GREEN, WHITE, GRAY = (80, 220, 100), (255, 255, 255), (170, 170, 170)
FONT_PATHS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
TONE_KEYS = {ord("1"): "sac", ord("2"): "huyen", ord("3"): "hoi", ord("4"): "nga", ord("5"): "nang"}
HINT = "1 sac 2 huyen 3 hoi 4 nga 5 nang | C xoa | BKSP | SPACE | S doc | Q"


def _font(size: int):
    for p in FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_ui(frame, sess: VietnameseSession, fps: float, note: str = "", hint: str = HINT):
    """Vẽ bằng PIL vì OpenCV không hiển thị được chữ có dấu tiếng Việt."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
    cv2.rectangle(frame, (15, 52), (215, 62), GRAY, 1)
    cv2.rectangle(frame, (15, 52), (15 + int(200 * sess.progress()), 62), GREEN, -1)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    d.text((15, 8), sess.status or "-", font=_font(30), fill=GREEN)
    d.text((w - 110, 8), f"{fps:.0f} FPS", font=_font(22), fill=WHITE)
    d.text((15, h - 62), (sess.text[-34:] or " ") + "_", font=_font(40), fill=WHITE)
    d.text((15, h - 100), "tokens: " + " ".join(sess.tokens[-10:]), font=_font(18), fill=GRAY)
    d.text((15, 76), note or sess.tone_info, font=_font(17), fill=(0, 220, 255))   # lý do dấu thanh chưa chốt / thông báo lưu
    d.text((240, 44), hint, font=_font(14), fill=GRAY)
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main(composer=compose_text, hint: str = HINT):
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hold", type=float, default=1.0, help="giây giữ dấu tĩnh để chốt")
    ap.add_argument("--no-tone", action="store_true", help="tắt nhận dạng dấu thanh bằng clip (chỉ Telex/phím 1-5)")
    ap.add_argument("--tone-conf", type=float, default=0.5, help="ngưỡng tin cậy dấu thanh (hạ xuống nếu khó ăn)")
    ap.add_argument("--no-save", action="store_true", help="không lưu clip thật khi bấm phím 1-5")
    args = ap.parse_args()

    import mediapipe as mp
    tone_fn = None if args.no_tone else load_tone_predictor()
    print("Dấu thanh bằng cử chỉ:", "BẬT (vẽ dấu bằng tay sau chữ, ~2 giây)" if tone_fn else "TẮT (dùng Telex hoặc phím 1-5)")
    sess = VietnameseSession(load_static_predictor(), tone_fn, hold_seconds=args.hold, tone_conf=args.tone_conf,
                             composer=composer)
    note, note_t = "", 0.0
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam số {args.camera} (thử --camera 1)")
    last, fps = time.time(), 0.0
    shown_text, changed_at, prefetched = "", time.time(), True
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm = None
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            lm = np.array([[p.x, p.y, p.z] for p in hl.landmark], np.float32)
            mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
        sess.update(lm)
        now = time.time()
        if sess.text != shown_text:                       # chữ đổi: chờ ổn định ~1,2 giây rồi tổng hợp sẵn giọng đọc
            shown_text, changed_at, prefetched = sess.text, now, False
        elif not prefetched and sess.text.strip() and now - changed_at > 1.2:
            prefetch(sess.text.strip()); prefetched = True
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6); last = now
        draw_ui(frame, sess, fps, note if time.time() - note_t < 3 else "", hint)
        cv2.imshow("VSL webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in TONE_KEYS:
            # Bấm phím ngay sau khi vẽ dấu = bạn xác nhận nhãn đúng -> lưu clip 2 giây vừa rồi làm dữ liệu thật.
            clip = sess.last_clip() if (tone_fn is not None and not args.no_save) else None
            if clip is not None:
                n = append_live_clip(TONE_KEYS[key], clip)
                note, note_t = f"da luu clip thuc te '{TONE_KEYS[key]}' (tong {n})", time.time()
            sess.add_tone(TONE_KEYS[key])
        elif key == ord("c"):
            sess.clear()
        elif key == 8:
            sess.backspace()
        elif key == ord(" "):
            sess.tokens.append("space")
        elif key == ord("s") and sess.text.strip():
            speak(sess.text.strip())
    cap.release(); cv2.destroyAllWindows(); hands.close()


if __name__ == "__main__":
    main()
