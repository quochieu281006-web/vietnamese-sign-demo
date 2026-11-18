"""Ứng dụng webcam nhận dạng chữ cái ASL theo thời gian thực (MediaPipe + MLP trên điểm mốc).

Chạy:   .venv\\Scripts\\python.exe src\\webcam_app.py [--camera 0] [--model B_mlp]
Phím:   C = xóa chữ | BACKSPACE = xóa 1 ký tự | SPACE = thêm khoảng trắng | S = đọc thành tiếng | Q/ESC = thoát
Cử chỉ: dấu `space` và `del` của bộ dữ liệu cũng thêm khoảng trắng / xóa ký tự.
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from data_processing import PROC_DIR, normalize_landmarks
from letter_stabilizer import LetterStabilizer

ROOT = Path(__file__).resolve().parents[1]
GREEN, WHITE, GRAY, RED = (80, 220, 100), (255, 255, 255), (170, 170, 170), (60, 60, 230)


def load_classes() -> list[str]:
    return [str(c) for c in np.load(PROC_DIR / "asl_features.npz")["classes"]]


def load_model(name: str):
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    return tf.keras.models.load_model(ROOT / "models" / f"{name}.keras")


def speak(text: str) -> None:
    """Đọc to bằng pyttsx3 trong luồng riêng để không làm đứng video. Bỏ qua nếu chưa cài."""
    def _run():
        try:
            import pyttsx3
            eng = pyttsx3.init(); eng.say(text); eng.runAndWait()
        except Exception as exc:  # noqa: BLE001 - tính năng phụ, không được làm sập ứng dụng
            print("Không đọc được thành tiếng:", exc)
    threading.Thread(target=_run, daemon=True).start()


def predict_landmarks(model, lm: np.ndarray) -> tuple[int, float]:
    lm = lm.copy(); lm[:, 0] = 1.0 - lm[:, 0]   # khung webcam đã lật gương; ảnh huấn luyện ASL thì không
    probs = model.predict(normalize_landmarks(lm)[None], verbose=0)[0]
    k = int(probs.argmax())
    return k, float(probs[k])


def draw_ui(frame, label, conf, progress, text, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)
    shown = "-" if label is None else label.upper()
    cv2.putText(frame, shown, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, GREEN if label else GRAY, 4)
    cv2.putText(frame, f"conf {conf:.0%}", (190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.rectangle(frame, (190, 42), (190 + 200, 58), GRAY, 1)
    cv2.rectangle(frame, (190, 42), (190 + int(200 * progress), 58), GREEN, -1)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.rectangle(frame, (0, h - 60), (w, h), (30, 30, 30), -1)
    cv2.putText(frame, text[-40:] + "_", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    cv2.putText(frame, "C xoa | BKSP xoa 1 | SPACE cach | S doc | Q thoat", (15, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--model", default="B_mlp")
    ap.add_argument("--hold", type=float, default=1.0, help="giây giữ dấu để chốt chữ")
    ap.add_argument("--min-conf", type=float, default=0.8)
    args = ap.parse_args()

    import mediapipe as mp
    classes = load_classes()
    model = load_model(args.model)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    stab = LetterStabilizer(hold_seconds=args.hold, min_conf=args.min_conf)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam số {args.camera} (thử --camera 1)")
    last, fps = time.time(), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # soi gương cho tự nhiên
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, conf = None, 0.0
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            lm = np.array([[p.x, p.y, p.z] for p in hl.landmark], np.float32)
            k, conf = predict_landmarks(model, lm)
            label = classes[k]
            mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)
        stab.update(label, conf)
        now = time.time(); fps = 0.9 * fps + 0.1 / max(now - last, 1e-6); last = now
        draw_ui(frame, label, conf, stab.progress(), stab.text, fps)
        cv2.imshow("ASL webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("c"):
            stab.clear()
        elif key == 8:
            stab.text = stab.text[:-1]
        elif key == ord(" "):
            stab.text += " "
        elif key == ord("s") and stab.text.strip():
            speak(stab.text)
    cap.release(); cv2.destroyAllWindows(); hands.close()


if __name__ == "__main__":
    main()
