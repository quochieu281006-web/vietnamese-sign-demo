"""Thu CLIP chuyển động (dấu thanh) bằng webcam: mỗi clip là một lần quay, lưu vào data/vsl_clips/.

Chạy:  .venv\\Scripts\\python.exe src\\collect_vsl_clips.py --person huong --hand right
Mỗi clip: SPACE -> đếm ngược 2 giây (đưa tay vào tư thế bắt đầu) -> ghi `--clip-seconds` giây (mặc định 2)
-> tự lưu. Làm trọn một lần dấu thanh trong khoảng ghi, bắt đầu ngay sau khi hiện "GHI".
Nhãn `none` = KHÔNG làm dấu thanh (giữ yên tay hoặc đưa tay vào/ra) để mô hình biết "không có dấu".
Phím:  SPACE = quay 1 clip | U = xóa clip cuối của nhãn | N / B = nhãn kế / trước | Q = thoát
Chỉ lưu 21 điểm mốc (không lưu ảnh).
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from vsl_clip_dataset import (DEFAULT_LABELS, MIN_HAND_SCALE, append_clip, count_clips, delete_last_clip,
                              hand_scale, resample_clip)

COUNTDOWN = 2.0
GREEN, RED, WHITE, GRAY, YELLOW = (80, 220, 100), (60, 60, 230), (255, 255, 255), (170, 170, 170), (0, 220, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True)
    ap.add_argument("--hand", choices=["right", "left"], default="right")
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--clips", type=int, default=40, help="số clip mục tiêu mỗi nhãn")
    ap.add_argument("--clip-seconds", type=float, default=2.0)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam số {args.camera} (thử --camera 1)")

    idx, state, t0, frames, msg, msg_t = 0, "idle", 0.0, [], "", 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, now = args.labels[idx], time.time()
        lm = None
        if res.multi_hand_landmarks:
            hl = res.multi_hand_landmarks[0]
            lm = np.array([[p.x, p.y, p.z] for p in hl.landmark], np.float32)
            mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)

        if state == "countdown" and now - t0 >= COUNTDOWN:
            state, t0, frames = "recording", now, []
        elif state == "recording":
            frames.append(lm)
            if now - t0 >= args.clip_seconds:
                clip = resample_clip(frames)
                if clip is None:
                    msg = "BO clip: mat tay qua nhieu, quay lai"
                elif hand_scale(clip) < MIN_HAND_SCALE:
                    msg = "BO clip: tay qua nho, ngoi gan camera hon"
                else:
                    append_clip(args.person, label, args.hand, clip)
                    msg = f"DA LUU clip ({len(frames)} khung goc)"
                state, msg_t = "idle", now

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.putText(frame, f"DAU: {label.upper()}  ({idx + 1}/{len(args.labels)})", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 3)
        cv2.putText(frame, f"{args.person} / tay {args.hand} | clip: {count_clips(args.person, label)}/{args.clips}",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        if state == "countdown":
            cv2.putText(frame, f"CHUAN BI... {COUNTDOWN - (now - t0):.0f}", (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, YELLOW, 4)
        elif state == "recording":
            cv2.putText(frame, f"GHI! {args.clip_seconds - (now - t0):.1f}s", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 3)
        else:
            cv2.putText(frame, "SAN SANG" + ("" if lm is not None else "  - KHONG THAY TAY"), (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GRAY, 2)
        if msg and now - msg_t < 2.5:
            cv2.putText(frame, msg, (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        cv2.putText(frame, "SPACE quay | U xoa clip cuoi | N tiep | B lui | Q thoat", (15, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        cv2.imshow("Thu clip dau thanh VSL", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and state == "idle":
            state, t0 = "countdown", now
        elif key in (ord("n"), ord("b")) and state == "idle":
            idx = (idx + (1 if key == ord("n") else -1)) % len(args.labels)
        elif key == ord("u") and state == "idle":
            msg, msg_t = f"Da xoa {delete_last_clip(args.person, label)} clip cuoi", now
        elif key in (ord("q"), 27):
            break
    cap.release(); cv2.destroyAllWindows(); hands.close()
    print("Đã lưu. Xem tổng quan: .venv\\Scripts\\python.exe src\\vsl_clip_dataset.py")


if __name__ == "__main__":
    main()
