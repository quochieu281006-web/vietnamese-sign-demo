"""Thu dữ liệu điểm mốc ký hiệu tiếng Việt (hình tay TĨNH) bằng webcam, mỗi người một file data/vsl/.

Chạy:  .venv\\Scripts\\python.exe src\\collect_vsl_landmarks.py --person huong --hand right --labels oo ow uw
Quy trình mỗi LẦN QUAY: SPACE -> đếm ngược 2 giây (đưa tay vào dáng đúng) -> ghi tự động
`--take-seconds` giây (mặc định 3, ~35 mẫu) -> tự lưu thành 1 take. Quay đủ `--takes` lần mỗi nhãn.
Phím:  SPACE = quay 1 lần | U = xóa lần quay gần nhất của nhãn này | N / B = nhãn kế / trước | Q = thoát
Chỉ lưu 21 điểm mốc (không lưu ảnh).
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from vsl_dataset import DEFAULT_LABELS, append_samples, count_takes, delete_last_take, load_person

INTERVAL, COUNTDOWN, MIN_FRACTION = 0.08, 2.0, 0.6
GREEN, RED, WHITE, GRAY, YELLOW = (80, 220, 100), (60, 60, 230), (255, 255, 255), (170, 170, 170), (0, 220, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, help="tên/biệt danh người ra dấu (mỗi người 1 file)")
    ap.add_argument("--hand", choices=["right", "left"], default="right", help="tay đang dùng ra dấu")
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--takes", type=int, default=10, help="số lần quay mục tiêu mỗi nhãn")
    ap.add_argument("--take-seconds", type=float, default=3.0, help="độ dài mỗi lần quay (giây)")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    import mediapipe as mp
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                     min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được webcam số {args.camera} (thử --camera 1)")

    expected = int(args.take_seconds / INTERVAL)
    idx, state, t0, last_t, buf, msg, msg_t = 0, "idle", 0.0, 0.0, [], "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        label, now = args.labels[idx], time.time()
        seen = bool(res.multi_hand_landmarks)
        if seen:
            hl = res.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(frame, hl, mp.solutions.hands.HAND_CONNECTIONS)

        if state == "countdown" and now - t0 >= COUNTDOWN:
            state, t0, buf = "recording", now, []
        elif state == "recording":
            if seen and now - last_t >= INTERVAL:
                buf.append(np.array([[p.x, p.y, p.z] for p in hl.landmark], np.float32)); last_t = now
            if now - t0 >= args.take_seconds:
                if len(buf) >= MIN_FRACTION * expected:
                    append_samples(args.person, label, args.hand, buf)
                    msg = f"DA LUU lan quay ({len(buf)} mau)"
                else:
                    msg = f"BO lan quay: chi thay tay {len(buf)}/{expected} mau, quay lai"
                state, msg_t = "idle", now

        takes = count_takes(args.person, label)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.putText(frame, f"NHAN: {label.upper()}  ({idx + 1}/{len(args.labels)})", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GREEN, 3)
        cv2.putText(frame, f"{args.person} / tay {args.hand} | lan quay: {takes}/{args.takes}",
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        if state == "countdown":
            cv2.putText(frame, f"CHUAN BI... {COUNTDOWN - (now - t0):.0f}", (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, YELLOW, 4)
        elif state == "recording":
            cv2.putText(frame, f"DANG GHI {args.take_seconds - (now - t0):.1f}s", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3)
        else:
            cv2.putText(frame, "SAN SANG" + ("" if seen else "  - KHONG THAY TAY"), (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, GRAY, 2)
        if msg and now - msg_t < 2.5:
            cv2.putText(frame, msg, (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        cv2.putText(frame, "SPACE quay | U xoa lan cuoi | N tiep | B lui | Q thoat", (15, h - 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
        cv2.imshow("Thu du lieu VSL", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and state == "idle":
            state, t0 = "countdown", now
        elif key in (ord("n"), ord("b")) and state == "idle":
            idx = (idx + (1 if key == ord("n") else -1)) % len(args.labels)
        elif key == ord("u") and state == "idle":
            n = delete_last_take(args.person, label)
            msg, msg_t = f"Da xoa lan quay cuoi ({n} mau)", now
        elif key in (ord("q"), 27):
            break
    cap.release(); cv2.destroyAllWindows(); hands.close()
    print("Đã lưu. Xem tổng quan: .venv\\Scripts\\python.exe src\\vsl_dataset.py")


if __name__ == "__main__":
    main()
