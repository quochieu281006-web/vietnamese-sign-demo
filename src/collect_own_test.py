"""Chụp nhanh ảnh tự thu bằng webcam cho tập test độc lập (data/own_test/<chữ cái>/*.jpg).

Chạy:  python src/collect_own_test.py --letters A B C --per-letter 20
Phím:  SPACE = chụp 1 ảnh, N = chuyển chữ tiếp theo, Q = thoát.
"""
import argparse
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parents[1] / "data" / "own_test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", nargs="+", default=list("ABCDEFGHIKLMNOPQRSTUVWXY"))
    ap.add_argument("--per-letter", type=int, default=20)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Không mở được webcam")
    for letter in args.letters:
        folder = OUT / letter
        folder.mkdir(parents=True, exist_ok=True)
        n = len(list(folder.glob("*.jpg")))
        while n < args.per_letter:
            ok, frame = cap.read()
            if not ok:
                break
            view = frame.copy()
            cv2.putText(view, f"Chu: {letter}  {n}/{args.per_letter}  [SPACE chup, N bo qua, Q thoat]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("collect", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                cv2.imwrite(str(folder / f"{letter}_{n:03d}.jpg"), frame)
                n += 1
            elif key == ord("n"):
                break
            elif key == ord("q"):
                cap.release(); cv2.destroyAllWindows(); return
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
