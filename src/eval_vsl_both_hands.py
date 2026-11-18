"""Kiểm tra từng chữ / dấu chữ có nhận được ở CẢ HAI TAY không (tay trái mô phỏng bằng lật gương điểm mốc).

So sánh: (A) huấn luyện chỉ trên tay phải  vs  (B) huấn luyện thêm bản lật gương (cả hai tay).
Chia 5 fold theo nhóm (lần quay / mẫu ASL), bản lật luôn cùng fold với bản gốc -> không rò rỉ.
Chạy:  .venv\\Scripts\\python.exe src\\eval_vsl_both_hands.py   ->  reports/vsl_both_hands.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from train_vsl_unified_static import ROOT, load_training_landmarks, make_model, mirror
from vsl_models import static_features


def main():
    Xr, y, g = load_training_landmarks()
    F_r, F_l = static_features(Xr), static_features(mirror(Xr))
    pred = {k: np.empty(len(y), dtype=y.dtype) for k in ("A_phai", "A_trai", "B_phai", "B_trai")}
    for tr, te in GroupKFold(5).split(F_r, y, g):
        a = make_model().fit(F_r[tr], y[tr])                                   # chỉ tay phải
        b = make_model().fit(np.concatenate([F_r[tr], F_l[tr]]), np.concatenate([y[tr], y[tr]]))
        pred["A_phai"][te], pred["A_trai"][te] = a.predict(F_r[te]), a.predict(F_l[te])
        pred["B_phai"][te], pred["B_trai"][te] = b.predict(F_r[te]), b.predict(F_l[te])
    rows = [{"nhan": c, "n": int((y == c).sum()), **{k: round(float((p[y == c] == c).mean()), 3) for k, p in pred.items()}}
            for c in sorted(set(y))]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports" / "vsl_both_hands.csv", index=False)
    print(df.to_string(index=False))
    print("\nTONG  ", {k: round(float((p == y).mean()), 3) for k, p in pred.items()})
    print("Nhan nhan kem tay trai khi chi hoc tay phai (A_trai < 0.9):", list(df[df.A_trai < 0.9].nhan))
    print("Nhan van kem sau khi hoc ca hai tay (B_trai < 0.9):", list(df[df.B_trai < 0.9].nhan))


if __name__ == "__main__":
    main()
