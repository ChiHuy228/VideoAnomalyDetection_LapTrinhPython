"""
Xuất điểm MIL (32 điểm/video) thành điểm theo từng frame, độ dài mỗi video = số đặc trưng C3D * 16,
nối theo thứ tự Anomaly_Test.txt, để so sánh cùng nhãn gt với RTFM/URDMU/VadCLIP.
Ngoài frame_scores.npy (mảng nối), còn ghi video_scores.npz (mỗi video một mảng, khóa = tên video).

Chạy (ở thư mục gốc HuanLuyen):
  python scripts/mil/export_mil_scores.py \
      --results_dir results/mil_sultani/scores \
      --test_txt data/annotations/Anomaly_Test.txt \
      --counts_csv data/features_c3d/test_c3d_counts.csv \
      --out results/mil_sultani/frame_scores.npy
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mil_common import T_SEG          # noqa: E402
from mil_eval import frame_scores_matlab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--test_txt", required=True)
    ap.add_argument("--counts_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt", default=None,
                    help="nếu đặt: đệm cuối (giá trị cuối) cho đủ độ dài gt (clip thừa nằm trong khối Normal)")
    ap.add_argument("--extra_json", default=None,
                    help="json từ find_gt_mismatch.py: video nào thừa 1 clip trong gt thì đệm thêm 16 frame cuối")
    a = ap.parse_args()

    counts = {r["video"]: int(r["num_c3d_feats"]) for r in csv.DictReader(open(a.counts_csv))}
    names = [os.path.splitext(os.path.basename(l.split()[0].replace("\\", "/")))[0]
             for l in open(a.test_txt, encoding="utf-8", errors="ignore") if l.strip()]

    extra = set(json.load(open(a.extra_json))["sure"]) if a.extra_json else set()
    out, missing = [], []
    per = {}                                                          # +
    for n in names:
        mp = next((os.path.join(a.results_dir, n + s) for s in ("_C.mat", ".mat")
                   if os.path.exists(os.path.join(a.results_dir, n + s))), None)
        if mp is None or n not in counts:
            missing.append(n)
            continue
        pred = np.ravel(sio.loadmat(mp)["predictions"]).astype(np.float32)
        if pred.size != T_SEG:
            missing.append(n)
            continue
        fs = frame_scores_matlab(pred, counts[n], counts[n] * 16)
        if n in extra:
            fs = np.concatenate([fs, np.full(16, fs[-1], np.float32)])
        out.append(fs)
        per[n] = fs.astype(np.float32)                                # + điểm theo frame của từng video
    if missing:
        print(f"Thiếu {len(missing)} video, ví dụ: {missing[:5]}")
    arr = np.concatenate(out)
    if a.gt:
        n_gt = len(np.load(a.gt).ravel())
        if len(arr) > n_gt or (n_gt - len(arr)) % 16:
            raise SystemExit(f"Độ dài lệch bất thường: pred={len(arr)} gt={n_gt}")
        if len(arr) < n_gt:
            print(f"Đệm {n_gt - len(arr)} frame cuối cho khớp gt")
            arr = np.pad(arr, (0, n_gt - len(arr)), mode="edge")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out, arr)
    vs = os.path.join(os.path.dirname(os.path.abspath(a.out)), "video_scores.npz")   # +
    np.savez(vs, **per)                                               # +
    print("Đã ghi", a.out, "| số frame =", len(arr))
    print("Đã ghi", vs, "| số video =", len(per))                     # +


if __name__ == "__main__":
    main()