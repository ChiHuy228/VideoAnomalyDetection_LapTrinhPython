#!/usr/bin/env python3
"""
<- Eval_Anomaly_Detector.m (MIL gốc).
AUC toàn cục: ánh xạ 32 điểm -> điểm từng khung hình theo số đặc trưng C3D (mỗi đặc trưng = 16 khung hình),
gộp mọi video, sort giảm dần + cumsum + trapz(fpr, rec) đúng như MATLAB (không xử lý đồng hạng).

Chạy:
  python scripts/mil_eval.py --results_dir results/mil_sultani/scores \
      --ann_txt data/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
      --videos data/UCF_Crime --counts_csv data/features_c3d/test_c3d_counts.csv \
      --out_dir results/mil_sultani/eval
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import scipy.io as sio

from mil_common import T_SEG, matlab_round

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")


def load_ann_txt(path):
    """Temporal_Anomaly_Annotation_for_Testing_Videos.txt: tên, lớp, s1 e1 s2 e2 (-1 = không có)."""
    ann = {}
    for line in open(path):
        t = line.split()
        if len(t) < 6:
            continue
        s1, e1, s2, e2 = (int(x) for x in t[-4:])
        ann[os.path.splitext(t[0])[0]] = [(s, e) for s, e in ((s1, e1), (s2, e2)) if s >= 0 and e >= 0]
    return ann


def frame_scores_matlab(pred32, n_c3d, actual_frames):
    """Ánh xạ 32 điểm -> điểm từng khung hình đúng như MATLAB (mỗi đặc trưng C3D = 16 khung hình)."""
    det = np.zeros(n_c3d * 16, np.float32)
    shots = matlab_round(np.linspace(1, n_c3d, T_SEG + 1)).astype(int)
    for i in range(T_SEG):
        ss, ee = shots[i], shots[i + 1] - 1
        if ee < ss:
            det[(ss - 1) * 16: (ss - 1) * 16 + 16] = pred32[i]
        else:
            det[(ss - 1) * 16: (ee - 1) * 16 + 16] = pred32[i]
    if actual_frames >= len(det):                                        # repmat(det(end), ...) cho đủ số khung hình
        return np.concatenate([det, np.full(actual_frames - len(det), det[-1], np.float32)])
    return det[:actual_frames]                                           # (MATLAB gốc sẽ báo lỗi chỗ này; ta cắt bớt)


def matlab_auc(scores, gt):
    """AUC kiểu MATLAB: sort giảm dần (ổn định), tp/fp cộng dồn, trapz(fpr, rec). Không thêm điểm (0,0)."""
    order = np.argsort(-scores, kind="stable")
    g = gt[order]
    tp = np.cumsum(g > 0)
    fp = np.cumsum(g == 0)
    rec = tp / gt.sum()
    fpr = fp / (gt == 0).sum()
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(rec, fpr))


def main():
    import cv2
    from sklearn.metrics import roc_auc_score

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--ann_txt", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--counts_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ann = load_ann_txt(args.ann_txt)
    counts = {r["video"]: int(r["num_c3d_feats"]) for r in csv.DictReader(open(args.counts_csv))}
    vids = {}
    for f in glob.glob(os.path.join(args.videos, "**", "*"), recursive=True):
        if f.lower().endswith(VIDEO_EXTS):
            vids.setdefault(os.path.splitext(os.path.basename(f))[0], f)

    all_s, all_y, used, skipped = [], [], [], []
    for mp in sorted(glob.glob(os.path.join(args.results_dir, "*.mat"))):
        name = os.path.basename(mp)
        for suf in ("_C.mat", ".mat"):
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        if name not in ann or name not in counts or name not in vids:
            skipped.append(name)
            continue
        cap = cv2.VideoCapture(vids[name])
        nf, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        actual = int(matlab_round(nf / fps * 30)) if fps and fps > 0 else int(nf)   # round(Duration*30) như bản gốc
        pred = np.ravel(sio.loadmat(mp)["predictions"]).astype(np.float32)
        if pred.size != T_SEG:
            skipped.append(name)
            continue
        s = frame_scores_matlab(pred, counts[name], actual)
        y = np.zeros(actual, np.uint8)
        for st, en in ann[name]:
            y[max(st, 1) - 1: min(en, actual)] = 1
        all_s.append(s)
        all_y.append(y)
        used.append(name)
    if not used:
        raise RuntimeError("Không video nào khớp giữa .mat, annotation, counts_csv và thư mục video")

    S, Y = np.concatenate(all_s), np.concatenate(all_y)
    res = {"auc_matlab_style": matlab_auc(S, Y), "auc_sklearn_tie_aware": float(roc_auc_score(Y, S)),
           "n_video_evaluated": len(used), "n_skipped": len(skipped), "skipped_examples": skipped[:5]}
    print(f"Số video đánh giá: {len(used)} (bỏ qua {len(skipped)})")
    print(f"AUC kiểu MATLAB (đúng bài báo): {res['auc_matlab_style']:.4f}")
    print(f"AUC sklearn (xử lý đồng hạng, để tham khảo): {res['auc_sklearn_tie_aware']:.4f}")
    json.dump(res, open(os.path.join(args.out_dir, "auc_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()