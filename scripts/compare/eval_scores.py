"""
Đánh giá một hoặc nhiều file điểm theo frame (.npy) bằng các chỉ số tách được "tách video" và "định vị".

  python scripts/compare/eval_scores.py \
      --gt scripts/vadclip/list/gt_ucf.npy \
      --lens results/rtfm/gt_lens.npy \
      --scores results/rtfm/frame_scores.npy results/rtfm/frame_scores_scale7.npy

--lens là số CLIP theo gt của từng video (gt_lens.npy), mỗi clip = 16 frame.
"""
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    a = ap.parse_args()

    gt = np.load(a.gt).ravel()
    lens = np.load(a.lens).ravel().astype(int) * 16
    b = np.r_[0, np.cumsum(lens)]
    nv = len(lens)
    if b[-1] != len(gt):
        raise SystemExit(f"Tổng frame theo lens ({b[-1]}) khác gt ({len(gt)})")
    vlabel = np.array([gt[b[i]:b[i + 1]].max() for i in range(nv)])
    print(f"{nv} video: {int(vlabel.sum())} bất thường, {int((1 - vlabel).sum())} bình thường\n")

    for path in a.scores:
        s = np.load(path).ravel().astype(np.float64)
        if len(s) != len(gt):
            print(f"{path}: độ dài {len(s)} khác gt {len(gt)}, bỏ qua")
            continue
        vmean = np.array([s[b[i]:b[i + 1]].mean() for i in range(nv)])
        within = []
        s_centered = s.copy()
        for i in range(nv):
            seg, g = s[b[i]:b[i + 1]], gt[b[i]:b[i + 1]]
            s_centered[b[i]:b[i + 1]] = seg - seg.mean()
            if 0 < g.sum() < len(g):
                within.append(roc_auc_score(g, seg))
        c0 = np.array([s[b[i]] for i in range(nv)])
        c1 = np.array([s[b[i] + 16] for i in range(nv) if lens[i] > 16])
        print(f"== {path}")
        print(f"  AUC mức frame                         : {roc_auc_score(gt, s):.4f}")
        print(f"  AUC mức video (điểm trung bình video) : {roc_auc_score(vlabel, vmean):.4f}")
        print(f"  AUC trong từng video (TB, bất thường) : {np.mean(within):.4f}   (0.5 = không định vị)")
        print(f"  AUC frame sau khi trừ TB mỗi video    : {roc_auc_score(gt, s_centered):.4f}")
        print(f"  Điểm TB video bình thường / bất thường: {vmean[vlabel == 0].mean():.3f} / {vmean[vlabel == 1].mean():.3f}")
        print(f"  Điểm TB clip 0 / clip 1               : {c0.mean():.3f} / {c1.mean():.3f}")
        print()


if __name__ == "__main__":
    main()