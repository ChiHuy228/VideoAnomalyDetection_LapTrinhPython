"""
Xuất ROC riêng cho MIL (mặc định): ảnh ROC (vi/en) + bảng CSV (fpr, tpr, threshold).
Đọc dữ liệu giống calc_auc.py. Cần URDMU làm tham chiếu thứ tự/độ dài video.

Chạy từ thư mục HuanLuyen:
    python scripts/compare/roc_mil.py                          # ROC toàn bộ tập test
    python scripts/compare/roc_mil.py --video_name Arrest001   # thêm ROC của riêng video đó
    python scripts/compare/roc_mil.py --method VadCLIP         # đổi phương pháp
"""
import os, re, csv, argparse
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALL_METHODS = {'MIL':     'results/mil_sultani/video_scores.npz',
               'RTFM':    'results/rtfm/video_scores.npz',
               'URDMU':   'results/urdmu/video_scores.npz',
               'VadCLIP': 'results/vadclip/video_scores.npz'}
GT      = 'scripts/urdmu/frame_label/gt-ucf.npy'
OUT_DIR = 'results/compare/roc'
COLORS  = {'MIL': '#7f7f7f', 'RTFM': '#2ca02c', 'URDMU': '#1f4e9c', 'VadCLIP': '#030303'}
TEXT = {
    'vi': {'x': 'Tỉ lệ dương giả (FPR)', 'y': 'Tỉ lệ dương thật (TPR)', 'title': 'Đường ROC - %s (%s)'},
    'en': {'x': 'False Positive Rate',   'y': 'True Positive Rate',     'title': 'ROC curve - %s (%s)'},
}

norm = lambda k: re.sub(r'(_x264)?(_i3d)?$', '', k)


def resample(s, L):
    s = np.asarray(s, dtype=np.float64)
    if len(s) == L: return s
    if len(s) == 1: return np.full(L, s[0])
    return np.interp(np.linspace(0, len(s) - 1, L), np.arange(len(s)), s)


def load(path):
    return {norm(k): v for k, v in np.load(path).items()}


def export_roc(y, p, m, tag, desc):
    """Lưu CSV + ảnh ROC (vi, en). Trả về AUC."""
    fpr, tpr, thr = roc_curve(y, p)
    a = auc(fpr, tpr)

    csv_path = os.path.join(OUT_DIR, 'roc_%s_%s.csv' % (m.lower(), tag))
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['fpr', 'tpr', 'threshold'])
        for a_, b_, c_ in zip(fpr, tpr, thr):
            w.writerow(['%.6f' % a_, '%.6f' % b_, '%.6f' % c_])
    print('Đã lưu:', csv_path, '(%d điểm)' % len(fpr))

    for lang in ('vi', 'en'):
        T = TEXT[lang]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.plot(fpr, tpr, color=COLORS.get(m, 'black'), lw=2.2, label='%s (AUC %.1f)' % (m, a * 100))
        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
        ax.set_xlabel(T['x']); ax.set_ylabel(T['y'])
        ax.set_title(T['title'] % (m, desc), fontsize=11, fontweight='bold')
        ax.legend(loc='lower right'); ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, 'roc_%s_%s_%s.png' % (m.lower(), tag, lang))
        plt.savefig(out, dpi=200); plt.close()
        print('Đã lưu:', out)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='MIL', choices=list(ALL_METHODS))
    ap.add_argument('--video_name', default='', help='nếu có, vẽ thêm ROC riêng cho video này')
    ap.add_argument('--video_index', type=int, default=-1, help='hoặc chọn video theo index (0..289)')
    args = ap.parse_args()
    m = args.method

    for need in {'URDMU', m}:
        assert os.path.exists(ALL_METHODS[need]), 'Thiếu file: ' + ALL_METHODS[need]
    os.makedirs(OUT_DIR, exist_ok=True)

    ref = load(ALL_METHODS['URDMU'])
    S = ref if m == 'URDMU' else load(ALL_METHODS[m])
    names = list(ref)
    miss = [n for n in names if n not in S]
    assert not miss, '%s thiếu %d video (vd: %s)' % (m, len(miss), miss[:3])

    gt = np.load(GT).ravel()
    off, pos = {}, 0
    for n in names:
        off[n] = pos; pos += len(ref[n])
    assert pos == len(gt), (pos, len(gt))

    # ---- ROC toàn bộ tập test ----
    y = np.concatenate([gt[off[n]:off[n] + len(ref[n])] for n in names])
    p = np.concatenate([resample(S[n], len(ref[n])) for n in names])
    a = export_roc(y, p, m, 'all', 'toàn bộ test / all test videos')
    print('AUC toàn bộ: %.2f' % (a * 100))

    # ---- ROC riêng một video (tuỳ chọn) ----
    idx = args.video_index
    if args.video_name:
        key = norm(args.video_name).lower()
        hit = [i for i, n in enumerate(names) if n.lower() == key] or \
              [i for i, n in enumerate(names) if key in n.lower()]
        if not hit:
            print("Không thấy video chứa '%s'" % args.video_name)
            return
        idx = hit[0]
    if idx >= 0:
        assert idx < len(names), 'index ngoài khoảng 0..%d' % (len(names) - 1)
        v = names[idx]
        yv = gt[off[v]:off[v] + len(ref[v])]
        if len(np.unique(yv)) < 2:
            print('Video %s chỉ có một lớp (toàn bình thường hoặc toàn bất thường) nên không vẽ được ROC.' % v)
            return
        pv = resample(S[v], len(yv))
        av = export_roc(yv, pv, m, v, v)
        print('AUC video %s: %.2f' % (v, av * 100))


if __name__ == '__main__':
    main()