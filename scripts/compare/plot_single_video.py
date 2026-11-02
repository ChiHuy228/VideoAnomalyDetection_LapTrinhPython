"""
Vẽ điểm dự đoán theo frame của các phương pháp trên MỘT video, tô vùng Ground Truth bất thường.
Đọc dữ liệu giống calc_auc.py: results/<method>/video_scores.npz, GT từ gt-ucf.npy,
URDMU làm tham chiếu độ dài/thứ tự video.

Chạy từ thư mục HuanLuyen:
    python scripts/compare/plot_single_video.py --video_name Arrest001
    python scripts/compare/plot_single_video.py --video_index 5
    python scripts/compare/plot_single_video.py --list          # in danh sách video
"""
import os, re, argparse
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Cấu hình (giống calc_auc.py) ----------------
ALL_METHODS = {'MIL':     'results/mil_sultani/video_scores.npz',
               'RTFM':    'results/rtfm/video_scores.npz',
               'URDMU':   'results/urdmu/video_scores.npz',
               'VadCLIP': 'results/vadclip/video_scores.npz'}
GT      = 'scripts/urdmu/frame_label/gt-ucf.npy'
OUT_DIR = 'results/compare/video_plots'
COLORS = {'MIL': '#7f7f7f', 'RTFM': '#2ca02c', 'URDMU': '#1f4e9c', 'VadCLIP': "#030303"}
TEXT = {
    'vi': {'gt': 'Ground Truth (bất thường)',
           'title': 'Điểm dự đoán theo frame: %s (index %d)',
           'xlabel': 'Frame', 'ylabel': 'Điểm bất thường'},
    'en': {'gt': 'Ground Truth (anomaly)',
           'title': 'Frame-level anomaly scores: %s (index %d)',
           'xlabel': 'Frame', 'ylabel': 'Anomaly score'},
}

norm = lambda k: re.sub(r'(_x264)?(_i3d)?$', '', k)


def resample(s, L):
    s = np.asarray(s, dtype=np.float64)
    if len(s) == L: return s
    if len(s) == 1: return np.full(L, s[0])
    return np.interp(np.linspace(0, len(s) - 1, L), np.arange(len(s)), s)


def runs(y):
    """Các đoạn [start, end) liên tiếp có nhãn bất thường (y == 1)."""
    d = np.diff(np.concatenate([[0], (np.asarray(y) > 0.5).astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_name', default='', help='một phần tên video, vd: Arrest001')
    ap.add_argument('--video_index', type=int, default=10, help='thứ tự video (0..289) theo URDMU')
    ap.add_argument('--list', action='store_true', help='in danh sách video rồi thoát')
    args = ap.parse_args()

    methods = {m: p for m, p in ALL_METHODS.items() if os.path.exists(p)}
    for m in ALL_METHODS:
        if m not in methods:
            print('Bỏ qua (chưa có file):', m)
    assert 'URDMU' in methods, 'Cần có URDMU làm tham chiếu độ dài'

    S = {m: {norm(k): v for k, v in np.load(p).items()} for m, p in methods.items()}
    ref = S['URDMU']
    names = list(ref)

    if args.list:
        for i, n in enumerate(names):
            print(i, n)
        return

    # ---------- Chọn video ----------
    idx = args.video_index
    if args.video_name:
        key = norm(args.video_name).lower()
        exact = [i for i, n in enumerate(names) if n.lower() == key]
        part = [i for i, n in enumerate(names) if key in n.lower()]
        hit = exact or part
        if hit:
            idx = hit[0]
            if len(hit) > 1:
                print('Có %d video khớp, dùng cái đầu: %s' % (len(hit), [names[i] for i in hit[:5]]))
        else:
            print("Không thấy video chứa '%s', dùng index %d" % (args.video_name, idx))
    assert 0 <= idx < len(names), 'index ngoài khoảng 0..%d' % (len(names) - 1)
    vname = names[idx]

    # ---------- GT theo video (giống calc_auc.py) ----------
    gt = np.load(GT).ravel()
    start = sum(len(ref[n]) for n in names[:idx])
    L = len(ref[vname])
    y = gt[start:start + L]
    assert len(y) == L, 'GT ngắn hơn dự kiến, kiểm tra GT path'

    # ---------- Điểm từng phương pháp ----------
    P = {}
    for m in methods:
        if vname not in S[m]:
            print('!!! %s thiếu video %s -> bỏ qua' % (m, vname))
            continue
        P[m] = resample(S[m][vname], L)

       # ---------- Vẽ (xuất 2 ảnh: tiếng Việt và tiếng Anh) ----------
    os.makedirs(OUT_DIR, exist_ok=True)
    x = np.arange(L)
    has_both = len(np.unique(y)) == 2

    for lang in ('vi', 'en'):
        T = TEXT[lang]
        plt.figure(figsize=(13, 5.5), dpi=150)

        for k, (s, e) in enumerate(runs(y)):
            plt.axvspan(s, e, color='orange', alpha=0.25, label=T['gt'] if k == 0 else None)

        for m, s in P.items():
            lab = m + (' (AUC %.1f)' % (roc_auc_score(y, s) * 100) if has_both else '')
            plt.plot(x, s, label=lab, color=COLORS.get(m, 'black'), linewidth=1.8, alpha=0.9)

        plt.title(T['title'] % (vname, idx), fontsize=13, fontweight='bold')
        plt.xlabel(T['xlabel'], fontsize=11, fontweight='bold')
        plt.ylabel(T['ylabel'], fontsize=11, fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   ncol=len(P) + 1, fontsize=10, frameon=False)
        plt.tight_layout()

        out = os.path.join(OUT_DIR, '%s_comparison_%s.png' % (vname, lang))
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        print('Đã lưu:', out)


if __name__ == '__main__':
    main()