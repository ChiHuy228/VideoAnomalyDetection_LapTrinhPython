"""
evt_analysis.py - phân tích kết quả của evtmil_suite.py (chỉ numpy/sklearn/pandas/matplotlib).
  Bench          : tải thứ tự test + nhãn frame, tính AUC / AP / FAR@TPR / AUC-chỉ-video-bất-thường
  summary_table  : trung bình ± độ lệch chuẩn qua các seed cho từng cấu hình
  paired_bootstrap : khoảng tin cậy 95% cho hiệu số (EVT-MIL - phương pháp tham chiếu), lấy mẫu lại theo video
  per_class      : AUC theo lớp bất thường (cộng các video Normal, như calc_auc.py)
  plot_video     : đồ thị điểm theo frame của một video (vd Arson035)

Quy ước dữ liệu xuyên suốt file:
  - "pv" (per-video) = danh sách các mảng điểm, mỗi video một mảng (độ dài = số frame của video),
    sắp theo đúng thứ tự bench.order. Ghép (concatenate) các mảng này sẽ khớp từng frame với bench.gt.
"""
import os, re, glob, json
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')   # backend không cần màn hình, chỉ lưu ảnh
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

# Bỏ hậu tố '_x264' / '_i3d' để tên video của các phương pháp khác nhau trùng khớp
norm_name = lambda k: re.sub(r'(_x264)?(_i3d)?$', '', k)
# Lớp của video: 'Normal' nếu tên bắt đầu bằng Normal, ngược lại bỏ phần số (Arson011 -> Arson)
cls_of = lambda n: 'Normal' if n.startswith('Normal') else re.sub(r'\d+.*$', '', n)


def video_name(path):
    """Lấy tên video chuẩn hóa từ một đường dẫn/tên file trong test_csv
    (bỏ thư mục, đuôi file, hậu tố '__<số>' như '__5' của crop/augment, và '_x264'/'_i3d')."""
    s = os.path.splitext(os.path.basename(path))[0]
    return norm_name(re.sub(r'__\d+$', '', s))


def resample(s, L):
    """Nội suy tuyến tính mảng điểm s về độ dài L (dùng khi độ dài của baseline lệch nhẹ so với chuẩn)."""
    s = np.asarray(s, dtype=np.float64)
    if len(s) == L: return s                              # đã đúng độ dài, giữ nguyên
    if len(s) == 1: return np.full(L, s[0])               # chỉ có 1 điểm: lặp lại cho đủ L
    return np.interp(np.linspace(0, len(s) - 1, L), np.arange(len(s)), s)


def ap_urdmu(y, p):
    """Average Precision tính bằng diện tích dưới đường PR (trapezoid)."""
    pr, rc, _ = precision_recall_curve(y, p)
    return auc(rc, pr)                                  # cùng công thức với ucf_test.py của URDMU


class Bench:
    """Bộ đánh giá chung: giữ thứ tự video test, độ dài, nhãn GT của từng video,
    để mọi phương pháp được chấm trên cùng một lưới frame."""

    def __init__(self, test_csv, gt_path, any_scores_npz):
        d = pd.read_csv(test_csv)
        # Thứ tự video test (cột đầu của csv), bỏ trùng nhưng giữ thứ tự xuất hiện
        self.order = list(dict.fromkeys(video_name(p) for p in d.iloc[:, 0]))
        z = np.load(any_scores_npz)
        # Độ dài (frame) mỗi video = số điểm theo clip trong file điểm bất kỳ * 16 frame/clip
        self.L = [len(z[n]) * 16 for n in self.order]
        self.gt = np.load(gt_path).ravel()
        # Kiểm tra: tổng frame các video phải bằng độ dài nhãn GT, nếu lệch là sai thứ tự/độ dài
        assert sum(self.L) == len(self.gt), 'tổng frame %d != len(gt) %d' % (sum(self.L), len(self.gt))
        off = np.concatenate([[0], np.cumsum(self.L)])      # mốc bắt đầu của từng video trong gt
        self.gt_v = [self.gt[off[i]:off[i + 1]] for i in range(len(self.order))]   # GT cắt theo video
        self.normal = np.array([n.startswith('Normal') for n in self.order])        # cờ video Normal
        self.cls = [cls_of(n) for n in self.order]                                  # lớp của từng video

    # ---- điểm theo video (frame) ----
    def from_segments(self, npz_path):
        """Đọc điểm theo CLIP (mỗi clip 16 frame) của EVT-MIL, lặp mỗi điểm 16 lần để thành điểm theo frame."""
        z = np.load(npz_path)
        return [np.repeat(z[n], 16).astype(np.float64) for n in self.order]

    def from_frames(self, npz_path):                    # file video_scores.npz của các baseline
        """Đọc điểm theo FRAME của các baseline (MIL/URDMU/VadCLIP...), nội suy về đúng độ dài L."""
        z = {norm_name(k): v for k, v in np.load(npz_path).items()}
        return [resample(z[n], L) for n, L in zip(self.order, self.L)]

    # ---- chỉ số ----
    def metrics(self, pv, tpr_target=0.5):
        """Tính các chỉ số trên toàn bộ frame:
          AUC       : AUC ghép mọi frame của mọi video (micro, giống bài báo)
          AP        : average precision
          FAR_TPR50 : tỉ lệ báo động nhầm trên video Normal tại ngưỡng cho TPR >= tpr_target
          AUC_abn   : AUC chỉ trên các video bất thường (bỏ video Normal)
        """
        p, y = np.concatenate(pv), self.gt
        fpr, tpr, thr = roc_curve(y, p)
        # Ngưỡng nhỏ nhất đạt TPR >= tpr_target; rồi đo FAR trên video Normal ở ngưỡng đó
        t = thr[np.searchsorted(tpr, tpr_target)]
        pn = np.concatenate([a for a, m in zip(pv, self.normal) if m])
        ab = [i for i in range(len(pv)) if not self.normal[i]]   # chỉ số các video bất thường
        return dict(AUC=100 * auc(fpr, tpr), AP=100 * ap_urdmu(y, p), FAR_TPR50=100 * float((pn >= t).mean()),
                    AUC_abn=100 * roc_auc_score(np.concatenate([self.gt_v[i] for i in ab]),
                                                np.concatenate([pv[i] for i in ab])))

    def per_class(self, pv):
        """AUC theo từng lớp bất thường; mỗi lớp được gộp thêm toàn bộ video Normal (như calc_auc.py).
        Lớp nào không có đủ cả frame bình thường và bất thường thì bị bỏ qua."""
        out = {}
        for c in sorted(set(self.cls) - {'Normal'}):
            idx = [i for i, k in enumerate(self.cls) if k in (c, 'Normal')]
            y = np.concatenate([self.gt_v[i] for i in idx])
            if len(set(y)) == 2:                          # cần cả nhãn 0 và 1 mới tính được AUC
                out[c] = 100 * roc_auc_score(y, np.concatenate([pv[i] for i in idx]))
        return out


def find_runs(out_dir):
    """Trả về danh sách (cấu hình, seed, thư mục) từ <out_dir>/<cfg>_s<seed>/summary.json."""
    runs = []
    for d in sorted(glob.glob(os.path.join(out_dir, '*_s*'))):
        m = re.match(r'(.+)_s(\d+)$', os.path.basename(d))    # tách 'tên_cấu_hình' và 'số seed'
        if m and os.path.exists(os.path.join(d, 'summary.json')):   # chỉ nhận lần chạy đã hoàn tất
            runs.append((m.group(1), int(m.group(2)), d))
    return runs


def evaluate_all(bench, out_dir, ckpts=('best', 'last'), verbose=True):
    """Chấm mọi lần chạy (cấu hình x seed x loại điểm x checkpoint) và gom thành một DataFrame.
    File điểm có dạng scores_<loại điểm>_<best|last>.npz; 'best' là checkpoint có val_auc tốt nhất,
    'last' là checkpoint cuối cùng."""
    rows = []
    for name, seed, d in find_runs(out_dir):
        info = json.load(open(os.path.join(d, 'summary.json')))
        for f in sorted(glob.glob(os.path.join(d, 'scores_*_*.npz'))):
            m = re.match(r'scores_(\w+?)_(best|last)\.npz$', os.path.basename(f))
            if not m or m.group(2) not in ckpts:          # bỏ file không đúng mẫu hoặc checkpoint không yêu cầu
                continue
            r = dict(config=name, score=m.group(1), ckpt=m.group(2), seed=seed,
                     val_auc=info['best_val'] if m.group(2) == 'best' else info['last_val'])
            r.update(bench.metrics(bench.from_segments(f)))   # thêm AUC, AP, FAR_TPR50, AUC_abn
            rows.append(r)
            if verbose:
                print('%-10s s%-4d %s/%s  AUC %.2f  AP %.2f' % (name, seed, r['score'], r['ckpt'], r['AUC'], r['AP']), flush=True)
    return pd.DataFrame(rows)


def summary_table(df, metrics=('AUC', 'AP', 'FAR_TPR50', 'AUC_abn', 'val_auc')):
    """Bảng tóm tắt: với mỗi (cấu hình, loại điểm, checkpoint) in 'trung bình ± độ lệch chuẩn' qua các seed.
    Chỉ có 1 seed thì độ lệch chuẩn là NaN và được in là 0."""
    g = df.groupby(['config', 'score', 'ckpt'])
    out = g[list(metrics)].agg(['mean', 'std']).round(2)
    out['n_seed'] = g.size()                              # số seed của từng nhóm
    fmt = pd.DataFrame(index=out.index)
    for m in metrics:
        fmt[m] = [('%.2f ± %.2f' % (a, 0 if np.isnan(b) else b)) for a, b in zip(out[(m, 'mean')], out[(m, 'std')])]
    fmt['n_seed'] = out['n_seed']
    return fmt


def paired_bootstrap(bench, a_pvs, b_pvs, n_boot=300, stride=8, seed=0):
    """Hiệu số  mean_seed(A) - mean_seed(B)  lấy mẫu lại theo VIDEO (A, B: danh sách các lần chạy, mỗi lần là danh sách điểm theo video).
    B có thể chỉ là một lần chạy (vd VadCLIP: b_pvs=[ref_pv]).
    Để nhanh, mỗi video giữ 1 frame trong mỗi `stride` frame; ước lượng điểm vẫn tính trên toàn bộ frame ở nơi khác.

    Trả về, cho AUC và AP: hiệu số trung bình, khoảng tin cậy 95% (2.5%-97.5%),
    tỉ lệ lần lấy mẫu mà A > B (frac_positive) và số lần lấy mẫu hợp lệ n."""
    rng = np.random.default_rng(seed)
    # Thưa hóa: giữ 1 frame mỗi `stride` frame để bootstrap chạy nhanh
    y = [g[::stride] for g in bench.gt_v]
    A = [[p[::stride] for p in pv] for pv in a_pvs]
    Bs = [[p[::stride] for p in pv] for pv in b_pvs]
    n = len(y)
    res = {'AUC': [], 'AP': []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)                       # chọn lại n video có hoàn lại; A và B dùng CÙNG idx (ghép cặp)
        yy = np.concatenate([y[i] for i in idx])
        if len(set(yy)) < 2:                              # mẫu chỉ có một loại nhãn thì không tính được AUC
            continue
        def mean_metrics(runs):
            """Trung bình AUC/AP qua các lần chạy (seed) trên cùng mẫu video idx."""
            au, apv = [], []
            for pv in runs:
                pp = np.concatenate([pv[i] for i in idx])
                au.append(roc_auc_score(yy, pp)); apv.append(ap_urdmu(yy, pp))
            return np.mean(au), np.mean(apv)
        a_auc, a_ap = mean_metrics(A)
        b_auc, b_ap = mean_metrics(Bs)
        res['AUC'].append(100 * (a_auc - b_auc)); res['AP'].append(100 * (a_ap - b_ap))
    out = {}
    for k, v in res.items():
        v = np.array(v)
        out[k] = dict(diff_mean=float(v.mean()), ci95_low=float(np.percentile(v, 2.5)),
                      ci95_high=float(np.percentile(v, 97.5)), frac_positive=float((v > 0).mean()), n=len(v))
    return out


def plot_video(bench, name, series, path, title=None):
    """series: {nhãn: [điểm theo video]} ; tô vùng Ground Truth bất thường.
    name: tên video cần vẽ (phải nằm trong bench.order); path: nơi lưu ảnh PNG."""
    i = bench.order.index(name)
    y = bench.gt_v[i]
    # Tìm các đoạn liên tiếp có nhãn bất thường: +1 là điểm bắt đầu, -1 là điểm kết thúc
    d = np.diff(np.concatenate([[0], (y > 0.5).astype(int), [0]]))
    plt.figure(figsize=(13, 5), dpi=130)
    for k, (s, e) in enumerate(zip(np.where(d == 1)[0], np.where(d == -1)[0])):
        plt.axvspan(s, e, color='orange', alpha=0.25, label='Ground Truth' if k == 0 else None)   # chỉ gắn nhãn đoạn đầu
    for lab, pv in series.items():
        extra = ''
        if len(set(y)) == 2:                              # video có cả frame bình thường và bất thường mới có AUC
            extra = ' (AUC %.1f)' % (100 * roc_auc_score(y, pv[i]))
        plt.plot(pv[i], lw=1.6, label=lab + extra)
    plt.ylim(-0.05, 1.05); plt.xlabel('Frame'); plt.ylabel('Điểm bất thường')
    plt.title(title or name); plt.grid(True, ls='--', alpha=0.5); plt.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(path); plt.close()