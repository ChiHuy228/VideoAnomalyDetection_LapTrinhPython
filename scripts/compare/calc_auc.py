import os, re, csv, glob, argparse, itertools
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Tham số dòng lệnh ----------------
ap_ = argparse.ArgumentParser()
ap_.add_argument('--evt-glob', default='results/evtmil/*/video_scores_*.npz',
                 help='mẫu đường dẫn các file điểm EVT-MIL (mỗi file là một dòng trong bảng)')
ap_.add_argument('--no-evt', action='store_true', help='không đưa EVT-MIL vào so sánh')
ap_.add_argument('--out-dir', default='results/compare')
args = ap_.parse_args()

# ---------------- Cấu hình ----------------
ALL_METHODS = {'MIL':     'results/mil_sultani/video_scores.npz',
               'RTFM':    'results/rtfm/video_scores.npz',
               'URDMU':   'results/urdmu/video_scores.npz',
               'VadCLIP': 'results/vadclip/video_scores.npz'}

# EVT-MIL: results/evtmil/<tag>/video_scores_<p|d>.npz  ->  'EVT <tag> <p|d>'
if not args.no_evt:
    for f in sorted(glob.glob(args.evt_glob)):
        tag = os.path.basename(os.path.dirname(f))
        which = re.sub(r'^video_scores_|\.npz$', '', os.path.basename(f))
        ALL_METHODS['EVT %s %s' % (tag, which)] = f

GT      = 'scripts/urdmu/frame_label/gt-ucf.npy'   # 1109680 frame
OUT_DIR = args.out_dir
THR     = 0.5      # ngưỡng cố định cho FAR@thr (lưu ý: thang điểm của từng phương pháp có thể khác nhau)
TARGET_TPR = 0.5   # mức TPR cho FAR@TPR

os.makedirs(OUT_DIR, exist_ok=True)
METHODS = {m: p for m, p in ALL_METHODS.items() if os.path.exists(p)}
for m in ALL_METHODS:
    if m not in METHODS:
        print('Bỏ qua (chưa có file):', m)
assert 'URDMU' in METHODS, "Cần có URDMU làm tham chiếu độ dài"

# ---------------- Đọc điểm ----------------
norm = lambda k: re.sub(r'(_x264)?(_i3d)?$', '', k)
S = {m: {norm(k): v for k, v in np.load(p).items()} for m, p in METHODS.items()}
ref = S['URDMU']
names = list(ref)

for m in list(METHODS):
    miss = [n for n in names if n not in S[m]]
    if miss:
        print('!!! %s thiếu %d video (vd: %s) -> bỏ qua' % (m, len(miss), miss[:3]))
        del METHODS[m]; del S[m]

# ---------------- GT theo video ----------------
gt = np.load(GT).ravel()
pos, G = 0, {}
for n in names:
    L = len(ref[n]); G[n] = gt[pos:pos + L]; pos += L
assert pos == len(gt), (pos, len(gt))

def resample(s, L):
    s = np.asarray(s, dtype=np.float64)
    if len(s) == L: return s
    if len(s) == 1: return np.full(L, s[0])
    return np.interp(np.linspace(0, len(s) - 1, L), np.arange(len(s)), s)

cat = lambda n: 'Normal' if n.startswith('Normal') else re.sub(r'\d+.*$', '', n)
cats = sorted({cat(n) for n in names} - {'Normal'})
normals = [n for n in names if cat(n) == 'Normal']

# Gom sẵn điểm đã nội suy để khỏi tính lại
P = {m: {n: resample(S[m][n], len(G[n])) for n in names} for m in METHODS}

def cat_arrays(vids, m):
    y = np.concatenate([G[n] for n in vids])
    p = np.concatenate([P[m][n] for n in vids])
    return y, p

# ---------------- Chỉ số ----------------
def auc_of(vids, m):
    y, p = cat_arrays(vids, m)
    return roc_auc_score(y, p)

def ap_of(m):
    y, p = cat_arrays(names, m)
    pr, rc, _ = precision_recall_curve(y, p)
    return auc(rc, pr)                      # cùng công thức với ucf_test.py của URDMU

def far_at_thr(m, thr=THR):
    # tỉ lệ frame bị báo động nhầm trên các video Normal
    p = np.concatenate([P[m][n] for n in normals])
    return float((p > thr).mean())

def far_at_tpr(m, tpr_target=TARGET_TPR):
    # chọn ngưỡng sao cho TPR >= tpr_target trên toàn bộ frame,
    # rồi đo FAR trên các video Normal ở ngưỡng đó
    y, p = cat_arrays(names, m)
    fpr, tpr, thr = roc_curve(y, p)
    t = thr[np.searchsorted(tpr, tpr_target)]
    pn = np.concatenate([P[m][n] for n in normals])
    return float((pn >= t).mean())

res = {m: {'All': auc_of(names, m)} for m in METHODS}
for m in METHODS:
    for c in cats:
        res[m][c] = auc_of([n for n in names if cat(n) == c] + normals, m)

ap      = {m: ap_of(m) for m in METHODS}
far_thr = {m: far_at_thr(m) for m in METHODS}
far_tpr = {m: far_at_tpr(m) for m in METHODS}

# ---------------- Lưu CSV ----------------
labels = ['All'] + cats
with open(os.path.join(OUT_DIR, 'metrics.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Method', 'AUC', 'AP', 'FAR@%.1f' % THR, 'FAR@TPR=%.1f' % TARGET_TPR])
    for m in METHODS:
        w.writerow([m, '%.2f' % (res[m]['All'] * 100), '%.2f' % (ap[m] * 100),
                    '%.2f' % (far_thr[m] * 100), '%.2f' % (far_tpr[m] * 100)])

with open(os.path.join(OUT_DIR, 'auc_per_class.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Lớp'] + list(METHODS))
    for l in labels:
        w.writerow([l] + ['%.2f' % (res[m][l] * 100) for m in METHODS])

# ---------------- Màu: 4 phương pháp gốc cố định, EVT-MIL lấy màu khác ----------------
COLORS = {'MIL': '#9ecae1', 'RTFM': '#3182bd', 'URDMU': '#08519c', 'VadCLIP': '#de2d26'}
_extra = itertools.cycle(['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f'])
for m in METHODS:
    if m not in COLORS:
        COLORS[m] = next(_extra)

# ---------------- Biểu đồ 1: AUC theo lớp ----------------
n = len(METHODS)
x = np.arange(len(labels)); w = 0.8 / n
fig, ax = plt.subplots(figsize=(max(14, 1.0 * n * len(labels) / 3), 5))
for i, m in enumerate(METHODS):
    ax.bar(x + (i - (n - 1) / 2) * w, [res[m][l] * 100 for l in labels], w, label=m, color=COLORS[m])
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('AUC (%)'); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'auc_bar.png'), dpi=200); plt.close()

# ---------------- Hình 2a: ROC ----------------
fig, ax = plt.subplots(figsize=(5.5, 5))
for m in METHODS:
    y, p = cat_arrays(names, m)
    fpr, tpr, _ = roc_curve(y, p)
    ax.plot(fpr, tpr, color=COLORS[m], lw=2.2,
            label='%s  (AUC %.1f)' % (m, res[m]['All'] * 100))
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right', fontsize=8); ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'roc.png'), dpi=200); plt.close()

# ---------------- Hình 2b: bảng nhiệt các chỉ số ----------------
cols = [('AUC ↑',                          [res[m]['All'] * 100 for m in METHODS], True),
        ('AP ↑',                           [ap[m] * 100 for m in METHODS],         True),
        ('FAR@%.1f ↓' % THR,               [far_thr[m] * 100 for m in METHODS],    False),
        ('FAR@TPR=%.1f ↓' % TARGET_TPR,    [far_tpr[m] * 100 for m in METHODS],    False)]

vals  = np.array([c[1] for c in cols]).T            # (số phương pháp, số chỉ số)
score = np.zeros_like(vals)                         # 1 = tốt nhất, 0 = kém nhất trong cột
for j, (_, v, hb) in enumerate(cols):
    v = np.asarray(v); s = (v - v.min()) / (v.max() - v.min() + 1e-9)
    score[:, j] = s if hb else 1 - s

fig, ax = plt.subplots(figsize=(1.9 * len(cols) + 1.5, 0.9 * len(METHODS) + 1.2))
ax.imshow(score, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        ax.text(j, i, '%.2f' % vals[i, j], ha='center', va='center', fontsize=12,
                fontweight='bold' if score[i, j] == 1 else 'normal')
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=11)
ax.set_yticks(range(len(METHODS))); ax.set_yticklabels(list(METHODS), fontsize=11)
ax.xaxis.tick_top()
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'metrics_heatmap.png'), dpi=200); plt.close()

# ---------------- In gọn ----------------
w0 = max(10, max(len(m) for m in METHODS) + 1)
print('\n%-*s %8s %8s %10s %14s' % (w0, 'Method', 'AUC', 'AP', 'FAR@%.1f' % THR, 'FAR@TPR=%.1f' % TARGET_TPR))
for m in METHODS:
    print('%-*s %8.2f %8.2f %10.2f %14.2f' %
          (w0, m, res[m]['All'] * 100, ap[m] * 100, far_thr[m] * 100, far_tpr[m] * 100))
print('\nĐã lưu vào', OUT_DIR, ': metrics.csv, auc_per_class.csv, auc_bar.png, roc.png, metrics_heatmap.png')