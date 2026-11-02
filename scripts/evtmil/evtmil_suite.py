"""
evtmil_suite.py - chạy lưới cấu hình x seed cho Memory-Masked EVT-MIL trong MỘT tiến trình.
Dữ liệu được nạp vào RAM một lần (có thể cache ra đĩa), mỗi lần huấn luyện chỉ mất vài chục giây.

Cấu hình có sẵn:
  full     biến thể 2, POT + mặt nạ            (đề xuất đầy đủ)
  no_mask  tắt mặt nạ Bernoulli
  no_pot   gộp top-k thay POT
  v1       biến thể 1 (đặc trưng cố định, M chỉ học từ túi bình thường)
  nomem    ĐỐI CHỨNG KHÔNG BỘ NHỚ: MIL top-k, cùng tầng chiếu, cùng hinge, cùng đặc trưng
Cấu hình tùy ý:  tên:khóa=giá_trị,khóa=giá_trị   (kế thừa từ full), ví dụ  K16:K=16  hoặc  rho0:rho=0

Mỗi lần chạy ghi vào <out-dir>/<cấu hình>_s<seed>/ :
  scores_{p,d,s}_{best,last}.npz  (điểm theo PHÂN ĐOẠN; nhân 16 để ra frame), summary.json
  best = epoch có AUC video trên validation cao nhất; last = epoch cuối (quy tắc cố định, không nhìn test)

Ý tưởng mô hình (tóm tắt để đọc code):
  - Bộ nhớ M (K vector) học "mẫu bình thường". Mỗi phân đoạn được tái tạo từ M qua truy xuất thưa;
    sai số tái tạo d càng lớn thì càng bất thường.
  - EVT/POT: phần đuôi của d trên video bình thường (vượt ngưỡng u) được khớp bằng phân phối Pareto tổng quát (GPD);
    p = xác suất "d này là cực trị bất thường" dùng làm trọng số khi gộp điểm cả túi (video).
  - Học đa thể hiện (MIL): hinge xếp hạng sao cho điểm túi bất thường > điểm túi bình thường.
  - Ký hiệu: Sn/dn = túi bình thường (Normal), Sa/da = túi bất thường (Abnormal); "túi" = một video gồm T phân đoạn.
"""
import os, re, glob, math, random, copy, pickle, argparse, json, time
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

# Siêu tham số mặc định của cấu hình 'full'
DEFAULTS = dict(mode='mem',          # 'mem' = có bộ nhớ (EVTMIL); 'nomem' = đối chứng MIL top-k (MILHead)
                variant=2,           # 2: có tầng chiếu học được (d0 -> dim); 1: dùng đặc trưng gốc cố định
                no_pot=False,        # True: bỏ POT, gộp bằng top-k
                no_mask=False,       # True: tắt mặt nạ Bernoulli khi huấn luyện
                T=32,                # số phân đoạn mỗi video
                dim=256,             # số chiều sau tầng chiếu
                K=32,                # số vector trong bộ nhớ
                tau=0.1,             # nhiệt độ softmax khi truy xuất bộ nhớ
                lam_scale=1.0,       # ngưỡng co thưa lam = lam_scale / K
                q=0.9,               # phân vị dùng làm ngưỡng POT u (trên khoảng cách của túi bình thường)
                c=10.0,              # độ "sắc" của softplus làm trơn (d - u)+
                ema=0.95,            # hệ số trung bình trượt mũ cho u, mu, s2
                warmup=300,          # số bước đầu dùng top-k trước khi bật POT
                topk=3,              # số phân đoạn lớn nhất được lấy trung bình khi gộp top-k
                rho=0.5,             # cường độ mặt nạ Bernoulli
                tau_high=0.9,        # chỉ phân đoạn có p > tau_high mới có thể bị che
                gamma=0.5,           # biên (margin) của hinge xếp hạng
                lam_n=0.1,           # trọng số phạt điểm túi bình thường
                lam_mem=1.0,         # trọng số tái tạo trên túi bình thường (dạy M)
                lam_orth=0.01,       # trọng số phạt trực giao các vector bộ nhớ
                epochs=60, bs=30, lr=5e-4, hidden=128)   # hidden: kích thước lớp ẩn của MILHead
# Ghi đè cho từng cấu hình có sẵn (so với DEFAULTS)
BUILTIN = {'full': {}, 'no_mask': {'no_mask': True}, 'no_pot': {'no_pot': True},
           'v1': {'variant': 1}, 'nomem': {'mode': 'nomem', 'gamma': 1.0}}

# Bỏ hậu tố '_x264' / '_i3d' để tên video thống nhất
norm_name = lambda k: re.sub(r'(_x264)?(_i3d)?$', '', k)


def make_cfg(spec, global_over):
    """Tạo (tên, cấu hình) từ một chuỗi spec.
    - spec là tên có sẵn ('full', 'nomem'...): lấy BUILTIN[spec].
    - spec dạng 'tên:khóa=giá_trị,...': kế thừa từ 'full' rồi ghi đè theo các cặp khóa=giá_trị.
    Thứ tự ưu tiên: DEFAULTS < (full nếu tùy ý) < ghi đè toàn cục (--set) < ghi đè riêng của cấu hình."""
    name, over = spec, {}
    if ':' in spec:
        name, kv = spec.split(':', 1)
        for item in kv.split(','):
            k, v = item.split('=')
            over[k.strip()] = v.strip()
    else:
        over = dict(BUILTIN[spec])
    base = dict(DEFAULTS)
    if ':' in spec:
        base.update(BUILTIN['full'])
    base.update(global_over); base.update(over)        # cấu hình riêng thắng ghi đè toàn cục
    for k, v in list(base.items()):                    # ép kiểu theo giá trị mặc định
        if k not in DEFAULTS:
            raise KeyError("tham số '%s' không có trong cấu hình mô hình. Hợp lệ: %s" % (k, sorted(DEFAULTS)))
        d = DEFAULTS[k]
        if isinstance(v, str):                         # giá trị đến từ dòng lệnh là chuỗi -> ép về đúng kiểu
            base[k] = (v.lower() in ('1', 'true', 'yes')) if isinstance(d, bool) else type(d)(v)
    return name, SimpleNamespace(**base)


# ------------------------------------------------------------------ dữ liệu
def video_name(path):
    """Tên video chuẩn hóa từ đường dẫn file đặc trưng (bỏ đuôi, hậu tố '__<số>' của crop, '_x264'/'_i3d')."""
    s = os.path.splitext(os.path.basename(path))[0]
    return norm_name(re.sub(r'__\d+$', '', s))


def to_crops(x):
    """Đưa mảng đặc trưng về dạng (n_crop, n_clip, dim).
    Chấp nhận (n_clip, dim), (n_crop, n_clip, dim) hoặc (n_clip, n_crop, dim) (đoán theo chiều nhỏ <= 10 là số crop)."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        return x[None]
    if x.ndim == 3:
        if x.shape[0] <= 10 and x.shape[0] < x.shape[1]:
            return x
        if x.shape[1] <= 10:
            return x.transpose(1, 0, 2)
    raise ValueError('shape đặc trưng lạ: %s' % (x.shape,))


def sample_segments(x, T):
    """Chia chuỗi clip (n, dim) thành T đoạn đều nhau và lấy trung bình mỗi đoạn -> (T, dim).
    Đoạn nào rỗng (video ngắn hơn T) thì lấy ít nhất 1 clip."""
    n = len(x)
    edges = np.linspace(0, n, T + 1).astype(int)
    out = []
    for j in range(T):
        lo = min(edges[j], n - 1)
        hi = max(edges[j + 1], lo + 1)
        out.append(x[lo:hi].mean(0))
    return np.stack(out).astype(np.float32)


def read_groups(csv_path, index):
    """Đọc csv (cột đầu = đường dẫn file đặc trưng) và gom các file cùng video vào một nhóm.
    index: {tên file -> đường dẫn thật} tìm trong feat_root; nếu không có thì dùng chính đường dẫn trong csv."""
    groups = OrderedDict()
    for p in pd.read_csv(csv_path).iloc[:, 0]:
        src = index.get(os.path.basename(p)) or (p if os.path.exists(p) else None)
        assert src is not None, 'không thấy file đặc trưng: ' + p
        groups.setdefault(video_name(p), []).append(src)
    return groups


def load_train_bag(paths, T, train_crops):
    """Túi huấn luyện của một video: chọn đều tối đa `train_crops` file, mỗi crop -> T phân đoạn.
    Trả về (số_crop_tổng, T, dim); lúc train mỗi lần lấy ngẫu nhiên 1 crop (tăng cường dữ liệu)."""
    k = min(train_crops, len(paths))
    pick = sorted({int(round(i)) for i in np.linspace(0, len(paths) - 1, k)})
    arrs = []
    for i in pick:
        for c in to_crops(np.load(paths[i], mmap_mode='r')):
            arrs.append(sample_segments(c, T))
    return np.stack(arrs)


def load_seq(paths):
    """Chuỗi đặc trưng đầy đủ của một video test (n_clip, dim): trung bình qua các crop, rồi nối các file."""
    cs = [to_crops(np.load(p, mmap_mode='r')) for p in paths]
    t = min(c.shape[1] for c in cs)                    # cắt về độ dài chung nhỏ nhất giữa các file
    return np.concatenate([c[:, :t] for c in cs], 0).mean(0).astype(np.float32)


def parallel(fn, n, threads, desc):
    """Chạy fn(i) cho i = 0..n-1 bằng nhiều luồng (đọc file là việc I/O) và in tiến độ mỗi 300 video."""
    from concurrent.futures import ThreadPoolExecutor
    out, done = [None] * n, [0]
    def work(i):
        out[i] = fn(i); done[0] += 1
        if done[0] % 300 == 0:
            print('  nạp %s: %d/%d' % (desc, done[0], n), flush=True)
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(work, range(n)))
    return out


def load_data(a):
    """Nạp toàn bộ dữ liệu vào RAM: train (bình thường/bất thường), validation, test.
    Nếu có --cache và tham số khớp thì đọc lại từ cache để khỏi nạp từ đĩa."""
    key = dict(T=a.T, train_crops=a.train_crops, val_ratio=a.val_ratio, split_seed=a.split_seed,
               train_csv=a.train_csv, test_csv=a.test_csv)
    if a.cache and os.path.exists(a.cache):
        with open(a.cache, 'rb') as f:
            blob = pickle.load(f)
        if blob['key'] == key:                         # chỉ dùng cache khi mọi tham số liên quan trùng khớp
            print('Nạp cache dữ liệu:', a.cache, flush=True)
            return blob['data']
        print('Cache không khớp tham số -> nạp lại', flush=True)

    # Lập chỉ mục mọi file .npy trong feat_root: tên file -> đường dẫn
    index = {}
    for p in glob.glob(os.path.join(a.feat_root, '**', '*.npy'), recursive=True):
        index.setdefault(os.path.basename(p), p)
    train_all, test_all = read_groups(a.train_csv, index), read_groups(a.test_csv, index)
    names = sorted(train_all)
    cls_of = lambda n: 'Normal' if n.startswith('Normal') else re.sub(r'\d+.*$', '', n)
    by_cls = {}
    for n in names:
        by_cls.setdefault(cls_of(n), []).append(n)
    rng, val_names = random.Random(a.split_seed), set()
    for c, lst in by_cls.items():                      # validation phân tầng theo lớp, cố định theo split_seed
        lst = lst[:]; rng.shuffle(lst)
        val_names.update(lst[:max(1, int(round(len(lst) * a.val_ratio)))])   # mỗi lớp lấy ít nhất 1 video
    tr = [(n, train_all[n]) for n in names if n not in val_names]
    va = [(n, train_all[n]) for n in names if n in val_names]
    tr_n = [it for it in tr if it[0].startswith('Normal')]       # túi bình thường để train
    tr_a = [it for it in tr if not it[0].startswith('Normal')]   # túi bất thường để train
    print('train: %d bình thường + %d bất thường | val: %d | test: %d video'
          % (len(tr_n), len(tr_a), len(va), len(test_all)), flush=True)

    def ld_train(items, desc):
        arrs = parallel(lambda i: load_train_bag(items[i][1], a.T, a.train_crops), len(items), a.io_threads, desc)
        return [(items[i][0], arrs[i]) for i in range(len(items))]
    data = {'tr_n': ld_train(tr_n, 'train bình thường'), 'tr_a': ld_train(tr_a, 'train bất thường')}
    # Validation: mỗi video chỉ dùng 1 file ở giữa danh sách, cố định (không ngẫu nhiên) -> (1, T, dim)
    vv = parallel(lambda i: sample_segments(load_seq([va[i][1][len(va[i][1]) // 2]]), a.T)[None],
                  len(va), a.io_threads, 'validation')
    data['va'] = [(va[i][0], vv[i]) for i in range(len(va))]
    # Test: chuỗi đầy đủ (không gộp phân đoạn) để ra điểm theo từng clip
    tn = list(test_all)
    ts = parallel(lambda i: load_seq(test_all[tn[i]]), len(tn), a.io_threads, 'test')
    data['test'] = [(tn[i], ts[i]) for i in range(len(tn))]
    data['d0'] = data['tr_n'][0][1].shape[-1]          # số chiều đặc trưng gốc
    if a.cache:
        with open(a.cache, 'wb') as f:
            pickle.dump({'key': key, 'data': data}, f, protocol=4)
        print('Đã lưu cache:', a.cache, flush=True)
    return data


# ------------------------------------------------------------------ mô hình
class EVTMIL(nn.Module):
    """Bộ nhớ M (K vector) + thống kê cực trị (u, mu, s2) cho POT/GPD."""
    def __init__(self, cfg, d0):
        super().__init__()
        self.cfg = cfg
        if cfg.variant == 2:
            self.proj, D = nn.Linear(d0, cfg.dim), cfg.dim     # biến thể 2: tầng chiếu học được
        else:
            self.proj, D = nn.Identity(), d0                   # biến thể 1: đặc trưng cố định
        self.M = nn.Parameter(F.normalize(torch.randn(cfg.K, D), dim=1))
        # Các buffer (không học bằng gradient): u = ngưỡng POT; mu, s2 = trung bình/phương sai phần vượt;
        # ready = đã có đủ thống kê chưa; dscale = hệ số chuẩn hóa d về [0,1] lúc suy luận
        for n_, v in [('u', 0.), ('mu', 0.), ('s2', 0.), ('ready', 0.), ('dscale', 1.)]:
            self.register_buffer(n_, torch.tensor(v))

    def Mn(self):
        """Bộ nhớ đã chuẩn hóa L2 theo từng vector."""
        return F.normalize(self.M, dim=1)

    def encode(self, v):
        """Chiếu rồi chuẩn hóa L2 đặc trưng phân đoạn."""
        return F.normalize(self.proj(v), dim=-1)

    def dist(self, f, detach_M):
        """Sai số tái tạo bình phương từ bộ nhớ (truy xuất thưa kiểu hard-shrinkage).
        detach_M=True: chặn gradient vào M (dùng cho túi bất thường để M chỉ học từ túi bình thường)."""
        c = self.cfg
        M = self.Mn()
        if detach_M:
            M = M.detach()
        w = torch.softmax(f @ M.t() / c.tau, -1)               # trọng số truy xuất tới K vector
        lam, eps = c.lam_scale / c.K, 1e-6
        wt = F.relu(w - lam) * w / ((w - lam).abs() + eps)     # co thưa: trọng số <= lam về 0
        wh = wt / (wt.sum(-1, keepdim=True) + eps)             # chuẩn hóa lại tổng = 1
        return ((f - wh @ M) ** 2).sum(-1)                     # ||f - tái tạo||^2

    @torch.no_grad()
    def update_stats(self, dn):
        """Cập nhật u, mu, s2 từ khoảng cách của túi bình thường dn (trung bình trượt mũ).
        u = phân vị q của dn; mu, s2 = trung bình/phương sai của phần vượt dn - u (chỉ khi dn > u)."""
        c = self.cfg
        dn = dn.flatten()
        ub = torch.quantile(dn, c.q)
        first = self.ready.item() == 0
        self.u.copy_(ub if first else c.ema * self.u + (1 - c.ema) * ub)
        r = dn[dn > self.u] - self.u                           # các phần vượt ngưỡng
        if r.numel() >= 5:                                     # cần đủ mẫu để ước lượng phương sai
            mb, sb = r.mean(), r.var()
            if first or self.mu.item() == 0:
                self.mu.copy_(mb); self.s2.copy_(sb)
            else:
                self.mu.copy_(c.ema * self.mu + (1 - c.ema) * mb)
                self.s2.copy_(c.ema * self.s2 + (1 - c.ema) * sb)
            self.ready.fill_(1.)

    def evt(self):
        """Ước lượng tham số GPD bằng phương pháp moment: xi (hình dạng), sigma (tỉ lệ)."""
        mu, s2 = self.mu.clamp_min(1e-6), self.s2.clamp_min(1e-8)
        xi = (0.5 * (1 - mu ** 2 / s2)).clamp(-0.5, 0.5)
        sigma = (mu * (1 - xi)).clamp_min(1e-6)
        return float(xi), float(sigma)

    def p_of(self, d):
        """p = hàm phân phối GPD của phần vượt (d - u)+ (làm trơn bằng softplus); p gần 1 = khoảng cách cực trị."""
        xi, sigma = self.evt()
        z = F.softplus(d - self.u, beta=self.cfg.c / sigma)    # (d - u)+ khả vi
        r = z / sigma
        if abs(xi) < 1e-4:
            return 1 - torch.exp(-r)                           # xi ~ 0: GPD suy biến thành hàm mũ
        return 1 - (1 + xi * r).clamp_min(1e-8).pow(-1.0 / xi)

    def topk(self, d):
        """Điểm túi = trung bình k phân đoạn có khoảng cách lớn nhất."""
        return d.topk(min(self.cfg.topk, d.shape[-1]), dim=-1)[0].mean(-1)

    def pool(self, d, mask):
        """Gộp có trọng số POT: điểm túi = sum(p*d) / (1 + sum(p)).
        mask=True: mặt nạ Bernoulli ngẫu nhiên che bớt các phân đoạn p cao (chống chỉ dựa vào vài đoạn)."""
        c = self.cfg
        p = self.p_of(d)
        if mask:
            keep = (1 - c.rho * (p.detach() > c.tau_high).float() * p.detach()).clamp(0, 1)
            p = torch.bernoulli(keep) * p
        return (p * d).sum(-1) / (1 + p.sum(-1)), p


class MILHead(nn.Module):
    """Đối chứng không bộ nhớ: cùng tầng chiếu, đầu chấm điểm MLP, gộp top-k, hinge xếp hạng."""
    def __init__(self, cfg, d0):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(d0, cfg.dim)
        self.head = nn.Sequential(nn.Linear(cfg.dim, cfg.hidden), nn.ReLU(), nn.Linear(cfg.hidden, 1))

    def encode(self, v):
        return F.normalize(self.proj(v), dim=-1)

    def score(self, f):
        """Điểm bất thường mỗi phân đoạn trong [0,1]."""
        return torch.sigmoid(self.head(f)).squeeze(-1)

    def topk(self, s):
        return s.topk(min(self.cfg.topk, s.shape[-1]), dim=-1)[0].mean(-1)


# ------------------------------------------------------------------ huấn luyện một lần chạy
def train_one(cfg, seed, data, device):
    """Huấn luyện một (cấu hình, seed). Trả về model, state tốt nhất (theo val AUC), state cuối, thông tin/lịch sử."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    d0 = data['d0']
    mem = cfg.mode == 'mem'
    model = (EVTMIL(cfg, d0) if mem else MILHead(cfg, d0)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    tr_n, tr_a, va = data['tr_n'], data['tr_a'], data['va']
    B = min(cfg.bs, len(tr_n), len(tr_a))              # mỗi batch: B túi bình thường + B túi bất thường
    spe = min(len(tr_n), len(tr_a)) // B               # số bước mỗi epoch
    pick = lambda arr: arr[random.randrange(len(arr))]  # chọn ngẫu nhiên 1 crop của một video
    batch = lambda items, idx: torch.from_numpy(np.stack([pick(items[i][1]) for i in idx])).to(device)
    va_x = torch.from_numpy(np.stack([a[0] for _, a in va])).to(device)      # (n_val, T, dim)
    va_y = np.array([int(not n.startswith('Normal')) for n, _ in va])        # nhãn video: 1 = bất thường
    state = {'step': 0}                                # đếm số bước huấn luyện toàn cục

    def bag_scores(x):
        """Điểm túi dùng cho validation; cùng quy tắc gộp như lúc train (top-k trong warmup, POT sau đó)."""
        f = model.encode(x)
        if not mem:
            return model.topk(model.score(f))
        d = model.dist(f, False)
        use_pot = (not cfg.no_pot) and model.ready.item() == 1 and state['step'] >= cfg.warmup
        return model.pool(d, False)[0] if use_pot else model.topk(d)

    @torch.no_grad()
    def val_auc():
        """AUC mức video trên tập validation (không dùng test để chọn checkpoint)."""
        model.eval()
        s = bag_scores(va_x).cpu().numpy()
        model.train()
        return roc_auc_score(va_y, s) if len(set(va_y)) == 2 else float('nan')

    best, best_ep, best_state, hist = -1.0, -1, None, {'val': [], 'd_norm': [], 'd_abn': []}
    # Chỉ cho phép chọn 'best' sau warmup (khi POT đã bật) để so sánh công bằng giữa các epoch
    selectable_from = 0 if (not mem or cfg.no_pot) else cfg.warmup
    t0 = time.time()
    for ep in range(cfg.epochs):
        pn_, pa_ = np.random.permutation(len(tr_n)), np.random.permutation(len(tr_a))
        a_dn = a_da = 0.0                              # tích lũy khoảng cách trung bình để theo dõi
        for it in range(spe):
            vn, vaa = batch(tr_n, pn_[it * B:(it + 1) * B]), batch(tr_a, pa_[it * B:(it + 1) * B])
            fn, fa = model.encode(vn), model.encode(vaa)
            if mem:
                # dn: túi bình thường (gradient vào M); da: túi bất thường (M bị detach, M chỉ học từ bình thường)
                dn, da = model.dist(fn, False), model.dist(fa, True)
                if state['step'] == cfg.warmup:
                    model.ready.fill_(0.)              # hết warmup: ước lượng lại u, mu, s2 từ đầu cho POT
                model.update_stats(dn.detach())
                warm = state['step'] < cfg.warmup or model.ready.item() == 0 or cfg.no_pot
                if warm:
                    Sn, Sa = model.topk(dn), model.topk(da)            # giai đoạn warmup / no_pot: gộp top-k
                else:
                    Sn = model.pool(dn, False)[0]                      # túi bình thường: POT, không che
                    Sa = model.pool(da, not cfg.no_mask)[0]            # túi bất thường: POT + mặt nạ Bernoulli
                # Hinge xếp hạng: mọi cặp (bất thường j, bình thường i) phải cách nhau >= gamma
                hinge = F.relu(cfg.gamma - Sa[:, None] + Sn[None, :]).mean()
                Mn = model.Mn()
                orth = ((Mn @ Mn.t() - torch.eye(cfg.K, device=device)) ** 2).sum()   # các vector bộ nhớ gần trực giao
                # loss = xếp hạng + giữ điểm túi bình thường thấp + tái tạo tốt dữ liệu bình thường + trực giao
                loss = hinge + cfg.lam_n * Sn.mean() + cfg.lam_mem * dn.mean() + cfg.lam_orth * orth
                a_dn += float(dn.detach().mean()); a_da += float(da.detach().mean())
            else:
                # Đối chứng không bộ nhớ: MIL top-k + hinge, cùng tầng chiếu
                Sn, Sa = model.topk(model.score(fn)), model.topk(model.score(fa))
                hinge = F.relu(cfg.gamma - Sa[:, None] + Sn[None, :]).mean()
                loss = hinge + cfg.lam_n * Sn.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            state['step'] += 1
        v = val_auc()
        hist['val'].append(v); hist['d_norm'].append(a_dn / max(spe, 1)); hist['d_abn'].append(a_da / max(spe, 1))
        if not math.isnan(v) and v > best and state['step'] >= selectable_from:   # lưu epoch tốt nhất theo val
            best, best_ep = v, ep + 1
            best_state = copy.deepcopy(model.state_dict())
        if (ep + 1) % 20 == 0 or ep == 0:
            print('    ep %3d | val AUC(video) %.4f | d_norm %.3f d_abn %.3f | %.0fs'
                  % (ep + 1, v, a_dn / max(spe, 1), a_da / max(spe, 1), time.time() - t0), flush=True)
    last_state = copy.deepcopy(model.state_dict())
    if best_state is None:                             # không epoch nào hợp lệ: dùng luôn state cuối
        best_state, best_ep = last_state, cfg.epochs
    return model, best_state, last_state, dict(best_val=best, best_epoch=best_ep, last_val=hist['val'][-1], hist=hist)


@torch.no_grad()
def infer_all(model, cfg, state_dict, data, device):
    """Suy luận trên mọi video test với một checkpoint. Trả về điểm theo phân đoạn:
       mode 'mem'  : out['d'] = khoảng cách tái tạo chuẩn hóa về [0,1]; out['p'] = xác suất GPD (nếu không no_pot)
       mode 'nomem': out['s'] = điểm sigmoid của MILHead."""
    model.load_state_dict(state_dict)
    model.eval()
    out = {}
    if cfg.mode == 'mem':
        # Hệ số chuẩn hóa dscale = phân vị 99.9% của khoảng cách trên dữ liệu train bình thường
        xs = torch.from_numpy(np.stack([a[0] for _, a in data['tr_n']])).to(device)
        d_tr = model.dist(model.encode(xs), False).flatten().cpu()
        model.dscale.fill_(float(torch.quantile(d_tr, 0.999)))
        out['d'] = {}
        if not cfg.no_pot:
            out['p'] = {}
    else:
        out['s'] = {}
    for name, seq in data['test']:
        x = torch.from_numpy(seq).to(device)
        f = model.encode(x)
        if cfg.mode == 'mem':
            d = model.dist(f, False)
            out['d'][name] = (d / model.dscale).clamp(0, 1).cpu().numpy().astype(np.float32)
            if 'p' in out:
                out['p'][name] = model.p_of(d).cpu().numpy().astype(np.float32)
        else:
            out['s'][name] = model.score(f).cpu().numpy().astype(np.float32)
    return out


def quick_auc(out_which, order, gt):
    """AUC nhanh để THEO DÕI: lặp mỗi điểm clip 16 lần ra frame, ghép theo `order`, so với gt.
    Trả về NaN nếu tổng độ dài không khớp gt. Không dùng để chọn checkpoint."""
    pred = np.concatenate([np.repeat(out_which[n], 16) for n in order])
    return roc_auc_score(gt, pred) if len(pred) == len(gt) else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-csv', required=True)
    ap.add_argument('--test-csv', required=True)
    ap.add_argument('--feat-root', required=True)
    ap.add_argument('--gt', default=None, help='gt_ucf.npy: chỉ để IN AUC theo dõi, không dùng để chọn checkpoint')
    ap.add_argument('--out-dir', default='results_evtmil')
    ap.add_argument('--configs', nargs='+', default=['full'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[234])
    ap.add_argument('--set', nargs='*', default=[], help='ghi đè toàn cục, ví dụ epochs=30 warmup=200')
    ap.add_argument('--cache', default=None)
    ap.add_argument('--train-crops', type=int, default=3)
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--split-seed', type=int, default=234, help='cố định cách chia validation cho mọi seed')
    ap.add_argument('--io-threads', type=int, default=16)
    ap.add_argument('--skip-done', action='store_true')
    a = ap.parse_args()
    a.T = DEFAULTS['T']
    glob_over = dict(kv.split('=') for kv in a.set)
    if 'T' in glob_over:                               # T ảnh hưởng tới cách nạp dữ liệu nên xử lý riêng
        a.T = int(glob_over['T'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device, flush=True)
    data = load_data(a)
    order = [n for n, _ in data['test']]               # thứ tự video test (khớp thứ tự trong gt)
    gt = np.load(a.gt).ravel() if a.gt else None
    os.makedirs(a.out_dir, exist_ok=True)

    for spec in a.configs:                             # lưới: mọi cấu hình x mọi seed
        name, cfg = make_cfg(spec, glob_over)
        for seed in a.seeds:
            run_dir = os.path.join(a.out_dir, '%s_s%d' % (name, seed))
            if a.skip_done and os.path.exists(os.path.join(run_dir, 'summary.json')):
                print('bỏ qua (đã có):', run_dir); continue
            print('== %s | seed %d' % (name, seed), flush=True)
            os.makedirs(run_dir, exist_ok=True)
            model, best_state, last_state, info = train_one(cfg, seed, data, device)
            line = []
            for ck, st in (('best', best_state), ('last', last_state)):
                out = infer_all(model, cfg, st, data, device)
                for which, sc in out.items():          # which = 'd' | 'p' | 's'
                    np.savez(os.path.join(run_dir, 'scores_%s_%s.npz' % (which, ck)), **sc)
                    if gt is not None:
                        line.append('%s/%s AUC %.2f' % (which, ck, 100 * quick_auc(sc, order, gt)))
            info.update(cfg=vars(cfg), seed=seed, name=name)
            with open(os.path.join(run_dir, 'summary.json'), 'w') as f:
                json.dump(info, f)
            print('   val(best ep %d) %.4f | test(chỉ theo dõi): %s' % (info['best_epoch'], info['best_val'], ' | '.join(line)), flush=True)
    print('XONG ->', a.out_dir)


if __name__ == '__main__':
    main()