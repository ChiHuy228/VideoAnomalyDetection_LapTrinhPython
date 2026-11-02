"""
Xuất điểm RTFM theo từng video -> video_scores.npz (key = tên video, value = điểm theo frame).
Cùng định dạng với bản xuất của VadCLIP (mỗi điểm segment được lặp 16 lần thành điểm frame).

Đặt file này vào scripts/rtfm/ (cùng chỗ với model.py, dataset.py, thư mục list/) và chạy từ chính thư mục đó:

    python export_scores.py --ckpt ../../results/rtfm/rtfm5600-i3d.pkl \
                            --out  ../../results/rtfm/video_scores.npz
"""
import argparse
import glob
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import dataset as dataset_mod
from dataset import Dataset
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, help='checkpoint RTFM (vd: rtfm5600-i3d.pkl)')
parser.add_argument('--out', default='video_scores.npz')
parser.add_argument('--gt', default='list/gt-ucf.npy', help='dùng để kiểm tra lại AUC sau khi xuất')
parser.add_argument('--feat-root', default=None,
                    help='thư mục chứa đặc trưng I3D TEST trên máy (nơi có Abuse028_x264.npy ...). '
                         'Dùng khi list vẫn ghi đường dẫn Kaggle')
parser.add_argument('--feature-size', type=int, default=None,
                    help='mặc định: tự đọc từ checkpoint')
parser.add_argument('--batch-size', type=int, default=32, help='chỉ để khởi tạo Model, không ảnh hưởng suy luận')
parser.add_argument('--strip-suffix', default='_i3d',
                    help="bỏ hậu tố này khỏi tên video nếu có, để khớp tên với VadCLIP/MIL (vd Abuse028_x264)")
parser.add_argument('--ref', default=None,
                    help='npz khác (vd video_scores.npz của VadCLIP) để kiểm tra khớp tên và độ dài')


def install_remap(feat_root):
    """Đổi đường dẫn trong list sang đường dẫn thật trên máy (khớp theo tên file), không sửa dataset.py."""
    index, dup = {}, set()
    for p in glob.glob(os.path.join(feat_root, '**', '*.npy'), recursive=True):
        b = os.path.basename(p)
        if b in index:
            dup.add(b)
        index.setdefault(b, p)
    print(f'--feat-root: {len(index)} file .npy')
    orig = dataset_mod.group_videos

    def patched(list_file):
        res = []
        for base, paths in orig(list_file):
            new = []
            for p in paths:
                if os.path.exists(p):
                    new.append(p)
                    continue
                b = os.path.basename(p)
                if b not in index:
                    raise FileNotFoundError(f'không thấy {b} dưới {feat_root}')
                if b in dup:
                    raise RuntimeError(f'{b} xuất hiện nhiều lần dưới {feat_root}; '
                                       'trỏ --feat-root sâu hơn, vào đúng thư mục I3D test')
                new.append(index[b])
            res.append((base, new))
        return res

    dataset_mod.group_videos = patched


def score_video(model, x):
    """x: (1,10,T,F). Chạy cả 10 crop một lượt; nếu hết VRAM (non-local tốn bộ nhớ ~T^2 mỗi crop)
    thì chia thành nhóm crop nhỏ hơn rồi lấy trung bình. Kết quả không đổi vì các crop độc lập khi eval
    và điểm cuối là trung bình đều trên crop (nhóm luôn bằng nhau: 10, 5, 2, 1)."""
    ncrops = x.shape[1]
    for c in (10, 5, 2, 1):
        if ncrops % c:
            continue
        try:
            parts = [model(inputs=x[:, i:i + c])[6][0, :, 0] for i in range(0, ncrops, c)]
            if c < ncrops:
                print(f'  (T={x.shape[2]}: chạy {c} crop mỗi lượt do VRAM)')
            return torch.stack(parts).mean(0)
        except RuntimeError as e:                          # CUDA OOM
            if 'out of memory' not in str(e).lower():
                raise
            torch.cuda.empty_cache()
    raise RuntimeError(f'hết VRAM ngay cả với 1 crop (T={x.shape[2]}); chạy lại với CUDA_VISIBLE_DEVICES="" để dùng CPU')


@torch.no_grad()
def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    if args.feat_root:
        install_remap(args.feat_root)
    ds = Dataset(None, test_mode=True)                     # đọc list/UCF_Test.list, giữ đúng thứ tự
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    assert len(ds.names) == len(loader), (len(ds.names), len(loader))
    print('số video test:', len(ds.names))
    data_dim = ds.cache[0].shape[-1]

    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:                                      # torch cũ chưa có weights_only
        state = torch.load(args.ckpt, map_location=device)
    if isinstance(state, dict):
        for k in ('model', 'state_dict'):
            if k in state:
                state = state[k]
                break
    ckpt_dim = state['fc1.weight'].shape[1]                # chiều đầu vào mà checkpoint được train
    print(f'đặc trưng dữ liệu: {data_dim} chiều | checkpoint: {ckpt_dim} chiều')
    if args.feature_size is not None and args.feature_size != ckpt_dim:
        raise SystemExit(f'--feature-size={args.feature_size} không khớp checkpoint ({ckpt_dim})')
    if data_dim != ckpt_dim:
        raise SystemExit(f'đặc trưng {data_dim} chiều nhưng checkpoint train với {ckpt_dim} chiều: '
                         'sai thư mục --feat-root hoặc sai checkpoint')

    model = Model(ckpt_dim, args.batch_size).to(device)
    model.load_state_dict(state)
    model.eval()

    keys, out = [], {}
    for name, x in zip(ds.names, loader):
        x = x.to(device).permute(0, 2, 1, 3)               # (1,T,10,F) -> (1,10,T,F), như test_10crop.py
        seg = score_video(model, x).float().cpu().numpy()  # điểm từng segment, đã trung bình 10 crop
        key = name
        if args.strip_suffix and key.endswith(args.strip_suffix):
            key = key[:-len(args.strip_suffix)]
        assert key not in out, f'trùng tên video: {key}'
        keys.append(key)
        out[key] = np.repeat(seg, 16).astype(np.float32)   # segment -> frame

    print('ví dụ tên:', keys[:3], '...', keys[-2:])

    # ---- kiểm tra lại với GT: phải khớp AUC đã báo cáo ở bước 5600 ----
    try:
        gt = np.load(args.gt)
        pred = np.concatenate([out[k] for k in keys])
        if len(pred) != len(gt):
            print(f'[CẢNH BÁO] len(pred)={len(pred)} != len(gt)={len(gt)}, bỏ qua kiểm tra AUC')
        else:
            vlabel = np.array([0 if k.startswith('Normal') else 1 for k in keys])
            vmax = np.array([out[k].max() for k in keys])
            print('kiểm tra: AUC {:.4f} | video-AUC {:.4f}'.format(
                roc_auc_score(gt, pred), roc_auc_score(vlabel, vmax)))
    except FileNotFoundError:
        print('không thấy', args.gt, '-> bỏ qua kiểm tra AUC')

    # ---- so khớp với file điểm của mô hình khác ----
    if args.ref:
        ref = np.load(args.ref)
        both = sorted(set(out) & set(ref.files))
        print(f'khớp tên với {args.ref}: {len(both)}/{len(out)}')
        print('  thiếu ở ref:', sorted(set(out) - set(ref.files))[:5])
        print('  thừa ở ref :', sorted(set(ref.files) - set(out))[:5])
        diff = [(k, len(out[k]), len(ref[k])) for k in both if len(out[k]) != len(ref[k])]
        print('  video khác độ dài frame:', len(diff), diff[:5])

    np.savez(args.out, **out)
    print(len(out), 'video ->', args.out)


if __name__ == '__main__':
    main()