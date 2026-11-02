"""
Xuất điểm VadCLIP theo từng video -> video_scores.npz (key = tên video, value = điểm theo frame).
Đặt file này vào thư mục src của VadCLIP (cùng chỗ với model.py, ucf_option.py, thư mục clip/, utils/).
"""
import os
import glob
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import ucf_option
from model import CLIPVAD
from utils.dataset import UCFDataset
from utils.tools import get_batch_mask, get_prompt_text

ucf_option.parser.add_argument('--feat-root', default=None,
                               help='thư mục chứa đặc trưng .npy trên máy (vd: .../UCFClipFeatures)')
ucf_option.parser.add_argument('--score', default=1, type=int, choices=[1, 2],
                               help='1: sigmoid(logits1); 2: 1 - P(Normal) của nhánh CLIP')
ucf_option.parser.add_argument('--out', default='video_scores.npz')


def remap_csv(csv_path, feat_root):
    """Đổi đường dẫn trong csv sang đường dẫn thật trên máy (khớp theo tên file)."""
    d = pd.read_csv(csv_path)
    col = d.columns[0]
    if feat_root is not None:
        index = {}
        for p in glob.glob(os.path.join(feat_root, '**', '*.npy'), recursive=True):
            index.setdefault(os.path.basename(p), p)
        new = d[col].map(lambda x: index.get(os.path.basename(x)))
        print('khớp file:', int(new.notna().sum()), '/', len(d))
        d[col] = new.fillna(d[col])
    missing = (~d[col].map(os.path.exists)).sum()
    if missing:
        raise FileNotFoundError(f'{missing} file đặc trưng không tồn tại. Kiểm tra --feat-root')
    tmp = os.path.join(tempfile.gettempdir(), 'vadclip_test_tmp.csv')
    d.to_csv(tmp, index=False)
    return tmp, d[col].tolist()


@torch.no_grad()
def run(model, loader, maxlen, prompt_text, device):
    model.to(device).eval()
    p1_list, p2_list = [], []
    for item in loader:
        visual = item[0].squeeze(0)
        length = int(item[2])
        len_cur = length
        if len_cur < maxlen:
            visual = visual.unsqueeze(0)
        visual = visual.to(device)

        lengths = torch.zeros(int(length / maxlen) + 1)
        for j in range(int(length / maxlen) + 1):
            if j == 0 and length < maxlen:
                lengths[j] = length
            elif j == 0 and length > maxlen:
                lengths[j] = maxlen
                length -= maxlen
            elif length > maxlen:
                lengths[j] = maxlen
                length -= maxlen
            else:
                lengths[j] = length
        lengths = lengths.to(int)
        padding_mask = get_batch_mask(lengths, maxlen).to(device)

        _, logits1, logits2 = model(visual, padding_mask, prompt_text, lengths)
        logits1 = logits1.reshape(-1, logits1.shape[2])
        logits2 = logits2.reshape(-1, logits2.shape[2])
        p1_list.append(torch.sigmoid(logits1[:len_cur].squeeze(-1)).cpu().numpy())
        sm = logits2[:len_cur].softmax(dim=-1)
        p2_list.append((1 - sm[:, 0]).cpu().numpy())
    return p1_list, p2_list


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = ucf_option.parser.parse_args()
    print('device:', device)

    label_map = {k: k for k in ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
                                'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
                                'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']}

    csv_tmp, paths = remap_csv(args.test_list, args.feat_root)
    vid_names = [os.path.basename(p).rsplit('__', 1)[0] for p in paths]   # vd: Abuse028_x264

    ds = UCFDataset(args.visual_length, csv_tmp, True, label_map)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    assert len(loader) == len(vid_names), (len(loader), len(vid_names))
    print('số mẫu test:', len(loader))

    model = CLIPVAD(args.classes_num, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    args.prompt_prefix, args.prompt_postfix, device)
    try:
        state = torch.load(args.model_path, map_location=device, weights_only=False)
    except TypeError:                       # torch cũ chưa có weights_only
        state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)

    p1, p2 = run(model, loader, args.visual_length, get_prompt_text(label_map), device)
    scores = p1 if args.score == 1 else p2

    # gộp crop cùng video (nếu có nhiều dòng/video) bằng trung bình
    groups = {}
    for n, s in zip(vid_names, scores):
        groups.setdefault(n, []).append(s)
    out = {}
    for n, lst in groups.items():
        assert len(set(len(a) for a in lst)) == 1, f'{n}: các crop khác độ dài'
        out[n] = np.repeat(np.mean(lst, axis=0), 16).astype(np.float32)   # segment -> frame

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(len(out), 'video ->', args.out)