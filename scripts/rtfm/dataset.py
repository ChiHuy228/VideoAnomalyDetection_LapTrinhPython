import os, re
import torch.utils.data as data
import numpy as np
import torch
from tqdm import tqdm
from utils import process_feat

def group_videos(list_file):
    """Gom các dòng list (mỗi video 10 file crop) thành [(tên_video, [10 đường dẫn theo thứ tự crop])]."""
    groups, order = {}, []
    for l in open(list_file):
        if not l.strip():
            continue
        p = l.split()[0]
        m = re.match(r'^(.*?)(?:__(\d+))?\.npy$', os.path.basename(p))
        base, k = m.group(1), int(m.group(2) or 0)
        if base not in groups:
            groups[base] = {}
            order.append(base)
        groups[base][k] = p
    return [(b, [groups[b][k] for k in sorted(groups[b])]) for b in order]

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.test_mode = test_mode
        vids = group_videos('list/UCF_Test.list' if test_mode else 'list/UCF_Train.list')
        if not test_mode:
            vids = [v for v in vids if v[0].startswith('Normal') == is_normal]
        self.names = [v[0] for v in vids]
        name = 'test' if test_mode else ('normal' if is_normal else 'abnormal')
        self.cache = [self._load(v[1]) for v in tqdm(vids, desc='cache ' + name, mininterval=30)]

    def _load(self, paths):
        assert len(paths) == 10, "mỗi video cần đúng 10 crop, thấy %d: %s" % (len(paths), paths[0])
        crops = [np.asarray(np.load(p, allow_pickle=True), dtype=np.float32) for p in paths]
        T = min(c.shape[0] for c in crops)
        crops = [c[:T] for c in crops]
        if self.test_mode:
            return np.stack(crops, 1).astype(np.float16)                                # (T, 10, F)
        return np.array([process_feat(c, 32) for c in crops], dtype=np.float16)       # (10, 32, F)

    def __getitem__(self, index):
        x = torch.from_numpy(self.cache[index].astype(np.float32))
        return x if self.test_mode else (x, self.get_label())

    def get_label(self):
        return torch.tensor(0.0) if self.is_normal else torch.tensor(1.0)

    def __len__(self):
        return len(self.cache)
