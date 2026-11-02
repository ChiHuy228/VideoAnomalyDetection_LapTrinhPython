import torch
import numpy as np
import random

class Visualizer(object):
    """Bản rỗng thay cho visdom (Kaggle không có server visdom)."""
    def __init__(self, *a, **k): pass
    def plot_lines(self, *a, **k): pass
    def lines(self, *a, **k): pass
    def disp_image(self, *a, **k): pass
    def scatter(self, *a, **k): pass

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_perturb(feature_len, length):
    r = np.linspace(0, feature_len, length + 1, dtype=np.uint16)
    return r

def norm(data):
    l2 = torch.norm(data, p=2, dim=-1, keepdim=True)
    return torch.div(data, l2)

def save_best_record(test_info, file_path):
    with open(file_path, "w") as fo:
        fo.write("Step: {}\n".format(test_info["step"][-1]))
        fo.write("auc: {:.4f}\n".format(test_info["auc"][-1]))
        fo.write("ap: {:.4f}\n".format(test_info["ap"][-1]))
        fo.write("ac: {:.4f}\n".format(test_info["ac"][-1]))