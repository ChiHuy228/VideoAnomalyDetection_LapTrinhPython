import numpy as np

class Visualizer(object):
    def __init__(self, *a, **k): pass
    def plot_lines(self, *a, **k): pass
    def lines(self, *a, **k): pass
    def disp_image(self, *a, **k): pass
    def scatter(self, *a, **k): pass

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=int)
    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat

def save_best_record(test_info, file_path):
    with open(file_path, "w") as fo:
        fo.write("epoch: {}\n".format(test_info["epoch"][-1]))
        fo.write(str(test_info["test_AUC"][-1]))
