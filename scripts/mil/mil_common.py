"""Phần dùng chung của các script MIL gốc (Sultani et al., CVPR 2018): hằng số, đọc đặc trưng, mô hình."""
import glob
import os

import numpy as np
import torch.nn as nn

T_SEG = 32
DIM = 4096


def matlab_round(x):
    """MATLAB round(): làm tròn nửa ra xa số 0 (numpy mặc định làm tròn nửa về số chẵn)."""
    return np.floor(np.asarray(x, dtype=np.float64) + 0.5)


def load_features(path, dim=DIM):
    """Đọc *_C.txt -> (32, dim)."""
    arr = np.array(open(path, "r").read().split(), dtype=np.float32)
    if arr.size != T_SEG * dim:
        raise ValueError(f"{path}: có {arr.size} số, cần {T_SEG}*{dim}={T_SEG * dim}")
    return arr.reshape(T_SEG, dim)


def load_dir(path, dim=DIM):
    files = sorted(glob.glob(os.path.join(path, "*_C.txt")))
    if not files:
        raise FileNotFoundError(f"Không thấy *_C.txt trong {path}")
    return files, np.stack([load_features(f, dim) for f in files])


class MIL_Model(nn.Module):
    """Dense(512,relu)-Dropout(0.6)-Dense(32)-Dropout(0.6)-Dense(1,sigmoid); glorot_normal; bias = 0."""
    def __init__(self, input_dim=DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 1)
        self.drop = nn.Dropout(0.6)
        for m in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):                                  # x: (B, 32, dim) -> (B, 32)
        import torch
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(self.fc2(x))                          # lớp 32 không có hàm kích hoạt, giống bản gốc
        return torch.sigmoid(self.fc3(x)).squeeze(-1)

    def l2_penalty(self, c=1e-3):                          # Keras W_regularizer=l2(0.001): c * sum(W^2) mỗi lớp
        return c * sum((m.weight ** 2).sum() for m in (self.fc1, self.fc2, self.fc3))