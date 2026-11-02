#!/usr/bin/env python3
"""
<- Training_AnomalyDetector_public.py (MIL gốc, Keras/Theano) chuyển sang PyTorch.

GIỮ NGUYÊN: kiến trúc 4096-512-32-1 (dropout 0.6, glorot_normal), L2 = 0.001*sum(W^2) mỗi lớp, Adagrad lr=0.01 eps=1e-8,
batch 60 (30 bất thường + 30 bình thường), 20000 vòng, hinge loss lấy tổng theo mọi cặp (bất thường, bình thường),
sparsity/smoothness 8e-5 chỉ tính trên video BẤT THƯỜNG và lấy TỔNG.
Khác duy nhất về cách chạy: nạp sẵn toàn bộ đặc trưng vào RAM thay vì đọc lại file mỗi vòng (kết quả không đổi).

Chạy:
  python scripts/mil_train.py --train_abnormal data/features_c3d/Train/Abnormal \
      --train_normal data/features_c3d/Train/Normal --out_dir results/mil_sultani
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.optim as optim

from mil_common import DIM, MIL_Model, load_dir


def custom_objective(s_abn, s_nor, model, lam1=8e-5, lam2=8e-5):
    """Dịch từng dòng của custom_objective (Theano). s_abn, s_nor: (n_exp, 32)."""
    max_a = s_abn.max(dim=1)[0]                              # (n_exp,)
    max_n = s_nor.max(dim=1)[0]
    # với MỖI video bình thường i: tổng theo MỌI video bất thường j của max(0, 1 - maxA_j + maxN_i); rồi lấy trung bình theo i
    hinge = torch.clamp(1 - max_a[None, :] + max_n[:, None], min=0).sum(dim=1).mean()
    sparsity = s_abn.sum()                                   # sub_sum_l1[:n_exp] rồi T.sum -> chỉ video bất thường, TỔNG
    smooth = ((s_abn[:, 1:] - s_abn[:, :-1]) ** 2).sum()     # sub_l2[:n_exp] rồi T.sum -> chỉ video bất thường, TỔNG
    return hinge + lam1 * sparsity + lam2 * smooth + model.l2_penalty()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_abnormal", required=True)
    ap.add_argument("--train_normal", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dim", type=int, default=DIM)
    ap.add_argument("--batch_size", type=int, default=60)
    ap.add_argument("--num_iters", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=1000)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    _, Xa = load_dir(args.train_abnormal, args.dim)
    _, Xn = load_dir(args.train_normal, args.dim)
    print(f"Tổng số video Abnormal: {len(Xa)} | Normal: {len(Xn)}")
    n_exp = args.batch_size // 2
    if len(Xa) < n_exp or len(Xn) < n_exp:
        raise ValueError(f"Mỗi lớp cần >= {n_exp} video")

    model = MIL_Model(args.dim).to(device)
    opt = optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8)   # Keras Adagrad(lr=0.01, epsilon=1e-8)

    loss_history, t0 = [], time.time()
    print("Bắt đầu huấn luyện...")
    for it in range(1, args.num_iters + 1):
        model.train()
        xa = torch.from_numpy(Xa[np.random.choice(len(Xa), n_exp, replace=False)]).to(device)
        xn = torch.from_numpy(Xn[np.random.choice(len(Xn), n_exp, replace=False)]).to(device)
        loss = custom_objective(model(xa), model(xn), model)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(loss.item())

        if it % args.log_every == 0:
            print(f"Iteration [{it}/{args.num_iters}] - Loss: {loss.item():.4f} - Thời gian: {time.time() - t0:.2f}s")
            t0 = time.time()
        if it % args.save_every == 0:
            p = os.path.join(args.out_dir, f"weights_MIL_{it}.pth")
            torch.save(model.state_dict(), p)
            print(f"-> Đã lưu mô hình tại: {p}")
    np.save(os.path.join(args.out_dir, "loss_history.npy"), np.array(loss_history))
    print("Hoàn tất huấn luyện!")


if __name__ == "__main__":
    main()