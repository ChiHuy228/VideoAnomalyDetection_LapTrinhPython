#!/usr/bin/env python3
"""
<- Test_Anomaly_Detector_public.py (MIL gốc).
Mỗi video test (*_C.txt, 32 đoạn) -> 32 điểm bất thường, ghi thành <tên>_C.mat với khóa 'predictions' (32,1).

Chạy:
  python scripts/mil_test.py --test_dir data/features_c3d/Test \
      --weights results/mil_sultani/weights_MIL_20000.pth --out_dir results/mil_sultani/scores
"""
import argparse
import glob
import os

import numpy as np
import scipy.io as sio
import torch

from mil_common import DIM, MIL_Model, load_features


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dim", type=int, default=DIM)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    model = MIL_Model(args.dim).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    files = sorted(glob.glob(os.path.join(args.test_dir, "*_C.txt")))
    if not files:
        raise FileNotFoundError(f"Không thấy *_C.txt trong {args.test_dir}")
    for f in files:
        x = torch.from_numpy(load_features(f, args.dim))[None].to(device)
        pred = model(x)[0].cpu().numpy().astype(np.float32)              # (32,)
        name = os.path.basename(f)[: -len(".txt")]                       # 'Abuse028_x264_C' (giống aa[0:-4] gốc)
        sio.savemat(os.path.join(args.out_dir, name + ".mat"), {"predictions": pred.reshape(-1, 1)})
    print(f"Đã ghi {len(files)} file .mat vào {args.out_dir}")


if __name__ == "__main__":
    main()