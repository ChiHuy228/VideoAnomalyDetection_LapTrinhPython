
import argparse
import csv
import glob
import os

import numpy as np

from mil_common import T_SEG, matlab_round


def read_binary_blob(path):
    """Đọc file .fc6-1 của C3D: 5 số int32 (num,channels,length,height,width) rồi float32.
    (theo hàm read_binary_blob.m của C3D)"""
    with open(path, "rb") as f:
        s = np.fromfile(f, dtype=np.int32, count=5)
        m = int(np.prod(s))
        data = np.fromfile(f, dtype=np.float32, count=m)
    if data.size != m:
        raise ValueError(f"{path}: header báo {m} số nhưng chỉ đọc được {data.size}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3d_dir", required=True, help="thư mục chứa các thư mục con (mỗi video một thư mục *.fc6-1)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--counts_csv", default=None, help="ghi số file C3D mỗi video (cần cho mil_eval)")
    args = ap.parse_args()

    folders = sorted(d for d in glob.glob(os.path.join(args.c3d_dir, "*")) if os.path.isdir(d))
    if not folders:
        raise FileNotFoundError(f"Không thấy thư mục video con nào trong {args.c3d_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    counts = []
    for fol in folders:
        name = os.path.basename(fol)
        files = sorted(glob.glob(os.path.join(fol, "*.fc6-1")))
        if not files:
            print(f"[cảnh báo] {name}: không có file .fc6-1, bỏ qua")
            continue
        F = np.stack([read_binary_blob(f) for f in files])               # (n, 4096)
        if F.sum() == 0 or (F.sum(1) == 0).any() or not np.isfinite(F).all():
            raise ValueError(f"{name}: đặc trưng toàn 0 / NaN / Inf")    # giống các lệnh error('??') gốc
        n = len(files)
        shots = matlab_round(np.linspace(1, n, T_SEG + 1)).astype(int)
        seg = np.zeros((T_SEG, F.shape[1]), np.float32)
        for i in range(T_SEG):
            ss, ee = shots[i], shots[i + 1] - 1                          # giống bản gốc (ee = đầu đoạn sau - 1)
            v = F[ss - 1] if ee <= ss else F[ss - 1: ee].mean(axis=0)
            nv = np.linalg.norm(v)
            if nv == 0:
                raise ValueError(f"{name}: vector đoạn {i} có norm 0")
            seg[i] = v / nv                                              # chuẩn hóa L2 từng đoạn
        with open(os.path.join(args.out_dir, name + "_C.txt"), "w") as f:
            for row in seg:                                              # '%f ' mỗi số, mỗi đoạn một dòng (như MATLAB)
                f.write("".join(f"{v:f} " for v in row) + "\n")
        counts.append((name, n))

    if args.counts_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.counts_csv)), exist_ok=True)
        with open(args.counts_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video", "num_c3d_feats"])
            w.writerows(counts)
    print(f"Đã ghi {len(counts)} video vào {args.out_dir}")


if __name__ == "__main__":
    main()