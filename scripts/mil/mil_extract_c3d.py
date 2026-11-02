#!/usr/bin/env python3
"""
Bước 1 của pipeline MIL gốc: video -> đặc trưng C3D fc6 cho MỖI đoạn 16 khung hình liên tiếp (không chồng lấn).
Đầu ra đúng cấu trúc mà mil_features.py đọc:

  data/c3d_raw/Train/Abnormal/<tên_video>/000000.fc6-1, 000016.fc6-1, ...
  data/c3d_raw/Train/Normal/<tên_video>/...
  data/c3d_raw/Test/<tên_video>/...

Mỗi file .fc6-1 = header 5 số int32 (1,4096,1,1,1) + 4096 số float32.

Tiền xử lý THEO C3D Caffe GỐC (cái mà Sultani et al. dùng, extract_image_features):
  1. khung BGR (đúng thứ tự OpenCV/Caffe, KHÔNG đổi sang RGB)
  2. resize về 128 (cao) x 171 (rộng)
  3. trừ mean cube Sports-1M (16 x 128 x 171, sport1m_train16_128_mean) tại từng vị trí
  4. crop GIỮA 112x112 (offset cao 8, rộng 29, như Caffe center crop)
  5. lấy blob fc6-1 (sau ReLU chạy tại chỗ), bỏ phần dư < 16 khung cuối video

Mean cube: cần file .npy shape (3,16,128,171), kênh theo thứ tự BGR, đơn vị 0-255
(chuyển từ sport1m_train16_128_mean.binaryproto). Không có --mean thì script chỉ cảnh báo và chạy không trừ mean
(KHÔNG còn đúng tác giả).

Trọng số: c3d.pickle (state_dict torch của DavideA/c3d-pytorch, chuyển từ Caffe c3d_sports1m).

Chạy thử 2 video:
  python scripts/mil/mil_extract_c3d.py --videos data/UCF_Crime --list data/annotations/Anomaly_Test.txt \
      --split Test --out_dir data/c3d_raw --weights weights/c3d.pickle --mean weights/c3d_mean.npy --limit 2
Rồi chạy đủ (bỏ --limit) cho Anomaly_Test.txt (--split Test) và Anomaly_Train.txt (--split Train).
Chạy lại được sau khi bị ngắt: video đã xong (có file DONE) sẽ được bỏ qua.
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

EXTS = (".mp4", ".avi", ".mkv", ".mov")
CLIP = 16
H, W = 128, 171          # kích thước resize của C3D gốc
CROP = 112
OFF_H = (H - CROP) // 2  # 8
OFF_W = (W - CROP) // 2  # 29


class C3D(nn.Module):
    """C3D tới hết fc6 (bỏ fc7, fc8). Tên lớp trùng repo DavideA/c3d-pytorch để nạp được c3d.pickle."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv4a = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv5a = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        # prototxt: pool5 kernel 2, stride 2, KHÔNG pad; Caffe làm tròn lên (ceil) nên 7 -> 4 với cửa sổ [0,1],[2,3],[4,5],[6]
        self.pool5 = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.fc6 = nn.Linear(8192, 4096)
        self.relu = nn.ReLU()

    def forward(self, x):                                   # x: (B, 3, 16, 112, 112)
        r = self.relu
        h = self.pool1(r(self.conv1(x)))
        h = self.pool2(r(self.conv2(h)))
        h = self.pool3(r(self.conv3b(r(self.conv3a(h)))))
        h = self.pool4(r(self.conv4b(r(self.conv4a(h)))))
        h = self.pool5(r(self.conv5b(r(self.conv5a(h)))))
        return r(self.fc6(h.reshape(h.size(0), -1)))        # (B, 4096)


def build_model(weights, device, half):
    model = C3D()
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    res = model.load_state_dict(sd, strict=False)           # fc7/fc8 trong file sẽ bị bỏ qua
    if res.missing_keys:
        raise RuntimeError(f"File trọng số thiếu các khóa: {res.missing_keys}. Kiểm tra lại định dạng c3d.pickle.")
    model = model.to(device).eval()
    return model.half() if half else model


def load_mean(path):
    if not path:
        print("[cảnh báo] không có --mean: chạy KHÔNG trừ mean, đặc trưng sẽ lệch so với tác giả")
        return None
    m = np.load(path).astype(np.float32)
    if m.shape != (3, CLIP, H, W):
        raise ValueError(f"mean phải có shape (3,{CLIP},{H},{W}), nhận {m.shape}")
    return m


def prep(frame_bgr):
    """Giữ nguyên BGR, resize về 128x171 (cao x rộng). Trả về (128,171,3) float32 0-255."""
    return cv2.resize(frame_bgr, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def make_clip(buf, mean):
    """16 khung (128,171,3) -> (3,16,112,112): trừ mean cube rồi crop giữa (như Caffe)."""
    c = np.stack(buf).transpose(3, 0, 1, 2)                 # (3,16,128,171)
    if mean is not None:
        c = c - mean
    return np.ascontiguousarray(c[:, :, OFF_H:OFF_H + CROP, OFF_W:OFF_W + CROP])


@torch.no_grad()
def run_batch(model, clips, device, half):
    x = torch.from_numpy(np.stack(clips)).to(device)         # (B, 3, 16, 112, 112)
    if half:
        x = x.half()
    return model(x).float().cpu().numpy()


def extract_video(path, model, mean, device, batch, half):
    """Giải mã tuần tự, cắt thành các clip 16 khung hình liên tiếp (bỏ phần dư < 16, như C3D gốc)."""
    cap = cv2.VideoCapture(path)
    buf, pending, feats, last = [], [], [], None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        last = prep(fr)
        buf.append(last)
        if len(buf) == CLIP:
            pending.append(make_clip(buf, mean))
            buf = []
            if len(pending) == batch:
                feats.append(run_batch(model, pending, device, half))
                pending = []
    cap.release()
    if last is None:
        raise RuntimeError("không giải mã được khung hình nào")
    if not feats and not pending:                                 # video ngắn hơn 16 khung: lặp khung cuối cho đủ 1 clip
        buf += [last] * (CLIP - len(buf))
        pending.append(make_clip(buf, mean))
    if pending:
        feats.append(run_batch(model, pending, device, half))
    return np.concatenate(feats)                                  # (n_clip, 4096)


def write_blob(path, vec):
    with open(path, "wb") as f:
        np.array([1, vec.size, 1, 1, 1], dtype=np.int32).tofile(f)
        vec.astype(np.float32).tofile(f)


def name_of(path):
    return os.path.splitext(os.path.basename(path.strip().split()[0]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="thư mục gốc chứa video (tìm đệ quy)")
    ap.add_argument("--list", required=True, help="Anomaly_Train.txt hoặc Anomaly_Test.txt")
    ap.add_argument("--split", choices=["Train", "Test"], required=True)
    ap.add_argument("--out_dir", required=True, help="thường là data/c3d_raw")
    ap.add_argument("--weights", required=True, help="c3d.pickle (Sports-1M, từ DavideA/c3d-pytorch)")
    ap.add_argument("--mean", default=None, help="mean cube .npy shape (3,16,128,171), BGR, 0-255")
    ap.add_argument("--batch", type=int, default=16, help="số clip mỗi lần chạy mạng")
    ap.add_argument("--half", action="store_true", help="dùng fp16 (nhanh hơn, đặc trưng lệch nhẹ, không đúng tác giả)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N video đầu (để thử)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[cảnh báo] không có GPU, chạy bằng CPU sẽ rất chậm")
    half = args.half and device.type == "cuda"
    model = build_model(args.weights, device, half)
    mean = load_mean(args.mean)

    found = {}
    for f in glob.glob(os.path.join(args.videos, "**", "*"), recursive=True):
        if f.lower().endswith(EXTS):
            found.setdefault(name_of(f), f)
    wanted = [name_of(l) for l in open(args.list) if l.strip()]
    missing = [w for w in wanted if w not in found]
    if missing:
        print(f"[cảnh báo] {len(missing)} video trong danh sách không thấy trong {args.videos}, ví dụ: {missing[:3]}")
    todo = [(w, found[w]) for w in wanted if w in found]
    if args.limit:
        todo = todo[: args.limit]
    print(f"device {device} | {args.split} | {len(todo)} video")

    for k, (name, path) in enumerate(todo, 1):
        if args.split == "Train":
            sub = "Normal" if name.lower().startswith("normal") else "Abnormal"
            vdir = os.path.join(args.out_dir, "Train", sub, name)
        else:
            vdir = os.path.join(args.out_dir, "Test", name)
        if os.path.exists(os.path.join(vdir, "DONE")):
            continue                                              # chạy lại được sau khi bị ngắt
        os.makedirs(vdir, exist_ok=True)
        t0 = time.time()
        try:
            feats = extract_video(path, model, mean, device, args.batch, half)
        except Exception as e:
            print(f"[lỗi] {name}: {e}")
            continue
        for i, v in enumerate(feats):
            write_blob(os.path.join(vdir, f"{i * CLIP:06d}.fc6-1"), v)
        open(os.path.join(vdir, "DONE"), "w").close()
        print(f"  {k}/{len(todo)} {name}: {len(feats)} clip, {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()