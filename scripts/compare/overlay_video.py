"""
Vẽ điểm dự đoán của các phương pháp (kể cả EVT-MIL) lên VIDEO gốc, xuất mp4.
Dữ liệu giống plot_single_video.py / calc_auc.py:
    results/<method>/video_scores.npz, results/evtmil/<tag>/video_scores_<p|d>.npz, gt-ucf.npy

Chạy từ thư mục HuanLuyen:
    python scripts/compare/overlay_video.py --video_name Arson035
    python scripts/compare/overlay_video.py --video_index 14 --video_dir ~/UCF_Crime
    python scripts/compare/overlay_video.py --video_name Arson035 --max_frames 300      # chạy thử nhanh
    python scripts/compare/overlay_video.py --video_name Arson035 --only "URDMU,VadCLIP,EVT tagA p"
    python scripts/compare/overlay_video.py --video_name Arson035 --no-evt --h264
"""
import os, re, glob, argparse, shutil, subprocess, itertools
import numpy as np
import cv2

ALL_METHODS = {'MIL':     'results/mil_sultani/video_scores.npz',
               'RTFM':    'results/rtfm/video_scores.npz',
               'URDMU':   'results/urdmu/video_scores.npz',
               'VadCLIP': 'results/vadclip/video_scores.npz'}
GT      = 'scripts/urdmu/frame_label/gt-ucf.npy'
COLORS  = {'MIL': '#7f7f7f', 'RTFM': '#2ca02c', 'URDMU': '#1f4e9c', 'VadCLIP': '#030303'}
# màu cho EVT-MIL (tránh trùng xanh lá của RTFM và cam của vùng GT)
EXTRA_COLORS = ['#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#ff9896', '#aec7e8', '#c5b0d5']
FONT    = cv2.FONT_HERSHEY_SIMPLEX

norm = lambda k: re.sub(r'(_x264)?(_i3d)?$', '', k)


def hex2bgr(h):
    h = h.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def resample(s, L):
    s = np.asarray(s, dtype=np.float64)
    if len(s) == L: return s
    if len(s) == 1: return np.full(L, s[0])
    return np.interp(np.linspace(0, len(s) - 1, L), np.arange(len(s)), s)


def runs(y):
    """Các đoạn [start, end) liên tiếp có nhãn bất thường (y == 1)."""
    d = np.diff(np.concatenate([[0], (np.asarray(y) > 0.5).astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def find_video(video_dir, name):
    for pat in ('%s_x264.mp4', '%s.mp4'):
        hits = glob.glob(os.path.join(video_dir, '**', pat % name), recursive=True)
        if hits:
            return hits[0]
    return None


def darken(img, x0, y0, x1, y1, k=0.4):
    x1 = min(x1, img.shape[1]); y1 = min(y1, img.shape[0])
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (sub * k).astype(np.uint8)


def build_panel(W, P, y, L, thr, fs, plot_h=150):
    """Vẽ sẵn phần đồ thị tĩnh (GT + các đường điểm). Chú thích tự xuống hàng,
    chiều cao panel tự tính. Trả về (ảnh panel, khung vẽ)."""
    ml, mr, mb = 52, 16, 20
    items = [('GT', (0, 165, 255))] + [(m, hex2bgr(COLORS.get(m, '#000000'))) for m in P]

    pos, xl, row = [], ml, 0                            # bố trí chú thích
    for lab, col in items:
        tw = cv2.getTextSize(lab, FONT, fs, 1)[0][0]
        if xl + 28 + tw > W - mr and xl > ml:
            xl, row = ml, row + 1
        pos.append((lab, col, xl, row, tw))
        xl += 28 + tw + 22
    mt = 12 + 22 * (row + 1)
    PH = mt + plot_h + mb
    PH += PH % 2                                        # giữ chiều cao chẵn cho video
    x0, x1, y0, y1 = ml, W - mr, mt, PH - mb

    panel = np.full((PH, W, 3), 255, np.uint8)
    ov = panel.copy()                                   # vùng GT bất thường
    for s, e in runs(y):
        xs = x0 + int(s * (x1 - x0) / L); xe = x0 + int(e * (x1 - x0) / L)
        cv2.rectangle(ov, (xs, y0), (xe, y1), (0, 165, 255), -1)
    panel = cv2.addWeighted(ov, 0.28, panel, 0.72, 0)

    for v in (0.0, 0.5, 1.0):                           # lưới + nhãn trục
        yy = y1 - int(v * (y1 - y0))
        cv2.line(panel, (x0, yy), (x1, yy), (215, 215, 215), 1)
        cv2.putText(panel, '%.1f' % v, (8, yy + 4), FONT, fs * 0.8, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (x0, y0), (x1, y1), (150, 150, 150), 1)

    yt = y1 - int(thr * (y1 - y0))                      # ngưỡng (nét đứt)
    for xx in range(x0, x1, 10):
        cv2.line(panel, (xx, yt), (min(xx + 5, x1), yt), (120, 120, 120), 1)

    cols = np.arange(x1 - x0 + 1)                       # các đường điểm
    idx = (cols * (L - 1) / max(x1 - x0, 1)).astype(int)
    for m, s in P.items():
        ys = y1 - (np.clip(s[idx], 0, 1) * (y1 - y0)).astype(int)
        pts = np.stack([x0 + cols, ys], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(panel, [pts], False, hex2bgr(COLORS.get(m, '#000000')), 2, cv2.LINE_AA)

    for lab, col, lx, r, tw in pos:                     # chú thích
        ly = 14 + 22 * r
        cv2.line(panel, (lx, ly), (lx + 22, ly), col, 3 if lab != 'GT' else 8)
        cv2.putText(panel, lab, (lx + 28, ly + 5), FONT, fs, (40, 40, 40), 1, cv2.LINE_AA)
    return panel, (x0, x1, y0, y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_name', default='', help='một phần tên video, vd: Arson035')
    ap.add_argument('--video_index', type=int, default=10, help='thứ tự video (0..289) theo URDMU')
    ap.add_argument('--video_dir', default='~/UCF_Crime', help='thư mục chứa video gốc')
    ap.add_argument('--out_dir', default='results/compare/video_overlay')
    ap.add_argument('--thr', type=float, default=0.5, help='ngưỡng vẽ trên đồ thị và thanh điểm')
    ap.add_argument('--width', type=int, default=960, help='bề rộng video xuất ra')
    ap.add_argument('--max_frames', type=int, default=0, help='chỉ xuất N frame đầu (chạy thử)')
    ap.add_argument('--h264', action='store_true', help='mã hóa lại bằng ffmpeg (libx264) để phát được ở mọi nơi')
    ap.add_argument('--evt-glob', default='results/evtmil/*/video_scores_*.npz',
                    help='mẫu đường dẫn các file điểm EVT-MIL')
    ap.add_argument('--no-evt', action='store_true', help='không vẽ EVT-MIL')
    ap.add_argument('--only', default='',
                    help='chỉ vẽ các phương pháp này (tên đúng, cách nhau dấu phẩy), vd: "URDMU,VadCLIP,EVT tagA p"')
    ap.add_argument('--list', action='store_true', help='in danh sách video rồi thoát')
    args = ap.parse_args()

    # ---------- Gom phương pháp (giống calc_auc.py) ----------
    all_methods = dict(ALL_METHODS)
    if not args.no_evt:
        for f in sorted(glob.glob(args.evt_glob)):
            tag = os.path.basename(os.path.dirname(f))
            which = re.sub(r'^video_scores_|\.npz$', '', os.path.basename(f))
            all_methods['EVT %s %s' % (tag, which)] = f

    methods = {m: p for m, p in all_methods.items() if os.path.exists(p)}
    for m in all_methods:
        if m not in methods:
            print('Bỏ qua (chưa có file):', m)
    assert 'URDMU' in methods, 'Cần có URDMU làm tham chiếu độ dài'

    extra = itertools.cycle(EXTRA_COLORS)
    for m in methods:
        if m not in COLORS:
            COLORS[m] = next(extra)

    S = {m: {norm(k): v for k, v in np.load(p).items()} for m, p in methods.items()}
    ref = S['URDMU']
    names = list(ref)
    if args.list:
        for i, n in enumerate(names):
            print(i, n)
        return

    # ---------- Chọn video ----------
    idx = args.video_index
    if args.video_name:
        key = norm(args.video_name).lower()
        hit = [i for i, n in enumerate(names) if n.lower() == key] or \
              [i for i, n in enumerate(names) if key in n.lower()]
        if hit:
            idx = hit[0]
        else:
            print("Không thấy video chứa '%s', dùng index %d" % (args.video_name, idx))
    assert 0 <= idx < len(names), 'index ngoài khoảng 0..%d' % (len(names) - 1)
    vname = names[idx]

    # ---------- GT + điểm từng phương pháp ----------
    gt = np.load(GT).ravel()
    start = sum(len(ref[n]) for n in names[:idx])
    L = len(ref[vname])
    y = gt[start:start + L]
    assert len(y) == L, 'GT ngắn hơn dự kiến, kiểm tra GT path'

    only = [s.strip() for s in args.only.split(',') if s.strip()]
    P = {}
    for m in methods:
        if only and m not in only:
            continue
        if vname not in S[m]:
            print('!!! %s thiếu video %s -> bỏ qua' % (m, vname))
            continue
        P[m] = resample(S[m][vname], L)
    assert P, 'Không có phương pháp nào để vẽ (kiểm tra --only)'
    print('Phương pháp được vẽ:', list(P))

    # ---------- Mở video ----------
    vdir = os.path.expanduser(args.video_dir)
    vpath = find_video(vdir, vname)
    assert vpath, 'Không thấy video %s_x264.mp4 trong %s' % (vname, vdir)
    cap = cv2.VideoCapture(vpath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Video: %s | %d frame | %.1f fps | điểm/GT dài %d frame' % (vpath, n_frames, fps, L))
    if abs(n_frames - L) > 32:
        print('!!! Số frame video (%d) lệch nhiều so với độ dài điểm (%d): điểm sẽ được co giãn theo tỉ lệ' % (n_frames, L))

    W = args.width - args.width % 2
    H = int(h0 * W / w0); H -= H % 2
    fs = W / 1280 * 0.6 + 0.2
    panel, (px0, px1, py0, py1) = build_panel(W, P, y, L, args.thr, fs * 0.9)
    PH = panel.shape[0]

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, '%s_overlay.mp4' % vname)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H + PH))

    # kích thước thanh điểm trên video
    lh = int(26 * fs / 0.8)
    name_w = max(cv2.getTextSize(m, FONT, fs, 1)[0][0] for m in P) + 10
    bar_w = int(W * 0.14)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        j = min(int(i * L / max(n_frames, 1)), L - 1)
        canvas = np.empty((H + PH, W, 3), np.uint8)
        canvas[:H] = cv2.resize(frame, (W, H))
        canvas[H:] = panel

        gt_on = y[j] > 0.5
        if gt_on:
            cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (0, 0, 255), 6)

        # HUD: tiêu đề + thanh điểm từng phương pháp
        box_w = name_w + bar_w + 70
        box_h = (len(P) + 1) * lh + 10
        darken(canvas, 6, 6, 6 + box_w, min(6 + box_h, H))
        head = 'frame %d/%d   GT: %s' % (i + 1, n_frames, 'ANOMALY' if gt_on else 'normal')
        cv2.putText(canvas, head, (12, 6 + lh - 6), FONT, fs,
                    (0, 0, 255) if gt_on else (120, 255, 120), 1, cv2.LINE_AA)
        for r, (m, s) in enumerate(P.items()):
            yy = 6 + (r + 2) * lh - 6
            col = hex2bgr(COLORS.get(m, '#000000'))
            bcol = (255, 255, 255) if col == (3, 3, 3) else col      # màu đen khó thấy trên nền tối
            cv2.putText(canvas, m, (12, yy), FONT, fs, (255, 255, 255), 1, cv2.LINE_AA)
            bx = 12 + name_w
            cv2.rectangle(canvas, (bx, yy - lh + 10), (bx + bar_w, yy + 2), (255, 255, 255), 1)
            cv2.rectangle(canvas, (bx, yy - lh + 10),
                          (bx + int(bar_w * float(np.clip(s[j], 0, 1))), yy + 2), bcol, -1)
            tx = bx + int(bar_w * args.thr)                         # vạch ngưỡng
            cv2.line(canvas, (tx, yy - lh + 6), (tx, yy + 6), (0, 255, 255), 1)
            cv2.putText(canvas, '%.2f' % s[j], (bx + bar_w + 6, yy), FONT, fs,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # con trỏ thời gian + chấm điểm trên đồ thị
        cx = px0 + int(j * (px1 - px0) / L)
        cv2.line(canvas, (cx, H + py0), (cx, H + py1), (0, 0, 200), 1)
        for m, s in P.items():
            cy = H + py1 - int(float(np.clip(s[j], 0, 1)) * (py1 - py0))
            cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(canvas, (cx, cy), 4, hex2bgr(COLORS.get(m, '#000000')), -1)

        writer.write(canvas)
        i += 1
        if i % 200 == 0:
            print('  %d/%d frame' % (i, n_frames))

    cap.release(); writer.release()
    print('Đã lưu:', out)

    if args.h264:
        if shutil.which('ffmpeg'):
            out2 = out.replace('.mp4', '_h264.mp4')
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out, '-c:v', 'libx264',
                            '-pix_fmt', 'yuv420p', out2], check=True)
            print('Đã lưu (H.264):', out2)
        else:
            print('Không có ffmpeg trong PATH, bỏ qua --h264')


if __name__ == '__main__':
    main()