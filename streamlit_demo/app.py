
import os
import re
import sys
import glob
import time
import importlib
import pickle
from contextlib import contextmanager
from types import SimpleNamespace

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

st.set_page_config(page_title="Video Anomaly Detection (End-to-End)", layout="wide")
st.title("Hệ thống Nhận diện Bất thường Video (Không dùng Dataset)")

# ==========================================
# 1. ĐƯỜNG DẪN CƠ SỞ (TỰ ĐỘNG ĐỊNH VỊ)
# ==========================================
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(DEMO_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

MODEL_DIR = os.path.join(DEMO_DIR, "model")
SCRIPTS_DIR = os.path.join(SRC_DIR, "scripts")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")    

MIL_DIR = os.path.join(SCRIPTS_DIR, "mil")
EVT_DIR = os.path.join(SCRIPTS_DIR, "evtmil")
RTFM_DIR = os.path.join(SCRIPTS_DIR, "rtfm")
URDMU_DIR = os.path.join(SCRIPTS_DIR, "urdmu")
VADCLIP_DIR = os.path.join(SCRIPTS_DIR, "vadclip", "src")
if not os.path.isfile(os.path.join(VADCLIP_DIR, "model.py")):
    VADCLIP_DIR = os.path.join(SCRIPTS_DIR, "vadclip")

# Thư mục lưu trữ tạm thời trong quá trình hệ thống chạy
FEAT_DIR = os.path.join(DEMO_DIR, "temp_features")
VIDEO_DIR = os.path.join(DEMO_DIR, "temp_videos")
for _d in (FEAT_DIR, VIDEO_DIR):
    os.makedirs(_d, exist_ok=True)

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

MODELS = {
    "EVTMIL":      os.path.join(MODEL_DIR, "evtmil", "best.pt"),
    "MIL Sultani": os.path.join(MODEL_DIR, "mil_sultani", "weights_MIL_20000.pth"),
    "RTFM":        os.path.join(MODEL_DIR, "rtfm", "rtfm5600-i3d.pkl"),
    "URDMU":       os.path.join(MODEL_DIR, "urdmu", "ucf_trans_2022.pkl"),
    "VADCLIP":     os.path.join(MODEL_DIR, "vadclip", "model_ucf.pth"),
}
FEAT_BY_DIM = {4096: "c3d", 2048: "i3d", 1024: "i3d", 512: "clip"}
UCF_CLASSES = ['Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion',
               'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']

# ==========================================
# 2. CÁC HÀM TIỆN ÍCH CORE
# ==========================================
@contextmanager
def repo_context(repo_dir, mod_names):
    if not os.path.isdir(repo_dir): raise FileNotFoundError(f"Không thấy: {repo_dir}")
    mine = lambda n: any(n == m or n.startswith(m + ".") for m in mod_names)
    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if mine(n)}
    sys.path.insert(0, repo_dir)
    try: yield
    finally:
        try: sys.path.remove(repo_dir)
        except ValueError: pass
        for n in list(sys.modules):
            if mine(n): sys.modules.pop(n, None)
        sys.modules.update(saved)

def load_ckpt(path):
    if not os.path.isfile(path): raise FileNotFoundError(f"Không thấy file trọng số: {path}")
    try: return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError: return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as f: return pickle.load(f)

def find_sd(obj, markers):
    if isinstance(obj, nn.Module): obj = obj.state_dict()
    if isinstance(obj, dict):
        clean = {(k.replace("module.", "", 1) if isinstance(k, str) else k): v for k, v in obj.items()}
        if any(m in clean for m in markers): return clean
        for v in obj.values():
            r = find_sd(v, markers)
            if r is not None: return r
    return None

def load_strict(net, sd, ignore_prefix=()):
    try: net.load_state_dict(sd)
    except RuntimeError:
        res = net.load_state_dict(sd, strict=False)
        bad = [k for k in list(res.missing_keys) + list(res.unexpected_keys) if not any(k.startswith(p) for p in ignore_prefix)]
        if bad: raise RuntimeError(f"Trọng số không khớp kiến trúc: {bad[:4]}")

def norm_video(name):
    s = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r'(_x264)?(_i3d)?$', '', re.sub(r'__\d+$', '', s))

# ==========================================
# 3. LÕI TIỀN XỬ LÝ (EXTRACTORS) TỰ ĐỘNG
# ==========================================
# 3.1. TRÍCH XUẤT C3D (MIL Sultani)
@st.cache_resource(show_spinner=False)
def get_c3d():
    with repo_context(MIL_DIR, ["mil_extract_c3d", "mil_common", "mil_eval"]):
        m = importlib.import_module("mil_extract_c3d")
    w = os.path.join(WEIGHTS_DIR, "c3d.pickle")
    mp = os.path.join(WEIGHTS_DIR, "c3d_mean.npy")
    net = m.build_model(w, DEVICE, False)
    mean = m.load_mean(mp if os.path.isfile(mp) else None)
    return m, net, mean

def extract_c3d(video_path):
    m, net, mean = get_c3d()
    feats = m.extract_video(video_path, net, mean, DEVICE, 16, False) 
    return feats[None].astype(np.float32)

# 3.2. TRÍCH XUẤT CLIP (EVTMIL, VADCLIP)
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

def _clip_prep(rgb):
    h, w = rgb.shape[:2]
    s = 224.0 / min(h, w)
    nh, nw = max(224, int(round(h * s))), max(224, int(round(w * s)))
    im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    t, l = (nh - 224) // 2, (nw - 224) // 2
    im = im[t:t + 224, l:l + 224].astype(np.float32) / 255.0
    return ((im - _CLIP_MEAN) / _CLIP_STD).transpose(2, 0, 1)

def extract_clip(video_path):
    cm = load_model("VADCLIP")["net"].clipmodel
    dt = next(cm.parameters()).dtype
    cap, frames, i = cv2.VideoCapture(video_path), [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % 16 == 0: frames.append(_clip_prep(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    if not frames: raise RuntimeError("Không đọc được frame nào")
    out = []
    with torch.no_grad():
        for k in range(0, len(frames), 64):
            x = torch.from_numpy(np.stack(frames[k:k + 64])).to(DEVICE).to(dt)
            out.append(cm.encode_image(x).float().cpu().numpy())
    return np.concatenate(out)[None].astype(np.float32)

# 3.3. TRÍCH XUẤT CNN THAY THẾ I3D (RTFM, URDMU)
@st.cache_resource(show_spinner=False)
def get_cnn_extractor():
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    modules = list(resnet.children())[:-1]
    return nn.Sequential(*modules).to(DEVICE).eval()

def extract_i3d(video_path):
    """Trích xuất 10-crop thay thế I3D trực tiếp từ video."""
    extractor = get_cnn_extractor()
    cap, frames, i = cv2.VideoCapture(video_path), [], 0
    transform = T.Compose([
        T.ToPILImage(), T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % 16 == 0: frames.append(transform(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    
    out = []
    with torch.no_grad():
        for k in range(0, len(frames), 32):
            batch = torch.stack(frames[k:k+32]).to(DEVICE)
            feats = extractor(batch).squeeze(-1).squeeze(-1) 
            out.append(feats.cpu().numpy())
            
    feat_2d = np.concatenate(out) # Shape: (T, 2048)
    # Sao chép thành 10-crop để phù hợp với định dạng RTFM/URDMU
    return np.repeat(feat_2d[None, ...], 10, axis=0).astype(np.float32)

EXTRACTORS = {"c3d": extract_c3d, "clip": extract_clip, "i3d": extract_i3d}

def get_crops(video_path, feat, dim):
    """Quản lý luồng lấy đặc trưng: Ưu tiên Cache -> Tự động trích xuất"""
    base = norm_video(os.path.basename(video_path))
    cache_path = os.path.join(FEAT_DIR, f"{base}_{feat}_{dim}.npy")
    
    if os.path.exists(cache_path): 
        return np.load(cache_path)
        
    st.toast(f"Đang phân tích video gốc, trích xuất đặc trưng {feat.upper()} (Dim: {dim})...")
    arr = EXTRACTORS[feat](video_path)
    
    # Cắt chính xác số chiều (ví dụ: cắt 2048 thành 1024 nếu mô hình yêu cầu)
    if arr.shape[-1] > dim: 
        arr = arr[..., :dim]
        
    np.save(cache_path, arr)
    return arr

# ==========================================
# 4. LUỒNG CHẤM ĐIỂM CHÍNH XÁC CỦA MODEL
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model(name):
    ck = load_ckpt(MODELS[name])
    
    if name == "MIL Sultani":
        with repo_context(MIL_DIR, ["mil_common", "mil_eval", "mil_extract_c3d"]):
            common = importlib.import_module("mil_common")
            try: frame_fn = importlib.import_module("mil_eval").frame_scores_matlab
            except Exception: frame_fn = None
        sd = find_sd(ck, ["fc1.weight"])
        dim = int(sd["fc1.weight"].shape[1]) if sd else common.DIM
        net = common.MIL_Model(dim)
        load_strict(net, sd)
        return dict(net=net.to(DEVICE).eval(), dim=dim, frame_fn=frame_fn)

    if name == "EVTMIL":
        with repo_context(EVT_DIR, ["evtmil_suite"]):
            evt = importlib.import_module("evtmil_suite")
        sd = find_sd(ck, ["M", "proj.weight", "head.0.weight"])
        over = {}
        variant = 2 if "proj.weight" in sd else 1
        if "M" in sd:
            over.update(mode="mem", variant=variant, K=int(sd["M"].shape[0]), dim=int(sd["M"].shape[1]))
            d0 = int(sd["proj.weight"].shape[1]) if variant == 2 else int(sd["M"].shape[1])
            cfg = SimpleNamespace(**{**evt.DEFAULTS, **over})
            net = evt.EVTMIL(cfg, d0)
        else:
            over.update(mode="nomem", dim=int(sd["proj.weight"].shape[0]), hidden=int(sd["head.0.weight"].shape[0]))
            d0 = int(sd["proj.weight"].shape[1])
            cfg = SimpleNamespace(**{**evt.DEFAULTS, **over})
            net = evt.MILHead(cfg, d0)
        load_strict(net, sd)
        return dict(net=net.to(DEVICE).eval(), dim=d0, cfg=cfg)

    if name == "RTFM":
        with repo_context(RTFM_DIR, ["model"]):
            Model = importlib.import_module("model").Model
        sd = find_sd(ck, ["Aggregate.conv_1.0.weight"])
        dim = int(sd["Aggregate.conv_1.0.weight"].shape[1])
        net = Model(dim, 32)                           
        load_strict(net, sd)
        return dict(net=net.to(DEVICE).eval(), dim=dim)

    if name == "URDMU":
        with repo_context(URDMU_DIR, ["model", "memory", "translayer", "utils", "config", "options", "dataset_loader"]):
            WSAD = importlib.import_module("model").WSAD
        sd = find_sd(ck, ["embedding.conv_1.0.weight"])
        dim = int(sd["embedding.conv_1.0.weight"].shape[1])
        def nums(prefix):                              
            for k, v in sd.items():
                if k.startswith(prefix) and torch.is_tensor(v) and v.ndim == 2 and v.shape[1] == 512:
                    return int(v.shape[0])
            return 60
        net = WSAD(dim, flag="Test", a_nums=nums("Amemory."), n_nums=nums("Nmemory."))
        load_strict(net, sd)
        net.flag = "Test"
        return dict(net=net.to(DEVICE).eval(), dim=dim)

    if name == "VADCLIP":
        with repo_context(VADCLIP_DIR, ["model", "clip", "utils", "ucf_option"]):
            mdl = importlib.import_module("model")
            opt = importlib.import_module("ucf_option")
            tools = importlib.import_module("utils.tools")
            a = opt.parser.parse_args([])
            net = mdl.CLIPVAD(a.classes_num, a.embed_dim, a.visual_length, a.visual_width, a.visual_head,
                              a.visual_layers, a.attn_window, a.prompt_prefix, a.prompt_postfix, DEVICE_STR)
        sd = find_sd(ck, ["classifier.weight"])
        load_strict(net, sd, ignore_prefix=("clipmodel.",))
        label_map = {k: k for k in UCF_CLASSES}
        return dict(net=net.to(DEVICE).eval(), dim=a.embed_dim, maxlen=a.visual_length,
                    get_mask=tools.get_batch_mask, prompt=tools.get_prompt_text(label_map))

def _shots(n): return np.floor(np.linspace(0, n, 33) + 0.5).astype(int)

def score_mil(L, crops, opts):
    seq = crops.mean(0)                                            
    n = len(seq)
    shots = _shots(n)
    seg = np.zeros((32, seq.shape[1]), np.float32)
    for k in range(32):                                            
        lo, hi = min(shots[k], n - 1), max(shots[k + 1], min(shots[k]+1, n))
        v = seq[lo:hi].mean(0)
        seg[k] = v / (np.linalg.norm(v) + 1e-12)
    with torch.no_grad():
        pred = L["net"](torch.from_numpy(seg)[None].to(DEVICE))[0].cpu().numpy()
    if L["frame_fn"] is not None: return np.asarray(L["frame_fn"](pred, n, n * 16), dtype=np.float32)
    fs = np.zeros(n * 16, np.float32)
    for c in range(n):
        k = int(np.clip(np.searchsorted(shots, c, side="right") - 1, 0, 31))
        fs[c * 16:(c + 1) * 16] = pred[k]
    return fs

def score_evtmil(L, crops, opts):
    net, cfg = L["net"], L["cfg"]
    feats = torch.from_numpy(crops.mean(0).astype(np.float32)).to(DEVICE)
    feats = F.normalize(feats, p=2, dim=-1) 
    
    with torch.no_grad():
        f = net.encode(feats)
        if cfg.mode == "nomem": s = net.score(f)
        else:
            d = net.dist(f, False)
            use_p = opts.get("evt_kind", "p").startswith("p") and (not getattr(cfg, 'no_pot', False)) and getattr(net, 'ready', torch.tensor(1)).item() == 1
            if use_p: s = net.p_of(d)
            else: s = (d / getattr(net, 'dscale', torch.tensor(1.0)).clamp_min(1e-6)).clamp(0, 1)
    return np.repeat(s.cpu().numpy().astype(np.float32), 16)

def score_vadclip(L, crops, opts):
    net, maxlen = L["net"], L["maxlen"]
    res = []
    for feats in crops:                                            
        T, n = len(feats), int(len(feats) / maxlen) + 1
        vis = torch.zeros(n, maxlen, feats.shape[1])
        for i in range(n):
            seg = feats[i * maxlen:(i + 1) * maxlen]
            vis[i, :len(seg)] = torch.from_numpy(np.asarray(seg, np.float32))
        lengths, rem = torch.zeros(n), T                           
        for j in range(n):
            if rem < maxlen: lengths[j] = rem; rem = 0
            else: lengths[j] = maxlen; rem -= maxlen
        lengths = lengths.to(int)
        mask = L["get_mask"](lengths, maxlen).to(DEVICE)
        with torch.no_grad():
            _, logits1, _ = net(vis.to(DEVICE), mask, L["prompt"], lengths)
        res.append(torch.sigmoid(logits1.reshape(-1, logits1.shape[-1])[:T].squeeze(-1)).float().cpu().numpy())
    t = min(len(r) for r in res)
    return np.repeat(np.mean([r[:t] for r in res], axis=0), 16)

def score_rtfm(L, crops, opts):
    x = torch.from_numpy(crops).float().to(DEVICE)[None]           
    net, nc = L["net"], x.shape[1]
    with torch.no_grad():
        for c in (10, 5, 2, 1):                                        
            if nc % c: continue
            try:
                parts = [net(inputs=x[:, i:i + c])[6][0, :, 0] for i in range(0, nc, c)]
                return np.repeat(torch.stack(parts).mean(0).float().detach().cpu().numpy(), 16)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower(): raise
                torch.cuda.empty_cache()
    raise RuntimeError("Hết VRAM khi chạy RTFM")

def score_urdmu(L, crops, opts):
    x = torch.from_numpy(crops).float().to(DEVICE)[None]  
    with torch.no_grad():         
        out = L["net"](x)["frame"]                                     
    return np.repeat(out[0].float().detach().cpu().numpy(), 16)

SCORERS = {"MIL Sultani": score_mil, "EVTMIL": score_evtmil, "RTFM": score_rtfm, "URDMU": score_urdmu, "VADCLIP": score_vadclip}

def process_inference_with_speed(model_name, video_path, num_frames, opts):
    t0 = time.time()
    try:
        L = load_model(model_name)
        feat_type = FEAT_BY_DIM.get(L["dim"])
        
        # Bắt đầu tự động trích xuất / lấy từ cache (Không cần Dataset)
        crops = get_crops(video_path, feat_type, L["dim"])
        
        scores = np.asarray(SCORERS[model_name](L, crops, opts), dtype=np.float32)
        if len(scores) < num_frames: scores = np.pad(scores, (0, num_frames - len(scores)), mode="edge")
        scores = scores[:num_frames]
        
        dt = time.time() - t0
        fps_infer = num_frames / dt if dt > 0 else 0
        return scores, fps_infer
    except Exception as e:
        st.sidebar.error(f"[{model_name}] Lỗi: {str(e)}")
        return None, 0

# ==========================================
# 5. GIAO DIỆN STREAMLIT
# ==========================================
st.sidebar.header("BẢNG ĐIỀU KHIỂN")
available_models = [m for m, p in MODELS.items() if os.path.exists(p)]
selected_models = st.sidebar.multiselect("1. Chọn Mô hình:", available_models, default=[available_models[0]] if available_models else [])
uploaded_file = st.sidebar.file_uploader("2. Chọn Video Test (.mp4, .avi)", type=["mp4", "avi"])

# ĐÃ XÓA BỎ HOÀN TOÀN Ô NHẬP "Thư mục Dataset" ĐỂ ĐẢM BẢO CHUẨN END-TO-END

threshold = st.sidebar.slider("Ngưỡng bất thường", 0.05, 0.95, 0.5, 0.05)

opts = {}
if "EVTMIL" in selected_models:
    opts["evt_kind"] = st.sidebar.selectbox("Điểm EVTMIL", ["p (POT/GPD)", "d (sai số tái tạo chuẩn hóa)"])

if not selected_models: st.stop()

cols = st.columns(len(selected_models))
ui = {}
for idx, m in enumerate(selected_models):
    with cols[idx]:
        st.subheader(f"{m}")
        ui[m] = {"video": st.empty(), "status": st.empty(), "score": st.empty(), "speed": st.empty(), "stats": st.empty(), "time_info": st.empty()}
        ui[m]["log_text"] = st.expander("📝 Nhật ký", expanded=True).empty()

def format_time(seconds): return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"

if uploaded_file and st.sidebar.button("▶ XỬ LÝ & CHẠY DEMO THỰC TẾ", type="primary"):
    vpath = os.path.join(VIDEO_DIR, os.path.basename(uploaded_file.name))
    with open(vpath, "wb") as f: f.write(uploaded_file.read())

    cap = cv2.VideoCapture(vpath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_str = format_time(total_frames / fps)

    model_scores = {}
    with st.spinner("Đang tiền xử lý video tự động & chấm điểm bằng AI..."):
        for m in selected_models:
            sc, speed = process_inference_with_speed(m, vpath, total_frames, opts)
            if sc is not None: model_scores[m] = {"scores": sc, "speed": speed}

    active = list(model_scores)
    if not active: st.stop()

    tracking = {m: {"in": False, "start": 0.0, "logs": [], "anomaly_count": 0} for m in active}
    frame_idx = 0
    
    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        cur = frame_idx / fps
        tstr = format_time(cur)
        frames_left = total_frames - frame_idx

        for m in active:
            sc = model_scores[m]["scores"]
            ai_speed = model_scores[m]["speed"]
            score = float(sc[frame_idx]) if frame_idx < len(sc) else 0.0
            
            is_anom = score > threshold
            color = (0, 0, 255) if is_anom else (0, 255, 0)
            status = "BẤT THƯỜNG!" if is_anom else "Bình thường"

            tr = tracking[m]
            if is_anom: tr["anomaly_count"] += 1
            anomaly_ratio = (tr["anomaly_count"] / max(1, frame_idx)) * 100

            if is_anom and not tr["in"]: tr["in"], tr["start"] = True, cur
            elif not is_anom and tr["in"]:
                tr["in"] = False
                tr["logs"].append(f"- Từ **{format_time(tr['start'])}** đến **{tstr}**")

            disp = frame.copy()
            cv2.rectangle(disp, (10, 10), (450, 70), (0, 0, 0), -1)
            cv2.putText(disp, f"Model: {m} | {tstr}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(disp, f"Status: {'ANOMALY!' if is_anom else 'Normal'}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            bar_len = int(min(max(score, 0), 1) * 300)
            cv2.rectangle(disp, (10, 80), (10 + bar_len, 95), color, -1)
            cv2.rectangle(disp, (10, 80), (310, 95), (255, 255, 255), 1)

            p = ui[m]
            p["video"].image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            p["status"].markdown(f"**Trạng thái:** :{'red' if is_anom else 'green'}[{status}] {'🚨' if is_anom else '✅'}")
            p["score"].text(f"Điểm số: {score:.4f}")
            p["speed"].markdown(f"⚡ **Tốc độ AI:** `{ai_speed:.1f} FPS`")
            p["stats"].markdown(f"📊 **Dư:** `{frames_left} frame` | **Tỉ lệ lỗi:** `{anomaly_ratio:.1f}%`")
            p["time_info"].markdown(f"**Thời gian:** `{tstr}` / `{total_str}`")

            if tr["logs"] or tr["in"]:
                txt = "\n".join(tr["logs"])
                if tr["in"]: txt += f"\n- Từ **{format_time(tr['start'])}** đến **đang diễn ra...**"
                p["log_text"].markdown(txt)
            else:
                p["log_text"].markdown("_Chưa phát hiện vấn đề..._")

        frame_idx += 1
        
        delay = (1.0 / fps) - (time.time() - t0)
        if delay > 0: time.sleep(delay)

    for m in active:
        tr = tracking[m]
        if tr["in"]:
            tr["logs"].append(f"- Từ **{format_time(tr['start'])}** đến **{total_str}**")
            ui[m]["log_text"].markdown("\n".join(tr["logs"]))
            
    cap.release()
