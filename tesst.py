import os, glob, cv2, numpy as np, torch
from PIL import Image
from torchvision import transforms as T
import clip

ROOT = '/home/danhkay/UCF_Crime'
NAME = 'Abuse028_x264'
OFF0 = 'data/UCFClipFeatures/Abuse/Abuse028_x264__0.npy'
ref = np.load(OFF0)

# tìm video hoặc thư mục frame
hit = None
for r, ds, fs in os.walk(ROOT):
    for f in fs:
        if f.startswith(NAME) and f.lower().endswith(('.mp4', '.avi')):
            hit = os.path.join(r, f)
    for d in ds:
        if d == NAME:
            hit = os.path.join(r, d)
print('nguồn:', hit)
assert hit, 'không tìm thấy ' + NAME

frames = []
if os.path.isdir(hit):
    for p in sorted(glob.glob(os.path.join(hit, '*.png')) + glob.glob(os.path.join(hit, '*.jpg'))):
        frames.append(Image.open(p).convert('RGB'))
else:
    cap = cv2.VideoCapture(hit)
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
print('số frame:', len(frames), '| số clip chính thức:', len(ref), '| frame/16 =', len(frames) / 16)

dev = 'cuda'
model, pre_center = clip.load('ViT-B/16', device=dev); model.eval()
norm = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
pre_squash = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm])

cos = lambda x, y: ((x*y).sum(1) / (np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1))).mean()

@torch.no_grad()
def enc(imgs, pre):
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.stack([pre(im) for im in imgs[i:i+64]]).to(dev).half()
        out.append(model.encode_image(x).float().cpu().numpy())
    return np.concatenate(out)

for name, pre in [('center-crop', pre_center), ('squash 224x224', pre_squash)]:
    res = []
    for off in range(16):
        idx = [min(off + 16*t, len(frames)-1) for t in range(len(ref))]
        res.append((cos(enc([frames[i] for i in idx], pre), ref), off))
    res.sort(reverse=True)
    print(name, '| tốt nhất:', [(round(float(c), 4), o) for c, o in res[:3]])