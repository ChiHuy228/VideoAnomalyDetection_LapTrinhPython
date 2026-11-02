import os, re
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE = "/run/media/danhkay/Data/Project/PPNCKH/HuanLuyen/results/urdmu/video_scores.npz"


def _vname(x):
    x = x[0] if isinstance(x, (list, tuple)) else x
    b = os.path.basename(str(x))
    return re.sub(r'(__\d+)?\.npy$', '', b)      # vd: Abuse028_x264


def test(net, config, wind, test_loader, test_info, step, model_file=None,
         save_path=DEFAULT_SAVE):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        if model_file is not None:
            print(">>> loading", model_file, flush=True)
            net.load_state_dict(torch.load(model_file, map_location="cuda"))

        load_iter = iter(test_loader)
        gt_path = os.path.join(HERE, "frame_label", "gt-ucf.npy")
        if not os.path.exists(gt_path):
            gt_path = "frame_label/gt-ucf.npy"    # fallback: theo cwd
        frame_gt = np.load(gt_path)
        frame_predict = None

        cls_label, cls_pre = [], []
        temp_predict = torch.zeros((0)).cuda()
        per_video = {}

        for i in range(len(test_loader.dataset)):
            _data, _label, _name = next(load_iter)
            _data = _data.cuda()
            _label = _label.cuda()

            res = net(_data)
            a_predict = res["frame"]
            temp_predict = torch.cat([temp_predict, a_predict], dim=0)

            if (i + 1) % 10 == 0:
                cls_label.append(int(_label))
                a_predict = temp_predict.mean(0).cpu().numpy()

                cls_pre.append(1 if a_predict.max() > 0.5 else 0)
                fpre_ = np.repeat(a_predict, 16)
                per_video[_vname(_name)] = fpre_.astype(np.float32)
                frame_predict = fpre_ if frame_predict is None else np.concatenate([frame_predict, fpre_])
                temp_predict = torch.zeros((0)).cuda()

        print(">>> samples:", len(test_loader.dataset),
              "| videos:", len(per_video),
              "| frames pred/gt:", len(frame_predict), len(frame_gt), flush=True)

        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        correct_num = np.sum(np.array(cls_label) == np.array(cls_pre), axis=0)
        accuracy = correct_num / len(cls_pre)

        precision, recall, th = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)

        # Lưu điểm theo video khi đạt best AUC (hoặc khi đánh giá từ model_file)
        prev_best = max(test_info["auc"]) if len(test_info["auc"]) else -1
        if save_path and (auc_score > prev_best or model_file is not None):
            assert len(frame_predict) == len(frame_gt), (len(frame_predict), len(frame_gt))
            if len(per_video) != 290:
                print("!!! cảnh báo: len(per_video) =", len(per_video),
                      "| vài key:", list(per_video)[:5], flush=True)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **per_video)
            print("  -> đã lưu", save_path, "| AUC", round(float(auc_score), 4), flush=True)

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
        print("step {} | AUC {:.4f} | AP {:.4f} | ACC {:.4f}".format(
            step, auc_score, ap_score, accuracy), flush=True)