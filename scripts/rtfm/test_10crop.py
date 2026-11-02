import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, roc_auc_score
from dataset import group_videos

_names = [b for b, _ in group_videos('list/UCF_Test.list')]
_vlabel = np.array([0 if n.startswith('Normal') else 1 for n in _names])

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        frames, vmax = [], []
        for i, input in enumerate(dataloader):
            input = input.to(device).permute(0, 2, 1, 3)           # (1,T,10,F) -> (1,10,T,F)
            out = model(inputs=input)
            logits = torch.mean(torch.squeeze(out[6], 1), 0)        # (T,)
            vmax.append(logits.max().item())
            frames.append(np.repeat(logits.cpu().numpy(), 16))
        pred = np.concatenate(frames)
        gt = np.load('list/gt-ucf.npy')
        assert len(pred) == len(gt), "pred %d != gt %d: gt không khớp độ dài feature" % (len(pred), len(gt))
        fpr, tpr, _ = roc_curve(list(gt), pred)
        rec_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(list(gt), pred)
        pr_auc = auc(recall, precision)
        vauc = roc_auc_score(_vlabel, vmax)
        print('AUC {:.4f} | AP {:.4f} | video-AUC {:.4f}'.format(rec_auc, pr_auc, vauc), flush=True)
        return rec_auc, pr_auc
