import torch

def sparsity(arr, batch_size, lamda2):
    return lamda2 * torch.mean(torch.norm(arr, dim=0))

def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    return lamda1 * torch.sum((arr2 - arr) ** 2)

class RTFM_loss(torch.nn.Module):
    def __init__(self, alpha, margin):
        super(RTFM_loss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.criterion = torch.nn.BCELoss()

    def forward(self, score_normal, score_abnormal, nlabel, alabel, feat_n, feat_a):
        label = torch.cat((nlabel, alabel), 0)
        score = torch.cat((score_normal, score_abnormal), 0).squeeze()
        label = label.cuda()
        loss_cls = self.criterion(score, label)
        loss_abn = torch.abs(self.margin - torch.norm(torch.mean(feat_a, dim=1), p=2, dim=1))
        loss_nor = torch.norm(torch.mean(feat_n, dim=1), p=2, dim=1)
        loss_rtfm = torch.mean((loss_abn + loss_nor) ** 2)
        return loss_cls + self.alpha * loss_rtfm

def train(nloader, aloader, model, batch_size, optimizer, viz, device):
    with torch.set_grad_enabled(True):
        model.train()
        ninput, nlabel = next(nloader)
        ainput, alabel = next(aloader)
        input = torch.cat((ninput, ainput), 0).to(device)

        score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, \
        feat_normal_bottom, scores, scores_nor_bottom, scores_nor_abn_bag, _ = model(input)

        scores = scores.view(batch_size * 32 * 2, -1).squeeze()
        abn_scores = scores[batch_size * 32:]
        nlabel = nlabel[0:batch_size]
        alabel = alabel[0:batch_size]

        loss_criterion = RTFM_loss(0.0001, 100)
        loss_sparse = sparsity(abn_scores, batch_size, 8e-3)
        loss_smooth = smooth(abn_scores, 8e-4)
        loss_main = loss_criterion(score_normal, score_abnormal, nlabel, alabel, feat_select_normal, feat_select_abn)
        cost = loss_main + loss_smooth + loss_sparse

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        return {"total": cost.item(), "main": loss_main.item(),
                "smooth": loss_smooth.item(), "sparse": loss_sparse.item()}
