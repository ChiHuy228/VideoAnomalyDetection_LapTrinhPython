import time
START = time.time()
import sys, os
import numpy as np
import torch
import torch.utils.data as data
import utils
from options import parse_args
from config import Config
from train import train, AD_Loss
from ucf_test import test
from model import WSAD
from utils import Visualizer
from dataset_loader import UCF_crime
from tqdm import tqdm

if __name__ == "__main__":
    args = parse_args()
    config = Config(args)
    worker_init_fn = None
    torch.cuda.set_device('cuda:0')
    if config.seed >= 0:
        utils.set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)

    config.len_feature = 1024
    net = WSAD(config.len_feature, flag="Train", a_nums=60, n_nums=60).cuda()

    normal_train_loader = data.DataLoader(
        UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal, num_segments=200,
                  len_feature=config.len_feature, is_normal=True),
        batch_size=64, shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn, drop_last=True)
    abnormal_train_loader = data.DataLoader(
        UCF_crime(root_dir=config.root_dir, mode='Train', modal=config.modal, num_segments=200,
                  len_feature=config.len_feature, is_normal=False),
        batch_size=64, shuffle=True, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn, drop_last=True)
    test_loader = data.DataLoader(
        UCF_crime(root_dir=config.root_dir, mode='Test', modal=config.modal,
                  num_segments=config.num_segments, len_feature=config.len_feature),
        batch_size=1, shuffle=False, num_workers=config.num_workers,
        worker_init_fn=worker_init_fn)

    test_info = {"step": [], "auc": [], "ap": [], "ac": []}
    best_auc = 0
    criterion = AD_Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr[0], betas=(0.9, 0.999), weight_decay=0.00005)
    wind = Visualizer()
    LOG_EVERY = 10

    print("normal videos:", len(normal_train_loader.dataset),
          "| abnormal videos:", len(abnormal_train_loader.dataset),
          "| test files:", len(test_loader.dataset), flush=True)

    resume_path = os.path.join(args.model_path, 'resume.pth')

    def save_resume(step):
        tmp = resume_path + '.tmp'
        torch.save({'model': net.state_dict(), 'opt': optimizer.state_dict(), 'step': step,
                    'best_auc': best_auc, 'test_info': test_info}, tmp)
        os.replace(tmp, resume_path)          # ghi nguyên tử, không để file dở dang

    start_step = 1
    if os.path.exists(resume_path) and not args.fresh:
        ck = torch.load(resume_path, map_location='cuda:0', weights_only=False)
        net.load_state_dict(ck['model']); optimizer.load_state_dict(ck['opt'])
        start_step, best_auc, test_info = ck['step'] + 1, ck['best_auc'], ck['test_info']
        print("RESUME từ step", start_step, "| best AUC", best_auc, flush=True)
    else:
        test(net, config, wind, test_loader, test_info, 0)

    normal_loader_iter = iter(normal_train_loader)
    abnormal_loader_iter = iter(abnormal_train_loader)
    stopped_early = False
    last_step = start_step - 1

    for step in tqdm(range(start_step, config.num_iters + 1),
                     total=max(config.num_iters - start_step + 1, 0), dynamic_ncols=True, mininterval=60):
        for g in optimizer.param_groups:
            g["lr"] = config.lr[step - 1]
        try:
            loss = train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion, wind, step)
        except StopIteration:
            normal_loader_iter = iter(normal_train_loader)
            abnormal_loader_iter = iter(abnormal_train_loader)
            loss = train(net, normal_loader_iter, abnormal_loader_iter, optimizer, criterion, wind, step)
        last_step = step

        if step % LOG_EVERY == 0:
            tqdm.write("step {:4d} | total {:.4f} | att {:.4f} | triplet {:.4f} | kl {:.4f}".format(
                step, loss['total_loss'], loss['att_loss'], loss['triplet'], loss['kl_loss']))

        if (step % config.eval_every == 0) or step == config.num_iters:
            test(net, config, wind, test_loader, test_info, step)
            if test_info["auc"][-1] > best_auc:
                best_auc = test_info["auc"][-1]
                utils.save_best_record(test_info, os.path.join(config.output_path, "ucf_best_record_{}.txt".format(config.seed)))
                torch.save(net.state_dict(), os.path.join(args.model_path, "ucf_trans_{}.pkl".format(config.seed)))
                tqdm.write("  -> saved best model (AUC {:.4f})".format(best_auc))

        if step % args.save_every == 0:
            save_resume(step)

        if args.max_hours > 0 and (time.time() - START) / 3600 >= args.max_hours:
            tqdm.write("HẾT NGÂN SÁCH {} giờ: lưu resume và dừng ở step {}".format(args.max_hours, step))
            stopped_early = True
            break

    save_resume(last_step)
    if not stopped_early:
        torch.save(net.state_dict(), os.path.join(args.model_path, "ucf_trans_{}.pkl".format(config.num_iters)))
    print("DỪNG SỚM" if stopped_early else "XONG", "| step cuối:", last_step, "| BEST AUC:", best_auc)