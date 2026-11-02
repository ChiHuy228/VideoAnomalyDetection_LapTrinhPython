import os, time, glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import option
from config import Config
from dataset import Dataset
from model import Model
from train import train
from test_10crop import test
from utils import Visualizer, save_best_record

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    device = torch.device("cuda")
    viz = Visualizer()

    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True), batch_size=args.batch_size,
                               shuffle=True, num_workers=args.workers, persistent_workers=True,
                               pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False), batch_size=args.batch_size,
                               shuffle=True, num_workers=args.workers, persistent_workers=True,
                               pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True), batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)
    print("normal:", len(train_nloader.dataset), "| abnormal:", len(train_aloader.dataset),
          "| test:", len(test_loader.dataset), flush=True)

    model = Model(args.feature_size, args.batch_size).to(device)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_auc = -1
    t0 = time.time(); test(test_loader, model, args, viz, device)
    print("thời gian 1 lần đánh giá: {:.1f}s".format(time.time() - t0), flush=True)

    for step in tqdm(range(1, args.max_epoch + 1), total=args.max_epoch, dynamic_ncols=True, mininterval=300):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for g in optimizer.param_groups:
                g["lr"] = config.lr[step - 1]
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)
        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        loss = train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        if step % args.log_every == 0:
            tqdm.write("step {:5d} | total {:.4f} | main {:.4f}".format(step, loss["total"], loss["main"]))

        if step % args.eval_every == 0 and step >= args.eval_start:
            auc, ap = test(test_loader, model, args, viz, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            if auc > best_auc:
                best_auc = auc
                for old in glob.glob(os.path.join(args.ckpt_dir, '*')):   # chỉ giữ bản best mới nhất
                    os.remove(old)
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, '{}{}-i3d.pkl'.format(args.model_name, step)))
                save_best_record(test_info, os.path.join(args.ckpt_dir, '{}-step-AUC.txt'.format(step)))
                tqdm.write("step {:5d} | AUC {:.4f} -> best".format(step, auc))

    print("BEST AUC:", best_auc)
