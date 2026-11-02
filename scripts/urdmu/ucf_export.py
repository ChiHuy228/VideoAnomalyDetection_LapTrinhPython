import os, sys, traceback
print(">>> start", flush=True)

import numpy as np
import torch
import torch.utils.data as data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import options
from utils import *
from config import *
from model import *
from dataset_loader import *
from ucf_test import test

if __name__ == "__main__":
    try:
        args = options.parse_args()
        config = Config(args)
        print(">>> args ok | model_file =", args.model_file, flush=True)

        net = WSAD(config.len_feature, flag="Test", a_nums=60, n_nums=60).cuda()
        test_loader = data.DataLoader(
            UCF_crime(root_dir=config.root_dir, mode='Test', modal=config.modal,
                      num_segments=config.num_segments, len_feature=config.len_feature),
            batch_size=1, shuffle=False, num_workers=config.num_workers)
        print(">>> net + loader ok | dataset size =", len(test_loader.dataset), flush=True)

        test_info = {"step": [], "auc": [], "ap": [], "ac": []}
        print(">>> before test", flush=True)
        test(net, config, None, test_loader, test_info, step=0,
             model_file=args.model_file)
        print(">>> done |", {k: v for k, v in test_info.items()}, flush=True)
    except SystemExit as e:
        print(">>> SystemExit:", e.code, flush=True)
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)