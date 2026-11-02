import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Pytorch Implementation of UR-DMU')
    parser.add_argument('--output_path', type=str, default='outputs/')
    parser.add_argument('--root_dir', type=str, default='outputs/')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--modal', type=str, default='rgb', choices=["rgb", "flow", "both"])
    parser.add_argument('--model_path', type=str, default='models/')
    parser.add_argument('--lr', type=str, default='[0.0001]*3000')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_segments', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--model_file', type=str, default="trans_2022.pkl")
    parser.add_argument('--eval_every', type=int, default=50)
    parser.add_argument('--max_hours', type=float, default=0, help='tự dừng sạch sau N giờ (0 = không giới hạn)')
    parser.add_argument('--save_every', type=int, default=100, help='lưu resume.pth mỗi N step')
    parser.add_argument('--fresh', type=int, default=0, help='1 = bỏ qua resume.pth, train lại từ đầu')
    parser.add_argument('--debug', action='store_true')

    return init_args(parser.parse_args())

def init_args(args):
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True)
    return args