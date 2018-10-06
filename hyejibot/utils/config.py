import argparse


args = argparse.ArgumentParser()
args.add_argument('--batch', type=int, default=1)
args.add_argument('--epoch', type=int, default=200)
args.add_argument('--hidden', type=int, default=256)
args.add_argument('--dropout', type=float, default=0.8)
args.add_argument('--layers', type=int, default=1)
args.add_argument('--save_path', type=str, default='hyejibot/data/save')
args.add_argument('--ckpt_name', type=str, default='ckpt_35_256.pth')
config = args.parse_args()