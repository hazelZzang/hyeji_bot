import argparse


args = argparse.ArgumentParser()
args.add_argument('--batch', type=int, default=1)
args.add_argument('--epoch', type=int, default=200)
args.add_argument('--hidden', type=int, default=256)
args.add_argument('--dropout', type=float, default=0.8)
args.add_argument('--layers', type=int, default=1)
args.add_argument('--save_path', type=str, default='data/save')
config = args.parse_args()