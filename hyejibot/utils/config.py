import argparse


args = argparse.ArgumentParser()
args.add_argument('--batch', type=int, default=1)
args.add_argument('--epoch', type=int, default=200)
args.add_argument('--hidden', type=int, default=128)
args.add_argument('--dropout', type=float, default=0.8)
args.add_argument('--layers', type=int, default=2)
config = args.parse_args()