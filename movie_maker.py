from common_func import *
import os
import numpy as np

# iid = 24590
import argparse
# python3 movie_maker.py --mode nh --kind d --type 0
parser = argparse.ArgumentParser(description='(syj3514@yonsei.ac.kr)')
parser.add_argument("-i", "--iid", required=False, help='target halo id', type=int, default=None)
parser.add_argument("-m", "--mode", required=False, help='simmode', type=str, default='nh')
parser.add_argument("-k", "--kind", required=True, help='kind', type=str, choices=['d', 'p', 'u'], default='d')
parser.add_argument("-t", "--type", required=True, help='type', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7,8], default=0)
args = parser.parse_args()

mode = args.mode
pkind = args.kind
ptype = args.type

iid = args.iid
directory = f"/home/jeon/MissingSat/database/{mode}/photo/evolution/{pkind}/{ptype}"
if(iid is None):
    ff = os.listdir(directory)
    iid = int(ff[0])
    print(f"Firstly selected `{iid}`")

ff = os.listdir(f"{directory}/{iid:07d}")
ff.sort()
fnames = [f"{directory}/{iid:07d}/{f}" for f in ff if f.endswith(".png")]
print(f"{len(fnames)} images found in `{directory}/{iid:07d}`\n {ff[:3]+['...']+ff[-3:]}")


output = f"{directory}/{mode}_{iid}_l{len(fnames)}_{pkind}_{ptype}.mp4"
makemovie(fnames, output, fps=15)
print(f"See `{output}`")