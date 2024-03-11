from common_func import *
import os
import numpy as np

# iid = 24590
import argparse
parser = argparse.ArgumentParser(description='(syj3514@yonsei.ac.kr)')
parser.add_argument("-i", "--iid", required=True, help='target halo id', type=int)
args = parser.parse_args()

iid = args.iid
directory = f"/home/jeon/MissingSat/database/nh/photo"

fnames = os.listdir(f"{directory}/{iid:07d}")
fnames = [f"{directory}/{iid:07d}/{f}" for f in fnames if f.endswith(".png")]
fnames.sort()
print(len(fnames), fnames[:3], fnames[-3:])

output = f"{directory}/{iid}_l{len(fnames)}.mp4"
makemovie(fnames, output, fps=15)