from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize

import numpy as np
import os, glob
import time
import warnings

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
from rur.sci.photometry import measure_luminosity
from rur.sci.geometry import get_angles, euler_angle
from rur.utool import rotate_data
from scipy.ndimage import gaussian_filter
uri.timer.verbose=1
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind, isin
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle
import argparse, subprocess
from importlib import reload
import cmasher as cmr
from copy import deepcopy
from multiprocessing import Pool, shared_memory

mode = 'nh2'
iout = 797
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)
database = f"/home/jeon/MissingSat/database/nh2"




LG = pklload(f"{database}/LG")




allsubs = None
for key in LG.keys():
    subs = LG[key]['subs']
    allsubs = subs if allsubs is None else np.hstack((allsubs, subs))
argsort = np.argsort(allsubs['id'])
allsubs = allsubs[argsort]





lastids = {}
maxid = 0
for sub in tqdm( allsubs ):
    lastids[sub['id']] = uhmi.HaloMaker.read_member_part(snaps.get_snap(sub['timestep']), sub['id'], galaxy=False, simple=True)
    maxid = max(maxid, lastids[sub['id']].max())




iouts = os.listdir("/storage7/NH2/halo")
iouts = [int(f[-5:]) for f in iouts if f.startswith("tree_bricks")]
iouts.sort()
iouts = np.array(iouts[::-1])
iouts = iouts[iouts < iout]
print(f"{np.max(iouts)} ~ {np.min(iouts)}")




dtype = [("lastid", np.int16), ("timestep", np.int16), ("id", np.int16), ("give_score", np.float64), ("take_score", np.float64)]
nstep = len(iouts)
nsub = len(allsubs)

result = np.zeros(nstep*nsub, dtype=dtype)
result[:nsub]['lastid'] = allsubs['id']
result[:nsub]['timestep'] = allsubs['timestep']
result[:nsub]['id'] = allsubs['id']
result[:nsub]['give_score'] = 1
result[:nsub]['take_score'] = 1

cursor = nsub

if(os.path.exists(f"{database}/02_main_progenitors.tmp.pickle")):
    lastout, result = pklload(f"{database}/02_main_progenitors.tmp.pickle")
else:
    pklsave((iout,result), f"{database}/02_main_progenitors.tmp.pickle")
    lastout = iout
    cursor = np.where(result['id']==0)[0][0]

uri.timer.verbose=0
for pout in tqdm(iouts[1:]):
    if(pout >= lastout): continue
    psnap = snaps.get_snap(pout)
    phals, ppids = uhmi.HaloMaker.load(psnap, galaxy=False, load_parts=True)
    cparts = phals['nparts']; cparts = np.cumsum(cparts); cparts = np.hstack((0, cparts))
    table = np.repeat(phals['id'], phals['nparts']).astype(np.int16)
    pids = np.zeros(max(maxid, ppids.max()), dtype=np.int32)
    pids[ppids-1] = table

    result[cursor:cursor+nsub]['lastid'] = allsubs['id']
    result[cursor:cursor+nsub]['timestep'] = pout

    for key in lastids.keys():
        pid = lastids[key]

        # Find Prog
        progs = pids[pid-1]
        unique, counts = np.unique(progs, return_counts=True)
        if( (len(unique)==1)&(0 in unique) ):
            prog=0
            give_score=0
            take_score=0
        else:
            if(0 in unique):
                mask = unique!=0
                unique = unique[mask]; counts = counts[mask]
            
            # Give Score
            argmax = np.argmax(counts)
            prog = unique[argmax]
            give_score = counts[argmax] / len(pid)
            # print(prog, give_score)

            # Take Score
            part_of_prog = ppids[ cparts[prog-1]:cparts[prog] ]
            ind = isin(part_of_prog, pid, assume_unique=True)
            take_score = np.sum(ind)/len(part_of_prog)
        result[cursor]['id'] = prog
        result[cursor]['give_score'] = give_score
        result[cursor]['take_score'] = take_score
        cursor += 1
    psnap.clear()
    del snaps.snaps[pout]
    pklsave((pout,result), f"{database}/02_main_progenitors.tmp.pickle", overwrite=True)
    lastout = pout
pklsave(result, f"{database}/02_main_progenitors.pickle", overwrite=True)