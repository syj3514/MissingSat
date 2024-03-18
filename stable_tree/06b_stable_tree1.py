from IPython import get_ipython


def type_of_script():
    """
    Detects and returns the type of python kernel
    :return: string 'jupyter' or 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


if type_of_script() == 'jupyter':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import cmasher as cmr

import numpy as np
import os, glob, atexit, signal
import time
import warnings

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
from rur.sci.photometry import measure_luminosity
from rur.sci.geometry import get_angles, euler_angle
from rur.utool import rotate_data
from scipy.ndimage import gaussian_filter
uri.timer.verbose=0
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind, isin
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle, ax_change_color
from importlib import reload
from copy import deepcopy
from multiprocessing import Pool, shared_memory, Value
from common_func import *


mode1 = 'nh'
database1 = f"/home/jeon/MissingSat/database/{mode1}"
iout1 = 1026
repo1, rurmode1, dp1 = mode2repo(mode1)
snap1 = uri.RamsesSnapshot(repo1, iout1, mode=rurmode1)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']; nout=nout1[nout1 <= iout1]
gals1 = uhmi.HaloMaker.load(snap1, galaxy=True, double_precision=dp1)
hals1 = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp1)

LG1 = pklload(f"{database1}/LocalGroup.pickle")
allsats1 = None; allsubs1 = None; states1 = None
keys1 = list(LG1.keys())
for key in keys1:
    sats = LG1[key]['sats']; subs = LG1[key]['subs']; real = LG1[key]['real']
    dink = real[real['state']=='dink']['hid']
    ind = isin(subs['id'], dink)
    subs['dink'][ind] = True; subs['dink'][~ind] = False
    state = np.zeros(len(subs), dtype='<U7')
    state[ind] = 'dink'; state[~ind] = 'pair'
    
    upair = real[real['state']=='upair']['hid']
    ind = isin(subs['id'], upair)
    state[ind] = 'upair'

    allsats1 = sats if allsats1 is None else np.hstack((allsats1, sats))
    allsubs1 = subs if allsubs1 is None else np.hstack((allsubs1, subs))
    states1 = state if states1 is None else np.hstack((states1, state))
argsort = np.argsort(allsubs1['id'])
allsubs1 = allsubs1[argsort]; states1 = states1[argsort]
dinks1 = allsubs1[states1 == 'dink']
pairs1 = allsubs1[states1 == 'pair']
upairs1 = allsubs1[states1 == 'upair']

print(len(allsubs1), np.unique(states1, return_counts=True))  


# mode2 = 'nh2'
# database2 = f"/home/jeon/MissingSat/database/{mode2}"
# iout2 = 797
# repo2, rurmode2, dp2 = mode2repo(mode2)
# snap2 = uri.RamsesSnapshot(repo2, iout2, mode=rurmode2)
# snap2s = uri.TimeSeries(snap2)
# snap2s.read_iout_avail()
# nout2 = snap2s.iout_avail['iout']; nout=nout2[nout2 <= iout2]
# gals2 = uhmi.HaloMaker.load(snap2, galaxy=True, double_precision=dp2)
# hals2 = uhmi.HaloMaker.load(snap2, galaxy=False, double_precision=dp2)

# LG2 = pklload(f"{database2}/LocalGroup.pickle")
# allsats2 = None; allsubs2 = None; states2 = None
# keys2 = list(LG2.keys())
# for key in keys2:
#     sats = LG2[key]['sats']; subs = LG2[key]['subs']; real = LG2[key]['real']
#     dink = real[real['state']=='dink']['hid']
#     ind = isin(subs['id'], dink)
#     subs['dink'][ind] = True; subs['dink'][~ind] = False
#     state = np.zeros(len(subs), dtype='<U7')
#     state[ind] = 'dink'; state[~ind] = 'pair'
    
#     upair = real[real['state']=='upair']['hid']
#     ind = isin(subs['id'], upair)
#     state[ind] = 'upair'

#     allsats2 = sats if allsats2 is None else np.hstack((allsats2, sats))
#     allsubs2 = subs if allsubs2 is None else np.hstack((allsubs2, subs))
#     states2 = state if states2 is None else np.hstack((states2, state))
# argsort = np.argsort(allsubs2['id'])
# allsubs2 = allsubs2[argsort]; states2 = states2[argsort]
# dinks2 = allsubs2[states2 == 'dink']
# pairs2 = allsubs2[states2 == 'pair']
# upairs2 = allsubs2[states2 == 'upair']

# print(len(allsubs2), np.unique(states2, return_counts=True))







################################################## NH1
print("\nNewHorizon1\n")
shared_particles1 = pklload(f"{database1}/06b_shared_particles.pickle")
leng = len(allsubs1)
fnames = os.listdir(f"/storage6/NewHorizon/halo/")
bout = [int(fname[-5:]) for fname in fnames if(fname.startswith("tree_bricks"))]; bout.sort()
bout = np.array(bout); bout = bout[bout <= iout1]
# print(bout)

lastids = {}
maxid = 0
uri.timer.verbose=1
hparts = uhmi.HaloMaker.read_member_parts(snap1, allsubs1, galaxy=False, target_fields=['id'], nthread=24)
uri.timer.verbose=0
for sub in tqdm( allsubs1, desc='Find maximum pID' ):
    lastids[sub['id']] = hparts[hparts['hmid']==sub['id']]['id']
    maxid = max(maxid, lastids[sub['id']].max())


dtype = [("lastid", np.int16), ("timestep", np.int16), ("id", np.int16), ("give_score", np.float64), ("take_score", np.float64)]
nstep = len(bout)
nsub = len(allsubs1)
result = np.zeros(nstep*nsub, dtype=dtype)

result[:nsub]['lastid'] = allsubs1['id']
result[:nsub]['timestep'] = iout1
result[:nsub]['id'] = allsubs1['id']
result[:nsub]['give_score'] = 1
result[:nsub]['take_score'] = 1
cursor = nsub
if(os.path.exists(f"{database1}/stable_progenitors.tmp.pickle")):
    lastout, result = pklload(f"{database1}/stable_progenitors.tmp.pickle")
else:
    pklsave((iout1,result), f"{database1}/stable_progenitors.tmp.pickle")
    lastout = iout1
    cursor = np.where(result['id']==0)[0][0]

print("Calculate Scores...")
for pout in bout[::-1][1:]:
    if(pout >= lastout): continue
    psnap = snap1s.get_snap(pout)
    phals, ppids = uhmi.HaloMaker.load(psnap, galaxy=False, load_parts=True)
    cparts = phals['nparts']; cparts = np.cumsum(cparts); cparts = np.hstack((0, cparts))
    table = np.repeat(phals['id'], phals['nparts']).astype(np.int16)
    pids = np.zeros(max(maxid, ppids.max()), dtype=np.int32)
    pids[ppids-1] = table
    result[cursor:cursor+nsub]['lastid'] = allsubs1['id']
    result[cursor:cursor+nsub]['timestep'] = pout

    for key in lastids.keys():
        shared_particles = shared_particles1[key]

        # Find Prog
        progs = pids[shared_particles-1]
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
            give_score = counts[argmax] / len(shared_particles)
            # print(prog, give_score)

            # Take Score
            part_of_prog = ppids[ cparts[prog-1]:cparts[prog] ]
            ind = isin(part_of_prog, shared_particles, assume_unique=True)
            take_score = np.sum(ind)/len(part_of_prog)
        result[cursor]['id'] = prog
        result[cursor]['give_score'] = give_score
        result[cursor]['take_score'] = take_score
        cursor += 1
    psnap.clear()
    del snap1s.snaps[pout]
    pklsave((pout,result), f"{database1}/stable_progenitors.tmp.pickle", overwrite=True)
    print(f"[{pout:04d}] `{database1}/stable_progenitors.tmp.pickle` save done")
    lastout = pout
pklsave(result, f"{database1}/stable_progenitors.pickle", overwrite=True)
snap1s.clear()

# ################################################## NH2
# print("\nNewHorizon2\n")
# shared_particles2 = pklload(f"{database2}/06b_shared_particles.pickle")
# leng = len(allsubs2)
# fnames = os.listdir(f"/storage7/NH2/halo/")
# bout = [int(fname[-5:]) for fname in fnames if(fname.startswith("tree_bricks"))]; bout.sort()
# bout = np.array(bout); bout = bout[bout <= iout2]
# # print(bout)

# lastids = {}
# maxid = 0
# uri.timer.verbose=1
# hparts = uhmi.HaloMaker.read_member_parts(snap2, allsubs2, galaxy=False, target_fields=['id'], nthread=24)
# uri.timer.verbose=0
# for sub in tqdm( allsubs2, desc='Find maximum pID' ):
#     lastids[sub['id']] = hparts[hparts['hmid']==sub['id']]['id']
#     maxid = max(maxid, lastids[sub['id']].max())


# dtype = [("lastid", np.int16), ("timestep", np.int16), ("id", np.int16), ("give_score", np.float64), ("take_score", np.float64)]
# nstep = len(bout)
# nsub = len(allsubs2)
# result = np.zeros(nstep*nsub, dtype=dtype)

# result[:nsub]['lastid'] = allsubs2['id']
# result[:nsub]['timestep'] = iout2
# result[:nsub]['id'] = allsubs2['id']
# result[:nsub]['give_score'] = 1
# result[:nsub]['take_score'] = 1
# cursor = nsub
# if(os.path.exists(f"{database2}/stable_progenitors.tmp.pickle")):
#     lastout, result = pklload(f"{database2}/stable_progenitors.tmp.pickle")
# else:
#     pklsave((iout2,result), f"{database2}/stable_progenitors.tmp.pickle")
#     lastout = iout2
#     cursor = np.where(result['id']==0)[0][0]

# print("Calculate Scores...")
# for pout in bout[::-1][1:]:
#     if(pout >= lastout): continue
#     psnap = snap2s.get_snap(pout)
#     phals, ppids = uhmi.HaloMaker.load(psnap, galaxy=False, load_parts=True)
#     cparts = phals['nparts']; cparts = np.cumsum(cparts); cparts = np.hstack((0, cparts))
#     table = np.repeat(phals['id'], phals['nparts']).astype(np.int16)
#     pids = np.zeros(max(maxid, ppids.max()), dtype=np.int32)
#     pids[ppids-1] = table
#     result[cursor:cursor+nsub]['lastid'] = allsubs2['id']
#     result[cursor:cursor+nsub]['timestep'] = pout

#     for key in lastids.keys():
#         shared_particles = shared_particles2[key]

#         # Find Prog
#         progs = pids[shared_particles-1]
#         unique, counts = np.unique(progs, return_counts=True)
#         if( (len(unique)==1)&(0 in unique) ):
#             prog=0
#             give_score=0
#             take_score=0
#         else:
#             if(0 in unique):
#                 mask = unique!=0
#                 unique = unique[mask]; counts = counts[mask]
            
#             # Give Score
#             argmax = np.argmax(counts)
#             prog = unique[argmax]
#             give_score = counts[argmax] / len(shared_particles)
#             # print(prog, give_score)

#             # Take Score
#             part_of_prog = ppids[ cparts[prog-1]:cparts[prog] ]
#             ind = isin(part_of_prog, shared_particles, assume_unique=True)
#             take_score = np.sum(ind)/len(part_of_prog)
#         result[cursor]['id'] = prog
#         result[cursor]['give_score'] = give_score
#         result[cursor]['take_score'] = take_score
#         cursor += 1
#     psnap.clear()
#     del snap2s.snaps[pout]
#     pklsave((pout,result), f"{database2}/stable_progenitors.tmp.pickle", overwrite=True)
#     print(f"[{pout:04d}] `{database2}/stable_progenitors.tmp.pickle` save done")
#     lastout = pout
# pklsave(result, f"{database2}/stable_progenitors.pickle", overwrite=True)
# snap2s.clear()





