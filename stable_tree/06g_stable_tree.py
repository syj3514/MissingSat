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



def massbranch(branch):
    val = np.log10(branch['mvir'])
    mask1 = np.full(len(branch), True, dtype=bool)
    for i in range(len(mask1)):
        if(val[i] < 9): continue
        arr = val[max(0,i-100):i+100]
        mask1[i] = val[i] <= (np.mean(arr) + 4*np.std(arr))
    upper = (np.median(val) + 3*np.std(val))
    if(upper>8.5):
        mask2 = val <= upper
        return mask1&mask2
    return mask1

def velbranch(branch, snaps):
    iout = snaps.iout_avail['iout']
    fsnap = snaps.get_snap(iout[-1]); unitl_com = fsnap.unit_l/fsnap.aexp
    aexp = snaps.iout_avail['aexp']
    age = snaps.iout_avail['age']
    mask = np.full(len(branch), False, dtype=bool)
    mask[0] = True
    factor = 1
    for i in range(len(mask)-1):
        if(mask[i]):
            nb = branch[i]
            niout = nb['timestep']; nwhere = np.where(iout == niout)[0][0]; nage = age[nwhere]
            unit_l = unitl_com * aexp[nwhere]
        pb = branch[i+1]
        piout = pb['timestep']; pwhere = np.where(iout == piout)[0][0]; page = age[pwhere]
        dt = (nage - page)*1e9 * 365*24*3600 # sec
        dx = (nb['vx']*dt) * 1e5 # cm
        dy = (nb['vy']*dt) * 1e5
        dz = (nb['vz']*dt) * 1e5
        nnx = nb['x'] - dx/unit_l
        nny = nb['y'] - dy/unit_l
        nnz = nb['z'] - dz/unit_l
        dist2 = np.sqrt( (nnx-pb['x'])**2 + (nny-pb['y'])**2 + (nnz-pb['z'])**2 )
        if(dist2 < factor*(nb['rvir']+pb['rvir'])) and (pb['mvir'] < nb['mvir']*1e2):
            mask[i+1] = True
            factor = 1
        else:
            factor += 0.01
    return mask

def polybranch(branch, vmask=None, return_poly=False):
    if(vmask is None): vmask = np.full(len(branch), 0)
    score = (branch['take_score']*branch['give_score']) * (vmask+0.5)
    polyx = np.polynomial.polynomial.Polynomial.fit(branch['timestep'], branch['x'], 20, w=score)
    resix = branch['x'] - polyx(branch['timestep'])
    stdx = np.std(resix)
    polyy = np.polynomial.polynomial.Polynomial.fit(branch['timestep'], branch['y'], 20, w=score)
    resiy = branch['y'] - polyy(branch['timestep'])
    stdy = np.std(resiy)
    polyz = np.polynomial.polynomial.Polynomial.fit(branch['timestep'], branch['z'], 20, w=score)
    resiz = branch['z'] - polyz(branch['timestep'])
    stdz = np.std(resiz)

    resi = np.sqrt(resix**2 + resiy**2 + resiz**2)
    where1 = (resi > np.sqrt(stdx**2 + stdy**2 + stdz**2))
    where2 = resi/np.sqrt(3) > 1e-4
    where = where1&where2
    if(return_poly):
        return (~where), polyx, polyy, polyz
    return (~where)

####################################################
# New Horizon
####################################################
print('\nNew Horizon\n')
mode1 = 'nh'
database1 = f"/home/jeon/MissingSat/database/{mode1}"
iout1 = 1026
repo1, rurmode1, dp1 = mode2repo(mode1)
snap1 = uri.RamsesSnapshot(repo1, iout1, mode=rurmode1)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']; nout=nout1[nout1 <= iout1]

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
stree1 = pklload(f"{database1}/stable_tree.pickle")
rtree1 = pklload(f"{database1}/stable_tree_raw.pickle")

keys = list(rtree1.keys())
ntree1 = {}
for key in tqdm(keys, desc='NewHorizon'):
    branch = rtree1[key]
    mmask = massbranch(branch)
    mbranch = branch[mmask]
    vmask = velbranch(mbranch, snap1s)
    pmask = polybranch(mbranch, vmask, return_poly=False)
    mask = (vmask|pmask)
    ntree1[key] = mbranch[mask]
pklsave(ntree1, f"{database1}/stable_tree_new.pickle", overwrite=True)
snap1s.clear()

####################################################
# New Horizon2
####################################################
print('\nNew Horizon2\n')
mode2 = 'nh2'
database2 = f"/home/jeon/MissingSat/database/{mode2}"
iout2 = 797
repo2, rurmode2, dp2 = mode2repo(mode2)
snap2 = uri.RamsesSnapshot(repo2, iout2, mode=rurmode2)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']; nout=nout2[nout2 <= iout2]

LG2 = pklload(f"{database2}/LocalGroup.pickle")
allsats2 = None; allsubs2 = None; states2 = None
keys2 = list(LG2.keys())
for key in keys2:
    sats = LG2[key]['sats']; subs = LG2[key]['subs']; real = LG2[key]['real']
    dink = real[real['state']=='dink']['hid']
    ind = isin(subs['id'], dink)
    subs['dink'][ind] = True; subs['dink'][~ind] = False
    state = np.zeros(len(subs), dtype='<U7')
    state[ind] = 'dink'; state[~ind] = 'pair'
    
    upair = real[real['state']=='upair']['hid']
    ind = isin(subs['id'], upair)
    state[ind] = 'upair'

    allsats2 = sats if allsats2 is None else np.hstack((allsats2, sats))
    allsubs2 = subs if allsubs2 is None else np.hstack((allsubs2, subs))
    states2 = state if states2 is None else np.hstack((states2, state))
argsort = np.argsort(allsubs2['id'])
allsubs2 = allsubs2[argsort]; states2 = states2[argsort]
dinks2 = allsubs2[states2 == 'dink']
pairs2 = allsubs2[states2 == 'pair']
upairs2 = allsubs2[states2 == 'upair']

print(len(allsubs2), np.unique(states2, return_counts=True))  
stree2 = pklload(f"{database2}/stable_tree.pickle")
rtree2 = pklload(f"{database2}/stable_tree_raw.pickle")

keys = list(rtree2.keys())
ntree2 = {}
for key in tqdm(keys, desc='NewHorizon'):
    branch = rtree2[key]
    mmask = massbranch(branch)
    mbranch = branch[mmask]
    vmask = velbranch(mbranch, snap2s)
    pmask = polybranch(mbranch, vmask, return_poly=False)
    mask = (vmask|pmask)
    ntree2[key] = mbranch[mask]
pklsave(ntree2, f"{database2}/stable_tree_new.pickle", overwrite=True)
snap2s.clear()
