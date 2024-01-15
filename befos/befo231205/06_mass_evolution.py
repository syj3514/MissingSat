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

mode = 'nh'
iout = 1026
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)
database = f"/home/jeon/MissingSat/database"

from common_func import *







LG = pklload(f"{database}/LG")
allsubs = None
states = None
for key in LG.keys():
    subs = LG[key]['subs']
    real = LG[key]['real']
    dink = real[real['state']=='dink']['hid']
    ind = isin(subs['id'], dink)
    subs['dink'][ind] = True
    subs['dink'][~ind] = False
    state = np.zeros(len(subs), dtype='<U7')
    state[ind] = 'dink'
    state[~ind] = 'pair'
    upair = real[real['state']=='upair']['hid']
    ind = isin(subs['id'], upair)
    state[ind] = 'upair'

    allsubs = subs if allsubs is None else np.hstack((allsubs, subs))
    states = state if states is None else np.hstack((states, state))
argsort = np.argsort(allsubs['id'])
allsubs = allsubs[argsort]
states = states[argsort]

tree = pklload(f"{database}/02_main_progenitors.pickle")
fnames = os.listdir(f"{database}/main_prog")
fnames = [fname for fname in fnames if(fname.startswith('subhalos'))]
fnames.sort(reverse=True)
tmp = pklload(f"{database}/main_prog/{fnames[0]}")[0]
trees = np.empty(tmp.shape[0]*len(fnames), dtype=tmp.dtype)
cursor = 0
uri.timer.verbose=0
for fname in tqdm( fnames ):
    tmp = pklload(f"{database}/main_prog/{fname}")[0]
    trees[cursor : cursor+len(tmp)] = tmp
    cursor += len(tmp)

colors = {'dink':'salmon', 'pair':'lightskyblue', 'upair':'yellowgreen'}
zorders = {'dink':0, 'pair':2, 'upair':1}



for key in LG.keys():
    if(os.path.exists(f"{database}/photo/06_evolution/nh_{key:04d}_2dorbit.png")): continue
    fig, ax = fancy_axis(figsize=(6,6), dpi=300)
    ax.set_aspect(1)

    BGG = LG[key]['BGG']
    subs = LG[key]['subs']
    real = LG[key]['real']

    for ireal in tqdm(real, desc=f'[{key:04d}]'):
        if(ireal['hid']>0):
            sub = subs[subs['id'] == ireal['hid']][0]
            state = ireal['state']
            color = colors[state]
            zorder = zorders[state]
            tmp = trees[trees['lastid'] == sub['id']]
            tmp = tmp[tmp['give_score']>0]
            mask = tmp['give_score']*tmp['take_score'] > 0.1
            if(np.sum(mask)/len(mask) > 0.9):
                ax.plot(tmp[mask]['x'], tmp[mask]['y'], color=color, lw=0.1, zorder=zorder)
    cir = circle(BGG, rname='r200_code', zorder=2.5)
    ax.add_patch(cir)
    ax.text(BGG['x'],BGG['y']+BGG['r200_code'],f"Group{key}",color='white',ha='center',va='bottom',zorder=3)
    
    plt.savefig(f"{database}/photo/06_evolution/nh_{key:04d}_2dorbit.png", dpi=400, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close()