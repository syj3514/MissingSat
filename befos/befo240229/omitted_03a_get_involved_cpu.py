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

from common_func import *





tree = pklload(f"{database}/02_main_progenitors.pickle")
if(os.path.exists(f"{database}/halo_dict.pickle")):
    halos = pklload(f"{database}/halo_dict.pickle")
else:
    halos = {'catalog':{}, 'index':{}}
    uri.timer.verbose=0
    for iout in tqdm(np.unique(tree['timestep'])):
        isnap = snaps.get_snap(iout)
        ihals = uhmi.HaloMaker.load(isnap, galaxy=False, double_precision=dp)
        indicies = np.zeros(len(ihals), dtype=int)
        iids = tree[tree['timestep'] == iout]['id']
        ihals = ihals[iids-1]
        indicies[iids-1] = np.arange(len(iids))
        halos['catalog'][iout] = ihals
        halos['index'][iout] = indicies   
    pklsave(halos, f"{database}/halo_dict.pickle")



def _ibox(h, factor=1):
    return np.array([
                    [h['x']-factor*h['r'], h['x']+factor*h['r']],
                    [h['y']-factor*h['r'], h['y']+factor*h['r']],
                    [h['z']-factor*h['r'], h['z']+factor*h['r']]
                    ])

uri.timer.verbose=0
for iout in np.unique(tree['timestep'])[::-1]:
    if(os.path.exists(f"{database}/main_prog/cpulist/cpulist_{iout:05d}.pickle")): continue
    cpudict = {}
    targets = halos['catalog'][iout]
    isnap = snaps.get_snap(iout)

    cpulists = []
    with Pool(32) as pool:
        async_result = [
                    pool.apply_async(
                            uri.get_cpulist, 
                            (_ibox(h,factor=1.1), None, isnap.levelmax, isnap.bound_key, isnap.ndim, 5, isnap.ncpu)
                            ) for h in targets
                    ]
        iterobj = tqdm(async_result, total=len(targets), desc=f"iout={iout:05d}")
        for r in iterobj:
            cpulists.append(r.get())
    cpulists = np.unique( np.concatenate(cpulists) )
    cpudict['all'] = cpulists
    pklsave(cpudict, f"{database}/main_prog/cpulist/cpulist_{iout:05d}.pickle")
    print(f"`{database}/main_prog/cpulist/cpulist_{iout:05d}.pickle` save done")
    isnap.clear()