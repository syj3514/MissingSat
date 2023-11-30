'''
Purpose:

I mistake that in some snapshots, I mis-calculate the `mdm` and `mdm_vir` without unit conversion.
So, Here I want to fix it.
'''

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




fnames = os.listdir(f"{database}/main_prog")
fnames = [fname for fname in fnames if(fname.startswith('subhalos'))]
fnames.sort(reverse=True)


uri.timer.verbose=0
for fname in tqdm( fnames ):
    tmp = pklload(f"{database}/main_prog/{fname}")
    tmp,readme = tmp[0],tmp[1]
    if(np.min(tmp['mdm_vir'])<600000):
        iout = int(fname[9:14])
        isnap = snaps.get_snap(iout)
        tmp['mdm_vir'] = tmp['mdm_vir']/isnap.unit['Msol']
        tmp['mdm'] = tmp['mdm']/isnap.unit['Msol']
        os.rename(f"{database}/main_prog/{fname}", f"{database}/main_prog/backup/{fname}")
        pklsave((tmp,readme), f"{database}/main_prog/{fname}")
        print(f"`{fname}` is fixed.")