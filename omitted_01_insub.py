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



mode = 'nh2'
database = f"/home/jeon/MissingSat/database/nh2"
iout = 797
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)



LG = pklload(f"{database}/LG")
print(LG.keys())
print(LG[2].keys())



for key in LG.keys():
    subs = LG[key]['subs']
    if(os.path.exists(f"./database/parts/insub/nh2_dm_{key:04d}_{subs[-1]['id']:07d}.pickle")): continue

    star = uri.Particle(pklload(f"./database/parts/nh2_star_{key:04d}.pickle"), snap)
    cell = uri.Particle(pklload(f"./database/parts/nh2_cell_{key:04d}.pickle"), snap)
    dm = uri.Particle(pklload(f"./database/parts/nh2_dm_{key:04d}.pickle"), snap)

    for i, sub in tqdm(enumerate(subs), total=len(subs)):
        if(os.path.exists(f"./database/parts/insub/nh2_dm_{key:04d}_{sub['id']:07d}.pickle")): continue
        istar = cut_sphere(star, sub['x'],sub['y'],sub['z'],sub['r'])
        icell = cut_sphere(cell, sub['x'],sub['y'],sub['z'],sub['r'])
        idm = cut_sphere(dm, sub['x'],sub['y'],sub['z'],sub['r'])
        pklsave(istar.table, f"./database/parts/insub/nh2_star_{key:04d}_{sub['id']:07d}.pickle")
        pklsave(icell.table, f"./database/parts/insub/nh2_cell_{key:04d}_{sub['id']:07d}.pickle")
        pklsave(idm.table, f"./database/parts/insub/nh2_dm_{key:04d}_{sub['id']:07d}.pickle")
