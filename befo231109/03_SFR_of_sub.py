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


mode = 'nh'
iout = 1026
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)

LG = pklload("./database/00_LocalGroup_fix.pickle")
print(LG.keys())
print(LG[2].keys())


subss = None
hosts = None
isdinkss = None
sfrss = None
sfrss_rvir = None
for key in LG.keys():
    print(f"\nID={key:04d}")
    myLG = LG[key]
    BGG = myLG['BGG']

    reals = myLG['real']
    dinks = reals[reals['state']=='dink']
    pairs = reals[reals['state']=='pair']

    hids = reals[reals['hid']>0]['hid']
    subs = hals[hids-1]
    host = np.full(len(subs), key, dtype=np.int32)
    isdinks = isin(hids, dinks['hid'])
    sfrs = np.zeros(len(hids))
    sfrs_rvir = np.zeros(len(hids))
    for i, sub in tqdm(enumerate(subs), total=len(subs)):
        istar = uri.Particle(pklload(f"./database/parts/insub/nh_star_{key:04d}_{sub['id']:07d}.pickle"), snap)
        if(len(istar)>0):
            sfrs[i] = np.sum(istar[istar['age','Myr'] < 100]['m','Msol'])/1e8
            istar_rvir = cut_sphere(istar, sub['x'],sub['y'],sub['z'],sub['rvir'])
            if(len(istar_rvir)>0):
                sfrs_rvir[i] = np.sum(istar_rvir[istar_rvir['age','Myr'] < 100]['m','Msol'])/1e8
    
    sfrss = sfrs if(sfrss is None) else np.hstack((sfrss, sfrs))
    sfrss_rvir = sfrs_rvir if(sfrss_rvir is None) else np.hstack((sfrss_rvir, sfrs_rvir))
    subss = subs if(subss is None) else np.hstack((subss, subs))
    hosts = host if(hosts is None) else np.hstack((hosts, host))
    isdinkss = isdinks if(isdinkss is None) else np.hstack((isdinkss, isdinks))

SFRs = np.rec.fromarrays((hosts, subss['id'],isdinkss, sfrss, sfrss_rvir), names=('host','sub','dink','sfr','sfr_rvir'))
pklsave(SFRs, f"./database/03_SFR_in_sub.pickle", overwrite=True)