from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker

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



subids = None
for key in LG.keys():
    tmp = LG[key]['real']['hid']
    tmp = tmp[tmp>0]
    subids = tmp if(subids is None) else np.union1d(subids, tmp)
print( subids.shape )

for subid in tqdm( subids ):
    if(os.path.exists(f"./database/submember/z017/DM_{subid:07d}.pickle")): continue
    mem = uhmi.HaloMaker.read_member_part(snap, subid, galaxy=False, target_fields=['x','y','z','id'])
    pklsave(mem.table, f"./database/submember/z017/DM_{subid:07d}.pickle")



zreds = np.array([0.17, 0.3, 0.5,0.7, 1,1.2, 1.5, 2,2.5, 3])
aexps = 1/(1+zreds)
iouts = np.zeros(len(zreds), dtype=np.int32)
for i in range(len(zreds)):
    arg = np.argmin(np.abs(snaps.iout_avail['aexp']-aexps[i]))
    iouts[i] = snaps.iout_avail['iout'][arg]
print(iouts)



for iout in iouts:
    isnap = snaps.get_snap(iout)
    dirname = f"z{round(isnap.z*100):03d}"
    if(not os.path.isdir(f"./database/submember/{dirname}")):
        os.makedirs(f"./database/submember/{dirname}")
    lis = os.listdir(f"./database/submember/{dirname}")
    if(len(lis) == len(subids)): continue
    print(f"[{iout:04d} ({dirname})]")
    isnap.get_part(pname='dm', nthread=32, target_fields=['x','y','z','id'])
    print("Get part done")
    assert (isnap.part['id']>0).all()
    argsort = np.argsort(isnap.part['id'])
    dm = isnap.part[argsort]
    print("Argsort done")

    for subid in tqdm(subids):
        if(os.path.exists(f"./database/submember/{dirname}/DM_{subid:07d}.pickle")): continue
        memid = pklload(f"./database/submember/z017/DM_{subid:07d}.pickle")['id']
        nowmem = dm[memid-1]
        pklsave(nowmem.table, f"./database/submember/{dirname}/DM_{subid:07d}.pickle")
    isnap.clear()