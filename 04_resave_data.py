from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import numpy as np
import os, glob
import time
import warnings

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind
from icl_draw import drawsnap, add_scalebar, addtext
import argparse, subprocess




print("\n\n[Data Preparation]")
mode = 'nh'
fout = 1026
repo,rurmode,dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, fout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)



uri.timer.verbose=0
LG = pklload(f"./database/11_LocalGroup.pickle")
MWAs = None
for key in LG.keys():
    if(MWAs is None): MWAs = LG[key]['BGG']
    else: MWAs = np.hstack((MWAs, LG[key]['BGG']))

for MWA in tqdm( MWAs, total=len(MWAs) ):
    rrange = MWA['r200_code']
    snap.set_box_halo(MWA, radius=3*rrange, use_halo_radius=False)
    snap.get_part(nthread=36)
    pklsave(snap.part['star'].table, f"./database/parts/nh_star_{MWA['id']:04d}.pickle", overwrite=True)
    pklsave(snap.part['dm'].table, f"./database/parts/nh_dm_{MWA['id']:04d}.pickle", overwrite=True)
    snap.get_cell(nthread=36)
    pklsave(snap.cell.table, f"./database/parts/nh_cell_{MWA['id']:04d}.pickle", overwrite=True)
    snap.clear()