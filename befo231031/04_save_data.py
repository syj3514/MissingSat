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
mode1 = 'nh'
fout1 = 1026
repo1,rurmode1,dp1 = mode2repo(mode1)
snap1 = uri.RamsesSnapshot(repo1, fout1, mode=rurmode1)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']
gal1s = uhmi.HaloMaker.load(snap1, galaxy=True, double_precision=dp1)
hal1s = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp1)

mode2 = 'nh2'
fout2 = 797
repo2,rurmode2,dp2 = mode2repo(mode2)
snap2 = uri.RamsesSnapshot(repo2, fout2, mode=rurmode2)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']
gal2s = uhmi.HaloMaker.load(snap2, galaxy=True, double_precision=dp2)
hal2s = uhmi.HaloMaker.load(snap2, galaxy=False, double_precision=dp2)



uri.timer.verbose=0
MWA1s = pklload(f"./database/03_MWA1s.pickle")
MWA2s = pklload(f"./database/03_MWA2s.pickle")

for MWA1, MWA2 in tqdm( zip(MWA1s, MWA2s), total=len(MWA1s) ):
    rrange = max(MWA1['r200_code'], MWA2['r200_code'])
    pass1 = rrange <= MWA1['r200_code']
    pass2 = rrange <= MWA2['r200_code']
    snap1.set_box_halo(MWA1, radius=1.5*rrange, use_halo_radius=False)
    snap2.set_box_halo(MWA2, radius=1.5*rrange, use_halo_radius=False)
    # if(not os.path.exists(f"./database/parts/nh_star_{MWA1['id']:04d}.pickle")):
    if(pass1): print(f"Pass NH part of {MWA1['id']}")
    else:
        snap1.get_part(nthread=36)
        pklsave(snap1.part['star'].table, f"./database/parts/nh_star_{MWA1['id']:04d}.pickle", overwrite=True)
        pklsave(snap1.part['dm'].table, f"./database/parts/nh_dm_{MWA1['id']:04d}.pickle", overwrite=True)
    # if(not os.path.exists(f"./database/parts/nh2_star_{MWA2['id']:04d}.pickle")):
    if(pass2): print(f"Pass NH2 part of {MWA2['id']}")
    else:
        snap2.get_part(nthread=36)
        pklsave(snap2.part['star'].table, f"./database/parts/nh2_star_{MWA2['id']:04d}.pickle", overwrite=True)
        pklsave(snap2.part['dm'].table, f"./database/parts/nh2_dm_{MWA2['id']:04d}.pickle", overwrite=True)
    # if(not os.path.exists(f"./database/parts/nh_cell_{MWA1['id']:04d}.pickle")):
    if(pass1): print(f"Pass NH cell of {MWA1['id']}")
    else:
        snap1.get_cell(nthread=36)
        pklsave(snap1.cell.table, f"./database/parts/nh_cell_{MWA1['id']:04d}.pickle", overwrite=True)
    # if(not os.path.exists(f"./database/parts/nh2_cell_{MWA2['id']:04d}.pickle")):
    if(pass2): print(f"Pass NH2 cell of {MWA2['id']}")
    else:
        snap2.get_cell(nthread=36)
        pklsave(snap2.cell.table, f"./database/parts/nh2_cell_{MWA2['id']:04d}.pickle", overwrite=True)
    snap1.clear()
    snap2.clear()