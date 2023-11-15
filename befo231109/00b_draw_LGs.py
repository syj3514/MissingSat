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



uri.timer.verbose=0
for key in tqdm(LG.keys()):
    BGG = LG[key]['BGG']
    sats = LG[key]['sats']
    subs = LG[key]['subs']
    real = LG[key]['real']

    
    star = pklload(f"./database/parts/nh_star_{key:04d}.pickle")
    dm = pklload(f"./database/parts/nh_dm_{key:04d}.pickle")
    cell = pklload(f"./database/parts/nh_cell_{key:04d}.pickle")
    x1 = min(star['x'].min(), dm['x'].min(), cell['x'].min())
    x2 = max(star['x'].max(), dm['x'].max(), cell['x'].max())
    y1 = min(star['y'].min(), dm['y'].min(), cell['y'].min())
    y2 = max(star['y'].max(), dm['y'].max(), cell['y'].max())
    z1 = min(star['z'].min(), dm['z'].min(), cell['z'].min())
    z2 = max(star['z'].max(), dm['z'].max(), cell['z'].max())
    box = np.array([[x1, x2], [y1, y2], [z1, z2]])
    LG[key]['box'] = box
    if(os.path.exists(f"./database/photo/00_LG_image/NH_{key:04d}_total.png")):
        continue
    snap.box = box
    star = uri.Particle(star,snap)
    dm = uri.Particle(dm,snap)
    cell = uri.Cell(cell,snap)
    
    rband = measure_luminosity(star, 'SDSS_r')
    starmap = painter.partmap(star, box=snap.box, weights=rband, shape=1080)
    cellmap = painter.gasmap(cell, box=snap.box, shape=1080)
    dmmap_raw = painter.partmap(dm, box=snap.box, shape=1080)
    dmmap = gaussian_filter(dmmap_raw, sigma=3)

    cmap_star = drawer.make_cmap([(0,0,0),(1,0,0),(1,1,0),(1,1,1)], position=[0,0.4,0.8,1])
    cmap_dm = drawer.make_cmap([(0,0,0),(0.1,0.25,0.15),(0,0.5,0)], position=[0,0.5,1])
    composite = painter.composite_image(
        [starmap, cellmap, dmmap], 
        cmaps=[cmap_star, cmr.neutral, cmap_dm],
        qscales=[4.5,4,3.5],
        mode='screen',
        vmaxs = [np.nanmax(starmap)*0.9, np.nanmax(cellmap)*10, np.nanmax(dmmap)*5]
        )
    fig, ax = fancy_axis(figsize=(8,8), dpi=200)
    ax.imshow(composite, origin='lower', extent=snap.box[:2].flatten(), aspect='equal')
    cir = plt.Circle((BGG['x'], BGG['y']), BGG['r200_code'], color='w', fill=False, lw=0.5, ls=':')
    ax.add_patch(cir)
    cir = plt.Circle((BGG['x'], BGG['y']), 1.5*BGG['r200_code'], color='w', fill=False, lw=0.5, ls=':')
    ax.add_patch(cir)

    for ireal in real:
        if(ireal['state'])=='pair':
            isat = sats[sats['id']==ireal['gid']][0]
            cir = plt.Circle((isat['x'], isat['y']), isat['r'], color='cyan', fill=False, lw=0.5, ls='-')
            ax.add_patch(cir)
            isub = subs[subs['id']==ireal['hid']][0]
            cir = plt.Circle((isub['x'], isub['y']), isub['rvir'], color='tomato', fill=False, lw=0.5, ls='-')
            ax.add_patch(cir)
        elif(ireal['state'])=='orphan':
            isat = sats[sats['id']==ireal['gid']][0]
            cir = plt.Circle((isat['x'], isat['y']), isat['r'], color='cyan', fill=False, lw=0.25, ls='--')
            ax.add_patch(cir)
        elif(ireal['state'])=='dink':
            isub = subs[subs['id']==ireal['hid']][0]
            cir = plt.Circle((isub['x'], isub['y']), isub['rvir'], color='tomato', fill=False, lw=0.25, ls='--')
            ax.add_patch(cir)
        

    add_scalebar(ax, snap.unit_l)
    addtext(f"{np.sum(real['state']=='dink')} DINKs", f"{np.sum(real['state']=='orphan')} Orphans", f"{np.sum(real['state']=='pair')} pairs", ax=ax, loc='lower left', color='white', fontsize=12, offset=0.025, dx=0.04)
    addtext(f"NewHorizon\n\nGalaxy {key}\nHalo {BGG['halo_id']}",ax=ax, loc='upper left', color='white', fontsize=12, offset=0.025)

    plt.savefig(f"./database/photo/00_LG_image/NH_{key:04d}_total.png", dpi=400, facecolor='none', bbox_inches='tight', pad_inches=0)
    plt.close()
pklsave(LG, "./database/00_LocalGroup_fix.pickle", overwrite=True)