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



mdms = None
mstars = None
mcolds = None
mcells = None

subss = None
hosts = None

isdinkss = None
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
    mstar = np.zeros(len(hids))
    mcell = np.zeros(len(hids))
    mcold = np.zeros(len(hids))
    mdm = np.zeros(len(hids))
    for i, sub in tqdm(enumerate(subs), total=len(subs)):
        istar = uri.Particle(pklload(f"./database/parts/insub/nh_star_{key:04d}_{sub['id']:07d}.pickle"), snap)
        istar = cut_sphere(istar, sub['x'],sub['y'],sub['z'],sub['rvir'])
        if(len(istar)>0): mstar[i] = np.sum(istar['m','Msol'])

        idm = uri.Particle(pklload(f"./database/parts/insub/nh_dm_{key:04d}_{sub['id']:07d}.pickle"), snap)
        idm = cut_sphere(idm, sub['x'],sub['y'],sub['z'],sub['rvir'])
        if(len(idm)>0): mdm[i] = np.sum(idm['m','Msol'])

        icell = uri.Cell(pklload(f"./database/parts/insub/nh_cell_{key:04d}_{sub['id']:07d}.pickle"), snap)
        icell = cut_sphere(icell, sub['x'],sub['y'],sub['z'],sub['rvir'])
        if(len(icell)>0): mcell[i] = np.sum(icell['m','Msol'])

        icold = icell[icell['T','K'] < 2e4]
        if(len(icold)>0): mcold[i] = np.sum(icold['m','Msol'])


    where = mstar==0
    mstar[where] += 1e3
    where = mcell==0
    mcell[where] += 1e3
    where = mcold==0
    mcold[where] += 1e3
    where = mdm==0
    mdm[where] += 1e3

    mdms = mdm if(mdms is None) else np.hstack((mdms, mdm))
    mstars = mstar if(mstars is None) else np.hstack((mstars, mstar))
    mcolds = mcold if(mcolds is None) else np.hstack((mcolds, mcold))
    mcells = mcell if(mcells is None) else np.hstack((mcells, mcell))
    subss = subs if(subss is None) else np.hstack((subss, subs))
    hosts = host if(hosts is None) else np.hstack((hosts, host))
    isdinkss = isdinks if(isdinkss is None) else np.hstack((isdinkss, isdinks))


    if(not os.path.exists(f"./database/photo/01_compare_pairs/NH_{key:04d}_mratio_rvir.png")):
        fig, ax = plt.subplots(dpi=300, figsize=(8,6))
        for i,sub in enumerate(subs):
            isdink = sub['id'] in dinks['hid']
            color = 'k' if(mcold[i]<=1e3) else 'deepskyblue'
            facecolor='none' if(isdink) else color
            edgecolor=color if(isdink) else 'none'
            linecolor=color
            lw = 0.1 if(isdink) else 0.3

            ax.scatter(np.log10(mdm[i]), np.log10(mstar[i]), facecolor=facecolor, edgecolor=edgecolor, s=50, lw=0.1, marker='*')
            ax.scatter(np.log10(mdm[i]), np.log10(mcold[i]), facecolor=facecolor, edgecolor=edgecolor, s=50, lw=0.1, marker='H')
            ax.scatter(np.log10(mdm[i]), np.log10(mcell[i]), facecolor=facecolor, edgecolor=edgecolor, s=50, lw=0.1)
            vmax = np.max([np.log10(mstar[i]), np.log10(mcell[i]), np.log10(mcold[i])])
            vmin = np.min([np.log10(mstar[i]), np.log10(mcell[i]), np.log10(mcold[i])])
            ax.plot([np.log10(mdm[i]),np.log10(mdm[i])], [vmin, vmax], color=linecolor, lw=lw)

        ax.fill_between([7.9,12.1],3, np.log10(np.min(gals['m'])), ec='none', fc='grey', alpha=0.2, zorder=-1)

        ax.set_xlim(8,12)
        xlim = ax.get_xlim()
        ax.plot(xlim, xlim, ls=':', color='grey', zorder=-1)
        ax.set_ylim(3.01,12)
        ax.set_xlabel(r"$\log\ M_{halo}/M_{\odot}$")
        ax.set_ylabel(r"$\log\ M/M_{\odot}$")

        hascold = mcold > 1e3
        haspair = isin(subs['id'], pairs['hid'])
        ax.text(0.03, 0.95, fr"$\mathtt{{ N_{{subh}}: {len(subs)} }}$", ha='left', va='top', color='k', fontsize=9, transform=ax.transAxes)
        ax.text(0.03, 0.90, fr"$\mathtt{{ N_{{dink}}: {len(dinks)} }}$", ha='left', va='top', color='k', fontsize=9, transform=ax.transAxes)
        ax.text(0.03, 0.85, fr"$\mathtt{{ N_{{pair}}: {len(pairs)} }}$", ha='left', va='top', color='k', fontsize=9, transform=ax.transAxes)
        ax.text(0.03, 0.80, fr"$\mathtt{{ N_{{has cold}}: {np.sum(hascold)} }}$", ha='left', va='top', color='deepskyblue', fontsize=9, transform=ax.transAxes)
        ax.text(0.25, 0.95, fr"$\mathtt{{ P(pair|cold): {100*np.sum(haspair&hascold)/np.sum(hascold):.1f}\% }}$", ha='left', va='top', color='k', fontsize=11, transform=ax.transAxes)
        ax.text(0.25, 0.90, fr"$\mathtt{{ P(cold|pair): {100*np.sum(haspair&hascold)/np.sum(haspair):.1f}\% }}$", ha='left', va='top', color='k', fontsize=11, transform=ax.transAxes)

        ax.text(0.97, 0.15, fr"$M_{{200}}: 10^{{{np.log10(BGG['m200']):.1f}}} M_{{\odot}}$", ha='right', va='top', color='k', fontsize=9, transform=ax.transAxes)
        ax.text(0.97, 0.10, fr"$M_{{BGG}}: 10^{{{np.log10(BGG['m']):.1f}}} M_{{\odot}}$", ha='right', va='top', color='k', fontsize=9, transform=ax.transAxes)

        # Making Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0],[0], marker='.',color='none', markeredgecolor='k', markerfacecolor='none', markersize=0, label=r'$\mathcal{Which\ Mass?}$'),
            Line2D([0],[0], marker='o',color='none', markeredgecolor='k', markerfacecolor='none', markersize=8, label=r'  $M_{gas}$'),
            Line2D([0],[0], marker='*',color='none', markeredgecolor='k', markerfacecolor='none', markersize=8, label=r'  $M_{star}$'),
            Line2D([0],[0], marker='H',color='none', markeredgecolor='k', markerfacecolor='none', markersize=8, label=r'  $M_{cold}$'),
            Line2D([0],[0], marker='.',color='none', markeredgecolor='k', markerfacecolor='none', markersize=0, label="----------\n"+r'$\mathcal{Has\ Galaxy?}$'),
            Line2D([0],[0], marker='o',color='none', markeredgecolor='k', markerfacecolor='none', markersize=8, label='  Dink'),
            Line2D([0],[0], marker='o',color='none', markeredgecolor='none', markerfacecolor='k', markersize=8, label='  Pair'),
            Line2D([0],[0], marker='.',color='none', markeredgecolor='k', markerfacecolor='none', markersize=0, label="----------\n"+r'$\mathcal{Has\ Cold\ Gas?}$'),
            Line2D([0],[0], marker='o',color='none', markeredgecolor='none', markerfacecolor='deepskyblue', markersize=8, label='  Yes'),
            Line2D([0],[0], marker='o',color='none', markeredgecolor='none', markerfacecolor='k', markersize=8, label='  No'),
        ]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        ax.set_aspect(1/2.5)
        fig.savefig(f"./database/photo/01_compare_pairs/NH_{key:04d}_mratio_rvir.png", dpi=400, facecolor='white')
        print(f"`./database/photo/01_compare_pairs/NH_{key:04d}_mratio_rvir.png` save done")
        plt.close()

mratios = np.rec.fromarrays((hosts, subss['id'],isdinkss, mdms, mstars, mcolds, mcells), names=('host','sub','dink','mdm','mstar','mcold','mcell'))
pklsave(mratios, f"./database/01_mass_in_sub_rvir.pickle", overwrite=True)