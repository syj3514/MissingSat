from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize
from common_func import *

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




LG = pklload(f"{database}/00_LocalGroup_final_h_addudg.pickle")


mindm = 1e10
uri.timer.verbose=0
cmap_dm = cmr.jungle
for key in LG.keys():
    
    BGG = LG[key]['BGG']
    sats = LG[key]['sats']
    subs = LG[key]['subs']
    real = LG[key]['real']

    for sub in tqdm(subs, desc=f"[{key:04d}]"):
        myreal = real[real['hid'] == sub['id']][0]
        state = myreal['state']

        isstar = False
        dm = pklload(f"{database}/parts/insub/nh_dm_{key:04d}_{sub['id']:07d}.pickle")
        mindm = min(mindm, 1.1*dm['m'].min())
        lowdm = dm[dm['m']  >  mindm]
        x1 = dm['x'].min(); x2 = dm['x'].max()
        y1 = dm['y'].min(); y2 = dm['y'].max()
        z1 = dm['z'].min(); z2 = dm['z'].max()
        star = pklload(f"{database}/parts/insub/nh_star_{key:04d}_{sub['id']:07d}.pickle")
        if(len(star)>0):
            isstar = True
            x1 = min(x1, star['x'].min()); x2 = max(x2, star['x'].max())
            y1 = min(y1, star['y'].min()); y2 = max(y2, star['y'].max())
            z1 = min(z1, star['z'].min()); z2 = max(z2, star['z'].max())
        cell = pklload(f"{database}/parts/insub/nh_cell_{key:04d}_{sub['id']:07d}.pickle")
        if(len(cell)>0):
            x1 = min(x1, cell['x'].min()); x2 = max(x2, cell['x'].max())
            y1 = min(y1, cell['y'].min()); y2 = max(y2, cell['y'].max())
            z1 = min(z1, cell['z'].min()); z2 = max(z2, cell['z'].max())
        box = np.array([[x1, x2], [y1, y2], [z1, z2]])
        write_subbox(sub['id'], key=key, newpos=(x1,x2,y1,y2,z1,z2))
        
        snap.box = box
        star = uri.Particle(star,snap)
        dm = uri.Particle(dm,snap)
        cell = uri.Cell(cell,snap)
        
        if(isstar): rband = measure_luminosity(star, 'SDSS_r')
        cellmap = painter.gasmap(cell, box=snap.box, shape=480)
        plarge,psmall = np.percentile(cellmap[cellmap>0], [99.9999,5])
        dmmap_raw = painter.partmap(dm, box=snap.box, shape=480)
        dmmap = gaussian_filter(dmmap_raw, sigma=3)
        if(isstar): starmap = painter.partmap(star, box=snap.box, weights=rband, shape=480)
        else: starmap = np.zeros_like(dmmap)

        cmap_star = drawer.make_cmap([(0,0,0),(1,0,0),(1,1,0),(1,1,1)], position=[0,0.4,0.8,1])
        cmap_dm = drawer.make_cmap([(0,0,0),(0.1,0.25,0.15),(0,0.5,0)], position=[0,0.5,1])

        titles = ["All Composite Image", "Star+DM Composite Image", "Star+Gas Composite Image", "DM+Gas Composite Image"]
        suffixs = ["All", "SD", "SG", "DG"]
        for ith in range(4):
            title = titles[ith]; suffix = suffixs[ith]
            if(ith==0):
                composite = painter.composite_image(
                    [starmap, cellmap, dmmap], 
                    cmaps=[cmap_star, cmr.neutral, cmap_dm],
                    qscales=[4.5,np.log10(plarge/psmall),3.5],
                    mode='screen',
                    vmaxs = [np.nanmax(starmap), plarge, np.nanmax(dmmap)*2]
                    )
            elif(ith==1):
                composite = painter.composite_image(
                    [starmap, dmmap], 
                    cmaps=[cmap_star, cmap_dm],
                    qscales=[4.5,3.5],
                    mode='screen',
                    vmaxs = [np.nanmax(starmap), np.nanmax(dmmap)*2]
                    )
            elif(ith==2):
                composite = painter.composite_image(
                    [starmap, cellmap], 
                    cmaps=[cmap_star, cmr.neutral],
                    qscales=[4.5,np.log10(plarge/psmall)],
                    mode='screen',
                    vmaxs = [np.nanmax(starmap), plarge]
                    )
            elif(ith==3):
                composite = painter.composite_image(
                    [dmmap, cellmap], 
                    cmaps=[cmap_dm, cmr.neutral],
                    qscales=[3.5,np.log10(plarge/psmall)],
                    mode='screen',
                    vmaxs = [np.nanmax(dmmap)*2, plarge]
                    )
            

            fig, ax = fancy_axis(figsize=(8,8), dpi=200)
            fig1, ax1 = fancy_axis(figsize=(8,8), dpi=200)
            ax.imshow(composite, origin='lower', extent=snap.box[:2].flatten(), aspect='equal')
            ax1.imshow(composite, origin='lower', extent=snap.box[:2].flatten(), aspect='equal')
            if(len(lowdm)>0):
                ax.scatter(lowdm['x'], lowdm['y'], s=20, marker='p', ec='none', fc='magenta', alpha=1)
                ax1.scatter(lowdm['x'], lowdm['y'], s=20, marker='p', ec='none', fc='magenta', alpha=1)
                # in rvir
                if(ith==0):
                    lowdm_vir = cut_sphere(lowdm, sub['x'], sub['y'], sub['z'], sub['rvir'])
                    dm_vir = cut_sphere(dm, sub['x'], sub['y'], sub['z'], sub['rvir'])
                    frac_vir = np.sum(lowdm_vir['m'])/np.sum(dm_vir['m'])
                    addtext(
                        "Contaminated!", 
                        ax=ax, loc='upper right', color='magenta', fontsize=10, offset=0.025, dx=0.04)
                    addtext(
                        "","",
                        f"in Rmax: {np.sum(lowdm['m'])/np.sum(dm['m'])*100:.2f}%", 
                        f"in Rvir: {frac_vir*100:.2f}%", 
                        f"Of member: {sub['mcontam']/sub['m']*100:.2f}%",
                        ax=ax, loc='upper right', color='magenta', fontsize=8, offset=0.025, dx=0.02)
            cir = plt.Circle((sub['x'], sub['y']), sub['rvir'], color='w', fill=False, lw=0.5, ls=':', zorder=10)
            ax.add_patch(cir)
            if(ith==0): ax.text(sub['x'], sub['y']+sub['rvir'], fr"$R_{{vir}}={sub['rvir']/snap.unit['kpc']:.2f}$ kpc", color='white', fontsize=12, ha='center', va='bottom')
            if(ith!=2):
                cir = plt.Circle((sub['x'], sub['y']), sub['r50_mem'], color='w', fill=False, lw=0.5, ls=':', zorder=10)
                ax.add_patch(cir)
                if(ith==0): ax.text(sub['x'], sub['y']-sub['r50_mem'], fr"$R_{{50,ofMem}}={sub['r50_mem']/snap.unit['kpc']:.2f}$ kpc", color='white', fontsize=12, ha='center', va='top')
            if(state=='pair')or(state=='upair'):
                isat = sats[sats['id']==myreal['gid']][0]
                cir = plt.Circle((isat['x'], isat['y']), isat['r'], color='cyan', fill=False, lw=0.5, ls='-', zorder=10)
                ax.add_artist(cir)
                if(ith==0): ax.text(isat['x'], isat['y']+isat['r'], fr"$R_{{max}}={isat['r']/snap.unit['kpc']:.2f}$ kpc", color='cyan', fontsize=12, ha='center', va='bottom')
                cir = plt.Circle((isat['x'], isat['y']), isat['r50r'], color='cyan', fill=False, lw=0.5, ls='-', zorder=10)
                ax.add_artist(cir)
                if(ith==0): ax.text(isat['x'], isat['y']-isat['r50r'], fr"$R_{{eff}}={isat['r50r']/snap.unit['kpc']:.2f}$ kpc", color='cyan', fontsize=12, ha='center', va='top')
            
            neighbors = cut_sphere(subs[subs['id']!=sub['id']], sub['x'], sub['y'], sub['z'], sub['r'], both_sphere=True, rname='rvir')
            for neigh in neighbors:
                neighreal = real[real['hid']==neigh['id']][0]
                if(neighreal['state']=='pair'):
                    ls = '-'
                    if(ith!=3):
                        isat = sats[sats['id']==neighreal['gid']][0]
                        cir = plt.Circle((isat['x'], isat['y']), isat['r'], color='cyan', fill=False, lw=0.5, ls=ls, zorder=10)
                        ax.add_artist(cir)
                elif(neighreal['state']=='upair'):
                    ls = '-'
                    if(ith!=3):
                        isat = sats[sats['id']==neighreal['gid']][0]
                        cir = plt.Circle((isat['x'], isat['y']), isat['r'], color='yellow', fill=False, lw=0.5, ls=ls, zorder=10)
                        ax.add_artist(cir)
                else:
                    ls = '--'
                cir = plt.Circle((neigh['x'], neigh['y']), neigh['rvir'], color='tomato', fill=False, lw=0.5, ls=ls, zorder=10)
                ax.add_artist(cir)               

            add_scalebar(ax, snap.unit_l)
            if(ith==0): addtext(
                f"log$M_{{200}}$={np.log10(sub['m200']):.2f}", 
                f"log$M_{{vir}}$={np.log10(sub['mvir']):.2f}", 
                f"log$M_{{HaloMaker}}$={np.log10(sub['m']):.2f}", 
                f"log$M_{{DM,Rmax}}$={np.log10(sub['mdm']):.2f}", 
                f"log$M_{{Star,Rmax}}$={np.log10(sub['mstar']):.2f}" if(sub['mstar']>1e3) else f"log$M_{{Star,Rmax}}$=None", 
                f"log$M_{{Gas,Rmax}}$={np.log10(sub['mcell']):.2f}", 
                f"log$M_{{Cold,Rmax}}$={np.log10(sub['mcold']):.2f}" if(sub['mcold']>1e3) else f"log$M_{{Cold,Rmax}}$=None", 
                ax=ax, loc='lower left', color='white', fontsize=10, offset=0.025, dx=0.04)
            addtext(
                f"NewHorizon",
                f"  LocalGroup {key}",
                ax=ax, loc='upper left', color='white', fontsize=12, offset=0.025, dx=0.04)

            ax.text(0.5, 0.95, title, ha='center', va='center', fontsize=15, color='white', transform=ax.transAxes, family='serif')
            ax.text(0.5, 0.90, f"Subhalo {sub['id']} ({state})", ha='center', va='center', fontsize=11, color='white', transform=ax.transAxes, family='monospace')
            if(state=='pair'):
                ax.text(0.5, 0.87, f"Satellite {myreal['gid']}", ha='center', va='center', fontsize=11, color='cyan', transform=ax.transAxes, family='monospace')

            ax.set_xlim(*snap.box[0])
            ax.set_ylim(*snap.box[1])
            ax1.set_xlim(*snap.box[0])
            ax1.set_ylim(*snap.box[1])

            fig.savefig(f"{database}/photo/gallery/info/{suffix}/NH_sub{sub['id']:07d}.png", dpi=400, facecolor='none', bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            fig1.savefig(f"{database}/photo/gallery/clean/{suffix}/NH_sub{sub['id']:07d}.png", dpi=400, facecolor='none', bbox_inches='tight', pad_inches=0)
            plt.close(fig1)