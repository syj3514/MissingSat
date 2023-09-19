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
uri.timer.verbose=0
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind, isin
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle
import argparse, subprocess
from importlib import reload
import cmasher as cmr






mode = 'nh'
iout = 1026
repo, rurmode, dp = mode2repo(mode)
snap1 = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']
gal1s = pklload("./database/01_nh_ghmatch.pickle")
hal1s = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp)

result1s = pklload(f"./database/03_MWA1s.pickle")
pair1s = pklload(f"./database/06_nh_subhalo_pairs.pickle")
scores = pklload(f"./database/08_nh_scores.pickle")
dm_scores = pklload("./database/08_nh_dm_scores.pickle")

cols = [
    "Host", "Sat", "r50m", "r90m", "r50r", "r90r", 
    "SFR_mem", "u_mem", "g_mem", "r_mem", "i_mem", "z_mem", "metal_mem", "ager_mem", "t50_mem", "t90_mem"] 
category = ["r50m", "r90m", "r50r", "r90r", "rmax"]


for icate in category:
    cols = cols+[f"SFR_{icate}", f"u_{icate}", f"g_{icate}", f"r_{icate}", f"i_{icate}", f"z_{icate}", f"metal_{icate}", f"ager_{icate}", f"t50_{icate}", f"t90_{icate}", f"mgas_{icate}", f"mcold_{icate}", f"mdm_{icate}"]

dtype = [(col, np.float64) for col in cols]
dtype[0] = ("Host", np.int32)
dtype[1] = ("Sat", np.int32)
dtype = np.dtype(dtype)

vad = np.genfromtxt("./database/09_value_added.txt", delimiter="\t", dtype=dtype, skip_header=1)


realmembers = {}
if(os.path.exists(f"./database/10_real_members.pickle")):
    realmembers = pklload(f"./database/10_real_members.pickle")

for MWA in tqdm(result1s):
    if(MWA['id'] in realmembers.keys()): continue
    if(MWA['r200_code'] < MWA['r']):
        print("WARNING! r200_code < r")
        print(f"\tID={MWA['id']}\n\tr={MWA['r']}\n\tr200_code={MWA['r200_code']}")
        continue
    print(f" > Data Processing {MWA['id']}...")
    target_id = MWA['id']
    target_hid = MWA['halo_id']
    snap1.set_box_halo(MWA, 1.5, radius_name='r200_code')

    gal_gives = scores['give'][target_id]
    gal_takes = scores['take'][target_id]
    hal_gives = dm_scores['give'][target_id]
    hal_takes = dm_scores['take'][target_id]
    values = vad[vad['Host'] == target_id]

    star = uri.Particle(pklload(f"./database/parts/nh_star_{target_id:04d}.pickle"), snap1 )
    dm = uri.Particle(pklload(f"./database/parts/nh_dm_{target_id:04d}.pickle"), snap1 )
    cell = uri.Particle(pklload(f"./database/parts/nh_cell_{target_id:04d}.pickle"), snap1 )
    rband = measure_luminosity(star, 'SDSS_r')
    starmap = painter.partmap(star, box=snap1.box, weights=rband, shape=1080)
    cellmap = painter.gasmap(cell, box=snap1.box, shape=1080)
    dmmap_raw = painter.partmap(dm, box=snap1.box, shape=1080)
    dmmap = gaussian_filter(dmmap_raw, sigma=3)



    print(f" > Drawing {MWA['id']}...")
    # DM map
    cmap_star = drawer.make_cmap([(0,0,0),(1,0,0),(1,1,0),(1,1,1)], position=[0,0.4,0.8,1])
    composite = painter.composite_image(
        [cellmap, dmmap], 
        cmaps=[cmr.neutral, cmr.jungle],
        qscales=[4,2.5],
        mode='screen',
        vmaxs = [np.nanmax(cellmap)*10, np.nanmax(dmmap)*1.2]
        )
    
    # All subhalos
    fig, ax = fancy_axis(figsize=(8,8), dpi=200)
    snap1.set_box_halo(MWA, 1.5, radius_name='r200_code')
    ax.imshow(composite, origin='lower', extent=snap1.box[:2].flatten(), aspect='equal')
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), MWA['r200_code'], color='w', fill=False, lw=0.5, ls=':')
    ax.add_artist(cir)
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), 1.5*MWA['r200_code'], color='w', fill=False, lw=0.25, ls=':')
    ax.add_artist(cir)
    cmap = plt.cm.bwr
    norm = plt.Normalize(vmin=0, vmax=1)

    nhalo = 0
    for hkey in hal_takes.keys():
        if(hkey == MWA['halo_id']): continue
        sub = hal1s[hkey-1]
        take_scos = hal_takes[hkey]
        take_sco = 2 * np.median(take_scos%1)
        cir = plt.Circle((sub['x'], sub['y']), sub['rvir'], color=cmap(norm(take_sco)), fill=False, lw=0.5, ls='-')
        ax.add_artist(cir)
        nhalo += 1

    add_scalebar(ax, snap1.unit_l)
    addtext(f"{nhalo} subhalos",ax=ax, loc='lower left', color='white', fontsize=12, offset=0.025)
    addtext(f"NewHorizon\n\nGalaxy {target_id}\nHalo {target_hid}",ax=ax, loc='upper left', color='white', fontsize=12, offset=0.025)
    plt.savefig(f"./database/photo/10_test_saved/NH_{target_id:04d}_sub_all.png", dpi=400)
    plt.close()



    # Good subhalos
    fig, ax = fancy_axis(figsize=(8,8), dpi=200)
    snap1.set_box_halo(MWA, 1.5, radius_name='r200_code')
    ax.imshow(composite, origin='lower', extent=snap1.box[:2].flatten(), aspect='equal')
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), MWA['r200_code'], color='w', fill=False, lw=0.5, ls=':')
    ax.add_artist(cir)
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), 1.5*MWA['r200_code'], color='w', fill=False, lw=0.25, ls=':')
    ax.add_artist(cir)
    cmap = plt.cm.bwr
    norm = plt.Normalize(vmin=0, vmax=1)

    nhalo = 0
    realsubs = np.array([])
    for hkey in hal_takes.keys():
        if(hkey == MWA['halo_id']): continue
        sub = hal1s[hkey-1]
        dist = distance(sub, MWA['halo_x'], MWA['halo_y'], MWA['halo_z'])
        if(dist > (1.5*MWA['r200_code'] + sub['rvir'])): continue
        take_scos = hal_takes[hkey]
        take_sco = 2 * np.median(take_scos%1)
        if(take_sco < 0.5): continue
        cir = plt.Circle((sub['x'], sub['y']), sub['rvir'], color='tomato', fill=False, lw=0.5, ls='-')
        ax.add_artist(cir)
        nhalo += 1
        realsubs = np.array([sub]) if(len(realsubs)==0) else np.hstack((realsubs, sub))
    add_scalebar(ax, snap1.unit_l)
    addtext(f"{nhalo} subhalos",ax=ax, loc='lower left', color='white', fontsize=12, offset=0.025)
    addtext(f"NewHorizon\n\nGalaxy {target_id}\nHalo {target_hid}",ax=ax, loc='upper left', color='white', fontsize=12, offset=0.025)
    plt.savefig(f"./database/photo/10_test_saved/NH_{target_id:04d}_sub_good.png", dpi=400)
    plt.close()



    # Star map
    cmap_star = drawer.make_cmap([(0,0,0),(1,0,0),(1,1,0),(1,1,1)], position=[0,0.4,0.8,1])
    composite = painter.composite_image(
        [starmap, cellmap], 
        cmaps=[cmap_star, cmr.neutral],
        qscales=[4.5,4],
        mode='screen',
        vmaxs = [np.nanmax(starmap)*0.9, np.nanmax(cellmap)*10]
        )
    
    # All satellites
    fig, ax = fancy_axis(figsize=(8,8), dpi=200)
    snap1.set_box_halo(MWA, 1.5, radius_name='r200_code')
    ax.imshow(composite, origin='lower', extent=snap1.box[:2].flatten(), aspect='equal')
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), MWA['r200_code'], color='w', fill=False, lw=0.5, ls=':')
    ax.add_artist(cir)
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), 1.5*MWA['r200_code'], color='w', fill=False, lw=0.25, ls=':')
    ax.add_artist(cir)
    cmap = plt.cm.cool_r
    norm = plt.Normalize(vmin=0, vmax=1)

    ngal = 0
    for val in values:
        if(not val['Sat'] in gal_takes.keys()): continue
        take_scos = gal_takes[val['Sat']]
        take_sco = 2 * np.median(take_scos%1)
        sat = gal1s[val['Sat']-1]
        # if(take_sco < 0.5): continue
        cir = plt.Circle((sat['x'], sat['y']), sat['r'], color=cmap(norm(take_sco)), fill=False, lw=0.5, ls='-')
        ax.add_artist(cir)
        ngal += 1
    add_scalebar(ax, snap1.unit_l)
    addtext(f"{ngal} satellites",ax=ax, loc='lower left', color='white', fontsize=12, offset=0.025)
    addtext(f"NewHorizon\n\nGalaxy {target_id}\nHalo {target_hid}",ax=ax, loc='upper left', color='white', fontsize=12, offset=0.025)
    plt.savefig(f"./database/photo/10_test_saved/NH_{target_id:04d}_sat_all.png", dpi=400)
    plt.close()

    # Good satellites
    fig, ax = fancy_axis(figsize=(8,8), dpi=200)
    snap1.set_box_halo(MWA, 1.5, radius_name='r200_code')
    ax.imshow(composite, origin='lower', extent=snap1.box[:2].flatten(), aspect='equal')
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), MWA['r200_code'], color='w', fill=False, lw=0.5, ls=':')
    ax.add_artist(cir)
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), 1.5*MWA['r200_code'], color='w', fill=False, lw=0.25, ls=':')
    ax.add_artist(cir)
    cmap = plt.cm.cool_r
    norm = plt.Normalize(vmin=0, vmax=1)

    ngal = 0
    realsats = np.array([])
    for val in values:
        if(not val['Sat'] in gal_takes.keys()): continue
        take_scos = gal_takes[val['Sat']]
        take_sco = 2 * np.median(take_scos%1)
        sat = gal1s[val['Sat']-1]
        dist = distance(sat, MWA['halo_x'], MWA['halo_y'], MWA['halo_z'])
        if(dist > (1.5*MWA['r200_code'] + sat['r'])): continue
        if(take_sco < 0.3): continue
        cir = plt.Circle((sat['x'], sat['y']), sat['r'], color='cyan', fill=False, lw=0.5, ls='-')
        ax.add_artist(cir)
        ngal += 1
        realsats = np.array([sat]) if(len(realsats)==0) else np.hstack((realsats, sat))

    add_scalebar(ax, snap1.unit_l)
    addtext(f"{ngal} satellites",ax=ax, loc='lower left', color='white', fontsize=12, offset=0.025)
    addtext(f"NewHorizon\n\nGalaxy {target_id}\nHalo {target_hid}",ax=ax, loc='upper left', color='white', fontsize=12, offset=0.025)
    plt.savefig(f"./database/photo/10_test_saved/NH_{target_id:04d}_sat_good.png", dpi=400)
    plt.close()



    # Total map
    cmap_star = drawer.make_cmap([(0,0,0),(1,0,0),(1,1,0),(1,1,1)], position=[0,0.4,0.8,1])
    composite = painter.composite_image(
        [starmap, cellmap, dmmap], 
        cmaps=[cmap_star, cmr.neutral, cmr.jungle],
        qscales=[4.5,4,2.5],
        mode='screen',
        vmaxs = [np.nanmax(starmap)*0.9, np.nanmax(cellmap)*10, np.nanmax(dmmap)*1.2]
        )
    def point_in_sphere(point, sphere, rname='r', factor=1):
        dist = np.sqrt( (point['x'] - sphere['x'])**2 + (point['y'] - sphere['y'])**2 + (point['z'] - sphere['z'])**2 )
        return dist < sphere[rname]*factor

    # All pairs
    fig, ax = fancy_axis(figsize=(8,8), dpi=200)
    snap1.set_box_halo(MWA, 1.5, radius_name='r200_code')
    ax.imshow(composite, origin='lower', extent=snap1.box[:2].flatten(), aspect='equal')
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), MWA['r200_code'], color='w', fill=False, lw=0.5, ls=':')
    ax.add_artist(cir)
    cir = plt.Circle((MWA['halo_x'], MWA['halo_y']), 1.5*MWA['r200_code'], color='w', fill=False, lw=0.25, ls=':')
    ax.add_artist(cir)
    cmap = plt.cm.bwr
    norm = plt.Normalize(vmin=0, vmax=1)


    realpairs = np.zeros((max(len(realsubs), len(realsats)), 2), dtype=np.int32)
    for i in range(len(realsubs)):
        realpairs[i,0] = realsubs[i]['id']
    
    npair = 0
    for i, sub in enumerate(realsubs):
        ls = '--'
        if(len(realsats)>0):
            inner = point_in_sphere(realsats, sub, rname='r', factor=1)
            if(isinstance(inner, np.bool_)): inner = np.array([inner])
            if(True in inner):
                cands = realsats[inner]
                for cand in cands:
                    inside = point_in_sphere(sub, cand, rname='r', factor=1)
                    if(inside):
                        ls = '-'
                        npair += 1
                        realpairs[i,1] = cand['id']
                        break
        cir = plt.Circle((sub['x'], sub['y']), sub['rvir'], color='tomato', fill=False, lw=0.5, ls=ls)
        ax.add_artist(cir)
    for sat in realsats:
        ls = '--'
        inner = point_in_sphere(realsubs, sat, rname='r', factor=1)
        if(isinstance(inner, np.bool_)): inner = np.array([inner])
        if(True in inner):
            cands = realsubs[inner]
            for cand in cands:
                inside = point_in_sphere(sat, cand, rname='r', factor=1)
                if(inside):
                    ls = '-'
                    break
        cir = plt.Circle((sat['x'], sat['y']), sat['r'], color='cyan', fill=False, lw=0.7, ls=ls)
        ax.add_artist(cir)
    add_scalebar(ax, snap1.unit_l)
    addtext(f"{len(realsubs)-npair} DINKs", f"{len(realsats)-npair} Orphans", f"{npair} pairs", ax=ax, loc='lower left', color='white', fontsize=12, offset=0.025, dx=0.04)
    addtext(f"NewHorizon\n\nGalaxy {target_id}\nHalo {target_hid}",ax=ax, loc='upper left', color='white', fontsize=12, offset=0.025)
    plt.savefig(f"./database/photo/10_test_saved/NH_{target_id:04d}_total.png", dpi=400)
    plt.close()



    realmembers[MWA['id']] = realpairs
    pklsave(realmembers, f"./database/10_real_members.pickle", overwrite=True)
    snap1.clear()
    star.table = None
    dm.table = None
    cell.table = None










