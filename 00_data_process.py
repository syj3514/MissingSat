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




gals2 = pklload("./database/01_nh_ghmatch.pickle")
LG = pklload(f"./database/11_LocalGroup.pickle")
del LG[11]
del LG[136]
del LG[937]
del LG[168]
del LG[212]
subhalo_pairs = pklload(f"./database/06_nh_subhalo_pairs.pickle")
scores = pklload(f"./database/08_nh_scores.pickle")
dm_scores = pklload(f"./database/08_nh_dm_scores.pickle")
vac = pklload("./database/09_value_added.pickle")
MWAs = pklload("./database/03_MWA1s.pickle")



def point_in_sphere(point, sphere, rname='r', factor=1):
    dist = np.sqrt( (point['x'] - sphere['x'])**2 + (point['y'] - sphere['y'])**2 + (point['z'] - sphere['z'])**2 )
    # print(dist, sphere[rname]*factor)
    return dist < sphere[rname]*factor

def sphere_in_sphere(inner, outer, r1='r',r2='r', factor=1):
    dist = np.sqrt( (inner['x'] - outer['x'])**2 + (inner['y'] - outer['y'])**2 + (inner['z'] - outer['z'])**2 )
    # print(dist+inner[r1], outer[r2])
    return (dist+inner[r1]) < outer[r2]

def sphere_touch_sphere(sph1, sph2, r1='r',r2='r', factor=1):
    dist = np.sqrt( (sph1['x'] - sph2['x'])**2 + (sph1['y'] - sph2['y'])**2 + (sph1['z'] - sph2['z'])**2 )
    # print(dist, sph1[r1]+sph2[r2])
    return dist < (sph1[r1]+sph2[r2])



for key in LG.keys():
    print(f"\n[{key:04d}]")
    BGG = LG[key]['BGG']
    neighbor = LG[key]['neighbor']
    sats = LG[key]['sats']
    subs = LG[key]['subs']
    real = LG[key]['real']

    # Real Satelllites
    ##################
    ingals = cut_sphere(gals, BGG['x'], BGG['y'], BGG['z'], 1.5*BGG['r200_code'], both_sphere=True)
    ingals = ingals[ingals['id'] != BGG['id']]
    myscore = deepcopy( scores['take'][key] )
    if(BGG['id'] in myscore.keys()):
        del myscore[BGG['id']]
    if(len(neighbor)>0):
        for neigh in neighbor:
            assert neigh['id'] in MWAs['id']
            neighG = MWAs[MWAs['id'] == neigh['id']][0]
            tmp = cut_sphere(gals, neighG['x'], neighG['y'], neighG['z'], 1.5*neighG['r200_code'], both_sphere=True)
            tmp = tmp[tmp['id'] != BGG['id']]
            tmp = tmp[~isin(tmp['id'], ingals['id'])]
            ingals = np.hstack((ingals, tmp))

            tmp = scores['take'][neigh['id']]
            for ikey in tmp.keys():
                if(ikey == BGG['id']): continue
                if(not ikey in myscore.keys()):
                    myscore[ikey] = tmp[ikey]

    goodind = np.zeros(len(ingals), dtype=bool)
    for i, ingal in enumerate(ingals):
        try: # For satellite
            assert ingal['id'] in myscore.keys()
            goodind[i] = np.median(myscore[ingal['id']]%1) > 0.15
        except: # For neighbor BGG
            goodind[i] = True
    leng = np.sum(goodind)
    goodsats = ingals[goodind]

    satdtype = sats.dtype
    newsats = np.empty(leng, dtype=satdtype)
    for i, goodsat in enumerate(goodsats):
        tvac = vac[vac['Sat'] == goodsat['id']][0]
        for iname in satdtype.names:
            if(iname in gals.dtype.names):
                newsats[i][iname] = goodsat[iname]
            elif(iname in gals2.dtype.names):
                newsats[i][iname] = gals2[goodsat['id']-1][iname]
            elif(iname in vac.dtype.names):
                newsats[i][iname] = tvac[iname]
    LG[key]['sats'] = newsats

    # Real Subhalos
    ###############
    inhals = cut_sphere(hals, BGG['x'], BGG['y'], BGG['z'], 1.5*BGG['r200_code'], both_sphere=True)
    inhals = inhals[inhals['id'] != BGG['halo_id']]
    myscore = deepcopy( dm_scores['take'][key] )
    if(BGG['halo_id'] in myscore.keys()):
        del myscore[BGG['halo_id']]
    if(len(neighbor)>0):
        for neigh in neighbor:
            assert neigh['id'] in MWAs['id']
            neighG = MWAs[MWAs['id'] == neigh['id']][0]
            tmp = cut_sphere(hals, neighG['x'], neighG['y'], neighG['z'], 1.5*neighG['r200_code'], both_sphere=True)
            tmp = tmp[tmp['id'] != BGG['halo_id']]
            tmp = tmp[~isin(tmp['id'], inhals['id'])]
            inhals = np.hstack((inhals, tmp))

            tmp = dm_scores['take'][neigh['id']]
            for ikey in tmp.keys():
                if(ikey == BGG['halo_id']): continue
                if(not ikey in myscore.keys()):
                    myscore[ikey] = tmp[ikey]

    goodind = np.zeros(len(inhals), dtype=bool)
    goodind[:] = False
    for i, inhal in enumerate(inhals):
        try: # For subhalo
            assert inhal['id'] in myscore.keys()
            goodind[i] = np.median(myscore[inhal['id']]%1) > 0.25
        except: # For neighbor BGG
            print(f"\t! {inhal['id']} is not in myscore.keys()")
            goodind[i] = True
    leng = np.sum(goodind)
    goodsubs = inhals[goodind]
    newsubs = goodsubs
    LG[key]['subs'] = newsubs

    # Matching
    ##########
    # Initialize
    newreals = np.empty(len(newsats)+len(newsubs), dtype=real.dtype)
    newreals['hid'] = -1
    newreals['gid'] = -1
    for i, newsub in enumerate(newsubs):
        newreals[i]['hid'] = newsub['id']
        newreals[i]['state'] = 'dink'
    # 1. Both inside center
    count = 0
    argsort = np.argsort(-newsats['m'])
    inewsats = newsats[argsort]
    for i, newsat in enumerate(inewsats):
        if(newsat['id'] in newreals['gid']): continue
        already = newreals['hid'][newreals['gid']>0]
        inewsubs = newsubs[~isin(newsubs['id'], already)]
        insides = point_in_sphere(inewsubs, newsat, rname='r', factor=1)
        if(np.sum(insides)==0):
            insides = sphere_in_sphere(newsat, inewsubs, r2='rvir')
        # if(np.sum(insides)==0):
        #     insides = sphere_in_sphere(newsat, inewsubs, r2='r')
        if(np.sum(insides)==0):
            insides1 = sphere_touch_sphere(newsat, inewsubs, r2='rvir')
            insides2 = inewsubs['r']>newsat['r']
            insides = insides1&insides2
        # if(np.sum(insides)==0):
        #     insides = sphere_touch_sphere(newsat, inewsubs, r2='r')
        if(np.sum(insides)==0):
            # orphan
            where = np.where(newreals['hid'] == -1)[0][0]
            newreals[where]['hid'] = 0
            newreals[where]['gid'] = newsat['id']
            newreals[where]['state'] = 'orphan'
        
        cands = inewsubs[insides]
        if(len(cands)==0):
            pass
        elif(len(cands)==1):
            cand = cands[0]
            where = np.where(newreals['hid'] == cand['id'])[0][0]
            newreals[where]['gid'] = newsat['id']; count+=1
            newreals[where]['state'] = 'pair'
        else:
            dists = distance(newsat, cands)
            argmin = np.argmin(dists)
            cand = cands[argmin]
            where = np.where(newreals['hid'] == cand['id'])[0][0]
            newreals[where]['gid'] = newsat['id']; count+=1
            newreals[where]['state'] = 'pair'
    print(f"{count}/{len(inewsats)} are matched")

    mask = (newreals['hid']>0)|(newreals['gid']>0)
    newreals = newreals[mask]
    LG[key]['real'] = newreals




    fig, ax = plt.subplots(1,1, figsize=(5,5))
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.scatter(inewsats['x'], inewsats['y'], color='r', s=0)
    init_colors()
    pairs = newreals[newreals['state']=='pair']
    for pair in pairs:
        assert (pair['hid']>0)&(pair['gid']>0)
        sat = newsats[newsats['id']==pair['gid']][0]
        sub = newsubs[newsubs['id']==pair['hid']][0]
        color = get_color()
        cir = plt.Circle((sat['x'], sat['y']), sat['r'], fill=False, ls='-', lw=0.6, color=color)
        ax.add_patch(cir)
        cir = plt.Circle((sub['x'], sub['y']), sub['rvir'], fill=False, ls='--', lw=0.6, color=color)
        ax.add_patch(cir)
        cir = plt.Circle((sub['x'], sub['y']), sub['r'], fill=False, ls=':', lw=0.2, color=color)
        ax.add_patch(cir)
        ax.plot([sat['x'], sub['x']], [sat['y'], sub['y']], color=color, lw=0.6)

    cir = plt.Circle((BGG['x'], BGG['y']), BGG['r'], color='k', fill=False)
    ax.add_patch(cir)
    cir = plt.Circle((BGG['x'], BGG['y']), BGG['r200_code'], color='k', fill=False)
    ax.add_patch(cir)
    ax.set_aspect(1)
    plt.savefig(f"./database/photo/00_data_process/{key:04d}_pairs.png", dpi=300, facecolor='white')
    plt.close()
    # break

pklsave(LG, "./database/00_LocalGroup.pickle", overwrite=True)
