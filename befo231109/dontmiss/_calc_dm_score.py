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

dirname = '../database'

mode = 'nh'
iout = 1026
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)
subs_lets_checks = pklload(f"./nh_subs_lets_checks.pickle")
pouts = snaps.iout_avail['iout'][snaps.iout_avail['age'] >= snap.age-1]
pouts = pouts[pouts < snap.iout][::-1]
print(pouts)

def get_members(gal, galaxy=True):
    global members, snaps
    if(gal['timestep'] in members.keys()):
        if(gal['id'] in members[gal['timestep']].keys()):
            return members[gal['timestep']][gal['id']]
    else:
        members[gal['timestep']] = {}
    members[gal['timestep']][gal['id']] = uhmi.HaloMaker.read_member_part(snaps.get_snap(gal['timestep']), gal['id'], galaxy=galaxy, simple=True)
    return members[gal['timestep']][gal['id']]

LGs = pklload(f"{dirname}/00_LocalGroup_fix.pickle")
for LGkey in LGs.keys():
    subs_lets_check = subs_lets_checks[LGkey][0]
    if(len(subs_lets_check) == 0): continue
    LG = LGs[LGkey]

    BGG = LG['BGG']


    rrange = BGG['r']
    uri.timer.verbose=0
    centers = {}
    members = {}
    all_scores = {}
    give_scores = {}
    take_scores = {}
    for ip, pout in tqdm( enumerate(pouts), total=len(pouts) ):
        psnap = snaps.get_snap(pout)
        pgals = uhmi.HaloMaker.load(psnap, galaxy=False)

        for subid in subs_lets_check:
            sub = hals[subid-1]
            if(sub['id'] in centers.keys()):
                center = centers[sub['id']]
            else:
                center = [sub['x'], sub['y'], sub['z']]
                centers[sub['id']] = center
            my_member = get_members(sub, galaxy=False)
            pneighbors = cut_box(pgals, *center, rrange)


            if(len(pneighbors)==0): continue

            give_score = np.zeros(len(pneighbors))
            take_score = np.zeros(len(pneighbors))
            for i, pg in enumerate(pneighbors):
                pmember = get_members(pg, galaxy=False)
                intersect = np.sum( isin(pmember, my_member, assume_unique=True) )
                give_score[i] = intersect / len(my_member) / 2
                take_score[i] = intersect / len(pmember) / 2
            all_score = give_score * take_score
            
            argmax_all = np.argmax(all_score)
            argmax_give = np.argmax(give_score)
            argmax_take = np.argmax(take_score)
            if(not sub['id'] in all_scores.keys()):
                all_scores[sub['id']] = np.zeros(len(pouts))
                give_scores[sub['id']] = np.zeros(len(pouts))
                take_scores[sub['id']] = np.zeros(len(pouts))
            
            all_scores[sub['id']][ip] = pneighbors['id'][argmax_all] + all_score[argmax_all]
            give_scores[sub['id']][ip] = pneighbors['id'][argmax_give] + give_score[argmax_give]
            take_scores[sub['id']][ip] = pneighbors['id'][argmax_take] + take_score[argmax_take]
            centers[sub['id']] = [ pneighbors['x'][argmax_all], pneighbors['y'][argmax_all], pneighbors['z'][argmax_all] ]

    pklsave(give_scores, f"./nh_give_dm_scores_host{BGG['id']:04d}_new.pickle", overwrite=True)
    pklsave(take_scores, f"./nh_take_dm_scores_host{BGG['id']:04d}_new.pickle", overwrite=True)
    uri.timer.verbose=1