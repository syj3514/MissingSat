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

database = f"/home/jeon/MissingSat/database/nh2"

mode = 'nh2'
iout = 797
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)
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

LGs = pklload(f"{database}/LG")
for LGkey in LGs.keys():
    sats_lets_check = LGs[LGkey]['sats']['id']
    LG = LGs[LGkey]

    BGG = LG['BGG']


    rrange = BGG['r']
    uri.timer.verbose=0
    centers = {}
    members = {}
    all_scores = {}
    give_scores = {}
    take_scores = {}
    tmp = uhmi.HaloMaker.read_member_parts(snap, gals[sats_lets_check-1], galaxy=True, nthread=16, target_fields=['id'])
    my_members = {}
    for satid in sats_lets_check:
        my_members[satid] = tmp[tmp['hmid']==satid]['id']
    # for ip, pout in tqdm( enumerate(pouts), total=len(pouts) ):
    for ip, pout in enumerate(pouts):
        print(f"[{pout:04d}]")
        psnap = snaps.get_snap(pout)
        pgals = uhmi.HaloMaker.load(psnap, galaxy=True)

        for satid in sats_lets_check:
            sat = gals[satid-1]
            if(sat['id'] in centers.keys()):
                center = centers[sat['id']]
            else:
                center = [sat['x'], sat['y'], sat['z']]
                centers[sat['id']] = center
            my_member = my_members[satid]
            pneighbors = cut_box(pgals, *center, rrange)


            if(len(pneighbors)==0): continue

            give_score = np.zeros(len(pneighbors))
            take_score = np.zeros(len(pneighbors))
            pmembers = uhmi.HaloMaker.read_member_parts(psnap, pneighbors, galaxy=True, nthread=16, target_fields=['id'])
            for i, pg in enumerate(pneighbors):
                pmember = pmembers[pmembers['hmid']==pg['id']]['id']
                intersect = np.sum( isin(pmember, my_member, assume_unique=True) )
                give_score[i] = intersect / len(my_member) / 2
                take_score[i] = intersect / len(pmember) / 2
            all_score = give_score * take_score
            
            argmax_all = np.argmax(all_score)
            argmax_give = np.argmax(give_score)
            argmax_take = np.argmax(take_score)
            if(not sat['id'] in all_scores.keys()):
                all_scores[sat['id']] = np.zeros(len(pouts))
                give_scores[sat['id']] = np.zeros(len(pouts))
                take_scores[sat['id']] = np.zeros(len(pouts))
            
            all_scores[sat['id']][ip] = pneighbors['id'][argmax_all] + all_score[argmax_all]
            give_scores[sat['id']][ip] = pneighbors['id'][argmax_give] + give_score[argmax_give]
            take_scores[sat['id']][ip] = pneighbors['id'][argmax_take] + take_score[argmax_take]
            centers[sat['id']] = [ pneighbors['x'][argmax_all], pneighbors['y'][argmax_all], pneighbors['z'][argmax_all] ]
        psnap.clear(part=True, cell=False)
    pklsave(give_scores, f"{database}/scores/nh2_give_scores_host{BGG['id']:04d}.pickle", overwrite=True)
    pklsave(take_scores, f"{database}/scores/nh2_take_scores_host{BGG['id']:04d}.pickle", overwrite=True)
    uri.timer.verbose=1