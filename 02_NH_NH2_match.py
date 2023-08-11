from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import numpy as np
import os, glob
import time
import warnings

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
from scipy.ndimage import gaussian_filter
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind
from icl_draw import drawsnap, add_scalebar, addtext
import argparse, subprocess

#-------------------------------------------------
# Data Preparation
#-------------------------------------------------
mode1 = 'nh'
fout1 = 1026
repo1,rurmode1,dp1 = mode2repo(mode1)
snap1 = uri.RamsesSnapshot(repo1, fout1, mode=rurmode1)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']
gal1s = uhmi.HaloMaker.load(snap1, galaxy=True, double_precision=dp1)
hal1s, hmpid1s = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp1, load_parts=True)
halo_id1s = np.repeat(hal1s['id'], hal1s['nparts'])

mode2 = 'nh2'
fout2 = 797
repo2,rurmode2,dp2 = mode2repo(mode2)
snap2 = uri.RamsesSnapshot(repo2, fout2, mode=rurmode2)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']
gal2s = uhmi.HaloMaker.load(snap2, galaxy=True, double_precision=dp2)
hal2s, hmpid2s = uhmi.HaloMaker.load(snap2, galaxy=False, double_precision=dp2, load_parts=True)
halo_id2s = np.repeat(hal2s['id'], hal2s['nparts'])

matchid = pklload(f"{repo1}/DMID_NH_to_NH2.pickle")
matchid.shape

result1s = pklload("./database/01_nh_ghmatch.pickle")
result2s = pklload("./database/01_nh2_ghmatch.pickle")
print(result1s.shape, gal1s.shape)
print(result2s.shape, gal2s.shape)


#-------------------------------------------------
# Base Array
#-------------------------------------------------
target1s = gal1s[ (gal1s['nparts'] >= 10**6) & (result1s['halo_id'] > 0) & (result1s['central'])]
print(f"Initial {len(target1s)} galaxies from NH")

dt1 = target1s.dtype.descr
dt2 = result1s.dtype.descr
dt2 = [idt2 for idt2 in dt2 if(idt2 not in dt1)]
dt3 = [
       ('r200', 'f8'), ('m200', 'f8'), ('r200_code', 'f8'), 
       ('m_star_200', 'f8'), ('m_gas_200', 'f8'), ('fcontam_200', 'f8'),
       ('rp','f8'), ('sfr', 'f8'), ('sfr_tot', 'f8'), 
       ('galaxy_nh2', 'i8'), ('halo_nh2', 'i8'), ('matchrate', 'f8')]
dtype = np.dtype(dt1 + dt2 + dt3)
MWA1s = np.zeros(len(target1s), dtype=dtype)
for iname in dtype.names:
    if(iname in target1s.dtype.names):
        MWA1s[iname] = target1s[iname]
    elif(iname in result1s.dtype.names):
        MWA1s[iname] = result1s[target1s['id']-1][iname]
    else:
        pass
print(MWA1s.dtype.names)

dt2 = result2s.dtype.descr
dt2 = [idt2 for idt2 in dt2 if(idt2 not in dt1)]
dt3 = [
       ('r200', 'f8'), ('m200', 'f8'), ('r200_code', 'f8'), 
       ('m_star_200', 'f8'), ('m_gas_200', 'f8'), ('fcontam_200', 'f8'),
       ('rp','f8'), ('sfr', 'f8'), ('sfr_tot', 'f8'), 
       ('galaxy_nh', 'i8'), ('halo_nh', 'i8'), ('matchrate', 'f8')]
dtype = np.dtype(dt1 + dt2 + dt3)
MWA2s = np.zeros(len(target1s), dtype=dtype)
print(MWA2s.dtype.names)


#-------------------------------------------------
# Match
#-------------------------------------------------
for ith, MWA1 in enumerate(MWA1s):
    if(MWA1['galaxy_nh2'] > 0): continue
    dm1s = uhmi.HaloMaker.read_member_part(snap1, MWA1['halo_id'], galaxy=False, simple=False)
    dm1s = cut_sphere(dm1s, MWA1['x'], MWA1['y'], MWA1['z'], MWA1['r'])
    dmids1 = dm1s['id']

    dmids2 = matchid[1][dmids1-1]

    MWA2 = None
    arg = large_isin(hmpid2s, dmids2)
    if(True in arg):
        cand_id2s, occurence = np.unique(halo_id2s[arg], return_counts=True)
        matchrate = occurence**2 / len(dmids1) / hal2s[cand_id2s-1]['nparts'] # (A&B)^2 / A / B
        argsort = np.argsort(-matchrate)
        MWA1['halo_nh2'] = cand_id2s[argsort][0]
        MWA2s[ith]['halo_id'] = cand_id2s[argsort][0]
        MWA2s[ith]['halo_nh'] = MWA1['halo_id']
        MWA2s[ith]['galaxy_nh'] = MWA1['id']
        MWA1['matchrate'] = matchrate[argsort][0]
        for hid2, score in zip(cand_id2s[argsort], matchrate[argsort]):
            # argmax = np.argmax(matchrate)
            # hid2 = cand_id2s[argmax]; score = matchrate[argmax]
            if(hid2 in result2s['halo_id']):
                cands = result2s[result2s['halo_id']==hid2]
                if(len(cands) == 1):
                    MWA2 = cands[0]
                else:
                    if(True in cands['central']):
                        MWA2 = cands[cands['central']][0]
                    else:
                        MWA2 = cands[np.argmax(cands['r'])]
                MWA1['halo_nh2'] = hid2
                MWA1['matchrate'] = score
                MWA2s[ith]['matchrate'] = score
                MWA2s[ith]['halo_id'] = hid2
                break
        if(MWA2 is not None):
            MWA1['galaxy_nh2'] = MWA2['id']
            MWA2s[ith]['id'] = MWA2['id']
            print(f"{ith:03d}|[NH] (G{MWA1['id']}, H{MWA1['halo_id']}) <- {score:.2f} -> (G{MWA2['id']}, H{hid2}) [NH2]")
        else:
            print(f"{ith:03d}|[NH] (G{MWA1['id']}, H{MWA1['halo_id']}) <- {score:.2f} -> (H{hid2}) [NH2]")
    else:
        print(f"{ith:03d}|[NH] (G{MWA1['id']}, H{MWA1['halo_id']}) <- 0.00 -> (H?) [NH2]")
        print(f"{ith:03d}|No matched due to contamination ({MWA1['fcontam']:.2f})")    

names1 = MWA2s.dtype.names
names2 = result2s.dtype.names
names3 = gal2s.dtype.names
for MWA2 in MWA2s:
    if(MWA2['id']==0): continue
    gal = result2s[MWA2['id']-1]
    for iname in names1:
        if(iname in names2):
            MWA2[iname] = gal[iname]
        elif(iname in names3):
            MWA2[iname] = gal2s[gal['id']-1][iname]

#-------------------------------------------------
# Remove weird
#-------------------------------------------------
# Zero matched
where = np.where(MWA2s['id']>0)[0]
print(where)
MWA1s = MWA1s[where]
MWA2s = MWA2s[where]

# Duplicated
unique, counts = np.unique(MWA2s['id'], return_counts=True)
unis = unique[counts>1]

where = []
for uni in unis:
    samples = MWA2s[MWA2s['id']==uni]
    argmax = np.argmax(samples['matchrate'])
    winner = samples[argmax]
    losers = samples[samples['galaxy_nh']!=winner['galaxy_nh']]
    for loser in losers:
        where.append(loser['galaxy_nh'])
where = np.array(where)
where = np.where(np.isin(MWA1s['id'], where))[0]
where = np.isin(np.arange(len(MWA1s)), where, invert=True)
MWA1s = MWA1s[where]
MWA2s = MWA2s[where]

check_unique(MWA2s['id'])

#-------------------------------------------------
# Save
#-------------------------------------------------
pklsave(MWA1s, f"./database/02_MWA1s.pickle", overwrite=True)
pklsave(MWA2s, f"./database/02_MWA2s.pickle", overwrite=True)