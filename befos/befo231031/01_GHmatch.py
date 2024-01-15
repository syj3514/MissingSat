from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import numpy as np
import os, glob
import time
import warnings
from importlib import reload

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel
import argparse, subprocess
from func01 import make_banlist, cutting, find_cengal_of_lvl1hals, find_cengal_of_others, find_halos_for_other, find_halos_for_others, find_halos_for_otherss, final_job



#----------------------------------------------------------------------
# Load HaloMaker
#----------------------------------------------------------------------
print(" > Load HaloMaker")
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


#----------------------------------------------------------------------
# Make a base table
#----------------------------------------------------------------------
print(" > Make a base table")
dtype1 = [
    ('id', '<i4'), ('timestep', '<i4'), ('nparts', '<i4'), 
    ('level', '<i4'), ('host', '<i4'), ('hostsub', '<i4'), ('aexp', '<f8'),
    ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
    ('m', '<f8'), ('r', '<f8')]
dtype2 = [
    ('id', '<i4'), ('nparts', '<i4'), 
    ('level', '<i4'), ('host', '<i4'), ('hostsub', '<i4'), 
    ('x', '<f8'), ('y', '<f8'), ('z', '<f8'),
    ('vx', '<f8'), ('vy', '<f8'), ('vz', '<f8'),
    ('mvir', '<f8'), ('rvir', '<f8')]
dtype3 = [ ('fcontam', '<f8'), ('dist', '<f8'), ('central', bool), ('main', bool) ]
hdtype2 = np.dtype(dtype2)
hdtype2 = [(f"halo_{iname}", iformat) for iname, iformat in hdtype2.descr ]
dtype = dtype1 + hdtype2 + dtype3


result1_a = np.zeros(len(gal1s), dtype=dtype)
result1_a.dtype
for iname in result1_a.dtype.names:
    if(iname in gal1s.dtype.names):
        result1_a[iname] = gal1s[iname]

result2_a = np.zeros(len(gal2s), dtype=dtype)
result2_a.dtype
for iname in result2_a.dtype.names:
    if(iname in gal2s.dtype.names):
        result2_a[iname] = gal2s[iname]
print(result2_a[0].dtype.names)


#----------------------------------------------------------------------
# Exclude halos with no galaxy
#----------------------------------------------------------------------
print(" > Exclude halos with no galaxy")
ban1 = make_banlist(gal1s, hal1s); print(np.sum(ban1))
ban2 = make_banlist(gal2s, hal2s); print(np.sum(ban2))


#----------------------------------------------------------------------
# Find central galaxy of each lvl1 halo
#----------------------------------------------------------------------
print(" > Find central galaxy of each lvl1 halo")

result1_b = np.copy(result1_a)
gid1s, score1s = find_cengal_of_lvl1hals(gal1s, hal1s, ban1)
check_dupl(gid1s[gid1s>0])
names = result1_b.dtype.names
names = [name for name in names if(name[:4]=='halo')]
for i, gid1 in enumerate(gid1s):
    if(gid1>0):
        ihal = hal1s[i]
        check_order(result1_b['id'])
        result1_b[gid1-1]['central'] = True
        result1_b[gid1-1]['main'] = True
        for iname in names:
            result1_b[gid1-1][iname] = ihal[iname[5:]]

result2_b = np.copy(result2_a)
gid2s, score2s = find_cengal_of_lvl1hals(gal2s, hal2s, ban2)
check_dupl(gid2s[gid2s>0])
names = result2_b.dtype.names
names = [name for name in names if(name[:4]=='halo')]
for i, gid2 in enumerate(gid2s):
    if(gid2>0):
        ihal = hal2s[i]
        check_order(result2_b['id'])
        result2_b[gid2-1]['central'] = True
        result2_b[gid2-1]['main'] = True
        for iname in names:
            result2_b[gid2-1][iname] = ihal[iname[5:]]

#----------------------------------------------------------------------
# Find central galaxy of each lvl-high halo
#----------------------------------------------------------------------
print(" > Find central galaxy of each lvl-high halo")

result1_c = np.copy(result1_b)
check_order(result1_c['id'])
gid1s, score1s = find_cengal_of_others(hal1s, gal1s, ban1, result1_b)
check_dupl(gid1s[gid1s>0])
names = result1_b.dtype.names
names = [name for name in names if(name[:4]=='halo')]
count =0
for i, gid1 in enumerate(gid1s):
    if(gid1>0):
        count += 1
        if(result1_c[gid1-1]['halo_id']<=0):
            ihal = hal1s[i]
            result1_c[gid1-1]['central'] = True
            result1_c[gid1-1]['main'] = hal1s[i]['level']==1
            for iname in names:
                result1_c[gid1-1][iname] = ihal[iname[5:]]

result2_c = np.copy(result2_b)
check_order(result2_c['id'])
gid2s, score2s = find_cengal_of_others(hal2s, gal2s, ban2, result2_b)
check_dupl(gid2s[gid2s>0])
names = result2_b.dtype.names
names = [name for name in names if(name[:4]=='halo')]
count =0
for i, gid2 in enumerate(gid2s):
    if(gid2>0):
        count += 1
        if(result2_c[gid2-1]['halo_id']<=0):
            ihal = hal2s[i]
            result2_c[gid2-1]['central'] = True
            result2_c[gid2-1]['main'] = hal2s[i]['level']==1
            for iname in names:
                result2_c[gid2-1][iname] = ihal[iname[5:]]

#----------------------------------------------------------------------
# Find parent halo of remaining galaxies
#----------------------------------------------------------------------
print(" > Find parent halo of remaining galaxies")
arr = result1_c
print(f"NH: {np.sum(arr['halo_id']>0)} has host halos (of {len(arr)})")
arr = result2_c
print(f"NH2: {np.sum(arr['halo_id']>0)} has host halos (of {len(arr)})")





result1_d = find_halos_for_otherss(hal1s, result1_c)
result2_d = find_halos_for_otherss(hal2s, result2_c)

arr = result1_d
print(f"NH: {np.sum(arr['halo_id']>0)} has host halos (of {len(arr)})")
arr = result2_d
print(f"NH2: {np.sum(arr['halo_id']>0)} has host halos (of {len(arr)})")


#----------------------------------------------------------------------
# Final job
#----------------------------------------------------------------------
print(" > Final job")

result1_e = final_job(hal1s, result1_d)
result2_e = final_job(hal2s, result2_d)

arr = result1_e
print(f"NH: {np.sum(arr['halo_id']>0)} has host halos (of {len(arr)})")
arr = result2_e
print(f"NH2: {np.sum(arr['halo_id']>0)} has host halos (of {len(arr)})")


#----------------------------------------------------------------------
# Save
#----------------------------------------------------------------------
pklsave(result1_e, f"./database/01_nh_ghmatch.pickle", overwrite=True)
pklsave(result2_e, f"./database/01_nh2_ghmatch.pickle", overwrite=True)