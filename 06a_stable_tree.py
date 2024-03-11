from IPython import get_ipython


def type_of_script():
    """
    Detects and returns the type of python kernel
    :return: string 'jupyter' or 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


if type_of_script() == 'jupyter':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import cmasher as cmr

import numpy as np
import os, glob, atexit, signal
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
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle, ax_change_color
from importlib import reload
from copy import deepcopy
from multiprocessing import Pool, shared_memory, Value
from common_func import *



mode1 = 'nh'
database1 = f"/home/jeon/MissingSat/database/{mode1}"
iout1 = 1026
repo1, rurmode1, dp1 = mode2repo(mode1)
snap1 = uri.RamsesSnapshot(repo1, iout1, mode=rurmode1)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']; nout=nout1[nout1 <= iout1]
gals1 = uhmi.HaloMaker.load(snap1, galaxy=True, double_precision=dp1)
hals1 = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp1)

LG1 = pklload(f"{database1}/LocalGroup.pickle")
allsats1 = None; allsubs1 = None; states1 = None
keys1 = list(LG1.keys())
for key in keys1:
    sats = LG1[key]['sats']; subs = LG1[key]['subs']; real = LG1[key]['real']
    dink = real[real['state']=='dink']['hid']
    ind = isin(subs['id'], dink)
    subs['dink'][ind] = True; subs['dink'][~ind] = False
    state = np.zeros(len(subs), dtype='<U7')
    state[ind] = 'dink'; state[~ind] = 'pair'
    
    upair = real[real['state']=='upair']['hid']
    ind = isin(subs['id'], upair)
    state[ind] = 'upair'

    allsats1 = sats if allsats1 is None else np.hstack((allsats1, sats))
    allsubs1 = subs if allsubs1 is None else np.hstack((allsubs1, subs))
    states1 = state if states1 is None else np.hstack((states1, state))
argsort = np.argsort(allsubs1['id'])
allsubs1 = allsubs1[argsort]; states1 = states1[argsort]
dinks1 = allsubs1[states1 == 'dink']
pairs1 = allsubs1[states1 == 'pair']
upairs1 = allsubs1[states1 == 'upair']

print(len(allsubs1), np.unique(states1, return_counts=True))  


mode2 = 'nh2'
database2 = f"/home/jeon/MissingSat/database/{mode2}"
iout2 = 797
repo2, rurmode2, dp2 = mode2repo(mode2)
snap2 = uri.RamsesSnapshot(repo2, iout2, mode=rurmode2)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']; nout=nout2[nout2 <= iout2]
gals2 = uhmi.HaloMaker.load(snap2, galaxy=True, double_precision=dp2)
hals2 = uhmi.HaloMaker.load(snap2, galaxy=False, double_precision=dp2)

LG2 = pklload(f"{database2}/LocalGroup.pickle")
allsats2 = None; allsubs2 = None; states2 = None
keys2 = list(LG2.keys())
for key in keys2:
    sats = LG2[key]['sats']; subs = LG2[key]['subs']; real = LG2[key]['real']
    dink = real[real['state']=='dink']['hid']
    ind = isin(subs['id'], dink)
    subs['dink'][ind] = True; subs['dink'][~ind] = False
    state = np.zeros(len(subs), dtype='<U7')
    state[ind] = 'dink'; state[~ind] = 'pair'
    
    upair = real[real['state']=='upair']['hid']
    ind = isin(subs['id'], upair)
    state[ind] = 'upair'

    allsats2 = sats if allsats2 is None else np.hstack((allsats2, sats))
    allsubs2 = subs if allsubs2 is None else np.hstack((allsubs2, subs))
    states2 = state if states2 is None else np.hstack((states2, state))
argsort = np.argsort(allsubs2['id'])
allsubs2 = allsubs2[argsort]; states2 = states2[argsort]
dinks2 = allsubs2[states2 == 'dink']
pairs2 = allsubs2[states2 == 'pair']
upairs2 = allsubs2[states2 == 'upair']

print(len(allsubs2), np.unique(states2, return_counts=True))



if(not os.path.exists(f"{database1}/06b_shared_particles.pickle")):
    ptree_dm1 = pklload("/storage6/NewHorizon/ptree_dm/ptree_stable.pkl")
    lastouts1 = nout1[snap1s.iout_avail['age'] >= np.max(snap1s.iout_avail['age']-1)]
    ptree_dm1 = ptree_dm1[isin(ptree_dm1['timestep'], lastouts1)]
    halos1 = pklload(f"{database1}/halo_dict.pickle")
    prog_fromp1 = {}
    tmp = ptree_dm1[ptree_dm1['timestep']==1026]
    for sub in tqdm(allsubs1, desc='branch from ptree'):
        last = tmp[tmp['hmid'] == sub['id']][0]['last']
        branch = ptree_dm1[ptree_dm1['last'] == last]
        
        mybranch = None
        for ib in branch:
            index = halos1['index'][ib['timestep']][ib['hmid']-1]
            ihal = halos1['catalog'][ib['timestep']][index]
            mybranch = ihal if(mybranch is None) else np.hstack((mybranch, ihal))
        prog_fromp1[sub['id']] = mybranch
    pklsave(prog_fromp1, f"{database1}/06a_branch_from_ptree.pickle")

    count = 0
    shared_particles1 = {}
    for target in tqdm(allsubs1, desc='shared particle maximum'):
        count += 1
        progs = prog_fromp1[target['id']]
        parts = uhmi.HaloMaker.read_member_general(snap1s, progs, galaxy=False)
        ids, counts = np.unique(parts['id'], return_counts=True)
        thresh = len(progs)
        stables = np.array([], dtype=int)
        while len(stables) < target['nparts']/10:
            stables = ids[counts >= thresh]
            thresh -= 1
            if(thresh < 1):
                break
        shared_particles1[target['id']] = stables
        if(count%100 == 0):
            snap1s.clear()
    pklsave(shared_particles1, f"{database1}/06b_shared_particles.pickle")


if(not os.path.exists(f"{database2}/06b_shared_particles.pickle")):
    if(os.path.exists(f"{database2}/06a_branch_from_ptree.pickle")):
        prog_fromp2 = pklload(f"{database2}/06a_branch_from_ptree.pickle")
    else:
        ptree_dm2 = pklload("/storage7/NH2/ptree_dm/ptree_stable.pkl")
        lastouts2 = nout2[snap2s.iout_avail['age'] >= (snap2.age-1)]
        lastouts2 = lastouts2[lastouts2 <= iout2]
        ptree_dm2 = ptree_dm2[isin(ptree_dm2['timestep'], lastouts2)]
        halos2 = pklload(f"{database2}/halo_dict.pickle")
        prog_fromp2 = {}
        tmp = ptree_dm2[ptree_dm2['timestep']==797]
        for sub in tqdm(allsubs2, desc='branch from ptree'):
            last = tmp[tmp['hmid'] == sub['id']][0]['last']
            branch = ptree_dm2[ptree_dm2['last'] == last]
            
            mybranch = None
            for ib in branch:
                index = halos2['index'][ib['timestep']][ib['hmid']-1]
                ihal = halos2['catalog'][ib['timestep']][index]
                mybranch = ihal if(mybranch is None) else np.hstack((mybranch, ihal))
            prog_fromp2[sub['id']] = mybranch
        pklsave(prog_fromp2, f"{database2}/06a_branch_from_ptree.pickle")

    count = 0
    shared_particles2 = {}
    for target in tqdm(allsubs2, desc='shared particle maximum'):
        count += 1
        progs = prog_fromp2[target['id']]
        if(type(progs) == np.record):
            progs = np.array([progs])
        # print(type(progs), type(progs) == np.record)
        parts = uhmi.HaloMaker.read_member_general(snap2s, progs, galaxy=False)
        ids, counts = np.unique(parts['id'], return_counts=True)
        thresh = len(progs)
        stables = np.array([], dtype=int)
        while len(stables) < target['nparts']/10:
            stables = ids[counts >= thresh]
            thresh -= 1
            if(thresh < 1):
                break
        shared_particles2[target['id']] = stables
        if(count%100 == 0):
            snap2s.clear()
    pklsave(shared_particles2, f"{database2}/06b_shared_particles.pickle")