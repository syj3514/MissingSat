import sys
argv = sys.argv
if(len(argv)==1):
    mod = 0
else:
    mod = int(argv[1])
print(mod)



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

from common_func import *


LG = pklload(f"{database}/LG")
allsubs = None
states = None
for key in LG.keys():
    subs = LG[key]['subs']
    real = LG[key]['real']
    dink = real[real['state']=='dink']['hid']
    ind = isin(subs['id'], dink)
    subs['dink'][ind] = True
    subs['dink'][~ind] = False
    state = np.zeros(len(subs), dtype='<U7')
    state[ind] = 'dink'
    state[~ind] = 'pair'
    upair = real[real['state']=='upair']['hid']
    ind = isin(subs['id'], upair)
    state[ind] = 'upair'

    allsubs = subs if allsubs is None else np.hstack((allsubs, subs))
    states = state if states is None else np.hstack((states, state))
argsort = np.argsort(allsubs['id'])
allsubs = allsubs[argsort]
dtype = allsubs.dtype



tree = pklload(f"{database}/02_main_progenitors.pickle")
if(os.path.exists(f"{database}/halo_dict.pickle")):
    halos = pklload(f"{database}/halo_dict.pickle")
else:
    halos = {'catalog':{}, 'index':{}}
    uri.timer.verbose=0
    for iout in tqdm(np.unique(tree['timestep'])):
        isnap = snaps.get_snap(iout)
        ihals = uhmi.HaloMaker.load(isnap, galaxy=False, double_precision=dp)
        indicies = np.zeros(len(ihals), dtype=int)
        iids = tree[tree['timestep'] == iout]['id']
        ihals = ihals[iids-1]
        indicies[iids-1] = np.arange(len(iids))
        halos['catalog'][iout] = ihals
        halos['index'][iout] = indicies   
    pklsave(halos, f"{database}/halo_dict.pickle")

cursor = 0
nsub = len(allsubs)
uri.timer.verbose=0
for iout in np.unique(tree['timestep'])[::-1]:
    if(os.path.exists( f"{database}/main_prog/subhalos_{iout:05d}.pickle"))or(iout%10 != mod):
        cursor += nsub
        continue
    isnap = snaps.get_snap(iout)
    ihals = halos['catalog'][iout]
    indicies = halos['index'][iout]

    x1 = np.min(ihals['x']-ihals['r']); x2 = np.max(ihals['x']+ihals['r'])
    y1 = np.min(ihals['y']-ihals['r']); y2 = np.max(ihals['y']+ihals['r'])
    z1 = np.min(ihals['z']-ihals['r']); z2 = np.max(ihals['z']+ihals['r'])
    isnap.box = np.array([[x1,x2],[y1,y2],[z1,z2]])
    uri.timer.verbose=1
    isnap.get_cell(nthread=40, target_fields=['x','y','z','rho','P'], exact_box=False)
    isnap.get_part(nthread=40, target_fields=['x','y','z','m','epoch','id'], exact_box=False)
    uri.timer.verbose=0

    #--------------------------------------------------------------
    #--------------------------------------------------------------
    newsubs = np.zeros(len(allsubs), dtype=dtype.descr + [('lastid', '<i4'),('give_score', '<f8'), ('take_score', '<f8')])

    #--------------------------------------------------------------
    # 'nparts', 'id', 'timestep',
    # 'level', 'host', 'hostsub', 'nbsub', 'nextsub', 
    # 'aexp', 'm', 
    # 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz', 
    # 'r', 'a', 'b', 'c', 'ek', 'ep', 'et', 'spin', 'sigma', 
    # 'rvir', 'mvir', 'tvir', 'cvel', 'rho0', 'rc', 'mcontam' 
    #--------------------------------------------------------------
    for iname in dtype.names:
        if(iname in ihals.dtype.names):
            newsubs[iname] = ihals[iname]

    #--------------------------------------------------------------
    # 'lastid', 'give_score', 'take_score'
    #--------------------------------------------------------------
    itree = tree[cursor:cursor+nsub]
    newsubs['lastid'] = itree['lastid']
    newsubs['give_score'] = itree['give_score']
    newsubs['take_score'] = itree['take_score']
    
    for ith in tqdm( range(nsub), desc=f"IOUT{iout:05d} "):
        # timereport = []; ref = time.time()
        ihal = ihals[np.argsort(ihals['m'])][ith]
        #--------------------------------------------------------------
        # 'mdm', 'mstar', 'mcold', 'mcell', 
        # 'mdm_vir', 'mstar_vir', 'mcell_vir', 'mcold_vir'
        #--------------------------------------------------------------
        isnap.set_box_halo(ihal, 1, radius_name='r')
        isnap.get_part(nthread=40, target_fields=['x','y','z','m','epoch','id'])
        # timereport.append(("get_part", time.time()-ref)); ref = time.time()
        table = isnap.part['star']
        if(len(table)>0):
            table = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['r'])
            if(len(table)>0):
                newsubs['mstar'][ith] = np.sum(table['m','Msol'])
                table_vir = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'])
                if(len(table_vir)>0):
                    newsubs['mstar_vir'][ith] = np.sum(table_vir['m','Msol'])
        # timereport.append(("cutstar", time.time()-ref)); ref = time.time()
        table = isnap.get_cell(nthread=40, target_fields=['x','y','z','rho','P'])#,'level'])
        # timereport.append(("get_cell", time.time()-ref)); ref = time.time()
        table = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['r'])
        if(len(table)>0):
            newsubs['mcell'][ith] = np.sum(table['m','Msol'])
            ctable = table[table['T','K']<2e4]
            if(len(ctable)>0):
                newsubs['mcold'][ith] = np.sum(ctable['m','Msol'])
            table_vir = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'])
            if(len(table_vir)>0):
                newsubs['mcell_vir'][ith] = np.sum(table_vir['m','Msol'])
                ctable_vir = table_vir[table_vir['T','K']<2e4]
                if(len(ctable_vir)>0):
                    newsubs['mcold_vir'][ith] = np.sum(ctable_vir['m','Msol'])
        # timereport.append(("cut_cell", time.time()-ref)); ref = time.time()
        table = isnap.part['dm']
        table = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['r'])
        newsubs['mdm'][ith] = np.sum(table['m','Msol'])
        table_vir = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'])
        newsubs['mdm_vir'][ith] = np.sum(table_vir['m','Msol'])
        # timereport.append(("cut_dm", time.time()-ref)); ref = time.time()

        #--------------------------------------------------------------
        # 'r10_mem', 'r50_mem', 'r90_mem', 'r10_vir', 'r50_vir', 'r90_vir', 'r10_max', 'r50_max', 'r90_max', 
        #--------------------------------------------------------------
        all_dist = distance(ihal, table); argsort = np.argsort(all_dist)
        all_dist = all_dist[argsort]; all_mass = table['m'][argsort]
        memdm = uhmi.HaloMaker.read_member_part(isnap, ihal['id'], galaxy=False, target_fields=['x','y','z','m'])
        mem_dist = distance(ihal, memdm); argsort = np.argsort(mem_dist)
        mem_dist = mem_dist[argsort]; mem_mass = memdm['m'][argsort]

        newsubs['r10_mem'][ith] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.1)
        newsubs['r50_mem'][ith] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.5)
        newsubs['r90_mem'][ith] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.9)
        newsubs['r10_max'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.1)
        newsubs['r50_max'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.5)
        newsubs['r90_max'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.9)
        _, ind = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'], return_index=True)
        all_dist = all_dist[ind]; all_mass = table['m'][ind]
        newsubs['r10_vir'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.1)
        newsubs['r50_vir'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.5)
        newsubs['r90_vir'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.9)
        # timereport.append(("rhalfs", time.time()-ref)); ref = time.time()

        # for ireport in timereport:
        #     print(f"[{ireport[0]}] {ireport[1]:.2f} sec")
    isnap.clear()
    README = "`sub`, `dink`, `Host`, `r200kpc`, `m200`, `r200` are missed!"
    pklsave((newsubs, README), f"{database}/main_prog/subhalos_{iout:05d}.pickle")

    cursor += nsub
    # stop()