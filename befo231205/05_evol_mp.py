import sys
argv = sys.argv
if(len(argv)==1):
    mod = 0
else:
    mod = int(argv[1])
print(mod)

database = f"/home/jeon/MissingSat/database"
# database = f"/gem_home/jeon/MissingSat/database"

ncpu=32
memory=None

from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize

import numpy as np
import os, glob
os.nice(19)
import time
import warnings

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
from rur.sci.photometry import measure_luminosity
from rur.sci.geometry import get_angles, euler_angle
from rur.utool import rotate_data
from scipy.ndimage import gaussian_filter
uri.timer.verbose=1
import atexit, signal
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind, isin
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle
import argparse, subprocess
from importlib import reload
# import cmasher as cmr
from copy import deepcopy
from multiprocessing import Pool, shared_memory, Value

mode = 'nh'
iout = 1026
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)

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

keys = list(halos['index'].keys())
for key in keys:
    if((key%10 != mod)):
        del halos['index'][key]
        del halos['catalog'][key]


def _make_sub(ith, address, shape, dtype):
    global ihals, isnap, globalbox, reft, refn

    exist = shared_memory.SharedMemory(name=address)
    newsubs = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)
    # ihals, 
    ihal = ihals[ith]
    #--------------------------------------------------------------
    # 'mdm', 'mstar', 'mcold', 'mcell', 
    # 'mdm_vir', 'mstar_vir', 'mcell_vir', 'mcold_vir'
    #--------------------------------------------------------------
    assert np.array_equal(isnap.box, globalbox), f"{isnap.box} != {globalbox}"
    ibox = np.array([[ihal['x']-ihal['r'], ihal['x']+ihal['r']],
                        [ihal['y']-ihal['r'], ihal['y']+ihal['r']],
                        [ihal['z']-ihal['r'], ihal['z']+ihal['r']]])
    table = isnap.get_cell_instant(box=ibox, nthread=1, target_fields=['x','y','z','rho','P'])
    table = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['r'])
    if(len(table)>0):
        newsubs['mcell'][ith] = np.sum(table['m','Msol'])
        ctable = table[table['T','K']<2e4]
        if(len(ctable)>0):
            newsubs['mcold'][ith] = np.sum(ctable['m','Msol'])
        table = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'])
        if(len(table)>0):
            newsubs['mcell_vir'][ith] = np.sum(table['m','Msol'])
            ctable = table[table['T','K']<2e4]
            if(len(ctable)>0):
                newsubs['mcold_vir'][ith] = np.sum(ctable['m','Msol'])


    part = isnap.get_part_instant(box=ibox, nthread=1, target_fields=['x','y','z','m','epoch','id'])
    table = part['star']
    if(len(table)>0):
        table = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['r'])
        if(len(table)>0):
            newsubs['mstar'][ith] = np.sum(table['m','Msol'])
            table = cut_sphere(table, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'])
            if(len(table)>0):
                newsubs['mstar_vir'][ith] = np.sum(table['m','Msol'])
    del table
    part = part['dm']
    part = cut_sphere(part, ihal['x'], ihal['y'], ihal['z'], ihal['r'])
    

    #--------------------------------------------------------------
    # 'r10_mem', 'r50_mem', 'r90_mem', 'r10_vir', 'r50_vir', 'r90_vir', 'r10_max', 'r50_max', 'r90_max', 
    #--------------------------------------------------------------
    all_dist = distance(ihal, part); argsort = np.argsort(all_dist)
    part=part[argsort]; all_dist = all_dist[argsort]; all_mass = part['m']
    newsubs['mdm'][ith] = np.sum(all_mass)
    memdm = uhmi.HaloMaker.read_member_part(isnap, ihal['id'], galaxy=False, target_fields=['x','y','z','m'])
    mem_dist = distance(ihal, memdm); argsort = np.argsort(mem_dist)
    mem_dist = mem_dist[argsort]; mem_mass = memdm['m'][argsort]
    del argsort

    newsubs['r10_mem'][ith] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.1)
    newsubs['r50_mem'][ith] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.5)
    newsubs['r90_mem'][ith] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.9)
    del memdm, mem_dist, mem_mass
    newsubs['r10_max'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.1)
    newsubs['r50_max'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.5)
    newsubs['r90_max'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.9)
    _, ind = cut_sphere(part, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'], return_index=True)
    all_dist = all_dist[ind]; all_mass = all_mass[ind]
    del part, ind
    newsubs['mdm_vir'][ith] = np.sum(all_mass)
    newsubs['r10_vir'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.1)
    newsubs['r50_vir'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.5)
    newsubs['r90_vir'][ith] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.9)

    refn.value += 1
    if(refn.value%500==0)&(refn.value>0):
        print(f" > {refn.value}/{len(newsubs)} {time.time()-reft.value:.2f} sec (ETA: {(len(newsubs)-refn.value)*(time.time()-reft.value)/refn.value/60:.2f} min)")



assert not os.path.exists(f"{database}/main_prog/mod{mod}.pickle")
pklsave(["warning"], f"{database}/main_prog/mod{mod}.pickle")


def flush(msg=False, parent=''):
    global memory
    if(msg): print(f"{parent} Clearing memory")
    print(f"\tUnlink `{memory.name}`")
    memory.close()
    memory.unlink()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

def terminate(self, signum):
    flush(msg=True, parent=f'[Signal{signum}]')
    atexit.unregister(flush)
    exit(0)


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
    globalbox = np.array([[x1,x2],[y1,y2],[z1,z2]])
    uri.timer.verbose=1
    isnap.get_cell(nthread=40, target_fields=['x','y','z','rho','P'], exact_box=False, domain_slicing=False)
    isnap.get_part(nthread=40, target_fields=['x','y','z','m','epoch','id'], exact_box=False, domain_slicing=False)
    

    #--------------------------------------------------------------
    #--------------------------------------------------------------
    atexit.register(flush)
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGPIPE, terminate)
    newdtype = np.dtype( dtype.descr + [('lastid', '<i4'),('give_score', '<f8'), ('take_score', '<f8')] )
    newsubs = np.zeros(len(allsubs), dtype=newdtype)
    memory = shared_memory.SharedMemory(create=True, size=newsubs.nbytes)
    newsubs = np.ndarray(newsubs.shape, dtype=newdtype, buffer=memory.buf)

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
    
    reft = Value('f', 0); reft.value = time.time()
    refn = Value('i', 0)
    uri.timer.verbose=0
    print(f"[IOUT {iout:05d}]")
    with Pool(processes=ncpu) as pool:
        async_result = [pool.apply_async(_make_sub, (ith, memory.name, newsubs.shape, newdtype)) for ith in range(nsub)]
        # iterobj = tqdm(async_result, total=len(async_result), desc=f"IOUT{iout:05d} ")# if(uri.timer.verbose>=1) else async_result
        iterobj = async_result
        for r in iterobj:
            r.get()

    isnap.clear()
    README = "`sub`, `dink`, `Host`, `r200kpc`, `m200`, `r200` are missed!"
    pklsave((newsubs, README), f"{database}/main_prog/subhalos_{iout:05d}.pickle")
    print(f"`{database}/main_prog/subhalos_{iout:05d}.pickle` save done\n\n\n")
    flush(msg=True)
    del newsubs
    cursor += nsub
