from IPython import get_ipython

print("ex: $ python3 06c_stable_tree.py [--mod 7]")
import argparse
parser = argparse.ArgumentParser(description='(syj3514@yonsei.ac.kr)')
parser.add_argument("-m", "--mod", required=True, help='mod', type=int)
args = parser.parse_args()

home = "/home/jeon"
# home = "/gem_home/jeon"
mod = args.mod
ncpu=24

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
os.nice(19)
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
#from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle, ax_change_color
from importlib import reload
from copy import deepcopy
from multiprocessing import Pool, shared_memory, Value
from common_func import *


mode2 = 'nh2'
database2 = f"{home}/MissingSat/database/{mode2}"
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
dtype2 = allsubs2.dtype

print(len(allsubs2), np.unique(states2, return_counts=True))  


stree2 = pklload(f"{database2}/stable_progenitors.pickle")
if(0 in stree2['timestep']):
    stree2 = stree2[stree2['timestep']>0]
    pklsave(stree2, f"{database2}/stable_progenitors.pickle", overwrite=True)



def _make_sub(ith, address, shape, dtype):
    global ihals, isnap, globalbox, reft, refn

    exist = shared_memory.SharedMemory(name=address)
    newsubs = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)
    if(newsubs[ith]['mdm_vir']>0):
        refn.value += 1
        if(refn.value%500==0)&(refn.value>0):
            print(f" > {refn.value}/{len(newsubs)} {time.time()-reft.value:.2f} sec (ETA: {(len(newsubs)-refn.value)*(time.time()-reft.value)/refn.value/60:.2f} min)")
        return None
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

    if(isnap.mode == 'nh'):
        part = isnap.get_part_instant(box=ibox, nthread=1, target_fields=['x','y','z','m','epoch','id'])
    else:
        part = isnap.get_part_instant(box=ibox, nthread=1, target_fields=['x','y','z','m','family','id'])
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
    part=part[argsort]; all_dist = all_dist[argsort]; all_mass = part['m','Msol']
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



# assert not os.path.exists(f"{database2}/main_prog/mod{mod}.pickle")
# pklsave(["warning"], f"{database2}/main_prog/mod{mod}.pickle")


def flush(msg=False, parent=''):
    global memory
    if(msg): print(f"{parent} Clearing memory")
    print(f"\tUnlink `{memory.name}`")
    try:
        memory.close()
        memory.unlink()
    except: pass

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

def terminate(signum, *args):
    flush(msg=True, parent=f'[Signal{signum}]')
    atexit.unregister(flush)
    exit(0)


##############################################################
# New Horizon 2
##############################################################
print(f"\nNew Horizon 2\n")
halos = pklload(f"{database2}/halo_dict.pickle")
keys = list(halos.keys())
for key in keys:
    if(key%10 != mod):
        del halos[key]
cursor = 0
nsub = len(allsubs2)
uri.timer.verbose=0
for iout in np.unique(stree2['timestep'])[::-1]:
    if(os.path.exists( f"{database2}/stable_prog/subhalos_{iout:05d}.pickle"))or(iout%10 != mod):
        cursor += nsub
        continue
    isnap = snap2s.get_snap(iout)
    ihals = halos[iout]
    itree = stree2[cursor:cursor+nsub]
    ihals = ihals[itree['id']-1]

    x1 = np.min(ihals['x']-ihals['r']); x2 = np.max(ihals['x']+ihals['r'])
    y1 = np.min(ihals['y']-ihals['r']); y2 = np.max(ihals['y']+ihals['r'])
    z1 = np.min(ihals['z']-ihals['r']); z2 = np.max(ihals['z']+ihals['r'])
    isnap.box = np.array([[x1,x2],[y1,y2],[z1,z2]])
    globalbox = np.array([[x1,x2],[y1,y2],[z1,z2]])
    

    #--------------------------------------------------------------
    #--------------------------------------------------------------
    atexit.register(flush)
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGPIPE, terminate)
    newdtype = np.dtype( dtype2.descr + [('lastid', '<i4'),('give_score', '<f8'), ('take_score', '<f8')] )
    newsubs = np.zeros(len(allsubs2), dtype=newdtype)
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
    for iname in dtype2.names:
        if(iname in ihals.dtype.names):
            newsubs[iname] = ihals[iname]

    #--------------------------------------------------------------
    # 'lastid', 'give_score', 'take_score'
    #--------------------------------------------------------------
    newsubs['lastid'] = itree['lastid']
    newsubs['give_score'] = itree['give_score']
    newsubs['take_score'] = itree['take_score']
    if(os.path.exists(f"{database2}/main_prog/subhalos_{iout:05d}.pickle")):
        main_prog = pklload(f"{database2}/main_prog/subhalos_{iout:05d}.pickle")[0]
        for i in range(len(allsubs2)):
            newsub = newsubs[i]
            where1 = newsub['lastid'] == main_prog['lastid']
            where2 = newsub['id'] == main_prog['id']
            if(True in where1&where2):
                already = main_prog[where1&where2][0]
                newsubs[i] = already
    needed = newsubs[newsubs['mdm_vir']<=0]
    if(len(needed)==0):
        pass
    else:
        cpulist = isnap.get_halos_cpulist(needed, 1.2, radius_name='r', nthread=min(ncpu, len(needed)))
        uri.timer.verbose=1
        isnap.get_cell(nthread=ncpu, target_fields=['x','y','z','rho','P'], exact_box=False, domain_slicing=True, cpulist=cpulist)
        isnap.get_part(nthread=ncpu, target_fields=['x','y','z','m','family','id'], exact_box=False, domain_slicing=True, cpulist=cpulist)

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
    pklsave((newsubs, README), f"{database2}/stable_prog/subhalos_{iout:05d}.pickle")
    print(f"`{database2}/stable_prog/subhalos_{iout:05d}.pickle` save done\n\n\n")
    flush(msg=True)
    del newsubs
    cursor += nsub