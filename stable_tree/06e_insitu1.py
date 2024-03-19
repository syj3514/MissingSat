from IPython import get_ipython
ncpu = 32
ioutmax = 10000
ioutmin = -10000
# ioutmax = 10000
# ioutmin = 800

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

import os, glob, atexit, signal
if type_of_script() == 'jupyter':
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
    os.nice(19)
    
import matplotlib.pyplot as plt # type: module
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec

import numpy as np
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
gal1s = uhmi.HaloMaker.load(snap1, galaxy=True, double_precision=dp1)
hal1s = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp1)



LG1 = pklload(f"{database1}/LocalGroup.pickle")
allsubs1 = None
states1 = None
for key in LG1.keys():
    subs = LG1[key]['subs']
    real = LG1[key]['real']
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

    allsubs1 = subs if allsubs1 is None else np.hstack((allsubs1, subs))
    states1 = state if states1 is None else np.hstack((states1, state))
argsort = np.argsort(allsubs1['id'])
allsubs1 = allsubs1[argsort]
states1 = states1[argsort]




stree1 = pklload(f"{database1}/stable_tree_raw.pickle")
def _insitu_SF(ith, maxbstar, maxnstar, address, shape, dtype, reft, needed):
    global ihals, isnap, refn

    exist = shared_memory.SharedMemory(name=address)
    results = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)
    if(not needed[ith]):
        refn.value += 1
        if(refn.value%500==0)&(refn.value>0):
            print(f" > {refn.value}/{len(results)} {time.time()-reft:.2f} sec (ETA: {(len(results)-refn.value)*(time.time()-reft)/refn.value/60:.2f} min)")
        return None
    ihal = ihals[ith]
    ibox = np.array([[ihal['x']-ihal['r'], ihal['x']+ihal['r']],
                        [ihal['y']-ihal['r'], ihal['y']+ihal['r']],
                        [ihal['z']-ihal['r'], ihal['z']+ihal['r']]])

    if(isnap.mode == 'nh'):
        star = isnap.get_part_instant(box=ibox, nthread=1, target_fields=['x','y','z','m','epoch','id'], pname='star')
    else:
        star = isnap.get_part_instant(box=ibox, nthread=1, target_fields=['x','y','z','m','family','id'], pname='star')
    newstar = star[(np.abs(star['id']) > maxbstar)&(np.abs(star['id']) <= maxnstar)]
    if(len(newstar)>0):
        newstar = cut_sphere(newstar, ihal['x'], ihal['y'], ihal['z'], ihal['r'])
        if(len(newstar)>0):
            results['insitu'][ith] = np.sum(newstar['m','Msol'])
            newstar = cut_sphere(newstar, ihal['x'], ihal['y'], ihal['z'], ihal['rvir'])
            if(len(newstar)>0):
                results['insitu_vir'][ith] = np.sum(newstar['m','Msol'])
    del newstar, star

    refn.value += 1
    if(refn.value%500==0)&(refn.value>0):
        print(f" > {refn.value}/{len(results)} {time.time()-reft:.2f} sec (ETA: {(len(results)-refn.value)*(time.time()-reft)/refn.value/60:.2f} min)")

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

#############################################################
# NewHorizon
#############################################################
print("\nNewHorizon1\n")

fnames = os.listdir(f"{database1}/stable_prog/")
fnames = [fname for fname in fnames if(fname.startswith("subhalos"))]
fnames.sort()

nstar = pklload(f"{database1}/nstar.pickle")
if(not os.path.isdir(f"{database1}/stable_prog/props")):
    os.mkdir(f"{database1}/stable_prog/props")
for fname in fnames:
    iout = int(fname[-12:-7])
    if(os.path.exists(f"{database1}/stable_prog/props/insitu_{iout:05d}.pickle")):
        continue
    if(iout>=ioutmax)or(iout<ioutmin): continue
    isnap = snap1s.get_snap(iout)
    if(not isnap.star[0]):
        continue
    istep = np.where(nout1==iout)[0][0]
    maxbstar = nstar[istep-1]
    maxnstar = nstar[istep]

    ihals = pklload(f"{database1}/stable_prog/{fname}")[0]
    nsub = len(ihals)


    print(f"[IOUT {iout:05d}]")
    atexit.register(flush)
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGPIPE, terminate)
    newdtype = np.dtype( [('lastid', '<i4'),('insitu', '<f8'), ('insitu_vir', '<f8')] )
    results = np.zeros(len(ihals), dtype=newdtype)
    memory = shared_memory.SharedMemory(create=True, size=results.nbytes)
    results = np.ndarray(results.shape, dtype=newdtype, buffer=memory.buf)
    results['lastid'] = ihals['lastid']
    isneed = np.full(nsub, True, dtype=bool)
    if(os.path.exists(f"{database1}/stable_prog/old/props/insitu_{iout:05d}.pickle")):
        oldhals = pklload(f"{database1}/stable_prog/old/subhalos_{iout:05d}.pickle")[0]
        oldsitu = pklload(f"{database1}/stable_prog/old/props/insitu_{iout:05d}.pickle")
        for i in range(nsub):
            ihal = ihals[i]
            where1 = ihal['lastid'] == oldhals['lastid']
            where2 = ihal['id'] == oldhals['id']
            if(True in where1&where2):
                already = oldsitu[where1&where2][0]
                results[i] = already
                isneed[i] = False
    needed = ihals[isneed]
    if(len(needed)==0):
        pass
    else:
        cpulist = isnap.get_halos_cpulist(needed, 1, radius_name='r', nthread=ncpu)
        uri.timer.verbose=1
        if(isnap.mode=='nh'):
            isnap.get_part(pname='star', nthread=ncpu, target_fields=['x','y','z','m','epoch','id'], exact_box=False, domain_slicing=True, cpulist=cpulist)
        else:
            isnap.get_part(pname='star', nthread=ncpu, target_fields=['x','y','z','m','family','id'], exact_box=False, domain_slicing=True, cpulist=cpulist)
        uri.timer.verbose=0
        
        reft = time.time()
        refn = Value('i', 0)
        with Pool(processes=ncpu) as pool:
            async_result = [pool.apply_async(_insitu_SF, (ith, maxbstar, maxnstar, memory.name, results.shape, newdtype, reft, needed)) for ith in range(nsub)]
            # iterobj = tqdm(async_result, total=len(async_result), desc=f"IOUT{iout:05d} ")# if(uri.timer.verbose>=1) else async_result
            iterobj = async_result
            for r in iterobj:
                r.get()
    isnap.clear()
    pklsave(results, f"{database1}/stable_prog/props/insitu_{iout:05d}.pickle")
    print(f"`{database1}/stable_prog/props/insitu_{iout:05d}.pickle` save done")
    flush(msg=True)
