from IPython import get_ipython
ncpu = 24
# ioutmax = 10000
# ioutmin = -10000
ioutmax = 10000
ioutmin = 700

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



mode2 = 'nh2'
database2 = f"/home/jeon/MissingSat/database/{mode2}"
iout2 = 797
repo2, rurmode2, dp2 = mode2repo(mode2)
snap2 = uri.RamsesSnapshot(repo2, iout2, mode=rurmode2)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']; nout=nout2[nout2 <= iout2]
gal2s = uhmi.HaloMaker.load(snap2, galaxy=True, double_precision=dp2)
hal2s = uhmi.HaloMaker.load(snap2, galaxy=False, double_precision=dp2)



LG2 = pklload(f"{database2}/LocalGroup.pickle")
allsubs2 = None
states2 = None
for key in LG2.keys():
    subs = LG2[key]['subs']
    real = LG2[key]['real']
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

    allsubs2 = subs if allsubs2 is None else np.hstack((allsubs2, subs))
    states2 = state if states2 is None else np.hstack((states2, state))
argsort = np.argsort(allsubs2['id'])
allsubs2 = allsubs2[argsort]
states2 = states2[argsort]


stree2 = pklload(f"{database2}/stable_tree.pickle")
def _insitu_SF(ith, maxbstar, maxnstar, address, shape, dtype, reft):
    global ihals, isnap, refn

    exist = shared_memory.SharedMemory(name=address)
    results = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)

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
print("\nNewHorizon2\n")

fnames = os.listdir(f"{database2}/stable_prog/")
fnames = [fname for fname in fnames if(fname.startswith("subhalos"))]
fnames.sort()

nstar = pklload(f"{database2}/nstar.pickle")
if(not os.path.isdir(f"{database2}/stable_prog/props")):
    os.mkdir(f"{database2}/stable_prog/props")
for fname in fnames:
    iout = int(fname[-12:-7])
    if(os.path.exists(f"{database2}/stable_prog/props/insitu_{iout:05d}.pickle")):
        continue
    if(iout>=ioutmax)or(iout<ioutmin): continue
    isnap = snap2s.get_snap(iout)
    if(not isnap.star[0]):
        continue
    istep = np.where(nout2==iout)[0][0]
    maxbstar = nstar[istep-1]
    maxnstar = nstar[istep]

    ihals = pklload(f"{database2}/stable_prog/{fname}")[0]
    nsub = len(ihals)
    cpulist = isnap.get_halos_cpulist(ihals, 1, radius_name='r', nthread=ncpu)
    uri.timer.verbose=1
    if(isnap.mode=='nh'):
        isnap.get_part(pname='star', nthread=ncpu, target_fields=['x','y','z','m','epoch','id'], exact_box=False, domain_slicing=True, cpulist=cpulist)
    else:
        isnap.get_part(pname='star', nthread=ncpu, target_fields=['x','y','z','m','family','id'], exact_box=False, domain_slicing=True, cpulist=cpulist)
    uri.timer.verbose=0


    print(f"[IOUT {iout:05d}]")
    atexit.register(flush)
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGPIPE, terminate)
    newdtype = np.dtype( [('lastid', '<i4'),('insitu', '<f8'), ('insitu_vir', '<f8')] )
    results = np.zeros(len(ihals), dtype=newdtype)
    memory = shared_memory.SharedMemory(create=True, size=results.nbytes)
    results = np.ndarray(results.shape, dtype=newdtype, buffer=memory.buf)
    results['lastid'] = ihals['lastid']
    
    
    reft = time.time()
    refn = Value('i', 0)
    with Pool(processes=ncpu) as pool:
        async_result = [pool.apply_async(_insitu_SF, (ith, maxbstar, maxnstar, memory.name, results.shape, newdtype, reft)) for ith in range(nsub)]
        # iterobj = tqdm(async_result, total=len(async_result), desc=f"IOUT{iout:05d} ")# if(uri.timer.verbose>=1) else async_result
        iterobj = async_result
        for r in iterobj:
            r.get()
            
    pklsave(results, f"{database2}/stable_prog/props/insitu_{iout:05d}.pickle")
    print(f"`{database2}/stable_prog/props/insitu_{iout:05d}.pickle` save done")
    flush(msg=True)
