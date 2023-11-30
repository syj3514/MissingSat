import sys
argv = sys.argv
if('ipykernel' in argv[0]):
    mod=0
else:
    if(len(argv)==1):
        mod = 0
    else:
        mod = int(argv[1])
print(mod)

database = f"/home/jeon/MissingSat/database"
# database = f"/gem_home/jeon/MissingSat/database"

ncpu=32
memory=None


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
database = f"/home/jeon/MissingSat/database"

from common_func import *


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
keys = list(halos['index'].keys())
nhals = len( halos['catalog'][keys[0]] )


snap_star = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_stars = uri.TimeSeries(snap)
snap_dm = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_dms = uri.TimeSeries(snap)
snap_cell = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_cells = uri.TimeSeries(snap)


def calc_virial(cx,cy,cz, rmax_pkpc, pos_code, m_msol, params):
    '''
    input:
        cx,cy,cz : center of halo
        star, dm, cell : data
    output:
        rvir : virial radius
        mvir : virial mass
        rvir_code : virial radius in code unit
    '''
    H0 = params['H0']; aexp=params['aexp']; kpc=params['kpc']
    # critical density
    H02 = (H0 * 3.24078e-20)**2 # s-2
    G = 6.6743e-11 # N m2 kg-2 = kg m s-2 m2 kg-2 = m3 s-2 kg-1
    rhoc = 3 * H02 /8 /np.pi /G # kg m-3
    rhoc *= 5.02785e-31  * (3.086e+19)**3 # Msol ckpc-3
    rhoc /= (aexp**3) # Msol pkpc-3

    # Sorting
    dis = distance3d(pos_code[:,0], pos_code[:,1], pos_code[:,2], cx, cy, cz)/kpc # pkpc
    mask = dis<rmax_pkpc
    argsort = np.argsort(dis[mask])
    dis = dis[mask][argsort] # pkpc
    mas = m_msol[mask][argsort] # Msol

    # Inside density
    cmas = np.cumsum(mas) # Msol
    vols = 4/3*np.pi * dis**3 # pkpc^3
    rhos = cmas / vols # Msol pkpc-3

    arg = np.argmin(np.abs(rhos - 200*rhoc))
    rvir = dis[arg] # pkpc
    # if(rvir>=np.max(dis)):
    #     warnings.warn("rvir is larger than maximum distance!\nEnlarge the box size!")
    # elif(rvir<=np.min(dis)):
    #     warnings.warn("rvir is smaller than minimum distance!\nNot enough particles!")
    # else:
    #     pass
    rvir_code = rvir * kpc # code unit
    mvir = cmas[arg] # Msol
    return rvir, mvir, rvir_code




def calc_virial_mp(hal, kwargs):
    cx,cy,cz = hal['x'], hal['y'], hal['z']
    rmax_pkpc = kwargs['rmax_pkpc']
    pos_code = kwargs['pos_code']
    m_msol = kwargs['m_msol']
    params = {'H0':kwargs['H0'], 'aexp':kwargs['aexp'], 'kpc':kwargs['kpc']}
    rvir, mvir, rvir_code = calc_virial(cx,cy,cz, rmax_pkpc, pos_code, m_msol, params)
    return hal['id'], rvir, mvir, rvir_code


def _ibox(h, factor=1):
    return np.array([[h['x']-factor*h['r'], h['x']+factor*h['r']],
                        [h['y']-factor*h['r'], h['y']+factor*h['r']],
                        [h['z']-factor*h['r'], h['z']+factor*h['r']]])


def _calc_virial(ith, address, shape, dtype):
    global ihals, snap_star, snap_dm, snap_cell, reft, refn, params

    exist = shared_memory.SharedMemory(name=address)
    virials = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)

    ihal = ihals[ith]
    r200 = 1000
    factor = 0.5

    while(ihal['r']*factor < r200):
        if(factor>1): print(f'Enlarge the box size! {factor}->{factor*2}')
        factor *= 2
        ibox = _ibox(ihal, factor=factor)
        star = snap_star.get_part_instant(box=ibox, pname='star', target_fields=['x','y','z','m'], nthread=1)
        dm = snap_dm.get_part_instant(box=ibox, pname='dm', target_fields=['x','y','z','m'], nthread=1)
        cell = snap_cell.get_cell_instant(box=ibox, target_fields=['x','y','z','rho', 'level'], nthread=1)

        pos_star = star['pos']; mass_star = star['m','Msol']
        pos_dm = dm['pos']; mass_dm = dm['m','Msol']
        pos_cell = cell['pos']; mass_cell = cell['m','Msol']
        pos_code = np.vstack( (pos_star, pos_dm, pos_cell) )
        mass_msol = np.hstack( (mass_star, mass_dm, mass_cell) )

        
        r200kpc, m200, r200 = calc_virial(ihal['x'], ihal['y'], ihal['z'], factor*ihal['r']/snap_star.unit['kpc'], pos_code, mass_msol, params)

    virials['r200kpc'][ith] = r200kpc
    virials['m200'][ith] = m200
    virials['r200'][ith] = r200

    refn.value += 1
    if(refn.value%100==0)&(refn.value>0):
        print(f" > {refn.value}/{len(virials)} {time.time()-reft.value:.2f} sec (ETA: {(len(virials)-refn.value)*(time.time()-reft.value)/refn.value/60:.2f} min)")

assert not os.path.exists(f"{database}/main_prog/mod{mod}.pickle")
pklsave(["warning"], f"{database}/main_prog/mod{mod}.pickle")
for iout in keys[::-1]:
    if(iout==1026): continue
    ihals = halos['catalog'][iout]
    cpulist = pklload(f"{database}/main_prog/cpulist/cpulist_{iout:05d}.pickle")['all']
    uri.timer.verbose=0
    snap_star = snap_stars.get_snap(iout)
    snap_dm = snap_dms.get_snap(iout)
    snap_cell = snap_cells.get_snap(iout)

    params = {'H0':snap_star.H0,
        'aexp':snap_star.aexp,
        'kpc':snap_star.unit['kpc']}

    virials = np.zeros( nhals, dtype=[("r200kpc","<f8"), ("m200","<f8"), ("r200","<f8")])
    uri.timer.verbose=1
    snap_star.get_part(pname='star', target_fields=['x','y','z','m'], nthread=32, cpulist=cpulist, exact_box=False, domain_slicing=True)
    snap_dm.get_part(pname='dm', target_fields=['x','y','z','m'], nthread=32, cpulist=cpulist, exact_box=False, domain_slicing=True)
    snap_cell.get_cell(target_fields=['x','y','z','rho', 'level'], nthread=32, cpulist=cpulist, exact_box=False, domain_slicing=True)
    uri.timer.verbose=1

    atexit.register(flush)
    signal.signal(signal.SIGINT, terminate)
    signal.signal(signal.SIGPIPE, terminate)
    memory = shared_memory.SharedMemory(create=True, size=virials.nbytes)
    virials = np.ndarray(virials.shape, dtype=virials.dtype, buffer=memory.buf)

    reft = Value('f', 0); reft.value = time.time()
    print(reft.value)
    refn = Value('i', 0)
    uri.timer.verbose=0
    print(f"[IOUT {iout:05d}]")

    with Pool(processes=ncpu) as pool:
        async_result = [pool.apply_async(_calc_virial, (ith, memory.name, virials.shape, virials.dtype)) for ith in range(nhals)]
        iterobj = tqdm(async_result, total=len(async_result), desc=f"IOUT{iout:05d} ")# if(uri.timer.verbose>=1) else async_result
        # iterobj = async_result
        for r in iterobj:
            r.get()
    snap_star.clear()
    snap_dm.clear()
    snap_cell.clear()
    pklsave(virials, f"{database}/main_prog/virials_{iout:05d}.pickle")
    print(f"`{database}/main_prog/virials_{iout:05d}.pickle` save done\n\n\n")
    flush(msg=True)
    del virials