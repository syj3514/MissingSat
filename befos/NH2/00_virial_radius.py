from common_func import *
if type_of_script() == 'jupyter': from tqdm.notebook import tqdm
else: from tqdm import tqdm
    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colormaps
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D, proj3d
#import cmasher as cmr

import numpy as np
import os, glob, atexit, signal, time, warnings, argparse, subprocess

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
from rur.sci.photometry import measure_luminosity
from rur.sci.geometry import get_angles, euler_angle
from rur.utool import rotate_data
from scipy.ndimage import gaussian_filter
uri.timer.verbose=1

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind, isin
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle

from importlib import reload
from copy import deepcopy
from multiprocessing import Pool, shared_memory, Value

memory=[]


mode = 'nh2'
iout = 797
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)
database = f"/home/jeon/MissingSat/database/nh2"



lvl1s = hals[ (hals['level']==1) & (hals['mcontam'] < hals['m'])]
print(len(lvl1s))

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

def terminate(signum):
    flush(msg=True, parent=f'[Signal{signum}]')
    atexit.unregister(flush)
    exit(0)

def _ibox(h, factor=1, rname='r'):
    return np.array([[h['x']-factor*h[rname], h['x']+factor*h[rname]],
                        [h['y']-factor*h[rname], h['y']+factor*h[rname]],
                        [h['z']-factor*h[rname], h['z']+factor*h[rname]]])

def calc_virial(cx,cy,cz, rmax_pkpc, pos_code, m_msol, ns, params):
    '''
    input:
        cx,cy,cz : center of halo
        star, dm, cell : data
    output:
        rvir : virial radius
        mvir : virial mass
        rvir_code : virial radius in code unit
    '''
    global mindm
    nstar=ns[0]; ndm=ns[1]; ncell=ns[2]
    H0 = params['H0']; aexp=params['aexp']; kpc=params['kpc']
    # critical density
    H02 = (H0 * 3.24078e-20)**2 # s-2
    G = 6.6743e-11 # N m2 kg-2 = kg m s-2 m2 kg-2 = m3 s-2 kg-1
    rhoc = 3 * H02 /8 /np.pi /G # kg m-3
    rhoc *= 5.02785e-31  * (3.086e+19)**3 # Msol ckpc-3
    rhoc /= (aexp**3) # Msol pkpc-3

    # Sorting
    dis = distance3d(pos_code[:,0], pos_code[:,1], pos_code[:,2], cx, cy, cz)/kpc # pkpc
    # stardis = dis[:nstar]; dmdis = dis[nstar:nstar+ndm]; celldis = dis[nstar+ndm:]
    # starmas = m_msol[:nstar]; dmmas = m_msol[nstar:nstar+ndm]; cellmas = m_msol[nstar+ndm:]

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

    # mstar200 = np.sum(starmas[stardis<rvir])
    # mgas200 = np.sum(cellmas[celldis<rvir])
    # indm = dmdis<rvir
    # dmdis = dmdis[indm]; dmmas = dmmas[indm]
    # fcontam200 = np.sum(dmmas[dmmas > 1.5*mindm]) / np.sum(dmmas)

    return rvir, mvir, rvir_code#, mstar200, mgas200, fcontam200

def _calc_virial(sub, address, shape, dtype):
    global TREE, snap_star, snap_dm, snap_cell, reft, refn, params, keys

    exist = shared_memory.SharedMemory(name=address)
    results = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)

    # branch = TREE[key]
    ihal = sub
    # ith = np.where(keys == key)[0][0]
    r200 = 1000
    factor = 0.75

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
        ns = [len(pos_star), len(pos_dm), len(pos_cell)]

        
        r200kpc, m200, r200, = calc_virial(ihal['x'], ihal['y'], ihal['z'], factor*ihal['r']/snap_star.unit['kpc'], pos_code, mass_msol,ns, params)

    results['id'][sub['id']-1] = sub['id']
    results['r200kpc'][sub['id']-1] = r200kpc
    results['m200'][sub['id']-1] = m200
    results['r200'][sub['id']-1] = r200
    # virials['m_star_200'][ith] = mstar200
    # virials['m_gas_200'][ith] = mgas200
    # virials['fcontam_200'][ith] = fcontam200

    refn.value += 1
    if(refn.value%100==0)&(refn.value>0):
        print(f" > {refn.value}/{len(virials)} {time.time()-reft.value:.2f} sec (ETA: {(len(virials)-refn.value)*(time.time()-reft.value)/refn.value/60:.2f} min)")



snap_star = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_dm = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_cell = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_star.get_part(pname='star', target_fields=['x','y','z','m'], nthread=32, exact_box=False, domain_slicing=False)
snap_dm.get_part(pname='dm', target_fields=['x','y','z','m'], nthread=32, exact_box=False, domain_slicing=False)
snap_cell.get_cell(target_fields=['x','y','z','rho', 'level'], nthread=32, exact_box=False, domain_slicing=False)
params = {'H0':snap_star.H0,
        'aexp':snap_star.aexp,
        'kpc':snap_star.unit['kpc']}
virials = np.zeros( len(hals), dtype=[("r200kpc","<f8"), ("m200","<f8"), ("r200","<f8")])
if(os.path.exists(f"{database}/virial_radius_{mode}_{iout}.pickle")):
    virials = pklload(f"{database}/virial_radius_{mode}_{iout}.pickle")
uri.timer.verbose=0
for lvl1 in lvl1s:
    changed = False
    if(virials[lvl1['id']-1]['r200kpc']>0):
        # print(f"[{lvl1['id']:07d}] Skip {virials[lvl1['id']-1]['r200kpc']}")
        pass
    else:
        print(f"[{lvl1['id']:07d}] Main R200")
        r200 = 1000
        factor = 0.75
        ibox = _ibox(lvl1, factor=1)
        while(lvl1['r']*factor < r200):
            if(factor>1): print(f'Enlarge the box size! {factor}->{factor*2}')
            factor *= 2
            ibox = _ibox(lvl1, factor=factor)
            star = snap_star.get_part_instant(box=ibox, pname='star', target_fields=['x','y','z','m'], nthread=1)
            dm = snap_dm.get_part_instant(box=ibox, pname='dm', target_fields=['x','y','z','m'], nthread=1)
            cell = snap_cell.get_cell_instant(box=ibox, target_fields=['x','y','z','rho', 'level'], nthread=1)
            pos_star = star['pos']; mass_star = star['m','Msol']
            pos_dm = dm['pos']; mass_dm = dm['m','Msol']
            pos_cell = cell['pos']; mass_cell = cell['m','Msol']
            pos_code = np.vstack( (pos_star, pos_dm, pos_cell) )
            mass_msol = np.hstack( (mass_star, mass_dm, mass_cell) )
            ns = [len(pos_star), len(pos_dm), len(pos_cell)]
            r200kpc, m200, r200 = calc_virial(lvl1['x'], lvl1['y'], lvl1['z'], factor*lvl1['r']/snap.unit['kpc'], pos_code, mass_msol, ns, params)
        virials[lvl1['id']-1]['r200kpc'] = r200kpc
        virials[lvl1['id']-1]['m200'] = m200
        virials[lvl1['id']-1]['r200'] = r200
        changed=True
    subs = None
    subs = hals[hals['host'] == lvl1['id']]
    subs = subs[subs['id'] != lvl1['id']]
    yet = virials['r200kpc'][subs['id']-1]==0
    subs = subs[yet]
    if(len(subs)>0):
        if(len(subs)>1):
            print(f"[{lvl1['id']:07d}] Subs R200 {len(subs)}")
            nproc = min(32, len(subs))
            rdtype = np.dtype(virials.dtype.descr + [('id', '<i4')])
            results = np.empty(virials.shape, dtype=rdtype)
            memory = shared_memory.SharedMemory(create=True, size=results.nbytes)
            results = np.ndarray(virials.shape, dtype=rdtype, buffer=memory.buf)
            reft = Value('f', 0); reft.value = time.time()
            refn = Value('i', 0)
            with Pool(processes=nproc) as pool:
                async_result = [pool.apply_async(_calc_virial, (sub, memory.name, virials.shape, rdtype)) for sub in subs]
                iterobj = tqdm(async_result, total=len(async_result), desc=f"[{iout:04d}] Subhalos")# if(uri.timer.verbose>=1) else async_result
                # results = pool.starmap(calc_virial_mp, tqdm([(sub,kwargs) for sub in subs]))
                for r in iterobj:
                    r.get()
            for result in results:
                virials[result['id']-1]['r200kpc'] = result['r200kpc']
                virials[result['id']-1]['m200'] = result['m200']
                virials[result['id']-1]['r200'] = result['r200']
            changed=True
        else:
            tmp = subs[0]
            print(f"[{lvl1['id']:07d}] Subs R200 single")
            r200 = 1000
            factor = 0.75
            ibox = _ibox(tmp, factor=1)
            while(tmp['r']*factor < r200):
                if(factor>1): print(f'Enlarge the box size! {factor}->{factor*2}')
                factor *= 2
                ibox = _ibox(tmp, factor=factor)
                star = snap_star.get_part_instant(box=ibox, pname='star', target_fields=['x','y','z','m'], nthread=1)
                dm = snap_dm.get_part_instant(box=ibox, pname='dm', target_fields=['x','y','z','m'], nthread=1)
                cell = snap_cell.get_cell_instant(box=ibox, target_fields=['x','y','z','rho', 'level'], nthread=1)
                pos_star = star['pos']; mass_star = star['m','Msol']
                pos_dm = dm['pos']; mass_dm = dm['m','Msol']
                pos_cell = cell['pos']; mass_cell = cell['m','Msol']
                pos_code = np.vstack( (pos_star, pos_dm, pos_cell) )
                mass_msol = np.hstack( (mass_star, mass_dm, mass_cell) )
                ns = [len(pos_star), len(pos_dm), len(pos_cell)]
                r200kpc, m200, r200 = calc_virial(tmp['x'], tmp['y'], tmp['z'], factor*lvl1['r']/snap.unit['kpc'], pos_code, mass_msol, ns, params)
            virials[tmp['id']-1]['r200kpc'] = r200kpc
            virials[tmp['id']-1]['m200'] = m200
            virials[tmp['id']-1]['r200'] = r200
            changed=True

        
    if(changed): pklsave(virials, f"{database}/virial_radius_{mode}_{iout}.pickle", overwrite=True)
uri.timer.verbose=1
