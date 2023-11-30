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
len(lvl1s)



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
    if(rvir>=np.max(dis)):
        warnings.warn("rvir is larger than maximum distance!\nEnlarge the box size!")
    elif(rvir<=np.min(dis)):
        warnings.warn("rvir is smaller than minimum distance!\nNot enough particles!")
    else:
        pass
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



snap_star = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_dm = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_cell = uri.RamsesSnapshot(repo, iout, mode=rurmode)

virials = np.zeros( len(hals), dtype=[("r200kpc","<f8"), ("m200","<f8"), ("r200","<f8")])
if(os.path.exists(f"{database}/virial_radius_{mode}_{iout}.pickle")):
    virials = pklload(f"{database}/virial_radius_{mode}_{iout}.pickle")
uri.timer.verbose=0
for lvl1 in tqdm( lvl1s ):
    if(virials[lvl1['id']-1]['r200kpc']>0): continue
    if(len(snap_star.cpulist_part)>400)or(len(snap_dm.cpulist_part)>400)or(len(snap_cell.cpulist_cell)>400):
        print(f"Clearing cpulist {len(snap_star.cpulist_part)} {len(snap_dm.cpulist_part)} {len(snap_cell.cpulist_cell)}")
        snap_star.clear()
        snap_dm.clear()
        snap_cell.clear()
    snap_star.set_box_halo(lvl1, 2, radius_name='r'); snap_star.get_part(pname='star', target_fields=['x','y','z','m'], nthread=32)
    snap_dm.set_box_halo(lvl1, 2, radius_name='r'); snap_dm.get_part(pname='dm', target_fields=['x','y','z','m'], nthread=32)
    snap_cell.set_box_halo(lvl1, 2, radius_name='r'); snap_cell.get_cell(target_fields=['x','y','z','rho','level'], nthread=32)    

    pos_star = snap_star.part['pos']; mass_star = snap_star.part['m','Msol']
    pos_dm = snap_dm.part['pos']; mass_dm = snap_dm.part['m','Msol']
    pos_cell = snap_cell.cell['pos']; mass_cell = snap_cell.cell['m','Msol']
    pos_code = np.vstack( (pos_star, pos_dm, pos_cell) )
    mass_msol = np.hstack( (mass_star, mass_dm, mass_cell) )

    params = {'H0':snap_star.H0,
                'aexp':snap_star.aexp,
                'kpc':snap_star.unit['kpc']}
    r200kpc, m200, r200 = calc_virial(lvl1['x'], lvl1['y'], lvl1['z'], 2*lvl1['r']/snap.unit['kpc'], pos_code, mass_msol, params)
    virials[lvl1['id']-1]['r200kpc'] = r200kpc
    virials[lvl1['id']-1]['m200'] = m200
    virials[lvl1['id']-1]['r200'] = r200
    if(lvl1['nbsub']>0):
        if(lvl1['nbsub']>1):
            subs = None
            subs = hals[hals['host'] == lvl1['id']]
            subs = subs[subs['id'] != lvl1['id']]
            # now = lvl1
            # while(now['nextsub']>0):
            #     tmp = hals[now['nextsub']-1]
            #     assert tmp['host']==lvl1['id']
            #     subs = tmp if(subs is None) else np.hstack( (subs, tmp) )
            #     now = tmp
            #     # if(len(subs) > (lvl1['nbsub']+1)):
            #     #     print(len(subs), lvl1['nbsub'], now['id'], now['nextsub'])
            #     #     raise ValueError("Too many subhalos!")
            nproc = min(24, len(subs))
            kwargs = {
                'rmax_pkpc':2*lvl1['r']/snap.unit['kpc'], 
                'pos_code':pos_code, 
                'm_msol':mass_msol,
                'H0':snap_star.H0,
                'aexp':snap_star.aexp,
                'kpc':snap_star.unit['kpc']}
            with Pool(processes=nproc) as pool:
                results = pool.starmap(calc_virial_mp, tqdm([(sub,kwargs) for sub in subs]))
            for result in results:
                virials[result[0]-1]['r200kpc'] = result[1]
                virials[result[0]-1]['m200'] = result[2]
                virials[result[0]-1]['r200'] = result[3]
        else:
            tmp = hals[lvl1['nextsub']-1]
            r200kpc, m200, r200 = calc_virial(tmp['x'], tmp['y'], tmp['z'], 2*lvl1['r']/snap.unit['kpc'], pos_code, mass_msol, params)
            virials[tmp['id']-1]['r200kpc'] = r200kpc
            virials[tmp['id']-1]['m200'] = m200
            virials[tmp['id']-1]['r200'] = r200

        
    pklsave(virials, f"{database}/virial_radius_{mode}_{iout}.pickle", overwrite=True)
uri.timer.verbose=1
