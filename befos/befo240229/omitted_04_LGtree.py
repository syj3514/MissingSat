import argparse
database = f"/home/jeon/MissingSat/database/nh2"
# database = f"/gem_home/jeon/MissingSat/database"

print("ex: $ python3 08_LGtree.py [--ncpu 32] [--mod 0]")

parser = argparse.ArgumentParser(description='(syj3514@yonsei.ac.kr)')
parser.add_argument("-n", "--ncpu", required=False, help='The number of threads', type=int)
parser.add_argument("-m", "--mod", required=True, help='divide mod', type=int)
args = parser.parse_args()

mod = args.mod
ncpu = args.ncpu
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
import matplotlib.gridspec as gridspec

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
from icl_draw import drawsnap, add_scalebar, addtext, MakeSub_nolabel, label_to_in, fancy_axis, circle
from importlib import reload
from copy import deepcopy
from multiprocessing import Pool, shared_memory, Value




mode = 'nh2'
iout = 797
repo, rurmode, dp = mode2repo(mode)
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snaps = uri.TimeSeries(snap)
snaps.read_iout_avail()
nout = snaps.iout_avail['iout']; nout=nout[nout <= iout]
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)
hals = uhmi.HaloMaker.load(snap, galaxy=False, double_precision=dp)

from common_func import *


def flush(msg=False, parent=''):
    global memory
    if(msg):
        print(f"{parent} cClearing memory")
        print(f"\tUnlink `{memory.name}`")
    try:
        memory.close()
        memory.unlink()
    except: pass

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

def terminate(signum, frame):
    flush(msg=False, parent=f'[Signal{signum}]')
    atexit.unregister(flush)
    exit(0)

def id2hmid(pid):
    global gtree
    tmp = gtree[gtree['id']==pid][0]
    return tmp['timestep'], tmp['hmid']

def _ibox(h, factor=1):
    return np.array([[h['x']-factor*h['r'], h['x']+factor*h['r']],
                        [h['y']-factor*h['r'], h['y']+factor*h['r']],
                        [h['z']-factor*h['r'], h['z']+factor*h['r']]])

snap_star = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_stars = uri.TimeSeries(snap)
snap_dm = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_dms = uri.TimeSeries(snap)
snap_cell = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap_cells = uri.TimeSeries(snap)






LG = pklload(f"{database}/LG.pickle")
keys = []
for key in LG.keys():
    if(LG[key]['isLG']): keys.append(key)
keys = np.asarray(keys)
print( len(keys), keys )

gtree = pklload("/storage7/NH2/ptree/ptree_stable.pkl")
rgtree = {}
for key in tqdm( keys ):
    if(LG[key]['isLG']):
        BGG = LG[key]['BGG']
        target = gtree[ (gtree['hmid']==BGG['id'])&(gtree['timestep']==BGG['timestep']) ]
        tmp = gtree[gtree['last'] == target['last']]
        argsort = np.argsort(-tmp['timestep'])
        rgtree[key] = tmp[argsort]
        ind = isin(nout, rgtree[key]['timestep'])
        
dtype1 = gals.dtype
dtype2 = [('halo_id', '<i4'), ('halo_nparts', '<i4'), ('halo_level', '<i4'), ('halo_host', '<i4'), ('halo_hostsub', '<i4'), ('halo_x', '<f8'), ('halo_y', '<f8'), ('halo_z', '<f8'), ('halo_vx', '<f8'), ('halo_vy', '<f8'), ('halo_vz', '<f8'), ('halo_mvir', '<f8'), ('halo_rvir', '<f8')]
dtype3 = [('fcontam', '<f8'), ('dist', '<f8'), ('central', '?'), ('main', '?'), ('r200', '<f8'), ('m200', '<f8'), ('r200_code', '<f8'), ('m_star_200', '<f8'), ('m_gas_200', '<f8'), ('fcontam_200', '<f8'), ('rp', '<f8'), ('sfr', '<f8'), ('sfr_tot', '<f8'), ('galaxy_nh2', '<i8'), ('halo_nh2', '<i8'), ('matchrate', '<f8')]
dtype4 = [('fat', '<i8'), ('son', '<i8'), ('score_fat', '<f8'), ('score_son', '<f8')]
dtype = np.dtype(dtype1.descr + dtype2 + dtype3 + dtype4)

pure = hals[hals['mcontam']==0]
mindm = np.min(pure['m']/pure['nparts'])*snap.unit['Msol']

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
    stardis = dis[:nstar]; dmdis = dis[nstar:nstar+ndm]; celldis = dis[nstar+ndm:]
    starmas = m_msol[:nstar]; dmmas = m_msol[nstar:nstar+ndm]; cellmas = m_msol[nstar+ndm:]

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

    mstar200 = np.sum(starmas[stardis<rvir])
    mgas200 = np.sum(cellmas[celldis<rvir])
    indm = dmdis<rvir
    dmdis = dmdis[indm]; dmmas = dmmas[indm]
    fcontam200 = np.sum(dmmas[dmmas > 1.5*mindm]) / np.sum(dmmas)

    return rvir, mvir, rvir_code, mstar200, mgas200, fcontam200

def _calc_virial(key, address, shape, dtype):
    global TREE, snap_star, snap_dm, snap_cell, reft, refn, params, keys, donelist
    if(key in donelist):
        refn.value += 1
    else:
        exist = shared_memory.SharedMemory(name=address)
        virials = np.ndarray(shape=shape, dtype=dtype, buffer=exist.buf)

        branch = TREE[key]
        ihal = branch[-1]
        ith = np.where(keys == key)[0][0]
        r200 = 1000
        factor = 0.75

        while(ihal['halo_rvir']*factor < r200):
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
            if(ns[0]+ns[1] == 0)or(ns[1] == 0):
                print(f"! Nstar={ns[0]}, Ndm={ns[1]} !")
                r200 = 1000
            else:
                r200kpc, m200, r200, mstar200, mgas200, fcontam200 = calc_virial(ihal['x'], ihal['y'], ihal['z'], factor*ihal['halo_rvir']/snap_star.unit['kpc'], pos_code, mass_msol,ns, params)

        virials['r200'][ith] = r200kpc
        virials['m200'][ith] = m200
        virials['r200_code'][ith] = r200
        virials['m_star_200'][ith] = mstar200
        virials['m_gas_200'][ith] = mgas200
        virials['fcontam_200'][ith] = fcontam200

        refn.value += 1
    if(refn.value%100==0)&(refn.value>0):
        print(f" > {refn.value}/{len(virials)} {time.time()-reft.value:.2f} sec (ETA: {(len(virials)-refn.value)*(time.time()-reft.value)/refn.value/60:.2f} min)")


fname = f"{database}/main_prog/mainhalos_{mod}.pickle"
TREE = {}
for key in keys:
    TREE[key] = None
if(os.path.exists(fname)):
    TREE = pklload(fname)
    for key in keys:
        nan = np.isnan(TREE[key]['fcontam_200'])
        if(True in nan):
            TREE[key] = TREE[key][~nan]

for i, iout in enumerate(nout[::-1]):
    if(iout%10 != mod): continue

    donelist = []
    if(TREE[keys[-1]] is not None):
        for key in keys:
            if(iout in TREE[key]['timestep']):
                donelist.append(key)
    if(len(donelist) == len(keys)): 
        print(f"[{iout:04d}] All done")
        continue
    print(f"\n[{iout:04d}] {donelist}")
    if(iout==797):
        for key in tqdm( keys ):
            if(not LG[key]['isLG']): continue
            if(key in donelist): continue
            BGG = LG[key]['BGG']
            table = np.zeros(1, dtype=dtype)
            itree = rgtree[key]
            itree = itree[itree['timestep'] == iout]
            for iname in dtype.names:
                if(iname in BGG.dtype.names):
                    # dtype1, dtype2, dtype3
                    table[iname] = BGG[iname]
                else:
                    # dtype4
                    table['fat'] = id2hmid(itree['fat'])[1] if(itree['fat']>0) else itree['fat']
                    table['son'] = id2hmid(itree['son'])[1] if(itree['son']>0) else itree['son']
                    table['score_fat'] = itree['score_fat']
                    table['score_son'] = itree['score_son']
            TREE[key] = table
    else:
        snap_star = snap_stars.get_snap(iout)
        snap_dm = snap_dms.get_snap(iout)
        snap_cell = snap_cells.get_snap(iout)
        igals = uhmi.HaloMaker.load(snap_star, galaxy=True)
        ihals = uhmi.HaloMaker.load(snap_star, galaxy=False)
        if(len(igals)==0)and(len(ihals)==0): continue
        params = {'H0':snap_star.H0,
                'aexp':snap_star.aexp,
                'kpc':snap_star.unit['kpc']}
        ihals = ihals[ihals['mcontam'] < ihals['m']]
        for key in tqdm( keys, desc=f"[{iout:04d}] From Catalogs" ):
            if(not LG[key]['isLG']): continue
            if(key in donelist): continue
            #------------------------------------------
            # From TreeMaker
            #------------------------------------------
            ref = time.time(); tcount=0
            '''
            'fat', 'son', 'score_fat', 'score_son'
            '''
            itree = rgtree[key]
            if(not iout in itree['timestep']): continue
            itree = itree[itree['timestep'] == iout] 
                
            table = np.zeros(1, dtype=dtype)[0]
            table['fat'] = id2hmid(itree['fat'])[1] if(itree['fat']>0) else itree['fat']
            table['son'] = id2hmid(itree['son'])[1] if(itree['son']>0) else itree['son']
            table['score_fat'] = itree['score_fat']
            table['score_son'] = itree['score_son']
            #------------------------------------------
            # From GalaxyMaker
            #------------------------------------------
            '''
            'id', 'timestep', 'level', 'host', 'hostsub', 'nbsub', 'nextsub', 
            'aexp', 'age_univ', 'm', 'macc', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Lx', 'Ly', 'Lz', 
            'r', 'a', 'b', 'c', 'ek', 'ep', 'et', 'spin', 
            'rvir', 'mvir', 'tvir', 'cvel', 'rho0', 'rc'
            '''
            igal = igals[itree['hmid']-1]
            for iname in table.dtype.names:
                if(iname in igal.dtype.names):
                    table[iname] = igal[iname]
            #------------------------------------------
            # From Matched Halo
            #------------------------------------------
            '''
            'halo_id', 'halo_nparts', 'halo_level', 'halo_host', 'halo_hostsub', 
            'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_mvir', 'halo_rvir'
            'fcontam', 'dist'
            '''
            tmp = ihals[ihals['rvir'] > table['r']]
            cands = cut_sphere(tmp, table['x'], table['y'], table['z'], table['r'])
            if(len(cands)>0):
                dists = distance(cands, table)
                mask = dists < cands['rvir']
                if(True in mask):
                    cands = cands[mask]
                    dists = distance(cands, table)
                    mask = dists < (cands['rvir']-table['r'])
                    if(True in mask):
                        cands = cands[mask]
            else:
                factor=2
                while len(cands)==0:
                    cands = cut_sphere(tmp, table['x'], table['y'], table['z'], table['r']*factor)
                    factor *= 2
            ihal = cands[np.argmax(cands['mvir'])]
            for iname in table.dtype.names:
                if(iname[:5]=='halo_'):
                    if(iname[5:] in ihal.dtype.names):
                        table[iname] = ihal[iname[5:]]
            table['fcontam'] = ihal['mcontam']/ihal['m']
            table['dist'] = distance(ihal, table)
            TREE[key] = np.array([table]) if(TREE[key] is None) else np.hstack((TREE[key], table))

        #------------------------------------------
        # From Raw data
        #------------------------------------------
        '''
        'r200', 'm200', 'r200_code', 'm_star_200', 'm_gas_200', 'fcontam_200'
        '''
        cpulists = []
        for key in keys:
            if(key in donelist): continue
            ihal = TREE[key][-1]
            ibox = _ibox(ihal, factor=4)
            cpulists.append( snap_star.get_involved_cpu(box=ibox) )
        cpulists = np.unique(np.hstack(cpulists))
        virials = np.zeros( len(keys), dtype=[('key','<i4'),
            ("r200","<f8"), ("m200","<f8"), ("r200_code","<f8"), ("m_star_200","<f8"), ("m_gas_200","<f8"), ("fcontam_200","<f8")
            ])
        uri.timer.verbose=1
        snap_star.get_part(pname='star', target_fields=['x','y','z','m'], nthread=ncpu, exact_box=False, domain_slicing=True, cpulist=cpulists)
        snap_dm.get_part(pname='dm', target_fields=['x','y','z','m'], nthread=ncpu, exact_box=False, domain_slicing=True, cpulist=cpulists)
        snap_cell.get_cell(target_fields=['x','y','z','rho', 'level'], nthread=ncpu, exact_box=False, domain_slicing=True, cpulist=cpulists)
        uri.timer.verbose=0

        atexit.register(flush)
        signal.signal(signal.SIGINT, terminate)
        signal.signal(signal.SIGPIPE, terminate)
        memory = shared_memory.SharedMemory(create=True, size=virials.nbytes)
        virials = np.ndarray(virials.shape, dtype=virials.dtype, buffer=memory.buf)
        virials['key'] = keys


        reft = Value('f', 0); reft.value = time.time()
        refn = Value('i', 0)
        uri.timer.verbose=0
        signal.signal(signal.SIGTERM, terminate)
        with Pool(processes=len(keys)) as pool:
            async_result = [pool.apply_async(_calc_virial, (key, memory.name, virials.shape, virials.dtype)) for key in keys]
            iterobj = tqdm(async_result, total=len(async_result), desc=f"[{iout:04d}] From Raw data")# if(uri.timer.verbose>=1) else async_result
            # iterobj = async_result
            for r in iterobj:
                r.get()
        snap_star.clear()
        snap_dm.clear()
        snap_cell.clear()

        for key in keys:
            if(key in donelist): continue
            TREE[key][-1]['r200'] = virials['r200'][virials['key']==key][0]
            TREE[key][-1]['m200'] = virials['m200'][virials['key']==key][0]
            TREE[key][-1]['r200_code'] = virials['r200_code'][virials['key']==key][0]
            TREE[key][-1]['m_star_200'] = virials['m_star_200'][virials['key']==key][0]
            TREE[key][-1]['m_gas_200'] = virials['m_gas_200'][virials['key']==key][0]
            TREE[key][-1]['fcontam_200'] = virials['fcontam_200'][virials['key']==key][0]
        flush(msg=False)
    pklsave(TREE, fname, overwrite=True)
    print(f"`{fname}` save done")











