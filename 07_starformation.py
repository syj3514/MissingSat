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


rtree1 = pklload(f"{database1}/reduced_tree.pickle")
rtree2 = pklload(f"{database2}/reduced_tree.pickle")



###########################################################
# Functions
###########################################################
def get_dt(snap, snaps):
    istep = np.where(snaps.iout_avail['iout'] == snap.iout)[0][0]
    table = snaps.iout_avail['time']
    return np.abs(table[istep-1] - table[istep])

def get_nbor(icell, cells, return_nbor=False):
    dx = icell['dx']
    distx = np.abs(cells['x'] - icell['x'])
    if(len(cells)>300000):
        size = 128*dx
        indx = distx <= size
        cells = cells[indx]
        distx = np.abs(cells['x'] - icell['x'])
    disty = np.abs(cells['y'] - icell['y'])
    if(len(cells)>300000):
        size = 128*dx
        indy = disty <= size
        cells = cells[indy]
        distx = np.abs(cells['x'] - icell['x'])
        disty = np.abs(cells['y'] - icell['y'])
    distz = np.abs(cells['z'] - icell['z'])
    if(len(cells)>300000):
        size = 128*dx
        indz = distz <= size
        cells = cells[indz]
        distx = np.abs(cells['x'] - icell['x'])
        disty = np.abs(cells['y'] - icell['y'])
        distz = np.abs(cells['z'] - icell['z'])
    if(len(cells)>300000):
        size = 64*dx
        indx = distx <= size
        indy = disty <= size
        indz = distz <= size
        cells = cells[indx&indy&indz]
        distx = np.abs(cells['x'] - icell['x'])
        disty = np.abs(cells['y'] - icell['y'])
        distz = np.abs(cells['z'] - icell['z'])
    dxs = 1 / 2**cells['level'] # <--- main bottleneck
    size = (dx + dxs)/2
    indx = distx <= size
    indy = disty <= size
    indz = distz <= size
    neighs = cells[indx&indy&indz]
    neighs = neighs[neighs['rho'] != icell['rho']]
    
    

    samez = (neighs['z'] <= (icell['z'] + icell['dx']/2))&(neighs['z'] >= (icell['z'] - icell['dx']/2))
    samey = (neighs['y'] <= (icell['y'] + icell['dx']/2))&(neighs['y'] >= (icell['y'] - icell['dx']/2))
    samex = (neighs['x'] <= (icell['x'] + icell['dx']/2))&(neighs['x'] >= (icell['x'] - icell['dx']/2))

    # left right
    sameyz = samey & samez
    lrs = neighs[sameyz]
    ls = lrs[lrs['x'] < icell['x']]
    rs = lrs[lrs['x'] > icell['x']]
    # front back
    samezx = samez & samex
    fbs = neighs[samezx]
    fs = fbs[fbs['y'] < icell['y']]
    bs = fbs[fbs['y'] > icell['y']]
    # up down
    samexy = samex & samey
    uds = neighs[samexy]
    us = uds[uds['z'] < icell['z']]
    ds = uds[uds['z'] > icell['z']]
    if(return_nbor): return ls, rs, fs, bs, us, ds, neighs
    return ls, rs, fs, bs, us, ds

def wmean(vals,ws):
    if(len(vals)==1): return vals[0]
    return np.average(vals, weights=ws)

def cell_calc(target, snap):
    radii = 1.5
    snap.set_box_halo(target, radii, radius_name='r')
    snap.get_cell(nthread=16)
    allcells = snap.cell
    cells = cut_sphere(allcells, target['x'], target['y'], target['z'], target['r'])
    dtype = cells.dtype.descr + ndtype
    newcells = np.zeros(len(cells), dtype=dtype)
    for iname in cells.dtype.names:
        newcells[iname] = cells[iname]

    newcells['dense'] = newcells['rho'] > d_gmc
    if(np.sum(newcells['dense'])==0):
        snap.clear()
        return newcells
    print(np.sum(newcells['dense']))
    where = np.where(newcells['dense'])[0]
    # for i, icell in tqdm(enumerate(cells), total=len(cells)):
    for i, icell in tqdm(zip(where, cells[newcells['dense']]), total=np.sum(newcells['dense'])):
        if(not newcells[i]['dense']): continue
        ls,rs,fs,bs,us,ds = get_nbor(icell, allcells, return_nbor=False)
        while(len(ls)==0 or len(rs)==0 or len(fs)==0 or len(bs)==0 or len(us)==0 or len(ds)==0):
            radii += 0.5
            snap.set_box_halo(target, radii, radius_name='r')
            snap.get_cell(nthread=16)
            allcells = snap.cell
            ls,rs,fs,bs,us,ds = get_nbor(icell, allcells, return_nbor=False)
            if(radii > 4):
                newcells[i]['trgv'] = np.nan
                radii = 1.5
                snap.set_box_halo(target, radii, radius_name='r')
                snap.get_cell(nthread=16)
                allcells = snap.cell
                break
        if(np.isnan(newcells[i]['trgv'])): continue

        # The local 3D instantaneous velocity dispersion sig_g
        trgv = 0
        for val in ['vx','vy','vz']:
            d = icell['rho']
            dl = wmean(ls['rho'], ls['vol'])
            dr = wmean(rs['rho'], rs['vol'])
            df = wmean(fs['rho'], fs['vol'])
            db = wmean(bs['rho'], bs['vol'])
            du = wmean(us['rho'], us['vol'])
            dd = wmean(ds['rho'], ds['vol'])
            vl = ( dl*wmean(ls[val], ls['vol']) + icell[val]*d ) / ( dl + d )
            vr = ( dr*wmean(rs[val], rs['vol']) + icell[val]*d ) / ( dr + d )
            vf = ( df*wmean(fs[val], fs['vol']) + icell[val]*d ) / ( df + d )
            vb = ( db*wmean(bs[val], bs['vol']) + icell[val]*d ) / ( db + d )
            vu = ( du*wmean(us[val], us['vol']) + icell[val]*d ) / ( du + d )
            vd = ( dd*wmean(ds[val], ds['vol']) + icell[val]*d ) / ( dd + d )
            trgv += (vl-vr)**2 + (vf-vb)**2 + (vu-vd)**2

        # The sound speed squared c_s2
        P = icell['P']
        c_s = np.sqrt((gamma-1) * P / d)
        c_s2 = max(smallc**2, c_s**2)

        # # Mach number
        mach2 = trgv/c_s2

        ilevel = icell['level']
        dx = icell['dx']
        mcell = icell['rho'] * dx**3

        # The virial parameter
        alpha0 = 5*(trgv + c_s2)/(np.pi * factG * icell['rho'] * dx**2)

        # Variance of the logarithmic PDF
        # sigs  = np.log(1.0 + 0.16*mach2)
        sigs  = np.log(1.0 + 0.16*trgv/c_s2)

        # The criitical densisity contrast by Padoan & Nordlund (2011)
        # scrit = np.log(0.067 / theta**2 * alpha0 * mach2)
        scrit = np.log(0.067 / theta**2 * alpha0 * trgv/c_s2)

        # sigs > scrit -> star formation
        sfr_ff = e_cts/2*phi_t * np.exp(3/8*sigs) * (2 - erfc( (sigs-scrit)/np.sqrt(2*sigs) ))

        # The local free-fall time of the gas
        tstar     = 0.5427 * np.sqrt(1/( factG*icell['rho'] )) # 0.5427 = sqrt(3pi / 32)
        if(m_star < 0e0): mstar = n_star/(scale_nH*aexp**3)*vol_min*fstar_min
        else: mstar=m_star*mass_sph

        nstar_fine = 0; nstar_iout = 0
        dt_fine = dt_new[ilevel-1]
        PoissMean_fine = min( dt_fine*sfr_ff/tstar*mcell/mstar, 10)
        PoissMean_iout = min( dt_iout*sfr_ff/tstar*mcell/mstar, 10)
        nstar_fine = poissdev(localseed,PoissMean_fine,nstar_fine)
        if(PoissMean_fine>0.4): print(f"{PoissMean_fine=}, {nstar_fine=}")
        nstar_iout = poissdev(localseed,PoissMean_iout,nstar_iout)
        if(PoissMean_iout>0.4): print(f"{PoissMean_iout=}, {nstar_iout=}")
        nstar_corr_fine=nstar_fine; nstar_corr_iout=nstar_iout
        mgas_fine = nstar_fine*mstar
        if(mgas_fine > 0.9*mcell): nstar_corr_fine = int(0.9*mcell/mstar)
        mgas_iout = nstar_iout*mstar
        if(mgas_iout > 0.9*mcell): nstar_corr_iout = int(0.9*mcell/mstar)

        newcells[i]['trgv'] = trgv
        newcells[i]['c_s2'] = c_s2
        newcells[i]['mach2'] = mach2
        newcells[i]['alpha0'] = alpha0
        newcells[i]['sigs'] = sigs
        newcells[i]['scrit'] = scrit
        newcells[i]['sfr_ff'] = sfr_ff
        newcells[i]['tstar'] = tstar
        newcells[i]['mstar'] = mstar
        newcells[i]['nstar_fine'] = nstar_fine
        newcells[i]['dt_fine'] = dt_fine
        newcells[i]['PoissMean_fine'] = PoissMean_fine
        newcells[i]['nstar_corr_fine'] = nstar_corr_fine
        newcells[i]['nstar_iout'] = nstar_iout
        newcells[i]['dt_iout'] = dt_iout
        newcells[i]['PoissMean_iout'] = PoissMean_iout
        newcells[i]['nstar_corr_iout'] = nstar_corr_iout
        if(nstar_iout>0): print(newcells[i])
    snap.clear()
    return newcells
###########################################################
# Snapshot
###########################################################
for pair in pairs1:
    iid = pair['id']
    try:
        target = rtree1[iid][-1]
        tnext = rtree1[iid][-2]
        print(f"{iid}, {target['mstar_vir']=:}")
        print(f"{iid}, {tnext['mstar_vir']=:}")
    except:
        continue
    snap = snap1s.get_snap(target['timestep'])
    from ramses_function import *
    # Variable in this snapshot
    h0 = params('h0', snap)
    aexp = params('aexp', snap)
    omega_m = params('omega_m', snap)
    scale_nH = params('scale_nH', snap)
    nCOM = params('nCOM', snap)
    d_gmc = params('d_gmc', snap)
    factG = params('factG', snap)
    dt_old = params('dt_old', snap)
    dt_new = params('dt_new', snap)
    mass_sph = params('mass_sph', snap)
    localseed = params('localseed', snap)
    nlevelmax = snap.params['levelmax']
    dx_min   = 0.5**nlevelmax
    vol_min  = dx_min**snap.params['ndim']
    dt_iout = get_dt(snap, snap1s)
    ndtype = [('dense', bool),
        ('trgv', 'f8'), ('c_s2', 'f8'), ('mach2', 'f8'), 
        ('alpha0', 'f8'), ('sigs', 'f8'), ('scrit', 'f8'), 
        ('sfr_ff', 'f8'), ('tstar', 'f8'), ('mstar', 'f8'), 
        ('nstar_fine', 'i8'), ('dt_fine', 'f8'), ('PoissMean_fine', 'f8'), ('nstar_corr_fine', 'f8'),
        ('nstar_iout', 'i8'), ('dt_iout', 'f8'), ('PoissMean_iout', 'f8'), ('nstar_corr_iout', 'f8')
        ]


    newcells = cell_calc(target, snap)
    snap.clear()
    if(np.max(newcells['nstar_iout'])>0):
        where = newcells['sfr_ff']>0
        print(newcells[where]['sfr_ff'])
        print(newcells[where]['nstar_fine'])
        print(newcells[where]['nstar_iout'])