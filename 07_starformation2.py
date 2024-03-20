from IPython import get_ipython

ncpu=48
home = '/home/jeon'
if(not os.path.isdir(home)): home = '/gem_home/jeon'
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



mode2 = 'nh2'
database2 = f"{home}/MissingSat/database/{mode2}"
iout2 = 797
repo2, rurmode2, dp2 = mode2repo(mode2)
snap2 = uri.RamsesSnapshot(repo2, iout2, mode=rurmode2)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']; nout=nout2[nout2 <= iout2]

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



stree2 = pklload(f"{database2}/stable_tree_new.pickle")



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
        size = 1.5001*dx
        indx = distx <= size
        cells = cells[indx]
        distx = np.abs(cells['x'] - icell['x'])
    disty = np.abs(cells['y'] - icell['y'])
    if(len(cells)>300000):
        size = 1.5001*dx
        indy = disty <= size
        cells = cells[indy]
        distx = np.abs(cells['x'] - icell['x'])
        disty = np.abs(cells['y'] - icell['y'])
    distz = np.abs(cells['z'] - icell['z'])
    if(len(cells)>300000):
        size = 1.5001*dx
        indz = distz <= size
        cells = cells[indz]
        distx = np.abs(cells['x'] - icell['x'])
        disty = np.abs(cells['y'] - icell['y'])
        distz = np.abs(cells['z'] - icell['z'])
    if(len(cells)>300000):
        size = 1.5001*dx
        indx = distx <= size
        indy = disty <= size
        indz = distz <= size
        cells = cells[indx&indy&indz]
        distx = np.abs(cells['x'] - icell['x'])
        disty = np.abs(cells['y'] - icell['y'])
        distz = np.abs(cells['z'] - icell['z'])
    dxs = 1 / 2**cells['level'] # <--- main bottleneck
    size = 1.0001*(dx + dxs)/2
    indx = distx <= size
    indy = disty <= size
    indz = distz <= size
    neighs = cells[indx&indy&indz]
    # remove itself
    itself = (neighs['x'] == icell['x'])&(neighs['y'] == icell['y'])&(neighs['z'] == icell['z'])
    neighs = neighs[~itself]
    
    
    # Find aligned cells
    samez = (neighs['z'] <= (icell['z'] + icell['dx']/1.999))&(neighs['z'] >= (icell['z'] - icell['dx']/1.99))
    samey = (neighs['y'] <= (icell['y'] + icell['dx']/1.999))&(neighs['y'] >= (icell['y'] - icell['dx']/1.99))
    samex = (neighs['x'] <= (icell['x'] + icell['dx']/1.999))&(neighs['x'] >= (icell['x'] - icell['dx']/1.999))

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

ndtype = [('id','i4'), ('lastid','i4'), ('dense', bool),
            ('trgv', 'f8'), ('c_s2', 'f8'), ('mach2', 'f8'), 
            ('alpha0', 'f8'), ('sigs', 'f8'), ('scrit', 'f8'), 
            ('sfr_ff', 'f8'), ('tstar', 'f8'), ('mstar', 'f8'), 
            ('nstar_fine', 'i8'), ('dt_fine', 'f8'), ('PoissMean_fine', 'f8'), ('nstar_corr_fine', 'f8'),
            ('nstar_iout', 'i8'), ('dt_iout', 'f8'), ('PoissMean_iout', 'f8'), ('nstar_corr_iout', 'f8')
            ]

def cell_calc(target, snap):
    global ndtype
    radii = 1
    snap.set_box_halo(target, radii, radius_name='r')
    snap.get_cell(nthread=ncpu, target_fields=['x','y','z','vx','vy','vz','rho','level','P'])
    maxdx = 1 / 2**np.min(snap.cell['level']-1)
    snap.box[0,0] -= maxdx; snap.box[0,1] += maxdx
    snap.box[1,0] -= maxdx; snap.box[1,1] += maxdx
    snap.box[2,0] -= maxdx; snap.box[2,1] += maxdx
    snap.get_cell(nthread=ncpu, target_fields=['x','y','z','vx','vy','vz','rho','level','P'])
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
    where = np.where(newcells['dense'])[0]
    for i, icell in zip(where, cells[newcells['dense']]):
        if(not newcells[i]['dense']): continue
        ls,rs,fs,bs,us,ds = get_nbor(icell, allcells, return_nbor=False)
        if(len(ls)==0 or len(rs)==0 or len(fs)==0 or len(bs)==0 or len(us)==0 or len(ds)==0):
            print(len(ls), len(rs), len(fs), len(bs), len(us), len(ds))
            newcells[i]['trgv'] = np.nan
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
        if(snap.mode=='nh'):
            alpha0 = 5*(trgv + c_s2)/(np.pi * factG * icell['rho'] * dx**2)
        else:
            alpha0 = 5*trgv/(np.pi * factG * icell['rho'] * dx**2)

        # Variance of the logarithmic PDF
        # sigs  = np.log(1.0 + 0.16*mach2)
        sigs  = np.log(1.0 + 0.16*trgv/c_s2)

        # The criitical densisity contrast by Padoan & Nordlund (2011)
        # scrit = np.log(0.067 / theta**2 * alpha0 * mach2)
        scrit = np.log(0.067 / theta**2 * alpha0 * trgv/c_s2)

        # sigs > scrit -> star formation
        if(snap.mode=='nh'):
            sfr_ff = e_cts/2*phi_t * np.exp(3/8*sigs) * (2 - erfc( (sigs-scrit)/np.sqrt(2*sigs) ))
        else:
            sfr_ff = eps_star/2*phi_t * np.exp(3/8*sigs) * (2 - erfc( (sigs-scrit)/np.sqrt(2*sigs) ))

        # The local free-fall time of the gas
        tstar     = 0.5427 * np.sqrt(1/( factG*icell['rho'] )) # 0.5427 = sqrt(3pi / 32)
        if(m_star < 0e0): mstar = n_star/(scale_nH*aexp**3)*vol_min*fstar_min
        else: mstar=m_star*mass_sph

        nstar_fine = 0; nstar_iout = 0
        dt_fine = dt_new[ilevel-1]
        PoissMean_fine = min( dt_fine*sfr_ff/tstar*mcell/mstar, 10)
        PoissMean_iout = min( dt_iout*sfr_ff/tstar*mcell/mstar, 10)
        nstar_fine = poissdev(localseed,PoissMean_fine,nstar_fine)
        # if(PoissMean_fine>0.4): print(f"{PoissMean_fine=}, {nstar_fine=}")
        nstar_iout = poissdev(localseed,PoissMean_iout,nstar_iout)
        # if(PoissMean_iout>0.4): print(f"{PoissMean_iout=}, {nstar_iout=}")
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
        # if(nstar_iout>0): print(newcells[i])
    snap.clear()
    return newcells

###########################################################
# Who is the target?
###########################################################
finddict = pklload(f"{database2}/SF/finddict.pickle")
outs = np.array(list(finddict.keys()))

###########################################################
# Snapshot
###########################################################
for iout in outs:
    snap = snap2s.get_snap(iout)
    ids = finddict[iout]
    targets = None
    for isubid in ids:
        branch = stree2[isubid]
        if(iout in branch['timestep']):
            tmp = branch[branch['timestep'] == iout]
            targets = tmp if targets is None else np.hstack((targets, tmp))
    if(targets is None):
        print(f"iout={iout}, No target")
        continue
    else:
        if(not os.path.isdir(f"{database2}/SF/{iout:05d}")):
            os.makedirs(f"{database2}/SF/{iout:05d}")
    for target in tqdm(targets, desc=f"[iout={iout}]"):
        tid = target['id']; iid = target['lastid']
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
        dt_iout = get_dt(snap, snap2s)


        newcells = cell_calc(target, snap)
        newcells['id'] = tid; newcells['lastid'] = iid
        pklsave(newcells, f"{database2}/SF/{iout:05d}/{iid:07d}.pickle")
        snap.clear()