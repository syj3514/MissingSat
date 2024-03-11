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
import matplotlib.patheffects as patheffects
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
from scipy.optimize import curve_fit
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


print("ex: $ python3 04_case_study.py [--iid 21025]")
import argparse
parser = argparse.ArgumentParser(description='(syj3514@yonsei.ac.kr)')
parser.add_argument("-i", "--iid", required=True, help='target halo id', type=int)
parser.add_argument("-s", "--skip", required=True, help='target halo id', type=int)
args = parser.parse_args()

target_id = args.iid
skip = args.skip

if(os.path.isdir(f"/home/jeon/MissingSat/database/nh/photo/{target_id:07d}")):
    overwrite = input(f"Target {target_id} already exists. Overwrite? (y/n): ")
    if(overwrite == 'y'):
        os.system(f"rm -rf /home/jeon/MissingSat/database/nh/photo/{target_id:07d}")
        os.system(f"mkdir /home/jeon/MissingSat/database/nh/photo/{target_id:07d}")
    else:
        exit()
else:
    os.system(f"mkdir /home/jeon/MissingSat/database/nh/photo/{target_id:07d}")



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

m1d, m2d = np.nanpercentile(np.hstack((dinks1['mvir'],dinks2['mvir'])), q=[2.5,97.5])
m1p, m2p = np.nanpercentile(np.hstack((pairs1['mvir'],pairs2['mvir'])), q=[2.5,97.5])
m1 = np.max([m1d,m1p]); m2 = np.min([m2d,m2p])
rtree1 = pklload(f"{database1}/reduced_tree.pickle")
rtree2 = pklload(f"{database2}/reduced_tree.pickle")


ptarget = allsubs1[allsubs1['id'] == target_id][0]
print(np.log10(ptarget['mdm_vir']))


from functools import lru_cache
def out2gyr(outs, snaps):
    table = snaps.iout_avail
    gyrs = np.zeros(len(outs))

    iout_table = table['iout']
    gyr_table = table['age']
    @lru_cache(None)
    def gyrfromout(iout):
        arg = iout_table==iout
        return gyr_table[arg][0]
    
    for i, iout in enumerate(outs):
        gyrs[i] = gyrfromout(iout)#table[table['iout']==iout][0]['age']
    return gyrs

branch = rtree1[ptarget['id']]
if(os.path.exists(f"{database1}/photo/{target_id:07d}/evolution.pickle")):
    evolution = pklload(f"{database1}/photo/{target_id:07d}/evolution.pickle")
else:
    evolution = {}
    evolution['nsn'] = np.zeros(len(branch))
    evolution['newstar'] = np.array([])
    evolution['insitu'] = np.zeros(len(branch))

np.seterr(divide = 'ignore') 
zoom = 1.5
dpi = 240
nstars = pklload(f"{database1}/nstar.pickle")
for iout in branch['timestep'][::-1]:
    if(iout <= skip): continue
    istep = np.where(nout1 == iout)[0][0]
    
    nh = snap1s.get_snap(iout)
    fig = plt.figure(figsize=(12,9), layout="constrained", dpi=150, facecolor='k')
    gs = gridspec.GridSpec(3, 4, figure=fig)



    ###### DM Map
    # Data Process
    nh = snap1s.get_snap(iout)
    target = branch[branch['timestep'] == iout][0]
    iarg = np.where(branch['timestep'] == iout)[0][0]
    nh.set_box_halo(target, zoom*8)
    extent = nh.box[:2].flatten()
    xbins = np.linspace(extent[0], extent[1], int(dpi*2)); ybins = np.linspace(extent[2], extent[3], int(dpi*2))
    part = nh.get_part(target_fields=['x','y','z','id','epoch','m'])
    indm = cut_sphere(part['dm'], target['x'], target['y'], target['z'], target['rvir'])
    # Drawing
    ax_dm = fig.add_subplot(gs[0:2, 0:3], facecolor='k', xticks=[], yticks=[])
    hist, xe, ye = np.histogram2d(part['dm']['x'], part['dm']['y'], bins=[xbins, ybins], weights=part['dm']['m'])
    dA = (xe[1]-xe[0])*(ye[1]-ye[0]); hist /= dA
    vmax = np.log10(np.max(hist))
    ycen = 0.5*(extent[2]+extent[3]); dy = extent[3]-extent[2]; y1 = extent[2]; y2 = ycen + dy/6
    ax_dm.imshow(np.log10(hist.T), origin='lower', extent=extent, cmap=cmr.tree, vmax=vmax, vmin=vmax-3)
    ax_dm.set_ylim(y1,y2)
    # Info
    cir = circle(target, rname='rvir'); ax_dm.add_patch(cir)
    tmpstr = fr"$\mathbf{{10^{{{np.log10(np.sum(indm['m','Msol'])):.2f}}}}}$" if len(indm)>0 else "0"
    ax_dm.text(target['x'], target['y']+target['rvir'], tmpstr, color='lime', ha='center', va='bottom', fontsize=14)
    ax_dm.text(0.95, 0.93, f"z={1/nh.aexp-1:.3f}", color='w', ha='right', va='top', fontsize=18, transform=ax_dm.transAxes)
    ax_dm.text(0.5,0.99, "Dark Matter", color='w', ha='center', va='top', fontsize=15, transform=ax_dm.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])
    add_scalebar(ax_dm, nh.unit_l, color='w', fontsize=12, top=True)



    ###### Zoom-in
    nh.set_box_halo(target, zoom)
    extent = nh.box[:2].flatten()
    xbins = np.linspace(extent[0], extent[1], dpi)
    ybins = np.linspace(extent[2], extent[3], dpi)
    part = nh.get_part(target_fields=['x','y','z','id','epoch','m'])
    cell = nh.get_cell(target_fields=['x','y','z','rho','P','metal'])
    buffer = 0.01
    length = 0.5



    ###### Star Map
    # Data Process
    ##################CHECK FROM HERE
    instar = cut_sphere(part['star'], target['x'], target['y'], target['z'], target['rvir'])
    newids = evolution['newstar'] # <- negative ids at previous
    nowids = part['star']['id']
    evolution['newstar'] = nowids[nowids < 0]
    oldids = nowids[nowids > 0]
    nsn = np.sum(isin(np.abs(oldids), np.abs(newids))) # negative before, but positive now
    evolution['nsn'][iarg] = nsn
    maxid = nstars[istep-1]
    newstar = instar[instar['id'] < -maxid]
    # dt = snap1s.iout_avail['age'][istep]-snap1s.iout_avail['age'][istep-1] # Gyr
    evolution['insitu'][iarg] = np.sum(newstar['m','Msol'])
    ##################CHECK BY HERE
    # Drawing
    ax_star = ax_dm.inset_axes([buffer*2/3, 1-length+buffer, (length-buffer*2)*2/3, (length-buffer*2)], xticks=[], yticks=[], facecolor='k')
    for spine in ax_star.spines.values(): spine.set_edgecolor('orange')
    if(len(part['star'])<100)and(len(part['star'])>0):
        ax_star.scatter(part['star']['x'], part['star']['y'], s=200/len(part['star']), ec='none', fc='orange',marker="*")
        ax_star.set_xlim(*nh.box[0]); ax_star.set_ylim(*nh.box[1])
    else:
        hist, xe, ye = np.histogram2d(part['star']['x'], part['star']['y'], bins=[xbins, ybins], weights=part['star']['m'])
        dA = (xe[1]-xe[0])*(ye[1]-ye[0]); hist /= dA
        vmax = np.log10(np.max(hist)/2)
        ax_star.imshow(np.log10(hist.T), origin='lower', extent=extent, cmap=cmr.sunburst, vmax=vmax, vmin=vmax-3)
    # Info
    cir = circle(target, rname='rvir'); ax_star.add_patch(cir)
    tmpstr = fr"$\mathbf{{10^{{{np.log10(np.sum(instar['m','Msol'])):.2f}}}}}$" if len(instar)>0 else "0"
    ax_star.text(target['x'], target['y']+target['rvir'], tmpstr, color='orange', ha='center', va='bottom', fontsize=12)
    ax_star.text(0.5,0.01, "Star", color='orange', ha='center', va='bottom', fontsize=15, transform=ax_star.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])


    length = 0.4
    ###### Gas Map
    # Data Process
    incell = cut_sphere(cell, target['x'], target['y'], target['z'], target['rvir'])
    # Drawing
    ax_gas = ax_dm.inset_axes([buffer*2/3, buffer, (length-buffer*2)*2/3, (length-buffer*2)], xticks=[], yticks=[], facecolor='k')
    for spine in ax_gas.spines.values(): spine.set_edgecolor('white')
    gasmap = painter.gasmap(cell, shape=dpi)
    vmax = np.log10(np.max(gasmap))
    ax_gas.imshow(np.log10(gasmap), origin='lower', extent=extent, cmap=cmr.neutral, vmax=vmax, vmin=vmax-3)
    # Info
    cir = circle(target, rname='rvir'); ax_gas.add_patch(cir)
    tmpstr = fr"$\mathbf{{10^{{{np.log10(np.sum(incell['m','Msol'])):.2f}}}}}$" if len(incell)>0 else "0"
    ax_gas.text(target['x'], target['y']+target['rvir'], tmpstr, color='whitesmoke', ha='center', va='bottom', fontsize=12)
    ax_gas.set_title(fr"$f_b={(np.sum(incell['m'])+np.sum(instar['m']))/(np.sum(instar['m']) + np.sum(indm['m']) + np.sum(incell['m']))*100:.1f}$%", color='w',fontsize=13)
    ax_gas.text(0.5,0.01, "Gas", color='w', ha='center', va='bottom', fontsize=15, transform=ax_gas.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])



    ###### Cold Map
    # Data Process
    cold = cell[cell['T','K'] < 1e4]
    incold = cut_sphere(cold, target['x'], target['y'], target['z'], target['rvir'])
    # Drawing
    ax_cold = ax_dm.inset_axes([(length+buffer)*2/3, buffer, (length-buffer*2)*2/3, (length-buffer*2)], xticks=[], yticks=[], facecolor='k')
    for spine in ax_cold.spines.values(): spine.set_edgecolor('dodgerblue')
    if(len(cold)>0):
        coldmap = painter.gasmap(cold, shape=dpi)
        vmax = np.log10(np.max(coldmap))
        ax_cold.imshow(np.log10(coldmap), origin='lower', extent=extent, cmap=cmr.arctic, vmax=vmax, vmin=vmax-3)
    else:
        ax_cold.set_xlim(*nh.box[0]); ax_cold.set_ylim(*nh.box[1])
    # Info
    cir = circle(target, rname='rvir')
    ax_cold.add_patch(cir)
    tmpstr = fr"$\mathbf{{10^{{{np.log10(np.sum(incold['m','Msol'])):.2f}}}}}$" if len(incold)>0 else "0"
    ax_cold.text(target['x'], target['y']+target['rvir'], tmpstr, color='dodgerblue', ha='center', va='bottom', fontsize=12)
    ax_cold.text(0.5,0.01, "Cold Gas", color='dodgerblue', ha='center', va='bottom', fontsize=15, transform=ax_cold.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])



    ###### Metal Map
    # Data Process
    metalmap = painter.gasmap(cell, mode='metal', shape=dpi)
    metalmap = gaussian_filter(metalmap, int(dpi/70))
    metalmap = np.log10(metalmap/0.0142)
    # Drawing
    ax_metal = ax_dm.inset_axes([(2*length+buffer)*2/3, buffer, (length-buffer*2)*2/3, (length-buffer*2)], xticks=[], yticks=[], facecolor='k')
    for spine in ax_metal.spines.values(): spine.set_edgecolor('salmon')
    metal_cmap = cmr.redshift
    ax_metal.imshow(metalmap, origin='lower', extent=extent, cmap=metal_cmap, vmax=0, vmin=-3)
    # Info
    cir = circle(target, rname='rvir'); ax_metal.add_patch(cir)
    ax_metal.text(0.5,0.01, "Metallicity", color='salmon', ha='center', va='bottom', fontsize=15, transform=ax_metal.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])
    # Colorbar
    cax = ax_metal.inset_axes([1.01, 0.00, 0.04, 1.0])
    norm = Normalize(vmin=-3,vmax=0); sm = plt.cm.ScalarMappable(cmap=metal_cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical', ticks=[-3,-2,-1,0])
    cbar.set_label(r"$log(Z/Z_\odot)$", color='w', fontsize=11)
    cbar.outline.set_edgecolor('w')
    for spine in cax.spines.values(): spine.set_edgecolor('w')
    cax.tick_params(axis='x', colors='w', labelsize=8); cax.tick_params(axis='y', colors='w')



    ###### DM density profile
    def single(logr, const, gamma): return gamma * logr + const

    r1 = 34*nh.unit['pc']; r2 = zoom*target['rvir']
    r1 = max(r1, np.min(cell['dx']))
    bins = np.logspace(np.log10(r1), np.log10(r2), 20+1)
    xs = 0.5*(bins[:-1]+bins[1:])/nh.unit['kpc'] # kpc
    sax1 = fig.add_subplot(gs[0,3]); sax1.yaxis.tick_right()
    sax1.grid(True, zorder=0, color='lightgray'); sax1.set_axisbelow(True)
    # Member Particles
    ys = np.zeros(len(xs))
    member = uhmi.HaloMaker.read_member_part(nh, target['id'], galaxy=False, target_fields=['id','x','y','z','m'])
    dist = distance(member, target)
    for i in range(len(xs)):
        ind = (dist > bins[i]) & (dist < bins[i+1])
        vol = (4/3*np.pi*(bins[i+1]**3-bins[i]**3)) # code**3
        vol /= nh.unit['kpc']**3
        ys[i] = np.sum(member[ind]['m','Msol']) / vol # Msol/kpc3
    sax1.plot(xs, ys, color='darkgray', lw=1.5, ls='-')
    # All Particles
    ys = np.zeros(len(xs))
    ws = np.zeros(len(xs))
    dist = distance(part['dm'], target)
    for i in range(len(xs)):
        ind = (dist > bins[i]) & (dist < bins[i+1])
        vol = (4/3*np.pi*(bins[i+1]**3-bins[i]**3)) # code**3
        vol /= nh.unit['kpc']**3
        ys[i] = np.sum(part['dm'][ind]['m','Msol']) / vol # Msol/kpc3
        ws[i] = 1 / np.sqrt(np.sum(part['dm'][ind]['m']))
    ind = ys>0
    ws /= np.sum(ws)
    sax1.plot(xs, ys, color='dimgrey', lw=1.5, ls=':')
    from scipy.optimize import curve_fit
    try: popt, pcov = curve_fit(single, np.log10(xs[ind]), np.log10(ys[ind]), p0=[6, -2], bounds=([-np.inf, -3.5], [np.inf, 0.5]), sigma=ws[ind], maxfev=5000, method='trf')
    except: popt = np.zeros(2)*np.nan; pcov = np.zeros(2)*np.nan
    sax1.plot(xs, 10**single(np.log10(xs), *popt), color='royalblue', lw=1.5, ls='-')
    sax1.axvline(target['rvir']/nh.unit['kpc'], color='dimgrey', linestyle='--', lw=1.5)
    sax1.text(0.05, 0.15, fr"$\gamma={popt[1]:.3f}$", ha='left', va='bottom', fontsize=12, transform=sax1.transAxes, color='royalblue')
    sax1.text(0.05, 0.05, fr"$R_{{vir}}={target['rvir']/nh.unit['kpc']:.2f}$ [kpc]", ha='left', va='bottom', fontsize=12, transform=sax1.transAxes)
    sax1.text(0.5, 0.99, "Density Profile", ha='center', va='top', fontsize=12, transform=sax1.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])
    sax1.set_xscale('log'); sax1.set_xlabel('r [kpc]', fontsize=12)
    sax1.set_yscale('log'); sax1.set_ylabel(r'$\rho_{DM}$ [M$_\odot$ kpc$^{-3}$]', fontsize=12); sax1.yaxis.set_label_position("right")
    ax_change_color(sax1, 'w')
    # evolution['gamma'][iarg] = popt[1]



    ###### Phase Diagram
    sax2 = fig.add_subplot(gs[1,3], facecolor='w'); sax2.yaxis.tick_right()
    sax2.grid(True, zorder=0, color='lightgray'); sax2.set_axisbelow(True)
    sax2.scatter(cell['rho','H/cc'], cell['T','K'], s=0.5, fc='darkgray', ec='none', zorder=1)
    sax2.scatter(cold['rho','H/cc'], cold['T','K'], s=1.0, fc='lightskyblue', ec='none', zorder=1)
    sax2.scatter(incell['rho','H/cc'], incell['T','K'], s=1, fc='dimgrey', ec='none', zorder=1)
    sax2.scatter(incold['rho','H/cc'], incold['T','K'], s=1.5, fc='royalblue', ec='none', zorder=1)
    sax2.set_xscale('log'); sax2.set_xlim(2e-6,1e3); sax2.set_xlabel(r'$\rho$ [H/cc]', fontsize=12)
    sax2.set_yscale('log'); sax2.set_ylim(5e1,9e6); sax2.set_ylabel('T [K]', fontsize=12); sax2.yaxis.set_label_position("right")
    sax2.text(0.05, 0.05, fr"$\rho_{{max}}={np.max(incell['rho','H/cc']):.2f}\,$[H/cc]", ha='left', va='bottom', fontsize=12, transform=sax2.transAxes)
    sax2.text(0.5, 0.99, "Gas Phase", ha='center', va='top', fontsize=12, transform=sax2.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])
    ax_change_color(sax2, 'w')
    # evolution['rhomax'][iarg] = np.max(incell['rho','H/cc'])



    ###### Metallicity Histogram
    sax3 = fig.add_subplot(gs[2,3]); sax3.yaxis.tick_right()
    sax3.hist(np.log10(cell['metal']/0.0142), bins=np.linspace(-3,0,20), histtype='step', color='darkgray')
    sax3.hist(np.log10(cold['metal']/0.0142), bins=np.linspace(-3,0,20), histtype='step', color='lightskyblue')
    sax3.hist(np.log10(incell['metal']/0.0142), bins=np.linspace(-3,0,20), histtype='stepfilled', color='dimgrey')
    sax3.hist(np.log10(incold['metal']/0.0142), bins=np.linspace(-3,0,20), histtype='stepfilled', color='royalblue')
    sax3.set_xlabel(r'$log(Z/Z_\odot)$', fontsize=12)
    sax3.set_yscale('log'); sax3.set_ylim(7,); sax3.set_ylabel('$N_{cell}$', fontsize=12); sax3.yaxis.set_label_position("right")
    sax3.text(0.5, 0.99, "Metallicity", ha='center', va='top', fontsize=12, transform=sax3.transAxes, family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])
    ax_change_color(sax3, 'w')
    # evolution['metal'][iarg] = np.mean(incell['metal']/0.0142)



    ###### Evolutionary Track
    lax = fig.add_subplot(gs[2,0:3], facecolor='k')
    lax.grid(True, zorder=0, color='dimgrey'); lax.set_axisbelow(True)
    # Mass Evolution
    tmp = branch[branch['timestep'] < iout]
    l1=lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mdm_vir'], color='lime', label='DM', lw=2)
    l2=lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mstar_vir'], color='orange', label='Star', lw=2)
    l3=lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mcell_vir'], color='w', label='Gas', lw=2)
    l4=lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mcold_vir'], color='dodgerblue', label='Cold', lw=2)
    tmp = branch[branch['timestep'] > iout]
    lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mdm_vir'], color='lime', ls=':', lw=0.5)
    # lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mstar_vir'], color='orange', ls=':', lw=0.5)
    # lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mcell_vir'], color='w', ls=':', lw=0.5)
    # lax.plot(out2gyr(tmp['timestep'], snap1s), tmp['mcold_vir'], color='dodgerblue', ls=':', lw=0.5)
    tmp = branch[branch['timestep'] == iout]
    lax.scatter(out2gyr(tmp['timestep'], snap1s), tmp['mdm_vir'], s=100,ec='none',fc='lime', marker='*')
    lax.scatter(out2gyr(tmp['timestep'], snap1s), tmp['mstar_vir'], s=100,ec='none',fc='orange', marker='*')
    lax.scatter(out2gyr(tmp['timestep'], snap1s), tmp['mcell_vir'], s=100,ec='none',fc='w', marker='*')
    lax.scatter(out2gyr(tmp['timestep'], snap1s), tmp['mcold_vir'], s=100,ec='none',fc='dodgerblue', marker='*')
    # Star Formation
    tmp = branch[branch['timestep'] <= iout]
    l5=lax.fill_between(out2gyr(tmp['timestep'], snap1s), 0, np.cumsum(evolution['insitu'][iarg:][::-1])[::-1], color='darkorange', alpha=0.3, label='in-situ star')
    
    lax2 = lax.twinx(); lax2.set_zorder(-1); lax.patch.set_visible(False)
    stemcontainer = lax2.stem(out2gyr(branch['timestep'], snap1s), evolution['nsn'], linefmt='yellow', markerfmt='yellow', basefmt='none', label='SN event')
    markerline, stemlines, baseline = stemcontainer
    markerline.set_markersize(1)
    markerline.set_marker('*')
    stemlines.set_linewidth(0.4)
    stemlines.set_zorder(-1)
    stemlines.set_alpha(0.5)
    lax2.set_ylim(0.001,)

    lines = l1+l2+l3+l4+[l5]+[stemcontainer]
    labels = [l.get_label() for l in lines]
    lax.legend(lines, labels, loc='lower center', frameon=False, labelcolor='w', ncol=3)
    lax.set_xlabel("Age of Univ. [Gyr]", fontsize=12), lax.set_xlim(0,13)
    lax.set_yscale('log'); lax.set_ylabel("M [M$_\odot$]", fontsize=12); lax.set_ylim(5e3,)
    lax.set_title("Evolutionary Track", fontsize=15, color='w', family='DejaVu Serif', path_effects=[patheffects.withSimplePatchShadow(offset=(0.5,-0.5))])
    ax_change_color(lax, 'w')
    ax_change_color(lax2, 'w')


    pklsave(evolution, f"{database1}/photo/{target_id:07d}/evolution.pickle", overwrite=True)
    plt.savefig(f"{database1}/photo/{target_id:07d}/{target_id:07d}_{iout:04d}.png", dpi=300, facecolor='k', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    nh.clear()
    print(f"{target_id}: {iout:04d} done")