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


LG = pklload(f"{database}/00_LocalGroup_final.pickle")
filedone = []
if(os.path.exists(f"{database}/00_LocalGroup_final_a_realsubs.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_a_realsubs.pickle")
    filedone.append('a')
if(os.path.exists(f"{database}/00_LocalGroup_final_b_realsats.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_b_realsats.pickle")
    filedone.append('b')
if(os.path.exists(f"{database}/00_LocalGroup_final_c_realreal.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_c_realreal.pickle")
    filedone.append('c')
if(os.path.exists(f"{database}/00_LocalGroup_final_d_realparking.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_d_realparking.pickle")
    filedone.append('d')
if(os.path.exists(f"{database}/00_LocalGroup_final_e_removeunreal.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_e_removeunreal.pickle")
    filedone.append('e')
if(os.path.exists(f"{database}/00_LocalGroup_final_f_newbox.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_f_newbox.pickle")
    filedone.append('f')
if(os.path.exists(f"{database}/00_LocalGroup_final_g_addcat.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_g_addcat.pickle")
    filedone.append('g')
if(os.path.exists(f"{database}/00_LocalGroup_final_h_addudg.pickle")):
    LG = pklload(f"{database}/00_LocalGroup_final_h_addudg.pickle")
    filedone.append('h')


virials = pklload(f"{database}/virial_radius_nh_1026.pickle")


dtype1 = hals.dtype
dtype2 = virials.dtype
dtype = np.dtype(dtype1.descr + dtype2.descr)
nhals = np.zeros(hals.shape, dtype=dtype)
for iname in tqdm(hals.dtype.names):
    nhals[iname] = hals[iname]
for iname in tqdm(virials.dtype.names):
    nhals[iname] = virials[iname]
print(dtype)


def point_in_sphere(point, sphere, rname='r', factor=1):
    dist = np.sqrt( (point['x'] - sphere['x'])**2 + (point['y'] - sphere['y'])**2 + (point['z'] - sphere['z'])**2 )
    # print(dist, sphere[rname]*factor)
    return dist < sphere[rname]*factor

def sphere_in_sphere(inner, outer, r1='r',r2='r', factor=1):
    dist = np.sqrt( (inner['x'] - outer['x'])**2 + (inner['y'] - outer['y'])**2 + (inner['z'] - outer['z'])**2 )
    # print(dist+inner[r1], outer[r2])
    return (dist+inner[r1]) < outer[r2]

def sphere_touch_sphere(sph1, sph2, r1='r',r2='r', factor=1):
    dist = np.sqrt( (sph1['x'] - sph2['x'])**2 + (sph1['y'] - sph2['y'])**2 + (sph1['z'] - sph2['z'])**2 )
    # print(dist, sph1[r1]+sph2[r2])
    return dist < (sph1[r1]+sph2[r2])*factor



pouts = snaps.iout_avail['iout'][snaps.iout_avail['age'] >= snap.age-1]
pouts = pouts[pouts < snap.iout][::-1]

def get_members(gal, galaxy=True):
    global members, snaps
    if(gal['timestep'] in members.keys()):
        if(gal['id'] in members[gal['timestep']].keys()):
            return members[gal['timestep']][gal['id']]
    else:
        members[gal['timestep']] = {}
    members[gal['timestep']][gal['id']] = uhmi.HaloMaker.read_member_part(snaps.get_snap(gal['timestep']), gal['id'], galaxy=galaxy, simple=True)
    return members[gal['timestep']][gal['id']]


sat_dtype = LG[1]['sats'].dtype
cols = [
    "Host", "Sat", "r50m", "r90m", "r50r", "r90r", 
    "SFR_mem", "u_mem", "g_mem", "r_mem", "i_mem", "z_mem", "metal_mem", "ager_mem", "t50_mem", "t90_mem"] 
category = ["r50m", "r90m", "r50r", "r90r", "rmax"]


for icate in category:
    cols = cols+[f"SFR_{icate}", f"u_{icate}", f"g_{icate}", f"r_{icate}", f"i_{icate}", f"z_{icate}", f"metal_{icate}", f"ager_{icate}", f"t50_{icate}", f"t90_{icate}", f"mgas_{icate}", f"mcold_{icate}", f"mdm_{icate}"]

def calc_rhalf(gal, part, weights, ratio=0.5):
    dist = distance(gal, part)
    argsort = np.argsort(dist)
    sw = np.cumsum(weights[argsort])
    sw /= sw[-1]
    return dist[argsort][np.argmin(np.abs(sw-ratio))]

def calc_rhalf_sorted(sorted_dist, sorted_weights, ratio=0.5):
    sw = np.cumsum(sorted_weights)
    sw /= sw[-1]
    return sorted_dist[np.argmin(np.abs(sw-ratio))]

def calc_tform(part, weights, ratio=0.5):
    age = part['age','Gyr']
    argsort = np.argsort(age)
    sw = np.cumsum(weights[argsort])
    sw /= sw[-1]
    return age[argsort][np.argmin(np.abs(sw-ratio))]


def make_gcatalog(Hostkey:int, table:np.void, istar:uri.Particle, idm:uri.Particle, icell:uri.Cell) -> np.void:
    global snap, sat_dtype
    result = np.zeros(1, dtype=sat_dtype)[0]
    for iname in result.dtype.names:
        if(iname in table.dtype.names):
            result[iname] = table[iname]
    result['Host'] = Hostkey

    satid = table['id']
    pid = uhmi.HaloMaker.read_member_part(snap, satid, galaxy=True, simple=True).flatten()
    ind = isin(np.abs(istar['id']), pid)
    ibox = np.array([
        [table['x']-table['r'], table['x']+table['r']],
        [table['y']-table['r'], table['y']+table['r']],
        [table['z']-table['r'], table['z']+table['r']]
                     ])
    assert np.sum(ind) == len(pid), f"{np.sum(ind)}, {len(pid)}\n{snap.box} <-> ({ibox})"
    mem_star = istar[ind]
    mem_dist = distance(table, mem_star)
    argsort = np.argsort(mem_dist)
    mem_dist = mem_dist[argsort]
    mem_mass = mem_star['m'][argsort]
    mem_rband = measure_luminosity(mem_star, 'SDSS_r', model='cb07')[argsort]

    result['r50m'] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.5)
    result['r90m'] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.9)
    result['r50r'] = calc_rhalf_sorted(mem_dist, mem_rband, ratio=0.5)
    result['r90r'] = calc_rhalf_sorted(mem_dist, mem_rband, ratio=0.9)

    ind = mem_star['age', 'Myr'] < 100
    result['SFR_mem'] = np.sum(mem_star['m', 'Msol'][ind]) / 1e8
    result['u_mem'] = measure_luminosity(mem_star, 'SDSS_u', model='cb07', total=True)
    result['g_mem'] = measure_luminosity(mem_star, 'SDSS_g', model='cb07', total=True)
    result['r_mem'] = measure_luminosity(mem_star, 'SDSS_r', model='cb07', total=True)
    result['i_mem'] = measure_luminosity(mem_star, 'SDSS_i', model='cb07', total=True)
    result['z_mem'] = measure_luminosity(mem_star, 'SDSS_z', model='cb07', total=True)
    result['metal_mem'] = np.sum(mem_star['metal'] * mem_star['m']) / np.sum(mem_star['m'])
    result['ager_mem'] = np.average(mem_star['age', 'Gyr'], weights=mem_rband)
    result['t50_mem'] = calc_tform(mem_star, mem_rband, ratio=0.5)
    result['t90_mem'] = calc_tform(mem_star, mem_rband, ratio=0.9)

    radiis = [result['r50m'], result['r90m'], result['r50r'], result['r90r'], table['r']]
    for radii, rname in zip(radiis, category):
        cut_star = cut_sphere(istar, table['x'], table['y'], table['z'], radii)
        rband = measure_luminosity(cut_star, 'SDSS_r', model='cb07')
        result[f'SFR_{rname}'] = np.sum(cut_star['m', 'Msol'][cut_star['age', 'Myr'] < 100]) / 1e8
        result[f'u_{rname}'] = measure_luminosity(cut_star, 'SDSS_u', model='cb07', total=True)
        result[f'g_{rname}'] = measure_luminosity(cut_star, 'SDSS_g', model='cb07', total=True)
        result[f'r_{rname}'] = measure_luminosity(cut_star, 'SDSS_r', model='cb07', total=True)
        result[f'i_{rname}'] = measure_luminosity(cut_star, 'SDSS_i', model='cb07', total=True)
        result[f'z_{rname}'] = measure_luminosity(cut_star, 'SDSS_z', model='cb07', total=True)
        result[f'metal_{rname}'] = np.sum(cut_star['metal'] * cut_star['m']) / np.sum(cut_star['m'])
        result[f'ager_{rname}'] = np.average(cut_star['age', 'Gyr'], weights=rband)
        result[f't50_{rname}'] = calc_tform(cut_star, rband, ratio=0.5)
        result[f't90_{rname}'] = calc_tform(cut_star, rband, ratio=0.9)
        cut_gas = cut_sphere(icell, table['x'], table['y'], table['z'], radii)
        cut_dm = cut_sphere(idm, table['x'], table['y'], table['z'], radii)
        result[f'mgas_{rname}'] = np.sum(cut_gas['m', 'Msol'])
        coldind = cut_gas['T', 'K'] < 1e4
        result[f'mcold_{rname}'] = np.sum(cut_gas['m', 'Msol'][coldind])
        result[f'mdm_{rname}'] = np.sum(cut_dm['m', 'Msol'])

    return result

sub_dtype = LG[1]['subs'].dtype
sub_dtype = np.dtype(sub_dtype.descr + [('r200kpc', '<f8'), ('m200', '<f8'), ('r200', '<f8')])
def make_hcatalog(Hostkey:int, table:np.void, istar:uri.Particle, idm:uri.Particle, icell:uri.Cell) -> np.void:
    global snap, sub_dtype, database
    assert 'm200' in table.dtype.names
    result = np.zeros(1, dtype=sub_dtype)[0]
    for iname in result.dtype.names:
        if(iname in table.dtype.names):
            result[iname] = table[iname]

    result['Host'] = Hostkey
    result['mdm'] = np.sum(idm['m','Msol'])
    result['mstar'] = np.sum(istar['m','Msol'])
    result['mcell'] = np.sum(icell['m','Msol'])
    mask = icell['T','K'] < 2e4
    if(True in mask):
        result['mcold'] = np.sum(icell['m','Msol'][mask])

    all_dist = distance(table, idm); argsort = np.argsort(all_dist)
    all_dist = all_dist[argsort]; all_mass = idm['m'][argsort]
    memdm = uhmi.HaloMaker.read_member_part(snap, table['id'], galaxy=False, target_fields=['x','y','z','m'])
    mem_dist = distance(table, memdm); argsort = np.argsort(mem_dist)
    mem_dist = mem_dist[argsort]; mem_mass = memdm['m'][argsort]

    result['r10_mem'] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.1)
    result['r50_mem'] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.5)
    result['r90_mem'] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.9)
    result['r10_max'] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.1)
    result['r50_max'] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.5)
    result['r90_max'] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.9)
    virdm, ind = cut_sphere(idm, table['x'], table['y'], table['z'], table['rvir'], return_index=True)
    all_dist = all_dist[ind]; all_mass = idm['m'][ind]
    result['r10_vir'] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.1)
    result['r50_vir'] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.5)
    result['r90_vir'] = calc_rhalf_sorted(all_dist, all_mass, ratio=0.9)
    
    return result
#-----------------------------------------------------------------------------
#
#
# New Box Calculation
#
#
#-----------------------------------------------------------------------------
def box_in_box(sbox, lbox):
    return (sbox[0,0] >= lbox[0,0]) & (sbox[0,1] <= lbox[0,1]) & (sbox[1,0] >= lbox[1,0]) & (sbox[1,1] <= lbox[1,1]) & (sbox[2,0] >= lbox[2,0]) & (sbox[2,1] <= lbox[2,1])
if(not 'f' in filedone):
    uri.timer.verbose=1
    for key in LG.keys():
        print(f"[{key:04d}]")
        BGG = LG[key]['BGG']
        oldsats = LG[key]['sats']
        oldsubs = LG[key]['subs']
        satids = LG[key]['Final_sats']
        subids = LG[key]['Final_subs']
        
        newsats = np.zeros(len(satids), dtype=sat_dtype)
        newsubs = np.zeros(len(subids), dtype=sub_dtype)
        newreal = LG[key]['Final_real']

        snap.box = LG[key]['box']

        # Box check
        x1 = BGG['x'] - 1.5*BGG['r200_code']; x2 = BGG['x'] + 1.5*BGG['r200_code']
        y1 = BGG['y'] - 1.5*BGG['r200_code']; y2 = BGG['y'] + 1.5*BGG['r200_code']
        z1 = BGG['z'] - 1.5*BGG['r200_code']; z2 = BGG['z'] + 1.5*BGG['r200_code']
        if(len(satids)>0):
            gtmp = gals[satids-1]
            x1 = min(x1, np.min(gtmp['x'] - 1.1*gtmp['r'])); x2 = max(x2, np.max(gtmp['x'] + 1.1*gtmp['r']))
            y1 = min(y1, np.min(gtmp['y'] - 1.1*gtmp['r'])); y2 = max(y2, np.max(gtmp['y'] + 1.1*gtmp['r']))
            z1 = min(z1, np.min(gtmp['z'] - 1.1*gtmp['r'])); z2 = max(z2, np.max(gtmp['z'] + 1.1*gtmp['r']))
        if(len(subids)>0):
            htmp = nhals[subids-1]
            x1 = min(x1, np.min(htmp['x'] - 1.1*htmp['r'])); x2 = max(x2, np.max(htmp['x'] + 1.1*htmp['r']))
            y1 = min(y1, np.min(htmp['y'] - 1.1*htmp['r'])); y2 = max(y2, np.max(htmp['y'] + 1.1*htmp['r']))
            z1 = min(z1, np.min(htmp['z'] - 1.1*htmp['r'])); z2 = max(z2, np.max(htmp['z'] + 1.1*htmp['r']))
        sbox = np.array([[x1,x2],[y1,y2],[z1,z2]])

        if(box_in_box(sbox, snap.box)):
            print("Box check passed -> Shrink")
            LG[key]['box'] = sbox
            star = pklload(f"{database}/parts/nh_star_{key:04d}.pickle")
            mask = (star['x'] >= sbox[0,0]) & (star['x'] <= sbox[0,1]) & (star['y'] >= sbox[1,0]) & (star['y'] <= sbox[1,1]) & (star['z'] >= sbox[2,0]) & (star['z'] <= sbox[2,1])
            pklsave(star[mask], f"{database}/parts/nh_star_{key:04d}.pickle", overwrite=True)
            dm = pklload(f"{database}/parts/nh_dm_{key:04d}.pickle")
            mask = (dm['x'] >= sbox[0,0]) & (dm['x'] <= sbox[0,1]) & (dm['y'] >= sbox[1,0]) & (dm['y'] <= sbox[1,1]) & (dm['z'] >= sbox[2,0]) & (dm['z'] <= sbox[2,1])
            pklsave(dm[mask], f"{database}/parts/nh_dm_{key:04d}.pickle", overwrite=True)
            cell = pklload(f"{database}/parts/nh_cell_{key:04d}.pickle")
            mask = (cell['x'] >= sbox[0,0]) & (cell['x'] <= sbox[0,1]) & (cell['y'] >= sbox[1,0]) & (cell['y'] <= sbox[1,1]) & (cell['z'] >= sbox[2,0]) & (cell['z'] <= sbox[2,1])
            pklsave(cell[mask], f"{database}/parts/nh_cell_{key:04d}.pickle", overwrite=True)
        else:
            print("Box check failed -> Recalc box")
            print(sbox)
            print(snap.box)
            snap.box = sbox; LG[key]['box']=sbox
            print("Load particles")
            snap.get_part(nthread=32)
            pklsave(snap.part['star'].table, f"{database}/parts/nh_star_{key:04d}.pickle", overwrite=True)
            pklsave(snap.part['dm'].table, f"{database}/parts/nh_dm_{key:04d}.pickle", overwrite=True)
            snap.part=None; snap.part_data=None; snap.cpulist_part=[]
            print("Load cells")
            snap.get_cell(nthread=32)
            pklsave(snap.cell.table, f"{database}/parts/nh_cell_{key:04d}.pickle", overwrite=True)
            snap.cell=None; snap.cell_data=None; snap.cpulist_cell=[]
        snap.clear()
        pklsave(LG, f"{database}/00_LocalGroup_final_f_newbox.pickle", overwrite=True)
    uri.timer.verbose=0
    pklsave(LG, f"{database}/00_LocalGroup_final_f_newbox.pickle", overwrite=True)
    filedone.append('f')

#-----------------------------------------------------------------------------
#
#
# Make Value Added Catalog
#
#
#-----------------------------------------------------------------------------
if(not 'g' in filedone):
    for key in LG.keys():
        print(f"[{key:04d}]")
        BGG = LG[key]['BGG']
        oldsats = LG[key]['sats']
        oldsubs = LG[key]['subs']
        satids = LG[key]['Final_sats']
        subids = LG[key]['Final_subs']
        
        newsats = np.zeros(len(satids), dtype=sat_dtype)
        newsubs = np.zeros(len(subids), dtype=sub_dtype)
        newreal = LG[key]['Final_real']

        snap.box = LG[key]['box']
        star = uri.Particle(pklload(f"{database}/parts/nh_star_{key:04d}.pickle"), snap)
        dm = uri.Particle(pklload(f"{database}/parts/nh_dm_{key:04d}.pickle"), snap)
        cell = uri.Cell(pklload(f"{database}/parts/nh_cell_{key:04d}.pickle"), snap)

        for i, satid in tqdm( enumerate(satids), total=len(satids), desc=f"G[{key:04}]"):
            if(satid in oldsats['id']): newsats[i] = oldsats[oldsats['id'] == satid][0]
            else:
                sat = gals[satid-1]
                istar = cut_sphere(star, sat['x'], sat['y'], sat['z'], 1.5*sat['r'])
                idm = cut_sphere(dm, sat['x'], sat['y'], sat['z'], 1.5*sat['r'])
                icell = cut_sphere(cell, sat['x'], sat['y'], sat['z'], 1.5*sat['r'])

                newcat = make_gcatalog(key, sat, istar, idm, icell)
                newsats[i] = newcat
        
        for i, subid in tqdm( enumerate(subids), total=len(subids), desc=f"H[{key:04}]" ):
            sub = nhals[subid-1]
            if(subid in oldsubs['id']):
                old = oldsubs[oldsubs['id'] == subid][0]
                for iname in newsubs.dtype.names:
                    if(iname in old.dtype.names): newsubs[i][iname] = old[iname]
                    elif(iname in sub.dtype.names): newsubs[i][iname] = sub[iname]
                    else: raise TypeError(f"`{iname}` is not in sub dtype")
            else:
                fname = f"{database}/parts/insub/nh_dm_{key:04d}_{sub['id']:07d}.pickle"
                if(os.path.exists(fname)): idm = uri.Particle(pklload(fname), snap)
                else:
                    idm = cut_sphere(dm, sub['x'], sub['y'], sub['z'], sub['r'])
                    pklsave(idm.table, fname, overwrite=True)
                fname = f"{database}/parts/insub/nh_star_{key:04d}_{sub['id']:07d}.pickle"
                if(os.path.exists(fname)): istar = uri.Particle(pklload(fname), snap)
                else:
                    istar = cut_sphere(star, sub['x'], sub['y'], sub['z'], sub['r'])
                    pklsave(istar.table, fname, overwrite=True)
                fname = f"{database}/parts/insub/nh_cell_{key:04d}_{sub['id']:07d}.pickle"
                if(os.path.exists(fname)): icell = uri.Cell(pklload(fname), snap)
                else:
                    icell = cut_sphere(cell, sub['x'], sub['y'], sub['z'], sub['r'])
                    pklsave(icell.table, fname, overwrite=True)

                newsub = make_hcatalog(key, sub, istar, idm, icell)
                newsubs[i] = newsub
        
        LG[key]['sats'] = newsats
        LG[key]['subs'] = newsubs
        LG[key]['real'] = newreal[newreal['state'] != 'ban']

    pklsave(LG, f"{database}/00_LocalGroup_final_g_addcat.pickle", overwrite=True)
    filedone.append('g')


#-----------------------------------------------------------------------------
#
#
# Find UDG (Ultra Diffuse Galaxy)
#
#
#-----------------------------------------------------------------------------
def make_udg(host:int, istar_vir:uri.Particle, idm_vir:uri.Particle, icell_vir:uri.Cell) -> np.void:
    global snap, sat_dtype
    result = np.zeros(1, dtype=sat_dtype)[0]
    names = ['timestep', 'aexp', 'Host']
    for name in names:
        result[name] = host[name]
    result['id'] = -host['id']
    result['fcontam'] = host['mcontam']/host['m']
    result['nparts'] = len(istar_vir)
    names = ['halo_id', 'halo_nparts', 'halo_level', 'halo_host', 'halo_hostsub', 'halo_x', 'halo_y', 'halo_z', 'halo_vx', 'halo_vy', 'halo_vz', 'halo_mvir', 'halo_rvir']
    for name in names:
        result[name] = host[name[5:]]
    # Default Info
    result['m'] = np.sum(istar_vir['m', 'Msol'])
    result['x'] = np.average(istar_vir['x'], weights=istar_vir['m'])
    result['y'] = np.average(istar_vir['y'], weights=istar_vir['m'])
    result['z'] = np.average(istar_vir['z'], weights=istar_vir['m'])
    result['vx'] = np.average(istar_vir['vx','km/s'], weights=istar_vir['m'])
    result['vy'] = np.average(istar_vir['vy','km/s'], weights=istar_vir['m'])
    result['vz'] = np.average(istar_vir['vz','km/s'], weights=istar_vir['m'])
    result['dist'] = distance(result, host)
    # Calculate the angular momentum
    dx = istar_vir['x'] - result['x']; dx /= snap.unit['Mpc']
    dy = istar_vir['y'] - result['y']; dy /= snap.unit['Mpc']
    dz = istar_vir['z'] - result['z']; dz /= snap.unit['Mpc']
    dpx = (istar_vir['vx','km/s'] - result['vx'])*istar_vir['m','Msol']
    dpy = (istar_vir['vy','km/s'] - result['vy'])*istar_vir['m','Msol']
    dpz = (istar_vir['vz','km/s'] - result['vz'])*istar_vir['m','Msol']
    result['Lx'] = np.sum(dpy*dz - dpz*dy)/1e11 # in 10**11 Msun * km/s * Mpc
    result['Ly'] = np.sum(dpz*dx - dpx*dz)/1e11
    result['Lz'] = np.sum(dpx*dy - dpy*dx)/1e11
    result['r'] = np.max(np.sqrt(dx**2 + dy**2 + dz**2)) * snap.unit['Mpc']
    # Value Added
    mem_dist = distance(result, istar_vir)
    argsort = np.argsort(mem_dist)
    mem_dist = mem_dist[argsort]
    mem_mass = istar_vir['m'][argsort]
    mem_rband = measure_luminosity(istar_vir, 'SDSS_r', model='cb07')[argsort]
    result['r50m'] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.5)
    result['r90m'] = calc_rhalf_sorted(mem_dist, mem_mass, ratio=0.9)
    result['r50r'] = calc_rhalf_sorted(mem_dist, mem_rband, ratio=0.5)
    result['r90r'] = calc_rhalf_sorted(mem_dist, mem_rband, ratio=0.9)
    # Use member particles
    ind = istar_vir['age', 'Myr'] < 100
    result['SFR_mem'] = np.sum(istar_vir['m', 'Msol'][ind]) / 1e8
    result['u_mem'] = measure_luminosity(istar_vir, 'SDSS_u', model='cb07', total=True)
    result['g_mem'] = measure_luminosity(istar_vir, 'SDSS_g', model='cb07', total=True)
    result['r_mem'] = measure_luminosity(istar_vir, 'SDSS_r', model='cb07', total=True)
    result['i_mem'] = measure_luminosity(istar_vir, 'SDSS_i', model='cb07', total=True)
    result['z_mem'] = measure_luminosity(istar_vir, 'SDSS_z', model='cb07', total=True)
    result['metal_mem'] = np.sum(istar_vir['metal'] * istar_vir['m']) / np.sum(istar_vir['m'])
    result['ager_mem'] = np.average(istar_vir['age', 'Gyr'], weights=rband)
    result['t50_mem'] = calc_tform(istar_vir, rband, ratio=0.5)
    result['t90_mem'] = calc_tform(istar_vir, rband, ratio=0.9)
    radiis = [result['r50m'], result['r90m'], result['r50r'], result['r90r'], result['r']]
    for radii, rname in zip(radiis, category):
        cut_star, cutind = cut_sphere(istar_vir, result['x'], result['y'], result['z'], radii, return_index=True)
        rband = measure_luminosity(cut_star, 'SDSS_r', model='cb07')
        result[f'SFR_{rname}'] = np.sum(cut_star['m', 'Msol'][cut_star['age', 'Myr'] < 100]) / 1e8
        result[f'u_{rname}'] = measure_luminosity(cut_star, 'SDSS_u', model='cb07', total=True)
        result[f'g_{rname}'] = measure_luminosity(cut_star, 'SDSS_g', model='cb07', total=True)
        result[f'r_{rname}'] = measure_luminosity(cut_star, 'SDSS_r', model='cb07', total=True)
        result[f'i_{rname}'] = measure_luminosity(cut_star, 'SDSS_i', model='cb07', total=True)
        result[f'z_{rname}'] = measure_luminosity(cut_star, 'SDSS_z', model='cb07', total=True)
        result[f'metal_{rname}'] = np.sum(cut_star['metal'] * cut_star['m']) / np.sum(cut_star['m'])
        result[f'ager_{rname}'] = np.average(cut_star['age', 'Gyr'], weights=rband)
        result[f't50_{rname}'] = calc_tform(cut_star, rband, ratio=0.5)
        result[f't90_{rname}'] = calc_tform(cut_star, rband, ratio=0.9)
        cut_gas = cut_sphere(icell, result['x'], result['y'], result['z'], radii)
        cut_dm = cut_sphere(idm, result['x'], result['y'], result['z'], radii)
        result[f'mgas_{rname}'] = np.sum(cut_gas['m', 'Msol'])
        coldind = cut_gas['T', 'K'] < 1e4
        result[f'mcold_{rname}'] = np.sum(cut_gas['m', 'Msol'][coldind])
        result[f'mdm_{rname}'] = np.sum(cut_dm['m', 'Msol'])

    return result
if(not 'h' in filedone):
    for key in LG.keys():
        print(f"[{key:04d}]")
        BGG = LG[key]['BGG']
        sats = LG[key]['sats']
        subs = LG[key]['subs']
        real = LG[key]['real']
        ind = subs['mstar'] <= 1e3; subs[ind]['mstar'] = 0
        ind = subs['mcell'] <= 1e3; subs[ind]['mcell'] = 0
        ind = subs['mcold'] <= 1e3; subs[ind]['mcold'] = 0
        LG[key]['subs'] = subs
        LG[key]['UDG'] = []
        igals = gals[~isin(gals['id'], sats['id'])]
        igals = igals[igals['id'] != BGG['id']]


        dinks = real[real['gid'] < 0]['hid']

        star = uri.Particle(pklload(f"{database}/parts/nh_star_{key:04d}.pickle"), snap)
        dm = uri.Particle(pklload(f"{database}/parts/nh_dm_{key:04d}.pickle"), snap)
        cell = uri.Cell(pklload(f"{database}/parts/nh_cell_{key:04d}.pickle"), snap)

        newsats = np.copy(sats); newreal = np.copy(real)
        for i,dink in enumerate(dinks):
            sub = subs[subs['id'] == dink][0]
            if(sub['mstar'] < 6e5): continue

            insides1 = sphere_touch_sphere(sub, igals, r1='rvir', factor=0.75)
            insides2 = igals['r'] < 2*sub['rvir']
            insides3 = ~isin(igals['id'], newreal['gid'])
            insides = insides1&insides2&insides3
            gcands = igals[insides]
            if(len(gcands)==0):
                # Truly No gals in this sub
                if(len(star)>300):
                    LG[key]['UDG'].append(dink)   # <-------------- Find UDG candidates!
                continue
            elif(len(gcands)==1):
                # One son sat in this sub
                gcand = gcands[0]
                print(f"\t[{sub['id']}] 1 son")
            else:
                # Multiple son sats in this sub
                dist = distance(gcands, sub)
                argmin = np.argmin(dist)
                gcand = gcands[argmin]
                print(f"\t[{sub['id']}] {len(gcands)} sons")

            # If you reach here, then you must find `gcand`
            if(gcand['id'] in real['gid']):
                # Connect `sub` and `gcand`
                assert gcand['id'] in sats['id']
                hwhere = np.where(newreal['hid'] == sub['id'])[0][0]
                gwhere = np.where(newreal['gid'] == gcand['id'])[0][0]
                newreal[hwhere]['gid'] = gcand['id']
                newreal[hwhere]['state'] = 'pair'
                newreal[gwhere]['state'] = 'ban'
            else:
                # Make new `gcand`
                istar = cut_sphere(star, gcand['x'], gcand['y'], gcand['z'], 1.5*gcand['r'])
                idm = cut_sphere(dm, gcand['x'], gcand['y'], gcand['z'], 1.5*gcand['r'])
                icell = cut_sphere(cell, gcand['x'], gcand['y'], gcand['z'], 1.5*gcand['r'])
                newcat = make_gcatalog(key, gcand, istar, idm, icell)
                newsats = np.append(newsats, newcat)

                hwhere = np.where(newreal['hid'] == sub['id'])[0][0]
                newreal[hwhere]['gid'] = gcand['id']
                newreal[hwhere]['state'] = 'pair'

        if(len(LG[key]['UDG']) > 0):
            for udghostid in LG[key]['UDG']:
                udghost = subs[subs['id'] == udghostid][0]

                istar_vir = cut_sphere(star, udghost['x'], udghost['y'], udghost['z'], udghost['rvir'])
                if(len(istar_vir)>=100):
                    idm_vir = cut_sphere(dm, udghost['x'], udghost['y'], udghost['z'], udghost['rvir'])
                    icell_vir = cut_sphere(cell, udghost['x'], udghost['y'], udghost['z'], udghost['rvir'])
                    udg = make_udg(udghost, istar_vir:uri.Particle, idm_vir:uri.Particle, icell_vir:uri.Cell)
                    
                    newsats = np.append(newsats, udg)
                    hwhere = np.where(newreal['hid'] == udghost['id'])[0][0]
                    newreal[hwhere]['gid'] = udg['id']
                    newreal[hwhere]['state'] = 'upair'
        
        LG[key]['sats'] = newsats
        LG[key]['real'] = newreal[newreal['state'] != 'ban']
    pklsave(LG, f"{database}/00_LocalGroup_final_h_addudg.pickle", overwrite=True)