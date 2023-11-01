from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
import numpy as np
import os, glob
import time
import warnings

from rur.fortranfile import FortranFile
from rur import uri, uhmi, painter, drawer
from rur.sci.photometry import measure_luminosity
from scipy.ndimage import gaussian_filter
# from rur.sci.kinematics import f_getpot

from icl_IO import mode2repo, pklsave, pklload
from icl_tool import *
from icl_numba import large_isin, large_isind
from icl_draw import drawsnap, add_scalebar, addtext
import argparse, subprocess
uri.timer.verbose=0

###########################################################
#       Data Preparation
###########################################################
print("\n\n[Data Preparation]")
mode1 = 'nh'
fout1 = 1026
repo1,rurmode1,dp1 = mode2repo(mode1)
snap1 = uri.RamsesSnapshot(repo1, fout1, mode=rurmode1)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']
gal1s = uhmi.HaloMaker.load(snap1, galaxy=True, double_precision=dp1)
hal1s = uhmi.HaloMaker.load(snap1, galaxy=False, double_precision=dp1)

mode2 = 'nh2'
fout2 = 797
repo2,rurmode2,dp2 = mode2repo(mode2)
snap2 = uri.RamsesSnapshot(repo2, fout2, mode=rurmode2)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']
gal2s = uhmi.HaloMaker.load(snap2, galaxy=True, double_precision=dp2)
hal2s = uhmi.HaloMaker.load(snap2, galaxy=False, double_precision=dp2)


MWA1s = pklload(f"./database/02_MWA1s.pickle")
MWA2s = pklload(f"./database/02_MWA2s.pickle")





###########################################################
#       Virial Properties
###########################################################
print("\n\n[Virial Properties]")
count = 0
for MWA1, MWA2 in tqdm(zip(MWA1s, MWA2s), total=len(MWA1s)):
    count+=1
    ######################################################################
    #       NH I
    ######################################################################
    # Calculate Virial
    hal1 = hal1s[MWA1['halo_id']-1]
    r200_code = 1
    factor = 0.7
    while(r200_code > hal1['rvir']*factor):
        factor += 0.4
        snap1.set_box_halo(hal1, radius=factor, radius_name='rvir')
        snap1.get_part(nthread=24, target_fields=['x','y','z','m','id','epoch','metal'])
        snap1.get_cell(nthread=24, target_fields=['x','y','z','rho','level','cpu'])

        r200, m200, r200_code = calc_virial(MWA1['x'], MWA1['y'], MWA1['z'], snap1.part['star'], snap1.part['dm'], snap1.cell)
    MWA1['r200'] = r200
    MWA1['m200'] = m200
    MWA1['r200_code'] = r200_code

    # Calculate other properties
    star = snap1.part['star']
    gal_mem_ids = uhmi.HaloMaker.read_member_part(snap1, MWA1['id'], galaxy=True, simple=True)
    isin = large_isin(np.abs(star['id']), gal_mem_ids)
    gal_mem = star[isin]
    young_ind = np.where(gal_mem['age', 'Gyr'] < 0.1)[0]
    SFR = np.sum(gal_mem['m', 'Msol'][young_ind]) / 1e8 # Msol / yr
    MWA1['sfr'] = SFR

    instar = cut_sphere(star, MWA1['x'], MWA1['y'], MWA1['z'], r200_code)
    young_ind = np.where(instar['age', 'Gyr'] < 0.1)[0]
    SFR_tot = np.sum(instar['m', 'Msol'][young_ind]) / 1e8 # Msol / yr
    MWA1['sfr_tot'] = SFR_tot
    MWA1['m_star_200'] = np.sum(instar['m', 'Msol'])

    ingas = cut_sphere(snap1.cell, MWA1['x'], MWA1['y'], MWA1['z'], r200_code)
    MWA1['m_gas_200'] = np.sum(ingas['m', 'Msol']) # Msol

    dists = distance(instar, MWA1)
    rband = measure_luminosity(instar, 'SDSS_r')
    rp, _, _, _ = measure_petro_ratio(dists, rband)
    MWA1['rp'] = rp

    indm = cut_sphere(snap1.part['dm'], MWA1['x'], MWA1['y'], MWA1['z'], r200_code)
    minmass = np.min(snap1.part['dm']['m'])
    ind = indm['m'] > minmass*1.1
    if(True in ind):
        fcontam = np.sum(indm['m'][ind]) / np.sum(indm['m'])
    else:
        fcontam = 0
    MWA1['fcontam_200'] = fcontam


    # Drawing maps
    dpi = 1440
    starmap = painter.partmap(snap1.part['star'], box=snap1.box, shape=dpi, method='hist')
    vmax_star = np.nanmax(starmap)
    cellmap = painter.gasmap(snap1.cell, box=snap1.box, shape=dpi, minlvl=13, weights=snap1.cell['rho'])
    vmax_cell = np.nanmax(cellmap)
    dmmap1 = painter.partmap(snap1.part['dm'], box=snap1.box, shape=120, method='hist')
    vmax_dm = int(np.log10(np.nanmax(dmmap1)))
    dmmap = gaussian_filter(np.log10(dmmap1), 1.5)

    # Preparation
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    ax.set_facecolor('k')
    ax.axis('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


    # Background map
    vmin_dm = vmax_dm - 5

    smap = plt.cm.afmhot
    gmap = drawer.make_cmap([[0,0,0], [2/255,2/255,25/255], [25/255, 25/255, 112/255], [25/255, 191/255, 255/255], [255/255, 255/255, 255/255]], position=[0, 0.25, 0.5,0.8,1])
    combine = painter.composite_image([cellmap, starmap], [gmap, smap], mode='screen', qscales=[4.5, 3.7])
    ax.imshow(combine, origin='lower', extent=[snap1.box[0,0], snap1.box[0,1], snap1.box[1,0], snap1.box[1,1]])
    ax.contour(
        dmmap, levels=np.linspace(vmin_dm,vmax_dm,25), colors='grey', alpha=0.7, origin='lower', 
        extent=[snap1.box[0,0], snap1.box[0,1], snap1.box[1,0], snap1.box[1,1]], linewidths=0.25
        )


    # Circles
    cir_gal = plt.Circle((MWA1['x'], MWA1['y']), MWA1['r'], color='yellow', fill=False, lw=0.3)
    ax.add_patch(cir_gal)
    ax.text(MWA1['x'], MWA1['y']+MWA1['r'], "$R_{star, max}$", ha='center', va='bottom', color='yellow', fontsize=10)
    ax.text(MWA1['x'], MWA1['y']-MWA1['r'], f"{MWA1['r']/snap1.unit['kpc']:.1f} kpc", ha='center', va='top', color='yellow', fontsize=10)

    cir_hal1 = plt.Circle((hal1['x'], hal1['y']), hal1['rvir'], color='magenta', fill=False, lw=0.3)
    ax.add_patch(cir_hal1)
    ax.text(hal1['x'], hal1['y']+hal1['rvir'], "$R_{vir, HM}$", ha='center', va='bottom', color='magenta', fontsize=10)
    ax.text(hal1['x'], hal1['y']-hal1['rvir'], f"{hal1['rvir']/snap1.unit['kpc']:.1f} kpc", ha='center', va='top', color='magenta', fontsize=10)

    cir_vir = plt.Circle((MWA1['x'], MWA1['y']), r200_code, color='w', fill=False, lw=0.3)
    ax.add_patch(cir_vir)
    ax.text(MWA1['x'], MWA1['y']+r200_code, "$R_{200}$", ha='center', va='bottom', color='w', fontsize=10)
    ax.text(MWA1['x'], MWA1['y']-r200_code, f"{r200:.1f} kpc", ha='center', va='top', color='w', fontsize=10)


    # Infomation
    ax.text(0.05, 0.97, f"NewHorizon (z={1/snap1.aexp-1:.2f})", ha='left', va='top', color='w', fontsize=13, transform=ax.transAxes)

    ax.text(0.05, 0.39+0.04, f"Galaxy:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.34+0.04, f"$\mathtt{{\ >\ ID: {MWA1['id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    a = MWA1['m'] / 10**int(np.log10(MWA1['m']))
    b = int(np.log10(MWA1['m']))
    ax.text(0.05, 0.30+0.04, fr"$\mathtt{{\ >\ M_{{*}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.26+0.04, fr"$\mathtt{{\ >\ SFR: {SFR:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    
    ax.text(0.05, 0.18+0.04, f"Halo:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.13+0.04, f"$\mathtt{{\ >\ ID: {hal1['id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    a = m200 / 10**int(np.log10(m200))
    b = int(np.log10(m200))
    ax.text(0.05, 0.09+0.04, fr"$\mathtt{{\ >\ M_{{200}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.05+0.04, fr"$\mathtt{{\ >\ SFR_{{tot}}: {SFR_tot:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.05     , fr"$\mathtt{{\ >\ f_{{cont}}: {fcontam*100:.1f}\ \% }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)

    add_scalebar(ax, snap1.unit_l)

    plt.savefig(f"./database/photo/{count:02d}NH1_MWA_{MWA1['id']:05d}.png", dpi=400, bbox_inches='tight', pad_inches=0.1, facecolor='none')
    plt.close()
    snap1.clear()







    ######################################################################
    #       NH II
    ######################################################################
    # Calculate Virial
    hal2 = hal2s[MWA2['halo_id']-1]
    r200_code = 1
    factor = 0.7
    while(r200_code > hal2['rvir']*factor):
        factor += 0.4
        snap2.set_box_halo(hal2, radius=factor, radius_name='rvir')
        snap2.get_part(nthread=24, target_fields=['x','y','z','m','id','epoch','family','metal'])
        snap2.get_cell(nthread=24, target_fields=['x','y','z','rho','level','cpu'])

        r200, m200, r200_code = calc_virial(MWA2['x'], MWA2['y'], MWA2['z'], snap2.part['star'], snap2.part['dm'], snap2.cell)
    MWA2['r200'] = r200
    MWA2['m200'] = m200
    MWA2['r200_code'] = r200_code

    # Calculate other properties
    star = snap2.part['star']
    gal_mem_ids = uhmi.HaloMaker.read_member_part(snap2, MWA2['id'], galaxy=True, simple=True)
    isin = large_isin(np.abs(star['id']), gal_mem_ids)
    gal_mem = star[isin]
    young_ind = np.where(gal_mem['age', 'Gyr'] < 0.1)[0]
    SFR = np.sum(gal_mem['m', 'Msol'][young_ind]) / 1e8 # Msol / yr
    MWA2['sfr'] = SFR

    instar = cut_sphere(star, MWA2['x'], MWA2['y'], MWA2['z'], r200_code)
    young_ind = np.where(instar['age', 'Gyr'] < 0.1)[0]
    SFR_tot = np.sum(instar['m', 'Msol'][young_ind]) / 1e8 # Msol / yr
    MWA2['sfr_tot'] = SFR_tot
    MWA2['m_star_200'] = np.sum(instar['m', 'Msol'])

    ingas = cut_sphere(snap2.cell, MWA2['x'], MWA2['y'], MWA2['z'], r200_code)
    MWA2['m_gas_200'] = np.sum(ingas['m', 'Msol']) # Msol

    dists = distance(instar, MWA2)
    rband = measure_luminosity(instar, 'SDSS_r')
    rp, _, _, _ = measure_petro_ratio(dists, rband)
    MWA2['rp'] = rp

    indm = cut_sphere(snap2.part['dm'], MWA2['x'], MWA2['y'], MWA2['z'], r200_code)
    minmass = np.min(snap2.part['dm']['m'])
    ind = indm['m'] > minmass*1.1
    if(True in ind):
        fcontam = np.sum(indm['m'][ind]) / np.sum(indm['m'])
    else:
        fcontam = 0
    MWA2['fcontam_200'] = fcontam


    # Drawing maps
    dpi = 1440
    starmap = painter.partmap(snap2.part['star'], box=snap2.box, shape=dpi, method='hist')
    cellmap = painter.gasmap(snap2.cell, box=snap2.box, shape=dpi, minlvl=13, weights=snap2.cell['rho'])
    dmmap1 = painter.partmap(snap2.part['dm'], box=snap2.box, shape=120, method='hist')
    # vmax_dm = int(np.log10(np.nanmax(dmmap1)))
    dmmap = gaussian_filter(np.log10(dmmap1), 1.5)

    # Preparation
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)
    ax.set_facecolor('k')
    ax.axis('equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


    # Background map
    # vmin_dm = vmax_dm - 5

    smap = plt.cm.afmhot
    gmap = drawer.make_cmap([[0,0,0], [2/255,2/255,25/255], [25/255, 25/255, 112/255], [25/255, 191/255, 255/255], [255/255, 255/255, 255/255]], position=[0, 0.25, 0.5,0.8,1])
    combine = painter.composite_image([cellmap, starmap], [gmap, smap], mode='screen', qscales=[4.5, 3.7], vmaxs=[vmax_cell, vmax_star])
    ax.imshow(combine, origin='lower', extent=[snap2.box[0,0], snap2.box[0,1], snap2.box[1,0], snap2.box[1,1]])
    ax.contour(
        dmmap, levels=np.linspace(vmin_dm,vmax_dm,25), colors='grey', alpha=0.7, origin='lower', 
        extent=[snap2.box[0,0], snap2.box[0,1], snap2.box[1,0], snap2.box[1,1]], linewidths=0.25
        )


    # Circles
    cir_gal = plt.Circle((MWA2['x'], MWA2['y']), MWA2['r'], color='yellow', fill=False, lw=0.3)
    ax.add_patch(cir_gal)
    ax.text(MWA2['x'], MWA2['y']+MWA2['r'], "$R_{star, max}$", ha='center', va='bottom', color='yellow', fontsize=10)
    ax.text(MWA2['x'], MWA2['y']-MWA2['r'], f"{MWA2['r']/snap2.unit['kpc']:.1f} kpc", ha='center', va='top', color='yellow', fontsize=10)

    cir_hal2 = plt.Circle((hal2['x'], hal2['y']), hal2['rvir'], color='magenta', fill=False, lw=0.3)
    ax.add_patch(cir_hal2)
    ax.text(hal2['x'], hal2['y']+hal2['rvir'], "$R_{vir, HM}$", ha='center', va='bottom', color='magenta', fontsize=10)
    ax.text(hal2['x'], hal2['y']-hal2['rvir'], f"{hal2['rvir']/snap2.unit['kpc']:.1f} kpc", ha='center', va='top', color='magenta', fontsize=10)

    cir_vir = plt.Circle((MWA2['x'], MWA2['y']), r200_code, color='w', fill=False, lw=0.3)
    ax.add_patch(cir_vir)
    ax.text(MWA2['x'], MWA2['y']+r200_code, "$R_{200}$", ha='center', va='bottom', color='w', fontsize=10)
    ax.text(MWA2['x'], MWA2['y']-r200_code, f"{r200:.1f} kpc", ha='center', va='top', color='w', fontsize=10)


    # Infomation
    ax.text(0.05, 0.97, f"NH2 (z={1/snap2.aexp-1:.2f})", ha='left', va='top', color='w', fontsize=13, transform=ax.transAxes)

    ax.text(0.05, 0.39+0.04, f"Galaxy:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.34+0.04, f"$\mathtt{{\ >\ ID: {MWA2['id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    a = MWA2['m'] / 10**int(np.log10(MWA2['m']))
    b = int(np.log10(MWA2['m']))
    ax.text(0.05, 0.30+0.04, fr"$\mathtt{{\ >\ M_{{*}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.26+0.04, fr"$\mathtt{{\ >\ SFR: {SFR:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    
    ax.text(0.05, 0.18+0.04, f"Halo:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
    ax.text(0.05, 0.13+0.04, f"$\mathtt{{\ >\ ID: {hal2['id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    a = m200 / 10**int(np.log10(m200))
    b = int(np.log10(m200))
    ax.text(0.05, 0.09+0.04, fr"$\mathtt{{\ >\ M_{{200}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.05+0.04, fr"$\mathtt{{\ >\ SFR_{{tot}}: {SFR_tot:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
    ax.text(0.05, 0.05     , fr"$\mathtt{{\ >\ f_{{cont}}: {fcontam*100:.1f}\ \% }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)

    add_scalebar(ax, snap2.unit_l)

    plt.savefig(f"./database/photo/{count:02d}NH2_MWA_{MWA2['id']:05d}.png", dpi=400, bbox_inches='tight', pad_inches=0.1, facecolor='none')
    plt.close()
    snap2.clear()

pklsave(MWA1s, f"./database/03_MWA1s.pickle", overwrite=True)
snap1.clear()
pklsave(MWA2s, f"./database/03_MWA2s.pickle", overwrite=True)
snap1.clear()

print("\n\n[Done]")