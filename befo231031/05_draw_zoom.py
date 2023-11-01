from tqdm import tqdm
import matplotlib.pyplot as plt # type: module
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
from icl_numba import large_isin, large_isind
from icl_draw import drawsnap, add_scalebar, addtext
import argparse, subprocess
from importlib import reload
import cmasher as cmr



mode = 'nh'
iout = 1026
repo, rurmode, dp = mode2repo(mode)
snap1 = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap1s = uri.TimeSeries(snap1)
snap1s.read_iout_avail()
nout1 = snap1s.iout_avail['iout']

mode = 'nh2'
iout = 797
repo, rurmode, dp = mode2repo(mode)
snap2 = uri.RamsesSnapshot(repo, iout, mode=rurmode)
snap2s = uri.TimeSeries(snap2)
snap2s.read_iout_avail()
nout2 = snap2s.iout_avail['iout']

result1s = pklload(f"./database/02_MWA1s.pickle")
result2s = pklload(f"./database/02_MWA2s.pickle")
prepared03 = os.path.exists(f"./database/03_MWA1s.pickle")
if(prepared03):
    result1s = pklload(f"./database/03_MWA1s.pickle")
    result2s = pklload(f"./database/03_MWA2s.pickle")
prepared04 = os.path.exists(f"./database/parts/nh_star_{result1s[-1]['id']:04}.pickle")
uri.timer.verbose=0
cell_cmap = drawer.make_cmap([[0,0,0], [2/255,2/255,25/255], [25/255, 25/255, 112/255], [25/255, 191/255, 255/255], [255/255, 255/255, 255/255]], position=[0, 0.25, 0.5,0.8,1])


dpi=1440
ifname = 0
for ir1, ir2 in tqdm(zip(result1s, result2s), total=len(result1s)):
    if(os.path.exists(f"./database/photo/Composite_NH{ir1['id']:05d}_edge.png") and os.path.exists(f"./database/photo/Composite_NH{ir1['id']:05d}_face.png")):
        print(f"Skipping `Composite_NH{ir1['id']:05d}.png`...")
        continue
    # if(ir1['id'] <= 1300): continue
    #-----------------------------------------------------------------
    #   Data Load
    #-----------------------------------------------------------------
    # Load NH1
    print("Loading NH1...")
    rrange_big = max(ir1['r'], 40*snap1.unit['kpc'])
    rrange = min(0.5 * ir1['r'], 25*snap1.unit['kpc'])
    snap1.set_box_halo(ir1, radius=rrange_big, use_halo_radius=False)
    if(prepared04):
        snap1.get_part(nthread=36, target_fields=['x','y','z','vx','vy','vz','m','id','epoch', 'metal'], pname='star')
    else:
        snap1.get_part(nthread=36, target_fields=['x','y','z','vx','vy','vz','m','id','epoch', 'metal'])
    snap1.get_cell(nthread=36, target_fields=['x','y','z','rho','level','cpu','P'])
    snap1.set_box_halo(ir1, radius=rrange, use_halo_radius=False)

    # Calculate L
    print("Calculating L...")
    star1 = snap1.part['star']
    rband1 = measure_luminosity(star1, 'SDSS_r')
    closests = star1[star1['age','Gyr'] < 0.5]
    if(len(closests) < 100): closests = star1
    dist = distance(closests, ir1)
    yr50_1 = np.median(dist)
    closests = closests[(dist >= 0.3*yr50_1) & (dist <= 3*yr50_1)]
    r_x = closests['x'] - ir1['x']; r_y = closests['y'] - ir1['y']; r_z = closests['z'] - ir1['z']
    mv_x = closests['m','Msol']*(closests['vx','km/s'] - ir1['vx'])
    mv_y = closests['m','Msol']*(closests['vy','km/s'] - ir1['vy'])
    mv_z = closests['m','Msol']*(closests['vz','km/s'] - ir1['vz'])
    Lx = np.sum( r_y*mv_z - r_z*mv_y ); Ly = np.sum( r_z*mv_x - r_x*mv_z ); Lz = np.sum( r_x*mv_y - r_y*mv_x )
    
    # Rotate Paticles
    print("Rotating Particles...")
    cx1,cy1,cz1 = ir1['x'], ir1['y'], ir1['z']
    nx,ny,nz = rot(Lx,Ly,Lz, star1['x']-cx1,star1['y']-cy1,star1['z']-cz1)
    star1['x'] = nx+cx1; star1['y'] = ny+cy1; star1['z'] = nz+cz1
    if(prepared04):
        dm1 = pklload(f"./database/parts/nh_dm_{ir1['id']:04d}.pickle")
        dm1 = uri.Particle(dm1, snap1)
    else:
        dm1 = snap1.part['dm']
    nx,ny,nz = rot(Lx,Ly,Lz, dm1['x']-cx1,dm1['y']-cy1,dm1['z']-cz1)
    dm1['x'] = nx+cx1; dm1['y'] = ny+cy1; dm1['z'] = nz+cz1
    cell1 = snap1.cell
    nx,ny,nz = rot(Lx,Ly,Lz, cell1['x']-cx1,cell1['y']-cy1,cell1['z']-cz1)
    cell1['x'] = nx+cx1; cell1['y'] = ny+cy1; cell1['z'] = nz+cz1
    extent1_face = np.concatenate( (snap1.box[0], snap1.box[1]) )
    extent1_edge = np.concatenate( (snap1.box[0], snap1.box[2]) )


    # Load NH2
    print("Loading NH2...")
    snap2.set_box_halo(ir2, radius=rrange_big, use_halo_radius=False)
    if(prepared04):
        snap2.get_part(nthread=36, target_fields=['x','y','z','vx','vy','vz','m','id','epoch', 'family', 'metal'], pname='star')
    else:
        snap2.get_part(nthread=36, target_fields=['x','y','z','vx','vy','vz','m','id','epoch', 'family', 'metal'])
    snap2.get_cell(nthread=36, target_fields=['x','y','z','rho','level','cpu','P'])
    snap2.set_box_halo(ir2, radius=rrange, use_halo_radius=False)

    # Calculate L
    print("Calculating L...")
    star2 = snap2.part['star']
    rband2 = measure_luminosity(star2, 'SDSS_r')
    closests = star2[star2['age','Gyr'] < 0.5]
    if(len(closests) < 100): closests = star2
    dist = distance(closests, ir2)
    yr50_2 = np.median(dist)
    closests = closests[(dist >= 0.3*yr50_2) & (dist <= 3*yr50_2)]
    r_x = closests['x'] - ir2['x']; r_y = closests['y'] - ir2['y']; r_z = closests['z'] - ir2['z']
    mv_x = closests['m','Msol']*(closests['vx','km/s'] - ir2['vx'])
    mv_y = closests['m','Msol']*(closests['vy','km/s'] - ir2['vy'])
    mv_z = closests['m','Msol']*(closests['vz','km/s'] - ir2['vz'])
    Lx = np.sum( r_y*mv_z - r_z*mv_y ); Ly = np.sum( r_z*mv_x - r_x*mv_z ); Lz = np.sum( r_x*mv_y - r_y*mv_x )
    
    # Rotate Paticles
    print("Rotating Particles...")
    cx2,cy2,cz2 = ir2['x'], ir2['y'], ir2['z']
    nx,ny,nz = rot(Lx,Ly,Lz, star2['x']-cx2,star2['y']-cy2,star2['z']-cz2)
    star2['x'] = nx+cx2; star2['y'] = ny+cy2; star2['z'] = nz+cz2
    if(prepared04):
        dm2 = pklload(f"./database/parts/nh2_dm_{ir2['id']:04d}.pickle")
        dm2 = uri.Particle(dm2, snap2)
    else:
        dm2 = snap2.part['dm']
    nx,ny,nz = rot(Lx,Ly,Lz, dm2['x']-cx2,dm2['y']-cy2,dm2['z']-cz2)
    dm2['x'] = nx+cx2; dm2['y'] = ny+cy2; dm2['z'] = nz+cz2
    cell2 = snap2.cell
    nx,ny,nz = rot(Lx,Ly,Lz, cell2['x']-cx2,cell2['y']-cy2,cell2['z']-cz2)
    cell2['x'] = nx+cx2; cell2['y'] = ny+cy2; cell2['z'] = nz+cz2
    extent2_face = np.concatenate( (snap2.box[0], snap2.box[1]) )
    extent2_edge = np.concatenate( (snap2.box[0], snap2.box[2]) )

    poss = ['x','y','z']
    for jth in range(2):    
        ith = 1-jth
        # ith = jth
        # ith=0: face-on, ith=1: edge-on
        extent1 = extent1_face if ith==0 else extent1_edge
        extent2 = extent2_face if ith==0 else extent2_edge
        proj = [0,1] if ith==0 else [0,2]
        file_prefix = 'face' if ith==0 else 'edge'
        if(os.path.exists(f"./database/photo/Composite_NH{ir1['id']:05d}_{file_prefix}.png")):
            print(f"Skipping `Composite_NH{ir1['id']:05d}_{file_prefix}.png`...")
            continue
        print(f"\nPlotting {file_prefix}-on...")
        ncon = 4 if ith==0 else 9
        qcon = 2 if ith==0 else 4.5
            
        #-----------------------------------------------------------------
        #   Mapping
        #-----------------------------------------------------------------
        print("Mapping...")
        # Stellar map
        #   Old pops
        oldmap1 = painter.partmap(star1[star1['age','Gyr'] > 0.3], box=snap1.box, shape=dpi, method='hist', proj=proj)
        oldmap2 = painter.partmap(star2[star2['age','Gyr'] > 0.3], box=snap2.box, shape=dpi, method='hist', proj=proj)
        #   Young pops
        youngmap1 = painter.partmap(star1[star1['age','Gyr'] < 0.3], box=snap1.box, shape=dpi, method='hist', proj=proj)
        youngmap2 = painter.partmap(star2[star2['age','Gyr'] < 0.3], box=snap2.box, shape=dpi, method='hist', proj=proj)
        #   Composite
        vmax_old = max( np.nanmax(oldmap1), np.nanmax(oldmap2) )*2
        vmax_young = max( np.nanmax(youngmap1), np.nanmax(youngmap2) )/2
        starmap1 = painter.composite_image([oldmap1, youngmap1], [plt.cm.afmhot, cell_cmap], vmaxs=[vmax_old, vmax_young], mode='screen', qscales=[4, 4])
        starmap2 = painter.composite_image([oldmap2, youngmap2], [plt.cm.afmhot, cell_cmap], vmaxs=[vmax_old, vmax_young], mode='screen', qscales=[4, 4])
        #   r-band
        lumimap1 = painter.partmap(star1, weights=rband1, box=snap1.box, shape=dpi, method='cic', proj=proj)
        lumimap2 = painter.partmap(star2, weights=rband2, box=snap2.box, shape=dpi, method='cic', proj=proj)

        # Gas map
        #   Total
        cellmap1 = painter.gasmap(cell1, box=snap1.box, shape=dpi, method='cic', proj=proj)
        cellmap2 = painter.gasmap(cell2, box=snap2.box, shape=dpi, method='cic', proj=proj)
        #   Cold gas
        ind1 = (cell1['T','K'] < 1e4)&(cell1['rho','H/cc'] > 0.1)
        if(not np.any(ind1)): ind1 = (cell1['T','K'] < 1e4)
        coldmap1 = painter.gasmap(cell1[ind1], box=snap1.box, shape=dpi, method='cic', proj=proj)
        ind2 = (cell2['T','K'] < 1e4)&(cell2['rho','H/cc'] > 0.1)
        if(not np.any(ind2)): ind2 = (cell2['T','K'] < 1e4)
        coldmap2 = painter.gasmap(cell2[ind2], box=snap2.box, shape=dpi, method='cic', proj=proj) 
        
        # # DM map
        dmmap1 = painter.partmap(dm1, box=snap1.box, shape=dpi, method='hist', proj=proj)
        dmmap2 = painter.partmap(dm2, box=snap2.box, shape=dpi, method='hist', proj=proj)
        #   Composite
        vmax_cell = max( np.nanmax(cellmap1),np.nanmax(cellmap2) )*2
        vmax_dm = max( np.nanmax(dmmap1),np.nanmax(dmmap2) )
        compmap1 = painter.composite_image([cellmap1, dmmap1], [cmr.savanna, plt.cm.bone], vmaxs=[vmax_cell, vmax_dm], mode='screen', qscales=[2.5, 3])
        compmap2 = painter.composite_image([cellmap2, dmmap2], [cmr.savanna, plt.cm.bone], vmaxs=[vmax_cell, vmax_dm], mode='screen', qscales=[2.5, 3])


        #-----------------------------------------------------------------
        #   Drawing
        #-----------------------------------------------------------------
        print("Drawing...")
        fig, axes = plt.subplots(3, 2, figsize=(10, 15), dpi=300)
        fig.set_facecolor("none")
        for ax in axes.flatten():
            ax.set_facecolor('k')
            ax.axis('equal')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

        titles = ["Composite\n(r-band + HI gas)", "Stellar Density\n(Young+Old)", "Gas & DM Density"]
        # [Composite]
        ax = axes[0,0]
        vmax = max( np.nanmax(lumimap1), np.nanmax(lumimap2) )
        painter.draw_image(lumimap1, extent=extent1, ax=ax, cmap=cmr.amber, qscale=4, vmax=vmax)
        ax.text(0.5, 0.13, f"NewHorizon\nat {ir1['timestep']} (z={1/snap1.aexp-1:.2f})", ha='center', va='top', color='w', fontsize=12, transform=ax.transAxes)
        vmax_cold = int(np.log10( max(np.nanmax(coldmap1),np.nanmax(coldmap2))/2 ))
        levels = np.linspace(vmax_cold-qcon,vmax_cold,ncon)
        rcoldmap1 = gaussian_filter(coldmap1, 2)
        rcoldmap1 = gaussian_filter(np.log10(rcoldmap1), 2.5)
        ax.contour(
                rcoldmap1, levels=levels, colors='whitesmoke', alpha=1, origin='lower', 
                extent=extent1, linewidths=0.2 )
        ax = axes[0,1]
        painter.draw_image(lumimap2, extent=extent2, ax=ax, cmap=cmr.amber, qscale=4, vmax=vmax)
        ax.text(0.5, 0.13, f"NewHorizon2\nat {ir2['timestep']} (z={1/snap2.aexp-1:.2f})", ha='center', va='top', color='w', fontsize=12, transform=ax.transAxes)
        rcoldmap2 = gaussian_filter(coldmap2, 2)
        rcoldmap2 = gaussian_filter(np.log10(rcoldmap2), 2.5)
        ax.contour(
                rcoldmap2, levels=levels, colors='whitesmoke', alpha=1, origin='lower', 
                extent=extent2, linewidths=0.2 )
        
        # [Stellar Density]
        ax = axes[1,0]
        vmax = max( np.nanmax(starmap1), np.nanmax(starmap2) )
        painter.draw_image(starmap1, extent=extent1, ax=ax, qscale=4, vmax=vmax)
        ax.text(0.05, 0.07+0.05*4, f"Galaxy:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*3, f"$\mathtt{{\ >\ ID: {ir1['id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        a = ir1['m'] / 10**int(np.log10(ir1['m']))
        b = int(np.log10(ir1['m']))
        ax.text(0.05, 0.05+0.05*2, fr"$\mathtt{{\ >\ M_{{*}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*1, fr"$\mathtt{{\ >\ SFR: {ir1['sfr']:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        if(prepared03):
            ax.text(0.05, 0.05+0.05*0, fr"$\mathtt{{\ >\ R_{{p}}: {ir1['rp']/snap1.unit['kpc']:.2f}\ kpc }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        ax = axes[1,1]
        painter.draw_image(starmap2, extent=extent2, ax=ax, qscale=4, vmax=vmax)
        ax.text(0.05, 0.07+0.05*4, f"Galaxy:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*3, f"$\mathtt{{\ >\ ID: {ir2['id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        a = ir2['m'] / 10**int(np.log10(ir2['m']))
        b = int(np.log10(ir2['m']))
        ax.text(0.05, 0.05+0.05*2, fr"$\mathtt{{\ >\ M_{{*}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*1, fr"$\mathtt{{\ >\ SFR: {ir2['sfr']:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        if(prepared03):
            ax.text(0.05, 0.05+0.05*0, fr"$\mathtt{{\ >\ R_{{p}}: {ir2['rp']/snap2.unit['kpc']:.2f}\ kpc }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)


        # [Gas+DM Density]
        ax = axes[2,0]
        vmax = max( np.nanmax(compmap1), np.nanmax(compmap2) )
        painter.draw_image(compmap1, extent=extent1, ax=ax, qscale=4, vmax=vmax)
        dm1 = cut_box(dm1, cx1, cy1, cz1, rrange)
        check = dm1['m'] > np.min(dm1['m'])*1.1
        iscontam1 = True in check
        if(iscontam1):
            print(f"In NH1, Contaminated DM particles are found. ({np.sum(check)})")
            ax.scatter(dm1[check][poss[proj[0]]], dm1[check][poss[proj[1]]], s=20, c='magenta', edgecolors='none', marker='*', alpha=1, zorder=2.9)
        ax.text(0.05, 0.07+0.05*4, f"Halo:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*3, f"$\mathtt{{\ >\ ID: {ir1['halo_id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        if(prepared03):
            a = ir1['m200'] / 10**int(np.log10(ir1['m200']))
            b = int(np.log10(ir1['m200']))
            ax.text(0.05, 0.05+0.05*2, fr"$\mathtt{{\ >\ M_{{200}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*1, fr"$\mathtt{{\ >\ SFR_{{tot}}: {ir1['sfr_tot']:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*0, fr"$\mathtt{{\ >\ f_{{cont}}: {ir1['fcontam']*100:.1f}\ \% }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        add_scalebar(ax, snap1.unit_l)
        ax = axes[2,1]
        painter.draw_image(compmap2, extent=extent2, ax=ax, qscale=4, vmax=vmax)
        dm2 = cut_box(dm2, cx2, cy2, cz2, rrange)
        check = dm2['m'] > np.min(dm2['m'])*1.1
        iscontam2 = True in check
        if(iscontam2):
            print(f"In NH2, Contaminated DM particles are found. ({np.sum(check)})")
            ax.scatter(dm2[check][poss[proj[0]]], dm2[check][poss[proj[1]]], s=20, c='magenta', edgecolors='none', marker='*', alpha=1, zorder=2.9)
        ax.text(0.05, 0.07+0.05*4, f"Halo:", ha='left', va='top', color='w', fontsize=11, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*3, f"$\mathtt{{\ >\ ID: {ir2['halo_id']} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        if(prepared03):
            a = ir2['m200'] / 10**int(np.log10(ir2['m200']))
            b = int(np.log10(ir2['m200']))
            ax.text(0.05, 0.05+0.05*2, fr"$\mathtt{{\ >\ M_{{200}}: {a:.2f}\times 10^{{{b}}}\ M_{{\odot}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*1, fr"$\mathtt{{\ >\ SFR_{{tot}}: {ir2['sfr_tot']:.2f}\ M_{{\odot}}\ yr^{{-1}} }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        ax.text(0.05, 0.05+0.05*0, fr"$\mathtt{{\ >\ f_{{cont}}: {ir2['fcontam']*100:.1f}\ \% }}$", ha='left', va='top', color='w', fontsize=9, transform=ax.transAxes)
        add_scalebar(ax, snap2.unit_l)

        
        
        plt.subplots_adjust(wspace=0, hspace=0)

        for i,title in enumerate(titles):
            xy_fig = axes[i,0].transAxes.transform((1,0.95))
            xy_fig = fig.transFigure.inverted().transform(xy_fig)
            fig.text(xy_fig[0], xy_fig[1], title, ha='center', va='top', color='w', fontsize=13, family='DejaVu Serif')
            if(i==0):
                xy_fig = axes[i,0].transAxes.transform((1,0.05))
                xy_fig = fig.transFigure.inverted().transform(xy_fig)
                fig.text(xy_fig[0], xy_fig[1], f"Match score\n{ir1['matchrate']:.2f}", ha='center', va='bottom', color='w', fontsize=12, family='monospace')


        plt.savefig(f"./database/photo/Composite_NH{ir1['id']:05d}_{file_prefix}.png", dpi=400, bbox_inches='tight')
        plt.close()

    snap1.clear()
    snap2.clear()