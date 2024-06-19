import os
import shutil
import numpy as np

import ase
from ase import io
from ase.optimize import BFGS
from ase import vibrations
from ase.visualize import view

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

import sys
sys.path.insert(0, '/home/toepfer/Documents/Project_PhysNet3/Asparagus')

from asparagus import Asparagus
from asparagus.interface.orca_ase import ORCA

# Asparagus Models
configs = [
    'nh3_md_orca/nh3_md.json', 
    'nh3_meta_orca/nh3_meta.json', 
    'nh3_nmscan_orca/nh3_nms.json']
model_dir = [
    'nh3_md_orca/model_nh3_md', 
    'nh3_meta_orca/model_nh3_meta', 
    'nh3_nmscan_orca/model_nh3_nms']
ref_db = [
    'nh3_md_orca/nh3_md.db', 
    'nh3_meta_orca/nh3_meta.db', 
    'nh3_nmscan_orca/nh3_nms.db']
labels = [
    'MD Sampling',
    'Metadynamics Sampling',
    'Normal Mode Scanning']
colors = [
    'red',
    'blue',
    'green']

#======================================
# N-H Bond Distance Scan
#======================================

if False:

    # Plot preparation

    # Plot property: Fontsize
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE, weight='bold')
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Plot property: Figure size and arrangement
    figsize = (12, 4)
    sfig = float(figsize[0])/float(figsize[1])
    left = 0.08
    bottom = 0.15
    column = [0.25, 0.05]
    row = [column[0]*sfig]

    # Initialize plot figure
    fig = plt.figure(figsize=figsize)
    axs_list = []

    # Distance range
    drange = np.arange(0.7, 3.001, 0.02)
    dcenter = drange[1:] + (drange[1] - drange[0])/2.

    # Auxiliary parameters
    hist_scale = 4.0
    refmin = np.inf
    refmax = -np.inf
    panels = ['A', 'B', 'C']

    # Iterate over models
    for imodel, (config, mdir, db, label) in enumerate(
        zip(configs, model_dir, ref_db, labels)):
        
        print(f"Test model {mdir:s} trained by NH3 data from {label:s}.")
        
        # Read system
        system_model = io.read('../../data/nh3_c3v.xyz')
        system_ref = io.read('../../data/nh3_c3v.xyz')
        
        # Get calculators
        model = Asparagus(
            config=config,
            config_file='config_eval.json',
            model_directory=mdir)
        calc_model = model.get_ase_calculator()
        calc_ref = ORCA(
            charge=0,
            mult=1,
            orcasimpleinput='RI PBE D3BJ def2-SVP def2/J TightSCF',
            orcablocks='%pal nprocs 4 end',
            directory='orca')

        # Assign calculators
        system_model.calc = calc_model
        system_ref.calc = calc_ref

        # Optimize systems
        dyn = BFGS(system_model)
        dyn.run(fmax=0.0001)
        system_ref.set_positions(system_model.get_positions())
        dyn = BFGS(system_ref)
        dyn.run(fmax=0.0001)
        #view([system_model, system_ref])
        
        # Prepare N-H bond potential scan
        system_model_nhmin = system_model.get_distance(0, 1)
        system_ref_nhmin = system_ref.get_distance(0, 1)
        
        # Get energies for model minimum conformation
        energy_min_model = system_model.get_potential_energy()
        energy_min_model_ref = system_ref.get_potential_energy()
        
        # Scan N-H bond length
        energy_model = np.zeros_like(drange)
        energy_ref = np.zeros_like(drange)
        for idx, di in enumerate(drange):
            
            system_model.set_distance(0, 1, di, fix=0)
            energy_model[idx] = system_model.get_potential_energy()
            system_ref.set_distance(0, 1, di, fix=0)
            energy_ref[idx] = system_ref.get_potential_energy()
            
        # Read reference samples
        data = model.get_data_container(data_file=db)
        distances = []
        for data_i in data:
            distances.append(
                np.linalg.norm(
                    data_i['positions'][0] - data_i['positions'][1]))
            distances.append(
                np.linalg.norm(
                    data_i['positions'][0] - data_i['positions'][2]))
            distances.append(
                np.linalg.norm(
                    data_i['positions'][0] - data_i['positions'][3]))

        # Get histogram of sampled N-H bond distances
        histogram_distances, _ = np.histogram(distances, bins=drange)
        
        # Scale maximum to hist_scale = 4 (eV)
        histogram_distances = (
            histogram_distances.astype(float)
            / np.max(histogram_distances)
            * hist_scale)
        
        # Initialize plot axis
        axs = fig.add_axes(
            [left + imodel*np.sum(column), bottom, column[0], row[0]])
        
        # Plot distance histogram
        axs.bar(
            dcenter, histogram_distances, 
            width=(drange[1] - drange[0]), 
            edgecolor='grey', linewidth=1, fc='None')

        # Plot results
        axs.plot(
            drange, energy_model - energy_min_model, 
            color=colors[imodel], ls='-', 
            label='Model')
        axs.plot(
            drange, energy_ref - energy_min_model_ref, 
            color='black', ls='--', 
            label='PBE')
        
        # Plot options
        axs.set_xlim(drange[0], drange[-1])
        if refmin >= np.min(energy_ref - energy_min_model_ref):
            refmin = np.min(energy_ref - energy_min_model_ref)
        if refmax <= np.max(energy_ref - energy_min_model_ref):
            refmax = np.max(energy_ref - energy_min_model_ref)
        
        axs.set_title(label, fontweight='bold')
        axs.set_xlabel(r"N-H Distance ($\mathrm{\AA}$)", fontweight='bold')
        if not imodel:
            axs.set_ylabel("Potential (eV)", fontweight='bold')
            axs.get_yaxis().set_label_coords(-0.15, 0.50)
        
        axs.legend(loc='upper left')
        
        tbox = TextArea(
            f"{panels[imodel]:s}", 
            textprops=dict(color='k', fontsize=20))
        anchored_tbox = AnchoredOffsetbox(
            loc='lower left', child=tbox, pad=0., frameon=False,
            bbox_to_anchor=(-0.10, 1.00),
            bbox_transform=axs.transAxes, borderpad=0.)
        axs.add_artist(anchored_tbox)

        axs_list.append(axs)

    # Scale y axis
    for axs in axs_list:
        refdif = refmax - refmin
        axs.set_ylim(refmin - .1*refdif, refmax + .1*refdif)

    # Save plot
    plt.savefig('eval_nh3_distances.png', format='png', dpi=200)
    plt.close()

    # Remove temporary files
    if os.path.exists('orca'):
        shutil.rmtree('orca')
    if os.path.exists('config_eval.json'):
        os.remove('config_eval.json')

#======================================
# NH3 Harmonic Analysis
#======================================

if False:

    msg = "NH3 Harmonic Analysis\n"

    # Iterate over models
    for imodel, (config, mdir, db, label) in enumerate(
        zip(configs, model_dir, ref_db, labels)):
        
        print(f"Test model {mdir:s} trained by NH3 data from {label:s}.")
        
        # Read system
        system_model = io.read('../../data/nh3_c3v.xyz')
        system_ref = io.read('../../data/nh3_c3v.xyz')
        
        # Get calculators
        model = Asparagus(
            config=config,
            config_file='config_eval.json',
            model_directory=mdir)
        calc_model = model.get_ase_calculator()
        calc_ref = ORCA(
            charge=0,
            mult=1,
            orcasimpleinput='RI PBE D3BJ def2-SVP def2/J TightSCF',
            orcablocks='%pal nprocs 4 end',
            directory='orca')

        # Assign calculators
        system_model.calc = calc_model
        system_ref.calc = calc_ref

        # Optimize systems
        dyn = BFGS(system_model)
        dyn.run(fmax=0.0001)
        system_ref.set_positions(system_model.get_positions())
        dyn = BFGS(system_ref)
        dyn.run(fmax=0.0001)
        
        # Vibration
        vibrations_model = vibrations.Vibrations(
            system_model,
            name="vibfiles")
        vibrations_model.clean()
        vibrations_model.run()
        vibrations_model.summary()
        frequencies_model = vibrations_model.get_frequencies()
        vibrations_ref = vibrations.Vibrations(
            system_ref,
            name="vibfiles")
        vibrations_ref.clean()
        vibrations_ref.run()
        vibrations_ref.summary()
        frequencies_ref = vibrations_ref.get_frequencies()
        
        msg += label + "\n"
        msg += (
            f" {'Model Freq.':<15s} {'Ref Freq.':<15s} {'Difference':<15s}\n")
        for ifreq, (freq_model, freq_ref) in enumerate(
            zip(frequencies_model, frequencies_ref)
        ):
            error = freq_model - freq_ref
            msg += (
                f" {np.abs(freq_model):>15.2f}"
                + f" {np.abs(freq_ref):>15.2f}" 
                + f" {np.abs(error):>15.2f}\n")

    # Save result
    with open("eval_nh3_frequencies.txt", 'w') as f:
        f.write(msg)

    # Remove temporary files
    if os.path.exists('orca'):
        shutil.rmtree('orca')
    if os.path.exists('vibfiles'):
        shutil.rmtree('vibfiles')
    if os.path.exists('config_eval.json'):
        os.remove('config_eval.json')

#======================================
# NH3 Umbrella Potential
#======================================

if True:

    # Plot property: Fontsize
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    plt.rc('font', size=SMALL_SIZE, weight='bold')
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('axes', titlesize=MEDIUM_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)

    # Plot property: Figure size and arrangement
    figsize = (6, 6)
    sfig = float(figsize[0])/float(figsize[1])
    left = 0.15
    bottom = 0.15
    column = [0.75, 0.00]
    row = [column[0]*sfig]

    # Model style
    linestyle = ['--', 'dashdot', 'dotted']
    makerstyle = ['o', 's', '^']

    # Initialize plot figure
    fig = plt.figure(figsize=figsize)
    axs = fig.add_axes([left + 0.*np.sum(column), bottom, column[0], row[0]])

    # Umbrella angles
    angles = np.arange(52.0, 128.1, 2.0)

    # Auxiliary parameters
    refmin = np.inf
    refmax = -np.inf

    # Read system
    system_ref = io.read('../../data/nh3_c3v.xyz')

    # Get calculator
    calc_ref = ORCA(
        charge=0,
        mult=1,
        orcasimpleinput='RI PBE D3BJ def2-SVP def2/J TightSCF',
        orcablocks='%pal nprocs 4 end',
        directory='orca')

    # Assign calculators
    system_ref.calc = calc_ref

    # Optimize systems
    dyn = BFGS(system_ref)
    dyn.run(fmax=0.0001)
    energy_min_ref = system_ref.get_potential_energy()

    # Copy working system
    system_work = system_ref.copy()
    system_work.calc = calc_ref

    # Get Bisector vector
    vector = np.zeros(3, dtype=float)
    for ibond in range(1, 4):
        vector += system_work.positions[0] - system_work.positions[ibond]
    vector /= np.linalg.norm(vector)

    # Add dummy atom
    system_scan = system_work.copy()
    system_scan.append(
        ase.Atom('X', position=(system_scan.positions[0] + vector)))

    # Scan over umbrella angle
    potential = np.zeros_like(angles)
    for iangle, angle in enumerate(angles):
        for ibond in range(1, 4):
            system_scan.set_angle(4, 0, ibond, angle)
        system_work.set_positions(system_scan.positions[:4])
        potential[iangle] = system_work.get_potential_energy()

    axs.plot(
        angles, potential - energy_min_ref, 
        color='black', ls='-',
        label='PBE')

    refmin = np.min(potential - energy_min_ref)
    refmax = np.max(potential - energy_min_ref)

    # Iterate over models
    for imodel, (config, mdir, db, label) in enumerate(
        zip(configs, model_dir, ref_db, labels)):
        
        print(f"Test model {mdir:s} trained by NH3 data from {label:s}.")
        
        # Read system
        system_model = system_ref.copy()
        
        # Get calculators
        model = Asparagus(
            config=config,
            config_file='config_eval.json',
            model_directory=mdir)
        calc_model = model.get_ase_calculator()
        
        # Assign calculators
        system_model.calc = calc_model

        # Optimize systems
        #dyn = BFGS(system_model)
        #dyn.run(fmax=0.0001)
        energy_min_model = system_model.get_potential_energy()
        
        # Get Bisector vector
        vector = np.zeros(3, dtype=float)
        for ibond in range(1, 4):
            vector += system_model.positions[0] - system_model.positions[ibond]
        vector /= np.linalg.norm(vector)
        
        # Add dummy atom to scan atoms
        system_scan = system_model.copy()
        system_scan.append(
            ase.Atom('X', position=(system_scan.positions[0] + vector)))
        
        # Scan over umbrella angle
        potential = np.zeros_like(angles)
        for iangle, angle in enumerate(angles):
            for ibond in range(1, 4):
                system_scan.set_angle(4, 0, ibond, angle)
            system_model.set_positions(system_scan.positions[:4])
            potential[iangle] = system_model.get_potential_energy()

        axs.plot(
            angles, potential - energy_min_model, 
            color=colors[imodel], 
            #marker=makerstyle[imodel], markerfacecolor='None',
            ls=linestyle[imodel], #ls='None', 
            label=label)

    # Plot options
    axs.set_xlim(angles[0], angles[-1])
    refdif = refmax - refmin
    axs.set_ylim(refmin - .1*refdif, refmax + .1*refdif)

    #axs.set_title("Umbrella Potential", fontweight='bold')
    axs.set_xlabel(r"Umbrella Angle ($^\circ$)", fontweight='bold')
    axs.set_ylabel("Potential (eV)", fontweight='bold')
    axs.get_yaxis().set_label_coords(-0.15, 0.50)

    axs.legend(loc='upper center')

    # Save plot
    plt.savefig('eval_nh3_umbrella.png', format='png', dpi=200)
    plt.close()

    # Remove temporary files
    if os.path.exists('orca'):
        shutil.rmtree('orca')
    if os.path.exists('config_eval.json'):
        os.remove('config_eval.json')
