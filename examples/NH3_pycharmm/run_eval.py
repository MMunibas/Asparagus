# Basics
import os
import shutil
import numpy as np

# Trajectory reader
import MDAnalysis
from MDAnalysis.analysis.distances import distance_array, capped_distance

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea

directories = [
    'nh3_cgenff_pycharmm', 
    'nh3_physnet_pycharmm', 
    'nh3_painn_pycharmm']
dcd_files = [
    'charmm_data/dyna.0.dcd',
    'charmm_data/dyna.0.dcd',
    'charmm_data/dyna.0.dcd']
psf_files = [
    'charmm_data/ammonia_water.psf',
    'charmm_data/ammonia_water.psf',
    'charmm_data/ammonia_water.psf']
rdf_pairs = [
    ('N1', 'OH2'),
    ('N1', 'OH2'),
    ('N1', 'OH2')]
rdf_label = r'N(NH$_3$)-O(H$_2$O)'
labels = [
    'CGenFF',
    'PhysNet',
    'PaiNN']
colors = [
    'red',
    'blue',
    'green']
linestyle = ['--', 'dashdot', 'dotted']
makerstyle = ['o', 's', '^']

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
column = [0.375, 0.10]
row = [0.7]

# Initialize plot figure
fig = plt.figure(figsize=figsize)

#======================================
# Plot NVE data
#======================================

# Initialize plot axis
axs = fig.add_axes([left, bottom, column[0], row[0]])

# Result lists
etot_list = []
eavg_list = []
estd_list = []
edif_list = []
time_list = []

# Auxiliary variables
edif_max = 0.0
time_max = 0.0

# Iterate over sample data
for ii, diri in enumerate(directories):
    
    # Read data
    with open(os.path.join(diri, 'nve_run2.dat'), 'r') as fdat:
        lines_dat = fdat.readlines()

    # Extract total energy
    etot = np.zeros(len(lines_dat), dtype=float)
    time = np.zeros(len(lines_dat), dtype=float)
    for il, line in enumerate(lines_dat):
        etot[il] = float(line.split()[3])
        time[il] = float(line.split()[2])

    # Evaluate energy sequence
    eavg = np.mean(etot)
    estd = np.std(etot)
    edif = np.max(etot) - np.min(etot)

    # Append results
    etot_list.append(etot)
    eavg_list.append(eavg)
    estd_list.append(estd)
    edif_list.append(edif)
    time_list.append(time)

    # Check for largest energy difference
    if edif_max < edif:
        edif_max = edif

# Plot sample data
for ii, diri in enumerate(directories):
    
    # Shift energy and time
    etot_plot = (
        etot_list[ii] - eavg_list[ii] 
        + 2*(len(directories) - 1)*edif_max - 2*ii*edif_max)
    time_plot = time_list[ii] - time_list[ii][0]

    # Plot energy sequence
    axs.plot(
        time_plot, etot_plot, color=colors[ii],
        label=labels[ii])

    # Check for larges time length
    if time_max < time_plot[-1]:
        time_max = time_plot[-1]

# Plot energy average line
for ii, diri in enumerate(directories):
    
    # Shift average energy
    eavg_plot = 2*(len(directories) - 1)*edif_max - 2*ii*edif_max
    
    # Plot average energy
    axs.plot(
        [0.0, time_max], [eavg_plot]*2, '--k')

# Plot options
axs.set_xlim(0.0, time_max)
axs.set_ylim(-edif_max, 2*(ii + 1)*edif_max + edif_max)

axs.set_xlabel(r"Time (ps)", fontweight='bold')
axs.set_ylabel(r"Total Energy (kcal/mol)", fontweight='bold')
axs.get_yaxis().set_label_coords(-0.12, 0.50)

axs.legend(loc='upper left', ncol=3)

tbox = TextArea("A", textprops=dict(color='k', fontsize=20))
anchored_tbox = AnchoredOffsetbox(
    loc='lower left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(-0.10, 1.00),
    bbox_transform=axs.transAxes, borderpad=0.)
axs.add_artist(anchored_tbox)

#======================================
# Plot g(r) - N(NH3)-O(H2O)
#======================================

# Initialize plot axis
axs = fig.add_axes([left + np.sum(column), bottom, column[0], row[0]])

# Binning options
rad_lim = (1.50, 8.00)
rad_num = int((rad_lim[1] - rad_lim[0])/0.1) + 1
rad_bins = np.linspace(rad_lim[0], rad_lim[1], num=rad_num)
rad_dist = rad_bins[1] - rad_bins[0]
rad_cent = rad_bins[:-1] + rad_dist/2.

# Result lists
rdfs_list = []

# Auxiliary variables
rdfs_max = 0.0

# Iterate over sample data
for ii, diri in enumerate(directories):
    
    # Open dcd file
    dcd = MDAnalysis.Universe(
        os.path.join(diri, psf_files[ii]),
        os.path.join(diri, dcd_files[ii]))

    # Get trajectory parameter
    Nframes = len(dcd.trajectory)
    Nskip = int(dcd.trajectory.skip_timestep)
    dt = np.round(
        float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip

    # Get atom types
    atoms = np.array([ai for ai in dcd._topology.names.values])
    idx_pair1 = np.array(
        [ia for ia, ai in enumerate(atoms) if rdf_pairs[ii][0] == ai])
    idx_pair2 = np.array(
        [ia for ia, ai in enumerate(atoms) if rdf_pairs[ii][1] == ai])
    Natoms = len(idx_pair1) + len(idx_pair2)

    # Prepare position and cell array to read from trajectory
    # to compute distances just once for all timesteps
    positions = np.zeros(
        (Nframes, Natoms, 3), dtype=np.float32)
    cell = np.zeros(
        (Nframes, 6), dtype=np.float32)

    # Read pair atom positions from trajectory
    for ic, tc in enumerate(dcd.trajectory):

        # Get positions
        positions[ic, :len(idx_pair1)] = tc._pos[idx_pair1]
        positions[ic, len(idx_pair1):] = tc._pos[idx_pair2]

        # Get cell information
        cell[ic] = tc._unitcell

    # Prepare distances histogram
    distances_hist = np.zeros_like(rad_cent, dtype=np.int64)

    # Iterate over atom pair positions
    for ic, posi in enumerate(positions):

        # Compute pair distances
        _, distances = capped_distance(
            posi[:len(idx_pair1)], posi[len(idx_pair1):],
            rad_lim[-1], box=cell[ic])

        # Compute histogram and add to distance histogram
        distances_hist[:] += np.histogram(distances, bins=rad_bins)[0]

    # Compute g(r) form distance histogram
    volume = 4./3.*np.pi*rad_lim[1]**3
    N = np.sum(distances_hist)
    if N > 0.0:
        rdfs = (volume/N)*(distances_hist/rad_dist)/(4.0*np.pi*rad_cent**2)
    else:
        rdfs = np.zeros_like(distances_hist)

    # Append result
    rdfs_list.append(rdfs)

    # Check for highest g(r) peak
    if rdfs_max < np.max(rdfs):
        rdfs_max = np.max(rdfs)

    # Plot g(r)
    axs.plot(
        rad_cent, rdfs, color=colors[ii],
        ls=linestyle[ii],
        label=labels[ii])

# Plot options
axs.set_xlim(*rad_lim)
axs.set_ylim(0.0, rdfs_max*1.1)

axs.set_xlabel(r"Atom Pair Distance ($\mathrm{\AA}$)", fontweight='bold')
axs.set_ylabel(r"$g(r)$", fontweight='bold')
axs.get_yaxis().set_label_coords(-0.12, 0.50)

axs.legend(loc='upper right', title=rdf_label)

tbox = TextArea("B", textprops=dict(color='k', fontsize=20))
anchored_tbox = AnchoredOffsetbox(
    loc='lower left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(-0.10, 1.00),
    bbox_transform=axs.transAxes, borderpad=0.)
axs.add_artist(anchored_tbox)

# Save plot
plt.savefig('eval_pycharmm.png', format='png', dpi=200)
plt.close()

plt.show()
