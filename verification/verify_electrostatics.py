import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from asparagus import module

import matplotlib.pyplot as plt

# Create model setup
distances_pair = torch.tensor(
    np.arange(0.1, 12.01, 0.1),
    dtype=torch.float64)
vectors_x = torch.tensor(
    [
        [[dist, 0., 0.], [-dist, 0., 0.]]
        for dist in distances_pair],
    dtype=torch.float64).reshape(-1, 3)
vectors_z = torch.tensor(
    [
        [[0., 0., dist], [0., 0., -dist]]
        for dist in distances_pair],
    dtype=torch.float64).reshape(-1, 3)
rsqrt2 = 1./np.sqrt(2.)
vectors_yz = torch.tensor(
    [
        [[0., dist*rsqrt2, dist*rsqrt2], [0., -dist*rsqrt2, -dist*rsqrt2]] 
        for dist in distances_pair],
    dtype=torch.float64).reshape(-1, 3)
Ndist = len(distances_pair)


ci = 0.5
cj = -0.5
di = 0.1

properties = {}
properties['atomic_charges'] = torch.tensor(
    [ci, cj]*Ndist, dtype=torch.float64)

properties_x_x = properties.copy()
properties_x_x['atomic_dipoles'] = torch.tensor(
    [[di, 0.0, 0.0], [di, 0.0, 0.0]]*Ndist, dtype=torch.float64)

properties_x_mx = properties.copy()
properties_x_mx['atomic_dipoles'] = torch.tensor(
    [[di, 0.0, 0.0], [-di, 0.0, 0.0]]*Ndist, dtype=torch.float64)

properties_x_y = properties.copy()
properties_x_y['atomic_dipoles'] = torch.tensor(
    [[di, 0.0, 0.0], [0.0, di, 0.0]]*Ndist, dtype=torch.float64)

properties_x_z = properties.copy()
properties_x_z['atomic_dipoles'] = torch.tensor(
    [[di, 0.0, 0.0], [0.0, 0.0, di]]*Ndist, dtype=torch.float64)

properties_x_yz = properties.copy()
properties_x_yz['atomic_dipoles'] = torch.tensor(
    [[di, 0.0, 0.0], [0.0, di/rsqrt2, di/rsqrt2]]*Ndist, dtype=torch.float64)


distances = distances_pair.repeat_interleave(2)
idx_i = torch.tensor(np.arange(Ndist*2), dtype=torch.int64)
idx_j = torch.tensor(
    np.arange(Ndist*2).reshape(Ndist, 2)[:, ::-1].reshape(-1),
    dtype=torch.int64)

long_cutoff = 10.0
short_cutoff = 6.0

# Initialize electrostatic scheme
V_fn = module.PC_damped_electrostatics(
    long_cutoff,
    short_cutoff,
    'cpu',
    torch.float64,
    truncation='None'
    )

# Compute electrostatics
V_C = V_fn(
    properties,
    distances,
    idx_i,
    idx_j)

# Initialize electrostatic scheme
V_fn = module.PC_damped_electrostatics(
    long_cutoff,
    short_cutoff,
    'cpu',
    torch.float64,
    truncation='potential'
    )

# Compute electrostatics
V_ps = V_fn(
    properties,
    distances,
    idx_i,
    idx_j)

# Initialize electrostatic scheme
V_fn = module.PC_damped_electrostatics(
    long_cutoff,
    short_cutoff,
    'cpu',
    torch.float64,
    truncation='forces'
    )

# Compute electrostatics
V_fs = V_fn(
    properties,
    distances,
    idx_i,
    idx_j)

# Potential energy is the sum of the atom pair
V_C = torch.sum(V_C.reshape(Ndist, 2), axis=1)
V_ps = torch.sum(V_ps.reshape(Ndist, 2), axis=1)
V_fs = torch.sum(V_fs.reshape(Ndist, 2), axis=1)









# Initialize electrostatic scheme including atomic dipoles
V_fn = module.PC_Dipole_damped_electrostatics(
    long_cutoff,
    short_cutoff,
    'cpu',
    torch.float64,
    truncation='None'
    )

# Compute electrostatics
Vdip_C_x_x_x = V_fn(
    properties_x_x,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_C_x_mx_x = V_fn(
    properties_x_mx,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_C_x_x_z = V_fn(
    properties_x_x,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_C_x_mx_z = V_fn(
    properties_x_mx,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_C_x_y_x = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_C_x_y_z = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_C_x_y_yz = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_yz)
Vdip_C_x_yz_yz = V_fn(
    properties_x_yz,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_yz)


# Initialize electrostatic scheme including atomic dipoles
V_fn = module.PC_Dipole_damped_electrostatics(
    long_cutoff,
    short_cutoff,
    'cpu',
    torch.float64,
    truncation='Potential'
    )

# Compute electrostatics
Vdip_ps_x_x_x = V_fn(
    properties_x_x,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_ps_x_mx_x = V_fn(
    properties_x_mx,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_ps_x_x_z = V_fn(
    properties_x_x,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_ps_x_mx_z = V_fn(
    properties_x_mx,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_ps_x_y_x = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_ps_x_y_z = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_ps_x_y_yz = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_yz)
Vdip_ps_x_yz_yz = V_fn(
    properties_x_yz,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_yz)



# Initialize electrostatic scheme including atomic dipoles
V_fn = module.PC_Dipole_damped_electrostatics(
    long_cutoff,
    short_cutoff,
    'cpu',
    torch.float64,
    truncation='forces'
    )

# Compute electrostatics
Vdip_fs_x_x_x = V_fn(
    properties_x_x,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_fs_x_mx_x = V_fn(
    properties_x_mx,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_fs_x_x_z = V_fn(
    properties_x_x,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_fs_x_mx_z = V_fn(
    properties_x_mx,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_fs_x_y_x = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_x)
Vdip_fs_x_y_z = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_z)
Vdip_fs_x_y_yz = V_fn(
    properties_x_y,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_yz)
Vdip_fs_x_yz_yz = V_fn(
    properties_x_yz,
    distances,
    idx_i,
    idx_j,
    vectors=vectors_yz)


# Potential energy is twice i<->j contribution (j<->i not 'computed')
Vdip_C_x_x_x = torch.sum(Vdip_C_x_x_x.reshape(Ndist, 2), axis=1)
Vdip_C_x_mx_x = torch.sum(Vdip_C_x_mx_x.reshape(Ndist, 2), axis=1)
Vdip_C_x_x_z = torch.sum(Vdip_C_x_x_z.reshape(Ndist, 2), axis=1)
Vdip_C_x_mx_z = torch.sum(Vdip_C_x_mx_z.reshape(Ndist, 2), axis=1)
Vdip_C_x_y_x = torch.sum(Vdip_C_x_y_x.reshape(Ndist, 2), axis=1)
Vdip_C_x_y_z = torch.sum(Vdip_C_x_y_z.reshape(Ndist, 2), axis=1)
Vdip_C_x_y_yz = torch.sum(Vdip_C_x_y_yz.reshape(Ndist, 2), axis=1)
Vdip_C_x_yz_yz = torch.sum(Vdip_C_x_yz_yz.reshape(Ndist, 2), axis=1)

Vdip_ps_x_x_x = torch.sum(Vdip_ps_x_x_x.reshape(Ndist, 2), axis=1)
Vdip_ps_x_mx_x = torch.sum(Vdip_ps_x_mx_x.reshape(Ndist, 2), axis=1)
Vdip_ps_x_x_z = torch.sum(Vdip_ps_x_x_z.reshape(Ndist, 2), axis=1)
Vdip_ps_x_mx_z = torch.sum(Vdip_ps_x_mx_z.reshape(Ndist, 2), axis=1)
Vdip_ps_x_y_x = torch.sum(Vdip_ps_x_y_x.reshape(Ndist, 2), axis=1)
Vdip_ps_x_y_z = torch.sum(Vdip_ps_x_y_z.reshape(Ndist, 2), axis=1)
Vdip_ps_x_y_yz = torch.sum(Vdip_ps_x_y_yz.reshape(Ndist, 2), axis=1)
Vdip_ps_x_yz_yz = torch.sum(Vdip_ps_x_yz_yz.reshape(Ndist, 2), axis=1)

Vdip_fs_x_x_x = torch.sum(Vdip_fs_x_x_x.reshape(Ndist, 2), axis=1)
Vdip_fs_x_mx_x = torch.sum(Vdip_fs_x_mx_x.reshape(Ndist, 2), axis=1)
Vdip_fs_x_x_z = torch.sum(Vdip_fs_x_x_z.reshape(Ndist, 2), axis=1)
Vdip_fs_x_mx_z = torch.sum(Vdip_fs_x_mx_z.reshape(Ndist, 2), axis=1)
Vdip_fs_x_y_x = torch.sum(Vdip_fs_x_y_x.reshape(Ndist, 2), axis=1)
Vdip_fs_x_y_z = torch.sum(Vdip_fs_x_y_z.reshape(Ndist, 2), axis=1)
Vdip_fs_x_y_yz = torch.sum(Vdip_fs_x_y_yz.reshape(Ndist, 2), axis=1)
Vdip_fs_x_yz_yz = torch.sum(Vdip_fs_x_yz_yz.reshape(Ndist, 2), axis=1)






# Plot charge-charge electrostatics
plt.plot(distances_pair, V_C, '-', mfc='None', label='None')
plt.plot(distances_pair, V_ps, '--', mfc='None', label='Potential Shift')
plt.plot(distances_pair, V_fs, ':', mfc='None', label='Force Shift')

plt.ylim(-6.0, 1.0)

plt.title(
    'Damped Electrostatic Potential of diatom with charges '
    + r'$q_i$=' + f'{ci:.1f}, ' + r'$q_j$=' + f'{cj:.1f}\n' 
    + 'and different truncation methods in cutoff range '
    + f'{short_cutoff:.1f} to {long_cutoff:.1f}')

plt.xlabel('Distance (Ang)')
plt.ylabel('Damped Electrostatic Potential (eV)')

plt.legend(loc='lower right', title='Truncation method')
plt.show()

plt.close()



# Plot (charge+dipole)-(charge+dipole) electrostatics, no truncation
plt.plot(
    distances_pair, V_C, 'o-', mfc='None',
    label=r'None ($\mu_i$=0,$\mu_j$=0,$\vec{r}_{ij}$=arb.)')
plt.plot(
    distances_pair, Vdip_C_x_x_x, 'x-', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=x,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_C_x_mx_x, '+-', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=-x,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_C_x_x_z, 's--', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=x,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_C_x_mx_z, 'd--', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=-x,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_C_x_y_x, 'v:', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_C_x_y_z, '^:', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_C_x_y_yz, '<:', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=yz)')
plt.plot(
    distances_pair, Vdip_C_x_yz_yz, '>:', lw=2.0, mfc='None',
    label=r'None ($\vec{\mu}_i$=x,$\vec{\mu}_j$=yz,$\vec{r}_{ij}$=yz)')

plt.ylim(-6.0, 1.0)

plt.title(
    'Damped Electrostatic Potential of diatom with charges '
    + r'$q_i$=' + f'{ci:.1f}, ' + r'$q_j$=' + f'{cj:.1f}\n' 
    + 'and differently aligned dipole moments '
    + r'|$\mu_i$|=' + f'{di:.1f}, ' + r'|$\mu_j$|=' + f'{di:.1f}\n' 
    + 'and different truncation methods in cutoff range '
    + f'{short_cutoff:.1f} to {long_cutoff:.1f}',
    fontsize=8)

plt.xlabel('Distance (Ang)')
plt.ylabel('Damped Electrostatic Potential (eV)')

plt.legend(loc='lower right', title='Truncation method')
plt.show()


# Plot (charge+dipole)-(charge+dipole) electrostatics, potential truncation
plt.plot(
    distances_pair, V_ps, 'o-', mfc='None',
    label=r'Potential ($\mu_i$=0,$\mu_j$=0,$\vec{r}_{ij}$=arb.)')
plt.plot(
    distances_pair, Vdip_ps_x_x_x, 'x-', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=x,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_ps_x_mx_x, '+-', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=-x,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_ps_x_x_z, 's--', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=x,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_ps_x_mx_z, 'd--', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=-x,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_ps_x_y_x, 'v:', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_ps_x_y_z, '^:', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_ps_x_y_yz, '<:', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=yz)')
plt.plot(
    distances_pair, Vdip_ps_x_yz_yz, '>:', lw=2.0, mfc='None',
    label=r'Potential ($\vec{\mu}_i$=x,$\vec{\mu}_j$=yz,$\vec{r}_{ij}$=yz)')

plt.ylim(-6.0, 1.0)

plt.title(
    'Damped Electrostatic Potential of diatom with charges '
    + r'$q_i$=' + f'{ci:.1f}, ' + r'$q_j$=' + f'{cj:.1f}\n' 
    + 'and differently aligned dipole moments '
    + r'|$\mu_i$|=' + f'{di:.1f}, ' + r'|$\mu_j$|=' + f'{di:.1f}\n' 
    + 'and different truncation methods in cutoff range '
    + f'{short_cutoff:.1f} to {long_cutoff:.1f}',
    fontsize=8)

plt.xlabel('Distance (Ang)')
plt.ylabel('Damped Electrostatic Potential (eV)')

plt.legend(loc='lower right', title='Truncation method')
plt.show()


# Plot (charge+dipole)-(charge+dipole) electrostatics, forces truncation
# Plot (charge+dipole)-(charge+dipole) electrostatics, potential truncation
plt.plot(
    distances_pair, V_fs, 'o-', mfc='None',
    label=r'Force ($\mu_i$=0,$\mu_j$=0,$\vec{r}_{ij}$=arb.)')
plt.plot(
    distances_pair, Vdip_fs_x_x_x, 'x-', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=x,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_fs_x_mx_x, '+-', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=-x,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_fs_x_x_z, 's--', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=x,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_fs_x_mx_z, 'd--', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=-x,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_fs_x_y_x, 'v:', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=x)')
plt.plot(
    distances_pair, Vdip_fs_x_y_z, '^:', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=z)')
plt.plot(
    distances_pair, Vdip_fs_x_y_yz, '<:', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=y,$\vec{r}_{ij}$=yz)')
plt.plot(
    distances_pair, Vdip_fs_x_yz_yz, '>:', lw=2.0, mfc='None',
    label=r'Force ($\vec{\mu}_i$=x,$\vec{\mu}_j$=yz,$\vec{r}_{ij}$=yz)')

plt.ylim(-6.0, 1.0)

plt.title(
    'Damped Electrostatic Potential of diatom with charges '
    + r'$q_i$=' + f'{ci:.1f}, ' + r'$q_j$=' + f'{cj:.1f}\n' 
    + 'and differently aligned dipole moments '
    + r'|$\mu_i$|=' + f'{di:.1f}, ' + r'|$\mu_j$|=' + f'{di:.1f}\n' 
    + 'and different truncation methods in cutoff range '
    + f'{short_cutoff:.1f} to {long_cutoff:.1f}',
    fontsize=8)

plt.xlabel('Distance (Ang)')
plt.ylabel('Damped Electrostatic Potential (eV)')

plt.legend(loc='lower right', title='Truncation method')
plt.show()
