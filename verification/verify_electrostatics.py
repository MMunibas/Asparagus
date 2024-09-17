import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from asparagus import module

import matplotlib.pyplot as plt

# Create model setup
distances = torch.tensor(np.arange(0.1, 12.01, 0.1), dtype=torch.float64)
Ndist = len(distances)
properties = {}
properties['atomic_charges'] = torch.tensor(
    [1.0, -1.0]*Ndist, dtype=torch.float64)
idx_i = torch.tensor(np.arange(Ndist)*2, dtype=torch.int64)
idx_j = torch.tensor(np.arange(Ndist)*2 + 1, dtype=torch.int64)

long_cutoff = 10.0
short_cutoff = 8.0

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

# Potential energy is twice i<->j contribution (j<->i not 'computed')
V_C = V_C[::2]*2


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

# Potential energy is twice i<->j contribution (j<->i not 'computed')
V_ps = V_ps[::2]*2


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

# Potential energy is twice i<->j contribution (j<->i not 'computed')
V_fs = V_fs[::2]*2

plt.plot(distances, V_C, label='None')
plt.plot(distances, V_ps, label='Potential Shift')
plt.plot(distances, V_fs, label='Force Shift')

plt.title(f'Truncation methods at cutoff {long_cutoff:.1f}')

plt.xlabel('Distance (Ang)')
plt.ylabel('Electrostatic Potential (eV)')

plt.legend(loc='lower right', title='Truncation method')
plt.show()
