import os
import sys
import json

import numpy as np

from ase import units

# Function to store results in json format
result_file = "results.json"
def save_results(results, result_file):
    """
    Save result dictionary as json file
    """
    with open(result_file, 'w') as f:
        json.dump(results, f)


# Check Molpro output file
molpro_out = sys.argv[1]
if not os.path.exists(molpro_out):
    save_results({}, result_file)
    exit()

# Read Molpro output
with open(molpro_out, 'r') as fout:
    lines = fout.readlines()

# Get MP2 calculation method (Closed Shell or Restricted Open Shell)
# By default, assume Closed Shell MP2
mp2_closed_shell = True
for line in lines:
    if 'PROGRAM * MP2' in line:
        mp2_closed_shell = True
        break
    elif 'PROGRAM * RMP2' in line:
        mp2_closed_shell = False
        break

# Prepare result variables
energy = None
forces = []
dipole = None

# Iterate over output file lines
flag_forces = False
counter_forces = 0
for line in lines:
    
    # MP2 Energy line
    if 'MP2/' in line:
        energy = float(line.split()[-1])*units.Hartree

    # MP2 Forces lines
    # If forces section found, reduce counter until atom forces start
    if flag_forces and counter_forces > 0:
        counter_forces -= 1
        #print("Skip", counter_forces, line)
    # Else if forces section found and started but empty line, forces are done
    elif flag_forces and counter_forces == 0 and len(line.strip()) == 0:
        flag_forces = False
        #print("Done", line)
    # Else if forces section found and started, read closed shell MP2 forces
    elif mp2_closed_shell and flag_forces:
        #print("Read", line, len(line.strip()))
        forces.append(
            [-float(ll)*units.Hartree/units.Bohr for ll in line.split()[1:]])
    # Else if forces section found and started, read RMP2 forces
    elif (not mp2_closed_shell) and flag_forces:
        #print("Read", line, len(line.strip()))
        forces.append(
            [-float(ll)*units.Hartree/units.Bohr for ll in line.split()[1:4]])
    # Else if forces section not found yet, look for closed shell MP2 marker
    elif mp2_closed_shell and 'MP2 GRADIENT FOR STATE' in line:
        flag_forces = True
        counter_forces = 4
        #print("Start", counter_forces, line)
    # Else if forces section not found yet, look for RMP2 marker
    elif (not mp2_closed_shell) and 'Numerical gradient for MP2' in line:
        flag_forces = True
        counter_forces = 4
        #print("Start", counter_forces, line)

    # MP2 Dipole line
    # Only the Closed Shell MP2 method in Molpro can compute the MP2 Dipole
    # moment. That is not possible(?)/implemented for RMP2 on an analytic
    # level, only via finite difference methods applying small finite fields.
    # For that reason, the Restricted Open Shell Hartree Fock Dipole moments
    # will be read instead.
    if mp2_closed_shell and '!MP2 STATE' in line and 'Dipole moment' in line:
        dipole = [float(ll)*units.Bohr for ll in line.split()[-3:]]
    elif (
        (not mp2_closed_shell)
        and '!RHF STATE' in line
        and 'Dipole moment' in line
    ):
        dipole = [float(ll)*units.Bohr for ll in line.split()[-3:]]

# Collect results
results = {
    'energy': energy,
    'forces': forces,
    'dipole': dipole,
    }

# Save results to result file
save_results(results, result_file)
