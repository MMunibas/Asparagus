import os
import re
import json

import numpy as np

from ase import units

# Orca output and result file path
output_file = "run_orca.out"
gradient_file = "run_orca.engrad"
result_file = "results.json"

def save_results(results, result_file):
    """
    Save result dictionary as json file
    """
    with open(result_file, 'w') as f:
        json.dump(results, f)

# If output file not written, save empty result dictionary
if not os.path.exists("run_orca.out"):
    save_results({}, result_file)
    exit()

# Initialize result dictionary
results = {}

# Read output file
with open(output_file, 'r') as f:
    flines = f.read()

# Read energy
re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
re_not_converged = re.compile(r"Wavefunction not fully converged")
found_line = re_energy.search(flines)
if found_line and not re_not_converged.search(found_line.group()):
    results['energy'] = float(found_line.group().split()[-1])*units.Hartree

# Read dipole
re_dipole = re.compile(r"Total Dipole Moment.*\n")
re_not_converged = re.compile(r"Wavefunction not fully converged")
found_line = re_dipole.search(flines)
if found_line and not re_not_converged.search(found_line.group()):
    results['dipole'] = list(np.array(
        found_line.group().split()[-3:], dtype=float)*units.Bohr)

# Read gradient file
with open(gradient_file, 'r') as f:
    flines = f.readlines()

# Read forces
getgrad = False
gradients = []
tempgrad = []
for i, line in enumerate(flines):
    if line.find('# The current gradient') >= 0:
        getgrad = True
        gradients = []
        tempgrad = []
        continue
    if getgrad and "#" not in line:
        grad = line.split()[-1]
        tempgrad.append(float(grad))
        if len(tempgrad) == 3:
            gradients.append(tempgrad)
            tempgrad = []
    if '# The at' in line:
        getgrad = False
results['forces'] = [
    list(-np.array(grad_i)*units.Hartree/units.Bohr)
    for grad_i in gradients]

# Save results
save_results(results, result_file)
