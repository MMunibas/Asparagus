#!/usr/bin/env python
import torch
import sys
import os
import numpy as np
import traceback
from goptimizer import parse_ifile, write_ofile
from ase import Atoms
from asparagus import Asparagus

LOG_FILE = "asp_vpt2_debug.log"

def main():
    # The input and output filenames from gaussian (don't change).
    ifile = sys.argv[2]
    ofile = sys.argv[3]

    # Use absolute path for config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'NMS.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Config file not found: {model_path}")

    # What you get out from Gaussian.
    # Cartesian coordinates in angstrom, use 0.52917721092 to get Bohr.
    (natoms, deriv, charge, spin, atomtypes, coordinates) = parse_ifile(ifile)

#===============================================================================
#NN part, get nn energy and force and whatever needed.

    atoms = Atoms(atomtypes, coordinates)
    model = Asparagus(config=model_path)
    calc = model.get_ase_calculator()
    atoms.calc = calc

    #get energy prediction from NN
    # This call triggers the forward pass, calculating 'dipder' internally in modified physnet.py
    energy = atoms.get_potential_energy() #output in eV
    energy /= 27.211386024367243 #conversion to hartree

    gradient = -atoms.get_forces() #output in eV/angstrom
    gradient = (gradient*0.5291772105638411)/27.211386024367243 #converstion to hartree/bohr

    # Fallback if hessian is not computed automatically
    if 'hessian' not in calc.results:
        calc.calculate(atoms, properties=['hessian'])

    hessian = calc.results['hessian'].reshape(3*len(atoms),3*len(atoms)) #output in ev/angstr**2

    #gaussian needs only lower triangle of hessian, thus reformat
    hessiantmp = []

    count=1
    for i in range(3*natoms):
        for j in range(count):
            hessiantmp.append(hessian[i,j])
        count+=1

    hessian = np.reshape(np.array(hessiantmp), (-1,3))
    hessian = (hessian*0.5291772105638411**2)/27.211386024367243 #conversion to ha/bohr**2

    # Dipole calculation
    if 'atomic_charges' in calc.results:
        charges = calc.results['atomic_charges']
        mu = np.dot(charges, atoms.get_positions())/0.5291772105638411 #conversion to e*bohr
    else:
        mu = np.zeros(3)

    if 'dipder' in calc.results:
        # Get raw tensor (Shape: 3 components, N atoms, 3 coords)
        dipder_raw = calc.results['dipder']

        # Convert to numpy if it's still a tensor
        if isinstance(dipder_raw, torch.Tensor):
            dipder = dipder_raw.detach().cpu().numpy()
        else:
            dipder = dipder_raw

    else:
        raise KeyError(" 'dipder' not found in calc.results. Ensure physnet.py and model_ase.py are modified correctly.")

    dipdertmp = []
    for i in range(3*len(atoms)):
        dipdertmp.append(dipder[0].reshape(-1)[i])
        dipdertmp.append(dipder[1].reshape(-1)[i])
        dipdertmp.append(dipder[2].reshape(-1)[i])
    dipder = np.array(dipdertmp).reshape(-1, 3)

#===============================================================================
    # Produce the Gaussian input file with the supplied data
    write_ofile(ofile, energy, natoms, gradient=gradient, hessian=hessian, dipole=mu, dipole_derivative=dipder)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        with open(LOG_FILE, "w") as f:
            f.write("ASP_FREQ CRASH REPORT:\n")
            traceback.print_exc(file=f)
        sys.exit(1)
