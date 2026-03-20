#!/usr/bin/env python
import sys
import numpy as np
from goptimizer import parse_ifile, write_ofile
from ase import Atoms
from asparagus import Asparagus
import time

if __name__ == "__main__":

    # 1. The input and output filenames from gaussian
    ifile = sys.argv[2]
    ofile = sys.argv[3]

    # 2. Parse Gaussian Input
    (natoms, deriv, charge, spin, atomtypes, coordinates) = parse_ifile(ifile)

#===============================================================================
# NN part: Setup Atoms and Calculator
#===============================================================================

    atoms = Atoms(atomtypes, coordinates)
    model = Asparagus(config='.json') #Your .json file, make sure it contains in the model properties "hessian"
    calc = model.get_ase_calculator()

    if 'dipder' not in calc.implemented_properties:
        # Convert tuple to list if necessary, append, and convert back
        props = list(calc.implemented_properties)
        props.append('dipder')
        calc.implemented_properties = props

    calc.model_conversion['dipder'] = 1.0
    atoms.calc = calc
    calc.calculate(atoms, properties=['energy', 'forces', 'hessian', 'dipole', 'atomic_charges'])


#===============================================================================
# Extract and Convert Results
#===============================================================================

    # Energy conversion to hartree
    energy = calc.results['energy'] / 27.211386024367243
    print(energy)
    # Gradient conversion to hartree/bohr
    gradient = -calc.results['forces']
    gradient = (gradient * 0.5291772105638411) / 27.211386024367243
    print(gradient)
    # Hessian conversion
    hessian_raw = calc.results['hessian'].reshape(3*natoms, 3*natoms)
    hessiantmp = []
    count = 1
    for i in range(3*natoms):
        for j in range(count):
            hessiantmp.append(hessian_raw[i, j])
        count += 1
    hessian = np.reshape(np.array(hessiantmp), (-1, 3))
    hessian = (hessian * 0.5291772105638411**2) / 27.211386024367243
    #print(hessian)
    # Dipole conversion to e*bohr
    charges = calc.results['atomic_charges']
    mu = np.dot(charges, atoms.get_positions()) / 0.5291772105638411
    #print(f"DEBUG Dipole (mu): \n{mu}")

    # --- DIPOLE DERIVATIVE (APT) ---
    dipder_raw = calc.results.get('dipder')

    if dipder_raw is None:
        print("CRITICAL ERROR: 'dipder' is still returning None from the calculator!")
        sys.exit(1)

    print(f"DEBUG Raw Dipder Shape: {dipder_raw.shape}")

    # Gaussian expects a flattened (3*N, 3) matrix.
    # Rows: coordinate derivatives (x1, y1, z1, x2, y2, z2...)
    # Columns: dipole components (mu_x, mu_y, mu_z)
    # The transpose(1, 2, 0) perfectly maps (comp, atom, coord) to (atom, coord, comp)
    dipder_gauss = dipder_raw.transpose(1, 2, 0).reshape(-1, 3)


#===============================================================================
    # Produce the Gaussian input file with the supplied data
    write_ofile(ofile, energy, natoms,
                gradient=gradient,
                hessian=hessian,
                dipole=mu,
                dipole_derivative=dipder_gauss)
