#!/usr/bin/env python
import sys
import numpy as np
from goptimizer import parse_ifile, write_ofile
from ase import Atoms
from asparagus import Asparagus
import numpy as np

if __name__ == "__main__":
    # The input and output filenames from gaussian (don't change).
    ifile = sys.argv[2]
    ofile = sys.argv[3]
    # What you get out from Gaussian.
    # Cartesian coordinates in angstrom, use 0.52917721092 to get Bohr.
    (natoms, deriv, charge, spin, atomtypes, coordinates) = parse_ifile(ifile)
    
#===============================================================================
#NN part, get nn energy and force
    atoms = Atoms(atomtypes, coordinates)
    model = Asparagus(config='your_model.json') #Replace with your model .json file
    calc = model.get_ase_calculator()
    atoms.calc = calc

    energy = atoms.get_potential_energy() #output in eV
    energy /= 27.211386246 #conversion to hartree
    gradient = -atoms.get_forces() #output in eV/angstrom
    gradient = (gradient*0.52917721092)/27.211386245
    charges = atoms.calc.results['atomic_charges']
    mu = np.dot(charges, atoms.get_positions())/0.5291772105638411 #conversion to e*bohr
#===============================================================================
    # Produce the Gaussian input file with the supplied data
    write_ofile(ofile, energy, natoms, gradient=gradient, dipole=mu)

