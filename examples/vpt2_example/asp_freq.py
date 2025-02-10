#!/usr/bin/env python
import torch
import sys
import numpy as np
from goptimizer import parse_ifile, write_ofile
from ase import Atoms
from asparagus import Asparagus
import numpy as np
import time
#sys.stderr = open('errorlog.txt', 'w')
if __name__ == "__main__":

    # The input and output filenames from gaussian (don't change).
    ifile = sys.argv[2]
    ofile = sys.argv[3]

    # What you get out from Gaussian.
    # Cartesian coordinates in angstrom, use 0.52917721092 to get Bohr.
    (natoms, deriv, charge, spin, atomtypes, coordinates) = parse_ifile(ifile)

#===============================================================================
#NN part, get nn energy and force and whatever needed.

    atoms = Atoms(atomtypes, coordinates)
    model = Asparagus(config='your_model.json') #replace with your model .json file
    calc = model.get_ase_calculator()
    atoms.calc = calc

    #get energy prediction from NN
    energy = atoms.get_potential_energy() #output in eV
    energy /= 27.211386024367243 #conversion to hartree
    
    gradient = -atoms.get_forces() #output in eV/angstrom
    gradient = (gradient*0.5291772105638411)/27.211386024367243 #converstion to hartree/bohr

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
    charges = atoms.calc.results['atomic_charges']
    mu = np.dot(charges, atoms.get_positions())/0.5291772105638411 #conversion to e*bohr 
    
    dipder_tensor = torch.load('dipder.pth') #Should be changed to calc.results['dipder'] = dipder
    dipder = dipder_tensor.cpu().detach().numpy()
    dipdertmp = []
    for i in range(3*len(atoms)):
        dipdertmp.append(dipder[0].reshape(-1)[i])
        dipdertmp.append(dipder[1].reshape(-1)[i])
        dipdertmp.append(dipder[2].reshape(-1)[i])
    dipder = np.array(dipdertmp).reshape(-1, 3)
#===============================================================================
    # Produce the Gaussian input file with the supplied data
    write_ofile(ofile, energy, natoms, gradient=gradient, hessian=hessian, dipole=mu, dipole_derivative=dipder)
    #write_ofile(ofile, energy, natoms, gradient=gradient, hessian=hessian, dipole=mu)

#sys.stderr.close()
#sys.stderr = sys.__stderr__



