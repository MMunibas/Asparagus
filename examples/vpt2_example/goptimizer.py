# MIT License
#
# Copyright (c) 2018 Anders Steen Christensen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import print_function

import fortranformat as ff
import sys

import numpy as np

G09_ANGS_TO_BOHR = 0.52917721092

NAME = {
    1:  'H'  , 
    2:  'He' , 
    3:  'Li' , 
    4:  'Be' , 
    5:  'B'  , 
    6:  'C'  , 
    7:  'N'  , 
    8:  'O'  , 
    9:  'F'  , 
   10:  'Ne' , 
   11:  'Na' , 
   12:  'Mg' , 
   13:  'Al' , 
   14:  'Si' , 
   15:  'P'  , 
   16:  'S'  , 
   17:  'Cl' , 
   18:  'Ar' , 
   19:  'K'  , 
   20:  'Ca' , 
   21:  'Sc' , 
   22:  'Ti' , 
   23:  'V'  , 
   24:  'Cr' , 
   25:  'Mn' , 
   26:  'Fe' , 
   27:  'Co' , 
   28:  'Ni' , 
   29:  'Cu' , 
   30:  'Zn' , 
   31:  'Ga' , 
   32:  'Ge' , 
   33:  'As' , 
   34:  'Se' , 
   35:  'Br' , 
   36:  'Kr' , 
   37:  'Rb' , 
   38:  'Sr' , 
   39:  'Y'  , 
   40:  'Zr' , 
   41:  'Nb' , 
   42:  'Mo' , 
   43:  'Tc' , 
   44:  'Ru' , 
   45:  'Rh' , 
   46:  'Pd' , 
   47:  'Ag' , 
   48:  'Cd' , 
   49:  'In' , 
   50:  'Sn' , 
   51:  'Sb' , 
   52:  'Te' , 
   53:  'I'  , 
   54:  'Xe' , 
   55:  'Cs' , 
   56:  'Ba' , 
   57:  'La' , 
   58:  'Ce' , 
   59:  'Pr' , 
   60:  'Nd' , 
   61:  'Pm' , 
   62:  'Sm' , 
   63:  'Eu' , 
   64:  'Gd' , 
   65:  'Tb' , 
   66:  'Dy' , 
   67:  'Ho' , 
   68:  'Er' , 
   69:  'Tm' , 
   70:  'Yb' , 
   71:  'Lu' , 
   72:  'Hf' , 
   73:  'Ta' , 
   74:  'W'  , 
   75:  'Re' , 
   76:  'Os' , 
   77:  'Ir' , 
   78:  'Pt' , 
   79:  'Au' , 
   80:  'Hg' , 
   81:  'Tl' , 
   82:  'Pb' , 
   83:  'Bi' , 
   84:  'Po' , 
   85:  'At' , 
   86:  'Rn' , 
   87:  'Fr' , 
   88:  'Ra' , 
   89:  'Ac' , 
   90:  'Th' , 
   91:  'Pa' , 
   92:  'U'  , 
   93:  'Np' , 
   94:  'Pu' , 
   95:  'Am' , 
   96:  'Cm' , 
   97:  'Bk' , 
   98:  'Cf' , 
   99:  'Es' , 
  100:  'Fm' , 
  101:  'Md' , 
  102:  'No' , 
  103:  'Lr' , 
  104:  'Rf' , 
  105:  'Db' , 
  106:  'Sg' , 
  107:  'Bh' , 
  108:  'Hs' , 
  109:  'Mt' , 
  110:  'Ds' , 
  111:  'Rg' , 
  112:  'Cn' , 
  114:  'Uuq', 
  116:  'Uuh'}




def parse_ifile(ifile):
    f = open(ifile, "r")
    lines = f.readlines()
    f.close()

    tokens = lines[0].split()

    natoms = int(tokens[0])
    deriv  = int(tokens[1])
    charge = int(tokens[2])
    spin   = int(tokens[3])

    print("-------------------------------------")
    print("--  GOPTIMIZER INPUT  ---------------")
    print("-------------------------------------")
    print()
    print("  Number of atoms:     ", natoms)
    print("  Derivative requested:", deriv)
    print("  Total charge:        ", charge)
    print("  Spin:                ", spin)
    print()

    coords = np.zeros((natoms,3))
    atomtypes = []

    for i, line in enumerate(lines[1:1+natoms]):

        tokens = line.split()
        a = NAME[int(tokens[0])]

        c = np.array([float(tokens[1]), float(tokens[2]),float(tokens[3])])*G09_ANGS_TO_BOHR

        coords[i,0] = c[0]
        coords[i,1] = c[1]
        coords[i,2] = c[2]

        atomtypes.append(a)
    
    print("  Found the following atoms:")
    print("  --------------------------")
    print()


    for i in range(natoms):
        print("  Atom %3i  %-3s   %20.12f %20.12f %20.12f" % \
                (i, atomtypes[i], coords[i,0], coords[i,1], coords[i,2]))
        

    print()
    print("-------------------------------------")
    print("-------------------------------------")
    print("-------------------------------------")
    print()


    return natoms, deriv, charge, spin, atomtypes, coords


def write_ofile(ofile, energy, natoms, dipole=None, gradient=None, 
        polarizability=None, dipole_derivative=None, hessian=None):

    # Define output formats
    headformat = ff.FortranRecordWriter("4D20.12")
    bodyformat = ff.FortranRecordWriter("3D20.12")

    # Print output header:
    if dipole is None:
        dipole = np.zeros((3))

    head = [energy, dipole[0], dipole[1], dipole[2]]

    f = open(ofile, "w")

    # Write headers
    headstring = headformat.write(head)
    f.write(headstring + "\n")


    # Write gradient section
    if gradient is None:
        gradient = np.zeros((natoms,3))

    assert gradient.shape[0] == natoms, "ERROR: First dimension of gradient doesn't match natoms."
    assert gradient.shape[1] == 3, "ERROR: Second dimension of gradient is not 3."

    for i in range(natoms):
        output = bodyformat.write(gradient[i])
        f.write(output+ "\n")

    # Write polarization section
    if polarizability is None:
        polarizability = np.zeros((2,3))
    
    for i in range(2):
        output = bodyformat.write(polarizability[i])
        f.write(output+ "\n")

    # Write dipole derivative section
    if dipole_derivative is None:
        dipole_derivative = np.zeros((3*natoms,3))

    for i in range(3*natoms):
        output = bodyformat.write(dipole_derivative[i])
        f.write(output+ "\n")

    # Write hessian section
    if hessian is None:
        hessian = np.zeros((3*natoms, 3*natoms))
        print(hessian.shape)
        print(int(3*natoms*(3*natoms+1)/6))
        quit()
    for i in range(int(3*natoms*(3*natoms+1)/6)):
        output = bodyformat.write(hessian[i])
        f.write(output+  "\n")



    f.close()
