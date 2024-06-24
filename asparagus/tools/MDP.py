import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
import matplotlib.pyplot as plt
import warnings

import numpy as np

import ase
from ase.io import read, write
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin


from ase.io.trajectory import Trajectory

from .. import interface
from .. import settings
from .. import utils
from .. import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MDP']

class MDP:

    '''

    This class calculates the Minimum Dynamic Path (MDP) [J. Chem. Phys. 150, 074107 (2019)]
    for a given molecule starting from the Transition state.

    Parameters:
    -----------
    atoms: ase.Atoms
        Transition state geometry
    model_calculator: asparagus.Asparagus object
       An asparagus model.
    time_step: opt(float)
        Time step for the MD simulation
    langevin: opt(bool)
        If True use Langevin dynamics, if False use Velocity Verlet
    friction: opt(float)
        Friction coefficient for Langevin dynamics only if langevin=True
    temperature: opt(float)
        Temperature for Langevin dynamics only if langevin=True
    eps: opt(float)
        Initial displacement for the MDP
    number_of_steps: opt(int)
        Number of steps for the MDP
    output: opt(str)
        Output file name for the trajectory
    output_file: opt(str)
        Output file name for the log file of energies


    '''

    def __init__(self,atoms=None,
                 model_calculator=None,
                 forward=True,
                 time_step=0.1,
                 langevin=False,
                 friction=0.02,
                 temperature=300,
                 eps=0.0005,
                 number_of_steps=4000,
                 output='mdp.traj',
                 output_file=None,
                 **kwargs):

        # ASE object atoms
        self.atoms = atoms
        # Options for the dynamivs
        self.eps = eps
        self.number_of_steps = number_of_steps
        self.forward = forward
        self.langevin = langevin
        self.time_step = time_step
        # Output files
        self.output = output
        self.output_file = output_file

        if self.langevin:
            self.friction = friction
            self.temperature = temperature
            print('Using Langevin dynamics with friction %s and temperature %s' % (self.friction, self.temperature))
        elif self.langevin is True and (friction is None or temperature is None):
            raise ValueError('The friction and temperature are required for Langevin dynamics')
        else:
            self.friction = None
            self.temperature = None

        if self.output_file is None:
            self.output_file = self.output + '.txt'
        elif self.output_file is False:
            pass
        else:
            self.output_file = output_file

        if self.atoms is type(str):
            self.atoms = read(self.atoms)

        if self.atoms is None:
            raise ValueError('The transition state geometry is required')

        if model_calculator is None:
            raise ValueError('The model calculator is required')

        # Get the ASE calculator
        self.ase_calculator = model_calculator.get_ase_calculator()

        # Check the implemented properties
        self.implemented_properties = self.ase_calculator.implemented_properties

        if 'forces' not in self.implemented_properties:
            self.implemented_properties.append('forces')

        if 'hessian' not in self.implemented_properties:
            self.implemented_properties.append('hessian')


        # Initialize the trajectory file
        self.mdp_trajectory = Trajectory(self.output, 'w', self.atoms, properties=['energy', 'forces'])


    def run_mdp(self,forward=None):
        """
        This function runs the MDP calculation.

        Parameters
        ----------
        forward: opt(bool) If True forward, if False backward. This defines the direction of the momentum of the system.
                           Default is None because it is defined in the __init__ function.

        Returns
        -------


        """
        if forward is not None:
            warnings.WarningMessage('The forward parameter is rewroted to %s' % forward)
            self.forward = forward

        initial_system = self.atoms.copy()
        initial_system.set_calculator(self.ase_calculator)

        n_atom = len(initial_system)

        # Calculate the hessian of the transition state
        initial_system.get_potential_energy()
        hessian = np.reshape(initial_system.calc.results['hessian'], (n_atom * 3, n_atom * 3))

        # Calculate the eigenvalues and eigenvectors of the hessian

        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        eigenvectors = eigenvectors.T
        normdisp = eigenvectors[0].reshape(n_atom, 3)

        if self.forward:
            normdisp = normdisp
        else:
            normdisp = -normdisp

        initial_system.set_momenta(self.eps * normdisp)

        pos = initial_system.get_positions()

        new_pos = pos + self.eps * normdisp
        initial_system.set_positions(new_pos)

        if self.langevin:
            dyn = Langevin(initial_system, self.time_step * units.fs, self.temperature * units.kB, self.friction)
        else:
            dyn = VelocityVerlet(initial_system, self.time_step * units.fs)

        self.write_trajectory(initial_system)
        self.ase_calculator.implemented_properties.remove('hessian')

        step = []
        eners_pot = []
        eners_kin = []
        eners_tot = []
        for i in range(self.number_of_steps):
            dyn.run(1)
            ener_i = initial_system.get_potential_energy()
            epot = ener_i/n_atom
            ekin = initial_system.get_kinetic_energy()/n_atom
            etot = epot + ekin
            self.write_trajectory(initial_system)
            eners_tot.append(etot)
            eners_pot.append(epot)
            eners_kin.append(ekin)
            step.append(i)

        if self.output_file is not False:
            np.savetxt(self.output_file, np.c_[step,eners_pot,eners_kin,eners_tot],
                       header='Step,Energy Potential,Energy Kinetic,Energy Total')


    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """

        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.mdp_trajectory.write(system_noconstraint)


