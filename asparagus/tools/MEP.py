import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
import matplotlib.pyplot as plt

import numpy as np

import ase
from ase.io import read, write,iread
from ase import units
from ase.neb import NEB
from ase.neb import NEBTools
from ase.optimize import FIRE
from ase.constraints import FixAtoms

from ase.io.trajectory import Trajectory

from .. import interface
from .. import settings
from .. import utils
from .. import model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MEP']

class MEP:

    '''

    This class calculates the minimum energy path from the transtion state to the most favorable part of the PES.

    It follows the gradient of the vector of the Transition state, therefore it is necessary to calculate the hessian of the transition state.


    Parameters:
    -----------

    atoms: ase.Atoms object
        It is the transition state geometry.
    model_calculator: asparagus.Asparagus object
       An asparagus model.
    eps: float (optional)
        Step size to follow the gradient.
    number_of_steps: int (optional)
        Number of steps to follow the gradient.
    output: str (optional)
        Name of the output trajectory if not defined it is equal to mep.traj.
    output_file: str (optional)
        Name of the output file with the energy and steps of the MEP.
    '''

    def __init__(self,atoms=None,
                 model_calculator=None,
                 eps=0.001,
                 number_of_steps=4000,
                 output='mep.traj',
                 output_file=None,
                 **kwargs):

            #ASE object atoms
            self.atoms = atoms
            #Set up the dynamics
            self.eps = eps
            self.number_of_steps = number_of_steps
            #Output files
            self.output = output
            self.output_file = output_file

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
            self.implemented_properties =self.ase_calculator.implemented_properties

            if 'forces' not in self.implemented_properties:
                self.implemented_properties.append('forces')

            if 'hessian' not in self.implemented_properties:
                self.implemented_properties.append('hessian')

            #Initialize the trajectory file
            self.mep_trajectory = Trajectory(self.output, 'w', self.atoms,properties=['energy','forces'])

    def get_MEP(self):
        """
        This function calculates the minimum energy path from the transition state to the most favorable part of the PES.

        It saves the trajectory in a trajectory file and the energy and steps in a txt file.

        """

        initial_system = self.atoms.copy()
        initial_system.set_calculator(self.ase_calculator)

        n_atom = len(initial_system)

        # Calculate the hessian of the transition state
        initial_system.get_potential_energy()
        hessian = np.reshape(initial_system.calc.results['hessian'],(n_atom*3,n_atom*3))

        # Calculate the eigenvalues and eigenvectors of the hessian

        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        eigenvectors = eigenvectors.T
        v_to_follow = eigenvectors[0].reshape(n_atom,3)

        # get initial positions
        pos = initial_system.get_positions() - self.eps * v_to_follow
        initial_system.set_positions(pos)
        grad_old = v_to_follow

        self.ase_calculator.implemented_properties.remove('hessian')
        steps = []
        energies = []

        for i in range(self.number_of_steps):
            pos = initial_system.get_positions()

            grad = -self.ase_calculator.get_forces()
            grad /= np.linalg.norm(grad)
            pos = pos - self.eps * grad
            initial_system.set_positions(pos)
            ener_i = initial_system.get_potential_energy()
            energies.append(ener_i)
            steps.append(i)
            self.mep_trajectory.write(initial_system)
            if np.linalg.norm(grad - grad_old) < 1e-4:
                break
            grad_old = grad

        if self.output_file is not False:
            np.savetxt(self.output_file, np.c_[steps,energies],header='Step,Energy')



    def write_trajectory(self, system):
        """
        Write current image to trajectory file but without constraints
        """

        system_noconstraint = system.copy()
        system_noconstraint.set_constraint()
        self.mep_trajectory.write(system_noconstraint)

class MEP_NEB:

    '''

    TO BE FINISHED

    This class calculates the minimum energy path between two geometries in the PES using the nudged elastic band method.

    '''

    def __init__(self,react,
                 prod,
                 atoms_charge=None,
                 config=None,
                 config_file=None,
                 model_checkpoint=None,
                 implemented_properties=None,
                 use_neighbor_list=False,
                 label='mep_neb',
                 nimages=5,
                 fixed_atoms=None,
                 k=0.1,
                 climb=True,
                 fmax=0.05,
                 output='mep_neb.traj',
                 **kwargs
                 ):

        if react is type(str):
            react = read(react)
        if prod is type(str):
            prod = read(prod)

        self.react = react
        self.prod = prod

        if self.fixed_atoms is None:
            self.fixed_atoms = []
        else:
            self.fixed_atoms = fixed_atoms

        self.nimages = nimages
        self.k = k
        self.climb = climb
        self.fmax = fmax
        self.output = output

        ######################################
        # # # Check ASE Calculator Input # # #
        ######################################

        # Assign model parameter configuration library
        if config is None:
            config_data = settings.get_config(
                self.config, config_file, **kwargs)
        else:
            config_data = settings.get_config(
                config, config_file, **kwargs)

        # Check model parameter configuration and set default
        config_data.check()

        # Check for empty config dictionary
        if "model_directory" not in config_data:
            raise SyntaxError(
                "Configuration does not provide information for a model "
                + "calculator. Please check the input in 'config'.")

        ##################################
        # # # Prepare NNP Calculator # # #
        ##################################

        # Assign NNP calculator
        if self.model_calculator is None:
            # Assign NNP calculator model
            self.model_calculator = self._get_Calculator(
                config_data,
                **kwargs)

        # Add calculator info to configuration dictionary
        if hasattr(self.model_calculator, "get_info"):
            config_data.update(
                self.model_calculator.get_info(),
                verbose=False)

        # Initialize checkpoint file manager and load best model
        filemanager = utils.FileManager(config_data, **kwargs)
        if model_checkpoint is None:
            latest_checkpoint = filemanager.load_checkpoint(best=True)
        elif utils.is_integer(model_checkpoint):
            latest_checkpoint = filemanager.load_checkpoint(
                num_checkpoint=model_checkpoint)
        else:
            raise ValueError(
                "Input 'model_checkpoint' must be either None to load best "
                + "model checkpoint or an integer of a respective checkpoint "
                + "file.")
        self.model_calculator.load_state_dict(
            latest_checkpoint['model_state_dict'])

        ##################################
        # # # Prepare ASE Calculator # # #
        ##################################
        if implemented_properties is not None:
            self.implemented_properties = implemented_properties
        else:
            self.implemented_properties = ['energy', 'forces']

        if 'energy' not in self.implemented_properties:
            self.implemented_properties.append('energy')

        if 'forces' not in self.implemented_properties:
            self.implemented_properties.append('forces')


        # Create images

        self.images = [self.react]

        self.images += [self.react.copy() for i in range(self.nimages)]

        self.images += [self.prod]

        self.neb = NEB(self.images, k=self.k, climb=self.climb)

        self.neb.interpolate()

        # set_calculators
        self.calcs = [interface.ASE_Calculator(self.model_calculator,
                                                atoms=self.images[i],
                                                atoms_charge=atoms_charge,
                                                implemented_properties=self.implemented_properties,
                                                use_neighbor_list=use_neighbor_list,
                                                label=label) for i in range(self.nimages+2)]


    def get_mep_neb(self):

        # Setting up the constraints and the calculators

        for index, image in enumerate(self.images):
            image.calc = self.calcs[index]
            image.set_constraint(FixAtoms(mask=self.fixed_atoms))

        # Run optimization
        optimizer = FIRE(self.neb, trajectory=self.output)
        optimizer.run(fmax=self.fmax)

    def analyze_neb(self,output=None,show_plot=True,save_plot=False):

        '''
        This function analyzes the NEB calculation and returns the energy barrier and the reaction energy and a NEB plot.

        Parameters
        ----------
        output: The output trajectory from the NEB calculation.
        show_plot: bool (optional) show the plot of the NEB calculation.
        save_plot: bool (optional) save the plot of the NEB calculation.

        Returns
        -------

        '''
        if output is None:
            raise ValueError('The output trajectory is required')

        traj = Trajectory(output)

        # It takes the last trajectory of the NEB calculation, Note that ASE does not read the extreme images of the NEB calculation.
        traj_last = traj[-self.nimages+1:-2]

        nebtools = NEBTools(traj_last)

        Ef, dE = nebtools.get_barrier()
        print('Energy barrier:{} eV'.format(Ef))
        print('Reaction energy:{} eV'.format(dE))

        if show_plot:
            #Plotting set up from ASE NEB tutorial
            fig = plt.figure(figsize=(5.5, 4.0))
            ax = fig.add_axes((0.15, 0.15, 0.8, 0.75))
            nebtools.plot_band(ax)
            if save_plot:
                plt.savefig('neb_plot.png',dpi=300)

        if save_plot and not show_plot:
            fig = nebtools.plot_band()
            fig.savefig('neb_plot.png',dpi=300)














