import os
import queue
import logging
import threading
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase.constraints import FixAtoms
from ase import units

from .. import settings
from .. import utils
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MCSampler']


class MCSampler(sample.Sampler):
    """
    A very simple Monte Carlo (MC) sampler class.
    Uses the Metropolis algorithm to generate samples for a molecule.

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    mc_temperature: float, optional, default 300
        Temperature of the MC simulation in Kelvin.
    mc_steps: integer, optional, default 1000
        Number of MC simulations steps
    mc_save_interval: int, optional, default 1
        Step interval to store system properties of the current frame.
    mc_max_displacement: float
        Maximum displacement of the MC simulation in Angstrom.
    
    Returns
    -------
    object
        Monta-Carlo Sampler class object
    """

    # Default arguments for sample module
    sample.Sampler._default_args.update({
        'mc_temperature':               300.,
        'mc_steps':                     1000,
        'mc_max_displacement':          0.1,
        'mc_save_interval':             1,
        })
    
    # Expected data types of input variables
    sample.Sampler._dtypes_args.update({
        'mc_temperature':               [utils.is_numeric],
        'mc_steps':                     [utils.is_numeric],
        'mc_max_displacement':          [utils.is_numeric],
        'mc_save_interval':             [utils.is_integer],
        })

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        mc_temperature: Optional[float] = None,
        mc_steps: Optional[int] = None,
        mc_max_displacement: Optional[float] = None,
        mc_save_interval: Optional[int] = None,
        **kwargs,
    ):

        # Sampler class label
        self.sample_tag = 'mc'

        # Initialize parent class
        super().__init__(
            sample_tag=self.sample_tag,
            config=config,
            config_file=config_file,
            **kwargs
            )
        
        ################################
        # # # Check MC Class Input # # #
        ################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)
        
        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, sample),
            check_dtype=utils.get_dtype_args(self, sample)
        )
        
        # Check sample properties for energy properties which are required for 
        # MC sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')

        return

    def get_info(self):
        """
        Returns a dictionary with the information of the MC sampler.
        
        Returns
        -------
        dict
            Dictionary with the information of the MC sampler.
        """
        
        info = super().get_info()
        info.update({
            'mc_temperature': self.mc_temperature,
            'mc_steps': self.mc_steps,
            'mc_max_displacement': self.mc_max_displacement,
            'mc_save_interval': self.mc_save_interval,
            })

        return info

    def run_systems(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        **kwargs,
    ):
        """
        Perform Normal Mode Scanning on the sample system.
        
        Parameters
        ----------
        sample_systems_queue: queue.Queue, optional, default None
            Queue object including sample systems or to which 'sample_systems' 
            input will be added. If not defined, an empty queue will be 
            assigned.
        """

        # Check sample system queue
        if sample_systems_queue is None:
            sample_systems_queue = queue.Queue()

        # Initialize thread continuation flag
        self.thread_keep_going = np.array(
            [True for ithread in range(self.sample_num_threads)],
            dtype=bool
            )
        
        # Add stop flag
        for _ in range(self.sample_num_threads):
            sample_systems_queue.put('stop')

        if self.sample_num_threads == 1:
            
            self.run_system(sample_systems_queue)
        
        else:

            # Create threads
            threads = [
                threading.Thread(
                    target=self.run_system, 
                    args=(sample_systems_queue, ),
                    kwargs={
                        'ithread': ithread}
                    )
                for ithread in range(self.sample_num_threads)]

            # Start threads
            for thread in threads:
                thread.start()

            # Wait for threads to finish
            for thread in threads:
                thread.join()

        return

    def run_system(
        self, 
        sample_systems_queue: queue.Queue,
        ithread: Optional[int] = None,
    ):
        """
        Perform a very simple MC Simulation using the Metropolis algorithm with
        the sample system.

        Parameters
        ----------
        sample_systems_queue: queue.Queue
            Queue of sample system information providing tuples of ASE atoms
            objects, index number and respective sample source and the total
            sample index.
        ithread: int, optional, default None
            Thread number
        """

        while self.keep_going(ithread):
            
            # Get sample parameters or wait
            sample = sample_systems_queue.get()
            
            # Check for stop flag
            if sample == 'stop':
                self.thread_keep_going[ithread] = False
                continue
            
            # Extract sample system to optimize
            (system, isample, source, index) = sample

            # If requested, perform structure optimization
            if self.sample_systems_optimize:

                # Perform structure optimization
                system = self.run_optimization(
                    sample_system=system,
                    sample_index=isample,
                    ithread=ithread)

            # Initialize trajectory file
            if self.sample_save_trajectory:
                trajectory_file = self.sample_trajectory_file.format(isample)
            else:
                trajectory_file = None

            # Assign calculator
            system = self.assign_calculator(
                system,
                ithread=ithread)

            # Perform Monte-Carlo simulation
            Nsample = self.monte_carlo_steps(
                system, 
                trajectory_file=trajectory_file,
                ithread=ithread)
            
            # Print sampling info
            msg = f"Sampling method '{self.sample_tag:s}' complete for system "
            msg += f"of index {index:d} from '{source}!'\n"
            if Nsample == 0:
                msg += f"No samples written to "
            if Nsample == 1:
                msg += f"{Nsample:d} sample written to "
            else:
                msg += f"{Nsample:d} samples written to "
            msg += f"'{self.sample_data_file:s}'.\n"
            
            logger.info(f"INFO:\n{msg:s}")
        
        return

    def monte_carlo_steps(
        self,
        system: ase.Atoms,
        Nsteps: Optional[int] = None,
        temperature: Optional[float] = None,
        max_displacement: Optional[float] = None,
        trajectory_file: Optional[str] = None,
        ithread: Optional[int] = None,
    ):
        """
        This does a simple Monte-Carlo simulation using the Metropolis
        algorithm.

        In the future we could add more sophisticated sampling methods
        (e.g. MALA or HMC)

        Parameters
        ----------
        system: ase.Atoms
            System to be sampled.
        Nsteps: int, optional, default None
            Number of Monte-Carlo steps to perform.
        temperature: float, optional, default None
            Sample temperature 
        max_displacement: float, optional, default None
            Maximum displacement of the MC simulation in Angstrom.
        trajectory_file: str, optional, default None
            ASE Trajectory file path to append sampled system if requested
        ithread: int, optional, default None
            Thread number
        
        Return
        ------
        int
            Number of sampled systems to database
        """
    
        # Check sample steps
        if Nsteps is None:
            Nsteps = self.mc_steps
        
        # Check Temperature parameter
        if temperature is None:
            temperature = self.mc_temperature
        beta = 1.0/(units.kB*self.mc_temperature)
        
        # Check maximum atom displacement
        if max_displacement is None:
            max_displacement = self.mc_max_displacement
        
        # Monte-Carlo acceptance and stored sample system counter
        Naccept = 0
        Nsample = 0
        
        # Compute current energy
        system.calc.calculate(
            system,
            properties=self.sample_properties)
        system_properties = system.calc.results
        current_energy = system_properties['energy']

        # Store initial system properties
        Nsample = self.save_properties(system, Nsample)
        if self.sample_save_trajectory:
            self.write_trajectory(system, trajectory_file)
        
        # Get selectable system atoms
        atom_indices = np.arange(
            system.get_global_number_of_atoms(), dtype=int)
        for constraint in system.constraints:
            if isinstance(constraint, FixAtoms):
                atom_indices = [
                    idx for idx in atom_indices
                    if idx not in constraint.index]
        atom_indices = np.array(atom_indices)
        Natoms = len(atom_indices)
        
        # Iterate over MC steps
        for istep in range(Nsteps):

            # First, randomly select an atom
            selected_atom = atom_indices[np.random.randint(Natoms)]

            # Store current selected atom position
            old_position = system.positions[selected_atom].copy()

            # Propose a new random atom position
            displacement = np.random.uniform(
                -max_displacement, max_displacement, 3)
            system.positions[selected_atom] += displacement

            # Get the potential energy of the new system
            system.calc.calculate(
                system,
                properties=self.sample_properties)
            system_properties = system.calc.results
            new_energy = system_properties['energy']

            # Metropolis acceptance criterion
            threshold = np.exp(-beta*(new_energy - current_energy))

            # If accepted 
            if np.random.rand() < threshold:

                # Set the new position of the atom
                current_energy = new_energy
                Naccept += 1
                
                # Store system properties
                if not Nsample%self.mc_save_interval:
                    Nsample = self.save_properties(system, Nsample)
                    if self.sample_save_trajectory:
                        self.write_trajectory(
                            system, trajectory_file)
            
            # If not accepted
            else:
                
                # Reset the position of the atom
                system.positions[selected_atom] = old_position

        return Nsample
