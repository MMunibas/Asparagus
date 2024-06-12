import os
import queue
import logging
import itertools
import threading
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase.vibrations import Vibrations as Vibrations_ASE
from ase.vibrations.data import VibrationsData
from ase.constraints import FixAtoms
from ase.parallel import world, paropen
from ase import units

from .. import settings
from .. import utils
from .. import sample
from .. import interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['NormalModeScanner', 'NormalModeSampler']


class NormalModeScanner(sample.Sampler):
    """
    Normal Mode Scanning class


    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    nms_harmonic_energy_step: float, optional, default 0.05
        Within the harmonic approximation the initial normal mode
        displacement from the equilibrium positions is scaled to match
        the potential difference with the given energy value.
    nms_energy_limits: (float, list(float)), optional, default 1.0
        Potential energy limit in eV from the initial system conformation
        to which additional normal mode displacements steps are added.
        If one numeric value is give, the energy limit is used as upper
        potential energy limit and the lower limit in case the initial
        system conformation might not be the global minimum.
        If a list with two numeric values are given, the first two are the
        lower and upper potential energy limit, respectively.
    nms_number_of_coupling: int, optional, default 2
        Maximum number of coupled normal mode displacements to sample
        the system conformations.
    nms_limit_of_steps: int, optional, default 10
        Maximum limit of coupled normal mode displacements in one direction
        to sample the system conformations.
    nms_limit_com_shift: float, optional, default 0.1 Angstrom
        Center of mass shift threshold to identify translational normal
        modes from vibrational (and rotational). Normalized Normal modes
        with a center of mass shift larger than the threshold are not
        considered in the normal mode scan.
    nms_save_displacements: bool, optional, default False
        If True, add results of atom displacement calculations from the normal
        mode analysis to the dataset.

    Returns
    -------
    object
        Normal Mode Scanning class object
    """

    # Default arguments for sample module
    sample.Sampler._default_args.update({
        'nms_harmonic_energy_step':     0.05,
        'nms_energy_limits':            1.0,
        'nms_number_of_coupling':       1,
        'nms_limit_of_steps':           10,
        'nms_limit_com_shift':          0.1,
        'nms_save_displacements':       False
        })
    
    # Expected data types of input variables
    sample.Sampler._dtypes_args.update({
        'nms_harmonic_energy_step':     [utils.is_numeric],
        'nms_energy_limits':            [
            utils.is_numeric, utils.is_numeric_array],
        'nms_number_of_coupling':       [utils.is_numeric],
        'nms_limit_of_steps':           [utils.is_numeric],
        'nms_limit_com_shift':          [utils.is_numeric],
        'nms_save_displacements':       [utils.is_bool],
        })

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        nms_harmonic_energy_step: Optional[float] = None,
        nms_energy_limits: Optional[Union[float, List[float]]] = None,
        nms_number_of_coupling: Optional[int] = None,
        nms_limit_of_steps: Optional[int] = None,
        nms_limit_com_shift: Optional[float] = None,
        nms_save_displacements: Optional[bool] = None,
        **kwargs,
    ):

        # Sampler class label
        self.sample_tag = 'nmscan'

        # Initialize parent class
        super().__init__(
            sample_tag=self.sample_tag,
            config=config,
            config_file=config_file,
            **kwargs
            )

        #################################
        # # # Check NMS Class Input # # #
        #################################

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

        # Check sample properties for energy property which is required for
        # normal mode scanning
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')

        # Check potential energy limits
        if utils.is_numeric(self.nms_energy_limits):
            self.nms_energy_limits = [
                -abs(self.nms_energy_limits), abs(self.nms_energy_limits)]

        return

    def get_info(self):
        """
        Obtain information about the Normal Mode Scanning class object.

        Returns
        -------
        dict
            Dictionary with information about the Normal Mode Scanning class
        """

        info = super().get_info()
        info.update({
            'nms_harmonic_energy_step': self.nms_harmonic_energy_step,
            'nms_energy_limits': self.nms_energy_limits,
            'nms_number_of_coupling': self.nms_number_of_coupling,
            'nms_limit_com_shift': self.nms_limit_com_shift,
            'nms_limit_of_steps': self.nms_limit_of_steps,
            'nms_save_displacements': self.nms_save_displacements
            })
        return info

    def run_systems(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        nms_indices: Optional[List[int]] = None,
        nms_exclude_modes: Optional[List[int]] = None,
        nms_frequency_range: Optional[List[Tuple[str, float]]] = None,
        nms_clean: Optional[bool] = True,
        **kwargs,
    ):
        """
        Perform Normal Mode Scanning on the sample system.
        Iterate over systems using 'sample_num_threads' threads.
        
        Parameters
        ----------
        sample_systems_queue: queue.Queue, optional, default None
            Queue object including sample systems or where 'sample_systems' 
            input will be added. If not defined, an empty queue will be 
            assigned.
        nms_indices: list[int], optional, default None
            List of atom indices to include in normal mode analysis.
            If none, indices if a full list of atom indices with length to the
            atom number of the system.
            Atom indices from atoms constraint by FixAtoms are removed from
            index list and the normal mode analysis.
        nms_exclude_modes: list[int], optional, default None
            List of vibrational modes, sorted by wave number, to exclude
            from the sampling procedure.
        nms_frequency_range: list[tuple(str, float)], optional, default None
            Frequency range conditions for normal modes to be included in the
            scan.
        nms_clean: bool, optional, default True
            If True, checkpoint files for atom displacement calculations
            in {sample_directory}/vib_{isample} will be deleted.
            Else, results from available  checkpoint files will be used.
        """
        
        # Check sample system queue
        if sample_systems_queue is None:
            sample_systems_queue = queue.Queue()
        
        # Optimize sample systems or take as normal mode analysis input
        if self.sample_systems_optimize:
            
            # Add stop flag
            for _ in range(self.sample_num_threads):
                sample_systems_queue.put('stop')
            
            # Initialize continuation flag
            self.thread_keep_going = np.array(
                [True for ithread in range(self.sample_num_threads)],
                dtype=bool
                )

            # Initialize optimized sample system into queue
            sample_input_queue = queue.Queue()
            
            if self.sample_num_threads == 1:

                # Run sample system optimization
                self.run_optimization(
                    sample_systems_queue=sample_systems_queue,
                    sample_optimzed_queue=sample_input_queue)
            
            else:

                # Create threads for sample system optimization
                threads = [
                    threading.Thread(
                        target=self.run_optimization,
                        kwargs={
                            'sample_systems_queue': sample_systems_queue,
                            'sample_optimzed_queue': sample_input_queue,
                            'ithread': ithread}
                        )
                    for ithread in range(self.sample_num_threads)]

                # Start threads
                for thread in threads:
                    thread.start()

                # Wait for threads to finish
                for thread in threads:
                    thread.join()

        else:
            
            # Set sample system queue as optimized sample system queue
            sample_input_queue = sample_systems_queue

        # Run normal mode scanning
        while not sample_input_queue.empty():
            self.run_system(
                sample_input_queue,
                nms_indices,
                nms_exclude_modes,
                nms_frequency_range,
                nms_clean,
                **kwargs)
        
        return
    
    def run_system(
        self,
        sample_systems_queue: queue.Queue,
        nms_indices: List[int],
        nms_exclude_modes: List[int],
        nms_frequency_range: List[Tuple[str, float]],
        nms_clean: bool,
        **kwargs
    ):
        """
        Perform Normal Mode Scanning on the sample system.

        Parameters
        ----------
        sample_systems_queue: queue.Queue
            Queue object including sample systems.
        """

        # Initialize stored sample counter
        Nsample = 0
        
        # Initialize normal mode analysis queue
        sample_calculate_queue = queue.Queue()
        
        # Get sample system for normal mode analysis
        (system, isample, source, index) = sample_systems_queue.get()

        # Print sampler info
        msg = "INFO:\nStart normal mode scanning of the system "
        msg += f"from '{source}' of index {index:d}.\n"
        logger.info(msg)
        
        # Get non-fixed atoms indices
        if nms_indices is None:
            atom_indices = np.arange(
                system.get_global_number_of_atoms(), dtype=int)
        else:
            atom_indices = np.array(nms_indices, dtype=int)
        for constraint in system.constraints:
            if isinstance(constraint, FixAtoms):
                atom_indices = [
                    idx for idx in atom_indices
                    if idx not in constraint.index]
        atom_indices = np.array(atom_indices)
        
        # Prepare system parameter
        Natoms = len(atom_indices)
        Nmodes = 3*Natoms

        # Initialize Vibration class
        system_vibrations = Vibrations_Asparagus(
            self,
            system,
            self.sample_calculator,
            self.sample_calculator_args,
            self.nms_save_displacements,
            indices=atom_indices,
            name=os.path.join(self.sample_directory, f"vib_{isample:d}"),
            **kwargs
            )
        if nms_clean:
            system_vibrations.clean()
        
        # Add jobs to calculation queue
        system_vibrations.add_calculations(
            sample_calculate_queue)

        # Add stop flag
        for _ in range(self.sample_num_threads):
            sample_calculate_queue.put('stop')
        
        # Initialize continuation flag
        self.thread_keep_going = np.array(
            [True for ithread in range(self.sample_num_threads)],
            dtype=bool
            )
        
        if self.sample_num_threads == 1:
            
            # Run job calculations
            system_vibrations.run(
                sample_calculate_queue,
                sample_calculator=self.sample_calculator,
                sample_calculator_args=self.sample_calculator_args)
        
        else:

            # Create threads for job calculations
            threads = [
                threading.Thread(
                    target=system_vibrations.run,
                    args=(sample_calculate_queue, ),
                    kwargs={
                        'sample_calculator': self.sample_calculator,
                        'sample_calculator_args': self.sample_calculator_args,
                        'ithread': ithread}
                    )
                for ithread in range(self.sample_num_threads)]

            # Start threads
            for thread in threads:
                thread.start()

            # Wait for threads to finish
            for thread in threads:
                thread.join()

        # Print sampler info
        msg = "INFO:\nNormal mode analysis calculation completed for "
        msg += f"the system from '{source}' of index {index:d}.\n"
        logger.info(msg)
        
        # Finish normal mode analysis and diagonalize Hessian matrix
        system_vibrations.summary(**kwargs)
        
        # Get initial (equilibrium) energy of the sampling system
        system_initial_results = system_vibrations.get_initial_results()
        
        # (Trans. + Rot. + ) Vibrational frequencies in cm**-1
        system_frequencies = system_vibrations.get_frequencies()

        # (Trans. + Rot. + ) Vibrational modes normalized to 1
        system_modes = np.array([
            system_vibrations.get_mode(imode)[atom_indices].reshape(Natoms, 3)
            / np.sqrt(np.sum(
                system_vibrations.get_mode(imode)[atom_indices].reshape(
                    Natoms, 3)**2))
            for imode in range(Nmodes)])

        # Reduced mass per mode (in amu)
        system_redmass = np.array([
            1./np.sum(
                system_modes[imode]**2
                / system.get_masses()[atom_indices].reshape(Natoms, 1))
            for imode in range(Nmodes)])

        # Force constant per mode (in eV/Angstrom**2)
        system_forceconst = (
            4.0*np.pi**2*(np.abs(system_frequencies)*1.e2*units._c)**2
            * system_redmass*units._amu*units.J*1.e-20)

        # Compute and store equilibrium positions and center of mass,
        # moments of inertia and principle axis of inertia
        system_init_positions = system.get_positions()
        system_init_com = system[atom_indices].get_center_of_mass()

        # Compute and compare same quantities for displaced system
        system_com_shift = np.zeros(Nmodes, dtype=float)
        
        for imode, mode in enumerate(system_modes):

            # COM shift
            system_mode_positions = system_init_positions.copy()
            system_mode_positions[atom_indices] += mode
            system.set_positions(system_mode_positions, apply_constraint=False)
            system_displ_com = system[atom_indices].get_center_of_mass()
            system_com_shift[imode] = np.sqrt(np.sum(
                (system_init_com - system_displ_com)**2))

        # Reset system positions
        system.set_positions(system_init_positions)

        # Vibrational modes are assumed with center of mass shifts smaller
        # than 1.e-1 Angstrom displacement
        system_vib_modes = system_com_shift < self.nms_limit_com_shift
        
        # Apply exclusion list if defined
        if nms_exclude_modes is not None:

            for imode in nms_exclude_modes:
                if imode < len(system_vib_modes):
                    system_vib_modes[imode] = False
                else:
                    logger.warning(
                        f"WARNING:\nVibrational mode {imode:d} in the "
                        + "exclusion list is larger than the number of "
                        + "vibrational modes!")

        if nms_frequency_range is not None:
            
            # Initially include all modes
            include_modes = np.ones_like(system_vib_modes)
            
            # Iterate over all exclusion conditions
            for (condition, frequency) in nms_frequency_range:
                
                # Compare absolute if requested
                if '||' in condition:
                    comp_frequencies = np.abs(system_frequencies)
                else:
                    comp_frequencies = system_frequencies
                
                # Modes are still include if conditions are matched
                if '<=' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies <= frequency)
                elif '>=' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies >= frequency)
                elif '<' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies < frequency)
                elif '>' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies >= frequency)
                else:
                    raise SyntaxError(
                        f"Normal mode selection condition '{condition}' in "
                        + "'nms_frequency_range' selection input is not "
                        + "recognized! Choose between ('<', '<=', '>=', '>') "
                        + "or for comparing absolute frequencies "
                        + "('<||', '<=||', '>=||', '>||')")

            # Combine normal mode exclusion list
            system_vib_modes = np.logical_and(system_vib_modes, include_modes)

        # Displacement factor for energy step (in eV)
        system_displfact = np.sqrt(
            2.*self.nms_harmonic_energy_step/system_forceconst)

        # Add normal mode analysis results to log file
        msg = "\nStart Normal Mode Scanning at system: "
        if self.sample_data_file is None:
            msg += f"{system.get_chemical_formula():s}\n"
        else:
            msg += f"{self.sample_data_file:s}\n"
        msg += " Index | Frequency (cm**-1) | Vibration (CoM displacemnt)\n"
        msg += "---------------------------------------------------------\n"
        for ivib, freq in enumerate(system_frequencies):
            msg += f" {ivib + 1:5d} | {freq:18.2f} |"
            if system_vib_modes[ivib]:
                msg += f"   x   ({system_com_shift[ivib]:2.1e})\n"
            else:
                msg += f"       ({system_com_shift[ivib]:2.1e})\n"
        with open(self.sample_log_file.format(isample), 'a') as flog:
            flog.write(msg)
        
        # Iterate over number of normal mode combinations
        vib_modes = np.where(system_vib_modes)[0]
        
        # Initialize normal mode combination queue
        sample_calculate_queue = queue.Queue()
        
        # Prepare normal mode combination jobs
        irun = 0
        for icomp in range(1, self.nms_number_of_coupling + 1):

            # Prepare sign combinations
            all_signs = np.array(list(
                itertools.product((-1, 1), repeat=icomp)))

            # Prepare step size list
            steps = np.arange(1, self.nms_limit_of_steps + 1, 1)
            all_steps = np.array(list(
                itertools.product(steps, repeat=icomp)))
            Nsteps = all_steps.shape[0]

            # Iterate over vib. normal mode indices and their combinations
            for imodes in itertools.combinations(vib_modes, icomp):

                # Iterate over sign combinations
                for isign, signs in enumerate(all_signs):
                    
                    # Add mode combination job parameters
                    sample_calculate_queue.put(
                        (isample, irun, icomp, imodes, isign, signs))
                    irun += 1

        # Add stop flag
        for _ in range(self.sample_num_threads):
            sample_calculate_queue.put('stop')
        
        # Initialize continuation flag
        self.thread_keep_going = np.array(
            [True for ithread in range(self.sample_num_threads)],
            dtype=bool
            )

        # Run 
        Nsamples = [0]*self.sample_num_threads
        if self.sample_num_threads == 1:
                
            # Run job calculations
            self.run_scan(
                system,
                sample_calculate_queue,
                system_initial_results['energy'],
                system_displfact,
                system_modes,
                atom_indices,
                Nsamples)
        
        else:

            # Create threads for job calculations
            threads = [
                threading.Thread(
                    target=self.run_scan,
                    args=(
                        system,
                        sample_calculate_queue, 
                        system_initial_results['energy'],
                        system_displfact,
                        system_modes,
                        atom_indices,
                        Nsamples),
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
                
        # Print sampling info
        for ithread in range(self.sample_num_threads):

            msg = f"Sampling method '{self.sample_tag:s}' complete for "
            msg += f"system of index {index:d} from '{source}!'\n"
            if Nsamples[ithread] == 0:
                msg += f"No samples written to "
            if Nsamples[ithread] == 1:
                msg += f"{Nsamples[ithread]:d} sample written to "
            else:
                msg += f"{Nsamples[ithread]:d} samples written to "
            msg += f"'{self.sample_data_file:s}'.\n"
            logger.info(f"INFO:\n{msg:s}")

    def run_scan(
        self,
        sample_system: ase.Atoms,
        sample_calculate_queue: queue.Queue,
        system_initial_potential: float,
        system_displfact: List[float],
        system_modes: List[float],
        atom_indices: List[float],
        Nsamples: List[int],
        ithread: Optional[int] = None
    ):
        """
        Run normal mode scanning for normal mode combinations.
        
        Parameters
        ----------
        sample_system: ase.Atoms
            Initial/equilibrium sample system to apply on the normal mode
            combinations.
        sample_systems_queue: queue.Queue
            Queue containing normal mode combination parameters
        ithread: int, optional, default None
            Thread number
        """
        
        # Initialize stored sample counter
        Nsample = 0
        
        # Get ASE calculator
        ase_calculator, ase_calculator_tag = (
            interface.get_ase_calculator(
                self.sample_calculator,
                self.sample_calculator_args,
                ithread=ithread)
            )
        
        # Assign calculator
        system = sample_system.copy()
        system.set_calculator(ase_calculator)

        # Store equilibrium positions
        system_initial_positions = system.get_positions()
        
        while self.keep_going(ithread):
            
            # Get sample parameters or wait
            sample = sample_calculate_queue.get()

            # Check for stop flag
            if sample == 'stop':
                self.thread_keep_going[ithread] = False
                continue
            
            # Extract normal mode combination parameters
            (isample, irun, icomp, imodes, isign, signs) = sample
            
            # Prepare sign combinations
            all_signs = np.array(list(
                itertools.product((-1, 1), repeat=icomp)))

            # Prepare step size list
            steps = np.arange(1, self.nms_limit_of_steps + 1, 1)
            all_steps = np.array(list(
                itertools.product(steps, repeat=icomp)))
            Nsteps = all_steps.shape[0]
            
            # Iterate through steps
            istep = 0
            done = False
            while not done:

                # Get current step size
                step_size = np.array(all_steps[istep])

                # Get normal mode elongation step
                current_step = np.array(signs)*step_size

                # Set elongation step on initial system positions
                current_step_positions = system_initial_positions.copy()
                for imode, modei in enumerate(imodes):
                    current_step_positions[atom_indices] += (
                        system_displfact[modei]*current_step[imode]
                        * system_modes[modei])
                system.set_positions(
                    current_step_positions,
                    apply_constraint=False)

                # Compute observables and check potential threshold
                try:

                    ase_calculator.calculate(
                        system,
                        properties=self.sample_properties,
                        system_changes=system.calc.implemented_properties)
                    current_properties = system.calc.results
                    current_potential = current_properties['energy']

                    converged = True

                    if current_potential < system_initial_potential:
                        threshold_reached = (
                            (
                                current_potential 
                                - system_initial_potential
                            ) < self.nms_energy_limits[0])
                    else:
                        threshold_reached = (
                            (
                                current_potential 
                                - system_initial_potential
                            ) > self.nms_energy_limits[1])

                except ase.calculators.calculator.CalculationFailed:

                    converged = False
                    threshold_reached = True

                # Add to dataset
                if converged:
                    Nsample = self.save_properties(system, Nsample)

                # Attach to trajectory
                if converged and self.sample_save_trajectory:
                    self.write_trajectory(
                        system, self.sample_trajectory_file.format(isample))

                # Check energy threshold
                if threshold_reached:

                    # Check for next suitable step size index and avoid
                    # combination of step sizes which were already
                    # above the energy threshold before.
                    istep += 1
                    for jstep in range(istep, Nsteps):
                        if np.any(
                            np.array(all_steps[jstep]) < step_size
                        ):
                            istep = jstep
                            break
                    else:
                        done = True

                else:

                    # Increment step size index
                    istep += 1

                # Check step size progress
                if done or istep >= Nsteps:

                    # Update log file
                    msg = "Vib. modes: ("
                    for imode, isign in zip(imodes, signs):
                        if isign > 0:
                            msg += f"+{imode + 1:d}, "
                        else:
                            msg += f"-{imode + 1:d}, "
                    msg += f") - {istep:4d} steps added\n"
                    log_file = self.sample_log_file.format(isample)
                    with open(log_file, 'a') as flog:
                        flog.write(msg)

                    # Set flag in case maximum Nsteps is reached
                    done = True
        
        # Set number of stored samples
        if ithread is None:
            Nsamples[0] = Nsample
        else:
            Nsamples[ithread] = Nsample

        return


class NormalModeSampler(sample.Sampler):
    """
    Normal Mode Sampling class.

    This is the simple version of the normal mode class as implemented in:
    Chem. Sci., 2017, 8, 3192-3203

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    nms_temperature: float, optional, default 300
        Temperature in Kelvin to sample the normal modes.
    nms_nsamples: int, optional, default 100
        Number of samples to generate.
    nms_limit_com_shift: float, optional, default 0.1 Angstrom
        Center of mass shift threshold to identify translational normal
        modes from vibrational (and rotational). Normalized Normal modes
        with a center of mass shift larger than the threshold are not
        considered in the normal mode scan.
    nms_save_displacements: bool, optional, default False
        If True, add results of atom displacement calculations from the normal
        mode analysis to the dataset.
    
    Returns
    -------
    object
        Normal Mode Sampler class object
    """
    
    # Default arguments for sample module
    sample.Sampler._default_args.update({
        'nms_temperature':              300.0,
        'nms_nsamples':                 100,
        'nms_limit_com_shift':          0.1,
        'nms_save_displacements':       False,
        })
    
    # Expected data types of input variables
    sample.Sampler._dtypes_args.update({
        'nms_temperature':              [utils.is_numeric],
        'nms_nsamples':                 [utils.is_integer],
        'nms_limit_com_shift':          [utils.is_numeric],
        'nms_save_displacements':       [utils.is_bool],
        })

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        nms_temperature: Optional[float] = None,
        nms_nsamples: Optional[int] = None,
        nms_limit_com_shift: Optional[float] = None,
        nms_save_displacements: Optional[bool] = None,
        **kwargs
    ):

        # Sampler class label
        self.sample_tag = 'nmsmpl'

        # Initialize parent class
        super().__init__(
            sample_tag=self.sample_tag,
            config=config,
            config_file=config_file,
            **kwargs
            )

        #################################
        # # # Check NMS Class Input # # #
        #################################

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

        # Check sample properties for energy and forces property which is 
        # required for normal mode scanning
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')

        return

    def get_info(self):
        """
        Obtain information about the Normal Mode Sampling class object.

        Returns
        -------
        dict
            Dictionary with information about the Normal Mode Sampling class
        """
        
        info = super().get_info()
        info.update({
            'nms_temperature': self.nms_temperature,
            'nms_nsamples': self.nms_nsamples,
            'nms_limit_com_shift': self.nms_limit_com_shift,
            'nms_save_displacements': self.nms_save_displacements,
            })
        
        return info

    def run_systems(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        nms_indices: Optional[List[int]] = None,
        nms_exclude_modes: Optional[List[int]] = None,
        nms_frequency_range: Optional[List[Tuple[str, float]]] = None,
        nms_clean: Optional[bool] = True,
        **kwargs,
    ):
        """
        Perform Normal Mode Scanning on the sample system.
        Iterate over systems using 'sample_num_threads' threads.
        
        Parameters
        ----------
        sample_systems_queue: queue.Queue, optional, default None
            Queue object including sample systems or where 'sample_systems' 
            input will be added. If not defined, an empty queue will be 
            assigned.
        nms_indices: list[int], optional, default None
            List of atom indices to include in normal mode analysis.
            If none, indices if a full list of atom indices with length ot the
            atom number of the system.
            Atom indices from atoms constraint by FixAtoms are removed from
            index list and the normal mode analysis.
        nms_exclude_modes: list[int], optional, default None
            List of vibrational modes, sorted by wave number, to exclude
            from the sampling procedure.
        nms_frequency_range: list[tuple(str, float)], optional, default None
            Frequency range conditions for normal modes to be included in the
            sampling.
        nms_clean: bool, optional, default True
            If True, checkpoint files for atom displacement calculations
            in {sample_directory}/vib_{isample} will be deleted.
            Else, results from available  checkpoint files will be used.
        """

        # Check sample system queue
        if sample_systems_queue is None:
            sample_systems_queue = queue.Queue()

        # Optimize sample systems or take as normal mode analysis input
        if self.sample_systems_optimize:
            
            # Add stop flag
            for _ in range(self.sample_num_threads):
                sample_systems_queue.put('stop')
            
            # Initialize continuation flag
            self.thread_keep_going = np.array(
                [True for ithread in range(self.sample_num_threads)],
                dtype=bool
                )
            
            # Initialize optimized sample system into queue
            sample_input_queue = queue.Queue()
            
            if self.sample_num_threads == 1:

                # Run sample system optimization
                self.run_optimization(
                    sample_systems_queue=sample_systems_queue,
                    sample_optimzed_queue=sample_input_queue)
            
            else:

                # Create threads for sample system optimization
                threads = [
                    threading.Thread(
                        target=self.run_optimization,
                        kwargs={
                            'sample_systems_queue': sample_systems_queue,
                            'sample_optimzed_queue': sample_input_queue,
                            'ithread': ithread}
                        )
                    for ithread in range(self.sample_num_threads)]

                # Start threads
                for thread in threads:
                    thread.start()

                # Wait for threads to finish
                for thread in threads:
                    thread.join()

        else:
            
            # Set sample system queue as optimized sample system queue
            sample_input_queue = sample_systems_queue

        # Run normal mode sampling
        while not sample_input_queue.empty():
            self.run_system(
                sample_input_queue,
                nms_indices,
                nms_exclude_modes,
                nms_frequency_range,
                nms_clean,
                **kwargs)
        
        return

    def run_system(
        self,
        sample_systems_queue: queue.Queue,
        nms_indices: List[int],
        nms_exclude_modes: List[int],
        nms_frequency_range: List[Tuple[str, float]],
        nms_clean: bool,
        **kwargs
    ):
        """
        Perform Normal Mode Sampling on the sample system.

        Parameters
        ----------
        sample_systems_queue: queue.Queue
            Queue object including sample systems.
        """

        # Initialize stored sample counter
        self.Nsample = 0

        # Initialize normal mode analysis queue
        sample_calculate_queue = queue.Queue()

        # Get sample system for normal mode analysis
        (system, isample, source, index) = sample_systems_queue.get()

        # Print sampler info
        msg = "INFO:\nStart normal mode sampling of the system "
        msg += f"from '{source}' of index {index:d}.\n"
        logger.info(msg)
        
        # Get non-fixed atoms indices
        if nms_indices is None:
            atom_indices = np.arange(
                system.get_global_number_of_atoms(), dtype=int)
        else:
            atom_indices = np.array(nms_indices, dtype=int)
        for constraint in system.constraints:
            if isinstance(constraint, FixAtoms):
                atom_indices = [
                    idx for idx in atom_indices
                    if idx not in constraint.index]
        atom_indices = np.array(atom_indices)
        
        # Prepare system parameter
        Natoms = len(atom_indices)
        Nmodes = 3*Natoms

        # Initialize Vibration class
        system_vibrations = Vibrations_Asparagus(
            self,
            system,
            self.sample_calculator,
            self.sample_calculator_args,
            self.nms_save_displacements,
            indices=atom_indices,
            name=os.path.join(self.sample_directory, f"vib_{isample:d}"),
            **kwargs
            )
        if nms_clean:
            system_vibrations.clean()
        
        # Add jobs to calculation queue
        system_vibrations.add_calculations(
            sample_calculate_queue)

        # Add stop flag
        for _ in range(self.sample_num_threads):
            sample_calculate_queue.put('stop')
        
        # Initialize continuation flag
        self.thread_keep_going = np.array(
            [True for ithread in range(self.sample_num_threads)],
            dtype=bool
            )

        if self.sample_num_threads == 1:
            
            # Run job calculations
            system_vibrations.run(
                sample_calculate_queue,
                sample_calculator=self.sample_calculator,
                sample_calculator_args=self.sample_calculator_args)
        
        else:

            # Create threads for job calculations
            threads = [
                threading.Thread(
                    target=system_vibrations.run,
                    args=(sample_calculate_queue, ),
                    kwargs={
                        'sample_calculator': self.sample_calculator,
                        'sample_calculator_args': self.sample_calculator_args,
                        'ithread': ithread}
                    )
                for ithread in range(self.sample_num_threads)]

            # Start threads
            for thread in threads:
                thread.start()

            # Wait for threads to finish
            for thread in threads:
                thread.join()

        # Print sampler info
        msg = "INFO:\nNormal mode analysis calculation completed for "
        msg += f"the system from '{source}' of index {index:d}.\n"
        logger.info(msg)
        
        # Finish normal mode analysis and diagonalize Hessian matrix
        system_vibrations.summary(**kwargs)
        
        # Get initial (equilibrium) energy of the sampling system
        system_initial_results = system_vibrations.get_initial_results()
        
        # Store equilibrium positions
        system_initial_positions = system.get_positions()

        # (Trans. + Rot. + ) Vibrational frequencies in cm**-1
        system_frequencies = system_vibrations.get_frequencies()

        # (Trans. + Rot. + ) Vibrational modes normalized to 1
        system_modes = np.array([
            system_vibrations.get_mode(imode)[atom_indices].reshape(Natoms, 3)
            / np.sqrt(np.sum(
                system_vibrations.get_mode(imode)[atom_indices].reshape(
                    Natoms, 3)**2))
            for imode in range(Nmodes)])

        # Reduced mass per mode (in amu)
        system_redmass = np.array([
            1./np.sum(
                system_modes[imode]**2
                / system.get_masses()[atom_indices].reshape(Natoms, 1))
            for imode in range(Nmodes)])

        # Force constant per mode (in eV/Angstrom**2)
        system_forceconst = (
            4.0*np.pi**2*(np.abs(system_frequencies)*1.e2*units._c)**2
            * system_redmass*units._amu*units.J*1.e-20)

        # Compute and store equilibrium positions and center of mass,
        # moments of inertia and principle axis of inertia
        system_init_positions = system.get_positions()
        system_init_com = system[atom_indices].get_center_of_mass()

        # Compute and compare same quantities for displaced system
        system_com_shift = np.zeros(Nmodes, dtype=float)
        
        for imode, mode in enumerate(system_modes):

            # COM shift
            system_mode_positions = system_init_positions.copy()
            system_mode_positions[atom_indices] += mode
            system.set_positions(system_mode_positions, apply_constraint=False)
            system_displ_com = system[atom_indices].get_center_of_mass()
            system_com_shift[imode] = np.sqrt(np.sum(
                (system_init_com - system_displ_com)**2))

        # Vibrational modes are assumed with center of mass shifts smaller
        # than 1.e-1 Angstrom displacement
        system_vib_modes = system_com_shift < self.nms_limit_com_shift

        # Apply exclusion list if defined
        if nms_exclude_modes is not None:

            for imode in nms_exclude_modes:
                if imode < len(system_vib_modes):
                    system_vib_modes[imode] = False
                else:
                    logger.warning(
                        f"WARNING:\nVibrational mode {imode:d} in the "
                        + "exclusion list is larger than the number of "
                        + "vibrational modes!")

        if nms_frequency_range is not None:
            
            # Initially include all modes
            include_modes = np.ones_like(system_vib_modes)
            
            # Iterate over all exclusion conditions
            for (condition, frequency) in nms_frequency_range:
                
                # Compare absolute if requested
                if '||' in condition:
                    comp_frequencies = np.abs(system_frequencies)
                else:
                    comp_frequencies = system_frequencies
                
                # Modes are still include if conditions are matched
                if '<=' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies <= frequency)
                elif '>=' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies >= frequency)
                elif '<' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies < frequency)
                elif '>' in condition:
                    include_modes = np.logical_and(
                        include_modes, 
                        comp_frequencies >= frequency)
                else:
                    raise SyntaxError(
                        f"Normal mode selection condition '{condition}' in "
                        + "'nms_frequency_range' selection input is not "
                        + "recognized! Choose between ('<', '<=', '>=', '>') "
                        + "or for comparing absolute frequencies "
                        + "('<||', '<=||', '>=||', '>||')")

            # Combine normal mode exclusion list
            system_vib_modes = np.logical_and(system_vib_modes, include_modes)

        # Add normal mode analysis results to log file
        msg = "\nStart Normal Mode Sampling at system: "
        if self.sample_data_file is None:
            msg += f"{system.get_chemical_formula():s}\n"
        else:
            msg += f"{self.sample_data_file:s}\n"
        msg += " Index | Frequency (cm**-1) | Vibration (CoM displacemnt)\n"
        msg += "---------------------------------------------------------\n"
        for ivib, freq in enumerate(system_frequencies):
            msg += f" {ivib + 1:5d} | {freq:18.2f} |"
            if system_vib_modes[ivib]:
                msg += f"   x   ({system_com_shift[ivib]:2.1e})\n"
            else:
                msg += f"       ({system_com_shift[ivib]:2.1e})\n"
        with open(self.sample_log_file.format(isample), 'a') as flog:
            flog.write(msg)

        # Initialize normal mode sampling queue
        sample_calculate_queue = queue.Queue()
        
        # Initialize continuation flag
        self.thread_keep_going = np.array(
            [True for ithread in range(self.sample_num_threads)],
            dtype=bool
            )

        # Start calculation run
        Nsamples = [0]*self.sample_num_threads
        if self.sample_num_threads == 1:
            
            # Add calculation jobs
            for irun in range(self.nms_nsamples):
                sample_position = self.get_sample_positions(
                    system_initial_positions,
                    Nmodes, 
                    system_modes, 
                    system_redmass, 
                    system_forceconst,
                    system_vib_modes,
                    self.nms_temperature)
                sample_calculate_queue.put((isample, irun, sample_position))
            
            # Add stop flag
            for _ in range(self.sample_num_threads):
                sample_calculate_queue.put('stop')
                
            # Run job calculations
            self.run_sampling(
                system,
                sample_calculate_queue,
                Nsamples)
        
        else:

            # Create threads for job calculations
            threads = [
                threading.Thread(
                    target=self.run_sampling,
                    args=(
                        system,
                        sample_calculate_queue,
                        Nsamples),
                    kwargs={
                        'ithread': ithread}
                    )
                for ithread in range(self.sample_num_threads)]

            # Start threads
            for thread in threads:
                thread.start()

            # Add calculation jobs
            for irun in range(self.nms_nsamples):
                sample_position = self.get_sample_positions(
                    system_initial_positions,
                    Nmodes, 
                    system_modes, 
                    system_redmass, 
                    system_forceconst,
                    system_vib_modes,
                    self.nms_temperature)
                sample_calculate_queue.put((isample, irun, sample_position))
            
            # Add stop flag
            for _ in range(self.sample_num_threads):
                sample_calculate_queue.put('stop')

            # Wait for threads to finish
            for thread in threads:
                thread.join()

        # Print sampling info
        for ithread in range(self.sample_num_threads):

            msg = f"Sampling method '{self.sample_tag:s}' complete for "
            msg += f"system of index {index:d} from '{source}!'\n"
            if Nsamples[ithread] == 0:
                msg += f"No samples written to "
            if Nsamples[ithread] == 1:
                msg += f"{Nsamples[ithread]:d} sample written to "
            else:
                msg += f"{Nsamples[ithread]:d} samples written to "
            msg += f"'{self.sample_data_file:s}'.\n"
            logger.info(f"INFO:\n{msg:s}")

    def run_sampling(
        self,
        sample_system: ase.Atoms,
        sample_calculate_queue: queue.Queue,
        Nsamples: List[int],
        ithread: Optional[int] = None
    ):
        """
        Run normal mode scanning for normal mode combinations.
        
        Parameters
        ----------
        sample_system: ase.Atoms
            Initial/equilibrium sample system to apply on the normal mode
            combinations.
        sample_systems_queue: queue.Queue
            Queue containing normal mode combination parameters
        ithread: int, optional, default None
            Thread number
        """

        # Initialize stored sample counter
        Nsample = 0
        
        # Get ASE calculator
        ase_calculator, ase_calculator_tag = (
            interface.get_ase_calculator(
                self.sample_calculator,
                self.sample_calculator_args,
                ithread=ithread)
            )
        
        # Assign calculator
        system = sample_system.copy()
        system.set_calculator(ase_calculator)

        while self.keep_going(ithread):
            
            # Get sample parameters or wait
            sample = sample_calculate_queue.get()

            # Check for stop flag
            if sample == 'stop':
                self.thread_keep_going[ithread] = False
                continue
            
            # Extract normal mode combination parameters
            (isample, irun, sample_positions) = sample
            
            # Set sample positions to calculate
            system.set_positions(sample_positions)
            
            # Compute observables and check potential threshold
            try:

                ase_calculator.calculate(
                    system,
                    properties=self.sample_properties,
                    system_changes=system.calc.implemented_properties)
                converged = True
            
            except ase.calculators.calculator.CalculationFailed:

                converged = False

            # Add to dataset
            if converged:
                Nsample = self.save_properties(system, Nsample)

            # Attach to trajectory
            if converged and self.sample_save_trajectory:
                self.write_trajectory(
                    system, self.sample_trajectory_file.format(isample))

        # Set number of stored samples
        if ithread is None:
            Nsamples[0] = Nsample
        else:
            Nsamples[ithread] = Nsample

        return
        
    def get_sample_positions(
        self, 
        initial_positions, 
        Nmodes: int,
        vib_modes: List[float],
        vib_masses: List[float], 
        vib_fcnts: List[float],
        vib_include: List[bool],
        vib_temp: float
    ):
        """
        Create the new coordinates for the system.

        Parameters
        ----------
        initial_positions: np.ndarray(float)
            Sample system position to add normal mode displacement
        Nmodes: int
            Normal modes of the system
        vib_modes: np.ndarray(float)
            Nomralized vibrational modes
        vib_masses: np.ndarray(float)
            Reduced mass of the normal modes
        vib_fcnts: np.ndarray(float)
            Force constant per normal mode (in eV/Angstrom**2)
        vib_include: np.ndarray(bool)
            Selection of normal mode to include in sampling
        vib_temp: float
            Temperature in Kelvin to sample the normal modes.

        Returns
        -------
        np.ndarray
            Sample coordinates for the system
        """

        Rx = self.R(Nmodes, vib_fcnts, vib_temp)
        sample_positions = initial_positions.copy()

        for imode, mode_i in enumerate(vib_modes):
            if vib_include[imode]:
                disp_i = vib_masses[imode]*Rx[imode]*mode_i
                sample_positions += disp_i

        return sample_positions

    def R(self, Nmodes, fcnts, temp):
        """
        Made a random displacement for each of the modes in the system.
        The displacements follow a Bernoulli distribution with $P=0.5$

        The value R is given by:
        .. :math:

            R_{i} = \pm \sqrt{\dfrac{3c_{i}N_{a}k_{b}T}{K_{i}}}

        Parameters
        ----------
        Nmodes: int
            Number of modes in the system
        fcnts: np.ndarray(float)
            Force constant per mode (in eV/Angstrom**2)
        temp: float
            Temperature in Kelvin to sample the normal modes.

        Returns
        -------
        np.ndarray
            Array of displacements for each mode in the system
        """

        random_num = np.random.uniform(size=Nmodes)**2
        sign = [-1 if i < 0.5 else 1 for i in random_num]
        fix_fcnts = [0.05 if i < 0.05 else i for i in fcnts]
        R = []

        for i, j in enumerate(fix_fcnts):
            R_i = sign[i] * np.sqrt((3*random_num[i]*units.kB*temp)/j)
            R.append(R_i)

        return np.array(R)


class Vibrations_Asparagus(Vibrations_ASE):
    """
    Calculate vibrational modes using finite difference or calculated Hessian 
    directly.
    """
    
    def __init__(
        self,
        sampler,
        sample_atoms,
        sample_calculator,
        sample_calculator_args,
        nms_save_displacements,
        indices=None, 
        name='vib',
        delta=0.01, 
        nfree=2,
        **kwargs
    ):
        
        # Assign calculator to atoms object
        sample_atoms = sampler.assign_calculator(
            sample_atoms,
            sample_calculator=sample_calculator,
            sample_calculator_args=sample_calculator_args)
        
        # Initialize ASE Vibrations parent class
        super().__init__(
            sample_atoms, 
            indices=indices, 
            name=name, 
            delta=delta, 
            nfree=nfree
            )
        
        # Check for Hessian in calculator implemented properties
        if 'hessian' in self.calc.implemented_properties:
            self.hessian_avail = True
        else:
            self.hessian_avail = False
        
        # Assign class parameter
        self.sampler = sampler
        self.nms_save_displacements = nms_save_displacements

        # Initialize equilibrium/initial sample system result dictionary
        self.eq_results = {}

        return

    def add_calculations(
        self,
        calculate_queue: queue.Queue,
    ):
        """
        Add vibration calculations to job queue.
        
        Parameters
        ----------
        calculate_queue : queue.Queue
            Sample system queue
        """
        
        # Check for writing rights and old result files
        if not self.cache.writable:
            raise RuntimeError(
                "Cannot run calculation. "
                + "Cache must be removed or split in order "
                + "to have only one sort of data structure at a time.")
        self._check_old_pickles()
        
        # Add jobs to compute Hessian or forces at atom displacements to the
        # job queue
        if self.hessian_avail:
            
            displacements = self.displacements()
            eq_disp = next(displacements)
            assert eq_disp.name == 'eq'
            calculate_queue.put(
                (self.atoms, eq_disp.name))
            
        else:
            
            for disp, atoms in self.iterdisplace(inplace=False):
                calculate_queue.put(
                    (atoms, disp.name))
                
        return calculate_queue
    
    def run(
        self,
        calculate_queue: queue.Queue,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        ithread: Optional[int] = None,
    ):
        """
        Run the vibration calculations.
        
        Parameters
        ----------
        calculate_queue : queue.Queue
            Sample system queue
        sample_calculator : (str, object), optional, default None
            ASE calculator object or string of an ASE calculator class
            name to assign to the sample systems
        sample_calculator_args : dict, optional, default None
            Dictionary of keyword arguments to initialize the ASE
            calculator
        ithread: int, optional, default None
            Thread number
        """

        # Get ASE calculator
        ase_calculator, ase_calculator_tag = (
            interface.get_ase_calculator(
                sample_calculator,
                sample_calculator_args,
                ithread=ithread)
            )

        while self.sampler.keep_going(ithread):
        
            # Get job parameters or wait
            sample = calculate_queue.get()

            # Check for stop flag
            if sample == 'stop':
                self.sampler.thread_keep_going[ithread] = False
                continue

            # Get job to run
            (atoms, name) = sample
            
            # Assign calculator
            atoms.set_calculator(ase_calculator)

            # Run job
            with self.cache.lock(name) as handle:

                # Read results if result file exist
                if handle is None:

                    results = self.cache[name]
                    atoms.calc.results = results

                # Or run calculation
                else:

                    # Compute essential results
                    results = {}
                    if self.hessian_avail:
                        results['hessian'] = atoms.get_hessian()
                    else:
                        results['forces'] = atoms.get_forces()
                    
                    # Add additional results
                    for prop, result in ase_calculator.results.items():
                        results[prop] = result

                    if world.rank == 0:
                        handle.save(results)

            # Store results for equilibrium/initial sample system
            if name == 'eq':
                self.eq_results = results.copy()
                atoms_properties = self.sampler.get_properties(atoms)
                self.sampler.sample_dataset.add_atoms(
                    atoms, atoms_properties)
            else:
                # Store results of for atom displacement calculations if
                # requested
                if self.nms_save_displacements:
                    atoms_properties = self.sampler.get_properties(atoms)
                    self.sampler.sample_dataset.add_atoms(
                        atoms, atoms_properties)

        return

    def get_initial_results(
        self,
    ):
        """
        Return equilibrium/initial sample system results
        """
        return self.eq_results

    def summary(
        self,
        method='standard',
        direction='central',
        **kwargs,
    ):
        """
        Run modified summary function with respect to the ASE version.
        """
        
        energies = self.get_energies(method=method, direction=direction)
        summary_lines = VibrationsData._tabulate_from_energies(energies)
        log_text = '\n'.join(summary_lines) + '\n'
        
        logger.info("INFO:\n" + log_text)
        
        return
        
    def read(
        self,
        method='standard', 
        direction='central'
    ):
        """
        Read calculation results, get Hessian and solve eigenvalue problem.
        """
        
        self.method = method
        self.direction = direction
        assert self.method in ['standard', 'frederiksen']
        assert self.direction in ['central', 'forward', 'backward']

        # Initialize Hesse matrix
        n = 3 * len(self.indices)
        H = np.empty((n, n))
        
        eq_disp = self._eq_disp()
        
        if self.hessian_avail:
            
            Hfull = eq_disp.vib._cached['hessian']
            selection = np.zeros(n, dtype=int)
            for ii, index in enumerate(self.indices):
                selection[3*ii + 0] = 3*index + 0
                selection[3*ii + 1] = 3*index + 1
                selection[3*ii + 2] = 3*index + 2
            H = Hfull[selection, selection]

        else:
            
            r = 0
            
            if self.direction != 'central':
                feq = eq_disp.forces()

            for a, i in self._iter_ai():
                disp_minus = self._disp(a, i, -1)
                disp_plus = self._disp(a, i, 1)
                fminus = disp_minus.forces()
                fplus = disp_plus.forces()
                if self.method == 'frederiksen':
                    fminus[a] -= fminus.sum(0)
                    fplus[a] -= fplus.sum(0)
                if self.nfree == 4:
                    fminusminus = self._disp(a, i, -2).forces()
                    fplusplus = self._disp(a, i, 2).forces()
                    if self.method == 'frederiksen':
                        fminusminus[a] -= fminusminus.sum(0)
                        fplusplus[a] -= fplusplus.sum(0)
                if self.direction == 'central':
                    if self.nfree == 2:
                        H[r] = .5 * (fminus - fplus)[self.indices].ravel()
                    else:
                        assert self.nfree == 4
                        H[r] = H[r] = (-fminusminus +
                                    8 * fminus -
                                    8 * fplus +
                                    fplusplus)[self.indices].ravel() / 12.0
                elif self.direction == 'forward':
                    H[r] = (feq - fplus)[self.indices].ravel()
                else:
                    assert self.direction == 'backward'
                    H[r] = (fminus - feq)[self.indices].ravel()
                H[r] /= 2 * self.delta
                r += 1

            
        H += H.copy().T
        self.H = H
            
        masses = self.atoms.get_masses()
        if any(masses[self.indices] == 0):
            raise RuntimeError('Zero mass encountered in one or more of '
                               'the vibrated atoms. Use Atoms.set_masses()'
                               ' to set all masses to non-zero values.')

        self.im = np.repeat(masses[self.indices]**-0.5, 3)
        self._vibrations = self.get_vibrations(read_cache=False)

        omega2, modes = np.linalg.eigh(self.im[:, None] * H * self.im)
        self.modes = modes.T.copy()

        # Conversion factor:
        s = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
        self.hnu = s * omega2.astype(complex)**0.5
        
        return
