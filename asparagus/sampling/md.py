import os
import queue
import logging
import threading
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from asparagus import sampling

from asparagus import settings
from asparagus import utils

__all__ = ['MDSampler']


class MDSampler(sampling.Sampler):
    """
    Molecular Dynamics Sampler class

    Parameters
    ----------
    config: (str, dict, settings.Configuration)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    md_temperature: (float, dict(str, float)), optional, default 300
        Target temperature in Kelvin of the MD simulation controlled by a
        Langevin thermostat.
        Can be provided as dictionary for varying temperatures during the run.
        Expected dictionary keys and items are (only first four characters
        of the keys are considered and case insensitive):
            'Tstart': float     - Initial temperature in Kelvin
            'Tend': float       - Final temperature in Kelvin
            'Tincrement': float - Temperature step in K
            'Tinterval': float  - Time interval for temperature increase in fs
    md_time_step: float, optional, default 1.0 (1 fs)
        MD Simulation time step in fs
    md_simulation_time: float, optional, default 1E5 (100 ps)
        Total MD Simulation time in fs
    md_save_interval: int, optional, default 10
        MD Simulation step interval to store system properties of
        the current frame to dataset.
    md_langevin_friction: float, optional, default 0.01
        Langevin thermostat friction coefficient in Kelvin. Generally
        within the magnitude of 1E-2 (fast heating/cooling) to 1E-4 (slow)
    md_equilibration_time: float, optional, default 0 (no equilibration)
        Total MD Simulation time in fs for a equilibration run prior to
        the production run.
    md_initial_temperature: (float, bool), optional, default False
        Temperature for initial atom velocities according to a Maxwell-
        Boltzmann distribution.

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for sample module
    sampling.Sampler._default_args.update({
        'md_temperature':               300.,
        'md_time_step':                 1.,
        'md_simulation_time':           1.E5,
        'md_save_interval':             100,
        'md_langevin_friction':         1.E-2,
        'md_equilibration_time':        None,
        'md_initial_temperature':       False,
        })
    
    # Expected data types of input variables
    sampling.Sampler._dtypes_args.update({
        'md_temperature':               [
            utils.is_numeric, utils.is_dictionary],
        'md_time_step':                 [utils.is_numeric],
        'md_simulation_time':           [utils.is_numeric],
        'md_save_interval':             [utils.is_integer],
        'md_langevin_friction':         [utils.is_numeric],
        'md_equilibration_time':        [utils.is_numeric],
        'md_initial_temperature':       [utils.is_numeric, utils.is_bool],
        })

    def __init__(
        self,
        config: Optional[Union[str, dict, settings.Configuration]] = None,
        config_file: Optional[str] = None, 
        md_temperature: Optional[Union[float, Dict[str, float]]] = None,
        md_time_step: Optional[float] = None,
        md_simulation_time: Optional[float] = None,
        md_save_interval: Optional[float] = None,
        md_langevin_friction: Optional[float] = None,
        md_equilibration_time: Optional[float] = None,
        md_initial_temperature: Optional[Union[float, bool]] = None,
        **kwargs,
    ):
        """
        Initialize MD sampling class

        """
        
        # Sampler class label
        self.sample_tag = 'md'

        # Initialize parent class
        super().__init__(
            sample_tag=self.sample_tag,
            config=config,
            config_file=config_file,
            **kwargs
            )

        ################################
        # # # Check MD Class Input # # #
        ################################
        
        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)
        
        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, sampling),
            check_dtype=utils.get_dtype_args(self, sampling)
        )

        # Check sample properties for energy and forces properties which are 
        # required for MD sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')

        return
    
    def get_info(self):
        """
        Returns a dictionary with the information of the MD sampler.

        Returns
        -------
        dict
            Dictionary with the information of the MD sampler.
        """
        
        info = super().get_info()
        info.update({
            'md_temperature': self.md_temperature,        
            'md_time_step': self.md_time_step,
            'md_simulation_time': self.md_simulation_time,
            'md_save_interval': self.md_save_interval,
            'md_langevin_friction': self.md_langevin_friction,
            'md_equilibration_time': self.md_equilibration_time,
            'md_initial_temperature': self.md_initial_temperature,
            })
        
        return info

    def run_systems(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        **kwargs,
    ):
        """
        Perform Molecular Dynamics simulations on the sample system.
        
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

        # Initialize auxiliary array
        self.first_increment = [True]*sample_systems_queue.qsize()

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
        Perform MD Simulation with the sample system.

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

            # Initialize log file
            sample_log_file = self.sample_log_file.format(isample)
            
            # Initialize trajectory file
            if self.sample_save_trajectory:
                trajectory_file = self.sample_trajectory_file.format(isample)
            else:
                trajectory_file = None

            # Assign calculator
            system = self.assign_calculator(
                system,
                ithread=ithread)

            # Perform MD simulation
            Nsample = self.run_langevin_md(
                system,
                log_file=sample_log_file,
                trajectory_file=trajectory_file,
                ithread=ithread)
            
            # Print sampling info
            message = (
                f"Sampling method '{self.sample_tag:s}' complete for system "
                + f"of index {index:d} from '{source}!'\n")
            if Nsample == 0:
                message += f"No samples written to "
            if Nsample == 1:
                message += f"{Nsample:d} sample written to "
            else:
                message += f"{Nsample:d} samples written to "
            message += f"'{self.sample_data_file:s}'."
            self.logger.info(message)

        return

    def run_langevin_md(
        self,
        system: ase.Atoms,
        temperature: Optional[Union[float, Dict[str, float]]] = None,
        time_step: Optional[float] = None,
        simulation_time: Optional[float] = None,
        langevin_friction: Optional[float] = None,
        equilibration_time: Optional[float] = None,
        initial_velocities: Optional[bool] = None,
        initial_temperature: Optional[Union[float, bool]] = None,
        log_file: Optional[str] = None,
        trajectory_file: Optional[str] = None,
        ithread: Optional[int] = None,
    ) -> int:
        """
        This does a Molecular Dynamics simulation using Langevin thermostat
        and verlocity Verlet algorithm for an NVT ensemble.

        In the future we could add more sophisticated sampling methods
        (e.g. MALA or HMC)

        Parameters
        ----------
        system: ase.Atoms
            System to be sampled.
        temperature: (float, dict(str, float)), optional, default None
            MD Simulation temperature in Kelvin
            Can be provided as dictionary for varying temperatures during the
            run. Expected dictionary keys and items are (only first four
            characters of the keys are considered and case insensitive):
                'Tstart': float     - Initial temperature in Kelvin
                'Tend': float       - Final temperature in Kelvin
                'Tincrement': float - Temperature step in K
                'Tinterval': float  - Time interval for temperature increase in
                                      fs
        time_step: float, optional, default None
            MD Simulation time step in fs
        simulation_time: float, optional, default None
            Total MD Simulation time in fs
        langevin_friction: float, optional, default None
            Langevin thermostat friction coefficient in Kelvin.
        equilibration_time: float, optional, default None
            Total MD Simulation time in fs for a equilibration run prior to
            the production run.
        initial_temperature: (float, bool), optional, default None
            Temperature for initial atom velocities according to a Maxwell-
            Boltzmann distribution.
        log_file: str, optional, default None
            Log file for sampling information
        trajectory_file: str, optional, default None
            ASE Trajectory file path to append sampled system if requested
        ithread: int, optional, default None
            Thread number
        
        Return
        ------
        int
            Number of sampled systems to database

        """

        # Check input parameters
        if temperature is None:
            temperature = self.md_temperature
        if time_step is None:
            time_step = self.md_time_step
        if simulation_time is None:
            simulation_time = self.md_simulation_time
        if langevin_friction is None:
            langevin_friction = self.md_langevin_friction
        if equilibration_time is None:
            equilibration_time = self.md_equilibration_time
        if initial_temperature is None:
            initial_temperature = self.md_initial_temperature

        # Check temperature input
        temperature, temperature_program = self.check_temperature(
            temperature,
            simulation_time)

        # Initialize stored sample counter
        Nsample = 0

        # Set initial atom velocities if requested
        if utils.is_bool(initial_temperature):
            if initial_temperature:
                MaxwellBoltzmannDistribution(
                    system,
                    temperature_K=temperature)
        elif initial_temperature > 0.:
            MaxwellBoltzmannDistribution(
                system, 
                temperature_K=initial_temperature)
        
        # Initialize MD simulation propagator
        md_dyn = Langevin(
            system, 
            timestep=time_step*units.fs,
            temperature_K=temperature,
            friction=langevin_friction,
            logfile=log_file,
            loginterval=self.md_save_interval)

        # Perform MD equilibration simulation if requested
        if equilibration_time is not None and equilibration_time > 0.:
            
            # Run equilibration simulation
            equilibration_step = round(equilibration_time/time_step)
            md_dyn.run(equilibration_step)
            
        # Attach system properties saving function
        md_dyn.attach(
            self.save_properties,
            interval=self.md_save_interval,
            system=system,
            Nsample=Nsample)

        # Attach trajectory
        if self.sample_save_trajectory:
            md_dyn.attach(
                self.write_trajectory, 
                interval=self.md_save_interval,
                system=system,
                trajectory_file=trajectory_file)

        # If defined, attach temperature program function to change reference
        # temperature by an temperature increment every specified time interval
        if temperature_program:
            temperature_interval = int(temperature_program['tint']/time_step)
            if ithread is None:
                ithread = 0
            md_dyn.attach(
                self.update_temperature,
                interval=temperature_interval,
                dyn=md_dyn,
                increment=temperature_program['tinc'],
                ithread=ithread)

        # Run MD simulation
        simulation_steps = round(simulation_time/time_step)
        md_dyn.run(simulation_steps)
        
        # As function attachment to ASE Dynamics class does not provide a 
        # return option of Nsamples, guess attached samples
        Nsample = simulation_steps//self.md_save_interval + 1

        return Nsample

    def check_temperature(
        self,
        temperature: Union[float, Dict[str, float]],
        simulation_time: float,
    ) -> (float, Dict[str, float]):
        """
        Check temperature parameter input and, eventually, prepare temperature
        program parameters

        Parameters
        ----------
        temperature: (float, dict(str, float))
            MD Simulation temperature in Kelvin
        simulation_time: float
            Total MD Simulation time in fs

        Returns
        -------
        float
            MD Simulation temperature in Kelvin
        dict(str, float)
            Temperature program parameter to change the reference temperature
            during the MD simulation. If empty, reference temperature stays
            constant

        """

        # If temperature is numeric - constant reference temperature
        if utils.is_numeric(temperature):

            return temperature, {}

        elif utils.is_dictionary(temperature):

            # Reduce dictionary keys to 4 letter codes
            temperature_short = {}
            for key, item in temperature.items():
                key_short = f"{key.lower():<4s}"[:4]
                temperature_short[key_short] = float(item)

            # Check temperature program parameters
            # Essential starting temperature
            if 'tsta' in temperature_short:
                tstart = temperature_short['tsta']
            else:
                raise SyntaxError(
                    "Initial MD simulation temperature 'tstart 'is not "
                    + "defined!")
            # Optional final temperature if increment and interval is given
            if 'tend' in temperature_short:
                tend = temperature_short['tend']
            elif not 'tint' in temperature_short:
                raise SyntaxError(
                    "Final MD simulation temperature 'tend 'is not "
                    + "defined!")
            else:
                tend = None
            # Optional temperature increment if final temperature is given
            if 'tinc' in temperature_short:
                tincrement = temperature_short['tinc']
            elif (
                'tend' in temperature_short
                and 'tint' in temperature_short
            ):
                tsteps = int(simulation_time/temperature_short['tint']) - 1
                tincrement = (tstart - tend)/tsteps
            else:
                tincrement = 10.0
            # Optional temperature increment interval
            if 'tint' in temperature_short:
                tinterval = temperature_short['tint']
            elif 'tend' in temperature_short:
                tsteps = int((tstart - tend)/tinc) + 1
                tinterval = simulation_time/tsteps
            else:
                raise SyntaxError(
                    "Interval for MD simulation temperature increase 'tint' "
                    + "is not defined!")

            # Complete temperature program dictionary
            if tend is None:
                tsteps = int(simulation_time/tinterval) - 1
                tend = tstart + tsteps*tincrement
            temperature_program = {
                'tsta': tstart,
                'tend': tend,
                'tinc': tincrement,
                'tint': tinterval,
                }

            return tstart, temperature_program

        else:

            raise SyntaxError(
                "MD simulation temperature input is invalid!")

        return

    def update_temperature(self, dyn, increment, ithread):
        """
        Apply Langevin reference temperature step
        """
        if self.first_increment[ithread]:
            self.first_increment[ithread] = False
            return
        dyn.temp = dyn.temp + units.kB*increment
        dyn.updatevars()
        return
