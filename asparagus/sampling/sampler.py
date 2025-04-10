import os
import time
import queue
import logging
import threading
import subprocess
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase import optimize
from ase.parallel import world

from asparagus import sampling

from asparagus import data
from asparagus import settings
from asparagus import utils
from asparagus import interface

__all__ = ['Sampler']


# ======================================
# General Conformation Sampler Class
# ======================================

class Sampler:
    """
    Conformation Sampler main class for generation of reference structures.

    Parameters
    ----------
    config: (str, dict, settings.Configuration)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    sample_data_file: str, optional, default None
        Database file name to store a selected set of systems with
        computed reference data. If None, data file name is the respective
        sample method tag.
    sample_directory: str, optional, default None
        Working directory where to store eventually temporary ASE
        calculator files, ASE trajectory files and/or model calculator
        files. If None, files will be stored in parent directory.
    sample_system_queue: queue.Queue, optional, default None
        Queue object including sample systems or where 'sample_systems' input
        will be added. If not defined, an empty queue will be assigned.
    sample_systems: (str, list, object), optional, default ''
        System coordinate file or a list of system coordinate files or
        ASE atoms objects that are considered as initial conformations for
        reference structure sampling.
    sample_systems_format: (str, list), optional, default ''
        System coordinate file format string (e.g. 'xyz') for the
        definition in 'sample_systems' in case of file paths.
    sample_systems_indices: (int, list), optional, default None
        List of sample number indices for specific selection of systems
        in the sample system files.
    sample_system_fragments: (list, dict), optional, default None
        System specific fragment definition to assign system atoms to 
        difference fragments which are, e.g., compute at different level of
        theory or get treated differently by the model potential.
        It can be either a list containing lists of atom indices assigned to
        fragment 0, 1, ..., or a dictionary of fragment name (key) and atom
        indices (item).
        The atom indices can be given as an integer atom index or a string of
        a single index (e.g. '0') or an index range (e.g. '0-10').
        Not defined atoms are always added to a new fragment of the last index
        plus 1.
        If multiple sample systems are defined, the system fragment definition
        is applied on each system.
    sample_calculator: (str, callable object), optional, default 'XTB'
        Definition of the ASE calculator type for reference data
        computation. The input can be either directly a ASE calculator
        class object or a string with available ASE calculator classes.
    sample_calculator_args: dict, optional, default {}
        In case of string type input for 'sample_calculator', this
        dictionary is passed as keyword arguments at the initialization
        of the ASE calculator.
    sample_save_trajectory: bool, optional, default True
        If True, add sampled systems added to the database file also to an
        ASE trajectory file.
    sample_num_threads: int, optional, default 1
        Number of parallel threads of property calculation using the sample
        calculator. Default is 1 (serial computation). Parallel computation
        is not possible for all sampling methods but for:
            Sampler, NMSampler, NMScanner
    sample_properties: List[str], optional, default None
        List of system properties which are computed by the ASE
        calculator class. Requested properties will be checked with the
        calculator available property list and return an error when one
        requested property is unavailable. By default all available
        properties will be stored.
        If None, compute all properties implemented in the ASE calculator.
    sample_systems_optimize: bool, optional, default False
        Instruction flag if the system coordinates shall be
        optimized using the ASE calculator defined by 'sample_calculator'.
    sample_systems_optimize_fmax: float, optional, default 0.01
        Instruction flag, if the system coordinates shall be
        optimized using the ASE calculator defined by 'sample_calculator'.
    sample_data_overwrite: bool, optional, default False
        If False, add new sampling data to an eventually existing data
        file. If True, overwrite an existing one.
    sample_property_threshold: dict, optional, default None
        Dictionary of model property (key) minimum threshold values (item) to
        decide if samples or frames are added to the sample data file.
        For model ensembles, this could be the energy standard deviation
        ('str_energy').
    sample_nsamples_threshold: int, optional, default None
        If not None, the sampling will be terminated if the sample number
        threshold is reached.
    sample_tag: str, optional, default 'sample'
        Sampling method tag of the specific sampling methods for
        log and ASE trajectory files or the data file name if not defined.

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for sample module
    _default_args = {
        'sample_directory':             None,
        'sample_data_file':             None,
        'sample_systems_queue':         None,
        'sample_systems':               None,
        'sample_systems_format':        None,
        'sample_systems_indices':       None,
        'sample_system_fragments':      None,
        'sample_calculator':            'XTB',
        'sample_calculator_args':       {},
        'sample_save_trajectory':       True,
        'sample_num_threads':           1,
        'sample_properties':            None,
        'sample_systems_optimize':      False,
        'sample_systems_optimize_fmax': 0.001,
        'sample_data_overwrite':        False,
        'sample_property_threshold':    None,
        'sample_nsamples_threshold':    None,
        'sample_tag':                   'sample',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'sample_directory':             [utils.is_string, utils.is_None],
        'sample_data_file':             [
            utils.is_string, utils.is_string_array_inhomogeneous,
            utils.is_None],
        'sample_systems':               [
            utils.is_None, utils.is_string, utils.is_string_array,
            utils.is_ase_atoms, utils.is_ase_atoms_array],
        'sample_systems_format':        [
            utils.is_None, utils.is_string, utils.is_string_array],
        'sample_systems_indices':       [
            utils.is_integer, utils.is_integer_array],
        'sample_system_fragments':      [
            utils.is_array_like, utils.is_dictionary],
        'sample_calculator':            [utils.is_string, utils.is_object],
        'sample_calculator_args':       [utils.is_dictionary],
        'sample_save_trajectory':       [utils.is_bool],
        'sample_num_threads':           [utils.is_integer],
        'sample_properties':            [
            utils.is_None, utils.is_string, utils.is_string_array],
        'sample_systems_optimize':      [
            utils.is_bool, utils.is_boolean_array],
        'sample_systems_optimize_fmax': [utils.is_numeric],
        'sample_data_overwrite':        [utils.is_bool],
        'sample_property_threshold':    [utils.is_None, utils.is_dictionary],
        'sample_nsamples_threshold':    [utils.is_None, utils.is_integer],
        'sample_tag':                   [utils.is_string],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, settings.Configuration]] = None,
        config_file: Optional[str] = None, 
        sample_data_file: Optional[str] = None,
        sample_directory: Optional[str] = None,
        sample_systems_queue: Optional[queue.Queue] = None,
        sample_systems: Optional[Union[str, List[str], object]] = None,
        sample_systems_format: Optional[Union[str, List[str]]] = None,
        sample_systems_indices: Optional[Union[int, List[int]]] = None,
        sample_system_fragments: 
            Optional[Union[List[int], Dict[int, Any]]] = None,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        sample_save_trajectory: Optional[bool] = None,
        sample_num_threads: Optional[int] = None,
        sample_properties: Optional[List[str]] = None,
        sample_systems_optimize: Optional[bool] = None,
        sample_systems_optimize_fmax: Optional[float] = None,
        sample_data_overwrite: Optional[bool] = None,
        sample_property_threshold: Optional[Dict[str, float]] = None,
        sample_nsamples_threshold: Optional[int] = None,
        sample_tag: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Sampler class.
        """

        #####################################
        # # # Check Sampler Class Input # # #
        #####################################

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

        # Set global configuration as class parameter
        self.config = config

        # Check system input
        if self.sample_systems is None and self.sample_systems_queue is None:
            self.logger.warning(
                "No input in 'sample_systems' is given!\n"
                + "Please provide either a chemical structure file or "
                + "an ASE Atoms object as initial sample structure.")
            self.sample_systems = []

        #####################################
        # # # Prepare Sample Calculator # # #
        #####################################

        # Get ASE calculator
        ase_calculator, ase_calculator_tag = (
            interface.get_ase_calculator(
                self.sample_calculator,
                self.sample_calculator_args))

        # Assign calculator tag for info dictionary
        self.sample_calculator_tag = ase_calculator_tag

        # Check requested system properties
        self.sample_properties, self.sample_unit_properties = (
            self.check_properties(
                self.sample_properties,
                ase_calculator)
            )

        # Check number of calculation threads
        if self.sample_num_threads <= 0:
            raise ValueError(
                "Number of sample threads 'sample_num_threads' must be "
                + "larger or equal 1, but "
                + f"'{self.sample_num_threads:d}' is given!")

        # Check if sample calculator is thread safe
        if self.sample_num_threads > 1:
            message = (
                f"Sample calculator {self.sample_calculator_tag:s} is "
                + "selected for multi-threading sampling with "
                + f"{self.sample_num_threads:d} threads.")
            if interface.is_ase_calculator_threadsafe(
                self.sample_calculator_tag
            ):
                self.logger.info(message)
            else:
                message += (
                    "\nBut the selected calculator is NOT thread safe!"
                    "\nNumber of parallel threads is set back to just 1.")
                self.logger.warning(message)
                self.sample_num_threads = 1

        ############################
        # # # Prepare Sampling # # #
        ############################

        # Initialize sampling counter
        if config.get('sample_counter') is None:
            self.sample_counter = 1
        else:
            self.sample_counter = config.get('sample_counter') + 1

        # Generate working directory
        if self.sample_directory is None or not len(self.sample_directory):
            self.sample_directory = '.'
        elif not os.path.exists(self.sample_directory):
            os.makedirs(self.sample_directory)

        # Check sample data file
        if self.sample_data_file is None:
            self.sample_data_file = f'{self.sample_tag:s}.db'
        if utils.is_string(self.sample_data_file):
            self.sample_data_file = (
                self.sample_data_file,
                data.check_data_format(
                    self.sample_data_file, is_source_format=False)
                )
        elif utils.is_string_array(self.sample_data_file):
            self.sample_data_file = tuple(self.sample_data_file)
        else:
            raise ValueError(
                "Sample data file 'sample_data_file' must be a string "
                + "of a valid file path but is of type "
                + f"'{type(self.sample_data_file)}'!")

        # Define sample log file path and trajectory file
        self.sample_log_file = os.path.join(
            self.sample_directory,
            f'{self.sample_counter:d}_{self.sample_tag:s}_{{:d}}.log')
        self.sample_trajectory_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}_{{:d}}.traj')

        # Initialize the multithreading lock
        self.lock = threading.Lock()

        # Check property threshold parameter
        if self.sample_property_threshold is not None:
            self.check_property_threshold(
                self.sample_property_threshold,
                sample_calculator=ase_calculator)

        # Check sample number threshold parameter
        if self.sample_nsamples_threshold is not None:
            self.check_nsamples_threshold()

        #############################
        # # # Prepare Optimizer # # #
        #############################

        if self.sample_systems_optimize:

            # Assign ASE optimizer
            self.optimizer_tag = 'bfgs'
            self.ase_optimizer = optimize.BFGS

        #####################################
        # # # Initialize Sample DataSet # # #
        #####################################

        self.sample_dataset = data.DataSet(
            self.sample_data_file,
            data_properties=self.sample_properties,
            data_unit_properties=self.sample_unit_properties,
            data_overwrite=self.sample_data_overwrite)

        return

    def __str__(self):
        """
        Return class descriptor
        """
        return "Sampler class"

    def read_systems(
        self,
        sample_systems_queue,
        sample_systems,
        sample_systems_format,
        sample_systems_indices,
        sample_system_fragments,
    ):
        """
        Iterator to read next sample system and return as ASE atoms object
        """
        
        # Check system and format input
        if sample_systems is None or not len(sample_systems):
            return sample_systems_queue, ['Queue']
        
        if not utils.is_array_like(sample_systems):
            sample_systems = [sample_systems]
        
        if sample_systems_format is None:
            sample_systems_format = []
            for system in sample_systems:
                if utils.is_string(system):
                    sample_systems_format.append(system.split('.')[-1])
                else:
                    sample_systems_format.append(None)
        elif utils.is_string(sample_systems_format):
            sample_systems_format = (
                [sample_systems_format]*len(sample_systems))
        elif len(sample_systems) != len(sample_systems_format):
            raise ValueError(
                "Sample system input 'sample_systems' and "
                + "'sample_systems_format' have different input size of "
                + f"{len(sample_systems):d} and "
                + f"{len(sample_systems_format):d}, respectively.")
        
        # Check system index selection
        if utils.is_integer(sample_systems_indices):
            sample_systems_indices = [sample_systems_indices]
        if (
            sample_systems_indices is not None 
            and len(sample_systems_indices) == 0
        ):
            sample_systems_indices = None

        # If number of samples sources is larger 1, prepare system index 
        # selection for sample sources
        Nsystems = len(sample_systems)
        if Nsystems > 1 and sample_systems_indices is not None:
            indices = sample_systems_indices.copy()
            for ii, idx in enumerate(sample_systems_indices):
                if idx >= Nsystems or idx < (-1*Nsystems):
                    raise SyntaxError(
                        "System index selection 'sample_systems_indices' "
                        + f"contains index ({idx:d}) which is outside of "
                        + f"the sample number range of ({Nsystems:d})!")
                if idx < 0:
                    indices[ii] = Nsystems + idx

        # Iterate over system input and eventually read file to store as
        # (ASE Atoms object, index, sample source)
        for isample, (source, source_format) in enumerate(
            zip(sample_systems, sample_systems_format)
        ):
            
            # If sample index not in indices list in case of multiple sources
            if Nsystems > 1 and sample_systems_indices is not None:
                if isample not in indices:
                    continue

            # Check for ASE Atoms object or read system file
            if utils.is_ase_atoms(source):
                
                # Store ASE Atoms object as xyf file
                source_file = os.path.join(
                    self.sample_directory,
                    f"{self.sample_counter:d}_sample_system_{isample:d}.xyz")
                ase.io.write(source_file, source, format='xyz')

                # Assign system fragment definition to atoms object
                if sample_system_fragments is not None:
                    fragments = self.get_system_fragments(
                        sample_system_fragments,
                        source)
                    source.info['fragments'] = fragments

                # Add sample system to queue
                sample_systems_queue.put((source, isample, source_file, 1))
            
            # Check for an Asparagus dataset
            # elif source_format.lower() == 'db':
            elif (
                data.check_data_format(source_format, ignore_error=True)
                is not None
            ):
                
                # Open dataset
                dataset = data.DataSet(source)
                
                # Prepare system index selection in case of just one sample 
                # source
                if Nsystems == 1 and sample_systems_indices is not None:
                    Ndata = len(dataset)
                    indices = sample_systems_indices.copy()
                    for ii, idx in enumerate(sample_systems_indices):
                        if idx >= Ndata or idx < (-1*Ndata):
                            raise SyntaxError(
                                "System index selection "
                                + "'sample_systems_indices' contains index "
                                + f"({idx:d}) which is outside of the data "
                                + f"number range of ({Ndata:d}) in sample "
                                + f"file '{source:s}'!")
                        if idx < 0:
                            indices[ii] = Ndata + idx

                # Iterate over dataset
                for isys, data_i in enumerate(dataset):

                    # Skip if system index not in indices list 
                    if Nsystems == 1 and sample_systems_indices is not None:
                        if isys not in indices:
                            continue

                    # Check cell parameter
                    cell = data_i['cell'].numpy().reshape(-1)
                    if cell.shape == (9,):
                        cell = cell.reshape(3, 3)

                    # Create and append atoms object to sample queue
                    system = ase.Atoms(
                        data_i['atomic_numbers'],
                        positions=data_i['positions'],
                        pbc=data_i['pbc'].numpy().reshape(-1),
                        cell=cell)
                    if 'charge' in data_i:
                        system.info['charge'] = int(
                            data_i['charge'].numpy()[0])
                    
                    # Assign system fragment definition to atoms object
                    if sample_system_fragments is not None:
                        fragments = self.get_system_fragments(
                            sample_system_fragments,
                            system)
                        system.info['fragments'] = fragments
                    elif (
                        'fragments' in data_i
                        and data_i['fragments'] is not None
                    ):
                        system.info['fragments'] = data_i['fragments'].numpy()
                    
                    sample_systems_queue.put((system, isample, source, isys))
            
            # Else, use ase.read function with respective format
            else:
                
                counter=0
                complete = False
                while not complete:
                    try:
                        if (
                            Nsystems == 1
                            and sample_systems_indices is not None
                        ):
                            isys = sample_systems_indices[counter]
                        else:
                            isys = counter

                        # Read sample system from source file
                        system = ase.io.read(
                            source, index=isys, format=source_format)

                        # Assign system fragment definition to atoms object
                        if sample_system_fragments is not None:
                            fragments = self.get_system_fragments(
                                sample_system_fragments,
                                system)
                            system.info['fragments'] = fragments

                        sample_systems_queue.put(
                            (system, isample, source, isys))
                    
                    except (StopIteration, AssertionError):
                        complete = True
                    
                    else:
                        counter += 1
                        if (
                            Nsystems == 1
                            and sample_systems_indices is not None
                            and counter >= len(sample_systems_indices)
                        ):
                            complete = True

        return sample_systems_queue, sample_systems

    def assign_calculator(
        self,
        sample_system: ase.Atoms,
        sample_calculator: Optional[Union[str, object]] = None,
        sample_calculator_args: Optional[Dict[str, Any]] = None,
        ithread: Optional[int] = None,
    ):
        """
        Assign calculator to a list of sample ASE Atoms objects

        Parameters
        ----------
        sample_system : ase.Atoms
            ASE Atoms object to assign the calculator
        sample_calculator : (str, object), optional, default None
            ASE calculator object or string of an ASE calculator class
            name to assign to the sample systems
        sample_calculator_args : dict, optional, default None
            Dictionary of keyword arguments to initialize the ASE
            calculator
        ithread: int, optional, default None
            Thread number to avoid conflict between files written by the
            calculator.

        """

        # Check calculator input
        if sample_calculator is None:
            sample_calculator = self.sample_calculator
        if sample_calculator_args is None:
            sample_calculator_args = self.sample_calculator_args

        # Get ASE calculator
        # Special case: Asparagus model calculator is stored to avoid repeating
        # initialization for multiple sample systems
        if (
            utils.is_string(sample_calculator)
            and sample_calculator.lower() in ['asparagus', 'model']
        ):

            if hasattr(self, 'model_calculator'):

                # Get stored model calculator
                ase_calculator = self.model_calculator
                ase_calculator_tag = self.model_calculator_tag

            else:

                # Initialize and store model calculator
                ase_calculator, ase_calculator_tag = (
                    interface.get_ase_calculator(
                        sample_calculator,
                        sample_calculator_args,
                        ithread=ithread)
                    )
                self.model_calculator = ase_calculator
                self.model_calculator_tag = ase_calculator_tag

        else:

            ase_calculator, ase_calculator_tag = (
                interface.get_ase_calculator(
                    sample_calculator,
                    sample_calculator_args,
                    ithread=ithread)
                )

        # Check requested system properties
        self.sample_properties, self.sample_unit_properties = (
            self.check_properties(
                self.sample_properties,
                ase_calculator)
            )

        # Assign ASE calculator
        sample_system.calc = ase_calculator

        return sample_system

    def check_properties(
        self,
        sample_properties: List[str],
        sample_calculator: 'ase.Calculator'
    ):
        """
        Check requested sample properties and units with implemented properties
        of the calculator
        
        Parameters
        ----------
        sample_properties: list
            List of system properties to check for availability by the
            calculator.
        sample_calculator: ase.Calculator
            ASE Calculator to compare requested and available properties
            from the calculator.

        Returns
        -------
        list:
            Checked sample properties
        list:
            Sample properties (+ positions, charge) units

        """

        # If sample properties are not defined (None), take all available
        # properties from ASE calculator
        if sample_properties is None:
            sample_properties = []
            for prop in sample_calculator.implemented_properties:
                if prop in settings._valid_properties:
                    sample_properties.append(prop)
                elif 'std_' in prop and prop[4:] in settings._valid_properties:
                    sample_properties.append(prop)

        # Check requested system properties
        for prop in sample_properties:
            # TODO Special calculator properties list for special properties
            # not supported by ASE such as, e.g., charge, hessian, etc.
            if prop not in sample_calculator.implemented_properties:
                raise ValueError(
                    f"Requested property '{prop:s}' is not implemented "
                    + f"in the ASE calculator '{sample_calculator}'! "
                    + "Available ASE calculator properties are:\n"
                    + f"{sample_calculator.implemented_properties}")

        # Define positions and property units
        sample_unit_properties = {}
        for prop in sample_properties:
            if 'std_' in prop:
                sample_unit_properties[prop] = (
                    interface.ase_calculator_units.get(prop[4:]))
            else:
                sample_unit_properties[prop] = (
                    interface.ase_calculator_units.get(prop))
        if 'positions' not in sample_unit_properties:
            sample_unit_properties['positions'] = (
                interface.ase_calculator_units.get('positions'))
        if 'charge' not in sample_unit_properties:
            sample_unit_properties['charge'] = (
                interface.ase_calculator_units.get('charge'))

        return sample_properties, sample_unit_properties

    def get_info(self):
        """
        Dummy function for sampling parameter dictionary
        """
        return {            
            'sample_data_file': self.sample_data_file,
            'sample_directory': self.sample_directory,
            'sample_systems': self.sample_systems,
            'sample_systems_format': self.sample_systems_format,
            'sample_system_fragments': self.sample_system_fragments,
            'sample_calculator': self.sample_calculator_tag,
            'sample_calculator_args': self.sample_calculator_args,
            'sample_properties': self.sample_properties,
            'sample_systems_optimize': self.sample_systems_optimize,
            'sample_systems_optimize_fmax': self.sample_systems_optimize_fmax,
            'sample_data_overwrite': self.sample_data_overwrite,
        }

    def run(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        sample_systems: Optional[Union[str, List[str], object]] = None,
        sample_systems_format: Optional[Union[str, List[str]]] = None,
        sample_systems_indices: Optional[Union[int, List[int]]] = None,
        sample_system_fragments: 
            Optional[Union[List[int], Dict[int, Any]]] = None,
        **kwargs
    ):
        """
        Perform sampling of all sample systems or a selection of them.
        """

        ################################
        # # # Check Sampling Input # # #
        ################################
        
        # Check input
        if sample_systems_queue is None:
            sample_systems_queue = self.sample_systems_queue
        if sample_systems is None:
            sample_systems = self.sample_systems
        if sample_systems_format is None:
            sample_systems_format = self.sample_systems_format
        if sample_systems_indices is None:
            sample_systems_indices = self.sample_systems_indices
        if sample_system_fragments is None:
            sample_system_fragments = self.sample_system_fragments

        # Collect sampling parameters
        config_sample_tag = f'{self.sample_counter}_{self.sample_tag}'
        config_sample = {
            config_sample_tag: self.get_info()
            }
        
        # Read sample systems into queue
        if sample_systems_queue is None:
            sample_systems_queue = queue.Queue()
        sample_systems_queue, sample_systems = self.read_systems(
            sample_systems_queue,
            sample_systems,
            sample_systems_format,
            sample_systems_indices,
            sample_system_fragments)

        # Check if sample systems queue is empty
        if sample_systems_queue.empty():
            raise SyntaxError(
                "No sample systems defined (empty queue)! "
                + "Check 'sample_systems' input.")

        # Update configuration file with sampling parameters
        if 'sampler_schedule' in self.config:
            config_sample = {
                **self.config['sampler_schedule'],
                **config_sample,
                }
        self.config.update({
            'sampler_schedule': config_sample,
            'sample_counter': self.sample_counter
            })

        # Print sampling overview
        message = (
            f"Perform sampling method '{self.sample_tag:s}' on systems:\n")
        for isys, system in enumerate(sample_systems):
            if utils.is_ase_atoms(system):
                system = system.get_chemical_formula()
            message += f" {isys + 1:3d}. '{system:s}'\n"
        self.logger.info(message)

        ##########################
        # # # Start Sampling # # #
        ##########################
        
        self.run_systems(
            sample_systems_queue=sample_systems_queue,
            **kwargs
            )

        # Increment sample counter
        self.sample_counter += 1

        return

    def run_systems(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        **kwargs,
    ):
        """
        Apply sample calculation on sample systems.
        
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

        # Initialize sample number list
        self.Nsamples = [0 for ithread in range(self.sample_num_threads)]

        # Run sampling over sample systems
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
        Apply sample calculator on system input and write properties to 
        database.
        
        Parameters
        ----------
        sample_systems_queue: queue.Queue
            Queue of sample system information providing tuples of ase Atoms
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

            # Assign calculator
            system = self.assign_calculator(
                system, 
                ithread=ithread)

            # If requested, perform structure optimization
            if self.sample_systems_optimize:

                # Perform structure optimization
                system = self.run_optimization(
                    sample_system=system,
                    sample_index=isample)

            # Compute system properties
            system, converged = self.run_calculation(
                system, isample, source, index)

            # Store results
            if converged:
                self.save_properties(system, ithread)

            if converged and self.sample_save_trajectory:
                self.write_trajectory(
                    system, self.sample_trajectory_file.format(isample))

        # Print sampling info
        if ithread is None:
            isample = 0
        else:
            isample = ithread
        if self.Nsamples[isample] == 0:
            message += f"No samples written to "
        else:
            message = (
                f"Sampling method '{self.sample_tag:s}' complete for system "
                + f"from '{source}!'\n")
            message += f"{self.Nsamples[isample]:d} sample written to "
            if self.Nsamples[isample] == 1:
                message += "sample "
            else:
                message += f"samples "
            message += "written to "
        message += f"'{self.sample_data_file[0]:s}'."
        self.logger.info(message)

        return

    def run_calculation(
        self,
        system: ase.Atoms,
        isample: Optional[int] = None,
        source: Optional[str] = None,
        index: Optional[int] = None,
        try_again: Optional[bool] = True,
    ) -> (ase.Atoms, bool):
        """
        Apply sample calculator on system input and write properties to 
        database.
        
        Parameters
        ----------
        system: ase.Atoms
            ASE atoms object with linked calculator to run the reference
            calculation with.
        isample: int, optional, default None
            For log file, sample number from 'source'.
        source: str, optional, default None
            For log file, sample system 'source'.
        index: int, optional, default None
            For log file, sample system 'source' index.
        try_again: bool, optional, default True
            If the calculation failed (converged == False), try the reference
            calculation for a second time with 'try_again=False' then.

        Returns
        -------
        ase.Atoms
            ASE atoms object with linked calculator after reference calculation
            was run.
        bool
            Flag for converged (True) or failed (False) reference calculation

        """

        # Run calculation
        try:

            system.calc.calculate(
                system,
                properties=self.sample_properties,
                system_changes=system.calc.implemented_properties)

            if hasattr(system.calc, 'converged'):
                converged = system.calc.converged
            elif (
                self.sample_properties[0] in system.calc.results
                and (
                    system.calc.results[self.sample_properties[0]] is None
                    or np.any(np.isnan(
                        system.calc.results[self.sample_properties[0]]))
                )
            ):
                converged = False
            else:
                converged = True

        except (
            ase.calculators.calculator.CalculationFailed
            or subprocess.CalledProcessError
        ):

            converged = False

        if not converged and try_again:
            system, converged = self.run_calculation(system, try_again=False)

        # Store calculation state to log file
        if isample is None:
            isample = -1
        if source is None:
            source = 'None'
        if index is None:
            index = -1
        message = f"{time.asctime():s} "
        if converged:
            message += "CONVERGED "
        else:
            message += "FAILED "
        message += (
            f"calculation for sample {isample:d} of '{source:s}' ({index:d})")
        with self.lock:
            with open(self.sample_log_file.format(isample), 'a') as flog:
                flog.write(message)

        return system, converged

    def run_optimization(
        self,
        sample_system: Optional[ase.Atoms] = None,
        sample_index: Optional[int] = None,
        sample_systems_queue: Optional[queue.Queue] = None,
        sample_optimzed_queue: Optional[queue.Queue] = None,
        ithread: Optional[int] = None,
    ):
        """
        Perform structure optimization on sample system
        
        Parameters
        ----------
        sample_system: ase.Atoms, optional, default None
            ASE Atoms object which will be optimized using the optimizer 
            defined in self.ase_optimizer.
        sample_index: int, optional, default None
            Sample index number of the ASE atoms object to optimize.
        sample_systems_queue: queue.Queue, optional, default None
            Sample system queue cotaining ASE Atoms object which will be 
            optimized using the optimizer defined in self.ase_optimizer.
            If sample_system is not None, this queue will be ignored.
        sample_optimzed_queue: queue.Queue, optional, default None
            If defined, the optimized sample system will be put into the
            queue.
        ithread: int, optional, default None
            Thread number
        
        Returns
        -------
        ase.Atoms
            Optimized ASE atoms object

        """
        
        # Check sample system input
        if sample_system is None and sample_systems_queue is None:
            return None
        
        # Optimize sample system
        if sample_system is not None:
            
            # Prepare optimization log and trajectory file name
            if sample_index is None:
                ase_optimizer_log_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}.log')
                ase_optimizer_trajectory_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}.traj')
            else:
                ase_optimizer_log_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}'
                    + f'_{sample_index:d}.log')
                ase_optimizer_trajectory_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}'
                    + f'_{sample_index:d}.traj')

            # Assign calculator
            system = self.assign_calculator(
                sample_system,
                ithread=ithread)

            # Perform structure optimization
            self.ase_optimizer(
                system,
                logfile=ase_optimizer_log_file,
                trajectory=ase_optimizer_trajectory_file,
                ).run(
                    fmax=self.sample_systems_optimize_fmax)

            # Add optimized ASE atoms object to the queue if defined
            if sample_optimzed_queue is not None:
                sample_optimzed_queue.put((system, isample, str(system), 1))

            return system
        
        else:
            
            while self.keep_going(ithread):
                
                # Get sample parameters or wait
                sample = sample_systems_queue.get()
                
                # Check for stop flag
                if sample == 'stop':
                    self.thread_keep_going[ithread] = False
                    continue
                
                # Extract sample system to optimize
                (system, isample, source, index) = sample

                # Prepare optimization log and trajectory file name
                ase_optimizer_log_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}'
                    + f'_{isample:d}.log')
                ase_optimizer_trajectory_file = os.path.join(
                    self.sample_directory,
                    f'{self.sample_counter:d}_{self.optimizer_tag:s}'
                    + f'_{isample:d}.traj')

                # Assign calculator
                system = self.assign_calculator(
                    system,
                    ithread=ithread)

                # Perform structure optimization
                try:

                    ase_optimizer = self.ase_optimizer(
                        system,
                        logfile=ase_optimizer_log_file,
                        trajectory=ase_optimizer_trajectory_file,
                        )
                    ase_optimizer.run(fmax=self.sample_systems_optimize_fmax)
                
                except ase.calculators.calculator.CalculationFailed:

                    message = (
                        "Single point calculation of the system "
                        + f"from '{source}' of index {index:d} "
                        + "is not converged "
                        + "during structure optimization for sampling method "
                        + f"'{self.sample_tag:s}' "
                        + f"(see log file '{ase_optimizer_log_file:s}')!")
                    if sample_optimzed_queue is not None:
                        message += (
                            "\nSystem will be skipped for further sampling.")
                    self.logger.error(message)
                    
                else:

                    message = (
                        "Optimization of system "
                        + f"from '{source}' of index {index:d} is converged "
                        + f"(see log file '{ase_optimizer_log_file:s}').")
                    self.logger.info(message)

                # Add optimized ASE atoms object to the queue if defined
                if sample_optimzed_queue is not None:
                    sample_optimzed_queue.put((system, isample, source, index))

        return

    def keep_going(
        self,
        ithread
    ) -> bool:
        """
        Return thread continuation flag
        
        Parameters
        ----------
        ithread: int
            Thread number

        """
        
        # Check if number of sample threshold is reached
        if (
            self.sample_nsamples_threshold is not None
            and np.sum(self.Nsamples) >= self.sample_nsamples_threshold
        ):
            if ithread is None:
                self.thread_keep_going[0] = False
            else:
                self.thread_keep_going[ithread] = False
            return False

        # Return thread continuation flag
        if ithread is None:
            return self.thread_keep_going[0]
        else:
            return self.thread_keep_going[ithread]

        return False

    def get_properties(self, system) -> List[str]:
        """
        Collect system properties and calculator results

        """
        return interface.get_ase_properties(system, self.sample_properties)

    def save_properties(self, system, ithread) -> int:
        """
        Save system properties
        """

        # Collect system properties
        system_properties = self.get_properties(system)

        # Sample list index
        if ithread is None:
            isample = 0
        else:
            isample = ithread

        # Check for property thresholds
        if self.sample_property_threshold is not None:
            for prop, value in self.sample_property_threshold.items():
                if system.calc.results[prop] < value:
                    return

        # Check for number of sample threshold
        if (
            self.sample_nsamples_threshold is not None
            and np.sum(self.Nsamples) >= self.sample_nsamples_threshold
        ):
            return

        # Add to sample data file
        with self.lock:
            self.sample_dataset.add_atoms(system, system_properties)
            self.Nsamples[isample] += 1

        return

    def write_trajectory(self, system, trajectory_file):
        """
        Write current image to trajectory file but without constraints
        """
        
        # Check trajectory file
        if trajectory_file is None:
            return

        # Check for property thresholds
        # If accepted, write to another trajectory file
        threshold_trajectory_file = None
        if self.sample_property_threshold is not None:
            accepted = True
            for prop, value in self.sample_property_threshold.items():
                if system.calc.results[prop] < value:
                    accepted = False
            if accepted:
                threshold_trajectory_file = (
                    f'{trajectory_file:s}.threshold.traj')
                threshold_trajectory_properties = (
                    self.sample_properties
                    + list(self.sample_property_threshold.keys()))

        # Save system without constraint to the trajectory file
        with self.lock:
            trajectory = ase.io.Trajectory(
                trajectory_file, atoms=system,
                mode='a', properties=self.sample_properties)
            system_noconstraint = system.copy()
            system_noconstraint.calc = system.calc
            system_noconstraint.set_constraint()
            trajectory.write(system_noconstraint)
            trajectory.close()
            if threshold_trajectory_file is not None:
                trajectory = ase.io.Trajectory(
                    threshold_trajectory_file, atoms=system_noconstraint,
                    mode='a', properties=threshold_trajectory_properties)
                trajectory.write(system_noconstraint)
                trajectory.close()

        return

    def check_property_threshold(
        self,
        sample_property_threshold: Dict[str, float],
        sample_calculator: Optional['ase.Calculator'] = None,
    ):
        """
        Check property threshold input

        Parameters
        ----------
        sample_property_threshold: dict
            Dictionary of model property (key) minimum threshold values (item)
            to decide if samples or frames are added to the sample data file.
        sample_calculator: ASE.Calculator, optional, default None
            ASE Calculator used for sampling. If defined, check if the
            calculators implemented properties contain the requested property,
            else check with sample properties.

        """

        # Get available property list
        if sample_calculator is None:
            available_properties = self.sample_properties
        else:
            available_properties = sample_calculator.implemented_properties

        # Get available property unit dictionary
        available_unit_properties = {}
        for prop in available_properties:
            if prop in self.sample_unit_properties:
                available_unit_properties[prop] = (
                    self.sample_unit_properties[prop])
            elif 'std_' in prop:
                available_unit_properties[prop] = (
                    interface.ase_calculator_units.get(prop[4:]))
            else:
                available_unit_properties[prop] = (
                    interface.ase_calculator_units.get(prop))

        if (
            sample_property_threshold is not None
            and utils.is_dictionary(sample_property_threshold)
        ):
            message = (
                "Property threshold filter is applied to store samples in "
                + f"'{self.sample_data_file[0]:s}'!\n"
                + f" {'Property':<12s}   Threshold\n"
                + "-"*30 + "\n")
            for prop, value in sample_property_threshold.items():
                if prop not in available_properties:
                    raise ValueError(
                        "Property threshold dictionary "
                        + "'sample_property_threshold' contains invalid "
                        + f"property '{prop:s}' which is not a calculator "
                        + f"property:\n{available_properties}")
                if utils.is_numeric(value):
                    sample_property_threshold[prop] = float(value)
                    message += (
                        f" {prop:>12} > "
                        + f"{sample_property_threshold[prop]:.2E}"
                        + f" {available_unit_properties[prop]:s}\n")
                else:
                    raise ValueError(
                        "Property threshold dictionary "
                        + "'sample_property_threshold' contains invalid "
                        + f"threshold value for property '{prop:s}' of type "
                        + f"'{value}'!")
            self.logger.info(message)
        elif sample_property_threshold is not None:
            raise ValueError(
                "Property threshold dictionary 'sample_property_threshold' "
                + "must be a dictionary of a valid property (key) and "
                + "threshold value (value), but is of type "
                + f"'{type(sample_property_threshold)}'!")

        return

    def check_nsamples_threshold(
        self
    ):
        """
        Check sample number threshold input

        """

        if (
            self.sample_nsamples_threshold is not None
            and utils.is_integer(self.sample_nsamples_threshold)
            and self.sample_nsamples_threshold > 0
        ):
            message = (
                "Sample number threshold filter for a maximum of "
                + f"{self.sample_nsamples_threshold:d} is applied that are "
                + f"stored in '{self.sample_data_file[0]:s}'!\n")
            self.logger.info(message)
        elif self.sample_nsamples_threshold is not None:
            raise ValueError(
                "Sample number threshold input 'sample_nsamples_threshold' "
                + "must be a positive integer but is of type "
                + f"'{type(self.sample_nsamples_threshold)}'!")

        return
