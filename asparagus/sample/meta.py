import os
import queue
import logging
import threading
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import ase
from ase.constraints import Hookean
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from .. import settings
from .. import utils
from .. import sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['MetaSampler']


class MetaSampler(sample.Sampler):
    """
    Meta(-Dynamic) Sampler class

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    meta_cv: list(list(int)), optional, default []
        List of sublists defining collective variables (CVs) / reaction
        coordinates to add Gaussian potentials. The number of atom indices
        in the sublist defines either bonds (2), angles (3) or
        dihedrals (4). Example [[1, 2], [1, 2, 3], [4, 1, 2, 3]]
    meta_gaussian_height: float, optional, default 0.05
        Potential energy height in eV of the Gaussian potential.
    meta_gaussian_widths: (float, list(floats)), optional, default 0.1
        Gaussian width for all CVs or a list of widths per CV that define
        the FWHM of Gaussian potential.
    meta_gaussian_interval: int, optional, default 10
        Step interval to add Gaussian potential at current set of
        collective variable.
    meta_hookean: list(list(int,float)), optional, default []
        It is always recommended for bond type collective variables to
        define a hookean constraint that limit the distance between two
        atoms. Otherwise gaussian potential could be only added to an ever
        increasing atoms bond distance. Hookean are defined by a list of
        sublists containing first two atom indices followed by one upper
        distance limit and, optionally, a Hookean force constant k (default
        is defined by meta_hookean_force_constant).
        For example: [1, 2, 4.0] or [1, 2, 4.0, 5.0]
    meta_hookean_force_constant: float, optional, default 5.0
        Default Hookean force constant if not specifically defined in
        Hookean constraint list meta_hookean.
    meta_temperature: float, optional, default 300
        Target temperature in Kelvin of the MD simulation controlled by a
        Langevin thermostat
    meta_time_step: float, optional, default 1.0 (1 fs)
        MD Simulation time step in fs
    meta_simulation_time: float, optional, default 1E5 (100 ps)
        Total MD Simulation time in fs
    meta_save_interval: int, optional, default 10
        MD Simulation step interval to store system properties of
        the current frame to dataset.
    meta_langevin_friction: float, optional, default 0.1
        Langevin thermostat friction coefficient in Kelvin. The coefficient
        should be much higher than in classical MD simulations  (1E-2 to
        1E-4) due to the fast heating of the systems due to the Gaussian
        potentials.
    meta_initial_velocities: bool, optional, default False
        Instruction flag if initial atom velocities are assigned with
        respect to a Maxwell-Boltzmann distribution at temperature
        'md_initial_temperature'.
    meta_initial_temperature: float, optional, default 300
        Temperature for initial atom velocities according to a Maxwell-
        Boltzmann distribution.

    Returns
    -------
    object
        Meta(-Dynamics) Sampler class object
    """

    # Default arguments for sample module
    sample.Sampler._default_args.update({
        'meta_cv':                      [],
        'meta_gaussian_height':         0.05,
        'meta_gaussian_widths':         0.1,
        'meta_gaussian_interval':       10,
        'meta_hookean':                 [],
        'meta_hookean_force_constant':  5.0,
        'meta_temperature':             300.,
        'meta_time_step':               1.,
        'meta_simulation_time':         1.E5,
        'meta_save_interval':           100,
        'meta_langevin_friction':       1.E-0,
        'meta_initial_velocities':      False,
        'meta_initial_temperature':     300.,
        'meta_parallel':                False,
        })
    
    # Expected data types of input variables
    sample.Sampler._dtypes_args.update({
        'meta_cv':                      [utils.is_array_like],
        'meta_gaussian_height':         [utils.is_numeric],
        'meta_gaussian_width':          [
            utils.is_numeric, utils.is_numeric_array],
        'meta_gaussian_interval':       [utils.is_integer],
        'meta_hookean':                 [utils.is_array_like],
        'meta_hookean_force_constant':  [utils.is_numeric],
        'meta_temperature':             [utils.is_numeric],
        'meta_time_step':               [utils.is_numeric],
        'meta_simulation_time':         [utils.is_numeric],
        'meta_save_interval':           [utils.is_integer],
        'meta_langevin_friction':       [utils.is_numeric],
        'meta_initial_velocities':      [utils.is_bool],
        'meta_initial_temperature':     [utils.is_numeric],
        'meta_parallel':                [utils.is_bool],
        })

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        meta_cv: Optional[List[int]] = None,
        meta_gaussian_height: Optional[float] = None,
        meta_gaussian_widths: Optional[Union[float, List[float]]] = None,
        meta_gaussian_interval: Optional[int] = None,
        meta_hookean: Optional[List[Union[int, float]]] = None,
        meta_hookean_force_constant: Optional[float] = None,
        meta_temperature: Optional[float] = None,
        meta_time_step: Optional[float] = None,
        meta_simulation_time: Optional[float] = None,
        meta_save_interval: Optional[float] = None,
        meta_langevin_friction: Optional[float] = None,
        meta_initial_velocities: Optional[bool] = None,
        meta_initial_temperature: Optional[float] = None,
        meta_parallel: Optional[bool] = None,
        **kwargs,
    ):
        """
        Initialize Normal Mode Scanning class

        """
        
        # Sampler class label
        self.sample_tag = 'meta'

        # Initialize parent class
        super().__init__(
            sample_tag=self.sample_tag,
            config=config,
            config_file=config_file,
            **kwargs
            )
        
        ##################################
        # # # Check Meta Class Input # # #
        ##################################

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

        # Get number of collective variables
        Ncv = len(self.meta_cv)
        
        # Check collective variables
        self.cv_type = []
        self.cv_type_dict = {
            2: 'bond',
            3: 'angle',
            4: 'dihedral',
            5: 'reactive_bond'}
        for icv, cv in enumerate(self.meta_cv):
            
            # Check cv data type
            if not utils.is_array_like(cv):
                raise ValueError(
                    f"Collective variable number {icv:d} is not an integer "
                    + f"list but of type '{type(cv)}'!")
            
            # Get cv type
            if self.cv_type_dict.get(len(cv)) is None:
                raise ValueError(
                    f"Collective variable number {icv:} is not of valid "
                    + f"length but of length '{len(cv):d}'!")
            else:
                self.cv_type.append(self.cv_type_dict.get(len(cv)))
        
        # Check meta sampling input format
        if utils.is_numeric(self.meta_gaussian_widths):
            self.meta_gaussian_widths = [self.meta_gaussian_widths]*Ncv
        elif Ncv > len(self.meta_gaussian_widths):
            raise ValueError(
                f"Unsufficient number of gaussian width defined "
                + f"({len(self.meta_gaussian_widths):d}) for the number "
                + f"of collective variables with {Ncv:d}!")
        
        # Check Hookean constraints
        for ihk, hk in enumerate(self.meta_hookean):
            
            # Check Hookean list data type
            if not utils.is_numeric_array(hk):
                raise ValueError(
                    f"Hookean constraint number {ihk:d} is not a numeric "
                    + f"list but of type '{type(hk):s}'!")
            
            # Check Hookean constraint definition validity
            if len(hk)==3:
                # Add default Hookean force constant
                hk.append(self.meta_hookean_force_constant)
            elif not len(hk)==4:
                raise ValueError(
                    f"Hookean constraint number {ihk:d} is expected to be "
                    + f"length 3 or 4 but has a length of {len(hk):d}!")
            
            # Check atom definition type
            for ii, idx in enumerate(hk[:2]):
                if not utils.is_integer(idx):
                    raise ValueError(
                        f"Atom index {ii:d} in Hookean constraint number "
                        + f"{ihk:d} is not an integer but of type "
                        + f"{type(idx):s}!")

        # Define collective variable log file paths
        self.meta_gaussian_log_file = os.path.join(
            self.sample_directory, 
            f'{self.sample_counter:d}_{self.sample_tag:s}_{{:d}}_gaussian.log')
        
        # Check sample properties for energy and forces properties which are 
        # required for Meta sampling
        if 'energy' not in self.sample_properties:
            self.sample_properties.append('energy')
        if 'forces' not in self.sample_properties:
            self.sample_properties.append('forces')

        return

    def get_info(self):
        """
        Get information about the current sampler object

        Returns
        -------
        dict
            Dictionary with sampler information
        """
        
        info = super().get_info()
        info.update({
            'meta_cv': self.meta_cv,        
            'meta_gaussian_height': self.meta_gaussian_height,
            'meta_gaussian_widths': self.meta_gaussian_widths,
            'meta_gaussian_interval': self.meta_gaussian_interval,
            'meta_hookean': self.meta_hookean,
            'meta_hookean_force_constant': self.meta_hookean_force_constant,
            'meta_temperature': self.meta_temperature,
            'meta_time_step': self.meta_time_step,
            'meta_simulation_time': self.meta_simulation_time,
            'meta_save_interval': self.meta_save_interval,
            'meta_langevin_friction': self.meta_langevin_friction,
            'meta_initial_velocities': self.meta_initial_velocities,
            'meta_initial_temperature': self.meta_initial_temperature,
            'meta_parallel': self.meta_parallel,
            })
        
        return info

    def run_systems(
        self,
        sample_systems_queue: Optional[queue.Queue] = None,
        **kwargs,
    ):
        """
        Perform Meta-Dynamics simulations on the sample system.
        
        Parameters
        ----------
        sample_systems_queue: queue.Queue, optional, default None
            Queue object including sample systems or to which 'sample_systems' 
            input will be added. If not defined, an empty queue will be 
            assigned.
        """

        # Check sample system queue
        if sample_systems_queue is None:
            sample_systems_queue = queue.Queue
        
        # Initialize thread continuation flag
        self.thread_keep_going = np.array(
            [True for ithread in range(self.sample_num_threads)],
            dtype=bool
            )
        
        # Add stop flag
        for _ in range(self.sample_num_threads):
            sample_systems_queue.put('stop')

        if self.sample_num_threads == 1 or self.meta_parallel:
            
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

    def run_system(
        self, 
        sample_systems_queue: queue.Queue,
        ithread: Optional[int] = None,
    ):
        """
        Perform a coarse Meta-Dynamics simulation with the sample systems.

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

            # Initialize Gaussian log file
            gaussian_log_file = self.meta_gaussian_log_file.format(isample)
            
            # Perform parallel meta dynamics simulation
            if self.meta_parallel and self.sample_num_threads > 1:
                
                # Prepare parallel meta dynamics simulations
                Nsamples = [0]*self.sample_num_threads
                systems = [
                    system.copy() for _ in range(self.sample_num_threads)]

                # Create threads for job calculations
                threads = [
                    threading.Thread(
                        target=self.run_meta,
                        args=(systems[ithread], ),
                        kwargs={
                            'gaussian_log_file': gaussian_log_file,
                            'log_file': sample_log_file,
                            'trajectory_file': trajectory_file,
                            'Nsamples': Nsamples,
                            'ithread': ithread},
                        )
                    for ithread in range(self.sample_num_threads)]

                # Start threads
                for thread in threads:
                    thread.start()
                    
                # Wait for threads to finish
                for thread in threads:
                    thread.join()
                    
                # Sum number of stored system samples
                Nsample = np.sum(Nsamples)
                
            # Perform single meta dynamics simulation
            else:
            
                Nsample = self.run_meta(
                    system, 
                    gaussian_log_file=gaussian_log_file,
                    log_file=sample_log_file,
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
            
    def run_meta(
        self,
        system: ase.Atoms,
        cv: Optional[List[int]] = None,
        gaussian_height: Optional[float] = None,
        gaussian_widths: Optional[float] = None,
        gaussian_interval: Optional[int] = None,
        gaussian_log_file: Optional[str] = None,
        hookean: Optional[List[Union[int, float]]] = None,
        hookean_force_constant: Optional[float] = None,
        temperature: Optional[float] = None,
        time_step: Optional[float] = None,
        simulation_time: Optional[float] = None,
        langevin_friction: Optional[float] = None,
        initial_velocities: Optional[bool] = None,
        initial_temperature: Optional[float] = None,
        log_file: Optional[str] = None,
        trajectory_file: Optional[str] = None,
        Nsamples: Optional[List[int]] = None,
        ithread: Optional[int] = None,
    ):
        """
        This does a Meta Dynamics simulation using a Meta constraint with a
        Langevin thermostat and verlocity Verlet algorithm for an NVT ensemble.

        Parameters
        ----------
        system: ase.Atoms
            System to be sampled.
        cv: list(int), optional, default None
            List of sublists defining collective variables (CVs).
        gaussian_height: float, optional, default None
            Potential energy height in eV of the Gaussian potential.
        gaussian_widths: (float, list(floats)), optional, default None
            Gaussian width for all CVs or a list of widths per CV that define
            the FWHM of Gaussian potential.
        gaussian_interval: int, optional, default None
            Step interval to add Gaussian potential.
        gaussian_log_file: str, optional, default None
            Collective variable position log file path
        hookean: list(list(int,float)), optional, default None
            Hookean constraint informations that limit bond distances.
        hookean_force_constant: float, optional, default None
            Default Hookean force constant if not specifically defined in
            Hookean constraint list 'meta_hookean'.
        temperature: float, optional, default None
            Meta dynamics simulation temperature in Kelvin
        time_step: float, optional, default None
            Meta dynamics simulation time step in fs
        simulation_time: float, optional, default None
            Total meta dynamics simulation time in fs
        langevin_friction: float, optional, default None
            Langevin thermostat friction coefficient in Kelvin.
        initial_velocities: bool, optional, default None
            Instruction flag if initial atom velocities are assigned.
        initial_temperature: float, optional, default None
            Temperature for initial atom velocities according to a Maxwell-
            Boltzmann distribution.
        log_file: str, optional, default None
            Log file for sampling information
        trajectory_file: str, optional, default None
            ASE Trajectory file path to append sampled system if requested
        Nsamples: list(int), optional, default None
            List for number of sampled systems per thread
        ithread: int, optional, default None
            Thread number
        
        Return
        ------
        int
            Number of sampled systems to database
        """

        # Check input parameters
        if cv is None:
            cv = self.meta_cv
        if gaussian_height is None:
            gaussian_height = self.meta_gaussian_height
        if gaussian_widths is None:
            gaussian_widths = self.meta_gaussian_widths
        if gaussian_interval is None:
            gaussian_interval = self.meta_gaussian_interval
        if gaussian_log_file is None:
            gaussian_log_file = self.meta_gaussian_log_file.format(0)
        if hookean is None:
            hookean = self.meta_hookean
        if hookean_force_constant is None:
            hookean_force_constant = self.meta_hookean_force_constant
        if temperature is None:
            temperature = self.meta_temperature
        if time_step is None:
            time_step = self.meta_time_step
        if simulation_time is None:
            simulation_time = self.meta_simulation_time
        if langevin_friction is None:
            langevin_friction = self.meta_langevin_friction
        if initial_velocities is None:
            initial_velocities = self.meta_initial_velocities
        if initial_temperature is None:
            initial_temperature = self.meta_initial_temperature    
    
        # Initialize stored sample counter
        Nsample = 0
        
        # Assign calculator
        system = self.assign_calculator(
            system,
            ithread=ithread)

        # Current system constraints
        system_constraints = system.constraints

        # Initialize meta dynamic constraint
        meta_constraint = MetaConstraint(
            cv,
            gaussian_widths,
            gaussian_height,
            gaussian_log_file,
            gaussian_interval)
        
        # Initialize Hookean constraint
        hookean_constraint = []
        for hk in hookean:
            hookean_constraint.append(
                Hookean(hk[0], hk[1], hk[2], rt=hk[3]))
        
        # Set constraints to system
        system.set_constraint(
            [meta_constraint] + hookean_constraint + system_constraints)

        # Set initial atom velocities if requested
        if initial_velocities:
            MaxwellBoltzmannDistribution(
                system, 
                temperature_K=initial_temperature)
        
        # Initialize MD simulation propagator
        meta_dyn = Langevin(
            system, 
            timestep=time_step*units.fs,
            temperature_K=temperature,
            friction=langevin_friction,
            logfile=log_file,
            loginterval=self.meta_save_interval)
        
        # Attach system properties saving function
        meta_dyn.attach(
            self.save_properties,
            interval=self.meta_save_interval,
            system=system,
            Nsample=Nsample)
        
        # Attach trajectory
        if self.sample_save_trajectory:
            meta_dyn.attach(
                self.write_trajectory, 
                interval=self.meta_save_interval,
                system=system,
                trajectory_file=trajectory_file)

        # Attach collective variables writer
        meta_cv_logger = MetaDynamicLogger(
            meta_constraint, 
            system)
        meta_dyn.attach(
            meta_cv_logger.log, 
            interval=gaussian_interval)
        
        # Run MD simulation
        simulation_steps = round(
            simulation_time/time_step)
        meta_dyn.run(simulation_steps)
        
        # As function attachment to ASE Dynamics class does not provide a 
        # return option of Nsample, guess attached samples
        Nsample = simulation_steps//self.meta_save_interval + 1
        
        # Assign Nsample
        if Nsamples is not None:
            Nsamples[ithread] = Nsample

        return Nsample
        

class MetaConstraint:
    """
    Constraint class to perform Meta Dynamics simulations by adding
    artificial Gaussian potentials.

    Forces atoms of cluster to stay close to the center.

    Parameters
    ----------
    cv : list
        List of collective variables - e.g.:
        [[1,2], [2,4], [1,2,3], [2,3,4,1]]
        [a,b] bond variable between atom a and b
        [a,b,c] angle variable between angle a->b->c
        [a,b,c,d] dihedral angle variable between angle a->b->c->d
    widths : array
        Width if Gaussian distribution for cv i
    heights : array
        Maximum height of the artificial Gaussian potential
    """
    
    def __init__(
        self, 
        cv, 
        widths, 
        height, 
        logfile, 
        logwrite,
    ):
        """
        Initialize meta-dynamics constraint class
        """
        
        self.cv = cv
        self.Ncv = len(cv)
        self.widths = np.asarray(widths)
        self.height = height
        self.logfile = logfile
        self.logwrite = logwrite
        
        self.cv_list = np.array([], dtype=float)
        self.last_cv = None
        self.ilog = 0
        
        if self.Ncv != len(self.widths):
            raise IOError(
                'Please provide an array of width values of them' +
                'same lengths as the number of collective variables cv.')
        
        self.removed_dof = 0
                
    def get_removed_dof(self, atoms):
        return 0

    def todict(self):
        dct = {'name': 'MetaConstraints'}
        dct['kwargs'] = {
            'cv': self.cv,
            'widths': self.widths,
            'height': self.height}
        return dct
    
    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_momenta(self, atoms, momenta):
        pass
    
    def adjust_forces(self, atoms, forces):
        """
        Adjust the forces related to artificial Gaussian potential

        Parameters
        ----------
        atoms : ase.Atoms
            Atoms object
        forces : array
            Forces array of atoms object

        """
        
        # Get collective variable i and partial derivative dcvdR
        cv = np.zeros(self.Ncv, dtype=float)
        dcvdR = np.zeros([self.Ncv, *forces.shape], dtype=float)
        for icv, cvi in enumerate(self.cv):
            # Bond length
            if len(cvi)==2:
                cv[icv] = np.linalg.norm(
                    atoms.positions[cvi[0]] - atoms.positions[cvi[1]])
                dcvdR[icv, cvi[0]] = (
                    (atoms.positions[cvi[1]] - atoms.positions[cvi[0]])/cv[icv])
                dcvdR[icv, cvi[1]] = (
                    (atoms.positions[cvi[0]] - atoms.positions[cvi[1]])/cv[icv])
            # Angle
            elif len(cvi)==3:
                # Connection vectors
                ab = atoms.positions[cvi[0]] - atoms.positions[cvi[1]]
                cb = atoms.positions[cvi[2]] - atoms.positions[cvi[1]]
                # Vector distances
                d2ab = np.sum(ab**2)
                dab = np.sqrt(d2ab)
                d2cb = np.sum(cb**2)
                dcb = np.sqrt(d2cb)
                # Vector angles
                cabbc = np.sum(ab*cb)/(dab*dcb)
                cabbc = np.clip(cabbc, -1.0, 1.0)
                sabbc = np.sqrt(1.0 - cabbc**2)
                sabbc = np.clip(sabbc, 1.e-8, None)
                tabbc = np.arccos(cabbc)
                # Add to current collective variable
                cv[icv] = tabbc
                # Compute gradients dcvdRi
                a11 = sabbc*cabbc/d2ab
                a12 = -sabbc/(dab*dcb)
                a22 = sabbc*cabbc/d2cb
                fab = a11*ab + a12*cb
                fcb = a22*cb + a12*ab
                dcvdR[icv, cvi[0]] = -fab
                dcvdR[icv, cvi[1]] = fab + fcb
                dcvdR[icv, cvi[2]] = -fcb
            # Dihedral angle
            elif len(cvi)==4:
                # Connection vectors
                ab = atoms.positions[cvi[0]] - atoms.positions[cvi[1]]
                cb = atoms.positions[cvi[2]] - atoms.positions[cvi[1]]
                dc = atoms.positions[cvi[3]] - atoms.positions[cvi[2]]
                bc = -cb
                ## Plane vectors
                aa = np.cross(ab, bc)
                bb = np.cross(dc, bc)
                ## Vector distances
                d2aa = np.sum(aa**2)
                d2bb = np.sum(bb**2)
                d2bc = np.sum(bc**2)
                dbc = np.sqrt(d2bc)
                dinvab = 1./np.sqrt(d2aa*d2bb)
                # Vector angles
                cabcd = dinvab*np.sum(aa*bb)
                cabcd = np.clip(cabcd, -1.0, 1.0)
                sabcd = dbc*dinvab*np.sum(aa*dc)
                ## Dihedral multiplicity scalars 
                ## (multiplicity: number of minima in dihedral 0° to 360°)
                ## Here: multiplicity equal 1
                #p, dfab, ddfab = 1.0, 0.0, 0.0
                #for ii in range(1):
                    #ddfab = p*cabcd - dfab*sabcd
                    #dfab = p*sabcd + dfab*cabcd
                    #p = ddfab
                raise NotImplementedError
            # Reaction coordinate
            elif len(cvi)==5:
                if cvi[0]=='-':
                    a = np.linalg.norm(
                        atoms.positions[cvi[1]] - atoms.positions[cvi[2]])
                    b = np.linalg.norm(
                        atoms.positions[cvi[3]] - atoms.positions[cvi[4]])
                    cv[icv] = a - b
                    dcvdR[icv, cvi[1]] = (
                        (atoms.positions[cvi[2]] - atoms.positions[cvi[1]])/a)
                    dcvdR[icv, cvi[2]] = (
                        (atoms.positions[cvi[1]] - atoms.positions[cvi[2]])/a)
                    dcvdR[icv, cvi[3]] = (
                        (atoms.positions[cvi[3]] - atoms.positions[cvi[4]])/b)
                    dcvdR[icv, cvi[4]] = (
                        (atoms.positions[cvi[4]] - atoms.positions[cvi[3]])/b)
                else:
                    raise NotImplementedError
            else:
                raise IOError('Check cv list of Metadynamic constraint!')
        
        # Put to last cv
        self.last_cv = cv.copy()
        
        # If CV list is empty return zero forces
        # (Important to put it after last_cv update for the add_to_cv function)
        if self.cv_list.shape[0]==0:
            return np.zeros_like(forces)
        
        # Compute Gaussian exponents
        exponents = np.sum(
            (self.cv_list - np.expand_dims(cv, axis=0))**2
            /np.expand_dims(2.0*self.widths**2, axis=0),
            axis=1)
        #(Nlist)
        
        # Compute Gaussians
        gaussians = -1.0*self.height*np.exp(-exponents)
        #(Nlist)
        
        # Compute partial derivative d exponent d cv
        dexpdcv = (
            (self.cv_list -  np.expand_dims(cv, axis=0))
            /np.expand_dims(self.widths**2, axis=0))
        #(Nlist, Ncv)
        
        # Add up gradient with respective to cv
        dgausdcv = np.sum(np.expand_dims(gaussians, axis=1)*dexpdcv, axis=0)
        #(Ncv)
        
        # Compute gradient with respect to Cartesian
        gradient = np.sum(np.expand_dims(dgausdcv, axis=(1,2))*dcvdR, axis=0)
        #(Natoms, Ncart)
        
        forces -= gradient
        
        return
        
    def adjust_potential_energy(self, atoms):
        """Returns the artificial Gaussian potential"""
        
        # Get collective variable i
        cv = np.zeros(self.Ncv, dtype=float)
        for icv, cvi in enumerate(self.cv):
            
            # Bond length
            if len(cvi)==2:
                cv[icv] = np.linalg.norm(
                    atoms.positions[cvi[0]] - atoms.positions[cvi[1]])
            # Angle
            elif len(cvi)==3:
                # Connection vectors
                ab = atoms.positions[cvi[0]] - atoms.positions[cvi[1]]
                cb = atoms.positions[cvi[2]] - atoms.positions[cvi[1]]
                # Vector distances
                d2ab = np.sum(ab**2)
                dab = np.sqrt(d2ab)
                d2cb = np.sum(cb**2)
                dcb = np.sqrt(d2cb)
                # Vector angles
                cabbc = np.sum(ab*cb)/(dab*dcb)
                cabbc = np.clip(cabbc, -1.0, 1.0)
                tabbc = np.arccos(cabbc)
                # Add to current collective variable
                cv[icv] = tabbc
            # Dihedral angle
            elif len(cvi)==4:
                raise NotImplementedError
            # Special
            elif len(cvi)==5:
                if cvi[0]=='-':
                    a = np.linalg.norm(
                        atoms.positions[cvi[1]] - atoms.positions[cvi[2]])
                    b = np.linalg.norm(
                        atoms.positions[cvi[3]] - atoms.positions[cvi[4]])
                    cv[icv] = a - b
                else:
                    raise NotImplementedError
            else:
                raise IOError('Check cv list of Metadynamic constraint!')
        
        # Update to last cv
        self.last_cv = cv.copy()
        
        # If CV list is empty return zero potential
        # (Important to put it after last_cv update for the add_to_cv function)
        if self.cv_list.shape[0]==0:
            return 0.0
        
        # Compute Gaussian exponents
        exponents = np.sum(
            (self.cv_list - np.expand_dims(cv, axis=0))**2
            /np.expand_dims(2.0*self.widths**2, axis=0),
            axis=1)
        #(Nlist)
        
        # Compute gaussians
        gaussians = self.height*np.exp(-exponents)
        #(Nlist)
        
        # Add up potential
        potential = np.sum(gaussians)
        
        return potential
    
    def add_to_cv(self, atoms):
        
        # Get collective variable i
        if self.last_cv is None:
            return
        else:
            cv = self.last_cv
        
        # Add cv to list
        if self.cv_list.shape[0]==0:
            self.cv_list = np.array([cv], dtype=float)
        else:
            self.cv_list = np.append(self.cv_list, [cv], axis=0)
        
        # Write cv list but just every logwrite time to avoid repeated writing 
        # of large files
        if not self.ilog%self.logwrite:
            np.savetxt(self.logfile, self.cv_list)
        self.ilog += 1
        
    def get_indices(self):
        raise NotImplementedError
        
    def index_shuffle(self, atoms, ind):
        raise NotImplementedError

    def __repr__(self):
        return (
            'Metadynamics constraint') 


class MetaDynamicLogger:
    """ 
    Store defined amount of images of a trajectory and save the images 
    within a defined time window around a point when an action happened.

    Parameters:
    ----------

    metadynamic: object
        Metadynamic constraint class
    """
    
    def __init__(self, metadynamic, system):
        """
        Initialize meta-dynamics collective variable position logger class.
        """
        
        # Allocate parameter
        self.metadynamic = metadynamic
        self.system = system
        
    def __exit__(self, exc_type, exc_value, tb):
        self.close()
        
    def log(self, system=None, **kwargs):
        """
        Execute 
        """
        
        if system is None:
            system = self.system
        self.metadynamic.add_to_cv(system)
        
        return
    
