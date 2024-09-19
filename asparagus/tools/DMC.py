# Adaptation of diffusion Monte Carlo code from 
# https://github.com/MMunibas/dmc_gpu_PhysNet/tree/main
import os
import time
from datetime import datetime
from typing import Optional, List, Dict, Callable, Tuple, Union, Any

import numpy as np

import torch

import ase
from ase import io
from ase.optimize import BFGS

from asparagus import Asparagus
from asparagus import interface
from asparagus import utils

__all__ = ['DMC','Logger_DMC']

class DMC:
    """
    This code adapt the diffusion Monte Carlo code from 
    https://github.com/MMunibas/dmc_gpu_PhysNet/tree/main to the asparagus 
    framework.

    As a difference with the original implementation, this code reads
    the initial coordinates and the equilibrium coordinates from the atoms
    object in ASE.

    **Original Message**
    DMC code for the calculation of zero-point energies using PhysNet based 
    PESs on GPUs/CPUs. The calculation is performed in Cartesian coordinates.
    See e.g. American Journal of Physics 64, 633 (1996);
    https://doi.org/10.1119/1.18168 for DMC implementation and 
    https://scholarblogs.emory.edu/bowman/diffusion-monte-carlo/ for Fortran 90 
    implementation.

    Parameters
    ----------
    atoms: (str, ase.Atoms)
        File path to a structure file or an ase.Atoms object of the system
    model_calculator: (str, Callable)
        Either a file path to an Asparagus config file ('.json') or an
        Asparagus main class instance or initialized calculator.
    charge: float, optional, default None
        Total charge of the system. If the charge is required for the
        calculator initialization but 'None', a charge of 0 is assumed.
    optimize: bool, optional, default True
        Perform a structure optimization to determine the minimum potential.
    nwalker: int, optional, default 100
        Number of walkers for the DMC
    nsteps: int, optional, default 1000
        Number of steps for each of the DMC walkers
    eqsteps: int, optional, default 100
        Number of initial equilibration steps for the DMC walkers
    stepsize: float, optional, default 0.1
        Step size for the DMC in imaginary time
    alpha: float, optional, default 1.0
        Alpha parameter for the DMC feed-back parameter, usually proportional
        to 1/stepsize
    max_batch: int, optional, default 128
        Maximum size of the batch to compute walker energies
    seed: int, optional, default: np.random.randint(1E6)
        Specify random seed for atom positions shuffling
    initial_positions: (str, list(float), ase.Atoms), optional, default None
        Initial coordinates for the DMC system as a list of the correct 
        shape in Angstrom, a file path to the respective ASE Atoms file or
        even an ASE Atoms object at initial coordinates itself.
        If None, 'atoms' positions are taken as initial coordinates.
    filename: str, optional, default 'dmc'
        Name tag of the output file where results are going to be stored. 
        The DMC code create 4 files with the same name but different 
        extensions: '.pot', '.log' and '.xyz'. 
        '.pot': Potential energies of the DMC runs.
        '.log': Log information of the runs.
        '.xyz': Two '.xyz' files are generated where the last 10 steps of the
            simulation and the defective geometries are saved respectively.

    """

    def __init__(self,
        atoms: Union[str, ase.Atoms],
        model_calculator: Union[str, Callable] = None,
        charge: Optional[float] = None,
        optimize: Optional[bool] = True,
        nwalker: Optional[int] = 100,
        nsteps: Optional[int] = 1000,
        eqsteps: Optional[int] = 100,
        stepsize: Optional[float] = 0.1,
        alpha: Optional[float] = 1.0,
        max_batch: Optional[int] = 128,
        seed: Optional[int] = np.random.randint(1E6),
        initial_positions: Optional[Union[str, List[float], ase.Atoms]] = None,
        filename: Optional[str] = None,
        **kwargs
    ):

        # Check atoms input
        if utils.is_string(atoms):
            self.atoms = io.read(atoms)
        elif utils.is_ase_atoms(atoms):
            self.atoms = atoms
        else:
            raise ValueError(
                f"Invalid 'atoms' input of type '{type(atoms)}'.\n"
                + "Required is either a path to a structure file "
                + "(e.g. '.xyz') or an ASE Atoms object.")

        # Check model calculator input
        if model_calculator is None:
            raise ValueError(
                "A model calculator must be assigned!\nEither as a file path"
                + "to an Asparagus config file, as a Asparagus instance or an "
                + "Aspargus calculator itself.")
        if utils.is_string(model_calculator):
            model = Asparagus(config=model_calculator)
            self.model_calculator = model.get_model_calculator()
            self.model_unit_properties = model.model_unit_properties
        else:
            if hasattr(model_calculator, 'get_model_calculator'):
                self.model_calculator = model_calculator.get_model_calculator()
                self.model_unit_properties = (
                    self.model_calculator.model_unit_properties)
            elif hasattr(model_calculator, 'model_unit_properties'):
                self.model_calculator = model_calculator
                self.model_unit_properties = (
                    model_calculator.model_unit_properties)
            else:
                raise ValueError(
                    f"The model calculator is of unknown type!\n")
        
        # Read device and dtype information
        self.device = self.model_calculator.device
        self.dtype = self.model_calculator.dtype

        #self.mcalc = torch.compile(model_calculator.get_model_calculator())
        #self.mcalc = model_calculator.get_model_calculator()

        # Check system charge
        if charge is None:
            try:
                atomic_charges = atoms.get_charges()
            except RuntimeError:
                atomic_charges = atoms.get_initial_charges()
            self.charge = np.sum(atomic_charges)
        elif utils.is_numeric(charge):
            self.charge = charge
        else:
            raise ValueError(
                f"Invalid 'charge' input of type '{type(charge)}'.\n"
                + "Required is either a numeric system charge input or no "
                + "input (None) to assign a charge from the ASE Atoms object.")

        # Check optimization flag
        self.optimize = bool(optimize)

        # Check DMC input
        # Number of walkers for the DMC
        if utils.is_numeric(nwalker):
            self.nwalker = int(nwalker)
        else:
            raise ValueError(
                f"Invalid 'nwalker' input of type '{type(nwalker)}'.\n"
                + "Required is a numeric input!")
        # Number of steps for the DMC
        if utils.is_numeric(nsteps):
            self.nsteps = int(nsteps)
        else:
            raise ValueError(
                f"Invalid 'nsteps' input of type '{type(nsteps)}'.\n"
                + "Required is a numeric input!")
        # Number of equilibration steps for the DMC
        if utils.is_numeric(eqsteps):
            self.eqsteps = eqsteps
        else:
            raise ValueError(
                f"Invalid 'eqsteps' input of type '{type(eqsteps)}'.\n"
                + "Required is a numeric input!")
        # Step size for the DMC in imaginary time
        if utils.is_numeric(stepsize):
            self.stepsize = stepsize
        else:
            raise ValueError(
                f"Invalid 'stepsize' input of type '{type(stepsize)}'.\n"
                + "Required is a numeric input!")
        # Alpha parameter for the DMC
        if utils.is_numeric(alpha):
            self.alpha = alpha
        else:
            raise ValueError(
                f"Invalid 'alpha' input of type '{type(alpha)}'.\n"
                + "Required is a numeric input!")
        # Alpha parameter for the DMC
        if utils.is_numeric(alpha):
            self.alpha = alpha
        else:
            raise ValueError(
                f"Invalid 'alpha' input of type '{type(alpha)}'.\n"
                + "Required is a numeric input!")
        # Size of the batch
        if utils.is_numeric(max_batch):
            self.max_batch = int(max_batch)
        else:
            raise ValueError(
                f"Invalid 'max_batch' input of type '{type(max_batch)}'.\n"
                + "Required is a numeric input!")
        # Seed for random generator
        if utils.is_numeric(seed):
            self.seed = int(seed)
        else:
            raise ValueError(
                f"Invalid 'seed' input of type '{type(seed)}'.\n"
                + "Required is a numeric input!")

        # Check initial coordinates for the DMC runs
        if initial_positions is None:
            self.atoms_initial = self.atoms.copy()
        elif utils.is_string(initial_positions):
            self.atoms_initial = io.read(initial_positions)
        elif utils.is_numeric_array(initial_positions):
            if initial_positions.shape == (len(self.atoms), 3,):
                initial_positions = np.array(
                    initial_positions, dtype=float)
                self.atoms_initial = self.atoms.copy()
                self.atoms_initial.set_positions(initial_positions)
            else:
                raise ValueError(
                    "Invalid 'initial_positions' input shape "
                    + f"{initial_positions.shape:} but {(len(atoms), 3,):} is"
                    + "expected!")
        elif utils.is_ase_atoms(initial_positions):
            self.atoms_initial = initial_positions
        else:
            raise ValueError(
                "Invalid 'initial_positions' input of type "
                + f"'{type(initial_positions)}'!")

        # Check model properties and get unit conversion from model units to
        # atomic units.
        self.prepare_dmc()
        
        # Check output file name tag
        if filename is None:
            self.filename = "dmc"
        elif utils.is_string(filename):
            self.filename = filename
        else:
            raise ValueError(
                f"Invalid 'filename' input of type '{type(filename)}'!\n"
                + "Required is a string for a filename tag.")

        # Initialize DMC logger
        self.logger = Logger_DMC(
            self.filename,
            maxsteps=self.nsteps,
            maxwalker=3*self.nwalker)

        return

    def prepare_dmc(
        self,
    ):
        """
        Finalize the DMC preparation

        """

        # Check implemented properties
        if 'energy' not in self.model_calculator.model_properties:
            raise ValueError(
                "The model property 'energy ' is not predicted but required!")

        # ASE to model position and charge conversion
        self.ase_conversion = {}
        self.ase_conversion['positions'], _ = utils.check_units(
            None, self.model_unit_properties.get('positions'))
        self.ase_conversion['charge'], _ = utils.check_units(
            None, self.model_unit_properties.get('charge'))

        # Initialize conversion dictionary
        self.dmc_conversion = {}
        
        # Get positions and energy conversion factors from model to atomic
        # units
        self.dmc_conversion['positions'], _ = utils.check_units(
            self.model_unit_properties.get('positions'), 'Bohr')
        self.dmc_conversion['energy'], _ = utils.check_units(
            self.model_unit_properties.get('energy'), 'Hartree')

        # Get mass conversion factor from u to atomic units (electron mass)
        self.dmc_conversion['mass'] = 1822.88848

        # Prepare DMC related system information
        self.natoms = len(self.atoms)
        self.atomic_numbers = self.atoms.get_atomic_numbers()
        self.atomic_symbols = self.atoms.get_chemical_symbols()
        self.atomic_masses = (
            self.atoms.get_masses()*self.dmc_conversion['mass'])

        return

    def run(
        self,
        optimize: Optional[bool] = None,
        optimize_method: Optional[Callable] = BFGS,
        optimize_fmax: Optional[float] = 0.01,
        nwalker: Optional[int] = None,
        nsteps: Optional[int] = None,
        eqsteps: Optional[int] = None,
        stepsize: Optional[float] = None,
        alpha: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        """
        Run the diffusion Monte-Carlo simulation.

        Parameters
        ----------
        optimize: bool, optional, default None
            Perform a structure optimization to determine the minimum 
            potential. If None, initial option is taken.
        optimize_method: ase.optimizer, optional, default BFGS
            ASE optimizer to use for the structure optimization.
        optmize_fmax: float, optional, default 0.01
            Maximum force component value used as convergence threshold for
            the optimization.
        nwalker: int, optional, default None
            Number of walkers for the DMC
        nsteps: int, optional, default None
            Number of steps for each of the DMC walkers
        eqsteps: int, optional, default None
            Number of initial equilibration steps for the DMC walkers
        stepsize: float, optional, default None
            Step size for the DMC in imaginary time
        alpha: float, optional, default None
            Alpha parameter for the DMC feed-back parameter, usually proportional
            to 1/stepsize
        seed: int
            Specify random seed for atom positions shuffling

        """

        # Check input parameter
        if nwalker is None:
            nwalker = self.nwalker
        if nsteps is None:
            nsteps = self.nsteps
        if eqsteps is None:
            eqsteps = self.eqsteps
        if stepsize is None:
            stepsize = self.stepsize
        if alpha is None:
            alpha = self.alpha
        if seed is None:
            seed = self.seed

        # Write log file header
        self.logger.log_begin(
            nwalker,
            nsteps,
            eqsteps,
            stepsize,
            alpha,
            seed)

        # Set random generator seed
        np.random.seed(seed)

        # If requested, perform structure optimization
        if optimize is None:
            optimize = self.optimize
        if optimize:
            self.atoms = self.optimize_atoms(
                self.atoms,
                self.charge,
                optimize_method,
                optimize_fmax)

        # Create initial system batch information
        batch_initial = self.model_calculator.create_batch(
            self.atoms_initial,
            charge=self.charge,
            conversion=self.ase_conversion)
        batch_minimum = self.model_calculator.create_batch(
            self.atoms,
            charge=self.charge,
            conversion=self.ase_conversion)

        # First evaluation of the potential energy
        results_initial = self.model_calculator(
            batch_initial,
            no_derivation=True)
        energy_initial = (
            results_initial['energy'].cpu().detach().numpy()
            * self.dmc_conversion['energy'])
        results_minimum = self.model_calculator(
            batch_minimum,
            no_derivation=True)
        energy_minimum = (
            results_minimum['energy'].cpu().detach().numpy()
            * self.dmc_conversion['energy'])

        # Initialize the DMC
        w_positions, dmc_energy, w_status, w_stepsize = self.init_walker(
            energy_initial,
            energy_minimum,
            self.atoms_initial.get_positions(),
            nwalker,
            nsteps,
            eqsteps,
            stepsize)

        # Write initial state to log file
        self.logger.write_pot(np.sum(w_status), dmc_energy, step=0)

        # Run main DMC loop
        dmc_avg_energy = 0.0
        dmc_total_energy = 0.0
        w_positions_step = np.zeros_like(w_positions)
        for i in range(self.nsteps):

            # Start timer
            start_time = time.time()

            # Execute walk step
            w_positions_step = self.walk(
                w_positions,
                w_status,
                w_stepsize,
                w_positions_step)

            # Execute branching step
            w_positions, w_status, dmc_energy = self.branch(
                w_positions,
                w_positions_step,
                dmc_energy,
                w_status,
                dmc_total_energy,
                energy_minimum,
                nwalker,
                stepsize,
                alpha)

        exit()
        # Main loop of the DMC
        v_tot = 0.0
        for i in range(self.nsteps):
            start_time = time.time()
            psips[:psips_f[0], :, :] = self.walk(psips[:psips_f[0], :, :])
            psips, psips_f, v_ref = self.branch(
                self.initial_coord,
                self.atomic_masses,
                self.atomic_symbols,
                vmin,
                psips,
                psips_f,
                v_ref,
                v_tot)
            self.logger.write_pot(psips_f[0], v_ref, step=i + 1)

            if i > self.eqsteps:
                v_ave += v_ref

            if i > self.nsteps - 10:  # record the last 10 steps of the DMC simulation for visual inspection.
                self.logger.write_last(psips_f, psips, self.natoms, self.atomic_symbols)
            if i % 10 == 0:
                print("step:  ", i, "time/step:  ", time.time() - start_time, "nalive:   ", psips_f[0])      
        v_ave = v_ave / (self.nsteps - self.eqsteps)
        

        self.logger.write_log(v_ave)

        # terminate code and close log/pot files
        self.logger.log_end()
        self.logger.close_files()

        return

    def optimize_atoms(
        self,
        atoms: ase.Atoms,
        charge: float,
        optimize_method: Callable,
        optimize_fmax: float,
    ) -> ase.Atoms:
        """
        Perform a structure optimization of the atoms object using the
        model calculator
        
        Parameters
        ----------
        atoms: ase.Atoms
            ASE Atoms object to optimize
        charge: float
            System charge
        optimize_method: ase.optimizer
            ASE optimizer to use for the structure optimization.
        optimize_fmax: float
            Maximum force component value used as convergence threshold for
            the optimization.

        Returns
        -------
        ase.Atoms
            Optimized ASE atoms object

        """

        # Check for forces in model properties
        if not 'forces' in self.model_calculator.model_properties:
            self.logger.log_write(
                "Requested structure optimization are not possible!\n"
                + "The model calculator does not predict forces.\n")
            return atoms

        # Prepare ASE calculator of the model calculator
        ase_calculator = interface.ASE_Calculator(
            self.model_calculator,
            atoms=atoms,
            charge=charge)

        # Assign ASE calculator
        atoms.calc = ase_calculator
        
        # Initialize optimizer
        dyn = optimize_method(atoms)
        
        # Perform structure optimization
        dyn.run(fmax=optimize_fmax)

        return atoms

    def init_walker(
        self,
        energy_initial: float,
        energy_minimum: float,
        positions_initial: List[float],
        nwalker: int,
        nsteps: int,
        eqsteps: int,
        stepsize: float,
    ):
        """
        Initialize DMC simulation variables such as the atoms positions for
        each walker (w_positions) and the status of the walker (w_status).

        Parameters
        ----------
        energy_initial: float
            Potential energy of the initial atom configuration
        energy_minimum:
            Minimum potential energy of the atom configuration
        positions_initial: list(float)
            Initial atom positions assigned to the walkers
        nwalker: int
            Number of walkers for the DMC
        nsteps: int
            Number of steps for each of the DMC walkers
        eqsteps: int
            Number of initial equilibration steps for the DMC walkers
        stepsize: float
            Step size for the DMC in imaginary time

        Returns
        -------

        """

        # Prepare atom mass weighted step size
        w_stepsize = np.sqrt(stepsize)/self.atomic_masses

        # Initialize walker positions and walker status
        w_status = np.ones(3*nwalker, dtype=bool)
        w_status[nwalker:] = False
        w_positions = np.zeros([3*nwalker, self.natoms, 3], dtype=float)

        # Set initial atom positions to each walker
        w_positions[:] = positions_initial

        # Assign DMC energy shifted by the minimum potential energy
        dmc_energy = energy_initial - energy_minimum

        return w_positions, dmc_energy, w_status, w_stepsize

    def walk(
        self,
        w_positions: List[float],
        w_status: List[bool],
        w_stepsize: List[float],
        w_positions_step: List[float],
    ) -> List[float]:
        """
        Walk routine performs the diffusion process of the replicas/walkter by
        adding random displacement sqrt(deltatau)*rho to the atom positions of
        the alive replicas, where rho is random number from a Gaussian
        distribution

        Parameters
        ----------
        w_positions: list(float)
            Atom coordinates of the walkers
        w_status: list(bool)
            Walker status
        w_stepsize: list(float)
            Atom mass weighted step size
        w_positions_step: list(float)
            Array for the processed atom coordinates of the walkers

        Returns
        -------
        list(float)
            Processed atom coordinates of the walker

        """

        # Get size of the random number array
        size = w_positions[w_status].shape

        # Get random Gaussian distributed numbers
        rho = np.random.normal(size=size)

        # Get processed atom positions
        w_positions_step[w_status] = (
            w_positions[w_status]
            + rho*w_stepsize.reshape(1, -1, 1))

        return w_positions_step

    def compute_energies(
        self,
        w_positions: List[float],
        w_status: List[bool],
        w_nactive: Optional[int] = None,
    ) -> List[float]:
        """
        Compute walker potential energies

        Parameters
        ----------
        w_positions: list(float)
            Atom coordinates of the walkers
        w_status: list(bool)
            Walker status
        w_nactive: int, optional, default None
            Number of active walkers

        Returns
        -------
        list(float)
            Walker potential energies

        """

        # If not given, compute number of walkers which are active
        if w_nactive is None:
            w_nactive = np.sum(w_status)

        if w_nactive < self.max_batch:

            # If number of walkers is below the maximum batch size limit,
            # compute the walker energies in one batch
            w_batch = self.model_calculator.create_batch_copies(
                self.atoms,
                ncopies=w_nactive,
                positions=w_positions[w_status],
                charge=self.charge,
                conversion=self.ase_conversion)

            w_results = self.model_calculator(
                w_batch,
                no_derivation=True)
            w_energies = (
                w_results['energy'].cpu().detach().numpy()
                * self.dmc_conversion['energy'])

        else:

            # If number of walkers is larger than maximum batch size limit,
            # compute the walker energies in multiple batches
            w_energies = None
            w_end = 0
            while w_end < w_nactive:

                # Prepare batch start and end point parameters
                w_start = w_end
                w_end = w_end + self.max_batch
                if w_end > w_nactive:
                    w_end = w_nactive
                w_nbatch = w_end - w_start

                # Prepare and run batch calculation
                w_batch = self.model_calculator.create_batch_copies(
                    self.atoms,
                    ncopies=w_nbatch,
                    positions=w_positions[w_status][w_start:w_end],
                    charge=self.charge,
                    conversion=self.ase_conversion)

                w_results_batch = self.model_calculator(
                                w_batch,
                                no_derivation=True)
                w_energies_batch = (
                    w_results_batch['energy'].cpu().detach().numpy()
                    * self.dmc_conversion['energy'])

                # Assign batch energies
                if w_energies is None:
                    w_energies = torch.zeros(
                        w_nactive,
                        dtype=w_energies_batch.dtype,
                        device=w_energies_batch.device)
                w_energies[w_start:w_end] = w_energies_batch

        return w_energies

    def branch(
        self,
        w_positions: List[float],
        w_positions_step: List[float],
        dmc_energy: float,
        w_status: List[bool],
        dmc_total_energy: float,
        energy_minimum: float,
        nwalker: int,
        stepsize: float,
        alpha: float,
        defective_threshold: Optional[float] = -1.e-5,
    ):
        """
        Perform The birth-death (branching) process, which follows the
        diffusion step

        Parameters
        ----------
        w_positions: list(float)
            Atom coordinates of the walkers
        w_positions_step: list(float)
            Array for the processed atom coordinates of the walkers
        dmc_energy: float
            Current DMC reference energy to determine walker dying probability
        w_status: list(bool)
            Walker status
        dmc_total_energy:
            Total DMC run energy
        energy_minimum: float
            Atoms system minimum potential energy
        nwalker: int, optional, default None
            Number of walkers for the DMC
        stepsize: float, optional, default None
            Step size for the DMC in imaginary time
        alpha: float, optional, default None
            Alpha parameter for the DMC feed-back parameter, usually proportional
            to 1/stepsize
        defective_threshold: float, optional, default -1.0e-5
            Negative threshold of potential energy below the minimum potential
            energy to decide if walker atom positions predict wrong/defective
            model calculator results.

        """

        # Compute number of active walkers
        w_nactive = np.sum(w_status)

        # Update model calculator batch for new walker atom systems
        # and compute walker potential energies
        w_energies = self.compute_energies(
            w_positions_step,
            w_status,
            w_nactive=w_nactive)

        # Shift walker energies by the minimum potential energy
        w_energies = w_energies - energy_minimum

        # Check for energies that are lower than the minimum potential energy
        selection_defective = w_energies < defective_threshold
        flag_defective = np.any(selection_defective)

        # Write defective walkers to file and kill the respective walkers
        if flag_defective:
            self.logger.write_error(
                w_positions_step[w_status][selection_defective],
                w_energies[selection_defective],
                self.atomic_symbols)
            w_status[w_status][selection_defective] = False

        # DMC step acceptance criteria
        probability_threshold = (
            1.0 - np.exp((dmc_energy - w_energies)*stepsize))

        # Test whether one of the walkers has to die, most likely due to
        # high potential energy atom configuration
        probability_dicerole = np.random.uniform(size=w_nactive)
        probability_failed = probability_dicerole < probability_threshold
        if np.any(probability_failed):

            # Set walker status to dead and walker energies to zero
            w_status[probability_failed] = False
            w_energies[probability_failed] = 0.0

        # Compute new total DMC energy
        dmc_total_energy = dmc_total_energy + np.sum(w_energies)

        # Give birth to new walkers if walker energies are lower than the
        # reference DMC energy shown by negative probability threshold.
        probability_birth = probability_threshold < 0.0
        if np.any(probability_birth):

            for iw in np.where(probability_birth)[0]:

                # Skip defective walkers
                if selection_defective[iw]:
                    continue

                # Walker birth criteria
                threshold = -1.0*probability_threshold[iw]
                nbirth = int(threshold)

                # Test wether new walker(s) is(are) born
                dicerole = np.random.uniform()
                if dicerole < (threshold - nbirth):
                    nbirth += 1

                # Initialize new walker eventually
                for ib in range(nbirth):

                    w_positions[w_nactive] = w_positions[iw]
                    w_positions_step[w_nactive] = w_positions_step[iw]
                    w_status[w_nactive] = True
                    dmc_total_energy = dmc_total_energy + w_energies[iw]
                    w_nactive += 1

        # Assign accepted atom positions of active walkers as new walker
        # positions
        for iw, positions_new in enumerate(w_positions_step[w_status]):
            w_positions[iw] = positions_new
            w_status[iw] = True
        w_nactive = iw + 1
        w_positions[iw:] = 0.0
        w_status[iw:] = False
        w_positions_step[:] = 0.0

        # Update DMC reference energy
        dmc_energy = (
            dmc_total_energy/w_nactive
            + alpha*(1.0 - float(w_nactive)/float(nwalker)))

        return w_positions, w_status, dmc_energy

    def get_batch_energy(self,coor, batch_size):
        """
        Function to predict energies given the coordinates of the molecule. Depending on the max_batch and nwalkers,
        the energy prediction are done all at once or in multiple iterations.

        Parameters
        ----------
        coor : array of shape (natoms,3)
        batch_size: int


        """
        if batch_size <= self.max_batch:  # predict everything at once

            #Create the batch
            batch = self.create_batch(
                coor.reshape(-1,3)*self.au2ang,batch_size,max_size=False)
            results = self.mcalc(batch)

            e = results['energy'].cpu().detach().numpy()

        else:
            e = np.array([])
            counter = 0
            for i in range(int(batch_size/ self.max_batch) - 1):
                counter += 1
                # print(i*max_batch, (i+1)*max_batch)
                batch = self.create_batch(
                    coor[i*self.max_batch:(i + 1)*self.max_batch, :].reshape(
                        -1,3)
                    * self.au2ang,
                    self.max_batch)
                results = self.mcalc(batch)
                etmp = results['energy'].cpu().detach().numpy()
                e = np.append(e, etmp)

            # calculate missing geom according to batch_size - counter * max_batch
            remaining = batch_size - counter * self.max_batch
            # print(remaining)
            if remaining < 0:  # just to be sure...
                print("someting went wrong with the loop in get_batch_energy")
                quit()

            batch = self.create_batch(
                coor[-remaining:, :].reshape(-1,3)*self.au2ang,
                remaining, max_size=False)
            results = self.mcalc(batch)
            etmp = results['energy'].cpu().detach().numpy()
            e = np.append(e, etmp)

        # print("time:  ", time.time() - start_time)
        return e * 0.0367493


class Logger_DMC:
    """
    Class to write the log files of the DMC simulation.

    Parameters
    ----------
    filename: str, optional, default 'dmc'
        Name tag of the output file where results are going to be stored. 
        The DMC code create 4 files with the same name but different 
        extensions: '.pot', '.log' and '.xyz'. 
        '.log': Log information of the runs.
        '.pot': Potential energies of the DMC runs.
        '.xyz': Two '.xyz' files are generated where the last 10 steps of the
            simulation and the defective geometries are saved respectively.
    maxsteps: int, optional, default None
        Expected maximum number of DMC steps just column formatting
    maxwalker: int, optional, default None
        Expected maximum number of DMC walkers just column formatting
    write_interval: int, optional, default 10
        Interval of writing log and potential output to the respective file

    """

    def __init__(
        self,
        filename: str,
        maxsteps: Optional[int] = None,
        maxwalker: Optional[int] = None,
        write_interval: Optional[int] = 10,
    ):

        # File name tag
        self.filename = filename
        
        # Logger files
        self.logfile = open(self.filename + ".log", 'w')
        self.potfile = open(self.filename + ".pot", 'w')
        self.errorfile = open("defective_" + self.filename + ".xyz", 'w')
        self.lastfile = open("configs_" + self.filename + ".xyz", 'w')

        # Check maximum DMC step and walker input and get number of expected
        # digits for the step counter and alive walkers
        if maxsteps is None:
            self.dimstep = 6
        else:
            self.dimstep = len(str(maxsteps))
        if maxwalker is None:
            self.dimwalker = 4
        else:
            self.dimwalker = len(str(maxwalker))

        # Check write interval parameter
        self.write_interval = int(write_interval)
        if self.write_interval < 1:
            self.write_interval = self.write_interval

        # Initialize log and potential file output message variables
        self.log_message = ""
        self.pot_message = ""

        # Units conversion
        self.au2ang = 0.5291772083
        self.au2cm = 219474.6313710

        return

    def log_begin(
        self,
        nwalker: int,
        nstep: int,
        eqstep: int,
        stepsize: float,
        alpha: float,
        seed: int,
    ):
        """
        Subroutine to write header of log file
        logging all job details and the initial parameters of the DMC simulation

        Parameters
        ----------
        nwalker: int
            Number of walkers for the DMC
        nsteps: int
            Number of steps for each of the DMC walkers
        eqsteps: int
            Number of initial equilibration steps for the DMC walkers
        stepsize: float
            Step size for the DMC in imaginary time
        alpha: float
            Alpha parameter for the DMC feed-back parameter, usually 
            proportional to 1/stepsize
        seed: int
            Random seed for atom positions shuffling

        """

        # Write log file header
        message = (
            "  Diffusion Monte-Carlo Run\n"
            + f"    stored in {self.filename:s}\n\n"
            + f"DMC Simulation started at {str(datetime.now()):s}\n"
            + f"Number of random walkers: {nwalker:d}\n"
            + f"Number of total steps: {nstep:d}\n"
            + f"Number of steps before averaging: {eqstep:d}\n"
            + f"Stepsize: {stepsize:.6e}\n"
            + f"Alpha: {alpha:.6e}\n"
            + f"Random seed: {seed:d}\n\n")
        self.logfile.write(message)

        return

    def log_end(self):
        """
        Function to write footer of logfile

        """
        
        # Write log file footer
        message = (
            f"DMC Simulation terminated at {str(datetime.now()):s}\n"
            + "DMC calculation terminated successfully\n")
        self.logfile.write(message)

        return

    def log_write(
        self,
        message: str
    ):
        """
        Function to write custom message to log file

        """
        self.logfile.write(message)
        return

    def write_error(
        self,
        w_positions: List[float],
        w_energies: List[float],
        symbols: List[str],
    ):
        """
        Subroutine to write '.xyz' file of defective configurations

        Parameters
        ----------
        w_positions: list(float)
            Defective atom positions
        w_energies: list(float)
            Defective potential energies
        symbols: list(str)
            Element symbols of the atom system

        """

        # Iteration over defective configurations
        message = ""
        for ip, (positions, energies) in enumerate(
            zip(w_positions, w_energies)
        ):
            message += (
                f"{positions.shape[0]:d}\n"
                + f"{energies*self.au2cm:8.2f}\n")
            for ia, (pi, si) in enumerate(zip(positions, symbols)):
                message += (
                    f"{si:s}  "
                    + f"{pi[0]:.8f}  "
                    + f"{pi[1]:.8f}  "
                    + f"{pi[2]:.8f}\n")

        # Write defective configurations
        self.errorfile.write(message)

        return

    def write_last(
        self,
        w_alive: int,
        w_positions: List[float],
        natoms: int,
        symbols: List[str],
    ):
        """
        Subroutine to write xyz file of last 10 steps of DMC simulation

        Parameters
        ----------
        w_alive: int
            Number of walkers which are alive
        w_positions: list(float)
            Atom positions of each walker
        natoms: int
            Number of atoms in the system
        symbols: list(str)
            Element symbols of the atom system

        """

        # Iteration over walkers
        message = ""
        for iw, status in enumerate(w_alive):
            message += (
                f"{natoms:d}\n"
                + "\n")
            for ia, si in enumerate(symbols):
                message += (
                    f"{si:s}  "
                    + f"{w_positions[iw, ia, 1]:.8f}  "
                    + f"{w_positions[iw, ia, 2]:.8f}  "
                    + f"{w_positions[iw, ia, 3]:.8f}\n"
                    )

        # Write last configurations
        self.lastfile.write(message)

        return

    def write_pot(
        self,
        w_alive: int,
        dmc_energy: float,
        step: int,
    ):
        """
        Write potential file

        Parameters
        ----------
        w_alive: int
            Number of walker which are alive
        dmc_energy: float
            DMC potential energy
        step: int
            DMC progression step

        """

        # Add step potential information
        self.pot_message += (
            f"{step:{self.dimstep:d}d}  "
            + f"{w_alive:{self.dimwalker:d}d}  "
            + f"{dmc_energy:8.7f} Hartree  "
            + f"{dmc_energy*self.au2cm:8.2f} cm**-1\n")

        # If write interval is reached, write potential information to file
        if step % self.write_interval == 0:
            self.potfile.write(self.pot_message)
            self.pot_message = ""

        return

    def write_log(
        self,
        w_avg_energy):
        """
        Write average to log file

        Parameters
        ----------
        w_avg_energy: float
            Average energy of the trajectory

        """
        
        self.logfile.write(
            "Average energy of trajectory:  "
            + f"{w_avg_energy:8.7f} Hartree  "
            + f"{w_avg_energy*self.au2cm:8.2f} cm**-1\n")

        return

    def close_files(self):
        """
        Close all files

        """

        self.potfile.close()
        self.logfile.close()
        self.errorfile.close()
        self.lastfile.close()
