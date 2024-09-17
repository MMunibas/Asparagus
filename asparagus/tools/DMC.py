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
    max_batch: int, optional, default 6000
        Size of the batch
    seed: (int, float), optional, default: np.random.randint(1E6)
        Specify random seed
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
        max_batch: Optional[int] = 6000,
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
        # Random seed
        self.seed = seed

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
        self.logger = Logger_DMC(self.filename)

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

        #self.mass = np.sqrt(self.atoms.get_masses()*self.emass)
        #self.nucl_charge = self.atoms.get_atomic_numbers()
        #self.au2ang = 0.5291772083
        #self.coord_min = self.atoms.get_positions()/self.au2ang

        return

    def run(
        self,
        optimize: Optional[bool] = None,
        optimize_method: Optional[Callable] = BFGS,
        optimize_fmax: Optional[float] = 0.01,
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

        """

        # Write log file header
        self.logger.log_begin(
            self.nwalker,
            self.nsteps,
            self.eqsteps,
            self.stepsize,
            self.alpha)

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
        results_minimum = self.model_calculator(
            batch_minimum,
            no_derivation=True)
        energy_initial = (
            results_initial['energy'].cpu().detach().numpy()
            * self.dmc_conversion['energy'])
        energy_minimum = (
            results_minimum['energy'].cpu().detach().numpy()
            * self.dmc_conversion['energy'])

        #Initialize the DMC
        psips, psips_f, v_ave, v_ref = self.init_walker(
            energy_initial, energy_minimum)
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
            self.logger.write_pot(psips_f[0], v_ref, step=i + 1,initial=False)

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
    ):
        """
        Initialize DMC simulation variables such as the atoms positions for
        each walker (psi_positions) and the status of the walker (psi_status).

        Parameters
        ----------
        energy_initial: float
            initial energy
        energy_minimum:
            minimum energy

        Returns
        -------

        """
        # Not sure this function is going to work,
        # There are some variables that are not defined in the original code. LIVS 14/12/2023

        #define stepsize
        self.deltax = np.sqrt(self.stepsize/self.atomic_masses)

        # define psips and psips_f
        dim = self.natoms * 3
        psips_f = np.zeros([3 * self.nwalker + 1], dtype=int)
        psips = np.zeros([3 * self.nwalker, dim, 2], dtype=float)

        # psips_f keeps track of how many walkers are alive (psips_f[0]) and which ones (psips_f[1:], 1 for alive and 0 for dead)
        psips_f[:] = 1
        psips_f[0] = self.nwalker
        psips_f[self.nwalker + 1:] = 0

        # psips keeps track of atomic positions of all walkers
        # is initialized to some molecular geometry defined in the input xyz file
        psips[:, :, 0] = self.initial_coord.reshape(-1)

        # reference energy (which is updated throughout the DMC simulation) is initialized to energy of v0, referenced to energy
        # of minimum geometry
        v_ref = energy_initial
        v_ave = 0
        v_ref = v_ref - energy_minimum
        self.logger.write_pot(psips_f[0], v_ref, initial=True)

        return psips, psips_f, v_ave, v_ref

    def walk(self,psips):
        """
        Walk routine performs the diffusion process of the replicas by adding to the
        coordinates of the alive replicas sqrt(deltatau)rho, rho is random number
        from Gaussian distr

        Parameters
        ----------
        psips: array
            coordinates of the walkers

        """
        # print(psips.shape)
        dim = len(psips[0, :, 0])
        for i in range(dim):
            x = np.random.normal(size=(len(psips[:, 0, 0])))
            psips[:, i, 1] = psips[:, i, 0] + x * self.deltax[int(np.ceil((i + 1) / 3.0)) - 1]
            # print(psips[:,i-1,1])

        return psips

    def branch(self,refx, mass, symb, vmin, psips, psips_f, v_ref,v_tot):
        """

        The birth-death (branching) process, which follows the diffusion step

        Parameters
        ----------
        refx: array
            reference positions of atoms
        mass: array
            atomic masses
        symb: array
            atomic symbols
        vmin: float
            minimum energy
        psips: array
            coordinates of the walkers
        psips_f: array
            flag to know which walkers are alive
        v_ref: float
            reference energy
        v_tot: float
            total energy

        """

        nalive = psips_f[0]

        psips[:, :, 1], psips_f, v_tot, nalive = self.gbranch(
            refx, mass, symb, vmin, psips[:, :, 1],
            psips_f, v_ref, v_tot, nalive)

        # after doing the statistics in gbranch remove all dead replicas.
        count_alive = 0
        psips[:, :, 0] = 0.0  # just to be sure we dont use "old" walkers
        for i in range(nalive):
            """update psips and psips_f using the number of alive walkers (nalive). 
            """
            if psips_f[i + 1] == 1:
                count_alive += 1
                psips[count_alive - 1, :, 0] = psips[i, :, 1]
                psips_f[count_alive] = 1
        psips_f[0] = count_alive
        psips[:, :, 1] = 0.0  # just to be sure we dont use "old" walkers
        psips_f[count_alive + 1:] = 0  # set everything beyond index count_alive to zero

        # update v_ref
        v_ref = v_tot / psips_f[0] + self.alpha * (1.0 - 3.0 * psips_f[0] / (len(psips_f) - 1))

        return psips, psips_f, v_ref

    def gbranch(self,refx, mass, symb, vmin, psips, psips_f, v_ref, v_tot, nalive):
        """
        The birth-death criteria for the ground state energy. Note that psips is of shape
        (3*nwalker, 3*natm) as only the progressed coordinates (i.e. psips[:,i,1]) are
        given to gbranch

        Parameters
        ----------

        refx: array
            reference positions of atoms
        mass: array
            atomic masses
        symb: array
            atomic symbols
        vmin: float
            minimum energy
        psips: array
            coordinates of the walkers
        psips_f: array
            flag to know which walkers are alive
        v_ref: float
            reference energy
        v_tot: float
            total energy
        nalive: int
            number of alive walkers

        """

        birth_flag = 0
        error_checker = 0
        # print(psips.shape) #-> (3*nwalker, 3*natm)


        #RE-DEFINE THIS
        v_psip = self.get_batch_energy(psips[:nalive, :], nalive)  # predict energy of all alive walkers.

        # reference energy with respect to minimum energy.
        v_psip = v_psip - vmin

        # check for holes, i.e. check for energies that are lower than the one for the (global) min
        if np.any(v_psip < -1e-5):
            error_checker = 1
            idx_err = np.where(v_psip < -1e-5)
            self.logger.write_error(refx, mass, symb, psips[idx_err, :], v_psip, idx_err)
            print("Defective geometry is written to file")
            # kill defective walker idx_err + one as index 0 is counter of alive walkers
            psips_f[idx_err[0] + 1] = 0  # idx_err[0] as it is some stupid array...

        prob = np.exp((v_ref - v_psip) * self.stepsize)
        sigma = np.random.uniform(size=nalive)

        if np.any((1.0 - prob) > sigma):
            """test whether one of the walkers has to die given the probabilites
               and then set corresponding energies v_psip to zero as they
               are summed up later.
               geometries with high energies are more likely to die.
            """
            idx_die = np.array(np.where((1.0 - prob) > sigma)) + 1
            psips_f[idx_die] = 0
            v_psip[idx_die - 1] = 0.0

        v_tot = v_tot + np.sum(v_psip)  # sum energies of walkers that are alive (i.e. fullfill conditions)

        if np.any(prob > 1):
            """give birth to new walkers given the probabilities and update psips, psips_f
               and v_tot accordingly.
            """
            idx_prob = np.array(np.where(prob > 1)).reshape(-1)

            for i in idx_prob:
                if error_checker == 0:

                    probtmp = prob[i] - 1.0
                    n_birth = int(probtmp)
                    sigma = np.random.uniform()

                    if (probtmp - n_birth) > sigma:
                        n_birth += 1
                    if n_birth > 2:
                        birth_flag += 1

                    while n_birth > 0:
                        nalive += 1
                        n_birth -= 1
                        psips[nalive - 1, :] = psips[i, :]
                        psips_f[nalive] = 1
                        v_tot = v_tot + v_psip[i]

                else:
                    if np.any(i == idx_err[0]):  # to make sure none of the defective geom are duplicated
                        pass
                    else:

                        probtmp = prob[i] - 1.0
                        n_birth = int(probtmp)
                        sigma = np.random.uniform()

                        if (probtmp - n_birth) > sigma:
                            n_birth += 1
                        if n_birth > 2:
                            birth_flag += 1

                        while n_birth > 0:
                            nalive += 1
                            n_birth -= 1
                            psips[nalive - 1, :] = psips[i, :]
                            psips_f[nalive] = 1
                            v_tot = v_tot + v_psip[i]

        #error_checker = 0
        return psips, psips_f, v_tot, nalive

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
        '.pot': Potential energies of the DMC runs.
        '.log': Log information of the runs.
        '.xyz': Two '.xyz' files are generated where the last 10 steps of the
            simulation and the defective geometries are saved respectively.

    """

    def __init__(
        self,
        filename: str,
    ):

        # File name tag
        self.filename = filename
        
        # Logger files
        self.potfile = open(self.filename + ".pot", 'w')
        self.logfile = open(self.filename + ".log", 'w')
        self.errorfile = open("defective_" + self.filename + ".xyz", 'w')
        self.lastfile = open("configs_" + self.filename + ".xyz", 'w')

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

        """

        # Write log file header
        message = (
            "     DMC for " + self.filename + "\n\n"
            + f"DMC Simulation started at {str(datetime.now()):s}\n"
            + f"Number of random walkers: {nwalker:d}\n"
            + f"Number of total steps: {nstep:d}\n"
            + f"Number of steps before averaging: {eqstep:d}\n"
            + f"Stepsize: {stepsize:.6e}\n"
            + f"Alpha: {alpha:.6e}\n\n")
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
        refx: np.ndarray,
        mass: np.ndarray,
        symb: np.ndarray,
        errq: np.ndarray,
        v: np.ndarray,
        idx: np.ndarray,
    ):
        """
        Subroutine to write '.xyz' file of defective configurations

        Parameters
        ----------
        refx: np.ndarray
            Reference positions of atoms
        mass: np.ndarray
            Atomic masses
        symb: np.ndarray
            Atomic symbols
        errq: np.ndarray
            Error in positions
        v: np.ndarray
            Potential energy
        idx: np.ndarray
            Index of defective configurations

        """

        # Data preparation
        natoms = len(refx)
        errx = errq[0]*self.au2ang
        errx = errx.reshape(len(idx[0]), natoms, 3)
        
        # Iteration over defective configurations
        message = ""
        for ix, xi in enumerate(errx):
            message += (
                f"{natoms:d}\n"
                + f"{v[idx[0][ix]]*self.au2cm:.6e}\n")
            for ia in range(natoms):
                message += (
                    f"{symb[ia]:s}  "
                    + f"{errx[ix, ia, 0]:.8f}  "
                    + f"{errx[ix, ia, 1]:.8f}  "
                    + f"{errx[ix, ia, 2]:.8f}\n")
        
        # Write defective configurations
        self.errorfile.write(message)

        return

    def write_last(
        self,
        psips_f: np.ndarray,
        psips: np.ndarray,
        natoms: int,
        symb: np.ndarray,
    ):
        """
        Subroutine to write xyz file of last 10 steps of DMC simulation

        Parameters
        ----------
        psips_f: np.ndarray
            Flag to know which walkers are alive
        psips: np.ndarray
            Coordinates of the walkers
        natm: int
            Number of atoms
        symb: np.ndarray
            Atomic symbols

        """

        # Iteration over walkers
        message = ""
        for iw, status in enumerate(psips_f[0]):
            message += (
                f"{natoms:d}\n"
                + "\n")
            for ia, si in enumerate(symb):
                message += (
                    f"{si:s}  "
                    + f"{psips[iw, 3*(ia - 1) - 3, 0]:.8f}  "
                    + f"{psips[iw, 3*(ia - 1) - 2, 0]:.8f}  "
                    + f"{psips[iw, 3*(ia - 1) - 1, 0]:.8f}\n"
                    )

        # Write last configurations
        self.lastfile.write(message)

        return

    def write_pot(
        self,
        psips_f,
        v_ref: float,
        step=None,
        initial=False,
    ):
        """
        Write potential file

        Parameters
        ----------
        psips_f: np.ndarray
            Flag to know which walkers are alive
        v_ref: float
            Potential energy

        Returns
        -------

        """
        if initial:
           self.potfile.write("0  " + str(psips_f) + "  " + str(v_ref) + "  " + str(v_ref * self.au2cm) + "\n")
        else:
           self.potfile.write(str(step) + "  " + str(psips_f) + "  " + str(v_ref) + "  " + str(v_ref * self.au2cm) + "\n")

        return

    def write_log(self,v_ave):
        """
        Write average to log file

        Parameters
        ----------
        v_ave: float
            Average energy of the trajectory

        """
        
        self.logfile.write(
            "Average energy of trajectory:  "
            + f"{v_ave:.6e} Hartree,  "
            + f"{v_ave*self.au2cm:.6e} cm**-1\n")

        return

    def close_files(self):
        """
        Close all files

        """

        self.potfile.close()
        self.logfile.close()
        self.errorfile.close()
        self.lastfile.close()
