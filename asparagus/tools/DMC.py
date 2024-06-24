# Adaptation of diffusion Monte Carlo code from https://github.com/MMunibas/dmc_gpu_PhysNet/tree/main
import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any
import warnings

#Time importations
import time
from datetime import datetime

#Basic importations
import numpy as np
import torch

#ASE importations
import ase
from ase.io import read, write

#Internal importations
from .. import model
from .. import settings
from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DMC','Logger_DMC']

class DMC:
    """

    This code adapt the diffusion Monte Carlo code from https://github.com/MMunibas/dmc_gpu_PhysNet/tree/main
    to the asparagus framework.

    **NOTE**: As a difference with the original implementation, this code reads the initial coordinates and the
    equilibrium coordinates from the atoms object in ASE.

    **Original Message**
    DMC code for the calculation of zero-point energies using PhysNet based PESs on GPUs/CPUs. The calculation
    is performed in Cartesian coordinates.
    See e.g. American Journal of Physics 64, 633 (1996); https://doi.org/10.1119/1.18168 for DMC implementation
    and https://scholarblogs.emory.edu/bowman/diffusion-monte-carlo/ for Fortran 90 implementation.

    Parameters
    ----------
    natoms: int
        Number of atoms of the system
    nwalker: int
        Number of walkers for the DMC
    stepsize: float
        Step size for the DMC in imaginary time
    nsteps: int
        Number of steps for the DMC
    eqsteps: int
        Number of equilibration steps for the DMC
    alpha: float
        Alpha parameter for the DMC: Feed-back parameter, usually proportional to 1/stepsize
    max_batch: int
        Size of the batch
    initial_coord: str or array
        Initial coordinates for the DMC
    atoms: str or ase.Atoms
        Atoms object
    total_charge: int
        Total charge of the system
    config: dct
        Parameters for the model
    config_file: str
        Configuration file for asparagus
    model_checkpoint: int
        Checkpoint for the model
    implemented_properties: list
        List of implemented properties
    filename: str
        Name of the file where the results are going to be saved. **NOTE**: The code create 4 files with the same name
        but different extensions: .pot, .log, .xyz and .xyz. The first two are the files where the potential energy
        and the log of the simulation are saved. The last two are the files where the last 10 steps of the simulation
        and the defective geometries are saved respectively.



    """
    def __init__(self,natoms,
                 atoms=None,
                 model_calculator=None,
                 total_charge=None,
                 nwalker=100,
                 stepsize=0.1,
                 nsteps=1000,
                 eqsteps=100,
                 alpha=1,
                 max_batch=6000,
                 initial_coord=None,
                 filename=None,**kwargs):

        # Number of atoms
        self.natoms = natoms
        # Number of walkers for the DMC
        self.nwalker = nwalker
        # Step size for the DMC in imaginary time
        self.stepsize = stepsize
        # Number of steps for the DMC
        self.nsteps = nsteps
        # Number of equilibration steps for the DMC
        self.eqsteps = eqsteps
        # Alpha parameter for the DMC: Feed-back parameter, usually propotional to 1/stepsize"
        self.alpha = alpha
        # Initial coordinates for the DMC
        self.initial_coord = initial_coord
        # Charge of the atoms
        if total_charge is None:
            self.total_charge = 0
        else:
            self.total_charge = total_charge

        #Unit conversion (Maybe move to the settings file?)
        self.emass = 1822.88848

        # Atoms object
        self.atoms = atoms
        if self.atoms is type(str):
            self.atoms = read(self.atoms)

        if self.atoms is None:
            raise ValueError('A geometry is required')

        # Get the ASE calculator
        self.ase_calculator = model_calculator.get_ase_calculator()

        # Check the implemented properties
        self.implemented_properties = self.ase_calculator.implemented_properties

        # Check implemented properties
        if 'energy' not in self.implemented_properties:
            raise ValueError('The energy is not implemented in the model calculator')

        # Seed is defined according to the time
        self.seed = np.random.seed(int(time.time()))

        # Informtation about the system

        self.natm = len(self.atoms)

        self.mass = np.sqrt(self.atoms.get_masses()*self.emass)
        self.nucl_charge = self.atoms.get_atomic_numbers()

        self.coord_min = self.atoms.get_positions()

        #Size of the batch
        self.max_batch = max_batch

        #Here we define the initial coordinates
        if self.initial_coord is None:
            atoms2 = self.atoms.copy()
            #Here we add a random displacement to the initial coordinates, I hardcoded the seed from the original code
            atoms2.rattle(stdev=0.1,seed=1453)
            self.initial_coord = atoms2.get_positions()
        elif type(self.initial_coord) is str:
            atoms2 = read(self.initial_coord)
            self.initial_coord = atoms2.get_positions()
        else:
            self.initial_coord = self.initial_coord

        if filename is None:
            self.filename = "dmc"
        else:
            self.filename = filename
        self.logger = Logger_DMC(self.filename)

    def init_dmc(self, v0, vmin):

        """
        Initialize the DMC simulation. It creates the psips and psips_f arrays, which are the coordinates of the walkers

        Parameters
        ----------
        v0: float
            initial energy
        vmin:
            minimum energy

        Returns
        -------

        """
        #Not sure this function is going to work,
        # There are some variables that are not defined in the original code. LIVS 14/12/2023

        #define stepsize
        self.deltax = np.sqrt(self.stepsize)/self.mass

        # define psips and psips_f
        dim = self.natm * 3
        psips_f = np.zeros([3 * self.nwalker + 1], dtype=int)
        psips = np.zeros([3 * self.nwalker, dim, 2], dtype=float)

        # psips_f keeps track of how many walkers are alive (psips_f[0]) and which ones (psips_f[1:], 1 for alive and 0 for dead)
        psips_f[:] = 1
        psips_f[0] = self.nwalker
        psips_f[self.nwalker + 1:] = 0

        # psips keeps track of atomic positions of all walkers
        # is initialized to some molecular geometry defined in the input xyz file
        psips[:, :, 0] = self.initial_coord[:]

        # reference energy (which is updated throughout the DMC simulation) is initialized to energy of v0, referenced to energy
        # of minimum geometry
        v_ref = v0
        v_ave = 0
        v_ref = v_ref - vmin
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
            psips[:, i, 1] = psips[:, i, 0] + x * self.deltax[np.ceil((i + 1) / 3.0) - 1]
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

        psips[:, :, 1], psips_f, v_tot, nalive = self.gbranch(refx, mass, symb, vmin, psips[:, :, 1], psips_f, v_ref, v_tot,
                                                         nalive)

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

    def create_batch(self,coor, batch_size,max_size=False):
        """
        It creates the batch to be passed to the model.

        Parameters
        ----------
        coor : array of shape (natoms,3)
        batch_size: int
        max_size: bool
            If True, it creates a batch with the specified batch size. If False, it creates a batch with the size
            of the max_batch.

        Returns
        -------

        """
        batch = {}
        batch['atoms_number'] = self.natm
        batch['coordinates'] = torch.Tensor(coor)
        batch['atomic_numbers'] = torch.Tensor(self.nucl_charge)
        if not max_size:
            batch['idx_i'] = torch.Tensor(idx_ilarge[:(self.natm * (self.natm - 1)) * batch_size])
            batch['idx_j'] = torch.Tensor(idx_jlarge[:(self.natm * (self.natm - 1)) * batch_size])
            batch['batch_seg'] = torch.Tensor(batch_seglarge[:self.natm * batch_size])
        else:
            batch['idx_i'] = torch.Tensor(idx_ilarge[:(self.natm * (self.natm - 1)) * self.max_batch])
            batch['idx_j'] = torch.Tensor(idx_jlarge[:(self.natm * (self.natm - 1)) * self.max_batch])
            batch['batch_seg'] = torch.Tensor(batch_seglarge[:self.natm * self.max_batch])

        return batch


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
            batch = self.create_batch(coor,batch_size,max_size=True)
            results = self.ase_calculator(batch)
            e = results['energy'].detach().numpy()

        else:
            e = np.array([])
            counter = 0
            for i in range(int(batch_size/ self.max_batch) - 1):
                counter += 1
                # print(i*max_batch, (i+1)*max_batch)
                batch = self.create_batch(coor[i * self.max_batch:(i + 1) * self.max_batch, :],self.max_batch)
                results = self.ase_calculator(batch)
                etmp = results['energy'].detach().numpy()
                e = np.append(e, etmp)

            # calculate missing geom according to batch_size - counter * max_batch
            remaining = batch_size - counter * self.max_batch
            # print(remaining)
            if remaining < 0:  # just to be sure...
                print("someting went wrong with the loop in get_batch_energy")
                quit()

            batch = self.create_batch(coor[-remaining:, :],remaining,max_size=False)
            results = self.ase_calculator(batch)
            etmp = results['energy'].detach().numpy()
            e = np.append(e, etmp)

        # print("time:  ", time.time() - start_time)
        return e * 0.0367493

    def look_up_table(self,):
        '''
        Create "look up table" for the needed inputs. This is needed because the
        batch size is not always constant (walkers die).
        '''

        N = len(self.nucl_charge)
        i_ = []
        j_ = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    i_.append(i)
                    j_.append(j)
        global idx_ilarge
        idx_ilarge = []
        global idx_jlarge
        idx_jlarge = []
        global batch_seglarge
        batch_seglarge = []
        for batch in range(self.max_batch*2):  # largest batchsize
            idx_ilarge += [i + N * batch for i in i_]
            idx_jlarge += [j + N * batch for j in j_]
            batch_seglarge += [batch] * N
        return

    def run(self):
        '''
        Run the difussion montecarlo simulation.

        Returns
        -------

        '''

        #Some basic variables
        symb = self.nucl_charge
        self.look_up_table()
        # Basic evaluation of the potential energy
        batch_v0 = self.create_batch(self.initial_coord,1)
        batch_vmin = self.create_batch(self.coord_min,1)
        calc_v0 = self.ase_calculator(batch_v0)
        calc_vmin = self.ase_calculator(batch_vmin)

        v0 = calc_v0['energy'].detach().numpy()
        vmin = calc_vmin['energy'].detach().numpy()

        self.logger.log_begin(self.nwalker, self.nsteps, self.eqsteps, self.stepsize, self.alpha)

        #Initialize the DMC
        psips, psips_f, v_ave, v_ref = self.init_dmc(v0, vmin)

        v_tot = 0.0
        # Main loop of the DMC
        for i in range(self.nsteps):
            start_time = time.time()
            psips[:psips_f[0], :, :] = self.walk(psips[:psips_f[0], :, :])
            psips, psips_f, v_ref = self.branch(self.initial_coord, self.mass, symb, vmin, psips, psips_f, v_ref,v_tot)
            self.logger.write_pot(psips_f[0], v_ref, step=i + 1,initial=False)

            if i > self.eqsteps:
                v_ave += v_ref

            if i > self.nsteps - 10:  # record the last 10 steps of the DMC simulation for visual inspection.
                self.logger.write_last(psips_f, psips, self.natm, symb)
            if i % 10 == 0:
                print("step:  ", i, "time/step:  ", time.time() - start_time, "nalive:   ", psips_f[0])
        v_ave = v_ave / (self.nsteps - self.eqsteps)

        self.logger.write_log(v_ave)

        # terminate code and close log/pot files
        self.logger.log_end()
        self.logger.close_files()


class Logger_DMC:

    """
    Class to write the log files of the DMC simulation.

    Parameters
    ----------

    filename: str
        The name of the file where the results are going to be saved. **NOTE**: The code create 4 files with the same name
        but different extensions: .pot, .log, .xyz and .xyz. The first two are the files where the potential energy
        and the log of the simulation are saved. The last two are the files where the last 10 steps of the simulation
        and the defective geometries are saved respectively.

    """

    def __init__(self,filename):

        self.filename = filename
        self.potfile = open(self.filename + ".pot",'w')
        self.logfile = open(self.filename + ".log",'w')
        self.errorfile = open("defective" + self.filename + ".xyz",'w')
        self.lastfile = open("configs" + self.filename + ".xyz",'w')


        # Units conversion
        self.auang = 0.5291772083
        self.aucm = 219474.6313710

    def log_begin(self,nwalker,nstep,eqstep,stepsize,alpha):
        """
        Subroutine to write header of log file
        logging all job details and the initial parameters of the DMC simulation

        Parameters
        ----------
        nwalker: int
            Number of walkers for the DMC
        nstep: int
            Number of steps for the DMC
        eqstep: int
            Number of equilibration steps for the DMC
        stepsize: float
            Step size for the DMC in imaginary time
        alpha: float
            Alpha parameter for the DMC: Feed-back parameter, usually propotional to 1/stepsize

        """

        self.logfile.write("                  DMC for " + self.filename + "\n\n")
        self.logfile.write("DMC Simulation started at " + str(datetime.now()) + "\n")
        self.logfile.write("Number of random walkers: " + str(nwalker) + "\n")

        self.logfile.write("Number of total steps: " + str(nstep) + "\n")
        self.logfile.write("Number of steps before averaging: " + str(eqstep) + "\n")
        self.logfile.write("Stepsize: " + str(stepsize) + "\n")
        self.logfile.write("Alpha: " + str(alpha) + "\n\n")

    def log_end(self):
        """
        Function to write footer of logfile
        """
        self.logfile.write("DMC Simulation terminated at " + str(datetime.now()) + "\n")
        self.logfile.write("DMC calculation terminated successfully\n")

    def write_error(self,refx,mass,symb,errq,v,idx):
        """
        Subroutine to write xyz file of defective configurations

        Parameters
        ----------

        refx: array
            reference positions of atoms
        mass: array
            atomic masses
        symb: array
            atomic symbols
        errq: array
            error in positions
        v: array
            potential energy
        idx: array
            index of defective configurations

        """

        if len(idx[0]) == 1:
            natm = int(len(refx) / 3)
            errx = errq[0] * self.auang
            errx = errx.reshape(natm, 3)
            self.errorfile.write(str(int(natm)) + "\n")
            self.errorfile.write(str(v[idx[0]] * self.aucm) + "\n")
            for i in range(int(natm)):
                self.errorfile.write(
                    str(symb[i]) + "  " + str(errx[i, 0]) + "  " + str(errx[i, 1]) + "  " + str(errx[i, 2]) + "\n")

        else:
            natm = int(len(refx) / 3)
            errx = errq[0] * self.auang
            errx = errx.reshape(len(idx[0]), natm, 3)

            for j in range(len(errx)):
                self.errorfile.write(str(int(natm)) + "\n")
                self.errorfile.write(str(v[idx[0][j]] * self.aucm) + "\n")
                for i in range(int(natm)):
                    self.errorfile.write(str(symb[i]) + "  " + str(errx[j, i, 0]) + "  " + str(errx[j, i, 1]) + "  " + str(
                        errx[j, i, 2]) + "\n")

    def write_last(self,psips_f,psips,natm,symb):
        """
        Subroutine to write xyz file of last 10 steps of DMC simulation

        Parameters
        ----------
        psips_f: array
            flag to know which walkers are alive
        psips: array
            coordinates of the walkers
        natm: int
            number of atoms
        symb:
            atomic symbols

        Returns
        -------

        """

        for j in range(psips_f[0]):
            self.lastfile.write(str(natm) + "\n\n")
            for l in range(int(natm)):
                l = l + 1
                self.lastfile.write(str(symb[l - 1]) + "  " + str(psips[j, 3 * l - 3, 0] * self.auang) + "  " + str(
                    psips[j, 3 * l - 2, 0] * self.auang) + "  " + str(psips[j, 3 * l - 1, 0] * self.auang) + "\n")

    def write_pot(self,psips_f,v_ref,step=None,initial=False):
        """
        subroutine to write potential file

        Parameters
        ----------
        psips_f
        v_ref

        Returns
        -------

        """
        if initial:
           self.potfile.write("0  " + str(psips_f) + "  " + str(v_ref) + "  " + str(v_ref * self.aucm) + "\n")
        else:
           self.potfile.write(str(step) + "  " + str(psips_f) + "  " + str(v_ref) + "  " + str(v_ref * self.aucm) + "\n")


    def write_log(self,v_ave):
        """
        subroutine to write average log file

        Parameters
        ----------
        v_ave: float
        Average energy of the trajectory

        Returns
        -------

        """
        self.logfile.write("AVERAGE ENERGY OF TRAJ   " + "   " + str(v_ave) + " hartree   " + str(v_ave * self.aucm) + " cm**-1\n")

    def close_files(self):
        """
        Subroutine to close all files


        -------

        """
        self.potfile.close()
        self.logfile.close()
        self.errorfile.close()
        self.lastfile.close()

