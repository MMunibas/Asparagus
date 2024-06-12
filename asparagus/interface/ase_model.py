import os
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase

from .. import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = [
    'ase_calculator_units', 'get_ase_calculator', 'get_ase_properties']


#======================================
# ASE Calculator Units
#======================================

ase_calculator_units = {
    'positions':        'Ang',
    'energy':           'eV',
    'forces':           'eV/Ang',
    'hessian':          'eV/Ang/Ang',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'eAng',
    }

#======================================
# ASE Calculator Provision
#======================================

def get_xtb(**kwargs):
    from xtb.ase.calculator import XTB
    return XTB, {}

def get_orca(**kwargs):
    # Check ASE version
    version = ase.__version__
    # For ASE<3.23.0, use modified ORCA calculator
    #if (
        #int(version.split('.')[-3]) < 3
        #or (
            #int(version.split('.')[-3]) == 3
            #and int(version.split('.')[-2]) <= 22
        #)
    #):
    from .orca_ase import ORCA
    return ORCA, {}
    #else:
        #from ase.calculators.orca import ORCA
        #mkwargs = {}
        ## Check for engrad
        #if (
            #kwargs.get('orcasimpleinput') is not None
            #and not 'engrad'.lower() in kwargs.get('orcasimpleinput').lower()
        #):
            #mkwargs['orcasimpleinput'] = (
                #kwargs.get('orcasimpleinput') + ' engrad')
        ## Check for ORCA profile
        #if kwargs.get('profile') is None:
            #orca_command = os.environ.get('ORCA_COMMAND')
            #if orca_command is None:
                #return ORCA, {}
            #else:
                #from ase.calculators.orca import OrcaProfile
                #mkwargs['profile'] = OrcaProfile(command=orca_command)
        #elif utils.is_string(kwargs.get('profile')):
            #from ase.calculators.orca import OrcaProfile
            #mkwargs['profile'] = OrcaProfile(command=kwargs.get('profile'))
        #else:
            #mkwargs['profile'] = kwargs.get('profile')
        #return ORCA, mkwargs

def get_shell(**kwargs):
    from .shell_ase import ShellCalculator
    return ShellCalculator, {}

def get_slurm(**kwargs):
    from .slurm_ase import SlurmCalculator
    return SlurmCalculator, {}


#======================================
# ASE Calculator Assignment
#======================================

ase_calculator_avaiable = {
    'XTB'.lower(): get_xtb,
    'ORCA'.lower(): get_orca,
    'Shell'.lower(): get_shell,
    'Slurm'.lower(): get_slurm,
    }

def get_ase_calculator(
    calculator,
    calculator_args,
    ithread=None,
):
    """
    ASE Calculator interface

    Parameters
    ----------
    calculator: (str, object)
        Calculator label of an ASE calculator to initialize or an
        ASE calculator object directly returned.
    calculator_args: dict
        ASE calculator arguments if ASE calculator will be initialized.
    ithread: int, optional, default None
        Thread number to avoid conflict between files written by the
        calculator.

    Returns
    -------
    callable object
        ASE Calculator object to compute atomic systems
    """

    # Initialize calculator name parameter
    calculator_tag = None

    # In case of calculator label, initialize ASE calculator
    if utils.is_string(calculator):

        # Check avaiability
        if calculator.lower() not in ase_calculator_avaiable:
            raise ValueError(
                f"ASE calculator '{calculator}' is not avaiable!\n"
                + "Choose from:\n" +
                str(ase_calculator_avaiable.keys()))

        # Initialize ASE calculator
        try:
            calculator_tag = calculator
            calc, args = ase_calculator_avaiable[calculator.lower()](
                **calculator_args)
            calculator_args.update(args)
            calculator = calc(**calculator_args)
        except TypeError as error:
            logger.error(error)
            raise TypeError(
                f"ASE calculator '{calculator}' does not accept "
                + "arguments in 'calculator_args'")

    else:

        # Check for calculator name parameter in calculator class
        if hasattr(calculator, 'calculator_tag'):
            calculator_tag = calculator.calculator_tag

    # For application with multi threading (ithread not None), modify directory
    # by adding subdirectory 'thread_{ithread:d}'
    if ithread is not None:
        calculator.directory = os.path.join(
            calculator.directory,
            f'thread_{ithread:d}')

    # Return ASE calculator and name label
    return calculator, calculator_tag


# ======================================
# ASE Calculator Properties
# ======================================

def get_ase_properties(
    system,
    calc_properties,
):
    """
    ASE Calculator interface

    Parameters
    ----------
    system: ASE Atoms object
        ASE Atoms object with assigned ASE calculator
    calc_properties: list(str)
        List of computed properties to return

    Returns
    -------
    dict
        Property dictionary containing:
        atoms number, atomic numbers, positions, cell size, periodic boundary
        conditions, total charge and computed properties.
    """

    # Initialize property dictionary
    properties = {}

    # Atoms number
    properties['atoms_number'] = system.get_global_number_of_atoms()

    # Atomic numbers
    properties['atomic_numbers'] = system.get_atomic_numbers()

    # Atomic numbers
    properties['positions'] = system.get_positions()

    # Periodic boundary conditions
    properties['cell'] = system.get_cell()[:]
    properties['pbc'] = system.get_pbc()

    # Total charge
    # Try from calculator parameters
    if 'charge' in system.calc.parameters:
        charge = system.calc.parameters['charge']
    # If charge is still None, try from computed atomic charges
    elif 'charges' in system.calc.results:
        charge = sum(system.calc.results['charges'])
    else:
        charge = 0
    properties['charge'] = charge

    for ip, prop in enumerate(calc_properties):
        if prop in properties:
            continue
        properties[prop] = system._calc.results.get(prop)

    return properties
