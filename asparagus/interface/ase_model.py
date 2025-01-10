import os
import logging
from typing import Optional, List, Dict, Callable, Tuple, Union, Any

import numpy as np

import ase

from asparagus import utils

__all__ = [
    'ase_calculator_units', 'get_ase_calculator', 'get_ase_properties',
    'is_ase_calculator_threadsafe']

# Initialize logger
name = f"{__name__:s}"
logger = utils.set_logger(logging.getLogger(name))

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

def get_model_calculator(**kwargs):
    from asparagus import Asparagus
    model = Asparagus(**kwargs)
    return model.get_ase_calculator, {}


def get_xtb(**kwargs) -> "ASE.Calculator":
    from xtb.ase.calculator import XTB
    return XTB, {}


def get_orca_threadsafe() -> bool:
    ase_version = ase.__version__
    return (
        int(ase_version.split('.')[-3]) < 3
        or (
            int(ase_version.split('.')[-3]) == 3 
            and int(ase_version.split('.')[-2]) <= 22
        )
    )


def get_orca(**kwargs) -> "ASE.Calculator":
    
    # For ASE<=3.22.1, use modified ORCA calculator
    if get_orca_threadsafe():
            
        from .orca_ase import ORCA
        return ORCA, {}

    # Otherwise, use ORCA calculator (not thread safe anymore)
    else:
        
        from ase.calculators.orca import ORCA
        mkwargs = {}
        
        # Check for engrad
        if (
            kwargs.get('orcasimpleinput') is not None
            and not 'engrad'.lower() in kwargs.get('orcasimpleinput').lower()
        ):
            mkwargs['orcasimpleinput'] = (
                kwargs.get('orcasimpleinput') + ' engrad')
        
        # Check for ORCA profile
        if kwargs.get('profile') is None:
            orca_command = os.environ.get('ORCA_COMMAND')
            if orca_command is None:
                return ORCA, {}
            else:
                from ase.calculators.orca import OrcaProfile
                mkwargs['profile'] = OrcaProfile(command=orca_command)
        elif utils.is_string(kwargs.get('profile')):
            from ase.calculators.orca import OrcaProfile
            mkwargs['profile'] = OrcaProfile(command=kwargs.get('profile'))
        else:
            mkwargs['profile'] = kwargs.get('profile')

        return ORCA, mkwargs


def get_shell(**kwargs) -> "ASE.Calculator":
    from .shell_ase import ShellCalculator
    return ShellCalculator, {}


def get_slurm(**kwargs) -> "ASE.Calculator":
    from .slurm_ase import SlurmCalculator
    return SlurmCalculator, {}


#======================================
# ASE Calculator Assignment
#======================================

# ASE calculator grep functions
ase_calculator_avaiable = {
    'Asparagus'.lower(): get_model_calculator,
    'Model'.lower(): get_model_calculator,
    'XTB'.lower(): get_xtb,
    'ORCA'.lower(): get_orca,
    'Shell'.lower(): get_shell,
    'Slurm'.lower(): get_slurm,
    }

def get_ase_calculator(
    calculator: Union[str, Callable],
    calculator_args: Dict[str, Any],
    ithread: Optional[int] = None,
) -> ("ASE.Calculator", str):
    """
    ASE Calculator interface

    Parameters
    ----------
    calculator: (str, Callable)
        Calculator label of an ASE calculator to initialize or an
        ASE calculator object directly returned.
    calculator_args: dict
        ASE calculator arguments if ASE calculator will be initialized.
    ithread: int, optional, default None
        Thread number to avoid conflict between files written by the
        calculator.

    Returns
    -------
    ASE.Calculator object
        ASE Calculator object to compute atomic systems
    str
        ASE calculator label tag

    """

    # Initialize calculator name parameter
    calculator_tag = None

    # In case of calculator label, initialize ASE calculator
    if utils.is_string(calculator):

        # A column separates the calculator label from a argument of the
        # keyword 'case', that is quick parameter linked to predefined
        # template files in asparagus/template/
        calculator_tag = calculator[:]
        if ":" in calculator:
            calculator_label = calculator.split(":")[0]
            calculator_case = ":".join(calculator.split(":")[1:])
        else:
            calculator_label = calculator
            calculator_case = None

        # Check availability
        if calculator_label.lower() not in ase_calculator_avaiable:
            raise ValueError(
                f"ASE calculator '{calculator_label}' is not avaiable!\n"
                + "Choose from:\n" +
                str(ase_calculator_avaiable.keys()))

        # Initialize ASE calculator
        try:
            
            # Get calculator class
            calc, args = ase_calculator_avaiable[calculator_label.lower()](
                **calculator_args)
            
            # Update predefined arguments and calculator case
            calculator_args.update(args)
            if calculator_case is not None and 'case' not in calculator_args:
                calculator_args.update({'case': calculator_case})
            
            # Initialize ASE calculator
            calculator = calc(**calculator_args)

        except TypeError as error:
            logger.error(error)
            raise TypeError(
                f"ASE calculator '{calculator}' is not accepted!")

    else:

        # Check for calculator name parameter in calculator class
        if hasattr(calculator, 'calculator_tag'):
            calculator_tag = calculator.calculator_tag
        else:
            calculator_tag = None

    # For application with multi threading (ithread not None), modify directory
    # by adding subdirectory 'thread_{ithread:d}'
    if ithread is not None:
        calculator.directory = os.path.join(
            calculator.directory,
            f'thread_{ithread:d}')

    # Return ASE calculator and name label
    return calculator, calculator_tag


# ASE calculator thread safe state
ase_calculator_threadsafe = {
    'XTB'.lower(): False,
    'ORCA'.lower(): get_orca_threadsafe(),
    'Shell'.lower(): True,
    'Slurm'.lower(): True,
    }


def is_ase_calculator_threadsafe(
    calculator_tag: str,
) -> bool:
    """
    ASE Calculator interface

    Parameters
    ----------
    calculator_tag: str
        ASE calculator label tag

    Returns
    -------
    bool
        Status if ASE calculator is thread safe or not

    """

    # Get and check thread safe status
    threadsafe = ase_calculator_threadsafe.get(calculator_tag.lower())
    if threadsafe is None:
        threadsafe = False

    return threadsafe


# ======================================
# ASE Calculator Properties
# ======================================

def get_ase_properties(
    system: ase.Atoms,
    calc_properties: List[str],
) -> Dict[str, Any]:
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
