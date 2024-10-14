# ASE calculator class modifying and executing a template shell file(s) which
# compute atoms properties and provide them as a .json or .npy file.
import os
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable, Union, Any

from asparagus import utils

__all__ = ['shell_cases_available']


#======================================
# Shell Calculator Cases Details
#======================================

template_path = os.path.split(__file__)[0]

def get_case_mp2(
    basis: Optional[str] = 'aug-cc-pVDZ',
    nproc: Optional[Union[int, str]] = 4,
) -> (List[str], Dict[str, str], str):
    """
    Predefined shell calculator ORCA MP2.

    Parameters
    ----------
    basis: str, optional, default 'aug-cc-pVDZ'
        Basis set. Needs to be available as auxiliary basis /C in ORCA
    nproc: (int, str), optional, default 4
        Number of CPUs

    Returns
    -------
    list(str)
        ORCA Template files
    dict(str, str)
        ORCA Template files keyword replacements
    str
        Shell execution file

    """
    
    # Template file path
    files = [
        os.path.join(template_path, 'shell', 'run_orca_mp2.sh'),
        os.path.join(template_path, 'shell', 'run_orca_mp2.inp'),
        os.path.join(template_path, 'shell', 'run_orca_mp2.py')
        ]
    
    # Default template keywords
    files_replace = {
        '%orca%': os.environ.get('ORCA_COMMAND'),
        '%xyz%': '$xyz',
        '%charge%': '$charge',
        '%multiplicity%': '$multiplicity',
        '%basis%': basis,
        '%nproc%': str(nproc),
        }
    
    # Execution file path
    execute_file = os.path.join(template_path, 'shell', 'run_orca_mp2.sh')

    return files, files_replace, execute_file


#======================================
# Shell Calculator Cases Assignment
#======================================

shell_cases_available = {
    'mp2'.lower(): get_case_mp2,
    }
