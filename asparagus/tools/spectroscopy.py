# Importing modules
from typing import Optional
from ase.io import read
from ase.optimize import BFGS
from ase.vibrations import Vibrations

#TODO: Add more options


def get_harmonic_freqs(model_calculator: object,
                       initial_geometry: str,
                       tolerance_opt: Optional[float] = 0.001,
                       tolerance_freq: Optional[float] = 0.001):
    '''
    This function calculates harmonic frequencies with ASE.

    Parameters
    ----------
    model_calculator
    initial_geometry: An .xyz or any way to initialize the geometry of the molecule

    Returns
    -------

    '''

    # Read the initial geometry

    if initial_geometry.endswith('.xyz'):
        initial_geometry = read(initial_geometry)
    else:
        raise ValueError('The initial geometry must be an .xyz file')

    # Set the calculator
    initial_geometry.calc = model_calculator

    # It does an initial geometry optimization
    opt = BFGS(initial_geometry)
    opt.run(fmax=tolerance_opt)

    # Calculate the harmonic frequencies
    harmonic_freqs = Vibrations(initial_geometry, delta=tolerance_freq)
    harmonic_freqs.clean()
    harmonic_freqs.run()
    harmonic_freqs.summary()
    system_frequencies = harmonic_freqs.get_frequencies()

    return system_frequencies
