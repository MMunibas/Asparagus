import numpy as np
from typing import Optional, List, Dict, Callable, Tuple, Union, Any

import ase
import ase.calculators.calculator as ase_calc

import openmm
from openmm import app
from openmm import unit

import torch

from asparagus import utils

__all__ = ['ASE_OpenMM_Calculator']


class ASE_OpenMM_Calculator(ase_calc.Calculator):
    """
    ASE calculator interface for an OpenMM system.

    Parameters
    ----------
    openmm_system: openmm.openmm.System
        OpenMM system instance equivalent to the ASE atoms object.
    openmm_topology: openmm.app.topology.Topology
        OpenMM topology instance of the OpenMM system instance.
    atoms: ASE Atoms object, optional, default None
        ASE Atoms object to which the calculator will be attached.
    label: str, optional, default 'openmm'
        Label for the ASE calculator

    """

    # ASE specific calculator information
    default_parameters = {
        "method": "OpenMM",
    }
    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        openmm_system: openmm.openmm.System,
        openmm_topology: app.topology.Topology,
        atoms: Optional[ase.Atoms] = None,
        platform: Optional[openmm.openmm.Platform] = None,
        properties: Optional[Dict[str, str]] = None,
        label: Optional[str] = 'openmm',
        **kwargs
    ):
        """
        Initialize ASE Calculator of an OpenMM system.

        """

        # Initialize parent Calculator class
        ase_calc.Calculator.__init__(self, atoms=atoms, **kwargs)

        #####################################
        # # # Prepare OpenMM Calculator # # #
        #####################################

        # Instantiate a dummy integrator
        openmm_integrator = openmm.LangevinMiddleIntegrator(
            0*unit.kelvin,
            0/unit.picosecond,
            0.001*unit.picosecond,
        )

        # Check platform variable
        if platform is None:
            if torch.cuda.is_available():
                platform = openmm.Platform.getPlatform('CUDA')
                self.device = 'cuda'
            else:
                platform = openmm.Platform.getPlatform('CPU')
                self.device = 'cpu'
        elif isinstance(platform, openmm.openmm.Platform):
            if platform.getName() == 'CUDA':
                self.device = 'cuda'
            elif platform.getName() == 'CPU':
                self.device = 'cpu'
            else:
                self.device = platform.getName().lower()
        else:
            raise ValueError(
                f"OpenMM Platform is not of correct type ({type(platform)})!"
            )

        # Check properties variable
        if properties is None:
            if self.device == 'cuda':
                properties = {'Precision': 'single'}
            else:
                properties = {}
        elif not utils.is_dictionary(properties):
            raise ValueError(
                "OpenMM Platform properties is not a dicationary!"
            )

        # Instantiate the Simulation instance
        self.openmm_simulation = app.Simulation(
            openmm_topology,
            openmm_system,
            openmm_integrator,
            platform,
            properties,
        )

        # Set the initial atoms position and, eventually, cell parameter with
        # respect to the ASE Atoms object, if given.
        if atoms is not None:
            self.openmm_simulation.context.setPositions(
                atoms.get_positions()*unit.angstrom
            )
            if np.any(atoms.get_pbc()):
                self.openmm_simulation.context.setPeriodicBoxVectors(
                    *[
                        openmm.Vec3(*edge*unit.angstrom)
                        for edge in atoms.get_cell()
                    ]
                )

        # Conversion factors from kJ/mol to eV for energy values and 
        # from kJ/mol/nm to eV/Ang for forces
        self.openmm_conversion_energy = ase.units.kJ/ase.units.mol
        self.openmm_conversion_forces = ase.units.kJ/ase.units.mol/ase.units.nm

        ##################################
        # # # Set Calculator Options # # #
        ##################################

        # Initialize result dictionary
        state = self.openmm_simulation.context.getState(
            getEnergy=True,
            getForces=True,
        )
        self.results = {
            'energy': (
                state.getPotentialEnergy()._value*self.openmm_conversion_energy
            ),
            'forces': np.array(
                [forces_i[:] for forces_i in state.getForces()._value]
            )*self.openmm_conversion_forces,
        }

        # Initialize convergence flag
        self.converged = True

        # Set flag to use stored results if atoms property did not change
        self.use_cache = True

    def calculate(
        self,
        atoms: Optional[Union[ase.Atoms, List[ase.Atoms]]] = None,
        properties: Optional[List[str]] = None,
        system_changes: Optional[List[str]] = ase_calc.all_changes,
    ) -> Dict[str, Any]:
        """
        Calculate model properties

        Parameter
        ---------
        atoms: (ase.Atoms, list(ase.Atoms)), optional, default None
            Optional ASE Atoms object or list of ASE Atoms objects to which the
            properties will be calculated. If given, atoms setup to prepare
            model calculator input will be run again.
        properties: list(str), optional, default None
            List of properties to be calculated. If None, all implemented
            properties will be calculated (will be anyways ...).
        system_changes: list(str), optional, default all changes
            List of what has changed since last calculation.

        Results
        -------
        dict(str, any)
            ASE atoms property predictions

        """

        # Collect atoms object
        if atoms is None and self.atoms is None:
            raise ase_calc.CalculatorSetupError(
                "ASE Atoms object is not defined!"
            )
        elif atoms is None:
            atoms = self.atoms
        elif self.atoms is None:
            self.atoms = atoms

        # Set the atoms position and, eventually, cell parameter with
        # respect to the ASE Atoms object, if given.
        self.openmm_simulation.context.setPositions(
            atoms.get_positions()*unit.angstrom
        )
        if np.any(atoms.get_pbc()):
            self.openmm_simulation.context.setPeriodicBoxVectors(
                *[
                    openmm.Vec3(*edge*unit.angstrom)
                    for edge in atoms.get_cell()
                ]
            )

        # Compute result properties
        state = self.openmm_simulation.context.getState(
            getEnergy=True,
            getForces=True,
        )
        self.results = {
            'energy': (
                state.getPotentialEnergy()._value*self.openmm_conversion_energy
            ),
            'forces': np.array(
                [forces_i[:] for forces_i in state.getForces()._value]
            )*self.openmm_conversion_forces,
        }

        return self.results
