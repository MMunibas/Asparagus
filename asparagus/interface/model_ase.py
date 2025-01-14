import numpy as np
from typing import Optional, List, Dict, Callable, Tuple, Union, Any

import ase
import ase.calculators.calculator as ase_calc

import torch

from asparagus import module
from asparagus import utils
from asparagus import settings

__all__ = ['ASE_Calculator']


class ASE_Calculator(ase_calc.Calculator):
    """
    ASE calculator interface for a Asparagus model potential.

    Parameters
    ----------
    model_calculator: (callable object, list of callable objects)
        Model calculator(s) to predict model properties. If an ensemble
        is given in form of a list of model calculators, the average value
        is returned as model prediction.
    atoms: ASE Atoms object, optional, default None
        ASE Atoms object to which the calculator will be attached.
    charge: float, optional, default 0.0
        Total charge of the respective ASE Atoms object.
    implemented_properties: (str, list(str)), optional, default None
        Properties predicted by the model calculator. If None, then
        all model properties (of the first model if ensemble) are
        available.
    label: str, optional, default 'asparagus'
        Label for the ASE calculator

    """

    # ASE specific calculator information
    default_parameters = {
        "method": "Asparagus"}

    def __init__(
        self,
        model_calculator: Union[Callable, List[Callable]],
        atoms: Optional[ase.Atoms] = None,
        charge: Optional[float] = None,
        implemented_properties: Optional[List[str]] = None,
        label: Optional[str] = 'asparagus',
        **kwargs
    ):
        """
        Initialize ASE Calculator class.

        """

        # Initialize parent Calculator class
        ase_calc.Calculator.__init__(self, atoms=atoms, **kwargs)

        ###################################
        # # # Check NNP Calculator(s) # # #
        ###################################

        # Assign NNP calculator model(s)
        self.model_calculator = model_calculator
        if hasattr(self.model_calculator, 'model_ensemble'):
            self.model_ensemble = self.model_calculator.model_ensemble
        else:
            self.model_ensemble = False

        # Assign model calculator variables
        self.model_device = self.model_calculator.device
        self.model_dtype = self.model_calculator.dtype

        # Set implemented properties
        if implemented_properties is None:
            self.implemented_properties = (
                self.model_calculator.model_properties)
        else:
            if utils.is_string(implemented_properties):
                self.implemented_properties = [implemented_properties]
            else:
                self.implemented_properties = implemented_properties

        # Check if model calculator has loaded a checkpoint file
        if not self.model_calculator.checkpoint_loaded:
            raise SyntaxError(
                "The model calculator does not seem to have a "
                + "proper parameter set loaded from a checkpoint file."
                + "\nMake sure parameters are loaded otherwise "
                + "model predictions are random.")

        ##################################
        # # # Set Calculator Options # # #
        ##################################

        # Initialize neighbor list function
        cutoffs = self.model_calculator.get_cutoff_ranges()
        self.neighbor_list = module.TorchNeighborListRangeSeparated(
            cutoffs,
            self.model_device,
            self.model_dtype)

        # Get unit conversion dictionary
        self.model_conversion = self.check_model_units(
            self.model_calculator.model_unit_properties)

        # Initialize atoms object batch
        if atoms is None:
            self.atoms_batch = {}
        else:
            self.atoms_batch = self.model_calculator.create_batch(
                atoms,
                charge=charge,
                conversion=self.model_conversion)

        # Initialize result dictionary
        self.results = {}

        # Initialize convergence flag
        self.converged = False

        return

    def check_model_units(
        self,
        model_units: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Check model units with respect to ASE units and return conversion
        factors dictionary

        Parameter
        ---------
        model_units: dict(str, str)
            Dictionary of model property units.

        Returns
        -------
        dict(str, float)
            Dictionary of model to data property unit conversion factors

        """

        # Initialize model conversion factor dictionary
        model_conversion = {}

        # Check positions unit (None = ASE units by default)
        conversion, _ = utils.check_units(None, model_units['positions'])
        model_conversion['positions'] = conversion

        # Check implemented property units (None = ASE units by default)
        for prop in self.implemented_properties:
            conversion, _ = utils.check_units(None, model_units[prop])
            model_conversion[prop] = conversion

        return model_conversion

    def update_model_input(
        self, 
        batch: Dict[str, torch.Tensor],
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        charge: Optional[Union[float, List[float]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Update atoms data bach by the changes in the assigned ASE atoms system.
        
        Parameter
        ---------
        batch: dict(str, torch.Tensor)
            ASE atoms data batch
        atoms: (ase.Atoms, list(ase.Atoms))
            ASE Atoms object or list of ASE Atoms objects to update the data
            batch.
        charge: (float, list(float)), optional, default None
            Optional total charge of the respective ASE Atoms object or
            objects.
        
        Returns
        -------
        dict(str, torch.Tensor)
            Updated atoms data batch

        """

        # Check atoms input
        if utils.is_ase_atoms(atoms):
            atoms = [atoms]
        elif not utils.is_ase_atoms_array(atoms):
            raise ValueError(
                "Input 'atoms' is not an ASE Atoms object or list of ASE "
                + "atoms objects!")

        # If model input is not initialized
        if not batch:
            batch = self.model_calculator.create_batch(
                atoms,
                charge=charge,
                conversion=self.model_conversion)
            return batch

        # Update atom positions and cell parameters
        fconv = self.model_conversion['positions']
        batch['positions'] = torch.cat(
            [
                torch.tensor(atms.get_positions()*fconv, dtype=self.dtype)
                for atms in atoms
            ], 0).to(
                device=self.model_device, dtype=self.model_dtype)
        batch['cell'] = torch.tensor(
            np.array([atms.get_cell()[:]*fconv for atms in atoms]),
            dtype=self.model_dtype, device=self.model_device)

        # Create and assign atom pair indices and periodic offsets
        batch = self.neighbor_list(batch)
        
        return batch

    def calculate(
        self,
        atoms: Optional[Union[ase.Atoms, List[ase.Atoms]]] = None,
        charge: Optional[Union[float, List[float]]] = None,
        properties: List[str] = None,
        system_changes: List[str] = ase_calc.all_changes,
        verbose_results: Optional[bool] = False
    ) -> Dict[str, Any]:
        """
        Calculate model properties

        Parameter
        ---------
        atoms: (ase.Atoms, list(ase.Atoms)), optional, default None
            Optional ASE Atoms object or list of ASE Atoms objects to which the
            properties will be calculated. If given, atoms setup to prepare
            model calculator input will be run again.
        charge: (float, list(float)), optional, default None
            Optional total charge of the respective ASE Atoms object or
            objects. If the atoms charge is given as a float, the charge is
            assumed for all ASE atoms objects given.
        properties: list(str), optional, default None
            List of properties to be calculated. If None, all implemented
            properties will be calculated (will be anyways ...).
        verbose_results: bool, optional, default False
            If True, store extended model property contributions in the result
            dictionary.

        Results
        -------
        dict(str, any)
            ASE atoms property predictions

        """
        
        # Prepare or update atoms data batch
        if atoms is None and self.atoms is None:

            raise ase_calc.CalculatorSetupError(
                "ASE atoms object is not defined!")

        elif atoms is None:

            # Update data batch with linked atoms object(s)
            self.atoms_batch = self.update_model_input(
                self.atoms_batch,
                self.atoms,
                charge)

        else:

            # Reset data batch with the input atoms object(s)
            self.atoms_batch = self.update_model_input(
                {},
                atoms,
                charge)

        # Compute model properties
        prediction = self.model_calculator(
            self.atoms_batch,
            verbose_results=verbose_results)

        # Convert model properties
        self.assign_prediction(prediction, self.atoms_batch)

        return self.results

    def assign_prediction(
        self,
        prediction: Dict[str, torch.Tensor],
        atoms_batch: Dict[str, torch.Tensor],
    ):
        """
        Convert and assign and model prediction to the results dictionary.
        
        Parameter
        ---------
        prediction: list(float)
            Model prediction dictionary for ASE atoms system in batch
        atoms_batch: dict(str, torch.Tensor)
            ASE atoms systems data batch

        """

        # Check for multiple system prediction
        multi_sys = len(atoms_batch['atoms_number']) > 1
        if multi_sys:
            Nsys = atoms_batch['atoms_number'].shape[0]
            Natoms = atoms_batch['atomic_numbers'].shape[0]
            Npairs = atoms_batch['idx_i'].shape[0]

        # Iterate over model properties
        for prop in self.implemented_properties:

            # Convert property prediction
            pred = (
                prediction[prop].cpu().detach().numpy()
                *self.model_conversion[prop])

            # Resolve prediction system-wise
            if multi_sys:

                if not pred.shape:
                    pass
                elif pred.shape[0] == Nsys:
                    pred = [pred_i for pred_i in pred]
                elif pred.shape[0] == Natoms:
                    pred = [
                        pred[atoms_batch['sys_i'] == i_sys]
                        for i_sys in range(Nsys)]
                elif pred.shape[0] == Npairs:
                    pred = [
                        pred[
                            atoms_batch['sys_i'][atoms_batch['idx_i']] == i_sys
                        ] for i_sys in range(Nsys)]

            # Assign to results
            self.results[prop] = pred

        return
