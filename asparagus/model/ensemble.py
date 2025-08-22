import logging
from typing import Optional, List, Dict, Tuple, Callable, Union, Any

import numpy as np

import ase
import torch

from asparagus import data
from asparagus import model
from asparagus import settings
from asparagus import utils

__all__ = ['EnsembleModel']

#======================================
# Calculator Models
#======================================

class EnsembleModel(torch.nn.Module):
    """
    Asparagus ensemble model calculator

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to config json file (str)
    model_calculator_class: callable, optional, default None
        Number of model calculator in ensemble. If None and
        'model_ensemble' is True, take all available models found.
    model_ensemble_num: int, optional, default None
        Number of model calculator in ensemble. If None and
        'model_ensemble' is True, take all available models found.

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for graph module
    _default_args = {
        'model_calculator_class':       None,
        'model_ensemble_num':           3,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'model_calculator_class':       [utils.is_None, utils.is_callable],
        'model_ensemble_num':           [utils.is_integer],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_calculator_class: Optional[Callable] = None,
        model_ensemble_num: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize EnsembleModel Calculator


        """

        super(EnsembleModel, self).__init__()

        #############################
        # # # Check Class Input # # #
        #############################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, model),
            check_dtype=utils.get_dtype_args(self, model))

        # Update global configuration dictionary
        config.update(
            config_update,
            verbose=verbose)

        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)

        #############################
        # # # Check Model Input # # #
        #############################

        # Check model calculator class
        if self.model_calculator_class is None:
            if (
                kwargs.get('model_type') is None
                and config.get('model_type') is None
            ):
                model_type = settings._default_calculator_model
            elif config.get('model_type') is None:
                model_type = kwargs.get('model_type')
            else:
                model_type = config.get('model_type')

            # Get requested calculator model
            self.model_calculator_class = (
                model.calculator._get_model_calculator(model_type))

        #######################################
        # # # Initialize Model Calculator # # #
        #######################################

        # Initialize model calculator list
        model_calculator_list = []

        # Iterate over number of model calculator
        for _ in range(self.model_ensemble_num):

            # Initialize model calculator
            model_calculator_list.append(self.model_calculator_class(
                config=config,
                verbose=verbose,
                device=self.device,
                dtype=self.dtype,
                **kwargs)
            )

        # Convert model calculator list to torch module list
        self.model_calculator_list = torch.nn.ModuleList(
            model_calculator_list)

        ##################################
        # # # Adopt Model Parameters # # #
        ##################################

        # Assign model ensemble flag
        self.model_ensemble = True

        # Model type label
        self.model_type = self.model_calculator_list[0].model_type
        self._model_type = self.model_calculator_list[0]._model_type

        # Model properties and units
        self.model_properties = (
            self.model_calculator_list[0].model_properties.copy())
        self.model_unit_properties = (
            self.model_calculator_list[0].model_unit_properties.copy())
        for prop in self.model_calculator_list[0].model_properties:
            prop_std = f"std_{prop:s}"
            self.model_properties.append(prop_std)
            self.model_unit_properties[prop_std] = (
                self.model_calculator_list[0].model_unit_properties[prop])

        # Model cutoff ranges
        if hasattr(self.model_calculator_list[0], 'model_cutoff'):
            self.model_cutoff = (
                self.model_calculator_list[0].model_cutoff)
        if hasattr(self.model_calculator_list[0], 'model_cuton'):
            self.model_cuton = (
                self.model_calculator_list[0].model_cuton)
        if hasattr(self.model_calculator_list[0], 'model_switch_range'):
            self.model_switch_range = (
                self.model_calculator_list[0].model_switch_range)

        return

    def __str__(self):
        return f"{self._model_type:s} Ensemble"

    def __len__(
        self
    ) -> int:
        return len(self.model_calculator_list)

    def __getitem__(
        self,
        idx: int
    ):
        return self.model_calculator_list[idx]

    def __iter__(
        self,
    ):
        # Start model counter and model ensemble number
        self.counter = 0
        self.Nmodels = len(self)
        return self

    def __next__(
        self
    ):
        # Check counter within number of data range
        if self.counter < self.Nmodels:
            model = self.model_calculator_list[self.counter]
            self.counter += 1
            return model
        else:
            raise StopIteration

    def get_info(self) -> Dict[str, Any]:
        """
        Return model ensemble, model and module information
        """

        # Initialize info dictionary
        info = {}

        # Collect model and module info
        if hasattr(self.model_calculator_list[0], "get_info"):
            info = {**info, **self.model_calculator_list[0].get_info()}

        return {
            **info,
            'model_type': self._model_type,
            'model_ensemble': self.model_ensemble,
            'model_ensemble_num': self.model_ensemble_num,
        }

    @property
    def checkpoint_loaded(self):
        return all([
            model_calculator.checkpoint_loaded
            for model_calculator in self.model_calculator_list])

    @property
    def checkpoint_file(self):
        return [
            model_calculator.checkpoint_file
            for model_calculator in self.model_calculator_list]

    def load(
        self,
        checkpoint: List[Dict[str, Any]],
        checkpoint_file: Optional[List[str]] = None,
        verbose: Optional[bool] = True,
    ):
        """
        Load model parameters from checkpoint file.

        Parameters
        ----------
        checkpoint: list(dict(str, Any))
            Torch module checkpoint files for each model calculator in the
            ensemble
        checkpoint_file: list(str), optional, default None
            List of torch checkpoint file pathways for logger info

        """

        # Load model checkpoint file
        if (
            (checkpoint is None or utils.is_None_array(checkpoint))
            and verbose
        ):

            # Prepare and print loading info
            if (
                checkpoint_file is None
                or utils.is_None_array(checkpoint_file)
            ):
                checkpoint_state = "."
            else:
                checkpoint_state = " from files:\n"
                for ickpt, ckpt_file in enumerate(checkpoint_file):
                    checkpoint_state += (
                        f" Model {ickpt:d} - '{ckpt_file:s}'\n")
                checkpoint_state = checkpoint_state[:-1]
            self.logger.info(
                f"No checkpoint files are loaded{checkpoint_state:s}")

        else:

            # Prepare loading info
            if checkpoint_file is None:
                checkpoint_state = "."
            else:
                checkpoint_state = " from files:\n"

            # Iterate over number of model calculator
            for ickpt, (model_calculator, ckpt_i) in enumerate(zip(
                self.model_calculator_list,
                checkpoint
            )):

                if ckpt_i is None:
                    
                    checkpoint_state += (f" Model {ickpt:d} - not loaded!\n")
                
                else:

                    model_calculator.load_state_dict(
                        ckpt_i['model_state_dict'])
                    model_calculator.checkpoint_loaded = True
                    if (
                        checkpoint_file is None 
                        or checkpoint_file[ickpt] is None
                    ):
                        model_calculator.checkpoint_file = None
                        checkpoint_state += (
                            f" Model {ickpt:d} - loaded!\n")
                    else:
                        model_calculator.checkpoint_file = (
                            checkpoint_file[ickpt])
                        checkpoint_state += (
                            f" Model {ickpt:d} - "
                            + f"'{checkpoint_file[ickpt]:s}' loaded!\n")

            # Print loading info
            if verbose:
                checkpoint_state = checkpoint_state[:-1]
                self.logger.info(
                    f"Checkpoint files are loaded{checkpoint_state:s}")

        return

    def get_cutoff_ranges(self) -> List[float]:
        """
        Get model cutoff or, eventually, short range descriptor and long
        range cutoff list.

        Return
        ------
        list(float)
            List of the long range model and, eventually, short range
            descriptor cutoff (if defined and not short range equal long range
            cutoff).

        """
        return self.model_calculator_list[0].get_cutoff_ranges()

    # @torch.compile # Not supporting backwards propagation with torch.float64
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        no_derivation: Optional[bool] = False,
        verbose_results: Optional[bool] = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model ensemble calculator.

        Parameters
        ----------
        batch: dict(str, torch.Tensor)
            Dictionary of input data tensors for forward pass.
            Basic keys are:
                'atoms_number': torch.Tensor(n_systems)
                    Number of atoms per molecule in batch
                'atomic_numbers': torch.Tensor(n_atoms)
                    Atomic numbers of the batch of molecules
                'positions': torch.Tensor(n_atoms, 3)
                    Atomic positions of the batch of molecules
                'charge': torch.Tensor(n_systems)
                    Total charge of molecules in batch
                'idx_i': torch.Tensor(n_pairs)
                    Atom i pair index
                'idx_j': torch.Tensor(n_pairs)
                    Atom j pair index
                'sys_i': torch.Tensor(n_atoms)
                    System indices of atoms in batch
            Extra keys are:
                'pbc_offset': torch.Tensor(n_pairs)
                    Periodic boundary atom pair vector offset
                'pbc_atoms': torch.Tensor(n_atoms)
                    Primary atom indices for the supercluster approach
                'pbc_idx': torch.Tensor(n_pairs)
                    Image atom to primary atom index pointer for the atom
                    pair indices in a supercluster
                'pbc_idx_j': torch.Tensor(n_pairs)
                    Atom j pair index pointer from image atom to respective
                    primary atom index in a supercluster
        no_derivation: bool, optional, default False
            If True, only predict non-derived properties.
            Else, predict all properties even if backwards derivation is
            required (e.g. forces).
        verbose_results: bool, optional, default False
            If True, store single model property predictions and extended model
            property contributions.

        Returns
        -------
        dict(str, torch.Tensor)
            Model ensemble average property predictions, uncertainties and,
            if requested, single model property predictions.

        """

        # Initialize model and model ensemble result dictionaries
        model_results = {}
        ensemble_results = {}

        # Iterate over ensemble models
        for ic, model_calculator in enumerate(self.model_calculator_list):
            model_results[ic] = model_calculator(
                batch,
                no_derivation=no_derivation,
                verbose_results=verbose_results)

        # Accumulate model results
        for prop in self.model_properties:
            prop_std = f"std_{prop:s}"
            ensemble_results[prop_std], ensemble_results[prop] = (
                torch.std_mean(
                    torch.stack(
                        [results[prop] for results in model_results.values()]),
                    dim=0)
                )

        # Update model ensemble results with single results, if requested
        if verbose_results:
            ensemble_results.update(model_results)

        return ensemble_results

    def calculate(
        self,
        atoms: ase.Atoms,
        charge: Optional[float] = 0.0,
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of the ensemble model calculator with an ASE Atoms
        object.

        Parameters
        ----------
        atoms: ase.Atoms
            ASE Atoms object to calculate properties
        charge: float, optional, default 0.0
            Total system charge

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Create atoms batch dictionary
        atoms_batch = self.create_batch(
            atoms,
            charge=charge)

        return self.forward(atoms_batch)

    def create_batch(
        self,
        atoms: Union[ase.Atoms, List[ase.Atoms]],
        charge: Optional[Union[float, List[float]]] = None,
        conversion: Optional[Dict[str, float]] = {},
    ) -> Dict[str, torch.Tensor]:
        """
        Create a systems batch dictionary as input for the model calculator.

        Parameters
        ----------
        atoms: (ase.Atoms, list(ase.Atoms))
            ASE Atoms object or list of ASE Atoms object to prepare batch
            dictionary.
        charge: (float list(float)), optional, default 0.0
            Total system charge or list of total system charge.
            If None, charge is estimated from the ASE Atoms objects, mostly
            set as zero.
        conversion: dict(str, float), optional, default {}
            ASE Atoms conversion dictionary from ASE units to model units.

        Returns
        -------
        dict(str, torch.Tensor)
            System(s) batch dictionary used as model calculator input

        """

        return self.model_calculator_list[0].create_batch(
            atoms,
            charge=charge,
            conversion=conversion)

    def create_batch_copies(
        self,
        atoms: ase.Atoms,
        ncopies: Optional[int] = None,
        positions: Optional[List[float]] = None,
        cell: Optional[List[float]] = None,
        charge: Optional[float] = None,
        conversion: Optional[Dict[str, float]] = {},
    ) -> Dict[str, torch.Tensor]:
        """
        Create a systems batch dictionary as input for the model calculator.

        Parameters
        ----------
        atoms: (ase.Atoms)
            ASE Atoms object to prepare multiple copies in a batch dictionary.
        ncopies: int, optional, default None
            Number of copies of the ASE atoms system in the batch.
            If None, number of copies are taken from 'positions' or 'cell'
            input, otherwise is 1.
        positions: list(float), optional, default None
            Array of shape ('Ncopies', 'Natoms', 3) where 'Ncopies' is the
            number of copies of the ASE Atoms system in the batch and 'Natoms'
            is the number of atoms.
            If None, the positions of the ASE Atoms object is taken.
        cell: list(float), optional, default None
            Array of ASE Atoms cell parameter of shape ('Ncopies', 3).
            If None, the cell parameters from the ASE Atoms object is taken.
        charge: float, optional, default 0.0
            Total system charge of the ASE atoms object.
            If None, charge is estimated from the ASE Atoms objects, mostly
            set as zero.
        conversion: dict(str, float), optional, default {}
            ASE Atoms conversion dictionary from ASE units to model units.

        Returns
        -------
        dict(str, torch.Tensor)
            System(s) batch dictionary used as model calculator input

        """

        return self.model_calculator_list[0].create_batch_copies(
            atoms,
            ncopies=ncopies,
            positions=positions,
            cell=cell,
            charge=charge,
            conversion=conversion)

    def calculate_data(
        self,
        dataset: Union[data.DataContainer, data.DataSet, data.DataSubSet],
        batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 1,
        verbose_results: Optional[bool] = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        """
        Forward pass of the calculator model with an Asparagus data set.

        Parameters
        ----------
        dataset: (data.DataContainer, data.DataSet, data.DataSubSet)
            Asparagus DataContainer or DataSet object
        batch_size: int, optional, default 32
            Data loader batch size
        num_workers: int, optional, default 1
            Number of data loader workers
        verbose_results: bool, optional, default False
            If True, store single model property predictions and extended model
            property contributions.

        Returns
        -------
        dict(str, torch.Tensor)
            Model property predictions

        """

        # Prepare data loader for the data set
        dataloader = data.DataLoader(
            dataset,
            batch_size,
            [],
            False,
            num_workers,
            self.device,
            self.dtype)

        # Initialize model ensemble result dictionary
        results = {}

        # Iterate over data batches
        for ib, batch in enumerate(dataloader):
            
            # Predict model properties from data batch
            prediction = self.forward(
                batch, 
                verbose_results=verbose_results,
                **kwargs)

            # Append prediction to result dictionary
            for prop in self.model_properties:
                if verbose_results and utils.is_dictionary(prediction[prop]):
                    if results.get(prop) is None:
                        results[prop] = {}
                    for sub_prop, sub_result in prediction[prop].items():
                        if results[prop].get(sub_prop) is None:
                            results[prop][sub_prop] = [
                                sub_result.cpu().detach()]
                        else:
                            results[prop][sub_prop].append(
                                sub_result.cpu().detach())
                elif results.get(prop) is None:
                    results[prop] = [prediction[prop].cpu().detach()]
                else:
                    results[prop].append(prediction[prop].cpu().detach())

        # Concatenate results
        for prop in self.model_properties:
            if verbose_results and utils.is_dictionary(prediction[prop]):
                for sub_prop, sub_result in prediction[prop].items():
                    if ib and sub_result[0].shape:
                        results[prop][sub_prop] = torch.cat(sub_result)
                    elif ib:
                        results[prop][sub_prop] = torch.cat(
                            [result.reshape(1) for result in sub_result],
                            dim=0)
                    else:
                        results[prop][sub_prop] = sub_result[0]
            elif ib and prediction[prop][0].shape:
                results[prop] = torch.cat(prediction[prop])
            elif ib:
                results[prop] = torch.cat(
                    [result.reshape(1) for result in prediction[prop]], dim=0)
            else:
                results[prop] = prediction[prop][0]

        return results
