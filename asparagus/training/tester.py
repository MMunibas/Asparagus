import os
import sys

import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import torch

from asparagus import data
from asparagus import settings
from asparagus import utils
from asparagus import training

# These packages are required for all functions of plotting and analysing
# the model.
try:
    import pandas as pd
except ImportError:
    raise UserWarning(
        "You need to install pandas to use all "
        + "plotting and analysis functions")
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise UserWarning(
        "You need to install matplotlib to use all "
        + "plotting and analysis functions")
try:
    from scipy import stats
except ImportError:
    raise UserWarning(
        "You need to install scipy to use all plotting and analysis functions")

__all__ = ['Tester']

class Tester:
    """
    Model Prediction Tester Class

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    data_container: data.DataContainer, optional
        Data container object of the reference test data set.
        If not provided, the data container will be initialized according
        to config input.
    test_datasets: (str, list(str)) optional, default ['test']
        A string or list of strings to define the data sets ('train',
        'valid', 'test') of which the evaluation will be performed.
        By default it is just the test set of the data container object.
        Inputs 'full' or 'all' requests the evaluation of all sets.
    test_properties: (str, list(str)), optional, default None
        Model properties to evaluate which must be available in the
        model prediction and the reference test data set. If None, all
        model properties will be evaluated if available in the test set.
    test_batch_size: int, optional, default None
        Reference dataloader batch size
    test_num_batch_workers: int, optional, default 1
        Number of data loader workers.
    test_directory: str, optional, default '.'
        Directory to store evaluation graphics and data.

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for tester class
    _default_args = {
        'test_datasets':                ['test'],
        'test_properties':              None,
        'test_batch_size':              128,
        'test_num_batch_workers':       0,
        'test_directory':               '.',
        }

    # Expected data types of input variables
    _dtypes_args = {
        'test_datasets':                [
            utils.is_string, utils.is_string_array],
        'test_properties':            [
            utils.is_string, utils.is_string_array, utils.is_None],
        'test_batch_size':              [utils.is_integer],
        'test_num_batch_workers':       [utils.is_integer],
        'test_directory':               [utils.is_string],
        }

    # Model properties available for all fragment atoms, which will be 
    # additionally evaluated separately
    _fragment_properties = ['forces']

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        data_container: Optional[data.DataContainer] = None,
        test_datasets: Optional[Union[str, List[str]]] = None,
        test_properties: Optional[Union[str, List[str]]] = None,
        test_batch_size: Optional[int] = None,
        test_num_batch_workers: Optional[int] = None,
        test_directory: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize model tester.

        """

        ####################################
        # # # Check Model Tester Input # # #
        ####################################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, training),
            check_dtype=utils.get_dtype_args(self, training)
        )

        # Update global configuration dictionary
        config.update(
            config_update,
            config_from=self,
            verbose=verbose)

        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)

        ################################
        # # # Check Data Container # # #
        ################################

        # Assign DataContainer and test data loader
        if self.data_container is None:
            self.data_container = data.DataContainer(
                config=config,
                **kwargs)

        # Get reference data properties
        self.data_properties = self.data_container.data_properties
        self.data_units = self.data_container.data_unit_properties

        ##########################
        # # # Prepare Tester # # #
        ##########################

        # Check test properties if defined
        self.test_properties = self.check_test_properties(
            self.test_properties,
            self.data_properties)

        # Check test directory
        if not os.path.isdir(self.test_directory):
            os.makedirs(self.test_directory)

        #########################################
        # # # Prepare Reference Data Loader # # #
        #########################################

        # Initialize training, validation and test data loader
        self.data_container.init_dataloader(
            self.test_batch_size,
            self.test_batch_size,
            self.test_batch_size,
            reference_properties=self.test_properties,
            num_workers=self.test_num_batch_workers,
            device=self.device,
            dtype=self.dtype)

        # Prepare list of data set definition for evaluation
        if utils.is_string(self.test_datasets):
            self.test_datasets = [self.test_datasets]
        if 'full' in self.test_datasets or 'all' in self.test_datasets:
            self.test_datasets = self.data_container.get_datalabels()

        # Collect requested data loader
        self.test_data = {
            label: self.data_container.get_dataloader(label)
            for label in self.test_datasets}

        return

    def check_test_properties(
        self,
        test_properties: Union[str, List[str]],
        data_properties: List[str],
    ) -> List[str]:
        """
        Check availability of 'test_properties' in 'data_properties' and
        return eventually corrected test_properties as list.

        Parameters
        ----------
        test_properties: (str, list(str)), optional, default None
            Model properties to evaluate which must be available in the
            model prediction and the reference test data set. If None, model
            properties will be evaluated as initialized.
        data_properties: list(str), optional, default None
            List of properties available in the reference data set.

        Returns
        -------
        List[str]
            Test properties

        """

        # If not defined, take all reference properties, else check
        # availability
        if test_properties is None:
            test_properties = data_properties
        else:
            if utils.is_string(test_properties):
                test_properties = [test_properties]
            checked_properties = []
            for prop in test_properties:
                if prop not in data_properties:
                    self.logger.warning(
                        f"Requested property '{prop}' in " +
                        "'test_properties' for the model evaluation " +
                        "is not avaible in the reference data set and " +
                        "will be ignored!")
                else:
                    checked_properties.append(prop)
            test_properties = checked_properties

        return test_properties

    def test(
        self,
        model_calculator: torch.nn.Module,
        model_conversion: Optional[Dict[str, float]] = None,
        mlmm_inf_cutoff: Optional[bool] = True,
        test_properties: Optional[Union[str, List[str]]] = None,
        test_directory: Optional[str] = None,
        test_plot_correlation: Optional[bool] = True,
        test_plot_histogram: Optional[bool] = True,
        test_plot_residual: Optional[bool] = True,
        test_plot_format: Optional[str] = 'png',
        test_plot_dpi: Optional[int] = 300,
        test_save_csv: Optional[bool] = False,
        test_csv_file: Optional[str] = 'results.csv',
        test_save_npz: Optional[bool] = False,
        test_npz_file: Optional[str] = 'results.npz',
        test_scale_per_atom: Optional[Union[str, List[str]]] = ['energy'],
        test_outlier: Optional[bool] = False,
        test_outlier_property: Optional[str] = 'energy',
        test_outlier_num_max: Optional[int] = 10,
        test_outlier_threshold: Optional[float] = None,
        test_outlier_metric: Optional[str] = 'mse',
        verbose: Optional[bool] = True,
        **kwargs,
    ):
        """

        Main function to evaluate the model prediction on the test data set.

        Parameters
        ----------
        model_calculator: torch.nn.Module
            NNP model calculator to predict test properties. The prediction
            are done with the given state of parametrization, no checkpoint
            files will be loaded.
        model_conversion: dict(str, float), optional, default None
            Model prediction to reference data unit conversion.
        mlmm_inf_cutoff: bool, optional, default True
            For comparisons with reference data, set the ML-MM (electrostatic) 
            interaction cutoff to infinity, as usually the case in reference
            QM-MM calculation (e.g. ORCA). 
            Note that in case of an infinite cutoff, the periodic boundary
            conditions are ignored and only atom pairs within the primary
            cell (if one is defined) are considered.
        test_properties: (str, list(str)), optional, default None
            Model properties to evaluate which must be available in the
            model prediction and the reference test data set. If None, model
            properties will be evaluated as initialized.
        test_directory: str, optional, default '.'
            Directory to store evaluation graphics and data.
        test_plot_correlation: bool, optional, default True
            Show evaluation in property correlation plots
            (x-axis: reference property; y-axis: predicted property).
        test_plot_histogram: bool, optional, default True
            Show prediction error spread in histogram plots.
        test_plot_residual: bool, optional, default True
            Show evaluation in residual plots.
            (x-axis: reference property; y-axis: prediction error).
        test_plot_format: str, optional, default 'png'
            Plot figure format (for options see matplotlib.pyplot.savefig()).
        test_plot_dpi: int, optional, default 300
            Plot figure dpi.
        test_save_csv: bool, optional, default False
            Save all model prediction results and respective reference values
            in a csv file.
        test_csv_file: str, optional, default 'results.csv'
            Name tag of the csv file. The respective data set label and 
            property will be added as prefix to the tag:
            "{label:s}_{property:s}_{test_csv_file:s}"
        test_save_npz: bool, optional, default False
            Save all model prediction results and respective reference values
            in a binary npz file.
        test_npz_file: str, optional, default 'results.npz'
            Name tag of the npz file. The respective data set label and 
            property will be added as prefix to the tag:
            "{label:s}_{property:s}_{test_csv_file:s}"
        test_scale_per_atom: (str list(str), optional, default ['energy']
            List of properties where the results will be scaled by the number
            of atoms in the particular system.
        test_outlier: bool, optional, default False
            If True, show outlier in correlation plots with database id.
            By default, outliers are depicted as the 'test_num_outlier'
            number of systems with the largest system error 
            'test_outlier_metric' of the property 'test_outlier_property'.
            If 'test_outlier_threshold' is given (by default 'None'), outliers
            are system where the error of the property
            'test_outlier_property' is by a factor 'test_outlier_threshold'
            larger than the average error of the dataset to test.
        test_outlier_property: bool, optional, default 'energy'
            System property to detect outliers.
        test_num_outlier: int, optional, default 10
            Number of system with the largest system error
            'test_outlier_metric' of the property 'test_outlier_property' to
            depict as outlier.
        test_outlier_threshold: float, optional, default None
            Threshold factor when system error 'test_outlier_metric' of
            property 'test_outlier_property' is this much larger than the
            average error of the dataset to test.
        test_outlier_metric: str, optional, default 'mse'
            Error metric ('mse', 'rmse' or 'mae') to use for outlier detection.
            Only effects the outcome if 'test_outlier_threshold' is used.
        verbose: bool, optional, default True
            Print test metrics.

        """

        #################################
        # # # Check Test Properties # # #
        #################################

        # Get model properties
        if hasattr(model_calculator, "model_properties"):
            model_properties = model_calculator.model_properties
        else:
            raise AttributeError(
                "Model calculator has no 'model_properties' attribute")
        if hasattr(model_calculator, "model_ensemble"):
            model_ensemble = model_calculator.model_ensemble
            model_ensemble_num = model_calculator.model_ensemble_num
        else:
            model_ensemble = False
            model_ensemble_num = None

        # Check test properties if defined or take initialized ones
        if test_properties is None:
            test_properties = self.test_properties
        else:
            test_properties = self.check_test_properties(
                test_properties,
                self.data_properties)

        # Check the model for additional required reference properties
        additional_properties = []
        if hasattr(model_calculator, '_required_input_properties'):
            for prop in model_calculator._required_input_properties:
                if prop in self.data_properties:
                    additional_properties.append(prop)

        # Update test properties
        for dataloader in self.test_data.values():
            dataloader.set_reference_properties(
                test_properties + additional_properties)
            dataloader.set_fragments(
                model_calculator.model_mlmm_embedding)

        # Check test properties model to reference data conversion
        test_conversion = {}
        for prop in test_properties:
            if model_conversion is None or model_conversion.get(prop) is None:
                test_conversion[prop] = 1.0
            else:
                test_conversion[prop] = model_conversion.get(prop)

        # Check test output directory
        if test_directory is None:
            test_directory = self.test_directory
        elif utils.is_string(test_directory):
            if not os.path.exists(test_directory):
                os.makedirs(test_directory)
        else:
            raise SyntaxError(
                "Test results output directory input 'test_directory' is not "
                + "a string of a valid file path.")

        ##############################
        # # # Compute Properties # # #
        ##############################

        # Get reference atomic energy shifts
        metadata = self.data_container.get_metadata()
        if 'data_atomic_energies_scaling' in metadata:
            data_atomic_energies_scaling_str = metadata.get(
                'data_atomic_energies_scaling')
            data_atomic_energies_scaling = {}
            for key, item in data_atomic_energies_scaling_str.items():
                data_atomic_energies_scaling[int(key)] = item
            max_atomic_number = max([
                int(atomic_number)
                for atomic_number in data_atomic_energies_scaling.keys()])
            atomic_energies_shift = np.zeros(
                max_atomic_number + 1, dtype=float)
            for atomic_number in range(max_atomic_number + 1):
                if atomic_number in data_atomic_energies_scaling:
                    atomic_energies_shift[atomic_number] = (
                        data_atomic_energies_scaling[atomic_number][0])
        else:
            atomic_energies_shift = None

        # Loop over all requested data set
        for label, datasubset in self.test_data.items():

            # Get model cutoffs
            cutoffs = model_calculator.get_cutoff_ranges()

            # Set maximum model cutoff for neighbor list calculation
            if datasubset.neighbor_list is None:
                datasubset.init_neighbor_list(
                    cutoff=cutoffs,
                    device=self.device,
                    dtype=self.dtype)
            else:
                datasubset.neighbor_list.set_cutoffs(cutoffs)

            # Get model ML/MM cutoffs
            mlmm_cutoffs = model_calculator.get_mlmm_cutoff_ranges(
                mlmm_inf_cutoff=mlmm_inf_cutoff
            )

            # Set maximum model cutoff for neighbor list calculation
            if datasubset.mlmm_neighbor_list is None:
                datasubset.init_mlmm_neighbor_list(
                    cutoff=mlmm_cutoffs,
                    device=self.device,
                    dtype=self.dtype)
            else:
                datasubset.mlmm_neighbor_list.set_cutoffs(mlmm_cutoffs)

            # Check fragment flag
            datasubset.set_fragments(model_calculator.model_mlmm_embedding)

            # Prepare dictionary for property values with system information
            # and a dictionary with reference energy shifts
            test_prediction = {}
            test_reference = {}

            test_prediction['atoms_number'] = np.array([], dtype=int)
            test_prediction['mlmm_atoms_number'] = np.array([], dtype=int)
            if test_outlier:
                for prop in test_properties:
                    prop_sys = 'sys_' + prop + '_i'
                    test_prediction[prop_sys] = np.array([], dtype=int)

            test_shifts = {
                prop: np.array([], dtype=float)
                for prop in ['energy', 'atomic_energies']
            }

            # Check for number of fragments in the dataset systems
            fragments_available = np.all([
                'fragment_numbers' in batch
                for batch in datasubset
                ])
            if fragments_available:
                test_fragments = []
                for batch in datasubset:
                    fragments = torch.unique(batch['fragment_numbers'])
                    for fragment in fragments:
                        if fragment not in test_fragments:
                            test_fragments.append(
                                fragment.detach().cpu().numpy())
                    test_prediction['fragment_numbers'] = (
                        np.array([], dtype=int)
                    )
            else:
                test_fragments = None

            # Reset property metrics
            metrics_test = self.reset_metrics(
                test_properties,
                model_ensemble,
                model_ensemble_num,
                test_fragments=test_fragments,
                test_outlier=test_outlier,
            )

            # Loop over data batches
            for batch in datasubset:

                # Predict model properties from data batch
                batch = model_calculator(
                    batch,
                    verbose_results=True)

                # Detach back-propagation graph from tensors and send to
                # CPU memory to save, potentially, GPU memory
                for key, item in batch.items():
                    if key == 'reference':
                        for key_ref, item_ref in batch[key].items():
                            batch[key][key_ref] = item_ref.detach().cpu()
                    else:
                        batch[key] = item.detach().cpu()
                if model_ensemble:
                    for imodel in range(model_ensemble_num):
                        for key, item in batch[imodel].items():
                            batch[imodel][key] = (
                                batch[imodel][key].detach().cpu()
                            )

                # Compute metrics for test properties
                metrics_batch = self.compute_metrics(
                    batch,
                    batch['reference'],
                    test_properties,
                    test_conversion,
                    model_ensemble,
                    model_ensemble_num,
                    test_fragments=test_fragments,
                    test_outlier=test_outlier,
                )

                # Update average metrics
                self.update_metrics(
                    metrics_test,
                    metrics_batch,
                    test_properties,
                    model_ensemble,
                    model_ensemble_num,
                    test_fragments=test_fragments,
                    test_outlier=test_outlier,
                )

                # Store prediction and reference data system resolved
                Nsys = batch['atoms_number'].shape[0]
                if 'ml_idx' in batch:
                    Natoms = batch['mlmm_sys_i'].shape[0]
                    sys_i = batch['mlmm_sys_i'].numpy()
                else:
                    Natoms = batch['sys_i'].shape[0]
                    sys_i = batch['sys_i'].numpy()
                Npairs = len(batch['idx_i'])
                for prop in test_properties:
                    
                    # Convert from torch tensors to numpy arrays
                    data_prediction = batch[prop].numpy()
                    data_reference = batch['reference'][prop].numpy()

                    # Ensure same prediction and reference data shape
                    data_shape = data_prediction.shape
                    data_reference = data_reference.reshape(data_shape)

                    # Apply unit conversion of model prediction
                    data_prediction *= test_conversion[prop]

                    # If data are numeric (no idea when)
                    if (not data_shape) and sys_i[-1] == 0:
                        data_prediction = np.array(
                            [data_prediction],
                            dtype=float
                        ).reshape(1, -1)
                        data_reference = np.array(
                            [data_reference],
                            dtype=float
                        ).reshape(1, -1)
                    # If data are system resolved
                    elif data_shape[0] == Nsys:
                        if test_outlier:
                            if data_prediction.ndim > 1:
                                nvals = np.prod(data_prediction.shape[1:])
                            else:
                                nvals = 1
                            prop_sys = 'sys_' + prop + '_i'
                            if test_prediction[prop_sys].size:
                                next_sys_i = test_prediction[prop_sys][-1] + 1
                            else:
                                next_sys_i = 0
                            test_prediction[prop_sys] = np.concatenate(
                                (
                                    test_prediction[prop_sys],
                                    np.arange(Nsys).repeat(nvals) + next_sys_i
                                ),
                                axis=0                                
                            )
                        data_prediction = np.array(
                            data_prediction,
                            dtype=float
                        ).reshape(Nsys, -1)
                        data_reference = np.array(
                            data_reference,
                            dtype=float
                        ).reshape(Nsys, -1)
                    # If data are atom resolved
                    elif data_shape[0] == Natoms:
                        if test_outlier:
                            if data_prediction.ndim > 1:
                                nvals = np.prod(data_prediction.shape[1:])
                            else:
                                nvals = 1
                            prop_sys = 'sys_' + prop + '_i'
                            if test_prediction[prop_sys].size:
                                next_sys_i = test_prediction[prop_sys][-1] + 1
                            else:
                                next_sys_i = 0
                            test_prediction[prop_sys] = np.concatenate(
                                (
                                    test_prediction[prop_sys],
                                    sys_i.repeat(nvals) + next_sys_i
                                ),
                                axis=0                                
                            )
                        data_prediction = np.array(
                            data_prediction,
                            dtype=float
                        ).reshape(Natoms, -1)
                        data_reference = np.array(
                            data_reference,
                            dtype=float
                        ).reshape(Natoms, -1)
                    # If data are atom pair resolved
                    elif data_shape[0] == Npairs:
                        sys_pair_i = sys_i[batch['idx_i']].numpy()
                        if test_outlier:
                            if data_prediction.ndim > 1:
                                nvals = np.prod(data_prediction.shape[1:])
                            else:
                                nvals = 1
                            prop_sys = 'sys_' + prop + '_i'
                            if test_prediction[prop_sys].size:
                                next_sys_i = test_prediction[prop_sys][-1] + 1
                            else:
                                next_sys_i = 0
                            test_prediction[prop_sys] = np.concatenate(
                                (
                                    test_prediction[prop_sys],
                                    sys_pair_i.repeat(nvals) + next_sys_i
                                ),
                                axis=0                                
                            )
                        data_prediction = np.array(
                            data_prediction,
                            dtype=float
                        ).reshape(Natoms, -1)
                        data_reference = np.array(
                            data_reference,
                            dtype=float
                        ).reshape(Natoms, -1)

                    # Assign prediction and reference data
                    if prop not in test_prediction:
                        test_prediction[prop] = np.empty(
                            (0, data_prediction.shape[1]),
                            dtype=float
                        )
                    if prop not in test_reference:
                        test_reference[prop] = np.empty(
                            (0, data_reference.shape[1]),
                            dtype=float
                        )                    
                    test_prediction[prop] = np.concatenate(
                        (test_prediction[prop], data_prediction),
                        axis=0
                    )
                    test_reference[prop] = np.concatenate(
                        (test_reference[prop], data_reference),
                        axis=0
                    )

                    if model_ensemble:

                        for imodel in range(model_ensemble_num):

                            # Detach prediction and reference data
                            data_prediction = (
                                batch[imodel][prop].numpy()
                                )
                            
                            # Ensure same prediction data shape
                            data_prediction = data_prediction.reshape(
                                data_shape
                            )

                            # Apply unit conversion of model prediction
                            data_prediction *= test_conversion[prop]

                            # If data are numeric (no idea when)
                            if (not data_shape) and sys_i[-1] == 0:
                                data_prediction = np.array(
                                    [data_prediction],
                                    dtype=float
                                ).reshape(1, -1)
                            # If data are system resolved
                            elif data_shape[0] == Nsys:
                                data_prediction = np.array(
                                    data_prediction,
                                    dtype=float
                                ).reshape(Nsys, -1)
                            # If data are atom resolved
                            elif data_shape[0] == Natoms:
                                data_prediction = np.array(
                                    data_prediction,
                                    dtype=float
                                ).reshape(Natoms, -1)
                            # If data are atom pair resolved
                            elif data_shape[0] == Npairs:
                                sys_pair_i = sys_i[batch['idx_i']].numpy()
                                data_prediction = np.array(
                                    data_prediction,
                                    dtype=float
                                ).reshape(Natoms, -1)
                            
                            
                            # Assign prediction data
                            if imodel not in test_prediction:
                                test_prediction[imodel] = {}
                            if prop not in test_prediction:
                                test_prediction[imodel][prop] = np.empty(
                                    (0, data_prediction.shape[1]),
                                    dtype=float
                                )
                            test_prediction[imodel][prop] = np.concatenate(
                                (
                                    test_prediction[imodel][prop],
                                    data_prediction
                                ),
                                axis=0
                            )

                # Store atom numbers
                test_prediction['atoms_number'] = np.concatenate(
                    (
                        test_prediction['atoms_number'],
                        batch['atoms_number'].numpy()
                    ),
                    axis=0
                )
                if 'mlmm_atoms_number' in batch:
                    test_prediction['mlmm_atoms_number'] = np.concatenate(
                        (
                            test_prediction['mlmm_atoms_number'],
                            batch['mlmm_atoms_number'].numpy()
                        ),
                        axis=0
                    )
                else:
                    test_prediction['mlmm_atoms_number'] = np.concatenate(
                        (
                            test_prediction['mlmm_atoms_number'],
                            batch['atoms_number'].numpy()
                        ),
                        axis=0
                    )

                # Store fragment numbers
                if 'fragment_numbers' in test_prediction:
                    test_prediction['fragment_numbers'] = np.concatenate(
                    (
                        test_prediction['fragment_numbers'],
                        batch['fragment_numbers'].numpy()
                    ),
                    axis=0
                )

                # Compute energy and atomic energies shifts
                test_shifts_energy = np.zeros(Nsys, dtype=float)
                test_shifts_atomic_energies = np.zeros(Natoms, dtype=float)
                if atomic_energies_shift is not None:
                    atomic_numbers = batch['atomic_numbers'].numpy()
                    sys_i = batch['sys_i'].numpy()
                    test_shifts_atomic_energies = (
                        atomic_energies_shift[atomic_numbers])
                    np.add.at(
                        test_shifts_energy,
                        sys_i,
                        test_shifts_atomic_energies)
                test_shifts['energy']  = np.concatenate(
                    (test_shifts['energy'], test_shifts_energy),
                    axis=0
                )
                test_shifts['atomic_energies']  = np.concatenate(
                    (
                        test_shifts['atomic_energies'],
                        test_shifts_atomic_energies
                    ),
                    axis=0
                )

            # Print metrics
            if verbose:
                self.print_metric(
                    metrics_test,
                    test_properties,
                    label,
                    model_ensemble,
                    model_ensemble_num,
                    test_fragments=test_fragments)

            ###########################
            # # # Save Properties # # #
            ###########################

            # Save test prediction to files
            if test_save_csv:
                self.save_csv(
                    test_prediction,
                    test_reference,
                    test_shifts,
                    label,
                    test_directory,
                    test_csv_file)
                if model_ensemble:
                    for imodel in range(model_ensemble_num):
                        test_directory_model = os.path.join(
                            test_directory, f"{imodel:d}")
                        if not os.path.exists(test_directory_model):
                            os.makedirs(test_directory_model)
                        self.save_csv(
                            test_prediction[imodel],
                            test_reference,
                            test_shifts,
                            label,
                            test_directory_model,
                            test_csv_file,
                            imodel=imodel)

            if test_save_npz:
                self.save_npz(
                    test_prediction,
                    test_reference,
                    test_shifts,
                    label,
                    test_directory,
                    test_npz_file)
                if model_ensemble:
                    for imodel in range(model_ensemble_num):
                        test_directory_model = os.path.join(
                            test_directory, f"{imodel:d}")
                        if not os.path.exists(test_directory_model):
                            os.makedirs(test_directory_model)
                        self.save_npz(
                            test_prediction[imodel],
                            test_reference,
                            test_shifts,
                            label,
                            test_directory_model,
                            test_npz_file,
                            imodel=imodel)

            ###########################
            # # # Plot Properties # # #
            ###########################

            # Check input for scaling per atom and prepare atom number scaling
            if utils.is_string(test_scale_per_atom):
                test_scale_per_atom = [test_scale_per_atom]
            test_property_atoms_scaling = {}
            for prop in test_properties:
                if prop in test_scale_per_atom:
                    test_property_atoms_scaling[prop] = (
                        1./np.array(
                            test_prediction['atoms_number'], dtype=float)
                        )
                else:
                    test_property_atoms_scaling[prop] = None

            # Plot correlation between model and reference properties
            if test_plot_correlation:
                for prop in test_properties:
                    self.plot_correlation(
                        label,
                        prop,
                        self.plain_data(test_prediction[prop]),
                        self.plain_data(test_reference[prop]),
                        self.data_units[prop],
                        metrics_test[prop],
                        test_property_atoms_scaling[prop],
                        test_directory,
                        test_plot_format,
                        test_plot_dpi)
                    if model_ensemble:
                        for imodel in range(model_ensemble_num):
                            test_directory_model = os.path.join(
                                test_directory, f"{imodel:d}")
                            if not os.path.exists(test_directory_model):
                                os.makedirs(test_directory_model)
                            self.plot_correlation(
                                label,
                                prop,
                                self.plain_data(test_prediction[imodel][prop]),
                                self.plain_data(test_reference[prop]),
                                self.data_units[prop],
                                metrics_test[prop][imodel],
                                test_property_atoms_scaling[prop],
                                test_directory_model,
                                test_plot_format,
                                test_plot_dpi)
                    if (
                        test_fragments is not None
                        and len(test_fragments) > 1
                        and prop in self._fragment_properties
                    ):
                        for fragment in test_fragments:
                            fragment_prop = f'{prop:s}_{fragment:d}'
                            fragment_selection = (
                                fragment == np.array(
                                    test_prediction['fragment_numbers'])
                                )
                            self.plot_correlation(
                                label,
                                fragment_prop,
                                self.plain_data([
                                    prediction
                                    for ii, prediction in enumerate(
                                        test_prediction[prop])
                                    if fragment_selection[ii]
                                    ]),
                                self.plain_data([
                                    reference
                                    for ii, reference in enumerate(
                                        test_reference[prop])
                                    if fragment_selection[ii]
                                    ]),
                                self.data_units[prop],
                                metrics_test[fragment_prop],
                                test_property_atoms_scaling[prop],
                                test_directory,
                                test_plot_format,
                                test_plot_dpi)
                            

            # Plot histogram of the prediction error
            if test_plot_histogram:
                for prop in test_properties:
                    self.plot_histogram(
                        label,
                        prop,
                        self.plain_data(test_prediction[prop]),
                        self.plain_data(test_reference[prop]),
                        self.data_units[prop],
                        metrics_test[prop],
                        test_directory,
                        test_plot_format,
                        test_plot_dpi)
                    if model_ensemble:
                        for imodel in range(model_ensemble_num):
                            test_directory_model = os.path.join(
                                test_directory, f"{imodel:d}")
                            if not os.path.exists(test_directory_model):
                                os.makedirs(test_directory_model)
                            self.plot_histogram(
                                label,
                                prop,
                                self.plain_data(test_prediction[imodel][prop]),
                                self.plain_data(test_reference[prop]),
                                self.data_units[prop],
                                metrics_test[prop][imodel],
                                test_directory_model,
                                test_plot_format,
                                test_plot_dpi)
                    if (
                        test_fragments is not None
                        and len(test_fragments) > 1
                        and prop in self._fragment_properties
                    ):
                        for fragment in test_fragments:
                            fragment_prop = f'{prop:s}_{fragment:d}'
                            fragment_selection = (
                                fragment == np.array(
                                    test_prediction['fragment_numbers'])
                                )
                            self.plot_histogram(
                                label,
                                fragment_prop,
                                self.plain_data([
                                    prediction
                                    for ii, prediction in enumerate(
                                        test_prediction[prop])
                                    if fragment_selection[ii]
                                    ]),
                                self.plain_data([
                                    reference
                                    for ii, reference in enumerate(
                                        test_reference[prop])
                                    if fragment_selection[ii]
                                    ]),
                                self.data_units[prop],
                                metrics_test[fragment_prop],
                                test_directory,
                                test_plot_format,
                                test_plot_dpi)

            # Plot histogram of the prediction error
            if test_plot_residual:
                for prop in test_properties:
                    self.plot_residual(
                        label,
                        prop,
                        self.plain_data(test_prediction[prop]),
                        self.plain_data(test_reference[prop]),
                        self.data_units[prop],
                        metrics_test[prop],
                        test_property_atoms_scaling[prop],
                        test_directory,
                        test_plot_format,
                        test_plot_dpi)
                    if model_ensemble:
                        for imodel in range(model_ensemble_num):
                            test_directory_model = os.path.join(
                                test_directory, f"{imodel:d}")
                            if not os.path.exists(test_directory_model):
                                os.makedirs(test_directory_model)
                            self.plot_residual(
                                label,
                                prop,
                                self.plain_data(test_prediction[imodel][prop]),
                                self.plain_data(test_reference[prop]),
                                self.data_units[prop],
                                metrics_test[prop][imodel],
                                test_property_atoms_scaling[prop],
                                test_directory_model,
                                test_plot_format,
                                test_plot_dpi)
                    if (
                        test_fragments is not None
                        and len(test_fragments) > 1
                        and prop in self._fragment_properties
                    ):
                        for fragment in test_fragments:
                            fragment_prop = f'{prop:s}_{fragment:d}'
                            fragment_selection = (
                                fragment == np.array(
                                    test_prediction['fragment_numbers'])
                                )
                            self.plot_residual(
                                label,
                                fragment_prop,
                                self.plain_data([
                                    prediction
                                    for ii, prediction in enumerate(
                                        test_prediction[prop])
                                    if fragment_selection[ii]
                                    ]),
                                self.plain_data([
                                    reference
                                    for ii, reference in enumerate(
                                        test_reference[prop])
                                    if fragment_selection[ii]
                                    ]),
                                self.data_units[prop],
                                metrics_test[fragment_prop],
                                test_property_atoms_scaling[prop],
                                test_directory,
                                test_plot_format,
                                test_plot_dpi)

            #########################
            # # # Plot Outliers # # #
            #########################

            # If requested, detect and plot outlier systems
            if test_outlier:

                # Check outlier metric
                outlier_metric = test_outlier_metric.strip().lower()
                if not outlier_metric in ['mae', 'mse', 'rmse']:
                    raise SyntaxError(
                        f"Metric label '{test_outlier_metric:s}' is unkown!\n"
                        + "Choose between 'mae', 'mse' or 'rmse'."
                    )

                # Sort system metrics
                outlier_sys_property = f"sys_{test_outlier_property:s}"
                if outlier_metric == 'rmse':
                    sys_prop_metric = torch.sqrt(
                        metrics_test[outlier_sys_property]['mse']
                    )
                    sys_prop_metric_mean = torch.sqrt(
                        metrics_test[test_outlier_property]['mse']
                    )
                else:
                    sys_prop_metric = (
                        metrics_test[outlier_sys_property][outlier_metric]
                    )
                    sys_prop_metric_mean = (
                        metrics_test[test_outlier_property][outlier_metric]
                    )
                sys_prop_metric_sorted, sys_prop_metric_sorted_idx = (
                    torch.sort(sys_prop_metric)
                )
                sys_prop_metric_sorted = sys_prop_metric_sorted[
                    torch.logical_not(torch.isnan(sys_prop_metric))].flip(0)
                sys_prop_metric_sorted_idx = sys_prop_metric_sorted_idx[
                    torch.logical_not(torch.isnan(sys_prop_metric))].flip(0)

                # Either get the nth outlier with largest metric
                if test_outlier_threshold is None:
                    num_outlier = test_outlier_num_max
                # Or get the outlier with metric larger than n times the mean
                else:
                    num_outlier = torch.sum(
                        sys_prop_metric_sorted
                        >= test_outlier_threshold*sys_prop_metric_mean
                    )
                sys_prop_metric_sorted = (
                    sys_prop_metric_sorted[:num_outlier]
                )
                sys_prop_metric_sorted_idx = (
                    sys_prop_metric_sorted_idx[:num_outlier]
                )

                # Get auxiliary parameter
                Nsys = test_prediction['mlmm_atoms_number'].shape[0]
                Natoms = np.sum(test_prediction['mlmm_atoms_number'])

                # Iterate over outlier
                for ii, outlier_idx in enumerate(sys_prop_metric_sorted_idx):

                    # Prepare directory for outlier plots
                    outlier_row_id = metrics_test['row_ids'][outlier_idx]
                    outlier_directory = os.path.join(
                        test_directory,
                        (
                            f'outlier_{ii + 1:0{len(str(num_outlier)):d}d}_'
                            + f'{test_outlier_property:s}_'
                            + f'row_id_{outlier_row_id:d}'
                        )
                    )
                    if not os.path.isdir(outlier_directory):
                        os.makedirs(outlier_directory)

                    # Write system to file
                    datasubset.dataset.write_xyz(
                        os.path.join(
                            outlier_directory,
                            f'outlier_row_id_{outlier_row_id:d}.xyz'
                        ),
                        (outlier_row_id - 1),
                        global_idx=True,
                    )

                    # Iterate over test properties
                    for prop in test_properties:
                    
                        # Get outlier selection mask
                        data_shape = test_prediction[prop].shape
                        if data_shape[0] == Nsys:
                            outlier_selection = outlier_idx
                        elif data_shape[0] == Natoms:
                            outlier_selection = np.concatenate(
                                (
                                    np.zeros(1, dtype=int),
                                    np.cumsum(
                                        test_prediction['mlmm_atoms_number']
                                    )
                                )
                            )
                            outlier_selection = np.arange(
                                outlier_selection[outlier_idx],
                                outlier_selection[outlier_idx + 1]
                            )
                        outlier_prediction = (
                            test_prediction[prop][outlier_selection]
                        )
                        outlier_reference = (
                            test_reference[prop][outlier_selection]
                        )
                    
                        # Plot correlation with highlighted outliers
                        prop_sys = f"sys_{prop:s}"
                        outlier_metrics = {
                            'mae': metrics_test[prop_sys]['mae'][outlier_idx],
                            'mse': metrics_test[prop_sys]['mse'][outlier_idx],
                        }
                        if test_property_atoms_scaling[prop] is None:
                            outlier_scaling = None
                        else:
                            outlier_scaling = (
                                test_property_atoms_scaling[prop][outlier_idx]
                            )
                        self.plot_correlation(
                            label,
                            prop,
                            self.plain_data(test_prediction[prop]),
                            self.plain_data(test_reference[prop]),
                            self.data_units[prop],
                            metrics_test[prop],
                            test_property_atoms_scaling[prop],
                            outlier_directory,
                            test_plot_format,
                            test_plot_dpi,
                            outlier_data=[
                                self.plain_data(outlier_prediction),
                                self.plain_data(outlier_reference),
                            ],
                            outlier_metrics=outlier_metrics,
                            outlier_row_id=outlier_row_id,
                            outlier_atoms_scaling=outlier_scaling,
                        )
                        if (
                            test_fragments is not None
                            and len(test_fragments) > 1
                            and prop in self._fragment_properties
                        ):
                            for fragment in test_fragments:
                                fragment_prop = f'{prop:s}_{fragment:d}'
                                frag_prop_sys = f'sys_{fragment_prop:s}'
                                fragment_selection = (
                                    fragment == np.array(
                                        test_prediction['fragment_numbers'])
                                    )
                                outlier_fragment_selection = (
                                    fragment_selection[outlier_selection]
                                )
                                outlier_fragment_metrics = {
                                    'mae': (
                                        metrics_test[
                                            frag_prop_sys]['mae'][outlier_idx]
                                    ),
                                    'mse': (
                                        metrics_test[
                                            frag_prop_sys]['mse'][outlier_idx]
                                    ),
                                }
                                self.plot_correlation(
                                    label,
                                    fragment_prop,
                                    self.plain_data([
                                        prediction
                                        for ii, prediction in enumerate(
                                            test_prediction[prop])
                                        if fragment_selection[ii]
                                        ]),
                                    self.plain_data([
                                        reference
                                        for ii, reference in enumerate(
                                            test_reference[prop])
                                        if fragment_selection[ii]
                                        ]),
                                    self.data_units[prop],
                                    metrics_test[fragment_prop],
                                    test_property_atoms_scaling[prop],
                                    outlier_directory,
                                    test_plot_format,
                                    test_plot_dpi,
                                    outlier_data=[
                                        self.plain_data(
                                            outlier_prediction[
                                                outlier_fragment_selection]
                                        ),
                                        self.plain_data(
                                            outlier_reference[
                                                outlier_fragment_selection]
                                        ),
                                    ],
                                    outlier_metrics=outlier_fragment_metrics,
                                    outlier_row_id=outlier_row_id,
                                    outlier_atoms_scaling=outlier_scaling,
                                )

        return

    @staticmethod
    def is_imported(
        module: str
    ) -> bool:
        """
        Check if a module is imported.
        
        Parameters
        ----------
        module: str
            Module name
            
        Returns
        -------
        bool
            Module availability flag on the system

        """

        return module in sys.modules

    def save_npz(
        self,
        prediction: Dict[str, np.ndarray],
        reference: Dict[str, np.ndarray],
        shifts: Dict[str, np.ndarray],
        label: str,
        test_directory: str,
        npz_file: str,
        imodel: Optional[int] = None,
    ):
        """
        Save results of the test set to a binary npz file.

        Parameters
        ----------
        prediction: dict
            Dictionary of the property predictions to save.
        prediction: dict
            Dictionary of the reference property values to save.
        shifts: dict
            Dictionary of the reference property shifts.
        label: str
            Dataset label for the npz file prefix.
        test_directory: str
            Directory to save the npz file.
        npz_file: str
            Name tag of the npz file.
        imodel: int, optional, default None
            Model index for the respective model in the model ensemble.

        """

        # Check for .npz file extension
        if 'npz' != npz_file.split('.')[-1]:
            npz_file += '.npz'

        # Iterate over properties
        for prop, pred in prediction.items():
            
            # Check property in reference data, if not skip
            if not prop in reference:
                continue

            # Prepare npz file name
            npz_file_prop = os.path.join(
                test_directory, f"{label:s}_{prop:s}_{npz_file:s}")

            # Store data in npz format
            if prop in shifts:
                np.savez(
                    npz_file_prop,
                    prediction=pred,
                    reference=reference[prop],
                    shift=shifts[prop],
                )
            else:
                np.savez(
                    npz_file_prop,
                    prediction=pred,
                    reference=reference[prop],
                )

            # Print info
            if imodel is None:
                addition = ""
            else:
                addition = f" of model {imodel:d}"
            self.logger.info(
                f"Prediction results{addition:s} and reference data for the "
                + f"dataset '{label:s}' and property '{prop:s}' are saved in:"
                + f"\n'{npz_file_prop:s}'.")

        return

    def save_csv(
        self,
        prediction: Dict[str, np.ndarray],
        reference: Dict[str, np.ndarray],
        shifts: Dict[str, np.ndarray],
        label: str,
        test_directory: str,
        csv_file: str,
        imodel: Optional[int] = None,
    ):
        """
        Save results of the data set to a csv file.

        Parameters
        ----------
        prediction: dict
            Dictionary of the property predictions to save.
        prediction: dict
            Dictionary of the reference property values to save.
        shifts: dict
            Dictionary of the reference property shifts.
        label: str
            Dataset label for the csv file prefix.
        test_directory: str
            Directory to save the csv file.
        csv_file: str
            Name tag of the csv file.
        imodel: int, optional, default None
            Model index for the respective model in the model ensemble.

        """

        # Check for .csv file extension
        if 'csv' != csv_file.split('.')[-1]:
            csv_file += '.csv'

        # Iterate over properties
        for prop, pred in prediction.items():
            
            # Check property in reference data, if not skip
            if not prop in reference:
                continue
            
            # Prepare csv file name
            csv_file_prop = os.path.join(
                test_directory, f"{label:s}_{prop:s}_{csv_file:s}")

            # Prepare data
            if prop in shifts:
                results_np = np.column_stack((
                    np.array(pred).reshape(-1),
                    np.array(reference[prop]).reshape(-1),
                    np.array(shifts[prop]).reshape(-1))
                )
                columns_np=[
                    f"{prop:s} prediction", " reference", " shift"]
            else:
                results_np = np.column_stack((
                    np.array(pred).reshape(-1),
                    np.array(reference[prop]).reshape(-1))
                )
                columns_np=[
                    f"{prop:s} prediction", " reference"]
            
            # Store data in csv format generated via the pandas data frame
            if self.is_imported("pandas"):
                results = pd.DataFrame(
                    results_np,
                    columns=columns_np
                    )
                results.to_csv(csv_file_prop, index=False)
            else:
                self.logger.warning(
                    "Module 'pandas' is not available. "
                    + "Test properties are not written to a csv file!")

            # Print info
            if imodel is None:
                addition = ""
            else:
                addition = f" of model {imodel:d}"
            self.logger.info(
                f"Prediction results{addition:s} and reference data for the "
                + f"dataset '{label:s}' and property '{prop:s}' are saved in:"
                + f"\n'{csv_file_prop:s}'.")

        return

    def reset_metrics(
        self,
        test_properties: List[str],
        model_ensemble: bool,
        model_ensemble_num: int,
        test_fragments: List[int] = None,
        test_outlier: bool = False,
    ) -> Dict[str, float]:
        """
        Reset the metrics dictionary.

        Parameters
        ----------
        test_properties: list(str)
            List of properties to restart
        model_ensemble: bool
            Model calculator or model ensemble flag
        model_ensemble_num: int
            Model ensemble calculator number
        test_fragments: list(int)
            List of occurring fragment indices in the dataset systems.
            If None or just one, properties metrics are not separately 
            evaluated for each fragment.
            If two or more indices, possible properties are evaluated for each
            fragment also individually.
        test_outlier: bool, optional, default False
            If True, prepare for metrics per system.
            System specific property metrics are stored as 'sys_{property}'.

        Returns
        -------
        dict(str, dict(str, float))
            Property metrics dictionary

        """

        # Initialize metrics dictionary
        metrics = {}

        # Add data counter and, eventually, system information
        metrics['Ndata'] = 0
        if test_outlier:
            metrics['row_ids'] = torch.tensor(
                [],
                device='cpu',
                dtype=torch.int64,
            )

        # Add property metrics
        for prop in test_properties:
            
            metrics[prop] = {
                'mae': 0.0,
                'mse': 0.0,
            }
            if test_outlier:
                prop_sys = f"sys_{prop:s}"
                metrics[prop_sys] = {
                    'mae': torch.tensor([], device='cpu', dtype=self.dtype),
                    'mse': torch.tensor([], device='cpu', dtype=self.dtype),
                }
            
            # For model ensemble, reset individual model metrics
            if model_ensemble:
                metrics[prop]['std'] = 0.0
                for imodel in range(model_ensemble_num):
                    metrics[prop][imodel] = {
                        'mae': 0.0,
                        'mse': 0.0,
                    }

            # If multiple fragments are available, reset metrics for each
            # fragment system but just for suited properties
            if (
                test_fragments is not None
                and len(test_fragments) > 1
                and prop in self._fragment_properties
            ):

                # Iterate over fragment indices
                for fragment in test_fragments:
                    
                    fragment_prop = f'{prop:s}_{fragment:d}'
                    metrics[fragment_prop] = {
                        'mae': 0.0,
                        'mse': 0.0,
                    }
                    if test_outlier:
                        frag_prop_sys = f'sys_{fragment_prop:s}'
                        metrics[frag_prop_sys] = {
                            'mae': torch.tensor(
                                [],
                                device='cpu',
                                dtype=self.dtype
                            ),
                            'mse': torch.tensor(
                                [],
                                device='cpu',
                                dtype=self.dtype
                            ),
                        }

        return metrics

    def compute_metrics(
        self,
        prediction: Dict[str, Any],
        reference: Dict[str, Any],
        test_properties: List[str],
        test_conversion: Dict[str, float],
        model_ensemble: bool,
        model_ensemble_num: int,
        test_fragments: List[int] = None,
        test_outlier: bool = False
    ) -> Dict[str, float]:
        """
        Compute the metrics mean absolute error (MAE) and mean squared error
        (MSE) for the test set.

        Parameters
        ----------
        prediction: dict
            Dictionary of the model predictions
        reference: dict
            Dictionary of the reference data
        test_properties: list(str)
            List of properties to evaluate.
        test_conversion: dict(str, float)
            Model prediction to test data unit conversion.
        model_ensemble: bool
            Model calculator or model ensemble flag
        model_ensemble_num: int
            Model ensemble calculator number
        test_fragments: list(int)
            List of occurring fragment indices in the dataset systems.
            If None or just one, properties metrics are not separately 
            evaluated for each fragment.
            If two or more indices, possible properties are evaluated for each
            fragment also individually.
        test_outlier: bool, optional, default False
            If True, additionally compute metrics per system.
            System specific property metrics are stored as 'sys_{property}'.

        Returns
        -------
        dict(str, dict(str, float))
            Property metrics dictionary

        """

        # Initialize metrics dictionary
        metrics = {}

        # Add system information
        metrics['Ndata'] = reference[test_properties[0]].size()[0]
        if test_outlier:
            metrics['row_ids'] = prediction['row_ids']
            Nsys = prediction['atoms_number'].size()[0]
            if 'ml_idx' in prediction:
                Natoms = prediction['mlmm_sys_i'].size()[0]
                sys_i = prediction['mlmm_sys_i']
                atoms_number = prediction['mlmm_atoms_number']
            else:
                Natoms = prediction['sys_i'].size()[0]
                sys_i = prediction['sys_i']
                atoms_number = prediction['atoms_number']

        # Prepare metic functions
        if test_outlier:
            mae_fn = torch.nn.L1Loss(reduction='none')
            mse_fn = torch.nn.MSELoss(reduction='none')
        else:
            mae_fn = torch.nn.L1Loss(reduction='mean')
            mse_fn = torch.nn.MSELoss(reduction='mean')

        # Iterate over metric type
        for metric_fn, metric in zip([mae_fn, mse_fn], ['mae', 'mse']):

            # Iterate over test properties
            for ip, prop in enumerate(test_properties):

                # Initialize single property metrics dictionary
                if prop not in metrics:
                    metrics[prop] = {}

                # Compute MAE and MSE
                if test_outlier:

                    # Initialize system resolved property metrics dictionary
                    prop_sys = f"sys_{prop:s}"
                    if prop_sys not in metrics:
                        metrics[prop_sys] = {}

                    # Compute element-wise property metrics
                    metrics[prop_sys][metric] = metric_fn(
                        torch.flatten(prediction[prop])
                        * test_conversion[prop],
                        torch.flatten(reference[prop])
                    ).detach()

                    # Compute mean property metrics
                    metrics[prop][metric] = torch.mean(
                        metrics[prop_sys][metric]
                    )

                    # Reduce atom-wise property metric to system-wise metric
                    if prediction[prop].shape[0] == Natoms:
                        metrics[prop_sys][metric] = torch.zeros(
                            Nsys,
                            device=prediction[prop].device,
                            dtype=prediction[prop].dtype,
                            ).scatter_add_(
                                0, sys_i,
                                metrics[prop_sys][metric]
                            )/atoms_number
                    elif (
                        prediction[prop].shape[0] == Nsys
                        and prediction[prop].dim() > 1
                    ):
                        metrics[prop_sys][metric] = torch.zeros(
                            Nsys,
                            device=prediction[prop].device,
                            dtype=prediction[prop].dtype,
                            ).scatter_add_(
                                0, 
                                torch.arange(
                                    Nsys, device='cpu', dtype=torch.int64
                                    ).repeat(prediction[prop].shape[1:].numel()
                                ),
                                metrics[prop_sys][metric]
                            )/prediction[prop].shape[1:].numel()

                else:

                    # Compute mean property metrics
                    metrics[prop][metric] = metric_fn(
                        torch.flatten(prediction[prop])
                        * test_conversion[prop],
                        torch.flatten(reference[prop])
                    ).detach()
                
                # For model ensembles
                if model_ensemble:
                    
                    # Compute mean standard deviation
                    prop_std = f"std_{prop:s}"
                    metrics[prop]['std'] = torch.mean(prediction[prop_std])
                    
                    # Compute single calculator statistics
                    for imodel in range(model_ensemble_num):

                        # Initialize model resolved property metrics dictionary
                        if imodel not in metrics[prop]:
                            metrics[prop][imodel] = {}
                        
                        # Compute mean property metrics
                        metrics[prop][imodel][metric] = metric_fn(
                            torch.flatten(prediction[imodel][prop])
                            * test_conversion[prop],
                            torch.flatten(reference[prop]))

                # If multiple fragments are available, compute metrics for each
                # fragment system but just for suited properties
                if (
                    test_fragments is not None
                    and len(test_fragments) > 1
                    and prop in self._fragment_properties
                ):
                    
                    # Iterate over fragment indices
                    for fragment in test_fragments:
                        
                        # Fragment property tag
                        fragment_prop = f'{prop:s}_{fragment:d}'

                        # Initialize single property metrics dictionary
                        if fragment_prop not in metrics:
                            metrics[fragment_prop] = {}

                        # Fragment selection
                        fragment_selection = (
                            fragment == prediction['fragment_numbers'])

                        # Compute MAE and MSE
                        if test_outlier:

                            # Initialize system resolved property metrics 
                            # dictionary
                            frag_prop_sys = f'sys_{fragment_prop:s}'
                            if frag_prop_sys not in metrics:
                                metrics[frag_prop_sys] = {}

                            # Compute element-wise property metrics
                            metrics[frag_prop_sys][metric] = metric_fn(
                                torch.flatten(
                                    prediction[prop][fragment_selection]
                                )*test_conversion[prop],
                                torch.flatten(
                                    reference[prop][fragment_selection])
                            )

                            # Compute mean property metrics
                            metrics[fragment_prop][metric] = torch.mean(
                                metrics[frag_prop_sys][metric])

                            # Get fragments number of atoms
                            fragment_atoms_number = torch.bincount(
                                sys_i[fragment_selection],
                                minlength=Nsys,
                            )

                            # Reduce atom-wise property metric to system-wise
                            # metric
                            if prediction[prop].shape[0] == Natoms:
                                metrics[frag_prop_sys][metric] = (
                                    torch.zeros(
                                        Nsys,
                                        device=prediction[prop].device,
                                        dtype=prediction[prop].dtype,
                                    ).scatter_add_(
                                        0, sys_i[fragment_selection],
                                        metrics[frag_prop_sys][metric]
                                    )
                                )
                                metrics[frag_prop_sys][metric] = (
                                    torch.where(
                                        fragment_atoms_number.to(
                                            dtype=torch.bool
                                        ),
                                        (
                                            metrics[frag_prop_sys][metric]
                                            / fragment_atoms_number
                                        ),
                                        metrics[frag_prop_sys][metric]
                                    ).detach()
                                )
                            elif (
                                prediction[prop].shape[0] == Nsys
                                and prediction[prop].dim() > 1
                            ):
                                metrics[frag_prop_sys][metric] = (
                                    torch.zeros(
                                        Nsys,
                                        device=prediction[prop].device,
                                        dtype=prediction[prop].dtype,
                                    ).scatter_add_(
                                        0, 
                                        torch.arange(
                                            Nsys,
                                            device='cpu',
                                            dtype=torch.int64
                                        ).repeat(
                                            prediction[prop].shape[1:].numel()
                                        ),
                                        metrics[frag_prop_sys][metric]
                                    )/prediction[prop].shape[1:].numel()
                                )

                        else:
                            
                            # Compute mean property metrics                            
                            metrics[fragment_prop][metric] = metric_fn(
                                torch.flatten(
                                    prediction[prop][fragment_selection]
                                )*test_conversion[prop],
                                torch.flatten(
                                    reference[prop][fragment_selection]
                                )
                            )

        return metrics

    def update_metrics(
        self,
        metrics: Dict[str, float],
        metrics_update: Dict[str, float],
        test_properties: List[str],
        model_ensemble: bool,
        model_ensemble_num: int,
        test_fragments: List[int] = None,
        test_outlier: bool = False,
    ) -> Dict[str, float]:
        """
        Update the metrics dictionary.

        Parameters
        ----------
        metrics: dict
            Dictionary of the metrics which gets updated
        metrics_update: dict
            Dictionary of the new metrics for the update
        test_properties: list(str)
            List of properties to evaluate
        model_ensemble: bool
            Model calculator or model ensemble flag
        model_ensemble_num: int
            Model ensemble calculator number
        test_fragments: list(int)
            List of occurring fragment indices in the dataset systems.
            If None or just one, properties metrics are not separately 
            evaluated for each fragment.
            If two or more indices, possible properties are evaluated for each
            fragment also individually.
        test_outlier: bool, optional, default False
            If True, additionally update metrics per system.
            System specific property metrics are stored as 'sys_{property}'.

        Returns
        -------
        dict(str, dict(str, float))
            Property metrics dictionary

        """

        # Get data sizes and metric ratio
        Ndata = metrics['Ndata']
        Ndata_update = metrics_update['Ndata']
        fdata = float(Ndata)/float((Ndata + Ndata_update))
        fdata_update = 1. - fdata

        # Update system information
        metrics['Ndata'] = metrics['Ndata'] + metrics_update['Ndata']
        if test_outlier:
            metrics['row_ids'] = torch.cat(
                (metrics['row_ids'], metrics_update['row_ids']),
                dim=0
            )

        # Update metrics
        for prop in test_properties:

            # Skip if metric is not available
            # print(prop, metrics[prop]['mae'])
            # print(prop, metrics[prop]['mse'])
            # if torch.any(
            #     [
            #         torch.isnan(metrics[prop][metric])
            #         for metric in ['mae', 'mse']
            #     ]
            # ):
            #     continue

            for metric in ['mae', 'mse']:
                metrics[prop][metric] = (
                    fdata*metrics[prop][metric]
                    + fdata_update*metrics_update[prop][metric].detach().item()
                )
                if test_outlier:
                    prop_sys = f"sys_{prop:s}"
                    metrics[prop_sys][metric] = torch.cat(
                        (
                            metrics[prop_sys][metric],
                            metrics_update[prop_sys][metric]
                        ),
                        dim=0
                    )
            
            if model_ensemble:
                metrics[prop]['std'] = (
                    fdata*metrics[prop]['std']
                    + fdata_update*metrics_update[prop]['std'].detach().item()
                )
                for imodel in range(model_ensemble_num):
                    for metric in ['mae', 'mse']:
                        metrics[prop][imodel][metric] = (
                            fdata*metrics[prop][imodel][metric]
                            + fdata_update
                            * (metrics_update
                               )[prop][imodel][metric].detach().item()
                        )

            if (
                test_fragments is not None
                and len(test_fragments) > 1
                and prop in self._fragment_properties
            ):
                for fragment in test_fragments:
                    fragment_prop = f'{prop:s}_{fragment:d}'
                    for metric in ['mae', 'mse']:
                        if not np.isnan(metrics_update[fragment_prop][metric]):
                            metrics[fragment_prop][metric] = (
                                fdata*metrics[fragment_prop][metric]
                                + fdata_update*metrics_update[
                                    fragment_prop][metric].detach().item()
                            )
                        if test_outlier:
                            frag_prop_sys = f'sys_{fragment_prop:s}'
                            metrics[frag_prop_sys][metric] = torch.cat(
                                (
                                    metrics[frag_prop_sys][metric],
                                    metrics_update[frag_prop_sys][metric]
                                ),
                                dim=0
                            )
                            

        return metrics

    def print_metric(
        self,
        metrics: Dict[str, float],
        test_properties: List[str],
        test_label: str,
        model_ensemble: bool,
        model_ensemble_num: int,
        test_fragments: List[int] = None,
    ):
        """
        Print the values of MAE and RMSE for the test set.

        Parameters
        ----------
        metrics: dict
            Dictionary of the property metrics
        test_properties: list(str)
            List of properties to evaluate
        test_label: str
            Label of the reference data set
        model_ensemble: bool
            Model calculator or model ensemble flag
        model_ensemble_num: int
            Model ensemble calculator number
        test_fragments: list(int)
            List of occuring fragment indices in the dataset systems.
            If two or more indices, properties are shown for each
            fragment if possible.

        """

        # Prepare label input
        if len(test_label):
            msg_label = f" for {test_label:s} set"
        else:
            msg_label = ""

        # Prepare header
        message = f"Summary{msg_label:s}:\n"
        message += f"   {'Property Metrics':<18s} "
        message += f"{'MAE':<8s}   "
        message += f"{'RMSE':<8s}"
        if model_ensemble:
            message += f"   {'Std':<8s}\n"
        else:
            message += "\n"
        
        # Add property metrics
        for prop in test_properties:
            
            # Mean metrics
            message += f"   {prop:<18s} "
            message += f"{metrics[prop]['mae']:3.2e},  "
            message += f"{np.sqrt(metrics[prop]['mse']):3.2e}"
            if model_ensemble:
                message += f",  {metrics[prop]['std']:3.2e}"
            message += f" {self.data_units[prop]:s}\n"
            
            # For model ensemble, single model metrics
            if model_ensemble:
                for imodel in range(model_ensemble_num):
                    if imodel:
                        message += f"     {f'      {imodel:d}':<16s}  "
                    else:
                        message += f"     {f'Model {imodel:d}':<16s}  "
                    message += f"{metrics[prop][imodel]['mae']:3.2e},  "
                    message += f"{np.sqrt(metrics[prop][imodel]['mse']):3.2e}"
                    message += f" {self.data_units[prop]:s}\n"

            # For multiple fragments and possible property
            if (
                test_fragments is not None
                and len(test_fragments) > 1
                and prop in self._fragment_properties
            ):
                for fragment in test_fragments:
                    fragment_prop = f'{prop:s}_{fragment:d}'
                    message += f"     {f'Fragment {fragment:d}':<16s}  "
                    message += f"{metrics[fragment_prop]['mae']:3.2e},  "
                    message += f"{np.sqrt(metrics[fragment_prop]['mse']):3.2e}"
                    message += f" {self.data_units[prop]:s}\n"

        # Print metrics
        self.logger.info(message)

        return

    def plain_data(
        self,
        data_nd: List[Any],
    ) -> List[Any]:

        return np.array([
            data_i
            for data_sys in data_nd
            for data_i in np.array(data_sys).reshape(-1)])

    def plot_correlation(
        self,
        label_dataset: str,
        label_property: str,
        data_prediction: List[float],
        data_reference: List[float],
        unit_property: str,
        data_metrics: Dict[str, float],
        test_atoms_scaling: List[float],
        test_directory: str,
        test_plot_format: str,
        test_plot_dpi: int,
        outlier_data: Optional[List[float]] = None,
        outlier_metrics: Optional[Dict[str, float]] = None,
        outlier_row_id: Optional[int] = None,
        outlier_atoms_scaling: Optional[float] = None,
    ):
        """
        Plot property data correlation data.
        (x-axis: reference data; y-axis: predicted data)

        Some pre-defined plot properties are:
        figsize = (6, 6)
        fontsize = 12

        Parameters
        ----------
        label_dataset: str
            Label of the data set
        label_property: str
            Label of the property
        data_prediction: list(float)
            List of the predicted data
        data_reference: list(float)
            List of the reference data
        unit_property: str
            Unit of the property
        data_metrics: dict
            Dictionary of the metrics
        test_atoms_scaling: list(float)
            List of the scaling factors
        test_directory: str
            Directory to save the plot
        test_plot_format: str
            Format of the plot
        test_plot_dpi: int
            DPI of the plot
        outlier_data: list(float), optional, default None
            List of the predicted and reference data of the outlier
        outlier_metrics: dict, optional, default None
            Dictionary of the outlier metrics
        outlier_row_id: int, optional, default None
            Database row id of the outlier
        outlier_atoms_scaling: float, optional, default None
            Outlier data scaling factor

        """

        # Plot property: Fontsize
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE, weight='bold')
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('axes', titlesize=MEDIUM_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

        # Plot property: Figure size and arrangement
        figsize = (6, 6)
        sfig = float(figsize[0])/float(figsize[1])
        left = 0.20
        bottom = 0.15
        column = [0.75, 0.00]
        row = [column[0]*sfig]

        # Initialize figure
        fig = plt.figure(figsize=figsize)
        axs1 = fig.add_axes(
            [left + 0.*np.sum(column), bottom, column[0], row[0]])

        # Data label
        metrics_label = (
            f"{label_property:s} ({label_dataset:s})\n"
            + f"MAE = {data_metrics['mae']:3.2e} {unit_property:s}\n"
            + f"RMSE = {np.sqrt(data_metrics['mse']):3.2e} {unit_property:s}")
        if self.is_imported("scipy"):
            r2 = stats.pearsonr(data_prediction, data_reference).statistic
            metrics_label += (
                "\n" + r"1 - $R^2$ = " + f"{1.0 - r2:3.2e}")
        if 'std' in data_metrics:
            metrics_label += (
                f"\nStd = {data_metrics['std']:3.2e} {unit_property:s}")

        # If requested, outlier data label
        if outlier_data is not None:
            outlier_metrics_label = (
                f"row_id: {outlier_row_id}\nMAE = "
                + f"{outlier_metrics['mae']:3.2e} {unit_property:s}\n"
                + f"RMSE = {np.sqrt(outlier_metrics['mse']):3.2e} "
                + f"{unit_property:s}"
            )

        # Scale data if requested
        if test_atoms_scaling is not None:
            data_prediction = data_prediction*test_atoms_scaling
            data_reference = data_reference*test_atoms_scaling
            scale_label = "/atom"
        else:
            scale_label = ""
        if outlier_data is not None and outlier_atoms_scaling is not None:
            outlier_data = [
                data_i*outlier_atoms_scaling for data_i in outlier_data
            ]

        # Plot data
        data_min = np.min(
            (np.nanmin(data_reference), np.nanmin(data_prediction)))
        data_max = np.max(
            (np.nanmax(data_reference), np.nanmax(data_prediction)))
        data_dif = data_max - data_min
        axs1.plot(
            [data_min - data_dif*0.05, data_max + data_dif*0.05],
            [data_min - data_dif*0.05, data_max + data_dif*0.05],
            color='black',
            marker='None', linestyle='--')
        axs1.plot(
            data_reference,
            data_prediction,
            color='blue', markerfacecolor='None',
            marker='o', linestyle='None',
            label=metrics_label)

        # If requested, plot outlier data
        if outlier_data is not None:
            axs1.plot(
                outlier_data[1],
                outlier_data[0],
                color='red', markerfacecolor='None',
                marker='o', linestyle='None',
                label=outlier_metrics_label)

        # Axis range
        axs1.set_xlim(data_min - data_dif*0.05, data_max + data_dif*0.05)
        axs1.set_ylim(data_min - data_dif*0.05, data_max + data_dif*0.05)

        # Figure title
        title = f"Correlation plot\n{label_property:s} ({label_dataset:s})"
        if 'std' in data_metrics:
            title = title[:-1] + ", ensemble average)"
        axs1.set_title(title, fontweight='bold')

        # Axis labels
        axs1.set_xlabel(
            f"Reference {label_property:s} ({unit_property:s}{scale_label:s})",
            fontweight='bold')
        axs1.get_xaxis().set_label_coords(0.5, -0.12)
        axs1.set_ylabel(
            f"Model {label_property:s} ({unit_property:s}{scale_label:s})",
            fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.18, 0.5)

        # Figure legend
        axs1.legend(loc='upper left')

        # Save figure
        plt.savefig(
            os.path.join(
                test_directory,
                f"{label_dataset:s}_correlation_{label_property:s}"
                + f".{test_plot_format:s}"),
            format=test_plot_format,
            dpi=test_plot_dpi)
        plt.close()

        return

    def plot_histogram(
        self,
        label_dataset: str,
        label_property: str,
        data_prediction: List[float],
        data_reference: List[float],
        unit_property: str,
        data_metrics: Dict[str, float],
        test_directory: str,
        test_plot_format: str,
        test_plot_dpi: int,
        test_binnum: Optional[int] = 101,
        test_histlog: Optional[bool] = False,
    ):
        """
        Plot prediction error spread as histogram.

        Parameters
        ----------
        label_dataset: str
            Label of the data set
        label_property: str
            Label of the property
        data_prediction: list(float)
            List of the predicted data
        data_reference: list(float)
            List of the reference data
        unit_property: str
            Unit of the property
        data_metrics: dict
            Dictionary of the metrics
        test_directory: str
            Directory to save the plot
        test_plot_format: str
            Format of the plot
        test_plot_dpi: int
            DPI of the plot

        """

        # Plot property: Fontsize
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE, weight='bold')
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('axes', titlesize=MEDIUM_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

        # Plot property: Figure size and arrangement
        figsize = (6, 6)
        sfig = float(figsize[0])/float(figsize[1])
        left = 0.20
        bottom = 0.15
        column = [0.75, 0.00]
        row = [column[0]*sfig]

        # Initialize figure
        fig = plt.figure(figsize=figsize)
        axs1 = fig.add_axes(
            [left + 0.*np.sum(column), bottom, column[0], row[0]])

        # Data label
        metrics_label = (
            f"{label_property:s} ({label_dataset:s})\n"
            + f"MAE = {data_metrics['mae']:3.2e} {unit_property:s}\n"
            + f"RMSE = {np.sqrt(data_metrics['mse']):3.2e} {unit_property:s}")
        if self.is_imported("scipy"):
            r2 = stats.pearsonr(data_prediction, data_reference).statistic
            metrics_label += (
                "\n" + r"1 - $R^2$ = " + f"{1.0 - r2:3.2e}")
        if 'std' in data_metrics:
            metrics_label += (
                f"\nStd = {data_metrics['std']:3.2e} {unit_property:s}")

        # Plot data
        data_dif = data_reference - data_prediction
        data_min = np.nanmin(data_dif)
        data_max = np.nanmax(data_dif)
        data_absmax = np.max((np.abs(data_min), np.abs(data_max)))
        data_absmax += data_absmax/(2.0*test_binnum)
        data_bin = np.linspace(-data_absmax, data_absmax, num=test_binnum)
        axs1.hist(
            data_reference - data_prediction,
            bins=data_bin,
            density=True,
            color='red',
            log=test_histlog,
            label=metrics_label)

        # Axis range
        axs1.set_xlim(-data_absmax, data_absmax)

        # Figure title
        title = (
            "Prediction error distribution\n"
            + f"{label_property:s} ({label_dataset:s})")
        if 'std' in data_metrics:
            title = title[:-1] + ", ensemble average)"
        axs1.set_title(title, fontweight='bold')

        # Axis labels
        axs1.set_xlabel(
            f"Error in {label_property:s} ({unit_property:s})",
            fontweight='bold')
        axs1.get_xaxis().set_label_coords(0.5, -0.12)
        if test_histlog:
            ylabel = "log(Probability)"
        else:
            ylabel = "Probability"
        axs1.set_ylabel(
            ylabel,
            fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.18, 0.5)

        # Figure legend
        axs1.legend(loc='upper left')

        # Save figure
        plt.savefig(
            os.path.join(
                test_directory,
                f"{label_dataset:s}_histogram_{label_property:s}"
                + f".{test_plot_format:s}"),
            format=test_plot_format,
            dpi=test_plot_dpi)
        plt.close()

        return

    def plot_residual(
        self,
        label_dataset: str,
        label_property: str,
        data_prediction: List[float],
        data_reference: List[float],
        unit_property: str,
        data_metrics: Dict[str, float],
        test_scaling: List[float],
        test_directory: str,
        test_plot_format: str,
        test_plot_dpi: int,
    ):
        """
        Plot property data residual data.
        (x-axis: reference data; y-axis: prediction error)

        Parameters
        ----------
        label_dataset: str
            Label of the data set
        label_property: str
            Label of the property
        data_prediction: list(float)
            List of the predicted data
        data_reference: list(float)
            List of the reference data
        unit_property: str
            Unit of the property
        data_metrics: dict
            Dictionary of the metrics
        test_scaling: list(float)
            List of the scaling factors
        test_directory: str
            Directory to save the plot
        test_plot_format: str
            Format of the plot
        test_plot_dpi: int
            DPI of the plot

        """

        # Plot property: Fontsize
        SMALL_SIZE = 12
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 16
        plt.rc('font', size=SMALL_SIZE, weight='bold')
        plt.rc('xtick', labelsize=SMALL_SIZE)
        plt.rc('ytick', labelsize=SMALL_SIZE)
        plt.rc('legend', fontsize=SMALL_SIZE)
        plt.rc('axes', titlesize=MEDIUM_SIZE)
        plt.rc('axes', labelsize=MEDIUM_SIZE)
        plt.rc('figure', titlesize=BIGGER_SIZE)

        # Plot property: Figure size and arrangement
        figsize = (12, 6)
        left = 0.10
        bottom = 0.15
        column = [0.85, 0.00]
        row = [0.75, 0.00]

        # Initialize figure
        fig = plt.figure(figsize=figsize)
        axs1 = fig.add_axes(
            [left + 0.*np.sum(column), bottom, column[0], row[0]])

        # Data label
        metrics_label = (
            f"{label_property:s} ({label_dataset:s})\n"
            + f"MAE = {data_metrics['mae']:3.2e} {unit_property:s}\n"
            + f"RMSE = {np.sqrt(data_metrics['mse']):3.2e} {unit_property:s}")
        if self.is_imported("scipy"):
            r2 = stats.pearsonr(data_prediction, data_reference).statistic
            metrics_label += (
                "\n" + r"1 - $R^2$ = " + f"{1.0 - r2:3.2e}")
        if 'std' in data_metrics:
            metrics_label += (
                f"\nStd = {data_metrics['std']:3.2e} {unit_property:s}")

        # Scale data if requested
        if test_scaling is not None:
            data_prediction = data_prediction*test_scaling
            data_reference = data_reference*test_scaling
            scale_label = "per atom "
        else:
            scale_label = ""

        # Plot data
        data_min = np.nanmin(data_reference)
        data_max = np.nanmax(data_reference)
        data_dif = data_max - data_min
        axs1.plot(
            [data_min - data_dif*0.05, data_max + data_dif*0.05],
            [0.0, 0.0],
            color='black',
            marker='None', linestyle='--')
        data_deviation = data_reference - data_prediction
        data_devmaxabs = np.nanmax(np.abs(data_deviation))
        axs1.plot(
            data_reference,
            data_deviation,
            color='darkgreen', markerfacecolor='None',
            marker='o', linestyle='None',
            label=metrics_label)

        # Axis range
        axs1.set_xlim(
            data_min - data_dif*0.05, data_max + data_dif*0.05)
        axs1.set_ylim(
            -data_devmaxabs*1.05, +data_devmaxabs*1.05)

        # Figure title
        title = f"Residual plot\n{label_property:s} ({label_dataset:s})"
        if 'std' in data_metrics:
            title = title[:-1] + ", ensemble average)"
        axs1.set_title(title, fontweight='bold')

        # Axis labels
        axs1.set_xlabel(
            f"Reference {label_property:s} {scale_label:s}({unit_property:s})",
            fontweight='bold')
        axs1.get_xaxis().set_label_coords(0.5, -0.12)
        axs1.set_ylabel(
            f"Prediction error {label_property:s} {scale_label:s}"
            + f"({unit_property:s})",
            fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.08, 0.5)

        # Figure legend
        axs1.legend(loc='upper left')

        # Save figure
        plt.savefig(
            os.path.join(
                test_directory,
                f"{label_dataset:s}_residual_{label_property:s}"
                + f".{test_plot_format:s}"),
            format=test_plot_format,
            dpi=test_plot_dpi)
        plt.close()

        return
