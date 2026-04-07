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
        if not os.path.exists(self.test_directory):
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
        test_properties: Optional[Union[str, List[str]]] = None,
        test_directory: Optional[str] = None,
        test_plot_correlation: Optional[bool] = True,
        test_plot_histogram: Optional[bool] = False,
        test_plot_residual: Optional[bool] = False,
        test_plot_format: Optional[str] = 'png',
        test_plot_dpi: Optional[int] = 300,
        test_save_csv: Optional[bool] = False,
        test_csv_file: Optional[str] = 'results.csv',
        test_save_npz: Optional[bool] = False,
        test_npz_file: Optional[str] = 'results.npz',
        test_scale_per_atom: Optional[Union[str, List[str]]] = ['energy'],
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
        test_properties: (str, list(str)), optional, default None
            Model properties to evaluate which must be available in the
            model prediction and the reference test data set. If None, model
            properties will be evaluated as initialized.
        test_directory: str, optional, default '.'
            Directory to store evaluation graphics and data.
        test_plot_correlation: bool, optional, default True
            Show evaluation in property correlation plots
            (x-axis: reference property; y-axis: predicted property).
        test_plot_histogram: bool, optional, default False
            Show prediction error spread in histogram plots.
        test_plot_residual: bool, optional, default False
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
            mlmm_cutoffs = model_calculator.get_mlmm_cutoff_ranges()

            # Set maximum model cutoff for neighbor list calculation
            if datasubset.mlmm_neighbor_list is None:
                datasubset.init_mlmm_neighbor_list(
                    cutoff=mlmm_cutoffs,
                    device=self.device,
                    dtype=self.dtype)
            else:
                datasubset.mlmm_neighbor_list.set_cutoffs(mlmm_cutoffs)

            # Prepare dictionary for property values, number of atoms per
            # system, and reference energy shifts
            test_prediction = {prop: [] for prop in test_properties}
            if model_ensemble:
                test_prediction.update(
                    {
                        imodel: {prop: [] for prop in test_properties}
                        for imodel in range(model_ensemble_num)
                    })
            test_reference = {prop: [] for prop in test_properties}
            test_prediction['atoms_number'] = []
            test_shifts = {prop: [] for prop in ['energy', 'atomic_energies']}

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
                    test_prediction['fragment_numbers'] = []
            else:
                test_fragments = None

            # Reset property metrics
            metrics_test = self.reset_metrics(
                test_properties,
                model_ensemble,
                model_ensemble_num,
                test_fragments=test_fragments)

            # Loop over data batches
            for batch in datasubset:

                # Predict model properties from data batch
                batch = model_calculator(
                    batch,
                    verbose_results=True)

                # Compute metrics for test properties
                metrics_batch = self.compute_metrics(
                    batch,
                    batch['reference'],
                    test_properties,
                    test_conversion,
                    model_ensemble,
                    model_ensemble_num,
                    test_fragments=test_fragments)

                # Update average metrics
                self.update_metrics(
                    metrics_test,
                    metrics_batch,
                    test_properties,
                    model_ensemble,
                    model_ensemble_num,
                    test_fragments=test_fragments)

                # Store prediction and reference data system resolved
                Nsys = len(batch['atoms_number'])
                Natoms = len(batch['atomic_numbers'])
                Npairs = len(batch['idx_i'])
                for prop in test_properties:
                    
                    # Detach prediction and reference data
                    data_prediction = batch[prop].detach().cpu().numpy()
                    data_reference = (
                        batch['reference'][prop].detach().cpu().numpy())

                    # Apply unit conversion of model prediction
                    data_prediction *= test_conversion[prop]

                    # If data are atom resolved
                    if not data_prediction.shape:
                        data_prediction = [data_prediction]
                        data_reference = list(data_reference)
                    elif data_prediction.shape[0] == Natoms:
                        sys_i = batch['sys_i'].cpu().numpy()
                        data_prediction = [
                            list(data_prediction[sys_i == isys])
                            for isys in range(Nsys)]
                        data_reference = [
                            list(data_reference[sys_i == isys])
                            for isys in range(Nsys)]
                    # If data are atom pair resolved
                    elif data_prediction.shape[0] == Npairs:
                        sys_pair_i = (
                            batch['sys_i'][batch['idx_i']].cpu().numpy())
                        data_prediction = [
                            list(data_prediction[sys_pair_i == isys])
                            for isys in range(Nsys)]
                        data_reference = [
                            list(data_reference[sys_pair_i == isys])
                            for isys in range(Nsys)]
                    # Else, it is already system resolved
                    else:
                        data_prediction = list(data_prediction)
                        data_reference = list(data_reference)

                    # Assign prediction and reference data
                    test_prediction[prop] += data_prediction
                    test_reference[prop] += data_reference

                    if model_ensemble:
                        
                        for imodel in range(model_ensemble_num):
                            
                            # Detach prediction and reference data
                            data_prediction = (
                                batch[imodel][prop].detach().cpu().numpy()
                                )

                            # Apply unit conversion of model prediction
                            data_prediction *= test_conversion[prop]

                            # If data are atom resolved
                            if not data_prediction.shape:
                                data_prediction = [data_prediction]
                            elif data_prediction.shape[0] == Natoms:
                                sys_i = batch['sys_i'].cpu().numpy()
                                data_prediction = [
                                    list(data_prediction[sys_i == isys])
                                    for isys in range(Nsys)]
                            # If data are atom pair resolved
                            elif data_prediction.shape[0] == Npairs:
                                sys_pair_i = (
                                    batch['sys_i'][
                                        batch['idx_i']].cpu().numpy())
                                data_prediction = [
                                    list(data_prediction[sys_pair_i == isys])
                                    for isys in range(Nsys)]
                            # Else, it is already system resolved
                            else:
                                data_prediction = list(data_prediction)

                            # Assign prediction data
                            test_prediction[imodel][prop] += data_prediction

                # Store atom numbers
                test_prediction['atoms_number'] += list(
                    batch['atoms_number'].cpu().numpy())

                # Store fragment numbers
                if 'fragment_numbers' in test_prediction:
                    test_prediction['fragment_numbers'] += list(
                        batch['fragment_numbers'].cpu().numpy())

                # Compute energy and atomic energies shifts
                test_shifts_energy = np.zeros(Nsys, dtype=float)
                test_shifts_atomic_energies = np.zeros(Natoms, dtype=float)
                if atomic_energies_shift is None:
                    test_shifts['energy'] += list(test_shifts_energy)
                    test_shifts['atomic_energies'] += list(
                        test_shifts_atomic_energies)
                else:
                    atomic_numbers = batch['atomic_numbers'].cpu().numpy()
                    sys_i = batch['sys_i'].cpu().numpy()
                    test_shifts_atomic_energies = (
                        atomic_energies_shift[atomic_numbers])
                    np.add.at(
                        test_shifts_energy,
                        sys_i,
                        test_shifts_atomic_energies)
                    test_shifts['energy'] += list(test_shifts_energy)
                    test_shifts['atomic_energies'] += list(
                        test_shifts_atomic_energies)

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

            # Prepare data
            if prop in shifts:
                results_np = np.column_stack((
                    np.array(pred).reshape(-1),
                    np.array(reference[prop]).reshape(-1),
                    np.array(shifts[prop]).reshape(-1))
                )
                columns_np=[
                    "prediction", "reference", "shift"]
            else:
                results_np = np.column_stack((
                    np.array(pred).reshape(-1),
                    np.array(reference[prop]).reshape(-1))
                )
                columns_np=[
                    "prediction", "reference"]
            
            # Store data in npz format generated via the pandas data frame
            if self.is_imported("pandas"):
                results = pd.DataFrame(
                    results_np,
                    columns=columns_np
                    )
                np.savez(
                    npz_file_prop,
                    **{
                        column: results[column].values
                        for column in results.columns}
                    )
            else:
                self.logger.warning(
                    "Module 'pandas' is not available. "
                    + "Test properties are not written to a npz file!")

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
            List of occuring fragment indices in the dataset systems.
            If None or just one, properties metrics are not separately 
            evaluated for each fragment.
            If two or more indices, possible properties are evaluated for each
            fragment also individually.

        Returns
        -------
        dict(str, dict(str, float))
            Property metrics dictionary

        """

        # Initialize metrics dictionary
        metrics = {}

        # Add data counter
        metrics['Ndata'] = 0

        # Add property metrics
        for prop in test_properties:
            
            metrics[prop] = {
                'mae': 0.0,
                'mse': 0.0}
            
            # For model ensemble, reset individual model metrics
            if model_ensemble:
                metrics[prop]['std'] = 0.0
                for imodel in range(model_ensemble_num):
                    metrics[prop][imodel] = {
                        'mae': 0.0,
                        'mse': 0.0}
            
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
                        'mse': 0.0}

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
            List of occuring fragment indices in the dataset systems.
            If None or just one, properties metrics are not separately 
            evaluated for each fragment.
            If two or more indices, possible properties are evaluated for each
            fragment also individually.

        Returns
        -------
        dict(str, dict(str, float))
            Property metrics dictionary

        """

        # Initialize metrics dictionary
        metrics = {}

        # Add batch size
        metrics['Ndata'] = reference[
            test_properties[0]].size()[0]

        # Iterate over test properties
        mae_fn = torch.nn.L1Loss(reduction="mean")
        mse_fn = torch.nn.MSELoss(reduction="mean")
        for ip, prop in enumerate(test_properties):

            # Initialize single property metrics dictionary
            metrics[prop] = {}

            # Compute MAE and MSE
            metrics[prop]['mae'] = mae_fn(
                torch.flatten(prediction[prop])
                * test_conversion[prop],
                torch.flatten(reference[prop]))
            metrics[prop]['mse'] = mse_fn(
                torch.flatten(prediction[prop])
                * test_conversion[prop],
                torch.flatten(reference[prop]))

            # For model ensembles
            if model_ensemble:
                
                # Compute mean standard deviation
                prop_std = f"std_{prop:s}"
                metrics[prop]['std'] = torch.mean(prediction[prop_std])
                
                #  Compute single calculator statistics
                for imodel in range(model_ensemble_num):
                    metrics[prop][imodel] = {}
                    metrics[prop][imodel]['mae'] = mae_fn(
                        torch.flatten(prediction[imodel][prop])
                        * test_conversion[prop],
                        torch.flatten(reference[prop]))
                    metrics[prop][imodel]['mse'] = mse_fn(
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
                    metrics[fragment_prop] = {}

                    # Fragment selection
                    fragment_selection = (
                        fragment == prediction['fragment_numbers'])

                    # Compute MAE and MSE
                    metrics[fragment_prop]['mae'] = mae_fn(
                        torch.flatten(prediction[prop][fragment_selection])
                        * test_conversion[prop],
                        torch.flatten(reference[prop][fragment_selection]))
                    metrics[fragment_prop]['mse'] = mse_fn(
                        torch.flatten(prediction[prop][fragment_selection])
                        * test_conversion[prop],
                        torch.flatten(reference[prop][fragment_selection]))

        return metrics

    def update_metrics(
        self,
        metrics: Dict[str, float],
        metrics_update: Dict[str, float],
        test_properties: List[str],
        model_ensemble: bool,
        model_ensemble_num: int,
        test_fragments: List[int] = None,
    ) -> Dict[str, float]:
        """
        Update the metrics dictionary.

        Parameters
        ----------
        metrics: dict
            Dictionary of the new metrics
        metrics_update: dict
            Dictionary of the metrics to update
        test_properties: list(str)
            List of properties to evaluate
        model_ensemble: bool
            Model calculator or model ensemble flag
        model_ensemble_num: int
            Model ensemble calculator number
        test_fragments: list(int)
            List of occuring fragment indices in the dataset systems.
            If None or just one, properties metrics are not separately 
            evaluated for each fragment.
            If two or more indices, possible properties are evaluated for each
            fragment also individually.

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

        # Update metrics
        metrics['Ndata'] = metrics['Ndata'] + metrics_update['Ndata']
        for prop in test_properties:
            
            for metric in ['mae', 'mse']:
                metrics[prop][metric] = (
                    fdata*metrics[prop][metric]
                    + fdata_update*metrics_update[prop][metric].detach().item()
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
                        metrics[fragment_prop][metric] = (
                            fdata*metrics[fragment_prop][metric]
                            + fdata_update*metrics_update[
                                fragment_prop][metric].detach().item()
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

        # Scale data if requested
        if test_atoms_scaling is not None:
            data_prediction = data_prediction*test_atoms_scaling
            data_reference = data_reference*test_atoms_scaling
            scale_label = "/atom"
        else:
            scale_label = ""

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
