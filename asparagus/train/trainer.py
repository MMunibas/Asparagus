import time
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

import numpy as np

from .. import settings
from .. import utils
from .. import data
from .. import model
from .. import train

from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .tester import Tester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Trainer']


# ======================================
# NNP Model Trainer
# ======================================

class Trainer:
    """
    NNP model Trainer class

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to config json file (str)
    data_container: data.DataContainer, optional, default None
        Reference data container object providing training, validation and
        test data for the model training.
    model_calculator: torch.nn.Module, optional, default None
        Model calculator to train matching training and validation
        data in the reference data set. If not provided, the model
        calculator will be initialized according to config input.
    trainer_restart: bool, optional, default False
        Restart the model training from state in config['model_directory']
    trainer_max_epochs: int, optional, default 10000
        Maximum number of training epochs
    trainer_properties: list, optional, default None
        Properties contributing to the prediction quality value.
        If the list is empty or None, all properties which are both predicted
        by the model calculator and available in the data container will be 
        considered for the loss function.
    trainer_properties_metrics: dict, optional, default 'MSE' for all
        Quantification of the property prediction quality only for
        properties in the reference data set.
        Can be given for each property individually and by keyword 'all'
        for every property else wise.
    trainer_properties_weights: dict, optional, default {...}
        Weighting factors for the combination of single property loss
        values to total loss value.
    trainer_optimizer: (str, object), optional, default 'AMSgrad'
        Optimizer class for the NNP model training
    trainer_optimizer_args: dict, optional, default {}
        Additional optimizer class arguments
    trainer_scheduler: (str, object), optional, default 'ExponentialLR'
        Learning rate scheduler class for the NNP model training
    trainer_scheduler_args: dict, optional, default {}
        Additional learning rate scheduler class arguments
    trainer_ema: bool, optional, default True
        Apply exponential moving average scheme for NNP model training
    trainer_ema_decay: float, optional, default 0.999
        Exponential moving average decay rate
    trainer_max_gradient_norm: float, optional, default 1000.0
        Maximum model parameter gradient norm to clip its step size.
        If None, parameter gradient clipping is deactivated.
    trainer_save_interval: int, optional, default 5
        Interval between epoch to save current and best set of model
        parameters.
    trainer_validation_interval: int, optional, default 5
        Interval between epoch to evaluate model performance on
        validation data.
    trainer_evaluate_testset: bool, optional, default True
        Each validation interval and in case of a new best loss function,
        apply Tester class on the test set.
    trainer_max_checkpoints: int, optional, default 1
        Maximum number of checkpoint files stored before deleting the
        oldest ones up to the number threshold.
    trainer_summary_writer: bool, optional, default False
        Write training process to a tensorboard summary writer instance
    trainer_print_progress_bar: bool, optional, default True
        Print progress bar to stout.
        

    """

    # Default arguments for trainer class
    _default_args = {
        'trainer_restart':              False,
        'trainer_max_epochs':           10_000,
        'trainer_properties':           None,
        'trainer_properties_metrics':   {'else': 'mse'},
        'trainer_properties_weights':   {
            'energy': 1., 'forces': 50., 'dipole': 25., 'else': 1.},
        'trainer_optimizer':            'AMSgrad',
        'trainer_optimizer_args':       {'lr': 0.001, 'weight_decay': 1.e-5},
        'trainer_scheduler':            'ExponentialLR',
        'trainer_scheduler_args':       {'gamma': 0.99},
        'trainer_ema':                  True,
        'trainer_ema_decay':            0.99,
        'trainer_max_gradient_norm':    1000.0,
        'trainer_save_interval':        5,
        'trainer_validation_interval':  5,
        'trainer_evaluate_testset':     True,
        'trainer_max_checkpoints':      1,
        'trainer_summary_writer':       False,
        'trainer_print_progress_bar':   True,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'trainer_restart':              [utils.is_bool],
        'trainer_max_epochs':           [utils.is_integer],
        'trainer_properties':           [utils.is_string_array],
        'trainer_properties_metrics':   [utils.is_dictionary],
        'trainer_properties_weights':   [utils.is_dictionary],
        'trainer_optimizer':            [utils.is_string, utils.is_callable],
        'trainer_optimizer_args':       [utils.is_dictionary],
        'trainer_scheduler':            [utils.is_string, utils.is_callable],
        'trainer_scheduler_args':       [utils.is_dictionary],
        'trainer_ema':                  [utils.is_bool],
        'trainer_ema_decay':            [utils.is_numeric],
        'trainer_max_gradient_norm':    [utils.is_numeric, utils.is_None],
        'trainer_save_interval':        [utils.is_integer],
        'trainer_validation_interval':  [utils.is_integer],
        'trainer_evaluate_testset':     [utils.is_bool],
        'trainer_max_checkpoints':      [utils.is_integer],
        'trainer_summary_writer':       [utils.is_bool],
        'trainer_print_progress_bar':   [utils.is_bool],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        data_container: Optional[data.DataContainer] = None,
        model_calculator: Optional[torch.nn.Module] = None,
        trainer_restart: Optional[int] = None,
        trainer_max_epochs: Optional[int] = None,
        trainer_properties: Optional[List[str]] = None,
        trainer_properties_metrics: Optional[Dict[str, str]] = None,
        trainer_properties_weights: Optional[Dict[str, float]] = None,
        trainer_optimizer: Optional[Union[str, object]] = None,
        trainer_optimizer_args: Optional[Dict[str, float]] = None,
        trainer_scheduler: Optional[Union[str, object]] = None,
        trainer_scheduler_args: Optional[Dict[str, float]] = None,
        trainer_ema: Optional[bool] = None,
        trainer_ema_decay: Optional[float] = None,
        trainer_max_gradient_norm: Optional[float] = None,
        trainer_save_interval: Optional[int] = None,
        trainer_validation_interval: Optional[int] = None,
        trainer_evaluate_testset: Optional[bool] = None,
        trainer_max_checkpoints: Optional[int] = None,
        trainer_summary_writer: Optional[bool] = None,
        trainer_print_progress_bar: Optional[bool] = None,
        **kwargs
    ):
        """
        Initialize Model Calculator Trainer.

        """

        ##########################################
        # # # Check Calculator Trainer Input # # #
        ##########################################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, train),
            check_dtype=utils.get_dtype_args(self, train)
        )

        # Update global configuration dictionary
        config.update(config_update)
        
        # Assign module variable parameters from configuration
        self.device = config.get('device')
        self.dtype = config.get('dtype')

        ################################
        # # # Check Data Container # # #
        ################################

        # Assign DataContainer if not done already
        if data_container is None:
            self.data_container = data.DataContainer(
                config=config,
                **kwargs)

        # Assign training and validation data loader
        self.data_train = self.data_container.train_loader
        self.data_valid = self.data_container.valid_loader

        # Get reference data properties
        self.data_properties = self.data_container.data_load_properties
        self.data_units = self.data_container.data_unit_properties

        ##################################
        # # # Check Model Calculator # # #
        ##################################

        # Assign model calculator model if not done already
        if self.model_calculator is None:
            self.model_calculator, _ = model.get_calculator(
                config=config,
                **kwargs)

        # Get model properties
        self.model_properties = self.model_calculator.model_properties
        self.model_units = self.model_calculator.model_unit_properties

        ############################
        # # # Check Properties # # #
        ############################

        # Check property definition for the loss function evaluation
        self.trainer_properties = self.check_properties(
            self.trainer_properties,
            self.data_properties,
            self.model_properties)

        # Check property metrics and weights for the loss function
        self.trainer_properties_metrics, self.trainer_properties_weights = ( 
            self.check_properties_metrics_weights(
                self.trainer_properties,
                self.trainer_properties_metrics,
                self.trainer_properties_weights)
            )

        # Check model units and model to data unit conversion
        self.model_units, self.data_units, self.model_conversion = (
            self.check_model_units(
                self.trainer_properties,
                self.model_units,
                self.data_units)
            )

        # Show assigned property units and 
        msg = (
              f"  {'Property ':<17s}|  {'Model Unit':<12s}|"
            + f"  {'Data Unit':<12s}|  {'Conv. fact.':<12s}|"
            + f"  {'Loss Metric':<12s}|  {'Loss Weight':<12s}\n")
        msg += "-"*len(msg) + "\n"
        for prop, unit in self.model_units.items():
            msg += f"  {prop:<16s} |  {unit:<11s} |"
            msg += f"  {self.data_units.get(prop):<11s} |"
            msg += f"  {self.model_conversion.get(prop):> 8.4e} |"
            if prop in self.trainer_properties_metrics:
                msg += f"  {self.trainer_properties_metrics[prop]:<11s} |"
            else:
                msg += f"  {'':<11s} |"
            if prop in self.trainer_properties_weights:
                msg += f"  {self.trainer_properties_weights[prop]:> 10.4f}"
            msg += "\n"
        logger.info("INFO:\nTraining Properties\n" + msg + "\n")

        # Assign potentially new property units to the model
        if hasattr(self.model_calculator, 'set_unit_properties'):
            self.model_calculator.set_unit_properties(self.model_units)

        ######################################
        # # # Set Model Property Scaling # # #
        ######################################

        # Initialize model property scaling dictionary
        model_properties_scaling = {}

        # Get property scaling guess from reference data and convert from data
        # to model units.
        for prop, item in self.data_container.get_property_scaling().items():
            if prop in self.model_properties:
                model_properties_scaling[prop] = (
                    np.array(item)/self.model_conversion[prop])

        # Set current model property scaling
        self.model_calculator.set_property_scaling(model_properties_scaling)

        #############################
        # # # Prepare Optimizer # # #
        #############################

        # Assign model parameter optimizer
        self.trainer_optimizer = get_optimizer(
            self.trainer_optimizer,
            self.model_calculator.get_trainable_parameters(),
            self.trainer_optimizer_args)


        # Check maximum gradient norm
        if self.trainer_max_gradient_norm is None:
            self.gradient_clipping = False
        else:
            self.gradient_clipping = True

        #############################
        # # # Prepare Scheduler # # #
        #############################

        # Assign learning rate scheduler
        self.trainer_scheduler = get_scheduler(
            self.trainer_scheduler,
            self.trainer_optimizer,
            self.trainer_scheduler_args)

        #######################
        # # # Prepare EMA # # #
        #######################

        # Assign Exponential Moving Average model
        if self.trainer_ema:
            from torch_ema import ExponentialMovingAverage
            self.trainer_ema_model = ExponentialMovingAverage(
                self.model_calculator.parameters(),
                decay=self.trainer_ema_decay)

        ################################
        # # # Prepare File Manager # # #
        ################################

        # Initialize checkpoint file manager and summary writer
        self.filemanager = model.FileManager(
            config=config,
            max_checkpoints=self.trainer_max_checkpoints)
        
        # Initialize training summary writer
        if self.trainer_summary_writer:
            from torch.utils.tensorboard import SummaryWriter
            self.summary_writer = SummaryWriter(
                log_dir=self.filemanager.logs_dir)

        ##########################
        # # # Prepare Tester # # #
        ##########################

        # Assign model prediction tester if test set evaluation is requested
        if self.trainer_evaluate_testset:
            self.tester = Tester(
                config=config,
                data_container=self.data_container,
                test_datasets='test')

        #############################
        # # # Save Model Config # # #
        #############################

        # Save a copy of the current model configuration in the model directory
        self.filemanager.save_config(config)

        return

    def check_properties(
        self,
        trainer_properties: List[str],
        data_properties: List[str],
        model_properties: List[str],
    ) -> List[str]:
        """
        Check properties for the contribution to the loss function between 
        predicted model properties and available properties in the reference
        data container.
        
        Parameter
        ---------
        trainer_properties: list(str)
            Properties contributing to the loss function. If empty or None,
            take all matching properties between model and data container.
        data_properties: list(str)
            Properties available in the reference data container
        model_properties: list(str)
            Properties predicted by the model calculator

        Returns:
        --------
        list(str)
            List of loss function property contributions.

        """
        
        # Check matching data and model properties 
        matching_properties = []
        for prop in data_properties:
            if prop in model_properties:
                matching_properties.append(prop)

        # Check training properties are empty, use all matching properties
        if trainer_properties is None or not len(trainer_properties):
            trainer_properties = matching_properties
        else:
            for prop in trainer_properties:
                if prop not in matching_properties:
                    if prop in data_properties:
                        msg = "model calculator!"
                    else:
                        msg = "data container!"
                    raise SyntaxError(
                        f"Requested property '{prop:s}' as loss function "
                        + "contribution is not available in " + msg)
        
        return trainer_properties

    def check_properties_metrics_weights(
        self,
        trainer_properties: List[str],
        trainer_properties_metrics: Dict[str, float],
        trainer_properties_weights: Dict[str, float],
        default_property_metrics: Optional[str] = 'mse',
        default_property_weights: Optional[float] = 1.0,
    ) -> (Dict[str, float], Dict[str, float]):
        """
        Prepare property loss metrics and weighting factors for the loss 
        function contributions.
        
        Parameter
        ---------
        trainer_properties: list(str)
            Properties contributing to the loss function
        trainer_properties_metrics: dict(str, float)
            Metrics functions for property contribution in the loss function
        trainer_properties_weights: dict(str, float)
            Weighting factors for property metrics in the loss function
        default_property_metrics: str, optional, default 'mse'
            Default option, if the property not in metrics dictionary and no 
            other default value is defined by key 'else'. 
            Default: mean scare error (mse) 
            Alternative: mean absolute error (mae)
        default_property_weights: str, optional, default 1.0
            Default option, if the property not in weights dictionary and no 
            other default value is defined by key 'else'.

        Returns:
        --------
        dict(str, float)
            Prepared property metrics dictionary
        dict(str, float)
            Prepared property weighting factors dictionary

        """

        # Check property metrics
        for prop in trainer_properties:
            if (
                trainer_properties_metrics.get(prop) is None
                and trainer_properties_metrics.get('else') is None
            ):
                trainer_properties_metrics[prop] = default_property_metrics
            elif trainer_properties_metrics.get(prop) is None:
                trainer_properties_metrics[prop] = (
                    trainer_properties_metrics.get('else'))

        # Check property weights
        for prop in trainer_properties:
            if (
                trainer_properties_weights.get(prop) is None
                and trainer_properties_weights.get('else') is None
            ):
                trainer_properties_weights[prop] = default_property_weights
            elif trainer_properties_weights.get(prop) is None:
                trainer_properties_weights[prop] = (
                    trainer_properties_weights.get('else'))

        return trainer_properties_metrics, trainer_properties_weights

    def check_model_units(
        self,
        trainer_properties: List[str],
        model_units: Dict[str, str],
        data_units: Dict[str, str],
        related_properties: Dict[str, str] = {
            'energy': 'atomic_energies',
            'charge': 'atomic_charges'}
    ) -> [Dict[str,str], Dict[str,str], Dict[str,float]]:
        """
        Check the definition of the model units or assign units from the
        reference dataset

        Parameter
        ---------
        trainer_properties: list(str)
            Properties contributing to the loss function
        model_units: dict(str, str)
            Dictionary of model property units.
        data_units: dict(str, str)
            Dictionary of data property units.

        Returns
        -------
        dict(str, str)
            Dictionary of adopted model property units
        dict(str, str)
            Dictionary of adopted data property units
        dict(str, float)
            Dictionary of model to data property unit conversion factors

        """

        # Initialize mode to data unit conversion dictionary
        model_conversion = {}
        
        # Check basic properties - positions, charge
        for prop in ['positions', 'charge']:
            
            # Check model property unit
            if model_units.get(prop) is None:
                model_units[prop] = data_units.get(prop)
                model_conversion[prop] = 1.0
            else:
                model_conversion[prop], _ = utils.check_units(
                    model_units[prop], data_units.get(prop))

            # Append related property units
            if prop in related_properties:
                related_prop = related_properties[prop]
                model_units, data_units, model_conversion = (
                    self._check_model_units_related(
                        prop, 
                        related_prop,
                        model_units,
                        data_units,
                        model_conversion)
                    )

        # Iterate over training properties
        for prop in trainer_properties:
            
            # Check model property unit
            if model_units.get(prop) is None:
                model_units[prop] = data_units.get(prop)
                model_conversion[prop] = 1.0
            else:
                model_conversion[prop], _ = utils.check_units(
                    model_units[prop], data_units.get(prop))

            # Append related property units
            if prop in related_properties:
                related_prop = related_properties[prop]
                model_units, data_units, model_conversion = (
                    self._check_model_units_related(
                        prop, 
                        related_prop,
                        model_units,
                        data_units,
                        model_conversion)
                    )

        return model_units, data_units, model_conversion

    def _check_model_units_related(
        self,
        prop,
        related_prop,
        model_units,
        data_units,
        model_conversion
    ) -> [Dict[str,str], Dict[str,str], Dict[str,float]]:
        """
        Check and add the related property label to the model and data units
        dictionary and add the unit conversion.

        Parameter
        ---------
        prop: str
            Property label
        related_prop: str
            Related property label sharing same property unit
        model_units: dict(str, str)
            Dictionary of model property units.
        data_units: dict(str, str)
            Dictionary of data property units.
        model_conversion: dict(str, float)
            Dictionary of model to data property unit conversion factors

        Returns
        -------
        dict(str, str)
            Dictionary of adopted model property units
        dict(str, str)
            Dictionary of adopted data property units
        dict(str, float)
            Dictionary of model to data property unit conversion factors

        """

        # Add related property to lists and conversion dictionary
        if (
            related_prop in model_units
            and related_prop in data_units
        ):
            model_conversion[related_prop], _ = utils.check_units(
                model_units[related_prop], 
                data_units.get(related_prop))
        elif related_prop in data_units:
            model_units[related_prop] = model_units[prop]
            model_conversion[related_prop], _ = utils.check_units(
                model_units[prop], data_units[related_prop])
        elif related_prop in model_units:
            data_units[related_prop] = model_units[related_prop]
            model_conversion[related_prop], _ = utils.check_units(
                model_units[related_prop], data_units[related_prop])
            model_conversion[related_prop] = model_conversion[prop]
        else:
            model_units[related_prop] = model_units[prop]
            data_units[related_prop] = model_units[prop]
            model_conversion[related_prop] = model_conversion[prop]
        
        return model_units, data_units, model_conversion

    def run(
        self,
        reset_best_loss=False,
        verbose=True,
        debug=False,
        **kwargs,
    ):
        """
        Train model calculator.

        Parameters
        ----------
        reset_best_loss: bool, optional, default False
            If False, continue model potential validation from stored best
            loss value. Else, reset best loss value to None.
        verbose: bool, optional, default True
            Show progress bar for the current epoch.
        debug: bool, optional, dafault False
            Enable torch autograd anomaly detection.

        """

        ####################################
        # # # Prepare Model and Metric # # #
        ####################################

        # Load, if exists, latest model calculator and training state
        # checkpoint file
        latest_checkpoint = self.filemanager.load_checkpoint(
            checkpoint_label='last')

        trainer_epoch_start = 1
        best_loss = None
        if latest_checkpoint is not None:
            
            # Assign model parameters
            self.model_calculator.load_state_dict(
                latest_checkpoint['model_state_dict'])
            
            # Assign optimizer, scheduler and epoch parameter if available
            if latest_checkpoint.get('optimizer_state_dict') is not None:
                self.trainer_optimizer.load_state_dict(
                    latest_checkpoint['optimizer_state_dict'])
            if latest_checkpoint.get('scheduler_state_dict') is not None:
                self.trainer_scheduler.load_state_dict(
                    latest_checkpoint['scheduler_state_dict'])
            if latest_checkpoint.get('epoch') is not None:
                trainer_epoch_start = latest_checkpoint['epoch'] + 1
            
            # Initialize best total loss value of validation reference data
            if reset_best_loss or latest_checkpoint.get('best_loss') is None:
                best_loss = None
            else:
                best_loss = latest_checkpoint['best_loss']
            
        # Initialize training mode for calculator
        self.model_calculator.train()
        torch.set_grad_enabled(True)
        if debug:
            torch.autograd.set_detect_anomaly(True)

        # Reset property metrics
        metrics_best = self.reset_metrics()

        # Define loss function
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')

        # Count number of training batches
        Nbatch_train = len(self.data_train)

        # Initialize training time estimation per epoch
        train_time_estimation = np.nan

        # Get model and descriptor cutoffs
        model_cutoff = self.model_calculator.model_cutoff
        if hasattr(self.model_calculator.input_module, 'input_radial_cutoff'):
            input_cutoff = (
                self.model_calculator.input_module.input_radial_cutoff)
            if input_cutoff != model_cutoff:
                cutoff = [input_cutoff, model_cutoff]
        else:
            cutoff = [model_cutoff]

        # Set model and descriptor cutoffs for neighbor list calculation
        self.data_train.init_neighbor_list(
            cutoff=cutoff,
            device=self.device,
            dtype=self.dtype)
        self.data_valid.init_neighbor_list(
            cutoff=cutoff,
            device=self.device,
            dtype=self.dtype)

        ##########################
        # # # Start Training # # #
        ##########################

        # Skip if max epochs are already reached
        if trainer_epoch_start > self.trainer_max_epochs:
            return

        # Loop over epochs
        for epoch in torch.arange(
            trainer_epoch_start, self.trainer_max_epochs + 1
        ):

            # Start epoch train timer
            train_time_epoch_start = time.time()

            # Reset property metrics
            metrics_train = self.reset_metrics()

            # Loop over training batches
            for ib, batch in enumerate(self.data_train):

                # Start batch train timer
                train_time_batch_start = time.time()

                # Eventually show training progress
                if verbose:
                    utils.printProgressBar(
                        ib, Nbatch_train,
                        prefix=f"Epoch {epoch: 5d}",
                        suffix=(
                            "Complete - Remaining Epoch Time: "
                            + f"{train_time_estimation: 4.1f} s     "
                            ),
                        length=42)

                # Reset optimizer gradients
                self.trainer_optimizer.zero_grad(
                    set_to_none=(not self.gradient_clipping))

                # Predict model properties from data batch
                prediction = self.model_calculator(batch)

                # Compute total and single loss values for training properties
                metrics_batch = self.compute_metrics(
                    prediction, batch, loss_fn=loss_fn)
                loss = metrics_batch['loss']

                # Predict parameter gradients by backwards propagation
                loss.backward()

                # Clip parameter gradients
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_calculator.parameters(),
                        self.trainer_max_gradient_norm)

                # Update model parameters
                self.trainer_optimizer.step()

                # Apply Exponential Moving Average
                if self.trainer_ema:
                    self.trainer_ema_model.update()

                # Update average metrics
                self.update_metrics(metrics_train, metrics_batch)

                # End batch train timer
                train_time_batch_end = time.time()

                # Eventually update training batch time estimation
                if verbose:
                    train_time_batch = (
                        train_time_batch_end - train_time_batch_start)
                    if ib:
                        train_time_estimation = (
                            0.5*(train_time_estimation - train_time_batch)
                            + 0.5*train_time_batch*(Nbatch_train - ib - 1))
                    else:
                        train_time_estimation = (
                            train_time_batch*(Nbatch_train - 1))

            # Increment scheduler step
            self.trainer_scheduler.step()

            # Stop epoch train timer
            train_time_epoch_end = time.time()
            train_time_epoch = train_time_epoch_end - train_time_epoch_start

            # Eventually show final training progress
            if verbose:
                utils.printProgressBar(
                    Nbatch_train, Nbatch_train,
                    prefix=f"Epoch {epoch: 5d}",
                    suffix=(
                        "Done - Epoch Time: " +
                        f"{train_time_epoch: 4.1f} s, " +
                        f"Loss: {metrics_train['loss']: 4.4f}   "),
                    length=42)

            # Save current model each interval
            if not (epoch % self.trainer_save_interval):
                self.filemanager.save_checkpoint(
                    model=self.model_calculator,
                    optimizer=self.trainer_optimizer,
                    scheduler=self.trainer_scheduler,
                    epoch=epoch,
                    best_loss=best_loss)

            # Perform model validation each interval
            if not (epoch % self.trainer_validation_interval):

                # Change to evaluation mode for calculator
                self.model_calculator.eval()

                # Reset property metrics
                metrics_valid = self.reset_metrics()

                # Loop over validation batches
                for batch in self.data_valid:

                    # Predict model properties from data batch
                    prediction = self.model_calculator(batch)

                    # Compute total and single loss values for training
                    # properties
                    metrics_batch = self.compute_metrics(
                        prediction, batch, loss_fn=loss_fn, loss_only=False)

                    # Update average metrics
                    self.update_metrics(metrics_valid, metrics_batch)

                # Change back to training mode for calculator
                self.model_calculator.train()

                # Check for model improvement and save as best model eventually
                if (
                    best_loss is None
                    or metrics_valid['loss'] < best_loss
                ):

                    # Store best metrics
                    metrics_best = metrics_valid

                    # Update best total loss value
                    best_loss = metrics_valid['loss']

                    # Save model calculator state
                    self.filemanager.save_checkpoint(
                        model=self.model_calculator,
                        optimizer=self.trainer_optimizer,
                        scheduler=self.trainer_scheduler,
                        epoch=epoch,
                        best=True,
                        best_loss=best_loss)

                    # Evaluation of the test set if requested
                    if self.trainer_evaluate_testset:
                        self.tester.test(
                            self.model_calculator,
                            test_directory=self.filemanager.best_dir,
                            test_plot_correlation=True,
                            test_plot_histogram=True,
                            test_plot_residual=True)

                    # Add process to training summary writer
                    if self.trainer_summary_writer:
                        for prop, value in metrics_best.items():
                            if utils.is_dictionary(value):
                                for metric, val in value.items():
                                    self.summary_writer.add_scalar(
                                        prop + '_' + metric,
                                        metrics_best[prop][metric],
                                        global_step=epoch)
                            else:
                                self.summary_writer.add_scalar(
                                    prop, metrics_best[prop],
                                    global_step=epoch)

                # Print validation metrics summary
                msg = (
                    f"Summary Epoch: {epoch:d}/" +
                    f"{self.trainer_max_epochs:d}\n" +
                    "  Loss   train / valid: " +
                    f" {metrics_train['loss']:.2E} /" +
                    f" {metrics_valid['loss']:.2E}" +
                    f"  Best Loss valid: {metrics_best['loss']:.2E}\n"
                    f"  Property Metrics (valid):\n")
                for prop in self.trainer_properties:
                    msg += (
                        f"    {prop:10s}  MAE (Best) / RMSE (Best): " +
                        f" {metrics_valid[prop]['mae']:.2E}" +
                        f" ({metrics_best[prop]['mae']:.2E}) /" +
                        f" {np.sqrt(metrics_valid[prop]['mse']):.2E}" +
                        f" ({np.sqrt(metrics_best[prop]['mse']):.2E})" +
                        f" {self.model_units[prop]:s}\n")
                logger.info("INFO:\n" + msg)

        return

    def predict_batch(self, batch):
        """
        Predict properties from data batch.

        Parameters
        ----------
        batch: dict
            Data batch dictionary

        Returns
        -------
        dict(str, torch.Tensor)
            Model Calculator prediction of properties

        """

        # Predict properties
        return self.model_calculator(
            batch['atoms_number'],
            batch['atomic_numbers'],
            batch['positions'],
            batch['idx_i'],
            batch['idx_j'],
            batch['charge'],
            batch['atoms_seg'],
            batch['pbc_offset'])

    def reset_metrics(self):

        '''
        Reset metrics dictionary.
        Returns
        -------

        '''

        # Initialize metrics dictionary
        metrics = {}

        # Add loss total value
        metrics['loss'] = 0.0

        # Add data counter
        metrics['Ndata'] = 0

        # Add training property metrics
        for prop in self.trainer_properties:
            metrics[prop] = {
                'loss': 0.0,
                'mae': 0.0,
                'mse': 0.0}

        return metrics

    def update_metrics(
        self,
        metrics: Dict[str, float],
        metrics_update: Dict[str, float],
    ) -> Dict[str, float]:

        '''
        Update metrics dictionary.

        Parameters
        ----------
        metrics: dict
            Metrics dictionary
        metrics_update: dict
            Metrics dictionary to update

        Returns
        -------

        '''

        # Get data sizes and metric ratio
        Ndata = metrics['Ndata']
        Ndata_update = metrics_update['Ndata']
        fdata = float(Ndata)/float((Ndata + Ndata_update))
        fdata_update = 1. - fdata

        # Update metrics
        metrics['Ndata'] = metrics['Ndata'] + metrics_update['Ndata']
        metrics['loss'] = (
            fdata*metrics['loss']
            + fdata_update*metrics_update['loss'].detach().item())
        for prop in self.trainer_properties:
            for metric in metrics_update[prop].keys():
                metrics[prop][metric] = (
                    fdata*metrics[prop][metric]
                    + fdata_update*metrics_update[prop][metric].detach().item()
                    )

        return metrics

    def compute_metrics(
        self,
        prediction: Dict[str, Any],
        reference: Dict[str, Any],
        loss_fn: Optional[object] = None,
        loss_only: Optional[bool] = True,
    ) -> Dict[str, float]:

        '''
        Compute metrics. This function evaluates the loss function.

        Parameters
        ----------
        prediction: dict
            Model prediction dictionary
        reference:
            Reference data dictionary
        loss_fn:
            Loss function if not defined it is set to torch.nn.L1Loss
        loss_only
            Compute only loss function or compute MAE and MSE as well

        Returns
        -------

        '''

        # Check loss function input
        if loss_fn is None:
            loss_fn = torch.nn.L1Loss(reduction="mean")

        # Initialize MAE calculator function if needed
        if not loss_only:
            mae_fn = torch.nn.L1Loss(reduction="mean")
            mse_fn = torch.nn.MSELoss(reduction="mean")

        # Initialize metrics dictionary
        metrics = {}

        # Add batch size
        metrics['Ndata'] = reference['atoms_number'].size()[0]

        # Iterate over training properties
        for ip, prop in enumerate(self.trainer_properties):

            # Initialize single property metrics dictionary
            metrics[prop] = {}

            # Compute loss value per atom
            metrics[prop]['loss'] = loss_fn(
                torch.flatten(prediction[prop])
                * self.model_conversion[prop],
                torch.flatten(reference[prop]))

            # Weight and add to total loss
            if ip:
                metrics['loss'] = metrics['loss'] + (
                    self.trainer_properties_weights[prop]
                    * metrics[prop]['loss'])
            else:
                metrics['loss'] = (
                    self.trainer_properties_weights[prop]
                    * metrics[prop]['loss'])

            # Compute MAE and MSE if requested
            if not loss_only:
                metrics[prop]['mae'] = mae_fn(
                    torch.flatten(prediction[prop])
                    * self.model_conversion[prop],
                    torch.flatten(reference[prop]))
                metrics[prop]['mse'] = mse_fn(
                    torch.flatten(prediction[prop])
                    * self.model_conversion[prop],
                    torch.flatten(reference[prop]))

        return metrics
