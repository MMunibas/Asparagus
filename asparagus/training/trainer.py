import sys
import time
import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import torch

import numpy as np

import asparagus
from asparagus import training
from asparagus import settings
from asparagus import utils
from asparagus import data
from asparagus import model

from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .tester import Tester

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
        Model calculator to train matching model properties with the reference
        data set. If not provided, the model calculator will be initialized
        according to config input.
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
    trainer_batch_size: int, optional, default None
        Default dataloader batch size
    trainer_train_batch_size: int, optional, default None
        Training dataloader batch size. If None, default batch size is used.
    trainer_valid_batch_size: int, optional, default None
        Validation dataloader batch size. If None, default batch size is used.
    trainer_test_batch_size:  int, optional, default None
        Test dataloader batch size. If None, default batch size is used.
    trainer_num_batch_workers: int, optional, default 1
        Number of dataloader workers.
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
    trainer_max_gradient_norm: float, optional, default 1.0
        Maximum model parameter gradient norm to clip its step size.
        If None, parameter gradient norm clipping is deactivated.
    trainer_max_gradient_value: float, optional, default 10.0
        Maximum model parameter gradient value to clip its step size.
        If None, parameter gradient value clipping is deactivated.
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
    trainer_debug_mode: bool, optional, default False
        Perform model training in debug mode, which check repeatedly for
        'NaN' results.

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for trainer class
    _default_args = {
        'trainer_restart':              False,
        'trainer_max_epochs':           10_000,
        'trainer_properties':           None,
        'trainer_properties_metrics':   {'else': 'mse'},
        'trainer_properties_weights':   {
            'energy': 1., 'forces': 50., 'dipole': 25., 'else': 1.},
        'trainer_batch_size':           32,
        'trainer_train_batch_size':     None,
        'trainer_valid_batch_size':     None,
        'trainer_test_batch_size':      None,
        'trainer_num_batch_workers':    0,
        'trainer_optimizer':            'Adam',
        'trainer_optimizer_args':       {'lr': 0.001},
        'trainer_scheduler':            'ReduceLROnPlateau',
        'trainer_scheduler_args':       {},
        'trainer_ema':                  True,
        'trainer_ema_decay':            0.99,
        'trainer_max_gradient_norm':    1.0,
        'trainer_max_gradient_value':   10.0,
        'trainer_save_interval':        5,
        'trainer_validation_interval':  5,
        'trainer_evaluate_testset':     True,
        'trainer_max_checkpoints':      1,
        'trainer_summary_writer':       True,
        'trainer_print_progress_bar':   True,
        'trainer_debug_mode':           False,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'trainer_restart':              [utils.is_bool],
        'trainer_max_epochs':           [utils.is_integer],
        'trainer_properties':           [utils.is_string_array],
        'trainer_properties_metrics':   [utils.is_dictionary],
        'trainer_properties_weights':   [utils.is_dictionary],
        'trainer_batch_size':           [utils.is_integer],
        'trainer_train_batch_size':     [utils.is_integer, utils.is_None],
        'trainer_valid_batch_size':     [utils.is_integer, utils.is_None],
        'trainer_test_batch_size':      [utils.is_integer, utils.is_None],
        'trainer_num_batch_workers':    [utils.is_integer],
        'trainer_optimizer':            [utils.is_string, utils.is_callable],
        'trainer_optimizer_args':       [utils.is_dictionary],
        'trainer_scheduler':            [utils.is_string, utils.is_callable],
        'trainer_scheduler_args':       [utils.is_dictionary],
        'trainer_ema':                  [utils.is_bool],
        'trainer_ema_decay':            [utils.is_numeric],
        'trainer_max_gradient_norm':    [utils.is_numeric, utils.is_None],
        'trainer_max_gradient_value':   [utils.is_numeric, utils.is_None],
        'trainer_save_interval':        [utils.is_integer],
        'trainer_validation_interval':  [utils.is_integer],
        'trainer_evaluate_testset':     [utils.is_bool],
        'trainer_max_checkpoints':      [utils.is_integer],
        'trainer_summary_writer':       [utils.is_bool],
        'trainer_print_progress_bar':   [utils.is_bool],
        'trainer_debug_mode':           [utils.is_bool],
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
        trainer_batch_size: Optional[int] = None,
        trainer_train_batch_size: Optional[int] = None,
        trainer_valid_batch_size: Optional[int] = None,
        trainer_test_batch_size: Optional[int] = None,
        trainer_num_batch_workers: Optional[int] = None,
        trainer_optimizer: Optional[Union[str, object]] = None,
        trainer_optimizer_args: Optional[Dict[str, float]] = None,
        trainer_scheduler: Optional[Union[str, object]] = None,
        trainer_scheduler_args: Optional[Dict[str, float]] = None,
        trainer_ema: Optional[bool] = None,
        trainer_ema_decay: Optional[float] = None,
        trainer_max_gradient_norm: Optional[float] = None,
        trainer_max_gradient_value: Optional[float] = None,
        trainer_save_interval: Optional[int] = None,
        trainer_validation_interval: Optional[int] = None,
        trainer_evaluate_testset: Optional[bool] = None,
        trainer_max_checkpoints: Optional[int] = None,
        trainer_summary_writer: Optional[bool] = None,
        trainer_print_progress_bar: Optional[bool] = None,
        trainer_debug_mode: Optional[bool] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        verbose: Optional[bool] = True,
        **kwargs,
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

        # Assign DataContainer if not done already
        if data_container is None:
            self.data_container = data.DataContainer(
                config=config,
                **kwargs)

        # Check dataloader batch input
        if self.trainer_train_batch_size is None:
            self.trainer_train_batch_size = self.trainer_batch_size
        if self.trainer_valid_batch_size is None:
            self.trainer_valid_batch_size = self.trainer_batch_size
        if self.trainer_test_batch_size is None:
            self.trainer_test_batch_size = self.trainer_batch_size

        # Get reference data property units
        self.data_units = self.data_container.data_unit_properties

        #########################################
        # # # Prepare Reference Data Loader # # #
        #########################################

        # Initialize training, validation and test data loader
        self.data_container.init_dataloader(
            self.trainer_train_batch_size,
            self.trainer_valid_batch_size,
            self.trainer_test_batch_size,
            num_workers=self.trainer_num_batch_workers,
            device=self.device,
            dtype=self.dtype)

        # Get training and validation data loader
        self.data_train = self.data_container.get_dataloader('train')
        self.data_valid = self.data_container.get_dataloader('valid')

        ##################################
        # # # Check Model Calculator # # #
        ##################################

        # Assign model calculator model if not done already
        if self.model_calculator is None:
            self.model_calculator, _, _ = model.get_model_calculator(
                config=config,
                **kwargs)

        # Check for model ensemble calculator
        if hasattr(self.model_calculator, 'model_ensemble'):
            self.model_ensemble = self.model_calculator.model_ensemble
        else:
            self.model_ensemble = False
        if self.model_ensemble:
            raise SyntaxError(
                "The Trainer class cannot handle model ensembles!")

        # Get model properties
        self.model_properties = self.model_calculator.model_properties
        self.model_units = self.model_calculator.model_unit_properties

        ############################
        # # # Check Properties # # #
        ############################

        # Check property definition for the loss function evaluation
        self.trainer_properties = self.check_properties(
            self.trainer_properties,
            self.data_container.data_properties,
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

        # Show assigned properties, their units and contribution to loss value
        if verbose:
            self.print_trainer_info()

        # Assign potentially new property units to the model
        if hasattr(self.model_calculator, 'set_unit_properties'):
            self.model_calculator.set_unit_properties(self.model_units)

        ###################################################################
        # # # Prepare Optimizer, Scheduler and Gradient Normalization # # #
        ###################################################################

        # Assign model parameter optimizer
        self.trainer_optimizer, trainer_optimizer_args = get_optimizer(
            self.trainer_optimizer,
            self.model_calculator.get_trainable_parameters(),
            self.trainer_optimizer_args)

        # Assign learning rate scheduler
        self.trainer_scheduler, trainer_scheduler_args = get_scheduler(
            self.trainer_scheduler,
            self.trainer_optimizer,
            self.trainer_scheduler_args)

        # Check maximum gradient norm and value clipping input
        if self.trainer_max_gradient_norm is None:
            self.gradient_clipping_norm = False
        else:
            self.gradient_clipping_norm = True
        if self.trainer_max_gradient_value is None:
            self.gradient_clipping_value = False
        else:
            self.gradient_clipping_value = True

        # Update global configuration dictionary with optimizer and scheduler
        # options
        config.update(
            {
                "trainer_optimizer_args": trainer_optimizer_args,
                "trainer_scheduler_args": trainer_scheduler_args,
            },
            config_from=self,
            verbose=verbose
        )

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

        # Initialize checkpoint file manager
        self.filemanager = model.FileManager(
            config=config,
            model_calculator=self.model_calculator,
            max_checkpoints=self.trainer_max_checkpoints,
            verbose=verbose)

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
                test_datasets='test',
                test_batch_size=self.trainer_test_batch_size,
                test_num_batch_workers=self.trainer_num_batch_workers)

        #############################
        # # # Save Model Config # # #
        #############################

        # Save a copy of the current model configuration in the model directory
        self.filemanager.save_config(config)

        return

    def __str__(self):
        messgage = "Trainer"
        if hasattr(self, 'model_calculator'):
            messgage += f" of {str(self.model_calculator):s} model"
        if hasattr(self, 'filemanager'):
            messgage += f" in {str(self.filemanager):s}"
        return messgage

    def run(
        self,
        checkpoint: Optional[Union[str, int]] = 'last',
        restart: Optional[bool] = True,
        reset_best_loss: Optional[bool] = False,
        reset_energy_shift: Optional[bool] = False,
        skip_property_scaling: Optional[bool] = False,
        skip_initial_testing: Optional[bool] = False,
        print_progress: Optional[bool] = True,
        ithread: Optional[int] = None,
        verbose: Optional[bool] = True,
        **kwargs,
    ):
        """
        Train model calculator.

        Parameters
        ----------
        checkpoint: (str, int), optional, default 'last'
            If string and a valid file path, load the respective checkpoint 
            file.
            If string 'best' or 'last', load respectively the best checkpoint 
            file (as with None) or the with the highest epoch number for each.
            If integer, load the checkpoint file of the respective epoch 
            If None, load checkpoint file with best loss function value.
        restart: bool, optional, default True
            If True, restart the model training from the last checkpoint file,
            if available. If False or no checkpoint file exist, start training
            from scratch.
        reset_best_loss: bool, optional, default False
            If False, continue model potential validation from stored best
            loss value. Else, reset best loss value to None.
        reset_energy_shift: bool, optional, default False
            If True and a model checkpoint file is successfully loaded, only
            the (atomic) energy shifts will be initially optimized to match
            best the reference training set. This function is used, e.g., for
            "transfer learning", when a trained model is retrained on a
            different reference data set with a changed total energy shift.
        skip_property_scaling: bool, optional, default False
            Skip the initial model properties scaling factor and shift term
            optimization to match best the reference training set.
        skip_initial_testing: bool, optional, default False
            Skip the initial model evaluation on the reference test set, if the
            model evaluation is enabled anyways (see trainer_evaluate_testset).
        print_progress: bool, optional, default True
            Show  progress report.
        ithread: int, optional, default None
            Thread number

        """

        #################################
        # # # Load Model Checkpoint # # #
        #################################

        # Load checkpoint file
        loaded_checkpoint, checkpoint_file = self.filemanager.load_checkpoint(
            checkpoint_label=checkpoint,
            return_name=True,
            verbose=verbose)

        trainer_epoch_start = 1
        best_loss = None
        if loaded_checkpoint is not None:

            # Assign model parameters
            self.model_calculator.load_state_dict(
                loaded_checkpoint['model_state_dict'])

            self.model_calculator.checkpoint_loaded = True
            self.model_calculator.checkpoint_file = checkpoint_file
            if verbose:
                self.logger.info(
                    f"Checkpoint file '{checkpoint_file:s}' loaded.")

            # If restart training enabled, assign optimizer, scheduler and
            # epoch parameter if available
            optimizer_state, scheduler_state = "None", "None"
            if restart:
                if loaded_checkpoint.get('optimizer_state_dict') is not None:
                    self.trainer_optimizer.load_state_dict(
                        loaded_checkpoint['optimizer_state_dict'])
                    optimizer_state = "Loaded"
                if loaded_checkpoint.get('scheduler_state_dict') is not None:
                    self.trainer_scheduler.load_state_dict(
                        loaded_checkpoint['scheduler_state_dict'])
                    scheduler_state = "Loaded"
                if loaded_checkpoint.get('epoch') is not None:
                    trainer_epoch_start = loaded_checkpoint['epoch'] + 1

                # Initialize best total loss value of validation reference data
                if (
                    reset_best_loss
                    or loaded_checkpoint.get('best_loss') is None
                ):
                    best_loss = None
                elif loaded_checkpoint['best_loss'] == 0.0:
                    best_loss = None
                    self.logger.warning(
                        "Loaded best loss value from the checkpoint file "
                        + f"'{checkpoint_file:s}' is zero indicating an error "
                        + "during generation of the checkpoint file.\n"
                        + "The best loss value is reset!")
                else:
                    best_loss = loaded_checkpoint['best_loss']

            # Print checkpoint file info
            if verbose:               
                if restart:
                    message = "Restart"
                else:
                    message = "Start"
                message += (
                    "training from checkpoint file "
                    + f"'{checkpoint_file:s}':\n"
                    + f" Current Epoch: {trainer_epoch_start:d}\n")
                if best_loss is None:
                    message += " Best validation loss: None\n"
                else:
                    message += f" Best validation loss: {best_loss:.2E}\n"
                message += (
                    f" Optimizer state: {optimizer_state:s}\n"
                    + f" Scheduler state: {scheduler_state:s}")
                self.logger.info(message)

        # Skip if max epochs are already reached
        if trainer_epoch_start > self.trainer_max_epochs:
            if verbose:
                self.logger.info(
                    f"Max Epochs ({self.trainer_max_epochs:d}) already "
                    + f"reached ({trainer_epoch_start:d}).")
            return

        ################################
        # # # Prepare Model Cutoff # # #
        ################################

        # Get model and descriptor cutoffs
        cutoffs = self.model_calculator.get_cutoff_ranges()

        # Set model and descriptor cutoffs for neighbor list calculation
        self.data_train.init_neighbor_list(
            cutoff=cutoffs,
            device=self.device,
            dtype=self.dtype)
        self.data_valid.init_neighbor_list(
            cutoff=cutoffs,
            device=self.device,
            dtype=self.dtype)

        ####################################
        # # # Prepare Property Scaling # # #
        ####################################

        # Either, if a model checkpoint file is loaded, re-optimize energy or
        # atomic energy shifts if specifically requested.
        # If no model checkpoint file is loaded or it is still the first epoch,
        # run model property scaling for the model calculator.
        if self.model_calculator.checkpoint_loaded and reset_energy_shift:

            # Get loaded checkpoint file path
            checkpoint_file = self.model_calculator.checkpoint_file
            if checkpoint_file is None:
                checkpoint_file = "unknown(?)"

            if verbose:
                self.logger.info(
                    "Model calculator checkpoint file already loaded!\n"
                    + f"Checkpoint file: '{checkpoint_file:s}'\n"
                    + "Model energy shifts will be re-evaluated.")
            
            # Get model energy properties
            properties_scaleable = []
            for prop in self.model_calculator.get_scaleable_properties():
                if prop in ['energy', 'atomic_energies']:
                    properties_scaleable.append(prop)

            # Get model energy scaling
            training.set_property_scaling_estimation(
                model_calculator=self.model_calculator,
                data_loader=self.data_train,
                properties=properties_scaleable,
                set_shift_term=True,
                set_scaling_factor=False,
                )

        elif (
            self.model_calculator.checkpoint_loaded
            and not trainer_epoch_start == 1
        ):

            # Get loaded checkpoint file path
            checkpoint_file = self.model_calculator.checkpoint_file
            if checkpoint_file is None:
                checkpoint_file = "unknown(?)"

            if verbose:
                self.logger.info(
                    "Model calculator checkpoint file already loaded!\n"
                    + f"Checkpoint file: '{checkpoint_file:s}'\n"
                    + "No model property scaling parameter are set.")

        elif not skip_property_scaling:

            if verbose:
                self.logger.info(
                    "No Model calculator checkpoint file loaded!\n"
                    "Model property scaling parameter will be set.")

            # Get model property scaling
            training.set_property_scaling_estimation(
                model_calculator=self.model_calculator,
                data_loader=self.data_train,
                properties=self.model_calculator.get_scaleable_properties())

        #############################################
        # # # Prepare Model Training and Metric # # #
        #############################################

        # Initialize training mode for calculator
        self.model_calculator.train()
        torch.set_grad_enabled(True)
        if self.trainer_debug_mode:
            torch.autograd.set_detect_anomaly(True)

        # Reset best property metrics
        metrics_best = self.reset_metrics()
        metrics_best['loss'] = best_loss

        # Define loss function
        loss_fn = torch.nn.SmoothL1Loss(reduction='mean')

        # Get scheduler argument list for correct parameter passing
        scheduler_arguments = (
            self.trainer_scheduler.step.__code__.co_varnames)

        # Count number of training batches
        Nbatch_train = torch.tensor(len(self.data_train), dtype=torch.int64)

        # Initialize training time estimation per epoch
        train_time_estimation = torch.tensor(0.0, dtype=torch.float64)

        # Prepare progress status thread prefix
        if ithread is None:
            thread_prefix = ""
        else:
            thread_prefix = f"(Thread {ithread:d}) "
                
        ##########################
        # # # Start Training # # #
        ##########################

        # Initial evaluation of the test set if requested
        if self.trainer_evaluate_testset and not skip_initial_testing:
            self.tester.test(
                self.model_calculator,
                model_conversion=self.model_conversion,
                test_directory=self.filemanager.best_dir,
                test_plot_correlation=True,
                test_plot_histogram=True,
                test_plot_residual=True,
                **kwargs)

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
                if print_progress and self.trainer_print_progress_bar:
                    utils.print_ProgressBar(
                        ib, Nbatch_train,
                        prefix=f"{thread_prefix:s}Epoch {epoch: 5d}",
                        suffix=(
                            "Complete - Remaining Epoch Time: "
                            + f"{train_time_estimation: 4.1f} s     "
                            ),
                        length=42)

                # Reset optimizer gradients
                self.trainer_optimizer.zero_grad(
                    set_to_none=(
                        not (
                            self.gradient_clipping_norm
                            or self.gradient_clipping_value)
                        )
                    )

                # Predict model properties from data batch
                prediction = self.model_calculator(batch)

                # Check for NaN predictions
                if self.trainer_debug_mode:
                    for prop, item in prediction.items():
                        if torch.any(torch.isnan(item)):
                            raise SyntaxError(
                                f"Property prediction of '{prop:s}' contains "
                                + f"{torch.sum(torch.isnan(item))} elements "
                                + "of value 'NaN'!")

                # Compute total and single loss values for training properties
                metrics_batch = self.compute_metrics(
                    prediction, batch, loss_fn=loss_fn)
                loss = metrics_batch['loss']

                # Check for NaN loss value
                if torch.isnan(loss):
                    raise SyntaxError(
                        "Loss value of training batch is 'NaN'!")

                # Predict parameter gradients by backwards propagation
                loss.backward()
                
                # Clip parameter gradients norm and values
                if self.gradient_clipping_value:
                    torch.nn.utils.clip_grad_value_(
                        self.model_calculator.parameters(),
                        self.trainer_max_gradient_value)
                if self.gradient_clipping_norm:
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
            if 'metrics' in scheduler_arguments:
                self.trainer_scheduler.step(metrics_train['loss'])
            else:
                self.trainer_scheduler.step()

            # Stop epoch train timer
            train_time_epoch_end = time.time()
            train_time_epoch = train_time_epoch_end - train_time_epoch_start

            # Eventually show final training progress
            if print_progress and self.trainer_print_progress_bar:
                utils.print_ProgressBar(
                    Nbatch_train, Nbatch_train,
                    prefix=f"{thread_prefix:s}Epoch {epoch: 5d}",
                    suffix=(
                        "Done - Epoch Time: "
                        + f"{train_time_epoch: 4.1f} s, "
                        + f"Loss: {metrics_train['loss']: 4.4f}   "),
                    length=42)
            elif print_progress:
                utils.print_Progress(
                    f"{thread_prefix:s}Done Epoch {epoch: 5d}, "
                    + f"Epoch Time: {train_time_epoch: 4.1f} s, "
                    + f"Loss: {metrics_train['loss']: 4.4f}   ")

            # Add process to training summary writer
            if self.trainer_summary_writer:
                for prop, value in metrics_train.items():
                    if utils.is_dictionary(value):
                        for metric, val in value.items():
                            self.summary_writer.add_scalar(
                                '_'.join(('train', prop, metric)),
                                metrics_train[prop][metric],
                                global_step=epoch)
                    else:
                        self.summary_writer.add_scalar(
                            '_'.join(('train', prop)),
                            metrics_train[prop],
                            global_step=epoch)

            # Save current model each interval
            if not (epoch % self.trainer_save_interval):
                self.filemanager.save_checkpoint(
                    model_calculator=self.model_calculator,
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

                # If EMA is active
                if self.trainer_ema:
                    # Store last model parameter set
                    self.trainer_ema_model.store(
                        self.model_calculator.parameters())
                    # Load EMA model parameter set
                    self.trainer_ema_model.copy_to(
                        self.model_calculator.parameters())
    
                # Loop over validation batches
                for batch in self.data_valid:

                    # Predict model properties from data batch
                    prediction = self.model_calculator(batch)

                    # Compute total and single loss values for
                    # validation properties
                    metrics_batch = self.compute_metrics(
                        prediction, batch,
                        loss_fn=loss_fn, loss_only=False)

                    # Update average metrics
                    self.update_metrics(metrics_valid, metrics_batch)

                # Change back to training mode for calculator
                self.model_calculator.train()

                # Add process to training summary writer
                if self.trainer_summary_writer:
                    for prop, value in metrics_valid.items():
                        if utils.is_dictionary(value):
                            for metric, val in value.items():
                                self.summary_writer.add_scalar(
                                    '_'.join(('valid', prop, metric)),
                                    metrics_valid[prop][metric],
                                    global_step=epoch)
                        else:
                            self.summary_writer.add_scalar(
                                '_'.join(('valid', prop)),
                                metrics_valid[prop],
                                global_step=epoch)

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
                        model_calculator=self.model_calculator,
                        optimizer=self.trainer_optimizer,
                        scheduler=self.trainer_scheduler,
                        epoch=epoch,
                        best=True,
                        best_loss=best_loss)

                    # Evaluation of the test set if requested
                    if self.trainer_evaluate_testset:
                        self.tester.test(
                            self.model_calculator,
                            model_conversion=self.model_conversion,
                            test_directory=self.filemanager.best_dir,
                            test_plot_correlation=True,
                            test_plot_histogram=True,
                            test_plot_residual=True,
                            verbose=verbose,
                            **kwargs)

                    # Add process to training summary writer
                    if self.trainer_summary_writer:
                        for prop, value in metrics_best.items():
                            if utils.is_dictionary(value):
                                for metric, val in value.items():
                                    self.summary_writer.add_scalar(
                                        '_'.join(('best', prop, metric)),
                                        metrics_best[prop][metric],
                                        global_step=epoch)
                            else:
                                self.summary_writer.add_scalar(
                                    '_'.join(('best', prop)),
                                    metrics_best[prop],
                                    global_step=epoch)

                    # If EMA is active
                    if self.trainer_ema:
                        # Restore last model parameter set
                        self.trainer_ema_model.restore(
                            self.model_calculator.parameters())

                # Print validation metrics summary
                if print_progress and verbose:

                    msg = (
                        f"{thread_prefix:s}Summary Epoch: {epoch:d}/"
                        + f"{self.trainer_max_epochs:d}\n"
                        + "  Loss   train / valid: "
                        + f" {metrics_train['loss']:.2E} /"
                        + f" {metrics_valid['loss']:.2E}"
                        + f"  Best Loss valid: {metrics_best['loss']:.2E}\n"
                        + f"  Property Metrics (valid):\n")
                    for prop in self.trainer_properties:
                        msg += (
                            f"    {prop:10s}  MAE (Best) / RMSE (Best): "
                            + f" {metrics_valid[prop]['mae']:.2E}"
                            + f" ({metrics_best[prop]['mae']:.2E}) /"
                            + f" {np.sqrt(metrics_valid[prop]['mse']):.2E}"
                            + f" ({np.sqrt(metrics_best[prop]['mse']):.2E})"
                            + f" {self.model_units[prop]:s}\n")
                    self.logger.info(msg)

                elif print_progress:

                    utils.print_Progress(
                        f"{thread_prefix:s}Summary Epoch: {epoch:d}/"
                        + f"{self.trainer_max_epochs:d}: "
                        + "  Loss   train / valid: "
                        + f" {metrics_train['loss']:.2E} /"
                        + f" {metrics_valid['loss']:.2E}"
                        + f"  Best Loss valid: {metrics_best['loss']:.2E}")

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
        for prop in model_properties:
            if prop in data_properties:
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
            Default: mean squared error (mse)
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
    ) -> ([Dict[str, str], Dict[str, str], Dict[str, float]]):
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
            Dictionary of model property units
        dict(str, str)
            Dictionary of data property units
        dict(str, float)
            Dictionary of model to data property unit conversion factors

        """

        # Initialize model to data unit conversion dictionary
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

        # Iterate over training properties
        for prop in trainer_properties:

            # Check model property unit
            if model_units.get(prop) is None:
                model_units[prop] = data_units.get(prop)
                model_conversion[prop] = 1.0
            else:
                model_conversion[prop], _ = utils.check_units(
                    model_units[prop], data_units.get(prop))

        return model_units, data_units, model_conversion

    def reset_metrics(self):
        """
        Reset metrics dictionary.

        Returns
        -------
        dict(str, float)
            Metric values dictionary set to zero.

        """

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
        """
        Update metrics dictionary.

        Parameters
        ----------
        metrics: dict
            Metrics dictionary
        metrics_update: dict
            Metrics dictionary to update

        Returns
        -------
        dict(str, float)
            Updated metric values dictionary with new batch results

        """

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
        """
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
        dict(str, float)
            Metric values dictionary

        """

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

            # Check for NaN loss value
            if torch.isnan(metrics[prop]['loss']):
                raise SyntaxError(
                    f"Loss value for property '{prop:s}' is 'NaN'!")

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

    def print_trainer_info(self):
        """
        Print trainer properties summary

        """

        message = (
            f" {'Property ':<17s} |"
            + f" {'Model Unit':<12s} |"
            + f" {'Data Unit':<12s} |"
            + f" {'Conv. fact.':<12s} |"
            + f" {'Loss Metric':<12s} |"
            + f" {'Loss Weight':<12s}\n")
        message += "-"*len(message) + "\n"
        for prop, model_unit in self.model_units.items():
            if self.data_units.get(prop) is None:
                data_unit = "None"
            else:
                data_unit = self.data_units.get(prop)
            message += (
                f" {prop:<17s} |"
                + f" {model_unit:<12s} |"
                + f" {data_unit:<12s} |")
            if self.model_conversion.get(prop) is None:
                message += f" {'None':<12s} |"
            else:
                message += f" {self.model_conversion.get(prop):>12.4e} |"
            if prop in self.trainer_properties:
                message += f" {self.trainer_properties_metrics[prop]:<12s} |"
            else:
                message += f" {'':<12s} |"
            if prop in self.trainer_properties:
                message += f" {self.trainer_properties_weights[prop]:> 11.4f}"
            message += "\n"
        self.logger.info(
            "Model and data properties, and model to data conversion factors "
            + "(model to data)."
            + "\nError metric and weight are shown for the properties "
            + "included in the training loss function.\n"
            + message)

        return

    def reset_logger(
        self,
        level: Optional[Callable] = None,
        stream: Optional[Union[Callable, str]] = sys.stdout,
        verbose: Optional[bool] = True,
    ):
        """
        Reset Training class logger, e.g, in model ensemble training to stream
        model training to model specific output files.

        Parameters
        ----------
        level: callable, optional, default 'logging.INFO'
            Print level for output (e.g. logging.DEBUG, logging.INFO, ...)
        stream: (callable, str), optional, default 'sys.stdout'
            Output channel to print or file path to write
        verbose: bool, optional, default True
            Start logger output with header for information

        """

        # Set logger options
        self.logger = utils.set_logger(
            logging.getLogger(self.name),
            level=level,
            stream=stream,
            verbose=verbose)

        return
