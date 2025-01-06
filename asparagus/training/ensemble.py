import os
import time
import queue
import logging
import threading
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import torch

import numpy as np

import asparagus
from asparagus import settings
from asparagus import utils
from asparagus import data
from asparagus import model
from asparagus import training

class EnsembleTrainer:
    """
    Wrapper for ensemle training of NNP models.

    Here, a number ()'ensemble_num_models') of NNP models ('model_calculator')
    will be trained serially by looping through the models and train for
    'ensemble_epochs_step' epochs until maximum number of epochs is reached.

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to config json file (str)
    ensemble_num_models: int, optional, default None
        Number of models potentials to train.
    ensemble_epochs_step: int, optional, default None
        Number of epochs to train single NNP models each step before looping
        through the next until maximum number of epochs is reached.
    ensemble_num_threads: int, optional, default 1
        Number of model training threads for parallel model training.
    trainer_max_epochs: int, optional, default 10000
        Maximum number of training epochs
    trainer_save_interval: int, optional, default 5
        Interval between epoch to save current and best set of model
        parameters.

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for ensemble trainer class
    _default_args = {
        'ensemble_num_models':          5,
        'ensemble_epochs_step':         100,
        'ensemble_num_threads':         1,
        'trainer_max_epochs':           10_000,
        'trainer_save_interval':        5,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'ensemble_num_models':          [utils.is_integer],
        'ensemble_epochs_step':         [utils.is_integer],
        'ensemble_num_threads':         [utils.is_integer],
        'trainer_max_epochs':           [utils.is_integer],
        'trainer_save_interval':        [utils.is_integer],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        data_container: Optional[data.DataContainer] = None,
        model_calculator: Optional[torch.nn.Module] = None,
        ensemble_num_models: Optional[int] = None,
        ensemble_epochs_step: Optional[int] = None,
        ensemble_num_threads: Optional[int] = None,
        trainer_max_epochs: Optional[int] = None,
        trainer_save_interval: Optional[int] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
        **kwargs
    ):
        """
        Initialize Ensemble Model Calculator training.

        """

        ########################################
        # # # Check Ensemble Trainer Input # # #
        ########################################

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
        config.update(config_update)

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

        ##################################
        # # # Check Model Calculator # # #
        ##################################

        # Assign model calculator model if not done already
        if self.model_calculator is None:
            self.model_calculator, _, _ = model.get_model_calculator(
                config=config,
                **kwargs)

        # Initialize checkpoint file manager
        self.filemanager = model.FileManager(
            config=config,
            model_calculator=self.model_calculator)

        # Get model directory as main directory in which to store multiple
        # models
        self.model_directory = self.filemanager.model_directory

        #############################################
        # # # Check Ensemble Trainer Parameters # # #
        #############################################

        # Check that 'ensemble_epochs_step' is a multiple of
        # 'trainer_save_interval' and update in config if necessary.
        ensemble_epochs_step = int(
            self.trainer_save_interval*
            (self.ensemble_epochs_step//self.trainer_save_interval)
        )
        if ensemble_epochs_step != self.ensemble_epochs_step:
            self.logger.warning(
                "Number of epochs per training step "
                + f"({self.ensemble_epochs_step:d}) is not a multiple of the "
                + f"Trainer's checkpoint save interval "
                + f"({self.trainer_save_interval:d})!\n"
                + "Number of epochs per training step is changed to "
                + f"{ensemble_epochs_step:d}.")
            self.ensemble_epochs_step = ensemble_epochs_step

        ##########################################
        # # # Prepare Model Ensemble Configs # # #
        ##########################################

        # Get model subdirectories
        self.ensemble_model_subdirectories = [
            os.path.join(self.model_directory, f"{imodel:d}")
            for imodel in range(self.ensemble_num_models)]

        # Create model subdirectories
        for model_directory in self.ensemble_model_subdirectories:
            if not os.path.exists(model_directory):
                os.makedirs(model_directory)

        # Assign logger output files
        self.ensemble_logger_files = [
            os.path.join(
                self.ensemble_model_subdirectories[imodel],
                f"training_model_{imodel:d}.out")
            for imodel in range(self.ensemble_num_models)]

        # Generate model configurations
        self.ensemble_model_configs = []
        for imodel, model_subdirectory in enumerate(
            self.ensemble_model_subdirectories
        ):
            
            # Get a copy of the configuration dictionary
            model_config = config.config_dict.copy()

            # Reset ensemble flag in model configuration
            model_config['model_ensemble'] = False
            model_config['model_ensemble_num'] = None

            # Store single model configuration
            model_config_file = os.path.join(model_subdirectory, "config.json")
            self.ensemble_model_configs.append(
                settings.get_config(
                    config=model_config,
                    config_file=model_config_file,
                    config_from=f"Ensemble model {imodel:d}",
                    verbose=False,
                    )
                )

        #################################################
        # # # Initialize Ensemble Training Schedule # # #
        #################################################

        # Initialize step number list per model
        self.ensemble_model_step = np.zeros(
            self.ensemble_num_models, dtype=int)

        # Initialize active training flag list per model
        self.ensemble_model_is_training = np.zeros(
            self.ensemble_num_models, dtype=bool)

        # Assign training schedule steps
        self.ensemble_epoch_steps_list = np.arange(
            self.ensemble_epochs_step,
            self.trainer_max_epochs + self.ensemble_epochs_step,
            self.ensemble_epochs_step)
        self.ensemble_epoch_steps_number = len(self.ensemble_epoch_steps_list)

        # Check number of model training threads
        if self.ensemble_num_threads > self.ensemble_num_models:
            self.logger.warning(
                "Number of model training threads "
                + f"({self.ensemble_num_threads:d}) cannot be larger than "
                + " the number of ensemble models "
                + f"({self.ensemble_num_models:d})!\n"
                + "Number of model training threads is lowered to "
                + f"({ensemble_num_models:d}).")
            self.ensemble_num_threads = self.ensemble_num_models

        return

    def run(
        self,
        checkpoint: Optional[Union[str, int, List[str], List[int]]] = 'last',
        restart: Optional[bool] = True,
        reset_best_loss: Optional[bool] = False,
        reset_energy_shift: Optional[bool] = False,
        skip_property_scaling: Optional[bool] = False,
        skip_initial_testing: Optional[bool] = False,
        **kwargs,
    ):
        """
        Train ensemble of model calculators.

        Parameters
        ----------
        checkpoint: (str, int, list(str), list(int)), optional, default 'last'
            If string 'best' or 'last', load respectively the best checkpoint 
            file (as with None) or the with the highest epoch number for each.
            If a string of valid file path, load the checkpoint file in all
            models of the ensemble (not recommended for ensemble learning).
            If a list of strings of valid file paths of the same length as the
            number of models in the ensemble, load the respective checkpoint 
            files.
            If integer or list of integer of the same length as the number of
            models in the ensemble, load the checkpoint file of the respective
            epoch.
        restart: bool, optional, default True
            If True, restart the ensemble model training from each of the last
            checkpoint files, respectively, if available. If False or no
            checkpoint file exist, start each training from scratch.
        reset_best_loss: bool, optional, default False
            If False, continue each model potential validation from stored best
            loss value. Else, reset best loss value to None.
        reset_energy_shift: bool, optional, default False
            If True and a model checkpoint file is successfully loaded, only
            the (atomic) energy shifts will be initially optimized for each 
            model in the ensemble to match best the reference training set.
            This function is used, e.g., for "transfer learning", when a
            trained model is retrained on a different reference data set with a
            changed total energy shift.
        skip_property_scaling: bool, optional, default False
            Skip for each model the initial model properties scaling factor and
            shift term optimization to match best the reference training set.
        skip_initial_testing: bool, optional, default False
            Skip for each model  the initial model evaluation on the reference
            test set, if the model evaluation is enabled anyways (see
            trainer_evaluate_testset).

        """

        # Check checkpoint input
        if checkpoint is None:
            checkpoint_list = [None]*self.ensemble_num_models
        elif utils.is_string(checkpoint) or utils.is_integer(checkpoint):
            checkpoint_list = [checkpoint]*self.ensemble_num_models
        elif (
            utils.is_string_array(checkpoint)
            or utils.is_integer_array(checkpoint)
        ):
            if len(checkpoint) != self.ensemble_num_models:
                raise ValueError(
                    "Checkpoint in 'checkpoint' is a list but of different "
                    + f"length ({len(checkpoint):d}) than the number of "
                    + f"ensemble models ({self.ensemble_num_models:d})!")
            checkpoint_list = checkpoint
        else:
            raise ValueError(
                "Checkpoint input 'checkpoint' is of unkown format.\n"
                + "Provide a string, integer or list of strings or integers.")

        # Initialize the multithreading lock
        self.lock = threading.Lock()

        # Print ensemble training information
        self.logger.info(
            "Start model ensemble training:\n"
            + f" Number of models: {self.ensemble_num_models:d}\n"
            + f" Models directory: {self.model_directory:s}\n"
            + f" Epochs per training step: {self.ensemble_epochs_step:d}\n"
            + f" Number of training threads: {self.ensemble_num_threads:d}")

        # Run sampling over sample systems
        if self.ensemble_num_threads == 1:

            self.run_training(
                checkpoint_list,
                restart,
                reset_best_loss,
                reset_energy_shift,
                skip_property_scaling,
                skip_initial_testing)
        
        else:

            # Create threads
            threads = [
                threading.Thread(
                    target=self.run_training, 
                    args=(
                        checkpoint_list,
                        restart,
                        reset_best_loss,
                        reset_energy_shift,
                        skip_property_scaling,
                        skip_initial_testing, ),
                    kwargs={
                        'ithread': ithread}
                    )
                for ithread in range(self.ensemble_num_threads)]

            # Start threads
            for thread in threads:
                thread.start()

            # Wait for threads to finish
            for thread in threads:
                thread.join()
        
        return

    def run_training(
        self, 
        checkpoint_list: Union[List[str], List[int]],
        restart: bool,
        reset_best_loss: bool,
        reset_energy_shift: bool,
        skip_property_scaling: bool,
        skip_initial_testing: bool,
        ithread: Optional[int] = None,
    ):
        """
        Run training step for one model calculator in ensemble queue.
        
        Parameters
        ----------
        checkpoint_list: list(str, int)
            Checkpoint option only for the first training step of each model.
            For later steps, checkpoint option is 'last'.
        restart: bool
            Restart option only for the first training step of each model.
            For later steps, restart is True.
        reset_best_loss: bool
            Reset best lost option only for the first training step of each
            model. For later steps, parameter is False.
        reset_energy_shift: bool
            Reset energy shift option only for the first training step of each
            model. For later steps, parameter is False.
        skip_property_scaling: bool
            Skip property scaling option only for the first training step of 
            each model. For later steps, parameter is True.
        skip_initial_testing: bool
            Skip initial testing option only for the first training step of 
            each model. For later steps, parameter is True.
        ithread: int, optional, default None
            Thread number

        """

        while self.keep_going():
            
            # Select next model and step
            imodel, istep = self.next_step()

            # Print training thread information
            if ithread is None:
                self.logger.info(
                    f"Start training of model {imodel:d} up to epoch "
                    + f"{self.ensemble_epoch_steps_list[istep]:d}.")
            else:
                self.logger.info(
                    f"Start training of model {imodel:d} "
                    + f"(thread {ithread:d}) up to epoch "
                    + f"{self.ensemble_epoch_steps_list[istep]:d}.")

            # Initialize ensemble model potential
            model_calculator, _, _ = model.get_model_calculator(
                config=self.ensemble_model_configs[imodel],
                model_directory=self.ensemble_model_subdirectories[imodel],
                verbose=False)

            # Initialize Trainer instance
            trainer = training.Trainer(
                config=self.ensemble_model_configs[imodel],
                data_container=self.data_container,
                model_calculator=model_calculator,
                trainer_max_epochs=self.ensemble_epoch_steps_list[istep],
                trainer_evaluate_testset=False,
                trainer_print_progress_bar=False,
                verbose=False)

            # Run training
            if istep:
                trainer.run(
                    checkpoint='last',
                    restart=True,
                    reset_best_loss=False,
                    reset_energy_shift=False,
                    skip_property_scaling=True,
                    skip_initial_testing=True,
                    ithread=ithread,
                    verbose=False)
            else:
                trainer.run(
                    checkpoint=checkpoint_list[imodel],
                    restart=restart,
                    reset_best_loss=reset_best_loss,
                    reset_energy_shift=reset_energy_shift,
                    skip_property_scaling=skip_property_scaling,
                    skip_initial_testing=skip_initial_testing,
                    ithread=ithread,
                    verbose=False)

            # Set model training status to idle
            with self.lock:
                self.ensemble_model_is_training[imodel] = False
                self.ensemble_model_step[imodel] += 1

            # Print training thread information
            if ithread is None:
                self.logger.info(
                    f"Done training of model {imodel:d} up to epoch "
                    + f"{self.ensemble_epoch_steps_list[istep]:d}.")
            else:
                self.logger.info(
                    f"Done training of model {imodel:d} "
                    + f"(thread {ithread:d}) up to epoch "
                    + f"{self.ensemble_epoch_steps_list[istep]:d}.")

        return
        
    def keep_going(self):
        """
        Check if training steps are left.
        
        """
        
        # Hold further threads until check is complete
        with self.lock:

            # Models, which are currently not actively training
            idle_models = np.logical_not(self.ensemble_model_is_training)

            # Get training steps for idle models
            idle_model_steps = self.ensemble_model_step[idle_models]

            if np.any(
                idle_model_steps < self.ensemble_epoch_steps_number
            ):
                return True
            else:
                return False

    def next_step(self):
        """
        Select next training step
        
        """

        # Hold further threads until selection is done
        with self.lock:

            # Models, which are currently not actively training
            idle_models = np.logical_not(self.ensemble_model_is_training)

            # Select model with lowest step number which is currently not
            # actively training
            min_step = np.min(self.ensemble_model_step[idle_models])
            imodel = np.where(
                np.logical_and(
                    idle_models,
                    self.ensemble_model_step == min_step
                )
            )[0]
            if len(imodel):
                imodel = imodel[0]
            else:
                raise SyntaxError(
                    "No inactive model found!")

            # Get next epoch step of the selected model
            istep = self.ensemble_model_step[imodel]

            # Set model status to actively training
            self.ensemble_model_is_training[imodel] = True

        return imodel, istep
