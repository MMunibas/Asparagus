import time
import logging
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

    Here, a number ()'ensemble_number') of NNP models ('model_calculator')
    will be trained serially by looping through the models and train for
    'ensemble_epoch_steps' epochs until maximum number of epochs is reached.

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to config json file (str)
    ensemble_model_trainer: training.Trainer, optional, default None
        Single model Trainer class object which is used to train the model
        potential. If None, a trainer will be initialized.
        Here, the Trainer and assigned FileManager parameters will be
        manipulated to train an ensemble of models under consistent conditions.
    ensemble_number: int, optional, default None
        Number of models potentials to train.
    ensemble_epoch_steps: int, optional, default None
        Number of epochs to train single NNP models each step before looping
        through the next until maximum number of epochs is reached

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for trainer class
    _default_args = {
        'ensemble_number':              5,
        'ensemble_epoch_steps':         100,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'ensemble_number':              [utils.is_integer],
        'ensemble_epoch_steps':         [utils.is_integer],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        ensemble_model_trainer: Optional[training.Trainer] = None,
        ensemble_number: Optional[int] = None,
        ensemble_epoch_steps: Optional[int] = None,
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

        ###############################
        # # # Check Model Trainer # # #
        ###############################

        # Assign model calculator model if not done already
        if self.ensemble_model_trainer is None:
            self.ensemble_model_trainer = training.Trainer(
                config=config,
                **kwargs)

        #############################################
        # # # Check Ensemble Trainer Parameters # # #
        #############################################

        # Get model directory as main directory in which to store multiple 
        # models
        self.model_directory = (
            self.ensemble_model_trainer.filemanager.model_directory)

        # Get Trainer checkpoint save interval to match with ensemble learning
        # epoch steps
        self.trainer_save_interval = (
            self.ensemble_model_trainer.trainer_save_interval)

        # Check that 'ensemble_epoch_steps' is a multiple of
        # 'trainer_save_interval' and update in config if necessary.
        ensemble_epoch_steps = int(
            self.trainer_save_interval*
            (self.ensemble_epoch_steps//self.trainer_save_interval)
        )
        if ensemble_epoch_steps != self.ensemble_epoch_steps:
            self.logger.warning(
                "Number of epochs per training step "
                + f"({self.ensemble_epoch_steps:d}) is not a multiple of the "
                + f"Trainer's checkpoint save interval "
                + f"({self.trainer_save_interval:d})!\n"
                + "Number of epochs per training step is changed to "
                + f"({ensemble_epoch_steps:d}).")
            self.ensemble_epoch_steps = ensemble_epoch_steps
            config.update(
                {"ensemble_epoch_steps": self.ensemble_epoch_steps},
                config_from=self
            )

        


        exit()

        return
