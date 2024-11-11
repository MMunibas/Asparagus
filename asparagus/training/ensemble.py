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

from .tester import Tester

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
    data_container: data.DataContainer, optional, default None
        Reference data container object providing training, validation and
        test data for the model training.
    model_calculator: torch.nn.Module, optional, default None
        Model calculator to train matching model properties with the reference
        data set. If not provided, the model calculator will be initialized
        according to config input.
    ensemble_number: int, optional, default None
        Number of NNP models in the model ensemble
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
        data_container: Optional[data.DataContainer] = None,
        model_calculator: Optional[torch.nn.Module] = None,
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
            self.model_calculator = asparagus.get_model_calculator(
                config=config,
                **kwargs)

        # Manipulate model directory of the model calculator
        model_directory = self.model_calculator.model_directory
        print(model_directory)

        return
