import os
import sys
import inspect
import logging
import datetime
from typing import Optional, List, Dict, Tuple, Union, Callable

import numpy as np 

import ase

import torch
#import pytorch_lightning as pl
#from pytorch_lightning.accelerators import GPUAccelerator #TODO

from . import settings
from . import utils
from . import data
from . import model
from . import train as training
from . import interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['Asparagus']

class Asparagus():
    """
    Asparagus main class

    Parameters
    ----------
    config: (str, dict, object)
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    kwargs: dict, optional, default {}
        Additional model keyword input parameter

    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):

        super().__init__()

        #############################
        # # # Check Model Input # # #
        #############################

        # Initialize model parameter configuration dictionary
        # Keyword arguments overwrite entries in the configuration dictionary
        self.config = settings.get_config(
            config, config_file, config_from=self, **kwargs)

        # Check model parameter configuration and set default
        self.config.check(
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Get configuration file path
        self.config_file = self.config.get('config_file')

        # Print Asparagus header
        utils.header(self.config_file)

        ###########################
        # # # Class Parameter # # #
        ###########################

        # DataContainer of reference data
        self.data_container = None
        # Model calculator
        self.model_calculator = None
        # Model trainer
        self.trainer = None
        # Model testing
        self.tester = None

        return

    def __str__(self):
        """
        Return class descriptor
        """
        return "Asparagus Main"

    def __getitem__(self, args):
        """
        Return item(s) from configuration dictionary
        """
        return self.config.get(args)

    def get(self, args):
        """
        Return item(s) from configuration dictionary
        """
        return self.config.get(args)

    def set_data_container(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        data_container: Optional[data.DataContainer] = None,
        data_file: Optional[str] = None,
        data_file_format: Optional[str] = None,
        **kwargs
    ):
        """
        Set and, eventually, initialize DataContainer as class variable.

        Parameter:
        ----------
        config: (str, dict, object), optional, default None
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters.
        config_file: str, optional, default None
            Path to json file (str)
        data_container: data.DataContainer, optional, default None
            DataContainer object to assign to the Asparagus object
        data_file: str, optional, default None
            Reference Asparagus database file
        data_file_format: str, optional, default None
            Reference Asparagus database file format

        """
        
        ######################################
        # # # Check Data Container Input # # #
        ######################################
        
        if config is None:
            config = settings.get_config(
                self.config, config_file, config_from=self)
        else:
            config = settings.get_config(
                config, config_file, config_from=self)

        # Check model parameter configuration and set default
        config_update = config.set(
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Update configuration dictionary
        config.update(config_update)

        #################################
        # # # Assign Data Container # # #
        #################################

        self._set_data_container(
            config,
            data_container=data_container,
            data_file=data_file,
            data_file_format=data_file_format,
            **kwargs)
        
        return
    
    def _set_data_container(
        self,
        config: object,
        data_container: Optional[data.DataContainer] = None,
        data_file: Optional[str] = None,
        data_file_format: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize and set DataContainer as class variable

        Parameter:
        ----------
        config: object
            Asparagus parameter settings.config class object
        data_container: data.DataContainer, optional, default None
            DataContainer object to assign to the Asparagus object
        data_file: str, optional, default None
            Reference Asparagus database file
        data_file_format: str, optional, default None
            Reference Asparagus database file format
        
        """
        
        #################################
        # # # Assign Data Container # # #
        #################################

        # Check custom model calculator
        if data_container is not None:
            
            # Assign data container
            self.data_container = data_container

            # Add data container info to configuration dictionary
            if hasattr(data_container, "get_info"):
                config.update(data_container.get_info())
        
        else:
        
            # Get model calculator
            data_container = self._get_data_container(
                config,
                data_file=data_file,
                data_file_format=data_file_format,
                **kwargs)
            
            # Assign data container
            self.data_container = data_container

        return
    
    def get_data_container(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        data_file: Optional[str] = None,
        data_file_format: Optional[str] = None,
        **kwargs
    ) -> data.DataContainer:
        """
        Initialize and return DataContainer.

        Parameter:
        ----------
        config: (str, dict, object), optional, default None
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters.
        config_file: str, optional, default None
            Path to json file (str)
        data_file: str, optional, default None
            Reference Asparagus database file
        data_file_format: str, optional, default None
            Reference Asparagus database file format

        Returns
        -------
        data.DataContainer
            Asparagus data container object

        """
        
        ######################################
        # # # Check Data Container Input # # #
        ######################################
        
        if config is None:
            config = settings.get_config(
                self.config, config_file, config_from=self)
        else:
            config = settings.get_config(
                config, config_file, config_from=self)

        # Check model parameter configuration and set default
        config_update = config.set(
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Update configuration dictionary
        config.update(config_update)

        ##################################
        # # # Prepare Data Container # # #
        ##################################
        
        data_container = self._get_data_container(
            config,
            data_file=data_file,
            data_file_format=data_file_format,
            **kwargs)

        return data_container

    def _get_data_container(
        self,
        config: object,
        data_file: Optional[str] = None,
        data_file_format: Optional[str] = None,
        **kwargs
    ) -> data.DataContainer:
        """
        Initialize and set DataContainer as class variable

        Parameter:
        ----------
        config: object
            Asparagus parameter settings.config class object
        data_file: str, optional, default None
            Reference Asparagus database file
        data_file_format: str, optional, default None
            Reference Asparagus database file format

        Returns
        -------
        data.DataContainer
            Asparagus data container object

        """

        ##################################
        # # # Prepare Data Container # # #
        ##################################

        data_container = data.DataContainer(
            config,
            data_file=data_file,
            data_file_format=data_file_format,
            **kwargs)
        
        return data_container

    def set_model_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_calculator: Optional[torch.nn.Module] = None,
        model_type: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'best',
        **kwargs,
    ):
        """
        Set and, eventually, initialize the calculator model class object

        Parameters
        ----------
        config: (str, dict, object), optional, default 'self.config'
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_calculator: torch.nn.Module, optional, default None
            Model calculator object to assign as class model calculator.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_checkpoint: (int, str), optional, default 'best'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object

        """

        ########################################
        # # # Check Model Calculator Input # # #
        ########################################

        if config is None:
            config = settings.get_config(
                self.config, config_file, config_from=self)
        else:
            config = settings.get_config(
                config, config_file, config_from=self)

        # Check model parameter configuration and set default
        config_update = config.set(
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Update configuration dictionary
        config.update(config_update)

        ###################################
        # # # Assign Model Calculator # # #
        ###################################
        
        self._set_model_calculator(
            config,
            model_calculator=model_calculator,
            model_type=model_type,
            model_checkpoint=model_checkpoint,
            **kwargs)
        
        return

    def _set_model_calculator(
        self,
        config: object,
        model_calculator: Optional[torch.nn.Module] = None,
        model_type: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'best',
        **kwargs,
    ):
        """
        Initialize and set the calculator model class object

        Parameters
        ----------
        config: object
            Asparagus parameter settings.config class object
        model_calculator: torch.nn.Module, optional, default None
            Model calculator object to assign as class model calculator.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_checkpoint: (int, str), optional, default 'best'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object

        """
        
        ###################################
        # # # Assign Model Calculator # # #
        ###################################

        # Get model calculator
        model_calculator, restart = self._get_model_calculator(
            config,
            model_calculator=model_calculator,
            model_type=model_type,
            model_checkpoint=model_checkpoint,
            **kwargs)
        
        # Assign model calculator
        self.model_calculator = model_calculator
        self.model_restart = restart

        return

    def get_model_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_calculator: Optional[torch.nn.Module] = None,
        model_type: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'best',
        **kwargs,
    ) -> torch.nn.Module:
        """
        Return calculator model class object

        Parameters
        ----------
        config: (str, dict, object), optional, default 'self.config'
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_calculator: torch.nn.Module, optional, default None
            Model calculator object.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_checkpoint: (int, str), optional, default 'best'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object

        """

        ########################################
        # # # Check Model Calculator Input # # #
        ########################################

        # Assign model parameter configuration library
        if config is None:
            config = settings.get_config(
                self.config, config_file, config_from=self)
        else:
            config = settings.get_config(
                config, config_file, config_from=self)

        # Check model parameter configuration and set default
        config_update = config.set(
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Update configuration dictionary
        config.update(config_update)

        ####################################
        # # # Prepare Model Calculator # # #
        ####################################

        # Get model calculator
        model_calculator, restart = self._get_model_calculator(
            config=config,
            model_calculator=model_calculator,
            model_type=model_type,
            model_checkpoint=model_checkpoint,
            **kwargs,
            )

        return model_calculator

    def _get_model_calculator(
        self,
        config: object,
        model_calculator: Optional[torch.nn.Module] = None,
        model_type: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'last',
        **kwargs,
    ) -> (torch.nn.Module, bool):
        """
        Return calculator model class object and restart flag.

        Parameters
        ----------
        config: object
            Asparagus parameter settings.config class object
        model_calculator: torch.nn.Module, optional, default None
            Model calculator object.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_checkpoint: int, optional, default 'last'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        
        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object
        bool
            Restart flag, True if checkpoint file is loaded.

        """

        ####################################
        # # # Prepare Model Calculator # # #
        ####################################
        
        # Assign model calculator
        model_calculator, checkpoint = model.get_model_calculator(
            config,
            model_calculator=model_calculator,
            model_type=model_type,
            model_checkpoint=model_checkpoint,
            **kwargs)

        # Load model checkpoint file
        if checkpoint is None:
            logger.info(f"INFO:\nNo checkpoint file loaded.\n")
            restart = False
        else:
            model_calculator.load_state_dict(
                checkpoint['model_state_dict'])
            logger.info(f"INFO:\nCheckpoint file loaded.\n")
            restart = True

        return model_calculator, restart

    def get_trainer(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ) -> training.Trainer:
        """
        Initialize and return model calculator trainer.

        Parameters
        ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to config json file (str)

        Returns:
        --------
        train.Trainer
            Model calculator trainer object

        """

        ###############################
        # # # Check Trainer Input # # #
        ###############################

        # Assign model parameter configuration library
        if config is None:
            config = settings.get_config(
                self.config, config_file, config_from=self)
        else:
            config = settings.get_config(
                config, config_file, config_from=self)

        # Check model parameter configuration and set default
        config_update = config.set(
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))
        
        # Update configuration dictionary
        config.update(config_update)

        ###########################################
        # # # Assign Model Calculator Trainer # # #
        ###########################################

        # Assign model calculator trainer
        if self.trainer is None:
            trainer = self._get_trainer(
                config,
                **kwargs,)
        else:
            trainer = self.trainer

        return trainer

    def _get_trainer(
        self,
        config: object,
        **kwargs,
    ) -> training.Trainer:
        """
        Initialize and return model calculator trainer.

        Parameters
        ----------
        config: object
            Asparagus parameter settings.config class object

        Returns:
        --------
        train.Trainer
            Model calculator trainer object

        """

        #################################
        # # # Assign Reference Data # # #
        #################################
        
        if self.data_container is None:
            data_container = self.get_data_container(
                config=config,
                **kwargs)
        else:
            data_container = self.data_container
        
        ###################################
        # # # Assign Model Calculator # # #
        ###################################
        
        if self.model_calculator is None:
            model_calculator = self.get_model_calculator(
                config=config,
                **kwargs)
        else:
            model_calculator = self.model_calculator

        ###########################################
        # # # Assign Model Calculator Trainer # # #
        ###########################################
            
        trainer = training.Trainer(
            config=config,
            data_container=data_container,
            model_calculator=model_calculator,
            **kwargs)

        return trainer

    def train(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Quick command to initialize and start model calculator training.

        Parameters
        ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to config json file (str)

        """
        
        ###########################################
        # # # Assign Model Calculator Trainer # # #
        ###########################################
        
        trainer = self.get_trainer(
            config=config,
            config_file=config_file,
            **kwargs)

        ########################################
        # # # Run Model Calculator Trainer # # #
        ########################################

        trainer.run(**kwargs)

        return

    def get_tester(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ) -> training.Tester:
        """
        Initialize and return model calculator tester.

        Parameters
        ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to config json file (str)

        Returns:
        --------
        train.Tester
            Model calculator tester object

        """

        ##############################
        # # # Check Tester Input # # #
        ##############################

        # Assign model parameter configuration library
        if config is None:
            config = settings.get_config(
                self.config, config_file, config_from=self)
        else:
            config = settings.get_config(
                config, config_file, config_from=self)

        # Check model parameter configuration and set default
        config_update = config.set(
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))
        
        # Update configuration dictionary
        config.update(config_update)

        ##########################################
        # # # Assign Model Calculator Tester # # #
        ##########################################

        # Assign model calculator trainer
        if self.tester is None:
            tester = self._get_tester(
                config,
                **kwargs)
        else:
            tester = self.tester

        return tester

    def _get_tester(
        self,
        config: object,
        **kwargs,
    ) -> training.Tester:
        """
        Initialize and return model calculator tester.

        Parameters
        ----------
        config: object
            Asparagus parameter settings.config class object

        Returns:
        --------
        train.Tester
            Model calculator tester object

        """

        #################################
        # # # Assign Reference Data # # #
        #################################

        if self.data_container is None:
            data_container = self.get_data_container(
                config=config,
                **kwargs)
        else:
            data_container = self.data_container

        ###########################################
        # # # Assign Model Calculator Trainer # # #
        ###########################################

        tester = training.Tester(
            config=config,
            data_container=data_container,
            **kwargs)

        return tester

    def test(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        Quick command to initialize and start model calculator training.

        Parameters
        ----------
        config: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to config json file (str)

        """

        ###################################
        # # # Assign Model Calculator # # #
        ###################################
        
        if self.model_calculator is None:
            model_calculator = self.get_model_calculator(
                config=config,
                config_file=config_file,
                **kwargs)
        else:
            model_calculator = self.model_calculator
        
        ##########################################
        # # # Assign Model Calculator Tester # # #
        ##########################################
        
        tester = self.get_tester(
            config=config,
            config_file=config_file,
            **kwargs)

        #######################################
        # # # Run Model Calculator Tester # # #
        #######################################

        tester.test(
            model_calculator,
            **kwargs)

        return

    def get_ase_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'best',
        **kwargs,
    ) -> ase.calculators.calculator.Calculator:
        """
        Return ASE calculator class object of the model calculator

        Parameter
        ---------
        config: (str, dict, object), optional, default 'self.config'
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_checkpoint: (int, str), optional, default 'best'
            If None or 'best', load best model checkpoint. 
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.

        Returns
        -------
        ase.calculators.calculator.Calculator
            ASE calculator instance of the model calculator

        """

        ###################################
        # # # Assign Model Calculator # # #
        ###################################
        
        if self.model_calculator is None:
            model_calculator = self.get_model_calculator(
                config=config,
                config_file=config_file,
                model_checkpoint=model_checkpoint,
                **kwargs)
        else:
            model_calculator = self.model_calculator

        ##################################
        # # # Prepare ASE Calculator # # #
        ##################################

        ase_calculator = interface.ASE_Calculator(
            model_calculator,
            **kwargs)

        return ase_calculator

    def get_pycharmm_calculator(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_checkpoint: Optional[int] = None,
        **kwargs
    ) -> Callable:
        """
        Return PyCHARMM calculator class object of the initialized model 
        calculator.

        Parameters
        ----------
        config: (str, dict, object), optional, default 'self.config'
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters
        config_file: str, optional, default see settings.default['config_file']
            Path to json file (str)
        model_checkpoint: int, optional, default None
            If None, load best model checkpoint. Otherwise define a checkpoint
            index number of the respective checkpoint file.

        Returns
        -------
        callable object
            PyCHARMM calculator object
        """

        ###################################
        # # # Assign Model Calculator # # #
        ###################################
        
        if self.model_calculator is None:
            model_calculator = self.get_model_calculator(
                config=config,
                config_file=config_file,
                model_checkpoint=model_checkpoint,
                **kwargs)
        else:
            model_calculator = self.model_calculator

        #######################################
        # # # Prepare PyCHARMM Calculator # # #
        #######################################

        pycharmm_calculator = interface.PyCharmm_Calculator(
            model_calculator,
            **kwargs)

        return pycharmm_calculator
