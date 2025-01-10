import sys
import logging
from typing import Optional, Union, Any, List, Dict, Callable

import ase

import torch

from asparagus import settings
from asparagus import utils
from asparagus import data
from asparagus import model
from asparagus import training
from asparagus import interface

__all__ = ['Asparagus']


class Asparagus():
    """
    Asparagus main class

    Parameters
    ----------
    config: (str, dict, settings.Configuration), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    kwargs: dict, optional, default {}
        Additional model keyword input parameter

    """

    name = f"{__name__:s}: {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    def __init__(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
        config_file: Optional[str] = None,
        **kwargs
    ):

        super().__init__()

        #############################
        # # # Check Model Input # # #
        #############################

        # Initialize model parameter configuration dictionary
        # Keyword arguments overwrite entries in the configuration dictionary
        config = settings.get_config(
            config, config_file, config_from=self, **kwargs)

        # Check model parameter configuration and set default
        config.check(
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))

        # Get configuration file path
        self.config_file = config.get('config_file')

        # Print Asparagus header
        self.logger.info(utils.get_header(self.config_file))

        ###########################
        # # # Class Parameter # # #
        ###########################

        # DataContainer of reference data
        self.data_container = None
        # Model calculator
        self.model_calculator = None

        return

    def __str__(self) -> str:
        """
        Return class descriptor
        """
        return "Asparagus Main"

    def __getitem__(self, args: str) -> Any:
        """
        Return item(s) from configuration dictionary
        """
        config = settings.get_config(self.config)
        return config.get(args)

    def get(self, args: str) -> Any:
        """
        Return item(s) from configuration dictionary
        """
        config = settings.get_config(self.config)
        return config.get(args)

    def set_data_container(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
        config_file: Optional[str] = None,
        data_container: Optional[data.DataContainer] = None,
        **kwargs
    ):
        """
        Set and, eventually, initialize DataContainer as class variable.

        Parameter:
        ----------
        config: (str, dict, settings.Config), optional, default None
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters.
        config_file: str, optional, default None
            Path to json file (str)
        data_container: data.DataContainer, optional, default None
            DataContainer object to assign to the Asparagus object

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

        # Check custom data container
        if data_container is not None:

            # Assign data container
            self.data_container = data_container

            # Add data container info to configuration dictionary
            if hasattr(data_container, "get_info"):
                config.update(data_container.get_info())

        else:

            # Get data container
            data_container = self._get_data_container(
                config,
                data_container=data_container,
                **kwargs)

            # Assign data container
            self.data_container = data_container

        return

    def get_data_container(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
        config_file: Optional[str] = None,
        **kwargs
    ) -> data.DataContainer:
        """
        Initialize and return DataContainer.

        Parameter:
        ----------
        config: (str, dict, settings.Configuration), optional, default None
            Either the path to json file (str), dictionary (dict) or
            settings.config class object of model parameters.
        config_file: str, optional, default None
            Path to json file (str)

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
            **kwargs)

        return data_container

    def _get_data_container(
        self,
        config: settings.Configuration,
        **kwargs
    ) -> data.DataContainer:
        """
        Initialize and set DataContainer as class variable

        Parameter:
        ----------
        config: settings.Configuration
            Asparagus parameter settings.config class object

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
            **kwargs)

        return data_container

    def set_model_calculator(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
        config_file: Optional[str] = None,
        model_calculator:
            Optional[Union[List[torch.nn.Module], torch.nn.Module]] = None,
        model_type: Optional[str] = None,
        model_directory: Optional[str] = None,
        model_ensemble: Optional[bool] = None,
        model_ensemble_num: Optional[int] = None,
        model_checkpoint: Optional[Union[int, str]] = None,
        model_compile: Optional[bool] = None,
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
        model_calculator: (torch.nn.Module, list(torch.nn.Module)),
                optional, default None
            Model calculator object or list of model calculator objects.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_directory: str, optional, default None
            Model directory that contains checkpoint and log files.
        model_ensemble: bool, optional, default None
            Expect a model calculator ensemble. If None, check config or
            assume as False.
        model_ensemble_num: int, optional, default None
            Number of model calculator in ensemble.
        model_checkpoint: (int, str), optional, default None
            If None or 'best', load best model checkpoint.
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        model_compile: bool, optional, default None
            If True, the model calculator will get compiled at the first
            call to enhance the performance. Generally not applicable during
            training where multiple backwards calls are done (e.g. energy
            gradient and loss metrics gradient)

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

        # Get model calculator
        model_calculator = self._get_model_calculator(
            config,
            model_calculator=model_calculator,
            model_type=model_type,
            model_directory=model_directory,
            model_ensemble=model_ensemble,
            model_ensemble_num=model_ensemble_num,
            model_checkpoint=model_checkpoint,
            model_compile=model_compile,
            **kwargs,
            )

        # Assign model calculator
        self.model_calculator = model_calculator

        return

    def get_model_calculator(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
        config_file: Optional[str] = None,
        model_calculator: 
            Optional[Union[List[torch.nn.Module], torch.nn.Module]] = None,
        model_type: Optional[str] = None,
        model_directory: Optional[str] = None,
        model_ensemble: Optional[bool] = None,
        model_ensemble_num: Optional[int] = None,
        model_checkpoint: Optional[Union[int, str]] = None,
        model_compile: Optional[bool] = None,
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
        model_calculator: (torch.nn.Module, list(torch.nn.Module)),
                optional, default None
            Model calculator object or list of model calculator objects.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_directory: str, optional, default None
            Model directory that contains checkpoint and log files.
        model_ensemble: bool, optional, default None
            Expect a model calculator ensemble. If None, check config or
            assume as False.
        model_ensemble_num: int, optional, default None
            Number of model calculator in ensemble.
        model_checkpoint: (int, str), optional, default None
            If None or 'best', load best model checkpoint.
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        model_compile: bool, optional, default None
            If True, the model calculator will get compiled at the first
            call to enhance the performance. Generally not applicable during
            training where multiple backwards calls are done (e.g. energy
            gradient and loss metrics gradient)

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
        model_calculator = self._get_model_calculator(
            config,
            model_calculator=model_calculator,
            model_type=model_type,
            model_directory=model_directory,
            model_ensemble=model_ensemble,
            model_ensemble_num=model_ensemble_num,
            model_checkpoint=model_checkpoint,
            model_compile=model_compile,
            **kwargs,
            )

        return model_calculator

    def _get_model_calculator(
        self,
        config: settings.Configuration,
        model_calculator: 
            Optional[Union[List[torch.nn.Module], torch.nn.Module]] = None,
        model_type: Optional[str] = None,
        model_directory: Optional[str] = None,
        model_ensemble: Optional[bool] = None,
        model_ensemble_num: Optional[int] = None,
        model_checkpoint: Optional[Union[int, str]] = None,
        model_compile: Optional[bool] = False,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Return calculator model class object.

        Parameters
        ----------
        config: settings.Configuration
            Asparagus parameter settings.config class object
        model_calculator: (torch.nn.Module, list(torch.nn.Module)),
                optional, default None
            Model calculator object or list of model calculator objects.
        model_type: str, optional, default None
            Model calculator type to initialize, e.g. 'PhysNet'. The default
            model is defined in settings.default._default_calculator_model.
        model_directory: str, optional, default None
            Model directory that contains checkpoint and log files.
        model_ensemble: bool, optional, default None
            Expect a model calculator ensemble. If None, check config or
            assume as False.
        model_ensemble_num: int, optional, default None
            Number of model calculator in ensemble.
        model_checkpoint: int, optional, default 'best'
            If None or 'best', load best model checkpoint.
            Otherwise load latest checkpoint file with 'last' or define a
            checkpoint index number of the respective checkpoint file.
        model_compile: bool, optional, default False
            If True, the model calculator will get compiled at the first
            call to enhance the performance. Generally not applicable during
            training where multiple backwards calls are done (e.g. energy
            gradient and loss metrics gradient)

        Returns
        -------
        torch.nn.Module
            Asparagus calculator model object

        """

        ####################################
        # # # Prepare Model Calculator # # #
        ####################################

        # Assign model calculator
        model_calculator, checkpoint, checkpoint_file = (
            model.get_model_calculator(
                config=config,
                model_calculator=model_calculator,
                model_type=model_type,
                model_directory=model_directory,
                model_ensemble=model_ensemble,
                model_ensemble_num=model_ensemble_num,
                model_checkpoint=model_checkpoint,
                **kwargs)
            )

        # Load model checkpoint file
        model_calculator.load(
            checkpoint,
            checkpoint_file=checkpoint_file,
            **kwargs)

        # Compile model calculator if requested
        if model_compile:
            model_calculator.compile()

        return model_calculator

    def get_trainer(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
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

        # Assign model calculator trainer
        trainer = self._get_trainer(
            config=config,
            data_container=data_container,
            model_calculator=model_calculator,
            **kwargs)

        return trainer

    def _get_trainer(
        self,
        config: settings.Configuration,
        data_container: Optional[data.DataContainer] = None,
        model_calculator:
            Optional[Union[model.BaseModel, model.EnsembleModel]] = None,
        **kwargs,
    ) -> training.Trainer:
        """
        Initialize and return model calculator trainer.

        Parameters
        ----------
        config: settings.Configuration
            Asparagus parameter settings.config class object
        data_container: data.DataContainer, optional, default None
            Reference data container object providing training, validation and
            test data for the model training.
        model_calculator: torch.nn.Module, optional, default None
            Model or ensemble model calculator for property predictions.

        Returns:
        --------
        train.Trainer
            Model calculator trainer object

        """

        ###########################################
        # # # Assign Model Calculator Trainer # # #
        ###########################################

        # Check for single model calculator or model ensemble calculator
        if model_calculator is None:
            if (
                kwargs.get('model_ensemble') is None
                and config.get('model_ensemble') is None
            ):
                model_ensemble = False
            elif kwargs.get('model_ensemble') is None:
                model_ensemble = config.get('model_ensemble')
            else:
                model_ensemble = kwargs.get('model_ensemble')
        else:
            if hasattr(model_calculator, 'model_ensemble'):
                model_ensemble = model_calculator.model_ensemble
            else:
                model_ensemble = False

        # Initialize single model or model ensemble trainer
        if model_ensemble:
            trainer = training.EnsembleTrainer(
                config=config,
                data_container=data_container,
                model_calculator=model_calculator,
                **kwargs)
        else:
            trainer = training.Trainer(
                config=config,
                data_container=data_container,
                model_calculator=model_calculator,
                **kwargs)

        return trainer

    def train(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
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
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
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

        #################################
        # # # Assign Reference Data # # #
        #################################

        if self.data_container is None:
            data_container = self.get_data_container(
                config=config,
                **kwargs)
        else:
            data_container = self.data_container

        ##########################################
        # # # Assign Model Calculator Tester # # #
        ##########################################

        # Assign model calculator trainer
        tester = self._get_tester(
            config,
            data_container=data_container,
            **kwargs)

        return tester

    def _get_tester(
        self,
        config: settings.Configuration,
        data_container: Optional[data.DataContainer] = None,
        **kwargs,
    ) -> training.Tester:
        """
        Initialize and return model calculator tester.

        Parameters
        ----------
        config: settings.Configuration
            Asparagus parameter settings.config class object
        data_container: data.DataContainer, optional, default None
            Reference data container object providing training, validation and
            test data for the model training.


        Returns:
        --------
        train.Tester
            Model calculator tester object

        """

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
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
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
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
        config_file: Optional[str] = None,
        model_checkpoint: Optional[Union[int, str]] = 'best',
        **kwargs,
    ) -> 'ase.Calculator':
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
        ase.Calculator
            ASE calculator instance of the model calculator

        """

        ###################################
        # # # Assign Model Calculator # # #
        ###################################

        model_calculator = self.get_model_calculator(
            config=config,
            config_file=config_file,
            model_checkpoint=model_checkpoint,
            **kwargs)

        ##################################
        # # # Prepare ASE Calculator # # #
        ##################################

        ase_calculator = interface.ASE_Calculator(
            model_calculator,
            **kwargs)

        return ase_calculator

    def get_pycharmm_calculator(
        self,
        config: Optional[
            Union[str, Dict[str, Any], settings.Configuration]] = None,
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

        model_calculator = self.get_model_calculator(
            config=config,
            config_file=config_file,
            model_checkpoint=model_checkpoint,
            **kwargs)

        #######################################
        # # # Prepare PyCHARMM Calculator # # #
        #######################################

        pycharmm_calculator = interface.PyCharmm_Calculator(
            model_calculator,
            **kwargs)

        return pycharmm_calculator

    @property
    def config(self):
        return self.config_file
