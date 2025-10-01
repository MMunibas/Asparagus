import sys
from typing import Optional, Union, Any

import torch

from asparagus import model
from asparagus import settings
from asparagus import utils

__all__ = ['get_model_calculator']

#======================================
# Calculator Model Provision
#======================================

def get_Model_PhysNet():
    from .physnet import Model_PhysNet
    return Model_PhysNet

def get_Model_PaiNN():
    from .painn import Model_PaiNN
    return Model_PaiNN

def get_Model_AMP():
    from .amp import Model_AMP
    return Model_AMP


#======================================
# Calculator Model Assignment
#======================================

model_available = {
    'PhysNet'.lower(): get_Model_PhysNet,
    'PaiNN'.lower(): get_Model_PaiNN,
    'AMP'.lower(): get_Model_AMP,
    }

def _get_model_calculator(
    model_type: str,
) -> torch.nn.Module:
    """
    Model calculator selection

    Parameters
    ----------
    model_type: str
        Model calculator type, e.g. 'PhysNet'

    Returns
    -------
    torch.nn.Module
        Calculator model object for property prediction

    """
    
    # Check input parameter
    if model_type is None:
        raise SyntaxError("No model type is defined by 'model_type'!")

    # Return requested calculator model
    if model_type.lower() in model_available:
        return model_available[model_type.lower()]()
    else:
        raise ValueError(
            f"Calculator model type input '{model_type:s}' is not known!\n"
            + "Choose from:\n" + str(model_available.keys()))
    
    return

def get_model_calculator(
    config: object,
    model_calculator: Optional[torch.nn.Module] = None,
    model_type: Optional[str] = None,
    model_directory: Optional[str] = None,
    model_ensemble: Optional[bool] = None,
    model_ensemble_num: Optional[int] = None,
    model_checkpoint: Optional[Union[int, str]] = None,
    verbose: Optional[bool] = True,
    **kwargs,
) -> (torch.nn.Module, Any):
    """
    Return calculator model class object and restart flag.

    Parameters
    ----------
    config: object
        Model parameter settings.config class object
    model_calculator: torch.nn.Module, optional, default None
        Model calculator object.
    model_type: str, optional, default None
        Model calculator type to initialize, e.g. 'PhysNet'. The default
        model is defined in settings.default._default_calculator_model.
    model_directory: str, optional, default None
        Model directory that contains checkpoint and log files.
    model_ensemble: bool, optional, default None
        Expect a model calculator ensemble. If None, check config or assume
        as False.
    model_ensemble_num: int, optional, default None
        Number of model calculator in ensemble. If None and
        'model_ensemble' is True, check config or take all available models
        found.
    model_checkpoint: (str, int), optional, default None
        If None or 'best', load checkpoint file with best loss function value.
        If string is 'last', load respectively the best checkpoint file
        (as with None) or the with the highest epoch number.
        If integer, load the checkpoint file of the respective epoch number.
    
    Returns
    -------
    torch.nn.Module
        Asparagus calculator model object
    Any
        Model parameter checkpoint object

    """

    # Initialize model calculator if not given
    if model_calculator is None:
    
        # Check requested model type
        if model_type is None and config.get('model_type') is None:
            model_type = settings._default_calculator_model
        elif model_type is None:
            model_type = config['model_type']

        # Get requested calculator model
        model_calculator_class = _get_model_calculator(model_type)

        # Check for model ensemble calculator option
        if model_ensemble is None and config.get('model_ensemble') is None:
            model_ensemble = False
        elif model_ensemble is None:
            model_ensemble = config.get('model_ensemble')

        # Initialize model calculator or ensemble calculator
        if model_ensemble:
            model_calculator = model.EnsembleModel(
                config=config,
                model_calculator_class=model_calculator_class,
                model_ensemble_num=model_ensemble_num,
                verbose=verbose,
                **kwargs)
        else:
            model_calculator = model_calculator_class(
                config=config,
                verbose=verbose,
                **kwargs)

    # Add calculator info to configuration dictionary
    if hasattr(model_calculator, "get_info"):
        config.update(
            model_calculator.get_info(), 
            config_from=utils.get_function_location(),
            verbose=verbose)

    # Initialize checkpoint file manager and load best or specific model
    # parameter checkpoint file
    filemanager = model.FileManager(
        config,
        model_directory=model_directory,
        verbose=verbose,
        **kwargs)

    # Get checkpoint file
    if model_checkpoint is None and config.get('model_checkpoint') is None:
        model_checkpoint = 'best'
    elif model_checkpoint is None:
        model_checkpoint = config.get('model_checkpoint')
    checkpoint, checkpoint_file = filemanager.load_checkpoint(
        model_checkpoint,
        return_name=True,
        verbose=verbose)

    return model_calculator, checkpoint, checkpoint_file
