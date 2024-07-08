import sys
from typing import Optional, Union

import torch

from .. import model
from .. import settings
from .. import utils

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


#======================================
# Calculator Model Assignment
#======================================

model_available = {
    'PhysNet'.lower(): get_Model_PhysNet,
    'PaiNN'.lower(): get_Model_PaiNN,
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
            f"Calculator model type input '{model_type:s}' is not known!\n" +
            "Choose from:\n" + str(model_available.keys()))
    
    return

def get_model_calculator(
    config: object,
    model_calculator: Optional[torch.nn.Module] = None,
    model_type: Optional[str] = None,
    model_checkpoint: Optional[Union[int, str]] = 'best',
    **kwargs,
) -> (torch.nn.Module, bool):
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
    model_checkpoint: (str, int), optional, default 'best'
        If None, load checkpoint file with best loss function value.
        If string 'best' or 'last', load respectively the best checkpoint file
        (as with None) or the with the highest epoch number.
        If integer, load the checkpoint file of the respective epoch number.
    
    Returns
    -------
    torch.nn.Module
        Asparagus calculator model object
    Any
        Torch module checkpoint file

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

        # Initialize model calculator
        model_calculator = model_calculator_class(
            config=config,
            **kwargs)

    # Add calculator info to configuration dictionary
    if hasattr(model_calculator, "get_info"):
        config.update(
            model_calculator.get_info(), 
            config_from=utils.get_function_location())

    # Initialize checkpoint file manager and load best model
    filemanager = model.FileManager(config, **kwargs)
    
    # Get checkpoint file
    checkpoint = filemanager.load_checkpoint(model_checkpoint)

    return model_calculator, checkpoint
    
    
