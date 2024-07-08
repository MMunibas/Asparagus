
import torch

#======================================
# Input Model Provision
#======================================

def get_Input_PhysNet():
    from .physnet_modules import Input_PhysNet
    return Input_PhysNet

def get_Input_PhysNet_original():
    raise NotImplementedError()
    from .physnet_modules import Input_PhysNet_original
    return Input_PhysNet_original

def get_Input_PaiNN():
    from .painn_modules import Input_PaiNN
    return Input_PaiNN


#======================================
# Input Model Assignment
#======================================

input_module_available = {
    'PhysNet'.lower(): get_Input_PhysNet,
    'PhysNet_original'.lower(): get_Input_PhysNet_original,
    'PaiNN'.lower(): get_Input_PaiNN,
    }

def get_input_module(
    input_type: str,
    **kwargs,
) -> torch.nn.Module:
    """
    Input module selection

    Parameters
    ----------
    input_type: str
        Input module representation of the atomistic structural information,
        e.g. 'PhysNet'.
    **kwargs: dict, optional
        Keyword arguments for input module initialization

    Returns
    -------
    torch.nn.Module
        Input model object to encode atomistic structural information
    """

    # Check input parameter
    if input_type is None:
        raise SyntaxError("No input module type is defined by 'input_type'!")
    
    # Return requested input module
    if input_type.lower() in input_module_available:
        return input_module_available[input_type.lower()]()(
            **kwargs)
    else:
        raise ValueError(
            f"Input model type input '{input_type:s}' is not known!\n" +
            "Choose from:\n" + str(input_module_available.keys()))
