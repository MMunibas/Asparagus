
import torch

#======================================
# Output Model Provision
#======================================

def get_Output_PhysNet():
    from .physnet_modules import Output_PhysNet
    return Output_PhysNet

def get_Output_PaiNN():
    from .painn_modules import Output_PaiNN
    return Output_PaiNN


#======================================
# Output Model Assignment
#======================================

output_module_available = {
    'PhysNet'.lower(): get_Output_PhysNet,
    'PaiNN'.lower(): get_Output_PaiNN,
    }

def get_output_module(
    output_type: str,
    **kwargs,
) -> torch.nn.Module:
    """
    Output module selection

    Parameters
    ----------
    output_type: str
        Output module representation for property predictions,
        e.g. 'PhysNet'
    **kwargs: dict, optional
        Keyword arguments for output module initialization

    Returns
    -------
    torch.nn.Module
        Output model object for property predictions
    """

    # Check input parameter
    if output_type is None:
        raise SyntaxError("No output module type is defined by 'output_type'!")
    
    # Return requested output module
    if output_type.lower() in output_module_available:
        return output_module_available[output_type.lower()]()(
            **kwargs)
    else:
        raise ValueError(
            f"Output model type input '{output_type:s}' is not known!\n" +
            "Choose from:\n" + str(output_module_available.keys()))
