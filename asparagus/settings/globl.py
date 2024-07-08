import torch

from .. import utils
from .. import settings

#======================================
# Global Parameters
#======================================

_global_directory = ''

def set_global_directory(directory):
    if utils.is_string(directory):
        _global_directory = directory    
    else:
        raise ValueError(
            f"Model directory path is not a string " + 
            f"but of type '{type(directory)}'!")
        

_global_mode = None

def set_global_mode(mode):
    if mode.lower() in ['train', 'ase']:
        _global_mode = mode.lower()
    else:
        raise ValueError(
            f"Model mode setting ({mode}) is not valid! " +
            f"Choose between 'train' or 'ase'.")


_global_config_file = settings._default_args['config_file']

def set_global_config_file(config_file):
    if utils.is_string(config_file):
        _global_config_file = config_file
    else:
        raise ValueError(
            f"Configuration file path is not a string " + 
            f"but of type '{type(config_file)}'!")


_global_dtype = settings._default_args['model_dtype']

def set_global_dtype(dtype):
    
    if (dtype in (torch.float32, torch.float64, torch.float)):
        _global_dtype = dtype
    else:
        raise ValueError(
            f"Model data type setting ({dtype}) is not valid! " +
            f"Choose between 'torch.float32' or 'torch.float64'.")


_global_device = settings._default_args['model_device']

def set_global_device(device):
    if device.lower() in ['cpu', 'gpu']:
        _global_device = device.lower()
    else:
        raise ValueError(
            f"Model device setting ({device}) is not valid! " +
            f"Choose between 'cpu' or 'gpu'.")


_global_rate = settings._default_args['trainer_dropout_rate']

def set_global_rate(rate):
    if utils.is_numeric(rate):
        _global_rate = rate
    else:
        raise ValueError(
            f"Training dropout rate setting ({rate}) is not valid! " +
            f"Choose a dropout rate between 0 (no dropout) and " + 
            f"1 (full dropout, actually senseless).")
