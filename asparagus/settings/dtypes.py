import numpy as np

import torch

from .. import utils

#======================================
# Input data types
#======================================

# Expected data types of input variables
_dtypes_args = {
    # Model
    'model_calculator':             [utils.is_callable],
    'model_type':                   [utils.is_string, utils.is_None],
    'model_path':                   [utils.is_string],
    'model_num_threads':            [utils.is_integer, utils.is_None],
    'model_device':                 [utils.is_string],
    'model_seed':                   [utils.is_integer],
    # Input module
    'input_model':                  [utils.is_callable],
    'input_type':                   [utils.is_string, utils.is_None],
    # Representation module
    'graph_model':                  [utils.is_callable],
    'graph_type':                   [utils.is_string, utils.is_None],
    'graph_stability_constant':     [utils.is_numeric],
    # Output module
    'output_model':                 [utils.is_callable],
    'output_type':                  [utils.is_string, utils.is_None],
    }

#======================================
# Python data type library
#======================================

_dtype_library = {
    'float': float, 
    'np.float16': np.float16, 
    'np.float32': np.float32, 
    'np.float64': np.float64,
    'torch.float': torch.float,
    'torch.float16': torch.float16,
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    }
