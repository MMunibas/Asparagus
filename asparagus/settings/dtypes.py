import numpy as np

import torch

from asparagus import utils

# ======================================
#  Input data types
# ======================================

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

# ======================================
#  Python data type library
# ======================================

_dtype_library_dump = {
    'float': float,
    'half': np.float16,
    'single': np.float32,
    'double': np.float64,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
    'np.float16': np.float16,
    'np.float32': np.float32,
    'np.float64': np.float64,
    'torch.float16': torch.float16,
    'torch.half': torch.float16,
    'torch.float32': torch.float32,
    'torch.float': torch.float32,
    'torch.float64': torch.float64,
    'torch.double': torch.float64,
    }

_dtype_library_read = {
    'half': torch.float16,
    'single': torch.float32,
    'double': torch.float64,
    'float': torch.float64,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
    'np.float16': torch.float16,
    'np.float32': torch.float32,
    'np.float64': torch.float64,
    'torch.float16': torch.float16,
    'torch.half': torch.float16,
    'torch.float32': torch.float32,
    'torch.float': torch.float32,
    'torch.float64': torch.float64,
    'torch.double': torch.float64,
    }
