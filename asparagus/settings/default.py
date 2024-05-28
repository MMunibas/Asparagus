import numpy as np
import torch

# ======================================
# General Model Type Settings
# ======================================

# Default calculator model
_default_calculator_model = 'PhysNet'

# ======================================
# Default Input
# ======================================

# Default arguments for input variables
_default_args = {
    'config':                       {},
    'config_file':                  'config.json',
    'device':                       'cpu',
    'dtype':                        torch.float64,
    # Model
    'model_calculator':             None,
    'model_type':                   None,
    'model_restart':                False,
    'model_device':                 'cpu',
    'model_dtype':                  torch.float64,
    'model_seed':                   np.random.randint(1E6),
    # Input module
    'input_calculator':             None,
    'input_type':                   None,
    # Graph module
    'graph_calculator':             None,
    'graph_type':                   None,
    'graph_stability_constant':     1.e-8,
    # Output module
    'output_calculator':            None,
    'output_type':                  None,
}
