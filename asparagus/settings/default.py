import numpy as np
import torch

#======================================
# General Settings
#======================================

# Default calculator model
_default_calculator_model = 'PhysNet'

# Default device
_default_device = 'cpu'

# Default floating point dtype
_default_dtype = torch.float64

# ======================================
# Default Input
# ======================================

# Default arguments for input variables
_default_args = {
    'config':                       {},
    'config_file':                  'config.json',
    # Model
    'model_calculator':             None,
    'model_type':                   None,
    'model_restart':                False,
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
