
import logging
from typing import Optional, List, Dict, Tuple, Callable, Union, Any

import torch

from asparagus import utils

# Initialize logger
name = f"{__name__:s}"
logger = utils.set_logger(logging.getLogger(name))

__all__ = ['get_optimizer']

# ======================================
#  Optimizer assignment
# ======================================

optimizer_avaiable = {
    'SGD'.lower(): torch.optim.SGD,
    'Adagrad'.lower(): torch.optim.Adagrad,
    'Adam'.lower(): torch.optim.Adam,
    'AdamW'.lower(): torch.optim.AdamW,
    'Adamax'.lower(): torch.optim.Adam,
    'AMSgrad'.lower(): torch.optim.Adam,
    }
optimizer_arguments = {
    'SGD'.lower(): {},
    'Adagrad'.lower(): {},
    'Adam'.lower(): {},
    'AdamW'.lower(): {},
    'Adamax'.lower(): {},
    'AMSgrad'.lower(): {
        'amsgrad': True},
    }


def get_optimizer(
    trainer_optimizer: Union[str, Callable],
    model_parameter: Optional[Union[List, Dict[str, List]]] = None,
    trainer_optimizer_args: Optional[Dict[str, Any]] = {},
):
    """
    Optimizer selection

    Parameter
    ---------
    trainer_optimizer: (str, callable)
        If name is a str than it checks for the corresponding optimizer
        and return the function object.
        The input will be given if it is already a callable object.
    model_parameter: list, optional, default None
        NNP model trainable parameter to optimize by optimizer.
        Optional if 'trainer_optimizer' is already a torch optimizer object
    trainer_optimizer_args: dict, optional, default {}
        Additional optimizer parameter

    Returns
    -------
    object
        Optimizer function

    """

    # Select calculator model
    if utils.is_string(trainer_optimizer):

        # Check required input for this case
        if model_parameter is None:
            raise SyntaxError(
                "In case of defining 'trainer_optimizer' as string, the " +
                "optional input parameter 'model_parameter' must be defined!")

        # Check optimizer availability
        if trainer_optimizer.lower() in optimizer_avaiable.keys():

            # Set mandatory optimizer options if required
            if trainer_optimizer.lower() in optimizer_arguments.keys():
                trainer_optimizer_args.update(
                    optimizer_arguments[trainer_optimizer.lower()])

            # Prepare optimiizer input
            if utils.is_dictionary(model_parameter):

                optimizer_input = []

                # Iterate over parameter sets
                for key, parameters in model_parameter.items():
                    if key == 'no_weight_decay':
                        no_weight_decay_args = trainer_optimizer_args.copy()
                        no_weight_decay_args['weight_decay'] = 0.0
                        no_weight_decay_args['params'] = parameters
                        optimizer_input.append(no_weight_decay_args)
                    else:
                        default_args = trainer_optimizer_args.copy()
                        default_args['params'] = parameters
                        optimizer_input.append(default_args)

            else:

                trainer_optimizer_args['params'] = model_parameter
                optimizer_input.append(trainer_optimizer_args)

            try:

                return optimizer_avaiable[trainer_optimizer.lower()](
                    optimizer_input)

            except TypeError as error:
                logger.error(error)
                raise TypeError(
                    f"Optimizer '{trainer_optimizer}' does not accept " +
                    "arguments in 'trainer_optimizer_args'")

        else:

            raise ValueError(
                f"Optimizer class '{trainer_optimizer}' is not valid!" +
                "Choose from:\n" +
                str(optimizer_avaiable.keys()))

    else:

        return trainer_optimizer
