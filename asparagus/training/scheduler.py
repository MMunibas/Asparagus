import logging
from typing import Optional, List, Dict, Tuple, Callable, Union, Any

import torch

from asparagus import utils

# Initialize logger
name = f"{__name__:s}"
logger = utils.set_logger(logging.getLogger(name))

__all__ = ['get_scheduler']

# ======================================
#  Optimizer assignment
# ======================================

scheduler_avaiable = {
    'ExponentialLR'.lower(): torch.optim.lr_scheduler.ExponentialLR,
    'LinearLR'.lower(): torch.optim.lr_scheduler.LinearLR,
    'ReduceLROnPlateau'.lower(): torch.optim.lr_scheduler.ReduceLROnPlateau,
    }
scheduler_arguments = {
    'ExponentialLR'.lower(): {'gamma': 0.99},
    'LinearLR'.lower(): {},
    'ReduceLROnPlateau'.lower(): {},
    }


def get_scheduler(
    trainer_scheduler: Union[str, Callable],
    trainer_optimizer: Optional[Callable] = None,
    trainer_scheduler_args: Optional[Dict[str, Any]] = {},
):
    """
    Scheduler selection

    Parameters
    ----------

    trainer_scheduler: (str, Callable)
        If name is a str than it checks for the corresponding scheduler
        and return the function object.
        The input will be given if it is already a callable object.
    trainer_optimizer: Callable, optional, default None
        Torch optimizer class object for the NNP training.
        Optional if 'trainer_scheduler' is already a torch scheduler object.
    trainer_scheduler_args: dict, optional, default {}
        Additional scheduler parameter

    Returns
    -------
    object
        Scheduler function
    """

    # Select calculator model
    if utils.is_string(trainer_scheduler):

        # Check required input for this case
        if trainer_optimizer is None:
            raise SyntaxError(
                "In case of defining 'trainer_scheduler' as string, the " +
                "optional 'trainer_optimizer' must be defined!")

        if trainer_scheduler.lower() in scheduler_avaiable.keys():

            if trainer_scheduler.lower() in scheduler_arguments.keys():
                trainer_scheduler_args = {
                    **scheduler_arguments[trainer_scheduler.lower()],
                    **trainer_scheduler_args}

            try:

                return (
                    scheduler_avaiable[trainer_scheduler.lower()](
                        optimizer=trainer_optimizer,
                        **trainer_scheduler_args),
                    trainer_scheduler_args
                    )

            except TypeError as error:

                logger.error(error)
                raise TypeError(
                    f"Scheduler '{trainer_scheduler}' does not accept one"
                    + "of the arguments in 'trainer_scheduler_args':\n"
                    + f"{list(trainer_scheduler_args.keys())}")

        else:

            raise ValueError(
                f"Scheduler class '{trainer_scheduler}' is not valid!"
                + "Choose from:\n" +
                str(scheduler_avaiable.keys()))

    else:

        return trainer_scheduler, trainer_scheduler_args
