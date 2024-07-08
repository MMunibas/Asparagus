
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch
import numpy as np
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['get_scheduler']

#======================================
# Optimizer assignment  
#======================================

scheduler_avaiable = {
    'ExponentialLR'.lower(): torch.optim.lr_scheduler.ExponentialLR,
    'LinearLR'.lower(): torch.optim.lr_scheduler.LinearLR,
    }
scheduler_argumens = {
    'ExponentialLR'.lower(): {},
    'LinearLR'.lower(): {}, 
    }

def get_scheduler(
    trainer_scheduler: Union[str, object],
    trainer_optimizer: Optional[object] = None,
    trainer_scheduler_args: Optional[Dict[str, Any]] = {},
):
    """
    Scheduler selection
    
    Parameters
    ----------
        
    trainer_scheduler: (str, object)
        If name is a str than it checks for the corresponding scheduler
        and return the function object.
        The input will be given if it is already a callable object.
    trainer_optimizer: object
        Torch optimizer class object for the NNP training.
    trainer_scheduler_args: dict, optional
        Additional scheduler parameter
        Optional if 'trainer_scheduler' is already a torch optimizer object
            
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
                f"In case of defining 'trainer_scheduler' as string, the " +
                f"optional 'trainer_optimizer' must be defined!")
        
        if trainer_scheduler.lower() in scheduler_avaiable.keys():

            if trainer_scheduler.lower() in scheduler_argumens.keys():
                trainer_scheduler_args.update(
                    scheduler_argumens[trainer_scheduler.lower()])
            
            try:
                
                #TODO Why again???
                #trainer_scheduler_args['gamma'] = np.power(
                    #trainer_scheduler_args['gamma'],
                    #1./trainer_scheduler_args['decay_steps'])
                
                ## Delete decay_steps
                ## TODO maybe a more elegant way to do this is needed
                #del trainer_scheduler_args['decay_steps'] 
                
                return scheduler_avaiable[trainer_scheduler.lower()](
                    optimizer=trainer_optimizer,
                    **trainer_scheduler_args)
            
            except TypeError as error:
                
                logger.error(error)
                raise TypeError(
                    f"Scheduler '{trainer_scheduler}' does not accept " +
                    f"arguments in 'trainer_scheduler_args'")

        else:
            
            raise ValueError(
                f"Scheduler class '{trainer_scheduler}' is not valid!" +
                f"Choose from:\n" +
                str(scheduler_avaiable.keys()))

    else:

        return trainer_scheduler
