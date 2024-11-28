# Interface for hyperparameter tuning using Ray Tune
# https://github.com/MMunibas/dmc_gpu_PhysNet/tree/main
import os
from typing import Optional, List, Dict, Callable, Tuple, Union, Any

import ray

__all__ = ['HyperParameterTuning']

class HyperParameterTuning:
    """
    Hyperparameter tuning using Ray Tune
    
    """
    
    def __init__(
        self,
        model_trainer: Callable,
        #parameter_labels: Union[str, List[str]]:
    ):
        
        config = {
            "batch_size": tune.choice([4, 8, 16, 32]),
        }
        
        scheduler = ray.tune.schedulers.ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=100,
            grace_period=1,
            reduction_factor=2,
        )

        result = ray.tune.run(
            #partial(train_cifar, data_dir=data_dir),
            resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
            config=config,
            num_samples=10,
            scheduler=scheduler,
        )

    return
