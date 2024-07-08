'''

Train module

It contains a module for setting up the optimizer, scheduler, trainer, and tester.


'''

from .trainer import (
    Trainer
)

from .tester import (
    Tester
)

from .optimizer import (
    get_optimizer
)

from .scheduler import (
    get_scheduler
)
