# This should manage checkpoint creation and loading writer to tensorboardX
import os
import re
import string
import random
import datetime
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import torch

from asparagus import model
from asparagus import settings
from asparagus import utils

__all__ = ['FileManager']


class FileManager():
    """
    File manager for loading and storing model parameter and training files.
    Manage checkpoint creation and loading writer to tensorboardX

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of Asparagus parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to config json file (str)
    model_directory: str, optional, default None
        Model directory that contains checkpoint and log files.
    model_max_checkpoints: int, optional, default 1
        Maximum number of checkpoint files.
    **kwargs: dict
        Additional keyword arguments for tensorboards 'SummaryWriter'

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for graph module
    _default_args = {
        'model_directory':              None,
        'model_max_checkpoints':        1,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'model_directory':              [utils.is_string, utils.is_None],
        'model_max_checkpoints':        [utils.is_integer],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None,
        model_directory: Optional[str] = None,
        model_max_checkpoints: Optional[int] = None,
        verbose: Optional[bool] = True,
        **kwargs
    ):
        """
        Initialize file manager class.

        """

        ####################################
        # # # Check File Manager Input # # #
        ####################################

        # Get configuration object
        config = settings.get_config(config, config_file, config_from=self)

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            check_default=utils.get_default_args(self, None),
            check_dtype=utils.get_dtype_args(self, None))
        
        # Update global configuration dictionary
        config.update(
            config_update,
            verbose=verbose)
        
        ###################################
        # # # Prepare Model Directory # # #
        ###################################

        # Take either defined model directory path or a generate a generic one
        if self.model_directory is None:
            if config.get('model_type') is None:
                model_type = ""
            else:
                model_type = config.get('model_type') + "_"
            self.model_directory = (
                model_type
                + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            config['model_directory'] = self.model_directory

        # Prepare model subdirectory paths
        self.best_dir = os.path.join(self.model_directory, 'best')
        self.ckpt_dir = os.path.join(self.model_directory, 'checkpoints')
        self.logs_dir = os.path.join(self.model_directory, 'logs')

        return

    def create_model_directory(self):
        """
        Create folders for checkpoints and tensorboardX
        """

        # Create model directory
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        # Create directory for best model checkpoints
        if not os.path.exists(self.best_dir):
            os.makedirs(self.best_dir)
        # Create directory for model parameter checkpoints
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        # Create directory for tensorboardX/logs
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        return

    def save_checkpoint(
        self,
        model_calculator: "model.BaseModel",
        optimizer: Optional["torch.Optimizer"] = None,
        scheduler: Optional["torch.Scheduler"] = None,
        epoch: Optional[int] = 0,
        best: Optional[bool] = False,
        best_loss: Optional[float] = None,
        num_checkpoint: Optional[int] = None,
        max_checkpoints: Optional[int] = None,
    ):
        """
        Save model parameters and training state to checkpoint file.

        Parameters
        ----------
        model_calculator: model.BaseModel
            Torch calculator model
        optimizer: torch.Optimizer, optional, default None
            Torch optimizer
        scheduler: torch.Scheduler, optional, default None
            Torch scheduler
        epoch: int, optional, default 0
            Training epoch of calculator model 
        best: bool, optional, default False
            If True, save as best model checkpoint file.
        best_loss: float, optional, default None
            Best loss value of the training run.
        num_checkpoint: int, optional, default None
            Alternative checkpoint index other than epoch.
        max_checkpoints: int, optional, default 1
            Maximum number of checkpoint files. If the threshold is reached and
            a checkpoint of the best model (best=True) or specific number
            (num_checkpoint is not None), respectively many checkpoint files
            with the lowest indices will be deleted.

        """

        # Check existence of the directories
        self.create_model_directory()

        # Prepare state dictionary to store
        state = {
            'model_state_dict': model_calculator.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
        }

        # For best model, just store model parameter, epoch and loss value
        if best:
            pass
        # Else the complete current model training state if available
        else:
            if optimizer is not None:
                state.update(
                    {'optimizer_state_dict': optimizer.state_dict()}
                )
            if scheduler is not None:
                state.update(
                    {'scheduler_state_dict': scheduler.state_dict()}
                )

        # Checkpoint file name
        if best:
            ckpt_name = os.path.join(self.best_dir, 'best_model.pt')
        elif num_checkpoint is None:
            ckpt_name = os.path.join(
                self.ckpt_dir, f'model_{epoch:d}.pt')
        else:
            if utils.is_integer(num_checkpoint):
                ckpt_name = os.path.join(
                    self.ckpt_dir, f'model_{num_checkpoint:d}.pt')
            else:
                raise ValueError(
                    "Checkpoint file index number 'num_checkpoint' is not "
                    + "an integer!")

        # Write checkpoint file
        torch.save(state, ckpt_name)

        # Store latest checkpoint file in model calculator
        model_calculator.checkpoint_loaded = True
        model_calculator.checkpoint_file = ckpt_name

        # Check number of epoch checkpoints
        if not best and num_checkpoint is None:
            self.check_max_checkpoints(max_checkpoints)

        return

    def load_checkpoint(
        self,
        checkpoint_label: Union[str, int],
        return_name: Optional[bool] = False,
        verbose: Optional[bool] = True,
    ) -> Dict[str, Any]:
        """
        Load model parameters and training state from checkpoint file.

        Parameters
        ----------
        checkpoint_label: (str, int)
            If None, load checkpoint file with best loss function value.
            If string and a valid file path, load the respective checkpoint 
            file.
            If string 'best' or 'last', load respectively the best checkpoint 
            file (as with None) or the with the highest epoch number.
            If integer, load the checkpoint file of the respective epoch 
            number.
        return_name: bool, optional, default False
            If True, return checkpoint state and checkpoint file batch.
            Else, only return checkpoint state

        Returns
        -------
        Any
            Torch module checkpoint file
        """

        # Check checkpoint label input
        if checkpoint_label is None:
            checkpoint_label = 'best'

        if (
            utils.is_string(checkpoint_label)
            and checkpoint_label.lower() == 'best'
        ):

            ckpt_name = os.path.join(self.best_dir, 'best_model.pt')

            # Check if best checkpoint file exists or return None
            if not os.path.exists(ckpt_name):
                ckpt_name = None
                if verbose:
                    self.logger.info(
                        "No best checkpoint file found in "
                        + f"{self.best_dir:s}.")

        elif (
            utils.is_string(checkpoint_label)
            and checkpoint_label.lower() == 'last'
        ):

            # Get highest index checkpoint file
            ckpt_max = -1
            if os.path.exists(self.ckpt_dir):
                for ckpt_file in os.listdir(self.ckpt_dir):
                    ckpt_num = re.findall("model_(\d+).pt", ckpt_file)
                    ckpt_num = (int(ckpt_num[0]) if ckpt_num else -1)
                    if ckpt_max < ckpt_num:
                        ckpt_max = ckpt_num

            # If no checkpoint files available return None
            if ckpt_max < 0:
                ckpt_name = None
                if verbose:
                    self.logger.info(
                        "No latest checkpoint file found in "
                        + f"{self.ckpt_dir:s}.")
            else:
                ckpt_name = os.path.join(
                    self.ckpt_dir, f'model_{ckpt_max:d}.pt')

        elif utils.is_integer(checkpoint_label):

            ckpt_name = os.path.join(
                self.ckpt_dir, f'model_{checkpoint_label:d}.pt')

            # Check existence
            if not os.path.exists(ckpt_name):
                raise FileNotFoundError(
                    f"Checkpoint file '{ckpt_name}' of index "
                    + f"{checkpoint_label:d} does not exist!")
        
        elif utils.is_string(checkpoint_label):

            ckpt_name = checkpoint_label

            # Check existence
            if not os.path.exists(ckpt_name):
                raise FileNotFoundError(
                    f"Checkpoint file '{ckpt_name}' does not exist!")

        else:
            
            raise SyntaxError(
                "Input for the model checkpoint label to load "
                + "'checkpoint_label' is not valid type!")

        # Load checkpoint
        if ckpt_name is None:
            checkpoint = None
        else:
            checkpoint = torch.load(ckpt_name, weights_only=False)
            if verbose:
                self.logger.info(
                    f"Checkpoint file '{ckpt_name:s}' will be loaded.")

        if return_name:
            return checkpoint, ckpt_name
        else:
            return checkpoint

    def check_max_checkpoints(
        self,
        max_checkpoints: Optional[int] = None,
    ):
        """
        Check number of checkpoint files and in case of exceeding the
        maximum checkpoint threshold, delete the ones with lowest indices.

        Parameters
        ----------
        max_checkpoints: int, optional, default None
             Maximum number of checkpoint files. If None, the threshold is
             taken from the class attribute 'self.model_max_checkpoints'.

        """
        
        # Skip in checkpoint threshold is None
        if max_checkpoints is None and self.model_max_checkpoints is None:
            return
        elif max_checkpoints is None:
            max_checkpoints = self.model_max_checkpoints

        # Gather checkpoint files
        num_checkpoints = []
        for ckpt_file in os.listdir(self.ckpt_dir):
            ckpt_num = re.findall("model_(\d+).pt", ckpt_file)
            if ckpt_num:
                num_checkpoints.append(int(ckpt_num[0]))
        num_checkpoints = sorted(num_checkpoints)

        # Delete in case the lowest checkpoint files
        if len(num_checkpoints) >= max_checkpoints:
            # Get checkpoints to delete
            if max_checkpoints > 0:
                remove_num_checkpoints = num_checkpoints[:-max_checkpoints]
            # If max_checkpoints is zero (or less), delete everyone
            else:
                remove_num_checkpoints = num_checkpoints
            # Delete checkpoint files
            for ckpt_num in remove_num_checkpoints:
                ckpt_name = os.path.join(
                    self.ckpt_dir, f'model_{ckpt_num:d}.pt')
                os.remove(ckpt_name)

        return

    def save_config(
        self,
        config: object,
        max_backup: Optional[int] = 1,
    ):
        """
        Save config object in current model directory with the default file
        name. If such file already exist, backup the old one and overwrite.

        Parameters
        ----------
        config: object
            Config class object
        max_backup: optional, int, default 100
            Maximum number of backup config files

        """

        # Config file path
        config_file = os.path.join(
            self.model_directory, settings._default_args['config_file'])

        # Check for old config files
        if os.path.exists(config_file):

            default_file = settings._default_args['config_file']

            # Check for backup config files
            list_backups = []
            for f in os.listdir(self.model_directory):
                num_backup = re.findall(
                    "(\d+)_" + default_file, f)
                num_backup = (int(num_backup[0]) if num_backup else -1)
                if num_backup >= 0:
                    list_backups.append(num_backup)
            list_backups = sorted(list_backups)

            # Rename old config file
            if len(list_backups):
                backup_file = os.path.join(
                    self.model_directory,
                    f"{list_backups[-1] + 1:d}_" + default_file)
            else:
                backup_file = os.path.join(
                    self.model_directory,
                    f"{1:d}_" + default_file)
            os.rename(config_file, backup_file)

            # If maximum number of back file reached, delete the oldest
            #if len(list_backups) >= max_backup:
                #for num_backup in list_backups[:-(max_backup - 1)]:
                    #backup_file = os.path.join(
                        #self.model_directory,
                        #f"{num_backup:d}_" + default_file)
                    #os.remove(backup_file)
            while len(list_backups) >= max_backup:
                backup_file = os.path.join(
                    self.model_directory,
                    f"{list_backups[0]:d}_" + default_file)
                os.remove(backup_file)
                del list_backups[0]

        # Dump config in file path
        config.dump(config_file=config_file)

        return
