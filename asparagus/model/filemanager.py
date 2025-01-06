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
    model_calculator: torch.nn.Module, optional, default None
        Model calculator to to manage.
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
        model_calculator: Optional[torch.nn.Module] = None,
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
        
        #######################################
        # # # Prepare Directory Parameter # # #
        #######################################

        # Get model parameter
        if self.model_calculator is None:

            self.model_type = config.get('model_type')
            self.model_ensemble = config.get('model_ensemble')
            self.model_ensemble_num = config.get('model_ensemble_num')

        else:

            if hasattr(self.model_calculator, 'model_type'):
                self.model_type = self.model_calculator.model_type
            else:
                self.model_type = config.get('model_type')
            if hasattr(self.model_calculator, 'model_ensemble'):
                self.model_ensemble = self.model_calculator.model_ensemble
            else:
                self.model_ensemble = False
            if hasattr(self.model_calculator, 'model_ensemble_num'):
                self.model_ensemble_num = (
                    self.model_calculator.model_ensemble_num)
            elif self.model_ensemble:
                self.model_ensemble_num = config.get('model_ensemble_num')
            else:
                self.model_ensemble_num = None

        # Take either defined model directory path or a generate a generic one
        if self.model_directory is None:
            if self.model_type is None:
                model_label = ""
            else:
                model_label = self.model_type + "_"
            self.model_directory = (
                model_label
                + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
            config.update(
                {'model_directory': self.model_directory},
                verbose=verbose)

        # Prepare model subdirectory paths
        if self.model_ensemble:
            self.best_dir = [
                os.path.join(self.model_directory, f'{imodel:d}', 'best')
                for imodel in range(self.model_ensemble_num)]
            self.ckpt_dir = [
                os.path.join(
                    self.model_directory, f'{imodel:d}', 'checkpoints')
                for imodel in range(self.model_ensemble_num)]
            self.logs_dir = [
                os.path.join(self.model_directory, f'{imodel:d}', 'logs')
                for imodel in range(self.model_ensemble_num)]
        else:

            self.best_dir = os.path.join(self.model_directory, 'best')
            self.ckpt_dir = os.path.join(self.model_directory, 'checkpoints')
            self.logs_dir = os.path.join(self.model_directory, 'logs')

        return

    def __str__(self):
        return f"FileManager '{self.model_directory:s}'"

    def create_model_directory(self):
        """
        Create folders for checkpoints and tensorboardX
        """

        # Create model directory
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        # Create directory for best model checkpoints
        if not os.path.exists(self.best_dir):
            if self.model_ensemble:
                for imodel in range(self.model_ensemble_num):
                    os.makedirs(self.best_dir[imodel])
            else:
                os.makedirs(self.best_dir)
        # Create directory for model parameter checkpoints
        if not os.path.exists(self.ckpt_dir):
            if self.model_ensemble:
                for imodel in range(self.model_ensemble_num):
                    os.makedirs(self.ckpt_dir[imodel])
            else:
                os.makedirs(self.ckpt_dir)
        # Create directory for tensorboardX/logs
        if not os.path.exists(self.logs_dir):
            if self.model_ensemble:
                for imodel in range(self.model_ensemble_num):
                    os.makedirs(self.logs_dir[imodel])
            else:
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
        imodel: Optional[int] = None,
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
        imodel: int, optional, default None
            Model index in case of a model ensemble

        """

        # Check 'imodel' input in case of a model ensemble
        if self.model_ensemble and imodel is None:
            raise SyntaxError(
                "Checkpoint state of a model in a model ensemble cannot be "
                + "stored without definition of the model index 'imodel'.")

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
            if self.model_ensemble:
                best_dir = self.best_dir[imodel]
            else:
                best_dir = self.best_dir
            ckpt_name = os.path.join(best_dir, 'best_model.pt')
        elif num_checkpoint is None:
            if self.model_ensemble:
                ckpt_dir = self.ckpt_dir[imodel]
            else:
                ckpt_dir = self.ckpt_dir
            ckpt_name = os.path.join(ckpt_dir, f'model_{epoch:d}.pt')
        else:
            if utils.is_integer(num_checkpoint):
                if self.model_ensemble:
                    ckpt_dir = self.ckpt_dir[imodel]
                else:
                    ckpt_dir = self.ckpt_dir
                ckpt_name = os.path.join(
                    ckpt_dir, f'model_{num_checkpoint:d}.pt')
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
            self.check_max_checkpoints(
                max_checkpoints=max_checkpoints,
                imodel=imodel)

        return

    def load_checkpoint(
        self,
        checkpoint_label: Union[str, int],
        return_name: Optional[bool] = False,
        imodel: Optional[int] = None,
        verbose: Optional[bool] = True,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
        imodel: int, optional, default None
            Model index in case of a model ensemble

        Returns
        -------
        Any
            Torch module checkpoint file

        """

        # Check checkpoint label input
        if checkpoint_label is None:
            checkpoint_label = 'best'

        # Load best model checkpoint
        if (
            utils.is_string(checkpoint_label)
            and checkpoint_label.lower() == 'best'
        ):

            ckpt_name = self.get_best_checkpoint(
                imodel=imodel,
                verbose=verbose)

        # Load last model checkpoint
        elif (
            utils.is_string(checkpoint_label)
            and checkpoint_label.lower() == 'last'
        ):

            ckpt_name = self.get_last_checkpoint(
                imodel=imodel,
                verbose=verbose)

        else:

            ckpt_name = self.get_specific_checkpoint(
                checkpoint_label=checkpoint_label,
                imodel=imodel,
                verbose=verbose)

        # Load checkpoint(s)
        if utils.is_array_like(ckpt_name):
            checkpoint = []
            for ckpt_name_i in ckpt_name:
                if ckpt_name_i is None:
                    checkpoint.append(None)
                else:
                    checkpoint.append(
                        torch.load(ckpt_name_i, weights_only=False))
        elif ckpt_name is None:
            checkpoint = None
        else:
            checkpoint = torch.load(ckpt_name, weights_only=False)

        if return_name:
            return checkpoint, ckpt_name
        else:
            return checkpoint

    def check_max_checkpoints(
        self,
        max_checkpoints: Optional[int] = None,
        imodel: Optional[int] = None,
    ):
        """
        Check number of checkpoint files and in case of exceeding the
        maximum checkpoint threshold, delete the ones with lowest indices.

        Parameters
        ----------
        max_checkpoints: int, optional, default None
             Maximum number of checkpoint files. If None, the threshold is
             taken from the class attribute 'self.model_max_checkpoints'.
        imodel: int, optional, default None
            Model index in case of a model ensemble

        """

        # Check 'imodel' input in case of a model ensemble
        if self.model_ensemble and imodel is None:
            raise SyntaxError(
                "Missing model index 'imodel' definition of a model in a "
                + "model ensemble.")
        
        # Skip in checkpoint threshold is None
        if max_checkpoints is None and self.model_max_checkpoints is None:
            return
        elif max_checkpoints is None:
            max_checkpoints = self.model_max_checkpoints

        # Get checkpoint directory
        if self.model_ensemble:
            ckpt_dir = self.ckpt_dir[imodel]
        else:
            ckpt_dir = self.ckpt_dir

        # Gather checkpoint files
        num_checkpoints = []
        for ckpt_file in os.listdir(ckpt_dir):
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
                ckpt_name = os.path.join(ckpt_dir, f'model_{ckpt_num:d}.pt')
                os.remove(ckpt_name)

        return

    def save_config(
        self,
        config: object,
        max_backup: Optional[int] = 1,
        imodel: Optional[int] = None,
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
        imodel: int, optional, default None
            Model index in case of a model ensemble

        """

        # Check 'imodel' input in case of a model ensemble
        if self.model_ensemble and imodel is None:
            raise SyntaxError(
                "Missing model index 'imodel' definition of a model in a "
                + "model ensemble.")

        # Model directory and config file path
        if self.model_ensemble:
            model_directory = os.path.join(self.model_directory, f'{imodel:d}')
        else:
            model_directory = self.model_directory
        config_file = os.path.join(
            model_directory, settings._default_args['config_file'])

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
                    model_directory,
                    f"{list_backups[-1] + 1:d}_" + default_file)
            else:
                backup_file = os.path.join(
                    model_directory, f"{1:d}_" + default_file)
            os.rename(config_file, backup_file)

            # If maximum number of back file reached, delete the oldest
            while len(list_backups) >= max_backup:
                backup_file = os.path.join(
                    model_directory,
                    f"{list_backups[0]:d}_" + default_file)
                os.remove(backup_file)
                del list_backups[0]

        # Dump config in file path
        config.dump(config_file=config_file)

        return

    def get_best_checkpoint(
        self,
        imodel: Optional[int] = None,
        verbose: Optional[bool] = True,
    ) -> Union[str, List[str]]:
        """
        Get best model checkpoint file path

        Parameters
        ----------
        imodel: int, optional, default None
            Model index in case of a model ensemble

        Returns
        -------
        (str, list(str))
            Torch module checkpoint file path(s)

        """

        # For model ensembles without certain model definition get all
        # best model checkpoint files
        if self.model_ensemble and imodel is None:

            # Get model checkpoint file paths
            ckpt_name = [
                os.path.join(best_dir, 'best_model.pt')
                for best_dir in self.best_dir]

            # Check if best checkpoint file exists or return None
            message = "Ensemble model best checkpoint files:\n"
            for imdl, ckpt_name_i in enumerate(ckpt_name):
                if os.path.exists(ckpt_name_i):
                    message += (
                        f" Found '{ckpt_name_i:s}' "
                        + f"for model {imdl:d}!\n")
                else:
                    ckpt_name[imdl] = None
                    message += (
                        f" No file found in {self.best_dir[imdl]:s} "
                        + f"for model {imdl:d}!\n")
            message = message[:-1]

            if verbose:
                self.logger.info(message)

        # Get best model checkpoint file
        else:

            # Get model checkpoint file path
            if self.model_ensemble:
                best_dir = self.best_dir[imodel]
            else:
                best_dir = self.best_dir
            ckpt_name = os.path.join(self.best_dir, 'best_model.pt')

            # Check if best checkpoint file exists or return None
            if not os.path.exists(ckpt_name):
                ckpt_name = None
                if verbose:
                    self.logger.info(
                        f"No best checkpoint file found in {best_dir:s}!")
            elif self.model_ensemble and verbose:
                self.logger.info(
                    f"Checkpoint file '{ckpt_name:s}' found "
                    + f"for model {imdl:d}!")
            elif verbose:
                self.logger.info(
                    f"Checkpoint file '{ckpt_name:s}' found.")

        return ckpt_name

    def get_last_checkpoint(
        self,
        imodel: Optional[int] = None,
        verbose: Optional[bool] = True,
    ) -> Union[str, List[str]]:
        """
        Get last or specific model checkpoint file path with the highest
        epoch number.

        Parameters
        ----------
        imodel: int, optional, default None
            Model index in case of a model ensemble

        Returns
        -------
        (str, list(str))
            Torch module checkpoint file path(s)

        """

        # For model ensembles without certain model definition get all
        # respective last model checkpoint files
        if self.model_ensemble and imodel is None:

            # Get model checkpoint file paths
            ckpt_name = [
                os.path.join(ckpt_dir, 'best_model.pt')
                for ckpt_dir in self.ckpt_dir]

            # Get highest index checkpoint file each
            message = "Latest model checkpoint files:\n"
            for imdl, ckpt_dir in enumerate(self.ckpt_dir):

                # Get highest index checkpoint file
                ckpt_max = -1
                if os.path.exists(ckpt_dir):
                    for ckpt_file in os.listdir(ckpt_dir):
                        ckpt_num = re.findall("model_(\d+).pt", ckpt_file)
                        ckpt_num = (int(ckpt_num[0]) if ckpt_num else -1)
                        if ckpt_max < ckpt_num:
                            ckpt_max = ckpt_num

                # If no checkpoint files available return None
                if ckpt_max < 0:
                    ckpt_name.append(None)
                    message += (
                        f" No file found in {ckpt_dir:s} for model "
                        + f"{imdl:d}!\n")
                else:
                    ckpt_name_i = os.path.join(
                        ckpt_dir, f'model_{ckpt_max:d}.pt')
                    ckpt_name.append(ckpt_name_i)
                    message += (
                        f" Found '{ckpt_name_i:s}' for model {imdl:d}!\n")
            message = message[:-1]

            if verbose:
                self.logger.info(message)

        # Get last model checkpoint file
        else:

            # Get model checkpoint file path
            if self.model_ensemble:
                ckpt_dir = self.ckpt_dir[imodel]
            else:
                ckpt_dir = self.ckpt_dir

            # Get highest index checkpoint file
            ckpt_max = -1
            if os.path.exists(ckpt_dir):
                for ckpt_file in os.listdir(ckpt_dir):
                    ckpt_num = re.findall("model_(\d+).pt", ckpt_file)
                    ckpt_num = (int(ckpt_num[0]) if ckpt_num else -1)
                    if ckpt_max < ckpt_num:
                        ckpt_max = ckpt_num

            # If no checkpoint files available return None
            if ckpt_max < 0:
                ckpt_name = None
                if verbose and self.model_ensemble:
                    self.logger.info(
                        f"No latest checkpoint file found in {ckpt_dir:s} "
                        + f"for model {imdl:d}!")
                elif verbose:
                    self.logger.info(
                        f"No latest checkpoint file found in {ckpt_dir:s}!")
            else:
                ckpt_name = os.path.join(ckpt_dir, f'model_{ckpt_max:d}.pt')
                if verbose and self.model_ensemble:
                    self.logger.info(
                        f"Latest checkpoint file '{ckpt_name}' found "
                        + f"for model {imdl:d}!")
                elif verbose:
                    self.logger.info(
                        f"Latest checkpoint file '{ckpt_name}' found")

        return ckpt_name

    def get_specific_checkpoint(
        self,
        checkpoint_label: Union[int, str],
        imodel: Optional[int] = None,
        verbose: Optional[bool] = True,
    ) -> Union[str, List[str]]:
        """
        Get last or specific model checkpoint file path with the highest
        epoch number.

        Parameters
        ----------
        checkpoint_label: (str, int)
            If string and a valid file path, load the respective checkpoint
            file.
            If integer, load the checkpoint file of the respective epoch
            number.
        imodel: int, optional, default None
            Model index in case of a model ensemble

        Returns
        -------
        (str, list(str))
            Torch module checkpoint file path(s)

        """

        # Load specific model checkpoint number
        if utils.is_integer(checkpoint_label):

            # For model ensembles without certain model definition check
            # in all model subdirectories
            if self.model_ensemble and imodel is None:

                # Get model checkpoint file paths
                ckpt_name = [
                    os.path.join(ckpt_dir, f'model_{checkpoint_label:d}.pt')
                    for ckpt_dir in self.ckpt_dir]

                # Check existence of each file
                message = ""
                for imdl, ckpt_name_i in enumerate(ckpt_name):
                    if not os.path.exists(ckpt_name_i):
                        ckpt_name[imdl] = None
                        message += (
                            f" Checkpoint file '{ckpt_name_i:s}' for model "
                            + f"{imdl:d} not found!\n")
                    else:
                        message += (
                            f" Found checkpoint file '{ckpt_name_i:s}' for "
                            + f"model {imdl:d}!\n")
                message = message[:-1]

                if np.any(
                    [ckpt_name_i is None for ckpt_name_i in ckpt_name]
                ):
                    message = "Checkpoint files not found:\n" + message
                    raise FileNotFoundError(message)
                elif verbose:
                    message = "Checkpoint files found:\n" + message
                    self.logger.info(message)

            else:

                # Get model checkpoint file path
                if self.model_ensemble:
                    ckpt_dir = self.ckpt_dir[imodel]
                else:
                    ckpt_dir = self.ckpt_dir
                ckpt_name = os.path.join(
                    ckpt_dir, f'model_{checkpoint_label:d}.pt')

                # Check existence
                if not os.path.exists(ckpt_name):
                    raise FileNotFoundError(
                        f"Checkpoint file '{ckpt_name}' not found!")
                elif verbose and self.model_ensemble:
                    self.logger.info(
                        f"Checkpoint file '{ckpt_name}' found "
                        + f"for model {imdl:d}!")
                elif verbose:
                    self.logger.info(
                        f"Checkpoint file '{ckpt_name}' found!")

        # Load specific model checkpoint file
        elif utils.is_string(checkpoint_label):

            # For model ensembles without certain model definition is not
            # supported
            if self.model_ensemble and imodel is None:
                raise NotImplementedError(
                    "For model ensembles, the definition of the model number "
                    + "'imodel' is mandatory")

            # Check for checkpoint file
            else:

                ckpt_name = checkpoint_label

                # Check existence
                if not os.path.exists(ckpt_name):
                    raise FileNotFoundError(
                        f"Checkpoint file '{ckpt_name}' does not exist!")
                elif verbose:
                    self.logger.info(
                        f"Checkpoint file '{ckpt_name}' found!")

        else:

            raise SyntaxError(
                "Input for the model checkpoint label 'checkpoint_label' "
                + "is not a valid data type!")

        return ckpt_name
