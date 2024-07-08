import os
import sys
import json
import logging
import argparse
from typing import Optional, List, Dict, Tuple, Union, Any

import numpy as np

import torch

# import tensorflow as tf

from .. import settings
from .. import utils

# from ... import Asparagus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_from_tensorflow(
    checkpoint,
    config,
    v1=None,
    model_directory=None,
    model_dtype=None,
):
    """
    Convert tensorflow checkpoint files into torch checkpoint files

    **Note**: This function is not yet fully implemented. Check other options in folder `convert_tf_to_torch`.
    """
    
    #tf.compat.v1.disable_eager_execution()
    #tf.compat.v1.reset_default_graph()
    
    # Check input
    if model_directory is None:
        head, tail = os.path.split(checkpoint)
        model_directory = tail
    if model_dtype is None:
        model_dtype = torch.float64
    
    # Load Tensorflow checkpoint file
    logger.info(f"INFO:\nLoad checkpoint '{checkpoint:s}'.\n")
    chkpt = tf.train.load_checkpoint(checkpoint)
    shape_from_key = chkpt.get_variable_to_shape_map()
    dtype_from_key = chkpt.get_variable_to_dtype_map()

    # Read additional information from config file
    config_data = get_tf_config(config)
    
    # Check PhysNet version
    if v1 is None:
        if 'model/_embeddings/.ATTRIBUTES/VARIABLE_VALUE' in shape_from_key:
            v1 = False
        else:
            v1 = True
    if v1:
        with open('physnet_tensorflow_v1_asparagus.json', 'r') as f:
            dct_conversion = json.load(f)
    else:
        with open('physnet_tensorflow_v2_asparagus.json', 'r') as f:
            dct_conversion = json.load(f)

    # Prepare torch model state
    model_state = {}
    for tf_name, shape in shape_from_key.items():

        # Find related torch parameter
        torch_key_match = [tf_name in name for name in dct_conversion.keys()]
        if sum(torch_key_match) > 1:
            raise SyntaxError(
                f"Tensorflow key '{tf_name:s}' was found multiple times!")
        elif sum(torch_key_match) == 0:
            logger.warning(
                f"WARNING:\nCheckpoint parameter '{tf_name:s}' not assigned!"
                + "\n")
            continue
        torch_name = np.array(
            list(dct_conversion.values()))[torch_key_match][0]

        # Get Tensorflow tensor and convert to Torch tensor
        tf_tensor = chkpt.get_tensor(tf_name)
        torch_tensor = torch.tensor(tf_tensor, dtype=model_dtype)

        # Add to model state dictionary
        model_state[torch_name] = torch_tensor

    # Cutoff radii
    cutoff = config_data.cutoff
    model_state['input_model.input_cutoff_fn.cutoff'] = cutoff
    model_state['input_model.input_descriptor_fn.rbf_cutoff_fn.cutoff'] = (
        cutoff)
    model_state['electrostatic_model.switch_fn.cutoff'] = cutoff
    logger.info(f"INFO:\nCutoff radii assigned!")

    # Energy and charge scaling
    if v1:
        escale = torch.tensor(chkpt.get_tensor('Escale:0'), dtype=model_dtype)
        eshift = torch.tensor(chkpt.get_tensor('Eshift:0'), dtype=model_dtype)
        qscale = torch.tensor(chkpt.get_tensor('Qscale:0'), dtype=model_dtype)
        qshift = torch.tensor(chkpt.get_tensor('Qshift:0'), dtype=model_dtype)
    else:
        escale = torch.tensor(
            chkpt.get_tensor('model/_Escale/.ATTRIBUTES/VARIABLE_VALUE'),
            dtype=model_dtype)
        eshift = torch.tensor(
            chkpt.get_tensor('model/_Eshift/.ATTRIBUTES/VARIABLE_VALUE'),
            dtype=model_dtype)
        qscale = torch.tensor(
            chkpt.get_tensor('model/_Qscale/.ATTRIBUTES/VARIABLE_VALUE'),
            dtype=model_dtype)
        qshift = torch.tensor(
            chkpt.get_tensor('model/_Qshift/.ATTRIBUTES/VARIABLE_VALUE'), 
            dtype=model_dtype)
    model_state['atomic_energies_scaling'] = torch.stack(
        [escale, eshift], dim=1)
    model_state['atomic_charges_scaling'] = torch.stack(
        [qscale, qshift], dim=1)
    logger.info(f"INFO:\nEnergy and charge scaling assigned!")

    # Unit conversion parameter
    if config_data.use_electrostatic:
        model_state['electrostatic_model.kehalf'] = 7.199822675975274
    if config_data.use_dispersion:
        model_state['dispersion_model.distances_model2Bohr'] = 1./0.52917726
        model_state['dispersion_model.energies_Hatree2model'] = 27.21138505

    # Get Asparagus config dictionary
    config_legacy = get_torch_config(config_data)

    model = Asparagus(config_legacy)
    model_calc = Asparagus(config_legacy)._get_Calculator(config_legacy)
    model_calc.load_state_dict(model_state)
    
    print(model)
    print(model_calc)
    
    exit()
    
    # Initialize Filemanager
    filemanager = utils.FileManager(
        model_directory=model_directory)

    

    ## Load one asparagus model
    #model_param = torch.load("formaldehyde_best_model.pt")
    #for key, param in model_param.items():
        #print()
        #print(key)
        #print("------------------------")
        ##print(param)
        #if key == 'epoch':
            #print(param, sys.getsizeof(param))
        #else:
            #for key2, param2 in param.items():
                #print(key2, param2.shape, sys.getsizeof(param2))
                ##print(param2)


def get_torch_config(config_data):
    """
    Prepare an Asparagus config dictionary compatible with Tensorflow PhysNet.
    """
    
    # Initialize Asparagus config dictionary
    config_legacy = {}
    
    # Calculator Model
    config_legacy['model_type'] = 'PhysNet_original'
    
    return config_legacy
    

def get_tf_config(config):
    """
    Read config file information
    """

    # Initiate parser
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    # Add arguments
    parser.add_argument(
        "--restart", type=str, default=None, 
        help="Restart training from a specific folder")
    parser.add_argument(
        "--num_features", default=128, type=int, 
        help="Dimensionality of feature vectors")
    parser.add_argument(
        "--num_basis", default=64, type=int, 
        help="Number of radial basis functions")
    parser.add_argument(
        "--num_blocks", default=5, type=int, 
        help="Number of interaction blocks")
    parser.add_argument(
        "--num_residual_atomic", default=2, type=int, 
        help="Number of residual layers for atomic refinements")
    parser.add_argument(
        "--num_residual_interaction", default=3, type=int, 
        help="Number of residual layers for the message phase")
    parser.add_argument(
        "--num_residual_output", default=1, type=int, 
        help="Number of residual layers for the output blocks")
    parser.add_argument(
        "--cutoff", default=10.0, type=float, 
        help="Cutoff distance for short range interactions")
    parser.add_argument(
        "--use_electrostatic", default=1, type=int, 
        help="Use electrostatics in energy prediction (0/1)")
    parser.add_argument(
        "--use_dispersion", default=1, type=int, 
        help="Use dispersion in energy prediction (0/1)")
    parser.add_argument(
        "--grimme_s6", default=None, type=float, 
        help="Grimme s6 dispersion coefficient")
    parser.add_argument(
        "--grimme_s8", default=None, type=float, 
        help="Grimme s8 dispersion coefficient")
    parser.add_argument(
        "--grimme_a1", default=None, type=float, 
        help="Grimme a1 dispersion coefficient")
    parser.add_argument(
        "--grimme_a2", default=None, type=float, 
        help="Grimme a2 dispersion coefficient")
    parser.add_argument(
        "--dataset", type=str, 
        help="File path to dataset")
    parser.add_argument(
        "--num_train", type=int, 
        help="Number of training samples")
    parser.add_argument(
        "--num_valid", type=int, 
        help="Number of validation samples")
    parser.add_argument(
        "--batch_size", type=int, 
        help="Batch size used per training step")
    parser.add_argument(
        "--valid_batch_size", type=int, 
        help="Batch size used for going through validation_set")
    parser.add_argument(
        "--seed", default=np.random.randint(1000000), type=int, 
        help="Seed for splitting dataset into training/validation/test")
    parser.add_argument(
        "--max_steps", default=10000, type=int, 
        help="Maximum number of training steps")
    parser.add_argument(
        "--learning_rate", default=0.001, type=float, 
        help="Learning rate used by the optimizer")
    parser.add_argument(
        "--decay_steps", default=1000, type=int, 
        help="Decay the learning rate every N steps by decay_rate")
    parser.add_argument(
        "--decay_rate", default=0.1, type=float, 
        help=(
            "Factor with which the learning rate gets multiplied by every "
            + "decay_steps steps")
        )
    parser.add_argument(
        "--keep_prob", default=1.0, type=float, 
        help="One minus dropout rate")
    parser.add_argument(
        "--max_norm", default=1000.0, type=float, 
        help="Max norm for gradient clipping")
    parser.add_argument(
        "--ema_decay", default=0.999, type=float, 
        help="Exponential moving average decay used by the trainer")
    parser.add_argument(
        "--rate", default=0.0, type=float, 
        help="Rate probability for dropout regularization of rbf layer")
    parser.add_argument(
        "--l2lambda", default=0.0, type=float, 
        help="Lambda multiplier for l2 loss (regularization)")
    parser.add_argument(
        "--nhlambda", default=0.1, type=float, 
        help="Lambda multiplier for non-hierarchicality loss (regularization)")
    parser.add_argument(
        "--force_weight", default=52.91772105638412, type=float,
        help=(
            "This defines the force contribution to the loss function "
            + "relative to the energy contribution (to take into account the "
            + "different numerical range)")
        )
    parser.add_argument(
        "--charge_weight", default=14.399645351950548, type=float,
        help=(
            "This defines the charge contribution to the loss function "
            + "relative to the energy contribution (to take into account the "
            + "different numerical range)")
        )
    parser.add_argument(
        "--dipole_weight", default=27.211386024367243, type=float,
        help=(
            "This defines the dipole contribution to the loss function "
            + "relative to the energy contribution (to take into account the "
            + "different numerical range)")
        )
    parser.add_argument(
        "--summary_interval", default=5, type=int, 
        help="Write a summary every N steps")
    parser.add_argument(
        "--validation_interval", default=5, type=int, 
        help="Check performance on validation set every N steps")
    parser.add_argument(
        "--show_progress", default=True, type=bool, 
        help="Show progress of the epoch")
    parser.add_argument(
        "--save_interval", default=5, type=int, 
        help="Save progress every N steps")
    parser.add_argument(
        "--record_run_metadata", default=0, type=int, 
        help="Records metadata like memory consumption etc.")
    
    # Read config file
    return parser.parse_args(["@" + config])
    
