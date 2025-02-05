import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import torch

import numpy as np

import asparagus
from asparagus import settings
from asparagus import utils
from asparagus import data
from asparagus import model

__all__ = ['set_property_scaling_estimation']

# Initialize logger
name = f"{__name__:s}"
logger = utils.set_logger(logging.getLogger(name))


def set_property_scaling_estimation(
    model_calculator: model.BaseModel,
    data_loader: data.DataLoader,
    properties: List[str],
    use_model_prediction: Optional[bool] = True,
    model_conversion: Optional[Dict[str, float]] = {},
    atomic_energies_guess: Optional[bool] = True,
    set_shift_term: Optional[bool] = True,
    set_scaling_factor: Optional[bool] = True,
):
    """
    Estimate model property scaling parameters and prepare respective
    dictionaries.

    Parameters
    ----------
    model_calculator: model.BaseModel
        Model calculator to set property scaling estimation.
    data_loader: data.Dataloader
        Reference data loader to provide reference system data
    properties: list(str)
        Properties list generally predicted by the output module
        where the property scaling is usually applied.
    use_model_prediction: bool, optional, default True
        Use the current prediction by the model calculator of the model
        properties to enhance the guess of the scaling parameter.
        If not, predictions are expected to be zero.
    model_conversion: dict(str, float), optional, default {}
        Dictionary of model to data property unit conversion factors
    atomic_energies_guess: bool, optional, default True
        In case atomic energies are not available by the reference data
        set, which is normally the case, predict anyways from a guess of
        the difference between total reference energy and predicted energy.
    set_shift_term: bool, optional, default True
        If True, set or update the shift term. Else, keep previous value.
    set_scaling_factor: bool, optional, default True
        If True, set or update the scaling factor. Else, keep previous value.

    """

    # Check if scaling parameter for a requested property can be estimated
    # from the available data
    properties_available = []
    for prop in properties:
        if (
            prop in data_loader.data_properties
            and prop not in properties_available
        ):
            properties_available.append(prop)
        elif (
            prop == 'atomic_energies'
            and atomic_energies_guess
            and 'energy' in data_loader.data_properties
            and 'energy' not in properties_available
        ):
            properties_available.append('energy')

    # Return empty result dictionary if no property scaling parameter guess
    # can be computed
    if not properties_available:
        return {}

    # Run 2-phase property scaling estimation by first predict and apply
    # scaling parameter for not directly energy related properties.
    # Energy or atomic energies scaling parameters are scaled in a second run.
    properties_first = [
        prop for prop in properties_available
        if prop not in ['energy', 'atomic_energies']]
    properties_second = [
        prop for prop in properties_available
        if prop in ['energy', 'atomic_energies']]

    # Run first property scaling estimation
    if properties_first:

        property_scaling = estimate_property_scaling(
            model_calculator,
            data_loader,
            properties_first,
            use_model_prediction,
            model_conversion,
            atomic_energies_guess,
            set_shift_term=set_shift_term,
            set_scaling_factor=set_scaling_factor)

        # Set model property scaling parameter to the model calculator output
        # module
        model_calculator.set_property_scaling(
            property_scaling=property_scaling,
            set_shift_term=set_shift_term,
            set_scaling_factor=set_scaling_factor)

        # Print scaling log information
        logger.info(
            "Scaling parameter are set for model properties:\n"
            + "\n".join(property_scaling.keys()))

    # Run second property scaling estimation
    if properties_second:

        property_scaling = estimate_property_scaling(
            model_calculator,
            data_loader,
            properties_second,
            use_model_prediction,
            model_conversion,
            atomic_energies_guess,
            set_shift_term=set_shift_term,
            set_scaling_factor=set_scaling_factor)

        # Set model property scaling parameter to the model calculator output
        # module
        model_calculator.set_property_scaling(
            property_scaling=property_scaling,
            set_shift_term=set_shift_term,
            set_scaling_factor=set_scaling_factor)

        # Print scaling log information
        logger.info(
            "Scaling parameter are set for model properties:\n"
            + "\n".join(property_scaling.keys()))

    return


def estimate_property_scaling(
    model_calculator: model.BaseModel,
    data_loader: data.DataLoader,
    properties: List[str],
    use_model_prediction: bool,
    model_conversion: Dict[str, float],
    atomic_energies_guess: bool,
    set_shift_term: Optional[bool] = True,
    set_scaling_factor: Optional[bool] = True,
) -> Dict[str, Union[List[float], Dict[int, List[float]]]]:
    """
    Estimate model property scaling parameters and prepare respective
    dictionaries.

    Parameters
    ----------
    model_calculator: model.BaseModel
        Model calculator used for model property prediction.
    data_loader: data.Dataloader
        Reference data loader to provide reference system data
    properties: list(str)
        Properties list generally predicted by the output module
        where the property scaling is usually applied.
    use_model_prediction: bool
        Use the current prediction by the model calculator of the model
        properties to enhance the guess of the scaling parameter.
        If not, predictions are expected to be zero.
    model_conversion: dict(str, float)
        Dictionary of model to data property unit conversion factors
    atomic_energies_guess: bool
        In case atomic energies are not available by the reference data
        set, which is normally the case, predict anyways from a guess of
        the difference between total reference energy and predicted energy.
    set_shift_term: bool, optional, default True
        If True, estimate shift terms.
    set_scaling_factor: bool, optional, default True
        If True, estimate the scaling factors.

    Returns
    -------
    dict(str, (list(float), dict(int, float))
        Model property or atomic resolved scaling parameter [shift, scale]

    """

    # If requested, predict just non-derived model property predictions from
    # the data loader systems to support properties
    if use_model_prediction:

        logger.info(
            "Reference property data and model property prediction will "
            + "be collected.\nThis might take a moment.")

        # Compute and get model prediction and reference data
        properties_reference, properties_prediction = (
            get_model_prediction_and_reference_properties(
                model_calculator,
                data_loader,
                properties,
                model_conversion)
            )

        # Get current property scaling parameter
        model_scaling = model_calculator.get_property_scaling()

    else:

        logger.info("Reference property data will be collected.")

        # Compute and get model prediction and reference data
        properties_reference = (
            get_reference_properties(
                data_loader,
                properties,
                model_conversion)
            )
        properties_prediction = {}
        model_scaling = {}

    property_scaling = compute_property_scaling(
        properties_reference,
        properties_prediction,
        model_scaling,
        atomic_energies_guess,
        set_shift_term=set_shift_term,
        set_scaling_factor=set_scaling_factor)

    return property_scaling


def get_model_prediction_and_reference_properties(
    model_calculator: model.BaseModel,
    data_loader: data.DataLoader,
    properties_available: List[str],
    model_conversion: Dict[str, float],
) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
    """
    Compute and provide model prediction and reference data.

    Parameters
    ----------
    model_calculator: model.BaseModel
        Asparagus model calculator object
    data_loader: data.Dataloader
        Reference data loader to provide reference system data
    properties_available: list(str)
        Properties list generally predicted by the output module
        where the property scaling is usually applied.
    model_conversion: dict(str, float)
        Dictionary of model to data property unit conversion factors

    Returns
    -------
    dict(str, numpy.ndarray)
        Reference system and property data
    dict(str, numpy.ndarray)
        Model property prediction

    """

    # Prepare reference dictionary
    properties_reference = {}
    properties_system = ['atoms_number', 'atomic_numbers', 'sys_i']
    for prop in properties_system:
        properties_reference[prop] = []
    for prop in properties_available:
        properties_reference[prop] = []

    # Prepare prediction dictionary
    properties_prediction = {}

    # Loop over training batches
    for ib, batch in enumerate(data_loader):

        # Predict model properties from data batch
        prediction = model_calculator(
            batch,
            no_derivation=True,
            verbose_results=True)

        # Append prediction to library
        for prop, item in prediction.items():
            if properties_prediction.get(prop) is None:
                properties_prediction[prop] = []
            properties_prediction[prop].append(
                item.cpu().detach().reshape(-1))

        # Append system information and reference properties
        for prop in properties_system + properties_available:
            properties_reference[prop].append(
                batch[prop].cpu().detach())

    # Concatenate prediction and reference results
    for prop, item in properties_prediction.items():
        if ib:
            properties_prediction[prop] = torch.cat(item).numpy()
        else:
            properties_prediction[prop] = (item[0].numpy())

    for prop in properties_system + properties_available:
        if ib:
            if prop == 'sys_i':
                sys_i = []
                next_sys_i = 0
                for batch_sys_i in properties_reference[prop]:
                    sys_i.append(batch_sys_i + next_sys_i)
                    next_sys_i = sys_i[-1][-1] + 1
                properties_reference[prop] = torch.cat(sys_i).numpy()
            else:
                properties_reference[prop] = torch.cat(
                    properties_reference[prop]
                    ).numpy()
        else:
            properties_reference[prop] = (
                properties_reference[prop][0].numpy())

        # Apply reference to model property unit conversion factor
        if prop in model_conversion:
            properties_reference[prop] = (
                properties_reference[prop]/model_conversion[prop])

    return properties_reference, properties_prediction


def get_reference_properties(
    data_loader: data.DataLoader,
    properties: List[str],
    model_conversion: Dict[str, float]
) -> (Dict[str, np.ndarray], Dict[str, np.ndarray]):
    """
    Provide reference data of properties.

    Parameters
    ----------
    data_loader: data.Dataloader
        Reference data loader to provide reference system data
    properties: list(str)
        Properties list generally predicted by the output module
        where the property scaling is usually applied.
    model_conversion: dict(str, float)
        Dictionary of model to data property unit conversion factors

    Returns
    -------
    dict(str, numpy.ndarray)
        Reference system and property data

    """

    # Prepare reference dictionary
    properties_reference = {}
    properties_system = ['atoms_number', 'atomic_numbers', 'sys_i']
    for prop in properties_system:
        properties_reference[prop] = []
    for prop in properties:
        properties_reference[prop] = []

    # Loop over training batches
    for ib, batch in enumerate(data_loader):

        # Append system information and reference properties
        for prop in properties_system + properties:
            properties_reference[prop].append(
                batch[prop].cpu().detach())

    # Concatenate reference results
    for prop in properties_system + properties:
        if ib:
            if prop == 'sys_i':
                sys_i = []
                next_sys_i = 0
                for batch_sys_i in properties_reference[prop]:
                    sys_i.append(batch_sys_i + next_sys_i)
                    next_sys_i = sys_i[-1][-1] + 1
                properties_reference[prop] = torch.cat(sys_i).numpy()
            else:
                properties_reference[prop] = torch.cat(
                    properties_reference[prop]
                    ).numpy()
        else:
            properties_reference[prop] = (
                properties_reference[prop].numpy())

        # Apply reference to model property unit conversion factor
        if prop in model_conversion:
            properties_reference[prop] = (
                properties_reference[prop]/model_conversion[prop])

    return properties_reference


def compute_property_scaling(
    properties_reference: Dict[str, np.ndarray],
    properties_prediction: Dict[str, np.ndarray],
    model_scaling: Dict[str, Union[List[float], Dict[int, List[float]]]],
    atomic_energies_guess: bool,
    set_shift_term: Optional[bool] = True,
    set_scaling_factor: Optional[bool] = True,
) -> Dict[str, Union[List[float], Dict[int, List[float]]]]:
    """
    Compute guess of property scaling parameters either from the reference
    data or even the deviation between reference and prediction.

    Parameters
    ----------
    properties_reference: dict(str, numpy.ndarray)
        Reference system and property data
    properties_prediction: dict(str, numpy.ndarray)
        Model calculator property prediction
    model_scaling: dict(str, (list(float), dict(int, float))
        Current Model property or atomic resolved scaling parameters
    atomic_energies_guess: bool
        Predict atomic energies scaling parameter eventually from the
        total system energy data.
    set_shift_term: bool, optional, default True
        If True, estimate shift terms.
    set_scaling_factor: bool, optional, default True
        If True, estimate the scaling factors.

    Returns
    -------
    dict(str, (list(float), dict(int, float))
        Model property or atomic resolved scaling parameter [shift, scale]

    """

    # Initialize property scaling parameters dictionary
    properties_scaling = {}

    # Iterate over property data
    properties_system = ['atoms_number', 'atomic_numbers', 'sys_i']
    for prop in properties_reference:

        # Skip system properties
        if prop in properties_system:
            continue

        # If model prediction is available scale model prediction to match
        # mean and deviation of the reference properties
        if (
            properties_prediction is not None
            and prop in properties_prediction
        ):

            # Compute scaling parameter for atom resolved properties
            if (
                properties_prediction[prop].shape[0]
                == properties_reference['sys_i'].shape[0]
            ):

                properties_scaling[prop] = (
                    data.compute_atomic_property_scaling(
                        properties_reference[prop],
                        properties_prediction[prop],
                        properties_reference['atomic_numbers'])
                    )

            # Compute scaling parameter for system resolved properties
            else:

                # Special case: Atomic energies scaling guess
                if prop == 'energy' and atomic_energies_guess:

                    # Prepare reference energy by subtracting energy 
                    # contribution which are not from the output module atomic
                    # energies
                    reference_output_energy = (
                        properties_reference['energy'].copy())
                    property_tag = '_energy'
                    property_list = []
                    for prop, item in properties_prediction.items():
                        if property_tag in prop[-len(property_tag):]:
                            if 'output_' in prop:
                                continue
                            else:
                                property_list.append(prop)
                                reference_output_energy -= item

                    # Scale atomic energies output to best match total system
                    # energy
                    properties_scaling['atomic_energies'] = (
                        data.compute_atomic_property_sum_scaling(
                            reference_output_energy,
                            properties_prediction.get(
                                'output_atomic_energies'),
                            model_scaling['atomic_energies'],
                            properties_reference['atoms_number'],
                            properties_reference['atomic_numbers'],
                            properties_reference['sys_i'],
                            set_shift_term=set_shift_term,
                            set_scaling_factor=set_scaling_factor)
                        )

                elif prop == 'atomic_energies':

                    # Prepare reference atomic energies to scale with the 
                    # output module atomic energies
                    reference_output_atomic_energies = ( 
                        properties_reference['atomic_energies'].copy())
                    property_tag = '_atomic_energies'
                    for prop, item in properties_prediction.items():
                        if property_tag in prop[-len(property_tag):]:
                            if 'output_' in prop:
                                continue
                            else:
                                reference_output_atomic_energies -= item

                    properties_scaling[prop] = (
                        data.compute_system_property_scaling(
                            reference_output_atomic_energies,
                            properties_prediction.get(
                                'output_atomic_energies'))
                        )

                else:

                    properties_scaling[prop] = (
                        data.compute_system_property_scaling(
                            properties_reference[prop],
                            properties_prediction[prop])
                        )

    # Combine with model property scaling parameters
    for prop in model_scaling:
        if prop in properties_scaling:
            if utils.is_dictionary(properties_scaling[prop]):
                for ai in properties_scaling[prop]:
                    properties_scaling[prop][ai][0] = (
                        model_scaling[prop][ai][0]
                        * properties_scaling[prop][ai][1]
                        + properties_scaling[prop][ai][0])
                    properties_scaling[prop][ai][1] *= (
                        model_scaling[prop][ai][1])
            else:
                properties_scaling[prop][0] = (
                    model_scaling[prop][0]*properties_scaling[prop][1]
                    + properties_scaling[prop][0])
                properties_scaling[prop][1] *= model_scaling[prop][1]

    return properties_scaling
