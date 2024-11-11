import logging
from typing import Optional, List, Tuple, Dict, Union, Any, Callable

import numpy as np

from asparagus import data
from asparagus import utils

__all__ = [
    'compute_property_scaling',
    'compute_system_property_scaling',
    'compute_atomic_property_scaling',
    'compute_atomic_property_sum_scaling'
    'compute_atomic_energies_scaling',
    'get_energy_properties_from_dataset']

# Initialize logger
name = f"{__name__:s}"
logger = utils.set_logger(logging.getLogger(name))


def compute_property_scaling(
    dataset: Union[data.DataSet, data.DataSubSet],
    data_properties: List[str],
    property_scaling: Dict[str, List[float]],
    property_atom_scaled: Dict[str, str],
) -> Dict[str, List[float]]:
    """
    Compute open property statistics.

    Parameters
    ----------
    dataset: (data.DataSet, data.DataSubSet)
        Dataset or subset object containing data to compute property scaling
        statistics.
    data_properties: list(str)
        Available data properties
    property_scaling: dict(str, list(float))
        Already computed property statistics dictionary
    property_atom_scaled: dict(str, str)
        Property statistics (key) will be scaled by the number of atoms
        per system and stored with new property label (item).
        e.g. {'energy': 'atomic_energies'}

    Return
    ------
    dict(str, list(float))
        Property statistics dictionary

    """

    # Check for missing properties
    open_properties = []
    for prop in data_properties:
        if prop not in property_scaling:
            open_properties.append(prop)
        if prop in property_atom_scaled:
            atom_prop = property_atom_scaled[prop]
            if (
                atom_prop not in property_scaling
                and prop not in open_properties
            ):
                open_properties.append(prop)

    # Do computation if properties are missing
    if not open_properties:
        return {}

    # Initialize result dictionary
    scaling_result = {}
    for prop in open_properties:
        scaling_result[prop] = [0.0, 0.0]
        if prop in property_atom_scaled:
            atom_prop = property_atom_scaled[prop]
            scaling_result[atom_prop] = [0.0, 0.0]

    # Announce start of property statistics calculation
    message = "Start computing data statistics for properties:\n"
    for prop in scaling_result:
        message += f" {prop:s}\n"
    message += "This might take a moment."
    logger.info(message)

    # Iterate over data properties and compute property mean
    Nsamples = 0.0
    for sample in dataset:

        # Iterate over sample properties
        for prop in open_properties:

            # Get property values
            vals = sample.get(prop).numpy().reshape(-1)

            # Compute average
            scalar = np.mean(vals)
            scaling_result[prop][0] = (
                scaling_result[prop][0]
                + (scalar - scaling_result[prop][0])/(Nsamples + 1.0)
                ).item()

            # Scale by atom number if requested
            if prop in property_atom_scaled:

                # Compute atom scaled average
                atom_prop = property_atom_scaled[prop]
                Natoms = sample.get('atoms_number').numpy().reshape(-1)
                vals /= Natoms.astype(float)

                # Compute statistics normalized by atom numbers
                scalar = np.mean(vals)
                scaling_result[atom_prop][0] = (
                    scaling_result[atom_prop][0]
                    + (scalar - scaling_result[atom_prop][0])
                    / (Nsamples + 1.0)
                    ).item()

        # Increment sample counter
        Nsamples += 1.0

    # Iterate over training data properties and compute standard
    # deviation
    for sample in dataset:

        # Iterate over sample properties
        for prop in open_properties:

            # Get property values
            vals = sample.get(prop).numpy().reshape(-1)
            Nvals = len(vals)

            # Compute standard deviation contribution
            for scalar in vals:
                scaling_result[prop][1] = (
                    scaling_result[prop][1]
                    + (scalar - scaling_result[prop][0])**2/Nvals)

            # Scale by atom number if requested
            if prop in property_atom_scaled:

                # Compute atom scaled standard deviation contribution
                atom_prop = property_atom_scaled[prop]
                Natoms = sample.get('atoms_number').numpy().reshape(-1)
                vals /= Natoms.astype(float)

                # Compute atom scaled standard deviation contribution
                for scalar in vals:
                    scaling_result[atom_prop][1] = (
                        scaling_result[atom_prop][1]
                        + (scalar - scaling_result[atom_prop][0])**2
                        / Nvals)

    # Iterate over sample properties to complete standard deviation
    for prop in open_properties:
        scaling_result[prop][1] = np.sqrt(
            scaling_result[prop][1]/Nsamples).item()
        if prop in property_atom_scaled:
            atom_prop = property_atom_scaled[prop]
            scaling_result[atom_prop][1] = np.sqrt(
                scaling_result[atom_prop][1]/Nsamples).item()

    return scaling_result


def compute_atomic_energies_scaling(
    dataset: Union[data.DataSet, data.DataSubSet],
    energy_unit: str
) -> (Dict[str, List[float]], str):
    """
    Compute atomic energies statistics.

    Parameters
    ----------
    dataset: (data.DataSet, data.DataSubSet)
        Dataset or subset object containing data to compute atomic energies
        scaling statistics.
    energy_unit: str
        Atomic energies unit label.

    Return
    ------
    dict(str, list(float))
        Property statistics dictionary
    str
        Computation details message

    """

    # Announce start of property statistics calculation
    message = (
        "Start computing atomic energies statistics!\n"
        + "This might take a moment.")
    logger.info(message)

    # Avoid estimating atomic energies from energy property if atomic energies
    # are available from the dataset - TODO

    # Collect system data
    energies, atoms_number, atomic_numbers, sys_i = (
        get_energy_properties_from_dataset(dataset))

    # Compute atomic energies scaling from the system energy
    atomic_energies_scaling, fit_rmse, fit_complete = (
        compute_atomic_property_sum_scaling(
            energies,
            None,
            None,
            atoms_number,
            atomic_numbers,
            sys_i,
            verbose=True)
        )

    # Prepare atomic energies computation information output
    if fit_complete is None:
        message = (
            "Atom number scaled system energy statistics is used.\n")
    elif fit_complete:
        message = (
            "Prediction done by atomic energies fit. "
            + "System energy prediction RMSE/atom = "
            + f"{fit_rmse:.3E} {energy_unit:s}\n"
            )
    else:
        message = (
            "Prediction by atomic energies fit failed. "
            + "Atom number scaled system energy statistics is used.\n")

    return atomic_energies_scaling, message


def get_energy_properties_from_dataset(
    dataset: Union[data.DataSet, data.DataSubSet],
) -> (List[float], List[float], List[float], List[float]):
    """
    Collect energy data together with other system data from a dataset

    Parameters
    ----------
    dataset: (data.DataSet, data.DataSubSet)
        Dataset or subset object containing data to collect data from.
    
    Return
    ------
    np.ndarray(float)
        System energies
    np.ndarray(float)
        System atoms numbers
    np.ndarray(float)
        System atomic numbers
    np.ndarray(float)
        System indices of atoms

    """

    # Initialize system and energy property lists
    Nsamples = len(dataset)
    atoms_number = np.zeros(Nsamples, dtype=int)
    energies = np.zeros(Nsamples, dtype=float)
    atomic_numbers = []
    sys_i = []
    
    # Iterate over data samples and collect
    # system and energy properties
    for ismpl, sample in enumerate(dataset):
        atoms_number[ismpl] = sample['atoms_number']
        energies[ismpl] = sample['energy']
        atomic_numbers += list(sample['atomic_numbers'])
        sys_i += [ismpl]*sample['atoms_number']
    atomic_numbers = np.array(atomic_numbers, dtype=int)
    sys_i = np.array(sys_i, dtype=int)
    Natoms = atomic_numbers.shape[0]
    
    return energies, atoms_number, atomic_numbers, sys_i


def compute_system_property_scaling(
    property_reference: np.ndarray,
    property_prediction: np.ndarray,
) -> List[float]:
    """
    Compute scaling parameter to match system property mean and distribution
    between reference and prediction data. If prediction data are not
    available, compute scaling parameter under the assumption that the
    prediction mean is zero and the distribution is one.

    Parameters
    ----------
    property_reference: np.ndarray
        Reference property values
    property_prediction: np.ndarray
        Predicted property values

    Return
    ------
    list(float)
        System property shift and scaling factor

    """

    # Compute average and standard deviation of reference property
    property_reference_shift = np.mean(property_reference)
    property_reference_scale = np.std(property_reference)

    # Compute average and standard deviation of prediction property
    if property_prediction is None:
        property_prediction_shift = 0.0
        property_prediction_scale = 1.0
    else:
        property_prediction_shift = np.mean(property_prediction)
        property_prediction_scale = np.std(property_prediction)

    # Compute property shift and scaling parameter
    property_shift = property_reference_shift - property_prediction_shift
    property_scale = property_reference_scale/property_prediction_scale

    return [property_shift, property_scale]


def compute_atomic_property_scaling(
    property_reference: np.ndarray,
    property_prediction: np.ndarray,
    atomic_numbers: np.ndarray,
) -> Dict[str, List[float]]:
    """
    Compute scaling parameter to match atomic property mean and distribution
    between reference and prediction data. If prediction data are not
    available, compute scaling parameter under the assumption that the
    prediction mean is zero and the distribution is one.

    Parameters
    ----------
    property_reference: np.ndarray
        Reference property values
    property_prediction: np.ndarray
        Predicted property values
    atomic_numbers: np.ndarray
        System atomic numbers

    Return
    ------
    dict(str, list(float))
        Atomic property shift and scaling factor

    """

    # Initialize atom energies scaling dictionary
    property_scaling = {}

    # Get list of available elements
    atomic_numbers_list, atomic_numbers_indices = np.unique(
        atomic_numbers, return_inverse=True)

    # Iterate over available elements
    for iatom, atomic_number in enumerate(atomic_numbers_list):

        # Compute average and standard deviation of reference atomic property
        property_reference_shift = np.mean(
            property_reference[atomic_number==atomic_numbers])
        property_reference_scale = np.std(
            property_reference[atomic_number==atomic_numbers])
        if property_reference_scale == 0.0:
            property_reference_scale = 1.0

        # Compute average and standard deviation of prediction atomic property
        if property_prediction is None:
            property_prediction_shift = 0.0
            property_prediction_scale = 1.0
        else:
            property_prediction_shift = np.mean(
                property_prediction[atomic_number==atomic_numbers])
            property_prediction_scale = np.std(
                property_prediction[atomic_number==atomic_numbers])
            if property_prediction_scale == 0.0:
                property_prediction_scale = 1.0

        # Compute atomic property shift and scaling parameter
        property_scale = property_reference_scale/property_prediction_scale
        property_shift = (
            property_reference_shift
            - property_prediction_shift*property_scale)

        # Assign atomic energies statistics
        property_scaling[atomic_number] = [
            property_shift,
            property_scale]

    return property_scaling


def compute_atomic_property_sum_scaling(
    property_reference: np.ndarray,
    property_prediction: np.ndarray,
    property_prediction_scaling: Dict[int, np.ndarray],
    atoms_number: np.ndarray,
    atomic_numbers: np.ndarray,
    sys_i: np.ndarray,
    verbose: Optional[bool] = False,
) -> Dict[str, List[float]]:
    """
    Compute atomic property statistics from system property values for
    scaling by the assumption that the system property is the sum of the
    atomic property.

    Parameters
    ----------
    property_reference: np.ndarray
        Reference system property values
    property_prediction: np.ndarray
        Predicted atomic property values
    property_prediction_scaling: dict(int, np.ndarray)
        Applied shift term and scaling factor for the property prediction
    atoms_number: np.ndarray
        System atoms numbers
    atomic_numbers: np.ndarray
        System atomic numbers
    sys_i: np.ndarray
        System indices of atoms

    Return
    ------
    dict(str, list(float))
        Atomic property statistics dictionary

    """

    # Initialize atomic property scaling dictionary
    property_scaling = {}

    # Collect system data
    Nsystems = atoms_number.shape[0]
    Natoms = atomic_numbers.shape[0]

    # Check if database has samples of various system compositions
    various_systems = False
    if len(np.unique(atoms_number)) == 1:
        for ismpl in range(Nsystems):
            if ismpl:
                # Get system composition information
                atomic_numbers_i, atom_counts_i = np.unique(
                    atomic_numbers[sys_i == ismpl], return_counts=True)
                # Check for same atom types and count
                if not (
                    atomic_numbers_0.shape == atomic_numbers_i.shape
                    and np.all(atomic_numbers_0 == atomic_numbers_i)
                    and np.all(atom_counts_0 == atom_counts_i)
                ):
                    various_systems = True
                    break
            else:
                # Get reference system to compare
                atomic_numbers_0, atom_counts_0 = np.unique(
                    atomic_numbers[sys_i == ismpl], return_counts=True)
    else:
        various_systems = True

    # Get list of available elements
    atomic_numbers_list, atomic_numbers_indices = np.unique(
        atomic_numbers, return_inverse=True)

    # Compute atomic energies average and standard deviation
    property_atomic_reference_mean = np.mean(property_reference/atoms_number)
    property_atomic_reference_stdv = np.std(
        property_reference/atoms_number)
    if property_atomic_reference_stdv == 0.0:
        property_atomic_reference_stdv = 1.0

    # Set initial property shift and scaling parameter as atom scaled average
    # and standard deviation of the system reference property
    property_scale = np.full(
        atomic_numbers_list.shape,
        property_atomic_reference_stdv,
        dtype=float)
    property_shift = np.full(
        atomic_numbers_list.shape,
        property_atomic_reference_mean,
        dtype=float)

    if property_prediction_scaling is None:
        model_scaling = np.full(
            atomic_numbers_list.shape + (2, ), [0.0, 1.0], dtype=float)
    else:
        model_scaling = np.zeros(
            atomic_numbers_list.shape + (2, ), dtype=float)
        for iatom, atomic_number in enumerate(atomic_numbers_list):
            model_scaling[iatom] = property_prediction_scaling[atomic_number]

    # Include atomic property prediction into shift and scaling parameter
    if property_prediction is not None:

        # Iterate over available elements
        for iatom, atomic_number in enumerate(atomic_numbers_list):

            property_atomic_prediction_shift = np.mean(
                property_prediction[atomic_number==atomic_numbers])
            property_atomic_prediction_scale = np.std(
                property_prediction[atomic_number==atomic_numbers])
            if property_atomic_prediction_scale == 0.0:
                property_atomic_prediction_scale = 1.0

            # Attune atomic property shift and scaling parameter
            property_scale[iatom] /= property_atomic_prediction_scale
            property_shift[iatom] -= (
                property_atomic_prediction_shift*property_scale[iatom])

    # Define energy computation and evaluation function
    def property_sum(
        shifts,
        scales,
        Nsys=Nsystems,
        sidcs=sys_i,
        aidcs=atomic_numbers_indices,
    ):

        # Initialize prediction array per system
        prediction = np.zeros(Nsys, dtype=float)

        # Collect property per atom type
        if property_prediction is None:
            atomic_property = np.array(shifts)[aidcs]
        else:
            atomic_property = (
                (property_prediction[aidcs] - model_scaling[aidcs, 0])
                * scales[aidcs]
                + shifts[aidcs] + model_scaling[aidcs, 0])

        # Sum up atomic property to system property
        np.add.at(prediction, sidcs, atomic_property)

        return prediction

    def system_property_eval(
        pars,
        reference=property_reference,
        aidcs=atomic_numbers_indices,
    ):

        # Collect property per atom type
        shifts, scales = pars.reshape(2, -1)
        prediction = property_sum(shifts, scales)

        # Compute root mean square error between reference and prediction
        # weighted by the systems atoms number
        rmse = np.sqrt(
            np.sum(atoms_number*(reference - prediction)**2)
            / np.sum(atoms_number))

        return rmse

    # If systems have different atom compositions,
    # fit atomic property to match best the system energies.
    if various_systems:

        # Start fitting procedure
        try:

            # Concatenate shift term and scaling factor
            pars = np.append(property_shift, property_scale)

            # Start fitting atomic energies shifts
            from scipy.optimize import minimize
            fit_result = minimize(
                system_property_eval,
                pars,
                method='bfgs')
            fit_rmse = fit_result.fun

            # Assign fit results
            property_shift, property_scale = fit_result.x.reshape(2, -1)

            # Set successful fitting flag
            fit_complete = True

        except (RuntimeError, ValueError):

            # Compute first guess RMSE
            fit_rmse = system_property_eval(system_property_eval)

            # Set unsuccessful fitting flag
            fit_complete = False

    else:

        # Set None fitting flag
        fit_complete = None
        fit_rmse = np.nan

    # Assign atomic energies statistics
    for iatom, atomic_number in enumerate(atomic_numbers_list):
        property_scaling[int(atomic_number)] = [
            property_shift[iatom],
            property_scale[iatom]]
    #
    # if property_prediction_scaling is not None:
    #     print("New scaling:")
    #     for iatom, atomic_number in enumerate(atomic_numbers_list):
    #         print(
    #             atomic_number,
    #             property_scaling[int(atomic_number)][0] +
    #             property_prediction_scaling[int(atomic_number)][0],
    #             property_scaling[int(atomic_number)][1] *
    #             property_prediction_scaling[int(atomic_number)][1])

    if verbose:
        return property_scaling, fit_rmse, fit_complete
    else:
        return property_scaling
