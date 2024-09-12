import logging
from typing import Optional

from ase import units

from .output import set_logger

# Initialize logger
name = f"{__name__:s}"
logger = set_logger(logging.getLogger(name))


def check_units(
    target_unit: Optional[str] = None,
    source_unit: Optional[str] = None,
    verbose: Optional[bool] = False,
):
    """
    Compare units and provide conversion factor from 'source_unit' to
    'target_unit'. If unit is None, conversion factor will be 1.0.

    Units are described by symbol (e.g. 'eV', 'kcal', 'e', 'Ang', 'Bohr' ...)
    and the proportionality symbol (e.g. '*', '/', '**')

    Parameters
    ----------
    target_unit: str, optional, default None
        Target property unit
    source_unit: str, optional, default None
        Source property unit

    Returns
    -------
    float
        Conversion factor from 'source_unit' to 'target_unit'
    bool
        If True: unit match; False: otherwise

    """

    # In case of undefined unit label
    if target_unit is None:
        target_unit = 'None'
    if source_unit is None:
        source_unit = 'None'

    if target_unit.strip().lower() == source_unit.strip().lower():

        # For matching units, return conversion factor 1.
        return 1.0, True

    else:

        # In case of mismatch, get conversion factors to ASE units
        target_factor_to_ase = convert_unit_ase(target_unit, verbose=verbose)
        source_factor_to_ase = convert_unit_ase(source_unit, verbose=verbose)

        # Get conversion factor from source to target unit
        factor_source_to_target = source_factor_to_ase/target_factor_to_ase

        # Print Extended information
        if verbose:
            logger.info(
                "Unit conversion from "
                + f"'{target_unit:s}' to '{source_unit:s}': "
                + f"{factor_source_to_target:.16e}")

        # Return Conversion factor
        return factor_source_to_target, False


def convert_unit_ase(
    unit_label: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> float:
    """
    Get conversion factor to ASE units.

    Parameters
    ----------
    unit: str, optional, default None
        Unit label
    verbose:
        Extensive information for unit recognition

    Returns
    -------
    float
        Conversion factor from unit to ASE units
    """

    # If unit is None, return default factor 1.0
    if unit_label is None or unit_label.lower() == 'None'.lower():
        return 1.0

    # Single proportional symbols
    proportional_symbol = ['*', '/']

    # Unit composition lists and auxiliaries
    unit_contributions = []
    unit_proportionality = []
    unit_exponent = []
    current_contribution = None
    current_proportion = '*'
    ic_current = 0
    ic_start = 0
    ic_end = len(unit_label)

    # Iterate over unit label
    while ic_current < ic_end:

        char = unit_label[ic_current]

        # If proportional symbol found, get contribution and check for
        # power symbol
        if char in proportional_symbol:

            current_contribution = unit_label[ic_start:ic_current]
            if ic_current < ic_end:
                char_next = unit_label[ic_current + 1]
            else:
                char_next = ''

            # Special Case: power symbol
            if char == '*' and char_next == '*':

                power_start = ic_current + 2
                power_end = ic_current + 2

                # Check length of power term
                while (
                        power_end < ic_end
                        and unit_label[power_end] not in proportional_symbol
                ):
                    power_end += 1
                power_end += 1

                # Convert power term to numeric value
                power_string = unit_label[power_start:power_end]
                try:
                    if len(power_string):
                        power_float = float(power_string)
                    else:
                        power_float = 1
                except ValueError:
                    raise ValueError(
                        f"Power term '{power_string:s}' in target unit "
                        + f"'{unit_label:s}' could not be identified as "
                        + "exponential number (float).")

                # Add unit contribution exponent
                unit_exponent.append(power_float)

                # Reset unit label start index
                ic_start = power_end

                # Increment current character index to end of exponent
                ic_current = power_end

            else:

                # Add unit contribution label
                unit_contributions.append(current_contribution)
                # Add proportion label
                unit_proportionality.append(current_proportion)

                # Add unit contribution exponent
                if len(unit_exponent) < len(unit_contributions):
                    unit_exponent.append(1)

                # Reset current contribution
                current_proportion = char[:]

                # Reset unit label start index
                ic_start = ic_current + 1

        ic_current += 1

        # Check if end of unit label
        if ic_current == ic_end:
            current_contribution = unit_label[ic_start:]

    # Final or single unit contribution label
    unit_contributions.append(current_contribution)
    # Final or single proportion label
    unit_proportionality.append(current_proportion)
    # Final or single contribution exponent
    if len(unit_exponent) < len(unit_contributions):
        unit_exponent.append(1)

    # Compute target conversion factor
    unit_factor = 1.0
    if verbose:
        conversion_factors = []
    for ic, contr in enumerate(unit_contributions):

        # Get contribution factor
        # Special case for elemental charge and Coulomb
        if contr == 'e':
            factor = 1.0
        elif contr == 'C':
            factor = 1./getattr(units, "_e")
        elif contr == 'eAng':
            factor = 1.0
        else:
            try:
                factor = float(contr)
            except ValueError:
                try:
                    factor = getattr(units, contr)
                except AttributeError:
                    raise AttributeError(
                        f"Unit contribution '{contr:s}' is not recognized "
                        + "as an ASE known unit.")

        # Apply proportionality
        proportion = unit_proportionality[ic]
        exponent = unit_exponent[ic]
        if proportion == '*':
            factor = 1.*factor**exponent
        elif proportion == '/':
            factor = 1./factor**exponent
        else:
            raise SyntaxError(
                "Contribution conversion failed!")

        # Combine contribution factors
        unit_factor *= factor

        if verbose:
            conversion_factors.append(factor)

    # Print extended information
    if verbose:
        logger.info(
            f"Unit conversion from {unit_label:s} to ASE units:\n"
            + f" Unit contributions: {unit_contributions}\n"
            + f" Unit proportionality: {unit_proportionality}\n"
            + f" Unit exponent: {unit_exponent}\n"
            + f" Unit conversion factors: {conversion_factors}\n"
            + f" Total Unit conversion factor: {unit_factor:.16e}"
            )

    return unit_factor
