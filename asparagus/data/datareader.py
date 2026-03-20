import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any

import ase
import ase.db as ase_db

import numpy as np

from asparagus import data
from asparagus import utils
from asparagus import settings

__all__ = ['DataReader', 'check_data_format']

# Possible data format labels (keys) to valid data format label (item)
# Asparagus database format labels
_data_file_formats = {
    'db':       'db.sql',
    'sql':      'db.sql',
    'sqlite':   'db.sql',
    'sqlite3':  'db.sql',
    'db.sql':   'db.sql',
    'sql.db':   'db.sql',
    'npz':      'db.npz',   # Would be updated by _data_source_format
    'db.npz':   'db.npz',
    'npz.db':   'db.npz',
    'h5':       'db.h5',
    'hdf5':     'db.h5',
    'h5py':     'db.h5',
    'db.h5':    'db.h5',
    'h5.db':    'db.h5',
    'db.hdf5':  'db.h5',
    'hdf5.db':  'db.h5',
    'db.h5py':  'db.h5',
    'h5py.db':  'db.h5',
    }
# External data format labels
_data_source_formats = {
    'npz':      'npz',
    'npy':      'npz',
    'ase':      'ase.db',
    'ase.db':   'ase.db',
    'db.ase':   'ase.db',
    'traj':     'ase.traj',
    'ase.traj': 'ase.traj',
    'traj.ase': 'ase.traj',
    'xyz':      'ase.xyz',
    'ase.xyz':  'ase.xyz',
    'xyz.ase':  'ase.xyz',
    }


def check_data_format(
    data_format: str,
    is_source_format: Optional[bool] = False,
    ignore_error: Optional[bool] = False,
) -> Union[str, bool]:
    """
    Check input for compatible data format labels

    Parameters
    ----------
    data_format: str
        Input string to check for compatible data format labels
    is_source_format: bool, optional, default False
        If False, 'data_format' input is only compared with Asparagus
        database labels. Else, update format library with source format labels.
    ignore_error: bool, optional, default False
        If data format not found, return None rather than an error

    Returns
    -------
    str
        Valid data format label
    """

    # Assign data format dictionary
    file_formats = _data_file_formats.copy()
    if is_source_format:
        file_formats.update(_data_source_formats)

    # Split path and file
    data_path, data_file = os.path.split(data_format)

    # Check data file suffix
    if data_file.lower() in file_formats:
        return file_formats[data_file.lower()]

    if '.' in data_file and data_file.count('.') >= 2:

        idcs = [
            idx for idx, char in enumerate(data_file)
            if char.lower() == '.']
        data_format_label = data_file[idcs[-2] + 1:]

        if data_format_label.lower() in file_formats:
            return file_formats[data_format_label.lower()]

    if '.' in data_file:

        idcs = [
            idx for idx, char in enumerate(data_file)
            if char.lower() == '.']
        data_format_label = data_file[idcs[-1] + 1:]

        if data_format_label.lower() in file_formats:
            return file_formats[data_format_label.lower()]

    if ignore_error:
        return None
    else:
        raise SyntaxError(
            f"Data file format for '{data_format:s}' could not identified!")

    return None


class DataReader():
    """
    DataReader class that contain functions to read and adapt data.

    Parameters
    ----------
    data_file: (str, list(str))
        File path or a tuple of file path and file format label of data
        source to file.
    data_properties: List(str), optional, default None
        Subset of properties to load to dataset file
    data_unit_properties: dict, optional, default None
        Dictionary from properties (keys) to corresponding unit as a string
        (item).
    data_alt_property_labels: dict, optional, default None
        Dictionary of alternative property labeling to replace non-valid
        property labels with the valid one if possible.

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    def __init__(
        self,
        data_file: Optional[str] = None,
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize DataReader.

        """

        # Assign data file and format
        if utils.is_string(data_file):
            self.data_file = (
                data_file, data.check_data_format(
                    data_file, is_source_format=False))
        else:
            self.data_file = data_file

        # Get reference database connect function
        self.connect = data.get_connect(self.data_file[1])

        # Assign property list and units
        self.data_properties = data_properties
        self.data_unit_properties = data_unit_properties

        # Check alternative property labels
        if data_alt_property_labels is None:
            self.data_alt_property_labels = settings._alt_property_labels
        else:
            self.data_alt_property_labels = data_alt_property_labels

        # Default property labels
        self.default_property_labels = [
            'atoms_number', 'atomic_numbers', 'positions', 'charge',
            'cell', 'pbc', 'fragment_numbers']

        # Assign dataset format with respective load function
        self.data_file_format_load = {
            'db.sql':   self.load_db,
            'db.npz':   self.load_db,
            'db.h5':    self.load_db,
            'npz':      self.load_npz,
            'ase.db':   self.load_ase,
            'ase.traj': self.load_traj,
            'ase.xyz':  self.load_xyz,
            }

        # Assign dataset format with respective metadata load function
        self.data_file_format_get_metadata = {
            'db.sql':   self.get_db_metadata,
            'db.npz':   self.get_db_metadata,
            'db.h5':    self.get_db_metadata,
            }

        return

    def load(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        data_source_unit_properties: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Load data from respective dataset format.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'positions': 'Ang',
                                    'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        data_source_unit_properties: dictionary, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item) in the source data files.

        """

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Load data from source of respective format
        if self.data_file_format_load.get(data_source[1]) is None:
            raise SyntaxError(
                f"Data format '{data_source[1]:s}' of data source file "
                + f"'{data_source[0]:s}' is unknown!\n"
                + "Supported data formats are: "
                + f"{self.data_file_format_load.keys()}"
                )
        else:
            _ = self.data_file_format_load[data_source[1]](
                data_source,
                data_properties=data_properties,
                data_unit_properties=data_unit_properties,
                data_alt_property_labels=data_alt_property_labels,
                data_source_unit_properties=data_source_unit_properties,
                **kwargs)

        return

    def get_metadata(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load data from respective dataset format.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'positions': 'Ang',
                                    'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        dict(str, any)
            Source database file metadata

        """

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Load data from source of respective format
        if self.data_file_format_get_metadata.get(data_source[1]) is None:
            metadata = {}
        else:
            metadata = self.data_file_format_get_metadata[data_source[1]](
                data_source,
                data_properties=data_properties,
                data_unit_properties=data_unit_properties,
                data_alt_property_labels=data_alt_property_labels)

        return metadata

    def load_db(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_skip_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Load data from asparagus dataset format.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file.
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_skip_properties: List(str), optional, default None
            Properties not to load even if in data source.
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        (str, list(dict(str, any)))
            Either data file path or list of source properties if data file is
            not defined.

        """

        # Check if data source is empty
        if os.path.isfile(data_source[0]):
            with data.connect(data_source[0], data_source[1], mode='r') as db:
                Ndata = db.count()
        else:
            Ndata = 0
        if Ndata == 0:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' is empty!")
            return

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels
        if data_skip_properties is None:
            data_skip_properties = []

        # Get data sample property labels for label to comparison
        with data.connect(data_source[0], data_source[1], mode='r') as db:
            source_properties = db.get(1)[0].keys()

        # Assign data source property labels to valid property labels.
        assigned_properties = self.assign_property_labels(
            data_source,
            source_properties,
            data_properties,
            data_skip_properties,
            data_alt_property_labels)

        # Get source property units
        with data.connect(data_source[0], data_source[1], mode='r') as db:
            source_unit_properties = db.get_metadata()['unit_properties']

        # Get property unit conversion
        unit_conversion = self.get_unit_conversion(
            assigned_properties,
            data_unit_properties,
            source_unit_properties,
        )

        # Property match summary
        self.print_property_summary(
            data_source,
            assigned_properties,
            unit_conversion,
            data_unit_properties,
            source_unit_properties,
        )

        # If not dataset file is given, load source data to memory
        if self.data_file is None:

            # Add atoms systems to list
            all_atoms_properties = []
            self.logger.info(
                f"Load {Ndata} data point from '{data_source[0]:s}'!")

            # Open source dataset
            with data.connect(
                data_source[0], data_source[1], mode='r'
            ) as db_source:

                # Iterate over source data
                for idx in range(Ndata):

                    # Get property source data
                    source = db_source.get(idx + 1)[0]

                    # Collect system data
                    atoms_properties = self.collect_from_source(
                        source,
                        unit_conversion,
                        data_properties,
                        assigned_properties)

                    # Add system data to database
                    all_atoms_properties.append(atoms_properties)

        # If dataset file is given, write to dataset
        else:

            # Add atom systems to database
            all_atoms_properties = self.data_file[0]
            with self.connect(self.data_file[0], mode='a') as db:

                self.logger.info(
                    f"Writing '{data_source[0]:s}' to database " +
                    f"'{self.data_file[0]:s}'!\n" +
                    f"{Ndata} data point will be added.")

                # Open source dataset
                with data.connect(
                    data_source[0], data_source[1], mode='r'
                ) as db_source:

                    # Iterate over source data
                    for idx in range(Ndata):

                        # Get atoms object and property data
                        source = db_source.get(idx + 1)[0]

                        # Collect system data
                        atoms_properties = self.collect_from_source(
                            source,
                            unit_conversion,
                            data_properties,
                            assigned_properties)

                        # Write to reference database file
                        db.write(properties=atoms_properties)

        # Print completion message
        self.logger.info(
            f"Loading from Asparagus dataset '{data_source[0]:s}' "
            + "complete!")

        return all_atoms_properties

    def load_ase(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_skip_properties: Optional[List[str]] = None,
        data_alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Load data from ASE database formats.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file.
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_skip_properties: List(str), optional, default None
            Properties not to load even if in data source.
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        (str, list(dict(str, any)))
            Either data file path or list of source properties if data file is
            not defined.

        """

        # Check if data source is empty
        if os.path.isfile(data_source[0]):
            with data.connect(data_source[0], data_source[1], mode='r') as db:
                Ndata = db.count()
        else:
            Ndata = 0
        if Ndata == 0:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' is empty!")
            return

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Check list of properties to skip
        default_skip_properties = [
            'atoms_number', 'atomic_numbers', 'cell', 'pbc', 'fragment_numbers']
        if data_skip_properties is None:
            data_skip_properties = default_skip_properties
        else:
            for prop in default_skip_properties:
                if prop not in data_skip_properties:
                    data_skip_properties.append(prop)

        # Get data sample property labels for label to comparison
        with data.connect(data_source[0], data_source[1], mode='r') as db:
            source_properties = db.get(1)[0].keys()

        # Assign data source property labels to valid property labels.
        assigned_properties = self.assign_property_labels(
            data_source,
            source_properties,
            data_properties,
            data_skip_properties,
            data_alt_property_labels)

        # Get source property units - default ASE units
        source_unit_properties = settings._ase_units

        # Get property unit conversion
        unit_conversion = self.get_unit_conversion(
            assigned_properties,
            data_unit_properties,
            source_unit_properties,
        )

        # Property match summary
        self.print_property_summary(
            data_source,
            assigned_properties,
            unit_conversion,
            data_unit_properties,
            source_unit_properties,
        )

        # If not dataset file is given, load source data to memory
        if self.data_file is None:

            # Add atoms systems to list
            all_atoms_properties = []
            self.logger.info(
                f"Load {Ndata:d} data point from '{data_source[0]:s}'!")

            # Open source dataset
            with ase_db.connect(data_source[0]) as db_source:

                # Iterate over source data
                for idx in range(Ndata):

                    # Get atoms object and property data
                    atoms = db_source.get_atoms(idx + 1)
                    source = db_source.get(idx + 1)

                    # Collect system data
                    atoms_properties = self.collect_from_atoms_source(
                        atoms,
                        source,
                        unit_conversion,
                        data_properties,
                        assigned_properties)

                    # Add atoms system data
                    all_atoms_properties.append(atoms_properties)

        # If dataset file is given, write to dataset
        else:

            # Add atom systems to database
            all_atoms_properties = self.data_file[0]
            with self.connect(self.data_file[0], mode='a') as db:

                self.logger.info(
                    f"Writing '{data_source[0]}' to database " +
                    f"'{self.data_file[0]}'!\n" +
                    f"{Ndata} data point will be added.")

                # Open source dataset
                with ase_db.connect(data_source[0]) as db_source:

                    # Iterate over source data
                    for idx in range(Ndata):

                        # Get atoms object and property data
                        atoms = db_source.get_atoms(idx + 1)
                        source = db_source.get(idx + 1)

                        # Collect system data
                        atoms_properties = self.collect_from_atoms_source(
                            atoms,
                            source,
                            unit_conversion,
                            data_properties,
                            assigned_properties)

                        # Write to reference database file
                        db.write(properties=atoms_properties)

        # Print completion message
        self.logger.info(
            f"Loading from ASE database '{data_source[0]}' complete!")

        return all_atoms_properties

    def load_npz(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_skip_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, str]] = None,
        data_source_unit_properties: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Load data from npz dataset format.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file.
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_skip_properties: List(str), optional, default None
            Properties not to load even if in data source.
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.
        data_source_unit_properties: dictionary, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item) in the source data files.

        Returns
        -------
        (str, list(dict(str, any)))
            Either data file path or list of source properties if data file is
            not defined.

        """

        # Check if data source is empty
        if os.path.isfile(data_source[0]):
            source = np.load(data_source[0])
            Ndata = len(source.keys())
        else:
            source = None
            Ndata = 0
        if Ndata == 0:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' is empty!")
            return

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels
        if data_skip_properties is None:
            data_skip_properties = []

        # Get data sample property labels for label to comparison
        source_properties = source.keys()

        # Assign data source property labels to valid property labels.
        assigned_properties = self.assign_property_labels(
            data_source,
            source_properties,
            data_properties,
            data_skip_properties,
            data_alt_property_labels)

        # Initialize source data dictionary and assign properties
        source_dict = {}

        # Atom numbers
        if 'atoms_number' in assigned_properties:
            source_dict['atoms_number'] = (
                source[assigned_properties['atoms_number']])
            Ndata = len(source_dict['atoms_number'])
        else:
            raise ValueError(
                "Property 'atoms_number' not found in npz dataset "
                + f"'{self.data_file[0]}'!\n")

        # Atomic number
        if 'atomic_numbers' in assigned_properties:
            source_dict['atomic_numbers'] = (
                source[assigned_properties['atomic_numbers']])
        else:
            raise ValueError(
                "Property 'atomic_numbers' not found in npz dataset "
                + f"'{self.data_file[0]}'!\n")

        # Atom positions
        if 'positions' in assigned_properties:
            source_dict['positions'] = source[assigned_properties['positions']]
        else:
            raise ValueError(
                "Property 'positions' not found in npz dataset "
                + f"'{self.data_file[0]}'!\n")

        # Total atoms charge
        if 'charge' in assigned_properties:
            source_dict['charge'] = source[assigned_properties['charge']]
        else:
            source_dict['charge'] = np.zeros(Ndata, dtype=float)
            self.logger.warning(
                "Property 'charge' not found in npz dataset "
                + f"'{self.data_file[0]}'!\nCharges are assumed to be zero.")

        # Cell information
        if 'cell' in assigned_properties:
            source_dict['cell'] = source[assigned_properties['cell']]
        else:
            source_dict['cell'] = np.zeros((Ndata, 3), dtype=float)
            self.logger.info(
                "No cell information in npz dataset "
                + f"'{self.data_file[0]}'!")

        # PBC information
        if 'pbc' in assigned_properties:
            source_dict['pbc'] = source[assigned_properties['pbc']]
        else:
            source_dict['pbc'] = np.zeros((Ndata, 3), dtype=bool)
            self.logger.info(
                "No pbc information in npz dataset "
                + f"'{self.data_file[0]}'!")

        # Check if all properties in 'data_properties' are found
        found_properties = [
            prop in assigned_properties.keys()
            for prop in data_properties]
        for ip, prop in enumerate(data_properties):
            if not found_properties[ip]:
                self.logger.error(
                    f"Requested property '{prop:s}' in "
                    + "'data_properties' is not found in Numpy "
                    + f"dataset '{data_source[0]}'!")
        if not all(found_properties):
            raise ValueError(
                "Not all properties in 'data_properties' are found "
                + f"in Numpy dataset '{data_source[0]}'!\n")

        # Get property unit conversion
        unit_conversion = self.get_unit_conversion(
            assigned_properties,
            data_unit_properties,
            data_source_unit_properties,
            )

        # Property match summary
        self.print_property_summary(
            data_source,
            assigned_properties,
            unit_conversion,
            data_unit_properties,
            data_source_unit_properties,
        )

        # Collect properties from source
        for prop, item in assigned_properties.items():
            if prop in self.default_property_labels:
                continue
            if prop in data_properties:
                try:
                    source_dict[prop] = source[item]
                except ValueError:
                    raise ValueError(
                        f"Property '{prop:s}' from the npz entry "
                        + f"'{item:s}' could not be loaded!")

        # If not dataset file is given, load source data to memory
        if self.data_file is None:

            # Add atoms systems to list
            self.logger.info(
                f"Load {Ndata:d} data point from '{data_source[0]:s}'!")
            all_atoms_properties = self.collect_from_source_list(
                source_dict,
                unit_conversion,
                data_properties,
                assigned_properties)

        # If dataset file is given, write to dataset
        else:

            # Add atom systems to database
            self.logger.info(
                f"Writing '{data_source[0]:s}' to database "
                + f"'{self.data_file[0]:s}'!\n"
                + f"{Ndata:d} data point will be added.")
            all_atoms_properties = self.collect_from_source_list(
                source_dict,
                unit_conversion,
                data_properties,
                assigned_properties)

            # Write to reference database file
            with self.connect(self.data_file[0], mode='a') as db:

                for idx in range(Ndata):
                    db.write(properties=all_atoms_properties[idx])

            # Reassign data file path as usual when written to database
            all_atoms_properties = self.data_file[0]

        # Print completion message
        self.logger.info(
            f"Loading from npz database '{data_source[0]}' complete!")

        return all_atoms_properties

    def load_traj(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_skip_properties: Optional[List[str]] = None,
        data_alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Load data from ASE trajectory file.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file.
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_skip_properties: List(str), optional, default None
            Properties not to load even if in data source.
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        (str, list(dict(str, any)))
            Either data file path or list of source properties if data file is
            not defined.

        """

        # Check if data source is empty
        if os.path.isfile(data_source[0]):
            source = ase.io.Trajectory(data_source[0])
            Ndata = len(source)
        else:
            source = None
            Ndata = 0
        if Ndata == 0:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' is empty!")
            return

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Check list of properties to skip
        default_skip_properties = [
            'atoms_number', 'atomic_numbers', 'cell', 'pbc', 'fragment_numbers']
        if data_skip_properties is None:
            data_skip_properties = default_skip_properties
        else:
            for prop in default_skip_properties:
                if prop not in data_skip_properties:
                    data_skip_properties.append(prop)

        # Get data sample to compare property labels
        data_sample = source[0]

        # Check if data source has properties
        if data_sample.calc is None:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' has no properties!")
            return

        # Check if data source has properties
        if source[0].calc is None:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' has no properties!")
            return

        # Get data sample property labels for label to comparison
        source_properties = ['positions', 'charge']
        source_properties += list(data_sample.calc.results)
        source_properties += list(data_sample.info.keys())
        source_properties += list(data_sample.arrays.keys())
        
        # Assign data source property labels to valid property labels.
        assigned_properties = self.assign_property_labels(
            data_source,
            source_properties,
            data_properties,
            data_skip_properties,
            data_alt_property_labels)

        # Get source property units - default ASE units
        source_unit_properties = settings._ase_units

        # Get property unit conversion
        unit_conversion = self.get_unit_conversion(
            assigned_properties,
            data_unit_properties,
            source_unit_properties,
        )

        # Property match summary
        self.print_property_summary(
            data_source,
            assigned_properties,
            unit_conversion,
            data_unit_properties,
            source_unit_properties,
        )

        # If not dataset file is given, load source data to memory
        if self.data_file is None:

            # Add atoms systems to list
            all_atoms_properties = []
            self.logger.info(
                f"Load {Ndata:d} data point from '{data_source[0]:s}'!")

            # Iterate over source data
            for idx in range(Ndata):

                # Atoms system data
                atoms_properties = {}

                # Get atoms object and property data
                atoms = source[idx]

                # Collect system data
                atoms_properties = self.collect_from_atoms(
                    atoms,
                    unit_conversion,
                    data_properties,
                    assigned_properties)

                # Add atoms system data
                all_atoms_properties.append(atoms_properties)

        # If dataset file is given, write to dataset
        else:

            # Add atom systems to database
            all_atoms_properties = self.data_file
            with self.connect(self.data_file[0], mode='a') as db:

                self.logger.info(
                    f"Writing '{data_source[0]}' to database " +
                    f"'{self.data_file[0]}'!\n" +
                    f"{Ndata} data point will be added.")

                # Iterate over source data
                for idx in range(Ndata):

                    # Atoms system data
                    atoms_properties = {}

                    # Get atoms object and property data
                    atoms = source[idx]

                    # Collect system data
                    atoms_properties = self.collect_from_atoms(
                        atoms,
                        unit_conversion,
                        data_properties,
                        assigned_properties)

                    # Write to reference database file
                    db.write(properties=atoms_properties)

        # Print completion message
        self.logger.info(
            f"Loading from ASE trajectory '{data_source[0]:s}' complete!")

        return all_atoms_properties


    def load_xyz(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_skip_properties: Optional[List[str]] = None,
        data_alt_property_labels: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        Load data from extended xyz files.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file.
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_skip_properties: List(str), optional, default None
            Properties not to load even if in data source.
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        (str, list(dict(str, any)))
            Either data file path or list of source properties if data file is
            not defined.

        """

        # Check if data source is empty
        if os.path.isfile(data_source[0]):
            source = ase.io.read(data_source[0], index=':', format='extxyz')
            Ndata = len(source)
        else:
            source = None
            Ndata = 0
        if Ndata == 0:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' is empty!")
            return

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Check list of properties to skip
        default_skip_properties = [
            'atoms_number', 'atomic_numbers', 'cell', 'pbc']
        if data_skip_properties is None:
            data_skip_properties = default_skip_properties
        else:
            for prop in default_skip_properties:
                if prop not in data_skip_properties:
                    data_skip_properties.append(prop)

        # Get data sample to compare property labels
        data_sample = source[0]

        # Check if data source has properties
        if data_sample.calc is None:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' has no properties!")
            return

        # Check if data source has properties
        if source[0].calc is None:
            self.logger.warning(
                f"Data source '{data_source[0]:s}' has no properties!")
            return

        # Get data sample property labels for label to comparison
        source_properties = ['positions', 'charge']
        source_properties += list(data_sample.calc.results)
        source_properties += list(data_sample.info.keys())
        source_properties += list(data_sample.arrays.keys())

        # Assign data source property labels to valid property labels.
        assigned_properties = self.assign_property_labels(
            data_source,
            source_properties,
            data_properties,
            data_skip_properties,
            data_alt_property_labels)

        # Get source property units - default ASE units
        source_unit_properties = settings._ase_units

        # Get property unit conversion
        unit_conversion = self.get_unit_conversion(
            assigned_properties,
            data_unit_properties,
            source_unit_properties,
        )

        # Property match summary
        self.print_property_summary(
            data_source,
            assigned_properties,
            unit_conversion,
            data_unit_properties,
            source_unit_properties,
        )

        # If not dataset file is given, load source data to memory
        if self.data_file is None:

            # Add atoms systems to list
            all_atoms_properties = []
            self.logger.info(
                f"Load {Ndata:d} data point from '{data_source[0]:s}'!")

            # Iterate over source data
            for idx in range(Ndata):

                # Atoms system data
                atoms_properties = {}

                # Get atoms object and property data
                atoms = source[idx]

                # Collect system data
                atoms_properties = self.collect_from_atoms(
                    atoms,
                    unit_conversion,
                    data_properties,
                    assigned_properties)

                # Add atoms system data
                all_atoms_properties.append(atoms_properties)

        # If dataset file is given, write to dataset
        else:

            # Add atom systems to database
            all_atoms_properties = self.data_file
            with self.connect(self.data_file[0], mode='a') as db:

                self.logger.info(
                    f"Writing '{data_source[0]}' to database " +
                    f"'{self.data_file[0]}'!\n" +
                    f"{Ndata} data point will be added.")

                # Iterate over source data
                for idx in range(Ndata):

                    # Atoms system data
                    atoms_properties = {}

                    # Get atoms object and property data
                    atoms = source[idx]

                    # Collect system data
                    atoms_properties = self.collect_from_atoms(
                        atoms,
                        unit_conversion,
                        data_properties,
                        assigned_properties)

                    # Write to reference database file
                    db.write(properties=atoms_properties)

        # Print completion message
        self.logger.info(
            f"Loading from extended xyz file '{data_source[0]:s}' complete!")

        return all_atoms_properties

    def load_atoms(
        self,
        atoms: object,
        atoms_properties: Dict[str, Any],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_skip_properties: Optional[List[str]] = None,
        data_alt_property_labels: Optional[Dict[str, str]] = None,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Load atoms object with properties to dataset format.

        Parameters
        ----------
        atoms: ASE Atoms object
            ASE Atoms object with conformation belonging to the properties.
        atoms_properties: dict
            Atoms object properties
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_skip_properties: List(str), optional, default None
            Properties not to load even if in data source.
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        (str, list(dict(str, any)))
            Either data file path or list of source properties if data file is
            not defined.

        """

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Check list of properties to skip
        default_skip_properties = [
            'atoms_number', 'atomic_numbers', 'cell', 'pbc']
        if data_skip_properties is None:
            data_skip_properties = default_skip_properties
        else:
            for prop in default_skip_properties:
                if prop not in data_skip_properties:
                    data_skip_properties.append(prop)

        # Get data sample property labels for label to comparison
        source_properties = atoms_properties.keys()

        # Assign data source property labels to valid property labels.
        assigned_properties = self.assign_property_labels(
            ('ase.Atoms', None),
            source_properties,
            data_properties,
            data_skip_properties,
            data_alt_property_labels)

        # Get source property units - default ASE units
        source_unit_properties = settings._ase_units

        # Get property unit conversion
        unit_conversion = self.get_unit_conversion(
            assigned_properties,
            data_unit_properties,
            source_unit_properties,
        )

        # If not dataset file is given, load data to memory
        if self.data_file is None:

            # Collect system data
            atoms_properties = self.collect_from_atoms_source(
                atoms,
                atoms_properties,
                unit_conversion,
                data_properties,
                assigned_properties)

        # If dataset file is given, write to dataset
        else:

            # Add atom systems to database
            with self.connect(self.data_file[0], mode='a') as db:

                # Collect system data
                atoms_properties = self.collect_from_atoms_source(
                    atoms,
                    atoms_properties,
                    unit_conversion,
                    data_properties,
                    assigned_properties)

                # Write to ASE database file
                db.write(properties=atoms_properties)

        return atoms_properties

    def assign_property_labels(
        self,
        data_source: List[str],
        source_properties: List[str],
        data_properties: List[str],
        skip_properties: List[str],
        data_alt_property_labels: Dict[str, List[str]],
    ) -> Dict[str, str]:
        """
        Assign data source property labels to valid property labels.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file
        source_properties: List(str)
            Properties list of source data
        data_properties: List(str)
            Subset of properties to load
        skip_properties: List(str)
            Properties not to load even if in source.
        data_alt_property_labels: dict
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        dict(str, str)
            Assigned data property label (key) to source label (item)

        """

        # Assign data source property labels to valid labels.
        assigned_properties = {}
        for source_label in source_properties:

            # Check source label for prefix
            if 'std_' in source_label:
                label = source_label[4:]
            else:
                label = source_label

            # Skip system properties, which are read otherwise
            if label in skip_properties:
                continue

            match, modified, valid_label = utils.check_property_label(
                label,
                valid_property_labels=settings._valid_properties,
                alt_property_labels=data_alt_property_labels)
            if (
                match
                and (
                    valid_label in self.default_property_labels
                    or valid_label in data_properties
                )
            ):
                if 'std_' in source_label:
                    valid_label = f"std_{valid_label:s}"
                if modified:
                    self.logger.warning(
                        f"Property key '{source_label:s}' in database "
                        + f"'{data_source[0]:s}' is not a valid label!\n"
                        + f"Property key '{source_label:s}' is assigned as "
                        + f"'{valid_label:s}'.")
                assigned_properties[valid_label] = label
            elif match:
                self.logger.warning(
                    f"Property key '{source_label:s}' in database "
                    + f"'{data_source[0]:s}' found as '{valid_label:s}', but "
                    + "not requested in 'data_properties'!\nProperty ignored.")
            else:
                self.logger.warning(
                    f"Unknown property '{source_label:s}' in "
                    + f"database '{data_source[0]:s}'!\nProperty ignored.")

        # Check if all properties in 'data_properties' are found
        found_properties = [
            prop in assigned_properties
            for prop in data_properties]
        for ip, prop in enumerate(data_properties):
            if not found_properties[ip]:
                self.logger.error(
                    f"Requested property '{prop}' in "
                    + "'data_properties' is not found in "
                    + f"database '{data_source[0]:s}'!")
        if not all(found_properties):
            raise ValueError(
                "Not all properties in 'data_properties' are found "
                + f"in database '{data_source[0]:s}'!")

        return assigned_properties

    def get_unit_conversion(
        self,
        data_properties: Dict[str, str],
        data_unit_properties: Dict[str, float],
        source_unit_properties: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Assign source property to data property unit conversion factors.

        Parameters
        ----------
        data_properties: dict(str, str)
            Property labels dictionary from data file (key) and data source
            (item)
        data_unit_properties: dict
            Dictionary from data properties (keys) to corresponding unit as a
            string (item)
        source_unit_properties: dict
            Dictionary from source properties (keys) to corresponding unit as a
            string (item)

        Returns
        -------
        dict(str, str)
            Assigned source to data property unit conversion factor

        """

        # Property match summary and unit conversion
        if data_unit_properties is None or source_unit_properties is None:

            # Set default conversion factor dictionary
            unit_conversion = {}
            for prop in data_properties.keys():
                unit_conversion[prop] = 1.0

        else:

            # Check units of positions and properties
            unit_conversion = {}
            for data_prop in data_properties.keys():

                # Check for property prefix
                if 'std_' in data_prop:
                    prop = data_prop[4:]
                else:
                    prop = data_prop

                if source_unit_properties.get(prop) is None:
                    source_unit_property = None
                else:
                    source_unit_property = source_unit_properties.get(prop)
                if (
                    data_unit_properties.get(data_prop) is None
                    and data_unit_properties.get(prop) is None
                ):
                    data_unit_property = None
                elif data_unit_properties.get(prop) is None:
                    data_unit_property = data_unit_properties.get(data_prop)
                else:
                    data_unit_property = data_unit_properties.get(prop)

                unit_conversion[data_prop], _ = (
                    utils.check_units(
                        data_unit_property,
                        source_unit_property)
                    )

        return unit_conversion

    def print_property_summary(
        self,
        data_source,
        assigned_properties,
        unit_conversion,
        data_unit_properties,
        source_unit_properties,
    ):
        """
        Print property match summary

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file
        assigned_properties: dict
            Assigned data property label (key) to source label (item)
        unit_conversion: dict
            Assigned source to data property unit conversion factor
        data_unit_properties: dict
            Dictionary from data properties (keys) to corresponding unit as a
            string (item)
        source_unit_properties: dict
            Dictionary from source properties (keys) to corresponding unit as a
            string (item)

        """

        # Prepare header
        message = (
            "Property assignment from database "
            + f"'{data_source[0]:s}'!\n"
            + f" {'Load':4s} |"
            + f" {'Property Label':<14s} |"
            + f" {'Data Unit':<14s} |"
            + f" {'Source Label':<14s} |"
            + f" {'Source Unit':<14s} |"
            + f" {'Conversion Fac.':<14s}\n"
            + "-"*(7 + 17*5)
            + "\n")

        # Iterate over properties
        for data_prop, source_prop in assigned_properties.items():

            # Check property labels
            if source_unit_properties is None:
                source_unit_property = "None"
            elif (
                source_unit_properties.get(data_prop) is None
                and source_unit_properties.get(source_prop) is None
            ):
                source_unit_property = "None"
            elif source_unit_properties.get(data_prop) is not None:
                source_unit_property = source_unit_properties.get(data_prop)
            elif source_unit_properties.get(source_prop) is not None:
                source_unit_property = source_unit_properties.get(source_prop)
            else:
                source_unit_property = "None"

            if (
                data_unit_properties is None or
                data_unit_properties.get(data_prop) is None
            ):
                data_unit_property = "None"
            else:
                data_unit_property = data_unit_properties.get(data_prop)

            if (
                data_prop in unit_conversion
                or data_prop in self.default_property_labels
            ):
                load_label = " x  "
            else:
                load_label = "    "

            message += (
                f" {load_label:4s} |"
                + f" {data_prop:<14s} |"
                + f" {data_unit_property:<14s} |"
                + f" {source_prop:<14s} |"
                + f" {source_unit_property:<14s} |"
                + f" {unit_conversion[data_prop]:11.9e}\n"
                )

        # Print property information
        self.logger.info(message)

        return

    def collect_from_source(
        self,
        source: Dict[str, Any],
        conversion: Dict[str, float],
        load_properties: List[str],
        property_labels: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Collect properties from database entry to property dictionary and
        apply unit conversion.

        Parameters
        ----------
        source: dict
            Database source dictionary
        conversion: dict
            Unit conversion factors dictionary.
        load_properties: list(str)
            Properties to load from source
        property_labels: dict(str, str), optional, default None
            List of additional source properties (key) to add from property
            source label (item). If None, all properties in source are added.

        Returns
        -------
        dict(str, any)
            System property dictionary

        """

        # Atoms system data
        atoms_properties = {}

        # Fundamental properties
        atoms_properties['atoms_number'] = source['atoms_number']
        atoms_properties['atomic_numbers'] = (
            source['atomic_numbers'])
        atoms_properties['positions'] = (
            conversion['positions']*source['positions'])
        atoms_properties['cell'] = (
            conversion['positions']*self.convert_cell(source['cell']))
        atoms_properties['pbc'] = source['pbc']
        if source.get('charge') is None:
            atoms_properties['charge'] = 0.0
        else:
            atoms_properties['charge'] = source['charge']
        if source.get('fragment_numbers') is None:
            atoms_properties['fragment_numbers'] = [0]*source['atoms_number']
        else:
            atoms_properties['fragment_numbers'] = source['fragment_numbers']

        # Collect properties
        for prop, item in property_labels.items():
            if prop in load_properties:
                if item is None:
                    raise SyntaxError(
                        f"Requested property '{prop:s}' not available "
                        + "in results dictionary (None)!")
                atoms_properties[prop] = (
                    conversion[prop]*source[item])

        return atoms_properties

    def collect_from_source_list(
        self,
        source: Dict[str, List[Any]],
        conversion: Dict[str, float],
        load_properties: List[str],
        property_labels: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Collect properties from complete data dictionary to property
        dictionaries and apply unit conversion.

        Parameters
        ----------
        source: dict
            Database source dictionary
        conversion: dict
            Unit conversion factors dictionary.
        load_properties: list(str)
            Properties to load from source
        property_labels: dict(str, str), optional, default None
            List of additional source properties (key) to add from property
            source label (item). If None, all properties in source are added.

        Returns
        -------
        dict(str, any)
            System property dictionary

        """

        # Initialize data list
        all_atoms_properties = []

        # Get Number of data entries and max atom number
        Ndata = source['atoms_number'].shape[0]
        max_atoms_number = source['atomic_numbers'].shape[1]

        # Iterate over all data entries
        for idx in range(Ndata):

            # Atoms system data
            atoms_properties = {}

            # Fundamental properties
            atoms_properties['atoms_number'] = source['atoms_number'][idx]
            atoms_properties['atomic_numbers'] = (
                source['atomic_numbers'][idx][:source['atoms_number'][idx]])
            atoms_properties['positions'] = (
                conversion['positions']
                * source['positions'][idx][:source['atoms_number'][idx]])
            atoms_properties['cell'] = (
                conversion['positions']*self.convert_cell(source['cell'][idx]))
            atoms_properties['pbc'] = source['pbc'][idx]
            if source.get('charge') is None:
                atoms_properties['charge'] = 0.0
            else:
                atoms_properties['charge'] = source['charge'][idx]
            if source.get('fragment_numbers') is None:
                atoms_properties['fragment_numbers'] = (
                    [0]*source['atoms_number'][idx])
            else:
                atoms_properties['fragment_numbers'] = (
                    source['fragment_numbers'][idx])

            # Collect properties
            for prop, item in source.items():
                if item is None:
                    raise SyntaxError(
                        f"Requested property '{prop:s}' not available "
                        + "in results dictionary (None)!")
                if (
                    prop in load_properties
                    and item[idx].shape
                    and source['atoms_number'][idx] != max_atoms_number
                    and item[idx].shape[0] == max_atoms_number
                    and np.all(item[idx][source['atoms_number'][idx]:] == 0.0)
                ):
                    atoms_properties[prop] = (
                        conversion[prop]
                        * item[idx][:source['atoms_number'][idx]])
                elif prop in load_properties:
                    atoms_properties[prop] = conversion[prop]*item[idx]

            # Add atoms system data
            all_atoms_properties.append(atoms_properties)

        return all_atoms_properties

    def collect_from_atoms_source(
        self,
        atoms: ase.Atoms,
        source: Dict[str, Any],
        conversion: Dict[str, float],
        load_properties: List[str],
        property_labels: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Collect properties from database entry to property dictionary and
        apply unit conversion.

        Parameters
        ----------
        atoms: ase.Atoms
            Database atoms object
        source: dict
            Database source dictionary
        conversion: dict
            Unit conversion factors dictionary.
        load_properties: list(str)
            Properties to load from source
        property_labels: dict(str, str), optional, default None
            List of additional source properties (key) to add from property
            source label (item). If None, all properties in source are added.

        Returns
        -------
        dict(str, any)
            System property dictionary

        """

        # Atoms system data
        atoms_properties = {}

        # Fundamental properties
        atoms_properties['atoms_number'] = (
            atoms.get_global_number_of_atoms())
        atoms_properties['atomic_numbers'] = (
            atoms.get_atomic_numbers())
        atoms_properties['positions'] = (
            conversion['positions']*atoms.get_positions())
        atoms_properties['cell'] = (
            conversion['positions']*self.convert_cell(atoms.get_cell()[:]))
        atoms_properties['pbc'] = atoms.get_pbc()
        if 'charge' in source:
            atoms_properties['charge'] = source['charge']
        elif 'charges' in source:
            atoms_properties['charge'] = sum(source['charges'])
        elif 'initial_charges' in source:
            atoms_properties['charge'] = sum(
                source['initial_charges'])
        else:
            atoms_properties['charge'] = 0.0
        if 'fragment_numbers' in source:
            atoms_properties['fragment_numbers'] = source['fragment_numbers']
        elif (
            'fragment_numbers' in property_labels
            and property_labels['fragment_numbers'] in source
        ):
            atoms_properties['fragment_numbers'] = (
                source[property_labels['fragment_numbers']])
        elif 'fragment_numbers' in atoms.arrays:
            atoms_properties['fragment_numbers'] = (
                atoms.arrays['fragment_numbers'])
        elif (
            'fragment_numbers' in property_labels
            and property_labels['fragment_numbers'] in atoms.arrays
        ):
            atoms_properties['fragment_numbers'] = (
                atoms.arrays[property_labels['fragment_numbers']])
        else:
            atoms_properties['fragment_numbers'] = (
                [0]*atoms.get_global_number_of_atoms())

        # Collect properties
        for prop, item in property_labels.items():
            if prop in load_properties:
                if item is None:
                    raise SyntaxError(
                        f"Requested property '{prop:s}' not available "
                        + "in results dictionary (None)!")
                atoms_properties[prop] = (
                    conversion[prop]*source[item])

        return atoms_properties

    def collect_from_atoms(
        self,
        atoms: ase.Atoms,
        conversion: Dict[str, float],
        load_properties: List[str],
        property_labels: Dict[str, str] = None,
    ) -> Dict[str, Any]:
        """
        Collect properties from database entry to property dictionary and
        apply unit conversion.

        Parameters
        ----------
        atoms: ase.Atoms
            Database atoms object including calculator object
        conversion: dict
            Unit conversion factors dictionary.
        load_properties: list(str)
            Properties to load from source
        property_labels: dict(str, str), optional, default None
            List of additional source properties (key) to add from property
            source label (item). If None, all properties in source are added.

        Returns
        -------
        dict(str, any)
            System property dictionary

        """

        # Atoms system data
        atoms_properties = {}

        # Get atoms property and general info data
        properties = atoms.calc
        infos = atoms.info
        infos.update(atoms.arrays)
        
        # Fundamental properties
        atoms_properties['atoms_number'] = (
            atoms.get_global_number_of_atoms())
        atoms_properties['atomic_numbers'] = (
            atoms.get_atomic_numbers())
        atoms_properties['positions'] = (
            conversion['positions']*atoms.get_positions())
        atoms_properties['cell'] = (
            conversion['positions']*self.convert_cell(atoms.get_cell()[:]))
        atoms_properties['pbc'] = atoms.get_pbc()
        if 'charge' in properties.parameters:
            atoms_properties['charge'] = (
                properties.parameters['charge'])
        elif 'charges' in properties.results:
            atoms_properties['charge'] = sum(
                properties.results['charges'])
        elif 'charge' in infos:
            if utils.is_numeric(infos['charge']):
                atoms_properties['charge'] = infos['charge']
            elif utils.is_numeric_array(infos['charge']):
                atoms_properties['charge'] = infos['charge'][0]
            else:
                raise SyntaxError(
                    f"Charge property '{prop:s}' cannot be read from "
                    + "atoms info dictionary!")
        else:
            atoms_properties['charge'] = 0.0
        if 'fragment_numbers' in infos:
            atoms_properties['fragment_numbers'] = infos['fragment_numbers']
        elif (
            'fragment_numbers' in property_labels
            and property_labels['fragment_numbers'] in infos
        ):
            atoms_properties['fragment_numbers'] = (
                infos[property_labels['fragment_numbers']])
        else:
            atoms_properties['fragment_numbers'] = (
                [0]*atoms.get_global_number_of_atoms())

        # Collect properties
        for prop, item in property_labels.items():
            if (
                prop in properties.results
                and prop not in ['charge', 'fragment_numbers']
            ):
                if item is None:
                    raise SyntaxError(
                        f"Requested property '{prop:s}' not available "
                        + "in results dictionary (None)!")
                atoms_properties[prop] = (
                    conversion[prop]*properties.results[prop])
            elif prop in infos and prop not in ['charge', 'fragment_numbers']:
                if item is None:
                    raise SyntaxError(
                        f"Requested property '{prop:s}' not available "
                        + "in atoms info dictionary (None)!")
                atoms_properties[prop] = (
                    conversion[prop]*infos[prop])

        return atoms_properties

    def convert_cell(
        self,
        cell,
        shape_out: Tuple[int] = (3, 3),
    ) -> np.ndarray:
        """
        Convert system cell information to 3x3 array

        Parameters
        ----------
        cell: list(float)
            Cell information in various shape
        shape_out: tuple(int)
            Returned cell shape

        Returns
        -------
        np.ndarray
            Cell information in requested shape

        """

        # Check input data type
        if cell is None:
            return np.zeros(shape_out, dtype=float)
        else:
            cell = np.array(cell, dtype=float)

        # Check if requested shape is correct otherwise flatten array for
        # further checks
        if cell.shape == shape_out:
            return cell
        else:
            cell = cell.reshape(-1)

        # Check if data shape is flatten
        shape_prod = np.prod(shape_out)
        if cell.shape == (shape_prod,):
            return cell.reshape(shape_out)

        # Check if data is diagonalized
        if len(shape_out) > 1 and np.diag(cell).shape == shape_out:
            return np.diag(cell)

        return SyntaxError(f"Invalid cell shape '{cell.shape:}'.")

    def get_db_metadata(
        self,
        data_source: List[str],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Load data from asparagus dataset format.

        Parameters
        ----------
        data_source: list(str)
            Tuple of file path and file format label of data source to file.
        data_properties: List(str), optional, default None
            Subset of properties to load
        data_unit_properties: dict, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item), e.g.:
                {property: unit}: { 'positions': 'Ang',
                                    'energy': 'eV',
                                    'force': 'eV/Ang', ...}
        data_alt_property_labels: dict, optional, default None
            Dictionary of alternative property labeling to replace
            non-valid property labels with the valid one if possible.

        Returns
        -------
        dict(str, any)
            Source database file metadata

        """

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Get source data metadata
        with data.connect(data_source[0], data_source[1], mode='r') as db:
            source_metadata = db.get_metadata()
            if db.get(1):
                source_properties = db.get(1)[0].keys()
            else:
                source_properties = source_metadata['load_properties']
            source_unit_properties = source_metadata['unit_properties']

        # Assign data source property labels to valid property labels.
        assigned_properties = self.assign_property_labels(
            data_source,
            source_properties,
            data_properties,
            [],
            data_alt_property_labels)

        # Get property unit conversion
        unit_conversion = self.get_unit_conversion(
            assigned_properties,
            data_unit_properties,
            source_unit_properties,
        )

        # Convert property scaling
        if source_metadata.get('data_property_scaling') is not None:
            for prop in source_metadata['data_property_scaling']:
                if unit_conversion.get(prop) is None:
                    conversion = 1.0
                else:
                    conversion = unit_conversion[prop]
                source_metadata['data_property_scaling'][prop] = [
                    conversion*item
                    for item in source_metadata['data_property_scaling'][prop]]

        # Convert atomic energies scaling
        if source_metadata.get('data_atomic_energies_scaling') is not None:
            if unit_conversion.get('energy') is None:
                conversion = 1.0
            else:
                conversion = unit_conversion['energy']
            for atom in source_metadata['data_atomic_energies_scaling']:
                source_metadata['data_atomic_energies_scaling'][atom] = [
                    conversion*item for item in
                    source_metadata['data_atomic_energies_scaling'][atom]]

        return source_metadata

