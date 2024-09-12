import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import numpy as np

import torch

from asparagus import data
from asparagus import utils
from asparagus import settings

__all__ = ['DataSet', 'DataSubSet']


class DataSet():
    """
    DataSet class containing and loading reference data from files

    Parameters
    ----------
    data_file: (str, tuple(str)), optional, default ('data.db', 'db.sql')
        Either a single string of the reference Asparagus database file name
        or a tuple of the filename first and the file format label second.
    data_label: str, optional, default 'all'
        Dataset label.
    data_properties: List(str), optional, default None
        Subset of properties to load.
    data_unit_properties: dict, optional, default None
        Dictionary from properties (keys) to corresponding unit as a
        string (item), e.g.:
            {property: unit}: { 'positions': 'Ang',
                                'energy': 'eV',
                                'force': 'eV/Ang', ...}
    data_alt_property_labels: dict, optional, default
            'settings._alt_property_labels'
        Dictionary of alternative property labeling to replace
        non-valid property labels with the valid one if possible.
    data_overwrite: bool, optional, default 'False'
        Overwrite database file

    Return
    ------
    callable
        Asparagus DataSet object
    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    def __init__(
        self,
        data_file: Union[str, Tuple[str, str]],
        data_label: Optional[Union[str]] = None,
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        data_overwrite: Optional[bool] = False,
        **kwargs
    ):
        """
        Initialize DataSet class

        """

        # Assign data file and format
        if utils.is_string(data_file):
            self.data_file = (
                data_file, data.check_data_format(
                    data_file, is_source_format=False))
        else:
            self.data_file = tuple(data_file)

        # Assign dataset label
        if data_label is None:
            self.data_label = 'all'
        else:
            self.data_label = data_label

        # Get database connect function
        self.connect = data.get_connect(self.data_file[1])

        # Check for data path existence
        path, _ = os.path.split(self.data_file[0])
        if path and not os.path.isdir(path):
            os.makedirs(path)

        # If overwrite, remove old DataSet file
        if os.path.exists(self.data_file[0]) and data_overwrite:
            with self.connect(self.data_file[0], 'w') as f:
                f.delete_file()

        # Copy current metadata
        metadata = self.get_metadata()

        # Check data property compatibility
        metadata = self.check_data_compatibility(
            metadata,
            data_properties,
            data_unit_properties)

        # Set metadata
        self.set_metadata(metadata)

        # Assign property list and units
        self.data_properties = data_properties
        self.data_unit_properties = data_unit_properties

        # Check alternative property labels
        if data_alt_property_labels is None:
            self.data_alt_property_labels = settings._alt_property_labels
        else:
            self.data_alt_property_labels = data_alt_property_labels

        # Assign database property dtype
        self.data_dtype_property = self.get_data_properties_dtype()

        # Initialize DataReader variable
        self.datareader = None

        return

    def __len__(
        self,
    ) -> int:

        if os.path.isfile(self.data_file[0]):
            with self.connect(self.data_file[0], mode='r') as db:
                return db.count()
        else:
            return 0

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(idx)

    def __setitem__(
        self,
        idx: int,
        properties: Dict[str, torch.tensor],
    ):

        return self._set_properties([idx], [properties])

    def __iter__(
        self
    ):
        # Start data counter and set dataset length
        self.counter = 0
        self.Ndata = len(self)
        
        # Open database
        self.db = self.connect(self.data_file[0], mode='r')
        
        return self

    def __next__(
        self
    ):
        # Check counter within number of data range
        if self.counter < self.Ndata:
            data = self.db.get(self.counter + 1)[0]
            self.counter += 1
            return data
        else:
            del self.db
            raise StopIteration

    def get(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(idx)

    def get_properties(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(idx)

    def _get_properties(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        with self.connect(self.data_file[0], mode='r') as db:
            row = db.get(idx + 1)[0]

        return row

    def set_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._set_properties(idx, properties)

    def _set_properties(
        self,
        idcs: List[int],
        properties: List[Dict[str, torch.tensor]],
    ):

        with self.connect(self.data_file[0], mode='a') as db:
            for idx, props in zip(idcs, properties):
                row_id = db.write(props, row_id=idx + 1)

        return row_id

    def update_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._update_properties(idx, properties)

    def _update_properties(
        self,
        idcs: List[int],
        properties: List[Dict[str, torch.tensor]],
    ):

        with self.connect(self.data_file[0], mode='a') as db:
            for idx, props in zip(idcs, properties):
                row_id = db.update(row_id=idx + 1, properties=props)

        return row_id

    @property
    def metadata(self):
        """
        DataSet metadata dictionary
        """
        return self.get_metadata()

    def get_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Get metadata from database

        Returns
        -------
        dict
            Metadata of the database
        """

        # Read metadata from database file
        with self.connect(self.data_file[0], mode='r') as db:
            return db.get_metadata()

    def set_metadata(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Add metadata to the ASE database file
        """

        # Check for custom metadata
        if metadata is None:
            metadata = self.metadata

        # Set metadata
        with self.connect(self.data_file[0], mode='a') as db:
            db.set_metadata(metadata)

    def reset_database(
        self,
    ):
        with self.connect(self.data_file[0], mode='a') as db:
            db.reset()

    def load_data(
        self,
        data_source: Union[str, List[str]],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, str]] = None,
        data_source_unit_properties: Optional[Dict[str, str]] = None,
    ):
        """
        Load properties from data source.

        Parameters:
        -----------
        data_source: (str, list(str))
            File path or a tuple of file path and file format label of data
            source to file.
        data_properties: list(str), optional, default None
            Set of properties to store in the DataSet
        data_unit_properties: dictionary, optional, default None
            Dictionary from properties (keys) to corresponding unit as a
            string (item).
        data_alt_property_labels: dict
            Alternative property labels to detect common mismatches.
        data_source_unit_properties: dictionary, optional, default None
            Dictionary from properties (keys) to corresponding unit as a 
            string (item) in the source data files.

        """

        # Get metadata from database file
        metadata = self.get_metadata()

        # Check data source file and format
        if utils.is_string(data_source):
            data_source = [
                data_source, data.check_data_format(
                    data_source, is_source_format=True)]
        else:
            data_source = list(data_source)

        # Check if data source already loaded
        if metadata.get('data_source') is None:
            metadata['data_source'] = []
        elif (
            tuple(data_source)
            in [tuple(source_i) for source_i in metadata['data_source']]
        ):
            self.logger.warning(
                f"Data source '{data_source[0]:s}' already "
                + f"written to dataset '{self.data_file[0]:s}'! "
                + "Loading data source is skipped.")
            return

        # Check property list, units and alternative labels
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels

        # Reset property scaling flag
        metadata['data_property_scaling_uptodate'] = False
        metadata['data_atom_energies_scaling_uptodate'] = False

        # Initialize DataReader
        datareader = data.DataReader(
            data_file=self.data_file,
            data_properties=data_properties,
            data_unit_properties=data_unit_properties,
            data_alt_property_labels=data_alt_property_labels)

        # Load data file
        datareader.load(
            data_source,
            data_source_unit_properties=data_source_unit_properties)

        # Append data source information
        metadata['data_source'].append(data_source)

        # If metadata properties is empty, initialize database
        if (
            metadata.get('load_properties') is None
            or metadata.get('unit_properties') is None
        ):
            metadata['load_properties'] = data_properties
            metadata['unit_properties'] = data_unit_properties

        # Set updated metadata
        self.set_metadata(metadata)

        return

    def add_atoms(
        self,
        atoms: object,
        properties: Dict[str, Any],
    ):
        """
        Add ASE Atoms system and properties

        """

        # In case, initialize DataReader with default properties
        if self.datareader is None:
            self.datareader = data.DataReader(
                data_file=self.data_file,
                data_properties=self.data_properties,
                data_unit_properties=self.data_unit_properties,
                data_alt_property_labels=self.data_alt_property_labels,
                )

        # Get dataset metadata
        metadata = self.get_metadata()

        # Load from ASE atoms object and property list
        self.datareader.load_atoms(
            atoms,
            properties,
            data_properties=metadata['load_properties'],
            data_unit_properties=metadata['unit_properties'],
            )

        return

    def check_data_compatibility(
        self,
        metadata: Dict[str, Any],
        data_properties: List[str],
        data_unit_properties: Dict[str, str],
    ):
        """
        Check compatibility between input for 'data_properties' and
        'data_unit_properties' with metadata.
        """

        if (
            metadata.get('load_properties') is None
            and data_properties is None
        ):
            metadata['load_properties'] = []
        elif metadata.get('load_properties') is None:
            metadata['load_properties'] = data_properties
        elif data_properties is None:
            data_properties = metadata['load_properties']
        else:
            mismatch_metadata = []
            mismatch_input = []
            # Check property match except for default properties always stored
            # in database
            for prop in metadata.get('load_properties'):
                if (
                    prop not in data_properties
                    and prop not in settings._default_property_labels
                ):
                    mismatch_metadata.append(prop)
            for prop in data_properties:
                if (
                    prop not in metadata.get('load_properties')
                    and prop not in settings._default_property_labels
                ):
                    mismatch_input.append(prop)
            if len(mismatch_metadata) or len(mismatch_input):
                message = (
                    "Mismatch between DataSet 'load_properties' "
                    + f"input and metadata in '{self.data_file[0]:s}'!\n")
                for prop in mismatch_metadata:
                    message += f"Property '{prop:s}' in metadata not in input."
                for prop in mismatch_input:
                    message += f"Property '{prop:s}' in input not in metadata."
                self.logger.error(message)
                raise SyntaxError(message)

        # Check compatibility between 'data_unit_properties' in metadata
        # and input
        if (
            metadata.get('unit_properties') is None
            and data_unit_properties is None
        ):
            metadata['unit_properties'] = {}
        elif metadata.get('unit_properties') is None:
            metadata['unit_properties'] = {}
            message = ""
            for prop in data_properties:
                if prop in data_unit_properties:
                    metadata['unit_properties'][prop] = (
                        data_unit_properties[prop])
                else:
                    message += f"No Property unit defined for '{prop:s}'."
            if len(message):
                raise SyntaxError(
                    "DataSet 'load_properties' input contains properties with"
                    + "unkown property units in 'unit_properties'!\n"
                    + message)
        elif data_unit_properties is None:
            data_unit_properties = metadata['unit_properties']
        else:
            mismatch = []
            # Check property unit match
            for prop, ui in data_unit_properties.items():
                um = metadata.get('unit_properties').get(ui)
                if um is not None and um.lower() != ui.lower():
                    mismatch.append((prop, um, ui))
            if len(mismatch):
                message = (
                    "Mismatch between DataContainer 'data_unit_properties' "
                    + f"input and metadata in '{self.data_file[0]:s}'!\n")
                for (prop, um, ui) in mismatch:
                    message += (
                        f"Unit for property for '{prop:s}' in metadata "
                        + f"'{um:s}' does not match with inout '{ui:s}'!")
                self.logger.error(message)
                raise SyntaxError(message)

        # Check for position and charge entry in 'unit_properties' in
        # metadata
        for prop in ['positions', 'charge']:
            if prop not in metadata['unit_properties']:
                if prop in data_unit_properties:
                    metadata['unit_properties'][prop] = (
                        data_unit_properties[prop])
                else:
                    metadata['unit_properties'][prop] = (
                        settings._default_units[prop])

        return metadata

    def get_data_properties_dtype(
        self,
    ) -> 'dtype':
        """
        Return database properties float dtype
        
        Returns:
        --------
        'dtype'
            Properties float dtype (most likely torch.float64)

        """
        with self.connect(self.data_file[0], mode='r') as db:
            return db.properties_torch_dtype

class DataSubSet(DataSet):
    """
    DataSubSet class iterating and returning over a subset of DataSet.

    Parameters
    ----------
    data_file: (str, tuple(str)), optional, default ('data.db', 'db.sql')
        Either a single string of the reference Asparagus database file name
        or a tuple of the filename first and the file format label second.
    data_label: str
        Data subset label
    subset_idx: List(int)
        List of reference data indices of this subset.

    Returns
    -------
    object
        DataSubSet to present training, validation or testing
    """

    def __init__(
        self,
        data_file: Union[str, Tuple[str, str]],
        data_label: str,
        subset_idx: List[int],
    ):
        """
        DataSubSet class
        """

        # Inherit from DataSet base class
        super().__init__(
            data_file,
            data_label,
            )

        # Assign arguments
        self.subset_idx = np.array(subset_idx, dtype=int)

        # Iterate over args
        for arg, item in locals().items():
            match = utils.check_input_dtype(
                arg, item, settings._dtypes_args, raise_error=True)

        # Check database file
        if not os.path.exists(data_file[0]):
            raise ValueError(
                f"File {data_file[0]:s} does not exists!\n")

        # Number of subset data points
        self.Nidx = len(self.subset_idx)

        # Get parent dataset parameters from metadata
        metadata = self.get_metadata()
        self.data_properties = metadata.get('load_properties')
        self.data_unit_properties = metadata.get('unit_properties')

        return

    def __len__(
        self,
    ) -> int:

        return self.Nidx

    def __getitem__(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(self.subset_idx[idx])

    def __setitem__(
        self,
        idx: int,
        properties: Dict[str, torch.tensor],
    ):

        return self._set_properties([self.subset_idx[idx]], [properties])

    def get(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        return self._get_properties(self.subset_idx[idx])

    def get_properties(
        self,
        idx: int,
    ) -> Dict[str, torch.tensor]:

        # Load property data
        properties = self._get_properties(self.subset_idx[idx])

        return properties

    def set_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._set_properties(
            [self.subset_idx[idxi] for idxi in idx], properties)

    def update_properties(
        self,
        idx: Union[int, List[int]],
        properties: Union[Dict[str, torch.tensor], List[Dict]],
    ):

        if utils.is_integer(idx):
            idx = [idx]
        if utils.is_dictionary(properties):
            properties = [properties]

        return self._update_properties(
            [self.subset_idx[idxi] for idxi in idx], properties)
