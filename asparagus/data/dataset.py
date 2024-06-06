import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import numpy as np

import torch

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataSet', 'DataSubSet']

class DataSet():
    """
    DataSet class containing and loading reference data from files
    
    Parameters
    ----------
    data_file: str
        Reference Asparagus database file
    data_file_format: str, optional, default 'data_file' prefix 
        Reference Asparagus database file format
    data_load_properties: List(str), optional, default None
        Subset of properties to load.
    data_unit_properties: dict, optional, default None
        Dictionary from properties (keys) to corresponding unit as a
        string (item), e.g.:
            {property: unit}: { 'positions': 'Ang',
                                'energy': 'eV',
                                'force': 'eV/Ang', ...}
    data_overwrite: bool, optional, default 'False'
        Overwrite database file
    
    Return
    ------
    callable
        Asparagus DataSet object
    """

    def __init__(
        self,
        data_file: str,
        data_file_format: Optional[str] = None,
        data_load_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_overwrite: Optional[bool] = False,
        **kwargs
    ):
        """
        Initialize DataSet class
        """

        # Assign arguments
        self.data_file = data_file
        if data_file_format is None:
            data_file_format = data_file.split('.')[-1]
        self.data_file_format = data_file_format

        # Check for data path existence
        path, _ = os.path.split(self.data_file)
        if path and not os.path.isdir(path):
            os.makedirs(path)

        # If overwrite, remove old DataSet file
        if os.path.exists(data_file) and data_overwrite:
            os.remove(data_file)

        # Copy current metadata
        metadata = self.get_metadata()
        
        # Check data property compatibility
        metadata = self.check_data_compatibility(
            metadata,
            data_load_properties,
            data_unit_properties)
        
        # Set metadata
        self.set_metadata(metadata)
        
        # Initialize DataReader variable
        self.datareader = None

        return

    def __len__(
        self,
    ) -> int:

        if os.path.isfile(self.data_file):
            with data.connect(self.data_file, self.data_file_format) as db:
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
        return self

    def __next__(
        self
    ):
        # Check counter within number of data range
        if self.counter < self.Ndata:
            data = self.get(self.counter)
            self.counter += 1
            return data
        else:
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

        with data.connect(self.data_file, mode='r') as db:
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

        with data.connect(self.data_file, mode='a') as db:
            for idx, props in zip(idcs, properties):
                row_id = db.write(props, row_id=idx + 1)

        return

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

        with data.connect(self.data_file, mode='a') as db:
            for idx, props in zip(idcs, properties):
                row_id = db.update(row_id=idx + 1, properties=props)

        return

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
        with data.connect(
            self.data_file, self.data_file_format, mode='r'
        ) as db:
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
        with data.connect(
            self.data_file, self.data_file_format, mode='a'
        ) as db:
            db.set_metadata(metadata)
            db.init_systems()

    def reset_database(
        self,
    ):
        with data.connect(
            self.data_file, self.data_file_format, mode='a'
        ) as db:
            db.reset()

    def load_data(
        self,
        data_source: str,
        data_source_format: str,
        data_load_properties: List[str],
        data_unit_properties: Dict[str, str],
        alt_property_labels: Optional[Dict[str, str]] = None,
    ):

        # Get metadata from database file
        metadata = self.get_metadata()

        # Check if data source already loaded
        if metadata.get('data_source') is None:
            metadata['data_source'] = []
            metadata['data_source_format'] = []
        elif data_source in metadata['data_source']:
            logger.warning(
                f"WARNING:\nData source '{data_source:s}' already "
                + f"written to dataset '{self.data_file:s}'! "
                + "Loading data source is skipped.\n")
            return
        
        # Append data source information
        metadata['data_source'].append(data_source)
        metadata['data_source_format'].append(data_source_format)
        
        # Reset property scaling flag
        metadata['data_property_scaling_uptodate'] = False
        
        # Initialize DataReader
        datareader = data.DataReader(
            data_file=self.data_file,
            data_file_format=self.data_file_format,
            alt_property_labels=alt_property_labels)
        
        # Load data file
        datareader.load(
            data_source,
            data_source_format,
            data_load_properties,
            data_unit_properties,
            alt_property_labels)
        
        # If metadata properties is empty, initialize database
        if (
            metadata.get('load_properties') is None
            or metadata.get('unit_properties') is None
        ):
            metadata['load_properties'] = data_load_properties
            metadata['unit_properties'] = data_unit_properties
        
        # Set updated metadata
        self.set_metadata(metadata)

    def add_atoms(
        self,
        atoms: object,
        properties: Dict[str, Any],
        #alt_property_labels: Optional[Dict[str, str]] = None,
    ):
        """
        Add ASE Atoms system and properties
        """

        # Check and, in case, initialize DataReader
        if self.datareader is None:
            self.datareader = data.DataReader(
                data_file=self.data_file,
                data_file_format=self.data_file_format,
                #alt_property_labels=alt_property_labels
                )

        # Get dataset metadata
        metadata = self.get_metadata()

        # Load from ASE atoms object and property list
        self.datareader.load_atoms(
            atoms,
            properties,
            data_load_properties=metadata['load_properties'],
            data_unit_properties=metadata['unit_properties'],
            #alt_property_labels=alt_property_labels
            )

        return

    def check_data_compatibility(
        self,
        metadata: Dict[str, Any],
        data_load_properties: List[str],
        data_unit_properties: Dict[str, str],
    ):
        """
        Check compatibility between input for 'data_load_properties' and 
        'data_unit_properties' with metadata.
        """
        if (
            metadata.get('load_properties') is None
            and data_load_properties is None
        ):
            metadata['load_properties'] = []
        elif metadata.get('load_properties') is None:
            metadata['load_properties'] = data_load_properties
        elif data_load_properties is None:
            data_load_properties = metadata['load_properties']
        else:
            mismatch_metadata = []
            mismatch_input = []
            # Check property match except for default properties always stored
            # in database 
            for prop in metadata.get('load_properties'):
                if (
                    prop not in data_load_properties
                    and prop not in settings._default_property_labels
                ):
                    mismatch_metadata.append(prop)
            for prop in data_load_properties:
                if (
                    prop not in metadata.get('load_properties')
                    and prop not in settings._default_property_labels
                ):
                    mismatch_input.append(prop)
            if len(mismatch_metadata) or len(mismatch_input):
                msg = (
                    "Mismatch between DataSet 'load_properties' "
                    + f"input and metadata in '{self.data_file:s}'!\n")
                for prop in mismatch_metadata:
                    msg += f"Property '{prop:s}' in metadata not in input.\n"
                for prop in mismatch_input:
                    msg += f"Property '{prop:s}' in input not in metadata.\n"
                logger.error("Error:\n" + msg)
                raise SyntaxError(msg)
        
        # Check compatibility between 'data_unit_properties' in metadata
        # and input
        if (
            metadata.get('unit_properties') is None
            and data_unit_properties is None
        ):
            metadata['unit_properties'] = {}
        elif metadata.get('unit_properties') is None:
            metadata['unit_properties'] = data_unit_properties
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
                msg = (
                    "Mismatch between DataContainer 'data_unit_properties' "
                    + f"input and metadata in '{self.data_file:s}'!\n")
                for (prop, um, ui) in mismatch:
                    msg += (
                        f"Unit for property for '{prop:s}' in metadata "
                        + f"'{um:s}' does not match with inout '{ui:s}'!\n")
                logger.error("Error:\n" + msg)
                raise SyntaxError(msg)

        return metadata


class DataSubSet(DataSet):
    """
    DataSubSet class iterating and returning over a subset of DataSet.
    
    Parameters
    ----------
    data_file: str
        Reference ASE database file
    subset_idx: List(int)
        List of reference data indices of this subset.

    Returns
    -------
    object
        DataSubSet to present training, validation or testing
    """

    def __init__(
        self,
        data_file: str,
        data_file_format: str,
        subset_idx: List[int],
    ):
        """
        DataSubSet class 
        """

        # Inherit from DataSet base class
        super().__init__(
            data_file,
            data_file_format=data_file_format
            )

        # Assign arguments
        self.subset_idx = [int(idx) for idx in subset_idx]

        # Iterate over args
        for arg, item in locals().items():
            match = utils.check_input_dtype(
                arg, item, settings._dtypes_args, raise_error=True)

        # Check database file
        if not os.path.exists(data_file):
            raise ValueError(
                f"File {data_file} does not exists!\n")

        # Number of subset data points
        self.Nidx = len(self.subset_idx)

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
