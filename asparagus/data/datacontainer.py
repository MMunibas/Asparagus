import time
import os
import logging
from typing import Optional, List, Dict, Tuple, Union, Any, Callable

import numpy as np

from .. import data
from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ['DataContainer']


class DataContainer():
    """
    DataContainer object that manage the distribution of the reference
    data from one or multiple databases into a DataSet object and provide
    DataSubSets for training, validation and test sets.

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    data_file: str, optional, default 'data.db'
        Reference Asparagus database file
    data_file_format: str, optional, default 'sql'
        Reference Asparagus database file format
    data_source: (str, List(str)), optional, default None
        Path to reference data set(s)
    data_source_format: (str, List(str)), optional, default file extension
        Dataset format of 'data_source'
    data_alt_property_labels: dictionary, optional, default
            'settings._alt_property_labels'
        Dictionary of alternative property labeling to replace
        non-valid property labels with the valid one if possible.
    data_unit_positions: str, optional, default 'Ang'
        Unit of the atom positions ('Ang' or 'Bohr') and other unit
        cell information.
    data_load_properties: List(str), optional,
            default ['energy', 'forces', 'dipole']
        Set of properties to store in the DataSet
    data_unit_properties: dictionary, optional, default {'energy':'eV'}
        Dictionary from properties (keys) to corresponding unit as a
        string (item), e.g.:
            {property: unit}: { 'energy', 'eV',
                                'forces', 'eV/Ang', ...}
    data_num_train: (int, float), optional, default 0.8 (80% of data)
        Number of training data points [absolute (>1) or relative
        (<= 1.0)].
    data_num_valid: (int, float), optional, default 0.1 (10% of data)
        Number of validation data points [absolute (>1) or relative
        (<= 1.0)].
    data_num_test: (int, float), optional, default 0.1 (10% of data)
        Number of test data points [absolute (>1) or relative (< 1.0)].
    data_batch_size
    data_train_batch_size: int, optional, default 128
        Training batch size
    data_valid_batch_size: int, optional, default 128
        Validation batch size
    data_test_batch_size:  int, optional, default 128
        Test batch size
    data_num_workers: int, optional, default 1
        Number of data loader workers
    data_overwrite: bool, optional, default False
        Overwrite database files with reference data from
        'data_source' if available.
    data_seed: (int, float), optional, default: np.random.randint(1E6)
        Define seed for random data splitting.

    """

    def __init__(
        self,
        config: Optional[Union[str, dict, object]] = None,
        config_file: Optional[str] = None, 
        data_file: Optional[str] = None,
        data_file_format: Optional[str] = None,
        data_source: Optional[Union[str, List[str]]] = None,
        data_source_format: Optional[Union[str, List[str]]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        data_unit_positions: Optional[str] = None,
        data_load_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_num_train: Optional[Union[int, float]] = None,
        data_num_valid: Optional[Union[int, float]] = None,
        data_num_test: Optional[Union[int, float]] = None,
        data_train_batch_size: Optional[int] = None,
        data_valid_batch_size: Optional[int] = None,
        data_test_batch_size: Optional[int] = None,
        data_num_workers: Optional[int] = None,
        data_overwrite: Optional[bool] = None,
        data_seed: Optional[int] = None,
        **kwargs,
    ):
        
        super().__init__()

        #####################################
        # # # Check DataContainer Input # # #
        #####################################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self, **kwargs)

        # If not to overwrite, get metadata from existing database
        if (
            data_overwrite or config.get('data_overwrite') 
            or data_file is None or config.get('data_file') is None
        ):
            metadata = {}
        else:
            # Get database reference file path and format
            if data_file is None:
                data_file = config.get('data_file')
            if data_file_format is None:
                data_file_format = config.get('data_file_format')
            metadata = data.get_metadata(data_file, data_file_format)
            # Check input with existing database properties
            data_load_properties, data_unit_properties = (
                self.get_from_metadata(
                    metadata,
                    config,
                    load_properties=data_load_properties,
                    unit_properties=data_unit_properties,
                    )
                )

        # Check input parameter, set default values if necessary and
        # update the configuration dictionary
        config_update = config.set(
            instance=self,
            argitems=utils.get_input_args(),
            argsskip=['metadata'],
            check_default=utils.get_default_args(self, data),
            check_dtype=utils.get_dtype_args(self, data)
        )

        # Update global configuration dictionary
        config.update(
            config_update, config_from=self)

        ########################################
        # # # Check DataSet Property Input # # #
        ########################################

        # Check and prepare data property input
        (self.data_load_properties, self.data_unit_properties,
        self.data_alt_property_labels) = (
            self.check_data_properties(
                self.data_load_properties, 
                self.data_unit_properties,
                self.data_unit_positions,
                self.data_alt_property_labels,
                )
            )

        #########################
        # # # DataSet Setup # # #
        #########################
        
        # Initialize data flag (False until setup is finished)
        self.data_avaiable = False

        # Initialize reference data set
        self.dataset = data.DataSet(
            self.data_file,
            data_file_format=self.data_file_format,
            data_load_properties=self.data_load_properties,
            data_unit_properties=self.data_unit_properties,
            data_overwrite=self.data_overwrite,
            **kwargs)

        # Reset dataset overwrite flag
        self.data_overwrite = False
        config['data_overwrite'] = False

        # Load source data
        self.dataset_load(
            self.data_source,
            self.data_source_format,
            self.data_alt_property_labels,
            **kwargs)
        
        # Finalize DataSet setup
        self.dataset_setup(
            **kwargs)

        return

    def __str__(self):
        """
        Return class descriptor
        """
        if hasattr(self, 'data_file'):
            return f"DataContainer '{self.data_file:s}'"
        else:
            return "DataContainer"

    def __len__(
        self,
    ) -> int:
        """
        Size of the complete data set
        """
        return len(self.dataset)

    def __getitem__(
        self,
        idx: int,
    ) -> Dict:
        """
        Get DataSet entry idx
        """
        return self.dataset.get(idx)

    def get(
        self,
        idx: int,
    ) -> Dict:
        """
        Get DataSet entry idx
        """
        return self.dataset.get(idx)

    def dataset_load(
        self,
        data_source: Union[str, List[str]],
        data_source_format: Optional[Union[str, List[str]]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        **kwargs,
    ):
        """
        Load source data to reference DataSet.
        """

        # Check data source input
        data_source, data_source_format = self.check_data_source(
            data_source,
            data_source_format,
        )

        # Load reference data set(s) from defined source data path(s)
        for source, source_format in zip(data_source, data_source_format):
            self.dataset.load_data(
                source,
                source_format,
                self.data_load_properties,
                self.data_unit_properties,
                self.data_alt_property_labels,
            )
            
        # Mirror DataSet parameters from metadata
        metadata = self.dataset.metadata
        self.data_source = metadata.get('data_source')
        self.data_source_format = metadata.get('data_source_format')

    def dataset_setup(
        self,
        **kwargs,
    ):
        """
        Setup the reference data set
        """

        # Prepare data split into training, validation and test set
        Ndata = len(self.dataset)

        # Stop further setup if no data are available
        if not Ndata:
            logger.error(
                f"ERROR:\nNo data are available in {self.data_file:s}!\n"
            )
            raise SyntaxError(
                f"No data are available in '{self.data_file:s}'!\n")

        ###########################
        # # # Data Separation # # #
        ###########################

        # Training set size
        if self.data_num_train < 0.0:
            logger.error(
                "ERROR:\nNumber of training set samples 'data_num_train'"
                + f"({self.data_num_train}) is lower then zero and invalid!\n"
            )
            raise ValueError(
                "Invalid Number of training set samples 'data_num_train'!")
        elif self.data_num_train <= 1.0:
            self.rel_train = float(self.data_num_train)
            self.data_num_train = int(Ndata*self.rel_train)
            self.rel_train = float(self.data_num_train)/float(Ndata)
        elif self.data_num_train <= Ndata:
            self.rel_train = float(self.data_num_train)/float(Ndata)
        else:
            logger.error(
                "ERROR:\nNumber of training set samples 'data_num_train' "
                + f"({self.data_num_train}) is larger than the total number "
                + f"of data samples ({Ndata})!\n"
            )
            raise ValueError(
                "Invalid Number of training set samples 'data_num_train'!")

        # Validation set size
        if self.data_num_valid < 0.0:
            raise ValueError(
                "Number of validation set samples 'data_num_valid' " +
                f"({self.data_num_valid}) is lower then zero and invalid!\n")
        elif self.data_num_valid < 1.0:
            self.rel_valid = float(self.data_num_valid)
            if (self.rel_train + self.rel_valid) > 1.0:
                rel_valid = 1.0 - float(self.data_num_train)/float(Ndata)
                logger.warning(
                    f"WARNING:\nRatio of training set ({self.rel_train})" +
                    f"and validation set samples ({self.rel_valid}) " +
                    "are larger 1.0!\n" +
                    "Ratio of validation set samples is set to " +
                    f"{rel_valid}.\n")
                self.rel_valid = rel_valid
            self.data_num_valid = int(round(Ndata*self.rel_valid))
            self.rel_valid = float(self.data_num_valid)/float(Ndata)
        elif self.data_num_valid <= (Ndata - self.data_num_train):
            self.rel_valid = float(self.data_num_valid)/float(Ndata)
        else:
            data_num_valid = int(Ndata - self.data_num_train)
            logger.warning(
                f"WARNING:\nNumber of training set ({self.data_num_train})" +
                "and validation set samples " +
                f"({self.data_num_valid}) are larger then number of " +
                f"data samples ({Ndata})!\n" +
                "Number of validation set samples is set to " +
                f"{data_num_valid}.\n")
            self.data_num_valid = data_num_valid
            self.rel_valid = float(self.data_num_valid)/float(Ndata)

        # Test set size
        if self.data_num_test is None:
            self.data_num_test = (
                Ndata - self.data_num_train - self.data_num_valid)
            self.rel_test = float(self.data_num_test)/float(Ndata)
        elif self.data_num_test < 0.0:
            raise ValueError(
                "Number of test set samples 'data_num_test' " +
                f"({self.data_num_test}) is lower then zero and invalid!\n")
        elif self.data_num_test < 1.0:
            self.rel_test = float(self.data_num_test)
            if (self.rel_test + self.rel_train + self.rel_valid) > 1.0:
                rel_test = (
                    1.0 - float(self.data_num_train + self.data_num_valid)
                    / float(Ndata))
                logger.warning(
                    f"WARNING:\nRatio of test set ({self.rel_test})" +
                    "with training and validation set samples " +
                    f"({self.rel_train}, {self.rel_valid}) " +
                    "are larger 1.0!\n" +
                    "Ratio of test set samples is set to " +
                    f"{rel_test}.\n")
                self.rel_test = rel_test
            self.data_num_test = int(round(Ndata*self.rel_test))
        elif self.data_num_test <= (
                Ndata - self.data_num_train - self.data_num_valid):
            self.rel_test = float(self.data_num_test)/float(Ndata)
        else:
            data_num_test = int(
                Ndata - self.data_num_train - self.data_num_valid)
            logger.warning(
                f"WARNING:\nNumber of training ({self.data_num_train}), " +
                f"validation set ({self.data_num_valid}) and " +
                f"test set samples ({self.data_num_test}) are larger " +
                f"then number of data samples ({Ndata})!\n" +
                "Number of test set samples is set to " +
                f"{data_num_test}.\n")
            self.data_num_test = data_num_test
            self.rel_test = float(self.data_num_test)/float(Ndata)

        ############################
        # # # DataSubSet Setup # # #
        ############################

        # Select training, validation and test data indices randomly
        np.random.seed(self.data_seed)
        idx_data = np.random.permutation(np.arange(Ndata))
        self.idx_train = idx_data[:self.data_num_train]
        self.idx_valid = idx_data[
            self.data_num_train:(self.data_num_train + self.data_num_valid)]
        self.idx_test = idx_data[
            (self.data_num_train + self.data_num_valid):
            (self.data_num_train + self.data_num_valid + self.data_num_test)]

        # Prepare training, validation and test subset
        self.train_set = data.DataSubSet(
            self.data_file,
            self.data_file_format,
            self.idx_train)
        self.valid_set = data.DataSubSet(
            self.data_file,
            self.data_file_format,
            self.idx_valid)
        self.test_set = data.DataSubSet(
            self.data_file,
            self.data_file_format,
            self.idx_test)
        logger.info(
            f"INFO:\n{Ndata:d} reference data points are distributed on " +
            "data subsets.\n" +
            f"Training data:   {len(self.train_set): 6d} " +
            f"({len(self.train_set)/Ndata*100: 3.1f}%)\n" +
            f"Validation data: {len(self.valid_set): 6d} " +
            f"({len(self.valid_set)/Ndata*100: 3.1f}%)\n" +
            f"Test data:       {len(self.test_set): 6d} " +
            f"({len(self.test_set)/Ndata*100: 3.1f}%)\n")

        ############################
        # # # DataLoader Setup # # #
        ############################

        # Prepare training, validation and test data loader
        self.train_loader = data.DataLoader(
            self.train_set,
            self.data_train_batch_size,
            True,
            self.data_num_workers)
        self.valid_loader = data.DataLoader(
            self.valid_set,
            self.data_valid_batch_size,
            False,
            self.data_num_workers)
        self.test_loader = data.DataLoader(
            self.test_set,
            self.data_test_batch_size,
            False,
            self.data_num_workers)

        # Prepare dictionaries as pointers between dataset label and the
        # respective DataSubSet and DataLoader objects
        self.all_data_sets = {
            'train': self.train_set,
            'valid': self.valid_set,
            'test': self.test_set}
        self.all_data_loder = {
            'train': self.train_loader,
            'valid': self.valid_loader,
            'test': self.test_loader}

        # Set data flag
        self.data_avaiable = True

    def get_from_metadata(
        self,
        metadata: Dict[str, Any],
        config: object,
        **kwargs,
    ) -> List[Any]:
        """
        Return input in kwargs from top priority source. Priority:
            1. Keyword argument input
            2. Config input
            3. Metadata properties
        """
        
        # Initialize top priority property list 
        properties = []

        # Iterate over properties
        for key, item in kwargs.items():
            
            # If not defined by input or in config, use property from metadata
            if item is None and config.get(key) is None:
                properties.append(metadata.get(key))
            # Else if defined by input, use property from config
            elif item is None:
                properties.append(config.get(key))
            # Else take input property
            else:
                properties.append(item)

        return properties

    def get_property_scaling(
        self,
        overwrite: Optional[bool] = False,
        property_atom_scaled: Optional[Dict[str, str]] = {
            'energy': 'atomic_energies'},
    ) -> Dict[str, List[float]]:
        """
        Compute property statistics with average and standard deviation.
        
        Parameters
        ----------
        overwrite: bool, optional, default False
            If property statistics already available and up-to-date, recompute
            them. The up-to-date flag will be reset to False if any database
            manipulation is done.
        property_atom_scaled: dict(str, str), optional, default ...
            Property statistics (key) will be scaled by the number of atoms
            per system and stored with new property label (item).
            Default: {'energy': 'atomic_energies'}

        Return
        ------
        dict(str, list(float))
            Property statistics dictionary
        """

        # Get current metadata dictionary
        metadata = self.dataset.get_metadata()

        # Initialize scaling and shift parameter dictionary
        property_scaling = {}
        for prop in metadata.get('load_properties'):
            # List of property mean value and standard deviation
            property_scaling[prop] = [0.0, 0.0]
            if prop in property_atom_scaled:
                atom_prop = property_atom_scaled[prop]
                property_scaling[atom_prop] = [0.0, 0.0]

        # Check property scaling status
        if (
            metadata.get('data_property_scaling_uptodate') is not None
            and metadata['data_property_scaling_uptodate']
            and not overwrite
        ):

            property_scaling = metadata.get('data_property_scaling')
        
        else:

            # Announce start of property statistics calculation
            logger.info(
                "INFO:\nStart computing training data property statistics. "
                + "This might take a moment.\n")

            # Iterate over training data properties and compute property mean
            Nsamples = 0.0
            for sample in self.train_set:

                # Iterate over sample properties
                for prop in metadata.get('load_properties'):

                    # Get property values
                    vals = sample.get(prop).numpy().reshape(-1)

                    # Compute average
                    scalar = np.mean(vals)
                    property_scaling[prop][0] = (
                        property_scaling[prop][0]
                        + (scalar - property_scaling[prop][0])/(Nsamples + 1.0)
                        ).item()
                    
                    # Scale by atom number if requested
                    if prop in property_atom_scaled:
                        
                        # Compute atom scaled average
                        atom_prop = property_atom_scaled[prop]
                        Natoms = sample.get('atoms_number').numpy().reshape(-1)
                        vals /= Natoms.astype(float)
                        
                        # Compute statistics normalized by atom numbers
                        scalar = np.mean(vals)
                        property_scaling[atom_prop][0] = (
                            property_scaling[atom_prop][0]
                            + (scalar - property_scaling[atom_prop][0])
                            / (Nsamples + 1.0)
                            ).item()

                # Increment sample counter
                Nsamples += 1.0

            # Iterate over training data properties and compute standard 
            # deviation
            for sample in self.train_set:

                # Iterate over sample properties
                for prop in metadata.get('load_properties'):

                    # Get property values
                    vals = sample.get(prop).numpy().reshape(-1)
                    Nvals = len(vals)

                    # Compute standard deviation contribution
                    for scalar in vals:
                        property_scaling[prop][1] = (
                            property_scaling[prop][1]
                            + (scalar - property_scaling[prop][0])**2/Nvals)

                    # Scale by atom number if requested
                    if prop in property_atom_scaled:
                        
                        # Compute atom scaled standard deviation contribution
                        atom_prop = property_atom_scaled[prop]
                        Natoms = sample.get('atoms_number').numpy().reshape(-1)
                        vals /= Natoms.astype(float)
                        
                        # Compute atom scaled standard deviation contribution
                        for scalar in vals:
                            property_scaling[atom_prop][1] = (
                                property_scaling[atom_prop][1]
                                + (scalar - property_scaling[atom_prop][0])**2
                                / Nvals)
            
            # Iterate over sample properties to complete standard deviation
            for prop in metadata.get('load_properties'):
                property_scaling[prop][1] = np.sqrt(
                    property_scaling[prop][1]/Nsamples).item()
                if prop in property_atom_scaled:
                    atom_prop = property_atom_scaled[prop]
                    property_scaling[atom_prop][1] = np.sqrt(
                        property_scaling[atom_prop][1]/Nsamples).item()

        # Collect and print property statistics information
        msg = f"  {'Property':<17s}|{'Average':>17s}  |"
        msg += f"{'Std. Deviation':>17s}  |  {'Unit':<12}\n"
        msg += "-"*len(msg) + "\n"
        # Iterate over sample properties
        for prop in metadata.get('load_properties'):
            msg += f"  {prop:<17s}|  {property_scaling[prop][0]:>15.3e}  |  "
            msg += f"{property_scaling[prop][1]:>15.3e}  |  "
            if prop in self.data_unit_properties:
                msg += f"{self.data_unit_properties[prop]:<15s}\n"
            else:
                msg += f"{'':<15s}\n"
            if prop in property_atom_scaled:
                atom_prop = property_atom_scaled[prop]
                msg += f"  {atom_prop:<17s}|  "
                msg += f"{property_scaling[atom_prop][0]:>15.3e}  |  "
                msg += f"{property_scaling[atom_prop][1]:>15.3e}  |  "
                if prop in self.data_unit_properties:
                    msg += f"{self.data_unit_properties[prop]:<10s}\n"
                elif atom_prop in self.data_unit_properties:
                    msg += f"{self.data_unit_properties[atom_prop]:<10}\n"
                else:
                    msg += f"{'':<15s}\n"
        logger.info("INFO:\nProperty Statistics\n" + msg + "\n")

        # Update property scaling
        if metadata.get('data_property_scaling') is None:
            metadata['data_property_scaling'] = property_scaling
        else:
            metadata['data_property_scaling'].update(property_scaling)
        metadata['data_property_scaling_uptodate'] = True
        self.dataset.set_metadata(metadata)

        return property_scaling

    def check_data_properties(
        self,
        data_load_properties, 
        data_unit_properties,
        data_unit_positions,
        data_alt_property_labels,
    ):
        """
        Check data property input.
        """

        # Combine alternative property label input 'data_alt_property_labels'
        # with default setting, check for repetitions.
        data_alt_property_labels = utils.merge_dictionary_lists(
            data_alt_property_labels, settings._alt_property_labels)

        # Check for unknown property labels in data_load_properties
        # and replace if possible with internally used property label in
        # *data_alt_property_labels'
        if utils.is_string(data_load_properties):
            data_load_properties = [data_load_properties]
        for ip, prop in enumerate(data_load_properties):
            match, modified, new_prop = utils.check_property_label(
                prop,
                valid_property_labels=settings._valid_properties,
                alt_property_labels=data_alt_property_labels)
            if match and modified:
                logger.warning(
                    f"WARNING:\nProperty key '{prop}' in "
                    + "'data_load_properties' is not a valid label!\n"
                    + f"Property key '{prop}' is replaced by '{new_prop}'.\n")
                data_load_properties[ip] = new_prop
            elif not match:
                raise ValueError(
                    f"Unknown property ('{prop}') in 'data_load_properties'!")

        # Check for unknown property labels in 'data_unit_properties'
        # and replace if possible with internally used property label in
        # 'data_alt_property_labels'
        props = list(data_unit_properties.keys())
        for prop in props:
            match, modified, new_prop = utils.check_property_label(
                prop,                valid_property_labels=settings._valid_properties,
                alt_property_labels=data_alt_property_labels)
            if match and modified:
                logger.warning(
                    f"WARNING:\nProperty key '{prop}' in "
                    + "'data_unit_properties' is not a valid label!\n"
                    + f"Property key '{prop}' is replaced by '{new_prop}'.\n")
                data_unit_properties[new_prop] = (
                    data_unit_properties.pop(prop))
            elif not match:
                raise ValueError(
                    f"Unknown property ('{prop}') in 'data_unit_properties'!")

        # Check if all units from data_load_properties are defined in
        # data_unit_properties, if not assign default units
        for prop in data_load_properties:
            if prop not in data_unit_properties.keys():
                logger.warning(
                    f"WARNING:\nNo unit defined for property '{prop}'!\n"
                    + f"Default unit of '{settings._default_units[prop]}' "
                    + "will be used.\n")
                data_unit_properties[prop] = settings._default_units[prop]

        # Check if positions unit is defined in data_unit_properties
        if (
            'positions' not in data_unit_properties 
            and data_unit_positions is not None
        ):
            data_unit_properties['positions'] = data_unit_positions
            
        return (
            data_load_properties, data_unit_properties, 
            data_alt_property_labels)

    def check_data_source(
        self,
        data_source: Union[str, List[str]],
        data_source_format: Optional[Union[str, List[str]]] = None,
    ):
        """
        Check data source input.
        """

        # Make data source iterable if necessary
        if not utils.is_array_like(data_source):
            data_source = [data_source]

        # Check data_source file extension
        if data_source_format is None or not len(data_source_format):
            data_source_format = []
            for path in data_source:
                data_source_format.append(path.split('.')[-1])
        elif utils.is_string(data_source_format):
            data_source_format = [data_source_format]*len(data_source)
        elif len(data_source_format) != len(data_source):
            raise ValueError(
                "Number of arguments in 'data_source_format' does not match " 
                + "number of arguments in 'data_source_format'!")
        
        # Check if DataSet file is part of data source
        if self.data_file in data_source:
            raise SyntaxError(
                f"DataSet file path '{self.data_file}' is part of data " 
                + "source list! The conflict must be avoided.")
        
        return data_source, data_source_format

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata
        """
        return self.dataset.get_metadata()

    def get_datalabels(self) -> List[str]:
        """
        Return the list of all available data set labels which return
        DataSubSet or DataLoader objects with the respective function
        (get_dataset and get_dataloader).
        """
        return self.all_data_sets.keys()

    def get_dataset(self, label) -> Callable:
        """
        Return as specific DataSubSet object ('train', 'valid' or 'test')
        """
        return self.all_data_sets.get(label)

    def get_dataloader(self, label) -> Callable:
        """
        Return as specific DataLoader object ('train', 'valid' or 'test')
        """
        return self.all_data_loder.get(label)

    def get_train(self, idx) -> Dict:
        """
        Get Training DataSubSet entry idx
        """
        return self.train_set.get(idx)

    def get_valid(self, idx) -> Dict:
        """
        Get Validation DataSubSet entry idx
        """
        return self.valid_set.get(idx)

    def get_test(self, idx) -> Dict:
        """
        Get Test DataSubSet entry idx
        """
        return self.test_set.get(idx)

    def get_info(self) -> Dict[str, Any]:
        """
        Return data information
        """

        return {
            'data_file': self.data_file,
            'data_file_format': self.data_file_format,
            'data_load_properties': self.data_load_properties,
            'data_unit_properties': self.data_unit_properties,
            'data_num_train': self.data_num_train,
            'data_num_valid': self.data_num_valid,
            'data_num_test': self.data_num_test,
            'data_overwrite': self.data_overwrite,
            }
