import os
import logging
from typing import Optional, List, Tuple, Dict, Union, Any, Callable

import numpy as np

from asparagus import data
from asparagus import utils
from asparagus import settings

__all__ = ['DataContainer']


class DataContainer():
    """
    DataContainer object that manage the distribution of the reference
    data from one or multiple databases into a DataSet object and provide
    DataSubSets for training, validation and test sets.

    Parameters
    ----------
    config: (str, dict, settings.Configuration), optional, default None
        Either the path to json file (str), dictionary (dict) or
        settings.config class object of model parameters
    config_file: str, optional, default see settings.default['config_file']
        Path to json file (str)
    data_file: (str, tuple(str)), optional, default ('data.db', 'db.sql')
        Either a single string of the reference Asparagus database file name
        or a tuple of the filename first and the file format label second.
    data_source: (str, list(str)), optional, default None
        List (or string) of paths to reference data files. Each entry can be
        either a string for the file path or a tuple with the filename first
        and the file format label second.
    data_alt_property_labels: dict, optional, default
            'settings._alt_property_labels'
        Dictionary of alternative property labeling to replace
        non-valid property labels with the valid one if possible.
    data_properties: list(str), optional,
            default ['energy', 'forces', 'dipole']
        Set of properties to store in the DataSet
    data_unit_properties: dictionary, optional,
            default {'energy': 'eV', 'forces': 'eV/Ang', 'dipole': 'e*Ang'}
        Dictionary from properties (keys) to corresponding unit as a
        string (item), e.g.:
            {property: unit}: { 'energy', 'eV',
                                'forces', 'eV/Ang', ...}
    data_source_unit_properties: dictionary, optional, default None
        Dictionary from properties (keys) to corresponding unit as a
        string (item) in the source data files.
        If None, the property units as defined in 'data_unit_properties'
        are assumed.
        This input is only regarded for data source format, where no property
        units are defined such as the Numpy npz files.
    data_num_train: (int, float), optional, default 0.8 (80% of data)
        Number of training data points [absolute (>1) or relative
        (<= 1.0)].
    data_num_valid: (int, float), optional, default 0.1 (10% of data)
        Number of validation data points [absolute (>1) or relative
        (<= 1.0)].
    data_num_test: (int, float), optional, default None
        Number of test data points [absolute (>1) or relative (< 1.0)].
        If None, remaining data in the database are used.
    data_seed: (int, float), optional, default: np.random.randint(1E6)
        Define seed for random data splitting.
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

    """

    # Initialize logger
    name = f"{__name__:s} - {__qualname__:s}"
    logger = utils.set_logger(logging.getLogger(name))

    # Default arguments for data modules
    _default_args = {
        'data_file':                    ('data.db', 'sql.db'),
        'data_source':                  None,
        'data_properties':              ['energy', 'forces', 'dipole'],
        'data_unit_properties':         {'energy': 'eV',
                                        'forces': 'eV/Ang',
                                        'dipole': 'e*Ang'},
        'data_source_unit_properties':  None,
        'data_alt_property_labels':     {},
        'data_num_train':               0.8,
        'data_num_valid':               0.1,
        'data_num_test':                None,
        'data_seed':                    np.random.randint(1E6),
        'data_overwrite':               False,
        }

    # Expected data types of input variables
    _dtypes_args = {
        'data_file':                    [
            utils.is_string, utils.is_string_array_inhomogeneous],
        'data_source':                  [
            utils.is_string, utils.is_string_array_inhomogeneous,
            utils.is_None],
        'data_properties':              [utils.is_array_like],
        'data_unit_properties':         [utils.is_dictionary],
        'data_source_unit_properties':  [utils.is_dictionary, utils.is_None],
        'data_alt_property_labels':     [utils.is_dictionary],
        'data_num_train':               [utils.is_numeric],
        'data_num_valid':               [utils.is_numeric],
        'data_num_test':                [utils.is_numeric, utils.is_None],
        'data_seed':                    [utils.is_numeric],
        'data_overwrite':               [utils.is_bool],
        }

    def __init__(
        self,
        config: Optional[Union[str, dict, settings.Configuration]] = None,
        config_file: Optional[str] = None,
        data_file: Optional[Union[str, Tuple[str, str]]] = None,
        data_source: Optional[Union[str, List[str]]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_source_unit_properties: Optional[Dict[str, str]] = None,
        data_num_train: Optional[Union[int, float]] = None,
        data_num_valid: Optional[Union[int, float]] = None,
        data_num_test: Optional[Union[int, float]] = None,
        data_seed: Optional[int] = None,
        data_train_batch_size: Optional[int] = None,
        data_valid_batch_size: Optional[int] = None,
        data_test_batch_size: Optional[int] = None,
        data_num_workers: Optional[int] = None,
        data_overwrite: Optional[bool] = None,
        device: Optional[str] = None,
        dtype: Optional[object] = None,
        **kwargs,
    ):

        super().__init__()

        #####################################
        # # # Check DataContainer Input # # #
        #####################################

        # Get configuration object
        config = settings.get_config(
            config, config_file, config_from=self, **kwargs)

        # Get database reference file path
        if data_file is None:
            data_file = config.get('data_file')

        # Check 'data_file' input for file format information
        data_file = self.check_data_files(data_file)
            
        # If not to overwrite, get metadata from existing database
        if (
            data_overwrite or config.get('data_overwrite') or data_file is None
        ):

            metadata = {}

        else:

            # Get, eventually the metadata dictionary from the data file
            metadata = data.get_metadata(data_file)

            # Check input with existing database properties
            data_properties, data_unit_properties = (
                self.get_from_metadata(
                    metadata,
                    config,
                    data_properties=data_properties,
                    unit_properties=data_unit_properties,
                    )
                )

        # Check data source file and format input
        if data_source is None:
            data_source = config.get('data_source')
        data_source = self.check_data_files(data_source, is_source=True)

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

        # Assign module variable parameters from configuration
        self.device = utils.check_device_option(device, config)
        self.dtype = utils.check_dtype_option(dtype, config)

        ######################################
        # # # Check Data Parameter Input # # #
        ######################################

        # Check data file format again, if updated
        if self.data_file != data_file:
            self.data_file = self.check_data_files(self.data_file)

        # Check and prepare data property input
        data_properties, data_unit_properties, data_alt_property_labels = (
            self.check_data_properties(
                self.data_properties,
                self.data_unit_properties,
                self.data_alt_property_labels,
                )
            )

        # Reassign data properties
        self.data_properties = data_properties
        self.data_unit_properties = data_unit_properties
        self.data_alt_property_labels = data_alt_property_labels

        # Check and reassign source property unit input
        if self.data_source_unit_properties is not None:
            _, data_source_unit_properties, _ = (
            self.check_data_properties(
                self.data_properties,
                self.data_source_unit_properties,
                self.data_alt_property_labels,
                )
            )
            self.data_source_unit_properties = data_source_unit_properties

        #########################
        # # # DataSet Setup # # #
        #########################

        # Initialize reference data set
        self.dataset = data.DataSet(
            self.data_file,
            data_properties=self.data_properties,
            data_unit_properties=self.data_unit_properties,
            data_alt_property_labels=self.data_alt_property_labels,
            data_overwrite=self.data_overwrite,
            **kwargs)

        # Reset dataset overwrite flag
        self.data_overwrite = False
        config['data_overwrite'] = False

        # Load source data
        self.data_source = self.load_data_source(
            self.data_source,
            **kwargs)

        # Split the dataset into data subsets
        self.split_dataset(
            data_num_train=self.data_num_train,
            data_num_valid=self.data_num_valid,
            data_num_test=self.data_num_test,
            data_seed=self.data_seed,
            )

        # Update global configuration dictionary
        config.update(
            {
                'data_file': self.data_file,
                'data_properties': self.data_properties,
                'data_unit_properties': self.data_unit_properties,
                'data_source': self.data_source,
                'data_source_unit_properties': self.data_source_unit_properties
                },
            config_from=self)

        return

    def __str__(self):
        """
        Return class descriptor
        """
        if hasattr(self, 'data_file'):
            return (
                f"DataContainer '{self.data_file[0]:s}' "
                + f" ({self.data_file[1]:s})")
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

    def check_data_files(
        self,
        files: Union[str, Tuple[str, str], List[str], List[Tuple[str, str]]],
        is_source: Optional[bool] = False,
    ):
        """
        Check files input for file format information.

        Parameters
        ----------
        files: (str, tuple(str), list(str), list(tuple(str)))
            Single or list of data files and, eventually, file formats
        is_source: bool, optional, default False
            If False, files should be a single data file path with format.
            Else, a list of data source files are expected.

        Returns
        -------
        tuple(str) or list(tuple(str))
            File name and format informations either as tuple of on file
            (is_source=False) or a list of tuples (is_source=True).

        """

        # Check files input
        if files is None:
            return None

        # Initialize (file, format) list
        files_formats = []

        # Initialize files input list
        files_input = []

        # Files input is string
        if utils.is_string(files):

            # Check file existense if source
            if is_source and not os.path.isfile(files):
                self.logger.warning(
                    f"Source data file name ('{files:s}') does not exist!")

            # Get file format
            file_format = data.check_data_format(
                files, is_source_format=is_source)
            files_formats.append([files, file_format])
            files_input.append(f" <- {files:s}")

        # Files is string list
        elif utils.is_string_array(files, inhomogeneity=True):

            # If data file, only file path and format is expected
            if not is_source:

                # Check assigned file format
                file_format = data.check_data_format(
                    files[0], is_source_format=is_source)
                format_format = data.check_data_format(
                    files[1], is_source_format=is_source)
                if file_format != format_format:
                    self.logger.warning(
                        f"Data file name ('{files[0]:s}') and format "
                        + f"definition ('{files[1]:s}') does not match "
                        + f"('{file_format:s}' != '{format_format:s}')!")

                files_formats.append([files[0], format_format])
                files_input.append(f" <- ({files[0]:s}, {files[1]:s})")

            # If source files, multiple definitions can be expected
            else:

                # Iterate over files (or files and format)
                for file_i in files:

                    # Check for (file, format) pair
                    if utils.is_string_array(file_i):

                        # Check assigned file format
                        file_format = data.check_data_format(
                            file_i[0], is_source_format=is_source)
                        format_format = data.check_data_format(
                            file_i[1], is_source_format=is_source)
                        if file_format != format_format:
                            self.logger.warning(
                                f"Data file name ('{file_i[0]:s}') and format "
                                + f"definition ('{file_i[1]:s}') does not "
                                + f"match ('{file_format:s}' != "
                                + f"'{format_format:s}')!")

                        # Check for existense, which is expected
                        if not os.path.isfile(file_i[0]):
                            self.logger.warning(
                                f"Source data file name ('{file_i[0]:s}') "
                                + "does not exist!")

                        files_formats.append([file_i[0], format_format])
                        files_input.append(
                            f" <- ({file_i[0]:s}, {file_i[1]:s})")

                    elif utils.is_string(file_i):

                        # Get file format
                        file_format = data.check_data_format(
                            file_i, is_source_format=is_source)

                        # Check for existense, if not it is most likely the
                        # file format definition of the former input.
                        if not os.path.isfile(file_i):

                            self.logger.warning(
                                f"Source data file name ('{file_i:s}') does "
                                + " not exist!")

                        else:

                            files_formats.append([file_i, file_format])
                            files_input.append(f" <- {file_i:s}")

        # Prepare check info
        msg = ""
        for file_output, file_input in zip(files_formats, files_input):
            msg += (
                f" ({file_output[0]:s}, {file_output[1]:s}) {file_input:s}\n")

        # Return either tuple or list of tuples
        if is_source:
            self.logger.info(
                f"Data source files and formats detected:\n{msg:s}")
            return files_formats
        else:
            self.logger.info(
                f"Data file and format detected:\n{msg:s}")
            if len(files_formats) > 1:
                self.logger.warning(
                    "Multiple files were defined as data files, but only one "
                    + "is supported!\n List of files and file formats: "
                    + f" {str(files_formats):s}\n"
                    + f" Returned file and format {str(files_formats[0]):s}")
            return files_formats[0]

    def get_from_metadata(
        self,
        metadata: Dict[str, Any],
        config: settings.Configuration,
        **kwargs,
    ) -> List[Any]:
        """
        Return input in kwargs from top priority source. Priority:
            1. Keyword argument input
            2. Config input
            3. Metadata properties

        Parameters
        ----------
        metadata: dict
            Database file metadata deictionary.
        config: setting.Configuration
            Configuaration object with parameters
        kwargs: dict
            Parameter dictionary to check

        Returns
        -------
        list
            Updated parameter list in the order of kwargs

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

    def check_data_properties(
        self,
        data_properties: Union[str, List[str]],
        data_unit_properties: Dict[str, str],
        data_alt_property_labels: Dict[str, List[str]],
    ) -> (List[str], Dict[str, str], Dict[str, List[str]]):
        """
        Check data property input.

        Parameters
        ----------
        data_properties: (str, list(str))
            Set of properties to store in the DataSet
        data_unit_properties: dict
            Dictionary from properties (keys) to corresponding unit as a
            string (item)
        data_alt_property_labels: dict
            Alternative property labels to detect common mismatches.

        Returns
        -------
        list
            Updated parameter list in the order of kwargs

        """

        # Combine alternative property label input 'data_alt_property_labels'
        # with default setting, check for repetitions.
        data_alt_property_labels = utils.merge_dictionary_lists(
            data_alt_property_labels, settings._alt_property_labels)

        # Check for unknown property labels in data_properties
        # and replace if possible with internally used property label in
        # *data_alt_property_labels'
        if utils.is_string(data_properties):
            data_properties = [data_properties]
        for ip, prop in enumerate(data_properties):
            match, modified, new_prop = utils.check_property_label(
                prop,
                valid_property_labels=settings._valid_properties,
                alt_property_labels=data_alt_property_labels)
            if match and modified:
                self.logger.warning(
                    f"Property key '{prop}' in "
                    + "'data_properties' is not a valid label!\n"
                    + f"Property key '{prop}' is replaced by '{new_prop}'.")
                data_properties[ip] = new_prop
            elif not match:
                raise ValueError(
                    f"Unknown property ('{prop}') in 'data_properties'!")

        # Check for unknown property labels in 'data_unit_properties'
        # and replace if possible with internally used property label in
        # 'data_alt_property_labels'
        for prop in data_unit_properties:
            match, modified, new_prop = utils.check_property_label(
                prop,
                valid_property_labels=settings._valid_properties,
                alt_property_labels=data_alt_property_labels)
            if match and modified:
                self.logger.warning(
                    f"Property key '{prop}' in "
                    + "'data_unit_properties' is not a valid label!\n"
                    + f"Property key '{prop}' is replaced by '{new_prop}'.")
                data_unit_properties[new_prop] = (
                    data_unit_properties.pop(prop))
            elif not match:
                raise ValueError(
                    f"Unknown property ('{prop}') in 'data_unit_properties'!")

        # Initialize checked property units dictionary
        checked_data_unit_properties = {}

        # Check if positions and charge units are defined in
        # 'data_unit_properties'.
        for prop in ['positions', 'charge']:
            if prop not in data_unit_properties:
                checked_data_unit_properties[prop] = (
                    settings._default_units[prop])
            else:
                checked_data_unit_properties[prop] = (
                    data_unit_properties[prop])

        # Check if all units from 'data_properties' are defined in
        # 'data_unit_properties', if not assign default units.
        for prop in data_properties:
            if prop not in data_unit_properties:
                self.logger.warning(
                    f"No unit defined for property '{prop}'!\n"
                    + f"Default unit of '{settings._default_units[prop]}' "
                    + "will be used.")
                checked_data_unit_properties[prop] = (
                    settings._default_units[prop])
            else:
                checked_data_unit_properties[prop] = data_unit_properties[prop]

        return (
            data_properties, checked_data_unit_properties,
            data_alt_property_labels)

    def load_data_source(
        self,
        data_source: Union[str, List[str]],
        data_properties: Optional[List[str]] = None,
        data_unit_properties: Optional[Dict[str, str]] = None,
        data_alt_property_labels: Optional[Dict[str, List[str]]] = None,
        data_source_unit_properties: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Load source data to reference DataSet.

        Parameters
        ----------
        data_source: (str, list(str))
            List (or string) of paths to reference data files. Each entry can
            be either a string for the file path or a tuple with the filename
            first and the file format label second.
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

        Returns
        -------
        list(tuple(str))
            File name and format informations as a list of tuples.

        """

        # Check data source input
        if data_source is None:
            data_source = []
        data_source = self.check_data_files(data_source, is_source=True)

        # Check property input
        if data_properties is None:
            data_properties = self.data_properties
        if data_unit_properties is None:
            data_unit_properties = self.data_unit_properties
        if data_alt_property_labels is None:
            data_alt_property_labels = self.data_alt_property_labels
        if data_source_unit_properties is None:
            data_source_unit_properties = self.data_source_unit_properties

        # Load reference data set(s) from defined source data path(s)
        for source in data_source:
            self.dataset.load_data(
                source,
                data_properties=data_properties,
                data_unit_properties=data_unit_properties,
                data_alt_property_labels=data_alt_property_labels,
                data_source_unit_properties=data_source_unit_properties,
            )

        # Return data source information from metadata
        metadata = self.dataset.get_metadata()

        return metadata.get('data_source')

    def split_dataset(
        self,
        data_num_train: Optional[Union[int, float]] = None,
        data_num_valid: Optional[Union[int, float]] = None,
        data_num_test: Optional[Union[int, float]] = None,
        data_seed: Optional[int] = None,
    ):
        """
        Split dataset into data subsets
        
        Parameters
        ----------
        data_num_train: (int, float), optional, default 0.8 (80% of data)
            Number of training data points [absolute (>1) or relative
            (<= 1.0)].
        data_num_valid: (int, float), optional, default 0.1 (10% of data)
            Number of validation data points [absolute (>1) or relative
            (<= 1.0)].
        data_num_test: (int, float), optional, default None
            Number of test data points [absolute (>1) or relative (< 1.0)].
            If None, remaining data in the database are used.
        data_seed: (int, float), optional, default: np.random.randint(1E6)
            Define seed for random data splitting.

        """

        # Prepare data split into training, validation and test set
        data_num_all = len(self.dataset)

        # Stop further setup if no data are available
        if not data_num_all:
            msg = f"No data are available in '{self.data_file[0]:s}!"
            self.logger.error(msg)
            raise SyntaxError(msg)

        # Check data split parameter
        if data_num_train is None:
            data_num_train = self.data_num_train
        if data_num_valid is None:
            data_num_valid = self.data_num_valid
        if data_num_test is None:
            data_num_test = self.data_num_test
        if data_seed is None:
            data_seed = self.data_seed

        # Prepare split parameters
        num_train, num_valid, num_test = self.prepare_split_parameter(
            data_num_all,
            data_num_train,
            data_num_valid,
            data_num_test)

        # Select training, validation and test data indices randomly
        np.random.seed(data_seed)
        idx_data = np.random.permutation(np.arange(data_num_all))
        idx_train = idx_data[:num_train]
        idx_valid = idx_data[num_train:(num_train + num_valid)]
        idx_test = idx_data[
            (num_train + num_valid):(num_train + num_valid + num_test)]

        # Initialize training, validation and test subset
        self.train_dataset = data.DataSubSet(
            self.data_file,
            'test',
            idx_train)
        self.valid_dataset = data.DataSubSet(
            self.data_file,
            'valid',
            idx_valid)
        self.test_dataset = data.DataSubSet(
            self.data_file,
            'test',
            idx_test)

        # Prepare dataset and subset label to objects dictionary
        self.all_datasets = {
            'all': self.dataset,
            'train': self.train_dataset,
            'training': self.train_dataset,
            'valid': self.valid_dataset,
            'validation': self.valid_dataset,
            'test': self.test_dataset,
            'testing': self.test_dataset,
            }

        # Prepare header
        message = (
            "Dataset and subset split information of database "
            + f"'{self.data_file[0]:s}'!\n"
            + f" {'Dataset':<17s} |"
            + f" {'Abs. Number':<14s} |"
            + f" {'Rel. Number':<14s}\n"
            + "-"*(20 + 17*2)
            + "\n")

        for label in ['All', 'Training', 'Validation', 'Test']:
            
            # Get dataset and subset sizes
            num_abs = len(self.all_datasets[label.lower()])
            num_rel = float(num_abs)/float(data_num_all)

            # Prepare information
            message += (
                f" {label + ' Data':<17s} |"
                + f" {num_abs:>14d} |"
                + f" {num_rel*100:>13.1f}%\n")

        # Print information
        self.logger.info(message)

        return

    def prepare_split_parameter(
        self,
        data_num_all: int,
        data_num_train: Union[int, float],
        data_num_valid: Union[int, float],
        data_num_test: Union[int, float],
    ) -> (int, int, int):
        """
        Split dataset into data subsets
        
        Parameters
        ----------
        data_num_all: int
            Total number of data.
        data_num_train: (int, float)
            Number of training data points [absolute (>1) or relative
            (<= 1.0)].
        data_num_valid: (int, float)
            Number of validation data points [absolute (>1) or relative
            (<= 1.0)].
        data_num_test: (int, float)
            Number of test data points [absolute (>1) or relative (< 1.0)].
            If None, remaining data in the database are used.

        Returns
        -------
        int
            Number of training data
        int
            Number of validation data
        int
            Number of test data

        """

        # Training set size
        if data_num_train < 0.0:
            msg = (
                "Number of training set samples 'data_num_train'"
                + f"({data_num_train}) is lower then zero and invalid!")
            self.logger.error(msg)
            raise ValueError(msg)
        elif data_num_train <= 1.0:
            rel_train = float(self.data_num_train)
            data_num_train = int(data_num_all*rel_train)
            rel_train = float(data_num_train)/float(data_num_all)
        elif data_num_train <= data_num_all:
            rel_train = float(data_num_train)/float(data_num_all)
        else:
            msg = (
                "Number of training set samples 'data_num_train' "
                + f"({data_num_train}) is larger than the total number "
                + f"of data samples ({data_num_all})!"
            )
            self.logger.error(msg)
            raise ValueError(msg)

        # Validation set size
        if data_num_valid < 0.0:
            raise ValueError(
                "Number of validation set samples 'data_num_valid' " +
                f"({data_num_valid}) is lower then zero and invalid!\n")
        elif data_num_valid < 1.0:
            rel_valid = float(data_num_valid)
            if (rel_train + rel_valid) > 1.0:
                new_rel_valid = 1.0 - float(data_num_train)/float(data_num_all)
                self.logger.warning(
                    f"Ratio of training set ({rel_train})" +
                    f"and validation set samples ({rel_valid}) " +
                    "are larger 1.0!\n" +
                    "Ratio of validation set samples is set to " +
                    f"{new_rel_valid}.")
                rel_valid = new_rel_valid
            data_num_valid = int(round(data_num_all*rel_valid))
            rel_valid = float(data_num_valid)/float(data_num_all)
        elif data_num_valid <= (data_num_all - data_num_train):
            rel_valid = float(data_num_valid)/float(data_num_all)
        else:
            new_data_num_valid = int(data_num_all - data_num_train)
            self.logger.warning(
                f"Number of training set ({data_num_train})" +
                "and validation set samples " +
                f"({data_num_valid}) are larger then number of " +
                f"data samples ({data_num_all})!\n" +
                "Number of validation set samples is set to " +
                f"{new_data_num_valid}")
            data_num_valid = new_data_num_valid
            rel_valid = float(data_num_valid)/float(data_num_all)

        # Test set size
        if data_num_test is None:
            data_num_test = (
                data_num_all - data_num_train - data_num_valid)
            rel_test = float(data_num_test)/float(data_num_all)
        elif data_num_test < 0.0:
            raise ValueError(
                "Number of test set samples 'data_num_test' " +
                f"({data_num_test}) is lower then zero and invalid!\n")
        elif data_num_test < 1.0:
            rel_test = float(data_num_test)
            if (rel_test + rel_train + rel_valid) > 1.0:
                new_rel_test = (
                    1.0 - float(data_num_train + data_num_valid)
                    / float(data_num_all))
                self.logger.warning(
                    f"Ratio of test set ({rel_test})" +
                    "with training and validation set samples " +
                    f"({rel_train}, {rel_valid}) " +
                    "are larger 1.0!\n" +
                    "Ratio of test set samples is set to " +
                    f"{new_rel_test}.")
                rel_test = new_rel_test
            data_num_test = int(round(data_num_all*rel_test))
        elif data_num_test <= (data_num_all - data_num_train - data_num_valid):
            rel_test = float(data_num_test)/float(data_num_all)
        else:
            new_data_num_test = int(
                data_num_all - data_num_train - data_num_valid)
            self.logger.warning(
                f"Number of training ({data_num_train}), " +
                f"validation set ({data_num_valid}) and " +
                f"test set samples ({data_num_test}) are larger " +
                f"then number of data samples ({data_num_all})!\n" +
                "Number of test set samples is set to " +
                f"{new_data_num_test}.")
            data_num_test = new_data_num_test
            rel_test = float(data_num_test)/float(data_num_all)

        return data_num_train, data_num_valid, data_num_test

    def init_dataloader(
        self,
        train_batch_size: int,
        valid_batch_size: int,
        test_batch_size: int,
        num_workers: Optional[int] = 1,
        apply_atomic_energies_shift: Optional[bool] = True,
        atomic_energies_shift_list: Optional[List[float]] = None,
        device: Optional[str] = None,
        dtype: Optional['dtype'] = None,
    ):
        """
        Initialize the data subset loader
        
        Parameters:
        -----------
        train_batch_size: int
            Training dataloader batch size
        valid_batch_size: int
            Validation dataloader batch size
        test_batch_size:  int
            Test dataloader batch size
        num_workers: int, optional, default 1
            Number of data loader workers
        apply_atomic_energies_shift: bool, optional, default True
            Whether to apply atomic energies shift to reference energy provided
            by the data loader
        atomic_energies_shift_list: list(float), optional, default None
            Atom type specific energy shift terms to shift the system energies.
        device: str, optional, default global setting
            Device type for model variable allocation
        dtype: dtype object, optional, default global setting
            Model variables data type

        """

        # Check atomic energies shift list
        metadata = self.get_metadata()
        if apply_atomic_energies_shift and atomic_energies_shift_list is None:
            atomic_energies_shift_list = self.get_atomic_energies_shift()
        elif not apply_atomic_energies_shift:
            atomic_energies_shift_list = None

        # Check module variable parameters from configuration
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        # Prepare training, validation and test data loader
        self.train_dataloader = data.DataLoader(
            self.train_dataset,
            train_batch_size,
            True,
            num_workers,
            device,
            dtype,
            data_atomic_energies_shift=atomic_energies_shift_list)
        self.valid_dataloader = data.DataLoader(
            self.valid_dataset,
            valid_batch_size,
            False,
            num_workers,
            device,
            dtype,
            data_atomic_energies_shift=atomic_energies_shift_list)
        self.test_dataloader = data.DataLoader(
            self.test_dataset,
            test_batch_size,
            False,
            num_workers,
            device,
            dtype,
            data_atomic_energies_shift=atomic_energies_shift_list)

        # Prepare dataset and subset label to objects dictionary
        self.all_dataloader = {
            'train': self.train_dataloader,
            'training': self.train_dataloader,
            'valid': self.valid_dataloader,
            'validation': self.valid_dataloader,
            'test': self.test_dataloader,
            'testing': self.test_dataloader}

        return

    def get_property_scaling(
        self,
        data_label: Optional[str] = 'all',
        overwrite: Optional[bool] = False,
        property_atom_scaled: Optional[Dict[str, str]] = None,
    ) -> Dict[str, List[float]]:
        """
        Compute property statistics with average and standard deviation.

        Parameters
        ----------
        data_label: str, optional, default 'train'
            Dataset or subset label using for computing property scaling
            statistics.
        overwrite: bool, optional, default False
            If property statistics already available and up-to-date, recompute
            them. The up-to-date flag will be reset to False if any database
            manipulation is done.
        property_atom_scaled: dict(str, str), optional, default None
            Property statistics (key) will be scaled by the number of atoms
            per system and stored with new property label (item).
            e.g. {'energy': 'atomic_energies'}

        Return
        ------
        dict(str, list(float))
            Property statistics dictionary

        """

        # Get current metadata dictionary
        metadata = self.dataset.get_metadata()

        # Check atom scaled properties dictionary
        if property_atom_scaled is None:
            property_atom_scaled = {}

        # Check for property scaling results or Initialize 
        # scaling and shift parameter dictionary
        if (
            metadata.get('data_property_scaling_uptodate') is not None
            and metadata['data_property_scaling_uptodate']
            and metadata.get('data_property_scaling_label') is not None
            and metadata['data_property_scaling_label'] == data_label
            and not overwrite
        ):
            property_scaling = metadata.get('data_property_scaling')
        else:
            property_scaling = {}

        # Compute property statistics
        scaling_result = data.compute_property_scaling(
            self.get_dataset(data_label),
            metadata.get('load_properties'),
            property_scaling,
            property_atom_scaled)

        # Update property scaling dictionary
        for prop, result in scaling_result.items():
            property_scaling[prop] = result

        # Prepare header
        message = (
            f"Property statistics of '{data_label:s}' data "
            + f"of the database '{self.data_file[0]:s}'!\n"
            + f" {'Property Label':<17s} |"
            + f" {'Average':<17s} |"
            + f" {'Std. Deviation':<17s} |"
            + f" {'Unit':<17s}\n"
            + "-"*(20*4)
            + "\n")
        
        # Prepare property statistics information output
        for prop in metadata.get('load_properties'):
            
            # Get property unit
            if prop in self.data_unit_properties:
                unit = self.data_unit_properties[prop]
            else:
                unit = "None"

            # Add property statistics
            message += (
                f" {prop:<17s} |"
                + f" {property_scaling[prop][0]:>17.3e} |"
                + f" {property_scaling[prop][1]:>17.3e} |"
                + f" {unit:<17s}\n")
        
            if prop in property_atom_scaled:

                # Atom scaled property label
                atom_prop = property_atom_scaled[prop]

                # Add atom scaled property statistics
                message += (
                    f" {atom_prop:<17s} |"
                    + f" {property_scaling[atom_prop][0]:>17.3e} |"
                    + f" {property_scaling[atom_prop][1]:>17.3e} |"
                    + f" {unit:<17s}\n")
        
        # Print property statistics information
        self.logger.info(message)

        # Update property scaling
        if metadata.get('data_property_scaling') is None:
            metadata['data_property_scaling'] = property_scaling
        else:
            metadata['data_property_scaling'].update(property_scaling)
        metadata['data_property_scaling_uptodate'] = True
        metadata['data_property_scaling_label'] = data_label
        self.dataset.set_metadata(metadata)

        return property_scaling

    def get_atomic_energies_scaling(
        self,
        data_label: Optional[str] = 'all',
        overwrite: Optional[bool] = False,
    ) -> Dict[int, List[float]]:
        """
        Compute property statistics with average and standard deviation.

        Parameters
        ----------
        data_label: str, optional, default 'train'
            Dataset or subset label using for computing property scaling
            statistics.
        overwrite: bool, optional, default False
            If property statistics already available and up-to-date, recompute
            them. The up-to-date flag will be reset to False if any database
            manipulation is done.

        Return
        ------
        dict(int, list(float))
            Atomic energies scaling dictionary

        """

        # Get current metadata dictionary
        metadata = self.dataset.get_metadata()

        # Check for atomic energies scaling results or compute atom energies
        # scaling factor and shift term
        if (
            metadata.get('data_atomic_energies_scaling_uptodate') is not None
            and metadata['data_atomic_energies_scaling_uptodate']
            and metadata.get('data_atomic_energies_scaling_label') is not None
            and metadata['data_atomic_energies_scaling_label'] == data_label
            and not overwrite
        ):

            # Load stored atomic energies scaling
            atomic_energies_scaling_str = metadata.get(
                'data_atomic_energies_scaling')

            # Convert string keys to integer keys
            atomic_energies_scaling = {}
            for key, item in atomic_energies_scaling_str.items():
                atomic_energies_scaling[int(key)] = item

        else:

            # Get energy unit
            if 'energy' in self.data_unit_properties:
                energy_unit = self.data_unit_properties['energy']
            else:
                energy_unit = "None"

            atomic_energies_scaling, computation_message = (
                data.compute_atomic_energies_scaling(
                    self.get_dataset(data_label),
                    energy_unit)
                )

            # Prepare atomic energies statistics information output
            message = (
                f"Atomic energies statistics of {data_label:s} data "
                + f"of the database '{self.data_file[0]:s}'!\n")
            message += computation_message
            message += (
                f" {'Element':<17s} |"
                + f" {'Energy Shift':<17s} |"
                + f" {'Energy Scaling':<17s} |"
                + f" {'Unit':<17s}\n"
                + "-"*(20*4)
                + "\n")

            for atomic_number, scaling in (atomic_energies_scaling.items()):

                # Add atomic energies statistics
                message += (
                    f" {utils.chemical_symbols[atomic_number]:<17s} |"
                    + f" {scaling[0]:>17.3e} |"
                    + f" {scaling[1]:>17.3e} |"
                    + f" {energy_unit:<17s}\n")

            # Print atomic energies statistics information
            self.logger.info(message)

            # Update atomic energies scaling
            if metadata.get('data_atomic_energies_scaling') is None:
                metadata['data_atomic_energies_scaling'] = (
                    atomic_energies_scaling)
            else:
                metadata['data_atomic_energies_scaling'].update(
                    atomic_energies_scaling)
            metadata['data_atomic_energies_scaling_uptodate'] = True
            metadata['data_atomic_energies_scaling_label'] = data_label
            self.dataset.set_metadata(metadata)

        return atomic_energies_scaling

    def get_atomic_energies_shift(
        self,
        data_label: Optional[str] = 'all',
    ) -> List[float]:
        """
        Compute reference atomic energies shift to center system energies
        around zero.
        
        Parameters
        ----------
        data_label: str, optional, default 'training'
            Reference dataset ('all') or subset (e.g. 'training', 'validation')
            used for the atomic energies shift computation.

        Returns
        -------
        dict(int, float)
            Reference data atomic energies shift list

        """
        
        # Get atomic energies scaling guess
        if 'energy' in self.data_properties:
            data_atomic_energies_scaling = (
                self.get_atomic_energies_scaling(data_label=data_label))
        else:
            data_atomic_energies_scaling = {}

        # Prepare atomic energies shift list
        max_atomic_number = max([
            int(atomic_number)
            for atomic_number in data_atomic_energies_scaling.keys()])
        atomic_energies_shift = np.zeros(max_atomic_number + 1, dtype=float)
        for atomic_number in range(max_atomic_number + 1):
            if atomic_number in data_atomic_energies_scaling:
                atomic_energies_shift[atomic_number] = (
                    data_atomic_energies_scaling[atomic_number][0])

        return atomic_energies_shift

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
        if hasattr(self, 'all_datasets'):
            return ['train', 'valid', 'test']
        else:
            raise AttributeError(
                "Dataset and subsets were not initialized yet.")

    def get_dataset(self, label: str) -> Callable:
        """
        Return as specific DataSubSet object ('train', 'valid' or 'test')
        """
        if hasattr(self, 'all_datasets'):
            return self.all_datasets.get(label)
        else:
            raise AttributeError(
                "Dataset and subsets were not initialized yet.")

    def get_dataloader(self, label: str) -> Callable:
        """
        Return as specific DataLoader object ('train', 'valid' or 'test')
        """
        if hasattr(self, 'all_dataloader'):
            return self.all_dataloader.get(label)
        else:
            raise AttributeError(
                "Dataloaders were not initialized yet.")

    def get_train(self, idx: int) -> Dict:
        """
        Get Training DataSubSet entry idx
        """
        return self.train_dataset.get(idx)

    def get_valid(self, idx: int) -> Dict:
        """
        Get Validation DataSubSet entry idx
        """
        return self.valid_dataset.get(idx)

    def get_test(self, idx: int) -> Dict:
        """
        Get Test DataSubSet entry idx
        """
        return self.test_dataset.get(idx)

    def get_info(self) -> Dict[str, Any]:
        """
        Return data information
        """

        return {
            'data_file': self.data_file,
            'data_file_format': self.data_file_format,
            'data_properties': self.data_properties,
            'data_unit_properties': self.data_unit_properties,
            'data_num_train': self.data_num_train,
            'data_num_valid': self.data_num_valid,
            'data_num_test': self.data_num_test,
            'data_overwrite': self.data_overwrite,
            }
