import os
import json
import logging
from typing import Optional, List, Dict, Tuple, Union, Callable, Iterator, Any

from .. import utils
from .. import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================================
# Configuration Functions
# ======================================

def get_config(
    config: Optional[Union[str, dict, object]] = None,
    config_file: Optional[str] = None,
    config_from: Optional[Union[str, object]] = None,
    **kwargs,
) -> Callable:
    """
    Initialize Configuration class object. If 'config' input is already
    a class object, return itself.

    Parameters
    ----------
    config: (str, dict, object), optional, default None
        Either the path to json file (str), dictionary (dict) or
        configuration object of the same class (object) containing
        global model parameters
    config_file: str, optional, default see settings.default['config_file']
        Store global parameter configuration in json file of given path.
    config_from: (object, str), optional, default None
        Location, defined as class instance or string, from where the new
        configuration parameter dictionary comes from.
    kwargs: dict, optional, default {}
        Keyword arguments for configuration parameter which are added to
        'config' or overwrite 'config' content.

    Returns
    -------
    Callable
        Configuration parameter object
    """

    # If 'config' being a config class object
    if utils.is_callable(config):

        # If config file is defined - update config file path
        config.set_config_file = (config_file)
        
        # Update configuration with keyword arguments
        config.update(
            kwargs,
            config_from=config_from)

        return config

    # Otherwise initialize Configuration class object
    else:

        return settings.Configuration(
            config=config,
            config_file=config_file,
            config_from=config_from,
            **kwargs)

    return

# ======================================
# Configuration Class
# ======================================

class Configuration():
    """
    Global configuration object that contains all parameter about the
    model and training procedure.

    Parameters
    ----------
    config: (str, dict), optional, default None
        Either the path to json file (str) or dictionary (dict) containing
        global model parameters
    config_file: str, optional, default see settings.default['config_file']
        Store global parameter configuration in json file of given path.
    config_from: (object, str), optional, default None
        Location, defined as class instance or string, from where the new
        configuration parameter dictionary comes from.
    kwargs: dict, optional, default {}
        Keyword arguments for configuration parameter which are added to
        'config' or overwrite 'config' content.

    Returns
    -------
    Callable
        Configuration parameter object
    """

    def __init__(
        self,
        config: Optional[Union[str, dict]] = None,
        config_file: Optional[str] = None,
        config_from: Optional[Union[str, object]] = None,
        **kwargs,
    ):
        """
        Initialize config object.
        """

        # Initialize class config dictionary
        self.config_dict = {}
        self.config_indent = 2

        # Check configuration source input
        # Case 1: Neither defined - Get config dict at default file path
        if config is None and config_file is None:
            self.config_file = settings._default_args.get('config_file')
            self.config_dict = self.read(self.config_file)
        # Case 2: Only config file is defined - Read config dict from file path
        elif config is None:
            if utils.is_string(config_file):
                self.config_file = config_file
                self.config_dict = self.read(self.config_file)
            else:
                raise SyntaxError(
                    "Config input 'config_file' is not a string of "
                    + "a valid file path!")
        # Case 3: Only config is defined - Check type
        elif config_file is None:
            # If config is actually a file path - use as file path like case 2
            if utils.is_string(config):
                self.config_file = config
                self.config_dict = self.read(self.config_file)
            # If config is a dictionary - assign and read file path
            elif utils.is_dictionary(config):
                self.config_dict = config
                if self.config_dict.get('config_file') is None:
                    config_file = settings._default_args.get('config_file')
                else:
                    config_file = self.config_dict.get('config_file')
                self.config_file = config_file
            else:
                raise SyntaxError(
                    "Config input 'config' is neither a dictionary nor a "
                    + "string of a  valid file path!")
        # Case 4: Both are defined - Update config dict from config file with 
        # config input
        else:
            # Read config dict from file path and assign as self.config_dict
            if utils.is_string(config_file):
                self.config_file = config_file
                self.config_dict = self.read(self.config_file)
            else:
                raise SyntaxError(
                    "Config input 'config_file' is not a string of "
                    + "a valid file path!")
            # If config is actually a file path - read config dict and update
            # self.config_dict
            if utils.is_string(config):
                self.update(self.read(config))
            # If config is a dictionary - update self.config_dict
            elif utils.is_dictionary(config):
                self.update(config)
            else:
                raise SyntaxError(
                    "Config input 'config' is neither a dictionary nor a "
                    + "string of a  valid file path!")
        # In all cases, config_dict and config_file are class variables and 
        # defined as dictionary and string.
        
        # Set config file path to dictionary
        self.set_config_file(self.config_file)

        # Generate, eventually, the directory for the config file
        config_dir = os.path.dirname(self.config_file)
        if not os.path.isdir(config_dir) and len(config_dir):
            os.makedirs(os.path.dirname(self.config_file))

        # Update configuration dictionary with keyword arguments
        if len(kwargs):
            self.update(
                kwargs,
                config_from=config_from,
                )
        
        # Save current configuration dictionary to file
        self.dump()

        # Adopt default settings arguments and their valid dtypes
        self.default_args = settings._default_args
        self.dtypes_args = settings._dtypes_args

    def __str__(self):
        msg = f"Config file in '{self.config_file:s}':\n"
        for arg, item in self.config_dict.items():
            msg += f"  '{arg:s}': {str(item):s}\n"
        return msg 

    def __getitem__(self, args):
        return self.config_dict.get(args)

    def __setitem__(self, arg, item):
        self.config_dict[arg] = item
        self.dump()

    def __contains__(self, arg):
        return arg in self.config_dict.keys()

    def __call__(self, args):
        return self.config_dict.get(args)

    def items(self):
        for key, item in self.config_dict.items():
            yield key, item

    def get(self, args):
        if utils.is_array_like(args):
            return [self.config_dict.get(arg) for arg in args]
        else:
            return self.config_dict.get(args)

    def keys(self):
        return self.config_dict.keys()

    def set_config_file(
        self,
        config_file: str,
    ):
        
        # Check input
        if utils.is_string(config_file):
            self.config_file = config_file
        else:
            raise SyntaxError(
                "Config input 'config_file' is not a string of "
                + "a valid file path!")
        
        # Set config file path to dictionary
        if self.config_dict.get('config_file') is None:
            self.config_dict['config_file'] = self.config_file
            logger.info(
                "INFO:\nConfiguration file path set to "
                + f"'{self.config_file:s}'!\n")
        else:
            if self.config_dict.get('config_file') != self.config_file:
                logger.info(
                    "INFO:\nConfiguration file path will be changed from "
                    + f"'{self.config_dict.get('config_file'):s}' to "
                    + f"'{self.config_file:s}'!\n")
                self.config_dict['config_file'] = self.config_file
        
        return
    
    def read(
        self, 
        config_file: str,
    ) -> Dict[str, Any]:

        # Read json file
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        else:
            config_dict = {}

        # Check for convertible parameter keys and convert 
        for key, item in config_dict.items():
            if self.is_convertible(key):
                config_dict[key] = self.convert(key, item, 'read')

        return config_dict

    def update(
        self,
        config_new: Union[str, dict, object],
        config_from: Optional[Union[object, str]] = None,
        overwrite: Optional[bool] = True,
        verbose: Optional[bool] = True,
    ):
        """
        Update configuration dictionary.

        Parameters
        ----------

        config_new: (str, dict, object)
            Either the path to json file (str), dictionary (dict) or
            configuration object of the same class (object) containing
            new model parameters.
        config_from: (object, str), optional, default None
            Location, defined as class instance or string, from where the new
            configuration parameter dictionary comes from.
        overwrite: bool, optional, default True
            If True, 'config_new' input will be added and eventually overwrite
            existing entries in the configuration dictionary.
            If False, each input in 'config_new' will be only added if it is
            not already defined in the configuration dictionary.
        verbose: bool, optional, default True
            For conflicting entries between 'config_new' and current
            configuration dictionary, return further information.

        """

        # Check config_new input
        if utils.is_string(config_new):
            config_new = self.read(config_new)
        elif utils.is_dictionary(config_new):
            pass
        elif utils.is_callable(config_new):
            config_new = config_new.get_dictionary()
        else:
            raise ValueError(
                "Input 'config_new' is not of valid data type!\n" +
                "Data type 'dict', 'str' or a config class object " +
                f"is expected but '{type(config_new)}' is given.")

        # Return if update dictionary is empty
        if not len(config_new):
            logger.info("INFO:\nEmpty update configuration dictionary!\n")
            return

        # Show update information
        msg = (
            f"Parameter update in '{self.config_file}'\n")
        if config_from is not None:
            msg += f"  (called from '{config_from}')\n"
        if overwrite:
            msg += "  (overwrite conflicts)\n"
        else:
            msg += "  (ignore conflicts)\n"
        
        # Iterate over new configuration dictionary
        n_all, n_add, n_equal, n_overwrite = 0, 0, 0, 0
        for key, item in config_new.items():

            # Skip if parameter value is None
            if item is None:
                continue
            else:
                n_all += 1

            # Check for conflicting keyword
            conflict = key in self.config_dict.keys()

            # For conflicts, check for changed parameter
            if conflict:
                equal = str(item) == str( self.config_dict[key])
                if equal:
                    n_equal += 1

            # Add or update parameter
            if conflict and overwrite and not equal:

                if self.is_convertible(key):
                    self.config_dict[key] = self.convert(
                        key, config_new.get(key), 'read')
                else:
                    self.config_dict[key] = config_new.get(key)
                n_overwrite += 1
                if verbose:
                    msg += f"Overwrite parameter '{key}'.\n"

            elif conflict and not equal:

                if verbose:
                    msg += f"Ignore parameter '{key}'.\n"

            elif not conflict:

                if self.is_convertible(key):
                    self.config_dict[key] = self.convert(
                        key, config_new.get(key), 'read')
                else:
                    self.config_dict[key] = config_new.get(key)
                n_add += 1
                if verbose:
                    msg += f"Adding parameter '{key}'.\n"

        # Add numbers
        msg += (
            f"{n_all:d} new parameter, {n_add:d} added, "
            + f"{n_equal:d} equal, {n_overwrite:d} overwritten\n")
        # Show additional information output
        if verbose:
            logger.info("INFO:\n" + msg)

        # Store changes in file
        self.dump()

        return

    def dump(
        self,
        config_file: Optional[str] = None
    ):
        """
        Save configuration dictionary to json file

        Parameters
        ----------
        config_file: str, optional, default None
            Dump current config dictionary in this file path.

        """

        # Convert config dictionary to json compatible dictionary
        config_dump = self.make_dumpable(self.config_dict)

        # Dumb converted config dictionary
        if config_file is None:
            config_file = self.config_file
        with open(config_file, 'w') as f:
            json.dump(
                config_dump, f, 
                indent=self.config_indent,
                default=str)
        
        return

    def make_dumpable(
        self,
        config_source: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Convert config items to json compatible dictionary
        """
        
        # Initialize dictionary with JSON compatible parameter types
        config_dump = {}

        # Iterate over configuration parameters
        for key, item in config_source.items():
        
            # Skip callable class objects
            if utils.is_callable(item):
                continue
            
            # Convert numeric values to integer or float
            if utils.is_integer(item):
                config_dump[key] = int(item)
            elif utils.is_numeric(item):
                config_dump[key] = float(item)
            # Also store dictionaries,
            elif utils.is_dictionary(item):
                config_dump[key] = self.make_dumpable(item)
            # strings or bools
            elif utils.is_string(item) or utils.is_bool(item):
                config_dump[key] = item
            # and converted arrays as python lists,
            # but nothing else which might be to fancy
            elif utils.is_array_like(item):
                config_dump[key] = list(item)
            elif self.is_convertible(key):
                config_dump[key] = self.convert(key, item, 'dump')
            else:
                continue

        return config_dump

    def check(
        self,
        check_default: Optional[Dict] = None,
        check_dtype: Optional[Dict] = None,
    ):
        """
        Check configuration parameter for correct data type and, eventually,
        set default values for parameters with entry None.

        Parameters
        ----------
        check_default: dict, optional, default None
            Default argument parameter dictionary.
        check_dtype: dict, optional, default None
            Default argument data type dictionary.
        """

        for arg, item in self.config_dict.items():

            # Check if input parameter is None, if so take default value
            if check_default is not None and item is None:
                if arg in check_default:
                    item = check_default[arg]
                    self[arg] = item

            # Check datatype of defined arguments
            if check_dtype is not None and arg in check_dtype:
                _ = utils.check_input_dtype(
                    arg, item, check_dtype, raise_error=True)

        # Save successfully checked configuration
        self.dump()
        
        return

    def set(
        self,
        instance: Optional[object] = None,
        argitems: Optional[Iterator] = None,
        argsskip: Optional[List[str]] = None,
        check_default: Optional[Dict] = None,
        check_dtype: Optional[Dict] = None,
    )  -> Dict[str, Any]:
        """
        Iterate over arg, item pair, eventually check for default and dtype, 
        and set as class variable of instance

        Parameters:
        -----------
        instance: object, optional, default None
            Class instance to set arg, item pair as class variable. If None,
            skip.
        argitems: iterator, optional, default None
            Iterator for arg, item pairs. If None, skip.
        argskipt: list(str), optional, default None
            List of arguments to skip. 
        check_default: dict, optional, default None
            Default argument parameter dictionary.
        check_dtype: dict, optional, default None
            Default argument data type dictionary.
        
        Return:
        -------
        dict[str, any]
            Updated config dictionary
        """

        # Return empty dictionary if no arg, item pair iterator is defined
        if argitems is None:
            return {}
        else:
            config_dict_update = {}
        
        # Check arguments to skip
        default_argsskip = [
            'self', 'config', 'config_file', 'kwargs', '__class__']
        if argsskip is None:
            argsskip = default_argsskip
        else:
            argsskip = default_argsskip + list(argsskip)
        argsskip.append('default_args')
        
        # Iterate over arg, item pairs
        for arg, item in argitems.items():

            # Skip exceptions
            if arg in argsskip:
                continue

            # If item is None, take from class config
            if item is None:
                item = self.get(arg)
            
            # Check if input parameter is None, if so take default value
            if check_default is not None and item is None:
                if arg in check_default:
                    item = check_default[arg]

            # Check datatype of defined arguments
            if check_dtype is not None and arg in check_dtype:
                _ = utils.check_input_dtype(
                    arg, item, check_dtype, raise_error=True)
            
            # Append arg, item pair to update dictionary
            config_dict_update[arg] = item

            # Set item as class parameter arg to instance
            if instance is not None:
                setattr(instance, arg, item)
        
        return config_dict_update

    def get_file_path(self):
        return self.config_file

    def get_dictionary(self):
        return self.config_dict

    def conversion_dict(self):
        """
        Generate conversion dictionary.
        
        """
        
        self.convertible_dict = {
            'dtype': self.convert_dtype
            }
        
        return
    
    def is_convertible(self, key):
        """
        Check if parameter 'key' is covert in the convertible dictionary.

        Parameters
        ----------
        key: str
            Parameter name

        """
        
        # Check if convertible dictionary is already initialized
        if not hasattr(self, 'convertible_dict'):
            self.conversion_dict()

        # Look for parameter in conversion dictionary
        return key in self.convertible_dict
    
    def convert(self, key, arg, operation):
        """
        Convert argument 'arg' of parameter 'key' between json compatible
        format and internal type.
        
        Parameters
        ----------
        key: str
            Parameter name
        arg: Any
            Parameter value
        operation: str
            Convert direction such as 'dump' (internal -> json) or 'read'
            (json -> internal).
        
        """
        
        # Check if convertible dictionary is already initialized
        if hasattr(self, 'convertible_dict'):
            self.conversion_dict()

        # Provide conversion result
        return self.convertible_dict[key](arg, operation)
    
    def convert_dtype(self, arg, operation):
        """
        Convert data type to data label

        """
        if operation == 'dump':
            for dlabel, dtype in settings._dtype_library.items():
                if arg is dtype:
                    return dlabel
            return None
        elif operation == 'read':
            for dlabel, dtype in settings._dtype_library.items():
                if arg == dlabel:
                    return dtype
            return None
        else:
            return None
        
        
        
        
        
