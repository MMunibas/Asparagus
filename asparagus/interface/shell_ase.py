# ASE calculator class modifying and executing a template shell file(s) which
# compute atoms properties and provide them as a .json or .npy file.
import os
import json
import subprocess
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import ase
from ase.calculators.calculator import Calculator, FileIOCalculator, Parameters

from .. import utils


class TagReplacement:
    """
    Class for pre-defined functions that return input strings with
    respect to the respective atoms object.
    """
    
    def __init__(
        self,
        calculator: ase.calculators.calculator.Calculator,
        **kwargs
    ):
        """
        Initialize class object.
        """

        # Assign calculator class
        self.calculator = calculator
        
        # Initialize function library
        self.functions = {
            '$xyz': self.get_xyz,
            '$charge': self.get_charge,
            '$mult': self.get_multiplicity,
            '$multiplicity': self.get_multiplicity,
            '$spin2': self.get_spin_times_2,
            '$dir': self.calculator.directory,
            '$directory': self.calculator.directory,
            }

        return
    
    def __getitem__(
        self,
        item,
    ):
        """
        Get replacement function from function dictionary.
        """
        return self.functions.get(item)

    def get_xyz(
        self,
        atoms: ase.Atoms,
        parameters: Optional[Dict[str, Any]] = {},
    ):
        """
        Return lines of atoms element symbols and its Cartesian coordinates
        """
        
        out = ""
        for ia, atom in enumerate(atoms):
            if ia:
                out += "\n"
            out += f"{atom.symbol:<3s} "
            for i in range(3):
                out += f"{atom.position[i]: 12.8f} "

        return out

    def get_charge(
        self,
        atoms: ase.Atoms,
        parameters: Optional[Dict[str, Any]] = {},
    ):
        """
        Return system charge as string of an integer.
        """
        if 'charge' in parameters:
            out = f"{parameters['charge']:d}"
        elif 'charge' in self.calculator.parameters:
            out = f"{self.calculator.parameters['charge']:d}"
        else:
            out = "0"

        return out

    def get_multiplicity(
        self,
        atoms: ase.Atoms,
        parameters: Optional[Dict[str, Any]] = {},
    ):
        """
        Return system multiplicity as string of an integer.
        """
        if 'multiplicity' in parameters:
            out = f"{parameters['multiplicity']:d}"
        elif 'multiplicity' in self.calculator.parameters:
            out = f"{self.calculator.parameters['multiplicity']:d}"
        else:
            out = "1"

        return out

    def get_spin_times_2(
        self,
        atoms: ase.Atoms,
        parameters: Optional[Dict[str, Any]] = {},
    ):
        """
        Return system multiplicity in form of 2*S = 2*(n*1/2) equal the number
        of unpaired spins n as string of an integer.
        """
        if 'multiplicity' in parameters:
            out = f"{parameters['multiplicity'] - 1:d}"
        elif 'multiplicity' in self.calculator.parameters:
            out = f"{self.calculator.parameters['multiplicity'] - 1:d}"
        else:
            out = "0"

        return out

    def get_directory(
        self,
        atoms: ase.Atoms,
        parameters: Optional[Dict[str, Any]] = {},
    ):
        """
        Return system charge as string of an integer.
        """
        if 'directory' in parameters:
            out = f"{parameters['directory']:s}"
        elif 'directory' in self.calculator:
            out = f"{self.calculator.directory:s}"
        else:
            out = "0"

        return out
    

class ShellCalculator(FileIOCalculator):
    """
    ASE Calculator class modifying and executing a template shell file which
    computes atoms properties and provide the results as compatible ASE format.

    Parameters
    ----------
    files: (str, list(str))
        Template input files to copy into working directory and regarding for
        tag replacement.
    files_replace: dict(str, any) or list(dict(str, any))
        Template file tag replacement commands in the form of a dictionary or
        a list of dictionaries. The keys of the dictionary is the tag in the
        template files which will be replaced by the respective item or
        its output if the item is a callable function. 
        
        If one dictionary is defined, the instructions are applied to all
        template files and if a list of dictionaries is given, each dictionary
        is applied on the template file of the same list index.
        
        The item of the dictionaries can be either a self defined callable
        function in form of 'func(ase.Atoms, **kwargs)' that returns a single
        string, a fix string itself or one of the following strings that will
        order the execution of one the pre-defined functions with the
        respective outputs:
            item        output
            '$xyz'      Lines of element symbols and Cartesian coordinates
            '$charge'   Integer of the ase.Atoms system charge
            '$dir'      Path of the working directory
            ...
    execute_file: str, optional, default files[0]
        Template file, which will be executed by the shell command.
        If not defined, the (first) template file in 'files' will be assumed
        as executable.
    result_properties: (str, list(str)), optional, default ['energy']
        List of system properties of the respective atoms object which are
        expected to be stored in the result file.
    result_file: str, optional, default 'result.json'
        Result file path where the calculation results are stored.
    result_file_path: str, optional, default 'json'
        Result file format to define the way of reading the results.
    atoms: ase.Atoms, optional, default None
        Optional Atoms object to which the calculator will be
        attached.  When restarting, atoms will get its positions and
        unit-cell updated from file.
    charge: int, optional, default 0
        Default atoms charge
    multiplicity: int, optional, default 1
        Default system spin multiplicity (2*S + 1)
    command: str, optional, default 'bash'
        Command to start the calculation.
    label: str, optional, default 'shell'
        Name used for all files.  Not supported by all calculators.
        May contain a directory, but please use the directory parameter
        for that instead.
        Asparagus: May be used as 'calculator_tag'.
    directory: str or PurePath
        Working directory in which to read and write files and
        perform calculations.

    
    Results
    -------

    """

    # Default parameters dictionary for initialization
    default_parameters = dict(
        files=[],
        files_replace={},
        execute_file=None,
        result_properties=['energy'],
        result_file='results.json',
        result_file_format='json',
        command='bash')

    # Discard any results if parameters were changed
    discard_results_on_any_change = True

    def __init__(
        self,
        files: Union[str, List[str]],
        files_replace: Union[List[Dict[str, Any]], Dict[str, Any]],
        execute_file: Optional[str] = None,
        result_properties: Optional[Union[str, List[str]]] = ['energy'],
        result_file: Optional[str] = 'results.json',
        result_file_format: Optional[str] = 'json',
        atoms: Optional[ase.Atoms] = None,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
        command: Optional[str] = 'bash',
        restart: Optional[bool] = None,
        label: Optional[str] = 'shell',
        directory: Optional[str] = 'calc',
        **kwargs
    ):
        """
        Initialize Shell Calculator class.
        """

        # Valid result file formats
        self._valid_result_file_format = {
            'npz': self.load_results_npz,
            'json': self.load_results_json,
            }

        # Initialize parent class
        FileIOCalculator.__init__(
            self,
            atoms=atoms,
            label=label,
            directory=directory,
            restart=restart,
            command=command,
            **kwargs)

        # Set calculator parameters
        self.set_shell(
            files,
            files_replace,
            execute_file,
            result_properties,
            result_file,
            result_file_format,
            charge,
            multiplicity,
            **kwargs)
        
        # Initialize implemented properties list
        self.implemented_properties = self.result_properties
        
        # Initialize tag replacement class
        self.tag_replacer = TagReplacement(self)
        
        # Initialize a point charge object
        self.pcpot = None

    def __str__(self):
        return f"ShellCalculator {self.execute_file:s}"

    def set_shell(
        self,
        files: Union[str, List[str]],
        files_replace: Union[List[Dict[str, Any]], Dict[str, Any]],
        execute_file: str,
        result_properties: Union[str, List[str]],
        result_file: str,
        result_file_format: str,
        charge: int,
        multiplicity: int,
        **kwargs,
    ):
        """
        Check and set calculator parameters.
        """
        
        # Check template files and executable file
        files, files_replace, execute_file = self.set_input_files(
            files,
            files_replace, 
            execute_file)

        # Check result files and properties
        result_properties, result_file, result_file_format = (
            self.set_result_files(
                result_properties,
                result_file, 
                result_file_format)
            )

        # Check default system properties
        charge, multiplicity = self.set_system_properties(
            charge,
            multiplicity)
        
        # Set parameters
        self.files = files
        self.files_replace = files_replace
        self.execute_file = execute_file
        self.result_properties = result_properties
        self.result_file = result_file
        self.result_file_format = result_file_format
        self.charge = charge
        self.multiplicity = multiplicity
        
        # Run parent file setup
        changed_parameters = FileIOCalculator.set(
            self, 
            files=self.files,
            files_replace=self.files_replace,
            execute_file=self.execute_file,
            result_properties=self.result_properties,
            result_file=self.result_file,
            result_file_format=self.result_file_format,
            charge=self.charge,
            multiplicity=self.multiplicity,
            **kwargs)
        if changed_parameters:
            self.reset()

        return

    def set_input_files(
        self,
        files: Union[str, List[str]],
        files_replace: Union[List[Dict[str, Any]], Dict[str, Any]],
        execute_file: str,
    ):
        """
        Check template files and executable file.
        """
        
        # Check if template file exists
        if utils.is_string(files):
            files = [files]
        elif not utils.is_array_like(files):
            raise SyntaxError(
                "Template files is neither a valid file path (str) or "
                "a list of file paths!")
        for file_i in files:
            if not utils.is_string(file_i):
                raise SyntaxError(
                    f"Template file path '{file_i}' is not a valid file path!")
            if not os.path.exists(file_i):
                raise SyntaxError(
                    f"Template file '{file_i:s}' does not exists!")

        # Check for directory conflict
        for file_i in files:
            if (
                os.path.abspath(self.directory)
                == os.path.abspath(os.path.split(file_i)[0])
            ):
                raise SyntaxError(
                    f"Working directory '{os.path.abspath(self.directory):s}' "
                    + "for the calculation is the same path as for at least "
                    + "one of the template files "
                    + f"'{os.path.abspath(file_i):s}'!\n"
                    + "Avoid such conflict and the potential overwriting of "
                    + "any template file.")

        # Check executable file
        if execute_file is None:
            execute_file = os.path.split(file_i)[1]
        elif utils.is_string(execute_file):
            if (not os.path.exists(execute_file) and not any([
                execute_file in os.path.split(file_i)[1] for file_i in files])
            ):
                raise SyntaxError(
                    f"Executable file '{execute_file:s}' does not exists!")
            else:
                execute_file = os.path.split(execute_file)[1]
        else:
            raise SyntaxError(
                f"Executable file '{execute_file}' is not a valid file path!")

        # Check if template file replacement input
        if utils.is_dictionary(files_replace):
            files_replace = [files_replace]
        elif not utils.is_array_like(files_replace):
            raise SyntaxError(
                "Template file replacement input is not of type dictionary "
                + "or a list of dictionaries!")
        if len(files_replace) == 1:
            files_replace = files_replace*len(files)
        elif not len(files_replace) == len(files):
            raise SyntaxError(
                "Mismatch between number of template file replacement " 
                + f" dictionaries ({len(files_replace)}) and numer of "
                + f"template files ({len(files)})!")
        for files_replace_i in files_replace:
            if not utils.is_dictionary(files_replace_i):
                raise SyntaxError(
                    f"Template file replacement input '{files_replace_i}' "
                    + "is not of type dictionary!")
            
        return files, files_replace, execute_file

    def set_result_files(
        self,
        result_properties: Union[str, List[str]],
        result_file: str,
        result_file_format: str,
    ):
        """
        Check result files and properties.
        """

        # Check result properties
        if utils.is_string(result_properties):
            result_properties = [result_properties]
        elif not utils.is_array_like(result_properties):
            raise SyntaxError(
                "System properties 'result_properties' is neither a property "
                "label (str) or a list of property labels!")
        for prop_i in result_properties:
            if not utils.is_string(prop_i):
                raise SyntaxError(
                    f"System property '{prop_i}' is not a valid property "
                    + "label (str)!")

        # Check result file path and format
        if not utils.is_string(result_file):
            raise SyntaxError(
                f"Result file '{result_file}' is not a valid file path!")
        if result_file_format[0] == '.':
            result_file_format = result_file_format[1:]
        if result_file_format.lower() not in self._valid_result_file_format:
            raise SyntaxError(
                f"Result file format '{result_file_format}' is not a valid "
                + " file format!\nValid options are:\n  "
                + str(list(result_file_format.keys())))

        return result_properties, result_file, result_file_format
        
    def set_system_properties(
        self,
        charge: int,
        multiplicity: int,
    ):
        """
        Check system properties
        """
        
        # Check charge
        if charge is not None and not utils.is_numeric(charge):
            raise SyntaxError(
                "System charge input is not a numeric input!")

        # Check multiplicity
        if multiplicity is not None and not utils.is_numeric(multiplicity):
            raise SyntaxError(
                "System spin multiplicity input is not a numeric input!")

        return charge, multiplicity

    def write_input(
        self,
        atoms: ase.Atoms,
        properties: Optional[List[str]] = None, 
        system_changes: Optional[List[str]] = None,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
    ):
        """
        Write input files by copying template files to the working directory
        while applying the replacement tasks.
        
        Parameters
        ----------
        atoms: ase.Atoms
            Reference atoms object
        properties: list(str), optional, default None
            List of to computing system properties
        system_changes: list(str), optional, default None
            Detailed list of changed system parameters with regard to 
            previously computed system.
        """
        
        # Execute parent class 'write_input' function that checks and
        # eventually create the working directory.
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        
        # Get parameter set
        parameters = Parameters(self.parameters.copy())
        
        # Set system charge and spin multiplicity
        if charge is not None:
            charge = self.set_system_properties(charge, multiplicity)[0]
        elif self.charge is None and 'charge' in atoms.info:
            charge = self.set_system_properties(atoms.info['charge'], None)[0]
        elif self.charge is None:
            charge = 0
        else:
            charge = self.charge
        if multiplicity is None:
            multiplicity = self.multiplicity
        elif self.multiplicity is None and 'multiplicity' in atoms.info:
            multiplicity = self.set_system_properties(
                None, atoms.info['multiplicity'])[1]
        elif self.multiplicity is None:
            multiplicity = 1
        else:
            multiplicity = self.multiplicity
        parameters['charge'] = charge
        parameters['multiplicity'] = multiplicity
        
        # Write calculator options to file
        parameters.write(self.label + '.ase')
        parameters['label'] = self.label
        
        # Prepare and write input files
        self.write_shell_input(
            atoms,
            parameters,
            self.files,
            self.files_replace)

        return

    def write_shell_input(
        self,
        atoms: ase.Atoms,
        parameters: Dict[str, Any],
        files: List[str], 
        files_replace: List[Dict[str, Any]],
        **kwargs
    ):
        """
        Prepare and write shell input files
        """
    
        # Iterate over template files
        for ifile, file_i in enumerate(files):
            
            # Get tag replacement instructions
            file_replace = files_replace[ifile]
            
            # Read template file
            with open(file_i, 'r') as f:
                flines = f.read()
            
            # Replace tags
            for tag, item in file_replace.items():
                # Check replacement
                if utils.is_string(item):
                    if item in self.tag_replacer.functions:
                        item_str = self.tag_replacer[item](
                            atoms, parameters=parameters)
                    else:
                        item_str = item                    
                elif utils.is_callable(item):
                    item_str = item(atoms)
                else:
                    raise SyntaxError(
                        f"Template file replacement item for label '{tag:s}' "
                        + "is neither a string nor a callable function!")
                # Replace label tag with item string
                flines = flines.replace(
                    tag, 
                    item_str)

            # Write working file
            wfile_i = os.path.split(file_i)[1]
            with open(os.path.join(self.directory, wfile_i), 'w') as f:
                f.write(flines)

        return

    def calculate(
        self,
        atoms: Optional[ase.Atoms] = None,
        properties: Optional[List[str]] = ['energy'],
        system_changes: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Execute calculation and read results
        """
        
        # Prepare calculation by execution parent class function
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Write input files
        self.write_input(
            self.atoms, 
            properties, 
            system_changes,
            **kwargs)
        
        # Check shell command 
        if self.command is None:
            command = 'bash'
        else:
            command = self.command

        # Check executable file
        if self.execute_file is None:
            execute_file = os.path.split(self.files[0])[1]
        else:
            execute_file = self.execute_file

        # Execute command with executable file
        proc = subprocess.Popen([command, execute_file], cwd=self.directory)
        errorcode = proc.wait()
        if errorcode:
            msg = (
                f"Calculator '{self.label:s}' failed with command "
                + f"'{command:s} {execute_file:s}' failed in "
                + f"'{self.directory:}' with error code '{errorcode}'!")
            raise OSError(msg)
        
        self.read_results()

        return

    def read_results(
        self,
    ):
        """
        Read results from the defined result file
        """
        
        # Read results from file
        self._valid_result_file_format[self.result_file_format](
            os.path.join(self.directory, self.result_file))
        
        return

    def load_results_npz(self):
        raise NotImplementedError

    def load_results_json(
        self,
        result_file: str,
    ):
        
        # Open result file
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)
        else:
            results = {}

        # Convert lists to np.ndarrays
        self.results = {}
        for prop_i, result in results.items():
            self.results[prop_i] = np.array(result, dtype=float)

        return
