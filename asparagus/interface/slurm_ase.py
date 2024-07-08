# ASE calculator class modifying and executing a template shell file(s) which
# compute atoms properties and provide them as a .json or .npy file.
import os
import json
import time
import subprocess
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import ase
from ase.calculators.calculator import Calculator, FileIOCalculator, Parameters

from .shell_ase import ShellCalculator, TagReplacement

from .. import utils


class SlurmCalculator(ShellCalculator):
    """
    ASE Calculator class modifying and executing a template slurm submission
    file which computes atoms properties and provide the results as compatible
    ASE format.

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
        Template slurm submission file, which will be executed by the shell 
        command. If not defined, the (first) template file in 'files' will be
        assumed as executable.
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
    scan_interval: int, optional, default 1
        Scan interval checking for completeness of the submitted slurm job
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
        command='sbatch',
        scan_interval=1)

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
        charge: Optional[int] = 0,
        multiplicity: Optional[int] = 1,
        command: Optional[str] = 'sbatch',
        scan_interval: Optional[int] = 1,
        scan_catch_id: Optional[callable] = None,
        scan_check_id: Optional[callable] = None,
        restart: Optional[bool] = None,
        label: Optional[str] = 'slurm',
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
        ShellCalculator.__init__(
            self,
            files=files,
            files_replace=files_replace,
            execute_file=execute_file,
            result_properties=result_properties,
            result_file=result_file,
            result_file_format=result_file_format,
            atoms=atoms,
            charge=charge,
            multiplicity=multiplicity,
            command=command,
            restart=restart,
            label=label,
            directory=directory,
            **kwargs)

        # Assign additional class parameters
        if utils.is_numeric(scan_interval):
            self.scan_interval = scan_interval
        else:
            raise SyntaxError(
                "Submitted job scan interval 'scan_interval' is not a "
                "numeric value!")
        
        if scan_catch_id is None:
            self.scan_catch_id = None
        elif utils.is_callable(scan_catch_id):
            self.scan_catch_id = scan_catch_id
        else:
            raise SyntaxError(
                "Submitted job id catch function 'scan_catch_id' is not a "
                "callable function!")
        
        if scan_check_id is None:
            self.scan_check_id = None
        elif utils.is_callable(scan_check_id):
            self.scan_check_id = scan_check_id
        else:
            raise SyntaxError(
                "Submitted job id check function 'scan_check_id' is not a "
                "callable function!")

    def __str__(self):
        return f"SlurmCalculator {self.execute_file:s}"

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
            command = 'sbatch'
        else:
            command = self.command

        # Check executable file
        if self.execute_file is None:
            execute_file = os.path.split(self.files[0])[1]
        else:
            execute_file = self.execute_file

        # Execute command with executable file
        proc = subprocess.run(
            [command, execute_file],
            cwd=self.directory,
            capture_output=True)
        
        # Get slurm id
        if self.scan_catch_id is None:
            slurm_id = int(proc.stdout.decode().split()[-1])
        else:
            slurm_id = self.scan_catch_id(proc.stdout.decode())

        # Check for job completeness
        done = False
        while not done:
            
            # Get and check task id with active task ids
            if self.scan_check_id is None:
                proc = subprocess.run(
                    ['squeue', '-u', os.environ['USER']],
                    capture_output=True)
                active_id = [
                    int(tasks.split()[0])
                    for tasks in proc.stdout.decode().split('\n')[1:-1]]
                done = not slurm_id in active_id
            else:
                done = self.scan_check_id(slurm_id)

            # Wait for next scan step
            time.sleep(self.scan_interval)
            
        # Read results from result file
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
