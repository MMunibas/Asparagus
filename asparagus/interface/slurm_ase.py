# ASE calculator class modifying and executing a template shell file(s) which
# compute atoms properties and provide them as a .json or .npy file.
import os
import json
import time
import socket
import subprocess
import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Any

import ase
from ase.calculators.calculator import Calculator

from .shell_ase import ShellCalculator, TagReplacement

from asparagus import utils

__all__ = ['SlurmCalculator']


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
    command: str, optional, default 'sbatch'
        Command to start the calculation.
    remote_client: str, optional, default None
        Remote client id (e.g. 'username@server.ch') to which a connection
        is established and the calculation are performed in the directory
        'username.hostname'. The connection is established by the 'ssh -tt',
        the working directory created by 'mkdir -p' and
        the calculation files are transferred via 'scp -r'.
    scan_interval: int, optional, default 5
        Scan interval checking for completeness of the submitted slurm job
    scan_command: str, optional, default f'squeue -u {os.environ['USER']}'
        Command to obtain the current slurm job list.
    label: str, optional, default 'shell'
        Name used for all files.  Not supported by all calculators.
        May contain a directory, but please use the directory parameter
        for that instead.
        Asparagus: May be used as 'calculator_tag'.
    directory: str or PurePath
        Working directory in which to read and write files and
        perform calculations.

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
        scan_interval=5)

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
        remote_client: Optional[str] = None,
        scan_interval: Optional[int] = 1,
        scan_command: Optional[str] = None,
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

        # Assign remote client address
        if remote_client is None or utils.is_string(remote_client):
            self.remote_client = remote_client
        else:
            raise SyntaxError(
                "Remote client input 'remote_client' is not reconginzed as"
                + "a client id!")
        
        # Assign job scanning time interval in seconds
        if utils.is_numeric(scan_interval):
            self.scan_interval = scan_interval
        else:
            raise SyntaxError(
                "Submitted job scan interval 'scan_interval' is not a "
                "numeric value!")

        # Assign command to obtain an output of active jobs and their ids
        # which are compared with the own job id number by the function
        # 'scan_check_id'.
        if scan_command is None:
            self.scan_command = f"squeue -u {os.environ['USER']:s}"
        elif utils.is_string(scan_command):
            self.scan_command = scan_command
        elif utils.is_string_array(scan_command):
            self.scan_command = " ".join(scan_command)
        else:
            raise SyntaxError(
                "Scan command input 'scan_command' is not reconginzed as"
                + "a command string or list of strings!")

        # Assign a user defined function to obtain job id of the submitted job
        # from the output of the submission command 'command'
        if scan_catch_id is None:
            self.scan_catch_id = None
        elif utils.is_callable(scan_catch_id):
            self.scan_catch_id = scan_catch_id
        else:
            raise SyntaxError(
                "Submitted job id catch function 'scan_catch_id' is not a "
                "callable function!")

        # Assign a user defined function to obtain job id list from the output
        # of 'scan_command'
        if scan_check_id is None:
            self.scan_check_id = None
        elif utils.is_callable(scan_check_id):
            self.scan_check_id = scan_check_id
        else:
            raise SyntaxError(
                "Submitted job id check function 'scan_check_id' is not a "
                "callable function!")

        return

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
        
        # Prepare shell command 
        if self.command is None:
            command = ['sbatch']
        else:
            command = self.command.split()

        # Prepare scan command
        if self.scan_command is None:
            scan_command = ['squeue']
        else:
            scan_command = self.scan_command.split()

        # Check executable file
        if self.execute_file is None:
            execute_file = [os.path.split(self.files[0])[1]]
        else:
            execute_file = [self.execute_file]

        # Run the calculation
        if self.remote_client is None:

            # Execute command with executable file
            proc = subprocess.run(
                command + execute_file,
                cwd=self.directory,
                capture_output=True)
            
            # Catch submission output
            stdout = proc.stdout.decode()

        else:

            # Execute command with executable file on remote client,
            # catching the submission output and the calculation directory
            # on the remote client
            stdout, calculation_directory = self.run_remote(
                command,
                execute_file)

        # Get slurm id
        if self.scan_catch_id is None:
            slurm_id = int(stdout.split()[-1])
        else:
            slurm_id = self.scan_catch_id(stdout)

        # Check for job completeness
        done = False
        while not done:
            
            # Get and check task id with active task ids
            if self.scan_check_id is None:

                # Run check command
                if self.remote_client is None:

                    # Catch submission output
                    proc = subprocess.run(
                        scan_command,
                        capture_output=True)
                    stdout = proc.stdout.decode()

                else:
                    
                    stdout = self.scan_remote(scan_command)

                # Check if job id is still job id list
                active_id = [
                    int(tasks.split()[0])
                    for tasks in stdout.split('\n')[1:-1]]
                done = not slurm_id in active_id

            else:

                done = self.scan_check_id(slurm_id)

            # Wait for next scan step
            time.sleep(self.scan_interval)

        # If calculation is done on a remote client, copy result file to
        # local machine
        if self.remote_client is not None:
            self.copy_remote(calculation_directory)

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

        # Check for completeness
        self.converged = True
        for prop_i in self.result_properties:
            if prop_i not in self.results:
                self.converged = False

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

    def run_remote(
        self,
        command: List[str],
        execute_file: List[str],
    ) -> str:
        """
        Run calculation on a remote client
        
        Parameters
        ----------
        command: str
            Submission command as string list
        execute_file: str
            Execution file as list

        Returns
        -------
        str
            Submission command standard output

        """
    
        # Working directory on remote client
        wdir = f"{os.environ['USER']:s}.{socket.gethostname():s}"

        # Target directory for working files on remote client
        tdir = f"{self.remote_client:s}:{wdir:}"

        # Calculation directory on remote client
        cdir = os.path.join(wdir, self.directory.split("/")[-1])

        # Execution command
        exe_command = (
            " ".join(command + execute_file)
            + " > job.id\n")

        # Echo job.id command
        cat_command = "cat job.id\n"

        # Logout command
        logout_command = f"logout\n"

        # First establish connection to remote client and create working
        # directory
        proc = subprocess.Popen(
            ['ssh', '-tt', self.remote_client],
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        proc.stdin.write(f"mkdir -p {wdir:s}\n".encode('utf-8'))
        proc.stdin.write(logout_command.encode('utf-8'))
        proc.stdin.close()

        # Second, copy working files to remote client working directory
        tdir = f"{self.remote_client:s}:{wdir:}"
        proc = subprocess.Popen(
            ['scp', '-r', self.directory, tdir],
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        # Third establish connection to remote client and run the execute
        # command with executable file
        proc = subprocess.Popen(
            ['ssh', '-tt', self.remote_client],
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        proc.stdin.write(f"cd {cdir:s}\n".encode('utf-8'))
        proc.stdin.write(exe_command.encode('utf-8'))
        proc.stdin.write(cat_command.encode('utf-8'))
        proc.stdin.write(logout_command.encode('utf-8'))
        proc.stdin.close()
        proc.wait()

        # Catch submission output
        stdout_list = proc.stdout.readlines()
        stdout = ""
        submission_flag = False
        for line_encoded in stdout_list:
            line_decoded = line_encoded.decode('utf-8')
            if logout_command[:-1] in line_decoded:
                submission_flag = False
            if submission_flag:
                if line_decoded[-1] == '\n':
                    stdout += line_decoded[:-1]
                else:
                    stdout += line_decoded
            if cat_command[:-1] in line_decoded:
                submission_flag = True
        
        return stdout, cdir
    
    def scan_remote(
        self,
        scan_command: List[str],
    ) -> str:
        """
        Run scan command on a remote client
        
        Parameters
        ----------
        scan_command: str
            Scan command as string list

        Returns
        -------
        str
            Scan command standard output

        """

        # Execution command
        scan_line_command = " ".join(scan_command) + "\n"

        # Logout command
        logout_command = f"logout\n"

        # First establish connection to remote client and request
        # job queue list
        proc = subprocess.Popen(
            ['ssh', '-tt', self.remote_client],
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        proc.stdin.write(scan_line_command.encode('utf-8'))
        proc.stdin.write(logout_command.encode('utf-8'))
        proc.stdin.close()

        # Catch scan output
        stdout_list = proc.stdout.readlines()
        stdout = ""
        scan_flag = False
        for line_encoded in stdout_list:
            line_decoded = line_encoded.decode('utf-8')
            if logout_command[:-1] in line_decoded:
                scan_flag = False
            if scan_flag:
                stdout += line_decoded
            if scan_line_command[:-1] in line_decoded:
                scan_flag = True

        return stdout

    def copy_remote(
        self,
        calculation_directory: List[str],
    ):
        """
        Run scan command on a remote client
        
        Parameters
        ----------
        calculation_directory: str
            Calculation directory on remote client

        """

        # Copy files from remote clients calculation directory to local 
        # machines working directory
        source_file = os.path.join(calculation_directory, self.result_file)
        source_command = f"{self.remote_client:s}:{source_file:}"
        target_command = os.path.join(self.directory, self.result_file)
        proc = subprocess.Popen(
            ['scp', source_command, target_command],
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        proc.wait()

        return
