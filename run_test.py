
import asparagus

import os
import torch
import numpy as np

#==============================================================================
# Test Parameter
#==============================================================================


flag_dictionary_initialization = True
flag_database_sql = True
flag_database_npz = True
flag_database_hdf5 = False
flag_sampler_all = True
flag_sampler_shell = True
flag_sampler_slurm = False

flag_model_physnet = True
flag_train_physnet_sql = True
flag_train_physnet_npz = True
flag_ase_physnet = True

flag_model_painn = True
flag_train_painn = True

flag_train_cuda = False

#==============================================================================
# Test Asparagus Main Class Initialization
#==============================================================================

config_file = 'test/init.json'
config = {
    'config_file': config_file}
device = 'cpu'
dtype = torch.float32

# Dictionary initialization
if flag_dictionary_initialization:

    model = asparagus.Asparagus(config)
    model = asparagus.Asparagus(config=config_file)
    model = asparagus.Asparagus(config_file=config_file)

    model = asparagus.Asparagus(
        config,
        model_device=device,
        model_dtype=dtype)
    model = asparagus.Asparagus(
        config,
        model_device=device,
        model_dtype=dtype)

#==============================================================================
# Test Asparagus DataContainer Class Initialization
#==============================================================================

config_file = 'test/data.json'
config = {
    'config_file': config_file}

# SQL
if flag_database_sql:

    # Open DataBase file
    model = asparagus.Asparagus(
        config=config_file,
        data_file='data/nms_nh3.db',
        data_file_format='sql',
        )

    # Create new DataBase file
    model.set_data_container(
        config=config_file,
        data_file='test/nms_nh3_test.db',
        data_source='data/nms_nh3.db',
        data_overwrite=True,
    )

    # Add same source to DataBase file, should be skipped
    model.set_data_container(
        config=config_file,
        data_file='test/nms_nh3_test.db',
        data_source='data/nms_nh3.db',
        data_overwrite=False,
    )

    # Create new DataBase file with itself as source, should return error
    try:
        model.set_data_container(
            config=config_file,
            data_file='test/nms_nh3_test.db',
            data_source='test/nms_nh3_test.db',
            data_overwrite=True,
        )
    except SyntaxError:
        print("\nSyntaxError as expected\n")

    # Get DataContainer (Reset data source)
    model = asparagus.Asparagus(
        config=config_file,
        data_file='test/nms_nh3_test.db',
        data_file_format='sql',
        data_source='data/nms_nh3.db',
        )
    data = model.get_data_container()
    print("\nDatabase path: ", model.get_data_container(), "\n")
    print("\nDatabase entry '0': ", data[0]['energy'])
    print("\nDatabase Train entry '1': ", data.get_train(1)['atoms_number'])
    print("\nDatabase Valid entry '2': ", data.get_valid(2)['cell'])
    print("\nDatabase Valid entry '2': ", data.get_valid(2)['pbc'])
    print("\nDatabase Test entry  '3': ", data.get_test(3)['positions'])
    print("\nDatabase Test entry  '3': ", data.get_test(4)['forces'])

    # Load Numpy .npz files
    model.set_data_container(
        config=config_file,
        data_file='test/h2co_test.db',
        data_source='data/h2co_B3LYP_cc-pVDZ_4001.npz',
        data_overwrite=True,
    )
    data = model.get_data_container(data_file='test/h2co_test.db')
    print("\nDatabase path: ", data, "\n")
    print("\nDatabase entry '0': ", data[0]['energy'])
    print("\nDatabase Train entry '1': ", data.get_train(1)['atomic_numbers'])
    print("\nDatabase Valid entry '2': ", data.get_valid(2)['charge'])
    print("\nDatabase Test entry  '3': ", data.get_test(3)['pbc'])

    # Load multiple source files files
    model.set_data_container(
        config=config_file,
        data_file='test/nh3_h2co_test.db',
        data_source=['data/h2co_B3LYP_cc-pVDZ_4001.npz', 'data/nms_nh3.db'],
        data_overwrite=True,
    )

    # Check if repeated data source is skipped
    model.set_data_container(
        config=config_file,
        data_file='test/nh3_h2co_test.db',
        data_source=['data/nms_nh3.db'],
        data_overwrite=False,
    )

    # Load ASE trajectory file
    model.set_data_container(
        config=config_file,
        data_file='test/meta_nh3_test.db',
        data_source='data/meta_nh3.traj',
        data_overwrite=True,
    )

    # Load ASE trajectory file with different property units
    model.set_data_container(
        config=config_file,
        data_file='test/meta_nh3_test_unit.db',
        data_source='data/meta_nh3.traj',
        data_load_properties=['energy', 'forces'],
        data_unit_properties={
            'positions': 'Bohr',
            'energy': 'kcal/mol',
            'forces': 'kcal/mol/Bohr'},
        data_overwrite=True,
    )

    # Test training initialization
    model.train(
        trainer_max_epochs=0,
        model_directory='test/test_model')

    # Check automatic model property assignment from data properties
    if os.path.exists(config_file):
        os.remove(config_file)
    model = asparagus.Asparagus(
        config=config_file,
        data_file='test/meta_nh3_test_unit.db',
        data_source='data/meta_nh3.traj',
        data_load_properties=['energy', 'forces'],
        data_unit_properties={
            'positions': 'Bohr',
            'energy': 'kcal/mol',
            'forces': 'kcal/mol/Bohr'},
        data_overwrite=True,
    )

    # Test training initialization
    model.train(
        trainer_max_epochs=0,
        model_directory='test/test_model')

# Numpy npz
if flag_database_npz:

    model = asparagus.Asparagus(
        config=config_file,
        data_file='test/nms_nh3_test.db.npz',
        data_file_format='npz',
        data_source='data/nms_nh3.db',
        data_source_format='db',
        data_overwrite=True,
        )

    # Create new DataBase file
    data = model.get_data_container(
        config=config_file,
        data_file='test/nms_nh3_test.db.npz',
        data_file_format='npz',
        data_source='data/nms_nh3.db',
        data_source_format='db',
        data_overwrite=False,
    )
    print("\nDatabase path: ", data, "\n")
    print("\nDatabase entry '0': ", data[0]['energy'])
    print("\nDatabase Train entry '1': ", data.get_train(1)['atoms_number'])
    print("\nDatabase Valid entry '2': ", data.get_valid(2)['cell'])
    print("\nDatabase Valid entry '2': ", data.get_valid(2)['pbc'])
    print("\nDatabase Test entry  '3': ", data.get_test(3)['positions'])
    print("\nDatabase Test entry  '3': ", data.get_test(4)['forces'])
    
    # Test training initialization
    model.train(
        trainer_max_epochs=0,
        model_directory='test/test_model')

# HDF5
if flag_database_hdf5:

    # Create new DataBase file
    model = asparagus.Asparagus(
        config=config_file,
        data_file='test/nms_nh3_test.db.h5',
        data_file_format='hdf5',
        data_source='data/nms_nh3.db',
        data_source_format='sql',
        data_overwrite=True,
    )
    data = model.get_data_container()
    
    # Test training initialization
    model.train(
        trainer_max_epochs=0,
        model_directory='test/test_model')

#==============================================================================
# Test Asparagus Sampler Methods
#==============================================================================

# Sampler - with XTB and ORCA
# Mind: XTB is not thread safe when using with ASE modules such as Optimizer
# or Vibrations, but simple Calculator call works
if flag_sampler_all:
    
    from asparagus.sample import Sampler
    
    # Load single system from xyz file and compute properties using XTB default 
    # calculator
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',
        sample_calculator_args = {
            'charge': 0,
            'directory': 'test/xtb'},
        sample_num_threads=1,
        )
    sampler.run()
    
    # Load two system from xyz file and compute properties using XTB default 
    # calculator in parallel (still works without using other ASE functions)
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='XTB',
        sample_calculator_args = {
            'charge': 0,
            'directory': 'test/xtb'},
        sample_num_threads=2,
        )
    sampler.run()

    # Load two systems from xyz file and a ASE trajectory file and compute 
    # properties using XTB default calculator
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/meta_nh3.traj'],
        sample_calculator='XTB',
        sample_calculator_args = {
            'charge': 0,
            'directory': 'test/xtb'},
        sample_num_threads=1,
        )
    sampler.run()
    
    # Load a selection of sample system from an Asparagus data file and compute
    # properties using XTB default calculator
    sampler = Sampler(
        config='test/smpl_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nms_nh3.db',
        sample_systems_format='db',
        sample_systems_indices=[0, 1, 2, 3, -4, -3, -2, -1],
        sample_calculator='XTB',
        sample_calculator_args = {
            'charge': 0,
            'directory': 'test/xtb'},
        sample_num_threads=1,
        )
    sampler.run()

    from asparagus.sample import MCSampler

    # Sample a single system loaded from a xyz file using the Monte-Carlo
    # sampling method with the XTB calculator
    sampler = MCSampler(
        config='test/mc_nh3.json',
        sample_directory='test',
        sample_data_file='test/mc_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        mc_temperature=300.0,
        mc_steps=100,
        mc_max_displacement=0.1,
        mc_save_interval=1,
        )
    sampler.run()
    
    # Sample two systems loaded from a xyz files in parallel using the
    # Monte-Carlo sampling method and the ORCA calculator (thread safe)
    sampler = MCSampler(
        config='test/mc_nh3.json',
        sample_directory='test',
        sample_data_file='test/mc_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        mc_temperature=300.0,
        mc_steps=10,
        mc_max_displacement=0.1,
        mc_save_interval=1,
        )
    sampler.run()

    from asparagus.sample import MDSampler

    # Sample a single system loaded from a xyz file using the Molecular 
    # Dynamics sampling method with the XTB calculator
    sampler = MDSampler(
        config='test/md_nh3.json',
        sample_directory='test',
        sample_data_file='test/md_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        md_temperature=500,
        md_time_step=1.0,
        md_simulation_time=100.0,
        md_save_interval=10,
        md_langevin_friction=0.01,
        md_equilibration_time=0,
        md_initial_velocities=False,
        )
    sampler.run()

    # Sample two systems loaded from a xyz files in parallel using the 
    # Molecular Dynamics sampling method and the ORCA calculator (thread safe)
    sampler = MDSampler(
        config='test/md_nh3.json',
        sample_directory='test',
        sample_data_file='test/md_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=2,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        md_temperature=500,
        md_time_step=1.0,
        md_simulation_time=20.0,
        md_save_interval=10,
        md_langevin_friction=0.01,
        md_equilibration_time=0,
        md_initial_velocities=True,
        md_initial_temperature=300,
        )
    sampler.run()

    from asparagus.sample import MetaSampler
    
    # Sample a single system loaded from a xyz file using the Meta Dynamics
    # sampling method with the XTB calculator
    sampler = MetaSampler(
        config='test/meta_nh3.json',
        sample_directory='test',
        sample_data_file='test/meta_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_save_trajectory=True,
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        meta_cv=[[0, 1], [0, 2], [0, 3]],
        meta_gaussian_height=0.10,
        meta_gaussian_widths=0.1,
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        meta_temperature=500,
        meta_time_step=1.0,
        meta_simulation_time=10_0.0,
        meta_save_interval=10,
        )
    sampler.run()

    # Sample a system loaded from a xyz files in parallel using the Meta
    # Dynamics sampling method and the ORCA calculator (thread safe)
    # Here each of the multiple runs store the Gaussian add-potentials into 
    # in the same list, affecting the other runs as well and decrease the
    # sample steps to reach higher potential areas.
    # Not yet working as planned
    sampler = MetaSampler(
        config='test/meta_nh3.json',
        sample_directory='test',
        sample_data_file='test/meta_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=4,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        meta_cv=[[0, 1], [0, 2], [0, 3]],
        meta_gaussian_height=0.10,
        meta_gaussian_widths=0.1,
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        meta_temperature=500,
        meta_time_step=1.0,
        meta_simulation_time=20.0,
        meta_save_interval=10,
        meta_parallel=True,
        )
    sampler.run()

    # Sample two system loaded from a xyz files in parallel using the Meta
    # Dynamics sampling method and the ORCA calculator (thread safe)
    sampler = MetaSampler(
        config='test/meta_nh3.json',
        sample_directory='test',
        sample_data_file='test/meta_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_save_trajectory=True,
        sample_num_threads=2,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        meta_cv=[[0, 1], [0, 2], [0, 3]],
        meta_gaussian_height=0.10,
        meta_gaussian_widths=0.1,
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        meta_temperature=500,
        meta_time_step=1.0,
        meta_simulation_time=20.0,
        meta_save_interval=10,
        )
    sampler.run()

    from asparagus.sample import NormalModeScanner
    
    # Sample a single system loaded from a xyz file using the Normal Mode
    # Scanner sampling method with the XTB calculator
    sampler = NormalModeScanner(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_harmonic_energy_step=0.10,
        nms_energy_limits=0.50,
        nms_number_of_coupling=1,
        nms_limit_of_steps=10,
        nms_limit_com_shift=0.01,
        nms_save_displacements=True,
        )
    sampler.run()

    # Sample two systems loaded from a xyz file using the Normal Mode
    # Scanner sampling method with the ORCA calculator (thread safe).
    # Here it parallelize over the (1) system optimizations, (2) atom
    # displacement calculations for numeric normal mode analysis and (3) the
    # scans along single or combinations of normal modes. Step (2) and (3) will
    # run in serial for each sample system.
    sampler = NormalModeScanner(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems=['data/nh3_c3v.xyz', 'data/nh3_d3h.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_num_threads=4,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_harmonic_energy_step=0.10,
        nms_energy_limits=0.50,
        nms_number_of_coupling=1,
        nms_limit_of_steps=10,
        nms_limit_com_shift=0.01,
        nms_save_displacements=False,
        )
    sampler.run()

    from asparagus.sample import NormalModeSampler
    
    # Sample a single system loaded from a xyz file using the Normal Mode
    # Sampler sampling method with the XTB calculator
    sampler = NormalModeSampler(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='XTB',   # Not thread save when using ASE modules
        sample_num_threads=1,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_temperature=500.0,
        nms_nsamples=100,
        )
    sampler.run()
    
    # Sample a system loaded from a xyz file using the Normal Mode
    # Sampler sampling method with the ORCA calculator (thread safe).
    # Here it parallelize over the (1) system optimizations, (2) atom
    # displacement calculations for numeric normal mode analysis and (3) the
    # number of randomly sampled system conformations. Step (2) and (3) will
    # run in serial for each potential sample system.
    sampler = NormalModeSampler(
        config='test/nms_nh3.json',
        sample_directory='test',
        sample_data_file='test/nms_nh3.db',
        sample_systems='data/nh3_d3h.xyz',
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 0,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'test/orca'},
        sample_num_threads=4,
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        nms_temperature=500.0,
        nms_nsamples=10,
        )
    sampler.run()

#==============================================================================
# Test Asparagus Calculators - Shell & Slurm
#==============================================================================

# Shell Calculator
if flag_sampler_shell:
    
    from asparagus.sample import Sampler
    
    # Calculate properties of a sample system with multiple conformations
    # using the Shell calculator with template files for an ORCA calculation.
    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='shell',
        sample_calculator_args = {
            'files': [
                'data/template/shell/run_orca.sh',
                'data/template/shell/run_orca.inp',
                'data/template/shell/run_orca.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%multiplicity%': '$multiplicity',
                },
            'execute_file': 'run_orca.sh',  # or 'data/template/run_orca.sh'
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/shell',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=1,
        )
    sampler.run()

    # Calculate properties of a sample system with multiple conformations
    # using the Shell calculator with template files for an ORCA calculation
    # and in parallel.
    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/meta_nh3.traj',
        sample_calculator='shell',
        sample_calculator_args = {
            'files': [
                'data/template/shell/run_orca.sh',
                'data/template/shell/run_orca.inp',
                'data/template/shell/run_orca.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%multiplicity%': '$multiplicity',
                },
            'execute_file': 'data/template/shell/run_orca.sh',
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/shell',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=4,
        sample_save_trajectory=True,
        )
    sampler.run()

# Slurm Calculator
if flag_sampler_slurm:

    from asparagus.sample import Sampler
    
    # Calculate properties of a sample system with multiple conformations
    # using the Slurm calculator with template files for a MOLPRO calculation.
    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='slurm',
        sample_calculator_args = {
            'files': [
                'data/template/slurm/run_molpro.sh',
                'data/template/slurm/run_molpro.inp',
                'data/template/slurm/run_molpro.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%spin2%': '$spin2',
                },
            'execute_file': 'run_molpro.sh',
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/slurm',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=1,
        )
    sampler.run()

    # Calculate properties of a sample system with multiple conformations
    # using the Slurm calculator with template files for a MOLPRO calculation.
    # Here, define own slurm task id catch and check function
    
    import subprocess
    
    def catch_id(
        stdout: str,
    ) -> int:
        """
        Catch slurm task id from the output when running:
          subrocess.run([command, execute_file], capture_output=True)
          (here [command, execute_file] -> 'sbatch run_molpro.sh')
        
        Parameters
        ----------
        stdout: str
            Decoded output line (e.g. 'Submitted batch job 10937679')
        
        Return
        ------
        int
            Task id
        """
        return int(proc.stdout.decode().split()[-1])
    
    def check_id(
        slurm_id: int,
    ) -> bool:
        """
        Check slurm task id with e.g. task id list extracted from squeue
        
        Parameters
        ----------
        slurm_id: int
            Slurm task id of the submitted job
        
        Return
        ------
        bool
            Answer if task is done:
            False, if task is still running (task id is found in squeue)
            True, if task is done (task id not found in squeue)
        """
        proc = subprocess.run(
            ['squeue', '-u', os.environ['USER']],
            capture_output=True)
        active_id = [
            int(tasks.split()[0])
            for tasks in proc.stdout.decode().split('\n')[1:-1]]
        return not slurm_id in active_id

    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nh3_c3v.xyz',
        sample_systems_format='xyz',
        sample_calculator='slurm',
        sample_calculator_args = {
            'files': [
                'data/template/slurm/run_molpro.sh',
                'data/template/slurm/run_molpro.inp',
                'data/template/slurm/run_molpro.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%spin2%': '$spin2',
                },
            'execute_file': 'run_molpro.sh',
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/slurm',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=1,
        scan_interval=1,
        scan_catch_id=catch_id,
        scan_check_id=check_id,
        )
    sampler.run()
    
    # Calculate properties of a sample system with multiple conformations
    # using the Slurm calculator with template files for a MOLPRO calculation.
    sampler = Sampler(
        config='test/calc_nh3.json',
        sample_directory='test',
        sample_data_file='test/smpl_nh3.db',
        sample_systems='data/nms_nh3.db',
        sample_systems_format='db',
        sample_systems_indices=[0, 1, 2, 3, -4, -3, -2, -1],
        sample_calculator='slurm',
        sample_calculator_args = {
            'files': [
                'data/template/slurm/run_molpro.sh',
                'data/template/slurm/run_molpro.inp',
                'data/template/slurm/run_molpro.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%spin2%': '$spin2',
                },
            'execute_file': 'run_molpro.sh',
            'charge': 0,
            'multiplicity': 1,
            'directory': 'test/slurm',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=4,
        )
    sampler.run()


#==============================================================================
# Test Asparagus Model Calculator - PhysNet
#==============================================================================

# Initialize PhysNet model calculator
config_file1 = 'test/model_physnet.json'
if flag_model_physnet:
    
    model = asparagus.Asparagus(
        config_file=config_file1,
        model_type='physnet')
    mcalc = model.get_model_calculator(
        model_directory='test/physnet') # Default model type: 'PhysNet'
    model.set_model_calculator(
        model_directory='test/physnet')
    model.set_model_calculator(
        model_calculator=mcalc)
    
# Initialize PhysNet model training
if flag_train_physnet_sql:
    
    config_file2 = 'test/train_physnet.json'
    model = asparagus.Asparagus(
        config=config_file1,
        config_file=config_file2,
        data_file='data/nms_nh3.db',
        model_directory='test/physnet_sql',
        model_num_threads=2,
        trainer_max_epochs=10,
        trainer_debug_mode=False,
        )
    trainer = model.get_trainer()
    model.train()
    model.test(test_directory='test/physnet_sql')

if flag_train_physnet_npz:
    
    config_file2 = 'test/train_physnet_npz.json'
    model = asparagus.Asparagus(
        config=config_file1,
        config_file=config_file2,
        data_file='test/nms_nh3.db.npz',
        data_source='data/nms_nh3.db',
        data_source_format='db',
        model_directory='test/physnet_npz',
        model_num_threads=2,
        trainer_max_epochs=10,
        trainer_debug_mode=False,
        )
    trainer = model.get_trainer()
    model.train()
    model.test(test_directory='test/physnet_npz')

# Test ASE calculator
if flag_ase_physnet:

    from ase import Atoms
    
    # Get ASE model calculator
    config_file = 'test/train_physnet.json'
    model = asparagus.Asparagus(
        config_file=config_file)
    calc = model.get_ase_calculator()
    
    # Get data container
    data = model.get_data_container()
    Ndata = len(data)
    results_energy = np.zeros([Ndata, 2], dtype=float)
    for idata, data_i in enumerate(data):
    
        # Set system from data container
        system = Atoms(
            data_i['atomic_numbers'],
            positions=data_i['positions'])
        system_energy = data_i['energy'].numpy()
        system_forces = data_i['forces'].numpy()
        system_dipole = data_i['dipole'].numpy()
        
        # Compute model properties
        model_energy = calc.get_potential_energy(system)
        model_forces = calc.get_forces(system)
        model_dipole = calc.get_dipole_moment(system)
        
        # Compare results
        if False:
            print(
                "Reference and model energy (error): "
                + f"{system_energy:.4f} eV, {model_energy:.4f} eV "
                + f"({system_energy - model_energy:.4f} eV)"
                )
            print(
                "Reference and model forces on nitrogen (mean error): "
                + f"{system_forces[0]} eV/Ang, {model_forces[0]} eV/Ang "
                + f"({np.mean(system_forces[0] - model_forces[0]):.4f} eV/Ang)"
                )
            print(
                "Reference and model dipole (mean error): "
                + f"{system_dipole} eAng, {model_dipole} eAng "
                + f"({np.mean(system_dipole - model_dipole):.4f} eAng)"
                )

        # Append to result list
        results_energy[idata, 0] = system_energy
        results_energy[idata, 1] = model_energy

    # Show RMSE
    rmse_energy = np.sqrt(
        np.mean((results_energy[:, 0] - results_energy[:, 1])**2))
    print(f"RMSE(energy) = {rmse_energy:.4f} eV")

# Initialize PaiNN model calculator
config_file1 = 'test/model_painn.json'
if flag_model_painn:

    model = asparagus.Asparagus(
        config_file=config_file1,
        model_type='painn')
    mcalc = model.get_model_calculator(
        model_directory='test/painn') # Default model type: 'PhysNet'
    model.set_model_calculator(
        model_directory='test/painn')
    model.set_model_calculator(
        model_calculator=mcalc)

# Initialize PaiNN model training
if flag_train_painn:

    config_file2 = 'test/train_painn.json'
    model = asparagus.Asparagus(
        config=config_file1,
        config_file=config_file2,
        data_file='data/nms_nh3.db',
        model_directory='test/painn',
        model_num_threads=2,
        trainer_max_epochs=10,
        )
    trainer = model.get_trainer()
    model.train()
    model.test(test_directory='test/painn')

#==============================================================================
# Test Asparagus Model Calculator - PhysNet in Cuda
#==============================================================================

# Initialize PhysNet model training
if flag_train_cuda:
    
    config_file = 'test/train_cuda.json'
    model = asparagus.Asparagus(
        config_file=config_file,
        data_file='data/nms_nh3.db',
        model_directory='test/cuda',
        trainer_max_epochs=10,
        device='cuda',
        model_device='cuda',
        )
    model.train()
    model.test(test_directory='test/cuda')
