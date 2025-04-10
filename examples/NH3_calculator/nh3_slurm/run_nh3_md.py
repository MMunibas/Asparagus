
from asparagus.sampling import MDSampler

# Initialize molecule dynamics sampler for an ammonia molecule
# using the ORCA program to compute PBE reference energies, forces and the
# molecular dipole moment. The reference calculator ORCA runs on 4 CPUs
# ('orcablocks': '%pal nprocs 4 end').
# The temperature of the Langevin dyanmics is set to 500 K and run for
# 10000 steps a 1 fs time steps (total of 10 ps). Every 10th step is written
# to the database yielding 1000 reference samples.
sampler = MDSampler(
    config='nh3_md.json',
    sample_data_file='nh3_md.db',
    sample_systems='nh3_c3v.xyz',
    sample_systems_format='xyz',
    sample_systems_optimize=True,
    sample_systems_optimize_fmax=0.001,
    sample_properties=['energy', 'forces', 'dipole'],
    sample_calculator='slurm',
    sample_calculator_args = {
        'files': [
            'templates/run_molpro.sh',
            'templates/run_molpro.inp',
            'templates/run_molpro.py',
            ],
        'files_replace': {
            '%xyz%': '$xyz',
            '%charge%': '$charge',
            '%multiplicity%': '$multiplicity',
            },
        'execute_file': 'run_orca.sh',
        'charge': 0,
        'multiplicity': 1,
        'directory': 'slurm',
        'result_properties': ['energy', 'forces', 'dipole']
        },
    sample_num_threads=1,
    sample_save_trajectory=True,
    md_temperature=500,
    md_time_step=1.0,
    md_simulation_time=10_000.0,
    md_save_interval=10,
    md_langevin_friction=0.01,
    md_equilibration_time=0,
    md_initial_velocities=False
    )
sampler.run()

# Start training a default PhysNet model.
from asparagus import Asparagus
model = Asparagus(
    config='nh3_md.json',
    data_file='nh3_md.db',
    model_directory='model_nh3_md',
    model_properties=['energy', 'forces', 'dipole'],
    trainer_max_epochs=1_000,
    )
model.train()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
