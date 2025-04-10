
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
    sample_calculator='ORCA',
    sample_calculator_args = {
        'charge': 0,
        'mult': 1,
        'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
        'orcablocks': '%pal nprocs 4 end',
        'directory': 'orca'},
    sample_num_threads=1,
    sample_save_trajectory=True,
    md_temperature=500,
    md_time_step=1.0,
    md_simulation_time=10.0,
    md_save_interval=10,
    md_langevin_friction=0.01,
    md_equilibration_time=0,
    md_initial_velocities=False
    )
sampler.run()

