
import sys
sys.path.insert(0, '/home/toepfer/Documents/Project_PhysNet3/Asparagus')

from asparagus.sample import MetaSampler

# Initialize meta dynamics sampler for an ammonia molecule
# using the ORCA program to compute PBE reference energies, forces and the
# molecular dipole moment. The reference calculator ORCA runs on 4 CPUs
# ('orcablocks': '%pal nprocs 1 end').
# The collective variables of the meta dyanmics are the three N-H bonds
# (meta_cv=[[0, 1], [0, 2], [0, 3]]) where Gaussian potentials are added 
# every 10 steps (meta_gaussian_interval=10) of 0.05 eV height with 0.1 Ang
# width (meta_gaussian_height=0.05 & meta_gaussian_widths=0.1).
# Additionally, hookean constraints are applied on the N-H bonds to prevent
# N-H bond dissociation larger than 4.0 Ang 
# (meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]]).
# The temperature of the meta dyanmics is set to 500 K and run for
# 10000 steps a 1 fs time steps (total of 10 ps). Every 10th step is written
# to the database yielding 1000 reference samples.
if False:

    sampler = MetaSampler(
        config='model_nh3/nh3_meta.json',
        sample_data_file='model_nh3/nh3_meta.db',
        sample_directory='model_nh3',
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
            'directory': 'model_nh3/orca'},
        sample_num_threads=1,
        sample_save_trajectory=True,
        meta_cv=[[0, 1], [0, 2], [0, 3]],
        meta_gaussian_height=0.05,
        meta_gaussian_widths=0.1,
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 4.0], [0, 2, 4.0], [0, 3, 4.0]],
        meta_temperature=500,
        meta_time_step=1.0,
        meta_simulation_time=10_000.0,
        meta_save_interval=10,
        )

    # Start sampling procedure
    sampler.run()

# Start training a default PhysNet model (model_type='physnet' [default]).
if True:

    from asparagus import Asparagus
    model = Asparagus(
        config='model_nh3/nh3_physnet.json',
        data_file='model_nh3/nh3_meta.db',
        model_type='physnet',
        model_directory='model_nh3',
        model_properties=['energy', 'forces', 'dipole'],
        trainer_max_epochs=1_000,
        )
    model.train()
    model.test(
        test_datasets='test',
        test_directory=model.get('model_directory'))
