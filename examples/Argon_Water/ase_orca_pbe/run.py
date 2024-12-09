from asparagus import Asparagus
from asparagus.sampling import NormalModeScanner, MDSampler, MetaSampler

if False:

    sampler = NormalModeScanner(
        config='sample.json',
        sample_data_file='ar_water.db',
        sample_systems='ar_h2o.xyz',
        sample_systems_format='xyz',
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        sample_properties=['energy', 'forces', 'dipole'],
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': +1,
            'mult': 2,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'orca'},
        sample_num_threads=4,
        sample_save_trajectory=True,
        nms_harmonic_energy_step=0.05,
        nms_energy_limits=1.00,
        nms_number_of_coupling=2,
        )

    try:
        sampler.run(nms_frequency_range=[('>', 50)])
    except:
        pass

    sampler = MDSampler(
        config='sample.json',
        sample_data_file='ar_water.db',
        sample_systems='ar_h2o.xyz',
        sample_systems_format='xyz',
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        sample_properties=['energy', 'forces', 'dipole'],
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': +1,
            'mult': 2,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 4 end',
            'directory': 'orca'},
        sample_num_threads=1,
        sample_save_trajectory=True,
        md_temperature=300,
        md_time_step=1.0,
        md_simulation_time=5000.0,
        md_save_interval=10,
        md_langevin_friction=0.01,
        md_equilibration_time=0,
        md_initial_velocities=False
        )

    try:
        sampler.run()
    except:
        pass

    sampler = MetaSampler(
        config='sample.json',
        sample_data_file='ar_water.db',
        sample_systems='ar_h2o.xyz',
        sample_systems_format='xyz',
        sample_systems_optimize=True,
        sample_systems_optimize_fmax=0.001,
        sample_properties=['energy', 'forces', 'dipole'],
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': +1,
            'mult': 2,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 4 end',
            'directory': 'orca'},
        sample_num_threads=1,
        sample_save_trajectory=True,
        meta_cv=[[0, 1], [1, 2], [1, 3], [2, 1, 3], [0, 1, 2]],
        meta_gaussian_height=0.05,
        meta_gaussian_widths=[0.2, 0.1, 0.1, 0.1, 0.2],
        meta_gaussian_interval=10,
        meta_hookean=[[0, 1, 6.0], [1, 2, 4.0], [1, 3, 4.0]],
        meta_temperature=300,
        meta_time_step=1.0,
        meta_simulation_time=50_000.0,
        meta_save_interval=10,
        )

    try:
        sampler.run()
    except:
        pass


# Start training a default PhysNet model.
model = Asparagus(
    config='train.json',
    data_file='ar_water.db',
    model_directory='model',
    model_properties=['energy', 'forces', 'dipole'],
    trainer_max_epochs=5_000,
    )
model.train()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
