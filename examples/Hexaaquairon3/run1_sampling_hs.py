
from asparagus.sampling import MetaSampler

sampler = MetaSampler(
    config='data/sampling_hs_config.json',
    sample_directory='data/sampling_hs',
    sample_data_file='data/hexaaquairon3_hs.db',
    sample_systems=['hexaaquairon3_hs.xyz'],
    sample_systems_format='xyz',
    sample_calculator='shell',
    sample_calculator_args = {
        'files': [
            'templates/run_orca.sh',
            'templates/run_orca.inp',
            'templates/run_orca.py',
            ],
        'files_replace': {
            '%xyz%': '$xyz',
            '%charge%': '$charge',
            '%multiplicity%': '$multiplicity',
            '%nproc%': 4
            },
        'execute_file': 'run_orca.sh',
        'charge': 3,
        'multiplicity': 6,
        'directory': 'data/sampling_hs/orca',
        'result_properties': ['energy', 'forces', 'dipole', 'atomic_charges']
        },
    sample_properties=['energy', 'forces', 'dipole', 'atomic_charges'],
    sample_systems_optimize=True,
    sample_save_trajectory=True,
    meta_cv=[[0, 1], [0, 4], [0, 7], [0, 10], [0, 13], [0, 16]],
    meta_gaussian_height=0.05,
    meta_gaussian_widths=0.2,
    meta_gaussian_interval=5,
    meta_time_step=1.0,
    meta_simulation_time=1000.0,
    meta_save_interval=2,
    meta_temperature=300,
    meta_langevin_friction=1.0,
    )
sampler.run()

