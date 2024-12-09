from asparagus import Asparagus
from asparagus.sampling import Sampler

if True:

    sampler = Sampler(
        config='sample.json',
        sample_data_file='mp2_ar_water.db',
        sample_systems='ar_water.db',
        sample_properties=['energy', 'forces', 'dipole'],
        sample_calculator='slurm',
        sample_calculator_args = {
            'files': [
                '../templates/slurm/run_molpro.sh',
                '../templates/slurm/run_molpro.inp',
                '../templates/slurm/run_molpro.py',
                ],
            'files_replace': {
                '%xyz%': '$xyz',
                '%charge%': '$charge',
                '%spin2%': '$spin2',
                },
            'execute_file': 'run_molpro.sh',
            'charge': 1,
            'multiplicity': 2,
            'directory': 'molpro_slurm',
            'result_properties': ['energy', 'forces', 'dipole']
            },
        sample_num_threads=40,
        sample_save_trajectory=True,
        )

    sampler.run()

# Start training a default PhysNet model.
model = Asparagus(
    config='train.json',
    data_file='ar_water.db',
    model_directory='model',
    model_properties=['energy', 'forces', 'dipole'],
    model_type='physnet',
    trainer_max_epochs=5_000,
    )
model.train()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
