from asparagus import Asparagus
from asparagus.sampling import Sampler

# Start training a default PhysNet model.
sampler = Sampler(
    config='sample.json',
    sample_data_file='sample.db',
    sample_systems='ar_water.db',
    sample_calculator='Asparagus',
    sample_calculator_args={
        'config': 'train.json'},
    )

sampler.run()

