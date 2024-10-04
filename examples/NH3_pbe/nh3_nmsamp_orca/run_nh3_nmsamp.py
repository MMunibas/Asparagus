
from asparagus.sampling import NormalModeSampler

# Initialize normal mode sampler for an equilibrated ammonia molecule
# using the ORCA program to compute PBE reference energies, forces and the
# molecular dipole moment. The reference calculation are divided into
# 4 threads each on a single CPU.
sampler = NormalModeSampler(
    config='nh3_nmsamp.json',
    sample_data_file='nh3_nmsamp.db',
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
        'orcablocks': '%pal nprocs 1 end',
        'directory': 'orca'},
    sample_num_threads=4,
    sample_save_trajectory=True,
    nms_temperature=500,
    nms_nsamples=1000,
    )

# Start sampling procedure 
sampler.run()

# Start training a default PhysNet model.
from asparagus import Asparagus
model = Asparagus(
    config='nh3_nms.json',
    data_file='nh3_nms.db',
    model_directory='model_nh3_nms',
    model_properties=['energy', 'forces', 'dipole'],
    trainer_max_epochs=1_000,
    )
model.train()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
