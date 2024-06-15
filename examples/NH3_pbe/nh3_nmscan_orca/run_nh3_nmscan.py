
import sys
sys.path.insert(0, '/home/toepfer/Documents/Project_PhysNet3/Asparagus')

from asparagus.sample import NormalModeScanner

# Initialize normal mode scanner sampler for an equilibrated ammonia molecule
# using the ORCA program to compute PBE reference energies, forces and the
# molecular dipole moment. The reference calculation are divided into
# 4 threads.
# The step size of one normal mode step or combination of two normal mode steps
# (nms_number_of_coupling=2) shall be harmonically about 0.05 eV 
# (nms_harmonic_energy_step=0.05) and be applied up to a energy limit of 1 eV
# (nms_energy_limits=1.00) above the equilibrium energy is reached.
sampler = NormalModeScanner(
    config='nh3_nms.json',
    sample_data_file='nh3_nms.db',
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
    nms_harmonic_energy_step=0.05,
    nms_energy_limits=1.00,
    nms_number_of_coupling=2,
    )

# Start sampling procedure but only for normal modes with respective
# frequencies above 100 cm-1 to avoid including translation and rotational 
# normal mode vectors (nms_frequency_range=[('>', 100)])
sampler.run(nms_frequency_range=[('>', 100)])

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
