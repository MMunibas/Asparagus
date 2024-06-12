import sys
sys.path.insert(0, '/home/toepfer/Documents/Project_PhysNet3/Asparagus')

from asparagus import Asparagus

# Sampling
if False:

    from asparagus.sample import NormalModeSampler

    sampler = NormalModeSampler(
        config='model_zundel/zundel_config.json',
        sample_directory='model_zundel/sampling',
        sample_data_file='model_zundel/zundel.db',
        sample_systems='zundel_h5o2.xyz',
        sample_systems_format='xyz',
        sample_systems_optimize=False,
        sample_properties=['energy', 'forces', 'dipole'],
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 1,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 1 end',
            'directory': 'model_zundel/sampling'},
        sample_num_threads=4,
        sample_save_trajectory=True,
        nms_temperature=500.0,
        nms_nsamples=100,
    )
    sampler.run(nms_frequency_range=[('>||', 100.0)])

if False:
    
    from asparagus.sample import MetaSampler
    
    sampler = MetaSampler(
        config='model_zundel/zundel_config.json',
        sample_directory='model_zundel/sampling',
        sample_data_file='model_zundel/zundel.db',
        sample_systems=['zundel_h5o2.xyz'],
        sample_systems_format='xyz',
        sample_calculator='ORCA',
        sample_calculator_args = {
            'charge': 1,
            'mult': 1,
            'orcasimpleinput': 'RI PBE D3BJ def2-SVP def2/J TightSCF',
            'orcablocks': '%pal nprocs 4 end',
            'directory': 'model_zundel/sampling'},
        sample_properties=['energy', 'forces', 'dipole'],
        sample_systems_optimize=False,
        sample_save_trajectory=True,
        meta_cv=[['-', 0, 1, 0, 4], [1, 4]],
        meta_hookean=[[1, 4, 6.0]],
        meta_gaussian_height=0.05,
        meta_gaussian_widths=0.2,
        meta_gaussian_interval=10,
        meta_time_step=1.0,
        meta_simulation_time=10000.0,
        meta_save_interval=10,
        meta_temperature=300,
        meta_langevin_friction=1.0,
        )
    sampler.run()

# Train
if True:
    
    model = Asparagus(
        config='model_zundel/zundel_config.json',
        data_file='model_zundel/zundel.db',
        model_directory='model_zundel',
        model_properties=['energy', 'forces', 'dipole'],
        model_electrostatic=True,
        model_interaction_cutoff=12.0,
        input_cutoff_descriptor=6.0,
        trainer_properties_weights={
            'energy': 1.,
            'forces': 50.,
            'dipole': 20.
            },
        )
    model.train()
    model.test(
        test_datasets='all',
        test_directory=model.get('model_directory'))

if True:

    import ase
    from ase.optimize import BFGS
    from ase.visualize import view
    from ase.vibrations import Vibrations
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet
    from ase.io.trajectory import Trajectory
    import ase.units as units
    
    model = Asparagus(
        config="model_zundel/config.json",
        model_directory="model_zundel"
        )
    calc = model.get_ase_calculator()
    
    zundel = ase.io.read("zundel_h5o2.xyz")
    zundel.calc = calc
    
    dyn = BFGS(zundel)
    dyn.run(fmax=0.01)
    
    view(zundel)
    
    vib = Vibrations(zundel)
    vib.clean()
    vib.run()
    vib.summary()
    vib.write_mode()
