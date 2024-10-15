from asparagus import Asparagus

# Start training a default PhysNet model.
model = Asparagus(
    config='model_hs/training.json',
    data_file='data/hexaaquairon3_hs.db',
    model_directory='model_hs',
    model_properties=['energy', 'forces', 'dipole', 'atomic_charges'],
    trainer_batch_size=32,
    trainer_max_epochs=1_000,
    )
model.train()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
