from asparagus import Asparagus

# Start training a default PhysNet model.
# Instead of training atomic charges to match best the dipole moment,
# here atomic charges are directly trained to match the previously
# computed Hirshfeld charges, see 'trainer_properties'. However,
# the model also compute the molecular dipole moment with respect to
# the atomic charge prediction, which is automatically compared with
# the reference dipole moment in the database (but not used for the
# loss function during training).
model = Asparagus(
    config='model_hs/training.json',
    data_file='data/hexaaquairon3_hs.db',
    model_directory='model_hs',
    model_properties=['energy', 'forces', 'atomic_charges', 'dipole'],
    trainer_properties=['energy', 'forces', 'atomic_charges'],
    trainer_batch_size=32,
    trainer_max_epochs=1_000,
    )
model.train()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
