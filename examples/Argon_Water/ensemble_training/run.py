from asparagus import Asparagus

# Start training a default PhysNet model.
model = Asparagus(
    config='train.json',
    data_file='ar_water.db',
    data_num_train=800,
    data_num_valid=100,
    data_num_test=100,
    model_directory='model_ensemble',
    trainer_max_epochs=1000,)
model.train(
    train_ensemble=True,
    ensemble_num_models=3,
    ensemble_epochs_step=5,
    ensemble_num_threads=3)
exit()
model.test(
    test_datasets='all',
    test_directory=model.get('model_directory'))
