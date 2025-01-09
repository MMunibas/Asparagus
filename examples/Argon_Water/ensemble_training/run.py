from asparagus import Asparagus

# Start training a default PhysNet model.
model = Asparagus(
    config='train.json',
    data_file='ar_water.db',
    data_num_train=0.8,
    data_num_valid=0.1,
    data_num_test=0.1,
    model_directory='model_ensemble',
    trainer_max_epochs=1000,)
model.train(
    model_ensemble=True,
    model_ensemble_num=5,
    model_num_threads=4,
    trainer_epochs_step=50,
    trainer_num_threads=3)
model.test(
    test_datasets='test',
    test_directory=model.get('model_directory'))
