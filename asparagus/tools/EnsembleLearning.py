import time
import os
from asparagus import Asparagus
from multiprocessing import Process
import numpy as np
from ase import io
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin

class EnsembleLearning:
    def __init__(self, model_name, source, properties, train_size, validation_size, test_size, epoch_step, num_epochs, par_mod, reag, temperature, interval, trajname, num_steps_md):
        self.model_name = model_name
        self.source = source
        self.properties = properties
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        self.epoch_step = epoch_step
        self.num_epochs = num_epochs
        self.par_mod = par_mod
        self.reag = reag
        self.temperature = temperature
        self.interval = interval
        self.trajname = trajname
        self.num_steps_md = num_steps_md
        self.models = []
        self.finished_epochs = []

    # Function for parallel training
    def par_train(self, models, batch_size, k):
        for i in range(0, len(models), batch_size):
            batch = models[i:i+batch_size]
            processes = []
            for mode in batch:
                model = Asparagus(config=f"{mode}.json", trainer_max_epochs=k)
                p = Process(target=model.train)
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

    # Function for creating new data
    def generate_data(self, model_traj_calc, reag, temperature, interval, trajname, num_steps_md):
        reag = io.read(reag)
        model = Asparagus(config=f"{self.models[model_traj_calc]}.json")
        calc = model.get_ase_calculator()
        reag.calc = calc
        dyn = Langevin(reag, 0.5 * units.fs, temperature * units.kB, 0.002)
        dyn.attach(self.printenergy, interval=interval)
        traj = Trajectory(trajname, 'w', reag)
        dyn.attach(traj.write, interval=interval)
        dyn.run(num_steps_md)

    # Function to estimate the model
    def model_estimator(self, config, reag):
        model = Asparagus(config=config)
        reag = io.read(reag)
        calc = model.get_ase_calculator()
        reag.calc = calc
        e = reag.get_potential_energy()
        return e

    # Method for creating configuration files for each model
    def create_model_configs(self, number_models):
        for i in range(number_models):
            Asparagus(
                config=f"{self.model_name}_{i+1}.json",
                data_source=f"{self.source}_{i+1}.npz",
                data_file=f"{self.model_name}_{i+1}.db",
                model_directory=f"{self.model_name}_{i+1}",
                model_properties=self.properties,
                data_num_train=self.train_size,
                data_num_valid=self.validation_size,
                data_num_test=self.test_size,
                trainer_max_epochs=self.epoch_step,
                trainer_max_gradient_norm=None,
                trainer_save_interval=self.epoch_step,
                data_seed=np.random.randint(1E6),
            )
            self.models.append(f"{self.model_name}_{i+1}")

    # Method to check finished epochs
    def check_finished_epochs(self):
        for mode in self.models:
            if os.path.exists(f"{mode}/checkpoints/"):
                if len(os.listdir(f'{mode}/checkpoints/')) == 0:
                    self.finished_epochs.append(5)
                else:
                    for file in os.listdir(f"{mode}/checkpoints/"):
                        if file.endswith(".pt"):
                            last_epoch = int(file.split('_')[-1].split('.')[0])
                            self.finished_epochs.append(last_epoch)
            else:
                self.finished_epochs.append(5)
        return min(self.finished_epochs)

    # Main training cycle
    def run_training_cycle(self):
        self.create_model_configs(len(self.models))
        finished_epoch = self.check_finished_epochs()

        for k in range(finished_epoch, self.num_epochs, self.epoch_step):
            start_time = time.time()
            self.par_train(self.models, self.par_mod, k)
            print(f"Training of {len(self.models)} models completed in {time.time() - start_time:.2f} seconds")

    # Method for model validation
    def validate_models(self):
        with open('energies_eval.dat', 'w') as f:
            for reag in self.trajname:
                energies = []
                for mode in self.models:
                    energy = self.model_estimator(f"{mode}.json", reag)
                    energies.append(energy)
                f.write(" ".join(map(str, energies)) + "\n")
"""
Example of the code how bootstrapping could be performed:

size_dict = data[list(data.keys())[0]].shape[0]
print("Keys in the npz file:", data.keys())
n_samples = 4
for i in range(n_samples):
    indices = np.random.choice(size_dict, size = size_dict, replace=True)
    bootstrapped_sample = {}
    for key in data.keys():
        bootstrapped_sample[key] = data[key][indices]
    np.savez(f"bootstrap_sample_{i+1}.npz", **bootstrapped_sample)
------------------------
Example of input parameters:

ensemble = EnsembleLearning(
model_name='test_ensemble',
source="bootstrap_sample",
properties=["energy", "force"],
train_size=0.8,
validation_size=0.1,
test_size=0.1,
epoch_step=5,
num_epochs=1000,
par_mod=4,
reag="reag.xyz",
temperature=500,
interval=50,
trajname="traj.traj",
num_steps_md=5000,
model_traj_calc = 1
)
Calling the function:
#First it could be called bootstrapping function

trainer.run_training_cycle()
trainer.generate_data()
trainer.validate_models()
"""
