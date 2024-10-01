import time
import os
import numpy as np
import torch
from asparagus import Asparagus
from ase.io import read
import re

class AdaptiveSampling:
    def __init__(self, initial_dataset = None, 
                 num_epochs=100, 
                 model_name='adaptive_sampling_test', 
                 properties=['energy', 'forces'], 
                 split_ratio=0.3, 
                 initial_train_size=0.9, 
                 initial_validation_size=0.1, 
                 initial_test_size=0.0, 
                 num_iterations=4):
        self.initial_dataset = None
        self.num_epochs = num_epochs
        self.model_name = model_name
        self.properties = properties
        self.split_ratio = split_ratio
        self.initial_train_size = initial_train_size
        self.initial_validation_size = initial_validation_size
        self.initial_test_size = initial_test_size
        self.num_iterations = num_iterations

    def dataset_split(self):
        data = np.load(self.initial_dataset)
        data_1 = {}
        data_2 = {}
        for key in data:
            array = data[key]
            split_idx = int(len(array) * self.split_ratio)
            array_1 = array[:split_idx]
            array_2 = array[split_idx:]
            data_1[key] = array_1
            data_2[key] = array_2
        np.savez('initial_train.npz', **data_1)
        np.savez('initial_test.npz', **data_2)
        print(f"Splitting initial data into two separate files was performed successfully.")

    def process_npz_to_dat(self, npz_file, dat_file, en_file):
        data = np.load(npz_file)
        N = data['N']
        R = data['R']
        Z = data['Z']
        E = data['E']

        with open(dat_file, 'w') as file:
            for idx in range(len(N)):
                N_line = ' '.join(map(str, N[idx])) if isinstance(N[idx], np.ndarray) else str(N[idx])
                file.write(N_line + '\n\n')

                for j in range(R[idx].shape[0]):
                    Z_str = {1: 'H', 6: 'C', 8: 'O', 0: 'X'}.get(Z[idx, j], 'X')
                    row_str = ' '.join(map(str, [Z_str] + list(R[idx][j])))
                    if '0.0 0.0 0.0' not in row_str:
                        file.write(row_str + '\n')

        with open(en_file, 'w') as file:
            for energy in E:
                E_line = ' '.join(map(str, energy)) if isinstance(energy, np.ndarray) else str(energy)
                file.write(E_line + '\n')

    def calc_energy_for_testing(self, model_config, struct_file, out_file):
        model = Asparagus(config=model_config)
        calc = model.get_ase_calculator()
        structures = read(struct_file, index=':')
        print("Calculating energies")
        with open(out_file, 'w') as out_f:
            for atom in structures:
                atom.calc = calc
                e = atom.get_potential_energy()
                calc.reset()
                out_f.write(str(e) + '\n')

    def estimation_extraction(self, train_npz, test_npz, calc_data, ref_data, N, i):
        npz1 = np.load(train_npz)
        npz2 = np.load(test_npz)
        val1 = np.loadtxt(calc_data)
        val2 = np.loadtxt(ref_data)
        diff = val1 - val2
        mean_diff = np.mean(diff)
        indices = np.argsort(np.abs(diff))[-N:]

        combined_data = {key: np.concatenate((npz2[key][indices], npz1[key]), axis=0) for key in npz2.files}
        np.savez(f"train_set_{i}_iteration.npz", **combined_data)

        new_test_data = {key: np.delete(npz2[key], indices, axis=0) for key in npz2.files}
        np.savez(f"test_set_{i}_iteration.npz", **new_test_data)
        print(f"Mean difference for iteration {i}: {mean_diff}")

    def run(self):
        if not os.path.exists('initial_train.npz'):
            self.dataset_split()

        if not os.path.exists(f"{self.model_name}.json"):
            model = Asparagus(
                config=f"{self.model_name}.json",
                data_source="initial_train.npz",
                data_file="initial_train.db",
                model_directory=f"{self.model_name}_0_iteration",
                model_properties=self.properties,
                data_num_train=self.initial_train_size,
                data_num_valid=self.initial_validation_size,
                data_num_test=self.initial_test_size,
                trainer_max_epochs=self.num_epochs,
                data_seed=np.random.randint(1E6),
                trainer_evaluate_testset=False,
            )
            model.train()

        with open(f"{self.model_name}.json", 'r') as file:
            for line in file:
                if '"data_seed":' in line:
                    match = re.search(r'"data_seed":\s*(\d+)', line)
                    if match:
                        data_seed_value = int(match.group(1))
                    break

        if not os.path.exists("ref_en_0_iteration.dat"):
            self.process_npz_to_dat('initial_test.npz', 'structures_0_iteration.xyz', 'ref_en_0_iteration.dat')

        if not os.path.exists("pred_energies_0_iteration.dat"):
            self.calc_energy_for_testing(f"{self.model_name}.json", 'structures_0_iteration.xyz', 'pred_energies_0_iteration.dat')

        self.estimation_extraction('initial_train.npz', 'initial_test.npz', 'pred_energies_0_iteration.dat', 'ref_en_0_iteration.dat', 1000, 1)

        for i in range(1, self.num_iterations + 1):
            if not os.path.exists(f"{self.model_name}_{i}_iteration.json"):
                model = Asparagus(
                    config=f"{self.model_name}_{i}_iteration.json",
                    data_file=f"{self.model_name}_{i}_iteration.db",
                    data_source=f"train_set_{i}_iteration.npz",
                    model_directory=f"{self.model_name}_{i}_iteration",
                    model_properties=['energy', 'forces'],
                    data_num_train=0.9,
                    data_num_valid=0.1,
                    trainer_evaluate_testset=False,
                    model_checkpoint=f"{self.model_name}_{i-1}_iteration/best/best_model.pt",
                    trainer_max_epochs=100,
                    data_seed=data_seed_value,
                )
                model.train(reset_best_loss=True)

                self.process_npz_to_dat(f"test_set_{i}_iteration.npz", f"structures_{i}_iteration.xyz", f"ref_en_{i}_iteration.dat")
                self.calc_energy_for_testing(f"{self.model_name}_{i}_iteration.json", f"structures_{i}_iteration.xyz", f"pred_energies_{i}_iteration.dat")
                self.estimation_extraction(f"train_set_{i}_iteration.npz", f"test_set_{i}_iteration.npz", f"pred_energies_{i}_iteration.dat", f"ref_en_{i}_iteration.dat", 1000, i + 1)



if __name__ == "__main__":
    adaptive_sampling = AdaptiveSampling(input_file='formic_mp2.npz')
    adaptive_sampling.run()
