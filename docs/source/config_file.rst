Configuration File Structure
=============================

The heart of asparagus is the configuration file. It saves the details of what asparagus did and what is going to do.
You do not need to generate it by hand, asparagus will do it for you, we take care of that. |:wink:|
Nevertheless, it is useful to know how it is structured.

The config file is a JSON file, that usually look like this:


.. code-block:: rst

    {
        "config_file": "nh3_config_nms.json",
        "sample_data_file": "sampling_nh3_nms/nms_nh3.db",
        "sample_directory": "sampling_nh3_nms",
        "sample_systems": [
            "data/nh3_c3v.xyz"
        ],
        "sample_systems_format": "xyz",
        "sample_calculator": "XTB",
        "sample_calculator_args": {},
        "sample_properties": [
            "energy",
            "forces",
            "dipole"
        ],
        "sample_systems_optimize": true,
        "sample_systems_optimize_fmax": 0.001,
        "sample_data_overwrite": false,
        "sample_tag": "nmscan",
        "nms_harmonic_energy_step": 0.03,
        "nms_energy_limits": 1.0,
        "nms_number_of_coupling": 2,
        "nms_limit_of_steps": 10,
        "nms_limit_com_shift": 0.1,
        "sampler_schedule": {
            "1_nmscan": {
                "sample_data_file": "sampling_nh3_nms/nms_nh3.db",
                "sample_directory": "sampling_nh3_nms",
                "sample_systems": [
                    "data/nh3_c3v.xyz"
                ],
                "sample_systems_format": [
                    "xyz"
                ],
                "sample_calculator": "XTB",
                "sample_calculator_args": {},
                "sample_properties": [
                    "energy",
                    "forces",
                    "dipole"
                ],
                "sample_systems_optimize": true,
                "sample_systems_optimize_fmax": 0.001,
                "sample_data_overwrite": false,
                "nms_harmonic_energy_step": 0.03,
                "nms_energy_limits": [
                    -1.0,
                    1.0
                ],
                "nms_number_of_coupling": 2,
                "nms_limit_com_shift": 0.1,
                "nms_limit_of_steps": 10
            },
            "sample_systems_idx": null,
            "2_nmscan": {
                "sample_data_file": "sampling_nh3_nms/nms_nh3.db",
                "sample_directory": "sampling_nh3_nms",
                "sample_systems": [
                    "data/nh3_c3v.xyz"
                ],
                "sample_systems_format": [
                    "xyz"
                ],
                "sample_calculator": "XTB",
                "sample_calculator_args": {},
                "sample_properties": [
                    "energy",
                    "forces",
                    "dipole"
                ],
                "sample_systems_optimize": true,
                "sample_systems_optimize_fmax": 0.001,
                "sample_data_overwrite": false,
                "nms_harmonic_energy_step": 0.03,
                "nms_energy_limits": [
                    -1.0,
                    1.0
                ],
                "nms_number_of_coupling": 2,
                "nms_limit_com_shift": 0.1,
                "nms_limit_of_steps": 10
            }
        },
        "sample_counter": 2,
        "data_file": "sampling_nh3_nms/nms_nh3.db",
        "model_directory": "sampling_nh3_nms",
        "model_properties": [
            "energy",
            "forces",
            "dipole",
            "atomic_charges"
        ],
        "model_interaction_cutoff": 8.0,
        "trainer_properties_weights": {
            "energy": 1.0,
            "forces": 50.0,
            "dipole": 25.0,
            "else": 1.0
        },
        "trainer_max_epochs": 1000,
        "data_source": [],
        "data_format": [],
        "data_alt_property_labels": {},
        "data_unit_positions": "Ang",
        "data_load_properties": [
            "energy",
            "forces",
            "dipole"
        ],
        "data_unit_properties": {
            "energy": "eV",
            "forces": "eV/Ang",
            "dipole": "eAng",
            "charge": "e",
            "positions": "Ang",
            "atomic_charges": "e"
        },
        "data_num_train": 888,
        "data_num_valid": 111,
        "data_num_test": 112,
        "data_train_batch_size": 128,
        "data_valid_batch_size": 128,
        "data_test_batch_size": 128,
        "data_num_workers": 1,
        "data_workdir": ".",
        "data_overwrite": false,
        "data_seed": 664521,
        "model_properties_scaling": {
            "energy": [
                0.09574281780168606,
                -29.932318049978853
            ],
            "forces": [
                2.5735118188374284,
                -1.6977695174545036e-18
            ],
            "dipole": [
                0.17839811105704,
                -0.10566497769812809
            ]
        },
        "model_type": "PhysNet",
        "model_unit_properties": {
            "energy": "eV",
            "forces": "eV/Ang",
            "dipole": "eAng",
            "charge": "e",
            "positions": "Ang",
            "atomic_charges": "e"
        },
        "model_cutoff_width": 2.0,
        "model_repulsion": false,
        "model_electrostatic": true,
        "model_dispersion": true,
        "model_dispersion_trainable": true,
        "input_type": "PhysNetRBF",
        "input_n_atombasis": 128,
        "input_n_radialbasis": 64,
        "input_cutoff_descriptor": 8.0,
        "input_cutoff_fn": "Poly6",
        "input_rbf_center_start": 1.0,
        "input_rbf_trainable": true,
        "input_n_maxatom": 94,
        "input_atom_features_range": 1.7320508075688772,
        "graph_type": "PhysNetMP",
        "graph_n_blocks": 5,
        "graph_n_residual_interaction": 3,
        "graph_n_residual_atomic": 2,
        "output_type": "PhysNetOut",
        "output_n_residual": 1,
        "output_properties": [
            "energy",
            "forces",
            "dipole",
            "atomic_charges"
        ],
        "trainer_restart": false,
        "trainer_properties_train": [],
        "trainer_properties_metrics": {
            "else": "MSE",
            "energy": "MSE",
            "forces": "MSE",
            "dipole": "MSE"
        },
        "trainer_optimizer": "AMSgrad",
        "trainer_optimizer_args": {
            "lr": 0.001,
            "weight_decay": 1e-05,
            "amsgrad": true
        },
        "trainer_scheduler": "ExponentialLR",
        "trainer_scheduler_args": {
            "gamma": 0.999
        },
        "trainer_ema": true,
        "trainer_ema_decay": 0.99,
        "trainer_max_gradient_norm": 1000.0,
        "trainer_save_interval": 5,
        "trainer_validation_interval": 5,
        "trainer_evaluate_testset": true,
        "trainer_max_checkpoints": 1,
        "trainer_store_neighbor_list": false,
        "test_datasets": "all",
        "test_store_neighbor_list": false
    }

It looks overwhelming, but it is not. Let's go through it step by step.

-----------------
General
-----------------

The first part of the config file is the general part. It contains the following keys:

- `config_file`: The name of the config file.
- `sample_data_file`: The name of the sample data file. This is where the data is saved and loaded from. Usually, a database file.
- `sample_directory`: The name of the sample directory. This is where the sample data is saved if you do sampling of a molecular structure.
- `sample_systems`: A list of molecular structures that are used for sampling.
- `sample_systems_format`: The format of the molecular structures.
- `sample_calculator`: The calculator that is used for sampling. This usually is a calculator that can calculate the energy and forces of a molecular structure from *ab initio*.
- `sample_calculator_args`: The arguments that are passed to the calculator. This is a dictionary of keywords that might be needed by the electronic structure code.
- `sample_properties`: The properties that are calculated by the calculator. This is a list of strings.
- `sample_systems_optimize`: If the molecular structures should be optimized before sampling.
- `sample_systems_optimize_fmax`: The maximum force that is allowed during optimization.
- `sample_data_overwrite`: If the sample data should be overwritten.
- `sample_tag`: A tag that is added to the sample data file name.

---------
Sampling
---------

The next part of the config file is the sampling part. It is specific to the method that is used for sampling. In this
example, it is the NMS method. For more information about the options of the sampling methods, please refer to the
documentation of the sampling methods.

**Note**: By default, asparagus used XTB for sampling. If you want to use another calculator, you need to specify it.

---------
Model
---------

The next part of the config file is the model part. It setup the model that will be trained. It contains the following
keys:

- `data_unit_properties`: The units of the properties in the data file. By default, the units are the same as in ASE.
- `data_num_train`: The number of training samples. Asparagus automatically splits the data into training, validation, and test set (80%, 10%, 10%).
- `data_num_valid`: The number of validation samples.
- `data_num_test`: The number of test samples.
- `data_train_batch_size`: The batch size for training.
- `data_valid_batch_size`: The batch size for validation.
- `data_test_batch_size`: The batch size for testing.
- `data_num_workers`: The number of workers that are used for loading the data. Note for large databases, you should increase this number.
- `data_workdir`: The working directory for the data. It is the folder where asparagus will look for the data.
- `data_overwrite`: If the data should be overwritten.
- `data_seed`: The seed for the random number generator. This is important for reproducibility because of the random spliting of the data performed by asparagus.
- `model_properties_scaling`: The scaling of the properties. The properties are normalized by taking the mean and standard deviation of the training set. This models are learn by asparagus and saved in this dictionary.
- `model_type`: The type of the model. For the moment, only PhysNet is supported.
- `model_unit_properties`: The units of the properties in the model. By default, the units are the same as in ASE.
- `model_cutoff_width`: The width of the cutoff function for the descriptor.
- `model_repulsion`: If the repulsion term is used.
- `model_electrostatic`: If the electrostatic corrections are used.
- `model_dispersion`: If the dispersion corrections are used. For the moment, only the D3 dispersion correction is supported.
- `model_dispersion_trainable`: If the dispersion correction is trainable. Otherwise, the parameters are fixed.
- `input_type`: The type of the input layer. This is the layer that transforms the molecular structure into a descriptor. For the moment, only RBF are supported.
- `input_n_atombasis`: Size of the feature vector.
- `input_n_radialbasis`: Number of radial basis functions.
- `input_cutoff_descriptor`: The cutoff for the descriptor at short distances.
- `input_cutoff_fn`: The cutoff function for the descriptor.
- `input_rbf_center_start`: The starting point for the radial basis functions.
- `input_rbf_trainable`: If the radial basis functions are trainable.
- `input_n_maxatom`: The maximum number of atoms in the molecular structure.
- `input_atom_features_range`: The range of the atom features. This is used to normalize the atom features.
- `graph_type`: The type of the graph layer. This is the layer that transforms the descriptor into a graph. For the moment, only Physnet Message Passing is supported.
- `graph_n_blocks`: The number of blocks in the graph layer. This is the number of blocks used by the network.
- `graph_n_residual_interaction`: The number of residual interactions in the graph layer.
- `graph_n_residual_atomic`: The number of residual atomic in the graph layer.
- `output_type`: The type of the output layer. This is the layer that transforms the graph into the properties. For the moment, only PhysNet Output is supported.
- `output_n_residual`: The number of residual layers in the output layer.
- `output_properties`: The properties that are predicted by the model.

---------
Trainer
---------

The next part of the config file is the trainer part. It contains the following keys:

- `trainer_restart`: If the training should be restarted from the last checkpoint.
- `trainer_properties_train`: The properties that are trained. If empty, all properties are trained (Energy, Forces, dipole, charges).
- `trainer_properties_metrics`: The metrics that are used for the properties.
- `trainer_optimizer`: The optimizer that is used for training.
- `trainer_optimizer_args`: The arguments that are passed to the optimizer. Here you can set the learning rate, weight decay, or other parameters of the optimizer.
- `trainer_scheduler`: The scheduler that is used for training.
- `trainer_scheduler_args`: The arguments that are passed to the scheduler. Here you can set the learning rate decay.
- `trainer_ema`: If the exponential moving average is used.
- `trainer_ema_decay`: The decay of the exponential moving average.
- `trainer_max_gradient_norm`: The maximum gradient norm. This is used to clip the gradients.
- `trainer_save_interval`: The interval at which the model is saved.
- `trainer_validation_interval`: The interval at which the model is validated.
- `trainer_evaluate_testset`: If the test set is evaluated during training.
- `trainer_max_checkpoints`: The maximum number of checkpoints that are saved.
- `trainer_store_neighbor_list`: If the neighbor list is stored during training.






