# Project Asparagus

**Authors**: K. Toepfer, L.I. Vazquez-Salazar

<img src="https://github.com/LIVazquezS/Asparagus/blob/main/data/logo.png" width="50%">

## What is this?
 - A refined implementation of PhysNet and PainNN (and more atomistic NN to come) in PyTorch. 
 - A Suit for the automatic construction of Potential Energy Surface (PES) from sampling to production.

## How to use? 

- Clone the repository
- Requirements:
  - Python <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.8
  - PyTorch <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 1.10
  - Atomic Simulation Environment (ASE) <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " == 3.22.1 ( 3.23.0, 3.24.0 is broken when using ORCA calculator)
  - Torch-ema <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 0.3
  - TensorBoardX <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 2.4
  - numpy, scipy, pandas, ...
  
### Setting up the environment

We recommend to use [Mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) for the creation of a virtual environment. 

Once in mamba, you can create a virtual enviroment called *asparagus* 

``` 
mamba create --name asparagus python=3.8
```
 
To activate the virtual environment use the command:

```
mamba activate asparagus
```

### Installation
Installation must be done in the virtual environment through pip. It is important to mention that the path where you are
working will be added to the *PYTHONPATH*, so you can import the modules from anywhere.

Install via pip:
``` 
python -m pip install .
```
Alternatively, but deprecated and will probably fail, install via setup.py:
``` 
python setup.py install
```

**BEWARE**: With this command any modification that is done to the code in the folder *asparagus* will be automatically reflected 
in the modules that you import.

**NOTE**: Everytime you want to import the module, you must use the following command:

```
from asparagus import Asparagus
```
Then Asparagus is a function that takes some arguments.

## Documentation

Please check our documentation [here](http://asparagus-bundle.readthedocs.io/en/latest/)

## What needs to be added?

- [x] Add more NN architectures (Low priority)
- [x] Add sampling methods:
    - [x] MD
    - [x] MC
    - [x] Normal Model Sampling
    - [x] Normal Model Scanning 
    - [x] Metadynamics Sampling
    - [ ] Umbrella Sampling (Low priority)
- [x] Electronic structure calculations:
   - [x] ASE calculator
   - [x] Custom Calculators (via shell run or slurm batch files)
- Trainer class:
  - [x] Training of model ensemble 
- Tester class: 
  - [x] Model and Moddel Ensemble automatic evaluation 
- Active learning
   - [x] Adaptive Sampling (first iteration)
   - [x] Uncertainty calculations
     - [x] via Model Ensembles
     - [ ] via Deep Evidential Regression (Low priority)
- Tools class:
  - [x] Normal mode calculation (via ASE)
  - [x] Minimum energy path and Minimum dynamic path
  - [x] Diffusion Monte-Carlo
  - [ ] Others(?)
- Production: 
  - [x] ASE calculator
  - [x] PyCharmm interface
  - [ ] Gaussian interface
- Documentation:
  - [ ] Improve documentation
  - [x] Add examples
  - [ ] Add tutorials
 - Others
   - [ ] Create a conda package
   - [ ] Create a pip package

## Contact

For any questions, please open an issue in the repository.

## How to cite
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2407.15175)
[![Comp. Phys. Comm.](https://img.shields.io/badge/Paper-CompPhysComm-blue?logo=elsevier&link=https%3A%2F%2Fdoi.org%2F10.1016%2Fj.cpc.2024.109446)](https://doi.org/10.1016/j.cpc.2024.109446)


If you find this work useful in your research, please cite it as: 
```latex
@article{asparagus_cpc,
title = {Asparagus: A toolkit for autonomous, user-guided construction of machine-learned potential energy surfaces},
author = {Kai Töpfer and Luis Itza Vazquez-Salazar and Markus Meuwly},
journal = {Computer Physics Communications},
volume = {308},
pages = {109446},
year = {2025},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2024.109446},
url = {https://www.sciencedirect.com/science/article/pii/S0010465524003692},
keywords = {Machine learning, Neural networks, Potential energy surfaces},
}
```

