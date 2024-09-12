# Project Asparagus

**Authors**: K. Toepfer, L.I. Vazquez-Salazar

<img src="https://github.com/LIVazquezS/Asparagus/blob/main/data/logo.png" width="50%">

## What is this?
 - A refined implementation of PhysNet NN (and other atomistic NN to come) in PyTorch. 
 - A Suit for the automatic construction of Potential Energy Surface (PES) from sampling to production.

## How to use? 

- Clone the repository
- Requirements:
  - Python <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.8 (I recommend to use 3.8.12, **DO NOT** use 3.9)
  - PyTorch <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 1.10
  - Torch-ema <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 0.3
  - TensorBoardX <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 2.4
  - Atomic Simulation Environment (ASE) <img src="https://latex.codecogs.com/svg.image?\geq&space;" title="\geq " /> 3.21
  - xtb
  - pandas
  - scipy
  
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
Alternatively, but deprecated, install via setup.py:
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

- [ ] Add more NN architectures (Low priority)
- [x] Read parameters from older PhysNet Versions (i.e. TF1 and TF2) (Luis)
- [ ] Add sampling methods:
    - [x] MD with XTB
    - [x] MC with XTB
    - [x] Normal Model Sampling (Vanilla with random generation) 
    - [x] Normal Model Scanning 
    - [ ] Umbrella Sampling (Low priority)
    - [x] Metadynamics Sampling 
- [ ] Electronic structure calculations:
   - [x] ASE calculator (As good as it can be)
   - [ ] Automatic generation of input files for commonly used codes (e.g. Gaussian, Orca, MOLPRO, etc.)
   - [ ] Automatic extraction of information from output files
- Trainer class:
  - [ ] Training of model ensemble 
- Tester class: 
  - [x] Finish automatic evaluation 
- Active learning
   - [ ] Adaptive Sampling
   - [ ] Uncertainty calculations
     - [x] Model ensemble via ASE calculator 
     - [ ] Deep Evidential Regression (Low priority)
- Tools class:
  - [x] Normal mode calculation (Luis)
  - [x] Minimum energy path and Minimum dynamic path
  - [x] Diffusion MonteCarlo
  - [ ] Others(?)
- Production: 
  - [x] PyCharmm 
  - [x] ASE calculator for dynamics
- Documentation:
  - [X] Improve documentation
  - [x] Add examples
  - [x] Add tutorials
  
## Contact

For any questions, please open an issue in the repository.
