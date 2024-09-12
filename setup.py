import sys
from setuptools import setup, find_packages

with open('README.md','r') as fh:
    long_description = fh.read()

setup(
    name='Asparagus',
    version='0.3.0',
    description='Function Bundle from Sampling, Training to Application of NN Potentials',
    author='L.I.Vazquez-Salazar, Silvan Kaeser and Kai Toepfer',
    long_description=long_description,
    author_email='luisitza.vazquezsalazar@unibas.ch',
    url='https://github.com/LIVazquezS/Asparagus/tree/main',
    license='MIT',
    packages=find_packages(include=['asparagus']),
    include_package_data=True,
    install_requires=[
        'ase', 
        'numpy',
        'scipy',
        'ctype',
        'torch',
        'torchvision',
        'torchaudio',
        'torch-ema',
        'tensorboard',
        'pandas',
        'h5py',
        #'xtb',
        ]
    #TODO: Add more dependencies and option to be read from a file
)
