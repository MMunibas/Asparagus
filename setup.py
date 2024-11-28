import sys
from setuptools import setup, find_packages

with open('README.md','r') as fh:
    long_description = fh.read()

setup(
    name='Asparagus',
    version='0.4.2',
    description='Function Bundle from Sampling, Training to Application of NN Potentials',
    author=(
        'L.I.Vazquez-Salazar, Silvan Kaeser, '
        + 'Valerii Andreichev and Kai Toepfer'),
    long_description=long_description,
    author_email='luisitza.vazquezsalazar@unibas.ch',
    url='https://github.com/MMunibas/Asparagus/releases',
    license='MIT',
    packages=find_packages(include=['asparagus']),
    include_package_data=True,
    install_requires=[
        'ase>=3.21.0',
        'numpy',
        'scipy',
        'ctype',
        'torch>2.1',
        'torchvision',
        'torchaudio',
        'torch-ema>=0.3',
        'tensorboard',
        'pandas',
        'h5py',
        'matplotlib',
        'seaborn',
        'pytest']
    #TODO: Add more dependencies and option to be read from a file
)
