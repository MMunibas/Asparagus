#!/bin/bash
#SBATCH --job-name=molpro_test
#SBATCH --partition=vshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000

#---------
# Modules
#---------

# PC-Beethoven
module load molpro/molpro2024-gcc-9.2.0
# PC-Bach
#module load molpro/molpro-2024.1

#------------
# Run Molpro
#------------

rm -f run_molpro.xml
molpro -o run_molpro.out run_molpro.inp

#-------------
# Read Result
#-------------

python run_molpro.py run_molpro.xml

