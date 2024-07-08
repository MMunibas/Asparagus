import os
import sys
import json

import numpy as np

import xml.etree.ElementTree as et

from ase import units

# Function to store results in json format
result_file = "results.json"
def save_results(results, result_file):
    """
    Save result dictionary as json file
    """
    with open(result_file, 'w') as f:
        json.dump(results, f)


# Molpro output XML file
molpro_xml = sys.argv[1]
if not os.path.exists(molpro_xml):
    save_results({}, result_file)
    exit()

# Open XML file
tree = et.parse(molpro_xml)
root = tree.getroot()

# Get Molpro attribute
molpro_out = root[0]

# Iterate over job results
for job in molpro_out:

    # Read MP2 results
    if (
        job.get('command') is not None 
        and job.get('command').lower() == 'MP2'.lower()
    ):

        # Loop over results
        for details in job:
            
            # Read total energy
            if (
                'property' in details.tag 
                and 'name' in details.attrib
                and 'total energy'.lower() == details.attrib['name'].lower()
            ):
                if details.attrib.get('value') is None:
                    energy = None
                else:
                    energy = float(details.attrib.get('value'))*units.Hartree

            # Read dipole moment
            if (
                'property' in details.tag 
                and 'name' in details.attrib
                and 'Dipole moment'.lower() == details.attrib['name'].lower()
            ):
                if details.attrib.get('value') is None:
                    dipole = None
                else:
                    dipole = [
                        float(value)*units.Bohr
                        for value in details.attrib.get('value').split()]

    # Read MP2 gradient
    if (
        job.get('command') is not None 
        and job.get('command').lower() == 'FORCES'.lower()
    ):

        # Loop over results
        for details in job:
            
            # Read forces
            if (
                'gradient' in details.tag 
                and 'name' in details.attrib
                and 'MP2 GRADIENT'.lower() == details.attrib['name'].lower()
            ):
                forces = []
                for line in details.text.split('\n'):
                    if len(line.strip()):
                        forces.append([
                            -float(value)*units.Hartree/units.Bohr
                            for value in line.split()])

# Collect results
results = {
    'energy': energy,
    'forces': forces,
    'dipole': dipole,
    }

# Save results to result file
save_results(results, result_file)
