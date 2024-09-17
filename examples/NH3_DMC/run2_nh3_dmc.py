from asparagus import Asparagus
from asparagus.tools import DMC

from ase import io


# Initialize Asparagus
model = Asparagus(
    config='nh3_md.json'
    )

# Read initial and reference structure
nh3_initial = io.read('nh3_d3h.xyz')
nh3_reference = io.read('nh3_c3v.xyz')

# Initialize DMC class
dmc = DMC(
    nh3_initial,
    model,
    charge=0,
    optmize=True,
    initial_positions=nh3_initial,
    )
dmc.run()
