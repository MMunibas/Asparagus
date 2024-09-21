from asparagus import Asparagus
from asparagus.tools import DMC

from ase import io


# Initialize Asparagus
model = Asparagus(
    config='nh3_md.json'
    )

# Read initial and reference structure
nh3_reference = io.read('nh3_c3v.xyz')
nh3_initial = io.read('nh3_d3h.xyz')

# Initialize DMC class
dmc = DMC(
    nh3_reference,
    model,
    charge=0,
    optmize=True,
    initial_positions=nh3_initial,
    )
dmc.run(
    nsteps=1000,
    eqsteps=100,
    )
