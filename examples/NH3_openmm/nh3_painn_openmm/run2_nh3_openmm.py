from sys import stdout

from openmm import LangevinMiddleIntegrator, Platform, Vec3
from openmm.app import (
    CharmmPsfFile, PDBFile, CharmmParameterSet,
    Simulation, PME, HBonds,
    DCDReporter, StateDataReporter)
from openmm.unit import kelvin, picosecond, nanometer, angstrom, degree

from asparagus import Asparagus

# Read the PSF
psf = CharmmPsfFile('charmm_data/ammonia_water.psf')

# Get the coordinates from the PDB
pdb = PDBFile('charmm_data/ammonia_water.pdb')

positions = pdb.getPositions()

# Load the parameter set.
params = CharmmParameterSet(
    'charmm_data/nh3_water.top',
    'charmm_data/nh3_water.par')

# MM System

box = 30.0
angle = 90.0
psf.setBox(
    a=box*angstrom,
    b=box*angstrom,
    c=box*angstrom,
    alpha=angle*degree,
    beta=angle*degree,
    gamma=angle*degree,
)
hbox = Vec3(box/2., box/2., box/2.)*angstrom
for atom, pos in enumerate(positions):
    positions[atom] += hbox

mm_system = psf.createSystem(
    params,
    nonbondedMethod=PME,
    nonbondedCutoff=12.0*angstrom,
    switchDistance=2.0*angstrom,
    constraints=HBonds)

# ML System

# Define ML atoms (First chain/segid)
residues = list(pdb.topology.residues())
ml_atoms = [atom.index for atom in residues[0].atoms()]

# Define PaiNN potential
model = Asparagus(
    config='model_nh3/config.json',
    device='cuda',
    )
ml_potential = model.get_openmm_potential()

# Define ML/MM system
mlmm_system = ml_potential.createMixedSystem(
    pdb.topology,
    mm_system,
    ml_atoms,
    ml_charge=0.0,
    ml_fluctuating_charges=True,
    removeConstraints=False,
    )

# Instantiate the integrator and simulation
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picosecond)

platform = Platform.getPlatform('CUDA')
properties = {'DeviceIndex': '0', 'Precision': 'mixed'}
#platform = Platform.getPlatform('CPU')
#properties = {}

simulation = Simulation(psf.topology, mlmm_system, integrator, platform, properties)
simulation.context.setPositions(pdb.getPositions())

simulation.minimizeEnergy()
PDBFile.writeFile(
    simulation.topology,
    simulation.context.getState(positions=True).getPositions(),
    'charmm_data/ammonia_water_opt.pdb')

simulation.reporters.append(DCDReporter('charmm_data/output.dcd', 10))

simulation.reporters.append(
    StateDataReporter(
        stdout,
        100,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        speed=True)
    )

simulation.step(10_000)
