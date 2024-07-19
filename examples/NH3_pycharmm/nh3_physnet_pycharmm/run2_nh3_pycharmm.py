# Test Script to import PhysNet as energy function in CHARMM via PyCHARMM

# Basics
import os
import sys
import ctypes
import pandas
import numpy as np

# ASE
from ase import Atoms
from ase import io
import ase.units as units

# PyCHARMM
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.lingo as stream
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.cons_fix as cons_fix
import pycharmm.cons_harm as cons_harm
from pycharmm.lib import charmm as libcharmm
import pycharmm.lib as lib

# Asparagus
sys.path.insert(0, '/home/toepfer/Documents/Project_PhysNet3/Asparagus')
from asparagus import Asparagus

# Step 0: Load parameter files
#-----------------------------------------------------------

stream.charmm_script("""
! protein topology and parameter
open read card unit 10 name charmm_data/nh3_water.top
read  rtf card unit 10

open read card unit 20 name charmm_data/nh3_water.par
read para card unit 20 flex
""")

settings.set_bomb_level(-2)
settings.set_warn_level(-1)

# Step 1: Generate System
#-----------------------------------------------------------

# Should ammonia be solvated in water:
solvation = True
# Two ways to setup ammonia are available (= 1 or = 2)
setup = 1

if setup == 1:

    # Read ML system
    read.psf_card("charmm_data/ammonia.psf")
    read.pdb("charmm_data/ammonia.pdb")

    # Manipulate coordinates
    pos = coor.get_positions().to_numpy(dtype=np.float32)
    pos[:, 0] -= 10.0 # Shift ammonia to the cell border to test pbc
    pandas_pos = pandas.DataFrame(
        {'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]}
        )
    coor.set_positions(pandas_pos)

    # Read solvent
    if solvation:

        # Add TIP3P water residues
        read.sequence_pdb("charmm_data/water_box.pdb") 
        add_water = """
        generate WAT setup noang nodihe
        open read card unit 10 name charmm_data/water_box.pdb
        read coor pdb  unit 10 resid
        """
        stream.charmm_script(add_water)
        
        # Delete overlapping water molecules
        remove_water = """
        delete atom select ( .byres. ( (segid AMM1 .around. 2.0 ) -
            .and. (segid WAT .and. type OH2 ))) end
        """
        stream.charmm_script(remove_water)

elif setup == 2:

    # Read solvent
    if solvation:

        # Add TIP3P water residues
        add_water = """
        open read card unit 10 name charmm_data/water_box.pdb
        read sequence pdb unit 10
        generate WAT setup noang nodihe

        open read card unit 10 name charmm_data/water_box.pdb
        read coor pdb  unit 10 resid
        """
        stream.charmm_script(add_water)


    # Read ML system
    add_ammonia = """
    open read card unit 10 name charmm_data/ammonia.pdb
    read sequence pdb unit 10
    generate AMM1 setup warn first none last none

    open read card unit 10 name charmm_data/ammonia.pdb
    read coor pdb  unit 10 resid
    """
    stream.charmm_script(add_ammonia)

    # Manipulate coordinates
    pos = coor.get_positions().to_numpy(dtype=np.float32)
    pos[-4:, 0] -= 10.0 # Shift ammonia to the cell border to test pbc
    pandas_pos = pandas.DataFrame(
        {'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]}
        )
    coor.set_positions(pandas_pos)

    if solvation:
        
        # Delete overlapping water molecules
        remove_water = """
        delete atom select ( .byres. ( (segid AMM1 .around. 2.0 ) -
            .and. (segid WAT .and. type OH2 ))) end
        """
        stream.charmm_script(remove_water)



write.coor_pdb("charmm_data/ammonia_water.pdb", title="Ammonia solvated")
write.coor_card("charmm_data/ammonia_water.crd", title="Ammonia solvated")
write.psf_card("charmm_data/ammonia_water.psf", title="Ammonia solvated")

# ASE atoms object (just to get atomic numbers)
ase_ammonia = io.read("charmm_data/ammonia.pdb", format="proteindatabank")

# Step 2: CHARMM Setup
#-----------------------------------------------------------

# Non-bonding parameter
dict_nbonds = {
    'atom': True,
    'vdw': True,
    'vswitch': True,
    'cutnb': 14,
    'ctofnb': 12,
    'ctonnb': 10,
    'cutim': 14,
    'lrc': True,
    'inbfrq': -1,
    'imgfrq': -1
    }
nbond = pycharmm.NonBondedScript(**dict_nbonds)
nbond.run()

if solvation:
    
    # PBC box
    crystal.define_cubic(length=30.0)
    crystal.build(cutoff=14.0)

    stream.charmm_script('image byres xcen 0.0 ycen 0.0 zcen 0.0 sele all end')

    # H-bonds constraint
    #shake.on(bonh=True, tol=1e-7)
    stream.charmm_script('shake bonh para sele resname TIP3 end')

else:

    # Default Setup
    pass

# Energy
energy.show()

# Step 3: Asparagus Setup
#-----------------------------------------------------------

# Load Asparagus model
ml_model = Asparagus(config='model_nh3/nh3_meta.json')

# Get atomic number from ASE atoms object
ml_Z = ase_ammonia.get_atomic_numbers()

# Prepare PhysNet input parameter
ml_selection = pycharmm.SelectAtoms(seg_id='AMM1')

# Initialize PhysNet calculator
calc = pycharmm.MLpot(
    ml_model,
    ml_Z,
    ml_selection,
    ml_charge=0,
    ml_fq=True,
)

# Custom energy
energy.show()

# Step 4: Minimization
#-----------------------------------------------------------

file_mini_pdb = "charmm_data/mini_ammonia.pdb"
file_mini_crd = "charmm_data/mini_ammonia.crd"

if False or not os.path.exists(file_mini_crd):

    # Fix ML atoms
    cons_fix.setup(pycharmm.SelectAtoms(seg_id='AMM1'))

    # Optimization with PhysNet parameter
    minimize.run_sd(**{
        'nstep': 500,
        'nprint': 10,
        'tolenr': 1e-5,
        'tolgrd': 1e-5})

    # Unfix ML atoms
    cons_fix.turn_off()
    
    # Optimization with PhysNet parameter
    minimize.run_sd(**{
        'nstep': 500,
        'nprint': 10,
        'tolenr': 1e-5,
        'tolgrd': 1e-5})

    # Write pdb file
    write.coor_pdb(file_mini_pdb, title="Mini SD")
    write.coor_card(file_mini_crd, title="Mini SD")

else:
    
    # Read optimized coordinates - Do not read from pdb files
    # as it yield a weird non-bonding atom pair bug where ML atoms
    # are not excluded from non-bonding interaction.
    read.coor_card(file_mini_crd)

# Minimized custom energy
energy.show()

# Step 5: Heating - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.00025   # 0.25 fs
    nsteps = 10.*1./timestep # 10 ps
    nsavc = 0.100*1./timestep # every 100 fs
    temp = 300.0

    res_file = pycharmm.CharmmFile(
        file_name='charmm_data/heat.res', file_unit=2, 
        formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='charmm_data/heat.dcd', file_unit=1, 
        formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'verlet': True,
        'new': True,
        'start': True,
        'timestep': timestep,
        'nstep': nsteps,
        'nsavc': nsavc,
        'inbfrq': -1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 1000,
        'ihtfrq': 200,
        'ieqfrq': 1000,
        'firstt': temp/2.,
        'finalt': temp,
        'tbath': temp,
        'echeck':-1}

    dyn_heat = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_heat.run()
    
    res_file.close()
    dcd_file.close()

# Step 6: NVE - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.00025   # 0.2 fs
    nsteps = 50.*1./timestep # 50 ps
    nsavc = 0.01*1./timestep # every 10 fs

    str_file = pycharmm.CharmmFile(
        file_name='charmm_data/heat.res', file_unit=3, 
        formatted=True, read_only=False)
    res_file = pycharmm.CharmmFile(
        file_name='charmm_data/nve.res', file_unit=2, 
        formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='charmm_data/nve.dcd', file_unit=1, 
        formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'verlet': True,
        'new': False,
        'start': False,
        'restart': True,
        'timestep': timestep,
        'nstep': nsteps,
        'nsavc': nsavc,
        'inbfrq': -1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': str_file.file_unit,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 0,
        'ihtfrq': 0,
        'ieqfrq': 0,
        'echeck':-1}

    dyn_nve = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_nve.run()

    str_file.close()
    res_file.close()
    dcd_file.close()

# Step 7: Equilibration - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
        
    timestep = 0.00025   # 0.25 fs
    nsteps = 50.*1./timestep # 50 ps
    nsavc = 0.01*1./timestep # every 10 fs
    temp = 300.0

    pmass = int(np.sum(select.get_property('mass'))/50.0)
    tmass = int(pmass*10)

    str_file = pycharmm.CharmmFile(
        file_name='charmm_data/heat.res', file_unit=3, 
        formatted=True, read_only=False)
    res_file = pycharmm.CharmmFile(
        file_name='charmm_data/equi.res', file_unit=2, 
        formatted=True, read_only=False)
    dcd_file = pycharmm.CharmmFile(
        file_name='charmm_data/equi.dcd', file_unit=1, 
        formatted=False, read_only=False)

    # Run some dynamics
    dynamics_dict = {
        'leap': True,
        'cpt': True,
        'new': False,
        'start': False,
        'restart': True,
        'timestep': timestep,
        'nstep': nsteps,
        'nsavc': nsavc,
        'inbfrq': -1,
        'ihbfrq': 50,
        'ilbfrq': 50,
        'imgfrq': 50,
        'ixtfrq': 1000,
        'iunrea': str_file.file_unit,
        'iunwri': res_file.file_unit,
        'iuncrd': dcd_file.file_unit,
        'nprint': 100, # Frequency to write to output
        'iprfrq': 500, # Frequency to calculate averages
        'isvfrq': 1000, # Frequency to save restart file
        'ntrfrq': 1000,
        'pint pconst pref': 1,
        'pgamma': 5,
        'pmass': pmass,
        'hoover reft': temp,
        'tmass': tmass,
        'echeck':-1}

    dyn_equi = pycharmm.DynamicsScript(**dynamics_dict)
    dyn_equi.run()
    
    str_file.close()
    res_file.close()
    dcd_file.close()

# Step 6: Production - CHARMM, PhysNet
#-----------------------------------------------------------

if True:
    
    timestep = 0.00025   # 0.25 fs
    nsteps = 100.*1./timestep # 50 ps
    nsavc = 0.01*1./timestep # every 10 fs
    temp = 300.0

    pmass = int(np.sum(select.get_property('mass'))/50.0)
    tmass = int(pmass*10)

    for ii in range(0, 10):
        
        if ii==0:

            str_file = pycharmm.CharmmFile(
                file_name='charmm_data/equi.res', 
                file_unit=3, formatted=True, read_only=False)
            res_file = pycharmm.CharmmFile(
                file_name='charmm_data/dyna.{:d}.res'.format(ii), 
                file_unit=2, formatted=True, read_only=False)
            dcd_file = pycharmm.CharmmFile(
                file_name='charmm_data/dyna.{:d}.dcd'.format(ii), 
                file_unit=1, formatted=False, read_only=False)
            
        else:
            
            str_file = pycharmm.CharmmFile(
                file_name='charmm_data/dyna.{:d}.res'.format(ii - 1), 
                file_unit=3, formatted=True, read_only=False)
            res_file = pycharmm.CharmmFile(
                file_name='charmm_data/dyna.{:d}.res'.format(ii), 
                file_unit=2, formatted=True, read_only=False)
            dcd_file = pycharmm.CharmmFile(
                file_name='charmm_data/dyna.{:d}.dcd'.format(ii), 
                file_unit=1, formatted=False, read_only=False)

        # Run some dynamics
        dynamics_dict = {
            'leap': True,
            'cpt': True,
            'new': False,
            'start': False,
            'restart': True,
            'timestep': timestep,
            'nstep': nsteps,
            'nsavc': nsavc,
            'inbfrq': -1,
            'ihbfrq': 50,
            'ilbfrq': 50,
            'imgfrq': 50,
            'ixtfrq': 1000,
            'iunrea': str_file.file_unit,
            'iunwri': res_file.file_unit,
            'iuncrd': dcd_file.file_unit,
            'nprint': 100, # Frequency to write to output
            'iprfrq': 500, # Frequency to calculate averages
            'isvfrq': 1000, # Frequency to save restart file
            'ntrfrq': 1000,
            'pint pconst pref': 1,
            'pgamma': 5,
            'pmass': pmass,
            'hoover reft': temp,
            'tmass': tmass,
            'echeck':-1}

        dyn_prod = pycharmm.DynamicsScript(**dynamics_dict)
        dyn_prod.run()
        
        str_file.close()
        res_file.close()
        dcd_file.close()
