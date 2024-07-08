
#======================================
# Cutoff function labels
#======================================

# Valid cutoff function labels
_valid_cutoff_fn = [
    'Poly6',
]

#======================================
# Property labels
#======================================

# Essential property labels
_default_property_labels = [
    'positions',
    'charge',
]

# Valid property labels
_valid_properties = [
    'positions',
    'energy',
    'atomic_energies',
    'forces',
    'hessian',
    'charge',
    'atomic_charges',
    'atomic_dipoles',
    'dipole',
    'polarizability',
]

# Alternative property label dictionary 
# (keys:    internally used property labels
#  items:   possible externally used property labels)
# Comparison always between lowercase form
_alt_property_labels = {
    'atoms_number':     ['N'],
    'atomic_numbers':   ['Z', 'atomic_number'],
    'positions':        ['R', 'position'],
    'cell':             ['unit_cell'],
    'pbc':              ['periodicity'],
    'energy':           ['E', 'energies', 'U0', 'V'],
    'atomic_energies':  [
        'Ea', 'Ei', 'atom_energy', 'atoms_energy', 'atomic_energy',
        'atom_energies', 'atoms_energies'],
    'forces':           ['F', 'force'],
    'hessian':          ['H', 'hessians'],
    'charge':           ['Q', 'charges', 'total_charge', 'total_charges'],
    'atomic_charges':    [
        'Qa', 'Qi', 'atom_charge', 'atom_charges', 
        'atomic_charge'],
    'atomic_dipoles':    [
        'Da', 'Fi', 'atom_dipole', 'atom_dipoles', 
        'atomic_dipole'],
    'dipole':           ['D', 'dipoles'],
    'polarizability':   ['polarizability'],
    }
    

# Default property units - ASE units
_default_units = {
    'atoms_number':     '',
    'atomic_numbers':   '',
    'positions':        'Ang',
    'energy':           'eV',
    'atomic_energies':  'eV',
    'forces':           'eV/Ang',
    'hessian':          'eV/Ang**2',
    'charge':           'e',
    'atomic_charges':   'e',
    'dipole':           'e*Ang',
    }

# Default output block options for properties
_default_output_block_options = {
    'energy':           {
        'aggregation':      ['sum', 'atomic_energies'],
        },
    'atomic_energies':      {
        'aggregation':      None,
        'n_outputneurons':  1,
        'n_hiddenlayers':   2,
        'n_hiddenneurons':  None,
        'output_bias':      True,
        'output_init_zero': True,
        },
    }
