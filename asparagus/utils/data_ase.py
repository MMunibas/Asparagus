from ase.data import (
    chemical_symbols, atomic_names, atomic_masses, atomic_numbers
)

# Switch to lower case atomic symbols
atomic_numbers = {
    symbol.lower(): Z for symbol, Z in atomic_numbers.items()}
