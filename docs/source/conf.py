# Configuration file for the Sphinx documentation builder.
import os
import sys
from unittest.mock import MagicMock
from importlib.metadata import version as pkg_version, PackageNotFoundError

# Make package importable for autodoc (adjust as needed)
sys.path.insert(0, os.path.abspath(os.path.join('.', '..', '..')))

MOCK_MODULES = [
    "numpy", "pandas", "scipy", "h5py", "matplotlib", "seaborn",
    "torch", "torchvision", "torchaudio", "tensorboard", "torch_ema",
    "ase", "xtb",
    "openmm", "openmmtorch", "openmmml",
    "openmm.unit", "torch.multiprocessing",
]

for m in MOCK_MODULES:
    sys.modules.setdefault(m, MagicMock())

# -- Project information

project = 'Asparagus Bundle'
copyright = '2025, L.I.Vazquez-Salazar, K. Toepfer & M. Meuwly'
author = 'L.I.Vazquez-Salazar & K. Toepfer'

try:
    release = pkg_version("asparagus")
    version = release
except PackageNotFoundError:
    release = "0.6"
    version  = "0.6.0"
    
# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinxemoji.sphinxemoji',
    'sphinx.ext.todo',
    'myst_parser',]
    
autosummary_generate = True


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

autodoc_mock_imports = [
    "numpy",
    "ase",
    "ctypes",
    "torchvision",
    "torchaudio",
    "torch",
    "torch_ema",
    "tensorboard",
    "xtb",
    "h5py",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "openmm",
    "openmmtorch",
    "openmmml",
    "pytest",
]

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'
# html_logo = 'logo_low.png'
# html_theme_options = {
#     'logo_only': True,
#     'display_version': False,
# }

# -- Options for EPUB output
epub_show_urls = 'footnote'
