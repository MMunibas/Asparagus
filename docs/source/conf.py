# Configuration file for the Sphinx documentation builder.
import os
import sys
import asparagus
sys.path.insert(0, os.path.abspath(os.path.join('.','..','..')))
# -- Project information

project = 'Asparagus Bundle'
copyright = '2024, L.I.Vazquez-Salazar, K. Toepfer & M. Meuwly'
author = 'L.I.Vazquez-Salazar & K. Toepfer'

release = '0.4'
version = '0.4.2'

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

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

autodoc_mock_imports = [
    'numpy',
    'ase',
    'ctype',
    'torchvision',
    'torchaudio',
    'torch',
    'torch-ema',
    'tensorboard',
    'xtb',
    'h5py',
    'pandas',
    'matplotlib',
    'seaborn',
    'scipy',
    'pytest',]

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
