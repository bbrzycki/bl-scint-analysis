# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path
import sys
blscint_path = Path(__file__).parents[2].resolve()
sys.path.insert(0, str(blscint_path))

version_dict = {}
with open(blscint_path / "blscint/_version.py") as fp:
    exec(fp.read(), version_dict)

# -- Project information -----------------------------------------------------

project = 'blscint'
copyright = '2022-2023, Bryan Brzycki'
author = 'Bryan Brzycki'

# The full version, including alpha/beta/rc tags
release = version_dict["__version__"]


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_theme
html_theme = 'stanford_theme'
html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# The master toctree document.
master_doc = 'index'

# Order of docstrings; by source or alphabetical.
autodoc_member_order = 'bysource'

autodoc_mock_imports = ["turbo_seti"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

intersphinx_mapping = {'NumPy': ('http://docs.scipy.org/doc/numpy/', None),
                       'matplotlib': ('http://matplotlib.org', None)}


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True