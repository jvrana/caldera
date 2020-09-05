import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
from caldera.__version__ import __version__
import caldera as pkg
import datetime

now = datetime.datetime.now()
project = pkg.__version__.__title__
authors = pkg.__version__.__authors__
copyright = "{year}, {authors}".format(year=now.year, authors=','.join(authors))
author = authors[0]
release = pkg.__version__.__version__


# -- General configuration ---------------------------------------------------
import sphinx_rtd_theme

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    'sphinx.ext.autodoc',
    "sphinx.ext.autosummary"
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
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']