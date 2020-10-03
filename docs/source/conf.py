# flake8: noqa
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
import caldera.gnn

# -- Project information -----------------------------------------------------
import caldera as pkg
from caldera.utils import deterministic_seed

deterministic_seed(0)

import datetime

now = datetime.datetime.now()
project = pkg.__title__
authors = pkg.__authors__
copyright = "{year}, {authors}".format(year=now.year, authors=",".join(authors))
author = authors[0]
release = pkg.__version__

# -- General configuration ---------------------------------------------------
import glob

autosummary_generate = (
    True
    # glob.glob("source/api/*.rst")  # Make _autosummary files and include them
)
autoclass_content = "both"  # include both class docstring and __init__

autodoc_default_options = {
    "member-order": "bysource",
    "special-members": "__init__",
    "exclude-members": "__weakref__",
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "jupyter_sphinx",
    # 'sphinxcontrib.katex',
]

# Disable docstring inheritance
autodoc_inherit_docstrings = False

always_document_param_types = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "_templates/autosummary"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
import sphinx_bootstrap_theme

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Guzzle theme options (see theme.conf for more information)
html_theme_options = {
    # Set the name of the project to appear in the sidebar
    "navbar_title": pkg.__title__.capitalize() + " " + str(pkg.__version__),
    "navbar_site_name": pkg.__title__.capitalize(),
    # A list of tuples containing pages or urls to link to.
    # Valid tuples should be in the following forms:
    #    (name, page)                 # a link to a page
    #    (name, "/aa/bb", 1)          # a link to an arbitrary relative url
    #    (name, "http://example.com", True) # arbitrary absolute url
    # Note the "1" or "True" value above as the third argument to indicate
    # an arbitrary url.
    "navbar_links": [
        ("Getting Started", "getting_started"),
        ("Examples", "examples/examples"),
        ("Gallery", "gallery/gallery"),
        ("API", "api/api"),
        ("Github", pkg.__homepage__, True),
    ],
    # Render the next and previous page links in navbar. (Default: true)
    "navbar_sidebarrel": False,
    # Render the current pages TOC in the navbar. (Default: true)
    "navbar_pagenav": True,
    # Tab name for the current pages TOC. (Default: "Page")
    "navbar_pagenav_name": "Page",
    "globaltoc_depth": 4,
    # Location of link to source.
    # Options are "nav" (default), "footer" or anything else to exclude.
    "source_link_position": "footer" "",
    # Bootswatch (http://bootswatch.com/) theme.
    #
    # Options are nothing (default) or the name of a valid theme
    # such as "cosmo" or "sandstone".
    #
    # The set of valid themes depend on the version of Bootstrap
    # that's used (the next config option).
    #
    # Currently, the supported themes are:
    # - Bootstrap 2: https://bootswatch.com/2
    # - Bootstrap 3: https://bootswatch.com/3
    "bootswatch_theme": "simplex",
    # Choose Bootstrap version.
    # Values: "3" (default) or "2" (in quotes)
    "bootstrap_version": "3",
}

## uncomment to add globaltoc sidebar
html_sidebars = {
    "api/**": ["api_sidebar.html"],
}

# Add the 'copybutton' javascript, to hide/show the prompt in code
# examples, originally taken from scikit-learn's doc/conf.py
def setup(app):
    app.add_js_file("copybutton.js")
    app.add_css_file("style.css")


# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "networkx": ("https://networkx.github.io/documentation/stable/", None),
}
